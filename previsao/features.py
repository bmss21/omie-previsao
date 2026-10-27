"""Construção das variáveis do modelo.

Ideia: prever o preço da hora T sabendo apenas os preços OMIE até ao fim
do dia K (o último dia publicado) e a meteorologia prevista para T.
horizonte = dias entre K e o dia de T (1 a 8).
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from .dados import MADRID, PONTOS

HORIZONTES = range(1, 9)


# ------------------------------------------------------------- feriados

def _pascoa(ano: int) -> date:
    a, b, c = ano % 19, ano // 100, ano % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    L = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * L) // 451
    mes = (h + L - 7 * m + 114) // 31
    dia = (h + L - 7 * m + 114) % 31 + 1
    return date(ano, mes, dia)


def feriados(anos) -> set[date]:
    """Feriados nacionais de Espanha (pesam mais no preço ibérico) e Portugal."""
    out = set()
    for a in anos:
        for m, d in [(1, 1), (1, 6), (5, 1), (8, 15), (10, 12), (11, 1), (12, 6), (12, 8), (12, 25),
                     (4, 25), (6, 10), (10, 5), (12, 1)]:
            out.add(date(a, m, d))
        p = _pascoa(a)
        out.add(p - timedelta(days=2))  # Sexta-feira Santa
    return out


# ----------------------------------------------------------- meteorologia

def agregar_meteo(m: pd.DataFrame) -> pd.DataFrame:
    """Reduz os pontos a poucos indicadores ibéricos."""
    def media(tipo, var):
        cols = [f"{n}__{var}" for n, (_, _, t) in PONTOS.items() if tipo in t]
        return m[cols].astype(float).mean(axis=1)

    out = pd.DataFrame({"ts": m["ts"]})
    ws = [f"{n}__wind_speed_100m" for n, (_, _, t) in PONTOS.items() if "v" in t]
    vento = m[ws].astype(float)
    # Curva de potência simplificada de um aerogerador (arranque 3 m/s, nominal 12, corte 25)
    pot = ((vento.clip(3, 12) - 3) / 9) ** 3
    pot[vento > 25] = 0
    out["vento_pot"] = pot.mean(axis=1)
    out["vento_ms"] = vento.mean(axis=1)
    out["sol"] = media("s", "shortwave_radiation")
    out["temp"] = media("t", "temperature_2m")
    out["temp_desvio"] = (out["temp"] - 18).abs()
    chuva = media("c", "precipitation").fillna(0)
    out = out.sort_values("ts").reset_index(drop=True)
    out["chuva_7d"] = chuva.rolling(168, min_periods=1).sum().values
    out["chuva_30d"] = chuva.rolling(720, min_periods=1).sum().values
    return out


# ---------------------------------------------------------------- preços

def tabela_diaria(precos: pd.DataFrame) -> pd.DataFrame:
    """Matriz dia local × hora local (média se houver hora repetida no fim do verão)."""
    p = precos.copy()
    local = p["ts"].dt.tz_convert(MADRID)
    p["dia"] = local.dt.date
    p["hora"] = local.dt.hour
    return p.pivot_table(index="dia", columns="hora", values="preco", aggfunc="mean").sort_index()


def lags_ate(tab: pd.DataFrame, k: date) -> dict | None:
    """Indicadores de preço conhecidos no fim do dia k."""
    janela = tab.loc[(tab.index > k - timedelta(days=7)) & (tab.index <= k)]
    if len(janela) < 5 or k not in tab.index:
        return None
    return {
        "nivel_7d": float(np.nanmean(janela.values)),
        "nivel_1d": float(np.nanmean(tab.loc[k].values)),
        "perfil": janela.mean(axis=0).to_dict(),  # média por hora local
        "min_7d": float(np.nanmin(janela.values)),
        "max_7d": float(np.nanmax(janela.values)),
    }


METEO_JANELA = ["vento_pot", "sol", "temp_desvio"]


def meteo_7d(met: pd.DataFrame) -> pd.DataFrame:
    """Meteorologia média dos 7 dias (locais) que terminam em cada dia."""
    m = met.copy()
    m["dia"] = m["ts"].dt.tz_convert(MADRID).dt.date
    return m.groupby("dia")[METEO_JANELA].mean().rolling(7, min_periods=5).mean()


def linhas(alvos: pd.DataFrame, met: pd.DataFrame, tab: pd.DataFrame, k: date, fer: set[date],
           r7: pd.DataFrame | None = None) -> pd.DataFrame:
    """Variáveis para as horas `alvos` (coluna ts) quando o último dia conhecido é k."""
    lg = lags_ate(tab, k)
    if r7 is None:
        r7 = meteo_7d(met)
    if lg is None or k not in r7.index:
        return pd.DataFrame()
    df = alvos[["ts"]].merge(met, on="ts", how="left")
    local = df["ts"].dt.tz_convert(MADRID)
    dia = local.dt.date
    df["hora"] = local.dt.hour
    df["dow"] = local.dt.dayofweek
    df["fim_semana"] = (df["dow"] >= 5).astype(int)
    df["feriado"] = dia.isin(fer).astype(int)
    doy = local.dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["horizonte"] = [(d - k).days for d in dia]
    df["nivel_7d"] = lg["nivel_7d"]
    df["nivel_1d"] = lg["nivel_1d"]
    df["min_7d"] = lg["min_7d"]
    df["max_7d"] = lg["max_7d"]
    df["perfil_hora"] = df["hora"].map(lg["perfil"]).astype(float)
    # Tudo relativo ao nível recente, para o modelo funcionar mesmo quando os preços
    # sobem ou descem para valores que nunca viu (ex.: verão/outono de 2026).
    df["perf_rel"] = df["perfil_hora"] - lg["nivel_7d"]
    df["min_rel"] = lg["min_7d"] - lg["nivel_7d"]
    df["max_rel"] = lg["max_7d"] - lg["nivel_7d"]
    df["d1_rel"] = lg["nivel_1d"] - lg["nivel_7d"]
    df["amplitude"] = max(lg["max_7d"] - lg["min_7d"], 20.0)
    # Meteorologia prevista face à da semana que formou o perfil de preços
    for c in METEO_JANELA:
        df[f"{c}_7d"] = float(r7.loc[k, c])
        df[f"{c}_anom"] = df[c] - df[f"{c}_7d"]
    return df


COLUNAS = [
    "hora", "dow", "fim_semana", "feriado", "doy_sin", "doy_cos", "horizonte",
    "vento_pot", "vento_ms", "sol", "temp", "temp_desvio", "chuva_7d", "chuva_30d",
    "perf_rel", "min_rel", "max_rel", "d1_rel", "amplitude",
    "vento_pot_7d", "vento_pot_anom", "sol_7d", "sol_anom", "temp_desvio_anom",
]


def conjunto_treino(precos: pd.DataFrame, met: pd.DataFrame, dias_alvo) -> pd.DataFrame:
    """Para cada dia alvo T e horizonte h: k = T - h. Uma linha por hora × horizonte."""
    tab = tabela_diaria(precos)
    fer = feriados(range(min(tab.index).year, max(tab.index).year + 2))
    p = precos.copy()
    p["dia"] = p["ts"].dt.tz_convert(MADRID).dt.date
    por_dia = dict(tuple(p.groupby("dia")))
    r7 = meteo_7d(met)
    partes = []
    for t in dias_alvo:
        if t not in por_dia:
            continue
        alvo = por_dia[t]
        for h in HORIZONTES:
            df = linhas(alvo, met, tab, t - timedelta(days=h), fer, r7)
            if df.empty:
                continue
            df["preco"] = alvo["preco"].values
            partes.append(df)
    return pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()
