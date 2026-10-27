"""Download e cache dos dados: preços OMIE (Portugal) e meteorologia (Open-Meteo).

Tudo é guardado em hora UTC. Os preços ficam em €/MWh, médias horárias
(desde out/2025 o OMIE publica a 15 minutos; aqui agregamos por hora).
"""

from __future__ import annotations

import io
import time
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

MADRID = ZoneInfo("Europe/Madrid")
UTC = timezone.utc

OMIE_URL = "https://www.omie.es/es/file-download"
OM_HIST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
OM_PREV_URL = "https://api.open-meteo.com/v1/forecast"

SESSAO = requests.Session()
SESSAO.headers["User-Agent"] = "omie-previsao (github.com/bmss21/omie-previsao)"

# Pontos representativos da produção ibérica.
# tipo: v = vento, s = solar, t = temperatura/consumo, c = chuva (hídrica)
PONTOS = {
    "corunha": (43.30, -8.40, "v"),
    "viana": (41.80, -8.45, "vc"),
    "burgos": (42.30, -3.70, "v"),
    "zaragoza": (41.60, -0.90, "v"),
    "albacete": (39.00, -1.90, "v"),
    "tarifa": (36.05, -5.60, "v"),
    "castelo_branco": (39.80, -7.50, "vc"),
    "douro": (41.20, -7.20, "c"),
    "badajoz": (38.90, -7.00, "s"),
    "sevilha": (37.40, -6.00, "st"),
    "ciudad_real": (38.98, -3.90, "s"),
    "alentejo": (38.00, -7.90, "s"),
    "murcia": (38.00, -1.10, "s"),
    "madrid": (40.40, -3.70, "t"),
    "lisboa": (38.72, -9.14, "t"),
    "porto": (41.15, -8.61, "t"),
}
VARIAVEIS = ["wind_speed_100m", "shortwave_radiation", "temperature_2m", "precipitation"]


# ---------------------------------------------------------------- OMIE

def _get(url: str, params: dict | None = None, tentativas: int = 4) -> requests.Response | None:
    for i in range(tentativas):
        try:
            r = SESSAO.get(url, params=params, timeout=60)
            if r.status_code == 200 and r.content:
                return r
            if r.status_code in (404, 410):
                return None
        except requests.RequestException:
            pass
        time.sleep(2 * (i + 1))
    return None


def parse_marginal(texto: str) -> pd.DataFrame:
    """Lê um ficheiro marginalpdbc do OMIE.

    Linhas de dados: ano;mes;dia;periodo;preco_PT;preco_ES;
    Os períodos são horas (1–25) ou, desde out/2025, quartos de hora (1–100).
    Devolve DataFrame com ts (UTC, início da hora) e preco (€/MWh) – médias horárias.
    """
    linhas = []
    for linha in texto.splitlines():
        p = linha.strip().split(";")
        if len(p) >= 6 and p[0].strip().isdigit() and len(p[0].strip()) == 4:
            try:
                y, m, d, per = (int(x) for x in p[:4])
                pt = float(p[4].replace(",", "."))
            except ValueError:
                continue
            linhas.append((y, m, d, per, pt))
    if not linhas:
        return pd.DataFrame(columns=["ts", "preco"])
    df = pd.DataFrame(linhas, columns=["y", "m", "d", "per", "preco"])
    saida = []
    for (y, m, d), g in df.groupby(["y", "m", "d"]):
        n = g["per"].max()
        dur = timedelta(minutes=15) if n > 25 else timedelta(hours=1)
        inicio = datetime(y, m, d, tzinfo=MADRID).astimezone(UTC)
        ts = [inicio + (int(k) - 1) * dur for k in g["per"]]
        saida.append(pd.DataFrame({"ts": ts, "preco": g["preco"].values}))
    out = pd.concat(saida)
    out["ts"] = pd.to_datetime(out["ts"], utc=True).dt.floor("h")
    return out.groupby("ts", as_index=False)["preco"].mean()


def omie_dia(dia: date) -> pd.DataFrame:
    nome = f"marginalpdbc_{dia:%Y%m%d}.1"
    r = _get(OMIE_URL, {"parents": "marginalpdbc", "filename": nome})
    if r is None:
        return pd.DataFrame(columns=["ts", "preco"])
    return parse_marginal(r.content.decode("latin-1"))


def omie_ano(ano: int) -> pd.DataFrame:
    """Ficheiro anual (zip com um ficheiro por dia). Só existe para anos fechados."""
    r = _get(OMIE_URL, {"parents": "marginalpdbc", "filename": f"marginalpdbc_{ano}.zip"})
    if r is None:
        return pd.DataFrame(columns=["ts", "preco"])
    partes = []
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        for nome in sorted(z.namelist()):
            partes.append(parse_marginal(z.read(nome).decode("latin-1")))
    return pd.concat(partes) if partes else pd.DataFrame(columns=["ts", "preco"])


def atualizar_precos(cache: Path, inicio: date, hoje: date) -> pd.DataFrame:
    """Mantém data/precos.csv atualizado até amanhã (se já publicado)."""
    if cache.exists():
        df = pd.read_csv(cache, parse_dates=["ts"])
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        ultimo = df["ts"].max().tz_convert(MADRID).date()
        desde = ultimo - timedelta(days=2)
    else:
        df = pd.DataFrame(columns=["ts", "preco"])
        desde = inicio
    novos = []
    dia = desde
    # Anos completos de uma vez (zip anual); o resto dia a dia.
    while dia.year < hoje.year and dia.month == 1 and dia.day == 1:
        anual = omie_ano(dia.year)
        if anual.empty:
            break
        novos.append(anual)
        print(f"OMIE {dia.year}: {len(anual)} horas (zip anual)")
        dia = date(dia.year + 1, 1, 1)
    while dia <= hoje + timedelta(days=1):
        d = omie_dia(dia)
        if not d.empty:
            novos.append(d)
        elif dia <= hoje:
            print(f"aviso: sem preços OMIE para {dia}")
        dia += timedelta(days=1)
        time.sleep(0.2)
    if novos:
        df = pd.concat([df] + novos)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.drop_duplicates("ts", keep="last").sort_values("ts").reset_index(drop=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


# ---------------------------------------------------------- Open-Meteo

def _meteo(url: str, params: dict) -> pd.DataFrame:
    nomes = list(PONTOS)
    base = {
        "latitude": ",".join(str(PONTOS[n][0]) for n in nomes),
        "longitude": ",".join(str(PONTOS[n][1]) for n in nomes),
        "hourly": ",".join(VARIAVEIS),
        "timezone": "GMT",
        "wind_speed_unit": "ms",
    }
    r = _get(url, {**base, **params})
    if r is None:
        raise RuntimeError(f"Open-Meteo não respondeu: {url} {params}")
    dados = r.json()
    if isinstance(dados, dict):
        dados = [dados]
    colunas = {}
    for nome, bloco in zip(nomes, dados):
        h = bloco["hourly"]
        colunas["ts"] = pd.to_datetime(h["time"], utc=True)
        for v in VARIAVEIS:
            colunas[f"{nome}__{v}"] = h[v]
    return pd.DataFrame(colunas)


def atualizar_meteo_hist(cache: Path, inicio: date, ate: date) -> pd.DataFrame:
    """Previsões meteorológicas arquivadas (o que se previa na altura), em blocos de 3 meses."""
    if cache.exists():
        df = pd.read_csv(cache, parse_dates=["ts"])
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        desde = df["ts"].max().date() - timedelta(days=3)
    else:
        df = pd.DataFrame()
        desde = inicio
    partes = [df] if not df.empty else []
    a = desde
    while a <= ate:
        b = min(a + timedelta(days=90), ate)
        partes.append(_meteo(OM_HIST_URL, {"start_date": a.isoformat(), "end_date": b.isoformat()}))
        print(f"meteo {a} → {b}")
        a = b + timedelta(days=1)
        time.sleep(1)
    df = pd.concat(partes)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.drop_duplicates("ts", keep="last").sort_values("ts").reset_index(drop=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def meteo_previsao(dias: int = 9) -> pd.DataFrame:
    return _meteo(OM_PREV_URL, {"forecast_days": dias, "past_days": 7})
