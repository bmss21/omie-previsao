"""Corre a previsão diária e grava docs/previsao.json.

Uso: python run.py
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from previsao import dados
from previsao.dados import MADRID, UTC
from previsao.features import agregar_meteo, conjunto_treino, feriados, linhas, tabela_diaria
from previsao.modelo import avaliar, prever, treinar

RAIZ = Path(__file__).parent
INICIO = date(2024, 1, 1)
DIAS_TESTE = 90
DIAS_FRENTE = 7


def horas_locais(d0: date, d1: date) -> pd.DataFrame:
    """Horas (UTC) desde as 00h locais de d0 até às 00h locais de d1."""
    a = datetime(d0.year, d0.month, d0.day, tzinfo=MADRID).astimezone(UTC)
    b = datetime(d1.year, d1.month, d1.day, tzinfo=MADRID).astimezone(UTC)
    return pd.DataFrame({"ts": pd.date_range(a, b, freq="h", inclusive="left")})


def calcular(precos: pd.DataFrame, met_hist: pd.DataFrame, met_prev: pd.DataFrame, hoje: date) -> dict:
    tab = tabela_diaria(precos)
    completos = tab.index[tab.notna().sum(axis=1) >= 23]
    k = max(completos)  # último dia com preços publicados

    met = pd.concat([met_hist[met_hist["ts"] < met_prev["ts"].min()], met_prev])
    met = agregar_meteo(met.drop_duplicates("ts", keep="last"))

    dias = [d for d in completos if d >= INICIO + timedelta(days=8)]
    corte = k - timedelta(days=DIAS_TESTE)
    treino = conjunto_treino(precos, met, [d for d in dias if d <= corte]).dropna(subset=["preco"])
    teste = conjunto_treino(precos, met, [d for d in dias if d > corte]).dropna(subset=["preco"])
    metricas = avaliar(treino, teste)
    usar_modelo = metricas["modelo"] < metricas["media_7d"]
    print(f"backtest {DIAS_TESTE} dias: modelo {metricas['modelo']} €/MWh, "
          f"média 7d {metricas['media_7d']} €/MWh → {'modelo' if usar_modelo else 'média'}")

    alvos = horas_locais(k + timedelta(days=1), hoje + timedelta(days=DIAS_FRENTE + 1))
    fer = feriados(range(hoje.year, hoje.year + 2))
    x = linhas(alvos, met, tab, k, fer)

    if usar_modelo:
        tudo = pd.concat([treino, teste], ignore_index=True)
        p = prever(treinar(tudo), x)
        metodo = "gradient boosting (vento, sol, temperatura, chuva, histórico)"
    else:
        p = pd.DataFrame({"ts": x["ts"], "p50": x["perfil_hora"]})
        p["p10"] = p["p50"]
        p["p90"] = p["p50"]
        metodo = "média dos últimos 7 dias à mesma hora"

    p = p.dropna()

    def serie(col):
        return [[int(t.timestamp()), round(float(v) / 1000, 5)] for t, v in zip(p["ts"], p[col])]

    return {
        "gerado": datetime.now(UTC).isoformat(timespec="seconds"),
        "ultimo_dia_real": k.isoformat(),
        "unidade": "EUR/kWh",
        "metodo": metodo,
        "erro_backtest_eur_mwh": {**metricas, "dias": DIAS_TESTE},
        "previsao": serie("p50"),
        "p10": serie("p10"),
        "p90": serie("p90"),
    }


def main() -> None:
    hoje = datetime.now(MADRID).date()
    precos = dados.atualizar_precos(RAIZ / "data/precos.csv", INICIO, hoje)
    met_hist = dados.atualizar_meteo_hist(RAIZ / "data/meteo_hist.csv", INICIO - timedelta(days=31), hoje)
    met_prev = dados.meteo_previsao()
    resultado = calcular(precos, met_hist, met_prev, hoje)
    saida = RAIZ / "docs/previsao.json"
    saida.parent.mkdir(exist_ok=True)
    saida.write_text(json.dumps(resultado, ensure_ascii=False, separators=(",", ":")))
    (RAIZ / "docs/metricas.json").write_text(
        json.dumps({k: resultado[k] for k in ("gerado", "ultimo_dia_real", "metodo", "erro_backtest_eur_mwh")},
                   ensure_ascii=False, indent=2))
    print(f"previsão gravada: {len(resultado['previsao'])} horas até "
          f"{datetime.fromtimestamp(resultado['previsao'][-1][0], MADRID):%d/%m %H:%M}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
