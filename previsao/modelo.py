"""Treino, avaliação e previsão."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import COLUNAS


def _modelo(perda: str = "absolute_error", quantil: float | None = None) -> HistGradientBoostingRegressor:
    kw = dict(max_iter=800, learning_rate=0.03, max_leaf_nodes=31, min_samples_leaf=100,
              l2_regularization=1.0, random_state=0)
    if quantil is not None:
        return HistGradientBoostingRegressor(loss="quantile", quantile=quantil, **kw)
    return HistGradientBoostingRegressor(loss=perda, **kw)


# O modelo aprende a correção à regra simples (média dos 7 dias à mesma hora):
# preço = perfil_hora + correção(vento, sol, temperatura, ...).
def _alvo(df: pd.DataFrame) -> pd.Series:
    return df["preco"] - df["perfil_hora"]


def treinar(df: pd.DataFrame) -> dict:
    X, y = df[COLUNAS], _alvo(df)
    return {
        "p50": _modelo().fit(X, y),
        "p10": _modelo(quantil=0.1).fit(X, y),
        "p90": _modelo(quantil=0.9).fit(X, y),
    }


def prever(modelos: dict, df: pd.DataFrame) -> pd.DataFrame:
    X = df[COLUNAS]
    out = pd.DataFrame({"ts": df["ts"]})
    for nome, m in modelos.items():
        out[nome] = df["perfil_hora"].values + m.predict(X)
    # Garante p10 ≤ p50 ≤ p90
    v = np.sort(out[["p10", "p50", "p90"]].values, axis=1)
    out["p10"], out["p50"], out["p90"] = v[:, 0], v[:, 1], v[:, 2]
    return out


def avaliar(treino: pd.DataFrame, teste: pd.DataFrame) -> dict:
    """Erro médio absoluto (€/MWh) do modelo e da regra simples (média dos 7 dias a essa hora)."""
    m = _modelo().fit(treino[COLUNAS], _alvo(treino))
    pred = teste["perfil_hora"].values + m.predict(teste[COLUNAS])
    erro_m = np.abs(pred - teste["preco"])
    erro_b = np.abs(teste["perfil_hora"] - teste["preco"])
    por_h = {
        int(h): {"modelo": round(float(erro_m[teste["horizonte"] == h].mean()), 1),
                 "media_7d": round(float(erro_b[teste["horizonte"] == h].mean()), 1)}
        for h in sorted(teste["horizonte"].unique())
    }
    return {
        "modelo": round(float(erro_m.mean()), 1),
        "media_7d": round(float(erro_b.mean()), 1),
        "por_horizonte": por_h,
    }
