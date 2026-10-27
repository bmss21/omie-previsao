"""Testes offline: formato OMIE e pipeline completo com dados sintéticos."""

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from previsao.dados import MADRID, PONTOS, UTC, VARIAVEIS, parse_marginal
from run import calcular, horas_locais


def test_parse_horario():
    txt = "MARGINALPDBC;\n" + "".join(f"2025;03;10;{h};{50 + h:.2f};{60 + h:.2f};\n" for h in range(1, 25)) + "*\n"
    df = parse_marginal(txt)
    assert len(df) == 24
    assert df["preco"].iloc[0] == 51  # coluna de Portugal
    assert df["ts"].iloc[0] == pd.Timestamp("2025-03-09 23:00", tz="UTC")


def test_parse_quartos_e_mudanca_hora():
    # 26/10/2025: dia com 25 horas → 100 quartos de hora
    txt = "".join(f"2025;10;26;{q};{q:.2f};0;\n" for q in range(1, 101))
    df = parse_marginal(txt)
    assert len(df) == 25
    assert df["preco"].iloc[0] == np.mean([1, 2, 3, 4])
    assert df["ts"].iloc[0] == pd.Timestamp("2025-10-25 22:00", tz="UTC")


def _sintetico(d0, d1, seed=1):
    rng = np.random.default_rng(seed)
    ts = pd.date_range(datetime(d0.year, d0.month, d0.day, tzinfo=UTC), datetime(d1.year, d1.month, d1.day, tzinfo=UTC), freq="h")
    n = len(ts)
    dias = np.arange(n) / 24
    vento = 7 + 4 * np.sin(2 * np.pi * dias / 5.3) + rng.normal(0, 1.5, n)
    hora = ts.tz_convert(MADRID).hour
    sol = np.clip(np.sin(np.pi * (hora - 7) / 12), 0, None) * 700 * (0.7 + 0.3 * np.sin(2 * np.pi * dias / 365))
    met = {"ts": ts}
    for nome in PONTOS:
        met[f"{nome}__wind_speed_100m"] = np.clip(vento + rng.normal(0, 1, n), 0, None)
        met[f"{nome}__shortwave_radiation"] = sol
        met[f"{nome}__temperature_2m"] = 15 + 8 * np.sin(2 * np.pi * (dias - 100) / 365)
        met[f"{nome}__precipitation"] = rng.exponential(0.1, n)
    met = pd.DataFrame(met)
    preco = 120 - 9 * (vento - 7) - 0.12 * sol + 25 * np.sin(2 * np.pi * dias / 180) + rng.normal(0, 6, n)
    precos = pd.DataFrame({"ts": ts, "preco": np.clip(preco, 0, None)})
    return precos, met


def test_pipeline():
    hoje = date(2025, 10, 27)
    precos, met = _sintetico(date(2024, 1, 1), hoje + timedelta(days=2))
    precos = precos[precos["ts"] < pd.Timestamp(datetime(2025, 10, 28, tzinfo=MADRID))]
    met_prev = met[met["ts"] >= pd.Timestamp("2025-10-20", tz="UTC")]
    met_prev = pd.concat([met_prev, _sintetico(hoje + timedelta(days=2), hoje + timedelta(days=10), 2)[1]])
    met_hist = met[met["ts"] < pd.Timestamp("2025-10-27", tz="UTC")]
    r = calcular(precos, met_hist, met_prev.drop_duplicates("ts"), hoje)
    assert r["ultimo_dia_real"] == "2025-10-27"
    e = r["erro_backtest_eur_mwh"]
    print(e)
    assert e["modelo"] < e["media_7d"]
    prim = datetime.fromtimestamp(r["previsao"][0][0], MADRID)
    ult = datetime.fromtimestamp(r["previsao"][-1][0], MADRID)
    assert (prim.date(), prim.hour) == (date(2025, 10, 28), 0)
    assert (ult.date(), ult.hour) == (date(2025, 11, 3), 23)
    assert len(r["previsao"]) == len(horas_locais(date(2025, 10, 28), date(2025, 11, 4)))
    assert all(a[1] <= b[1] <= c[1] for a, b, c in zip(r["p10"], r["previsao"], r["p90"]))
