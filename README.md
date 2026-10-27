# Previsão OMIE Portugal a 7 dias

Previsão horária do preço do mercado diário do OMIE (zona de Portugal) para os próximos 7 dias, usada no Home Assistant para programar consumos.

## Como funciona

Todos os dias, depois de o OMIE publicar os preços de amanhã, o GitHub Actions:

1. Atualiza o histórico de preços do OMIE (`data/precos.csv`, médias horárias em €/MWh).
2. Atualiza o histórico de previsões meteorológicas da [Open-Meteo](https://open-meteo.com/) (`data/meteo_hist.csv`) em 16 pontos da Península:
   - **vento a 100 m** (zonas eólicas: Galiza, Minho, Castela, Aragão, Albacete, Tarifa, Beira Interior), convertido em potência com uma curva de aerogerador;
   - **radiação solar** (Alentejo, Badajoz, Sevilha, Ciudad Real, Múrcia);
   - **temperatura** (Madrid, Lisboa, Porto, Sevilha), que mexe no consumo;
   - **chuva acumulada 7 e 30 dias** (Minho, Douro, Beira), indicador da produção hídrica.
3. Treina um modelo de gradient boosting com essas variáveis, o dia da semana, feriados e o nível de preços dos últimos 7 dias.
4. Compara-o, nos últimos 90 dias, com a regra simples (média dos últimos 7 dias à mesma hora). Se o modelo não for melhor, publica a regra simples.
5. Grava `docs/previsao.json` (previsão + intervalo P10–P90) e `docs/metricas.json` (erro do backtest).

## Ficheiro de saída

`https://raw.githubusercontent.com/bmss21/omie-previsao/main/docs/previsao.json`

```json
{
  "gerado": "2026-10-28T12:09:31+00:00",
  "ultimo_dia_real": "2026-10-29",
  "unidade": "EUR/kWh",
  "metodo": "...",
  "erro_backtest_eur_mwh": {"modelo": 18.4, "media_7d": 25.1, "por_horizonte": {}, "dias": 90},
  "previsao": [[1793228400, 0.08712], ...],
  "p10": [...],
  "p90": [...]
}
```

Cada par é `[hora em epoch UTC, preço em €/kWh]`, o mesmo formato do sensor `sensor.omie_previsao_semanal` do Home Assistant.

## Limitações

- As previsões meteorológicas usadas no treino são as de curto prazo arquivadas pela Open-Meteo; a 5–7 dias a meteorologia real é menos certa, por isso o erro nesses dias será maior do que no backtest.
- O preço do gás não entra diretamente; o nível de preços dos últimos 7 dias capta-o em parte.
- É uma previsão orientativa. O preço oficial é o que o OMIE publica.

## Testar localmente

```
pip install -r requirements.txt
python -c "import tests.test_sintetico as t; t.test_parse_horario(); t.test_parse_quartos_e_mudanca_hora(); t.test_pipeline()"
python run.py
```

Dados: OMIE; meteorologia Open-Meteo (CC BY 4.0).
