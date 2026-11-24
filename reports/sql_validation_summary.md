# Validacao das Queries Analiticas SQL (Inga Multimarcas)

Base validada em SQLite 3.53.1 aplicando `schema_postgresql.sql` e `analytical_queries.sql`.

## `q1_brand_year_tier_summary` (26 linhas)

| brand | faixa_ano | total_veiculos | preco_fipe_medio | preco_loja_medio | spread_medio_pct | km_medio | score_medio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Chevrolet | Seminovo (2022+) | 18 | 96959.28 | 99772.22 | 3.18 | 31872.0 | 89.1 |
| Chevrolet | Intermediario (2016-2021) | 14 | 62813.79 | 64535.71 | 2.18 | 96043.0 | 89.9 |
| Chevrolet | Veterano (<=2015) | 25 | 43912.32 | 44768.0 | 2.83 | 159524.0 | 85.6 |
| Dodge | Veterano (<=2015) | 17 | 77558.76 | 77682.35 | 0.13 | 263476.0 | 83.6 |
| Dodge | Intermediario (2016-2021) | 1 | 65009.0 | 66300.0 | 1.99 | 74900.0 | 95.0 |
| Fiat | Seminovo (2022+) | 17 | 97075.76 | 101241.18 | 4.36 | 20224.0 | 88.2 |
| Fiat | Intermediario (2016-2021) | 11 | 62154.27 | 64063.64 | 3.14 | 92900.0 | 88.1 |
| Fiat | Veterano (<=2015) | 10 | 21819.6 | 22620.0 | 3.76 | 231400.0 | 85.6 |

## `q2_annual_depreciation_by_brand` (9 linhas)

| brand | transicoes_anuais | variacao_media_anual_pct | menor_fipe_medio | maior_fipe_medio |
| --- | --- | --- | --- | --- |
| Ford | 23 | 14.82 | 21889.0 | 176371.0 |
| Jeep | 13 | 14.53 | 80354.5 | 225774.0 |
| Toyota | 19 | 14.0 | 45205.0 | 287302.67 |
| Volkswagen | 21 | 13.71 | 18272.0 | 148689.25 |
| Chevrolet | 19 | 12.63 | 16922.0 | 149678.5 |
| Hyundai | 25 | 11.52 | 7030.0 | 180998.0 |
| Honda | 17 | 10.26 | 42688.0 | 268972.0 |
| Fiat | 16 | 9.86 | 18049.0 | 115179.0 |

## `q3_regional_spread_maringa_vs_others` (10 linhas)

| region_pr | status_laudo | total_anuncios | fipe_medio_brl | loja_medio_brl | spread_medio_brl | spread_medio_pct | score_medio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Maringa - PR | Com Laudo | 117 | 99023.6 | 104139.32 | 5115.72 | 5.25 | 89.5 |
| Londrina - PR | Com Laudo | 42 | 93178.69 | 97442.86 | 4264.17 | 5.04 | 91.2 |
| Curitiba - PR | Com Laudo | 63 | 85912.29 | 89569.84 | 3657.56 | 4.79 | 90.6 |
| Cascavel - PR | Com Laudo | 26 | 86425.38 | 89057.69 | 2632.31 | 3.35 | 89.6 |
| Outras Pracas (SP/SC) | Com Laudo | 31 | 91605.55 | 92887.1 | 1281.55 | 2.39 | 89.5 |
| Maringa - PR | Sem Laudo | 35 | 54239.83 | 52965.71 | -1274.11 | -1.79 | 76.9 |
| Londrina - PR | Sem Laudo | 11 | 67953.09 | 66309.09 | -1644.0 | -3.25 | 76.5 |
| Curitiba - PR | Sem Laudo | 18 | 70990.22 | 67777.78 | -3212.44 | -3.32 | 76.2 |

## `q4_liquidity_ranking_and_moving_avg` (27 linhas)

| brand | rank_qualidade_marca | model | model_year | region_pr | quality_score | fipe_spread_pct | fipe_price_brl | market_asking_price | media_movel_fipe_3_registros |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Chevrolet | 1 | ONIX HATCH LTZ 1.0 12V TB Flex 5p Aut. | 2022 | Londrina - PR | 98 | 6.06 | 69675 | 73900 | 65979.67 |
| Chevrolet | 2 | Celta Life/ LS 1.0 MPFI 8V FlexPower 3p | 2014 | Londrina - PR | 98 | 8.94 | 26252 | 28600 | 43357.67 |
| Chevrolet | 3 | TRACKER LTZ 1.0 Turbo 12V Flex Aut. | 2021 | Maringa - PR | 98 | 10.62 | 85968 | 95100 | 72639.33 |
| Dodge | 1 | JOURNEY SXT 3.6 V6 Aut. | 2015 | Maringa - PR | 97 | 2.94 | 53719 | 55300 | 60717.0 |
| Dodge | 2 | JOURNEY RT 3.6 AWD V6 Aut. | 2015 | Maringa - PR | 96 | 4.85 | 63423 | 66500 | 56112.67 |
| Dodge | 3 | JOURNEY RT 3.6 AWD V6 Aut. | 2016 | Outras Pracas (SP/SC) | 95 | 1.99 | 65009 | 66300 | 59364.0 |
| Fiat | 1 | ARGO ENDURANCE 1.3 8V Flex | 2025 | Outras Pracas (SP/SC) | 98 | 3.37 | 79329 | 82000 | 83309.33 |
| Fiat | 2 | ARGO TREKKING 1.8 16V Flex Aut. | 2021 | Curitiba - PR | 95 | 5.93 | 74957 | 79400 | 68225.67 |

## `q5_mileage_regime_impact` (3 linhas)

| regime_uso | qtd_veiculos | idade_media_anos | km_anual_medio | score_medio | spread_medio_brl | spread_medio_pct |
| --- | --- | --- | --- | --- | --- | --- |
| 1. Baixa (<9k km/ano) | 55 | 10.2 | 6776.0 | 91.2 | 5150.02 | 6.54 |
| 2. Moderada (9k-15k km/ano) | 207 | 10.1 | 12237.0 | 89.5 | 3343.45 | 3.78 |
| 3. Intensa (>15k km/ano) | 105 | 8.6 | 17578.0 | 79.3 | -917.94 | -1.19 |

## `q6_ml_prediction_error_audit` (10 linhas)

| model_name | split_type | body_segment | n_amostras | preco_real_medio | preco_predito_medio | mae_brl | mape_pct | vies_medio_brl | acerto_dentro_5pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GradientBoostingRegressor | train | Minivan | 7 | 56042.86 | 56150.41 | 219.73 | 0.37 | -107.55 | 100.0 |
| GradientBoostingRegressor | train | Hatch | 82 | 53563.41 | 53543.05 | 243.28 | 0.48 | 20.37 | 100.0 |
| GradientBoostingRegressor | train | Sedan | 60 | 89743.33 | 89759.55 | 283.33 | 0.55 | -16.22 | 100.0 |
| GradientBoostingRegressor | train | Pickup | 43 | 112734.88 | 112716.42 | 307.99 | 0.4 | 18.46 | 100.0 |
| GradientBoostingRegressor | train | SUV | 101 | 110275.25 | 110274.39 | 314.65 | 0.39 | 0.85 | 100.0 |
| GradientBoostingRegressor | test | Hatch | 25 | 50932.0 | 51945.73 | 1590.27 | 3.02 | -1013.73 | 84.0 |
| GradientBoostingRegressor | test | Sedan | 13 | 112646.15 | 112370.52 | 2114.04 | 2.17 | 275.63 | 92.3 |
| GradientBoostingRegressor | test | Minivan | 2 | 61500.0 | 63777.6 | 2277.6 | 3.48 | -2277.6 | 100.0 |
