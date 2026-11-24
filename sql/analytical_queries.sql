-- Queries Analiticas de Precificacao Automotiva - Inga Multimarcas (AutoMatch)
-- Cada bloco e identificado por -- QUERY_N: <nome> para validacao automatizada.

-- QUERY_1: q1_brand_year_tier_summary
-- Agregacao por marca e faixa de ano comparando preco FIPE, preco de loja, spread e score tecnico.
SELECT
    f.brand,
    CASE
        WHEN f.model_year >= 2022 THEN 'Seminovo (2022+)'
        WHEN f.model_year >= 2016 THEN 'Intermediario (2016-2021)'
        ELSE 'Veterano (<=2015)'
    END AS faixa_ano,
    COUNT(*) AS total_veiculos,
    ROUND(AVG(f.fipe_price_brl), 2) AS preco_fipe_medio,
    ROUND(AVG(m.market_asking_price), 2) AS preco_loja_medio,
    ROUND(AVG(m.fipe_spread_pct), 2) AS spread_medio_pct,
    ROUND(AVG(m.mileage_km), 0) AS km_medio,
    ROUND(AVG(m.quality_score), 1) AS score_medio
FROM vehicles_fipe_reference f
INNER JOIN market_listings m
    ON f.vehicle_id = m.vehicle_id
GROUP BY f.brand, faixa_ano
ORDER BY f.brand ASC, preco_fipe_medio DESC;

-- QUERY_2: q2_annual_depreciation_by_brand
-- Taxa de depreciacao / variacao media anual por marca utilizando CTE e Window Function LAG.
WITH yearly_brand_stats AS (
    SELECT
        f.brand,
        f.model_year,
        COUNT(*) AS qtd_modelos,
        ROUND(AVG(f.fipe_price_brl), 2) AS fipe_medio_ano
    FROM vehicles_fipe_reference f
    GROUP BY f.brand, f.model_year
),
brand_yoy AS (
    SELECT
        brand,
        model_year,
        qtd_modelos,
        fipe_medio_ano,
        LAG(fipe_medio_ano) OVER (PARTITION BY brand ORDER BY model_year ASC) AS fipe_ano_anterior,
        LAG(model_year) OVER (PARTITION BY brand ORDER BY model_year ASC) AS ano_anterior
    FROM yearly_brand_stats
)
SELECT
    brand,
    COUNT(*) AS transicoes_anuais,
    ROUND(
        AVG(
            CASE
                WHEN fipe_ano_anterior > 0 AND (model_year - ano_anterior) > 0
                THEN ((fipe_medio_ano - fipe_ano_anterior) / fipe_ano_anterior) * 100.0 / (model_year - ano_anterior)
                ELSE NULL
            END
        ),
        2
    ) AS variacao_media_anual_pct,
    ROUND(MIN(fipe_medio_ano), 2) AS menor_fipe_medio,
    ROUND(MAX(fipe_medio_ano), 2) AS maior_fipe_medio
FROM brand_yoy
WHERE fipe_ano_anterior IS NOT NULL
GROUP BY brand
ORDER BY variacao_media_anual_pct DESC;

-- QUERY_3: q3_regional_spread_maringa_vs_others
-- Spread Loja vs FIPE por praca regional (Maringa/PR vs outras pracas) e status de laudo cautelar.
SELECT
    m.region_pr,
    CASE WHEN m.has_laudo_cautelar = 1 THEN 'Com Laudo' ELSE 'Sem Laudo' END AS status_laudo,
    COUNT(*) AS total_anuncios,
    ROUND(AVG(f.fipe_price_brl), 2) AS fipe_medio_brl,
    ROUND(AVG(m.market_asking_price), 2) AS loja_medio_brl,
    ROUND(AVG(m.fipe_spread_brl), 2) AS spread_medio_brl,
    ROUND(AVG(m.fipe_spread_pct), 2) AS spread_medio_pct,
    ROUND(AVG(m.quality_score), 1) AS score_medio
FROM market_listings m
INNER JOIN vehicles_fipe_reference f
    ON m.vehicle_id = f.vehicle_id
GROUP BY m.region_pr, status_laudo
ORDER BY spread_medio_pct DESC, total_anuncios DESC;

-- QUERY_4: q4_liquidity_ranking_and_moving_avg
-- Ranking de score/custo-beneficio por marca e media movel de preco FIPE em janela de 3 anos.
WITH ranked_vehicles AS (
    SELECT
        f.brand,
        f.model,
        f.model_year,
        m.region_pr,
        m.quality_score,
        m.fipe_spread_pct,
        f.fipe_price_brl,
        m.market_asking_price,
        DENSE_RANK() OVER (
            PARTITION BY f.brand
            ORDER BY m.quality_score DESC, m.fipe_spread_pct ASC, f.model_year DESC
        ) AS rank_qualidade_marca,
        ROUND(
            AVG(f.fipe_price_brl) OVER (
                PARTITION BY f.brand
                ORDER BY f.model_year ASC
                ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING
            ),
            2
        ) AS media_movel_fipe_3_registros
    FROM vehicles_fipe_reference f
    INNER JOIN market_listings m
        ON f.vehicle_id = m.vehicle_id
)
SELECT
    brand,
    rank_qualidade_marca,
    model,
    model_year,
    region_pr,
    quality_score,
    fipe_spread_pct,
    fipe_price_brl,
    market_asking_price,
    media_movel_fipe_3_registros
FROM ranked_vehicles
WHERE rank_qualidade_marca <= 3
ORDER BY brand ASC, rank_qualidade_marca ASC;

-- QUERY_5: q5_mileage_regime_impact
-- Impacto da quilometragem anual (km/ano) sobre quality_score e spread vs Tabela FIPE.
SELECT
    CASE
        WHEN m.km_per_year < 9000 THEN '1. Baixa (<9k km/ano)'
        WHEN m.km_per_year <= 15000 THEN '2. Moderada (9k-15k km/ano)'
        ELSE '3. Intensa (>15k km/ano)'
    END AS regime_uso,
    COUNT(*) AS qtd_veiculos,
    ROUND(AVG(m.vehicle_age), 1) AS idade_media_anos,
    ROUND(AVG(m.km_per_year), 0) AS km_anual_medio,
    ROUND(AVG(m.quality_score), 1) AS score_medio,
    ROUND(AVG(m.fipe_spread_brl), 2) AS spread_medio_brl,
    ROUND(AVG(m.fipe_spread_pct), 2) AS spread_medio_pct
FROM market_listings m
GROUP BY regime_uso
ORDER BY regime_uso ASC;

-- QUERY_6: q6_ml_prediction_error_audit
-- Auditoria das predicoes de Machine Learning por segmento e conjunto (train vs test).
SELECT
    p.model_name,
    p.split_type,
    m.body_segment,
    COUNT(*) AS n_amostras,
    ROUND(AVG(p.actual_price_brl), 2) AS preco_real_medio,
    ROUND(AVG(p.predicted_price_brl), 2) AS preco_predito_medio,
    ROUND(AVG(p.abs_error_brl), 2) AS mae_brl,
    ROUND(AVG(p.abs_pct_error) * 100.0, 2) AS mape_pct,
    ROUND(AVG(p.residual_brl), 2) AS vies_medio_brl,
    ROUND(
        100.0 * SUM(CASE WHEN p.abs_pct_error <= 0.05 THEN 1 ELSE 0 END) / COUNT(*),
        1
    ) AS acerto_dentro_5pct
FROM pricing_predictions p
INNER JOIN market_listings m
    ON p.vehicle_id = m.vehicle_id
GROUP BY p.model_name, p.split_type, m.body_segment
ORDER BY p.split_type DESC, mae_brl ASC;
