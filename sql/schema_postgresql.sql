-- Schema PostgreSQL para Precificacao de Veiculos - Inga Multimarcas (AutoMatch Analytics)
-- Compativel com PostgreSQL 14+ e validado em runtime via SQLite 3.x

CREATE TABLE IF NOT EXISTS vehicles_fipe_reference (
    vehicle_id TEXT PRIMARY KEY,
    fipe_code TEXT NOT NULL,
    brand_raw TEXT NOT NULL,
    brand TEXT NOT NULL,
    model TEXT NOT NULL,
    model_family TEXT NOT NULL,
    model_year INTEGER NOT NULL CHECK (model_year BETWEEN 1990 AND 2030),
    fuel_type TEXT NOT NULL,
    fuel_acronym TEXT,
    fipe_price_brl NUMERIC(12, 2) NOT NULL CHECK (fipe_price_brl > 0),
    reference_month TEXT NOT NULL,
    source_api TEXT NOT NULL DEFAULT 'fipe.parallelum.com.br/api/v2',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS market_listings (
    listing_id INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_id TEXT NOT NULL UNIQUE REFERENCES vehicles_fipe_reference(vehicle_id) ON DELETE CASCADE,
    body_segment TEXT NOT NULL CHECK (body_segment IN ('Hatch', 'Sedan', 'SUV', 'Pickup', 'Minivan')),
    year_fabrication INTEGER NOT NULL CHECK (year_fabrication BETWEEN 1989 AND 2030),
    vehicle_age INTEGER NOT NULL CHECK (vehicle_age >= 0),
    transmission TEXT NOT NULL CHECK (transmission IN ('Manual', 'Automatico')),
    engine_liters NUMERIC(3, 1) NOT NULL CHECK (engine_liters > 0),
    mileage_km INTEGER NOT NULL CHECK (mileage_km >= 0),
    km_per_year NUMERIC(10, 1) NOT NULL CHECK (km_per_year >= 0),
    region_pr TEXT NOT NULL,
    has_laudo_cautelar INTEGER NOT NULL CHECK (has_laudo_cautelar IN (0, 1)),
    quality_score INTEGER NOT NULL CHECK (quality_score BETWEEN 0 AND 100),
    liquidity_tier TEXT NOT NULL CHECK (liquidity_tier IN ('Alta', 'Media', 'Baixa')),
    market_asking_price NUMERIC(12, 2) NOT NULL CHECK (market_asking_price > 0),
    fipe_spread_brl NUMERIC(12, 2) NOT NULL,
    fipe_spread_pct NUMERIC(6, 2) NOT NULL,
    data_origin TEXT NOT NULL DEFAULT 'FIPE_REAL_API_ENRICHED',
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS pricing_predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_id TEXT NOT NULL REFERENCES vehicles_fipe_reference(vehicle_id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    split_type TEXT NOT NULL CHECK (split_type IN ('train', 'test')),
    actual_price_brl NUMERIC(12, 2) NOT NULL CHECK (actual_price_brl > 0),
    predicted_price_brl NUMERIC(12, 2) NOT NULL CHECK (predicted_price_brl > 0),
    residual_brl NUMERIC(12, 2) NOT NULL,
    abs_error_brl NUMERIC(12, 2) NOT NULL CHECK (abs_error_brl >= 0),
    abs_pct_error NUMERIC(8, 4) NOT NULL CHECK (abs_pct_error >= 0),
    predicted_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_fipe_brand_year
    ON vehicles_fipe_reference (brand, model_year);

CREATE INDEX IF NOT EXISTS idx_market_region_score
    ON market_listings (region_pr, quality_score);

CREATE INDEX IF NOT EXISTS idx_pred_model_split
    ON pricing_predictions (model_name, split_type);
