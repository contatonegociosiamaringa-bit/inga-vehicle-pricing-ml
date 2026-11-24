"""Testes automatizados do pipeline de dados, ML, metricas e validacao SQL."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.run_sql_validation import parse_named_queries
from src.train_pricing_model import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    build_preprocessor,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw_fipe_vehicles.csv"
MARKET_PATH = PROJECT_ROOT / "data" / "vehicles_market_dataset.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "pricing_predictions.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "vehicle_price_model.joblib"
METRICS_PATH = PROJECT_ROOT / "reports" / "model_metrics.json"
SCHEMA_PATH = PROJECT_ROOT / "sql" / "schema_postgresql.sql"
QUERIES_PATH = PROJECT_ROOT / "sql" / "analytical_queries.sql"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def test_dataset_schema_and_quality() -> None:
    assert RAW_PATH.exists(), "raw_fipe_vehicles.csv nao encontrado."
    assert MARKET_PATH.exists(), "vehicles_market_dataset.csv nao encontrado."

    df_raw = pd.read_csv(RAW_PATH)
    df_market = pd.read_csv(MARKET_PATH)

    assert len(df_raw) >= 250, f"Esperado >= 250 registros FIPE reais, obtido {len(df_raw)}"
    assert len(df_market) == len(df_raw)
    assert df_raw["vehicle_id"].is_unique
    assert df_market["vehicle_id"].is_unique

    expected_cols = set(FEATURE_COLUMNS + [TARGET_COLUMN, "vehicle_id", "fipe_code", "fipe_spread_brl", "fipe_spread_pct"])
    assert expected_cols.issubset(df_market.columns)
    assert df_market[FEATURE_COLUMNS + [TARGET_COLUMN]].isna().sum().sum() == 0

    assert (df_market["fipe_price_brl"] > 0).all()
    assert (df_market["market_asking_price"] > 0).all()
    assert df_market["quality_score"].between(0, 100).all()
    assert df_market["model_year"].between(1990, 2030).all()
    assert df_market["brand"].nunique() >= 6


def test_feature_preprocessor_transforms_cleanly() -> None:
    df_market = pd.read_csv(MARKET_PATH)
    X = df_market[FEATURE_COLUMNS]
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)

    assert transformed.shape[0] == len(df_market)
    assert transformed.shape[1] > len(NUMERIC_FEATURES)
    assert np.isfinite(transformed).all()


def test_saved_model_inference_and_metrics() -> None:
    assert MODEL_PATH.exists(), "Modelo treinado nao encontrado."
    assert METRICS_PATH.exists(), "Arquivo model_metrics.json nao encontrado."

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    assert len(metrics["models"]) >= 3
    best_name = metrics["best_model"]
    assert best_name in metrics["models"]

    best_holdout = metrics["best_model_holdout_metrics"]
    assert best_holdout["r2"] >= 0.90
    assert best_holdout["rmse_brl"] > 0
    assert best_holdout["mae_brl"] > 0
    assert best_holdout["mape_pct"] < 15.0

    model = joblib.load(MODEL_PATH)
    sample = pd.DataFrame(
        [
            {
                "fipe_price_brl": 31463.0,
                "model_year": 2005,
                "vehicle_age": 21,
                "mileage_km": 236533,
                "km_per_year": 11263.5,
                "quality_score": 86,
                "engine_liters": 1.6,
                "has_laudo_cautelar": 1,
                "brand": "Volkswagen",
                "body_segment": "Hatch",
                "fuel_type": "Flex",
                "transmission": "Manual",
                "region_pr": "Maringa - PR",
                "liquidity_tier": "Alta",
            },
            {
                "fipe_price_brl": 63000.0,
                "model_year": 2021,
                "vehicle_age": 5,
                "mileage_km": 55000,
                "km_per_year": 11000.0,
                "quality_score": 83,
                "engine_liters": 1.0,
                "has_laudo_cautelar": 1,
                "brand": "Chevrolet",
                "body_segment": "Hatch",
                "fuel_type": "Flex",
                "transmission": "Manual",
                "region_pr": "Maringa - PR",
                "liquidity_tier": "Alta",
            },
        ]
    )
    preds = model.predict(sample)
    assert preds.shape == (2,)
    assert np.all(np.isfinite(preds))
    assert np.all(preds > 10000.0)


def test_required_figures_and_sql_queries() -> None:
    expected_figs = [
        "price_distribution.png",
        "depreciation_by_age.png",
        "actual_vs_predicted.png",
        "feature_importance.png",
    ]
    for fig_name in expected_figs:
        fig_path = FIGURES_DIR / fig_name
        assert fig_path.exists() and fig_path.stat().st_size > 10_000, f"Figura ausente ou vazia: {fig_name}"

    df_raw = pd.read_csv(RAW_PATH)
    df_market = pd.read_csv(MARKET_PATH)
    df_preds = pd.read_csv(PREDICTIONS_PATH)

    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))

    df_raw[
        [
            "vehicle_id",
            "fipe_code",
            "brand_raw",
            "brand",
            "model",
            "model_family",
            "model_year",
            "fuel_type",
            "fuel_acronym",
            "fipe_price_brl",
            "reference_month",
            "source_api",
        ]
    ].to_sql("vehicles_fipe_reference", conn, if_exists="append", index=False)

    df_market[
        [
            "vehicle_id",
            "body_segment",
            "year_fabrication",
            "vehicle_age",
            "transmission",
            "engine_liters",
            "mileage_km",
            "km_per_year",
            "region_pr",
            "has_laudo_cautelar",
            "quality_score",
            "liquidity_tier",
            "market_asking_price",
            "fipe_spread_brl",
            "fipe_spread_pct",
            "data_origin",
        ]
    ].to_sql("market_listings", conn, if_exists="append", index=False)

    df_preds[
        [
            "vehicle_id",
            "model_name",
            "split_type",
            "actual_price_brl",
            "predicted_price_brl",
            "residual_brl",
            "abs_error_brl",
            "abs_pct_error",
        ]
    ].to_sql("pricing_predictions", conn, if_exists="append", index=False)

    queries = parse_named_queries(QUERIES_PATH.read_text(encoding="utf-8"))
    assert len(queries) >= 6
    for qname, sql_body in queries:
        res = pd.read_sql_query(sql_body, conn)
        assert len(res) >= 1, f"Query {qname} retornou 0 linhas."
    conn.close()
