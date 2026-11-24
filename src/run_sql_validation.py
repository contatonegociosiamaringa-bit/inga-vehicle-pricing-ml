"""Valida o schema SQL e executa todas as queries analiticas em SQLite 3.x."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = PROJECT_ROOT / "sql" / "schema_postgresql.sql"
QUERIES_PATH = PROJECT_ROOT / "sql" / "analytical_queries.sql"
RAW_PATH = PROJECT_ROOT / "data" / "raw_fipe_vehicles.csv"
MARKET_PATH = PROJECT_ROOT / "data" / "vehicles_market_dataset.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "pricing_predictions.csv"
SUMMARY_JSON_PATH = PROJECT_ROOT / "reports" / "sql_validation_summary.json"
SUMMARY_MD_PATH = PROJECT_ROOT / "reports" / "sql_validation_summary.md"


def df_to_markdown_table(df: pd.DataFrame) -> str:
    """Converte DataFrame em tabela Markdown sem depender de pacotes opcionais."""
    if df.empty:
        return "_Sem registros retornados._"
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = [
        "| " + " | ".join(str(val) for val in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, sep, *rows])


def parse_named_queries(sql_text: str) -> list[tuple[str, str]]:
    """Extrai queries identificadas por '-- QUERY_N: <nome>' do arquivo SQL."""
    pattern = re.compile(r"^--\s*QUERY_\d+:\s*([a-zA-Z0-9_]+)\s*$", re.MULTILINE)
    matches = list(pattern.finditer(sql_text))
    queries: list[tuple[str, str]] = []
    for i, match in enumerate(matches):
        name = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(sql_text)
        body = sql_text[start:end].strip()
        if body:
            queries.append((name, body))
    return queries


def run_sql_validation() -> dict[str, Any]:
    """Carrega dados reais e predicoes em SQLite, aplica DDL e valida todas as queries."""
    df_raw = pd.read_csv(RAW_PATH)
    df_market = pd.read_csv(MARKET_PATH)
    df_preds = pd.read_csv(PREDICTIONS_PATH)

    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON;")

    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    conn.executescript(schema_sql)

    fipe_cols = [
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
    df_raw[fipe_cols].to_sql("vehicles_fipe_reference", conn, if_exists="append", index=False)

    market_cols = [
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
    df_market[market_cols].to_sql("market_listings", conn, if_exists="append", index=False)

    pred_cols = [
        "vehicle_id",
        "model_name",
        "split_type",
        "actual_price_brl",
        "predicted_price_brl",
        "residual_brl",
        "abs_error_brl",
        "abs_pct_error",
    ]
    df_preds[pred_cols].to_sql("pricing_predictions", conn, if_exists="append", index=False)

    queries = parse_named_queries(QUERIES_PATH.read_text(encoding="utf-8"))
    results_summary: dict[str, Any] = {}
    md_sections: list[str] = [
        "# Validacao das Queries Analiticas SQL (Inga Multimarcas)",
        "",
        f"Base validada em SQLite {sqlite3.sqlite_version} aplicando `{SCHEMA_PATH.name}` e `{QUERIES_PATH.name}`.",
        "",
    ]

    for query_name, sql_body in queries:
        df_res = pd.read_sql_query(sql_body, conn)
        results_summary[query_name] = {
            "row_count": int(len(df_res)),
            "columns": list(df_res.columns),
            "preview_rows": df_res.head(5).to_dict(orient="records"),
        }
        md_sections.append(f"## `{query_name}` ({len(df_res)} linhas)")
        md_sections.append("")
        md_sections.append(df_to_markdown_table(df_res.head(8)))
        md_sections.append("")

    conn.close()

    SUMMARY_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON_PATH.write_text(json.dumps(results_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    SUMMARY_MD_PATH.write_text("\n".join(md_sections), encoding="utf-8")
    return results_summary


if __name__ == "__main__":
    summary = run_sql_validation()
    for qname, info in summary.items():
        print(f"{qname}: {info['row_count']} linhas | colunas={info['columns']}")
