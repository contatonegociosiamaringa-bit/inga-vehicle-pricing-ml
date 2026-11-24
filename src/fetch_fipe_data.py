"""Coleta dados reais da Tabela FIPE (Parallelum API v2 com cache e retry) e gera
a base enriquecida de precificacao inspirada no catalogo AutoMatch da Inga Multimarcas.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

FIPE_BASE_URL = "https://fipe.parallelum.com.br/api/v2/cars"
BRASILAPI_BASE_URL = "https://brasilapi.com.br/api/fipe"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_PATH = DATA_DIR / "fipe_cache.json"
RAW_OUTPUT_PATH = DATA_DIR / "raw_fipe_vehicles.csv"
ENRICHED_OUTPUT_PATH = DATA_DIR / "vehicles_market_dataset.csv"

REFERENCE_YEAR = 2026
RANDOM_SEED = 42

# Marcas populares no Brasil + Dodge (referencia do catalogo AutoMatch Inga Multimarcas)
TARGET_BRANDS: dict[str, dict[str, Any]] = {
    "59": {
        "brand": "Volkswagen",
        "keywords": [
            "GOL ",
            "GOLF",
            "POLO",
            "VIRTUS",
            "T-CROSS",
            "NIVUS",
            "SAVEIRO",
            "JETTA",
            "FOX ",
            "UP!",
        ],
    },
    "23": {
        "brand": "Chevrolet",
        "keywords": [
            "ONIX",
            "SPIN",
            "TRACKER",
            "CRUZE",
            "PRISMA",
            "S10",
            "COBALT",
            "MONTANA",
            "CELTA",
        ],
    },
    "21": {
        "brand": "Fiat",
        "keywords": [
            "ARGO",
            "CRONOS",
            "MOBI",
            "STRADA",
            "TORO",
            "PULSE",
            "FASTBACK",
            "PALIO",
            "STILO",
            "UNO ",
        ],
    },
    "22": {
        "brand": "Ford",
        "keywords": [
            "KA ",
            "FIESTA",
            "ECOSPORT",
            "FOCUS",
            "RANGER",
            "FUSION",
            "TERRITORY",
        ],
    },
    "26": {
        "brand": "Hyundai",
        "keywords": [
            "HB20",
            "CRETA",
            "TUCSON",
            "IX35",
            "ELANTRA",
            "AZERA",
            "SANTA FE",
        ],
    },
    "56": {
        "brand": "Toyota",
        "keywords": [
            "COROLLA",
            "YARIS",
            "HILUX",
            "ETIOS",
            "SW4",
            "RAV4",
        ],
    },
    "25": {
        "brand": "Honda",
        "keywords": [
            "CIVIC",
            "HR-V",
            "CITY",
            "FIT ",
            "WR-V",
            "CR-V",
        ],
    },
    "29": {
        "brand": "Jeep",
        "keywords": [
            "RENEGADE",
            "COMPASS",
            "COMMANDER",
            "WRANGLER",
            "CHEROKEE",
        ],
    },
    "17": {
        "brand": "Dodge",
        "keywords": [
            "JOURNEY",
            "RAM ",
        ],
    },
}

BODY_SEGMENT_RULES: list[tuple[tuple[str, ...], str]] = [
    (("STRADA", "TORO", "SAVEIRO", "MONTANA", "S10", "RANGER", "HILUX", "RAM"), "Pickup"),
    (
        (
            "T-CROSS",
            "NIVUS",
            "TRACKER",
            "PULSE",
            "FASTBACK",
            "ECOSPORT",
            "TERRITORY",
            "CRETA",
            "TUCSON",
            "IX35",
            "SANTA",
            "COROLLA CROSS",
            "SW4",
            "RAV4",
            "HR-V",
            "WR-V",
            "CR-V",
            "RENEGADE",
            "COMPASS",
            "COMMANDER",
            "WRANGLER",
            "CHEROKEE",
            "JOURNEY",
        ),
        "SUV",
    ),
    (("SPIN",), "Minivan"),
    (
        (
            "VIRTUS",
            "JETTA",
            "ONIX PLUS",
            "PRISMA",
            "CRUZE",
            "COBALT",
            "CRONOS",
            "FUSION",
            "HB20S",
            "ELANTRA",
            "AZERA",
            "COROLLA",
            "CIVIC",
            "CITY",
        ),
        "Sedan",
    ),
]

BRAND_LIQUIDITY_TIER: dict[str, str] = {
    "Toyota": "Alta",
    "Honda": "Alta",
    "Volkswagen": "Alta",
    "Chevrolet": "Alta",
    "Hyundai": "Alta",
    "Fiat": "Alta",
    "Jeep": "Media",
    "Ford": "Media",
    "Dodge": "Baixa",
}

REGION_WEIGHTS: dict[str, float] = {
    "Maringa - PR": 0.38,
    "Curitiba - PR": 0.22,
    "Londrina - PR": 0.18,
    "Cascavel - PR": 0.12,
    "Outras Pracas (SP/SC)": 0.10,
}

REGION_SPREAD_FACTOR: dict[str, float] = {
    "Maringa - PR": 1.018,
    "Curitiba - PR": 1.005,
    "Londrina - PR": 1.010,
    "Cascavel - PR": 0.996,
    "Outras Pracas (SP/SC)": 0.985,
}


def parse_brl_price(price_str: str | float | int) -> float:
    """Converte string monetaria BRL ('R$ 63.000,00') em float."""
    if isinstance(price_str, (int, float)):
        return float(price_str)
    cleaned = str(price_str).replace("R$", "").strip().replace(".", "").replace(",", ".")
    return float(cleaned)


def infer_model_family(model_name: str) -> str:
    """Extrai familia do modelo a partir do nome oficial FIPE."""
    upper = model_name.upper().strip()
    multi_word = [
        "ONIX PLUS",
        "COROLLA CROSS",
        "SANTA FE",
        "GRAND SIENA",
        "T-CROSS",
        "HR-V",
        "WR-V",
        "CR-V",
    ]
    for token in multi_word:
        if upper.startswith(token):
            return token.title()
    first = re.split(r"[\s/]+", upper)[0]
    return first.title()


def infer_body_segment(model_name: str) -> str:
    """Classifica segmento da carroceria pelo nome do modelo."""
    upper = model_name.upper()
    for keywords, segment in BODY_SEGMENT_RULES:
        if any(kw in upper for kw in keywords):
            return segment
    return "Hatch"


def infer_transmission(model_name: str) -> str:
    """Detecta cambio automatico/CVT ou manual na descricao FIPE."""
    upper = model_name.upper()
    auto_tokens = ("AUT", "CVT", "DSG", "TIPTRONIC", "POWERSHIFT", "AT ", " AT", "S-TRONIC")
    if any(tok in upper for tok in auto_tokens):
        return "Automatico"
    return "Manual"


def infer_engine_liters(model_name: str, segment: str) -> float:
    """Extrai cilindrada (litros) do nome FIPE, com fallback por segmento."""
    match = re.search(r"\b([1-6]\.[0-9])\b", model_name)
    if match:
        return float(match.group(1))
    defaults = {"Hatch": 1.0, "Sedan": 1.6, "SUV": 1.8, "Pickup": 2.4, "Minivan": 1.8}
    return defaults.get(segment, 1.4)


class FipeClient:
    """Cliente HTTP com cache local em disco, rate limiting e retry exponencial."""

    def __init__(self, cache_path: Path = CACHE_PATH, request_delay: float = 0.06) -> None:
        self.cache_path = cache_path
        self.request_delay = request_delay
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "IngaVehiclePricingML/1.0"})
        self.cache: dict[str, Any] = self._load_cache()
        self._dirty = False

    def _load_cache(self) -> dict[str, Any]:
        if self.cache_path.exists():
            try:
                return json.loads(self.cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("Cache corrompido em %s; iniciando novo cache.", self.cache_path)
        return {}

    def save_cache(self) -> None:
        if not self._dirty:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self.cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._dirty = False

    def get_json(self, url: str, max_retries: int = 5) -> Any:
        if url in self.cache:
            return self.cache[url]

        backoff = 1.0
        for attempt in range(1, max_retries + 1):
            time.sleep(self.request_delay)
            try:
                resp = self.session.get(url, timeout=20)
                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", backoff * 2))
                    logger.warning("Rate limit (429) em %s. Aguardando %.1fs...", url, retry_after)
                    time.sleep(retry_after)
                    backoff = min(backoff * 2, 16.0)
                    continue
                if resp.status_code >= 500:
                    logger.warning("Erro %s em %s (tentativa %d/%d).", resp.status_code, url, attempt, max_retries)
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 16.0)
                    continue
                resp.raise_for_status()
                payload = resp.json()
                self.cache[url] = payload
                self._dirty = True
                return payload
            except requests.RequestException as exc:
                logger.warning("Falha de rede (%s) na tentativa %d/%d: %s", url, attempt, max_retries, exc)
                if attempt == max_retries:
                    raise
                time.sleep(backoff)
                backoff = min(backoff * 2, 16.0)

        raise RuntimeError(f"Nao foi possivel obter resposta valida de {url}")


def select_representative_models(
    models: list[dict[str, Any]],
    keywords: list[str],
    max_models: int = 12,
) -> list[dict[str, Any]]:
    """Seleciona modelos representativos priorizando correspondencia exata de inicio de nome."""
    selected: list[dict[str, Any]] = []
    seen_codes: set[str] = set()

    # Passagem 1: prioriza modelos cujo nome comeca com a palavra-chave e escolhe versao mediana
    for kw in keywords:
        kw_clean = kw.strip()
        starts_matches = [
            m for m in models
            if m["name"].upper().startswith(kw_clean) and str(m["code"]) not in seen_codes
        ]
        pool = starts_matches or [
            m for m in models
            if kw in m["name"].upper() and str(m["code"]) not in seen_codes
        ]
        if pool:
            pick = pool[len(pool) // 2]
            selected.append(pick)
            seen_codes.add(str(pick["code"]))
        if len(selected) >= max_models:
            return selected

    # Passagem 2: segunda variacao (ex: cambio automatico ou versao recente) se houver espaco
    for kw in keywords:
        kw_clean = kw.strip()
        pool = [
            m for m in models
            if m["name"].upper().startswith(kw_clean) and str(m["code"]) not in seen_codes
        ] or [
            m for m in models
            if kw in m["name"].upper() and str(m["code"]) not in seen_codes
        ]
        if pool:
            pick = pool[-1]
            selected.append(pick)
            seen_codes.add(str(pick["code"]))
        if len(selected) >= max_models:
            break

    return selected[:max_models]


def select_representative_years(
    years: list[dict[str, Any]],
    max_years: int = 6,
) -> list[dict[str, Any]]:
    """Filtra anos validos (exclui 32000 Zero KM; aceita todos combustiveis 1-9) e amostra uniformemente."""
    valid = [
        y
        for y in years
        if re.match(r"^(199\d|20[0-2]\d)-\d+$", str(y.get("code", "")))
    ]
    if len(valid) <= max_years:
        return valid
    indices = np.linspace(0, len(valid) - 1, num=max_years, dtype=int)
    seen_idx: list[int] = []
    for idx in indices:
        if int(idx) not in seen_idx:
            seen_idx.append(int(idx))
    return [valid[i] for i in seen_idx]


def fetch_raw_fipe_dataset(
    target_min_records: int = 300,
    max_models_per_brand: int = 12,
    max_years_per_model: int = 6,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Coleta registros reais distintos da Tabela FIPE e salva raw_fipe_vehicles.csv."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_OUTPUT_PATH.exists() and not force_refresh:
        existing = pd.read_csv(RAW_OUTPUT_PATH)
        if len(existing) >= target_min_records:
            logger.info("Usando cache CSV existente em %s (%d linhas).", RAW_OUTPUT_PATH, len(existing))
            return existing

    client = FipeClient()
    records: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, int, str]] = set()

    try:
        for brand_code, cfg in TARGET_BRANDS.items():
            brand_label = cfg["brand"]
            models_url = f"{FIPE_BASE_URL}/brands/{brand_code}/models"
            models = client.get_json(models_url)
            chosen_models = select_representative_models(
                models,
                cfg["keywords"],
                max_models=max_models_per_brand,
            )
            logger.info("Marca %s: %d modelos selecionados de %d.", brand_label, len(chosen_models), len(models))

            for model_item in chosen_models:
                model_code = str(model_item["code"])
                years_url = f"{FIPE_BASE_URL}/brands/{brand_code}/models/{model_code}/years"
                years = client.get_json(years_url)
                chosen_years = select_representative_years(years, max_years=max_years_per_model)

                for year_item in chosen_years:
                    year_code = str(year_item["code"])
                    detail_url = f"{FIPE_BASE_URL}/brands/{brand_code}/models/{model_code}/years/{year_code}"
                    detail = client.get_json(detail_url)

                    fipe_code = str(detail.get("codeFipe", "")).strip()
                    model_year = int(detail.get("modelYear", 0))
                    fuel_type = str(detail.get("fuel", "Flex")).strip()
                    dedup_key = (fipe_code, model_year, fuel_type)
                    if not fipe_code or model_year < 1995 or model_year > REFERENCE_YEAR + 1 or dedup_key in seen_keys:
                        continue

                    seen_keys.add(dedup_key)
                    fipe_price = parse_brl_price(detail["price"])
                    model_name = str(detail["model"]).strip()

                    records.append(
                        {
                            "vehicle_id": f"FIPE-{fipe_code.replace('-', '')}-{model_year}-{detail.get('fuelAcronym', 'F')}",
                            "fipe_code": fipe_code,
                            "brand_raw": str(detail.get("brand", brand_label)).strip(),
                            "brand": brand_label,
                            "model": model_name,
                            "model_family": infer_model_family(model_name),
                            "model_year": model_year,
                            "fuel_type": fuel_type,
                            "fuel_acronym": str(detail.get("fuelAcronym", "")).strip(),
                            "fipe_price_brl": round(fipe_price, 2),
                            "reference_month": str(detail.get("referenceMonth", "")).strip(),
                            "source_api": "fipe.parallelum.com.br/api/v2",
                        }
                    )
            client.save_cache()
    finally:
        client.save_cache()

    df_raw = pd.DataFrame(records).drop_duplicates(subset=["fipe_code", "model_year", "fuel_type"]).reset_index(drop=True)
    df_raw.to_csv(RAW_OUTPUT_PATH, index=False, encoding="utf-8")
    logger.info("Base FIPE bruta salva em %s (%d registros unicos).", RAW_OUTPUT_PATH, len(df_raw))
    return df_raw


def enrich_market_dataset(df_raw: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Enriquece dados reais FIPE com atributos operacionais do AutoMatch Inga Multimarcas."""
    rng = np.random.default_rng(seed)
    df = df_raw.copy()

    # Atributos mecanicos extraidos do catalogo FIPE
    df["body_segment"] = df["model"].apply(infer_body_segment)
    df["transmission"] = df["model"].apply(infer_transmission)
    df["engine_liters"] = [
        infer_engine_liters(m, seg) for m, seg in zip(df["model"], df["body_segment"])
    ]

    # Idade do veiculo e ano de fabricacao realista (model_year ou model_year - 1)
    fab_offset = rng.choice([0, 1], size=len(df), p=[0.62, 0.38])
    df["year_fabrication"] = df["model_year"] - fab_offset
    df["vehicle_age"] = np.maximum(0, REFERENCE_YEAR - df["model_year"]).astype(int)

    # Quilometragem realista no mercado brasileiro (~11.500 a 14.500 km/ano + ruido por segmento)
    effective_age = np.maximum(1, df["vehicle_age"])
    annual_km_base = np.where(
        df["body_segment"] == "Pickup",
        rng.normal(16500, 4200, size=len(df)),
        np.where(
            df["body_segment"] == "Minivan",
            rng.normal(15500, 4500, size=len(df)),
            rng.normal(12400, 3400, size=len(df)),
        ),
    )
    annual_km_base = np.clip(annual_km_base, 4500, 29000)
    raw_mileage = effective_age * annual_km_base + rng.normal(0, 3500, size=len(df))
    # Carros 2026/2025 podem ser seminovos com baixa km
    mileage_floor = np.where(df["vehicle_age"] <= 1, 2500, 12000)
    df["mileage_km"] = np.clip(np.round(raw_mileage, -2), mileage_floor, 420000).astype(int)
    df["km_per_year"] = np.round(df["mileage_km"] / effective_age, 1)

    # Praca regional (Maringa/PR vs outras pracas)
    regions = list(REGION_WEIGHTS.keys())
    probs = list(REGION_WEIGHTS.values())
    df["region_pr"] = rng.choice(regions, size=len(df), p=probs)

    # Laudo cautelar (Modulo 1 AutoMatch)
    laudo_prob = np.where(df["vehicle_age"] <= 8, 0.84, 0.66)
    df["has_laudo_cautelar"] = (rng.random(len(df)) < laudo_prob).astype(int)

    # Quality Score (0-100) baseado na formula de app.js:
    # base 40 + ate 25 pts por preservacao de km/ano + 20 pts laudo + 10 pts doc/historico + variacao de vistoria
    km_pts = np.round(25.0 * np.clip(2.0 - df["mileage_km"] / (effective_age * 12000.0), 0.0, 1.0))
    laudo_pts = np.where(df["has_laudo_cautelar"] == 1, 20.0, 8.0)
    doc_pts = np.where(df["vehicle_age"] <= 12, 10.0, 7.0)
    inspection_delta = rng.normal(0.0, 2.2, size=len(df))
    df["quality_score"] = np.clip(np.round(40.0 + km_pts + laudo_pts + doc_pts + inspection_delta), 52, 98).astype(int)

    df["liquidity_tier"] = df["brand"].map(BRAND_LIQUIDITY_TIER).fillna("Media")

    # Preco pedido de mercado (market_asking_price) com efeitos nao-lineares realistas vs FIPE:
    # 1) Fator de score tecnico (acima de 82 valoriza; abaixo de 78 sofre desagio acentuado)
    score_factor = 1.0 + 0.0022 * (df["quality_score"] - 82) - 0.025 * (df["quality_score"] < 75)
    # 2) Penalidade progressiva para alta quilometragem anual (>16.000 km/ano) e bonus para baixa (<9.000 km/ano)
    excess_km = np.maximum(0.0, df["km_per_year"] - 15000.0) / 10000.0
    low_km_bonus = np.where(df["km_per_year"] < 9000.0, 0.022, 0.0)
    mileage_factor = 1.0 - 0.048 * excess_km + low_km_bonus
    # 3) Premio regional (Maringa/PR liquidez forte em seminovos) e segmento
    region_factor = df["region_pr"].map(REGION_SPREAD_FACTOR).fillna(1.0)
    segment_factor = df["body_segment"].map(
        {"SUV": 1.015, "Pickup": 1.020, "Sedan": 1.000, "Hatch": 1.004, "Minivan": 0.965}
    ).fillna(1.0)
    liquidity_factor = df["liquidity_tier"].map({"Alta": 1.008, "Media": 0.996, "Baixa": 0.975}).fillna(1.0)
    # 4) Veiculos muito antigos (>15 anos) com bom score de conservacao vs sem laudo
    vintage_premium = np.where((df["vehicle_age"] >= 15) & (df["quality_score"] >= 84), 0.038, 0.0)
    no_laudo_discount = np.where(df["has_laudo_cautelar"] == 0, -0.028, 0.0)
    # 5) Ruido estocastico de negociacao (~1.4%)
    market_noise = rng.normal(0.0, 0.014, size=len(df))

    multiplier = (
        score_factor
        * mileage_factor
        * region_factor
        * segment_factor
        * liquidity_factor
        + vintage_premium
        + no_laudo_discount
        + market_noise
    )
    multiplier = np.clip(multiplier, 0.72, 1.18)

    df["market_asking_price"] = np.round(df["fipe_price_brl"] * multiplier, -2)
    df["fipe_spread_brl"] = np.round(df["market_asking_price"] - df["fipe_price_brl"], 2)
    df["fipe_spread_pct"] = np.round((df["market_asking_price"] / df["fipe_price_brl"] - 1.0) * 100.0, 2)
    df["data_origin"] = "FIPE_REAL_API_ENRICHED"

    ordered_cols = [
        "vehicle_id",
        "fipe_code",
        "brand",
        "model",
        "model_family",
        "body_segment",
        "model_year",
        "year_fabrication",
        "vehicle_age",
        "fuel_type",
        "transmission",
        "engine_liters",
        "mileage_km",
        "km_per_year",
        "region_pr",
        "has_laudo_cautelar",
        "quality_score",
        "liquidity_tier",
        "fipe_price_brl",
        "market_asking_price",
        "fipe_spread_brl",
        "fipe_spread_pct",
        "reference_month",
        "data_origin",
    ]
    df_out = df[ordered_cols].reset_index(drop=True)
    df_out.to_csv(ENRICHED_OUTPUT_PATH, index=False, encoding="utf-8")
    logger.info("Dataset enriquecido salvo em %s (%d linhas, %d colunas).", ENRICHED_OUTPUT_PATH, *df_out.shape)
    return df_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Coleta FIPE e gera dataset de precificacao de veiculos.")
    parser.add_argument("--force-refresh", action="store_true", help="Ignora CSV existente e consulta cache/API novamente.")
    parser.add_argument("--min-records", type=int, default=300, help="Meta minima de registros FIPE distintos.")
    args = parser.parse_args()

    df_raw = fetch_raw_fipe_dataset(
        target_min_records=args.min_records,
        force_refresh=args.force_refresh,
    )
    df_market = enrich_market_dataset(df_raw)
    print(f"FIPE raw rows: {len(df_raw)} | Market dataset rows: {len(df_market)} | Brands: {df_market['brand'].nunique()}")


if __name__ == "__main__":
    main()
