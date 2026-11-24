"""Pipeline de Machine Learning para precificacao de veiculos (Inga Multimarcas / FIPE).

Compara multiplas familias de modelos com validacao cruzada K-Fold e avaliacao holdout,
salva o melhor pipeline em models/vehicle_price_model.joblib, registra metricas exatas
em reports/model_metrics.json e gera visualizacoes analiticas em reports/figures/.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "vehicles_market_dataset.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "pricing_predictions.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "vehicle_price_model.joblib"
METRICS_PATH = PROJECT_ROOT / "reports" / "model_metrics.json"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

NUMERIC_FEATURES = [
    "fipe_price_brl",
    "model_year",
    "vehicle_age",
    "mileage_km",
    "km_per_year",
    "quality_score",
    "engine_liters",
    "has_laudo_cautelar",
]

CATEGORICAL_FEATURES = [
    "brand",
    "body_segment",
    "fuel_type",
    "transmission",
    "region_pr",
    "liquidity_tier",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "market_asking_price"
RANDOM_SEED = 42


def format_brl_axis(value: float, _pos: int) -> str:
    """Formata eixo numerico em milhares de reais (R$ k)."""
    return f"R$ {value / 1000:,.0f}k".replace(",", ".")


def build_preprocessor() -> ColumnTransformer:
    """Constroi ColumnTransformer padronizado para features numericas e categoricas."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_candidate_models(seed: int = RANDOM_SEED) -> dict[str, Any]:
    """Define candidatos de regressao para comparacao justa no mesmo pre-processador."""
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=5.0),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=300,
            max_depth=14,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            n_estimators=260,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.85,
            random_state=seed,
        ),
    }


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula RMSE, MAE, R2 e MAPE (%) sobre predicoes."""
    return {
        "rmse_brl": round(float(root_mean_squared_error(y_true, y_pred)), 2),
        "mae_brl": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "r2": round(float(r2_score(y_true, y_pred)), 5),
        "mape_pct": round(float(mean_absolute_percentage_error(y_true, y_pred) * 100.0), 3),
    }


def configure_plot_style() -> None:
    """Define estilo visual limpo e legivel para relatorios."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def generate_figures(
    df: pd.DataFrame,
    y_test: pd.Series,
    y_pred_test: np.ndarray,
    best_model_name: str,
    best_pipeline: Pipeline,
    X_test: pd.DataFrame,
) -> dict[str, float]:
    """Gera os 4 graficos obrigatorios em reports/figures/ e retorna importancias por permutacao."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    configure_plot_style()

    # 1. price_distribution.png
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    sns.histplot(df["fipe_price_brl"], kde=True, color="#2b5c8f", alpha=0.45, label="Tabela FIPE", ax=axes[0])
    sns.histplot(df["market_asking_price"], kde=True, color="#e07a5f", alpha=0.45, label="Preco de Loja", ax=axes[0])
    axes[0].xaxis.set_major_formatter(mtick.FuncFormatter(format_brl_axis))
    axes[0].set_title("Distribuicao: Preco FIPE vs Preco de Mercado")
    axes[0].set_xlabel("Valor (BRL)")
    axes[0].set_ylabel("Frequencia")
    axes[0].legend()

    sns.histplot(df["fipe_spread_pct"], kde=True, bins=28, color="#3d405b", alpha=0.65, ax=axes[1])
    mean_spread = float(df["fipe_spread_pct"].mean())
    median_spread = float(df["fipe_spread_pct"].median())
    axes[1].axvline(0.0, color="#999999", linestyle="--", linewidth=1.2, label="Paridade FIPE (0%)")
    axes[1].axvline(mean_spread, color="#d62828", linestyle="-", linewidth=1.8, label=f"Media: {mean_spread:+.2f}%")
    axes[1].axvline(median_spread, color="#2a9d8f", linestyle=":", linewidth=1.8, label=f"Mediana: {median_spread:+.2f}%")
    axes[1].set_title("Distribuicao do Spread Percentual (Loja vs FIPE)")
    axes[1].set_xlabel("Spread vs FIPE (%)")
    axes[1].set_ylabel("Quantidade de Veiculos")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "price_distribution.png", bbox_inches="tight")
    plt.close(fig)

    # 2. depreciation_by_age.png
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    top_segments = df["body_segment"].value_counts().index.tolist()
    age_grouped = (
        df[df["body_segment"].isin(top_segments)]
        .groupby(["vehicle_age", "body_segment"], as_index=False)["fipe_price_brl"]
        .mean()
    )
    sns.lineplot(
        data=age_grouped,
        x="vehicle_age",
        y="fipe_price_brl",
        hue="body_segment",
        marker="o",
        linewidth=2.0,
        ax=axes[0],
    )
    axes[0].yaxis.set_major_formatter(mtick.FuncFormatter(format_brl_axis))
    axes[0].set_title("Curva de Depreciacao FIPE por Idade e Segmento")
    axes[0].set_xlabel("Idade do Veiculo (anos)")
    axes[0].set_ylabel("Preco FIPE Medio (BRL)")
    axes[0].legend(title="Segmento")

    scatter = axes[1].scatter(
        df["km_per_year"],
        df["fipe_spread_pct"],
        c=df["quality_score"],
        cmap="viridis",
        alpha=0.75,
        edgecolors="white",
        linewidth=0.5,
        s=48,
    )
    axes[1].axhline(0.0, color="#666666", linestyle="--", linewidth=1.1)
    cbar = fig.colorbar(scatter, ax=axes[1])
    cbar.set_label("Quality Score AutoMatch (0-100)")
    axes[1].set_title("Impacto do Uso Anual e Score Tecnico no Spread FIPE")
    axes[1].set_xlabel("Quilometragem Media Anual (km/ano)")
    axes[1].set_ylabel("Spread vs FIPE (%)")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "depreciation_by_age.png", bbox_inches="tight")
    plt.close(fig)

    # 3. actual_vs_predicted.png
    residuals = y_test.to_numpy() - y_pred_test
    metrics_test = evaluate_predictions(y_test.to_numpy(), y_pred_test)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))
    axes[0].scatter(y_test, y_pred_test, alpha=0.75, color="#1d3557", edgecolors="white", s=52)
    min_val = min(float(y_test.min()), float(y_pred_test.min()))
    max_val = max(float(y_test.max()), float(y_pred_test.max()))
    axes[0].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.8, label="Predicao Perfeita (y = x)")
    axes[0].xaxis.set_major_formatter(mtick.FuncFormatter(format_brl_axis))
    axes[0].yaxis.set_major_formatter(mtick.FuncFormatter(format_brl_axis))
    axes[0].set_title(
        f"Real vs Predito ({best_model_name})\n"
        f"R2 = {metrics_test['r2']:.4f} | RMSE = R$ {metrics_test['rmse_brl']:,.0f}"
    )
    axes[0].set_xlabel("Preco Real de Mercado (BRL)")
    axes[0].set_ylabel("Preco Predito pelo Modelo (BRL)")
    axes[0].legend()

    sns.histplot(residuals, kde=True, bins=22, color="#457b9d", alpha=0.65, ax=axes[1])
    axes[1].axvline(0.0, color="#d62828", linestyle="--", linewidth=1.6, label="Erro Zero")
    axes[1].xaxis.set_major_formatter(mtick.FuncFormatter(format_brl_axis))
    axes[1].set_title(f"Distribuicao dos Residuos no Teste (MAE = R$ {metrics_test['mae_brl']:,.0f})")
    axes[1].set_xlabel("Residuo: Real - Predito (BRL)")
    axes[1].set_ylabel("Frequencia")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "actual_vs_predicted.png", bbox_inches="tight")
    plt.close(fig)

    # 4. feature_importance.png (Permutation Importance nas colunas originais do Pipeline)
    perm = permutation_importance(
        best_pipeline,
        X_test,
        y_test,
        n_repeats=20,
        random_state=RANDOM_SEED,
        scoring="neg_root_mean_squared_error",
    )
    imp_df = (
        pd.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    imp_dict = {
        row["feature"]: round(float(row["importance_mean"]), 4)
        for _, row in imp_df.iterrows()
    }

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.5))
    top_all = imp_df.head(10).sort_values("importance_mean", ascending=True)
    axes[0].barh(
        top_all["feature"],
        top_all["importance_mean"],
        xerr=top_all["importance_std"],
        color="#264653",
        alpha=0.88,
    )
    axes[0].set_title(f"Importancia Global por Permutacao ({best_model_name})")
    axes[0].set_xlabel("Impacto Medio no RMSE ao Permutar (BRL)")

    non_fipe = imp_df[imp_df["feature"] != "fipe_price_brl"].head(10).sort_values("importance_mean", ascending=True)
    axes[1].barh(
        non_fipe["feature"],
        non_fipe["importance_mean"],
        xerr=non_fipe["importance_std"],
        color="#e76f51",
        alpha=0.88,
    )
    axes[1].set_title("Drivers Operacionais de Ajuste vs FIPE (Zoom sem fipe_price_brl)")
    axes[1].set_xlabel("Impacto Medio no RMSE ao Permutar (BRL)")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance.png", bbox_inches="tight")
    plt.close(fig)

    return imp_dict


def train_and_evaluate(dataset_path: Path = DATASET_PATH) -> dict[str, Any]:
    """Executa treino, cross-validation, selecao de modelo, salvamento e geracao de graficos."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset nao encontrado em {dataset_path}. Execute src/fetch_fipe_data.py primeiro.")

    df = pd.read_csv(dataset_path)
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    # Estratificacao por quintis de preco garante cobertura equilibrada entre faixas no treino e teste
    price_bins = pd.qcut(y, q=5, labels=False, duplicates="drop")
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        df.index,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=price_bins,
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
        "mape": "neg_mean_absolute_percentage_error",
    }

    candidates = build_candidate_models(seed=RANDOM_SEED)
    model_results: dict[str, dict[str, Any]] = {}
    fitted_pipelines: dict[str, Pipeline] = {}

    for name, estimator in candidates.items():
        pipe = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("regressor", estimator),
            ]
        )
        cv_scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        pipe.fit(X_train, y_train)
        y_pred_test = pipe.predict(X_test)
        holdout_metrics = evaluate_predictions(y_test.to_numpy(), y_pred_test)

        model_results[name] = {
            "cv_5fold": {
                "rmse_brl_mean": round(float(-cv_scores["test_rmse"].mean()), 2),
                "rmse_brl_std": round(float(cv_scores["test_rmse"].std()), 2),
                "mae_brl_mean": round(float(-cv_scores["test_mae"].mean()), 2),
                "r2_mean": round(float(cv_scores["test_r2"].mean()), 5),
                "mape_pct_mean": round(float(-cv_scores["test_mape"].mean() * 100.0), 3),
            },
            "holdout_test": holdout_metrics,
        }
        fitted_pipelines[name] = pipe
        logger.info(
            "Modelo %-25s | CV RMSE: R$ %8.2f | Test RMSE: R$ %8.2f | Test MAE: R$ %7.2f | Test R2: %.5f | Test MAPE: %.2f%%",
            name,
            model_results[name]["cv_5fold"]["rmse_brl_mean"],
            holdout_metrics["rmse_brl"],
            holdout_metrics["mae_brl"],
            holdout_metrics["r2"],
            holdout_metrics["mape_pct"],
        )

    best_model_name = min(
        model_results.keys(),
        key=lambda m: model_results[m]["holdout_test"]["rmse_brl"],
    )
    best_pipeline = fitted_pipelines[best_model_name]

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)
    logger.info("Melhor modelo (%s) salvo em %s", best_model_name, MODEL_PATH)

    y_pred_test = best_pipeline.predict(X_test)
    perm_importance = generate_figures(
        df=df,
        y_test=y_test,
        y_pred_test=y_pred_test,
        best_model_name=best_model_name,
        best_pipeline=best_pipeline,
        X_test=X_test,
    )

    # Salva tabela de predicoes (treino + teste) para auditoria via SQL e pytest
    pred_frames = []
    for split_name, indices, features, target in [
        ("train", idx_train, X_train, y_train),
        ("test", idx_test, X_test, y_test),
    ]:
        preds = best_pipeline.predict(features)
        actual = target.to_numpy()
        residual = actual - preds
        abs_err = np.abs(residual)
        abs_pct = abs_err / actual
        pred_frames.append(
            pd.DataFrame(
                {
                    "vehicle_id": df.loc[indices, "vehicle_id"].values,
                    "model_name": best_model_name,
                    "split_type": split_name,
                    "actual_price_brl": np.round(actual, 2),
                    "predicted_price_brl": np.round(preds, 2),
                    "residual_brl": np.round(residual, 2),
                    "abs_error_brl": np.round(abs_err, 2),
                    "abs_pct_error": np.round(abs_pct, 5),
                }
            )
        )
    df_preds = pd.concat(pred_frames, ignore_index=True)
    df_preds.to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8")

    metrics_payload = {
        "dataset": {
            "total_rows": int(len(df)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "n_brands": int(df["brand"].nunique()),
            "n_model_families": int(df["model_family"].nunique()),
            "year_min": int(df["model_year"].min()),
            "year_max": int(df["model_year"].max()),
            "fipe_price_min_brl": round(float(df["fipe_price_brl"].min()), 2),
            "fipe_price_max_brl": round(float(df["fipe_price_brl"].max()), 2),
            "fipe_price_mean_brl": round(float(df["fipe_price_brl"].mean()), 2),
            "market_price_mean_brl": round(float(df["market_asking_price"].mean()), 2),
            "mean_fipe_spread_pct": round(float(df["fipe_spread_pct"].mean()), 2),
        },
        "features": {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "target": TARGET_COLUMN,
        },
        "best_model": best_model_name,
        "best_model_holdout_metrics": model_results[best_model_name]["holdout_test"],
        "models": model_results,
        "permutation_importance_holdout_rmse_brl": perm_importance,
        "artifacts": {
            "model_path": str(MODEL_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "predictions_path": str(PREDICTIONS_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "figures": [
                "reports/figures/price_distribution.png",
                "reports/figures/depreciation_by_age.png",
                "reports/figures/actual_vs_predicted.png",
                "reports/figures/feature_importance.png",
            ],
        },
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Metricas consolidadas em %s", METRICS_PATH)
    return metrics_payload


if __name__ == "__main__":
    summary = train_and_evaluate()
    print(json.dumps({"best_model": summary["best_model"], "metrics": summary["best_model_holdout_metrics"]}, indent=2))
