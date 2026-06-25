from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


LOGGER = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_DIR / "housing_price_dataset.csv"
DEFAULT_ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
DEFAULT_PLOTS_DIR = PROJECT_DIR / "plots"

TARGET_COLUMN = "Price"
RAW_FEATURES = ["SquareFeet", "Bedrooms", "Bathrooms", "Neighborhood", "YearBuilt"]
NUMERIC_FEATURES = [
    "SquareFeet",
    "Bedrooms",
    "Bathrooms",
    "YearBuilt",
    "TotalRooms",
    "PropertyAge",
]
CATEGORICAL_FEATURES = ["Neighborhood"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLUMNS = RAW_FEATURES + [TARGET_COLUMN]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate housing price prediction models."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    parser.add_argument("--reference-year", type=int, default=2024)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def add_engineered_features(data: pd.DataFrame, reference_year: int) -> pd.DataFrame:
    data = data.copy()
    data["TotalRooms"] = data["Bedrooms"] + data["Bathrooms"]
    data["PropertyAge"] = reference_year - data["YearBuilt"]
    return data


def load_dataset(data_path: Path, reference_year: int) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    data = pd.read_csv(data_path)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Dataset is missing required columns: {missing}")

    return add_engineered_features(data, reference_year)


def make_one_hot_encoder() -> OneHotEncoder:
    encoder_options = {"drop": "first", "handle_unknown": "ignore"}
    try:
        return OneHotEncoder(sparse_output=False, **encoder_options)
    except TypeError:
        return OneHotEncoder(sparse=False, **encoder_options)


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_model_candidates(random_state: int) -> dict[str, object]:
    return {
        "Mean Baseline": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10.0),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=12,
            min_samples_leaf=5,
            random_state=random_state,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        ),
    }


def build_pipeline(model: object) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_true, y_pred),
    }


def train_and_evaluate_models(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, Pipeline], dict[str, np.ndarray]]:
    fitted_pipelines: dict[str, Pipeline] = {}
    predictions: dict[str, np.ndarray] = {}
    metric_rows: list[dict[str, float | str]] = []

    for model_name, model in build_model_candidates(random_state).items():
        LOGGER.info("Training %s", model_name)
        pipeline = build_pipeline(model)
        pipeline.fit(x_train, y_train)

        y_pred = pipeline.predict(x_test)
        metrics = evaluate_predictions(y_test, y_pred)

        fitted_pipelines[model_name] = pipeline
        predictions[model_name] = y_pred
        metric_rows.append({"Model": model_name, **metrics})

    metrics_frame = (
        pd.DataFrame(metric_rows)
        .sort_values(by=["RMSE", "MAE"], ascending=True)
        .reset_index(drop=True)
    )
    return metrics_frame, fitted_pipelines, predictions


def cross_validate_rmse(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[float, float]:
    scores = cross_val_score(
        pipeline,
        x_train,
        y_train,
        cv=5,
        scoring="neg_root_mean_squared_error",
    )
    rmse_scores = -scores
    return float(rmse_scores.mean()), float(rmse_scores.std())


def format_number(value: float) -> str:
    if abs(value) >= 1_000:
        return f"{value:,.0f}"
    return f"{value:,.3f}"


def add_bar_labels(ax: plt.Axes, bars: Iterable[plt.Rectangle], metric: str) -> None:
    values = [bar.get_width() for bar in bars]
    max_value = max(values) if values else 0
    padding = max_value * 0.015 if max_value else 1

    for bar in bars:
        value = bar.get_width()
        label = f"{value:.3f}" if metric == "R2" else format_number(value)
        ax.text(
            value + padding,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=9,
        )

    ax.set_xlim(left=0, right=max_value * 1.16 if max_value else 1)


def plot_single_metric(
    metrics: pd.DataFrame,
    metric: str,
    title: str,
    output_path: Path,
    color: str,
) -> None:
    ordered = metrics.sort_values(metric, ascending=metric != "R2")
    fig, ax = plt.subplots(figsize=(10, 5.5))

    bars = ax.barh(ordered["Model"], ordered[metric], color=color)
    if metric != "R2":
        ax.invert_yaxis()

    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel(metric)
    ax.grid(axis="x", alpha=0.25)
    add_bar_labels(ax, bars, metric)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_model_performance(metrics: pd.DataFrame, output_path: Path) -> None:
    ordered = metrics.sort_values("RMSE", ascending=True)
    x_positions = np.arange(len(ordered))
    bar_width = 0.36

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(
        x_positions - bar_width / 2,
        ordered["MAE"],
        width=bar_width,
        label="MAE",
        color="#3b82f6",
    )
    ax.bar(
        x_positions + bar_width / 2,
        ordered["RMSE"],
        width=bar_width,
        label="RMSE",
        color="#f97316",
    )

    ax.set_title("Model Error Comparison", fontsize=15, pad=12)
    ax.set_ylabel("Error in dollars")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(ordered["Model"], rotation=25, ha="right")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value / 1000:.0f}k"))
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_actual_vs_predicted(
    y_true: pd.Series,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    actual = y_true.to_numpy()
    lower = min(actual.min(), y_pred.min())
    upper = max(actual.max(), y_pred.max())

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(actual, y_pred, alpha=0.55, color="#2563eb", edgecolor="white", linewidth=0.4)
    ax.plot([lower, upper], [lower, upper], color="#ef4444", linewidth=2, label="Perfect prediction")

    ax.set_title("Actual vs Predicted Prices", fontsize=15, pad=12)
    ax.set_xlabel("Actual price")
    ax.set_ylabel("Predicted price")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value / 1000:.0f}k"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value / 1000:.0f}k"))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_residuals(
    y_true: pd.Series,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    residuals = y_true.to_numpy() - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    axes[0].scatter(y_pred, residuals, alpha=0.55, color="#16a34a", edgecolor="white", linewidth=0.4)
    axes[0].axhline(0, color="#ef4444", linewidth=2)
    axes[0].set_title("Residuals by Prediction")
    axes[0].set_xlabel("Predicted price")
    axes[0].set_ylabel("Actual minus predicted")
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value / 1000:.0f}k"))
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value / 1000:.0f}k"))
    axes[0].grid(alpha=0.25)

    axes[1].hist(residuals, bins=28, color="#64748b", edgecolor="white")
    axes[1].axvline(0, color="#ef4444", linewidth=2)
    axes[1].set_title("Residual Distribution")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Homes")
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value / 1000:.0f}k"))
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def clean_feature_name(feature_name: str) -> str:
    return (
        feature_name.replace("numeric__", "")
        .replace("categorical__", "")
        .replace("Neighborhood_", "Neighborhood: ")
    )


def get_feature_names(pipeline: Pipeline) -> list[str]:
    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        names = preprocessor.get_feature_names_out()
    except AttributeError:
        names = MODEL_FEATURES
    return [clean_feature_name(name) for name in names]


def plot_feature_importance(pipeline: Pipeline, output_path: Path) -> Path | None:
    model = pipeline.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        importances = np.abs(np.ravel(model.coef_))
    else:
        return None

    feature_names = get_feature_names(pipeline)
    if len(feature_names) != len(importances):
        return None

    importance_frame = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(12)
        .sort_values("Importance", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(importance_frame["Feature"], importance_frame["Importance"], color="#0f766e")
    ax.set_title("Top Feature Importance", fontsize=15, pad=12)
    ax.set_xlabel("Relative importance")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def create_plots(
    metrics: pd.DataFrame,
    y_test: pd.Series,
    best_predictions: np.ndarray,
    best_pipeline: Pipeline,
    plots_dir: Path,
) -> list[Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_paths = [
        plots_dir / "mean_absolute_error.png",
        plots_dir / "mean_squared_error.png",
        plots_dir / "root_mean_squared_error.png",
        plots_dir / "model_performance.png",
        plots_dir / "actual_vs_predicted.png",
        plots_dir / "residuals.png",
    ]

    plot_single_metric(
        metrics,
        metric="MAE",
        title="Mean Absolute Error by Model",
        output_path=plot_paths[0],
        color="#2563eb",
    )
    plot_single_metric(
        metrics,
        metric="MSE",
        title="Mean Squared Error by Model",
        output_path=plot_paths[1],
        color="#7c3aed",
    )
    plot_single_metric(
        metrics,
        metric="RMSE",
        title="Root Mean Squared Error by Model",
        output_path=plot_paths[2],
        color="#f97316",
    )
    plot_model_performance(metrics, plot_paths[3])
    plot_actual_vs_predicted(y_test, best_predictions, plot_paths[4])
    plot_residuals(y_test, best_predictions, plot_paths[5])

    feature_importance_path = plot_feature_importance(
        best_pipeline,
        plots_dir / "feature_importance.png",
    )
    if feature_importance_path is not None:
        plot_paths.append(feature_importance_path)

    return plot_paths


def save_outputs(
    best_pipeline: Pipeline,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    metadata: dict[str, object],
    artifacts_dir: Path,
) -> dict[str, Path]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "model": artifacts_dir / "best_housing_price_model.joblib",
        "metrics": artifacts_dir / "model_metrics.csv",
        "predictions": artifacts_dir / "model_predictions.csv",
        "metadata": artifacts_dir / "run_metadata.json",
    }

    joblib.dump(best_pipeline, output_paths["model"])
    metrics.to_csv(output_paths["metrics"], index=False)
    predictions.to_csv(output_paths["predictions"], index=False)

    with output_paths["metadata"].open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    return output_paths


def build_prediction_frame(
    x_test: pd.DataFrame,
    y_test: pd.Series,
    best_predictions: np.ndarray,
) -> pd.DataFrame:
    prediction_frame = x_test[MODEL_FEATURES].reset_index(drop=True).copy()
    prediction_frame["ActualPrice"] = y_test.reset_index(drop=True)
    prediction_frame["PredictedPrice"] = best_predictions
    prediction_frame["Residual"] = prediction_frame["ActualPrice"] - prediction_frame["PredictedPrice"]
    return prediction_frame


def make_sample_prediction(best_pipeline: Pipeline, reference_year: int) -> float:
    sample_home = pd.DataFrame(
        {
            "SquareFeet": [2500],
            "Bedrooms": [3],
            "Bathrooms": [2],
            "Neighborhood": ["Urban"],
            "YearBuilt": [2005],
        }
    )
    sample_home = add_engineered_features(sample_home, reference_year)
    return float(best_pipeline.predict(sample_home[MODEL_FEATURES])[0])


def print_summary(
    metrics: pd.DataFrame,
    best_model_name: str,
    cv_rmse_mean: float,
    cv_rmse_std: float,
    sample_prediction: float,
    output_paths: dict[str, Path],
    plot_paths: list[Path],
) -> None:
    display_metrics = metrics.copy()
    for column in ["MAE", "MSE", "RMSE"]:
        display_metrics[column] = display_metrics[column].map(lambda value: f"{value:,.2f}")
    display_metrics["R2"] = display_metrics["R2"].map(lambda value: f"{value:.3f}")

    print("\nModel performance sorted by RMSE:")
    print(display_metrics.to_string(index=False))
    print(f"\nBest model: {best_model_name}")
    print(f"Cross-validated RMSE: {cv_rmse_mean:,.2f} (+/- {cv_rmse_std:,.2f})")
    print(f"Example Urban home prediction: ${sample_prediction:,.0f}")
    print(f"Saved model: {output_paths['model']}")
    print(f"Saved metrics: {output_paths['metrics']}")
    print(f"Saved predictions: {output_paths['predictions']}")
    print("Saved plots:")
    for plot_path in plot_paths:
        print(f"  - {plot_path}")


def main() -> None:
    args = parse_args()
    configure_logging()

    data_path = resolve_path(args.data_path)
    artifacts_dir = resolve_path(args.artifacts_dir)
    plots_dir = resolve_path(args.plots_dir)

    data = load_dataset(data_path, args.reference_year)
    x = data[MODEL_FEATURES]
    y = data[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    metrics, pipelines, model_predictions = train_and_evaluate_models(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        random_state=args.random_state,
    )

    best_model_name = str(metrics.loc[0, "Model"])
    best_pipeline = pipelines[best_model_name]
    best_predictions = model_predictions[best_model_name]

    cv_rmse_mean, cv_rmse_std = cross_validate_rmse(best_pipeline, x_train, y_train)
    prediction_frame = build_prediction_frame(x_test, y_test, best_predictions)
    plot_paths = create_plots(metrics, y_test, best_predictions, best_pipeline, plots_dir)
    sample_prediction = make_sample_prediction(best_pipeline, args.reference_year)

    best_metrics = metrics.loc[0].to_dict()
    metadata = {
        "best_model": best_model_name,
        "reference_year": args.reference_year,
        "random_state": args.random_state,
        "test_size": args.test_size,
        "best_metrics": {
            key: float(value) if isinstance(value, (np.floating, float)) else value
            for key, value in best_metrics.items()
        },
        "cross_validation": {
            "rmse_mean": cv_rmse_mean,
            "rmse_std": cv_rmse_std,
        },
        "plots": [str(path) for path in plot_paths],
    }
    output_paths = save_outputs(
        best_pipeline=best_pipeline,
        metrics=metrics,
        predictions=prediction_frame,
        metadata=metadata,
        artifacts_dir=artifacts_dir,
    )

    print_summary(
        metrics=metrics,
        best_model_name=best_model_name,
        cv_rmse_mean=cv_rmse_mean,
        cv_rmse_std=cv_rmse_std,
        sample_prediction=sample_prediction,
        output_paths=output_paths,
        plot_paths=plot_paths,
    )


if __name__ == "__main__":
    main()
