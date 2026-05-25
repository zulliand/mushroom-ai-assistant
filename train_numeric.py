"""Train and evaluate the structured-data mushroom classifier."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from joblib import dump, load
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from utils import ensure_directory, setup_logging

LOGGER = logging.getLogger("mushroom.numeric")
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "Mushroom data.csv"
MODELS_DIR = PROJECT_ROOT / "models"
PIPELINE_PATH = MODELS_DIR / "mushroom_numeric_pipeline.pkl"
METRICS_PATH = MODELS_DIR / "numeric_metrics.json"
TARGET_CANDIDATES = ("class", "target", "label", "edible", "is_edible", "poisonous", "quality")
POSITIVE_LABELS = {"e", "edible", "poisonous", "p", "1", 1, True}


def load_dataset(csv_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the numeric mushroom dataset and validate that it exists."""

    if not csv_path.exists():
        raise FileNotFoundError(f"Structured dataset not found at {csv_path}")

    dataframe = pd.read_csv(csv_path)
    if dataframe.empty:
        raise ValueError(f"Dataset at {csv_path} is empty")
    return dataframe


def infer_target_column(dataframe: pd.DataFrame) -> str:
    """Infer the target column from column names and value patterns."""

    lower_map = {column.lower(): column for column in dataframe.columns}
    for candidate in TARGET_CANDIDATES:
        for lower_name, original_name in lower_map.items():
            if candidate in lower_name:
                return original_name

    binary_candidates: List[Tuple[str, int]] = []
    for column in dataframe.columns:
        unique_values = dataframe[column].dropna().astype(str).str.strip().str.lower().unique()
        if len(unique_values) == 2:
            binary_candidates.append((column, len(unique_values)))

    if binary_candidates:
        return binary_candidates[0][0]

    return dataframe.columns[0]


def normalize_target(series: pd.Series) -> Tuple[pd.Series, LabelEncoder]:
    """Encode the target column into a binary label vector."""

    encoder = LabelEncoder()
    encoded = pd.Series(encoder.fit_transform(series.astype(str).str.strip().str.lower()), index=series.index)
    return encoded, encoder


def print_eda(dataframe: pd.DataFrame, target_column: str) -> None:
    """Print the required EDA summary to stdout."""

    missing_values = int(dataframe.isna().sum().sum())
    target_distribution = dataframe[target_column].value_counts(dropna=False)
    feature_count = int(dataframe.shape[1] - 1)

    print("EDA Summary")
    print(f"Rows: {len(dataframe)}")
    print(f"Missing values: {missing_values}")
    print("Target distribution:")
    print(target_distribution.to_string())
    print(f"Feature count: {feature_count}")

    LOGGER.info("Rows: %s", len(dataframe))
    LOGGER.info("Missing values: %s", missing_values)
    LOGGER.info("Target distribution:\n%s", target_distribution.to_string())
    LOGGER.info("Feature count: %s", feature_count)


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build the preprocessing pipeline for mixed tabular data."""

    categorical_features = list(features.columns)
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - compatibility for older scikit-learn versions
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", encoder),
        ]
    )
    return ColumnTransformer(
        transformers=[("categorical", categorical_pipeline, categorical_features)],
        remainder="drop",
    )


def build_models() -> Dict[str, object]:
    """Return the model candidates we want to compare."""

    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    }


def make_pipeline(preprocessor: ColumnTransformer, model: object) -> Pipeline:
    """Create a reusable preprocessing + model pipeline."""

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def evaluate_model(name: str, pipeline: Pipeline, x_eval: pd.DataFrame, y_eval: pd.Series) -> Dict[str, float]:
    """Calculate the requested classification metrics for one fitted model."""

    predictions = pipeline.predict(x_eval)
    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(y_eval, predictions)),
        "precision": float(precision_score(y_eval, predictions, zero_division=0)),
        "recall": float(recall_score(y_eval, predictions, zero_division=0)),
        "f1": float(f1_score(y_eval, predictions, zero_division=0)),
    }
    return metrics


def print_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, label_encoder: LabelEncoder) -> None:
    """Print and plot the confusion matrix for the selected model."""

    matrix = confusion_matrix(y_true, y_pred)
    label_names = list(label_encoder.classes_)
    print("Confusion matrix:")
    print(pd.DataFrame(matrix, index=[f"actual_{label}" for label in label_names], columns=[f"pred_{label}" for label in label_names]).to_string())

    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=label_names, yticklabels=label_names)
    plt.title("Numeric Mushroom Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    ensure_directory(MODELS_DIR)
    plt.savefig(MODELS_DIR / "numeric_confusion_matrix.png", dpi=160)
    plt.close()


def train_numeric_pipeline(csv_path: Path = DATA_PATH) -> Dict[str, object]:
    """Train, compare, and save the numeric mushroom classifier."""

    setup_logging()
    dataframe = load_dataset(csv_path)
    target_column = infer_target_column(dataframe)
    print_eda(dataframe, target_column)

    features = dataframe.drop(columns=[target_column])
    target = dataframe[target_column]
    target_values, label_encoder = normalize_target(target)

    x_train_full, x_test, y_train_full, y_test = train_test_split(
        features,
        target_values,
        test_size=0.2,
        random_state=42,
        stratify=target_values,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=0.2,
        random_state=42,
        stratify=y_train_full,
    )

    preprocessor = build_preprocessor(x_train)
    model_candidates = build_models()
    comparison_rows: List[Dict[str, float]] = []
    fitted_pipelines: Dict[str, Pipeline] = {}

    for name, model in model_candidates.items():
        pipeline = make_pipeline(preprocessor, model)
        pipeline.fit(x_train, y_train)
        fitted_pipelines[name] = pipeline
        val_metrics = evaluate_model(name, pipeline, x_val, y_val)
        comparison_rows.append(val_metrics)
        print(f"{name} F1: {val_metrics['f1']:.2f}")
        LOGGER.info("%s validation metrics: %s", name, val_metrics)

    comparison_df = pd.DataFrame(comparison_rows).sort_values(by="f1", ascending=False).reset_index(drop=True)
    print("Model comparison:")
    print(comparison_df.to_string(index=False))

    best_model_name = comparison_df.iloc[0]["model"]
    print("Selected model:")
    print(best_model_name)

    final_model = model_candidates[best_model_name]
    final_pipeline = make_pipeline(build_preprocessor(x_train_full), final_model)
    final_pipeline.fit(x_train_full, y_train_full)
    test_metrics = evaluate_model(best_model_name, final_pipeline, x_test, y_test)
    test_predictions = final_pipeline.predict(x_test)

    print("Test metrics:")
    print(pd.DataFrame([test_metrics]).to_string(index=False))
    print_confusion_matrix(y_test, test_predictions, label_encoder)

    ensure_directory(MODELS_DIR)
    dump(final_pipeline, PIPELINE_PATH)

    metrics_payload = {
        "dataset_path": str(csv_path),
        "target_column": target_column,
        "rows": int(dataframe.shape[0]),
        "feature_count": int(features.shape[1]),
        "target_distribution": dataframe[target_column].value_counts(dropna=False).to_dict(),
        "model_comparison": comparison_df.to_dict(orient="records"),
        "selected_model": best_model_name,
        "validation_metrics": comparison_df.iloc[0].to_dict(),
        "test_metrics": test_metrics,
        "label_classes": label_encoder.classes_.tolist(),
        "pipeline_path": str(PIPELINE_PATH),
    }

    METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Saved pipeline to %s", PIPELINE_PATH)
    LOGGER.info("Saved metrics to %s", METRICS_PATH)

    return {
        "selected_model": best_model_name,
        "validation_metrics": comparison_df.iloc[0].to_dict(),
        "test_metrics": test_metrics,
        "pipeline_path": str(PIPELINE_PATH),
        "metrics_path": str(METRICS_PATH),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""

    parser = argparse.ArgumentParser(description="Train the mushroom numeric model.")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH, help="Path to data/Mushroom data.csv")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    try:
        result = train_numeric_pipeline(csv_path=args.data_path)
        LOGGER.info("Training complete: %s", result)
    except Exception as exc:  # pragma: no cover - explicit CLI failure surface
        LOGGER.exception("Numeric training failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
