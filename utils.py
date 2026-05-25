"""Shared helpers for the Mushroom AI Assistant project."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import joblib

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

NUMERIC_DATA_PATH = DATA_DIR / "mushroom_numeric.csv"
CV_MODEL_PATH = MODELS_DIR / "mushroom_cv.pt"
NUMERIC_MODEL_PATH = MODELS_DIR / "mushroom_rf.pkl"
NUMERIC_PIPELINE_PATH = MODELS_DIR / "mushroom_numeric_pipeline.pkl"
NUMERIC_METADATA_PATH = MODELS_DIR / "mushroom_numeric_metadata.json"
NUMERIC_METRICS_PATH = MODELS_DIR / "numeric_metrics.json"
CV_METRICS_PATH = MODELS_DIR / "cv_metrics.json"
# Species-specific artifacts
SPECIES_IMAGES_DIR = DATA_DIR / "species_images"
SPECIES_CV_MODEL_PATH = MODELS_DIR / "mushroom_species_cv.pt"
SPECIES_CV_METADATA_PATH = MODELS_DIR / "mushroom_species_cv_metadata.json"
SPECIES_CV_METRICS_PATH = MODELS_DIR / "species_cv_metrics.json"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure a predictable logging format for scripts and the app."""

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_directory(path: Path) -> Path:
    """Create a directory if it does not exist and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    """Persist a JSON document with stable formatting."""

    ensure_directory(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON document from disk."""

    return json.loads(path.read_text(encoding="utf-8"))


def save_artifact(path: Path, artifact: Any) -> None:
    """Save a Python artifact using joblib for consistent model serialization."""

    ensure_directory(path.parent)
    joblib.dump(artifact, path)


def load_artifact(path: Path) -> Any:
    """Load a Python artifact written by save_artifact."""

    return joblib.load(path)


def build_label(prefix: str, value: str) -> str:
    """Create a readable label for UI text."""

    return f"{prefix}: {value}"
