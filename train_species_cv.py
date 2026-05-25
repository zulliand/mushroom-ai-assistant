"""Train the species-level CV model (explicit command, no auto-run)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from train_cv import build_datasets, train_cv_pipeline
from utils import (
    SPECIES_CV_METADATA_PATH,
    SPECIES_CV_METRICS_PATH,
    SPECIES_CV_MODEL_PATH,
    SPECIES_IMAGES_DIR,
)

LOGGER = logging.getLogger("mushroom.species_cv")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for species-level CV training."""

    parser = argparse.ArgumentParser(description="Train species-level mushroom classifier with transfer learning.")
    parser.add_argument("--data-dir", type=Path, default=SPECIES_IMAGES_DIR, help="Image root containing train/val/test species folders")
    parser.add_argument("--checkpoint-path", type=Path, default=SPECIES_CV_MODEL_PATH, help="Path to save species CV checkpoint")
    parser.add_argument("--metadata-path", type=Path, default=SPECIES_CV_METADATA_PATH, help="Path to save species metadata JSON")
    parser.add_argument("--metrics-path", type=Path, default=SPECIES_CV_METRICS_PATH, help="Path to save species metrics JSON")
    parser.add_argument("--model-name", type=str, default="efficientnet_b0", choices=["resnet18", "efficientnet_b0"], help="Backbone model to use")
    parser.add_argument("--image-size", type=int, default=224, help="Image size for transfer learning")
    parser.add_argument("--batch-size", type=int, default=0, help="Batch size (0=auto)")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Optimizer learning rate")
    parser.add_argument("--num-workers", type=int, default=0, help="Data loader workers")
    parser.add_argument("--max-train-samples", type=int, default=12000, help="Maximum training samples to use (0 or negative = full dataset)")
    parser.add_argument("--max-val-samples", type=int, default=3000, help="Maximum validation samples to use (0 or negative = full dataset)")
    parser.add_argument("--max-test-samples", type=int, default=3000, help="Maximum test samples to use (0 or negative = full dataset)")
    parser.add_argument("--early-stopping", action="store_true", default=True, help="Enable early stopping based on validation accuracy")
    parser.add_argument("--no-early-stopping", action="store_false", dest="early_stopping", help="Disable early stopping")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (epochs)")
    return parser.parse_args()


def estimate_dataset_size(data_dir: Path, image_size: int, max_train_samples: int | None, max_val_samples: int | None, max_test_samples: int | None) -> dict:
    """Estimate the effective dataset sizes after applying the CLI limits."""

    train_dataset, val_dataset, test_dataset, _, _ = build_datasets(data_dir=data_dir, image_size=image_size)

    def _limit(size: int, max_samples: int | None) -> int:
        if not max_samples or max_samples <= 0:
            return size
        return min(size, max_samples)

    return {
        "train": _limit(len(train_dataset), max_train_samples),
        "val": _limit(len(val_dataset), max_val_samples),
        "test": _limit(len(test_dataset), max_test_samples),
    }


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    try:
        estimated_sizes = estimate_dataset_size(
            data_dir=args.data_dir,
            image_size=args.image_size,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            max_test_samples=args.max_test_samples,
        )
        LOGGER.info(
            "Estimated effective dataset size (%s) - train: %s, val: %s, test: %s",
            args.model_name,
            estimated_sizes["train"],
            estimated_sizes["val"],
            estimated_sizes["test"],
        )
        result = train_cv_pipeline(
            data_dir=args.data_dir,
            checkpoint_path=args.checkpoint_path,
            metadata_path=args.metadata_path,
            metrics_path=args.metrics_path,
            image_size=args.image_size,
            batch_size=args.batch_size or None,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            num_workers=args.num_workers,
            early_stopping=args.early_stopping,
            patience=args.patience,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            max_test_samples=args.max_test_samples,
            model_name=args.model_name,
        )
        LOGGER.info("Species CV training complete: %s", result)
    except Exception as exc:  # pragma: no cover - explicit CLI failure surface
        LOGGER.exception("Species CV training failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
