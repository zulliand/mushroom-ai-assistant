"""Train the species-level CV model (explicit command, no auto-run)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from train_cv import train_cv_pipeline
from utils import (
    SPECIES_CV_METADATA_PATH,
    SPECIES_CV_METRICS_PATH,
    SPECIES_CV_MODEL_PATH,
    SPECIES_IMAGES_DIR,
)

LOGGER = logging.getLogger("mushroom.species_cv")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for species-level CV training."""

    parser = argparse.ArgumentParser(description="Train species-level mushroom classifier with ResNet18 transfer learning.")
    parser.add_argument("--data-dir", type=Path, default=SPECIES_IMAGES_DIR, help="Image root containing train/val/test species folders")
    parser.add_argument("--checkpoint-path", type=Path, default=SPECIES_CV_MODEL_PATH, help="Path to save species CV checkpoint")
    parser.add_argument("--metadata-path", type=Path, default=SPECIES_CV_METADATA_PATH, help="Path to save species metadata JSON")
    parser.add_argument("--metrics-path", type=Path, default=SPECIES_CV_METRICS_PATH, help="Path to save species metrics JSON")
    parser.add_argument("--image-size", type=int, default=224, help="Image size for transfer learning")
    parser.add_argument("--batch-size", type=int, default=0, help="Batch size (0=auto)")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Optimizer learning rate")
    parser.add_argument("--num-workers", type=int, default=0, help="Data loader workers")
    parser.add_argument("--early-stopping", action="store_true", help="Enable early stopping based on validation accuracy")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (epochs)")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    try:
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
        )
        LOGGER.info("Species CV training complete: %s", result)
    except Exception as exc:  # pragma: no cover - explicit CLI failure surface
        LOGGER.exception("Species CV training failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
