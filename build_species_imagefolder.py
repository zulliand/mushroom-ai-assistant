"""Build a species-level ImageFolder dataset by copying files from existing splits.

This script never moves or renames source files. It creates a derived dataset at
`data/species_images/` (or a custom target root) with split-preserving layout.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from utils import SPECIES_IMAGES_DIR, ensure_directory, setup_logging

LOGGER = logging.getLogger("mushroom.species_builder")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
SPECIES_PATTERN = re.compile(r"^([A-Z][a-z]+)[_-]([a-z]+)(?:[_-]\d+.*)?$")


def infer_species_from_stem(stem: str) -> Optional[str]:
    """Extract species key from a filename stem like `Agaricus_bisporus_0001`."""

    match = SPECIES_PATTERN.match(stem)
    if not match:
        return None
    genus = match.group(1).capitalize()
    species = match.group(2).lower()
    return f"{genus}_{species}"


def iter_split_files(split_root: Path) -> Iterable[Path]:
    """Yield all image files under one split root."""

    for path in split_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def copy_species_dataset(
    source_root: Path,
    target_root: Path,
    overwrite: bool = False,
    max_unknown_report: int = 100,
) -> Dict[str, object]:
    """Copy split-preserving dataset into ImageFolder layout by species."""

    splits = ["train", "val", "test"]
    for split in splits:
        if not (source_root / split).exists():
            raise FileNotFoundError(f"Missing expected split folder: {source_root / split}")

    copied = 0
    skipped_existing = 0
    unknown: List[str] = []
    unknown_total = 0
    by_split: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    species_counts: Dict[str, int] = {}

    for split in splits:
        split_root = source_root / split
        for source_path in iter_split_files(split_root):
            species = infer_species_from_stem(source_path.stem)
            if species is None:
                unknown_total += 1
                if len(unknown) < max_unknown_report:
                    unknown.append(str(source_path))
                continue

            destination = target_root / split / species / source_path.name
            ensure_directory(destination.parent)
            if destination.exists() and not overwrite:
                skipped_existing += 1
                continue

            shutil.copy2(source_path, destination)
            copied += 1
            by_split[split] += 1
            species_counts[species] = species_counts.get(species, 0) + 1

    report = {
        "source_root": str(source_root),
        "target_root": str(target_root),
        "copied_images": copied,
        "skipped_existing": skipped_existing,
        "unknown_filename_pattern_count": unknown_total,
        "unknown_filename_pattern_sample": unknown,
        "split_counts": by_split,
        "species_counts": dict(sorted(species_counts.items(), key=lambda item: item[0])),
    }
    return report


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(description="Build species-level ImageFolder dataset from train/val/test splits.")
    parser.add_argument("--source-root", type=Path, default=Path("data"), help="Root containing train/val/test folders")
    parser.add_argument("--target-root", type=Path, default=SPECIES_IMAGES_DIR, help="Derived target root for species dataset")
    parser.add_argument("--report-path", type=Path, default=Path("models/species_dataset_build_report.json"), help="Where to save build report")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing copied files at destination")
    parser.add_argument("--max-unknown-report", type=int, default=100, help="Max unknown-pattern files listed in report sample")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    setup_logging()
    args = parse_args()

    report = copy_species_dataset(
        source_root=args.source_root,
        target_root=args.target_root,
        overwrite=args.overwrite,
        max_unknown_report=args.max_unknown_report,
    )

    ensure_directory(args.report_path.parent)
    args.report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Species dataset build complete.")
    LOGGER.info("Copied images: %s", report["copied_images"])
    LOGGER.info("Unknown pattern count: %s", report["unknown_filename_pattern_count"])
    LOGGER.info("Saved report to %s", args.report_path)


if __name__ == "__main__":
    main()
