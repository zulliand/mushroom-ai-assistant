#!/usr/bin/env python3
"""Prepare species ImageFolder layout from CSVs.

Reads train.csv, val.csv, test.csv from data/raw/mushroom_species_recognition/.
Uses columns `image_path` and `label`. Copies images with shutil.copy2 into
data/species_images/{split}/{label}/ while keeping the original split.
Writes a JSON report to models/species_dataset_import_report.json.

Do NOT move or modify files in data/raw/.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


RAW_DIR = Path("data") / "raw" / "mushroom_species_recognition"
MERGED_DIR = RAW_DIR / "merged_dataset"
TARGET_BASE = Path("data") / "species_images"
REPORT_PATH = Path("models") / "species_dataset_import_report.json"

CSV_FILES = {
    "train": "train.csv",
    "val": "val.csv",
    "test": "test.csv",
}


def find_source(path: str) -> Path | None:
    """Try to locate the source file for a path listed in the CSV.

    We try several likely locations but do NOT move or modify raw files.
    """
    if not path:
        return None

    normalized = path.strip().replace("\\", "/")

    if "merged_dataset/" in normalized:
        relative = normalized.split("merged_dataset/", 1)[1].lstrip("/")
        candidate = MERGED_DIR / relative
        if candidate.exists():
            return candidate

    # absolute
    candidate = Path(normalized)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    # relative to RAW_DIR
    candidate = RAW_DIR / normalized
    if candidate.exists():
        return candidate

    # relative to MERGED_DIR
    candidate = MERGED_DIR / normalized
    if candidate.exists():
        return candidate

    # maybe CSV contains only basenames -> try searching merged dir and raw dir top-level
    basename = os.path.basename(normalized)
    candidate = MERGED_DIR / basename
    if candidate.exists():
        return candidate

    candidate = RAW_DIR / basename
    if candidate.exists():
        return candidate

    return None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sanitize_label(label: str) -> str:
    """Turn a raw class label into a safe folder name."""

    value = label.strip().replace(" ", "_")
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown"


def clean_target_directory() -> None:
    """Remove old imported species images so the folder matches the CSV import."""

    if TARGET_BASE.exists():
        shutil.rmtree(TARGET_BASE)
    TARGET_BASE.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the CSV-based species ImageFolder dataset.")
    parser.add_argument("--max-images-per-class", type=int, default=300, help="Maximum copied images per class and split")
    return parser.parse_args()


def process_csv(split: str, csv_path: str, max_images_per_class: int, stats: dict, split_class_counts: dict):
    full_csv = RAW_DIR / csv_path
    if not full_csv.exists():
        print(f"Warning: CSV not found: {full_csv}")
        return

    with full_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            stats["total_images_processed"] += 1
            img_path = row.get("image_path") or row.get("image") or row.get("path")
            label = row.get("label") or row.get("species") or row.get("class")

            if not img_path or not label:
                stats["skipped_rows"] += 1
                stats["skipped_row_reasons"]["missing_image_or_label"] += 1
                continue

            folder_label = sanitize_label(label)
            if split_class_counts[(split, folder_label)] >= max_images_per_class:
                stats["skipped_rows"] += 1
                stats["skipped_row_reasons"]["max_images_per_class"] += 1
                continue

            src = find_source(img_path)
            if not src:
                stats["missing_files"] += 1
                stats["missing_files_list"].append({"requested": img_path, "split": split, "label": label})
                continue

            dest_dir = TARGET_BASE / split / folder_label
            ensure_dir(dest_dir)
            dest_path = dest_dir / src.name

            try:
                if dest_path.exists():
                    stats["skipped_rows"] += 1
                    stats["skipped_row_reasons"]["duplicate_destination"] += 1
                    continue

                shutil.copy2(src, dest_path)
                stats["copied_images"] += 1
                stats["copied_images_per_split"][split] += 1
                stats["copied_images_per_class"][folder_label] += 1
                stats["classes_seen"].add(folder_label)
                split_class_counts[(split, folder_label)] += 1
            except Exception as e:
                stats["errors"].append({"src": str(src), "dest": str(dest_path), "error": str(e)})


def main():
    args = parse_args()

    ensure_dir(REPORT_PATH.parent)
    clean_target_directory()

    stats = {
        "total_images_processed": 0,
        "copied_images": 0,
        "copied_images_per_split": defaultdict(int),
        "copied_images_per_class": defaultdict(int),
        "missing_files": 0,
        "missing_files_list": [],
        "skipped_rows": 0,
        "skipped_row_reasons": Counter(),
        "classes_seen": set(),
        "errors": [],
    }
    split_class_counts = defaultdict(int)

    for split, csvname in CSV_FILES.items():
        process_csv(split, csvname, args.max_images_per_class, stats, split_class_counts)

    report = {
        "number_of_classes": len(stats["classes_seen"]),
        "max_images_per_class": args.max_images_per_class,
        "copied_images_per_split": dict(stats["copied_images_per_split"]),
        "copied_images_per_class": dict(stats["copied_images_per_class"]),
        "missing_files": stats["missing_files"],
        "skipped_rows": stats["skipped_rows"],
        "skipped_row_reasons": dict(stats["skipped_row_reasons"]),
        "total_images_processed": stats["total_images_processed"],
        "missing_files_list": stats["missing_files_list"],
        "errors": stats["errors"],
    }

    with REPORT_PATH.open("w", encoding="utf-8") as out:
        json.dump(report, out, indent=2, ensure_ascii=False)

    print("Import finished.")
    print(json.dumps(
        {
            "classes": report["number_of_classes"],
            "total": report["total_images_processed"],
            "copied": sum(report["copied_images_per_split"].values()),
            "missing": report["missing_files"],
            "skipped": report["skipped_rows"],
            "max_images_per_class": report["max_images_per_class"],
        },
        indent=2,
    ))


if __name__ == '__main__':
    main()
