#!/usr/bin/env python3
"""Prepare species ImageFolder layout from CSVs.

Reads train.csv, val.csv, test.csv from data/raw/mushroom_species_recognition/
Uses columns `image_path` and `label`. Copies images (shutil.copy2) into
data/species_images/{split}/{label}/ keeping the exact split from CSVs.
Writes a JSON report to models/species_dataset_import_report.json.

Do NOT move or modify files in data/raw/.
"""
import csv
import os
import shutil
import json
from collections import defaultdict


RAW_DIR = os.path.join("data", "raw", "mushroom_species_recognition")
MERGED_DIR = os.path.join(RAW_DIR, "merged_dataset")
TARGET_BASE = os.path.join("data", "species_images")
REPORT_PATH = os.path.join("models", "species_dataset_import_report.json")

CSV_FILES = {
    "train": "train.csv",
    "val": "val.csv",
    "test": "test.csv",
}


def find_source(path):
    """Try to locate the source file for a path listed in the CSV.

    We try several likely locations but do NOT move or modify raw files.
    """
    if not path:
        return None

    # normalize
    path = path.strip()
    # absolute
    if os.path.isabs(path) and os.path.exists(path):
        return path

    # relative to RAW_DIR
    cand = os.path.join(RAW_DIR, path)
    if os.path.exists(cand):
        return cand

    # relative to MERGED_DIR
    cand = os.path.join(MERGED_DIR, path)
    if os.path.exists(cand):
        return cand

    # maybe CSV contains only basenames -> try searching merged dir and raw dir top-level
    basename = os.path.basename(path)
    cand = os.path.join(MERGED_DIR, basename)
    if os.path.exists(cand):
        return cand

    cand = os.path.join(RAW_DIR, basename)
    if os.path.exists(cand):
        return cand

    return None


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def process_csv(split, csv_path, stats):
    full_csv = os.path.join(RAW_DIR, csv_path)
    if not os.path.exists(full_csv):
        print(f"Warning: CSV not found: {full_csv}")
        return

    with open(full_csv, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            stats['total_images_processed'] += 1
            img_path = row.get('image_path') or row.get('image') or row.get('path')
            label = row.get('label') or row.get('species') or row.get('class')

            if not img_path or not label:
                stats['missing_meta'] += 1
                stats['missing_files_list'].append({'csv': full_csv, 'row': row})
                continue

            src = find_source(img_path)
            if not src:
                stats['missing_files_list'].append({'requested': img_path, 'split': split, 'label': label})
                stats['missing_files'] += 1
                continue

            dest_dir = os.path.join(TARGET_BASE, split, label)
            ensure_dir(dest_dir)
            dest_path = os.path.join(dest_dir, os.path.basename(src))

            try:
                if not os.path.exists(dest_path):
                    shutil.copy2(src, dest_path)
                    stats['copied_images'] += 1
                else:
                    # already present -> count as copied (idempotent)
                    stats['copied_images'] += 1

                stats['images_per_species'][label] += 1
                stats['splits'][split] += 1
                stats['species_set'].add(label)
            except Exception as e:
                stats['errors'].append({'src': src, 'dest': dest_path, 'error': str(e)})


def main():
    ensure_dir(os.path.dirname(REPORT_PATH) or '.')
    ensure_dir(TARGET_BASE)

    stats = {
        'total_images_processed': 0,
        'copied_images': 0,
        'missing_files': 0,
        'missing_files_list': [],
        'detected_species_count': 0,
        'images_per_species': defaultdict(int),
        'splits': defaultdict(int),
        'species_set': set(),
        'errors': [],
        'missing_meta': 0,
    }

    for split, csvname in CSV_FILES.items():
        process_csv(split, csvname, stats)

    report = {
        'total_images_processed': stats['total_images_processed'],
        'copied_images': stats['copied_images'],
        'missing_files': stats['missing_files'],
        'detected_species_count': len(stats['species_set']),
        'images_per_species': dict(stats['images_per_species']),
        'train_val_test_counts': dict(stats['splits']),
        'missing_files_list': stats['missing_files_list'],
        'errors': stats['errors'],
    }

    # write report
    ensure_dir(os.path.dirname(REPORT_PATH))
    with open(REPORT_PATH, 'w', encoding='utf-8') as out:
        json.dump(report, out, indent=2, ensure_ascii=False)

    print("Import finished.")
    print(json.dumps({
        'total': report['total_images_processed'],
        'copied': report['copied_images'],
        'missing': report['missing_files'],
        'species': report['detected_species_count'],
    }, indent=2))


if __name__ == '__main__':
    main()
