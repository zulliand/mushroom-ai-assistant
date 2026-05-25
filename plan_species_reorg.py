"""Dry-run planner for species-level CV dataset reorganization.

This script only analyzes the current dataset and prints/writes a plan.
It never moves, renames, copies, or deletes files.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
SPECIES_PATTERN = re.compile(r"^([A-Za-z]+)[_-]([A-Za-z]+)")
DEFAULT_ROOTS = [Path("data/train"), Path("data/val"), Path("data/test")]


def infer_binary_label(path: Path) -> str:
    """Infer edible/poisonous from parent folder names."""

    lowered_parts = [part.lower() for part in path.parts]
    if "edible" in lowered_parts:
        return "edible"
    if "poisonous" in lowered_parts:
        return "poisonous"
    return "unknown"


def infer_species_from_stem(stem: str) -> Optional[str]:
    """Extract Genus_species from a filename stem, if present."""

    match = SPECIES_PATTERN.match(stem)
    if not match:
        return None
    genus = match.group(1).capitalize()
    species = match.group(2).lower()
    return f"{genus}_{species}"


def iter_image_files(roots: Iterable[Path]) -> Iterable[Tuple[Path, str]]:
    """Yield image files and their split name from existing roots."""

    for root in roots:
        if not root.exists():
            continue
        split = root.name.lower()
        for candidate in root.rglob("*"):
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
                yield candidate, split


def resolve_species_label(edible_count: int, poisonous_count: int, unknown_count: int) -> str:
    """Resolve one species-to-label mapping from observed file locations."""

    if edible_count > 0 and poisonous_count == 0:
        return "edible"
    if poisonous_count > 0 and edible_count == 0:
        return "poisonous"
    if edible_count > 0 and poisonous_count > 0:
        return "mixed/conflict"
    if unknown_count > 0:
        return "unknown"
    return "unknown"


def build_report(
    roots: List[Path],
    target_root: Path,
    min_images: int,
    max_unknown_sample: int,
    max_plan_sample: int,
) -> Dict[str, Any]:
    """Build a full dry-run planning report from current dataset files."""

    rows: List[Dict[str, Any]] = []
    for file_path, split in iter_image_files(roots):
        species = infer_species_from_stem(file_path.stem)
        rows.append(
            {
                "source_path": str(file_path),
                "split": split,
                "binary_label": infer_binary_label(file_path),
                "species": species,
            }
        )

    detected_rows = [row for row in rows if row["species"] is not None]
    unknown_rows = [row for row in rows if row["species"] is None]

    species_counter: Counter[str] = Counter()
    by_species_label: Dict[str, Counter[str]] = defaultdict(Counter)
    by_species_split: Dict[str, Counter[str]] = defaultdict(Counter)

    for row in detected_rows:
        species = str(row["species"])
        species_counter[species] += 1
        by_species_label[species][str(row["binary_label"])] += 1
        by_species_split[species][str(row["split"])] += 1

    species_rows: List[Dict[str, Any]] = []
    for species, total in species_counter.items():
        edible_count = by_species_label[species]["edible"]
        poisonous_count = by_species_label[species]["poisonous"]
        unknown_label_count = by_species_label[species]["unknown"]
        species_rows.append(
            {
                "species": species,
                "total": total,
                "label": resolve_species_label(edible_count, poisonous_count, unknown_label_count),
                "edible_count": edible_count,
                "poisonous_count": poisonous_count,
                "unknown_label_count": unknown_label_count,
                "train_count": by_species_split[species]["train"],
                "val_count": by_species_split[species]["val"],
                "test_count": by_species_split[species]["test"],
            }
        )

    species_rows.sort(key=lambda item: (-int(item["total"]), str(item["species"])))
    top10 = species_rows[:10]
    too_few = sorted(
        [row for row in species_rows if int(row["total"]) < min_images],
        key=lambda item: (int(item["total"]), str(item["species"])),
    )

    enough_classes = sum(1 for row in species_rows if int(row["total"]) >= min_images)
    species_count = len(species_rows)
    feasible = species_count >= 10 and enough_classes >= max(8, int(species_count * 0.6))

    layout: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    operations_sample: List[Dict[str, str]] = []
    for row in detected_rows:
        species = str(row["species"])
        split = str(row["split"])
        source_path = Path(str(row["source_path"]))
        target_path = target_root / split / species / source_path.name
        layout[split][species] += 1
        if len(operations_sample) < max_plan_sample:
            operations_sample.append({"source": str(source_path), "target": str(target_path)})

    split_layout: Dict[str, Dict[str, int]] = {
        split: dict(sorted(species_counts.items(), key=lambda item: item[0]))
        for split, species_counts in layout.items()
    }

    summary = {
        "roots": [str(root) for root in roots if root.exists()],
        "total_images": len(rows),
        "detected_species_count": species_count,
        "unknown_filename_pattern_images": len(unknown_rows),
        "threshold_for_too_few": min_images,
        "species_with_enough_images": enough_classes,
        "feasible_species_training": feasible,
        "dry_run_only": True,
        "target_layout_root": str(target_root),
    }

    return {
        "summary": summary,
        "top10": top10,
        "species_with_too_few_images": too_few,
        "all_species": species_rows,
        "unknown_files_sample": [row["source_path"] for row in unknown_rows[:max_unknown_sample]],
        "planned_layout": split_layout,
        "planned_operations_sample": operations_sample,
    }


def print_summary(report: Dict[str, Any]) -> None:
    """Print concise CLI output from a generated report."""

    summary = report["summary"]
    print("Dry-run species reorganization planner")
    print(f"Roots: {', '.join(summary['roots'])}")
    print(f"Total images: {summary['total_images']}")
    print(f"Detected species: {summary['detected_species_count']}")
    print(f"Unknown filename pattern images: {summary['unknown_filename_pattern_images']}")
    print(f"Threshold (too few): < {summary['threshold_for_too_few']}")
    print(f"Species with enough images: {summary['species_with_enough_images']}")
    print(f"Species-level training feasible: {summary['feasible_species_training']}")

    print("\nTop 10 species:")
    for row in report["top10"]:
        print(f"- {row['species']}: {row['total']} images, label={row['label']}")

    if report["species_with_too_few_images"]:
        print("\nSpecies with too few images:")
        for row in report["species_with_too_few_images"]:
            print(f"- {row['species']}: {row['total']}")
    else:
        print("\nSpecies with too few images: none")


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""

    parser = argparse.ArgumentParser(
        description="Dry-run planner for reorganizing mushroom images by species.",
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        default=[str(path) for path in DEFAULT_ROOTS],
        help="Input dataset roots to scan (default: data/train data/val data/test)",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=Path("data/species"),
        help="Planned output root shown in dry-run operations.",
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=20,
        help="Species with fewer images than this are flagged as too few.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("models/species_dry_run_report.json"),
        help="Path to write the dry-run report JSON.",
    )
    parser.add_argument(
        "--max-unknown-sample",
        type=int,
        default=30,
        help="Maximum unknown-pattern files listed in report sample.",
    )
    parser.add_argument(
        "--max-plan-sample",
        type=int,
        default=50,
        help="Maximum planned source->target operations listed in report sample.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print full JSON report to stdout after summary.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    roots = [Path(root) for root in args.roots]

    report = build_report(
        roots=roots,
        target_root=args.target_root,
        min_images=int(args.min_images),
        max_unknown_sample=int(args.max_unknown_sample),
        max_plan_sample=int(args.max_plan_sample),
    )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print_summary(report)
    print(f"\nSaved dry-run report: {output_path}")
    if args.print_json:
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
