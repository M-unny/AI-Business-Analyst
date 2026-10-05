"""Milestone 1 demo:  python main.py <file.csv|file.xlsx> [--json]"""
import argparse
import json
import sys

from app.data.loader import DatasetLoadError, load_dataset_from_path
from app.data.profiler import profile_dataset


def main() -> int:
    ap = argparse.ArgumentParser(description="Profile a CSV/Excel dataset.")
    ap.add_argument("path")
    ap.add_argument("--json", action="store_true", help="print the full profile as JSON")
    args = ap.parse_args()

    try:
        loaded = load_dataset_from_path(args.path)
    except DatasetLoadError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    profile = profile_dataset(
        loaded.df, file_name=loaded.file_name, encoding=loaded.encoding,
        extra_warnings=loaded.warnings,
    )
    if args.json:
        print(json.dumps(profile.model_dump(mode="json"), indent=2))
        return 0

    m = profile.metadata
    print(f"\n{m.file_name}: {m.rows:,} rows x {m.columns} columns "
          f"({profile.memory_usage_mb} MB, encoding={m.encoding})")
    print(f"Missing: {profile.missing_pct}%  Duplicates: {m.duplicate_rows} "
          f"({profile.duplicate_pct}%)  Quality score: {profile.quality_score}/100\n")
    print(f"{'column':<22}{'type':<13}{'missing%':>9}{'unique':>9}  role")
    print("-" * 70)
    for c in profile.columns:
        role = profile.semantic.primary.get(c.name)
        role_txt = f"{role.role} ({role.confidence:.2f})" if role else "-"
        print(f"{c.name:<22}{c.inferred_type:<13}{c.missing_pct:>9}{c.unique_count:>9}  {role_txt}")
    if profile.warnings:
        print("\nWarnings:")
        for w in profile.warnings:
            print(f"  - {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
