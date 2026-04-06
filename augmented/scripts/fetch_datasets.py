"""Download external keystroke datasets for cross-dataset validation.

Usage:
    uv run python scripts/fetch_datasets.py --list
    uv run python scripts/fetch_datasets.py --dataset cmu_benchmark
    uv run python scripts/fetch_datasets.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scholawrite.datasets import REGISTRY, DatasetAccessRequired


def _out(*args: object, **kwargs: object) -> None:
    """Write to stdout (CLI output, not debug logging)."""
    sys.stdout.write(" ".join(str(a) for a in args) + kwargs.get("end", "\n"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download external keystroke datasets.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="List all datasets with access info")
    group.add_argument("--dataset", type=str, help="Download one dataset by short name")
    group.add_argument("--all", action="store_true", help="Download all automatable datasets")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "external",
        help="Directory for downloaded data",
    )
    args = parser.parse_args()

    if args.list:
        _out(f"{'Short Name':<20} {'Access':<15} {'Name'}")
        _out("-" * 70)
        for name, loader in sorted(REGISTRY.items()):
            m = loader.METADATA
            _out(f"{m.short_name:<20} {m.access.value:<15} {m.name}")
        return

    args.data_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset:
        if args.dataset not in REGISTRY:
            _out(f"Unknown dataset: {args.dataset}")
            _out(f"Available: {', '.join(sorted(REGISTRY.keys()))}")
            sys.exit(1)
        _download_one(args.dataset, args.data_dir)
    elif args.all:
        for name in sorted(REGISTRY.keys()):
            _download_one(name, args.data_dir)


def _download_one(name: str, data_dir: Path) -> None:
    loader = REGISTRY[name]
    meta = loader.METADATA
    _out(f"\n=== {meta.name} ({meta.short_name}) ===")
    _out(f"    URL: {meta.url}")
    _out(f"    Access: {meta.access.value}")
    try:
        dest = loader.download(data_dir)
        _out(f"    Downloaded to: {dest}")
    except DatasetAccessRequired as e:
        _out(f"    SKIPPED (manual action required):")
        for line in str(e).split("\n"):
            _out(f"      {line}")
    except Exception as e:
        _out(f"    FAILED: {e}")


if __name__ == "__main__":
    main()
