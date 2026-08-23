#!/usr/bin/env python3
"""Remove checkpoint payloads while preserving OSWorld experiment logs."""

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Delete checkpoint directories; without this flag, only list them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = args.results_root.resolve(strict=True)
    checkpoint_dirs = sorted(
        path
        for path in results_root.glob("*/checkpoints")
        if path.is_dir() and not path.is_symlink() and path.parent.parent == results_root
    )

    if not checkpoint_dirs:
        print("No OSWorld checkpoint directories found.")
        return

    action = "Deleting" if args.execute else "Would delete"
    for checkpoint_dir in checkpoint_dirs:
        print(f"{action}: {checkpoint_dir}")
        if args.execute:
            shutil.rmtree(checkpoint_dir)

    print(f"{action} {len(checkpoint_dirs)} checkpoint directories.")


if __name__ == "__main__":
    main()
