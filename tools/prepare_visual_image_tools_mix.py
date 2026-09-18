# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Combine existing training splits without changing rows or sampling weights."""

import argparse
import json
from pathlib import Path

from tools.check_visual_image_tools_mix import audit_mix, read_rows


def prepare(*, games: Path, images: Path, validation: Path, output: Path) -> dict:
    """Audit before writing; preserve row contents and refuse to overwrite."""
    rows = read_rows([games, images])
    report = audit_mix(
        train=rows, validation=read_rows([validation]), max_output_tokens=512
    )
    with output.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(**vars(args)), indent=2))
