#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prepare public Hugging Face MMPR-Tiny data for the existing GRPO loader.

This script converts a raw `hf download` snapshot of `OpenGVLab/MMPR-Tiny`
into the cache layout expected by
`nemo_rl.data.datasets.response_datasets.mmpr_tiny`.

Input directory requirements:
- `images.zip`
- `mmpr_tiny.parquet`

Output directory layout:
- `<output_dir>/MMPR-Tiny/images/...`
- `<output_dir>/mmpr_tiny.parquet`
- `<output_dir>/mmpr_tiny_preview.jsonl`
- `<output_dir>/prepare_mmpr_tiny_for_vision_rl_summary.json`
- `<output_dir>/.mmpr_ready`
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw MMPR-Tiny download into a GRPO-ready local cache."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing raw MMPR-Tiny files from `hf download`.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Target directory to write the processed MMPR-Tiny cache into.",
    )
    parser.add_argument(
        "--preview-name",
        type=str,
        default="mmpr_tiny_preview.jsonl",
        help="Name of the generated inspection JSONL inside --output-dir.",
    )
    parser.add_argument(
        "--summary-name",
        type=str,
        default="prepare_mmpr_tiny_for_vision_rl_summary.json",
        help="Name of the generated summary JSON file inside --output-dir.",
    )
    return parser.parse_args()


def ensure_required_inputs(input_dir: Path) -> dict[str, Path]:
    paths = {
        "images_zip": input_dir / "images.zip",
        "parquet": input_dir / "mmpr_tiny.parquet",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required MMPR-Tiny inputs: {missing}")
    return paths


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def flatten_prompt_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""

    text_parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            if isinstance(item.get("text"), str):
                text = item["text"].strip()
            elif isinstance(item.get("content"), str):
                text = item["content"].strip()
            else:
                text = ""
        else:
            text = ""

        if text:
            text_parts.append(text)

    return "\n".join(text_parts).strip()


def extract_question(prompt_value: Any) -> str:
    if not isinstance(prompt_value, list):
        return ""

    for message in prompt_value:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "user":
            continue

        question = flatten_prompt_content(message.get("content"))
        if question:
            return question

    return ""


def extract_answer(reward_model_value: Any) -> str:
    if not isinstance(reward_model_value, dict):
        return ""
    return normalize_text(reward_model_value.get("ground_truth"))


def normalize_cache_image_path(path_value: str) -> str:
    normalized = path_value.replace("\\", "/").strip().lstrip("/")
    if not normalized:
        return ""

    if "/MMPR-Tiny/" in normalized:
        normalized = "MMPR-Tiny/" + normalized.split("/MMPR-Tiny/", 1)[1]
    elif normalized.startswith("MMPR-Tiny/"):
        pass
    elif normalized.startswith("images/"):
        normalized = f"MMPR-Tiny/{normalized}"
    elif "/images/" in normalized:
        normalized = "MMPR-Tiny/images/" + normalized.split("/images/", 1)[1]
    else:
        normalized = f"MMPR-Tiny/images/{PurePosixPath(normalized).name}"

    return PurePosixPath(normalized).as_posix()


def extract_first_image_path(images_value: Any) -> tuple[str | None, int]:
    if not isinstance(images_value, list) or not images_value:
        return None, 0

    raw_paths: list[str] = []
    for image_value in images_value:
        if isinstance(image_value, dict) and isinstance(image_value.get("path"), str):
            raw_paths.append(image_value["path"])
        elif isinstance(image_value, str):
            raw_paths.append(image_value)

    if not raw_paths:
        return None, 0

    normalized_path = normalize_cache_image_path(raw_paths[0])
    if not normalized_path:
        return None, len(raw_paths)

    return normalized_path, len(raw_paths)


def find_extracted_images_dir(temp_dir: Path) -> Path:
    direct_candidates = [
        temp_dir / "images",
        temp_dir / "MMPR-Tiny" / "images",
    ]
    for candidate in direct_candidates:
        if candidate.is_dir():
            return candidate

    candidates = sorted(
        (path for path in temp_dir.rglob("images") if path.is_dir()),
        key=lambda path: (len(path.parts), str(path)),
    )
    if not candidates:
        raise FileNotFoundError("Could not locate extracted `images/` directory in zip.")
    return candidates[0]


def extract_images_if_needed(images_zip_path: Path, output_dir: Path) -> Path:
    images_dir = output_dir / "MMPR-Tiny" / "images"
    if images_dir.exists():
        print(f"Images already extracted at {images_dir}")
        return images_dir

    temp_dir = output_dir / "_mmpr_tiny_extract_tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting images from {images_zip_path} to {images_dir}...")
    with zipfile.ZipFile(images_zip_path) as zf:
        members = zf.infolist()
        with tqdm(
            members,
            total=len(members),
            desc="Extracting MMPR-Tiny images",
            unit="file",
        ) as progress:
            for member in progress:
                zf.extract(member, temp_dir)

    extracted_images_dir = find_extracted_images_dir(temp_dir)
    images_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(extracted_images_dir), str(images_dir))
    shutil.rmtree(temp_dir, ignore_errors=True)
    print("Finished extracting images.")
    return images_dir


def copy_parquet(input_parquet_path: Path, output_dir: Path) -> Path:
    output_parquet_path = output_dir / "mmpr_tiny.parquet"
    shutil.copy2(input_parquet_path, output_parquet_path)
    return output_parquet_path


def build_preview_jsonl(parquet_path: Path, output_dir: Path, preview_path: Path) -> dict[str, int]:
    df = pd.read_parquet(parquet_path)
    stats: Counter[str] = Counter()
    stats["total_rows"] = len(df)

    preview_df = df[["prompt", "reward_model", "images"]]
    with preview_path.open("w", encoding="utf-8") as fout:
        iterator = preview_df.itertuples(index=False, name=None)
        for prompt_value, reward_model_value, images_value in tqdm(
            iterator,
            total=len(preview_df),
            desc="Building MMPR-Tiny preview",
            unit="row",
        ):
            question = extract_question(prompt_value)
            if not question:
                stats["missing_question"] += 1

            answer = extract_answer(reward_model_value)
            if not answer:
                stats["missing_answer"] += 1

            image_rel_path, image_count = extract_first_image_path(images_value)
            if image_rel_path is None:
                stats["invalid_image_field"] += 1
                continue

            if image_count > 1:
                stats["multi_image_rows"] += 1

            image_abs_path = output_dir / image_rel_path
            if not image_abs_path.exists():
                stats["missing_image_file"] += 1
                continue

            preview_record = {
                "images": [image_rel_path],
                "question": question,
                "answer": answer,
                "task_name": "mmpr_tiny",
            }
            fout.write(json.dumps(preview_record, ensure_ascii=False) + "\n")
            stats["preview_rows"] += 1

    return dict(stats)


def write_ready_marker(output_dir: Path) -> Path:
    ready_marker = output_dir / ".mmpr_ready"
    ready_marker.write_text("ready\n", encoding="utf-8")
    return ready_marker


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    paths = ensure_required_inputs(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_parquet_path = copy_parquet(paths["parquet"], output_dir)
    images_dir = extract_images_if_needed(paths["images_zip"], output_dir)

    preview_path = output_dir / args.preview_name
    stats = build_preview_jsonl(output_parquet_path, output_dir, preview_path)
    ready_marker = write_ready_marker(output_dir)

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "parquet_path": str(output_parquet_path),
        "images_dir": str(images_dir),
        "preview_path": str(preview_path),
        "ready_marker": str(ready_marker),
        "totals": stats,
    }
    summary_path = output_dir / args.summary_name
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote processed parquet to {output_parquet_path}")
    print(f"Wrote preview JSONL to {preview_path}")
    print(f"Wrote summary to {summary_path}")
    print(f"Wrote ready marker to {ready_marker}")
    print(
        "Prepared "
        f"{stats.get('preview_rows', 0)} / {stats.get('total_rows', 0)} MMPR-Tiny rows "
        f"for cache validation."
    )

    skipped = {
        key: value
        for key, value in stats.items()
        if key
        not in {
            "total_rows",
            "preview_rows",
        }
        and value
    }
    if skipped:
        print("Rows flagged during validation:")
        for reason, count in sorted(skipped.items()):
            print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
