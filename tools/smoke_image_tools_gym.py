#!/usr/bin/env python3
"""Replay all image-tool executors against a tiny NeMo-Gym dataset."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from resources_servers.image_tools import (
    IMAGE_TOOL_NAMES,
    ImageToolsGymToolLogic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _initial_images(row: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for message in row.get("responses_create_params", {}).get("input", []):
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "input_image":
                paths.append(str(part.get("image_url", "")))
    if not paths:
        raise ValueError("Smoke dataset's first row has no input images")
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing smoke images: {missing}")
    return paths


def _xml_tool_call(name: str, arguments: dict[str, Any]) -> str:
    parameters = "".join(
        f"<parameter={key}>{json.dumps(value)}</parameter>"
        for key, value in arguments.items()
    )
    return f"<tool_call><function={name}>{parameters}</function></tool_call>"


def _smoke_calls(initial_image_count: int) -> list[tuple[str, dict[str, Any]]]:
    first_output = initial_image_count
    return [
        (
            "image_crop_tool",
            {"bbox_2d": [0, 0, 500, 500], "label": "smoke crop", "img_idx": 0},
        ),
        (
            "image_zoom_in_tool",
            {
                "bbox_2d": [0, 0, 1000, 1000],
                "factor": 2,
                "label": "smoke zoom",
                "img_idx": first_output,
            },
        ),
        (
            "image_rotate_tool",
            {"degrees": 90, "label": "smoke rotate", "img_idx": first_output + 1},
        ),
        (
            "image_flip_tool",
            {"axis": "horizontal", "label": "smoke flip", "img_idx": first_output + 2},
        ),
        (
            "image_diff_tool",
            {"img_idx_a": 0, "img_idx_b": first_output + 3, "label": "smoke diff"},
        ),
        (
            "image_side_by_side_tool",
            {
                "img_indices": [0, first_output, first_output + 4],
                "labels": ["original", "crop", "diff"],
                "label": "smoke comparison",
            },
        ),
        (
            "image_overlay_tool",
            {
                "img_idx_a": 0,
                "img_idx_b": first_output + 3,
                "alpha": 0.5,
                "label": "smoke overlay",
            },
        ),
        (
            "count_objects_tool",
            {"min_size": 30, "label": "smoke count", "img_idx": 0},
        ),
        (
            "find_color_tool",
            {
                "color": [255, 255, 255],
                "tolerance": 20,
                "label": "smoke color",
                "img_idx": 0,
            },
        ),
        (
            "color_at_tool",
            {"point_2d": [500, 500], "label": "smoke pixel", "img_idx": 0},
        ),
    ]


def run_smoke(dataset: Path, output_dir: Path) -> dict[str, Any]:
    with dataset.open() as source:
        rows = [json.loads(line) for line in source if line.strip()]
    if not rows:
        raise ValueError(f"Dataset is empty: {dataset}")
    covered = set().union(*(set(row.get("source_tool_names", [])) for row in rows))
    if covered != IMAGE_TOOL_NAMES:
        raise ValueError(
            f"Dataset tool coverage mismatch: missing={sorted(IMAGE_TOOL_NAMES - covered)} "
            f"extra={sorted(covered - IMAGE_TOOL_NAMES)}"
        )

    image_paths = _initial_images(rows[0])
    metadata: dict[str, Any] = {
        "ground_truth": rows[0].get("expected_answer", ""),
        "image_paths": image_paths,
        "dataset": "image-tools-smoke",
    }
    logic = ImageToolsGymToolLogic(
        {
            "crop_dir": str(output_dir),
            "crop_format": "png",
            "crop_min_pixels": 32 * 32,
            "crop_max_pixels": 1024 * 1024,
            "max_tool_calls": len(IMAGE_TOOL_NAMES),
            "max_tool_calls_per_turn": 1,
        }
    )
    executed: list[str] = []
    for expected_new_idx, (name, arguments) in enumerate(
        _smoke_calls(len(image_paths)), start=len(image_paths)
    ):
        observation, _, done, _, next_metadata, _, _ = logic.process_nonterminal_turn(
            [{"role": "assistant", "content": _xml_tool_call(name, arguments)}],
            metadata,
        )
        if done or next_metadata is None:
            raise AssertionError(
                f"{name} unexpectedly terminated the smoke rollout: {observation}"
            )
        payload = json.loads(observation["content"][1]["text"])
        if not payload.get("ok") or payload.get("new_img_indices") != [
            expected_new_idx
        ]:
            raise AssertionError(f"Unexpected {name} response: {payload}")
        output_path = Path(observation["content"][2]["image"])
        if not output_path.is_file():
            raise AssertionError(f"{name} did not produce an image: {output_path}")
        metadata = next_metadata
        executed.append(name)

    if set(executed) != IMAGE_TOOL_NAMES:
        raise AssertionError(f"Executor coverage mismatch: {executed}")
    return {
        "dataset": str(dataset),
        "dataset_rows": len(rows),
        "executed_tools": executed,
        "initial_images": len(image_paths),
        "generated_images": len(metadata.get("crop_paths", [])),
        "final_image_store_size": len(metadata.get("image_paths", [])),
        "ok": True,
    }


def main() -> None:
    args = parse_args()
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        result = run_smoke(args.dataset, args.output_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="image-tools-smoke-") as temp_dir:
            result = run_smoke(args.dataset, Path(temp_dir))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
