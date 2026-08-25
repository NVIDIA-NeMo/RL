#!/usr/bin/env python3
"""Build a self-contained image-tools GRPO smoke dataset.

The generated image and JSONL files use absolute paths so they can be mounted
unchanged into the training container. By default the training split contains
16 copies of one deterministic counting task, enough for a single Super Omni
GRPO step with the accompanying sample recipe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYSTEM_PROMPT = (
    REPO_ROOT / "examples/prompts/image_tools_system_prompt.txt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--train-repeats", type=int, default=16)
    parser.add_argument("--validation-repeats", type=int, default=1)
    return parser.parse_args()


def make_counting_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (768, 512), "white")
    draw = ImageDraw.Draw(image)
    colors = [
        "#e63946",
        "#f4a261",
        "#e9c46a",
        "#2a9d8f",
        "#457b9d",
        "#7b2cbf",
    ]
    positions = [(90, 90), (300, 90), (510, 90), (90, 300), (300, 300), (510, 300)]
    for color, (x, y) in zip(colors, positions, strict=True):
        draw.rectangle((x, y, x + 110, y + 110), fill=color, outline="black", width=6)
    image.save(path, format="PNG")


def make_row(*, row_id: str, image_path: Path, system_prompt: str) -> dict[str, Any]:
    return {
        "id": row_id,
        "agent_ref": {
            "type": "responses_api_agents",
            "name": "image_tools_simple_agent",
        },
        "image_tools_base_agent_ref": {
            "type": "responses_api_agents",
            "name": "string_match_simple_agent",
        },
        "expected_answer": "6",
        "extraction_mode": "final_answer",
        "responses_create_params": {
            "input": [
                {"role": "system", "type": "message", "content": system_prompt},
                {
                    "role": "user",
                    "type": "message",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Count the colored squares. Use count_objects_tool to verify "
                                "the count, then answer with \\boxed{<number>}."
                            ),
                        },
                        {
                            "type": "input_image",
                            "image_url": str(image_path.resolve()),
                            "detail": "auto",
                        },
                    ],
                },
            ],
            "tools": [],
            "parallel_tool_calls": False,
        },
        "dataset": "image_tools_grpo_sample",
        "source_tool_names": ["count_objects_tool"],
    }


def write_split(
    path: Path, *, count: int, image_path: Path, system_prompt: str
) -> None:
    if count < 1:
        raise ValueError("split repeat count must be positive")
    with path.open("w") as output:
        for index in range(count):
            row = make_row(
                row_id=f"image-tools-count-{index:04d}",
                image_path=image_path,
                system_prompt=system_prompt,
            )
            output.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


def build_sample(
    output_dir: Path,
    *,
    system_prompt_path: Path = DEFAULT_SYSTEM_PROMPT,
    train_repeats: int = 16,
    validation_repeats: int = 1,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "six_squares.png"
    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    system_prompt = system_prompt_path.read_text()

    make_counting_image(image_path)
    write_split(
        train_path,
        count=train_repeats,
        image_path=image_path,
        system_prompt=system_prompt,
    )
    write_split(
        validation_path,
        count=validation_repeats,
        image_path=image_path,
        system_prompt=system_prompt,
    )
    return {
        "image": str(image_path.resolve()),
        "train": str(train_path.resolve()),
        "validation": str(validation_path.resolve()),
        "train_rows": train_repeats,
        "validation_rows": validation_repeats,
    }


def main() -> None:
    args = parse_args()
    result = build_sample(
        args.output_dir,
        system_prompt_path=args.system_prompt,
        train_repeats=args.train_repeats,
        validation_repeats=args.validation_repeats,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
