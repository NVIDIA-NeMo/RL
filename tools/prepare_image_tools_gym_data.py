#!/usr/bin/env python3
"""Prepare NeMo-Gym VLM rows for image-zoom wrapper training."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(
    "/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/"
    "datasets/eagle-next/image_data/rl_data/random_blend_v8_gym.jsonl"
)
DEFAULT_SYSTEM_PROMPT = Path("examples/prompts/image_zoom_tool_system_prompt.txt")
TOOLS_AGENT_REF = {"type": "responses_api_agents", "name": "image_tools_simple_agent"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, default=None)
    parser.add_argument("--validation-size", type=int, default=512)
    parser.add_argument(
        "--system-prompt-file", type=Path, default=DEFAULT_SYSTEM_PROMPT
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def _as_message_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        return [{"role": "user", "type": "message", "content": value}]
    if isinstance(value, list):
        return value
    raise TypeError(
        f"responses_create_params.input must be str or list, got {type(value).__name__}"
    )


def _prepend_system_prompt(
    input_messages: list[dict[str, Any]], system_prompt: str
) -> list[dict[str, Any]]:
    messages = list(input_messages)
    while messages and messages[0].get("role") == "system":
        messages.pop(0)
    return [
        {
            "role": "system",
            "type": "message",
            "content": system_prompt,
        },
        *messages,
    ]


def convert_row(row: dict[str, Any], system_prompt: str) -> dict[str, Any]:
    converted = dict(row)
    base_agent_ref = dict(row.get("agent_ref") or {})
    if not base_agent_ref.get("name"):
        raise ValueError(f"Missing agent_ref.name in row id={row.get('id')!r}")

    responses_create_params = dict(converted.get("responses_create_params") or {})
    responses_create_params["input"] = _prepend_system_prompt(
        _as_message_list(responses_create_params.get("input", [])),
        system_prompt,
    )

    converted["responses_create_params"] = responses_create_params
    converted["image_tools_base_agent_ref"] = base_agent_ref
    converted["agent_ref"] = dict(TOOLS_AGENT_REF)
    return converted


def main() -> None:
    args = parse_args()
    system_prompt = args.system_prompt_file.read_text().strip()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.validation_output is not None:
        args.validation_output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    with args.input.open() as f:
        for idx, line in enumerate(f):
            if args.limit is not None and idx >= args.limit:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            base_name = row.get("agent_ref", {}).get("name", "<missing>")
            counts[base_name] = counts.get(base_name, 0) + 1
            rows.append(convert_row(row, system_prompt))

    with args.output.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    validation_rows: list[dict[str, Any]] = []
    if args.validation_output is not None and args.validation_size > 0:
        rng = random.Random(args.seed)
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        validation_rows = [
            rows[i] for i in indices[: min(args.validation_size, len(rows))]
        ]
        with args.validation_output.open("w") as f:
            for row in validation_rows:
                f.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )

    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "rows": len(rows),
                "validation_output": str(args.validation_output)
                if args.validation_output
                else None,
                "validation_rows": len(validation_rows),
                "base_agent_counts": counts,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
