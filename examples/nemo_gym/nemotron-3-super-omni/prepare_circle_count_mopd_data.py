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

"""Generate deterministic NeMo Gym circle-count rows for image MOPD."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

AGENT_REF = {
    "type": "responses_api_agents",
    "name": "circle_count_simple_agent",
}


def _load_circle_count_generator() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[3]
    generator_path = (
        repo_root
        / "3rdparty"
        / "Gym-workspace"
        / "Gym"
        / "resources_servers"
        / "circle_count"
        / "generate_data.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_nemo_gym_circle_count_generate_data", generator_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load circle-count generator: {generator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_example(example: dict[str, Any]) -> None:
    if example.get("agent_ref") != AGENT_REF:
        raise ValueError("circle-count MOPD row has an invalid agent_ref")

    responses_create_params = example.get("responses_create_params")
    if not isinstance(responses_create_params, dict):
        raise ValueError("row is missing responses_create_params")

    image_urls: list[str] = []
    for message in responses_create_params.get("input", []):
        content = message.get("content", []) if isinstance(message, dict) else []
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "input_image":
                image_urls.append(str(item.get("image_url", "")))

    if len(image_urls) != 1 or not image_urls[0].startswith("data:image/"):
        raise ValueError(
            "each circle-count MOPD row must contain exactly one data-URL input_image"
        )

    request_text = json.dumps(responses_create_params)
    if '"circles"' in request_text or '"target_color"' in request_text:
        raise ValueError("answer metadata leaked into responses_create_params")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate image MOPD data routed to circle_count_simple_agent."
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=1000)
    parser.add_argument("--radius-min", type=int, default=30)
    parser.add_argument("--radius-max", type=int, default=60)
    parser.add_argument("--num-circles-min", type=int, default=5)
    parser.add_argument("--num-circles-max", type=int, default=20)
    parser.add_argument("--num-colors-min", type=int, default=2)
    parser.add_argument("--num-colors-max", type=int, default=4)
    args = parser.parse_args()

    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")

    generator = _load_circle_count_generator()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as output:
        for index in range(args.num_samples):
            example = generator.make_example(
                args.seed_offset + index,
                img_size_range=(args.image_size, args.image_size),
                circle_radius_range=(args.radius_min, args.radius_max),
                num_circles_range=(
                    args.num_circles_min,
                    args.num_circles_max,
                ),
                num_colors_range=(args.num_colors_min, args.num_colors_max),
            )
            example["agent_ref"] = dict(AGENT_REF)
            _validate_example(example)
            output.write(json.dumps(example) + "\n")

    print(f"Generated {args.num_samples} image-MOPD rows: {args.out}")


if __name__ == "__main__":
    main()
