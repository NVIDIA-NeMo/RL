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
"""Adapt source-owned MM Coding task records to single, multi-harness or mixed Gym training rows.

Dataset acquisition, source-specific conversion and split selection remain with
the source repository. This adapter preserves prepared prompts and images and
keeps reference code in verifier-owned task data.
"""

import argparse
import base64
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExperimentTask(BaseModel):
    """The prepared task format produced by nemo-rl-mm-coding."""

    model_config = ConfigDict(extra="forbid")
    benchmark: Literal[
        "chart2code",
        "mmsvg_img2svg",
        "mmsvg_text2svg",
        "image2struct_latex",
        "design2code_full",
    ]
    task_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    references: list[str] = Field(default_factory=list)
    semantic_prompt: str = ""
    ground_truth: str = ""
    original_source: str = ""
    reference_patch: str = ""
    category: str = ""
    judge_references: list[str] = Field(default_factory=list)
    vision_manifest: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_task(self) -> "ExperimentTask":
        expected = 0 if self.benchmark == "mmsvg_text2svg" else 1
        if len(self.references) != expected or not self.prompt.strip():
            raise ValueError("A visual task needs one input image; Text2SVG needs none")
        if (
            self.original_source
            or self.reference_patch
            or self.judge_references
            or self.vision_manifest
        ):
            raise ValueError(
                "This adapter supports only the four visual-coding task families"
            )
        return self


class SourceGymRow(BaseModel):
    """The source experiment's existing flat JSONL envelope, consumed only for its task."""

    model_config = ConfigDict(extra="forbid")
    agent_ref: dict[str, str]
    task: ExperimentTask
    route: Literal["single", "opencode", "codex", "hermes"]
    responses_create_params: dict[str, object]

    def validate_image_binding(self, expected_sha256: str | None) -> None:
        """Reject drift between a legacy single row's inline image and its current asset."""
        if self.route != "single":
            return
        images = []
        messages = self.responses_create_params.get("input")
        if isinstance(messages, list):
            for message in messages:
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                if not isinstance(content, list):
                    continue
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "input_image":
                        images.append(part.get("image_url"))
        if len(images) != (1 if expected_sha256 is not None else 0):
            raise ValueError("Legacy input image count differs from the prepared task")
        if not images:
            return
        value = images[0]
        if (
            not isinstance(value, str)
            or not value.startswith("data:image/")
            or ";base64," not in value
        ):
            raise ValueError(
                "Legacy reference must be an inline image, not a network URL"
            )
        encoded = value.split(";base64,", 1)[1]
        if len(encoded) > 45 * 1024 * 1024:
            raise ValueError("Legacy reference exceeds the image budget")
        raw = base64.b64decode(encoded, validate=True)
        if (
            len(raw) > 32 * 1024 * 1024
            or hashlib.sha256(raw).hexdigest() != expected_sha256
        ):
            raise ValueError(
                "Legacy reference image bytes differ from the current asset"
            )


HARNESS_NAMES = ("opencode", "codex", "hermes")
MULTI_FRACTIONS = {
    "chart2code": 0.5,
    "mmsvg_img2svg": 0.5,
    "mmsvg_text2svg": 0.25,
    "image2struct_latex": 0.75,
    "design2code_full": 0.5,
}
ARTIFACTS = {
    "chart2code": "answer.py",
    "mmsvg_img2svg": "answer.svg",
    "mmsvg_text2svg": "answer.svg",
    "image2struct_latex": "answer.tex",
    "design2code_full": "answer.html",
}


def select_route(
    task: ExperimentTask,
    *,
    mode: str,
    seed: int = 42,
    epoch: int = 0,
    harnesses: tuple[str, ...] = HARNESS_NAMES,
) -> str:
    """Freeze the source protocol's deterministic routing before sibling expansion."""
    if mode not in {"single", "multi", "mixed"}:
        raise ValueError("mode must be single, multi or mixed")
    if (
        not harnesses
        or len(set(harnesses)) != len(harnesses)
        or set(harnesses) - set(HARNESS_NAMES)
    ):
        raise ValueError("Choose unique supported harnesses")
    identity = f"{seed}:{epoch}:{task.benchmark}:{task.task_id}"
    routing_bits = hashlib.sha256(identity.encode()).hexdigest()
    multi = mode == "multi" or (
        mode == "mixed"
        and int(routing_bits[:16], 16) / 2**64 < MULTI_FRACTIONS[task.benchmark]
    )
    return (
        harnesses[int(routing_bits[16:32], 16) % len(harnesses)] if multi else "single"
    )


def gym_row(
    task: ExperimentTask,
    *,
    reference_root: Path,
    split: Literal["train", "validation", "example"],
    max_output_tokens: int,
    selected: str = "single",
) -> dict:
    """Preserve the task prompt and input image without exposing the reference source."""
    if selected != "single" and selected not in HARNESS_NAMES:
        raise ValueError("Unknown harness route")
    if split not in {"train", "validation", "example"}:
        raise ValueError(
            "Choose an approved train/validation split; held-out evaluation data is not training input"
        )
    if not 1 <= max_output_tokens <= 131072:
        raise ValueError("max_output_tokens must be between 1 and 131072")
    content = [{"type": "input_text", "text": task.prompt}]
    metadata = {
        "benchmark": task.benchmark,
        "task_id": task.task_id,
        "prompt": task.prompt,
        "source_split": split,
        "semantic_prompt": task.semantic_prompt,
        "ground_truth": task.ground_truth,
    }
    if task.references:
        root = reference_root.resolve()
        path = Path(task.references[0])
        path = (path if path.is_absolute() else root / path).resolve()
        if (
            not path.is_relative_to(root)
            or not path.is_file()
            or path.stat().st_size > 32 * 1024 * 1024
        ):
            raise ValueError(
                "Input reference must be a bounded file under reference_root"
            )
        raw = path.read_bytes()
        with Image.open(path) as image:
            if (
                image.format not in {"PNG", "JPEG", "WEBP"}
                or image.width * image.height > 32_000_000
            ):
                raise ValueError("Reference must be a bounded PNG, JPEG or WebP image")
            mime = Image.MIME[image.format]
            image.verify()
        metadata.update(
            reference_image=path.relative_to(root).as_posix(),
            reference_sha256=hashlib.sha256(raw).hexdigest(),
        )
        content.append(
            {
                "type": "input_image",
                "image_url": f"data:{mime};base64," + base64.b64encode(raw).decode(),
                "detail": "auto",
            }
        )
    if selected != "single":
        # The resource-owned sandbox receives the same reference bytes; no private grade data enters the prompt.
        content = [{"type": "input_text", "text": task.prompt}]
        if task.references:
            content.append(
                {
                    "type": "input_text",
                    "text": "The task reference image is /workspace/reference"
                    + path.suffix.lower()
                    + ". Inspect it with your image tool before solving.",
                }
            )
        artifact = ARTIFACTS[task.benchmark]
        system_prompt = (
            f"Solve the user's task in /workspace. Save your final answer to /workspace/{artifact}. "
            "Use shell and file tools to inspect and revise. "
            f"Preview with `python /opt/mmrl/render.py {task.benchmark} /workspace/{artifact} /workspace/preview.png`. "
            "You may inspect the preview with your image tool. Finish with a final response. "
            "Only the final answer file is transferred to a fresh grading sandbox. "
            "Make it self-contained: it must run without any other task-workspace files. "
            "Do not read or embed the task reference image as the solution. "
            "Do not change the task requirements."
        )
        return {
            "task_id": {
                "taskset": f"mm-coding:{selected}",
                "task_id": f"{task.benchmark}:{task.task_id}",
            },
            "task_input": {
                "task_data": {"task": metadata},
                "responses_create_params": {
                    "input": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": content},
                    ],
                    "max_output_tokens": max_output_tokens,
                    "temperature": 1.0,
                    "tools": [],
                },
            },
        }
    return {
        "agent_ref": {
            "type": "responses_api_agents",
            "name": "mm_coding_training_simple_agent",
        },
        "task": metadata,
        "responses_create_params": {
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": max_output_tokens,
            "temperature": 1.0,
            "tools": [],
        },
    }


def export(
    input_path: Path,
    output_path: Path,
    *,
    reference_root: Path,
    split: Literal["train", "validation", "example"],
    max_output_tokens: int,
    input_format: Literal["task", "legacy-row"] = "task",
    mode: str = "single",
    seed: int = 42,
    epoch: int = 0,
    harnesses: tuple[str, ...] = HARNESS_NAMES,
) -> int:
    """Write once and reject duplicate task identities or malformed prepared records."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[str, str]] = set()
    with tempfile.NamedTemporaryFile(
        mode="w", dir=output_path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            with input_path.open() as source:
                for line in source:
                    if not line.strip():
                        continue
                    legacy_row = None
                    if input_format == "task":
                        task = ExperimentTask.model_validate_json(line)
                    elif input_format == "legacy-row":
                        legacy_row = SourceGymRow.model_validate_json(line)
                        task = legacy_row.task
                    else:
                        raise ValueError("input_format must be task or legacy-row")
                    identity = (task.benchmark, task.task_id)
                    if identity in seen:
                        raise ValueError(f"Duplicate task identity: {identity}")
                    seen.add(identity)
                    row = gym_row(
                        task,
                        reference_root=reference_root,
                        split=split,
                        max_output_tokens=max_output_tokens,
                        selected=select_route(
                            task, mode=mode, seed=seed, epoch=epoch, harnesses=harnesses
                        ),
                    )
                    if legacy_row is not None:
                        metadata = (
                            row["task_input"]["task_data"]["task"]
                            if "task_input" in row
                            else row["task"]
                        )
                        legacy_row.validate_image_binding(
                            metadata.get("reference_sha256")
                        )
                    stream.write(json.dumps(row) + "\n")
            if not seen:
                raise ValueError("Input task collection is empty")
            stream.flush()
            os.fsync(stream.fileno())
            os.link(temporary, output_path)
        finally:
            temporary.unlink(missing_ok=True)
    return len(seen)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument(
        "--split", choices=["train", "validation", "example"], required=True
    )
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument(
        "--input-format", choices=["task", "legacy-row"], default="task"
    )
    parser.add_argument(
        "--mode", choices=["single", "multi", "mixed"], default="single"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument(
        "--harnesses", nargs="+", choices=HARNESS_NAMES, default=list(HARNESS_NAMES)
    )
    args = parser.parse_args()
    count = export(
        args.input,
        args.output,
        reference_root=args.reference_root,
        split=args.split,
        max_output_tokens=args.max_output_tokens,
        input_format=args.input_format,
        mode=args.mode,
        seed=args.seed,
        epoch=args.epoch,
        harnesses=tuple(args.harnesses),
    )
    print(json.dumps({"rows": count, "output": str(args.output), "split": args.split}))


if __name__ == "__main__":
    main()
