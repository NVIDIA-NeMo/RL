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

"""Parse the aligned NeMo RL, Search-R1, and ZeroSearch benchmark logs."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any


ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
MEASURED_STEPS = (2, 3, 4)


def _clean(text: str) -> str:
    return ANSI_ESCAPE.sub("", text)


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _summarize(steps: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [step for step in steps if step["step"] in MEASURED_STEPS]
    if len(measured) != len(MEASURED_STEPS):
        found = [step["step"] for step in measured]
        raise ValueError(
            f"Expected measured steps {list(MEASURED_STEPS)}, found {found}"
        )

    scalar_keys = (
        "core_step_s",
        "generated_tokens",
        "generation_output_tokens_per_s",
        "e2e_output_tokens_per_s",
        "reward",
        "search_calls",
    )
    means = {}
    for key in scalar_keys:
        values = [float(step[key]) for step in measured if key in step]
        if values:
            means[key] = _mean(values)
    stage_names = measured[0]["stages_s"].keys()
    stage_means = {
        name: _mean([float(step["stages_s"][name]) for step in measured])
        for name in stage_names
    }
    stage_total = sum(stage_means.values())
    stage_shares = {
        name: value / stage_total if stage_total else 0.0
        for name, value in stage_means.items()
    }
    means["stages_s"] = stage_means
    means["stage_shares"] = stage_shares
    means["step_stdev_s"] = statistics.stdev(
        [float(step["core_step_s"]) for step in measured]
    )
    checkpoint_values = [
        float(step["checkpoint_s"])
        for step in steps
        if float(step.get("checkpoint_s", 0.0)) > 0.0
    ]
    result: dict[str, Any] = {"steps": steps, "measured_mean": means}
    if checkpoint_values:
        result["separate_checkpoint_s"] = max(checkpoint_values)
    return result


def _parse_metric_segments(line: str) -> tuple[int, dict[str, float]] | None:
    match = re.search(r"(?:^|\s)step:(\d+)\s+-\s+", line)
    if match is None:
        return None
    step = int(match.group(1))
    metrics: dict[str, float] = {}
    for segment in line[match.end() :].split(" - "):
        if ":" not in segment:
            continue
        key, raw_value = segment.rsplit(":", 1)
        if re.fullmatch(NUMBER, raw_value):
            metrics[key] = float(raw_value)
    if "timing_s/step" not in metrics:
        return None
    return step, metrics


def parse_verl_log(path: Path, samples_per_step: int) -> list[dict[str, Any]]:
    """Parse one Search-R1 or ZeroSearch veRL training log."""
    steps: list[dict[str, Any]] = []
    for line in _clean(path.read_text(encoding="utf-8", errors="replace")).splitlines():
        parsed = _parse_metric_segments(line)
        if parsed is None:
            continue
        step_number, metrics = parsed
        search_s = metrics.get("timing_s/search", 0.0)
        generation_s = metrics["timing_s/gen"]
        checkpoint_s = metrics.get("timing_s/save_checkpoint", 0.0)
        core_step_s = metrics["timing_s/step"] - checkpoint_s
        stages = {
            "generation": max(0.0, generation_s - search_s),
            "search": search_s,
            "policy_logprob": metrics.get("timing_s/policy_logprob", 0.0),
            "reference_logprob": metrics.get("timing_s/ref", 0.0),
            "reward": metrics.get("timing_s/reward", 0.0),
            "advantage": metrics.get("timing_s/advantage", 0.0),
            "actor_update": metrics.get("timing_s/update_actor", 0.0),
        }
        stages["other"] = max(0.0, core_step_s - sum(stages.values()))
        generated_tokens = metrics["state_tokens/total"]
        response_tokens = metrics["response_length/mean"] * samples_per_step
        steps.append(
            {
                "step": step_number,
                "core_step_s": core_step_s,
                "raw_step_s": metrics["timing_s/step"],
                "checkpoint_s": checkpoint_s,
                "generated_tokens": generated_tokens,
                "response_tokens_including_observations": response_tokens,
                "generation_output_tokens_per_s": generated_tokens / generation_s,
                "e2e_output_tokens_per_s": generated_tokens / core_step_s,
                "reward": metrics.get("critic/score/mean", 0.0),
                "search_calls": metrics.get("env/search_calls", 0.0),
                "stages_s": stages,
            }
        )
    return steps


def _extract_float(section: str, pattern: str, *, required: bool = True) -> float:
    match = re.search(pattern, section, flags=re.MULTILINE)
    if match is None:
        if required:
            raise ValueError(f"Missing metric matching {pattern!r}")
        return 0.0
    return float(match.group(1))


def parse_nemo_log(path: Path, samples_per_step: int) -> list[dict[str, Any]]:
    """Parse one NeMo RL GRPO training log."""
    text = _clean(path.read_text(encoding="utf-8", errors="replace"))
    markers = list(re.finditer(r"^=+ Step (\d+)/(\d+) =+$", text, re.MULTILINE))
    steps: list[dict[str, Any]] = []
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        section = text[marker.end() : end]
        step_number = int(marker.group(1))
        raw_step_s = _extract_float(
            section, rf"^\s*• Total step time:\s*({NUMBER})s"
        )
        checkpoint_s = _extract_float(
            section, rf"^\s*• checkpointing:\s*({NUMBER})s", required=False
        )
        core_step_s = raw_step_s - checkpoint_s
        generation_s = _extract_float(
            section, rf"^\s*• generation:\s*({NUMBER})s"
        )
        mean_generation_length = _extract_float(
            section, rf"^\s*• Mean Generation Length:\s*({NUMBER})"
        )
        generated_tokens = mean_generation_length * samples_per_step
        stages = {
            "weight_sync": _extract_float(
                section,
                rf"^\s*• prepare_for_generation/total:\s*({NUMBER})s",
                required=False,
            ),
            "generation": generation_s,
            "logprob_prep": _extract_float(
                section,
                rf"^\s*• logprob_inference_prep:\s*({NUMBER})s",
                required=False,
            ),
            "policy_logprob": _extract_float(
                section, rf"^\s*• policy_logprobs:\s*({NUMBER})s", required=False
            ),
            "reference_logprob": _extract_float(
                section,
                rf"^\s*• reference_logprobs:\s*({NUMBER})s",
                required=False,
            ),
            "training_prep": _extract_float(
                section, rf"^\s*• training_prep:\s*({NUMBER})s", required=False
            ),
            "actor_update": _extract_float(
                section, rf"^\s*• policy_training:\s*({NUMBER})s", required=False
            ),
            "reward": _extract_float(
                section,
                rf"^\s*• reward_calculation:\s*({NUMBER})s",
                required=False,
            ),
            "advantage": _extract_float(
                section,
                rf"^\s*• advantage_calculation:\s*({NUMBER})s",
                required=False,
            ),
        }
        stages["other"] = max(0.0, core_step_s - sum(stages.values()))
        steps.append(
            {
                "step": step_number,
                "core_step_s": core_step_s,
                "raw_step_s": raw_step_s,
                "checkpoint_s": checkpoint_s,
                "generated_tokens": generated_tokens,
                "generation_output_tokens_per_s": generated_tokens / generation_s,
                "e2e_output_tokens_per_s": generated_tokens / core_step_s,
                "reward": _extract_float(
                    section, rf"^\s*• Avg Reward:\s*({NUMBER})"
                ),
                "stages_s": stages,
            }
        )
    return steps


def analyze(
    search_r1_log: Path,
    zerosearch_log: Path,
    nemo_log: Path,
    samples_per_step: int,
) -> dict[str, Any]:
    """Build a normalized three-framework result object."""
    return {
        "schema_version": 1,
        "samples_per_step": samples_per_step,
        "measured_steps": list(MEASURED_STEPS),
        "implementations": {
            "NeMo RL AI-search": _summarize(
                parse_nemo_log(nemo_log, samples_per_step)
            ),
            "Search-R1": _summarize(
                parse_verl_log(search_r1_log, samples_per_step)
            ),
            "ZeroSearch": _summarize(
                parse_verl_log(zerosearch_log, samples_per_step)
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-r1-log", type=Path, required=True)
    parser.add_argument("--zerosearch-log", type=Path, required=True)
    parser.add_argument("--nemo-log", type=Path, required=True)
    parser.add_argument("--samples-per-step", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(
        search_r1_log=args.search_r1_log,
        zerosearch_log=args.zerosearch_log,
        nemo_log=args.nemo_log,
        samples_per_step=args.samples_per_step,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
