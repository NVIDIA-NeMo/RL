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

import json
from pathlib import Path
from typing import Any

from examples.swe_bench.benchmark_streaming_final_decode import (
    FinalGenerationMeasurement,
    _stable_streamed_prefix,
)
from examples.swe_bench.benchmark_streaming_trace_replay import (
    _build_recorded_prompt_trace,
    _next_function_call,
    _recorded_prompt_pairs,
    _replay_output_contract,
    audit_recorded_prompt_coverage,
    load_recorded_prompt_trace_sweep,
    load_recorded_prompt_traces,
)


class _CharacterTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [ord(character) for character in text]


def _measurement(token_id: int) -> FinalGenerationMeasurement:
    return FinalGenerationMeasurement(
        output_kind="delta",
        cached_tokens=0,
        prompt_tokens=10,
        output_token_ids=(token_id,),
        output_logprobs=(-0.1,),
        output_text=str(token_id),
        decoded_text=str(token_id),
        time_to_first_output_seconds=0.1,
        total_seconds=0.1,
    )


def test_next_function_call_does_not_cross_another_tool_output() -> None:
    function_call = {"type": "function_call", "prompt_str": "prompt"}

    assert _next_function_call([{"type": "message"}, function_call], 0) is function_call
    assert (
        _next_function_call(
            [{"type": "function_call_output", "output": "new"}, function_call],
            0,
        )
        is None
    )


def test_recorded_prompt_pairs_selects_immediate_next_model_call(
    tmp_path: Path,
) -> None:
    trajectory = tmp_path / "trajectory.jsonl"
    trajectory.write_text(
        '{"response":{"output":['
        '{"type":"function_call_output","output":"tool result"},'
        '{"type":"message"},'
        '{"type":"function_call","prompt_str":"before tool result after"}'
        "]}}\n"
    )

    pairs = list(_recorded_prompt_pairs(trajectory))

    assert len(pairs) == 1
    assert pairs[0].trajectory_index == 0
    assert pairs[0].label == "trajectory-0-output-0"
    assert pairs[0].tool_output == "tool result"
    assert pairs[0].final_prompt == "before tool result after"


def test_recorded_trace_prefills_only_tokens_stable_in_final_prompt() -> None:
    tokenizer: Any = _CharacterTokenizer()
    tool_output = "abcdefghijklmnopqrstuvwxyz"
    final_prompt = f"prefix:{tool_output}:suffix"

    recorded, reason = _build_recorded_prompt_trace(
        tokenizer,
        label="sample",
        tool_output=tool_output,
        final_prompt=final_prompt,
        snapshot_chars=16,
        stability_margin_tokens=3,
        max_model_len=1_024,
    )

    assert reason is None
    assert recorded is not None
    trace = recorded.trace
    stable_prefix = _stable_streamed_prefix(trace, stability_margin_tokens=3)
    assert stable_prefix == trace.final[: len(stable_prefix)]
    assert len(stable_prefix) == len("prefix:") + 16 - 3
    assert trace.final[len(stable_prefix) :]


def test_trace_selection_covers_distinct_trajectories(tmp_path: Path) -> None:
    trajectory = tmp_path / "trajectory.jsonl"
    rows = []
    for trajectory_index in range(2):
        output_items = []
        for output_index in range(2):
            tool_output = f"trajectory{trajectory_index}-output{output_index}-abcdefgh"
            output_items.extend(
                (
                    {"type": "function_call_output", "output": tool_output},
                    {
                        "type": "function_call",
                        "prompt_str": f"prefix:{tool_output}:suffix",
                    },
                )
            )
        rows.append(json.dumps({"response": {"output": output_items}}))
    trajectory.write_text("\n".join(rows) + "\n")

    traces, skipped = load_recorded_prompt_traces(
        _CharacterTokenizer(),
        trajectory_jsonl=trajectory,
        trace_limit=2,
        max_traces_per_trajectory=1,
        snapshot_chars=8,
        stability_margin_tokens=2,
        max_model_len=1_024,
    )

    assert [trace.label.split("-")[1] for trace in traces] == ["0", "1"]
    assert skipped["per_trajectory_limit"] == 1


def test_recorded_trace_rejects_snapshot_without_stable_dynamic_tokens() -> None:
    recorded, reason = _build_recorded_prompt_trace(
        _CharacterTokenizer(),
        label="sample",
        tool_output="abcdefgh",
        final_prompt="prefix:abcdefgh:suffix",
        snapshot_chars=4,
        stability_margin_tokens=4,
        max_model_len=1_024,
    )

    assert recorded is None
    assert reason == "no_stable_dynamic_tokens"


def test_trace_sweep_uses_same_transition_for_every_snapshot(tmp_path: Path) -> None:
    trajectory = tmp_path / "trajectory.jsonl"
    short_output = "abcdefgh"
    long_output = "abcdefghijklmnopqrstuvwxyz"
    trajectory.write_text(
        json.dumps(
            {
                "response": {
                    "output": [
                        {"type": "function_call_output", "output": short_output},
                        {
                            "type": "function_call",
                            "prompt_str": f"prefix:{short_output}:suffix",
                        },
                        {"type": "function_call_output", "output": long_output},
                        {
                            "type": "function_call",
                            "prompt_str": f"prefix:{long_output}:suffix",
                        },
                    ]
                }
            }
        )
        + "\n"
    )

    traces, skipped = load_recorded_prompt_trace_sweep(
        _CharacterTokenizer(),
        trajectory_jsonl=trajectory,
        trace_limit=1,
        max_traces_per_trajectory=1,
        snapshot_chars_values=(4, 16),
        stability_margin_tokens=2,
        max_model_len=1_024,
    )

    assert traces[4][0].label == traces[16][0].label == "trajectory-0-output-2"
    assert skipped["snapshot_16:tool_output_not_longer_than_snapshot"] == 1

    coverage = audit_recorded_prompt_coverage(
        _CharacterTokenizer(),
        trajectory_jsonl=trajectory,
        snapshot_chars_values=(4, 16),
        stability_margin_tokens=2,
        max_model_len=1_024,
    )
    assert coverage["next_model_call_pairs"] == 2
    assert coverage["by_snapshot_chars"]["4"]["replayable_pairs"] == 2
    assert coverage["by_snapshot_chars"]["16"]["replayable_pairs"] == 1


def test_replay_contract_uses_candidate_majority_against_control_mode() -> None:
    controls = [_measurement(1), _measurement(1), _measurement(2)]

    passing = _replay_output_contract(
        controls,
        [_measurement(1), _measurement(2), _measurement(1)],
    )
    failing = _replay_output_contract(
        controls,
        [_measurement(2), _measurement(2), _measurement(1)],
    )

    assert passing["method"] == "modal_control"
    assert passing["candidate_matches_control_modal"] == 2
    assert passing["all"] is True
    assert failing["candidate_matches_control_modal"] == 1
    assert failing["all"] is False
