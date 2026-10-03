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

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import torch

_HELPER_PATH = (
    Path(__file__).parents[2] / "functional" / "_gym_prefix_recovery_parity.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "gym_prefix_recovery_parity", _HELPER_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_HELPER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HELPER)

_SNAPSHOT_HELPER_PATH = (
    Path(__file__).parents[2] / "functional" / "_gym_prefix_recovery_snapshot.py"
)
_SNAPSHOT_SPEC = importlib.util.spec_from_file_location(
    "gym_prefix_recovery_snapshot", _SNAPSHOT_HELPER_PATH
)
assert _SNAPSHOT_SPEC is not None and _SNAPSHOT_SPEC.loader is not None
_SNAPSHOT_HELPER = importlib.util.module_from_spec(_SNAPSHOT_SPEC)
_SNAPSHOT_SPEC.loader.exec_module(_SNAPSHOT_HELPER)


def _row(*, token_ids: list[int] | None = None) -> dict[str, Any]:
    tokens = token_ids or [1, 2, 3]
    return {
        "sample_id": "physical-group_g0",
        "group_id": "physical-group",
        "prompt_idx": 7,
        "generation_index": 0,
        "input_length": len(tokens),
        "token_ids": tokens,
        "token_loss_mask": [0.0, 1.0, 1.0][: len(tokens)],
        "sample_loss_mask": 1.0,
        "reward": 1.0,
        "advantages": [0.0, 0.5, 0.5][: len(tokens)],
        "generation_logprobs": [-0.1, -0.2, -0.3][: len(tokens)],
        "prev_logprobs": [-0.1, -0.2, -0.3][: len(tokens)],
    }


def test_calendar_sentinel_count_matches_one_checkpointed_event() -> None:
    state = {
        "calendar": {
            "_calendar_events": {
                "columns": [
                    {"name": "event_id", "values": [1, 2]},
                    {
                        "name": "event_name",
                        "values": ["another event", "Recovery Sentinel"],
                    },
                ]
            }
        }
    }

    assert _SNAPSHOT_HELPER._calendar_sentinel_count(state, "recovery sentinel") == 1


def test_serialize_training_batch_trims_padding_and_records_identity() -> None:
    record = _HELPER.serialize_training_batch(
        train_step=3,
        chunk_index=1,
        sample_ids=["group-a_g0", "group-a_g1"],
        tags=[{"prompt_idx": 11}, {"prompt_idx": 11}],
        data={
            "input_ids": torch.tensor([[1, 2, 99], [3, 4, 5]]),
            "input_lengths": torch.tensor([2, 3]),
            "token_mask": torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 1.0]]),
            "sample_mask": torch.tensor([1.0, 0.5]),
            "total_reward": torch.tensor([1.0, 0.0]),
            "advantages": torch.tensor([[0.0, 2.0, 99.0], [0.0, 3.0, 4.0]]),
            "generation_logprobs": torch.tensor([[0.0, -0.1, 99.0], [0.0, -0.2, -0.3]]),
            "prev_logprobs": torch.tensor([[0.0, -0.1, 99.0], [0.0, -0.2, -0.3]]),
        },
        staging_sample_ids=["z", "a"],
    )

    assert record["train_step"] == 3
    assert record["staging_sample_ids"] == ["a", "z"]
    assert record["rows"][0]["group_id"] == "group-a"
    assert record["rows"][0]["prompt_idx"] == 11
    assert record["rows"][0]["generation_index"] == 0
    assert record["rows"][0]["token_ids"] == [1, 2]
    assert record["rows"][0]["advantages"] == [0.0, 2.0]


def test_compare_training_rows_is_order_independent_and_tolerates_floats() -> None:
    baseline_rows = [
        _row(),
        {
            **_row(),
            "generation_index": 1,
            "sample_id": "physical-group_g1",
        },
    ]
    recovery_rows = [
        {
            **baseline_rows[1],
            "group_id": "recovered-group",
            "sample_id": "recovered-group_g1",
            "prev_logprobs": [-0.1, -0.2, -0.300001],
        },
        {
            **baseline_rows[0],
            "group_id": "recovered-group",
            "sample_id": "recovered-group_g0",
            "advantages": [0.0, 0.500001, 0.5],
        },
    ]

    _HELPER._compare_training_rows(
        [{"train_step": 1, "rows": baseline_rows}],
        [{"train_step": 1, "rows": recovery_rows}],
        steps=1,
        prompts_per_step=1,
        generations_per_prompt=2,
        rtol=1e-4,
        atol=1e-5,
    )


def test_compare_training_rows_rejects_prefix_boundary_token_difference() -> None:
    with pytest.raises(AssertionError, match="token_ids"):
        _HELPER._compare_training_rows(
            [{"train_step": 1, "rows": [_row()]}],
            [{"train_step": 1, "rows": [_row(token_ids=[1, 2, 4])]}],
            steps=1,
            prompts_per_step=1,
            generations_per_prompt=1,
            rtol=1e-4,
            atol=1e-5,
        )


def test_prefix_completion_and_successor_checkpoint_retire_old_keys(
    tmp_path: Path,
) -> None:
    selection = {
        "model_call_id": "source-call",
        "generation_token_count": 5,
        "staging_keys": ["prefix-base", "prefix-routes"],
    }
    log = (
        "generation prefix completed: rollout_id=r model_call_id=continued "
        "source_model_call_id=source-call prefix_tokens=5 tail_tokens=7 "
        "total_generation_tokens=12"
    )
    successor = tmp_path / "step_1"
    successor.mkdir()
    (successor / "gym_checkpoint.json").write_text(
        json.dumps({"staging_keys": {"gym-0": ["new-active-key"]}})
    )

    _HELPER._assert_prefix_boundary(selection, log)
    _HELPER._assert_retired_prefix_keys(
        selection,
        [{"staging_sample_ids": ["new-active-key"], "rows": []}],
        successor,
    )


def test_prefix_completion_rejects_gap_or_overlap() -> None:
    selection = {"model_call_id": "source-call", "generation_token_count": 5}
    log = (
        "generation prefix completed: rollout_id=r model_call_id=continued "
        "source_model_call_id=source-call prefix_tokens=5 tail_tokens=7 "
        "total_generation_tokens=13"
    )

    with pytest.raises(AssertionError, match="gap or overlap"):
        _HELPER._assert_prefix_boundary(selection, log)


def _completion(step: int, prompt: int, generation: int) -> dict[str, Any]:
    return {
        "event": "completion",
        "target_step": step,
        "prompt_idx": prompt,
        "sibling": {"generation_index": generation},
    }


def test_rank_metrics_compare_only_pairs_within_a_step() -> None:
    baseline = [(0, 0, 0), (0, 1, 0), (1, 2, 0), (1, 3, 0)]
    recovery = [(0, 1, 0), (0, 0, 0), (1, 2, 0), (1, 3, 0)]

    metrics = _HELPER._within_step_rank_metrics(baseline, recovery)

    assert metrics["comparable_pairs"] == 2
    assert metrics["discordant_pairs"] == 1
    assert metrics["kendall_tau"] == 0.0
    assert not metrics["exact_match"]
    assert [step["exact_match"] for step in metrics["per_step"]] == [False, True]


def test_rank_metrics_reject_different_identities() -> None:
    with pytest.raises(AssertionError, match="identical logical identities"):
        _HELPER._within_step_rank_metrics([(0, 0, 0)], [(0, 0, 1)])


def test_prompt_group_is_ready_at_its_last_sibling() -> None:
    order = [(0, 1, 0), (0, 0, 0), (0, 0, 1), (0, 1, 1)]

    assert _HELPER._prompt_group_ready_order(order) == [(0, 0), (0, 1)]


def test_recovery_order_keeps_retained_rows_and_uses_redone_arrivals() -> None:
    phase1 = [(0, 0, 0), (0, 1, 0)]
    phase2 = [(0, 1, 1), (0, 1, 0), (0, 0, 1)]

    assert _HELPER._effective_recovery_order(phase1, phase2) == [
        (0, 0, 0),
        (0, 1, 1),
        (0, 1, 0),
        (0, 0, 1),
    ]


def test_ordering_report_ignores_steps_that_were_not_trained() -> None:
    baseline = [_completion(0, 0, 0), _completion(0, 1, 0), _completion(1, 2, 0)]
    recovery = [_completion(0, 0, 0), _completion(0, 1, 0)]

    report = _HELPER._ordering_report(
        baseline_events=baseline,
        phase1_events=[],
        recovery_events=recovery,
        steps=1,
    )

    assert report["sealed"]["individual"]["exact_match"]
    assert report["arrived"]["individual"]["comparable_pairs"] == 0


def test_reward_signal_rejects_constant_rewards() -> None:
    rows = [
        _row(),
        {**_row(), "generation_index": 1, "sample_id": "physical-group_g1"},
    ]

    with pytest.raises(AssertionError, match="rewards are constant"):
        _HELPER._assert_reward_signal([{"train_step": 1, "rows": rows}])

    rows[1]["reward"] = 0.0
    signal = _HELPER._assert_reward_signal([{"train_step": 1, "rows": rows}])
    assert signal["rows_with_nonzero_advantages"] == 2


def test_generation_logprob_report_measures_only_generated_tokens() -> None:
    baseline = _row(token_ids=[1, 2, 3])
    recovery = {**baseline, "generation_logprobs": [-9.0, -0.2, -0.31]}

    report = _HELPER._generation_logprob_report(
        [{"train_step": 1, "rows": [baseline]}],
        [{"train_step": 1, "rows": [recovery]}],
    )

    pooled = report["pooled"]
    assert pooled["generated_tokens"] == 2
    assert pooled["spearman"] == pytest.approx(1.0)
    assert pooled["bit_identical_fraction"] == 0.5
    assert pooled["max_abs_diff"] == pytest.approx(0.01)


def test_rollout_timeline_is_relative_to_the_first_event() -> None:
    events = [
        {
            "event": "dispatch",
            "time_s": 100.0,
            "target_step": 0,
            "prompt_idx": 0,
            "generation_indices": [0],
        },
        {
            "event": "completion_arrived",
            "time_s": 102.5,
            "target_step": 0,
            "prompt_idx": 0,
            "generation_index": 0,
        },
        {**_completion(0, 0, 0), "time_s": 103.0},
    ]

    assert _HELPER._rollout_timeline(events, steps=1) == {
        "(0, 0, 0)": {"dispatch": 0.0, "completion_arrived": 2.5, "completion": 3.0}
    }


def _generated_row(
    generation_index: int, token_ids: list[int], *, group_id: str = "physical-group"
) -> dict[str, Any]:
    """Prompt [0, 1], turn [2, 3], tool result [4], closing call [5, ...]."""
    mask = [0.0, 0.0, 1.0, 1.0, 0.0] + [1.0] * (len(token_ids) - 5)
    return {
        **_row(token_ids=token_ids),
        "sample_id": f"{group_id}_g{generation_index}",
        "group_id": group_id,
        "generation_index": generation_index,
        "token_loss_mask": mask,
        "advantages": [0.0] * len(token_ids),
        "generation_logprobs": [0.0] * len(token_ids),
        "prev_logprobs": [0.0] * len(token_ids),
    }


@pytest.mark.parametrize(
    ("recovered_tokens", "expected_kind", "expected_diff"),
    [
        ([1, 2, 3, 4, 5, 6, 7, 8], "match", None),
        ([1, 2, 3, 4, 5, 9, 7, 8], "pre_cut", 5),
        ([1, 2, 3, 4, 5, 6, 7, 9], "post_cut", 7),
    ],
)
def test_token_divergence_is_located_relative_to_the_restored_prefix(
    recovered_tokens: list[int], expected_kind: str, expected_diff: int | None
) -> None:
    baseline = _generated_row(0, [1, 2, 3, 4, 5, 6, 7, 8])
    recovery = _generated_row(0, recovered_tokens, group_id="recovered-group")
    selection = {
        "cuts": [
            {
                "group_id": "recovered-group",
                "generation_index": 0,
                "generation_token_count": 2,
            }
        ]
    }

    result = _HELPER._classify_token_divergence(
        [{"train_step": 1, "rows": [baseline]}],
        [{"train_step": 1, "rows": [recovery]}],
        selection,
    )

    assert result[(1, 7, 0)] == {
        "kind": expected_kind,
        "first_diff": expected_diff,
        "restored_prefix_end": 7,
    }


def test_token_divergence_without_a_cut_is_unrestored() -> None:
    result = _HELPER._classify_token_divergence(
        [{"train_step": 1, "rows": [_generated_row(0, [1, 2, 3, 4, 5, 6])]}],
        [{"train_step": 1, "rows": [_generated_row(0, [1, 2, 3, 4, 5, 9])]}],
        {"cuts": []},
    )

    assert result[(1, 7, 0)]["kind"] == "unrestored"


def test_compare_training_rows_skips_only_excused_rows() -> None:
    baseline = [
        _row(),
        {**_row(), "generation_index": 1, "sample_id": "physical-group_g1"},
    ]
    recovery = [
        {**baseline[0], "token_ids": [1, 2, 4]},
        {**baseline[1], "token_ids": [1, 2, 4]},
    ]
    records = {
        "baseline": [{"train_step": 1, "rows": baseline}],
        "recovery": [{"train_step": 1, "rows": recovery}],
    }

    with pytest.raises(AssertionError, match=r"\(1, 7, 1\) field 'token_ids'"):
        _HELPER._compare_training_rows(
            records["baseline"],
            records["recovery"],
            steps=1,
            prompts_per_step=1,
            generations_per_prompt=2,
            rtol=1e-4,
            atol=1e-5,
            excused=frozenset({(1, 7, 0)}),
        )


def test_every_restored_prefix_must_complete_exactly_once() -> None:
    selection = {
        "cuts": [
            {"model_call_id": "call-a", "generation_token_count": 5},
            {"model_call_id": "call-b", "generation_token_count": 3},
        ]
    }
    log = (
        "generation prefix completed: rollout_id=r model_call_id=x "
        "source_model_call_id=call-a prefix_tokens=5 tail_tokens=7 "
        "total_generation_tokens=12"
    )

    with pytest.raises(AssertionError, match="call-b"):
        _HELPER._assert_prefix_boundary(selection, log)


def test_generation_logprob_report_excludes_rows_whose_tokens_diverged() -> None:
    matched = _row(token_ids=[1, 2, 3])
    diverged = {
        **_row(token_ids=[1, 2, 3]),
        "generation_index": 1,
        "sample_id": "physical-group_g1",
    }

    report = _HELPER._generation_logprob_report(
        [{"train_step": 1, "rows": [matched, diverged]}],
        [{"train_step": 1, "rows": [matched, {**diverged, "token_ids": [1, 2, 4]}]}],
    )

    assert report["excluded_rows"] == ["(1, 7, 1)"]
    assert report["pooled"]["generated_tokens"] == 2


@pytest.mark.parametrize(
    ("pooled", "message"),
    [
        (
            {
                "generated_tokens": 10,
                "bit_identical_fraction": 0.98,
                "spearman": 1.0,
                "max_abs_diff": 0.1,
            },
            "bit_identical",
        ),
        (
            {
                "generated_tokens": 10,
                "bit_identical_fraction": 1.0,
                "spearman": 0.9,
                "max_abs_diff": 0.1,
            },
            "spearman",
        ),
        (
            {
                "generated_tokens": 10,
                "bit_identical_fraction": 1.0,
                "spearman": None,
                "max_abs_diff": 0.0,
            },
            "spearman",
        ),
        (
            {
                "generated_tokens": 10,
                "bit_identical_fraction": 1.0,
                "spearman": 1.0,
                "max_abs_diff": 0.6,
            },
            "max_abs_diff",
        ),
        ({"generated_tokens": 0}, "no token-matched rows"),
    ],
)
def test_generation_logprob_gate_rejects_disagreement(
    pooled: dict[str, Any], message: str
) -> None:
    with pytest.raises(AssertionError, match=message):
        _HELPER._assert_generation_logprob_agreement(
            pooled, min_bit_identical=0.99, min_spearman=0.99, max_abs_diff=0.5
        )


def test_generation_logprob_gate_accepts_bf16_noise() -> None:
    _HELPER._assert_generation_logprob_agreement(
        {
            "generated_tokens": 135492,
            "bit_identical_fraction": 0.997,
            "spearman": 0.995,
            "max_abs_diff": 0.174,
        },
        min_bit_identical=0.99,
        min_spearman=0.99,
        max_abs_diff=0.5,
    )
