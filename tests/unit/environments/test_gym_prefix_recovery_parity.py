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

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

_HELPER_PATH = (
    Path(__file__).parents[2] / "functional" / "_gym_prefix_recovery_parity.py"
)
# Both helpers import their shared sibling module by name, as they do when the
# functional scripts run them.
sys.path.insert(0, str(_HELPER_PATH.parent))
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

    assert _SNAPSHOT_HELPER.calendar_sentinel_count(state, "recovery sentinel") == 1


def test_training_records_load_from_the_trainer_dump(tmp_path: Path) -> None:
    """Rows come from ``async_rl.log_full_train_data``, written by the real dump."""
    from nemo_rl.utils.train_data_dump import TrainDataDump

    # The entrypoint writes into a new exp_NNN directory under logger.log_dir.
    dump = TrainDataDump(str(tmp_path / "exp_001"))
    dump.add_chunk(
        step=2,
        sample_ids=["group-a_g0", "group-a_g1"],
        tags=[{"prompt_idx": 11}, {"prompt_idx": 11}],
        input_lengths=torch.tensor([2, 3]),
        sequences={
            "token_ids": torch.tensor([[1, 2, 99], [3, 4, 5]]),
            "token_loss_mask": torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 1.0]]),
            "advantages": torch.tensor([[0.0, 2.0, 99.0], [0.0, 3.0, 4.0]]),
            "generation_logprobs": torch.tensor([[0.0, -0.1, 9.0], [0.0, -0.2, -0.3]]),
            "prev_logprobs": torch.tensor([[0.0, -0.1, 9.0], [0.0, -0.2, -0.3]]),
        },
        scalars={
            "sample_loss_mask": torch.tensor([1.0, 0.5]),
            "rewards": torch.tensor([1.0, 0.0]),
        },
    )
    dump.finish_step(2, expected_rows=2)
    staging_keys = tmp_path / "staging-keys.jsonl"
    staging_keys.write_text(
        json.dumps({"train_step": 3, "staging_sample_ids": ["z"]})
        + "\n"
        + json.dumps({"train_step": 3, "staging_sample_ids": ["a"]})
        + "\n"
    )

    (record,) = _HELPER.load_training_records(tmp_path, staging_keys=staging_keys)

    assert record["train_step"] == 3
    assert record["staging_sample_ids"] == ["a", "z"]
    first, second = record["rows"]
    assert (first["group_id"], first["prompt_idx"], first["generation_index"]) == (
        "group-a",
        11,
        0,
    )
    assert first["token_ids"] == [1, 2]
    assert first["advantages"] == [0.0, 2.0]
    assert (second["input_length"], second["reward"]) == (3, 0.0)
    assert second["sample_loss_mask"] == 0.5


def test_training_records_require_one_experiment_directory(tmp_path: Path) -> None:
    (tmp_path / "exp_001").mkdir()
    (tmp_path / "exp_002").mkdir()

    with pytest.raises(AssertionError, match="expected one experiment directory"):
        _HELPER.load_training_records(tmp_path)


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
            "advantages": [0.0, 0.5, 0.500001],
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


def _trained_records(
    generated: int, *, group_id: str = "g", masked: frozenset[int] = frozenset()
) -> list[dict[str, Any]]:
    """One trained row: a two-token prompt, then a closing call of ``generated``.

    The call's tokens are 100, 101, ...; ``masked`` lists call positions the
    trainer dropped from the loss (e.g. top-k/top-p ``-inf`` logprobs).
    """
    row = {
        "group_id": group_id,
        "generation_index": 0,
        "token_ids": [1, 2] + list(range(100, 100 + generated)),
        "token_loss_mask": [0.0, 0.0]
        + [0.0 if index in masked else 1.0 for index in range(generated)],
    }
    return [{"train_step": 1, "rows": [row]}]


def _completion_log(
    source: str, *, prefix_tokens: int = 5, tail_tokens: int = 7, total: int = 12
) -> str:
    """Restore and completion lines for a call whose prefix is 100, 101, ...."""
    return _restored_log(source, list(range(100, 100 + prefix_tokens))) + (
        "generation prefix completed: rollout_id=r model_call_id=continued "
        f"source_model_call_id={source} prefix_tokens={prefix_tokens} "
        f"tail_tokens={tail_tokens} total_generation_tokens={total}\n"
    )


def test_prefix_completion_retires_old_keys() -> None:
    selection = {
        "cuts": [
            {
                "model_call_id": "source-call",
                "group_id": "g",
                "generation_index": 0,
                "prefix_token_count": 5,
                "staging_keys": ["prefix-base", "prefix-routes"],
            }
        ]
    }
    log = _completion_log("source-call")
    assert _HELPER._assert_prefix_boundary(selection, log, _trained_records(12)) == {
        ("g", 0): 2
    }
    _HELPER._assert_retired_prefix_keys(
        selection, [{"staging_sample_ids": ["new-active-key"], "rows": []}]
    )


def _dump_restored_run(
    tmp_path: Path, *, live_keys: list[str], phase2_log_text: str | None = None
) -> argparse.Namespace:
    """A phase-two trainer dump, log, and selection for ``verify-completion``."""
    from nemo_rl.utils.train_data_dump import TrainDataDump

    log_dir = tmp_path / "recovery-logs"
    token_ids = [1, 2] + list(range(100, 112))
    dump = TrainDataDump(str(log_dir / "exp_001"))
    dump.add_chunk(
        step=0,
        sample_ids=["group-a_g0"],
        tags=[{"prompt_idx": 0}],
        input_lengths=torch.tensor([len(token_ids)]),
        sequences={
            "token_ids": torch.tensor([token_ids]),
            # The trainer masked one position inside the restored call.
            "token_loss_mask": torch.tensor(
                [[0.0, 0.0] + [1.0] * 6 + [0.0] + [1.0] * 5]
            ),
            "advantages": torch.zeros(1, len(token_ids)),
            "generation_logprobs": torch.zeros(1, len(token_ids)),
            "prev_logprobs": torch.zeros(1, len(token_ids)),
        },
        scalars={
            "sample_loss_mask": torch.tensor([1.0]),
            "rewards": torch.tensor([1.0]),
        },
    )
    dump.finish_step(0, expected_rows=1)
    staging_keys = tmp_path / "recovery-staging-keys.jsonl"
    staging_keys.write_text(
        json.dumps({"train_step": 1, "staging_sample_ids": live_keys}) + "\n"
    )
    selection = tmp_path / "selected-snapshot.json"
    selection.write_text(
        json.dumps(
            {
                "cuts": [
                    {
                        **_cut("call-a", 0, 5),
                        "group_id": "group-a",
                        "staging_keys": ["prefix-key"],
                    }
                ]
            }
        )
    )
    phase2_log = tmp_path / "phase2.log"
    phase2_log.write_text(phase2_log_text or _completion_log("call-a"))
    return argparse.Namespace(
        selection_file=str(selection),
        recovery_log_dir=str(log_dir),
        recovery_staging_keys=str(staging_keys),
        phase2_log=str(phase2_log),
        require_routes=False,
        kl_type="k3",
    )


def test_verify_completion_checks_a_reference_free_restore(tmp_path: Path, capsys):
    _SNAPSHOT_HELPER.verify_completion(
        _dump_restored_run(tmp_path, live_keys=["new-active-key"])
    )

    assert "('group-a', 0)" in capsys.readouterr().out


def test_verify_completion_rejects_a_restored_prefix_still_live(tmp_path: Path):
    with pytest.raises(AssertionError, match="still live when training began"):
        _SNAPSHOT_HELPER.verify_completion(
            _dump_restored_run(tmp_path, live_keys=["prefix-key"])
        )


def test_restored_prefix_keys_must_not_be_live_when_training_begins() -> None:
    selection = {
        "cuts": [{"model_call_id": "source-call", "staging_keys": ["prefix-base"]}]
    }

    _HELPER._assert_retired_prefix_keys(selection, [])

    with pytest.raises(AssertionError, match="still live when training began"):
        _HELPER._assert_retired_prefix_keys(
            selection, [{"staging_sample_ids": ["prefix-base"], "rows": []}]
        )


@pytest.mark.parametrize(
    ("trained", "match"),
    [
        (11, "does not start its closing call"),
        (13, "does not end with its 12 generated tokens"),
    ],
    ids=["dropped", "duplicated"],
)
def test_prefix_completion_rejects_trained_row_that_dropped_or_duplicated_a_token(
    trained: int, match: str
) -> None:
    # The worker's own total always equals prefix + tail; only the trained row
    # can expose a dropped or duplicated token, which moves where counting back
    # from the row's end puts the call.
    selection = {
        "cuts": [
            {
                "model_call_id": "source-call",
                "group_id": "g",
                "generation_index": 0,
                "prefix_token_count": 5,
            }
        ]
    }
    log = _completion_log("source-call")

    with pytest.raises(AssertionError, match=match):
        _HELPER._assert_prefix_boundary(selection, log, _trained_records(trained))


def test_prefix_completion_counts_back_past_positions_the_trainer_masked() -> None:
    # Masked call positions, including the call's first token, must not be
    # mistaken for the start of the call.
    selection = {"cuts": [_cut("source-call", 0, 5) | {"group_id": "g"}]}

    starts = _HELPER._assert_prefix_boundary(
        selection,
        _completion_log("source-call"),
        _trained_records(12, masked=frozenset({0, 6})),
    )

    assert starts == {("g", 0): 2}


def test_prefix_completion_rejects_token_accounting_with_a_gap() -> None:
    selection = {"cuts": [_cut("source-call", 0, 5) | {"group_id": "g"}]}

    with pytest.raises(AssertionError, match="gap or overlap"):
        _HELPER._assert_prefix_boundary(
            selection, _completion_log("source-call", total=13), _trained_records(12)
        )


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


def test_restored_row_must_match_exactly_outside_the_cut_turn() -> None:
    """A changed tool result earlier in a restored row is never excused."""
    baseline = _generated_row(0, [1, 2, 3, 4, 30, 40, 41, 42])
    recovery = _generated_row(0, [1, 2, 3, 4, 77, 40, 41, 42])

    with pytest.raises(AssertionError, match=r"token_ids.*first difference at index 4"):
        _HELPER._compare_training_rows(
            [{"train_step": 1, "rows": [baseline]}],
            [{"train_step": 1, "rows": [recovery]}],
            steps=1,
            prompts_per_step=1,
            generations_per_prompt=1,
            rtol=0.0,
            atol=0.0,
        )


def _restored_log(source: str, prefix_ids: list[int]) -> str:
    digest = hashlib.sha256(",".join(map(str, prefix_ids)).encode()).hexdigest()
    return (
        f"generation prefix restored: rollout_id=r model_call_id=new-{source} "
        f"source_model_call_id={source} prefix_tokens={len(prefix_ids)} "
        f"prefix_digest={'0' * 64} weight_version_span=[0,0] "
        f"prefix_ids_sha256={digest}\n"
    )


def _cut(source: str, generation_index: int, prefix_tokens: int) -> dict[str, Any]:
    return {
        "model_call_id": source,
        "group_id": "physical-group",
        "generation_index": generation_index,
        "prefix_token_count": prefix_tokens,
    }


def _diverged(
    baseline_rows: list[dict[str, Any]],
    recovery_rows: list[dict[str, Any]],
    cuts: list[dict[str, Any]],
    log: str,
) -> frozenset[tuple[int, int, int]]:
    # Each restored call closes its recovery row, after the five tokens that
    # _generated_row puts before the closing call.
    for cut in cuts:
        (row,) = [
            row
            for row in recovery_rows
            if row["generation_index"] == cut["generation_index"]
        ]
        total = len(row["token_ids"]) - 5
        prefix_tokens = cut["prefix_token_count"]
        log += (
            "generation prefix completed: rollout_id=r model_call_id=x "
            f"source_model_call_id={cut['model_call_id']} "
            f"prefix_tokens={prefix_tokens} tail_tokens={total - prefix_tokens} "
            f"total_generation_tokens={total}\n"
        )
    return _HELPER._rows_diverged_before_cut(
        {"cuts": cuts},
        log,
        [{"train_step": 1, "rows": baseline_rows}],
        [{"train_step": 1, "rows": recovery_rows}],
    )


def test_restored_row_whose_phase_one_prefix_differed_is_set_aside() -> None:
    """Phase 1 took the other side of a near-tie before the cut."""
    baseline = [
        _generated_row(0, [0, 1, 2, 3, 4, 10, 11, 12, 13]),
        _generated_row(1, [0, 1, 2, 3, 4, 10, 11, 12]),
    ]
    recovery = [
        _generated_row(0, [0, 1, 2, 3, 4, 20, 21, 22]),
        _generated_row(1, [0, 1, 2, 3, 4, 10, 11, 12]),
    ]
    cuts = [_cut("call-a", 0, 2), _cut("call-b", 1, 1)]
    log = _restored_log("call-a", [20, 21]) + _restored_log("call-b", [10])

    diverged = _diverged(baseline, recovery, cuts, log)

    assert diverged == {(1, 7, 0)}

    def compare(diverged_before_cut: frozenset[tuple[int, int, int]]) -> None:
        _HELPER._compare_training_rows(
            [{"train_step": 1, "rows": baseline}],
            [{"train_step": 1, "rows": recovery}],
            steps=1,
            prompts_per_step=1,
            generations_per_prompt=2,
            rtol=0.0,
            atol=0.0,
            diverged_before_cut=diverged_before_cut,
        )

    compare(diverged)
    with pytest.raises(AssertionError, match=r"row \(1, 7, 0\) field 'input_length'"):
        compare(frozenset())


def test_restored_row_must_start_with_the_restored_tokens() -> None:
    baseline = [_generated_row(0, [0, 1, 2, 3, 4, 10, 11, 12])]
    recovery = [_generated_row(0, [0, 1, 2, 3, 4, 20, 99, 22])]

    with pytest.raises(AssertionError, match="with its 2 restored prefix tokens"):
        _diverged(
            baseline,
            recovery,
            [_cut("call-a", 0, 2)],
            _restored_log("call-a", [20, 21]),
        )


def test_restored_row_that_matched_the_reference_prefix_is_not_set_aside() -> None:
    """Only a pre-cut difference is excused; a different tail still fails."""
    baseline = [_generated_row(0, [0, 1, 2, 3, 4, 10, 11, 12])]
    recovery = [_generated_row(0, [0, 1, 2, 3, 4, 10, 11, 99])]

    diverged = _diverged(
        baseline, recovery, [_cut("call-a", 0, 2)], _restored_log("call-a", [10, 11])
    )

    assert diverged == frozenset()


def test_set_aside_row_must_match_before_its_closing_call() -> None:
    baseline = [
        _generated_row(0, [0, 1, 2, 3, 4, 10, 11]),
        _generated_row(1, [0, 1, 2, 3, 4, 10, 11]),
    ]
    recovery = [
        _generated_row(0, [0, 1, 2, 3, 77, 20, 21]),
        _generated_row(1, [0, 1, 2, 3, 4, 10, 11]),
    ]
    cuts = [_cut("call-a", 0, 1), _cut("call-b", 1, 1)]
    log = _restored_log("call-a", [20]) + _restored_log("call-b", [10])

    with pytest.raises(AssertionError, match="differs before its restored closing"):
        _diverged(baseline, recovery, cuts, log)


def test_not_every_restored_row_may_be_set_aside() -> None:
    baseline = [_generated_row(0, [0, 1, 2, 3, 4, 10, 11])]
    recovery = [_generated_row(0, [0, 1, 2, 3, 4, 20, 21])]

    with pytest.raises(AssertionError, match="every restored prefix differed"):
        _diverged(
            baseline, recovery, [_cut("call-a", 0, 1)], _restored_log("call-a", [20])
        )


def test_every_restored_prefix_must_complete_exactly_once() -> None:
    selection = {
        "cuts": [
            {
                "model_call_id": "call-a",
                "group_id": "g",
                "generation_index": 0,
                "prefix_token_count": 5,
            },
            {"model_call_id": "call-b", "prefix_token_count": 3},
        ]
    }
    log = _completion_log("call-a")

    with pytest.raises(AssertionError, match="call-b"):
        _HELPER._assert_prefix_boundary(selection, log, _trained_records(12))


# 202 generated tokens: prompt [0, 1] and tool result [4] are not generated.
_LOGPROB_TOKENS = list(range(205))


def _logprob_records(
    generated_diffs: dict[int, float], *, prompt_diff: float = 0.0
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Baseline and recovery rows whose generation logprobs differ as given."""
    baseline = _generated_row(0, _LOGPROB_TOKENS)
    baseline["generation_logprobs"] = [-1.0] * len(_LOGPROB_TOKENS)
    recovery = {
        **baseline,
        "generation_logprobs": list(baseline["generation_logprobs"]),
    }
    recovery["generation_logprobs"][0] += prompt_diff
    for position, diff in generated_diffs.items():
        recovery["generation_logprobs"][position] += diff
    return (
        [{"train_step": 1, "rows": [baseline]}],
        [{"train_step": 1, "rows": [recovery]}],
    )


def test_generation_logprob_gate_accepts_bf16_noise(capsys) -> None:
    # One small difference in 202 generated tokens; prompt logprobs are ignored.
    baseline, recovery = _logprob_records({10: 0.125}, prompt_diff=5.0)

    _HELPER._assert_generation_logprob_agreement(
        baseline, recovery, min_bit_identical=0.99, max_abs_diff=0.5
    )

    assert (
        "bit_identical=0.9950, max_abs_diff=0.125, tokens=202"
        in capsys.readouterr().out
    )


@pytest.mark.parametrize(
    ("generated_diffs", "message"),
    [
        ({10: 0.125, 11: 0.125, 12: 0.125}, "bit_identical=0.9851"),
        ({10: 0.75}, "max_abs_diff=0.750"),
    ],
)
def test_generation_logprob_gate_rejects_disagreement(
    generated_diffs: dict[int, float], message: str
) -> None:
    baseline, recovery = _logprob_records(generated_diffs)

    with pytest.raises(AssertionError, match=message):
        _HELPER._assert_generation_logprob_agreement(
            baseline, recovery, min_bit_identical=0.99, max_abs_diff=0.5
        )


def test_generation_logprob_gate_needs_token_matched_rows() -> None:
    baseline, recovery = _logprob_records({})
    recovery[0]["rows"][0] = {**recovery[0]["rows"][0], "token_ids": [9] * 205}

    with pytest.raises(AssertionError, match="no token-matched rows"):
        _HELPER._assert_generation_logprob_agreement(
            baseline, recovery, min_bit_identical=0.99, max_abs_diff=0.5
        )


def test_cut_coverage_counts_distinct_prompt_groups() -> None:
    # Three cuts, but a refused sibling left group "c" without one.
    candidates = [{"group_id": "a"}, {"group_id": "a"}, {"group_id": "b"}]

    _SNAPSHOT_HELPER._require_cut_coverage(candidates, min_cuts=3, min_cut_groups=2)
    with pytest.raises(
        AssertionError, match="3 recoverable prefixes in 2 prompt groups"
    ):
        _SNAPSHOT_HELPER._require_cut_coverage(candidates, min_cuts=1, min_cut_groups=3)
    with pytest.raises(AssertionError, match="need 4 in 1"):
        _SNAPSHOT_HELPER._require_cut_coverage(candidates, min_cuts=4, min_cut_groups=1)


def _with_route_hashes(log: str, *, restored: str, completed: str) -> str:
    """Append router replay's prefix route hashes to restore and completion lines."""
    return "".join(
        f"{line} prefix_routes_sha256="
        f"{restored if 'prefix restored:' in line else completed}\n"
        for line in log.splitlines()
    )


@pytest.mark.parametrize(
    ("completed", "match"),
    [
        ("a" * 64, None),
        ("b" * 64, "does not carry its restored prefix routes"),
        ("none", "carried no routes"),
    ],
    ids=["staged", "recomputed", "dropped"],
)
def test_restored_prefix_routes_must_be_the_ones_staged_for_training(
    completed: str, match: str | None
) -> None:
    selection = {"cuts": [_cut("call-a", 0, 5)]}
    log = _with_route_hashes(
        _completion_log("call-a"), restored="a" * 64, completed=completed
    )

    if match is None:
        _SNAPSHOT_HELPER._assert_restored_routes_staged(selection, log)
    else:
        with pytest.raises(AssertionError, match=match):
            _SNAPSHOT_HELPER._assert_restored_routes_staged(selection, log)


def test_restored_prefix_routes_must_be_logged_on_restore_and_completion() -> None:
    # A run without router replay logs no route hashes at all.
    with pytest.raises(AssertionError, match="expected one each"):
        _SNAPSHOT_HELPER._assert_restored_routes_staged(
            {"cuts": [_cut("call-a", 0, 5)]}, _completion_log("call-a")
        )


def test_verify_completion_reports_gen_kl_error_by_span(tmp_path: Path, capsys):
    args = _dump_restored_run(
        tmp_path,
        live_keys=[],
        phase2_log_text=_with_route_hashes(
            _completion_log("call-a"), restored="a" * 64, completed="a" * 64
        ),
    )
    args.require_routes = True

    _SNAPSHOT_HELPER.verify_completion(args)

    report = json.loads(
        capsys.readouterr().out.split("gen_kl_error: ", 1)[1].splitlines()[0]
    )
    # The five restored prefix tokens, then the tail's trained tokens (one of
    # its seven positions is masked); the dump's logprobs agree exactly.
    assert report["restored_prefix"] == {"tokens": 5, "mean": 0.0, "max": 0.0}
    assert report["resumed_tail"] == {"tokens": 6, "mean": 0.0, "max": 0.0}
    assert report["other_rows"]["tokens"] == 0
