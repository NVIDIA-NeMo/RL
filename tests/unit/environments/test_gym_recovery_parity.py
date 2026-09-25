# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
import torch


_HELPER_PATH = Path(__file__).parents[2] / "functional" / "_gym_recovery_parity.py"
sys.path.insert(0, str(_HELPER_PATH.parent))
_SPEC = importlib.util.spec_from_file_location("gym_recovery_parity", _HELPER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_HELPER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HELPER)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_inspect_cut_candidate_selects_requested_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "step_2/rollout_snapshots/snapshot_000001"
    snapshot.mkdir(parents=True)
    (snapshot / "manifest.json").write_text(
        json.dumps(
            {
                "base_train_step": 2,
                "gym_checkpoint": {"checkpoint_id": "checkpoint-2"},
            }
        )
    )
    torch.save(
        {
            "groups": [
                {
                    "group_id": "simple-group",
                    "task_source": "simple",
                    "siblings": [
                        {
                            "generation_index": 0,
                            "attempts": [{"attempt_index": 1}],
                        }
                    ],
                },
                {
                    "group_id": "workplace-group",
                    "task_source": "workplace",
                    "siblings": [
                        {
                            "generation_index": 0,
                            "attempts": [{"attempt_index": 0}],
                        }
                    ],
                },
            ]
        },
        snapshot / "rollout_recovery.pt",
    )
    monkeypatch.setattr(
        _HELPER,
        "_active_prefixes",
        lambda *_: [
            {
                "rollout_id": "simple-group_g0",
                "attempt_index": 1,
                "model_call_id": "simple-call",
                "prefix_token_count": 12,
                "prefix_digest": "1" * 64,
                "staging_keys": ["simple-key"],
            },
            {
                "rollout_id": "workplace-group_g0",
                "attempt_index": 0,
                "model_call_id": "workplace-call",
                "prefix_token_count": 24,
                "prefix_digest": "2" * 64,
                "staging_keys": ["workplace-key"],
            },
        ],
    )
    monkeypatch.setattr(
        _HELPER,
        "_matching_agent_boundary",
        lambda *_args, **_kwargs: {
            "boundary_kind": "turn_complete",
            "boundary_index": 0,
            "pending_model": None,
            "last_committed_model_call_id": None,
            "resource_state_revisions": {"resources": 1},
        },
    )

    selected = _HELPER.inspect_cut_candidate(
        snapshot,
        min_train_step=2,
        task_source="simple",
        max_generation_tokens=256,
        boundary_requirement="root",
    )

    assert selected["task_source"] == "simple"
    assert selected["rollout_id"] == "simple-group_g0"
    assert selected["attempt_index"] == 1
    assert selected["prefix_token_count"] == 12
    assert selected["boundary_requirement"] == "root"


def test_inspect_cut_candidate_honors_max_train_step(tmp_path: Path) -> None:
    snapshot = tmp_path / "step_2/rollout_snapshots/snapshot_000001"
    snapshot.mkdir(parents=True)
    (snapshot / "manifest.json").write_text(json.dumps({"base_train_step": 2}))

    with pytest.raises(AssertionError, match="train step 2 exceeds 1"):
        _HELPER.inspect_cut_candidate(
            snapshot,
            min_train_step=0,
            max_train_step=1,
            task_source="simple",
            max_generation_tokens=256,
        )


def test_inspect_cut_candidate_skips_terminal_and_exhausted_cuts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "step_2/rollout_snapshots/snapshot_000001"
    snapshot.mkdir(parents=True)
    (snapshot / "manifest.json").write_text(
        json.dumps(
            {
                "base_train_step": 2,
                "gym_checkpoint": {"checkpoint_id": "checkpoint-2"},
            }
        )
    )
    torch.save(
        {
            "groups": [
                {
                    "group_id": "simple-group",
                    "task_source": "simple",
                    "siblings": [
                        {
                            "generation_index": 0,
                            "attempts": [{"attempt_index": 0}],
                        }
                    ],
                }
            ]
        },
        snapshot / "rollout_recovery.pt",
    )
    monkeypatch.setattr(
        _HELPER,
        "_active_prefixes",
        lambda *_: [
            {
                "rollout_id": "simple-group_g0",
                "attempt_index": 0,
                "model_call_id": "terminal-call",
                "prefix_token_count": 20,
                "prefix_digest": "1" * 64,
                "staging_keys": ["terminal-key"],
                "effective_output_limit": 64,
                "terminal_finish_reason": "stop",
            },
            {
                "rollout_id": "simple-group_g0",
                "attempt_index": 0,
                "model_call_id": "exhausted-call",
                "prefix_token_count": 32,
                "prefix_digest": "2" * 64,
                "staging_keys": ["exhausted-key"],
                "effective_output_limit": 32,
                "terminal_finish_reason": None,
            },
        ],
    )
    monkeypatch.setattr(
        _HELPER,
        "_matching_agent_boundary",
        lambda *_args, **_kwargs: {
            "boundary_kind": "turn_complete",
            "pending_model": None,
            "last_committed_model_call_id": None,
        },
    )

    with pytest.raises(
        AssertionError,
        match="no recoverable nonterminal active prefix",
    ):
        _HELPER.inspect_cut_candidate(
            snapshot,
            min_train_step=2,
            task_source="simple",
            max_generation_tokens=256,
        )


def test_matching_agent_boundary_requires_a_completed_saved_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = {
        "participants": [
            {
                "participant": {
                    "component": "responses_api_agents",
                    "participant_name": "simple",
                }
            }
        ]
    }
    record = {
        "rollout_id": "group_g0",
        "attempt_index": 0,
        "boundary_kind": "turn_complete",
        "pending_model": None,
    }
    monkeypatch.setattr(_HELPER, "_agent_records", lambda *_: [record])

    assert (
        _HELPER._matching_agent_boundary(
            tmp_path,
            checkpoint,
            rollout_id="group_g0",
            attempt_index=0,
        )
        == record
    )

    record["boundary_kind"] = "pending_model"
    record["pending_model"] = {"model_call_id": "call-1"}
    with pytest.raises(
        AssertionError,
        match="not anchored by a completed agent boundary",
    ):
        _HELPER._matching_agent_boundary(
            tmp_path,
            checkpoint,
            rollout_id="group_g0",
            attempt_index=0,
        )


@pytest.mark.parametrize(
    ("requirement", "boundary", "expected"),
    [
        (
            "root",
            {
                "boundary_index": 0,
                "last_committed_model_call_id": None,
                "resource_state_revisions": {"resources": 1},
            },
            True,
        ),
        (
            "root",
            {
                "boundary_index": 3,
                "last_committed_model_call_id": "call-1",
                "resource_state_revisions": {"resources": 2},
            },
            False,
        ),
        (
            "post_mutation",
            {
                "boundary_index": 3,
                "last_committed_model_call_id": "call-1",
                "resource_state_revisions": {"resources": 2},
            },
            True,
        ),
        (
            "post_mutation",
            {
                "boundary_index": 0,
                "last_committed_model_call_id": None,
                "resource_state_revisions": {"resources": 1},
            },
            False,
        ),
    ],
)
def test_boundary_satisfies_requirement(
    requirement: str,
    boundary: dict,
    expected: bool,
) -> None:
    assert _HELPER._boundary_satisfies_requirement(boundary, requirement) is expected


def test_published_snapshots_skips_snapshot_retired_mid_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    survivor = checkpoint_dir / "bootstrap/rollout_snapshots/snapshot_000001"
    retired = checkpoint_dir / "bootstrap/rollout_snapshots/snapshot_000002"
    for path in (survivor, retired):
        path.mkdir(parents=True)
        (path / "manifest.json").write_text("{}")

    real_stat = Path.stat

    def _stat(self: Path, **kwargs: object) -> os.stat_result:
        if self == retired:
            raise FileNotFoundError(2, "No such file or directory", str(self))
        return real_stat(self, **kwargs)

    monkeypatch.setattr(Path, "stat", _stat)

    assert _HELPER._published_snapshots(checkpoint_dir) == [survivor]


def test_published_snapshots_ignores_uncommitted_trainer_anchors(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    bootstrap = checkpoint_dir / "bootstrap/rollout_snapshots/snapshot_000001"
    published = checkpoint_dir / "step_2/rollout_snapshots/snapshot_000001"
    temporary = checkpoint_dir / "tmp_step_3/rollout_snapshots/snapshot_000001"
    replaced = checkpoint_dir / "old_step_4/rollout_snapshots/snapshot_000001"
    for path in (bootstrap, published, temporary, replaced):
        path.mkdir(parents=True)
        (path / "manifest.json").write_text("{}")

    os.utime(bootstrap, ns=(1_000_000_000, 1_000_000_000))
    os.utime(published, ns=(2_000_000_000, 2_000_000_000))
    os.utime(temporary, ns=(4_000_000_000, 4_000_000_000))
    os.utime(replaced, ns=(3_000_000_000, 3_000_000_000))

    assert _HELPER._published_snapshots(checkpoint_dir) == [published, bootstrap]


def test_prune_to_selection_removes_only_newer_progress(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    selected = checkpoint_dir / "step_2/rollout_snapshots/snapshot_000002"
    older_same_anchor = checkpoint_dir / "step_2/rollout_snapshots/snapshot_000001"
    older_step = checkpoint_dir / "step_1/rollout_snapshots/snapshot_000001"
    newer_step = checkpoint_dir / "step_3/rollout_snapshots/snapshot_000001"
    for path in (selected, older_same_anchor, older_step, newer_step):
        path.mkdir(parents=True)
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "snapshot_path": str(selected),
                "base_train_step": 2,
            }
        )
    )

    _HELPER.prune_to_selection(
        argparse.Namespace(
            checkpoint_dir=checkpoint_dir,
            selection=selection,
        )
    )

    assert selected.is_dir()
    assert not older_same_anchor.exists()
    assert older_step.is_dir()
    assert not newer_step.exists()


def _events(*, retried: bool) -> list[dict]:
    records = []
    for target_step, prompt_idx, task_source in (
        (0, 0, "simple"),
        (0, 1, "workplace"),
    ):
        suffix = "-a1" if retried else ""
        records.append(
            {
                "event": "dispatch",
                "timestamp_ns": (len(records) + 1) * 1_000_000_000,
                "group_id": f"random-{target_step}-{prompt_idx}",
                "target_step": target_step,
                "prompt_idx": prompt_idx,
                "task_source": task_source,
                "generation_indices": [0, 1],
                "rollout_ids": [
                    f"r{prompt_idx}_g0{suffix}",
                    f"r{prompt_idx}_g1{suffix}",
                ],
            }
        )
        for generation_index in range(2):
            completion = {
                "target_step": target_step,
                "prompt_idx": prompt_idx,
                "task_source": task_source,
                "generation_index": generation_index,
                "reward": float(prompt_idx),
                "rollout_id": f"r{prompt_idx}_g{generation_index}{suffix}",
            }
            for event_name in ("completion_arrived", "completion_forwarded"):
                records.append(
                    {
                        "event": event_name,
                        "timestamp_ns": (len(records) + 1) * 1_000_000_000,
                        **completion,
                    }
                )
    return records


def _event_identity(record: dict) -> tuple:
    return (
        record.get("target_step"),
        record.get("prompt_idx"),
        record.get("task_source"),
        record.get("generation_index"),
    )


def _retimestamp_events(records: list[dict]) -> None:
    for position, record in enumerate(records, start=1):
        record["timestamp_ns"] = position * 1_000_000_000


def _swap_first_two_completion_pairs(records: list[dict]) -> None:
    identities: list[tuple] = []
    for record in records:
        if record.get("event") != "completion_forwarded":
            continue
        identity = _event_identity(record)
        if identity not in identities:
            identities.append(identity)
        if len(identities) == 2:
            break
    assert len(identities) == 2

    selected_positions = [
        position
        for position, record in enumerate(records)
        if record.get("event") in ("completion_arrived", "completion_forwarded")
        and _event_identity(record) in identities
    ]
    by_identity = {
        identity: [
            record
            for record in records
            if record.get("event") in ("completion_arrived", "completion_forwarded")
            and _event_identity(record) == identity
        ]
        for identity in identities
    }
    replacement = by_identity[identities[1]] + by_identity[identities[0]]
    assert len(selected_positions) == len(replacement) == 4
    for position, record in zip(selected_positions, replacement, strict=True):
        records[position] = record
    _retimestamp_events(records)


def _training_record() -> dict:
    return {
        "idx": 0,
        "content": ["prompt", "answer"],
        "rewards": 1.0,
        "input_lengths": 2,
        "token_ids": [1, 2, 3],
        "token_loss_mask": [0, 1, 1],
        "sample_loss_mask": 1,
        "advantages": [0.0, 0.25, 0.25],
        "generation_logprobs": [0.0, -0.2, -0.3],
        "prev_logprobs": [0.0, -0.2, -0.3],
    }


def _write_compare_fixture(tmp_path: Path) -> argparse.Namespace:
    baseline_events = tmp_path / "baseline-events.jsonl"
    recovery_events = tmp_path / "recovery-events.jsonl"
    _write_jsonl(baseline_events, _events(retried=False))
    _write_jsonl(recovery_events, _events(retried=True))

    baseline_logs = tmp_path / "baseline-logs"
    recovery_logs = tmp_path / "recovery-logs"
    _write_jsonl(baseline_logs / "exp_001/train_data_step1.jsonl", [_training_record()])
    nearly_equal = _training_record()
    nearly_equal["advantages"][1] += 1e-8
    _write_jsonl(recovery_logs / "exp_002/train_data_step1.jsonl", [nearly_equal])

    metrics = {
        "train/reward": {"1": 1.0},
        "train/loss": {"1": 0.25},
        "train/gen_kl_error": {"1": 0.01},
        "timing/train/total_step_time": {"1": 99.0},
    }
    baseline_metrics = tmp_path / "baseline-metrics.json"
    recovery_metrics = tmp_path / "recovery-metrics.json"
    baseline_metrics.write_text(json.dumps(metrics))
    recovery_metrics.write_text(
        json.dumps(
            {
                **metrics,
                "timing/train/total_step_time": {"1": 999.0},
            }
        )
    )

    baseline_audit = tmp_path / "baseline-audit.jsonl"
    recovery_audit = tmp_path / "recovery-audit.jsonl"
    mutation = {"event": "mutation_applied", "sentinel_count": 1}
    _write_jsonl(baseline_audit, [mutation, mutation])
    _write_jsonl(
        recovery_audit,
        [mutation, mutation, {"event": "state_restored", "sentinel_count": 1}],
    )
    return argparse.Namespace(
        baseline_events=baseline_events,
        recovery_events=[recovery_events],
        baseline_log_dir=baseline_logs,
        recovery_log_dir=recovery_logs,
        baseline_metrics=baseline_metrics,
        recovery_metrics=recovery_metrics,
        baseline_audit=baseline_audit,
        recovery_audit=[recovery_audit],
        steps=1,
        prompts_per_step=2,
        generations_per_prompt=2,
        required_retried_task_source=["simple", "workplace"],
        rtol=1e-5,
        atol=1e-6,
        timeline_output=None,
        allow_missing_training_payloads=True,
        require_completion_order_match=False,
    )


def test_compare_runs_accepts_logically_identical_multi_crash_run(
    tmp_path: Path,
) -> None:
    _HELPER.compare_runs(_write_compare_fixture(tmp_path))


def test_compare_runs_accepts_stage_replayed_after_prune(tmp_path: Path) -> None:
    args = _write_compare_fixture(tmp_path)
    crashed_events = _events(retried=False)
    for record in crashed_events:
        if record["event"] == "completion_forwarded":
            record["reward"] = 99.0
    crashed_stage = tmp_path / "recovery-events-stage0.jsonl"
    _write_jsonl(crashed_stage, crashed_events)
    args.recovery_events.insert(0, crashed_stage)

    crashed_audit = tmp_path / "recovery-audit-stage0.jsonl"
    _write_jsonl(
        crashed_audit,
        [{"event": "mutation_applied", "sentinel_count": 1}],
    )
    args.recovery_audit.insert(0, crashed_audit)

    stale = args.recovery_log_dir / "exp_001/train_data_step1.jsonl"
    stale_record = _training_record()
    stale_record["token_ids"][-1] = 4
    _write_jsonl(stale, [stale_record])
    survivor = args.recovery_log_dir / "exp_002/train_data_step1.jsonl"
    os.utime(stale, ns=(1_000_000_000, 1_000_000_000))
    os.utime(survivor, ns=(2_000_000_000, 2_000_000_000))

    _HELPER.compare_runs(args)


def test_compare_runs_ignores_untrained_lookahead_events(tmp_path: Path) -> None:
    args = _write_compare_fixture(tmp_path)
    records = _events(retried=True)
    records.extend(
        [
            {
                "event": "dispatch",
                "timestamp_ns": 20_000_000_000,
                "target_step": 1,
                "prompt_idx": 2,
                "task_source": "simple",
                "generation_indices": [0],
                "rollout_ids": ["lookahead_g0"],
            },
            {
                "event": "completion_forwarded",
                "timestamp_ns": 21_000_000_000,
                "target_step": 1,
                "prompt_idx": 2,
                "task_source": "simple",
                "generation_index": 0,
                "rollout_id": "lookahead_g0",
                "reward": 123.0,
            },
        ]
    )
    _write_jsonl(args.recovery_events[0], records)

    _HELPER.compare_runs(args)


def test_compare_runs_rejects_changed_prompt_order(tmp_path: Path) -> None:
    args = _write_compare_fixture(tmp_path)
    records = _events(retried=True)
    first = next(
        record
        for record in records
        if record["event"] == "dispatch" and record["task_source"] == "simple"
    )
    records.remove(first)
    workplace_dispatch = next(
        position
        for position, record in enumerate(records)
        if record["event"] == "dispatch" and record["task_source"] == "workplace"
    )
    records.insert(workplace_dispatch + 1, first)
    _retimestamp_events(records)
    _write_jsonl(args.recovery_events[0], records)

    with pytest.raises(AssertionError, match="logical prompt order differs"):
        _HELPER.compare_runs(args)


def test_compare_runs_rejects_duplicate_completion(tmp_path: Path) -> None:
    args = _write_compare_fixture(tmp_path)
    records = _events(retried=True)
    forwarded = next(
        record for record in records if record["event"] == "completion_forwarded"
    )
    records.append(dict(forwarded))
    _write_jsonl(args.recovery_events[0], records)

    with pytest.raises(AssertionError, match="forwarded twice"):
        _HELPER.compare_runs(args)


def test_compare_runs_rejects_changed_completion_order(tmp_path: Path) -> None:
    args = _write_compare_fixture(tmp_path)
    args.require_completion_order_match = True
    records = _events(retried=True)
    _swap_first_two_completion_pairs(records)
    _write_jsonl(args.recovery_events[0], records)

    with pytest.raises(AssertionError, match="logical completion order differs"):
        _HELPER.compare_runs(args)


def test_compare_runs_reports_changed_completion_order_by_default(
    tmp_path: Path,
) -> None:
    args = _write_compare_fixture(tmp_path)
    args.timeline_output = tmp_path / "rollout-timeline.json"
    records = _events(retried=True)
    _swap_first_two_completion_pairs(records)
    _write_jsonl(args.recovery_events[0], records)

    _HELPER.compare_runs(args)

    timeline = json.loads(args.timeline_output.read_text())
    assert timeline["completion_order_matches"] is False
    individual = timeline["ordering_parity"]["forwarded_completion"]["individual"]
    assert individual["discordant_pairs"] == 1
    assert individual["comparable_pairs"] == 6
    assert individual["kendall_tau"] == pytest.approx(2 / 3)
    arrival = timeline["ordering_parity"]["arrival"]
    assert arrival["complete"] is True
    assert arrival["individual"]["discordant_pairs"] == 1
    assert arrival["individual"]["kendall_tau"] == pytest.approx(2 / 3)


def test_compare_runs_writes_rollout_timeline(tmp_path: Path) -> None:
    args = _write_compare_fixture(tmp_path)
    args.timeline_output = tmp_path / "rollout-timeline.json"

    _HELPER.compare_runs(args)

    timeline = json.loads(args.timeline_output.read_text())
    assert timeline["completion_order_matches"] is True
    assert [entry["effective_completion_rank"] for entry in timeline["baseline"]] == [
        0,
        1,
        2,
        3,
    ]
    assert timeline["recovery"][0]["dispatches"][0]["rollout_id"] == "r0_g0-a1"
    group_ready = timeline["ordering_parity"]["forwarded_completion"][
        "prompt_group_ready"
    ]
    assert group_ready["within_step_exact_match"] is True
    assert group_ready["kendall_tau"] == 1.0
    arrival = timeline["ordering_parity"]["arrival"]
    assert arrival["complete"] is True
    assert arrival["individual"]["within_step_exact_match"] is True
    assert arrival["prompt_group_last_sibling"]["kendall_tau"] == 1.0


def test_rank_metrics_compare_only_within_train_step() -> None:
    first = (0, 0, "simple", 0)
    second = (0, 1, "workplace", 0)
    later_step = (1, 2, "simple", 0)

    metrics = _HELPER._within_step_rank_metrics(
        [first, second, later_step],
        [later_step, second, first],
    )

    assert metrics["comparable_pairs"] == 1
    assert metrics["discordant_pairs"] == 1
    assert metrics["inversion_rate"] == 1.0
    assert metrics["kendall_tau"] == -1.0


def test_prompt_group_ready_order_uses_last_sibling() -> None:
    simple_0 = (0, 0, "simple", 0)
    simple_1 = (0, 0, "simple", 1)
    workplace_0 = (0, 1, "workplace", 0)
    workplace_1 = (0, 1, "workplace", 1)

    baseline = _HELPER._prompt_group_ready_order(
        [simple_0, workplace_0, simple_1, workplace_1]
    )
    recovery = _HELPER._prompt_group_ready_order(
        [simple_0, workplace_0, workplace_1, simple_1]
    )

    assert baseline == [simple_0[:3], workplace_0[:3]]
    assert recovery == [workplace_0[:3], simple_0[:3]]
    metrics = _HELPER._within_step_rank_metrics(baseline, recovery)
    assert metrics["discordant_pairs"] == 1
    assert metrics["kendall_tau"] == -1.0


def test_effective_arrival_order_uses_retained_recovery_attempt() -> None:
    def event(
        event_name: str,
        *,
        prompt_idx: int,
        rollout_id: str,
    ) -> dict:
        return {
            "event": event_name,
            "target_step": 0,
            "prompt_idx": prompt_idx,
            "task_source": "simple",
            "generation_index": 0,
            "rollout_id": rollout_id,
            "reward": 1.0,
        }

    stages = [
        [
            event("completion_arrived", prompt_idx=0, rollout_id="r0_g0"),
            event("completion_forwarded", prompt_idx=0, rollout_id="r0_g0"),
        ],
        [
            event("completion_arrived", prompt_idx=1, rollout_id="r1_g0-a1"),
            event("completion_forwarded", prompt_idx=1, rollout_id="r1_g0-a1"),
            event("completion_arrived", prompt_idx=0, rollout_id="r0_g0-a1"),
            event("completion_forwarded", prompt_idx=0, rollout_id="r0_g0-a1"),
        ],
    ]

    order, missing = _HELPER._effective_arrival_order(stages)

    assert order == [
        (0, 1, "simple", 0),
        (0, 0, "simple", 0),
    ]
    assert missing == []


def test_compare_runs_accepts_missing_training_payload_dumps(tmp_path: Path) -> None:
    args = _write_compare_fixture(tmp_path)
    for path in (
        args.baseline_log_dir / "exp_001/train_data_step1.jsonl",
        args.recovery_log_dir / "exp_002/train_data_step1.jsonl",
    ):
        path.unlink()

    _HELPER.compare_runs(args)


def test_compare_runs_requires_opt_in_for_missing_training_payload_dumps(
    tmp_path: Path,
) -> None:
    args = _write_compare_fixture(tmp_path)
    args.allow_missing_training_payloads = False
    for path in (
        args.baseline_log_dir / "exp_001/train_data_step1.jsonl",
        args.recovery_log_dir / "exp_002/train_data_step1.jsonl",
    ):
        path.unlink()

    with pytest.raises(AssertionError, match="allow-missing-training-payloads"):
        _HELPER.compare_runs(args)


def test_compare_runs_rejects_one_sided_training_payload_dumps(
    tmp_path: Path,
) -> None:
    args = _write_compare_fixture(tmp_path)
    (args.recovery_log_dir / "exp_002/train_data_step1.jsonl").unlink()

    with pytest.raises(AssertionError, match="present for only one run"):
        _HELPER.compare_runs(args)


def test_rollout_timeline_accepts_subset_redispatch() -> None:
    stages = [
        [
            {
                "event": "dispatch",
                "timestamp_ns": 1,
                "target_step": 0,
                "prompt_idx": 0,
                "task_source": "simple",
                "generation_indices": [1],
                "rollout_ids": ["r0_g1-a1"],
            },
            {
                "event": "completion_forwarded",
                "timestamp_ns": 2,
                "target_step": 0,
                "prompt_idx": 0,
                "task_source": "simple",
                "generation_index": 1,
                "rollout_id": "r0_g1-a1",
                "reward": 1.0,
            },
        ]
    ]
    order, _ = _HELPER._effective_completions(stages)

    timeline = _HELPER._rollout_timeline(stages, effective_order=order)

    assert timeline[0]["dispatches"][0]["rollout_id"] == "r0_g1-a1"


def test_compare_runs_rejects_changed_tokens(tmp_path: Path) -> None:
    args = _write_compare_fixture(tmp_path)
    path = args.recovery_log_dir / "exp_002/train_data_step1.jsonl"
    record = _training_record()
    record["token_ids"][-1] = 4
    _write_jsonl(path, [record])

    with pytest.raises(AssertionError, match="token_ids"):
        _HELPER.compare_runs(args)
