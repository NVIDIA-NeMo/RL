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

    selected = _HELPER.inspect_cut_candidate(
        snapshot,
        min_train_step=2,
        task_source="simple",
        max_generation_tokens=256,
    )

    assert selected["task_source"] == "simple"
    assert selected["rollout_id"] == "simple-group_g0"
    assert selected["attempt_index"] == 1
    assert selected["prefix_token_count"] == 12


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
                "group_id": f"random-{target_step}-{prompt_idx}",
                "target_step": target_step,
                "prompt_idx": prompt_idx,
                "task_source": task_source,
                "rollout_ids": [f"r{prompt_idx}{suffix}"],
            }
        )
        records.extend(
            {
                "event": "completion_forwarded",
                "target_step": target_step,
                "prompt_idx": prompt_idx,
                "task_source": task_source,
                "generation_index": generation_index,
                "reward": float(prompt_idx),
            }
            for generation_index in range(2)
        )
    return records


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
        recovery_events=recovery_events,
        baseline_log_dir=baseline_logs,
        recovery_log_dir=recovery_logs,
        baseline_metrics=baseline_metrics,
        recovery_metrics=recovery_metrics,
        baseline_audit=baseline_audit,
        recovery_audit=recovery_audit,
        steps=1,
        prompts_per_step=2,
        generations_per_prompt=2,
        required_retried_task_source=["simple", "workplace"],
        rtol=1e-5,
        atol=1e-6,
    )


def test_compare_runs_accepts_logically_identical_multi_crash_run(
    tmp_path: Path,
) -> None:
    _HELPER.compare_runs(_write_compare_fixture(tmp_path))


def test_compare_runs_rejects_changed_prompt_order(tmp_path: Path) -> None:
    args = _write_compare_fixture(tmp_path)
    records = _events(retried=True)
    first = records.pop(0)
    records.insert(3, first)
    _write_jsonl(args.recovery_events, records)

    with pytest.raises(AssertionError, match="logical prompt order differs"):
        _HELPER.compare_runs(args)


def test_compare_runs_rejects_duplicate_completion(tmp_path: Path) -> None:
    args = _write_compare_fixture(tmp_path)
    records = _events(retried=True)
    records.append(dict(records[1]))
    _write_jsonl(args.recovery_events, records)

    with pytest.raises(AssertionError, match="forwarded twice"):
        _HELPER.compare_runs(args)


def test_compare_runs_rejects_changed_tokens(tmp_path: Path) -> None:
    args = _write_compare_fixture(tmp_path)
    path = args.recovery_log_dir / "exp_002/train_data_step1.jsonl"
    record = _training_record()
    record["token_ids"][-1] = 4
    _write_jsonl(path, [record])

    with pytest.raises(AssertionError, match="token_ids"):
        _HELPER.compare_runs(args)
