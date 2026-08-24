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

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from nemo_rl.environments.nemo_gym import NemoGym


def _capture_env() -> NemoGym:
    env_cls = NemoGym.__ray_metadata__.modified_class
    return object.__new__(env_cls)


def _manifest_record(call_id: str, *, logical_request_id: str, parent: str | None = None) -> dict:
    prev_len = 0 if parent is None else 900
    return {
        "model_call_id": call_id,
        "parent_call_id": parent,
        "prev_len": prev_len,
        "delta_len": 100,
        "cum_len": prev_len + 100,
        "weight_version": 3,
        "digest": "a" * 64,
        "extras_digest": "b" * 64,
        "staging_key": f"r0/{call_id}",
        "mode": "text" if parent is None else "token_in",
        "logical_request_id": logical_request_id,
    }


def test_receipt_postprocess_without_a_terminal_logical_id_uses_the_heuristic() -> None:
    env = _capture_env()
    records = [
        _manifest_record("c1", logical_request_id="lr-1"),
        _manifest_record("c2", logical_request_id="lr-2", parent="c1"),
    ]
    env._control = AsyncMock(
        return_value={"rollout_id": "r0", "records": records, "failures": []}
    )

    result = asyncio.run(
        env._postprocess_receipt_mode(
            {"_ng_rollout_id": "r0"},
            {"reward": 1.0},
        )
    )

    env._control.assert_awaited()
    receipt = result["receipt"]
    assert receipt["terminal_model_call_id"] == "c2"
    assert receipt["terminal_selection"] == "heuristic"
    assert receipt["capture_poisoned"] is False


def test_receipt_postprocess_fetches_manifest_and_selects_terminal_row() -> None:
    env = _capture_env()
    records = [
        _manifest_record("c1", logical_request_id="lr-1"),
        _manifest_record("c2", logical_request_id="lr-2", parent="c1"),
    ]
    env._control = AsyncMock(
        return_value={"rollout_id": "r0", "records": records, "failures": []}
    )

    result = asyncio.run(
        env._postprocess_receipt_mode(
            {"_ng_rollout_id": "r0"},
            {"reward": 1.0, "terminal_logical_request_id": "lr-2"},
        )
    )

    call = env._control.await_args
    assert call.args == (
        "GET",
        "/training-token-capture/control/rollouts/r0/manifest",
    )
    receipt = result["receipt"]
    assert receipt["rollout_id"] == "r0"
    assert receipt["terminal_model_call_id"] == "c2"
    assert receipt["terminal_selection"] == "declared"
    assert receipt["capture_poisoned"] is False
    assert receipt["failure_reason"] is None
    assert receipt["reward"] == 1.0
    assert [r["model_call_id"] for r in receipt["manifest"]] == ["c1", "c2"]


def test_receipt_assembly_poisons_on_failure_rows() -> None:
    env = _capture_env()
    manifest = {
        "rollout_id": "r0",
        "records": [_manifest_record("c1", logical_request_id="lr-1")],
        "failures": [{"model_call_id": "c2", "reason": "worker_capture_failed"}],
    }
    receipt = env._assemble_receipt(
        "r0", manifest, terminal_logical_request_id="lr-1", reward=0.0
    )
    assert receipt["capture_poisoned"] is True
    assert receipt["failure_reason"] == "worker_capture_failed"


def test_receipt_assembly_ignores_uncommitted_call_failures_off_the_terminal_chain() -> None:
    """A call that died without coordinates never served a completion and can
    never be a lineage parent (no committed row to resolve against), so it is
    structurally off-chain — e.g. the doomed final call of a rollout that
    exhausted the context window. It must not poison the verified chain."""
    env = _capture_env()
    manifest = {
        "rollout_id": "r0",
        "records": [
            _manifest_record("c1", logical_request_id="lr-1"),
            _manifest_record("c2", logical_request_id="lr-2", parent="c1"),
        ],
        "failures": [
            {
                "model_call_id": "c3",
                "reason": "request_finished_without_staged_coordinates",
            }
        ],
    }
    receipt = env._assemble_receipt(
        "r0", manifest, terminal_logical_request_id="lr-2", reward=1.0
    )
    assert receipt["capture_poisoned"] is False
    assert receipt["failure_reason"] is None
    assert receipt["terminal_model_call_id"] == "c2"


def test_receipt_assembly_still_poisons_when_the_terminal_call_died_uncommitted() -> None:
    """If the reported terminal request itself died without coordinates there
    is no terminal row — the missing-terminal check must mask the rollout."""
    env = _capture_env()
    manifest = {
        "rollout_id": "r0",
        "records": [_manifest_record("c1", logical_request_id="lr-1")],
        "failures": [
            {
                "model_call_id": "c2",
                "reason": "request_finished_without_staged_coordinates",
            }
        ],
    }
    receipt = env._assemble_receipt(
        "r0", manifest, terminal_logical_request_id="lr-2", reward=0.0
    )
    assert receipt["capture_poisoned"] is True
    assert receipt["failure_reason"] == "missing_terminal_row"


def test_receipt_assembly_poisons_when_the_terminal_row_is_missing() -> None:
    env = _capture_env()
    manifest = {
        "rollout_id": "r0",
        "records": [_manifest_record("c1", logical_request_id="lr-1")],
        "failures": [],
    }
    receipt = env._assemble_receipt(
        "r0", manifest, terminal_logical_request_id="lr-lost", reward=0.0
    )
    assert receipt["capture_poisoned"] is True
    assert receipt["failure_reason"] == "missing_terminal_row"
    assert receipt["terminal_model_call_id"] is None
    # A declared id is authoritative: a miss never falls back to the heuristic
    # even when the manifest holds an unambiguous chain.
    assert receipt["terminal_selection"] == "declared"


def test_receipt_assembly_heuristic_eliminates_abandoned_retry() -> None:
    env = _capture_env()
    records = [
        _manifest_record("c1", logical_request_id="lr-1"),
        _manifest_record("c2", logical_request_id="lr-2", parent="c1"),
        _manifest_record("c2r", logical_request_id="lr-2r", parent="c1"),
        _manifest_record("c3", logical_request_id="lr-3", parent="c2"),
    ]
    manifest = {"rollout_id": "r0", "records": records, "failures": []}
    receipt = env._assemble_receipt(
        "r0", manifest, terminal_logical_request_id=None, reward=1.0
    )
    assert receipt["terminal_model_call_id"] == "c3"
    assert receipt["terminal_selection"] == "heuristic"
    assert receipt["capture_poisoned"] is False


def test_receipt_assembly_heuristic_masks_a_final_call_retry() -> None:
    env = _capture_env()
    records = [
        _manifest_record("c1", logical_request_id="lr-1"),
        _manifest_record("c2", logical_request_id="lr-2", parent="c1"),
        _manifest_record("c2r", logical_request_id="lr-2r", parent="c1"),
    ]
    manifest = {"rollout_id": "r0", "records": records, "failures": []}
    receipt = env._assemble_receipt(
        "r0", manifest, terminal_logical_request_id=None, reward=0.0
    )
    assert receipt["terminal_model_call_id"] is None
    assert receipt["capture_poisoned"] is True
    assert receipt["failure_reason"] == "ambiguous_terminal"


def test_receipt_assembly_heuristic_masks_an_empty_manifest() -> None:
    env = _capture_env()
    manifest = {"rollout_id": "r0", "records": [], "failures": []}
    receipt = env._assemble_receipt(
        "r0", manifest, terminal_logical_request_id=None, reward=0.0
    )
    assert receipt["terminal_model_call_id"] is None
    assert receipt["capture_poisoned"] is True
    assert receipt["failure_reason"] == "no_records"


def test_receipt_assembly_heuristic_masks_invalid_manifest_rows() -> None:
    env = _capture_env()
    bad = _manifest_record("c1", logical_request_id="lr-1")
    bad["delta_len"] = 0  # violates the CallRecord length contract
    manifest = {"rollout_id": "r0", "records": [bad], "failures": []}
    receipt = env._assemble_receipt(
        "r0", manifest, terminal_logical_request_id=None, reward=0.0
    )
    assert receipt["terminal_model_call_id"] is None
    assert receipt["capture_poisoned"] is True
    assert receipt["failure_reason"] == "invalid_manifest_row"


def test_receipt_assembly_keeps_dead_branch_siblings_in_the_manifest() -> None:
    """A retry sibling stays enumerable (its staged row must be cleaned) but
    never becomes the terminal call."""
    env = _capture_env()
    records = [
        _manifest_record("c1", logical_request_id="lr-1"),
        _manifest_record("c2", logical_request_id="lr-2", parent="c1"),
        _manifest_record("c2r", logical_request_id="lr-2r", parent="c1"),
    ]
    manifest = {"rollout_id": "r0", "records": records, "failures": []}
    receipt = env._assemble_receipt(
        "r0", manifest, terminal_logical_request_id="lr-2r", reward=1.0
    )
    assert receipt["terminal_model_call_id"] == "c2r"
    assert receipt["capture_poisoned"] is False
    assert {r["model_call_id"] for r in receipt["manifest"]} == {"c1", "c2", "c2r"}


def test_receipt_postprocess_returns_placeholder_on_fetch_failure() -> None:
    env = _capture_env()
    env._control = AsyncMock(side_effect=RuntimeError("control plane down"))

    result = asyncio.run(
        env._postprocess_receipt_mode(
            {"_ng_rollout_id": "r0"},
            {"reward": 1.0, "terminal_logical_request_id": "lr-1"},
        )
    )
    assert result["receipt"] is None
