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
import copy
import hashlib
from unittest.mock import AsyncMock

import pytest

from nemo_rl.environments.nemo_gym import NemoGym, _external_staging_backend

# Receipt assembly imports nemo_gym at call time (resolve_terminal etc.), so
# these tests must run in the Nemo_Gym shard, not the base-env Environments one.
pytestmark = pytest.mark.nemo_gym


def _capture_env() -> NemoGym:
    env_cls = NemoGym.__ray_metadata__.modified_class
    env = object.__new__(env_cls)
    env._token_capture_enabled = True
    env._retire_pending = {}
    env._retire_full = asyncio.Event()
    env._retire_task = None
    return env


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


@pytest.mark.parametrize(
    ("generation_backend", "expected"),
    [("vllm", "vllm_worker"), ("megatron", "megatron_worker")],
)
def test_external_staging_backend_maps_generation_backend(
    generation_backend: str, expected: str
) -> None:
    token_capture = {"generation_backend": generation_backend}

    assert _external_staging_backend(token_capture) == expected
    assert token_capture == {"generation_backend": generation_backend}


@pytest.mark.parametrize(
    "token_capture",
    [{}, {"generation_backend": None}, {"generation_backend": "sglang"}],
)
def test_external_staging_backend_rejects_missing_or_invalid_backend(
    token_capture: dict,
) -> None:
    with pytest.raises(ValueError, match="setup-derived generation_backend"):
        _external_staging_backend(token_capture)


@pytest.mark.parametrize(
    "content_part",
    [
        {"type": "image_url", "image_url": {"url": "data:,"}},
        {"type": "video_url", "video_url": {"url": "data:,"}},
    ],
    ids=["image", "video"],
)
def test_megatron_capture_handler_admits_media_requests(content_part: dict) -> None:
    """The pinned Gym must admit media on the Megatron capture path.

    Older Gym revisions reject image and video parts there, which would only
    surface on the first media request of a run.
    """
    from nemo_gym.token_id_capture.external_capture import (
        make_external_capture_handler,
    )
    from nemo_gym.token_id_capture.lineage import InMemoryLineageStore
    from nemo_gym.token_id_capture.sink import (
        CaptureContext,
        reset_token_sink,
        set_token_sink,
    )
    from nemo_gym.token_id_capture.staging.records import CaptureAdmission

    admission = CaptureAdmission(rollout_id="r0", model_call_id="c1", mode="text")
    context = CaptureContext(
        rollout_id="r0",
        model_call_id="c1",
        token_sink=None,
        lineage_store=InMemoryLineageStore(),
        external_staging=True,
        capture_admission=admission,
    )
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "describe"}, content_part],
        }
    ]
    handler = make_external_capture_handler(
        _external_staging_backend({"generation_backend": "megatron"})
    )
    token = set_token_sink(context)
    try:
        payload = handler.prepare_request({"messages": copy.deepcopy(messages)})
    finally:
        reset_token_sink(token)

    assert payload["messages"] == messages
    assert payload["offload_params"]["ng_capture"] == admission.model_dump(mode="json")


def _manifest_record(
    call_id: str,
    *,
    response_id: str | None = None,
    parent: str | None = None,
    cumulative_hash: str | None = None,
) -> dict:
    # CallRecord requires both chain digests (Gym e5780688); derive distinct
    # placeholders per call so identical-content collapsing stays off unless a
    # test opts in by passing the same cumulative_hash twice.
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
        "response_id": response_id or f"resp-{call_id}",
        "chain_hash": _digest(f"chain:{call_id}"),
        "cumulative_hash": cumulative_hash or _digest(f"cumulative:{call_id}"),
    }


def test_receipt_postprocess_without_a_terminal_response_id_uses_the_heuristic() -> (
    None
):
    env = _capture_env()
    records = [
        _manifest_record("c1"),
        _manifest_record("c2", parent="c1"),
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
        _manifest_record("c1"),
        _manifest_record("c2", parent="c1"),
    ]
    env._control = AsyncMock(
        return_value={"rollout_id": "r0", "records": records, "failures": []}
    )

    result = asyncio.run(
        env._postprocess_receipt_mode(
            {"_ng_rollout_id": "r0"},
            {"reward": 1.0, "terminal_response_id": "resp-c2"},
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
        "records": [_manifest_record("c1")],
        "failures": [{"model_call_id": "c2", "reason": "worker_capture_failed"}],
    }
    receipt = env._assemble_receipt(
        "r0", manifest, terminal_response_id="resp-c1", reward=0.0
    )
    assert receipt["capture_poisoned"] is True
    assert receipt["failure_reason"] == "worker_capture_failed"


def test_receipt_assembly_ignores_uncommitted_call_failures_off_the_terminal_chain() -> (
    None
):
    """A call that died without coordinates never served a completion and can
    never be a lineage parent (no committed row to resolve against), so it is
    structurally off-chain — e.g. the doomed final call of a rollout that
    exhausted the context window. It must not poison the verified chain."""
    env = _capture_env()
    manifest = {
        "rollout_id": "r0",
        "records": [
            _manifest_record("c1"),
            _manifest_record("c2", parent="c1"),
        ],
        "failures": [
            {
                "model_call_id": "c3",
                "reason": "request_finished_without_staged_coordinates",
            }
        ],
    }
    receipt = env._assemble_receipt(
        "r0", manifest, terminal_response_id="resp-c2", reward=1.0
    )
    assert receipt["capture_poisoned"] is False
    assert receipt["failure_reason"] is None
    assert receipt["terminal_model_call_id"] == "c2"


def test_receipt_assembly_still_poisons_when_the_terminal_call_died_uncommitted() -> (
    None
):
    """If the reported terminal request itself died without coordinates there
    is no terminal row — the missing-terminal check must mask the rollout."""
    env = _capture_env()
    manifest = {
        "rollout_id": "r0",
        "records": [_manifest_record("c1")],
        "failures": [
            {
                "model_call_id": "c2",
                "reason": "request_finished_without_staged_coordinates",
            }
        ],
    }
    receipt = env._assemble_receipt(
        "r0", manifest, terminal_response_id="resp-c2", reward=0.0
    )
    assert receipt["capture_poisoned"] is True
    assert receipt["failure_reason"] == "missing_terminal_row"


def test_receipt_assembly_poisons_when_the_terminal_row_is_missing() -> None:
    env = _capture_env()
    manifest = {
        "rollout_id": "r0",
        "records": [_manifest_record("c1")],
        "failures": [],
    }
    receipt = env._assemble_receipt(
        "r0", manifest, terminal_response_id="resp-lost", reward=0.0
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
        _manifest_record("c1"),
        _manifest_record("c2", parent="c1"),
        _manifest_record("c2r", parent="c1"),
        _manifest_record("c3", parent="c2"),
    ]
    manifest = {"rollout_id": "r0", "records": records, "failures": []}
    receipt = env._assemble_receipt(
        "r0", manifest, terminal_response_id=None, reward=1.0
    )
    assert receipt["terminal_model_call_id"] == "c3"
    assert receipt["terminal_selection"] == "heuristic"
    assert receipt["capture_poisoned"] is False


def test_receipt_assembly_heuristic_masks_a_final_call_retry() -> None:
    env = _capture_env()
    records = [
        _manifest_record("c1"),
        _manifest_record("c2", parent="c1"),
        _manifest_record("c2r", parent="c1"),
    ]
    manifest = {"rollout_id": "r0", "records": records, "failures": []}
    receipt = env._assemble_receipt(
        "r0", manifest, terminal_response_id=None, reward=0.0
    )
    assert receipt["terminal_model_call_id"] is None
    assert receipt["capture_poisoned"] is True
    assert receipt["failure_reason"] == "ambiguous_terminal"


def test_receipt_assembly_heuristic_masks_an_empty_manifest() -> None:
    env = _capture_env()
    manifest = {"rollout_id": "r0", "records": [], "failures": []}
    receipt = env._assemble_receipt(
        "r0", manifest, terminal_response_id=None, reward=0.0
    )
    assert receipt["terminal_model_call_id"] is None
    assert receipt["capture_poisoned"] is True
    assert receipt["failure_reason"] == "no_records"


def test_receipt_assembly_heuristic_masks_invalid_manifest_rows() -> None:
    env = _capture_env()
    bad = _manifest_record("c1")
    bad["delta_len"] = 0  # violates the CallRecord length contract
    manifest = {"rollout_id": "r0", "records": [bad], "failures": []}
    receipt = env._assemble_receipt(
        "r0", manifest, terminal_response_id=None, reward=0.0
    )
    assert receipt["terminal_model_call_id"] is None
    assert receipt["capture_poisoned"] is True
    assert receipt["failure_reason"] == "invalid_manifest_row"
    assert receipt["manifest"] == []


def test_receipt_assembly_ships_only_rows_that_parse() -> None:
    """One bad row masks the rollout but must not drop the good rows: the
    finalizer needs their staging keys to clean the staged TQ rows."""
    env = _capture_env()
    good = _manifest_record("c1")
    bad = _manifest_record("c2", parent="c1")
    bad["delta_len"] = 0  # violates the CallRecord length contract
    manifest = {"rollout_id": "r0", "records": [good, bad], "failures": []}
    receipt = env._assemble_receipt(
        "r0", manifest, terminal_response_id=None, reward=0.0
    )
    assert receipt["failure_reason"] == "invalid_manifest_row"
    assert receipt["terminal_model_call_id"] is None
    assert receipt["manifest"] == [good]


def test_finalizer_cleans_good_rows_when_a_manifest_row_is_invalid() -> None:
    """End to end through finalize_rollout: the receipt assembled from a
    manifest with one bad row must reject as rollout_failed (not
    invalid_receipt) and carry the good row's staging key."""
    from nemo_rl.experience.rollout_reassembler import RolloutReassembler

    env = _capture_env()
    good = _manifest_record("c1")
    bad = _manifest_record("c2", parent="c1")
    del bad["chain_hash"]
    manifest = {"rollout_id": "r0", "records": [good, bad], "failures": []}
    receipt = env._assemble_receipt(
        "r0", manifest, terminal_response_id=None, reward=0.0
    )
    # Rejection happens before any staging read, so no TQ client is needed.
    finalizer = RolloutReassembler(
        dp_client=None,
        partition_id="canonical",
        staging_partition="staging",
        pad_token_id=0,
        max_seq_len=64,
    )
    row = finalizer.finalize_rollout("r0", receipt, reward=0.0)
    assert row.valid is False
    assert row.rejection_reason == "rollout_failed:invalid_manifest_row"
    assert row.staging_keys == [good["staging_key"]]


def test_receipt_assembly_keeps_dead_branch_siblings_in_the_manifest() -> None:
    """A retry sibling stays enumerable (its staged row must be cleaned) but
    never becomes the terminal call."""
    env = _capture_env()
    records = [
        _manifest_record("c1"),
        _manifest_record("c2", parent="c1"),
        _manifest_record("c2r", parent="c1"),
    ]
    manifest = {"rollout_id": "r0", "records": records, "failures": []}
    receipt = env._assemble_receipt(
        "r0", manifest, terminal_response_id="resp-c2r", reward=1.0
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
            {"reward": 1.0, "terminal_response_id": "resp-c1"},
        )
    )
    assert result["receipt"] is None


def test_response_id_witness_resolves_a_final_call_retry() -> None:
    """The heuristic masks a retried final call; the scored response's served
    envelope id names the sibling the harness kept, recovering the rollout."""
    env = _capture_env()
    records = [
        _manifest_record("c1"),
        _manifest_record("c2", parent="c1", cumulative_hash="a" * 64),
        _manifest_record("c2r", parent="c1", cumulative_hash="c" * 64),
    ]
    manifest = {"rollout_id": "r0", "records": records, "failures": []}
    receipt = env._assemble_receipt(
        "r0",
        manifest,
        terminal_response_id=None,
        scored_response={"id": "resp-c2r", "output": []},
        reward=1.0,
    )
    assert receipt["terminal_model_call_id"] == "c2r"
    assert receipt["terminal_selection"] == "response_id"
    assert receipt["capture_poisoned"] is False


def test_unattributed_scored_response_falls_back_to_the_heuristic() -> None:
    env = _capture_env()
    records = [
        _manifest_record("c1"),
        _manifest_record("c2", parent="c1"),
    ]
    manifest = {"rollout_id": "r0", "records": records, "failures": []}
    receipt = env._assemble_receipt(
        "r0",
        manifest,
        terminal_response_id=None,
        scored_response={"id": "resp-unknown", "output": []},
        reward=1.0,
    )
    assert receipt["terminal_model_call_id"] == "c2"
    assert receipt["terminal_selection"] == "heuristic"
    assert "response_id_no_match" in (receipt["terminal_attribution_reason"] or "")


def test_receipt_assembly_leaves_terminal_selection_unset_on_invalid_row() -> None:
    env = _capture_env()
    invalid = _manifest_record("c2", parent="c1")
    del invalid["chain_hash"]  # CallRecord requires it: the manifest fails to parse
    records = [_manifest_record("c1"), invalid]
    manifest = {"rollout_id": "r0", "records": records, "failures": []}
    receipt = env._assemble_receipt(
        "r0", manifest, terminal_response_id=None, reward=0.0
    )
    assert receipt["capture_poisoned"] is True
    assert receipt["failure_reason"] == "invalid_manifest_row"
    assert receipt["terminal_model_call_id"] is None
    # No attribution stage ran, so the receipt must not be stamped with a
    # method (a "heuristic" label here would inflate that bucket's fraction).
    assert receipt["terminal_selection"] is None
    assert receipt["terminal_attribution_reason"] is None
    assert [row["model_call_id"] for row in receipt["manifest"]] == ["c1"]


def test_declared_and_response_id_witnesses_corroborate() -> None:
    env = _capture_env()
    records = [
        _manifest_record("c1"),
        _manifest_record("c2", parent="c1"),
    ]
    manifest = {"rollout_id": "r0", "records": records, "failures": []}
    receipt = env._assemble_receipt(
        "r0",
        manifest,
        terminal_response_id="resp-c2",
        scored_response={"id": "resp-c2", "output": []},
        reward=1.0,
    )
    assert receipt["terminal_model_call_id"] == "c2"
    assert receipt["terminal_selection"] == "declared"
    assert "corroborated_by=response_id" in (
        receipt["terminal_attribution_reason"] or ""
    )


def test_witness_disagreement_masks_a_retry_instead_of_guessing() -> None:
    """A declaration naming one retry sibling while the scored response's id
    names the other is a contradiction: attribution abstains, the declared
    path stays authoritative, and the rollout masks."""
    env = _capture_env()
    records = [
        _manifest_record("c1"),
        _manifest_record("c2", parent="c1", cumulative_hash="a" * 64),
        _manifest_record("c2r", parent="c1", cumulative_hash="c" * 64),
    ]
    manifest = {"rollout_id": "r0", "records": records, "failures": []}
    receipt = env._assemble_receipt(
        "r0",
        manifest,
        terminal_response_id="resp-c2",
        scored_response={"id": "resp-c2r", "output": []},
        reward=1.0,
    )
    assert receipt["terminal_model_call_id"] is None
    assert receipt["capture_poisoned"] is True
    assert "witness_disagreement[" in (receipt["terminal_attribution_reason"] or "")


def test_postprocess_passes_the_scored_response_to_attribution() -> None:
    env = _capture_env()
    records = [
        _manifest_record("c1"),
        _manifest_record("c2", parent="c1", cumulative_hash="a" * 64),
        _manifest_record("c2r", parent="c1", cumulative_hash="c" * 64),
    ]
    env._control = AsyncMock(
        return_value={"rollout_id": "r0", "records": records, "failures": []}
    )
    result = asyncio.run(
        env._postprocess_receipt_mode(
            {"_ng_rollout_id": "r0"},
            {"reward": 1.0, "response": {"id": "resp-c2", "output": []}},
        )
    )
    receipt = result["receipt"]
    assert receipt["terminal_model_call_id"] == "c2"
    assert receipt["terminal_selection"] == "response_id"


def test_setup_nemo_gym_config_megatron_keeps_async_rollout_check_satisfied() -> None:
    """setup_nemo_gym_config must not set mcore_generation_config.async_engine.

    should_use_async_rollouts asserts that key is absent for the megatron
    backend, and should_use_nemo_gym calls it after setup_nemo_gym_config has run.
    """
    from nemo_rl.algorithms.grpo import MasterConfig
    from nemo_rl.environments.nemo_gym import (
        setup_nemo_gym_config,
        should_use_nemo_gym,
    )

    config = MasterConfig.model_construct(
        env={"should_use_nemo_gym": True},
        policy={
            "generation": {
                "backend": "megatron",
                "mcore_generation_config": {"expose_http_server": False},
            }
        },
    )

    setup_nemo_gym_config(config, tokenizer=None)

    mcore_cfg = config.policy["generation"]["mcore_generation_config"]
    assert mcore_cfg["expose_http_server"] is True
    assert "async_engine" not in mcore_cfg
    assert should_use_nemo_gym(config) is True


def test_setup_nemo_gym_config_dynamo_exposes_http_server_via_vllm_cfg() -> None:
    """Dynamo serves Gym through the vllm_cfg HTTP server settings."""
    from nemo_rl.algorithms.grpo import MasterConfig
    from nemo_rl.environments.nemo_gym import (
        setup_nemo_gym_config,
        should_use_nemo_gym,
    )

    config = MasterConfig.model_construct(
        env={"should_use_nemo_gym": True},
        policy={
            "generation": {
                "backend": "dynamo",
                "vllm_cfg": {"async_engine": False, "expose_http_server": False},
            }
        },
    )

    setup_nemo_gym_config(config, tokenizer=None)

    vllm_cfg = config.policy["generation"]["vllm_cfg"]
    assert vllm_cfg["async_engine"] is True
    assert vllm_cfg["expose_http_server"] is True
    assert should_use_nemo_gym(config) is True


class _ControlResponse:
    def __init__(self, status: int, text: str = "") -> None:
        self.status = status
        self._text = text
        self.released = False

    async def text(self) -> str:
        return self._text

    def release(self) -> None:
        self.released = True


def _retired_batches(env) -> list[list[str]]:
    return [
        call.kwargs["json"]["rollout_ids"]
        for call in env._control_response.await_args_list
    ]


def test_receipt_postprocess_queues_its_rollout_for_retirement() -> None:
    env = _capture_env()
    env._control = AsyncMock(
        return_value={
            "rollout_id": "r0",
            "records": [_manifest_record("c1")],
            "failures": [],
        }
    )

    async def _scenario() -> None:
        await env._postprocess_receipt_mode({"_ng_rollout_id": "r0"}, {"reward": 1.0})
        # Queued without a request of its own; the flush loop sends it later.
        assert list(env._retire_pending) == ["r0"]
        assert env._control.await_count == 1
        env._retire_task.cancel()

    asyncio.run(_scenario())


def test_receipt_postprocess_retires_even_when_the_manifest_fetch_fails() -> None:
    env = _capture_env()
    env._control = AsyncMock(side_effect=RuntimeError("control plane down"))

    async def _scenario() -> None:
        await env._postprocess_receipt_mode({"_ng_rollout_id": "r0"}, {"reward": 1.0})
        assert list(env._retire_pending) == ["r0"]
        env._retire_task.cancel()

    asyncio.run(_scenario())


def test_retire_is_a_no_op_without_token_capture() -> None:
    env = _capture_env()
    env._token_capture_enabled = False

    asyncio.run(env.retire_rollout_ledgers(["r0"]))

    assert env._retire_pending == {}
    assert env._retire_task is None


def test_retire_flush_sends_every_queued_rollout_in_one_request() -> None:
    env = _capture_env()
    env._retire_pending = dict.fromkeys(["r0", "r1", "r2"])
    env._control_response = AsyncMock(return_value=_ControlResponse(200))

    assert asyncio.run(env._flush_retired_rollouts()) is True

    assert _retired_batches(env) == [["r0", "r1", "r2"]]
    call = env._control_response.await_args
    assert call.args == ("POST", "/training-token-capture/control/rollouts/retire")
    # Cleanup rides Gym's reserved control pool, not the model-call pool.
    assert call.kwargs["_control"] is True
    assert env._retire_pending == {}


def test_retire_flush_splits_at_the_gym_batch_limit(monkeypatch) -> None:
    from nemo_gym.token_id_capture import control_routes

    monkeypatch.setattr(control_routes, "MAX_LEDGER_BATCH", 2)
    env = _capture_env()
    env._retire_pending = dict.fromkeys(["r0", "r1", "r2", "r3", "r4"])
    env._control_response = AsyncMock(return_value=_ControlResponse(200))

    assert asyncio.run(env._flush_retired_rollouts()) is True

    assert _retired_batches(env) == [["r0", "r1"], ["r2", "r3"], ["r4"]]


@pytest.mark.parametrize(
    "outcome",
    [RuntimeError("control plane down"), OSError("reset"), _ControlResponse(409)],
    ids=["timeout", "transport", "refused_while_checkpointing"],
)
def test_retire_flush_keeps_rollouts_queued_on_a_retryable_failure(outcome) -> None:
    env = _capture_env()
    env._retire_pending = dict.fromkeys(["r0", "r1"])
    if isinstance(outcome, Exception):
        env._control_response = AsyncMock(side_effect=outcome)
    else:
        env._control_response = AsyncMock(return_value=outcome)

    assert asyncio.run(env._flush_retired_rollouts()) is False

    assert list(env._retire_pending) == ["r0", "r1"]


def test_retire_flush_keeps_later_batches_after_an_earlier_one_succeeds(
    monkeypatch,
) -> None:
    from nemo_gym.token_id_capture import control_routes

    monkeypatch.setattr(control_routes, "MAX_LEDGER_BATCH", 2)
    env = _capture_env()
    env._retire_pending = dict.fromkeys(["r0", "r1", "r2"])
    env._control_response = AsyncMock(
        side_effect=[_ControlResponse(200), RuntimeError("control plane down")]
    )

    assert asyncio.run(env._flush_retired_rollouts()) is False

    assert list(env._retire_pending) == ["r2"]


@pytest.mark.parametrize("status", [400, 401, 422])
def test_retire_flush_drops_a_batch_a_retry_cannot_fix(status: int) -> None:
    env = _capture_env()
    env._retire_pending = dict.fromkeys(["r0"])
    env._control_response = AsyncMock(return_value=_ControlResponse(status, "bad"))

    assert asyncio.run(env._flush_retired_rollouts()) is True

    assert env._retire_pending == {}


def test_retire_loop_retries_with_backoff_until_the_queue_drains(monkeypatch) -> None:
    from nemo_rl.environments import nemo_gym

    monkeypatch.setattr(nemo_gym, "_RETIRE_FLUSH_INTERVAL_S", 0.001)
    monkeypatch.setattr(nemo_gym, "_RETIRE_MAX_BACKOFF_S", 0.001)
    env = _capture_env()
    env._control_response = AsyncMock(
        side_effect=[RuntimeError("control plane down"), _ControlResponse(200)]
    )

    async def _scenario() -> None:
        await env.retire_rollout_ledgers(["r0", "r0", "r1"])
        await asyncio.wait_for(env._retire_task, timeout=5)

    asyncio.run(_scenario())

    # The failed batch was resent whole, and the duplicate ID was sent once.
    assert _retired_batches(env) == [["r0", "r1"], ["r0", "r1"]]
    assert env._retire_pending == {}


def test_retire_loop_flushes_a_full_batch_without_waiting(monkeypatch) -> None:
    from nemo_rl.environments import nemo_gym

    monkeypatch.setattr(nemo_gym, "_RETIRE_BATCH_SIZE", 2)
    monkeypatch.setattr(nemo_gym, "_RETIRE_FLUSH_INTERVAL_S", 60.0)
    env = _capture_env()
    env._control_response = AsyncMock(return_value=_ControlResponse(200))

    async def _scenario() -> None:
        await env.retire_rollout_ledgers(["r0"])
        await env.retire_rollout_ledgers(["r1"])
        await asyncio.wait_for(env._retire_task, timeout=5)

    asyncio.run(_scenario())

    assert _retired_batches(env) == [["r0", "r1"]]
