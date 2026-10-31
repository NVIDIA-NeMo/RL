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
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_rl.environments.nemo_gym import NemoGym, _external_staging_backend
from nemo_rl.experience.failures import GymTransportError

# Receipt assembly imports nemo_gym at call time (resolve_terminal etc.), so
# these tests must run in the Nemo_Gym shard, not the base-env Environments one.
pytestmark = pytest.mark.nemo_gym


def _capture_env() -> NemoGym:
    env_cls = NemoGym.__ray_metadata__.modified_class
    env = object.__new__(env_cls)
    env._forest_training = False
    return env


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _capture_stream_env(
    forest_training: bool,
    completed: Callable[[dict], Awaitable[tuple[dict, dict]]],
) -> tuple[NemoGym, list[dict]]:
    env = _capture_env()
    env._forest_training = forest_training
    env._token_capture_enabled = True
    env._tokenizer = object()
    env.cfg = {}
    env.head_server_config = object()
    env._require_spinup = lambda: None
    rows = [
        {
            "_rowidx": index,
            "agent_ref": {"name": "agent"},
            "responses_create_params": {"input": []},
        }
        for index in range(2)
    ]
    env.rch = SimpleNamespace(
        run_examples=lambda **kwargs: [completed(row) for row in kwargs["examples"]]
    )
    return env, rows


@pytest.mark.parametrize("forest_training", [False, True])
def test_capture_stream_yields_ready_sibling_before_slow_drain(
    forest_training: bool,
) -> None:
    async def run() -> None:
        drain_started = asyncio.Event()
        release_drain = asyncio.Event()

        async def completed(row: dict) -> tuple[dict, dict]:
            # The second Gym result arrives after the first starts draining.
            if row["_rowidx"] == 1:
                await drain_started.wait()
            return row, {"reward": 1.0}

        env, rows = _capture_stream_env(forest_training, completed)

        async def postprocess(row: dict, result: dict) -> dict:
            if row["_rowidx"] == 0:
                drain_started.set()
                await release_drain.wait()
            return {"owner": row["_rowidx"], **result}

        env._postprocess_receipt_mode = postprocess
        stream = env.run_rollouts(rows, "test")
        first = asyncio.create_task(anext(stream))
        try:
            await asyncio.wait_for(drain_started.wait(), timeout=1)
            ready = await asyncio.wait_for(first, timeout=1)
            assert ready[:3] == (1, {"name": "agent"}, {"owner": 1, "reward": 1.0})
            assert ready[3] is None
            assert not release_drain.is_set()
            release_drain.set()
            last = await asyncio.wait_for(anext(stream), timeout=1)
            assert last[:3] == (0, {"name": "agent"}, {"owner": 0, "reward": 1.0})
            assert 0 <= last[3]["test/postprocess_results_pct"] <= 100
            with pytest.raises(StopAsyncIteration):
                await anext(stream)
        finally:
            release_drain.set()
            first.cancel()
            await asyncio.gather(first, return_exceptions=True)
            await stream.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("forest_training", [False, True])
@pytest.mark.parametrize("stop", ["close", "cancel", "gym_error", "receipt_error"])
def test_capture_stream_stops_receipt_waits_and_closes_collection(
    forest_training: bool, stop: str
) -> None:
    async def run() -> None:
        drain_started = asyncio.Event()
        drain_cancelled = asyncio.Event()
        client_started = asyncio.Event()
        client_cancelled = asyncio.Event()

        async def completed(row: dict) -> tuple[dict, dict]:
            if row["_rowidx"] == 1:
                await drain_started.wait()
                client_started.set()
                if stop == "cancel":
                    try:
                        await asyncio.Event().wait()
                    finally:
                        client_cancelled.set()
                if stop == "gym_error":
                    error = RuntimeError("injected HTTP failure")
                    error.status = 503
                    raise error
            return row, {"reward": 1.0}

        env, rows = _capture_stream_env(forest_training, completed)
        collection = MagicMock()
        collection.__iter__.return_value = iter(completed(row) for row in rows)
        collection.aclose = AsyncMock()
        env.rch.run_examples = lambda **kwargs: collection

        async def postprocess(row: dict, result: dict) -> dict:
            if row["_rowidx"] == 0:
                drain_started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    drain_cancelled.set()
            if stop == "receipt_error":
                raise ValueError("injected receipt failure")
            return result

        env._postprocess_receipt_mode = postprocess
        stream = env.run_rollouts(rows, "test")
        first = asyncio.create_task(anext(stream))
        try:
            await asyncio.wait_for(client_started.wait(), timeout=1)
            if stop == "close":
                assert (await asyncio.wait_for(first, timeout=1))[0] == 1
                await stream.aclose()
            elif stop == "cancel":
                first.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await first
                assert client_cancelled.is_set()
            else:
                expected = GymTransportError if stop == "gym_error" else ValueError
                with pytest.raises(expected, match="HTTP 503|receipt failure"):
                    await asyncio.wait_for(first, timeout=1)
            assert drain_cancelled.is_set()
            collection.aclose.assert_awaited_once()
        finally:
            first.cancel()
            await asyncio.gather(first, return_exceptions=True)
            await stream.aclose()

    asyncio.run(run())


@pytest.mark.parametrize(
    ("generation_backend", "expected"),
    [
        ("vllm", "vllm_worker"),
        ("megatron", "megatron_worker"),
        ("dynamo", "dynamo_worker"),
    ],
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

    admission = CaptureAdmission(capture_key="r0", model_call_id="c1", mode="text")
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
        return_value={"capture_key": "r0", "records": records, "failures": []}
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
        return_value={"capture_key": "r0", "records": records, "failures": []}
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
    assert receipt["capture_key"] == "r0"
    assert receipt["terminal_model_call_id"] == "c2"
    assert receipt["terminal_selection"] == "declared"
    assert receipt["capture_poisoned"] is False
    assert receipt["failure_reason"] is None
    assert receipt["reward"] == 1.0
    assert [r["model_call_id"] for r in receipt["manifest"]] == ["c1", "c2"]


def test_receipt_assembly_poisons_on_failure_rows() -> None:
    env = _capture_env()
    manifest = {
        "capture_key": "r0",
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
        "capture_key": "r0",
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
        "capture_key": "r0",
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


def test_a_failed_call_keeps_pending_write_custody() -> None:
    """Failure masks the rollout without transferring unfinished writes to cleanup."""
    from nemo_rl.environments.nemo_gym import _mask_undrained_receipt
    from nemo_rl.experience.rollout_recovery import _receipt_staging_keys

    env = _capture_env()
    manifest = {
        "capture_key": "r0",
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
        "attempted_call_ids": ["c1", "c2", "c3"],
        "pending_call_ids": ["c3"],
    }
    receipt = env._assemble_receipt(
        "r0", manifest, terminal_response_id="resp-c2", reward=1.0
    )
    receipt = _mask_undrained_receipt(receipt)
    assert receipt["pending_call_ids"] == ["c3"]
    assert receipt["capture_poisoned"] is True
    staging_keys = _receipt_staging_keys(receipt)
    assert set(staging_keys) == {"r0/c1", "r0/c2"}


def test_a_truly_in_flight_pending_intent_keeps_the_receipt_unsealable() -> None:
    """A pending intent with no failure row means the request may still be
    executing; the receipt keeps it pending and the seal keeps refusing."""
    from nemo_rl.experience.rollout_recovery import _receipt_staging_keys

    env = _capture_env()
    manifest = {
        "capture_key": "r0",
        "records": [_manifest_record("c1")],
        "failures": [],
        "attempted_call_ids": ["c1", "c2"],
        "pending_call_ids": ["c2"],
    }
    receipt = env._assemble_receipt(
        "r0", manifest, terminal_response_id="resp-c1", reward=0.0
    )
    assert receipt["pending_call_ids"] == ["c2"]
    with pytest.raises(ValueError, match="Capture acknowledgement is unresolved"):
        _receipt_staging_keys(receipt)


def test_an_undrained_receipt_masks_instead_of_wedging() -> None:
    """When the drain budget lapses with the call still executing, the
    receipt must become sealable as a masked rollout, not abort the run."""
    from nemo_rl.environments.nemo_gym import _mask_undrained_receipt
    from nemo_rl.experience.rollout_recovery import _receipt_staging_keys

    env = _capture_env()
    manifest = {
        "capture_key": "r0",
        "records": [_manifest_record("c1")],
        "failures": [],
        "attempted_call_ids": ["c1", "c2"],
        "pending_call_ids": ["c2"],
    }
    receipt = _mask_undrained_receipt(
        env._assemble_receipt(
            "r0", manifest, terminal_response_id="resp-c1", reward=0.0
        )
    )
    assert receipt["pending_call_ids"] == ["c2"]
    assert receipt["capture_poisoned"] is True
    assert receipt["failure_reason"] == "capture_incomplete"
    assert _receipt_staging_keys(receipt) == ["r0/c1"]


@pytest.mark.parametrize("compaction", [False, True])
def test_the_terminal_manifest_read_drains_an_in_flight_call(
    monkeypatch: pytest.MonkeyPatch,
    compaction: bool,
) -> None:
    """A session ended at its budget can leave its last call executing in the
    engine; the terminal manifest read polls until the call resolves."""
    import nemo_rl.environments.nemo_gym as env_module

    env = _capture_env()
    env._forest_training = compaction
    pending = {
        "capture_key": "r0",
        "records": [],
        "failures": [],
        "attempted_call_ids": ["c1"],
        "pending_call_ids": ["c1"],
    }
    resolved = {
        "capture_key": "r0",
        "records": [_manifest_record("c1")],
        "failures": [],
        "attempted_call_ids": ["c1"],
        "pending_call_ids": [],
    }
    env._control = AsyncMock(side_effect=[pending, resolved])
    monkeypatch.setattr(env_module, "_CAPTURE_DRAIN_POLL_S", 0.0)
    manifest = asyncio.run(env._drained_manifest("r0"))
    assert manifest["pending_call_ids"] == []
    assert env._control.await_count == 2
    assert env._control.await_args_list[0].args == (
        "POST" if compaction else "GET",
        f"/training-token-capture/control/rollouts/r0/{'close-admission' if compaction else 'manifest'}",
    )
    assert env._control.await_args_list[1].args == (
        "GET",
        "/training-token-capture/control/rollouts/r0/manifest",
    )


def test_the_drain_budget_bounds_the_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    import nemo_rl.environments.nemo_gym as env_module

    env = _capture_env()
    pending = {
        "capture_key": "r0",
        "records": [],
        "failures": [],
        "attempted_call_ids": ["c1"],
        "pending_call_ids": ["c1"],
    }
    env._control = AsyncMock(return_value=pending)
    monkeypatch.setattr(env_module, "_CAPTURE_DRAIN_POLL_S", 0.0)
    monkeypatch.setattr(env_module, "_CAPTURE_DRAIN_TIMEOUT_S", 0.0)
    manifest = asyncio.run(env._drained_manifest("r0"))
    assert manifest["pending_call_ids"] == ["c1"]
    assert env._control.await_count == 1


@pytest.mark.parametrize("cancel_during", ["manifest_read", "poll_sleep"])
def test_capture_drain_propagates_cancellation(
    monkeypatch: pytest.MonkeyPatch, cancel_during: str
) -> None:
    import nemo_rl.environments.nemo_gym as env_module

    env = _capture_env()
    pending = {"pending_call_ids": ["c1"], "failures": []}
    env._control = AsyncMock(return_value=pending)
    sleep = AsyncMock(side_effect=asyncio.CancelledError)
    monkeypatch.setattr(env_module.asyncio, "sleep", sleep)
    if cancel_during == "manifest_read":
        env._control.side_effect = asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(env._drained_manifest("r0"))

    assert env._control.await_count == 1
    assert sleep.await_count == (cancel_during == "poll_sleep")
    assert pending["pending_call_ids"] == ["c1"]


def test_failure_ends_drain_without_acknowledging_pending_write() -> None:
    from nemo_rl.environments.nemo_gym import _mask_undrained_receipt
    from nemo_rl.experience.rollout_recovery import _receipt_staging_keys

    env = _capture_env()
    pending = {
        "capture_key": "r0",
        "records": [_manifest_record("c1")],
        "failures": [
            {
                "model_call_id": "c2",
                "reason": "request_finished_without_staged_coordinates",
            }
        ],
        "attempted_call_ids": ["c1", "c2"],
        "pending_call_ids": ["c2"],
    }
    env._control = AsyncMock(return_value=pending)
    manifest = asyncio.run(env._drained_manifest("r0"))
    assert env._control.await_count == 1
    receipt = _mask_undrained_receipt(
        env._assemble_receipt(
            "r0", manifest, terminal_response_id="resp-c1", reward=1.0
        )
    )
    assert receipt["capture_poisoned"]
    assert receipt["pending_call_ids"] == ["c2"]
    assert _receipt_staging_keys(receipt) == ["r0/c1"]


def test_receipt_assembly_poisons_when_the_terminal_row_is_missing() -> None:
    env = _capture_env()
    manifest = {
        "capture_key": "r0",
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
    manifest = {"capture_key": "r0", "records": records, "failures": []}
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
    manifest = {"capture_key": "r0", "records": records, "failures": []}
    receipt = env._assemble_receipt(
        "r0", manifest, terminal_response_id=None, reward=0.0
    )
    assert receipt["terminal_model_call_id"] is None
    assert receipt["capture_poisoned"] is True
    assert receipt["failure_reason"] == "ambiguous_terminal"


def test_receipt_assembly_heuristic_masks_an_empty_manifest() -> None:
    env = _capture_env()
    manifest = {"capture_key": "r0", "records": [], "failures": []}
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
    manifest = {"capture_key": "r0", "records": [bad], "failures": []}
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
    manifest = {"capture_key": "r0", "records": [good, bad], "failures": []}
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
    manifest = {"capture_key": "r0", "records": [good, bad], "failures": []}
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
    manifest = {"capture_key": "r0", "records": records, "failures": []}
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
    manifest = {"capture_key": "r0", "records": records, "failures": []}
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
    manifest = {"capture_key": "r0", "records": records, "failures": []}
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
    manifest = {"capture_key": "r0", "records": records, "failures": []}
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
    manifest = {"capture_key": "r0", "records": records, "failures": []}
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
    manifest = {"capture_key": "r0", "records": records, "failures": []}
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
        return_value={"capture_key": "r0", "records": records, "failures": []}
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
        data={},
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


@pytest.mark.parametrize(
    ("vllm_model", "refused"),
    [
        ({}, False),
        ({"chat_template_kwargs": {"enable_thinking": False}}, False),
        (
            {"extra_body": {"chat_template_kwargs": {"force_nonempty_content": True}}},
            False,
        ),
        # The frontend answers every request carrying bad_words with a 400.
        ({"extra_body": {"bad_words": ["x"]}}, True),
    ],
)
def test_gym_actor_config_dynamo_sends_the_native_default_template_kwargs(
    vllm_model: dict, refused: bool
) -> None:
    """The native server's default template kwargs go under every kwargs Gym has."""
    from responses_api_models.vllm_model.app import VLLMModelConfig

    from nemo_rl.algorithms.grpo import MasterConfig
    from nemo_rl.environments.nemo_gym import (
        build_nemo_gym_config,
        setup_nemo_gym_config,
        should_use_nemo_gym,
    )

    user_vllm_model = copy.deepcopy(vllm_model)

    config = MasterConfig.model_construct(
        env={
            "should_use_nemo_gym": True,
            "nemo_gym": {
                "policy_model": {"responses_api_models": {"vllm_model": vllm_model}}
            },
        },
        data={},
        policy={
            "generation": {
                "backend": "vllm",
                "dynamo": True,
                "vllm_cfg": {
                    "async_engine": True,
                    "http_server_serving_chat_kwargs": {
                        "default_chat_template_kwargs": {
                            "enable_thinking": True,
                            "truncate_history_thinking": False,
                        }
                    },
                },
            }
        },
    )

    setup_nemo_gym_config(config, tokenizer=None)
    assert should_use_nemo_gym(config) is True

    def build() -> dict:
        return build_nemo_gym_config(
            config.env,
            base_urls=["http://frontend"],
            model_name="test-model",
            enable_router_replay=False,
            use_fastokens=False,
        )["initial_global_config_dict"]

    if refused:
        with pytest.raises(ValueError, match="bad_words"):
            build()
        return
    gym_config = build()
    composed = gym_config["policy_model"]["responses_api_models"]["vllm_model"]
    # Read back through Gym's own config, which ignores keys it does not know.
    gym_model = VLLMModelConfig(
        host="localhost",
        port=8000,
        name="policy_model",
        entrypoint="app.py",
        base_url="http://frontend",
        api_key="dummy_key",
        model="test-model",
        return_token_id_information=True,
        uses_reasoning_parser=True,
        **composed,
    )
    assert gym_model.dynamo_frontend is True
    # The frontend refuses reasoning under both of Gym's keys.
    assert gym_model.reasoning_field == "reasoning_content"
    assert gym_model.default_chat_template_kwargs == {
        "enable_thinking": True,
        "truncate_history_thinking": False,
    }
    # Gym applies the defaults under its own kwargs, which stay as written; its
    # config merge would let a value written into them override a Gym YAML's.
    for key in ("chat_template_kwargs", "extra_body"):
        assert composed.get(key) == vllm_model.get(key)
    assert "dynamo_dialect" not in gym_config
    # The user's overlay is left as written.
    assert vllm_model == user_vllm_model


def test_public_postprocess_waits_for_commit_instead_of_masking(monkeypatch) -> None:
    import nemo_rl.environments.nemo_gym as mod
    from nemo_rl.experience.rollout_recovery import _receipt_staging_keys

    env = _capture_env()
    pending = {
        "capture_key": "r0",
        "records": [_manifest_record("c1")],
        "failures": [],
        "attempted_call_ids": ["c1", "c2"],
        "pending_call_ids": ["c2"],
    }
    done = {
        **pending,
        "records": [_manifest_record("c1"), _manifest_record("c2", parent="c1")],
        "pending_call_ids": [],
    }
    env._control = AsyncMock(side_effect=[pending, done])
    monkeypatch.setattr(mod, "_CAPTURE_DRAIN_POLL_S", 0, raising=False)
    result = asyncio.run(
        env._postprocess_receipt_mode(
            {"_ng_rollout_id": "r0"}, {"terminal_response_id": "resp-c2", "reward": 1.0}
        )
    )
    assert env._control.await_count == 2
    assert not result["receipt"]["capture_poisoned"]
    assert result["receipt"]["terminal_model_call_id"] == "c2"
    assert set(_receipt_staging_keys(result["receipt"])) == {"r0/c1", "r0/c2"}


def test_public_postprocess_timeout_masks_without_taking_pending_write(
    monkeypatch,
) -> None:
    import nemo_rl.environments.nemo_gym as mod
    from nemo_rl.experience.rollout_recovery import _receipt_staging_keys

    env = _capture_env()
    pending = {
        "capture_key": "r0",
        "records": [_manifest_record("c1")],
        "failures": [],
        "attempted_call_ids": ["c1", "c2"],
        "pending_call_ids": ["c2"],
    }
    env._control = AsyncMock(return_value=pending)
    monkeypatch.setattr(mod, "_CAPTURE_DRAIN_TIMEOUT_S", 0, raising=False)
    result = asyncio.run(
        env._postprocess_receipt_mode(
            {"_ng_rollout_id": "r0"}, {"terminal_response_id": "resp-c1", "reward": 1.0}
        )
    )
    receipt = result["receipt"]
    keys = _receipt_staging_keys(receipt)
    assert receipt["capture_poisoned"] and receipt["pending_call_ids"] == ["c2"]
    assert keys == ["r0/c1"]


@pytest.mark.parametrize("policy", ["returned_history", "all_captured"])
def test_forest_failed_pending_was_already_masked(policy: str) -> None:
    from nemo_rl.experience.rollout_recovery import _receipt_staging_keys

    env = _capture_env()
    env._forest_training, env.cfg = True, {}
    env._selection_policy = policy
    pending = {
        "capture_key": "r0",
        "records": [],
        "failures": [
            {
                "model_call_id": "c1",
                "reason": "request_finished_without_staged_coordinates",
            }
        ],
        "attempted_call_ids": ["c1"],
        "pending_call_ids": ["c1"],
    }
    env._control = AsyncMock(return_value=pending)
    result = asyncio.run(
        env._postprocess_receipt_mode(
            {"_ng_rollout_id": "r0"}, {"reward": 0.0, "response": {"output": []}}
        )
    )
    receipt = result["receipt"]
    assert receipt["capture_poisoned"] and receipt["pending_call_ids"] == ["c1"]
    assert _receipt_staging_keys(receipt) == []
