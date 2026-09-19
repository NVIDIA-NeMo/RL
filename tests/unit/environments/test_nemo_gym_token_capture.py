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
import hashlib
from unittest.mock import AsyncMock

import pytest

from nemo_rl.environments.nemo_gym import NemoGym

# Receipt assembly imports nemo_gym at call time (resolve_terminal etc.), so
# these tests must run in the Nemo_Gym shard, not the base-env Environments one.
pytestmark = pytest.mark.nemo_gym


def _capture_env() -> NemoGym:
    env_cls = NemoGym.__ray_metadata__.modified_class
    return object.__new__(env_cls)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


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


# ── control-plane call: private pool, bounded body read, retries ──────────────


class _FakeResponse:
    def __init__(self, status: int, payload: dict, delay_s: float = 0.0):
        self.status = status
        self._payload = payload
        self._delay_s = delay_s

    async def __aenter__(self):
        if self._delay_s:
            await asyncio.sleep(self._delay_s)
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self, content_type=None):
        return self._payload

    async def text(self):
        return str(self._payload)


class _FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url))
        return self._responses.pop(0)


def _control_env(session, *, timeout_s=0.2, retries=3) -> NemoGym:
    env = _capture_env()
    env._control_headers = {"Authorization": "Bearer t"}
    env._control_timeout_s = timeout_s
    env._control_retries = retries
    env._control_pool_size = 4
    env._control_http = session
    env._control_base = "http://policy:1234"
    return env


def test_control_call_uses_private_session_and_returns_json() -> None:
    session = _FakeSession([_FakeResponse(200, {"rollout_id": "r0", "records": []})])
    env = _control_env(session)
    manifest = asyncio.run(env._control("GET", "/x/rollouts/r0/manifest"))
    assert manifest == {"rollout_id": "r0", "records": []}
    assert session.calls == [("GET", "http://policy:1234/x/rollouts/r0/manifest")]


def test_control_call_retries_after_a_timeout_then_succeeds() -> None:
    session = _FakeSession(
        [
            _FakeResponse(200, {"late": True}, delay_s=5.0),  # exceeds 0.2 s deadline
            _FakeResponse(200, {"ok": True}),
        ]
    )
    env = _control_env(session, timeout_s=0.2, retries=2)
    manifest = asyncio.run(env._control("GET", "/m"))
    assert manifest == {"ok": True}
    assert len(session.calls) == 2


def test_control_call_does_not_retry_http_errors() -> None:
    session = _FakeSession([_FakeResponse(503, {"detail": "busy"}) for _ in range(3)])
    env = _control_env(session, retries=3)
    with pytest.raises(RuntimeError, match="HTTP 503"):
        asyncio.run(env._control("GET", "/m"))
    assert len(session.calls) == 1


def test_control_call_gives_up_after_the_last_timeout() -> None:
    session = _FakeSession(
        [_FakeResponse(200, {"late": True}, delay_s=5.0) for _ in range(2)]
    )
    env = _control_env(session, timeout_s=0.2, retries=2)
    with pytest.raises(RuntimeError, match="exceeded 0.2s"):
        asyncio.run(env._control("GET", "/m"))
    assert len(session.calls) == 2


class _FakeServerClient:
    """Stands in for Gym's ServerClient (shared-session path, pool_size=0)."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    async def request(self, *, server_name, url_path, method, headers, **kwargs):
        self.calls.append((method, url_path))
        response = self._responses.pop(0)
        async with response as r:
            return r


def test_control_call_defaults_to_the_shared_gym_session_with_retries() -> None:
    client = _FakeServerClient(
        [
            _FakeResponse(200, {"late": True}, delay_s=5.0),  # exceeds deadline
            _FakeResponse(200, {"ok": True}),
        ]
    )
    env = _control_env(None, timeout_s=0.2, retries=2)
    env._control_pool_size = 0
    env._server_client = client
    manifest = asyncio.run(env._control("GET", "/m"))
    assert manifest == {"ok": True}
    assert client.calls == [("GET", "/m"), ("GET", "/m")]
