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

import threading
from types import SimpleNamespace
from typing import Any

import pytest

nemo_gym = pytest.importorskip("nemo_gym.token_id_capture.staging")

from nemo_gym.token_id_capture.adapters.megatron import (  # noqa: E402
    MegatronCaptureAdapter,
)
from nemo_gym.token_id_capture.staging.capture import (  # noqa: E402
    RolloutTokenCapture,
)
from nemo_gym.token_id_capture.staging.digest import (  # noqa: E402
    compute_chain_hash,
    hash_token_ids,
)
from nemo_gym.token_id_capture.staging.records import (  # noqa: E402
    StagedCallRecord,
    StageResult,
)

from nemo_rl.models.generation.megatron.megatron_generation import (  # noqa: E402
    MegatronGeneration,
)
from nemo_rl.models.generation.megatron.megatron_worker import (  # noqa: E402
    MegatronGenerationMixin,
)

pytestmark = pytest.mark.nemo_gym


class _MemorySink:
    def __init__(self, *, reject: bool = False) -> None:
        self.reject = reject
        self.records: list[StagedCallRecord] = []

    def stage(self, record: StagedCallRecord) -> StageResult:
        if self.reject:
            return StageResult(ok=False, error="rejected")
        self.records.append(record)
        return StageResult(ok=True, staging_key=record.staging_key)


class _DeferredAdapter:
    """Structural deferred adapter that is deliberately not the MInf concrete type."""

    def __init__(self) -> None:
        self._adapter = MegatronCaptureAdapter()

    def enter_prefix(
        self, request_payload: dict[str, Any], prefix_ids: list[int]
    ) -> dict[str, Any]:
        return self._adapter.enter_prefix(request_payload, prefix_ids)

    def extract_prompt_ids(self, response_payload: dict[str, Any]) -> list[int]:
        return self._adapter.extract_prompt_ids(response_payload)

    def extract_generation(
        self, response_payload: dict[str, Any]
    ) -> tuple[list[int], list[float]]:
        return self._adapter.extract_generation(response_payload)

    def extract_extras(self, response_payload: dict[str, Any]) -> dict[str, Any] | None:
        return self._adapter.extract_extras(response_payload)

    def extract_weight_version(self, response_payload: dict[str, Any]) -> int:
        return self._adapter.extract_weight_version(response_payload)


class _WorkerGroup:
    def __init__(self, ledger: dict[str, dict]) -> None:
        self.ledger = ledger
        self.calls: list[tuple[str, dict]] = []

    def run_all_workers_single_data(self, method_name: str, **kwargs):
        self.calls.append((method_name, kwargs))
        if method_name == "fetch_token_capture_records":
            return [
                {
                    uid: self.ledger[uid]
                    for uid in kwargs["request_uids"]
                    if uid in self.ledger
                },
                {},
            ]
        if method_name == "discard_token_capture_records":
            for uid in kwargs["request_uids"]:
                self.ledger.pop(uid, None)
            return [len(kwargs["request_uids"]), 0]
        return [True, True]


def _generation(
    monkeypatch: pytest.MonkeyPatch,
    sink: _MemorySink,
    ledger: dict[str, dict],
    adapter: Any | None = None,
) -> MegatronGeneration:
    generation = object.__new__(MegatronGeneration)
    generation._policy = SimpleNamespace(worker_group=_WorkerGroup(ledger))
    generation._token_capture = RolloutTokenCapture(
        sink=sink,
        weight_version_fn=lambda: 0,
        adapter=adapter or MegatronCaptureAdapter(),
    )
    generation._token_capture_flush_lock = threading.Lock()
    monkeypatch.setattr(
        "nemo_rl.models.generation.megatron.megatron_generation.ray.get",
        lambda value: value,
    )
    return generation


def _receipt() -> dict:
    token_ids = [10, 11, 12]
    return {
        "rollout_id": "r0",
        "reward": 1.0,
        "terminal_model_call_id": "c1",
        "manifest": [],
        "pending_manifest": [
            {
                "model_call_id": "c1",
                "parent_call_id": None,
                "prev_len": 0,
                "delta_len": 3,
                "cum_len": 3,
                "mode": "text",
                "ledger_request_uid": "minf-1",
                "chain_hash": compute_chain_hash(None, token_ids),
                "cumulative_hash": hash_token_ids(token_ids),
                "response_id": "minf-1",
                "logical_request_id": "lr-1",
            }
        ],
        "capture_poisoned": False,
        "failure_reason": None,
        "terminal_selection": "declared",
    }


def _ledger_record() -> dict:
    return {
        "policy_epoch": [[0, 7]],
        "kv_cache_epoch": None,
        "num_evictions": 0,
        "prompt_token_ids": [10, 11],
        "generated_token_ids": [12],
        "generated_log_probs": [-0.25],
        "prompt_log_probs": None,
        "routing_indices": None,
    }


def test_rollout_flush_stages_whole_ledger_batch_then_discards(monkeypatch):
    sink = _MemorySink()
    ledger = {"minf-1": _ledger_record()}
    generation = _generation(monkeypatch, sink, ledger)

    finalized = generation.flush_token_capture(_receipt())

    assert "pending_manifest" not in finalized
    assert finalized["manifest"][0]["weight_version"] == 7
    assert finalized["manifest"][0]["staging_key"] == "r0/c1"
    assert finalized["manifest"][0]["response_id"] == "minf-1"
    assert sink.records[0].token_ids_delta == [10, 11, 12]
    assert ledger == {}
    assert [call[0] for call in generation._policy.worker_group.calls] == [
        "fetch_token_capture_records",
        "discard_token_capture_records",
    ]


def test_rollout_flush_accepts_a_structural_deferred_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = _MemorySink()
    ledger = {"minf-1": _ledger_record()}
    generation = _generation(monkeypatch, sink, ledger, adapter=_DeferredAdapter())

    finalized = generation.flush_token_capture(_receipt())

    assert finalized["manifest"][0]["weight_version"] == 7
    assert sink.records[0].token_ids_delta == [10, 11, 12]


def test_rollout_flush_preserves_multiturn_chain_custody(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_tokens = [10, 11, 12]
    child_delta = [13, 14]
    root_chain_hash = compute_chain_hash(None, root_tokens)
    child_chain_hash = compute_chain_hash(root_chain_hash, child_delta)
    receipt = _receipt()
    receipt["terminal_model_call_id"] = "c2"
    receipt["pending_manifest"].append(
        {
            "model_call_id": "c2",
            "parent_call_id": "c1",
            "prev_len": len(root_tokens),
            "delta_len": len(child_delta),
            "cum_len": len(root_tokens) + len(child_delta),
            "mode": "token_in",
            "ledger_request_uid": "minf-2",
            "chain_hash": child_chain_hash,
            "cumulative_hash": hash_token_ids(root_tokens + child_delta),
            "response_id": "minf-2",
            "logical_request_id": "lr-2",
        }
    )
    child_ledger_record = _ledger_record()
    child_ledger_record["prompt_token_ids"] = root_tokens + [13]
    child_ledger_record["generated_token_ids"] = [14]
    child_ledger_record["generated_log_probs"] = [-0.5]
    sink = _MemorySink()
    generation = _generation(
        monkeypatch,
        sink,
        {"minf-1": _ledger_record(), "minf-2": child_ledger_record},
    )

    finalized = generation.flush_token_capture(receipt)

    assert [record["response_id"] for record in finalized["manifest"]] == [
        "minf-1",
        "minf-2",
    ]
    assert finalized["manifest"][1]["chain_hash"] == child_chain_hash
    assert sink.records[1].token_ids_delta == child_delta


def test_rollout_flush_keeps_ledger_when_staging_fails(monkeypatch):
    sink = _MemorySink(reject=True)
    ledger = {"minf-1": _ledger_record()}
    generation = _generation(monkeypatch, sink, ledger)

    with pytest.raises(RuntimeError, match="failed to stage"):
        generation.flush_token_capture(_receipt())

    assert "minf-1" in ledger
    assert [call[0] for call in generation._policy.worker_group.calls] == [
        "fetch_token_capture_records"
    ]


def test_rollout_flush_validates_http_lengths_before_staging(monkeypatch):
    sink = _MemorySink()
    ledger = {"minf-1": _ledger_record()}
    generation = _generation(monkeypatch, sink, ledger)
    receipt = _receipt()
    receipt["pending_manifest"][0]["delta_len"] = 4
    receipt["pending_manifest"][0]["cum_len"] = 4

    with pytest.raises(RuntimeError, match="HTTP lineage lengths"):
        generation.flush_token_capture(receipt)

    assert sink.records == []
    assert "minf-1" in ledger
    assert [call[0] for call in generation._policy.worker_group.calls] == [
        "fetch_token_capture_records"
    ]


def test_worker_enables_and_reads_minf_ledger() -> None:
    record = SimpleNamespace(**_ledger_record())

    class _Engine:
        local_metadata_ledger_enabled = False
        local_metadata_ledger_offload_enabled = False

        def fetch_from_metadata_ledger(self, uids, pop=True):
            return {"minf-1": record} if "minf-1" in uids else {}

    worker = object.__new__(MegatronGenerationMixin)
    worker.dynamic_inference_engine = _Engine()
    worker._token_capture_enabled = False
    assert worker.setup_token_capture()
    assert worker.dynamic_inference_engine.local_metadata_ledger_enabled
    assert worker.dynamic_inference_engine.local_metadata_ledger_offload_enabled
    fetched = worker.fetch_token_capture_records(["minf-1"])
    assert fetched["minf-1"]["generated_token_ids"] == [12]
