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

"""S2 worker hosting: install_capture wiring, fan-outs, version stamping.

Marked nemo_gym (run with ``--nemo-gym-only``): the hosting seam imports
Gym's capture core. No engine or GPU is needed — the worker methods are
driven unbound against light fakes, and the VllmGeneration fan-outs against
a mock worker group.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

nemo_gym = pytest.importorskip("nemo_gym.token_id_capture.staging")

from nemo_gym._checkpoint.generation_cut import (  # noqa: E402
    GENERATION_CUT_ROUTE,
    GenerationCutInventory,
    GenerationCutPrefix,
    GenerationCutPrefixAck,
)
from nemo_gym.token_id_capture.staging.capture import (  # noqa: E402
    CaptureError,
    RolloutTokenCapture,
)
from nemo_gym.token_id_capture.staging.records import (  # noqa: E402
    CaptureAdmission,
    StagedCallBaseSnapshot,
    StagedCallRecord,
    StageResult,
)

from nemo_rl.data.captured_media import (  # noqa: E402
    MEDIA_TENSOR_DIGESTS_FIELD,
    CapturedMedia,
    CapturedMediaItem,
    MediaCaptureRejected,
    MediaColumnSpec,
    media_tensor_digests,
)
from nemo_rl.data_plane.background_prefix_cleanup import (  # noqa: E402
    BackgroundPrefixCleanup,
    GenerationPrefixCleanupConfig,
)
from nemo_rl.data_plane.schema import ROUTE_ENCODING_ENVELOPE  # noqa: E402
from nemo_rl.data_plane.tq_token_sink import (  # noqa: E402
    ChainPrefixCache,
    FetchedStagedCall,
    StagedPrefix,
    TQTokenSink,
)
from nemo_rl.experience.route_assembly import RouteFragment  # noqa: E402
from nemo_rl.models.generation.generation_cut_capture import (  # noqa: E402
    GenerationCutCaptureMixin,
    _remaining_generation_limits_after_prefix,
    _RequestCaptureState,
    _ResolvedGenerationCut,
    _TokenCaptureSnapshotGate,
)
from nemo_rl.models.generation.prefix_read_batcher import (  # noqa: E402
    PrefixReadBatcher,
)
from nemo_rl.models.generation.vllm.utils import (  # noqa: E402
    attach_routed_experts_to_chat_response_choices,
)
from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration  # noqa: E402
from nemo_rl.models.generation.vllm.vllm_worker_async import (  # noqa: E402
    _GENERATION_CUT_ROUTE,
    VllmAsyncGenerationWorkerImpl,
    _build_restored_prefix_terminal_output,
    _classify_restored_prefix_terminal,
    _count_restored_prefix_as_completion,
    _generation_cut_blockers,
    _restored_prefix_terminal_routes,
    _RestoredPrefixTerminal,
)
from nemo_rl.utils.routed_experts_codec import (  # noqa: E402
    decode_routed_experts,
    encode_routed_experts,
)

pytestmark = pytest.mark.nemo_gym

# ``setup_token_capture`` takes these from ``rollout_recovery`` with no defaults.
_PREFIX_BATCH_LIMITS = {
    "generation_prefix_batch_size": 256,
    "generation_prefix_batch_max_tokens": 4_194_304,
}


class _MemorySink:
    def __init__(self) -> None:
        self.records: list[StagedCallRecord] = []
        self.attachments: list[dict | None] = []
        self.generation_prefix_records: list[tuple[str, StagedCallRecord]] = []
        self.generation_prefix_attachments: list[dict | None] = []
        self.generation_prefix_keys: list[str] = []
        self.cleared_generation_prefix_keys: list[str] = []

    def stage(
        self, record: StagedCallRecord, *, attachments: dict | None = None
    ) -> StageResult:
        self.records.append(record)
        self.attachments.append(attachments)
        return StageResult(ok=True, staging_key=record.staging_key)

    def _stage_prefix_row(
        self,
        record: StagedCallRecord,
        *,
        checkpoint_id: str,
        chunk_sequence: int,
        attachments: dict | None = None,
    ) -> str:
        self.generation_prefix_records.append((checkpoint_id, record))
        self.generation_prefix_attachments.append(attachments)
        key = (
            f"__generation_cut__/{checkpoint_id}/{record.rollout_id}/"
            f"{record.model_call_id}/{chunk_sequence}"
        )
        self.generation_prefix_keys.append(key)
        return key

    def stage_generation_prefix_batch(
        self,
        records: list[StagedCallRecord],
        *,
        checkpoint_id: str,
        chunk_sequences: list[int],
        attachments: list[dict | None] | None = None,
    ) -> list[str]:
        if attachments is None:
            attachments = [None] * len(records)
        written: list[str] = []
        try:
            for record, sequence, row_attachments in zip(
                records, chunk_sequences, attachments, strict=True
            ):
                written.append(
                    self._stage_prefix_row(
                        record,
                        checkpoint_id=checkpoint_id,
                        chunk_sequence=sequence,
                        attachments=row_attachments,
                    )
                )
        except Exception:
            # Like TQTokenSink: a failed batch discards what it wrote, then raises.
            self.clear(written)
            raise
        return written

    def clear(self, staging_keys: list[str]) -> None:
        self.cleared_generation_prefix_keys.extend(staging_keys)


class _ByteTokenizer:
    """Byte-level stand-in: each token ID is one UTF-8 byte of the output."""

    def decode(self, token_ids: list[int], **kwargs) -> str:
        return bytes(token_ids).decode("utf-8", errors="replace")


def _fake_worker(*, is_model_owner: bool = True) -> SimpleNamespace:
    """The attribute surface setup_token_capture touches, minus the engine."""
    worker = SimpleNamespace(
        # The vLLM worker's router-replay surface, off unless a test sets it.
        _return_routed_experts_enabled=lambda: False,
        routed_experts_dtype=torch.int16,
        is_model_owner=is_model_owner,
        token_capture=None,
        _rollout_weight_version=0,
        _chain_prefix=ChainPrefixCache(),
        _capture_calls={},
        _capture_calls_by_model_call_id={},
        _completed_capture_calls={},
        _generation_cut_receipts={},
        _generation_prefix_batch_max_rows=256,
        _generation_prefix_batch_max_tokens=4_194_304,
        _capture_registry_lock=threading.Lock(),
        _capture_sink=None,
        _prefix_cleanup=None,
        _generation_prefix_cuts_enabled=False,
        _generation_cut_control_token=None,
        _generation_cut_control_timeout_s=None,
        _generation_cut_tokenizer=_ByteTokenizer(),
        _restored_generation_prefix_calls=0,
        _restored_generation_prefix_tokens=0,
        _token_capture_snapshot_gate=_TokenCaptureSnapshotGate(),
        _generation_cut_cleanup_executor=ThreadPoolExecutor(max_workers=1),
        _staging_source=None,
        _generation_prefix_reader=None,
        _capture_media=False,
        _http_engine_client=None,
    )
    worker.install_token_capture = lambda capture: setattr(
        worker, "token_capture", capture
    )
    return worker


def _real_worker() -> VllmAsyncGenerationWorkerImpl:
    """A real worker over the fake's attributes, so methods resolve on the class."""
    worker = object.__new__(VllmAsyncGenerationWorkerImpl)
    fake = vars(_fake_worker())
    fake.pop("install_token_capture")
    vars(worker).update(fake)
    return worker


def test_generation_cut_route_matches_gym():
    """The worker serves cuts where Gym sends them; a drifted route 404s every cut."""
    assert _GENERATION_CUT_ROUTE == GENERATION_CUT_ROUTE


def test_setup_token_capture_installs_capture_with_vllm_adapter(monkeypatch):
    sink = _MemorySink()
    segment_sizes = []
    monkeypatch.setattr(
        "nemo_rl.data_plane.build_data_plane_client",
        lambda dp_cfg, bootstrap, segment_size=None: (
            segment_sizes.append(segment_size) or MagicMock(name="dp_client")
        ),
    )
    monkeypatch.setattr(
        "nemo_rl.data_plane.tq_token_sink.TQTokenSink",
        lambda dp_client, *, staging_partition, capture_media, media_pixel_dtype=None: (
            sink
        ),
    )
    worker = _real_worker()

    installed = asyncio.run(
        VllmAsyncGenerationWorkerImpl.setup_token_capture(
            worker,
            dp_cfg={"backend": "simple"},
            staging_partition="rollout_staging",
            generation_prefix_batch_size=17,
            generation_prefix_batch_max_tokens=4096,
            generation_prefix_cleanup=GenerationPrefixCleanupConfig(),
        )
    )

    assert installed is None  # text-only capture pins no media column
    # The capture client owns no Mooncake memory, so no save waits on vLLM.
    assert segment_sizes == [0]
    assert isinstance(worker.token_capture, RolloutTokenCapture)
    assert isinstance(worker._generation_prefix_reader, PrefixReadBatcher)
    # The batch size bounds cut-time puts and restore-time gets; the token
    # limit bounds only cut-time puts.
    assert worker._generation_prefix_batch_max_rows == 17
    assert worker._generation_prefix_batch_max_tokens == 4096
    assert worker._generation_prefix_reader._max_rows == 17
    assert worker.token_capture.adapter is not None
    # The adapter is the vLLM one (prefix ids enter via the worker's field).
    payload = worker.token_capture.adapter.enter_prefix({}, [1, 2])
    assert payload["required_prefix_token_ids"] == [1, 2]


def test_setup_token_capture_skips_non_model_owners(monkeypatch):
    worker = _fake_worker(is_model_owner=False)
    installed = asyncio.run(
        VllmAsyncGenerationWorkerImpl.setup_token_capture(
            worker,
            dp_cfg={},
            staging_partition="rollout_staging",
            generation_prefix_cleanup=GenerationPrefixCleanupConfig(),
            **_PREFIX_BATCH_LIMITS,
        )
    )
    assert installed is None
    assert worker.token_capture is None


def test_weight_version_is_stamped_from_worker_state(monkeypatch):
    """The install closure reads _rollout_weight_version live: a
    set_rollout_weight_version between calls changes the stamp."""
    sink = _MemorySink()
    monkeypatch.setattr(
        "nemo_rl.data_plane.build_data_plane_client",
        lambda dp_cfg, bootstrap, segment_size=None: MagicMock(),
    )
    monkeypatch.setattr(
        "nemo_rl.data_plane.tq_token_sink.TQTokenSink",
        lambda dp_client, *, staging_partition, capture_media, media_pixel_dtype=None: (
            sink
        ),
    )
    worker = _real_worker()
    asyncio.run(
        VllmAsyncGenerationWorkerImpl.setup_token_capture(
            worker,
            dp_cfg={},
            staging_partition="rollout_staging",
            generation_prefix_cleanup=GenerationPrefixCleanupConfig(),
            **_PREFIX_BATCH_LIMITS,
        )
    )

    asyncio.run(VllmAsyncGenerationWorkerImpl.set_rollout_weight_version(worker, 4))
    first = worker.token_capture.begin_call(
        CaptureAdmission(rollout_id="r", model_call_id="c1", mode="text")
    )
    asyncio.run(VllmAsyncGenerationWorkerImpl.set_rollout_weight_version(worker, 5))
    second = worker.token_capture.begin_call(
        CaptureAdmission(rollout_id="r", model_call_id="c2", mode="text")
    )

    assert (first.weight_version, second.weight_version) == (4, 5)

    coords = worker.token_capture.complete_call(
        first, prompt_token_ids=[1], generated_token_ids=[2], generated_logprobs=[-0.1]
    )
    assert coords.weight_version == 4
    assert sink.records[0].weight_version == 4


def _generation_with_mock_group(*, async_engine: bool = True) -> VllmGeneration:
    gen = object.__new__(VllmGeneration)
    gen.cfg = {"vllm_cfg": {"async_engine": async_engine}}
    gen.worker_group = MagicMock()
    gen.worker_group.run_all_workers_single_data.return_value = []
    return gen


def test_generation_setup_token_capture_fans_out(monkeypatch):
    gen = _generation_with_mock_group()
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_generation.ray.get",
        lambda futures: futures,
    )
    gen.setup_token_capture(
        {"backend": "simple"},
        "rollout_staging",
        generation_cut_control_timeout_s=60.0,
        generation_prefix_batch_size=128,
        generation_prefix_batch_max_tokens=2048,
        generation_prefix_cleanup=GenerationPrefixCleanupConfig(),
    )
    gen.worker_group.run_all_workers_single_data.assert_called_once_with(
        "setup_token_capture",
        dp_cfg={"backend": "simple"},
        staging_partition="rollout_staging",
        capture_media=False,
        generation_prefix_cuts_enabled=False,
        generation_cut_control_token=None,
        generation_cut_control_timeout_s=60.0,
        generation_prefix_batch_size=128,
        generation_prefix_batch_max_tokens=2048,
        generation_prefix_cleanup=GenerationPrefixCleanupConfig(),
        run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
    )


def test_generation_setup_token_capture_requires_async_engine():
    gen = _generation_with_mock_group(async_engine=False)
    with pytest.raises(AssertionError, match="async vLLM engine"):
        gen.setup_token_capture(
            {},
            "rollout_staging",
            generation_prefix_cleanup=GenerationPrefixCleanupConfig(),
            **_PREFIX_BATCH_LIMITS,
        )


@pytest.mark.parametrize(
    ("operation", "worker_method"),
    [
        (
            "begin_token_capture_snapshot_fence",
            "begin_token_capture_snapshot_fence_async",
        ),
        (
            "end_token_capture_snapshot_fence",
            "end_token_capture_snapshot_fence_async",
        ),
    ],
)
def test_token_capture_snapshot_fence_control_fans_out(
    monkeypatch, operation: str, worker_method: str
):
    gen = _generation_with_mock_group()
    gen.worker_group.workers = [object()]
    gen.worker_group.run_all_workers_single_data.return_value = [True]
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_generation.ray.get",
        lambda futures, timeout=None: futures,
    )

    assert getattr(gen, operation)(timeout_s=17.0)
    gen.worker_group.run_all_workers_single_data.assert_called_once_with(
        worker_method,
        run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
    )


def test_generation_prefix_restore_counts_sum_over_workers(monkeypatch):
    gen = _generation_with_mock_group()
    gen.worker_group.workers = [object(), object()]
    gen.worker_group.run_all_workers_single_data.return_value = [
        {"restored_calls": 2, "restored_prefix_tokens": 50},
        {"restored_calls": 1, "restored_prefix_tokens": 7},
    ]
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_generation.ray.get",
        lambda futures, timeout=None: futures,
    )

    assert gen.get_generation_prefix_restore_counts() == {
        "restored_calls": 3,
        "restored_prefix_tokens": 57,
    }
    gen.worker_group.run_all_workers_single_data.assert_called_once_with(
        "get_generation_prefix_restore_counts",
        run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
    )


def test_prefix_capture_setup_requires_control_token(monkeypatch):
    monkeypatch.setattr(
        "nemo_rl.data_plane.build_data_plane_client",
        lambda dp_cfg, bootstrap, segment_size=None: MagicMock(),
    )
    worker = _fake_worker()
    with pytest.raises(ValueError, match="control bearer token"):
        asyncio.run(
            VllmAsyncGenerationWorkerImpl.setup_token_capture(
                worker,
                {},
                "rollout_staging",
                generation_prefix_cuts_enabled=True,
                generation_prefix_cleanup=GenerationPrefixCleanupConfig(),
                **_PREFIX_BATCH_LIMITS,
            )
        )


@pytest.mark.parametrize("http_client", ["missing", "unsupported_vllm"])
def test_prefix_capture_with_router_replay_fails_at_setup_without_route_state(
    monkeypatch, http_client: str
):
    """A cut reads an active request's routes through the HTTP client."""
    monkeypatch.setattr(
        "nemo_rl.data_plane.build_data_plane_client",
        lambda dp_cfg, bootstrap, segment_size=None: MagicMock(name="dp_client"),
    )
    monkeypatch.setattr(
        "nemo_rl.data_plane.tq_token_sink.TQTokenSink",
        lambda *args, **kwargs: _MemorySink(),
    )
    worker = _fake_worker()
    worker.llm = MagicMock(name="llm")
    worker._return_routed_experts_enabled = lambda: True
    if http_client == "missing":
        match = "requires the vLLM HTTP client"
    else:
        match = "verified on vLLM"
        worker._http_engine_client = MagicMock(name="http_engine_client")
        worker._http_engine_client.check_route_snapshot_support.side_effect = (
            RuntimeError("router-replay prefix recovery ... verified on vLLM 0.29.0")
        )

    with pytest.raises(RuntimeError, match=match):
        asyncio.run(
            VllmAsyncGenerationWorkerImpl.setup_token_capture(
                worker,
                {},
                "rollout_staging",
                generation_prefix_cuts_enabled=True,
                generation_cut_control_token="secret",
                generation_prefix_cleanup=GenerationPrefixCleanupConfig(),
                **_PREFIX_BATCH_LIMITS,
            )
        )


@pytest.mark.parametrize(
    ("max_tokens", "min_tokens", "prefix_token_count", "expected"),
    [
        (10, 6, 3, (7, 3)),
        (None, 6, 3, (None, 3)),
        (10, None, 3, (7, None)),
        (10, 2, 3, (7, 0)),
    ],
)
def test_restored_prefix_reduces_remaining_output_limits(
    max_tokens, min_tokens, prefix_token_count, expected
):
    assert (
        _remaining_generation_limits_after_prefix(
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            prefix_token_count=prefix_token_count,
        )
        == expected
    )


def _usage(prompt_tokens: int, completion_tokens: int, cached_tokens: int | None):
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
    )


def test_resumed_call_reports_its_restored_prefix_as_completion():
    # Prompt [1, 2, 3], restored generated prefix [4, 5], regenerated tail of 3.
    usage = _usage(prompt_tokens=5, completion_tokens=3, cached_tokens=5)

    prompt_token_ids = _count_restored_prefix_as_completion(
        usage, [1, 2, 3, 4, 5], restored_token_count=2, terminal=None
    )

    assert (usage.prompt_tokens, usage.completion_tokens) == (3, 5)
    assert usage.total_tokens == 8
    assert usage.prompt_tokens_details.cached_tokens == 3
    assert prompt_token_ids == [1, 2, 3]


def test_terminal_restored_prefix_is_the_whole_completion():
    terminal = _RestoredPrefixTerminal(
        prompt_token_ids=(1, 2, 3, 4, 5),
        prefix_token_count=2,
        reason="eos",
        finish_reason="stop",
    )
    # vLLM served a one-token request on the prefix-extended prompt.
    usage = _usage(prompt_tokens=4, completion_tokens=1, cached_tokens=None)

    prompt_token_ids = _count_restored_prefix_as_completion(
        usage, terminal.prompt_token_ids, restored_token_count=2, terminal=terminal
    )

    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (
        3,
        2,
        5,
    )
    assert prompt_token_ids == [1, 2, 3]


def test_restored_prefix_at_output_limit_is_terminal():
    terminal = _classify_restored_prefix_terminal(
        prompt_token_ids=[1, 2, 3, 4],
        prefix_token_count=2,
        requested_output_tokens=2,
        model_max_tokens=8,
    )
    assert terminal is not None
    assert terminal.reason == "output_limit"
    assert terminal.original_prompt_token_count == 2


def _stopped_terminal(
    finish_reason: str = "stop", stop_reason: str | int | None = None
) -> _RestoredPrefixTerminal:
    return _RestoredPrefixTerminal(
        prompt_token_ids=(1, 2),
        prefix_token_count=1,
        reason=f"observed_{finish_reason}",
        finish_reason=finish_reason,
        stop_reason=stop_reason,
    )


def test_restored_call_replays_its_prefix_with_the_original_settings(monkeypatch):
    """The prefix is replayed against the original prompt with the call's settings.

    The replay itself is vLLM's detokenizer; test_vllm_restored_prefix_replay.py
    checks it reproduces the uninterrupted response.
    """
    replays = []

    def replay(tokenizer, **kwargs):
        replays.append((tokenizer, kwargs))
        return "answer: 4"

    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_worker_async._detokenize_restored_prefix",
        replay,
    )
    tokenizer = object()
    terminal = _stopped_terminal(stop_reason="</answer>")
    state = SimpleNamespace(
        resumed_generation_token_ids=[12, 13],
        # The engine prompt ends with the spliced prefix.
        prompt_token_ids=[10, 11, 12, 13],
        lock=threading.Lock(),
    )
    worker = SimpleNamespace(_get_request_capture=lambda request: state)
    request = SimpleNamespace(
        _restored_prefix_terminal=terminal,
        # Admission reduces min_tokens by the prefix length; the replay needs
        # the call's original value, which gated its stop strings.
        _original_min_tokens=4,
        min_tokens=2,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
    )
    # The engine is skipped for a call that already stopped: its tail is empty.
    output = SimpleNamespace(token_ids=[], text="")

    VllmAsyncGenerationWorkerImpl._restore_response_prefix(
        worker,
        request,
        SimpleNamespace(outputs=[output]),
        tokenizer=tokenizer,
    )

    assert output.text == "answer: 4"
    assert replays == [
        (
            tokenizer,
            {
                "original_prompt_token_ids": [10, 11],
                "prefix_token_ids": [12, 13],
                "terminal": terminal,
                "stop": ["</answer>"],
                "min_tokens": 4,
                "include_stop_str_in_output": True,
                "skip_special_tokens": False,
                "spaces_between_special_tokens": False,
            },
        )
    ]


def test_cumulative_outputs_are_observed_once_per_token():
    """vLLM's cumulative outputs share its live lists; each token is taken once."""
    worker = _worker_with_capture(_MemorySink())
    request = _FakeRequest(
        ng_capture={"rollout_id": "r0", "model_call_id": "c1", "mode": "text"},
        stream=False,
    )
    VllmAsyncGenerationWorkerImpl._begin_request_capture(worker, request, [10, 11])
    token_ids = [12]
    logprobs = [{12: SimpleNamespace(logprob=-0.1)}]
    first = SimpleNamespace(
        outputs=[SimpleNamespace(token_ids=token_ids, logprobs=logprobs)]
    )

    VllmAsyncGenerationWorkerImpl._observe_request_capture(worker, request, first)
    # vLLM appends to the same lists; a late read of the old output sees them.
    token_ids += [13, 14]
    logprobs += [
        {13: SimpleNamespace(logprob=-0.2)},
        {14: SimpleNamespace(logprob=-0.3)},
    ]
    VllmAsyncGenerationWorkerImpl._observe_request_capture(worker, request, first)
    final = SimpleNamespace(
        outputs=[
            SimpleNamespace(
                token_ids=token_ids, logprobs=logprobs, finish_reason="stop"
            )
        ]
    )
    VllmAsyncGenerationWorkerImpl._observe_request_capture(worker, request, final)

    state = worker._capture_calls[id(request)]
    assert state.observation_error is None
    assert state.generated_token_ids == [12, 13, 14]
    assert state.generated_logprobs == [-0.1, -0.2, -0.3]
    assert state.terminal_finish_reason == "stop"


def test_token_read_before_its_logprob_is_observed_with_the_next_output():
    """vLLM's engine loop appends while the HTTP thread reads; a lagging logprob waits."""
    worker = _worker_with_capture(_MemorySink())
    request = _FakeRequest(
        ng_capture={"rollout_id": "r0", "model_call_id": "c1", "mode": "text"},
        stream=False,
    )
    VllmAsyncGenerationWorkerImpl._begin_request_capture(worker, request, [10, 11])
    # Caught between the engine appending token 13 and appending its logprob.
    token_ids = [12, 13]
    logprobs = [{12: SimpleNamespace(logprob=-0.1)}]
    output = SimpleNamespace(token_ids=token_ids, logprobs=logprobs)

    VllmAsyncGenerationWorkerImpl._observe_request_capture(
        worker, request, SimpleNamespace(outputs=[output])
    )

    state = worker._capture_calls[id(request)]
    assert state.observation_error is None
    assert state.generated_token_ids == [12]

    logprobs.append({13: SimpleNamespace(logprob=-0.2)})
    output.finish_reason = "stop"
    VllmAsyncGenerationWorkerImpl._observe_request_capture(
        worker, request, SimpleNamespace(outputs=[output])
    )

    assert state.observation_error is None
    assert state.generated_token_ids == [12, 13]
    assert state.generated_logprobs == [-0.1, -0.2]
    assert state.terminal_finish_reason == "stop"


def test_terminal_token_without_a_logprob_makes_its_call_uncuttable():
    worker = _worker_with_capture(_MemorySink())
    request = _FakeRequest(
        ng_capture={"rollout_id": "r0", "model_call_id": "c1", "mode": "text"},
        stream=False,
    )
    VllmAsyncGenerationWorkerImpl._begin_request_capture(worker, request, [10, 11])
    # Nothing appends after the terminal step, so this logprob never arrives.
    final = SimpleNamespace(
        outputs=[
            SimpleNamespace(
                token_ids=[12, 13],
                logprobs=[{12: SimpleNamespace(logprob=-0.1)}],
                finish_reason="stop",
            )
        ]
    )

    VllmAsyncGenerationWorkerImpl._observe_request_capture(worker, request, final)

    state = worker._capture_calls[id(request)]
    assert "token_count=2, logprob_count=1" in state.observation_error
    assert state.terminal_finish_reason is None


def test_generation_cut_stages_only_tokens_past_the_last_sealed_cut():
    state = _RequestCaptureState(call=None, prompt_token_ids=[1])
    state.observe([10, 11], [-0.1, -0.2])

    first = state.freeze_for_checkpoint("c1")
    assert first == ("active/c1/0", 0, [10, 11], [-0.1, -0.2], [10, 11], [-0.1, -0.2])
    state.observe([12], [-0.3])  # decoding continues during the TQ write
    state.seal_frozen_cut("c1")
    state.observe([13], [-0.4])

    key, sequence, chunk, chunk_logprobs, prefix, _ = state.freeze_for_checkpoint("c2")
    assert (key, sequence, chunk, chunk_logprobs, prefix) == (
        "active/c2/1",
        1,
        [12, 13],
        [-0.3, -0.4],
        [10, 11, 12, 13],
    )
    state.rollback_frozen_cut("c2")

    # A failed cut's tokens stay unstaged, and its chunk name is never reused.
    key, sequence, chunk, _, _, _ = state.freeze_for_checkpoint("c3")
    assert (key, sequence, chunk) == ("active/c3/2", 2, [12, 13])


def test_token_capture_snapshot_fence_does_not_pause_decoding():
    class FakeLLM:
        pause_calls = 0

        async def pause_generation(self, **kwargs):
            self.pause_calls += 1

    async def scenario():
        worker = object.__new__(VllmAsyncGenerationWorkerImpl)
        worker.cfg = {"vllm_cfg": {"async_engine": True}}
        worker.llm = FakeLLM()
        worker._prefix_cleanup = None
        worker._token_capture_snapshot_gate = _TokenCaptureSnapshotGate()
        worker._token_capture_fence_executor = ThreadPoolExecutor(max_workers=1)
        try:
            assert await worker.begin_token_capture_snapshot_fence_async()
            assert worker.llm.pause_calls == 0
            entered = threading.Event()
            released = threading.Event()

            def terminal_write():
                entered.set()
                worker._token_capture_snapshot_gate.enter()
                released.set()
                worker._token_capture_snapshot_gate.exit()

            thread = threading.Thread(target=terminal_write)
            thread.start()
            assert entered.wait(timeout=5)
            assert not released.wait(timeout=0.05)
            assert await worker.end_token_capture_snapshot_fence_async()
            assert released.wait(timeout=5)
            thread.join(timeout=5)
        finally:
            worker._token_capture_snapshot_gate.reopen()
            worker._token_capture_fence_executor.shutdown()

    asyncio.run(scenario())


def _gate_admits_write(gate: _TokenCaptureSnapshotGate, *, timeout: float) -> bool:
    """Whether a terminal write passes the gate within ``timeout`` seconds."""
    admitted = threading.Event()

    def terminal_write():
        gate.enter()
        admitted.set()
        gate.exit()

    threading.Thread(target=terminal_write, daemon=True).start()
    return admitted.wait(timeout=timeout)


def test_snapshot_gate_close_after_its_release_is_a_no_op():
    gate = _TokenCaptureSnapshotGate()
    epoch = gate.begin_epoch()
    # The driver timed out and released before the worker ran the close.
    gate.reopen()
    gate.close_and_wait(epoch)
    assert _gate_admits_write(gate, timeout=5)

    # A later fence still closes the gate normally.
    gate.close_and_wait(gate.begin_epoch())
    assert not _gate_admits_write(gate, timeout=0.05)
    gate.reopen()


def test_snapshot_gate_release_unblocks_a_close_still_draining():
    gate = _TokenCaptureSnapshotGate()
    gate.enter()
    closer = threading.Thread(
        target=gate.close_and_wait, args=(gate.begin_epoch(),), daemon=True
    )
    closer.start()
    closer.join(timeout=0.05)
    assert closer.is_alive()
    gate.reopen()
    closer.join(timeout=5)
    assert not closer.is_alive()
    gate.exit()
    assert _gate_admits_write(gate, timeout=5)


def _fence_worker() -> VllmAsyncGenerationWorkerImpl:
    worker = object.__new__(VllmAsyncGenerationWorkerImpl)
    worker.cfg = {"vllm_cfg": {"async_engine": True}}
    worker._token_capture_snapshot_gate = _TokenCaptureSnapshotGate()
    worker._prefix_cleanup = None
    worker._generation_cut_control_executor = ThreadPoolExecutor(max_workers=1)
    worker._token_capture_fence_executor = ThreadPoolExecutor(max_workers=1)
    return worker


def _shutdown_fence_worker(worker: VllmAsyncGenerationWorkerImpl) -> None:
    worker._token_capture_snapshot_gate.reopen()
    worker._generation_cut_control_executor.shutdown()
    worker._token_capture_fence_executor.shutdown()


def test_snapshot_fence_is_not_queued_behind_a_running_cut():
    async def scenario():
        worker = _fence_worker()
        stale_cut_running = threading.Event()
        release_stale_cut = threading.Event()
        try:
            stale_cut = asyncio.ensure_future(
                worker._run_generation_cut_control(
                    lambda: (stale_cut_running.set(), release_stale_cut.wait(10))
                )
            )
            assert await asyncio.to_thread(stale_cut_running.wait, 5)
            await asyncio.wait_for(
                worker.begin_token_capture_snapshot_fence_async(), timeout=5
            )
            assert not stale_cut.done()
            release_stale_cut.set()
            await stale_cut
        finally:
            release_stale_cut.set()
            _shutdown_fence_worker(worker)

    asyncio.run(scenario())


def test_snapshot_fence_closed_after_its_release_leaves_the_gate_open():
    async def scenario():
        worker = _fence_worker()
        release_fence_thread = threading.Event()
        try:
            # Hold the fence thread so the close is still queued when the
            # driver gives up and releases the fence.
            worker._token_capture_fence_executor.submit(release_fence_thread.wait, 10)
            begin = asyncio.ensure_future(
                worker.begin_token_capture_snapshot_fence_async()
            )
            await asyncio.sleep(0)
            assert not begin.done()
            assert await worker.end_token_capture_snapshot_fence_async()
            release_fence_thread.set()
            assert await asyncio.wait_for(begin, timeout=5)
            assert _gate_admits_write(worker._token_capture_snapshot_gate, timeout=5)
        finally:
            release_fence_thread.set()
            _shutdown_fence_worker(worker)

    asyncio.run(scenario())


def test_generation_set_rollout_weight_version_fans_out(monkeypatch):
    gen = _generation_with_mock_group()
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_generation.ray.get",
        lambda futures: futures,
    )
    gen.set_rollout_weight_version(7)
    gen.worker_group.run_all_workers_single_data.assert_called_once_with(
        "set_rollout_weight_version",
        version=7,
        run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
    )


# ---------------------------------------------------------------------------
# S4: the request-path hookup (begin -> finish/abort around a served call)
# ---------------------------------------------------------------------------


class _FakeRequest(SimpleNamespace):
    pass


def _worker_with_capture(sink: _MemorySink):
    """A real worker, minus the engine, with in-memory token capture."""
    from nemo_gym.token_id_capture.adapters.vllm import VLLMCaptureAdapter

    worker = _real_worker()
    worker._capture_sink = sink
    worker.token_capture = RolloutTokenCapture(
        sink=sink,
        weight_version_fn=lambda: worker._rollout_weight_version,
        adapter=VLLMCaptureAdapter(),
    )
    return worker


class _BatchPrefixSink(_MemorySink):
    def __init__(self) -> None:
        super().__init__()
        self.batches: list[list[int]] = []
        self.fail_after_write = False
        self.fail_multi_row = False
        self.before_write = lambda: None

    def stage_generation_prefix_batch(
        self, records, *, checkpoint_id, chunk_sequences, attachments=None
    ):
        self.batches.append([len(record.token_ids_delta) for record in records])
        self.before_write()
        keys = super().stage_generation_prefix_batch(
            records,
            checkpoint_id=checkpoint_id,
            chunk_sequences=chunk_sequences,
            attachments=attachments,
        )
        if self.fail_after_write or (self.fail_multi_row and len(records) > 1):
            # Like TQTokenSink after a failed put: discard the batch, then raise.
            self.clear(keys)
            raise RuntimeError("injected batch transport failure after writes")
        return keys


def _batch_worker_fixture(lengths, *, batch_size=256, max_tokens=4_194_304):
    sink = _BatchPrefixSink()
    worker = _worker_with_capture(sink)
    worker._generation_prefix_batch_max_rows = batch_size
    worker._generation_prefix_batch_max_tokens = max_tokens
    requests, prefixes = [], []
    for i, length in enumerate(lengths):
        request = _FakeRequest(
            ng_capture={
                "rollout_id": f"r{i}",
                "model_call_id": f"c{i}",
                "parent_call_id": None,
                "prev_len": 0,
                "mode": "text",
            },
            stream=False,
        )
        VllmAsyncGenerationWorkerImpl._begin_request_capture(worker, request, [10])
        state = worker._capture_calls[id(request)]
        state.effective_output_limit = 128
        with state.lock:
            state.observe(list(range(20, 20 + length)), [-0.1] * length)
        requests.append(request)
        prefixes.append(
            GenerationCutPrefix(
                ticket_id=f"t{i}",
                rollout_id=f"r{i}",
                attempt=0,
                model_call_id=f"c{i}",
                admitted_at=1.0,
            )
        )
    inventory = GenerationCutInventory.build(
        checkpoint_id="batch-cut",
        server_name="policy",
        active_prefixes=prefixes,
    )
    return worker, sink, requests, inventory


def test_worker_batches_1000_prefixes_and_caches_receipt():
    worker, sink, requests, inventory = _batch_worker_fixture([2] * 1000)
    receipt = worker._checkpoint_generation_cut(inventory)

    assert [len(batch) for batch in sink.batches] == [256, 256, 256, 232]
    assert [ack.ticket_id for ack in receipt.prefixes] == [
        prefix.ticket_id for prefix in inventory.active_prefixes
    ]
    assert all(ack.prefix_token_count == 2 for ack in receipt.prefixes)
    assert all(worker._capture_calls[id(r)].frozen_cut is None for r in requests)
    assert worker._checkpoint_generation_cut(inventory) is receipt
    assert len(sink.batches) == 4


def test_worker_batch_token_limit_handles_ragged_and_oversized_rows():
    worker, sink, _, inventory = _batch_worker_fixture([1, 2, 8, 1], max_tokens=5)
    worker._checkpoint_generation_cut(inventory)
    assert sink.batches == [[2, 3], [9], [2]]  # First chunks include the prompt.


def test_worker_failed_puts_fail_their_calls_and_roll_back():
    worker, sink, requests, inventory = _batch_worker_fixture([2, 3, 4])
    sink.fail_after_write = True

    receipt = worker._checkpoint_generation_cut(inventory)

    # The batch put fails, then each row's own put fails too.
    assert sink.batches == [[3, 4, 5], [3], [4], [5]]
    assert [ack.disposition for ack in receipt.prefixes] == ["durable_failure"] * 3
    assert set(sink.generation_prefix_keys) <= set(sink.cleared_generation_prefix_keys)
    for request, length in zip(requests, [2, 3, 4], strict=True):
        state = worker._capture_calls[id(request)]
        assert state.frozen_cut is None
        assert not state.generation_cut_staging_keys
        assert state.staged_token_count == 0
        assert len(state.generated_token_ids[state.staged_token_count :]) == length
        assert state.lifecycle_lock.acquire(blocking=False)
        state.lifecycle_lock.release()

    sink.fail_after_write = False
    retried = worker._checkpoint_generation_cut(
        GenerationCutInventory.build(
            checkpoint_id="retry-cut",
            server_name="policy",
            active_prefixes=inventory.active_prefixes,
        )
    )
    assert [ack.prefix_token_count for ack in retried.prefixes] == [2, 3, 4]


def test_worker_failed_batch_put_is_retried_row_by_row():
    worker, sink, _, inventory = _batch_worker_fixture([2, 3, 4])
    sink.fail_multi_row = True

    receipt = worker._checkpoint_generation_cut(inventory)

    assert sink.batches == [[3, 4, 5], [3], [4], [5]]
    assert [ack.disposition for ack in receipt.prefixes] == ["durable_prefix"] * 3
    assert [ack.prefix_token_count for ack in receipt.prefixes] == [2, 3, 4]
    # The failed batch was discarded before each row was written again.
    assert len(sink.cleared_generation_prefix_keys) == 3


def test_worker_batch_ack_failure_fails_only_its_call(monkeypatch):
    from nemo_gym._checkpoint import generation_cut

    worker, sink, requests, inventory = _batch_worker_fixture([2, 3, 4])
    original = generation_cut.GenerationCutPrefixAck

    def reject_second(**kwargs):
        if kwargs["model_call_id"] == "c1":
            raise ValueError("injected acknowledgement failure")
        return original(**kwargs)

    # The rejected call is still acknowledged, as a failure.
    reject_second.failure = original.failure
    monkeypatch.setattr(generation_cut, "GenerationCutPrefixAck", reject_second)
    receipt = worker._checkpoint_generation_cut(inventory)

    assert [ack.disposition for ack in receipt.prefixes] == [
        "durable_prefix",
        "durable_failure",
        "durable_prefix",
    ]
    states = [worker._capture_calls[id(request)] for request in requests]
    assert states[1].frozen_cut is None
    assert states[1].staged_token_count == 0
    assert not states[1].generation_cut_staging_keys
    assert [state.staged_token_count for state in (states[0], states[2])] == [2, 4]
    # Only the rejected call's row is cleared; the others stay advertised.
    assert sink.cleared_generation_prefix_keys == [sink.generation_prefix_keys[1]]


def test_worker_batch_interruption_clears_rows_no_call_received(monkeypatch):
    from nemo_gym._checkpoint import generation_cut

    class _Interrupted(BaseException):
        pass

    # The middle ack is interrupted, so the last row is written but never
    # reaches its call: only the batch's own cleanup can discard it.
    worker, sink, requests, inventory = _batch_worker_fixture([2, 3, 4])
    original = generation_cut.GenerationCutPrefixAck

    def interrupt_second(**kwargs):
        if kwargs["model_call_id"] == "c1":
            raise _Interrupted
        return original(**kwargs)

    monkeypatch.setattr(generation_cut, "GenerationCutPrefixAck", interrupt_second)
    with pytest.raises(_Interrupted):
        worker._checkpoint_generation_cut(inventory)

    for request in requests:
        state = worker._capture_calls[id(request)]
        assert state.frozen_cut is None
        assert state.staged_token_count == 0
        assert not state.generation_cut_staging_keys
    assert len(sink.generation_prefix_keys) == 3
    assert set(sink.generation_prefix_keys) <= set(sink.cleared_generation_prefix_keys)


def test_worker_later_batch_failure_keeps_earlier_cuts():
    worker, sink, requests, inventory = _batch_worker_fixture([2] * 3, batch_size=2)

    def fail_second():
        if len(sink.batches) == 2:
            sink.fail_after_write = True

    sink.before_write = fail_second
    receipt = worker._checkpoint_generation_cut(inventory)

    assert [ack.disposition for ack in receipt.prefixes] == [
        "durable_prefix",
        "durable_prefix",
        "durable_failure",
    ]
    earlier_keys = set(sink.generation_prefix_keys[:2])
    assert earlier_keys.isdisjoint(sink.cleared_generation_prefix_keys)
    sink.fail_after_write = False
    sink.before_write = lambda: None
    for request in requests:
        state = worker._capture_calls[id(request)]
        with state.lock:
            state.observe([99], [-0.2])
    retried = worker._checkpoint_generation_cut(
        GenerationCutInventory.build(
            checkpoint_id="retry-cut",
            server_name="policy",
            active_prefixes=inventory.active_prefixes,
        )
    )
    assert [ack.prefix_token_count for ack in retried.prefixes] == [3, 3, 3]
    assert [len(ack.staging_keys) for ack in retried.prefixes] == [2, 2, 1]
    assert earlier_keys.isdisjoint(sink.cleared_generation_prefix_keys)


def test_worker_batch_call_without_output_limit_fails_alone():
    worker, sink, requests, inventory = _batch_worker_fixture([2, 3])
    worker._capture_calls[id(requests[1])].effective_output_limit = None

    receipt = worker._checkpoint_generation_cut(inventory)

    assert [ack.disposition for ack in receipt.prefixes] == [
        "durable_prefix",
        "durable_failure",
    ]
    assert sink.batches == [[3]]
    state = worker._capture_calls[id(requests[1])]
    assert state.frozen_cut is None
    assert state.staged_token_count == 0
    assert state.lifecycle_lock.acquire(blocking=False)
    state.lifecycle_lock.release()


def test_worker_batch_past_deadline_fails_calls_not_yet_written(monkeypatch):
    from nemo_rl.models.generation import generation_cut_capture

    worker, sink, requests, inventory = _batch_worker_fixture([2, 3, 4], batch_size=2)
    clock = {"now": 0.0}
    monkeypatch.setattr(
        generation_cut_capture,
        "time",
        SimpleNamespace(monotonic=lambda: clock["now"], perf_counter=time.perf_counter),
    )

    def expire_during_first_write():
        clock["now"] = 10.0

    sink.before_write = expire_during_first_write
    receipt = worker._checkpoint_generation_cut(inventory, deadline=5.0)

    # The batch already writing completes; the call not yet reached fails.
    receipt.validate_for(inventory)
    assert sink.batches == [[3, 4]]  # First chunks include the prompt.
    assert [ack.disposition for ack in receipt.prefixes] == [
        "durable_prefix",
        "durable_prefix",
        "durable_failure",
    ]
    state = worker._capture_calls[id(requests[2])]
    assert state.frozen_cut is None
    assert not state.generation_cut_staging_keys
    assert len(state.generated_token_ids[state.staged_token_count :]) == 4


def test_worker_batch_past_deadline_rolls_back_pending_calls(monkeypatch):
    from nemo_rl.models.generation import generation_cut_capture

    worker, sink, requests, inventory = _batch_worker_fixture([2, 3, 4], batch_size=3)
    # Every call is frozen before the deadline; it passes before their write.
    readings = iter([0.0, 0.0, 0.0])
    monkeypatch.setattr(
        generation_cut_capture,
        "time",
        SimpleNamespace(
            monotonic=lambda: next(readings, 10.0), perf_counter=time.perf_counter
        ),
    )

    receipt = worker._checkpoint_generation_cut(inventory, deadline=5.0)

    receipt.validate_for(inventory)
    assert not sink.batches
    assert [ack.disposition for ack in receipt.prefixes] == ["durable_failure"] * 3
    for request, length in zip(requests, [2, 3, 4], strict=True):
        state = worker._capture_calls[id(request)]
        assert state.frozen_cut is None
        assert state.staged_token_count == 0
        assert len(state.generated_token_ids[state.staged_token_count :]) == length
        assert state.lifecycle_lock.acquire(blocking=False)
        state.lifecycle_lock.release()


def test_worker_batch_observation_continues_while_lifecycle_waits():
    worker, sink, requests, inventory = _batch_worker_fixture([2, 3])
    state = worker._capture_calls[id(requests[0])]
    write_started = threading.Event()
    release_write = threading.Event()
    lifecycle_done = threading.Event()

    def block_write():
        write_started.set()
        assert release_write.wait(5)

    sink.before_write = block_write
    with ThreadPoolExecutor(max_workers=2) as executor:
        cut = executor.submit(worker._checkpoint_generation_cut, inventory)
        assert write_started.wait(5)

        def terminal_operation():
            with state.lifecycle_lock:
                lifecycle_done.set()

        terminal = executor.submit(terminal_operation)
        try:
            assert not lifecycle_done.wait(0.05)
            with state.lock:
                state.observe([99], [-0.2])
        finally:
            release_write.set()
        receipt = cut.result(timeout=5)
        terminal.result(timeout=5)

    assert receipt.prefixes[0].prefix_token_count == 2
    assert state.generated_token_ids[state.staged_token_count :] == [99]
    assert state.generated_token_ids[: state.staged_token_count] == [20, 21]


@pytest.mark.parametrize("batch_size", [1, 256])
def test_worker_batch_consecutive_cuts_reuse_then_append(batch_size):
    worker, sink, requests, inventory = _batch_worker_fixture(
        [2, 3], batch_size=batch_size
    )
    first = worker._checkpoint_generation_cut(inventory)
    second_inventory = GenerationCutInventory.build(
        checkpoint_id="second-cut",
        server_name="policy",
        active_prefixes=inventory.active_prefixes,
    )
    second = worker._checkpoint_generation_cut(second_inventory)
    assert [ack.staging_keys for ack in second.prefixes] == [
        ack.staging_keys for ack in first.prefixes
    ]
    assert len(sink.generation_prefix_keys) == 2
    if batch_size == 1:
        assert sink.batches == [[3], [4]]

    for request in requests:
        state = worker._capture_calls[id(request)]
        with state.lock:
            state.observe([99], [-0.2])
    third_inventory = GenerationCutInventory.build(
        checkpoint_id="third-cut",
        server_name="policy",
        active_prefixes=inventory.active_prefixes,
    )
    third = worker._checkpoint_generation_cut(third_inventory)
    assert [ack.prefix_token_count for ack in third.prefixes] == [3, 4]
    assert all(len(ack.staging_keys) == 2 for ack in third.prefixes)
    assert [
        list(record.token_ids_delta)
        for _, record in sink.generation_prefix_records[-2:]
    ] == [[99], [99]]


def test_worker_batch_mixes_zero_token_and_active_calls():
    worker, sink, _, inventory = _batch_worker_fixture([2, 0, 3])
    receipt = worker._checkpoint_generation_cut(inventory)
    assert [len(batch) for batch in sink.batches] == [2]
    assert [ack.disposition for ack in receipt.prefixes] == [
        "durable_prefix",
        "durable_failure",
        "durable_prefix",
    ]


class _MemoryPrefixSource:
    def __init__(
        self,
        deltas: dict[str, list[int]],
        *,
        records: dict[str, StagedCallRecord] | None = None,
    ) -> None:
        self.deltas = deltas
        self.records = records or {}
        self.calls: list[list[str]] = []

    def fetch_prefix_chains(self, staging_keys: list[str]) -> StagedPrefix:
        """Text-only chains: the vLLM worker never stages media through this source."""
        self.calls.append(list(staging_keys))
        return StagedPrefix(
            expanded=[token for key in staging_keys for token in self.deltas[key]]
        )

    def fetch_for_finalization(
        self, staging_keys: list[str], *, include_route_fragments: bool = False
    ) -> list[FetchedStagedCall]:
        assert not include_route_fragments
        self.calls.append(list(staging_keys))
        return [
            FetchedStagedCall(
                staging_key=key,
                snapshot=self.records[key],
                routed_len=0,
                extras=self.records[key].extras,
            )
            for key in staging_keys
        ]


def _served_content(gen_ids, logprobs):
    return {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "x"},
                "logprobs": {
                    "content": [
                        {"token": f"token_id:{t}", "logprob": lp}
                        for t, lp in zip(gen_ids, logprobs)
                    ]
                },
            }
        ]
    }


@pytest.mark.parametrize("with_message_tokens", [False, True])
@pytest.mark.parametrize("with_routed_experts", [False, True])
def test_request_capture_round_trip_stages_and_rides_coords(
    with_message_tokens: bool, with_routed_experts: bool
) -> None:
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    request = _FakeRequest(
        ng_capture={
            "rollout_id": "r0",
            "model_call_id": "c1",
            "parent_call_id": None,
            "prev_len": 0,
            "mode": "text",
        },
        stream=False,
    )
    VllmAsyncGenerationWorkerImpl._begin_request_capture(worker, request, [10, 11, 12])
    content = _served_content([13, 14], [-0.1, -0.2])
    message = content["choices"][0]["message"]
    if with_message_tokens:
        # The HTTP serializer preserves these dynamic fields on the message.
        message.update(
            prompt_token_ids=[10, 11, 12],
            generation_token_ids=[13, 14],
            generation_log_probs=[-0.1, -0.2],
        )
    if with_routed_experts:
        message["routed_experts"] = [[[0]]] * 5
    content = VllmAsyncGenerationWorkerImpl._finish_request_capture(
        worker, request, content
    )
    # Bytes were staged before the coords existed (fail-closed ordering).
    assert len(sink.records) == 1
    assert sink.records[0].token_ids_delta == [10, 11, 12, 13, 14]
    assert sink.records[0].token_mask_delta == [0.0, 0.0, 0.0, 1.0, 1.0]
    assert sink.records[0].generation_log_probs_delta == [
        0.0,
        0.0,
        0.0,
        -0.1,
        -0.2,
    ]
    if with_routed_experts:
        assert sink.records[0].extras["routed_experts"]
    else:
        assert sink.records[0].extras is None
    coords = content["ng_commit_coords"]
    assert coords["disposition"] == "staged"
    assert (coords["delta_len"], coords["cum_len"]) == (5, 5)
    # Coords are token-free: hashes ride the wire, deltas stay in the sink.
    assert "token_ids_delta" not in coords
    assert coords["chain_hash"] == sink.records[0].chain_hash
    assert coords["cumulative_hash"] == sink.records[0].cumulative_hash
    # Only the ordinary response fields and coords transit worker -> gate.
    assert content["choices"] == [
        {"index": 0, "message": {"role": "assistant", "content": "x"}}
    ]
    assert worker._capture_calls == {}


def test_generation_cut_stages_prefix_then_terminal_row_replaces_it():
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    worker._generation_prefix_cuts_enabled = True
    request = _FakeRequest(
        ng_capture={
            "rollout_id": "r0",
            "model_call_id": "c1",
            "parent_call_id": None,
            "prev_len": 0,
            "mode": "text",
        },
        stream=False,
    )
    VllmAsyncGenerationWorkerImpl._begin_request_capture(worker, request, [10, 11])
    state = worker._capture_calls[id(request)]
    with state.lock:
        state.effective_output_limit = 8
        state.observe([20, 21], [-0.1, -0.2])

    inventory = GenerationCutInventory.build(
        checkpoint_id="checkpoint-1",
        server_name="policy_model",
        active_prefixes=[
            GenerationCutPrefix(
                ticket_id="ticket-1",
                rollout_id="r0",
                attempt=0,
                model_call_id="c1",
                admitted_at=1.0,
            )
        ],
    )
    receipt = worker._checkpoint_generation_cut(inventory)
    replayed = worker._checkpoint_generation_cut(inventory)

    assert replayed is receipt
    assert len(sink.generation_prefix_records) == 1
    assert receipt.prefixes[0].disposition == "durable_prefix"
    assert receipt.prefixes[0].prefix_token_count == 2
    assert sink.generation_prefix_records[0][1].token_ids_delta == [10, 11, 20, 21]
    assert state.generation_cut_staging_keys == sink.generation_prefix_keys

    content = VllmAsyncGenerationWorkerImpl._finish_request_capture(
        worker,
        request,
        _served_content([20, 21, 22], [-0.1, -0.2, -0.3]),
    )
    assert content["ng_commit_coords"]["disposition"] == "staged"
    assert sink.records[-1].token_ids_delta == [10, 11, 20, 21, 22]
    assert sink.cleared_generation_prefix_keys == sink.generation_prefix_keys
    assert worker._capture_calls == {}
    assert worker._capture_calls_by_model_call_id == {}


@pytest.mark.parametrize(
    ("completed_capture_key", "matches"), [("r0-a1", True), ("r0", False)]
)
def test_completed_retry_is_matched_by_its_attempt_qualified_capture_key(
    completed_capture_key: str, matches: bool
):
    """A retry's terminal row is keyed ``<rollout>-a<attempt>``, as Gym keys it."""
    worker = _worker_with_capture(_MemorySink())
    worker._remember_completed_capture(
        "c2",
        SimpleNamespace(
            rollout_id=completed_capture_key,
            disposition="staged",
            staging_key="terminal-key",
            digest="a" * 64,
        ),
        3,
        effective_output_limit=128,
        terminal_finish_reason="stop",
    )
    prefix = GenerationCutPrefix(
        ticket_id="ticket-1",
        rollout_id="r0",
        attempt=1,
        model_call_id="c2",
        admitted_at=1.0,
    )

    if not matches:
        with pytest.raises(RuntimeError, match="does not match the completed call"):
            worker._completed_generation_cut_ack(prefix, "checkpoint-1")
        return
    ack = worker._completed_generation_cut_ack(prefix, "checkpoint-1")

    assert ack.disposition == "durable_prefix"
    assert ack.cut_kind == "terminal_completion"
    assert ack.staging_keys == ("terminal-key",)
    assert ack.prefix_token_count == 3


def test_generation_cut_past_its_deadline_fails_uncut_calls_without_staging():
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    worker._generation_prefix_cuts_enabled = True
    request = _FakeRequest(
        ng_capture={"rollout_id": "r0", "model_call_id": "c1", "mode": "text"},
        stream=False,
    )
    VllmAsyncGenerationWorkerImpl._begin_request_capture(worker, request, [10])
    state = worker._capture_calls[id(request)]
    with state.lock:
        state.effective_output_limit = 8
        state.observe([11, 12], [-0.1, -0.2])
    inventory = GenerationCutInventory.build(
        checkpoint_id="checkpoint-1",
        server_name="policy_model",
        active_prefixes=[
            GenerationCutPrefix(
                ticket_id="ticket-1",
                rollout_id="r0",
                attempt=0,
                model_call_id="c1",
                admitted_at=1.0,
            )
        ],
    )

    # Gym has already stopped waiting: the queued cut must not stage rows.
    receipt = worker._checkpoint_generation_cut(
        inventory, deadline=time.monotonic() - 1.0
    )

    receipt.validate_for(inventory)
    assert [ack.disposition for ack in receipt.prefixes] == ["durable_failure"]
    assert sink.generation_prefix_records == []
    assert state.frozen_cut is None
    assert state.generated_token_ids == [11, 12]


def _active_call_inventory(
    worker, *calls: tuple[str, list[int]]
) -> tuple[GenerationCutInventory, list]:
    """Admit one live call per (model_call_id, generated IDs) and inventory them."""
    states = []
    prefixes = []
    for index, (model_call_id, generated) in enumerate(calls):
        request = _FakeRequest(
            ng_capture={
                "rollout_id": f"r{index}",
                "model_call_id": model_call_id,
                "mode": "text",
            },
            stream=False,
        )
        VllmAsyncGenerationWorkerImpl._begin_request_capture(worker, request, [10])
        state = worker._capture_calls[id(request)]
        with state.lock:
            state.effective_output_limit = 16
            state.observe(generated, [-0.1] * len(generated))
        states.append(state)
        prefixes.append(
            GenerationCutPrefix(
                ticket_id=f"ticket-{index}",
                rollout_id=f"r{index}",
                attempt=0,
                model_call_id=model_call_id,
                admitted_at=1.0,
            )
        )
    inventory = GenerationCutInventory.build(
        checkpoint_id="checkpoint-1",
        server_name="policy_model",
        active_prefixes=prefixes,
    )
    return inventory, states


def test_generation_cut_refuses_a_prefix_ending_inside_a_character():
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    worker._generation_prefix_cuts_enabled = True
    dinosaur = list("🦖".encode())
    inventory, (split, whole) = _active_call_inventory(
        worker,
        ("split", list(b"Sure ") + dinosaur[:2]),
        ("whole", list(b"Sure ") + dinosaur),
    )

    receipt = worker._checkpoint_generation_cut(inventory)

    # vLLM would prime the resumed tail on half a character and echo the
    # prompt into the response, so that call regenerates instead.
    assert [ack.disposition for ack in receipt.prefixes] == [
        "durable_failure",
        "durable_prefix",
    ]
    assert len(sink.generation_prefix_records) == 1
    assert split.frozen_cut is None
    assert split.generated_token_ids == list(b"Sure ") + dinosaur[:2]
    assert whole.generation_cut_staging_keys == sink.generation_prefix_keys


def test_dropped_output_delta_keeps_its_call_uncuttable():
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    worker._generation_prefix_cuts_enabled = True
    inventory, (poisoned, healthy) = _active_call_inventory(
        worker, ("poisoned", [11, 12]), ("healthy", [13, 14])
    )
    with poisoned.lock:
        poisoned.observe([15], [])
        # A later valid delta must not hide the tokens the bad one dropped.
        poisoned.observe([16], [-0.1])
        assert poisoned.observation_error is not None

    receipt = worker._checkpoint_generation_cut(inventory)

    assert [ack.disposition for ack in receipt.prefixes] == [
        "durable_failure",
        "durable_prefix",
    ]
    assert poisoned.frozen_cut is None
    assert healthy.generation_cut_staging_keys == sink.generation_prefix_keys


def _sampling_params(**overrides) -> SimpleNamespace:
    defaults = dict(stop=[], bad_words=[], presence_penalty=0.0, frequency_penalty=0.0)
    return SimpleNamespace(**{**defaults, **overrides})


def _record_cut_constraints(
    state: Any, sampling_params: Any, *, tokenizer: Any | None = None
) -> None:
    VllmAsyncGenerationWorkerImpl._record_request_cut_constraints(
        SimpleNamespace(
            _get_request_capture=lambda request: state,
            _generation_cut_tokenizer=tokenizer,
        ),
        object(),
        sampling_params,
    )


@pytest.mark.parametrize(
    ("resolved_ids", "blocked"),
    [
        (None, True),
        ([], True),
        ([[]], True),
        ([[18]], False),
        ([[18], [19], [20], [27], [28], [29]], False),
        ([[18, 19]], True),
        ([[18], [19, 20]], True),
    ],
)
def test_generation_cut_allows_only_resolved_single_token_bad_words(
    resolved_ids: list[list[int]] | None, blocked: bool
) -> None:
    params = _sampling_params(bad_words=["<image>"], bad_words_token_ids=resolved_ids)

    assert ("bad_words" in _generation_cut_blockers(params)) is blocked


class _BadWordsTokenizer:
    max_token_id = 255

    def __init__(self, encodings: dict[str, list[int]]) -> None:
        self.encodings = encodings

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return list(self.encodings[text])


@pytest.mark.parametrize(
    ("encodings", "blocked"),
    [
        ({"<image>": [18], " <image>": [32, 18]}, False),
        ({"<image>": [18], " <image>": [19]}, False),
        ({"<image>": [18, 19], " <image>": [20, 21]}, True),
    ],
)
def test_request_cut_constraints_resolve_bad_words_without_mutating_request(
    encodings: dict[str, list[int]], blocked: bool
) -> None:
    SamplingParams = pytest.importorskip("vllm").SamplingParams
    params = SamplingParams(bad_words=["<image>"])
    state = SimpleNamespace(lock=threading.Lock(), uncuttable_reason=None)

    _record_cut_constraints(state, params, tokenizer=_BadWordsTokenizer(encodings))

    assert (state.uncuttable_reason is not None) is blocked
    assert params.bad_words == ["<image>"]
    assert params.bad_words_token_ids is None


def test_single_token_bad_words_remain_banned_after_prefix_restore() -> None:
    SamplingParams = pytest.importorskip("vllm").SamplingParams
    from vllm.v1.sample.ops.bad_words import _apply_bad_words_single_batch

    bad_words = [
        "<image>",
        "<img>",
        "</img>",
        "<so_embedding>",
        "<so_start>",
        "<so_end>",
    ]
    forbidden_ids = [18, 19, 20, 27, 28, 29]
    encodings = {}
    for word, token_id in zip(bad_words, forbidden_ids, strict=True):
        encodings[word] = [token_id]
        encodings[" " + word] = [32, token_id]
    tokenizer = _BadWordsTokenizer(encodings)

    # The resumed call starts with empty output history: its saved output is
    # now prompt tokens. Single-token bans must mask the same logits either way.
    for output_history in ([65, 66], []):
        params = SamplingParams(bad_words=list(bad_words))
        params.update_from_tokenizer(tokenizer)
        assert params.bad_words_token_ids == [[i] for i in forbidden_ids]
        assert "bad_words" not in _generation_cut_blockers(params)
        logits = torch.zeros(256)
        _apply_bad_words_single_batch(
            logits, params.bad_words_token_ids, output_history
        )
        assert torch.isneginf(logits[forbidden_ids]).all()
        assert logits[65] == 0


@pytest.mark.parametrize(
    "setting",
    [
        {"stop": ["</answer>"]},
        {"bad_words": ["forbidden word"]},
        {"presence_penalty": 0.5},
        {"frequency_penalty": -0.2},
        {"seed": 7, "temperature": 1.0},
    ],
)
def test_sampling_settings_a_resumed_tail_cannot_honor_keep_the_call_uncut(
    setting, monkeypatch
):
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_worker_async._LOGGED_UNCUTTABLE_REASONS",
        set(),
    )
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    worker._generation_prefix_cuts_enabled = True
    inventory, (constrained, plain) = _active_call_inventory(
        worker, ("constrained", [11, 12]), ("plain", [13, 14])
    )
    _record_cut_constraints(constrained, _sampling_params(**setting))
    _record_cut_constraints(plain, _sampling_params())
    assert constrained.uncuttable_reason is not None
    assert plain.uncuttable_reason is None

    receipt = worker._checkpoint_generation_cut(inventory)

    # The constrained call regenerates after a restore instead of resuming a
    # tail that would miss a straddling stop string or uncount its prefix.
    assert [ack.disposition for ack in receipt.prefixes] == [
        "durable_failure",
        "durable_prefix",
    ]
    assert constrained.frozen_cut is None
    assert constrained.generated_token_ids == [11, 12]
    assert plain.generation_cut_staging_keys == sink.generation_prefix_keys


@pytest.mark.parametrize(
    ("seed", "temperature", "blocked"),
    [
        # A restored call would restart the seeded generator and reuse its draws.
        (7, 1.0, True),
        # Greedy decoding draws nothing, so the seed is irrelevant.
        (7, 0.0, False),
        # Unseeded calls draw from the engine's shared generator.
        (None, 1.0, False),
    ],
)
def test_only_a_seeded_sampled_call_is_kept_out_of_cuts(seed, temperature, blocked):
    blockers = _generation_cut_blockers(
        _sampling_params(seed=seed, temperature=temperature)
    )

    assert ("seed" in blockers) is blocked


def test_uncuttable_sampling_settings_warn_once_per_reason(monkeypatch, caplog):
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_worker_async._LOGGED_UNCUTTABLE_REASONS",
        set(),
    )
    states = [SimpleNamespace(uncuttable_reason=None, lock=threading.Lock())]
    states.append(SimpleNamespace(uncuttable_reason=None, lock=threading.Lock()))

    with caplog.at_level(logging.WARNING):
        for state in states:
            _record_cut_constraints(state, _sampling_params(stop=["x"]))

    assert all(state.uncuttable_reason for state in states)
    assert caplog.text.count("are not cut; they regenerate after a restore") == 1


def _captured_image_media() -> CapturedMedia:
    tensors = {
        "imgs": torch.ones(1, 2, 3),
        "imgs_sizes": torch.tensor([[1, 2]], dtype=torch.int32),
    }
    # As _capture_request_media returns it when prefix cuts are enabled.
    return CapturedMedia(
        items=(
            CapturedMediaItem(
                modality="image",
                placeholder_offset=0,
                placeholder_length=2,
                placeholder_digest=hashlib.sha256(
                    json.dumps([18, 18], separators=(",", ":")).encode()
                ).hexdigest(),
                token_id=18,
                embedding_spans=((0, 2),),
                imgs_sizes=((1, 2),),
            ),
        ),
        tensors=tensors,
        tensor_digests=media_tensor_digests(tensors),
    )


def _single_generation_cut_inventory(
    checkpoint_id: str, *, model_call_id: str = "c1"
) -> GenerationCutInventory:
    return GenerationCutInventory.build(
        checkpoint_id=checkpoint_id,
        server_name="policy_model",
        active_prefixes=[
            GenerationCutPrefix(
                ticket_id=f"ticket-{checkpoint_id}",
                rollout_id="r0",
                attempt=0,
                model_call_id=model_call_id,
                admitted_at=1.0,
            )
        ],
    )


def _worker_with_active_media_prefix(sink: _MemorySink):
    worker = _worker_with_capture(sink)
    request = _FakeRequest(
        ng_capture={
            "rollout_id": "r0",
            "model_call_id": "c1",
            "parent_call_id": None,
            "prev_len": 0,
            "mode": "text",
        },
        stream=False,
    )
    media = _captured_image_media()
    VllmAsyncGenerationWorkerImpl._begin_request_capture(
        worker, request, [18, 18], media=media
    )
    state = worker._capture_calls[id(request)]
    state.effective_output_limit = 8
    state.observe([20], [-0.1])
    return worker, request, state, media


def test_generation_cut_stages_media_digests_only_with_first_prefix_row():
    sink = _MemorySink()
    worker, _, state, media = _worker_with_active_media_prefix(sink)

    first_receipt = worker._checkpoint_generation_cut(
        _single_generation_cut_inventory("checkpoint-1")
    )
    state.observe([21], [-0.2])
    second_receipt = worker._checkpoint_generation_cut(
        _single_generation_cut_inventory("checkpoint-2")
    )

    first, second = [record for _, record in sink.generation_prefix_records]
    assert first.extras == {
        "media_spans": [media.items[0].to_dict()],
        MEDIA_TENSOR_DIGESTS_FIELD: media.tensor_digests,
    }
    assert second.extras is None
    # Prefix rows never hold pixels; only the terminal row stages them.
    assert sink.generation_prefix_attachments == [None, None]
    assert second_receipt.prefixes[0].staging_keys == (
        *first_receipt.prefixes[0].staging_keys,
        sink.generation_prefix_keys[-1],
    )
    assert state.generation_cut_staging_keys == list(
        second_receipt.prefixes[0].staging_keys
    )
    assert not sink.cleared_generation_prefix_keys


def test_generation_cut_refuses_media_captured_without_digests(caplog):
    sink = _MemorySink()
    worker, _, state, _ = _worker_with_active_media_prefix(sink)
    state.media = replace(state.media, tensor_digests=None)

    with caplog.at_level(logging.ERROR):
        receipt = worker._checkpoint_generation_cut(
            _single_generation_cut_inventory("checkpoint-1")
        )

    # Only this call is left uncut; it regenerates after a restore.
    assert [ack.disposition for ack in receipt.prefixes] == ["durable_failure"]
    assert "without tensor digests" in caplog.text
    assert not sink.generation_prefix_records
    assert state.frozen_cut is None


def _worker_with_active_text_prefix(sink: _MemorySink):
    worker = _worker_with_capture(sink)
    request = _FakeRequest(
        ng_capture={
            "rollout_id": "r0",
            "model_call_id": "c1",
            "parent_call_id": None,
            "prev_len": 0,
            "mode": "text",
        },
        stream=False,
    )
    VllmAsyncGenerationWorkerImpl._begin_request_capture(worker, request, [10, 11])
    state = worker._capture_calls[id(request)]
    state.effective_output_limit = 8
    state.observe([20], [-0.1])
    return worker, state


@pytest.mark.parametrize(
    ("media", "cuts"),
    [(False, 1), (True, 1), (True, 2)],
    ids=["text", "image_one_cut", "image_two_cuts"],
)
def test_generation_cut_digest_written_at_the_cut_is_the_one_restore_rebuilds(
    media: bool, cuts: int
):
    sink = _MemorySink()
    if media:
        worker, _, state, _ = _worker_with_active_media_prefix(sink)
    else:
        worker, state = _worker_with_active_text_prefix(sink)
    for index in range(cuts):
        if index:
            state.observe([21], [-0.2])
        receipt = worker._checkpoint_generation_cut(
            _single_generation_cut_inventory(f"checkpoint-{index + 1}")
        )
    (ack,) = receipt.prefixes
    records = dict(
        zip(
            sink.generation_prefix_keys,
            (record for _, record in sink.generation_prefix_records),
            strict=True,
        )
    )
    assert list(records) == list(ack.staging_keys)
    first = records[ack.staging_keys[0]]
    assert (MEDIA_TENSOR_DIGESTS_FIELD in (first.extras or {})) is media

    def resolve(staged: dict[str, StagedCallRecord]):
        resumed = _worker_with_capture(sink)
        resumed._rollout_weight_version = worker._rollout_weight_version
        resumed._staging_source = _MemoryPrefixSource({}, records=staged)
        request = _FakeRequest(
            ng_capture={
                "rollout_id": "r0-a1",
                "model_call_id": "c2",
                "mode": "text",
                "generation_cut": {
                    "source_capture_key": "r0",
                    "source_model_call_id": "c1",
                    "staging_keys": list(ack.staging_keys),
                    "prefix_token_count": ack.prefix_token_count,
                    "prefix_digest": ack.prefix_digest,
                    "effective_output_limit": ack.effective_output_limit,
                },
            },
            stream=False,
        )
        return resumed._resolve_generation_cut(
            resumed._capture_admission(request), []
        ).snapshot

    # Restore recomputes the cumulative digest from the staged chunks, the
    # first chunk's media extras included, and must reach the cut's digest:
    # Gym's begin_call rejects a restored snapshot whose digest differs.
    assert resolve(records).digest == ack.prefix_digest
    if media:
        # The tensor digests are covered: a changed one changes the digest.
        digests = dict(first.extras[MEDIA_TENSOR_DIGESTS_FIELD])
        digests["imgs"] = {**digests["imgs"], "sha256": "0" * 64}
        tampered = first.model_copy(
            update={"extras": {**first.extras, MEDIA_TENSOR_DIGESTS_FIELD: digests}}
        )
        assert (
            resolve({**records, ack.staging_keys[0]: tampered}).digest
            != ack.prefix_digest
        )


class _BlockingPrefixSink(_MemorySink):
    def __init__(self) -> None:
        super().__init__()
        self.prefix_write_started = threading.Event()
        self.release_prefix_write = threading.Event()

    def _stage_prefix_row(self, *args, **kwargs) -> str:
        self.prefix_write_started.set()
        assert self.release_prefix_write.wait(5)
        return super()._stage_prefix_row(*args, **kwargs)


def test_media_terminal_completion_waits_for_prefix_cut_then_replaces_it():
    sink = _BlockingPrefixSink()
    worker, request, _, media = _worker_with_active_media_prefix(sink)
    terminal_entered = threading.Event()
    terminal_finished = threading.Event()

    def complete():
        terminal_entered.set()
        try:
            return VllmAsyncGenerationWorkerImpl._finish_request_capture(
                worker, request, _served_content([20, 21], [-0.1, -0.2])
            )
        finally:
            terminal_finished.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        cut = executor.submit(
            worker._checkpoint_generation_cut,
            _single_generation_cut_inventory("checkpoint-1"),
        )
        assert sink.prefix_write_started.wait(5)
        terminal = executor.submit(complete)
        assert terminal_entered.wait(5)
        assert not terminal_finished.wait(0.05)
        sink.release_prefix_write.set()
        receipt = cut.result(timeout=5)
        content = terminal.result(timeout=5)

    assert receipt.prefixes[0].disposition == "durable_prefix"
    assert content["ng_commit_coords"]["disposition"] == "staged"
    assert sink.generation_prefix_attachments == [None]
    assert sink.attachments[0] is media.tensors
    assert sink.cleared_generation_prefix_keys == list(receipt.prefixes[0].staging_keys)


class _FailTerminalSink(_MemorySink):
    def __init__(self) -> None:
        super().__init__()
        self.terminal_attempts: list[tuple[StagedCallRecord, dict | None]] = []

    def stage(
        self, record: StagedCallRecord, *, attachments: dict | None = None
    ) -> StageResult:
        self.terminal_attempts.append((record, attachments))
        return StageResult(
            ok=False,
            staging_key=record.staging_key,
            error="injected terminal staging failure",
        )


def test_media_call_whose_terminal_staging_fails_clears_its_cut_rows():
    sink = _FailTerminalSink()
    worker, request, state, media = _worker_with_active_media_prefix(sink)
    receipt = worker._checkpoint_generation_cut(
        _single_generation_cut_inventory("checkpoint-1")
    )

    content = VllmAsyncGenerationWorkerImpl._finish_request_capture(
        worker, request, _served_content([20, 21], [-0.1, -0.2])
    )

    assert content["ng_commit_coords"]["disposition"] == "capture_failed"
    assert sink.terminal_attempts[0][1] is media.tensors
    # As for a text call, the failed call's cut rows are retired with it.
    assert sink.cleared_generation_prefix_keys == list(receipt.prefixes[0].staging_keys)


class _FailPrefixCleanupSink(_MemorySink):
    def clear(self, staging_keys: list[str]) -> None:
        raise OSError("injected prefix cleanup failure")


def test_canonical_media_row_survives_prefix_cleanup_failure(caplog):
    sink = _FailPrefixCleanupSink()
    worker, request, state, media = _worker_with_active_media_prefix(sink)
    receipt = worker._checkpoint_generation_cut(
        _single_generation_cut_inventory("checkpoint-1")
    )

    with caplog.at_level(logging.ERROR):
        content = VllmAsyncGenerationWorkerImpl._finish_request_capture(
            worker, request, _served_content([20, 21], [-0.1, -0.2])
        )

    assert content["ng_commit_coords"]["disposition"] == "staged"
    assert sink.records[-1].staging_key == "r0/c1"
    assert sink.attachments[-1] is media.tensors
    assert state.generation_cut_staging_keys == list(receipt.prefixes[0].staging_keys)
    assert "failed to clear obsolete generation chunks" in caplog.text


def test_generation_cut_during_media_preprocessing_redispatches_call():
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    worker._capture_media = True

    # begin_request_capture runs only after vLLM finishes media preprocessing.
    # A cut that arrives in that window sees no active state and must tell Gym
    # to redispatch the model call instead of advertising an empty prefix.
    receipt = worker._checkpoint_generation_cut(
        _single_generation_cut_inventory("checkpoint-before-preprocess")
    )

    assert receipt.prefixes[0].disposition == "durable_failure"
    assert not sink.generation_prefix_records
    assert not sink.generation_prefix_attachments


class _FailOncePrefixSink(_MemorySink):
    def __init__(self) -> None:
        super().__init__()
        self._fail_next_prefix = True

    def _stage_prefix_row(self, *args, **kwargs) -> str:
        if self._fail_next_prefix:
            self._fail_next_prefix = False
            raise RuntimeError("injected generation-prefix staging failure")
        return super()._stage_prefix_row(*args, **kwargs)


def test_generation_cut_rolls_back_tokens_after_staging_failure():
    sink = _FailOncePrefixSink()
    worker = _worker_with_capture(sink)
    request = _FakeRequest(
        ng_capture={
            "rollout_id": "r0",
            "model_call_id": "c1",
            "mode": "text",
        },
        stream=False,
    )
    VllmAsyncGenerationWorkerImpl._begin_request_capture(worker, request, [10])
    state = worker._capture_calls[id(request)]
    with state.lock:
        state.effective_output_limit = 8
        state.observe([11, 12], [-0.1, -0.2])

    def inventory(checkpoint_id: str) -> GenerationCutInventory:
        return GenerationCutInventory.build(
            checkpoint_id=checkpoint_id,
            server_name="policy_model",
            active_prefixes=[
                GenerationCutPrefix(
                    ticket_id="ticket-1",
                    rollout_id="r0",
                    attempt=0,
                    model_call_id="c1",
                    admitted_at=1.0,
                )
            ],
        )

    # The failed call is acknowledged, not raised, and its buffer rolls back.
    failed = worker._checkpoint_generation_cut(inventory("checkpoint-1"))
    assert [ack.disposition for ack in failed.prefixes] == ["durable_failure"]
    assert state.frozen_cut is None
    assert state.generated_token_ids == [11, 12]
    assert state.staged_token_count == 0

    receipt = worker._checkpoint_generation_cut(inventory("checkpoint-2"))
    assert receipt.prefixes[0].prefix_token_count == 2
    assert sink.generation_prefix_records[-1][1].token_ids_delta == [10, 11, 12]


class _FailingCallPrefixSink(_MemorySink):
    """Fails every prefix row of one call, as a corrupt row would."""

    def __init__(self, model_call_id: str) -> None:
        super().__init__()
        self._model_call_id = model_call_id

    def _stage_prefix_row(self, record, **kwargs) -> str:
        if record.model_call_id == self._model_call_id:
            raise RuntimeError("injected generation-prefix staging failure")
        return super()._stage_prefix_row(record, **kwargs)


def test_one_failed_call_does_not_fail_the_rest_of_the_cut():
    sink = _FailingCallPrefixSink("failing")
    worker = _worker_with_capture(sink)
    inventory, (failing, healthy) = _active_call_inventory(
        worker, ("failing", [11, 12]), ("healthy", [13, 14])
    )

    receipt = worker._checkpoint_generation_cut(inventory)

    assert [ack.disposition for ack in receipt.prefixes] == [
        "durable_failure",
        "durable_prefix",
    ]
    assert failing.frozen_cut is None
    assert healthy.generation_cut_staging_keys == sink.generation_prefix_keys


def test_call_without_an_output_limit_is_not_frozen_for_the_cut():
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    inventory, (state,) = _active_call_inventory(worker, ("c1", [11, 12]))
    with state.lock:
        state.effective_output_limit = None

    receipt = worker._checkpoint_generation_cut(inventory)

    assert [ack.disposition for ack in receipt.prefixes] == ["durable_failure"]
    assert state.frozen_cut is None
    assert sink.generation_prefix_records == []


def test_unexpected_error_after_the_freeze_rolls_the_call_back():
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    inventory, (state,) = _active_call_inventory(worker, ("c1", [11, 12]))

    class _BrokenTokenizer:
        def decode(self, token_ids, **kwargs):
            raise ValueError("injected decode failure")

    worker._generation_cut_tokenizer = _BrokenTokenizer()
    receipt = worker._checkpoint_generation_cut(inventory)

    assert [ack.disposition for ack in receipt.prefixes] == ["durable_failure"]
    # Without the rollback, every later cut of this call would fail to freeze.
    assert state.frozen_cut is None
    worker._generation_cut_tokenizer = _ByteTokenizer()
    retried = worker._checkpoint_generation_cut(
        GenerationCutInventory.build(
            checkpoint_id="checkpoint-2",
            server_name="policy_model",
            active_prefixes=list(inventory.active_prefixes),
        )
    )
    assert [ack.disposition for ack in retried.prefixes] == ["durable_prefix"]


def _restorable_cut(sink: _MemorySink):
    """Cut one call at weight version 7 and return what a restore would send."""
    worker = _worker_with_capture(sink)
    worker._rollout_weight_version = 7
    inventory, (state,) = _active_call_inventory(worker, ("c1", [12, 13]))
    receipt = worker._checkpoint_generation_cut(inventory)
    (ack,) = receipt.prefixes
    (cut_key,) = ack.staging_keys
    cut_record = sink.generation_prefix_records[-1][1]
    request = _FakeRequest(
        ng_capture={
            "rollout_id": "r0-a1",
            "model_call_id": "c2",
            "mode": "text",
            "generation_cut": {
                "source_capture_key": "r0",
                "source_model_call_id": "c1",
                "staging_keys": [cut_key],
                "prefix_token_count": ack.prefix_token_count,
                "prefix_digest": cut_record.digest,
                "effective_output_limit": ack.effective_output_limit,
            },
        },
        stream=False,
    )
    return cut_key, cut_record, request


def _resumed_worker(sink, *, records, weight_version: int = 9):
    worker = _worker_with_capture(sink)
    worker._rollout_weight_version = weight_version
    worker._staging_source = _MemoryPrefixSource({}, records=records)
    return worker


def test_usable_restored_prefix_is_resumed():
    sink = _MemorySink()
    cut_key, cut_record, request = _restorable_cut(sink)
    worker = _resumed_worker(sink, records={cut_key: cut_record})
    admission = worker._capture_admission(request)

    served, resolved = worker._resolve_usable_generation_cut(admission, [])

    assert served is admission
    assert resolved.snapshot.digest == cut_record.digest
    _drain_cleanup(worker)
    assert sink.cleared_generation_prefix_keys == []


def test_restored_prefix_uses_the_chunks_its_reader_fetched():
    """The worker's batched reader fetches a restore's chunks; resolving the
    prefix must use them instead of reading TQ again."""
    sink = _MemorySink()
    cut_key, cut_record, request = _restorable_cut(sink)
    worker = _resumed_worker(sink, records={cut_key: cut_record})
    admission = worker._capture_admission(request)
    chunks = worker._fetch_generation_cut_rows([cut_key])
    worker._staging_source = _MemoryPrefixSource({}, records={})

    served, resolved = worker._resolve_usable_generation_cut(admission, [], chunks)

    assert served is admission
    assert resolved.snapshot.digest == cut_record.digest


@pytest.mark.parametrize(
    "failure",
    [
        "missing chunk",
        "chunk newer than the worker",
        "digest mismatch",
        "media row unreadable",
    ],
)
def test_unusable_restored_prefix_regenerates_instead_of_failing(failure, caplog):
    sink = _MemorySink()
    cut_key, cut_record, request = _restorable_cut(sink)
    worker = _resumed_worker(
        sink,
        records={} if failure == "missing chunk" else {cut_key: cut_record},
        weight_version=0 if failure == "chunk newer than the worker" else 9,
    )
    admission = worker._capture_admission(request)
    if failure == "digest mismatch":
        admission = admission.model_copy(
            update={
                "generation_cut": admission.generation_cut.model_copy(
                    update={"prefix_digest": "0" * 64}
                )
            }
        )
    if failure == "media row unreadable":
        # The media comes from the same fetch, inside the same fallback.
        def unreadable_media(chunks):
            raise MediaCaptureRejected("Generation-prefix media metadata is missing")

        worker._restore_generation_cut_media = unreadable_media

    with caplog.at_level(logging.WARNING):
        served, resolved = worker._resolve_usable_generation_cut(admission, [])

    assert resolved is None
    assert served.generation_cut is None
    assert "regenerating it from its turn boundary" in caplog.text
    # The regenerated call is admitted as an ordinary call from its boundary.
    VllmAsyncGenerationWorkerImpl._begin_request_capture(
        worker, request, [10], admission=served, prefix_token_ids=[]
    )
    assert worker._capture_calls[id(request)].generation_cut_staging_keys == []
    # The restored rows it will never extend are retired through the gate.
    _drain_cleanup(worker)
    assert sink.cleared_generation_prefix_keys == [cut_key]


def test_restored_generation_cut_is_extended_and_retired_on_completion(
    caplog, monkeypatch
):
    caplog.set_level(logging.INFO)
    sink = _MemorySink()
    original_worker = _worker_with_capture(sink)
    original_worker._rollout_weight_version = 7
    original_request = _FakeRequest(
        ng_capture={
            "rollout_id": "r0",
            "model_call_id": "c1",
            "mode": "text",
        },
        stream=False,
    )
    VllmAsyncGenerationWorkerImpl._begin_request_capture(
        original_worker, original_request, [10, 11]
    )
    original_worker._capture_calls[id(original_request)].effective_output_limit = 128
    VllmAsyncGenerationWorkerImpl._observe_request_capture(
        original_worker,
        original_request,
        SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    token_ids=[12, 13],
                    logprobs=[
                        {12: SimpleNamespace(logprob=-0.1)},
                        {13: SimpleNamespace(logprob=-0.2)},
                    ],
                )
            ]
        ),
    )
    inventory = GenerationCutInventory.build(
        checkpoint_id="checkpoint-1",
        server_name="policy_model",
        active_prefixes=[
            GenerationCutPrefix(
                ticket_id="ticket-1",
                rollout_id="r0",
                attempt=0,
                model_call_id="c1",
                admitted_at=1.0,
            )
        ],
    )
    receipt = original_worker._checkpoint_generation_cut(inventory)
    (cut_key,) = receipt.prefixes[0].staging_keys
    cut_record = sink.generation_prefix_records[-1][1]

    resumed_worker = _worker_with_capture(sink)
    resumed_worker._rollout_weight_version = 9
    resumed_worker._staging_source = _MemoryPrefixSource(
        {}, records={cut_key: cut_record}
    )
    request = _FakeRequest(
        ng_capture={
            "rollout_id": "r0-a1",
            "model_call_id": "c2",
            "mode": "text",
            "generation_cut": {
                "source_capture_key": "r0",
                "source_model_call_id": "c1",
                "staging_keys": [cut_key],
                "prefix_token_count": 2,
                "prefix_digest": cut_record.digest,
                "effective_output_limit": receipt.prefixes[0].effective_output_limit,
            },
        },
        stream=False,
    )
    admission = resumed_worker._capture_admission(request)
    with caplog.at_level(logging.INFO):
        resolved_cut = resumed_worker._resolve_generation_cut(admission, [])
    assert resolved_cut is not None
    cut = resolved_cut.snapshot
    expected_sha = hashlib.sha256(b"12,13").hexdigest()
    assert f"prefix_ids_sha256={expected_sha}" in caplog.text
    # Gym's begin_call rejects a cut whose digest disagrees with its admission.
    bad_digest_admission = admission.model_copy(
        update={
            "generation_cut": admission.generation_cut.model_copy(
                update={"prefix_digest": "0" * 64}
            )
        }
    )
    with pytest.raises(CaptureError, match="identity does not match admission"):
        VllmAsyncGenerationWorkerImpl._begin_request_capture(
            resumed_worker,
            request,
            [10, 11, 12, 13],
            admission=bad_digest_admission,
            prefix_token_ids=[],
            generation_cut=resumed_worker._resolve_generation_cut(
                bad_digest_admission, []
            ).snapshot,
            resumed_generation_token_ids=[12, 13],
        )
    # Chunks prefetched by the worker's read batcher resolve without another read.
    prefetched = resumed_worker._fetch_generation_cut_rows([cut_key])
    resumed_worker._staging_source.calls.clear()
    prefetched_cut = resumed_worker._resolve_generation_cut(admission, [], prefetched)
    assert prefetched_cut == resolved_cut
    assert resumed_worker._staging_source.calls == []
    VllmAsyncGenerationWorkerImpl._begin_request_capture(
        resumed_worker,
        request,
        [10, 11, 12, 13],
        admission=admission,
        prefix_token_ids=[],
        generation_cut=cut,
        generation_cut_routed_experts=resolved_cut.routed_experts,
        resumed_generation_token_ids=[12, 13],
    )
    # Only the restore Gym accepted counts.
    assert VllmAsyncGenerationWorkerImpl.get_generation_prefix_restore_counts(
        resumed_worker
    ) == {"restored_calls": 1, "restored_prefix_tokens": 2}
    replays = []

    def replay(tokenizer, **kwargs):
        replays.append(kwargs)
        return "partial"

    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_worker_async._detokenize_restored_prefix",
        replay,
    )
    # vLLM's processed tail: EOS and any matched stop string already stripped.
    output = SimpleNamespace(token_ids=[14, 151645], text=" tail")
    resumed_worker._restore_response_prefix(
        request,
        SimpleNamespace(outputs=[output]),
        tokenizer=_ByteTokenizer(),
    )
    (replay_kwargs,) = replays
    assert replay_kwargs["original_prompt_token_ids"] == [10, 11]
    assert replay_kwargs["prefix_token_ids"] == [12, 13]
    assert replay_kwargs["terminal"] is None
    assert output.text == "partial tail"

    # A second checkpoint taken before the resumed call finishes must retain
    # the original cut key and append the newly generated tail. Gym v2 stores
    # this complete key list in the replacement attempt's model record.
    VllmAsyncGenerationWorkerImpl._observe_request_capture(
        resumed_worker,
        request,
        SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    token_ids=[14],
                    logprobs=[{14: SimpleNamespace(logprob=-0.3)}],
                )
            ]
        ),
    )
    second_receipt = resumed_worker._checkpoint_generation_cut(
        GenerationCutInventory.build(
            checkpoint_id="checkpoint-2",
            server_name="policy_model",
            active_prefixes=[
                GenerationCutPrefix(
                    ticket_id="ticket-2",
                    rollout_id="r0",
                    attempt=1,
                    model_call_id="c2",
                    admitted_at=2.0,
                )
            ],
        )
    )
    assert second_receipt.prefixes[0].prefix_token_count == 3
    assert second_receipt.prefixes[0].staging_keys[0] == cut_key
    assert len(second_receipt.prefixes[0].staging_keys) == 2

    content = VllmAsyncGenerationWorkerImpl._finish_request_capture(
        resumed_worker,
        request,
        _served_content([14], [-0.3]),
    )

    final_record = sink.records[-1]
    assert content["ng_commit_coords"]["rollout_id"] == "r0-a1"
    assert final_record.token_ids_delta == [10, 11, 12, 13, 14]
    assert final_record.token_mask_delta == [0.0, 0.0, 1.0, 1.0, 1.0]
    assert final_record.generation_log_probs_delta == [
        0.0,
        0.0,
        -0.1,
        -0.2,
        -0.3,
    ]
    assert final_record.weight_version == 7
    assert content["ng_commit_coords"]["weight_version"] == 7
    assert resumed_worker._completed_capture_calls["c2"].generation_token_count == 3
    assert sink.cleared_generation_prefix_keys == list(
        second_receipt.prefixes[0].staging_keys
    )
    assert "generation prefix restored:" in caplog.text


def _route_fetched_call(
    staging_key: str, record: StagedCallRecord
) -> FetchedStagedCall:
    extras = dict(record.extras or {})
    encoded = extras.pop("routed_experts")
    routes = decode_routed_experts(encoded, torch.int16)
    snapshot = StagedCallBaseSnapshot.model_validate(
        record.model_dump(exclude={"extras"})
    )
    return FetchedStagedCall(
        staging_key=staging_key,
        snapshot=snapshot,
        routed_len=int(routes.shape[0]),
        fragment=RouteFragment(
            routes=routes,
            encoding=ROUTE_ENCODING_ENVELOPE,
            extras_metadata_json=json.dumps(
                extras, sort_keys=True, separators=(",", ":")
            ).encode(),
        ),
        extras=extras,
    )


# Router replay: one layer and top-1, so each route row is a single expert ID.
_ROUTER_REPLAY_CALL = {"rollout_id": "r0", "model_call_id": "c1", "mode": "text"}


def _routes(*experts: int) -> torch.Tensor:
    return torch.tensor(experts, dtype=torch.int16).view(-1, 1, 1)


def _staged_routes(record: StagedCallRecord) -> list[int]:
    return decode_routed_experts(record.extras["routed_experts"]).flatten().tolist()


class _EngineRoutes:
    """vLLM's active per-request routes, as a generation cut reads them."""

    def __init__(self) -> None:
        self.by_request: dict[str, torch.Tensor] = {}
        self.reads: list[str] = []
        self.error: Exception | None = None

    def __call__(
        self, engine_request_id: str, *, timeout_s: float | None
    ) -> torch.Tensor | None:
        self.reads.append(engine_request_id)
        if self.error is not None:
            raise self.error
        return self.by_request.get(engine_request_id)


def _router_replay_worker(
    sink: _MemorySink, *, weight_version: int
) -> VllmAsyncGenerationWorkerImpl:
    worker = _worker_with_capture(sink)
    worker._return_routed_experts_enabled = lambda: True
    worker.routed_experts_dtype = torch.int16
    worker._rollout_weight_version = weight_version
    # These tests inject the single-call engine fake. Exercise the mixin's
    # compatibility adapter; grouped vLLM owner handoffs have separate tests.
    worker._snapshot_engine_routed_experts_many = (
        lambda request_ids,
        *,
        deadline: GenerationCutCaptureMixin._snapshot_engine_routed_experts_many(
            worker, request_ids, deadline=deadline
        )
    )
    return worker


def _router_replay_call(
    sink: _MemorySink,
    *,
    ng_capture: dict[str, Any] = _ROUTER_REPLAY_CALL,
    prompt: tuple[int, ...] = (10, 11),
) -> tuple[VllmAsyncGenerationWorkerImpl, _FakeRequest, _EngineRoutes]:
    """An active router-replay call whose cuts read ``_EngineRoutes``."""
    worker = _router_replay_worker(sink, weight_version=7)
    engine_routes = _EngineRoutes()
    worker._snapshot_engine_routed_experts = engine_routes
    request = _FakeRequest(ng_capture=dict(ng_capture), stream=False)
    VllmAsyncGenerationWorkerImpl._begin_request_capture(worker, request, list(prompt))
    worker._capture_calls[id(request)].effective_output_limit = 128
    return worker, request, engine_routes


def _cumulative_output(
    token_ids: list[int], *, request_id: str = "engine-request"
) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        outputs=[
            SimpleNamespace(
                token_ids=token_ids,
                logprobs=[
                    {token_id: SimpleNamespace(logprob=-0.1)} for token_id in token_ids
                ],
            )
        ],
    )


def _observe(
    worker: VllmAsyncGenerationWorkerImpl,
    request: _FakeRequest,
    token_ids: list[int],
    *,
    request_id: str = "engine-request",
) -> None:
    VllmAsyncGenerationWorkerImpl._observe_request_capture(
        worker, request, _cumulative_output(token_ids, request_id=request_id)
    )


def _cut(
    worker: VllmAsyncGenerationWorkerImpl,
    checkpoint_id: str,
    *,
    model_call_id: str = "c1",
) -> GenerationCutPrefixAck:
    (ack,) = worker._checkpoint_generation_cut(
        _single_generation_cut_inventory(checkpoint_id, model_call_id=model_call_id)
    ).prefixes
    return ack


def _staged_route_chunks(
    sink: _MemorySink, staging_keys: tuple[str, ...]
) -> list[FetchedStagedCall]:
    records = {
        key: record
        for key, (_, record) in zip(
            sink.generation_prefix_keys, sink.generation_prefix_records, strict=True
        )
    }
    return [_route_fetched_call(key, records[key]) for key in staging_keys]


def _resume(
    sink: _MemorySink,
    ack: GenerationCutPrefixAck,
    *,
    source: dict[str, Any] = _ROUTER_REPLAY_CALL,
    prefix_token_ids: tuple[int, ...] = (),
) -> tuple[VllmAsyncGenerationWorkerImpl, _FakeRequest, _ResolvedGenerationCut]:
    """Restore a cut on a new worker and begin its resumed call."""
    worker = _router_replay_worker(sink, weight_version=9)
    request = _FakeRequest(
        ng_capture={
            **source,
            "rollout_id": f"{source['rollout_id']}-a1",
            "model_call_id": f"{source['model_call_id']}-resumed",
            "generation_cut": {
                "source_capture_key": source["rollout_id"],
                "source_model_call_id": source["model_call_id"],
                "staging_keys": list(ack.staging_keys),
                "prefix_token_count": ack.prefix_token_count,
                "prefix_digest": ack.prefix_digest,
                "effective_output_limit": ack.effective_output_limit,
                "terminal_finish_reason": ack.terminal_finish_reason,
            },
        },
        stream=False,
    )
    admission = worker._capture_admission(request)
    resolved = worker._resolve_generation_cut(
        admission,
        list(prefix_token_ids),
        _staged_route_chunks(sink, ack.staging_keys),
    )
    cut = resolved.snapshot
    VllmAsyncGenerationWorkerImpl._begin_request_capture(
        worker,
        request,
        [*prefix_token_ids, *cut.token_ids_delta],
        admission=admission,
        prefix_token_ids=list(prefix_token_ids),
        generation_cut=cut,
        generation_cut_routed_experts=resolved.routed_experts,
        generation_cut_chunk_digests=resolved.chunk_digests,
        resumed_generation_token_ids=[
            token_id
            for token_id, mask in zip(
                cut.token_ids_delta, cut.token_mask_delta, strict=True
            )
            if mask == 1.0
        ],
    )
    return worker, request, resolved


def _finish_with_routes(
    worker: VllmAsyncGenerationWorkerImpl,
    request: _FakeRequest,
    generated_token_ids: list[int],
    routes: torch.Tensor | str | None,
) -> StagedCallRecord:
    content = _served_content(generated_token_ids, [-0.4] * len(generated_token_ids))
    if routes is not None:
        content["choices"][0]["message"]["routed_experts"] = (
            routes if isinstance(routes, str) else encode_routed_experts(routes)
        )
    VllmAsyncGenerationWorkerImpl._finish_request_capture(worker, request, content)
    return worker._capture_sink.records[-1]


def test_router_replay_survives_repeated_prefix_cuts_and_resume(caplog) -> None:
    sink = _MemorySink()
    worker, request, engine_routes = _router_replay_call(sink)
    _observe(worker, request, [12, 13])
    # vLLM routes both prompt tokens and the first generated one.
    engine_routes.by_request["engine-request"] = _routes(1, 2, 3)
    assert len(_cut(worker, "checkpoint-1").staging_keys) == 1
    # vLLM outputs are cumulative, and so are its routes.
    _observe(worker, request, [12, 13, 14])
    engine_routes.by_request["engine-request"] = _routes(1, 2, 3, 4)
    ack = _cut(worker, "checkpoint-2")
    # The first chunk carries the full routes and the final placeholder; the
    # next one route per new token, replacing that placeholder.
    assert [_staged_routes(record) for _, record in sink.generation_prefix_records] == [
        [1, 2, 3, -1],
        [4],
    ]

    with caplog.at_level(logging.INFO):
        resumed, resumed_request, resolved = _resume(sink, ack)
    assert resolved.routed_experts.flatten().tolist() == [1, 2, 3, 4, -1]

    first, second = _staged_route_chunks(sink, ack.staging_keys)
    tampered = first.fragment.routes.clone()
    tampered[0, 0, 0] = 99
    with pytest.raises(RuntimeError, match="failed integrity check"):
        resumed._resolve_generation_cut(
            resumed._capture_admission(resumed_request),
            [],
            [replace(first, fragment=replace(first.fragment, routes=tampered)), second],
        )

    with caplog.at_level(logging.INFO):
        staged = _finish_with_routes(
            resumed, resumed_request, [15], _routes(90, 91, 92, 93, 5, 0)
        )
    # The restored prefix keeps the routes it was sampled under.
    assert _staged_routes(staged) == [1, 2, 3, 4, 5, 0]
    assert staged.weight_version == 7
    # Restore and completion log the same kept rows: the generated prefix
    # tokens' routes except the last, whose route the resumed prefill computes.
    kept_sha = hashlib.sha256(
        torch.tensor([[[3]], [[4]]], dtype=torch.int32).numpy().tobytes()
    ).hexdigest()
    logged = re.findall(
        r"generation prefix (restored|completed): .*prefix_routes_sha256=(\S+)",
        caplog.text,
    )
    assert logged == [("restored", kept_sha), ("completed", kept_sha)]


def test_router_replay_cut_reads_routes_once_after_freezing_its_tokens() -> None:
    sink = _MemorySink()
    worker, request, engine_routes = _router_replay_call(sink)
    state = worker._capture_calls[id(request)]
    _observe(worker, request, [12, 13])
    # Decoding observes no routes; only a cut reads them.
    assert engine_routes.reads == []

    def read_while_decoding(engine_request_id, *, timeout_s):
        # The token count is frozen and the state lock released, so decoding
        # has moved on: vLLM routed two more tokens than the cut froze.
        assert not state.lock.locked()
        assert state.frozen_cut == ("checkpoint-1", 2)
        return _routes(1, 2, 3, 4, 5)

    worker._snapshot_engine_routed_experts = read_while_decoding

    assert _cut(worker, "checkpoint-1").disposition == "durable_prefix"
    (_, record) = sink.generation_prefix_records[0]
    # Prompt rows, the first generated token's row, then the final placeholder.
    assert _staged_routes(record) == [1, 2, 3, -1]


@pytest.mark.parametrize(
    "engine_failure",
    ["request_left_vllm", "engine_loop_timeout"],
)
def test_router_replay_cut_without_engine_routes_fails_only_that_call(
    engine_failure: str,
) -> None:
    sink = _MemorySink()
    worker, request, engine_routes = _router_replay_call(sink)
    _observe(worker, request, [12])
    if engine_failure == "engine_loop_timeout":
        engine_routes.error = TimeoutError()
    # Otherwise vLLM finished the request before the cut read its routes, and
    # the HTTP layer had not yet observed the final output.

    assert _cut(worker, "checkpoint-1").disposition == "durable_failure"
    assert engine_routes.reads == ["engine-request"]
    assert sink.generation_prefix_records == []
    state = worker._capture_calls[id(request)]
    assert state.frozen_cut is None
    assert state.staged_token_count == 0


def test_router_replay_bad_routes_fail_only_that_call_and_every_checkpoint_goes_on() -> (
    None
):
    sink = _MemorySink()
    worker, bad_request, engine_routes = _router_replay_call(sink)
    good_request = _FakeRequest(
        ng_capture={"rollout_id": "r1", "model_call_id": "c2", "mode": "text"},
        stream=False,
    )
    VllmAsyncGenerationWorkerImpl._begin_request_capture(worker, good_request, [10, 11])
    worker._capture_calls[id(good_request)].effective_output_limit = 128
    _observe(worker, bad_request, [12], request_id="bad")
    _observe(worker, good_request, [12], request_id="good")
    # Not [tokens, layers, topk]: a route error, not a caller bug.
    engine_routes.by_request["bad"] = torch.tensor([1, 2], dtype=torch.int16)
    engine_routes.by_request["good"] = _routes(1, 2)

    def cut(checkpoint_id: str) -> dict[str, str]:
        receipt = worker._checkpoint_generation_cut(
            GenerationCutInventory.build(
                checkpoint_id=checkpoint_id,
                server_name="policy_model",
                active_prefixes=[
                    GenerationCutPrefix(
                        ticket_id=f"{checkpoint_id}-{model_call_id}",
                        rollout_id=rollout_id,
                        attempt=0,
                        model_call_id=model_call_id,
                        admitted_at=1.0,
                    )
                    for rollout_id, model_call_id in (("r0", "c1"), ("r1", "c2"))
                ],
            )
        )
        return {ack.model_call_id: ack.disposition for ack in receipt.prefixes}

    assert cut("checkpoint-1") == {"c1": "durable_failure", "c2": "durable_prefix"}
    # The bad call is not poisoned: the next checkpoint tries it again.
    engine_routes.by_request["bad"] = _routes(1, 2)
    _observe(worker, good_request, [12, 13], request_id="good")
    engine_routes.by_request["good"] = _routes(1, 2, 3)
    assert cut("checkpoint-2") == {"c1": "durable_prefix", "c2": "durable_prefix"}


def test_vllm_route_chunks_with_mismatched_shapes_are_a_route_error() -> None:
    worker = object.__new__(VllmAsyncGenerationWorkerImpl)
    worker._http_engine_client = SimpleNamespace(
        snapshot_routed_experts=lambda request_id, *, timeout_s: [
            torch.zeros((2, 4, 2), dtype=torch.int16),
            torch.zeros((1, 3, 2), dtype=torch.int16),
        ]
    )

    pieces = worker._snapshot_engine_routed_experts("request", timeout_s=1.0)
    state = _RequestCaptureState(call=None, prompt_token_ids=[])
    with pytest.raises(ValueError, match="must share shape"):
        state.absorb_engine_route_pieces(pieces)


def test_router_replay_short_engine_routes_fail_the_cut_instead_of_padding() -> None:
    """Padding missing rows would put routes on the wrong tokens, and a later
    cut would stage rows that no longer line up, failing the restore digest."""
    sink = _MemorySink()
    worker, request, engine_routes = _router_replay_call(sink)
    _observe(worker, request, [12, 13])
    # Two prompt rows and the first generated token's row are needed.
    engine_routes.by_request["engine-request"] = _routes(1, 2)

    assert _cut(worker, "checkpoint-1").disposition == "durable_failure"
    assert sink.generation_prefix_records == []

    # Once vLLM's routes cover the frozen tokens, the call is cut again, and
    # each later cut's chunks rebuild exactly the digest it advertised.
    engine_routes.by_request["engine-request"] = _routes(1, 2, 3)
    first = _cut(worker, "checkpoint-2")
    assert _resume(sink, first)[2].snapshot.digest == first.prefix_digest
    _observe(worker, request, [12, 13, 14])
    engine_routes.by_request["engine-request"] = _routes(1, 2, 3, 4)
    second = _cut(worker, "checkpoint-3")
    assert len(second.staging_keys) == 2
    assert _resume(sink, second)[2].snapshot.digest == second.prefix_digest


@pytest.mark.parametrize("served_routes", ["missing", "other_model_dims"])
def test_router_replay_resumed_call_with_unfit_routes_still_finishes(
    served_routes: str, caplog
) -> None:
    """Like routes the delta alignment rejects, routes the restored prefix
    cannot be merged into are dropped, not raised: the handler stages after
    its abort path has closed, so a raise would fail the call and leak it."""
    sink = _MemorySink()
    worker, request, engine_routes = _router_replay_call(sink)
    _observe(worker, request, [12, 13])
    engine_routes.by_request["engine-request"] = _routes(1, 2, 3)
    resumed, resumed_request, _ = _resume(sink, _cut(worker, "checkpoint-1"))
    # The right length, so the delta alignment keeps it, but two layers where
    # the restored prefix has one: only the merge can reject it.
    routes = (
        torch.zeros((5, 2, 1), dtype=torch.int16)
        if served_routes == "other_model_dims"
        else None
    )

    with caplog.at_level(logging.WARNING):
        staged = _finish_with_routes(resumed, resumed_request, [14], routes)

    assert staged.model_call_id == "c1-resumed"
    assert "routed_experts" not in (staged.extras or {})
    assert resumed._capture_calls == {}
    assert (
        "restored prefix does not fit"
        if served_routes == "other_model_dims"
        else "staged no routed_experts"
    ) in caplog.text


def test_router_replay_prefix_cut_after_the_final_token_resumes_with_routes(
    monkeypatch,
) -> None:
    """A restored prefix that is already finished never reaches the engine;
    its built output must still carry routes, or router replay fails the call."""
    # vLLM's output types, minus vLLM: CompletionOutput has routed_experts.
    monkeypatch.setitem(
        sys.modules,
        "vllm.outputs",
        SimpleNamespace(
            CompletionOutput=lambda **kwargs: SimpleNamespace(**kwargs),
            RequestOutput=lambda **kwargs: SimpleNamespace(**kwargs),
        ),
    )
    sink = _MemorySink()
    worker, request, engine_routes = _router_replay_call(sink)
    # The cut lands after vLLM's final token, before terminal staging.
    final_output = _cumulative_output([12, 13])
    final_output.outputs[0].finish_reason = "length"
    final_output.outputs[0].stop_reason = None
    final_output.prompt_routed_experts = _routes(1, 2)
    final_output.outputs[0].routed_experts = _routes(3)
    VllmAsyncGenerationWorkerImpl._observe_request_capture(
        worker, request, final_output
    )
    ack = _cut(worker, "checkpoint-1")
    # The final output carried the routes; the engine was not asked.
    assert engine_routes.reads == []
    assert ack.terminal_finish_reason == "length"

    resumed, resumed_request, resolved = _resume(sink, ack)
    assert resolved.routed_experts.flatten().tolist() == [1, 2, 3, -1]
    terminal = replace(
        _classify_restored_prefix_terminal(
            prompt_token_ids=[10, 11, 12, 13],
            prefix_token_count=2,
            requested_output_tokens=ack.effective_output_limit,
            model_max_tokens=4096,
            terminal_finish_reason="length",
        ),
        routed_experts=_restored_prefix_terminal_routes(
            resolved.routed_experts, prev_len=0
        ),
    )
    choice = SimpleNamespace(index=0, message=SimpleNamespace(routed_experts=None))
    attach_routed_experts_to_chat_response_choices(
        SimpleNamespace(choices=[choice]),
        _build_restored_prefix_terminal_output("request", terminal),
        device=torch.device("cpu"),
        routed_experts_dtype=torch.int16,
    )
    # The restored prefix's routes, and no route for the final token, as an
    # uninterrupted call's response has.
    assert decode_routed_experts(choice.message.routed_experts).flatten().tolist() == [
        1,
        2,
        3,
        -1,
    ]

    staged = _finish_with_routes(
        resumed, resumed_request, [], choice.message.routed_experts
    )
    assert _staged_routes(staged) == [1, 2, 3, -1]


def test_router_replay_prefix_resume_keeps_the_parent_boundary_route() -> None:
    """A multi-turn call's cut chunks stage one route per token; its resumed
    canonical row also carries the parent's last-token route, which the
    reassembler writes back over the parent's placeholder."""
    sink = _MemorySink()
    # A real sink checks every row's route length; only its TQ store is mocked.
    validating_sink = TQTokenSink(MagicMock(), staging_partition="staging")
    validating_sink._store = MagicMock()
    ng_capture = {
        "rollout_id": "r0",
        "model_call_id": "c2",
        "parent_call_id": "c1",
        "prev_len": 3,
        "mode": "token_in",
        "required_prefix_token_ids": [10, 11, 12],
        "parent_chain_hash": "1" * 64,
    }
    worker, request, engine_routes = _router_replay_call(
        sink, ng_capture=ng_capture, prompt=(10, 11, 12, 20, 21)
    )
    # Engine rows cover positions 0..5; the parent's last token (position 2)
    # is routed by this call's prefill, but the delta starts at position 3.
    _observe(worker, request, [22, 23])
    engine_routes.by_request["engine-request"] = _routes(1, 2, 3, 4, 5, 6)
    _cut(worker, "checkpoint-1", model_call_id="c2")
    _observe(worker, request, [22, 23, 24])
    engine_routes.by_request["engine-request"] = _routes(1, 2, 3, 4, 5, 6, 7)
    ack = _cut(worker, "checkpoint-2", model_call_id="c2")
    cut_records = [record for _, record in sink.generation_prefix_records]
    assert [record.prev_len for record in cut_records] == [3, 3]
    assert [_staged_routes(record) for record in cut_records] == [[4, 5, 6, -1], [7]]
    for sequence, record in enumerate(cut_records):
        # Raises if the row's route length does not match its tokens.
        validating_sink.stage_generation_prefix_batch(
            [record], checkpoint_id="checkpoint", chunk_sequences=[sequence]
        )

    resumed, resumed_request, resolved = _resume(
        sink, ack, source=ng_capture, prefix_token_ids=(10, 11, 12)
    )
    assert resolved.routed_experts.flatten().tolist() == [4, 5, 6, 7, -1]

    # Current-policy routes for positions 0..7 plus the unrouted final token.
    staged = _finish_with_routes(
        resumed,
        resumed_request,
        [25],
        _routes(90, 91, 92, 93, 94, 95, 96, 97, 0),
    )
    # Parent boundary (position 2) and the generated tail come from this
    # call; the restored prefix keeps the routes it was sampled under.
    assert _staged_routes(staged) == [92, 4, 5, 6, 7, 97, 0]
    assert validating_sink.stage(staged).ok


def test_request_capture_token_in_prev_len_chains():
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    request = _FakeRequest(
        ng_capture={
            "rollout_id": "r0",
            "model_call_id": "c2",
            "parent_call_id": "c1",
            "prev_len": 3,
            "mode": "token_in",
            "required_prefix_token_ids": [10, 11, 12],
            "parent_chain_hash": "1" * 64,
        },
        stream=False,
    )
    spliced_prompt = [10, 11, 12, 20, 21]  # exact prefix + fresh suffix
    VllmAsyncGenerationWorkerImpl._begin_request_capture(
        worker, request, spliced_prompt
    )
    content = VllmAsyncGenerationWorkerImpl._finish_request_capture(
        worker, request, _served_content([22], [-0.5])
    )
    coords = content["ng_commit_coords"]
    assert coords["parent_call_id"] == "c1"
    assert (coords["delta_len"], coords["cum_len"]) == (3, 6)
    assert sink.records[0].token_ids_delta == [20, 21, 22]


def _staging_chain_request(prev_len: int = 3) -> _FakeRequest:
    return _FakeRequest(
        ng_capture={
            "rollout_id": "r0",
            "model_call_id": "c3",
            "parent_call_id": "c2",
            "prev_len": prev_len,
            "mode": "token_in",
            "staging_chain": ["r0/c1", "r0/c2"],
            "parent_chain_hash": "2" * 64,
        },
        stream=False,
    )


def test_staging_chain_prefix_flows_through_adapter_and_begin_call():
    """The resolved prefix reaches both the engine request and capture admission."""
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    source = _MemoryPrefixSource({"r0/c1": [10, 11], "r0/c2": [12]})
    worker._chain_prefix.install(source)
    request = _staging_chain_request()
    context_before = dict(request.ng_capture)

    admission = worker._capture_admission(request)
    prefix = worker._resolve_admission_prefix(admission)
    worker._enter_request_prefix(request, prefix)
    VllmAsyncGenerationWorkerImpl._begin_request_capture(
        worker,
        request,
        prefix + [20],
        admission=admission,
        prefix_token_ids=prefix,
    )

    assert prefix == [10, 11, 12]
    assert source.calls == [["r0/c1", "r0/c2"]]
    # Never patched back into the wire context.
    assert request.ng_capture == context_before
    assert admission.required_prefix_token_ids == []
    # enter_prefix is the production writer of the request field.
    assert request.required_prefix_token_ids == prefix
    state = worker._capture_calls[id(request)]
    assert state.call.prefix_token_ids == prefix
    assert state.prompt_token_ids == [10, 11, 12, 20]


def test_inline_prefix_admission_resolves_without_a_fetch():
    worker = _worker_with_capture(_MemorySink())
    source = _MemoryPrefixSource({})
    worker._chain_prefix.install(source)
    request = _FakeRequest(
        ng_capture={
            "rollout_id": "r0",
            "model_call_id": "c2",
            "parent_call_id": "c1",
            "prev_len": 2,
            "mode": "token_in",
            "required_prefix_token_ids": [10, 11],
            "parent_chain_hash": "1" * 64,
        },
        stream=False,
    )
    admission = worker._capture_admission(request)
    assert worker._resolve_admission_prefix(admission) == [10, 11]
    assert source.calls == []
    text_root = worker._capture_admission(
        _FakeRequest(
            ng_capture={"rollout_id": "r0", "model_call_id": "c1", "mode": "text"}
        )
    )
    assert worker._resolve_admission_prefix(text_root) == []


def test_staging_chain_cache_fetches_only_uncached_suffix():
    worker = _worker_with_capture(_MemorySink())
    source = _MemoryPrefixSource({"r0/c1": [10, 11], "r0/c2": [12]})
    worker._chain_prefix.install(source)

    first = VllmAsyncGenerationWorkerImpl._fetch_chain_prefix(worker, ["r0/c1"])
    second = VllmAsyncGenerationWorkerImpl._fetch_chain_prefix(
        worker, ["r0/c1", "r0/c2"]
    )

    assert first == [10, 11]
    assert second == [10, 11, 12]
    assert source.calls == [["r0/c1"], ["r0/c2"]]


def test_staging_chain_prefix_length_mismatch_is_rejected_by_begin_call():
    """Gym's begin_call rejects a fetched prefix whose length is not prev_len."""
    worker = _worker_with_capture(_MemorySink())
    worker._chain_prefix.install(_MemoryPrefixSource({"r0/c1": [10, 11], "r0/c2": []}))
    request = _staging_chain_request(prev_len=3)
    context_before = dict(request.ng_capture)

    admission = worker._capture_admission(request)
    prefix = worker._resolve_admission_prefix(admission)
    assert prefix == [10, 11]
    with pytest.raises(CaptureError, match="does not equal prev_len 3"):
        VllmAsyncGenerationWorkerImpl._begin_request_capture(
            worker, request, prefix + [20], admission=admission, prefix_token_ids=prefix
        )

    assert request.ng_capture == context_before
    assert worker._capture_calls == {}


def test_staging_chain_admission_requires_the_resolved_prefix_keyword():
    worker = _worker_with_capture(_MemorySink())
    request = _staging_chain_request()

    with pytest.raises(
        CaptureError,
        match="requires the caller to pass the resolved prefix_token_ids",
    ):
        VllmAsyncGenerationWorkerImpl._begin_request_capture(
            worker, request, [10, 11, 12, 20]
        )

    assert worker._capture_calls == {}


@pytest.mark.parametrize("capture_enabled", [False, True])
def test_request_capture_is_a_noop_without_context_or_capture(
    capture_enabled: bool,
) -> None:
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    plain = _FakeRequest(stream=False)  # no ng_capture attribute
    if not capture_enabled:
        worker.token_capture = None
        plain.ng_capture = {"rollout_id": "r0", "model_call_id": "c1", "mode": "text"}
    VllmAsyncGenerationWorkerImpl._begin_request_capture(worker, plain, [1, 2])
    content = _served_content([3], [-0.1])
    content["choices"][0]["message"].update(
        prompt_token_ids=[1, 2],
        generation_token_ids=[3],
        generation_log_probs=[-0.1],
        routed_experts=[[[0]]] * 3,
    )
    original = deepcopy(content)
    out = VllmAsyncGenerationWorkerImpl._finish_request_capture(worker, plain, content)
    assert out == original
    assert worker._capture_calls == {}
    assert sink.records == []


def _cut_active_call(worker, sink, model_call_id: str = "c1"):
    """Begin one captured call and stage its prefix at a generation cut."""
    request = _FakeRequest(
        ng_capture={
            "rollout_id": "r0",
            "model_call_id": model_call_id,
            "parent_call_id": None,
            "prev_len": 0,
            "mode": "text",
        },
        stream=False,
    )
    VllmAsyncGenerationWorkerImpl._begin_request_capture(worker, request, [10, 11])
    state = worker._capture_calls[id(request)]
    with state.lock:
        state.effective_output_limit = 8
        state.observe([20, 21], [-0.1, -0.2])
    inventory = GenerationCutInventory.build(
        checkpoint_id="checkpoint-1",
        server_name="policy_model",
        active_prefixes=[
            GenerationCutPrefix(
                ticket_id="ticket-1",
                rollout_id="r0",
                attempt=0,
                model_call_id=model_call_id,
                admitted_at=1.0,
            )
        ],
    )
    worker._checkpoint_generation_cut(inventory)
    assert state.generation_cut_staging_keys == sink.generation_prefix_keys
    return request


def _drain_cleanup(worker) -> None:
    worker._generation_cut_cleanup_executor.submit(lambda: None).result(timeout=5)


def test_aborted_call_clears_the_rows_its_cuts_staged():
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    worker._generation_prefix_cuts_enabled = True
    request = _cut_active_call(worker, sink)

    VllmAsyncGenerationWorkerImpl._abort_request_capture(
        worker, request, reason="client_disconnected"
    )
    _drain_cleanup(worker)

    assert sink.generation_prefix_keys
    assert sink.cleared_generation_prefix_keys == sink.generation_prefix_keys


def test_aborted_call_clears_its_rows_only_after_the_snapshot_fence():
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    worker._generation_prefix_cuts_enabled = True
    request = _cut_active_call(worker, sink)
    gate = worker._token_capture_snapshot_gate
    gate.close_and_wait(gate.begin_epoch())

    # The abort itself never waits for the fence.
    assert VllmAsyncGenerationWorkerImpl._abort_request_capture(
        worker, request, reason="client_disconnected", blocking=False
    )
    time.sleep(0.2)
    assert sink.cleared_generation_prefix_keys == []

    gate.reopen()
    _drain_cleanup(worker)
    assert sink.cleared_generation_prefix_keys == sink.generation_prefix_keys


class _FailingTerminalSink(_MemorySink):
    def stage(
        self, record: StagedCallRecord, *, attachments: dict | None = None
    ) -> StageResult:
        return StageResult(ok=False, error="tq unavailable")


def test_call_whose_terminal_staging_fails_clears_its_cut_rows():
    sink = _FailingTerminalSink()
    worker = _worker_with_capture(sink)
    worker._generation_prefix_cuts_enabled = True
    request = _cut_active_call(worker, sink)

    content = VllmAsyncGenerationWorkerImpl._finish_request_capture(
        worker,
        request,
        _served_content([20, 21, 22], [-0.1, -0.2, -0.3]),
    )

    assert content["ng_commit_coords"]["disposition"] == "capture_failed"
    assert sink.cleared_generation_prefix_keys == sink.generation_prefix_keys


def test_request_capture_abort_fails_the_call_and_drains_state():
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    request = _FakeRequest(
        ng_capture={
            "rollout_id": "r0",
            "model_call_id": "c1",
            "parent_call_id": None,
            "prev_len": 0,
            "mode": "text",
        },
        stream=False,
    )
    VllmAsyncGenerationWorkerImpl._begin_request_capture(worker, request, [1, 2])
    VllmAsyncGenerationWorkerImpl._abort_request_capture(
        worker, request, reason="engine_error"
    )
    assert worker._capture_calls == {}
    assert sink.records == []
    # A call that was never cut has no chunk rows to clear.
    _drain_cleanup(worker)
    assert sink.cleared_generation_prefix_keys == []
    # A late finish after abort is a no-op (state already drained).
    out = VllmAsyncGenerationWorkerImpl._finish_request_capture(
        worker, request, _served_content([3], [-0.1])
    )
    assert "ng_commit_coords" not in out


@pytest.mark.vllm
@pytest.mark.parametrize("prefix_cuts", [False, True])
@pytest.mark.parametrize("pruning_rate", [0.0, 0.5])
def test_omni_capture_setup_rejects_video_pruning(
    monkeypatch, pruning_rate, prefix_cuts
):
    # Both dependency markers select the combined vLLM + Gym lane.
    pytest.importorskip("vllm")
    from vllm.model_executor.models.nano_nemotron_vl import NanoNemotronVLProcessingInfo

    info = object.__new__(NanoNemotronVLProcessingInfo)
    monkeypatch.setattr(NanoNemotronVLProcessingInfo, "is_dynamic_tiler", True)
    monkeypatch.setattr(
        NanoNemotronVLProcessingInfo,
        "get_video_pruning_rate",
        lambda self: pruning_rate,
    )
    monkeypatch.setattr(
        NanoNemotronVLProcessingInfo,
        "get_hf_processor",
        lambda self: SimpleNamespace(_img_context_token_ids=[18]),
    )
    monkeypatch.setattr(
        NanoNemotronVLProcessingInfo,
        "get_hf_config",
        lambda self: SimpleNamespace(patch_size=2),
    )
    worker = _fake_worker()
    worker.llm = SimpleNamespace(
        renderer=SimpleNamespace(
            get_mm_processor=lambda: SimpleNamespace(info=info),
            get_tokenizer=lambda: SimpleNamespace(decode=lambda ids: ""),
        ),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
    )
    monkeypatch.setattr(
        "nemo_rl.data_plane.build_data_plane_client",
        lambda dp_cfg, bootstrap, segment_size=None: MagicMock(),
    )
    kwargs = {
        "staging_partition": "staging",
        "capture_media": True,
        "generation_prefix_cleanup": GenerationPrefixCleanupConfig(),
        **_PREFIX_BATCH_LIMITS,
    }
    if prefix_cuts:
        kwargs.update(
            generation_prefix_cuts_enabled=True,
            generation_cut_control_token="secret",
        )
    if pruning_rate:
        with pytest.raises(ValueError, match="video token pruning"):
            asyncio.run(
                VllmAsyncGenerationWorkerImpl.setup_token_capture(worker, {}, **kwargs)
            )
    else:
        spec = asyncio.run(
            VllmAsyncGenerationWorkerImpl.setup_token_capture(worker, {}, **kwargs)
        )
        # The engine's model dtype and the HF patch size, as pinned on the sink.
        assert spec == MediaColumnSpec(pixel_dtype=torch.bfloat16, patch_size=2)
        assert worker._capture_image_token_id == 18


class _RecordingCleanup:
    def __init__(self, gate: _TokenCaptureSnapshotGate) -> None:
        self.gate = gate
        self.submitted: list[list[str]] = []

    def submit(self, keys):
        # Submitting inside the gate would hold a permit the fence waits for.
        assert self.gate._active == 0
        self.submitted.append(list(keys))


@pytest.mark.parametrize("failure", [False, True])
def test_background_cleanup_only_receives_acknowledged_prefix_keys(failure):
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    state = SimpleNamespace(
        lifecycle_lock=threading.Lock(),
        terminal_started=False,
        call=SimpleNamespace(model_call_id="call"),
    )
    worker._get_request_capture = lambda request: state
    prefix = "__generation_cut__/checkpoint/rollout/call/0"
    content = {"payload": "unchanged"}

    def finish(state, request, response):
        if failure:
            raise RuntimeError("terminal put failed")
        # This return represents a successfully acknowledged terminal stage.
        return response, (prefix,)

    worker._finish_request_capture_with_lifecycle_owned = finish
    cleanup = _RecordingCleanup(worker._token_capture_snapshot_gate)
    worker._prefix_cleanup = cleanup
    if failure:
        with pytest.raises(RuntimeError, match="terminal put failed"):
            VllmAsyncGenerationWorkerImpl._finish_request_capture(
                worker, object(), content
            )
    else:
        out = VllmAsyncGenerationWorkerImpl._finish_request_capture(
            worker, object(), content
        )
        assert out is content
    # Only an acknowledged row's keys are queued, and nothing is cleared inline.
    assert cleanup.submitted == ([] if failure else [[prefix]])
    assert sink.cleared_generation_prefix_keys == []


@pytest.mark.parametrize("background", [False, True])
def test_obsolete_keys_never_include_a_restored_canonical_row(background):
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    cleanup = _RecordingCleanup(worker._token_capture_snapshot_gate)
    worker._prefix_cleanup = cleanup if background else None
    request = _FakeRequest(
        ng_capture={"rollout_id": "r0", "model_call_id": "c1", "mode": "text"},
        stream=False,
    )
    VllmAsyncGenerationWorkerImpl._begin_request_capture(worker, request, [10, 11])
    cut_key = "__generation_cut__/checkpoint/r0/c0/0"
    # A restored terminal-completion cut names the canonical row it replaced.
    canonical_key = "r0/c0"
    worker._capture_calls[id(request)].generation_cut_staging_keys = [
        cut_key,
        canonical_key,
    ]
    VllmAsyncGenerationWorkerImpl._finish_request_capture(
        worker, request, _served_content([12], [-0.1])
    )
    # Both paths delete the same rows: the cut chunk, never the canonical row.
    if background:
        assert cleanup.submitted == [[cut_key]]
        assert sink.cleared_generation_prefix_keys == []
    else:
        assert cleanup.submitted == []
        assert sink.cleared_generation_prefix_keys == [cut_key]


def test_background_cleanup_completion_does_not_wait_for_delete():
    worker = _fence_worker()
    entered, release = threading.Event(), threading.Event()
    keys = []

    def clear(batch):
        keys.extend(batch)
        entered.set()
        assert release.wait(5)

    cleanup = BackgroundPrefixCleanup(
        clear,
        worker._token_capture_snapshot_gate,
        config=GenerationPrefixCleanupConfig(batch_size=1),
    )
    worker._prefix_cleanup = cleanup
    worker._capture_sink = SimpleNamespace(clear=clear)
    worker._get_request_capture = lambda request: SimpleNamespace(
        lifecycle_lock=threading.Lock(),
        terminal_started=False,
        call=SimpleNamespace(model_call_id="call"),
    )
    first = "__generation_cut__/checkpoint/rollout/call/0"
    second = "__generation_cut__/checkpoint/rollout/call/1"
    worker._finish_request_capture_with_lifecycle_owned = (
        lambda state, request, content: (content, (first,))
    )
    try:
        assert worker._finish_request_capture(object(), {"done": True}) == {
            "done": True
        }
        # Terminal completion returned while its delete is still running.
        assert entered.wait(5)

        async def checkpoint():
            task = asyncio.create_task(
                worker.begin_token_capture_snapshot_fence_async()
            )
            await asyncio.sleep(0.1)
            # The fence waits for the running CLEAR.
            assert not task.done()
            entered.clear()
            release.set()
            assert await asyncio.wait_for(task, timeout=5)
            # No CLEAR starts while the fence holds the gate closed.
            cleanup.submit([second])
            assert not await asyncio.to_thread(entered.wait, 0.2)
            assert await worker.end_token_capture_snapshot_fence_async()
            assert await asyncio.to_thread(entered.wait, 5)

        asyncio.run(checkpoint())
    finally:
        release.set()
        _shutdown_fence_worker(worker)
    assert keys == [first, second]


def _background_cleanup_worker(
    clear, *, config: GenerationPrefixCleanupConfig
) -> tuple[VllmAsyncGenerationWorkerImpl, list[str]]:
    """A fence-capable worker whose completions each retire one new cut key."""
    worker = _fence_worker()
    worker._prefix_cleanup = BackgroundPrefixCleanup(
        clear, worker._token_capture_snapshot_gate, config=config
    )
    worker._capture_sink = SimpleNamespace(clear=clear)
    worker._get_request_capture = lambda request: SimpleNamespace(
        lifecycle_lock=threading.Lock(),
        terminal_started=False,
        call=SimpleNamespace(model_call_id="call"),
    )
    keys = [f"__generation_cut__/checkpoint/rollout/call/{index}" for index in range(8)]
    pending = iter(keys)
    worker._finish_request_capture_with_lifecycle_owned = (
        lambda state, request, content: (content, (next(pending),))
    )
    return worker, keys


def test_failed_background_clear_does_not_fail_later_completions(caplog):
    calls, failed, cleared = [], threading.Event(), threading.Event()

    def clear(keys):
        calls.append(keys)
        if len(calls) == 1:
            failed.set()
            raise RuntimeError("transport failed")
        cleared.set()

    worker, keys = _background_cleanup_worker(
        clear, config=GenerationPrefixCleanupConfig(batch_size=1)
    )
    try:
        with caplog.at_level(logging.ERROR):
            assert worker._finish_request_capture(object(), {"done": 1}) == {"done": 1}
            assert failed.wait(5)
            # A completion after the failed CLEAR still succeeds and is cleaned.
            assert worker._finish_request_capture(object(), {"done": 2}) == {"done": 2}
            assert cleared.wait(5)
        assert calls == [[keys[0]], [keys[1]]]
        assert "background prefix cleanup failed for 1 keys" in caplog.text

        async def checkpoint():
            # Nor does the failure fail or stall the next checkpoint fence.
            assert await asyncio.wait_for(
                worker.begin_token_capture_snapshot_fence_async(), timeout=5
            )
            assert await worker.end_token_capture_snapshot_fence_async()

        asyncio.run(checkpoint())
    finally:
        _shutdown_fence_worker(worker)


def test_full_cleanup_queue_cannot_block_the_snapshot_fence():
    calls, entered, release = [], threading.Event(), threading.Event()

    def clear(keys):
        calls.append(keys)
        entered.set()
        assert release.wait(10)

    worker, keys = _background_cleanup_worker(
        clear,
        config=GenerationPrefixCleanupConfig(
            batch_size=1, max_pending=1, wait_seconds=0
        ),
    )
    producers = [
        threading.Thread(
            target=worker._finish_request_capture, args=(object(), {"done": index})
        )
        for index in (3, 4)
    ]
    try:
        # The cleanup thread holds the first CLEAR and the second call fills
        # the queue, so the next two completions block on a full queue.
        worker._finish_request_capture(object(), {"done": 1})
        assert entered.wait(5)
        worker._finish_request_capture(object(), {"done": 2})
        for producer in producers:
            producer.start()
        for producer in producers:
            producer.join(0.2)
            assert producer.is_alive()

        async def checkpoint():
            task = asyncio.create_task(
                worker.begin_token_capture_snapshot_fence_async()
            )
            await asyncio.sleep(0.1)
            # The fence waits for the running CLEAR...
            assert not task.done()
            release.set()
            # ...and then completes even though a producer is still blocked
            # on the full queue: blocked producers hold no gate permit.
            assert await asyncio.wait_for(task, timeout=5)
            assert any(producer.is_alive() for producer in producers)
            assert await worker.end_token_capture_snapshot_fence_async()

        asyncio.run(checkpoint())
        for producer in producers:
            producer.join(5)
            assert not producer.is_alive()
        deadline = time.monotonic() + 5
        while len(calls) < 4 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert sorted(key for batch in calls for key in batch) == sorted(keys[:4])
    finally:
        release.set()
        _shutdown_fence_worker(worker)


def test_background_cleanup_setup_rejects_nonprefix_capture():
    with pytest.raises(ValueError, match="requires generation-prefix cuts"):
        asyncio.run(
            VllmAsyncGenerationWorkerImpl.setup_token_capture(
                _fake_worker(),
                dp_cfg={},
                staging_partition="rollout_staging",
                generation_prefix_cleanup=GenerationPrefixCleanupConfig(enabled=True),
                **_PREFIX_BATCH_LIMITS,
            )
        )


def test_generation_setup_forwards_background_cleanup_config(monkeypatch):
    gen = _generation_with_mock_group()
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_generation.ray.get",
        lambda futures: futures,
    )
    config = GenerationPrefixCleanupConfig(enabled=True, batch_size=8, max_pending=16)
    gen.setup_token_capture(
        {},
        "rollout_staging",
        generation_prefix_cuts_enabled=True,
        generation_cut_control_token="control",
        generation_prefix_cleanup=config,
        **_PREFIX_BATCH_LIMITS,
    )
    assert (
        gen.worker_group.run_all_workers_single_data.call_args.kwargs[
            "generation_prefix_cleanup"
        ]
        == config
    )
