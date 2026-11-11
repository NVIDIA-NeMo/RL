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
import logging
import threading
import time
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

nemo_gym = pytest.importorskip("nemo_gym.token_id_capture.staging")

from nemo_gym._checkpoint.generation_cut import (  # noqa: E402
    GENERATION_CUT_ROUTE,
    GenerationCutInventory,
    GenerationCutPrefix,
)
from nemo_gym.token_id_capture.staging.capture import (  # noqa: E402
    CaptureError,
    RolloutTokenCapture,
)
from nemo_gym.token_id_capture.staging.records import (  # noqa: E402
    CaptureAdmission,
    StagedCallRecord,
    StageResult,
)

from nemo_rl.data.captured_media import MediaColumnSpec  # noqa: E402
from nemo_rl.data_plane.tq_token_sink import (  # noqa: E402
    StagedPrefix,
    TQTokenSink,
)
from nemo_rl.models.generation.prefix_recovery import (  # noqa: E402
    NoPrefixRecovery,
    PrefixRecovery,
    PrefixRecoveryParts,
    ProgressRead,
)
from nemo_rl.models.generation.prefix_recovery.restore import (  # noqa: E402
    _classify_restored_prefix_terminal,
    _count_restored_prefix_as_completion,
    _generation_cut_blockers,
    _remaining_generation_limits_after_prefix,
    _RestoredPrefixTerminal,
    restore_reply_text,
)
from nemo_rl.models.generation.prefix_recovery.state import (  # noqa: E402
    _FrozenCut,
    _RequestCaptureState,
)
from nemo_rl.models.generation.vllm.generation_cut_route import (  # noqa: E402
    GENERATION_CUT_ROUTE as RL_GENERATION_CUT_ROUTE,
)
from nemo_rl.models.generation.vllm.progress_reader import (  # noqa: E402
    VllmProgressReader,
)
from nemo_rl.models.generation.vllm.utils import (  # noqa: E402
    extract_generation_logprobs,
)
from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration  # noqa: E402
from nemo_rl.models.generation.vllm.vllm_worker_async import (  # noqa: E402
    _CAPTURED_MODEL_CALL,
    VllmAsyncGenerationWorkerImpl,
    _AsyncLLMHTTPClient,
)
from nemo_rl.utils.routed_experts_codec import (  # noqa: E402
    routed_experts_tensor_metadata,
)

pytestmark = pytest.mark.nemo_gym


class _MemorySink:
    def __init__(self) -> None:
        self.records: list[StagedCallRecord] = []
        self.attachments: list[dict[str, torch.Tensor] | None] = []
        self.generation_prefix_records: list[tuple[str, StagedCallRecord]] = []
        self.generation_prefix_keys: list[str] = []
        self.cleared_generation_prefix_keys: list[str] = []

    def stage(
        self,
        record: StagedCallRecord,
        *,
        attachments: dict[str, torch.Tensor] | None = None,
    ) -> StageResult:
        self.records.append(record)
        self.attachments.append(attachments)
        return StageResult(ok=True, staging_key=record.staging_key)

    def stage_generation_prefix(
        self,
        record: StagedCallRecord,
        *,
        checkpoint_id: str,
        chunk_sequence: int,
        attachments: dict | None = None,
    ) -> StageResult:
        assert attachments is None
        self.generation_prefix_records.append((checkpoint_id, record))
        key = (
            f"__generation_cut__/{checkpoint_id}/{record.capture_key}/"
            f"{record.model_call_id}/{chunk_sequence}"
        )
        self.generation_prefix_keys.append(key)
        return StageResult(ok=True, staging_key=key)

    def clear(self, staging_keys: list[str]) -> None:
        self.cleared_generation_prefix_keys.extend(staging_keys)


class _ByteTokenizer:
    """Byte-level stand-in: each token ID is one UTF-8 byte of the output."""

    def decode(self, token_ids: list[int], **kwargs) -> str:
        return bytes(token_ids).decode("utf-8", errors="replace")


class _NoCallsInVllm:
    """A progress reader none of whose captured calls has reached the engine."""

    def read(self, calls, *, timeout_s):
        return ProgressRead({}, 0.0, 0.0)

    def logprobs(self, token_ids, logprobs, *, start):
        return extract_generation_logprobs(token_ids, logprobs, start=start)


def _fake_worker(*, is_model_owner: bool = True) -> VllmAsyncGenerationWorkerImpl:
    """A real worker's token-capture state, minus the engine."""
    worker = object.__new__(VllmAsyncGenerationWorkerImpl)
    worker._init_token_capture_state()
    worker.is_model_owner = is_model_owner
    worker._http_engine_client = None
    return worker


def _cuts(
    worker,
    *,
    staging_source=None,
    progress_reader=None,
    tokenizer=None,
) -> PrefixRecovery:
    """The worker's prefix recovery, turning generation-prefix cuts on if off.

    Passing a part rebuilds it with that part; the rest are the test worker's.
    """
    recovery = worker._prefix_recovery
    if isinstance(recovery, PrefixRecovery) and (
        staging_source is progress_reader is tokenizer is None
    ):
        return recovery
    recovery = PrefixRecovery(
        PrefixRecoveryParts(
            capture=worker.token_capture,
            sink=worker._capture_sink,
            staging_source=staging_source or worker._staging_source,
            registry=worker._capture_registry,
            tokenizer=tokenizer or _ByteTokenizer(),
            weight_version=lambda: worker._rollout_weight_version,
            control_token="secret",
            control_timeout_s=None,
            progress_reader=progress_reader or _NoCallsInVllm(),
        )
    )
    worker._prefix_recovery = recovery
    return recovery


def test_generation_cut_route_matches_gym():
    """The worker serves cuts where Gym sends them; a drifted route 404s every cut."""
    assert RL_GENERATION_CUT_ROUTE == GENERATION_CUT_ROUTE


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
    worker = _fake_worker()

    installed = asyncio.run(
        VllmAsyncGenerationWorkerImpl.setup_token_capture(
            worker,
            dp_cfg={"backend": "simple"},
            staging_partition="rollout_staging",
            top_logprobs=4,
        )
    )

    assert installed is None  # text-only capture pins no media column
    # The capture client owns no Mooncake memory, so no save waits on vLLM.
    assert segment_sizes == [0]
    assert isinstance(worker.token_capture, RolloutTokenCapture)
    assert worker.token_capture.adapter is not None
    # The top-k count stays on the worker for the request path.
    assert worker._capture_top_logprobs == 4
    # The adapter is the vLLM one (prefix ids enter via the worker's field).
    payload = worker.token_capture.adapter.enter_prefix({}, [1, 2])
    assert payload["required_prefix_token_ids"] == [1, 2]


def test_setup_token_capture_skips_non_model_owners(monkeypatch):
    worker = _fake_worker(is_model_owner=False)
    installed = asyncio.run(
        VllmAsyncGenerationWorkerImpl.setup_token_capture(
            worker, dp_cfg={}, staging_partition="rollout_staging", top_logprobs=0
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
    worker = _fake_worker()
    asyncio.run(
        VllmAsyncGenerationWorkerImpl.setup_token_capture(
            worker, dp_cfg={}, staging_partition="rollout_staging", top_logprobs=0
        )
    )

    asyncio.run(VllmAsyncGenerationWorkerImpl.set_rollout_weight_version(worker, 4))
    first = worker.token_capture.begin_call(
        CaptureAdmission(capture_key="r", model_call_id="c1", mode="text")
    )
    asyncio.run(VllmAsyncGenerationWorkerImpl.set_rollout_weight_version(worker, 5))
    second = worker.token_capture.begin_call(
        CaptureAdmission(capture_key="r", model_call_id="c2", mode="text")
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
        top_logprobs=4,
        generation_cut_control_timeout_s=60.0,
    )
    gen.worker_group.run_all_workers_single_data.assert_called_once_with(
        "setup_token_capture",
        dp_cfg={"backend": "simple"},
        staging_partition="rollout_staging",
        capture_media=False,
        top_logprobs=4,
        generation_prefix_cuts_enabled=False,
        generation_cut_control_token=None,
        generation_cut_control_timeout_s=60.0,
        run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
    )


def test_generation_setup_token_capture_requires_async_engine():
    gen = _generation_with_mock_group(async_engine=False)
    with pytest.raises(AssertionError, match="async vLLM engine"):
        gen.setup_token_capture({}, "rollout_staging", top_logprobs=0)


def test_obsolete_generation_cut_row_flush_fans_out(monkeypatch):
    gen = _generation_with_mock_group()
    gen.worker_group.workers = [object()]
    gen.worker_group.run_all_workers_single_data.return_value = [True]
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_generation.ray.get",
        lambda futures, timeout=None: futures,
    )

    assert gen.flush_obsolete_generation_cut_rows(3, timeout_s=17.0)
    gen.worker_group.run_all_workers_single_data.assert_called_once_with(
        "flush_obsolete_generation_cut_rows_async",
        run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
        epoch=3,
    )


def test_generation_prefix_restore_counts_sum_over_workers(monkeypatch):
    gen = _generation_with_mock_group()
    gen.worker_group.workers = [object(), object()]
    gen.worker_group.run_all_workers_single_data.return_value = [
        {"restored_calls": 2, "restored_prefix_tokens": 50, "progress_reads": 1},
        {"restored_calls": 1, "restored_prefix_tokens": 7, "progress_reads": 0},
    ]
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_generation.ray.get",
        lambda futures, timeout=None: futures,
    )

    assert gen.get_generation_prefix_counts() == {
        "restored_calls": 3,
        "restored_prefix_tokens": 57,
        "progress_reads": 1,
    }
    gen.worker_group.run_all_workers_single_data.assert_called_once_with(
        "get_generation_prefix_counts",
        run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
    )


def test_prefix_capture_setup_requires_a_control_token(monkeypatch):
    # Prefix recovery with media or sampler top-k is refused at config/setup.
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
            )
        )


@pytest.mark.parametrize(
    ("max_tokens", "min_tokens", "prefix_token_count", "expected"),
    [
        (10, 6, 3, (7, 3)),
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


@pytest.mark.parametrize(
    ("requested", "model_max", "finish", "reason"),
    [
        # Room left on both limits: the call resumes decoding.
        (8, 8, None, None),
        (2, 8, None, "output_limit"),
        (None, 4, None, "model_limit"),
        (2, 4, None, "output_and_model_limit"),
        # A finish Gym recorded wins over the limits.
        (2, 4, "stop", "observed_stop"),
    ],
)
def test_restored_prefix_terminal_outcomes(requested, model_max, finish, reason):
    # Two prompt tokens, then the two-token restored prefix.
    terminal = _classify_restored_prefix_terminal(
        prompt_token_ids=[1, 2, 3, 4],
        prefix_token_count=2,
        requested_output_tokens=requested,
        model_max_tokens=model_max,
        terminal_finish_reason=finish,
    )

    if reason is None:
        assert terminal is None
        return
    assert terminal.reason == reason
    assert terminal.finish_reason == (finish or "length")
    assert terminal.original_prompt_token_count == 2


@pytest.mark.parametrize(
    ("prefix", "requested", "model_max", "match"),
    [
        (5, None, 8, "more generated tokens than the restored engine prompt"),
        (3, 2, 8, r"exceeds the restored request output budget \(2\)"),
        (2, None, 3, r"exceeds restored model capacity \(3\)"),
    ],
)
def test_restored_prefix_that_cannot_fit_is_refused(
    prefix, requested, model_max, match
):
    with pytest.raises(ValueError, match=match):
        _classify_restored_prefix_terminal(
            prompt_token_ids=[1, 2, 3, 4],
            prefix_token_count=prefix,
            requested_output_tokens=requested,
            model_max_tokens=model_max,
        )


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
        "nemo_rl.models.generation.prefix_recovery.restore._detokenize_restored_prefix",
        replay,
    )
    tokenizer = object()
    terminal = _stopped_terminal(stop_reason="</answer>")
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

    restore_reply_text(
        request,
        SimpleNamespace(outputs=[output]),
        prefix_token_ids=[12, 13],
        # The engine prompt ends with the spliced prefix.
        engine_prompt_token_ids=[10, 11, 12, 13],
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


def _captured_text_call(
    worker, model_call_id: str = "c1", prompt_token_ids: tuple[int, ...] = (10, 11)
) -> _FakeRequest:
    """Begin capturing a parentless text call of rollout ``r0``."""
    request = _FakeRequest(
        ng_capture={"rollout_id": "r0", "model_call_id": model_call_id, "mode": "text"},
        stream=False,
    )
    VllmAsyncGenerationWorkerImpl._begin_request_capture(
        worker, request, list(prompt_token_ids)
    )
    return request


def _live_call(
    worker,
    generated_token_ids: list[int],
    logprobs: list[float],
    *,
    model_call_id: str = "c1",
    prompt_token_ids: tuple[int, ...] = (10, 11),
):
    """Begin one captured call that has decoded ``generated_token_ids``."""
    request = _captured_text_call(worker, model_call_id, prompt_token_ids)
    state = worker._capture_registry.get(request)
    state.set_effective_output_limit(8)
    state.record(generated_token_ids, logprobs)
    return request, state


def test_finished_output_records_only_tokens_past_the_last_cut():
    worker = _worker_with_capture(_MemorySink())
    _cuts(worker)
    request = _captured_text_call(worker)
    state = worker._capture_registry.get(request)
    state.record([12], [-0.1], start=0)  # read earlier by a cut
    final = SimpleNamespace(
        outputs=[
            SimpleNamespace(
                token_ids=[12, 13, 14],
                logprobs=[
                    {12: SimpleNamespace(logprob=-9.0)},
                    {13: SimpleNamespace(logprob=-0.2)},
                    {14: SimpleNamespace(logprob=-0.3)},
                ],
                finish_reason="stop",
            )
        ]
    )

    _cuts(worker).observe(request, final)

    progress = state.snapshot()
    assert progress.observation_error is None
    assert progress.generated_token_ids == [12, 13, 14]
    assert progress.generated_logprobs == [-0.1, -0.2, -0.3]
    assert progress.terminal_finish_reason == "stop"


def test_tokens_read_by_both_a_cut_and_the_finished_output_are_recorded_once():
    state = _RequestCaptureState(None, [1])
    state.record([12, 13], [-0.1, -0.2], start=0)
    state.record([12, 13, 14], [-0.1, -0.2, -0.3], start=0)
    state.record([14], [-0.3], start=2)

    progress = state.snapshot()
    assert progress.generated_token_ids == [12, 13, 14]
    assert progress.generated_logprobs == [-0.1, -0.2, -0.3]
    assert progress.observation_error is None


def _engine_request(
    token_ids: list[int],
    logprobs: list[float],
    *,
    request_id: str = "chatcmpl-1",
    finish_reason: str | None = None,
) -> SimpleNamespace:
    """The RequestState fields a progress read touches."""
    final = SimpleNamespace(
        outputs=[SimpleNamespace(finish_reason=finish_reason, stop_reason=None)]
    )
    return SimpleNamespace(
        request_id=f"{request_id}-internal",
        external_req_id=request_id,
        queue=SimpleNamespace(output=final),
        detokenizer=SimpleNamespace(output_token_ids=list(token_ids)),
        logprobs_processor=SimpleNamespace(
            logprobs=[
                {token_id: SimpleNamespace(logprob=logprob)}
                for token_id, logprob in zip(token_ids, logprobs, strict=True)
            ]
        ),
    )


class _OutputProcessor:
    """vLLM 0.29's OutputProcessor surface: live requests and the two hooks."""

    def __init__(self, *requests: SimpleNamespace) -> None:
        self.request_states = {r.request_id: r for r in requests}
        self.external_req_ids = {r.external_req_id: [r.request_id] for r in requests}
        self.processed_under_lock: list[bool] = []

    def process_outputs(self, reader) -> None:
        self.processed_under_lock.append(reader._lock._is_owned())

    def _finish_request(self, req_state) -> None:
        self.request_states.pop(req_state.request_id)
        del self.external_req_ids[req_state.external_req_id]


def _progress_client(processor) -> _AsyncLLMHTTPClient:
    engine = SimpleNamespace(
        output_processor=processor,
        model_config=None,
        renderer=None,
        input_processor=None,
        vllm_config=None,
    )
    return _AsyncLLMHTTPClient(engine, MagicMock(), flat_logprobs=False)


def _progress_reader(
    monkeypatch, client: _AsyncLLMHTTPClient, *, vllm_version: str = "0.29.0"
) -> VllmProgressReader:
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.progress_reader.importlib.metadata.version",
        lambda name: vllm_version,
    )
    return VllmProgressReader(client)


def test_vllm_processes_outputs_under_the_progress_lock(monkeypatch):
    processor = _OutputProcessor()
    reader = _progress_reader(monkeypatch, _progress_client(processor))

    processor.process_outputs(reader)

    assert processor.processed_under_lock == [True]


def test_finished_request_stays_readable_until_its_consumer_is_done(monkeypatch):
    request = _engine_request([20, 21], [-0.1, -0.2], finish_reason="stop")
    unserved = _engine_request([30], [-0.3], request_id="offline-1")
    processor = _OutputProcessor(request, unserved)
    client = _progress_client(processor)
    reader = _progress_reader(monkeypatch, client)
    client._captured_requests["chatcmpl-1"] = "c1"

    processor._finish_request(request)
    processor._finish_request(unserved)
    read = reader.read([("c1", 1), ("c2", 0)], timeout_s=None)

    finished = read.progress["c1"]
    assert (finished.token_ids, finished.finish_reason) == ([21], "stop")
    # Only captured requests are kept after they finish.
    assert "c2" not in read.progress
    assert "offline-1" not in reader._finished_requests


def test_finished_request_is_released_once_its_consumer_has_the_output(monkeypatch):
    request = _engine_request([20], [-0.1], finish_reason="length")
    processor = _OutputProcessor(request)
    client = _progress_client(processor)
    reader = _progress_reader(monkeypatch, client)

    async def consume() -> list:
        client._engine_loop = asyncio.get_running_loop()
        seen = []

        async def engine_generate(*args, **kwargs):
            processor._finish_request(request)
            yield "final output"

        client._engine_client.generate = engine_generate
        _CAPTURED_MODEL_CALL.set("c1")
        async for output in client._generate("prompt", None, "chatcmpl-1", {}):
            # The consumer records the output while vLLM's state is still kept.
            seen.append(reader._finished_requests["chatcmpl-1"].finish_reason)
        return seen

    assert asyncio.run(consume()) == ["length"]
    assert reader._finished_requests == {}
    assert client._captured_requests == {}


def test_progress_reads_refuse_an_unchecked_vllm(monkeypatch):
    with pytest.raises(RuntimeError, match="checked on vLLM 0.29.x"):
        _progress_reader(
            monkeypatch, _progress_client(_OutputProcessor()), vllm_version="0.30.1"
        )


def test_progress_reads_refuse_a_vllm_without_the_wrapped_hooks(monkeypatch):
    class Processor:  # no process_outputs / _finish_request
        pass

    with pytest.raises(RuntimeError, match="does not have"):
        _progress_reader(monkeypatch, _progress_client(Processor()))


def test_progress_reads_wrap_an_output_processor_once(monkeypatch):
    processor = _OutputProcessor()
    reader = _progress_reader(monkeypatch, _progress_client(processor))

    with pytest.raises(RuntimeError, match="already wraps"):
        _progress_reader(monkeypatch, _progress_client(processor))
    processor.process_outputs(reader)
    assert processor.processed_under_lock == [True]


def test_cut_reads_each_call_progress_from_vllm_request_state(monkeypatch):
    worker = _worker_with_capture(_MemorySink())
    decoding, resumed, finished, unsubmitted = (
        worker._capture_registry.get(_captured_text_call(worker, call_id))
        for call_id in ("c1", "c2", "c3", "c4")
    )
    resumed.record([30], [-0.3])  # recorded by an earlier cut
    done = _engine_request(
        [40, 41], [-0.5, -0.6], request_id="chatcmpl-3", finish_reason="stop"
    )
    processor = _OutputProcessor(
        _engine_request([20, 21], [-0.1, -0.2]),
        # The earlier cut's position is not read again.
        _engine_request([30, 31], [-9.0, -0.4], request_id="chatcmpl-2"),
        done,
    )
    client = _progress_client(processor)
    recovery = _cuts(worker, progress_reader=_progress_reader(monkeypatch, client))
    # c4 has not reached vLLM yet, so the HTTP client has no request for it.
    client._captured_requests.update(
        {"chatcmpl-1": "c1", "chatcmpl-2": "c2", "chatcmpl-3": "c3"}
    )
    processor._finish_request(done)

    recovery._record_progress([decoding, resumed, finished, unsubmitted], deadline=None)

    assert (
        decoding.snapshot().generated_token_ids,
        decoding.snapshot().generated_logprobs,
    ) == (
        [20, 21],
        [-0.1, -0.2],
    )
    assert (
        resumed.snapshot().generated_token_ids,
        resumed.snapshot().generated_logprobs,
    ) == (
        [30, 31],
        [-0.3, -0.4],
    )
    # Finished but not yet taken by its consumer: read whole, with its reason.
    assert finished.snapshot().generated_token_ids == [40, 41]
    assert finished.snapshot().terminal_finish_reason == "stop"
    # Never submitted: it keeps what it recorded.
    assert unsubmitted.snapshot().generated_token_ids == []
    assert all(
        state.snapshot().observation_error is None
        for state in (decoding, resumed, finished, unsubmitted)
    )
    counts = worker.get_generation_prefix_counts()
    assert (counts["progress_reads"], counts["progress_read_calls"]) == (1, 4)


def test_cut_past_its_deadline_skips_reading_vllm_progress(monkeypatch, caplog):
    worker = _worker_with_capture(_MemorySink())
    state = worker._capture_registry.get(_captured_text_call(worker))
    client = _progress_client(_OutputProcessor(_engine_request([20], [-0.1])))
    reader = _progress_reader(monkeypatch, client)
    client._captured_requests["chatcmpl-1"] = "c1"
    recovery = _cuts(worker, progress_reader=reader)
    vllm_busy, release = threading.Event(), threading.Event()

    def process_a_long_chunk():
        with reader._lock:
            vllm_busy.set()
            release.wait(timeout=5)

    vllm = threading.Thread(target=process_a_long_chunk)
    vllm.start()
    vllm_busy.wait(timeout=5)
    try:
        with caplog.at_level(logging.WARNING):
            recovery._record_progress([state], deadline=time.monotonic())
    finally:
        release.set()
        vllm.join(timeout=5)

    assert state.snapshot().generated_token_ids == []
    assert "could not read the engine's progress before its deadline" in caplog.text


def test_terminal_token_without_a_logprob_makes_its_call_uncuttable():
    worker = _worker_with_capture(_MemorySink())
    _cuts(worker)
    request = _captured_text_call(worker)
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

    _cuts(worker).observe(request, final)

    state = worker._capture_registry.get(request)
    assert "token_count=2, logprob_count=1" in state.snapshot().observation_error
    assert state.snapshot().terminal_finish_reason is None


def test_generation_cut_stages_only_tokens_past_the_last_sealed_cut():
    state = _RequestCaptureState(None, [1], effective_output_limit=8)
    state.record([10, 11], [-0.1, -0.2])

    with state.lifecycle() as owned:
        _, first = owned.freeze("c1")
        assert first == _FrozenCut(
            "active/c1/0", 0, [10, 11], [-0.1, -0.2], [10, 11], [-0.1, -0.2]
        )
        state.record([12], [-0.3])  # decoding continues during the TQ write
        owned.seal("c1", "chunk-0")
    state.record([13], [-0.4])

    with state.lifecycle() as owned:
        _, second = owned.freeze("c2")
        assert (
            second.chunk_key,
            second.chunk_sequence,
            second.chunk_token_ids,
            second.chunk_logprobs,
            second.generated_token_ids,
        ) == ("active/c2/1", 1, [12, 13], [-0.3, -0.4], [10, 11, 12, 13])
        owned.rollback("c2")

    # A failed cut's tokens stay unstaged, and its chunk name is never reused.
    with state.lifecycle() as owned:
        _, third = owned.freeze("c3")
        assert (third.chunk_key, third.chunk_sequence, third.chunk_token_ids) == (
            "active/c3/2",
            2,
            [12, 13],
        )
    progress = state.snapshot()
    assert progress.staged_token_count == 2
    assert progress.staging_keys == ("chunk-0",)
    # Leaving the block rolled the unsealed cut back.
    assert not progress.cut_in_progress


def test_call_lifecycle_works_only_while_it_is_held():
    state = _RequestCaptureState(SimpleNamespace(model_call_id="c1"), [1])
    with state.lifecycle() as owned:
        assert owned.begin_terminal() == ()
        with pytest.raises(RuntimeError, match="already started terminal"):
            owned.begin_terminal()
        # Another holder must wait, so a non-blocking try gets nothing.
        with state.try_lifecycle() as other:
            assert other is None

    with pytest.raises(RuntimeError, match="no longer held"):
        owned.seal("c1", None)
    with state.try_lifecycle() as again:
        assert again is not None and again.terminal_started


def test_only_the_call_record_touches_its_locks_and_fields():
    """A new call site must use the record's methods, never its locks or fields."""
    import ast
    import inspect

    from nemo_rl.models.generation.prefix_recovery import recovery, restore, state
    from nemo_rl.models.generation.vllm import progress_reader, vllm_worker_async

    private = set(_RequestCaptureState.__slots__)
    owners = {"_RequestCaptureState", "_CallLifecycle"}
    offenders = []
    for module in (state, recovery, restore, progress_reader, vllm_worker_async):
        for top in ast.parse(inspect.getsource(module)).body:
            if isinstance(top, ast.ClassDef) and top.name in owners:
                continue
            # A class's own fields are its business; another object's are not.
            offenders.extend(
                f"{module.__name__}:{node.lineno} {ast.unparse(node)}"
                for node in ast.walk(top)
                if isinstance(node, ast.Attribute)
                and node.attr in private
                and not (isinstance(node.value, ast.Name) and node.value.id == "self")
            )

    assert offenders == []


def test_call_state_cannot_be_written_from_outside():
    state = _RequestCaptureState(None, [1])
    with pytest.raises(AttributeError):
        state.call = "other"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        state.terminal_started = True  # type: ignore[attr-defined]


def _flush_retired_rows(worker) -> None:
    """Run the next snapshot's delete of retired chunk rows."""
    _cuts(worker)._flush_retired(_cuts(worker)._flushed_epoch + 1)


def test_flush_deletes_rows_retired_before_its_snapshot_once():
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    _cuts(worker).retire(("a/0", "a/1"))
    _cuts(worker).retire(("b/0",))

    assert asyncio.run(worker.flush_obsolete_generation_cut_rows_async(1))
    # One clear for every call that ended before the snapshot.
    assert sink.cleared_generation_prefix_keys == ["a/0", "a/1", "b/0"]

    # A late flush from that snapshot must not delete rows retired since.
    _cuts(worker).retire(("c/0",))
    assert asyncio.run(worker.flush_obsolete_generation_cut_rows_async(1))
    assert sink.cleared_generation_prefix_keys == ["a/0", "a/1", "b/0"]
    assert asyncio.run(worker.flush_obsolete_generation_cut_rows_async(2))
    assert sink.cleared_generation_prefix_keys == ["a/0", "a/1", "b/0", "c/0"]


class _BlockingClearSink(_MemorySink):
    def __init__(self) -> None:
        super().__init__()
        self.clearing = threading.Event()
        self.release = threading.Event()

    def clear(self, staging_keys: list[str]) -> None:
        self.clearing.set()
        assert self.release.wait(10)
        super().clear(staging_keys)


def test_flush_returns_only_once_its_clear_is_done():
    """TQ must not clear rows while the snapshot that follows is saved."""

    async def scenario():
        sink = _BlockingClearSink()
        worker = _worker_with_capture(sink)
        _cuts(worker).retire(("a/0",))
        try:
            flush = asyncio.ensure_future(
                worker.flush_obsolete_generation_cut_rows_async(1)
            )
            assert await asyncio.to_thread(sink.clearing.wait, 5)
            assert not flush.done()
            sink.release.set()
            assert await asyncio.wait_for(flush, timeout=5)
            assert sink.cleared_generation_prefix_keys == ["a/0"]
        finally:
            sink.release.set()

    asyncio.run(scenario())


def test_flush_is_not_queued_behind_a_running_cut():
    async def scenario():
        worker = _worker_with_capture(_MemorySink())
        recovery = _cuts(worker)
        stale_cut_running = threading.Event()
        release_stale_cut = threading.Event()
        try:
            # A cut still holding the control thread, as a stale one would.
            stale_cut = asyncio.get_running_loop().run_in_executor(
                recovery._control_executor,
                lambda: (stale_cut_running.set(), release_stale_cut.wait(10)),
            )
            assert await asyncio.to_thread(stale_cut_running.wait, 5)
            await asyncio.wait_for(
                worker.flush_obsolete_generation_cut_rows_async(1), timeout=5
            )
            assert not stale_cut.done()
            release_stale_cut.set()
            await stale_cut
        finally:
            release_stale_cut.set()
            recovery.shutdown()

    asyncio.run(scenario())


class _FailingClearSink(_MemorySink):
    def clear(self, staging_keys: list[str]) -> None:
        raise RuntimeError("tq unavailable")


def test_failed_flush_clear_is_logged_and_dropped(caplog):
    worker = _worker_with_capture(_FailingClearSink())
    _cuts(worker).retire(("a/0",))

    with caplog.at_level(logging.ERROR):
        assert asyncio.run(worker.flush_obsolete_generation_cut_rows_async(1))

    # The next restore clears what a failed delete left behind.
    assert "failed to clear 1 obsolete generation chunk rows" in caplog.text
    assert _cuts(worker)._obsolete_keys == []


@pytest.mark.parametrize(
    ("top_logprobs", "prefix_cuts"), [(0, False), (4, False), (0, True)]
)
def test_generation_setup_token_capture_keeps_its_arguments_for_restarts(
    monkeypatch, top_logprobs, prefix_cuts
):
    gen = _generation_with_mock_group()
    gen.worker_group.run_all_workers_single_data.return_value = [None, None]
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_generation.ray.get",
        lambda futures: futures,
    )
    cut_arguments = (
        {
            "generation_prefix_cuts_enabled": True,
            "generation_cut_control_token": "secret",
            "generation_cut_control_timeout_s": 60.0,
        }
        if prefix_cuts
        else {
            "generation_prefix_cuts_enabled": False,
            "generation_cut_control_token": None,
            "generation_cut_control_timeout_s": None,
        }
    )
    gen.setup_token_capture(
        {"backend": "simple"},
        "rollout_staging",
        top_logprobs=top_logprobs,
        **cut_arguments,
    )
    # A restarted shard replays these, so a prefix run's shard keeps cutting.
    assert asdict(gen._token_capture_setup) == {
        "dp_cfg": {"backend": "simple"},
        "staging_partition": "rollout_staging",
        "capture_media": False,
        "top_logprobs": top_logprobs,
        **cut_arguments,
    }
    assert gen._token_capture_media is None
    gen.worker_group.run_all_workers_single_data.assert_called_once_with(
        "setup_token_capture",
        **asdict(gen._token_capture_setup),
        run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
    )


def _generation_with_leaders(
    num_shards: int, *, membership=None, fleet_monitor=None
) -> tuple[VllmGeneration, list[MagicMock]]:
    """One worker per shard, so each worker is its shard's DP leader."""
    gen = object.__new__(VllmGeneration)
    gen.cfg = {"vllm_cfg": {"async_engine": True}}
    leaders = [MagicMock(name=f"leader{i}") for i in range(num_shards)]
    for i, leader in enumerate(leaders):
        leader.set_rollout_weight_version.remote.return_value = f"ref{i}"
    gen.worker_group = SimpleNamespace(workers=list(leaders))
    gen.dp_size = num_shards
    gen._refit_membership = membership
    gen.fleet_monitor = fleet_monitor
    gen._rollout_weight_version = None
    return gen, leaders


def test_generation_set_rollout_weight_version_fans_out(monkeypatch):
    gen, leaders = _generation_with_leaders(2)
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_generation.ray.get", lambda ref: ref
    )
    gen.set_rollout_weight_version(7)
    for leader in leaders:
        leader.set_rollout_weight_version.remote.assert_called_once_with(version=7)
    assert gen._rollout_weight_version == 7


def test_set_rollout_weight_version_skips_shards_outside_the_refit(monkeypatch):
    # Shard 1 is dead or still restarting: the rebuilt refit left it out.
    membership = SimpleNamespace(workers_per_shard=1, shard_prefixes={0: 0})
    gen, leaders = _generation_with_leaders(2, membership=membership)
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_generation.ray.get", lambda ref: ref
    )
    gen.set_rollout_weight_version(7)
    leaders[0].set_rollout_weight_version.remote.assert_called_once_with(version=7)
    leaders[1].set_rollout_weight_version.remote.assert_not_called()


def test_set_rollout_weight_version_skips_a_shard_condemned_since_the_refit(
    monkeypatch,
):
    # Shard 1 was a refit member, then died and is restarting: its handle may already be
    # a replacement that is still loading, and must not be waited on.
    monitor = MagicMock()
    monitor.absent_shards.return_value = [1]
    gen, leaders = _generation_with_leaders(2, fleet_monitor=monitor)
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_generation.ray.get", lambda ref: ref
    )
    gen.set_rollout_weight_version(7)
    leaders[0].set_rollout_weight_version.remote.assert_called_once_with(version=7)
    leaders[1].set_rollout_weight_version.remote.assert_not_called()
    assert gen._rollout_weight_version == 7


def test_set_rollout_weight_version_records_a_death_since_the_refit(monkeypatch):
    import ray

    def get(ref):
        if ref == "ref1":
            raise ray.exceptions.ActorDiedError()
        return ref

    monitor = MagicMock()
    gen, leaders = _generation_with_leaders(2, fleet_monitor=monitor)
    monkeypatch.setattr("nemo_rl.models.generation.vllm.vllm_generation.ray.get", get)
    gen.set_rollout_weight_version(7)
    leaders[0].set_rollout_weight_version.remote.assert_called_once_with(version=7)
    monitor.record_actor_death.assert_called_once()
    assert monitor.record_actor_death.call_args.args == (1,)


def test_set_rollout_weight_version_raises_a_death_without_fleet_health(
    monkeypatch,
):
    import ray

    def get(ref):
        raise ray.exceptions.ActorDiedError()

    gen, _ = _generation_with_leaders(1)
    monkeypatch.setattr("nemo_rl.models.generation.vllm.vllm_generation.ray.get", get)
    with pytest.raises(ray.exceptions.ActorDiedError):
        gen.set_rollout_weight_version(7)


# ---------------------------------------------------------------------------
# S4: the request-path hookup (begin -> finish/abort around a served call)
# ---------------------------------------------------------------------------


class _FakeRequest(SimpleNamespace):
    pass


def _uncut_reasons(caplog) -> dict[str, int]:
    """Why calls were left uncut, from the cut's one INFO summary line."""
    (summary,) = [
        record
        for record in caplog.records
        if record.levelno == logging.INFO
        and record.getMessage().startswith("generation cut ")
    ]
    return summary.args[-1]


def _single_call_inventory(
    model_call_id: str = "c1",
    *,
    checkpoint_id: str = "checkpoint-1",
    attempt: int = 0,
) -> GenerationCutInventory:
    """Gym's inventory naming one active call of rollout ``r0``."""
    return GenerationCutInventory.build(
        checkpoint_id=checkpoint_id,
        server_name="policy_model",
        active_prefixes=[
            GenerationCutPrefix(
                ticket_id="ticket-1",
                rollout_id="r0",
                attempt=attempt,
                model_call_id=model_call_id,
                admitted_at=1.0,
            )
        ],
    )


def _worker_with_capture(sink: _MemorySink):
    """A real worker, minus the engine, with in-memory token capture."""
    from nemo_gym.token_id_capture.adapters.vllm import VLLMCaptureAdapter

    worker = _fake_worker()
    worker._capture_sink = sink
    worker.token_capture = RolloutTokenCapture(
        sink=sink,
        weight_version_fn=lambda: worker._rollout_weight_version,
        adapter=VLLMCaptureAdapter(),
    )
    return worker


@pytest.mark.parametrize(
    ("logprobs", "top_logprobs"),
    [(False, 4), (True, None), (True, 0), (True, 2), (True, 8)],
)
def test_capture_rejects_mismatched_http_topk(logprobs, top_logprobs):
    worker = _worker_with_capture(_MemorySink())
    worker._capture_top_logprobs = 4
    request = SimpleNamespace(logprobs=logprobs, top_logprobs=top_logprobs)
    admission = CaptureAdmission(capture_key="r0", model_call_id="c1", mode="text")

    with pytest.raises(ValueError, match="top_logprobs=4 over HTTP"):
        VllmAsyncGenerationWorkerImpl._begin_request_capture(
            worker, request, [10, 11], admission=admission
        )
    assert len(worker._capture_registry) == 0


def test_capture_accepts_matching_http_topk():
    worker = _worker_with_capture(_MemorySink())
    worker._capture_top_logprobs = 4
    request = SimpleNamespace(logprobs=True, top_logprobs=4)
    admission = CaptureAdmission(capture_key="r0", model_call_id="c1", mode="text")

    VllmAsyncGenerationWorkerImpl._begin_request_capture(
        worker, request, [10, 11], admission=admission
    )
    assert request in worker._capture_registry
    assert request.top_logprobs == 4


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

    def fetch(self, staging_keys: list[str]) -> list[StagedCallRecord]:
        self.calls.append(list(staging_keys))
        return [self.records[key] for key in staging_keys]


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
    request = _captured_text_call(worker, "c1", (10, 11, 12))
    content = _served_content([13, 14], [-0.1, -0.2])
    content["choices"][0]["routed_experts"] = None  # vLLM's choice-level field
    message = content["choices"][0]["message"]
    if with_message_tokens:
        # The HTTP serializer preserves these dynamic fields on the message.
        message.update(
            prompt_token_ids=[10, 11, 12],
            generation_token_ids=[13, 14],
            generation_log_probs=[-0.1, -0.2],
        )
    if with_routed_experts:
        message["routed_experts"] = torch.zeros((5, 1, 1), dtype=torch.int16)
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
        metadata = sink.records[0].extras["routed_experts"]
        assert metadata["dtype"] == "int16"
        assert metadata["shape"] == [5, 1, 1]
        torch.testing.assert_close(
            sink.attachments[0]["routed_experts"],
            torch.zeros((5, 1, 1), dtype=torch.int16),
        )
    else:
        assert sink.records[0].extras is None
        assert sink.attachments[0] is None
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
    assert len(worker._capture_registry) == 0


def test_generation_cut_stages_prefix_then_terminal_row_replaces_it():
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    request, state = _live_call(worker, [20, 21], [-0.1, -0.2])

    inventory = _single_call_inventory()
    receipt = _cuts(worker).checkpoint_cut(inventory)
    replayed = _cuts(worker).checkpoint_cut(inventory)

    assert replayed is receipt
    assert len(sink.generation_prefix_records) == 1
    assert receipt.prefixes[0].disposition == "durable_prefix"
    assert receipt.prefixes[0].prefix_token_count == 2
    assert sink.generation_prefix_records[0][1].token_ids_delta == [10, 11, 20, 21]
    assert list(state.snapshot().staging_keys) == sink.generation_prefix_keys

    content = VllmAsyncGenerationWorkerImpl._finish_request_capture(
        worker,
        request,
        _served_content([20, 21, 22], [-0.1, -0.2, -0.3]),
    )
    assert content["ng_commit_coords"]["disposition"] == "staged"
    assert sink.records[-1].token_ids_delta == [10, 11, 20, 21, 22]
    # The snapshot in progress may name the chunk rows; the next one deletes them.
    assert sink.cleared_generation_prefix_keys == []
    _flush_retired_rows(worker)
    assert sink.cleared_generation_prefix_keys == sink.generation_prefix_keys
    assert len(worker._capture_registry) == 0
    assert worker._capture_registry.get_by_model_call_id("c1") is None


def test_generation_cut_stages_the_tokens_it_reads_from_vllm(monkeypatch):
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    request = _captured_text_call(worker)
    state = worker._capture_registry.get(request)
    state.set_effective_output_limit(8)
    client = _progress_client(_OutputProcessor(_engine_request([20, 21], [-0.1, -0.2])))
    _cuts(worker, progress_reader=_progress_reader(monkeypatch, client))
    client._captured_requests["chatcmpl-1"] = "c1"
    inventory = _single_call_inventory()

    receipt = _cuts(worker).checkpoint_cut(inventory)

    assert receipt.prefixes[0].disposition == "durable_prefix"
    assert receipt.prefixes[0].prefix_token_count == 2
    assert sink.generation_prefix_records[0][1].token_ids_delta == [10, 11, 20, 21]
    assert sink.generation_prefix_records[0][1].generation_log_probs_delta[2:] == [
        -0.1,
        -0.2,
    ]


@pytest.mark.parametrize(
    ("completed_capture_key", "matches"), [("r0-a1", True), ("r0", False)]
)
def test_completed_retry_is_matched_by_its_attempt_qualified_capture_key(
    completed_capture_key: str, matches: bool
):
    """A retry's terminal row is keyed ``<rollout>-a<attempt>``, as Gym keys it."""
    worker = _worker_with_capture(_MemorySink())
    _cuts(worker)
    _cuts(worker).remember_completed(
        "c2",
        SimpleNamespace(
            capture_key=completed_capture_key,
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
            _cuts(worker)._completed_ack(prefix, "checkpoint-1", Counter())
        return
    uncut: Counter[str] = Counter()
    ack = _cuts(worker)._completed_ack(prefix, "checkpoint-1", uncut)

    assert ack.disposition == "durable_prefix"
    assert ack.cut_kind == "terminal_completion"
    assert ack.staging_keys == ("terminal-key",)
    assert ack.prefix_token_count == 3
    assert not uncut


def test_generation_cut_past_its_deadline_fails_uncut_calls_without_staging(caplog):
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    _, state = _live_call(worker, [11, 12], [-0.1, -0.2], prompt_token_ids=(10,))
    inventory = _single_call_inventory()

    # Gym has already stopped waiting: the queued cut must not stage rows.
    with caplog.at_level(logging.INFO):
        receipt = _cuts(worker).checkpoint_cut(
            inventory, deadline=time.monotonic() - 1.0
        )

    receipt.validate_for(inventory)
    assert [ack.disposition for ack in receipt.prefixes] == ["durable_failure"]
    assert sink.generation_prefix_records == []
    assert not state.snapshot().cut_in_progress
    assert state.snapshot().generated_token_ids == [11, 12]
    assert "uncut=1 of 1 model_call_ids=['c1']" in caplog.text
    assert _uncut_reasons(caplog) == {"control_deadline": 1}


def _active_call_inventory(
    worker, *calls: tuple[str, list[int]], output_limit: int | None = 16
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
        state = worker._capture_registry.get(request)
        if output_limit is not None:
            state.set_effective_output_limit(output_limit)
        state.record(generated, [-0.1] * len(generated))
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


def test_generation_cut_refuses_a_prefix_ending_inside_a_character(caplog):
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    dinosaur = list("🦖".encode())
    inventory, (split, whole) = _active_call_inventory(
        worker,
        ("split", list(b"Sure ") + dinosaur[:2]),
        ("whole", list(b"Sure ") + dinosaur),
    )

    with caplog.at_level(logging.INFO):
        receipt = _cuts(worker).checkpoint_cut(inventory)

    # vLLM would prime the resumed tail on half a character and echo the
    # prompt into the response, so that call regenerates instead.
    assert [ack.disposition for ack in receipt.prefixes] == [
        "durable_failure",
        "durable_prefix",
    ]
    assert len(sink.generation_prefix_records) == 1
    assert not split.snapshot().cut_in_progress
    assert split.snapshot().generated_token_ids == list(b"Sure ") + dinosaur[:2]
    assert list(whole.snapshot().staging_keys) == sink.generation_prefix_keys
    assert (
        "cannot cut model call split: its 7 generated tokens end inside a "
        "multi-byte character"
    ) in caplog.text
    assert "model call whole" not in caplog.text
    assert _uncut_reasons(caplog) == {"incomplete_utf8_boundary": 1}


def test_generation_cut_of_a_call_with_no_tokens_logs_at_debug(caplog):
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    inventory, (empty,) = _active_call_inventory(worker, ("empty", []))

    with caplog.at_level(
        logging.DEBUG, logger="nemo_rl.models.generation.prefix_recovery.recovery"
    ):
        receipt = _cuts(worker).checkpoint_cut(inventory)

    assert [ack.disposition for ack in receipt.prefixes] == ["durable_failure"]
    assert _uncut_reasons(caplog) == {"zero_generated_tokens": 1}
    assert sink.generation_prefix_records == []
    assert not empty.snapshot().cut_in_progress
    # Still prefilling is expected, so it stays out of WARNING.
    [record] = [r for r in caplog.records if "model call empty" in r.getMessage()]
    assert record.levelno == logging.DEBUG
    assert "it has no generated tokens yet" in record.getMessage()


def test_completed_calls_are_not_kept_without_prefix_cuts():
    """Only a generation cut reads completed calls, so turn runs keep none."""
    worker = _worker_with_capture(_MemorySink())

    worker._prefix_recovery.remember_completed(
        "c2", SimpleNamespace(capture_key="r0", disposition="staged"), 3
    )

    # Without cuts the worker holds the do-nothing recovery, which keeps nothing.
    assert isinstance(worker._prefix_recovery, NoPrefixRecovery)
    assert not hasattr(worker._prefix_recovery, "_completed")


@pytest.mark.parametrize(
    ("completed", "reason", "category"),
    [
        (None, "this worker has no record of it", "no_capture_or_completed_state"),
        (
            {"disposition": "capture_failed"},
            "its terminal write ended as 'capture_failed'",
            "completed_not_staged",
        ),
        (
            {"effective_output_limit": None},
            "vLLM did not resolve its effective output limit",
            "completed_missing_output_limit",
        ),
        (
            {"terminal_finish_reason": None},
            "it finished without a stop or length finish reason",
            "completed_missing_finish_reason",
        ),
    ],
)
def test_completed_call_that_cannot_be_cut_logs_why(
    completed, reason, category, caplog
):
    worker = _worker_with_capture(_MemorySink())
    _cuts(worker)
    if completed is not None:
        _cuts(worker).remember_completed(
            "c2",
            SimpleNamespace(
                capture_key="r0",
                disposition=completed.get("disposition", "staged"),
                staging_key="terminal-key",
                digest="a" * 64,
            ),
            3,
            effective_output_limit=completed.get("effective_output_limit", 128),
            terminal_finish_reason=completed.get("terminal_finish_reason", "stop"),
        )
    prefix = GenerationCutPrefix(
        ticket_id="ticket-1",
        rollout_id="r0",
        attempt=0,
        model_call_id="c2",
        admitted_at=1.0,
    )

    uncut: Counter[str] = Counter()
    with caplog.at_level(logging.WARNING):
        ack = _cuts(worker)._completed_ack(prefix, "checkpoint-1", uncut)

    assert ack.disposition == "durable_failure"
    assert f"cannot cut completed model call c2: {reason}" in caplog.text
    assert uncut == {category: 1}


def test_call_with_a_recording_error_is_not_cut(caplog):
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    inventory, (poisoned, healthy) = _active_call_inventory(
        worker, ("poisoned", [11, 12]), ("healthy", [13, 14])
    )
    poisoned.record_error("RuntimeError: vLLM returned no logprobs")

    with caplog.at_level(logging.INFO):
        receipt = _cuts(worker).checkpoint_cut(inventory)

    assert [ack.disposition for ack in receipt.prefixes] == [
        "durable_failure",
        "durable_prefix",
    ]
    assert not poisoned.snapshot().cut_in_progress
    assert list(healthy.snapshot().staging_keys) == sink.generation_prefix_keys
    assert _uncut_reasons(caplog) == {"observation_error": 1}


class _WordTokenizer:
    """One token per whitespace-separated word, the shape vLLM's expansion reads."""

    max_token_id = 1000

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        return [len(word) for word in text.split()]


class _SamplingParams(SimpleNamespace):
    """The SamplingParams surface the cut checks read."""

    _bad_words_token_ids = None

    def clone(self) -> "_SamplingParams":
        return _SamplingParams(**vars(self))

    def update_from_tokenizer(self, tokenizer) -> None:
        self._bad_words_token_ids = [
            tokenizer.encode(word, add_special_tokens=False) for word in self.bad_words
        ]


def _sampling_params(**overrides) -> _SamplingParams:
    defaults = dict(stop=[], bad_words=[], seed=None, max_tokens=16)
    return _SamplingParams(**{**defaults, **overrides})


def _record_cut_constraints(state, sampling_params) -> None:
    """Run the sampling-settings hook for one call, as vLLM resolves its params."""
    PrefixRecovery(
        PrefixRecoveryParts(
            capture=None,
            sink=None,
            staging_source=None,
            registry=SimpleNamespace(get=lambda request: state),
            tokenizer=_WordTokenizer(),
            weight_version=lambda: 0,
            control_token=None,
            control_timeout_s=None,
            progress_reader=_NoCallsInVllm(),
        )
    ).on_sampling_params(object(), sampling_params)


@pytest.mark.parametrize(
    "setting",
    [
        {"stop": ["</answer>"]},
        {"bad_words": ["forbidden word"]},
    ],
)
def test_sampling_settings_a_resumed_tail_cannot_honor_keep_the_call_uncut(
    setting, monkeypatch, caplog
):
    monkeypatch.setattr(
        "nemo_rl.models.generation.prefix_recovery.recovery._LOGGED_UNCUTTABLE_REASONS",
        set(),
    )
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    inventory, (constrained, plain) = _active_call_inventory(
        worker, ("constrained", [11, 12]), ("plain", [13, 14])
    )
    _record_cut_constraints(constrained, _sampling_params(**setting))
    _record_cut_constraints(plain, _sampling_params())
    assert constrained.snapshot().uncuttable_reason is not None
    assert plain.snapshot().uncuttable_reason is None

    with caplog.at_level(logging.INFO):
        receipt = _cuts(worker).checkpoint_cut(inventory)

    # The constrained call regenerates after a restore instead of resuming a
    # tail that would miss a straddling stop string or uncount its prefix.
    assert [ack.disposition for ack in receipt.prefixes] == [
        "durable_failure",
        "durable_prefix",
    ]
    assert not constrained.snapshot().cut_in_progress
    assert constrained.snapshot().generated_token_ids == [11, 12]
    assert list(plain.snapshot().staging_keys) == sink.generation_prefix_keys
    assert _uncut_reasons(caplog) == {"uncuttable_request": 1}


@pytest.mark.vllm
@pytest.mark.parametrize(
    ("seed", "temperature", "blocked"),
    [
        # A restored call would restart the seeded generator and reuse its draws.
        (7, 1.0, True),
        # vLLM raises this to its minimum temperature and still samples.
        (7, 1e-6, True),
        # Greedy decoding draws nothing, so the seed is irrelevant.
        (7, 0.0, False),
        # Unseeded calls draw from the engine's shared generator.
        (None, 1.0, False),
    ],
)
def test_only_a_seeded_sampled_call_is_kept_out_of_cuts(seed, temperature, blocked):
    """vLLM's own sampling type decides, so a change to its rule fails here."""
    pytest.importorskip("vllm")
    from vllm import SamplingParams

    blockers = _generation_cut_blockers(
        SamplingParams(seed=seed, temperature=temperature), _WordTokenizer()
    )

    assert ("seed" in blockers) is blocked


@pytest.mark.parametrize(
    ("bad_words", "blocked"),
    [
        # Banned at every step whatever precedes it, before or after a cut.
        (["forbidden", "banned"], False),
        # Its first token can sit in the restored prefix, where vLLM no longer looks.
        (["forbidden", "two words"], True),
    ],
)
def test_only_multi_token_bad_words_keep_a_call_uncut(bad_words, blocked):
    sampling_params = _sampling_params(bad_words=bad_words)

    blockers = _generation_cut_blockers(sampling_params, _WordTokenizer())

    assert ("multi-token bad_words" in blockers) is blocked
    # The check runs on a copy; vLLM tokenizes the request's own params later.
    assert sampling_params._bad_words_token_ids is None


@pytest.mark.vllm
def test_bad_word_check_uses_vllms_own_expansion():
    # Both dependency markers select the combined vLLM + Gym lane.
    pytest.importorskip("vllm")
    from vllm import SamplingParams

    class SpaceSplitsTokenizer(_WordTokenizer):
        # " split" encodes as two tokens, "split" as one.
        def encode(self, text: str, add_special_tokens: bool) -> list[int]:
            ids = super().encode(text, add_special_tokens)
            return [999, *ids] if text == " split" else ids

    tokenizer = SpaceSplitsTokenizer()

    assert not _generation_cut_blockers(
        SamplingParams(bad_words=["forbidden"]), tokenizer
    )
    assert "multi-token bad_words" in _generation_cut_blockers(
        SamplingParams(bad_words=["two words"]), tokenizer
    )
    # vLLM drops a space-prefixed form whose length differs, so only "split"
    # (one token) is enforced and the call stays cuttable.
    assert not _generation_cut_blockers(SamplingParams(bad_words=["split"]), tokenizer)


def test_uncuttable_sampling_settings_warn_once_per_reason(monkeypatch, caplog):
    monkeypatch.setattr(
        "nemo_rl.models.generation.prefix_recovery.recovery._LOGGED_UNCUTTABLE_REASONS",
        set(),
    )
    states = [_RequestCaptureState(None, [1]), _RequestCaptureState(None, [1])]

    with caplog.at_level(logging.WARNING):
        for state in states:
            _record_cut_constraints(state, _sampling_params(stop=["x"]))

    assert all(state.snapshot().uncuttable_reason for state in states)
    assert caplog.text.count("are not cut; they regenerate after a restore") == 1


class _FailOncePrefixSink(_MemorySink):
    def __init__(self) -> None:
        super().__init__()
        self._fail_next_prefix = True

    def stage_generation_prefix(self, *args, **kwargs) -> StageResult:
        if self._fail_next_prefix:
            self._fail_next_prefix = False
            raise RuntimeError("injected generation-prefix staging failure")
        return super().stage_generation_prefix(*args, **kwargs)


def test_generation_cut_rolls_back_tokens_after_staging_failure(caplog):
    sink = _FailOncePrefixSink()
    worker = _worker_with_capture(sink)
    _, state = _live_call(worker, [11, 12], [-0.1, -0.2], prompt_token_ids=(10,))

    def inventory(checkpoint_id: str) -> GenerationCutInventory:
        return _single_call_inventory(checkpoint_id=checkpoint_id)

    # The failed call is acknowledged, not raised, and its buffer rolls back.
    with caplog.at_level(logging.INFO):
        failed = _cuts(worker).checkpoint_cut(inventory("checkpoint-1"))
    assert [ack.disposition for ack in failed.prefixes] == ["durable_failure"]
    assert not state.snapshot().cut_in_progress
    assert state.snapshot().generated_token_ids == [11, 12]
    assert state.snapshot().staged_token_count == 0
    assert _uncut_reasons(caplog) == {"cut_error": 1}

    with caplog.at_level(logging.INFO):
        receipt = _cuts(worker).checkpoint_cut(inventory("checkpoint-2"))
    assert receipt.prefixes[0].prefix_token_count == 2
    # A cut that leaves no call uncut adds no summary line.
    assert _uncut_reasons(caplog) == {"cut_error": 1}
    assert sink.generation_prefix_records[-1][1].token_ids_delta == [10, 11, 12]


def test_one_failed_call_does_not_fail_the_rest_of_the_cut():
    sink = _FailOncePrefixSink()
    worker = _worker_with_capture(sink)
    inventory, (failing, healthy) = _active_call_inventory(
        worker, ("failing", [11, 12]), ("healthy", [13, 14])
    )

    receipt = _cuts(worker).checkpoint_cut(inventory)

    assert [ack.disposition for ack in receipt.prefixes] == [
        "durable_failure",
        "durable_prefix",
    ]
    assert not failing.snapshot().cut_in_progress
    assert list(healthy.snapshot().staging_keys) == sink.generation_prefix_keys


def test_call_without_an_output_limit_is_not_frozen_for_the_cut(caplog):
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    inventory, (state,) = _active_call_inventory(
        worker, ("c1", [11, 12]), output_limit=None
    )

    with caplog.at_level(logging.INFO):
        receipt = _cuts(worker).checkpoint_cut(inventory)

    assert [ack.disposition for ack in receipt.prefixes] == ["durable_failure"]
    assert not state.snapshot().cut_in_progress
    assert sink.generation_prefix_records == []
    assert _uncut_reasons(caplog) == {"missing_output_limit": 1}


def test_call_that_finishes_while_being_cut_falls_back_to_its_completed_record(
    caplog,
):
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    inventory, (state,) = _active_call_inventory(worker, ("c1", [11, 12]))
    with state.lifecycle() as owned:  # its terminal write won the lifecycle
        owned.begin_terminal()

    with caplog.at_level(logging.INFO):
        receipt = _cuts(worker).checkpoint_cut(inventory)

    # It falls back to the completed-call path, which has no record of it yet.
    assert [ack.disposition for ack in receipt.prefixes] == ["durable_failure"]
    assert _uncut_reasons(caplog) == {"no_capture_or_completed_state": 1}


def test_unexpected_error_after_the_freeze_rolls_the_call_back():
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    inventory, (state,) = _active_call_inventory(worker, ("c1", [11, 12]))

    class _BrokenTokenizer:
        def decode(self, token_ids, **kwargs):
            raise ValueError("injected decode failure")

    receipt = _cuts(worker, tokenizer=_BrokenTokenizer()).checkpoint_cut(inventory)

    assert [ack.disposition for ack in receipt.prefixes] == ["durable_failure"]
    # Without the rollback, every later cut of this call would fail to freeze.
    assert not state.snapshot().cut_in_progress
    retried = _cuts(worker, tokenizer=_ByteTokenizer()).checkpoint_cut(
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
    receipt = _cuts(worker).checkpoint_cut(inventory)
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
    _cuts(worker, staging_source=worker._staging_source)
    return worker


def test_usable_restored_prefix_is_resumed():
    sink = _MemorySink()
    cut_key, cut_record, request = _restorable_cut(sink)
    worker = _resumed_worker(sink, records={cut_key: cut_record})
    admission = worker._capture_admission(request)

    served, snapshot = _cuts(worker).resolve_restore(admission, [])

    assert served is admission
    assert snapshot.digest == cut_record.digest
    _flush_retired_rows(worker)
    assert sink.cleared_generation_prefix_keys == []


@pytest.mark.parametrize(
    "failure",
    ["missing chunk", "chunk newer than the worker", "digest mismatch"],
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

    with caplog.at_level(logging.WARNING):
        served, snapshot = _cuts(worker).resolve_restore(admission, [])

    assert snapshot is None
    assert served.generation_cut is None
    assert "regenerating it from its turn boundary" in caplog.text
    # The regenerated call is admitted as an ordinary call from its boundary.
    VllmAsyncGenerationWorkerImpl._begin_request_capture(
        worker, request, [10], admission=served, prefix_token_ids=[]
    )
    assert list(worker._capture_registry.get(request).snapshot().staging_keys) == []
    # The restored rows it will never extend are retired for the next snapshot.
    _flush_retired_rows(worker)
    assert sink.cleared_generation_prefix_keys == [cut_key]


def test_restored_generation_cut_is_extended_and_retired_on_completion(
    caplog, monkeypatch
):
    caplog.set_level(logging.INFO)
    sink = _MemorySink()
    cut_key, cut_record, request = _restorable_cut(sink)
    resumed_worker = _resumed_worker(sink, records={cut_key: cut_record})
    admission = resumed_worker._capture_admission(request)
    with caplog.at_level(logging.INFO):
        cut = _cuts(resumed_worker)._rebuild_prefix(admission, [])
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
            [10, 12, 13],
            admission=bad_digest_admission,
            prefix_token_ids=[],
            generation_cut=_cuts(resumed_worker)._rebuild_prefix(
                bad_digest_admission, []
            ),
            resumed_generation_token_ids=[12, 13],
        )
    VllmAsyncGenerationWorkerImpl._begin_request_capture(
        resumed_worker,
        request,
        [10, 12, 13],
        admission=admission,
        prefix_token_ids=[],
        generation_cut=cut,
        resumed_generation_token_ids=[12, 13],
    )
    # Only the restore Gym accepted counts.
    counts = VllmAsyncGenerationWorkerImpl.get_generation_prefix_counts(resumed_worker)
    assert (counts["restored_calls"], counts["restored_prefix_tokens"]) == (1, 2)
    replays = []

    def replay(tokenizer, **kwargs):
        replays.append(kwargs)
        return "partial"

    monkeypatch.setattr(
        "nemo_rl.models.generation.prefix_recovery.restore._detokenize_restored_prefix",
        replay,
    )
    # vLLM's processed tail: EOS and any matched stop string already stripped.
    output = SimpleNamespace(token_ids=[14, 151645], text=" tail")
    resumed_worker._prefix_recovery.restore_reply_text(
        request,
        SimpleNamespace(outputs=[output]),
        tokenizer=_ByteTokenizer(),
    )
    (replay_kwargs,) = replays
    assert replay_kwargs["original_prompt_token_ids"] == [10]
    assert replay_kwargs["prefix_token_ids"] == [12, 13]
    assert replay_kwargs["terminal"] is None
    assert output.text == "partial tail"

    # A second checkpoint taken before the resumed call finishes must retain
    # the original cut key and append the newly generated tail. Gym v2 stores
    # this complete key list in the replacement attempt's model record.
    resumed = resumed_worker._capture_registry.get(request)
    resumed.record([14], [-0.3])
    second_receipt = _cuts(resumed_worker).checkpoint_cut(
        _single_call_inventory("c2", checkpoint_id="checkpoint-2", attempt=1)
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
    assert content["ng_commit_coords"]["capture_key"] == "r0-a1"
    assert final_record.token_ids_delta == [10, 12, 13, 14]
    assert final_record.token_mask_delta == [0.0, 1.0, 1.0, 1.0]
    assert final_record.generation_log_probs_delta == [0.0, -0.1, -0.1, -0.3]
    assert final_record.weight_version == 7
    assert content["ng_commit_coords"]["weight_version"] == 7
    assert _cuts(resumed_worker)._completed["c2"].generation_token_count == 3
    _flush_retired_rows(resumed_worker)
    assert sink.cleared_generation_prefix_keys == list(
        second_receipt.prefixes[0].staging_keys
    )
    assert "generation prefix restored:" in caplog.text


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32])
def test_request_capture_token_in_prev_len_chains_without_codec(monkeypatch, dtype):
    def reject_codec(*args, **kwargs):
        pytest.fail("worker-owned capture must not encode or decode route envelopes")

    monkeypatch.setattr(
        "nemo_rl.utils.routed_experts_codec.encode_routed_experts", reject_codec
    )
    monkeypatch.setattr(
        "nemo_rl.utils.routed_experts_codec.decode_routed_experts", reject_codec
    )
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    request = _FakeRequest(
        ng_capture={
            "capture_key": "r0",
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
    content = _served_content([22], [-0.5])
    routes = torch.arange(12, dtype=dtype).reshape(6, 1, 2)
    content["choices"][0]["message"]["routed_experts"] = routes
    content = VllmAsyncGenerationWorkerImpl._finish_request_capture(
        worker, request, content
    )
    coords = content["ng_commit_coords"]
    assert coords["parent_call_id"] == "c1"
    assert (coords["delta_len"], coords["cum_len"]) == (3, 6)
    assert sink.records[0].token_ids_delta == [20, 21, 22]
    staged_routes = sink.attachments[0]["routed_experts"]
    # Include the previous turn's last token, now routed by this call's prefill.
    torch.testing.assert_close(staged_routes, routes[2:])
    # Staging reuses the slice, preserving the compact dtype without a copy.
    assert staged_routes.data_ptr() == routes[2:].data_ptr()
    assert sink.records[0].extras["routed_experts"]["shape"] == [4, 1, 2]
    assert "routed_experts" not in content["choices"][0]["message"]


@pytest.mark.parametrize(
    "routes",
    [torch.zeros(3, dtype=torch.int16), torch.zeros((2, 1, 1), dtype=torch.int16)],
)
def test_request_capture_drops_routes_with_wrong_rank_or_length(routes, caplog):
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    request = _FakeRequest(
        ng_capture={"capture_key": "r", "model_call_id": "c", "mode": "text"},
        stream=False,
    )
    VllmAsyncGenerationWorkerImpl._begin_request_capture(worker, request, [1, 2])
    content = _served_content([3], [-0.1])
    content["choices"][0]["message"]["routed_experts"] = routes

    content = VllmAsyncGenerationWorkerImpl._finish_request_capture(
        worker, request, content
    )

    assert content["ng_commit_coords"]["disposition"] == "staged"
    assert sink.records[0].extras is None
    assert sink.attachments[0] is None
    assert "dropping invalid routed_experts" in caplog.text


def _staging_chain_request(prev_len: int = 3) -> _FakeRequest:
    return _FakeRequest(
        ng_capture={
            "capture_key": "r0",
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
    state = worker._capture_registry.get(request)
    assert state.call.prefix_token_ids == prefix
    assert state.prompt_token_ids == [10, 11, 12, 20]


def test_inline_prefix_admission_resolves_without_a_fetch():
    worker = _worker_with_capture(_MemorySink())
    source = _MemoryPrefixSource({})
    worker._chain_prefix.install(source)
    request = _FakeRequest(
        ng_capture={
            "capture_key": "r0",
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
            ng_capture={"capture_key": "r0", "model_call_id": "c1", "mode": "text"}
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
    assert len(worker._capture_registry) == 0


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

    assert len(worker._capture_registry) == 0


@pytest.mark.parametrize("capture_enabled", [False, True])
def test_request_capture_is_a_noop_without_context_or_capture(
    capture_enabled: bool,
) -> None:
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    plain = _FakeRequest(stream=False)  # no ng_capture attribute
    if not capture_enabled:
        worker.token_capture = None
        plain.ng_capture = {"capture_key": "r0", "model_call_id": "c1", "mode": "text"}
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
    assert len(worker._capture_registry) == 0
    assert sink.records == []


def _text_root_request() -> _FakeRequest:
    return _FakeRequest(
        ng_capture={
            "capture_key": "r0",
            "model_call_id": "c1",
            "parent_call_id": None,
            "prev_len": 0,
            "mode": "text",
        },
        stream=False,
    )


def _content_with_topk(gen_ids, logprobs):
    content = _served_content(gen_ids, logprobs)
    content["choices"][0]["message"].update(
        sampler_topk_token_ids=torch.tensor(
            [[token_id, 1] for token_id in gen_ids], dtype=torch.int32
        ),
        sampler_topk_logprobs=torch.tensor([[lp, -5.0] for lp in logprobs]),
    )
    return content


def test_request_capture_stages_topk_with_the_call():
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    request = _text_root_request()
    VllmAsyncGenerationWorkerImpl._begin_request_capture(worker, request, [10, 11, 12])
    content = _content_with_topk([13, 14], [-0.1, -0.2])
    ids = content["choices"][0]["message"]["sampler_topk_token_ids"]
    logprobs = content["choices"][0]["message"]["sampler_topk_logprobs"]

    content = VllmAsyncGenerationWorkerImpl._finish_request_capture(
        worker, request, content
    )

    assert content["ng_commit_coords"]["disposition"] == "staged"
    assert len(sink.records) == 1
    assert sink.attachments == [
        {"sampler_topk_token_ids": ids, "sampler_topk_logprobs": logprobs}
    ]
    # It never crosses the worker -> gate hop.
    assert content["choices"] == [
        {"index": 0, "message": {"role": "assistant", "content": "x"}}
    ]


def test_request_capture_skips_topk_when_the_call_did_not_stage():
    class _RejectingSink(_MemorySink):
        def stage(self, record: StagedCallRecord, *, attachments=None) -> StageResult:
            assert attachments["sampler_topk_token_ids"].shape == (1, 2)
            return StageResult(ok=False, staging_key=record.staging_key, error="down")

    sink = _RejectingSink()
    worker = _worker_with_capture(sink)
    request = _text_root_request()
    VllmAsyncGenerationWorkerImpl._begin_request_capture(worker, request, [10, 11, 12])

    content = VllmAsyncGenerationWorkerImpl._finish_request_capture(
        worker, request, _content_with_topk([13], [-0.1])
    )

    assert content["ng_commit_coords"]["disposition"] == "capture_failed"
    assert sink.records == []
    assert "sampler_topk_token_ids" not in content["choices"][0]["message"]


def test_request_capture_drops_topk_from_uncaptured_responses():
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    plain = _FakeRequest(stream=False)  # no ng_capture attribute

    out = VllmAsyncGenerationWorkerImpl._finish_request_capture(
        worker, plain, _content_with_topk([3], [-0.1])
    )

    assert "sampler_topk_token_ids" not in out["choices"][0]["message"]
    assert "sampler_topk_logprobs" not in out["choices"][0]["message"]
    assert sink.records == []


def test_request_capture_missing_required_topk_fails_before_staging():
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    request = _text_root_request()
    VllmAsyncGenerationWorkerImpl._begin_request_capture(worker, request, [10, 11, 12])
    worker._capture_top_logprobs = 2
    out = VllmAsyncGenerationWorkerImpl._finish_request_capture(
        worker, request, _served_content([13], [-0.1])
    )
    assert out["ng_commit_coords"]["disposition"] == "capture_failed"
    assert sink.records == []
    assert len(worker._capture_registry) == 0


def test_request_capture_topk_partial_write_returns_capture_failed():
    sink = object.__new__(TQTokenSink)
    sink._capture_media = False
    sink._store = MagicMock()

    def failed_put(key, fields, **kwargs):
        assert "token_ids_delta" in fields
        assert "sampler_topk_token_ids" in fields
        raise RuntimeError("top-k storage unavailable")

    sink._store.put.side_effect = failed_put
    worker = _worker_with_capture(sink)
    request = _text_root_request()
    VllmAsyncGenerationWorkerImpl._begin_request_capture(worker, request, [10, 11, 12])
    content = VllmAsyncGenerationWorkerImpl._finish_request_capture(
        worker, request, _content_with_topk([13], [-0.1])
    )
    assert content["ng_commit_coords"]["disposition"] == "capture_failed"
    sink._store.put.assert_called_once()
    sink._store.clear.assert_called_once_with(["r0/c1"])
    assert "sampler_topk_token_ids" not in content["choices"][0]["message"]
    assert "sampler_topk_logprobs" not in content["choices"][0]["message"]
    assert len(worker._capture_registry) == 0


def _cut_active_call(worker, sink, model_call_id: str = "c1"):
    """Begin one captured call and stage its prefix at a generation cut."""
    request, state = _live_call(
        worker, [20, 21], [-0.1, -0.2], model_call_id=model_call_id
    )
    inventory = _single_call_inventory(model_call_id)
    _cuts(worker).checkpoint_cut(inventory)
    assert list(state.snapshot().staging_keys) == sink.generation_prefix_keys
    return request


def test_aborted_call_clears_its_rows_at_the_next_snapshot():
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    request = _cut_active_call(worker, sink)

    assert VllmAsyncGenerationWorkerImpl._abort_request_capture(
        worker, request, reason="client_disconnected", blocking=False
    )
    assert sink.cleared_generation_prefix_keys == []

    _flush_retired_rows(worker)
    assert sink.generation_prefix_keys
    assert sink.cleared_generation_prefix_keys == sink.generation_prefix_keys


class _FailingTerminalSink(_MemorySink):
    def stage(
        self, record: StagedCallRecord, *, attachments: dict | None = None
    ) -> StageResult:
        return StageResult(ok=False, error="tq unavailable")


def test_call_whose_terminal_staging_fails_clears_its_cut_rows():
    sink = _FailingTerminalSink()
    worker = _worker_with_capture(sink)
    request = _cut_active_call(worker, sink)

    content = VllmAsyncGenerationWorkerImpl._finish_request_capture(
        worker,
        request,
        _served_content([20, 21, 22], [-0.1, -0.2, -0.3]),
    )

    assert content["ng_commit_coords"]["disposition"] == "capture_failed"
    _flush_retired_rows(worker)
    assert sink.cleared_generation_prefix_keys == sink.generation_prefix_keys


def test_request_capture_abort_fails_the_call_and_drains_state():
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    request = _captured_text_call(worker, "c1", (1, 2))
    VllmAsyncGenerationWorkerImpl._abort_request_capture(
        worker, request, reason="engine_error"
    )
    assert len(worker._capture_registry) == 0
    assert sink.records == []
    # A call that was never cut has no chunk rows to clear.
    _flush_retired_rows(worker)
    assert sink.cleared_generation_prefix_keys == []
    # A late finish after abort is a no-op (state already drained).
    out = VllmAsyncGenerationWorkerImpl._finish_request_capture(
        worker, request, _served_content([3], [-0.1])
    )
    assert "ng_commit_coords" not in out


@pytest.mark.vllm
@pytest.mark.parametrize("pruning_rate", [0.0, 0.5])
def test_omni_capture_setup_rejects_video_pruning(monkeypatch, pruning_rate):
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
        renderer=SimpleNamespace(get_mm_processor=lambda: SimpleNamespace(info=info)),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
    )
    monkeypatch.setattr(
        "nemo_rl.data_plane.build_data_plane_client",
        lambda dp_cfg, bootstrap, segment_size=None: MagicMock(),
    )
    if pruning_rate:
        with pytest.raises(ValueError, match="video token pruning"):
            asyncio.run(
                VllmAsyncGenerationWorkerImpl.setup_token_capture(
                    worker, {}, staging_partition="staging", capture_media=True
                )
            )
    else:
        spec = asyncio.run(
            VllmAsyncGenerationWorkerImpl.setup_token_capture(
                worker, {}, staging_partition="staging", capture_media=True
            )
        )
        # The engine's model dtype and the HF patch size, as pinned on the sink.
        assert spec == MediaColumnSpec(pixel_dtype=torch.bfloat16, patch_size=2)
        assert worker._capture_image_token_id == 18


def test_request_capture_text_call_in_media_run_stages_without_attachments():
    from nemo_rl.data.captured_media import CapturedMedia

    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    request = _text_root_request()
    # A text-only call in a media-capture run adds no media tensors.
    VllmAsyncGenerationWorkerImpl._begin_request_capture(
        worker, request, [10, 11, 12], media=CapturedMedia((), None)
    )

    content = VllmAsyncGenerationWorkerImpl._finish_request_capture(
        worker, request, _served_content([13], [-0.1])
    )

    assert content["ng_commit_coords"]["disposition"] == "staged"
    assert sink.attachments == [None]


@pytest.mark.parametrize("prev_len", [0, 3])
def test_route_delta_preserves_actual_predecessor_tail(prev_len):
    from nemo_gym.token_id_capture.adapters.vllm import VLLMCaptureAdapter

    routes = torch.arange(7, dtype=torch.int16).reshape(7, 1, 1)
    payload = {
        "choices": [
            {
                "message": {
                    "routed_experts": routes,
                }
            }
        ]
    }
    staged_routes = VllmAsyncGenerationWorkerImpl._delta_align_routed_experts(
        payload,
        prev_len=prev_len,
        prompt_len=5,
        generated_len=2,
    )
    extras = VLLMCaptureAdapter().extract_extras(payload)
    torch.testing.assert_close(staged_routes, routes[max(0, prev_len - 1) :])
    assert extras["routed_experts"] == routed_experts_tensor_metadata(staged_routes)
    assert "predecessor_tail_route" not in extras
