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

"""How far each running captured call got, read while paused for a refit."""

import asyncio
import inspect
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import AsyncMock, Mock, patch

import pytest

from nemo_rl.models.generation.vllm.vllm_worker_async import (
    _CAPTURED_MODEL_CALL,
    VllmAsyncGenerationWorkerImpl,
    _AsyncLLMHTTPClient,
)

pytestmark = pytest.mark.vllm


class _FakeEngine:
    """Records each request in vLLM's output-processor shape, then waits."""

    model_config = renderer = input_processor = vllm_config = None

    def __init__(self, generated_tokens: int) -> None:
        self.output_processor = SimpleNamespace(request_states={})
        self.release = asyncio.Event()
        self._generated_tokens = generated_tokens

    async def generate(self, prompt: Any, sampling_params: Any, request_id: str):
        # vLLM keys its state by an internal ID and keeps the caller's as
        # external_req_id.
        self.output_processor.request_states[f"internal-{request_id}"] = (
            SimpleNamespace(
                external_req_id=request_id,
                detokenizer=SimpleNamespace(
                    num_output_tokens=lambda: self._generated_tokens
                ),
            )
        )
        await self.release.wait()
        yield "output"

    def abort(self, request_id: str) -> None:
        pass


def test_running_captured_calls_report_their_generated_tokens() -> None:
    async def exercise() -> None:
        engine = _FakeEngine(generated_tokens=200)
        client = _AsyncLLMHTTPClient(
            engine, asyncio.get_running_loop(), flat_logprobs=False
        )
        sampling_params = SimpleNamespace(logprobs=None, prompt_logprobs=None)

        async def serve(model_call_id: Optional[str], request_id: str) -> None:
            # What _begin_request_capture does in the request's task.
            if model_call_id is not None:
                _CAPTURED_MODEL_CALL.set(model_call_id)
            async for _ in client.generate("prompt", sampling_params, request_id):
                pass

        requests = [
            asyncio.create_task(serve("call-1", "req-1")),
            # An uncaptured request, such as a validation call.
            asyncio.create_task(serve(None, "req-2")),
        ]
        while len(engine.output_processor.request_states) < 2:
            await asyncio.sleep(0)

        assert await client.captured_call_progress() == [["call-1", 200]]

        engine.release.set()
        await asyncio.gather(*requests)
        assert client._captured_requests == {}

    asyncio.run(exercise())


def test_an_engine_without_output_processor_state_reports_nothing() -> None:
    async def exercise() -> list:
        engine = SimpleNamespace(
            model_config=None, renderer=None, input_processor=None, vllm_config=None
        )
        client = _AsyncLLMHTTPClient(
            engine, asyncio.get_running_loop(), flat_logprobs=False
        )
        return await client.captured_call_progress()

    assert asyncio.run(exercise()) == []


@pytest.mark.parametrize("local_capture", [False, True])
@pytest.mark.parametrize("multiple_frontends", [False, True])
def test_worker_combines_frontend_progress(
    local_capture: bool, multiple_frontends: bool
) -> None:
    client = SimpleNamespace(
        captured_call_progress=AsyncMock(return_value=[["owner", 20]])
    )
    pool = SimpleNamespace(
        command=AsyncMock(
            return_value=[
                None,
                {"weight_version": 4, "calls": [["child", 30]]},
                {"weight_version": 4, "calls": []},
            ]
        )
    )
    worker = SimpleNamespace(
        _http_engine_client=client if local_capture else None,
        token_capture=object() if local_capture else None,
        _rollout_weight_version=4,
        _api_server_pool=pool if multiple_frontends else None,
    )
    result = asyncio.run(VllmAsyncGenerationWorkerImpl.captured_call_progress(worker))
    if local_capture or multiple_frontends:
        assert result == {
            "weight_version": 4,
            "calls": ([["owner", 20]] if local_capture else [])
            + ([["child", 30]] if multiple_frontends else []),
        }
    else:
        assert result is None
    if multiple_frontends:
        pool.command.assert_awaited_once_with("progress")
    else:
        pool.command.assert_not_awaited()


def test_worker_rejects_mixed_frontend_weight_versions() -> None:
    worker = SimpleNamespace(
        _http_engine_client=None,
        token_capture=None,
        _rollout_weight_version=4,
        _api_server_pool=SimpleNamespace(
            command=AsyncMock(return_value=[{"weight_version": 3, "calls": []}])
        ),
    )
    with pytest.raises(RuntimeError, match="different weight versions"):
        asyncio.run(VllmAsyncGenerationWorkerImpl.captured_call_progress(worker))


@pytest.mark.parametrize("multiple_frontends", [False, True])
def test_worker_collects_separate_frontend_snapshots(multiple_frontends: bool) -> None:
    owner = {"calls": 2, "by_op": {"put": {"n_bytes": 100, "step_max_ms": 2}}}
    child = {"calls": 3, "by_op": {"get": {"n_bytes": 200, "step_max_ms": 5}}}
    pool = SimpleNamespace(command=AsyncMock(return_value=[None, child]))
    worker = SimpleNamespace(
        get_data_plane_snapshot=Mock(return_value=owner),
        _api_server_pool=pool if multiple_frontends else None,
    )
    snapshots = asyncio.run(
        VllmAsyncGenerationWorkerImpl.get_data_plane_snapshots(worker)
    )
    assert snapshots == ([owner, child] if multiple_frontends else [owner])
    worker.get_data_plane_snapshot.assert_called_once_with()
    if multiple_frontends:
        pool.command.assert_awaited_once_with("snapshot")


def test_generation_flattens_frontend_snapshots() -> None:
    from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration

    snapshots = [{"calls": 2}, {"calls": 3}, {"calls": 5}]
    group = SimpleNamespace(run_all_workers_single_data=Mock(return_value="refs"))
    with patch(
        "nemo_rl.models.generation.vllm.vllm_generation.ray.get",
        return_value=[snapshots[:2], [], snapshots[2:]],
    ):
        assert (
            VllmGeneration.get_data_plane_snapshots(SimpleNamespace(worker_group=group))
            == snapshots
        )
    group.run_all_workers_single_data.assert_called_once_with(
        "get_data_plane_snapshots",
        run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
    )


@pytest.fixture
def multiprocess_metrics_reader(tmp_path, monkeypatch):
    """Initialize the pool's metrics reader without launching GPU processes."""
    from prometheus_client import CollectorRegistry, values
    from vllm.v1.metrics import reader

    from nemo_rl.models.generation.vllm.multi_api import MultiAPIServerPool

    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=1, data_parallel_backend="mp"
        ),
        scheduler_config=SimpleNamespace(max_num_queued_reqs=None),
    )
    engine_args = SimpleNamespace(
        create_engine_config=lambda: config, disable_log_stats=False
    )
    with (
        patch.object(reader, "REGISTRY", CollectorRegistry()),
        patch.object(values, "ValueClass"),
        patch(
            "ray.get_runtime_context", return_value=SimpleNamespace(namespace="test")
        ),
        patch(
            "vllm.v1.engine.utils.bind_engine_zmq_listeners",
            return_value=Mock(
                addresses=SimpleNamespace(inputs=["ipc://owner", "ipc://child"])
            ),
        ),
        patch("vllm.v1.executor.abstract.Executor.get_class", return_value=object),
        patch(
            "vllm.v1.engine.utils.launch_core_engines",
            side_effect=RuntimeError("stop before engine launch"),
        ),
    ):
        with pytest.raises(RuntimeError, match="stop before engine launch"):
            MultiAPIServerPool(SimpleNamespace(), 2).create_engine(engine_args, [])
        yield reader


def test_multiprocess_metrics_sum_tokens_and_keep_one_engine_gauge(
    tmp_path, multiprocess_metrics_reader
) -> None:
    from prometheus_client.mmap_dict import MmapedDict, mmap_key

    names = [
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:kv_cache_usage_perc",
    ]
    for pid, tokens, gauge_values in [
        (1, 20, [4, 8, 0.2]),
        (2, 30, [5, 9, 0.3]),
    ]:
        for kind, samples in [
            ("counter", [("vllm:generation_tokens", tokens)]),
            ("gauge_mostrecent", list(zip(names, gauge_values))),
        ]:
            data = MmapedDict(str(tmp_path / f"{kind}_{pid}.db"))
            try:
                for name, value in samples:
                    sample_name = name + "_total" if kind == "counter" else name
                    key = mmap_key(name, sample_name, ["engine"], ["0"], "test")
                    data.write_value(key, value, float(pid))
            finally:
                data.close()

    snapshot = multiprocess_metrics_reader.get_metrics_snapshot()
    assert {metric.name: metric.value for metric in snapshot} == {
        "vllm:generation_tokens": 50,
        "vllm:num_requests_running": 5,
        "vllm:num_requests_waiting": 9,
        "vllm:kv_cache_usage_perc": 0.3,
    }
    assert len(snapshot) == 4


def test_step_metrics_include_child_counters_and_request_outcomes(
    tmp_path, multiprocess_metrics_reader
) -> None:
    from prometheus_client.mmap_dict import MmapedDict, mmap_key

    from nemo_rl.models.generation.vllm.utils import (
        aggregate_spec_decode_counters,
        compute_engine_step_metrics,
    )

    worker = VllmAsyncGenerationWorkerImpl.__new__(VllmAsyncGenerationWorkerImpl)
    worker.llm = SimpleNamespace()
    start = aggregate_spec_decode_counters([worker._get_raw_spec_counters()])
    for pid, scale in [(1, 1), (2, 2)]:
        counter = MmapedDict(str(tmp_path / f"counter_{pid}.db"))
        histogram = MmapedDict(str(tmp_path / f"histogram_{pid}.db"))
        try:
            for name, labels, value in [
                ("vllm:generation_tokens", {}, 20),
                ("vllm:prompt_tokens", {}, 100),
                ("vllm:request_success", {"finished_reason": "stop"}, 2),
                ("vllm:request_success", {"finished_reason": "length"}, 1),
                ("vllm:request_success", {"finished_reason": "abort"}, 1),
                ("vllm:spec_decode_num_drafts", {}, 4),
                (
                    "vllm:spec_decode_num_accepted_tokens_per_pos",
                    {"position": "0"},
                    3,
                ),
                (
                    "vllm:spec_decode_num_accepted_tokens_per_pos",
                    {"position": "2"},
                    1,
                ),
            ]:
                key = mmap_key(
                    name,
                    name + "_total",
                    list(labels),
                    list(labels.values()),
                    "test",
                )
                counter.write_value(key, value * scale, 0.0)
            for name, total in [
                ("vllm:request_prompt_tokens", 100),
                ("vllm:request_generation_tokens", 20),
            ]:
                for part, value in [("sum", total), ("count", 2)]:
                    key = mmap_key(name, f"{name}_{part}", [], [], "test")
                    histogram.write_value(key, value * scale, 0.0)
                key = mmap_key(name, name + "_bucket", ["le"], ["+Inf"], "test")
                histogram.write_value(key, 2 * scale, 0.0)
        finally:
            counter.close()
            histogram.close()
    end = aggregate_spec_decode_counters([worker._get_raw_spec_counters()])

    assert compute_engine_step_metrics(start, end) == {
        "vllm/prompt_tokens": 300,
        "vllm/generation_tokens": 60,
        "vllm/prompt_length_mean": 50,
        "vllm/generation_length_mean": 10,
        "vllm/generations_ok": 9,
        "vllm/generations_failed": 3,
    }
    assert end["vllm:spec_decode_num_drafts"] == 12
    assert [
        end["vllm:spec_decode_num_accepted_tokens_per_pos", position]
        for position in (1, 2, 3)
    ] == [9, 0, 3]


def test_single_frontend_keeps_existing_step_counter_reader() -> None:
    worker = VllmAsyncGenerationWorkerImpl.__new__(VllmAsyncGenerationWorkerImpl)
    worker.llm = SimpleNamespace()
    with patch(
        "vllm.v1.metrics.reader.get_metrics_snapshot",
        return_value=[
            SimpleNamespace(name="vllm:generation_tokens", labels={}, value=20)
        ],
    ) as reader:
        assert worker._get_raw_spec_counters() == {"vllm:generation_tokens": 20}
    reader.assert_called_once_with()


def test_vllm_still_has_the_request_state_we_read() -> None:
    """The fakes above would pass a vLLM that renamed these; this checks the real ones.

    The read goes engine.output_processor.request_states[*].external_req_id and
    .detokenizer.num_output_tokens(), none of which is public API.
    """
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.engine.detokenizer import IncrementalDetokenizer
    from vllm.v1.engine.output_processor import OutputProcessor, RequestState

    assert "self.output_processor" in inspect.getsource(AsyncLLM.__init__)
    assert "self.request_states" in inspect.getsource(OutputProcessor.__init__)
    params = inspect.signature(RequestState.__init__).parameters
    assert "external_req_id" in params and "detokenizer" in params
    assert callable(IncrementalDetokenizer.num_output_tokens)
