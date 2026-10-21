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
"""NeMo RL's engine host around a stand-in for Dynamo's decode handler."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import numpy as np
import pytest

pytest.importorskip("nemo_gym")
pytestmark = pytest.mark.nemo_gym

from nemo_gym.token_id_capture.adapters.vllm import VLLMCaptureAdapter  # noqa: E402
from nemo_gym.token_id_capture.staging.capture import (  # noqa: E402
    RolloutTokenCapture,
)
from nemo_gym.token_id_capture.staging.records import (  # noqa: E402
    StagedCallRecord,
    StageResult,
)

from nemo_rl.data_plane.tq_token_sink import StagedPrefix  # noqa: E402
from nemo_rl.models.generation.dynamo.engine_host import (  # noqa: E402
    CAPTURE_ANNOTATION_PREFIX,
    PREFIX_ANNOTATION_PREFIX,
    ROLLOUT_ANNOTATION,
    EngineRequestError,
    NemoRLEngineHostMixin,
)

_EOS = 2
_MAX_MODEL_LEN = 16
_PROFILE = {"temperature": 0.7, "top_p": 0.95, "val_temperature": 0.6, "val_top_p": 1.0}
# As Dynamo's frontend delivers the profile's sampling: in f32.
_SAMPLING = {
    "temperature": float(np.float32(0.7)),
    "top_p": float(np.float32(0.95)),
}


class _MemorySink:
    def __init__(self) -> None:
        self.records: list[StagedCallRecord] = []

    def stage(
        self, record: StagedCallRecord, *, attachments: dict | None = None
    ) -> StageResult:
        self.records.append(record)
        return StageResult(ok=True, staging_key=record.staging_key)


class _PrefixSource:
    def fetch_prefix_chains(self, staging_keys: list[str]) -> StagedPrefix:
        assert staging_keys == ["r0/c1"]
        return StagedPrefix(expanded=[5, 6, _EOS])


class _Host(NemoRLEngineHostMixin):
    def __init__(self, sink: _MemorySink) -> None:
        self._init_nemo_rl_engine_host(
            max_model_len=_MAX_MODEL_LEN,
            eos_token_id=_EOS,
            sampling_profile=json.dumps(_PROFILE),
        )
        self.token_capture = RolloutTokenCapture(
            sink=sink,
            weight_version_fn=lambda: self._rollout_weight_version,
            adapter=VLLMCaptureAdapter(),
        )
        self._chain_prefix.install(_PrefixSource())


class _Context:
    def __init__(self) -> None:
        self.stopped = False

    def is_stopped(self) -> bool:
        return self.stopped

    def is_killed(self) -> bool:
        return False

    def id(self) -> str:
        return "request-0"


def _request(token_ids: list[int], *, admission: dict | None) -> dict[str, Any]:
    annotations = [ROLLOUT_ANNOTATION]
    if admission is not None:
        annotations.append(CAPTURE_ANNOTATION_PREFIX + json.dumps(admission))
    return {
        "token_ids": token_ids,
        "annotations": annotations,
        "sampling_options": dict(_SAMPLING),
        "stop_conditions": {"max_tokens": 100},
    }


def _generate(
    seen: list[dict], *, finish_reason: str = "stop"
) -> tuple[Any, list[dict]]:
    """Dynamo's handler as the host sees it: chunks, engine_data on the last."""
    yielded: list[dict] = []

    async def generate(request, context):
        seen.append(request)
        for chunk in (
            {"index": 0, "token_ids": [8], "log_probs": [-0.5]},
            {
                "index": 0,
                "token_ids": [_EOS],
                "log_probs": [-0.25],
                "finish_reason": finish_reason,
                "engine_data": {
                    "completion_token_ids": [8, _EOS],
                    "completion_logprobs": [-0.5, -0.25],
                    "prompt_token_ids": list(request["token_ids"]),
                    "finished": True,
                },
            },
        ):
            yielded.append(chunk)
            yield chunk

    return generate, yielded


def _run(host: _Host, request: dict, generate: Any, context: _Context) -> list[dict]:
    async def collect() -> list[dict]:
        return [
            chunk async for chunk in host._nemo_rl_generate(request, context, generate)
        ]

    return asyncio.run(collect())


@pytest.mark.parametrize(
    ("admission", "token_ids", "finish_reason", "engine_prompt", "staged_delta"),
    [
        pytest.param(
            {"rollout_id": "r0", "model_call_id": "c1", "mode": "text"},
            [5, 6],
            "stop",
            [5, 6],
            [5, 6, 8, _EOS],
            id="text-root",
        ),
        # The frontend re-rendered the first turn's output [6] as [9, 9], so the
        # boundary sits at the EOS, not at the staged prefix's length. The engine
        # runs the staged tokens [5, 6, EOS] and stages only the new suffix.
        pytest.param(
            {
                "rollout_id": "r0",
                "model_call_id": "c2",
                "parent_call_id": "c1",
                "prev_len": 3,
                "mode": "token_in",
                "staging_chain": ["r0/c1"],
                "parent_chain_hash": "1" * 64,
            },
            [5, 9, 9, _EOS, 11, 12],
            "stop",
            [5, 6, _EOS, 11, 12],
            [11, 12, 8, _EOS],
            id="token-in-splice",
        ),
        pytest.param(
            {"rollout_id": "r0", "model_call_id": "c1", "mode": "text"},
            [5, 6],
            "cancelled",
            [5, 6],
            None,
            id="cancelled-fails",
        ),
    ],
)
def test_captured_call_stages_from_the_engine_and_returns_only_coords(
    admission, token_ids, finish_reason, engine_prompt, staged_delta
) -> None:
    sink = _MemorySink()
    host = _Host(sink)
    asyncio.run(host.nrl_set_rollout_weight_version({"version": 7}))
    seen: list[dict] = []
    generate, _ = _generate(seen, finish_reason=finish_reason)

    chunks = _run(host, _request(token_ids, admission=admission), generate, _Context())

    [engine_request] = seen
    assert engine_request["token_ids"] == engine_prompt
    # Output clamped against the prompt the engine actually runs.
    assert engine_request["stop_conditions"]["max_tokens"] == (
        _MAX_MODEL_LEN - len(engine_prompt)
    )
    assert engine_request["output_options"]["logprobs"] == 0
    # Token arrays and logprobs stay in the engine; only coordinates leave it.
    assert all("log_probs" not in chunk for chunk in chunks)
    final_engine_data = chunks[-1]["engine_data"]
    assert not {"completion_token_ids", "completion_logprobs", "prompt_token_ids"} & (
        final_engine_data.keys()
    )
    assert host._capture_calls == {}
    if staged_delta is None:
        assert sink.records == []
        assert "ng_commit_coords" not in final_engine_data
        return
    [record] = sink.records
    assert record.token_ids_delta == staged_delta
    assert record.weight_version == 7
    assert final_engine_data["ng_commit_coords"]["disposition"] == "staged"


@pytest.mark.parametrize(
    ("request_overrides", "annotated", "expected", "engine_prompt"),
    [
        pytest.param({}, False, "untouched", [5, 6], id="non-gym-passthrough"),
        pytest.param({}, True, "clamped", [5, 6], id="rollout-clamped"),
        # Without capture, the last echoed turn's tokens replace the frontend's
        # re-render of it, as the native server's required_prefix_token_ids does.
        pytest.param(
            {
                "token_ids": [5, 9, 9, _EOS, 11, 12],
                "annotations": [
                    ROLLOUT_ANNOTATION,
                    f"{PREFIX_ANNOTATION_PREFIX}[5,6,{_EOS}]",
                ],
            },
            True,
            "clamped",
            [5, 6, _EOS, 11, 12],
            id="echoed-prefix-spliced",
        ),
        pytest.param(
            {"token_ids": list(range(_MAX_MODEL_LEN))},
            True,
            "maximum context length",
            None,
            id="overflow",
        ),
        pytest.param(
            {"sampling_options": {**_SAMPLING, "top_k": 5}},
            True,
            "Top k",
            None,
            id="off-policy",
        ),
        pytest.param(
            {"stop_conditions": {"max_tokens": 100, "stop": ["\n"]}},
            True,
            "stop strings",
            None,
            id="stop-strings",
        ),
    ],
)
def test_rollout_requests_get_native_checks_and_clamp(
    request_overrides, annotated, expected, engine_prompt
) -> None:
    host = _Host(_MemorySink())
    request = _request([5, 6], admission=None)
    request.update(request_overrides)
    if not annotated:
        request["annotations"] = []
        request["sampling_options"] = {"temperature": 0.0, "top_k": 1}
    seen: list[dict] = []
    generate, yielded = _generate(seen)

    if engine_prompt is None:
        with pytest.raises(EngineRequestError, match=expected) as error:
            _run(host, request, generate, _Context())
        assert error.value.status_code == 400
        assert seen == []
        return
    chunks = _run(host, request, generate, _Context())
    assert chunks == yielded  # without capture, engine_data reaches the client
    [engine_request] = seen
    assert engine_request["token_ids"] == engine_prompt
    if expected == "untouched":
        assert engine_request["stop_conditions"]["max_tokens"] == 100
        assert engine_request["sampling_options"] == {"temperature": 0.0, "top_k": 1}
    else:
        assert engine_request["stop_conditions"]["max_tokens"] == (
            _MAX_MODEL_LEN - len(engine_prompt)
        )
        # As the native server: Dynamo must not fill top_k from generation_config.json.
        assert engine_request["sampling_options"]["top_k"] == -1
    assert "output_options" not in engine_request
    assert chunks[-1]["engine_data"]["completion_token_ids"] == [8, _EOS]
