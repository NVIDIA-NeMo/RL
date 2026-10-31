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
"""NeMo-Gym token capture through the Dynamo engine adapter.

The worker's own capture lifecycle stages each call, as on the native chat
endpoint; only Gym's capture core, and in the media tests the worker's media
snapshot, are faked.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.nemo_gym, pytest.mark.vllm]
pytest.importorskip("dynamo._core")
pytest.importorskip("vllm")
# The vLLM lanes import this module without the nemo_gym extra.
pytest.importorskip("nemo_gym.token_id_capture.staging")

from nemo_gym.token_id_capture import NG_COMMIT_COORDS_FIELD  # noqa: E402
from nemo_gym.token_id_capture.adapters.vllm import VLLMCaptureAdapter  # noqa: E402
from nemo_gym.token_id_capture.external_capture import (  # noqa: E402
    DYNAMO_CAPTURE_ANNOTATION_PREFIX,
    DYNAMO_GYM_REQUEST_ANNOTATION,
    DYNAMO_PREFIX_ANNOTATION_PREFIX,
)
from nemo_gym.token_id_capture.staging.records import CaptureAdmission  # noqa: E402
from responses_api_models.vllm_model.app import (  # noqa: E402
    _DYNAMO_ENGINE_DATA_FIELDS,
)
from vllm.multimodal.inputs import PlaceholderRange  # noqa: E402

from nemo_rl.data.captured_media import CapturedMedia  # noqa: E402
from nemo_rl.data_plane.tq_token_sink import ChainPrefixCache  # noqa: E402
from nemo_rl.models.generation.dynamo import native_engine  # noqa: E402
from nemo_rl.models.generation.vllm.utils import R3_MISSING_ROUTE_SENTINEL  # noqa: E402
from nemo_rl.utils.routed_experts_codec import decode_routed_experts  # noqa: E402
from tests.unit.models.generation.test_dynamo_native_engine import (  # noqa: E402
    EOS,
    IMAGE_PLACEHOLDER,
    IMAGE_TOKEN,
    IMAGE_TOKENS,
    _collect,
    _Context,
    _completion,
    _FakeLLM,
    _gym_request,
    _host,
    _image_url,
    _output,
    _Tokenizer,
)

COORDS = {"staging_key": "rollout-1/c2"}


class _Capture:
    """Gym's capture core: records begin/complete/fail."""

    adapter = VLLMCaptureAdapter()

    def __init__(self) -> None:
        self.begun: list[tuple] = []
        self.staged: list[dict] = []
        self.failed: list[str] = []

    def begin_call(self, admission, *, prefix_token_ids, stream):
        self.begun.append((admission, prefix_token_ids, stream))
        return SimpleNamespace(admission=admission)

    def complete_call_from_response(self, call, payload, attachments=None):
        self.staged.append(payload)
        return SimpleNamespace(model_dump=lambda: dict(COORDS))

    def fail_call(self, call, *, reason) -> None:
        self.failed.append(reason)


def _capturing_host(llm, **cfg):
    host = _host(llm, **cfg)
    host.token_capture = _Capture()
    host._capture_calls = {}
    host._chain_prefix = ChainPrefixCache()
    return host


def _captured_request(admission: CaptureAdmission, **fields) -> dict:
    request = _gym_request(**fields)
    # Gym's dynamo_worker handler: no logprobs, the admission in an annotation.
    request["output_options"] = {}
    request["annotations"] = [
        DYNAMO_CAPTURE_ANNOTATION_PREFIX + admission.model_dump_json(),
        *request["annotations"],
    ]
    return request


def test_annotations_and_engine_data_are_the_ones_gyms_dynamo_dialect_uses() -> None:
    assert native_engine.GYM_ROLLOUT_ANNOTATION == DYNAMO_GYM_REQUEST_ANNOTATION
    assert native_engine.GYM_CAPTURE_ANNOTATION_PREFIX == (
        DYNAMO_CAPTURE_ANNOTATION_PREFIX
    )
    assert native_engine.GYM_PREFIX_ANNOTATION_PREFIX == (
        DYNAMO_PREFIX_ANNOTATION_PREFIX
    )
    # The engine_data keys Gym requires on a call without capture.
    llm = _FakeLLM([_output([1, 3, 4], _completion([10], "stop"))])
    engine = native_engine.NativeVllmEngine(_host(llm), leader_worker_idx=0)
    (chunk,) = _collect(engine, _gym_request())
    assert chunk["engine_data"].keys() == _DYNAMO_ENGINE_DATA_FIELDS.keys()


def test_a_captured_token_in_call_is_staged_from_the_engine() -> None:
    # The staged prefix [1, 5, 6, EOS] replaces the rendered history [1, 7, EOS].
    prefix = [1, 5, 6, EOS]
    admission = CaptureAdmission(
        capture_key="rollout-1",
        model_call_id="c2",
        parent_call_id="c1",
        prev_len=len(prefix),
        mode="token_in",
        required_prefix_token_ids=prefix,
        parent_chain_hash="0" * 64,
    )
    spliced = [*prefix, 3, 4]
    routes = np.arange(8 * 2 * 2, dtype=np.int16).reshape(8, 2, 2)
    final = _output(
        spliced,
        _completion([10, 11], "stop", routed_experts=routes[6:]),
        prompt_routed_experts=routes[:6],
    )
    llm = _FakeLLM([final])
    llm.tokenizer = _Tokenizer()
    host = _capturing_host(llm, vllm_kwargs={"enable_return_routed_experts": True})
    engine = native_engine.NativeVllmEngine(host, leader_worker_idx=0)

    (chunk,) = _collect(
        engine, _captured_request(admission, token_ids=[1, 7, EOS, 3, 4])
    )

    ((prompt, params, _),) = llm.calls
    assert prompt["prompt_token_ids"] == spliced
    # The sampled tokens' log probabilities are staged; Gym asked for none.
    assert params.logprobs == 0
    ((begun_admission, begun_prefix, stream),) = host.token_capture.begun
    assert (begun_admission, begun_prefix, stream) == (admission, prefix, False)
    (payload,) = host.token_capture.staged
    assert payload["prompt_token_ids"] == spliced
    message = payload["choices"][0]["message"]
    assert message["generation_token_ids"] == [10, 11]
    assert message["generation_log_probs"] == [-0.5, -0.5]
    # As the native endpoint stages them: the full sequence's routes (none for
    # the last token) from the boundary row, the prefix's last token, on.
    staged_routes = decode_routed_experts(message["routed_experts"], torch.int16)
    expected = torch.full((5, 2, 2), R3_MISSING_ROUTE_SENTINEL, dtype=torch.int16)
    expected[:4] = torch.from_numpy(routes[3:7])
    assert torch.equal(staged_routes, expected)
    # Only the commit coordinates go back through the frontend.
    assert chunk["engine_data"] == {NG_COMMIT_COORDS_FIELD: COORDS}
    assert "log_probs" not in chunk
    assert host._capture_calls == {}


class _FailingLLM(_FakeLLM):
    async def generate(self, prompt, sampling_params, request_id):
        raise RuntimeError("engine died")
        yield  # pragma: no cover


class _Disconnected(_Context):
    def is_stopped(self) -> bool:
        return True


@pytest.mark.parametrize(
    ("llm", "context", "error", "reason"),
    [
        (_FailingLLM(), _Context(), "engine died", "engine_error"),
        # As vLLM's chat server: a call vLLM ended with an error is never staged.
        (
            _FakeLLM([_output([1, 3, 4], _completion([10], "error"))]),
            _Context(),
            "Generation failed",
            "engine_error",
        ),
        # Dynamo aborts the vLLM request on a disconnect but lets the generator finish.
        (
            _FakeLLM([_output([1, 3, 4], _completion([10], "stop"))]),
            _Disconnected(),
            None,
            "client_disconnect",
        ),
    ],
    ids=["engine_error", "finished_with_error", "client_disconnect"],
)
def test_a_call_that_does_not_finish_cleanly_fails_its_capture(
    llm, context, error, reason
) -> None:
    admission = CaptureAdmission(
        capture_key="rollout-1", model_call_id="c1", mode="text"
    )
    host = _capturing_host(llm)
    engine = native_engine.NativeVllmEngine(host, leader_worker_idx=0)

    if error is None:
        assert _collect(engine, _captured_request(admission), context) == []
    else:
        with pytest.raises(RuntimeError, match=error):
            _collect(engine, _captured_request(admission), context)

    assert host.token_capture.staged == []
    assert host.token_capture.failed == [reason]
    assert host._capture_calls == {}


def test_a_captured_media_call_keeps_the_placeholders_capture_moved() -> None:
    # The staged prefix [1, 5, 6, EOS] replaces the rendered history [1, 7, EOS].
    prefix = [1, 5, 6, EOS]
    admission = CaptureAdmission(
        capture_key="rollout-1",
        model_call_id="c2",
        parent_call_id="c1",
        prev_len=len(prefix),
        mode="token_in",
        required_prefix_token_ids=prefix,
        parent_chain_hash="0" * 64,
    )
    spliced = [*prefix, *[IMAGE_TOKEN] * IMAGE_TOKENS, 4]
    llm = _FakeLLM([_output(spliced, _completion([10], "stop"))])
    llm.tokenizer = _Tokenizer()
    host = _capturing_host(llm)
    host._capture_media = True
    moved = {"image": [PlaceholderRange(len(prefix), IMAGE_TOKENS)]}

    def capture_request_media(engine_input, *, admission, splice):
        # As capture_processed_media, which moves the placeholders itself.
        engine_input["mm_placeholders"] = moved
        return CapturedMedia((), None)

    host._capture_request_media = capture_request_media
    engine = native_engine.NativeVllmEngine(host, leader_worker_idx=0)

    _collect(
        engine,
        _captured_request(
            admission,
            token_ids=[1, 7, EOS, IMAGE_PLACEHOLDER, 4],
            multi_modal_data={"image_url": [{"Url": _image_url()}]},
        ),
    )

    ((engine_input, _, _),) = llm.calls
    assert engine_input["prompt_token_ids"] == spliced
    # Not remapped a second time.
    assert engine_input["mm_placeholders"] == moved


def test_a_captured_media_call_stages_the_media_the_engine_ran_on() -> None:
    admission = CaptureAdmission(
        capture_key="rollout-1", model_call_id="c1", mode="text"
    )
    processed = [1, *[IMAGE_TOKEN] * IMAGE_TOKENS, 3]
    llm = _FakeLLM([_output(processed, _completion([10], "stop"))])
    host = _capturing_host(llm)
    host._capture_media = True
    captured = CapturedMedia((), None)
    snapshots = []

    def capture_request_media(engine_input, *, admission, splice):
        snapshots.append((list(engine_input["prompt_token_ids"]), admission, splice))
        return captured

    host._capture_request_media = capture_request_media
    engine = native_engine.NativeVllmEngine(host, leader_worker_idx=0)

    (chunk,) = _collect(
        engine,
        _captured_request(
            admission,
            token_ids=[1, IMAGE_PLACEHOLDER, 3],
            multi_modal_data={"image_url": [{"Url": _image_url()}]},
        ),
    )

    # The processor's own pixels, not cache references, as natively (#4124).
    ((_, skip_mm_cache),) = llm.renderer.prompts
    assert skip_mm_cache is True
    # Media is snapshotted from the processed prompt the engine runs.
    assert snapshots == [(processed, admission, None)]
    (payload,) = host.token_capture.staged
    assert payload["prompt_token_ids"] == processed
    assert chunk["engine_data"] == {NG_COMMIT_COORDS_FIELD: COORDS}
