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

The frontend's chat engine rendered and spliced each call; the adapter runs the
worker's own capture lifecycle on it, as the native chat endpoint does. Only
Gym's capture core, and in the media tests the worker's media snapshot, are
faked.
"""

import base64
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.nemo_gym, pytest.mark.vllm]
pytest.importorskip("dynamo._core")
pytest.importorskip("vllm")
# The vLLM lanes import this module without the nemo_gym extra.
pytest.importorskip("nemo_gym.token_id_capture.staging")

from dynamo.llm.exceptions import InvalidArgument  # noqa: E402
from nemo_gym._checkpoint.generation_cut import (  # noqa: E402
    GenerationCutInventory,
    GenerationCutPrefix,
)
from nemo_gym.token_id_capture import NG_COMMIT_COORDS_FIELD  # noqa: E402
from nemo_gym.token_id_capture.adapters.vllm import VLLMCaptureAdapter  # noqa: E402
from nemo_gym.token_id_capture.replay import (  # noqa: E402
    replay_context,
    summarize_replay,
)
from nemo_gym.token_id_capture.staging.capture import (  # noqa: E402
    RolloutTokenCapture,
)
from nemo_gym.token_id_capture.staging.records import CaptureAdmission  # noqa: E402
from vllm.multimodal.inputs import PlaceholderRange  # noqa: E402

from nemo_rl.data.captured_media import CapturedMedia  # noqa: E402
from nemo_rl.data_plane.tq_token_sink import ChainPrefixCache  # noqa: E402
from nemo_rl.models.generation.dynamo.vllm import (  # noqa: E402
    adapter as vllm_adapter,
)
from nemo_rl.models.generation.dynamo.vllm.generation_cut import (  # noqa: E402
    CUT_PREFIXES_KEY,
    GENERATION_CUT_ANNOTATION,
    INVENTORY_KEY,
)
from nemo_rl.models.generation.vllm.utils import R3_MISSING_ROUTE_SENTINEL  # noqa: E402
from nemo_rl.utils.routed_experts_codec import (  # noqa: E402
    routed_experts_tensor_metadata,
)
from tests.unit.models.generation.test_dynamo_vllm_adapter import (  # noqa: E402
    EOS,
    IMAGE_TOKEN,
    IMAGE_TOKENS,
    _chat,
    _collect,
    _completion,
    _Context,
    _FakeLLM,
    _gym_call,
    _host,
    _logprob,
    _output,
)
from tests.unit.models.generation.test_vllm_token_capture_hosting import (  # noqa: E402
    _active_call_inventory,
    _cuts,
    _MemorySink,
)

COORDS = {"staging_key": "rollout-1/c2"}
# Turn 2 of a rollout: the staged prefix [1, 5, 6, EOS] replaces the template's
# re-render of turn 1, [1, 7, EOS], before the new turn [3, 4].
PREFIX = [1, 5, 6, EOS]
TEMPLATE = [1, 7, EOS, 3, 4]
# Each side cut at its turn end: PREFIX[:3] + TEMPLATE[2:].
SPLICE = [3, 2]
SPLICED = [*PREFIX, 3, 4]
TOKEN_IN = CaptureAdmission(
    capture_key="rollout-1",
    model_call_id="c2",
    parent_call_id="c1",
    prev_len=len(PREFIX),
    mode="token_in",
    required_prefix_token_ids=PREFIX,
    parent_chain_hash="0" * 64,
)
TEXT = CaptureAdmission(capture_key="rollout-1", model_call_id="c1", mode="text")


class _Capture:
    """Gym's capture core: records begin/complete/fail."""

    adapter = VLLMCaptureAdapter()

    def __init__(self) -> None:
        self.begun: list[tuple] = []
        self.staged: list[dict] = []
        self.attachments: list[dict | None] = []
        self.failed: list[str] = []

    def begin_call(
        self,
        admission,
        *,
        prefix_token_ids,
        generation_cut,
        generation_cut_staging_keys,
        stream,
    ):
        del generation_cut_staging_keys
        self.begun.append((admission, prefix_token_ids, stream))
        return SimpleNamespace(
            admission=admission,
            model_call_id=admission.model_call_id,
            capture_key=admission.capture_key,
            generation_cut=generation_cut,
        )

    def complete_call_from_response(self, call, payload, attachments=None):
        self.staged.append(payload)
        self.attachments.append(attachments)
        return SimpleNamespace(model_dump=lambda: dict(COORDS))

    def fail_call(self, call, *, reason) -> None:
        self.failed.append(reason)


def _capturing_host(llm, **cfg):
    host = _host(llm, **cfg)
    host.token_capture = _Capture()
    host._chain_prefix = ChainPrefixCache()
    return host


def _captured_call(
    token_ids: list[int], admission: CaptureAdmission, top_logprobs: int = 0, **plan
) -> dict:
    """A captured Gym call, as the chat engine routes it after its render."""
    admission_json = admission.model_dump(mode="json")
    return _gym_call(
        token_ids,
        # Gym's dynamo_worker handler asks for the sampled tokens' logprobs.
        request=_chat(
            logprobs=True, top_logprobs=top_logprobs, ng_capture=admission_json
        ),
        admission=admission_json,
        serving_admission=admission_json,
        **plan,
    )


def test_a_captured_token_in_call_is_staged_from_the_engine() -> None:
    # vLLM routes every token but the last sampled one.
    routes = np.arange(7 * 2 * 2, dtype=np.int16).reshape(7, 2, 2)
    final = _output(
        SPLICED,
        _completion([10, 11], "stop", routed_experts=routes[6:]),
        prompt_routed_experts=routes[:6],
    )
    llm = _FakeLLM([final])
    host = _capturing_host(llm, vllm_kwargs={"enable_return_routed_experts": True})
    engine = vllm_adapter.DynamoVllmAdapter(host, leader_worker_idx=0)

    (chunk,) = _collect(
        engine,
        _captured_call(
            SPLICED,
            TOKEN_IN,
            prefix_len=len(PREFIX),
            template_token_ids=TEMPLATE,
            splice=SPLICE,
        ),
    )

    ((prompt, params, _, _),) = llm.calls
    assert prompt["prompt_token_ids"] == SPLICED
    assert params.logprobs == 0
    # The model's own prefix, as the frontend spliced it, begins the call.
    assert host.token_capture.begun == [(TOKEN_IN, PREFIX, False)]
    (payload,) = host.token_capture.staged
    assert payload["prompt_token_ids"] == SPLICED
    message = payload["choices"][0]["message"]
    assert message["generation_token_ids"] == [10, 11]
    assert message["generation_log_probs"] == [-0.5, -0.5]
    # As the native endpoint stages them: the full sequence's routes (none for
    # the last token) from the boundary row, the prefix's last token, on.
    (attachments,) = host.token_capture.attachments
    staged_routes = attachments["routed_experts"]
    expected = torch.full((5, 2, 2), R3_MISSING_ROUTE_SENTINEL, dtype=torch.int16)
    expected[:4] = torch.from_numpy(routes[3:7])
    assert torch.equal(staged_routes, expected)
    assert message["routed_experts"] == routed_experts_tensor_metadata(staged_routes)
    # Only the commit coordinates go back through the frontend, beside the text.
    assert chunk["engine_data"]["result"] == {NG_COMMIT_COORDS_FIELD: COORDS}
    assert "logprobs" not in chunk["engine_data"]["output"]
    assert len(host._capture_registry) == 0


def test_a_captured_call_stages_the_samplers_top_k_for_score_centering() -> None:
    completion = _completion([10, 11], "stop")
    completion.logprobs = [
        {10: _logprob(-0.5), 20: _logprob(-2.0, rank=2)},
        {11: _logprob(-0.7), 21: _logprob(-3.0, rank=2)},
    ]
    llm = _FakeLLM([_output([1, 3, 4], completion)])
    host = _capturing_host(llm)
    host._capture_top_logprobs = 2
    engine = vllm_adapter.DynamoVllmAdapter(host, leader_worker_idx=0)

    _collect(engine, _captured_call([1, 3, 4], TEXT, top_logprobs=2))

    ((_, params, _, _),) = llm.calls
    assert params.logprobs == 2
    (attachments,) = host.token_capture.attachments
    assert attachments["sampler_topk_token_ids"].tolist() == [[10, 20], [11, 21]]
    assert torch.equal(
        attachments["sampler_topk_logprobs"],
        torch.tensor([[-0.5, -2.0], [-0.7, -3.0]], dtype=torch.float32),
    )
    assert host.token_capture.failed == []


def test_an_undecided_candidate_admission_is_refused() -> None:
    history = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    candidate = CaptureAdmission(
        **{
            **TOKEN_IN.model_dump(),
            "mode": "candidate",
            "requires_empty_tool_content_equivalence": False,
            "candidate_replay": summarize_replay(
                replay_context(history, render_digest="b" * 64)
            ),
            "request_replay": replay_context(history, render_digest="b" * 64),
        }
    )
    llm = _FakeLLM()
    host = _capturing_host(llm)
    engine = vllm_adapter.DynamoVllmAdapter(host, leader_worker_idx=0)

    # The frontend's render decides a candidate; one reaching the engine skipped it.
    with pytest.raises(InvalidArgument, match="candidate"):
        _collect(engine, _captured_call(SPLICED, candidate))

    assert (llm.calls, host.token_capture.begun) == ([], [])


class _FailingLLM(_FakeLLM):
    async def generate(self, prompt, sampling_params, request_id, **kwargs):
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
    host = _capturing_host(llm)
    engine = vllm_adapter.DynamoVllmAdapter(host, leader_worker_idx=0)

    if error is None:
        assert _collect(engine, _captured_call([1, 3, 4], TEXT), context) == []
    else:
        with pytest.raises(InvalidArgument, match=error):
            _collect(engine, _captured_call([1, 3, 4], TEXT), context)

    assert host.token_capture.staged == []
    assert host.token_capture.failed == [reason]
    assert len(host._capture_registry) == 0


def _multimodal(placeholders: list[PlaceholderRange]) -> str:
    """The rendered engine input's media, as the chat engine ships it."""
    multimodal = {
        "type": "multimodal",
        "mm_kwargs": {"image": [f"pixels-{i}" for i in range(len(placeholders))]},
        "mm_hashes": {"image": [f"hash-{i}" for i in range(len(placeholders))]},
        "mm_placeholders": {"image": placeholders},
    }
    return base64.b64encode(pickle.dumps(multimodal)).decode()


def test_a_captured_media_call_keeps_the_placeholders_capture_moved() -> None:
    image = [IMAGE_TOKEN] * IMAGE_TOKENS
    template = [1, 7, EOS, *image, 4]
    spliced = [*PREFIX, *image, 4]
    llm = _FakeLLM([_output(spliced, _completion([10], "stop"))])
    host = _capturing_host(llm)
    host._capture_media = True
    moved = {"image": [PlaceholderRange(len(PREFIX), IMAGE_TOKENS)]}
    snapshots = []

    def capture_request_media(
        engine_input, *, admission, splice, verify_retained_media
    ):
        # As capture_processed_media, which moves the placeholders itself.
        snapshots.append((admission, splice, verify_retained_media))
        engine_input["mm_placeholders"] = moved
        return CapturedMedia((), None)

    host._capture_request_media = capture_request_media
    engine = vllm_adapter.DynamoVllmAdapter(host, leader_worker_idx=0)

    _collect(
        engine,
        _captured_call(
            spliced,
            TOKEN_IN,
            prefix_len=len(PREFIX),
            template_token_ids=template,
            splice=SPLICE,
            multimodal=_multimodal([PlaceholderRange(3, IMAGE_TOKENS)]),
            verify_retained_media=True,
        ),
    )

    ((admission, splice, verify),) = snapshots
    assert (admission, verify) == (TOKEN_IN, True)
    assert (splice.token_ids, splice.model_cut_end, splice.template_cut_start) == (
        spliced,
        *SPLICE,
    )
    ((engine_input, _, _, _),) = llm.calls
    assert engine_input["prompt_token_ids"] == spliced
    # Not remapped a second time.
    assert engine_input["mm_placeholders"] == moved


def test_a_captured_media_call_stages_the_media_the_engine_ran_on() -> None:
    processed = [1, *[IMAGE_TOKEN] * IMAGE_TOKENS, 3]
    llm = _FakeLLM([_output(processed, _completion([10], "stop"))])
    host = _capturing_host(llm)
    host._capture_media = True
    snapshots = []

    def capture_request_media(
        engine_input, *, admission, splice, verify_retained_media
    ):
        snapshots.append((list(engine_input["prompt_token_ids"]), admission, splice))
        return CapturedMedia((), None)

    host._capture_request_media = capture_request_media
    engine = vllm_adapter.DynamoVllmAdapter(host, leader_worker_idx=0)

    (chunk,) = _collect(
        engine,
        _captured_call(
            processed,
            TEXT,
            multimodal=_multimodal([PlaceholderRange(1, IMAGE_TOKENS)]),
        ),
    )

    # Media is snapshotted from the processed prompt the engine runs.
    assert snapshots == [(processed, TEXT, None)]
    (payload,) = host.token_capture.staged
    assert payload["prompt_token_ids"] == processed
    assert chunk["engine_data"]["result"] == {NG_COMMIT_COORDS_FIELD: COORDS}


def test_an_engine_answers_a_cut_with_the_calls_it_cut() -> None:
    sink = _MemorySink()
    host = _host(_FakeLLM())
    host._capture_sink = sink
    host.token_capture = RolloutTokenCapture(
        sink=sink, weight_version_fn=lambda: 0, adapter=VLLMCaptureAdapter()
    )
    _cuts(host)
    inventory, _ = _active_call_inventory(host, ("held", [12, 13]))
    # The frontend sends every engine the whole inventory: one call here,
    # one that another engine holds.
    inventory = GenerationCutInventory.build(
        checkpoint_id=inventory.checkpoint_id,
        server_name=inventory.server_name,
        active_prefixes=[
            *inventory.active_prefixes,
            GenerationCutPrefix(
                ticket_id="ticket-elsewhere",
                rollout_id="r9",
                attempt=0,
                model_call_id="elsewhere",
                admitted_at=1.0,
            ),
        ],
    )

    (chunk,) = _collect(
        vllm_adapter.DynamoVllmAdapter(host, leader_worker_idx=0),
        {
            "token_ids": [],
            "annotations": [GENERATION_CUT_ANNOTATION],
            "extra_args": {INVENTORY_KEY: inventory.model_dump(mode="json")},
        },
    )

    (ack,) = chunk["engine_data"][CUT_PREFIXES_KEY]
    assert (ack["model_call_id"], ack["disposition"]) == ("held", "durable_prefix")
    assert ack["prefix_token_count"] == 2
    # The cut's rows are in TQ before the engine answers.
    assert ack["staging_keys"] == sink.generation_prefix_keys
