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

"""CPU tests for the GPU-free slices of the megatron generation worker.

The end-to-end megatron generation tests need GPUs and several minutes, which
makes them a poor guard for the wire format of the engine's reply. That format
does change: megatron-core stopped echoing `prompt_tokens` back unless
`SamplingParams.return_prompt_tokens` is set, and every synchronous L1 functional
test began failing with `AttributeError: 'NoneType' object has no attribute
'tolist'` -- while two async ones swallowed the error per sample and still
reported success. These tests pin the packing logic without a GPU.

The same reasoning covers the NeMo-Gym port-reservation wiring
(test_http_server_port_reservation): its safety properties are pure
socket/plumbing contracts, pinned here without a GPU.
"""

import socket
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from megatron.core.inference.text_generation_server.dynamic_text_gen_server import (
    text_generation_server as mlm_text_gen_server,
)

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.held_port import (
    HeldPortReservation,
    receive_held_socket,
)
from nemo_rl.models.generation.megatron.megatron_worker import (
    MegatronGenerationMixin,
)

PAD = 0


class FakeInferenceReply:
    """The subset of mcore's DynamicInferenceRequest the parse path reads.

    `prompt_tokens` defaults to None to match what the inference client actually
    hands back: the engine drops it before serializing unless the caller opts in.
    """

    def __init__(
        self,
        generated_tokens: list[int],
        generated_log_probs: list[float],
        prompt_tokens: torch.Tensor | None = None,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.generated_tokens = generated_tokens
        self.generated_log_probs = generated_log_probs


def parse(data: BatchedDataDict, result: list[FakeInferenceReply]) -> BatchedDataDict:
    """Calls the mixin method unbound, so no engine or distributed setup is needed."""
    worker = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=PAD))
    return MegatronGenerationMixin._parse_result_to_batched_data_dict(
        worker, data, result
    )


@pytest.fixture
def two_sample_batch() -> BatchedDataDict:
    """Two right-padded prompts of different lengths (3 and 2 real tokens)."""
    return BatchedDataDict(
        {
            "input_ids": torch.tensor(
                [[5, 6, 7, PAD], [8, 9, PAD, PAD]], dtype=torch.long
            ),
            "input_lengths": torch.tensor([3, 2], dtype=torch.long),
        }
    )


@pytest.mark.mcore
def test_packs_replies_that_omit_prompt_tokens(two_sample_batch):
    """The regression: a reply with prompt_tokens=None must still pack correctly."""
    result = [
        FakeInferenceReply(generated_tokens=[11, 12], generated_log_probs=[-0.1, -0.2]),
        FakeInferenceReply(generated_tokens=[13], generated_log_probs=[-0.3]),
    ]

    out = parse(two_sample_batch, result)

    # Row width is the padded prompt length (4) plus the longest generation (2).
    torch.testing.assert_close(
        out["output_ids"],
        torch.tensor(
            [[5, 6, 7, 11, 12, PAD], [8, 9, 13, PAD, PAD, PAD]], dtype=torch.long
        ),
    )
    # Generation is placed at the true prompt length, not the padded length, so
    # sample 1's tokens must land at index 2 rather than 4.
    torch.testing.assert_close(out["generation_lengths"], torch.tensor([2, 1]))
    torch.testing.assert_close(out["unpadded_sequence_lengths"], torch.tensor([5, 3]))

    expected_logprobs = torch.zeros(2, 6)
    expected_logprobs[0, 3:5] = torch.tensor([-0.1, -0.2])
    expected_logprobs[1, 2:3] = torch.tensor([-0.3])
    torch.testing.assert_close(out["logprobs"], expected_logprobs)


@pytest.mark.mcore
def test_ignores_echoed_prompt_tokens(two_sample_batch):
    """An engine that does echo the prompt back must not change the output.

    The prompt is taken from the submitted batch either way, so the two mcore
    behaviours are indistinguishable downstream and no version branch is needed.
    """
    without_echo = [
        FakeInferenceReply(generated_tokens=[11, 12], generated_log_probs=[-0.1, -0.2]),
        FakeInferenceReply(generated_tokens=[13], generated_log_probs=[-0.3]),
    ]
    with_echo = [
        FakeInferenceReply(
            generated_tokens=[11, 12],
            generated_log_probs=[-0.1, -0.2],
            prompt_tokens=torch.tensor([5, 6, 7], dtype=torch.long),
        ),
        FakeInferenceReply(
            generated_tokens=[13],
            generated_log_probs=[-0.3],
            prompt_tokens=torch.tensor([8, 9], dtype=torch.long),
        ),
    ]

    baseline = parse(two_sample_batch, without_echo)
    echoed = parse(two_sample_batch, with_echo)

    for key in (
        "output_ids",
        "logprobs",
        "generation_lengths",
        "unpadded_sequence_lengths",
    ):
        torch.testing.assert_close(echoed[key], baseline[key])


@pytest.mark.mcore
def test_handles_a_reply_with_no_generated_tokens(two_sample_batch):
    """A request that produced nothing must yield the prompt and a zero length."""
    result = [
        FakeInferenceReply(generated_tokens=[], generated_log_probs=[]),
        FakeInferenceReply(generated_tokens=[13], generated_log_probs=[-0.3]),
    ]

    out = parse(two_sample_batch, result)

    torch.testing.assert_close(out["generation_lengths"], torch.tensor([0, 1]))
    torch.testing.assert_close(out["unpadded_sequence_lengths"], torch.tensor([3, 3]))
    assert out["output_ids"][0].tolist() == [5, 6, 7, PAD, PAD]


@pytest.mark.mcore
def test_http_server_port_reservation(monkeypatch):
    """The NeMo-Gym overlap contract, exercised back to back.

    The server URL is published to NeMo Gym before any worker exists, so: the
    reserved port is bound and listening from reservation time (early Gym
    probes queue instead of being refused), the worker adopts that same socket
    through the fd handoff — the port is never released in between — and the
    server falls back to a fresh port only when nothing was reserved.
    """
    # The holder resolves the node IP via held_port; the server resolves it via
    # virtual_cluster (megatron_worker imports it at call time). Patch both.
    monkeypatch.setattr(
        "nemo_rl.distributed.held_port._get_node_ip_local",
        lambda: "10.0.0.5",
    )
    monkeypatch.setattr(
        "nemo_rl.distributed.virtual_cluster._get_node_ip_local",
        lambda: "10.0.0.5",
    )
    holder = HeldPortReservation()
    node_ip, port = holder.address()
    assert node_ip == "10.0.0.5"

    # Held and listening from reservation time: early Gym probes queue
    # instead of being refused.
    with socket.create_connection(("127.0.0.1", port), timeout=5):
        pass

    # Worker-side adoption: the same live socket, duplicated across the
    # process boundary; still the same port, still accepting.
    reserved = receive_held_socket(port)
    try:
        assert reserved.getsockname()[1] == port
        with socket.create_connection(("127.0.0.1", port), timeout=5):
            pass

        # Server start with the network and MLM server stubbed out.
        started = {}
        monkeypatch.setattr(
            mlm_text_gen_server,
            "start_text_gen_server",
            lambda **kwargs: started.update(kwargs),
        )
        monkeypatch.setattr(
            "nemo_rl.distributed.virtual_cluster._get_free_port_local",
            lambda: 12345,
        )
        monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
        requests_mock = MagicMock()
        health_get = requests_mock.Session.return_value.__enter__.return_value.get
        health_get.return_value.status_code = 200
        monkeypatch.setattr(
            "nemo_rl.models.generation.megatron.megatron_worker.requests",
            requests_mock,
        )

        for reserved_socket, expected_port in ((reserved, port), (None, 12345)):
            worker = SimpleNamespace(
                coordinator_addr="tcp://127.0.0.1:5555",
                megatron_tokenizer=object(),
                rank=0,
                cfg={"generation": {"mcore_generation_config": {"parsers": []}}},
                _reserved_http_server_socket=reserved_socket,
            )
            base_url = MegatronGenerationMixin._setup_openai_api_server(worker)
            assert started["sock"] is reserved_socket
            assert started["server_port"] == expected_port
            assert base_url == f"http://10.0.0.5:{expected_port}/v1"
    finally:
        reserved.close()
