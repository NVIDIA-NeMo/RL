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

import contextlib
import importlib
import importlib.machinery
import sys
import types

import torch


def _stub_optional_module(name: str) -> None:
    try:
        importlib.import_module(name)
    except ModuleNotFoundError:
        module = types.ModuleType(name)
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        sys.modules[name] = module


for _module_name in ("decord", "mlflow", "swanlab", "wandb"):
    _stub_optional_module(_module_name)

from nemo_rl.models.huggingface.common import FlashAttentionKwargs  # noqa: E402
from nemo_rl.models.policy.workers import dtensor_policy_worker  # noqa: E402


class _FakeTokenizer:
    eos_token_id = 0


class _FakeBatch(dict):
    def get_multimodal_dict(self, *, as_tensors, device):
        return {}


class _FakePackableData(dict):
    size = 2

    def to(self, device):
        return self

    def make_microbatch_iterator_for_packable_sequences(self):
        yield _FakeBatch(self)

    def get_microbatch_iterator_for_packable_sequences_len(self):
        return 1, int(self["input_lengths"].max().item())


class _FakeModel:
    def __init__(self):
        self.forward_kwargs = None

    def eval(self):
        pass

    def __call__(self, **kwargs):
        self.forward_kwargs = kwargs
        input_ids = kwargs["input_ids"]
        vocab_size = int(input_ids.max().item()) + 3
        logits = torch.zeros(
            input_ids.shape[0], input_ids.shape[1], vocab_size, dtype=torch.float32
        )
        return types.SimpleNamespace(logits=logits)


def test_get_logprobs_packed_sequences_passes_flash_attention_kwargs(monkeypatch):
    worker = object.__new__(dtensor_policy_worker.DTensorPolicyWorkerImpl)
    worker.cfg = {
        "dynamic_batching": {"enabled": False},
        "logprob_batch_size": 2,
        "logprob_chunk_size": None,
    }
    worker.cp_size = 1
    worker.dtype = torch.float32
    worker.enable_seq_packing = True
    worker.model = _FakeModel()
    worker.sampling_params = None
    worker.tokenizer = _FakeTokenizer()

    data = _FakePackableData(
        {
            "input_ids": torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
            "input_lengths": torch.tensor([3, 2]),
        }
    )

    original_tensor = torch.tensor

    def tensor_on_cpu(*args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs = dict(kwargs)
            kwargs["device"] = "cpu"
        return original_tensor(*args, **kwargs)

    monkeypatch.setattr(
        dtensor_policy_worker,
        "unshard_fsdp2_model",
        lambda _model: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        dtensor_policy_worker.torch,
        "autocast",
        lambda **_kwargs: contextlib.nullcontext(),
    )
    monkeypatch.setattr(dtensor_policy_worker.torch, "tensor", tensor_on_cpu)
    monkeypatch.setattr(dtensor_policy_worker.torch.Tensor, "cuda", lambda self: self)
    monkeypatch.setattr(
        dtensor_policy_worker.torch.distributed,
        "all_reduce",
        lambda *_args, **_kwargs: None,
    )

    result = worker.get_logprobs(data)

    flash_attn_kwargs = worker.model.forward_kwargs["flash_attn_kwargs"]
    assert isinstance(flash_attn_kwargs, FlashAttentionKwargs)
    assert flash_attn_kwargs.cu_seqlens_q.tolist() == [0, 3, 5]
    assert worker.model.forward_kwargs["attention_mask"] is None
    assert result["logprobs"].shape == (2, 4)
