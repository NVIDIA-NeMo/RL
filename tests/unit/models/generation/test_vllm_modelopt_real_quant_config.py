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

import os
import types
from contextlib import contextmanager

import pytest
import torch

from nemo_rl.modelopt.models.generation.vllm_quant_backend import (
    VllmQuantInternalWorkerExtension,
)
from nemo_rl.modelopt.models.generation.vllm_quant_worker import (
    _configure_quant_engine_kwargs,
)
from nemo_rl.modelopt.utils import configure_modelopt_real_quant_generation
from nemo_rl.models.policy.utils import IPCProtocol
from nemo_rl.models.generation.vllm.utils import resolve_generation_worker_cls


def _valid_configs():
    policy = {
        "quant_cfg": "/tmp/modelopt-recipe.yaml",
        "megatron_cfg": {"enabled": True},
    }
    generation = {
        "backend": "vllm",
        "quant_cfg": None,
        "real_quant": True,
        "vllm_cfg": {
            "enforce_eager": True,
            "kv_cache_dtype": "auto",
        },
        "vllm_kwargs": {"hf_overrides": {"rope_theta": 1234}},
    }
    descriptor = {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "config_groups": {"group_0": {"targets": ["Linear"]}},
    }
    return policy, generation, descriptor


def test_configure_real_quant_passes_modelopt_descriptor_through():
    policy, generation, descriptor = _valid_configs()

    configure_modelopt_real_quant_generation(policy, generation, descriptor)

    hf_overrides = generation["vllm_kwargs"]["hf_overrides"]
    assert hf_overrides["rope_theta"] == 1234
    assert hf_overrides["quantization_config"] == descriptor
    assert hf_overrides["quantization_config"] is not descriptor


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda _p, g: g.update(quant_cfg="recipe"), "generation.quant_cfg"),
        (lambda _p, g: g.update(refit_transport="nixl"), "default IPC or collective"),
        (
            lambda _p, g: g["vllm_cfg"].update(enforce_eager=False),
            "enforce_eager=true",
        ),
        (
            lambda _p, g: g["vllm_cfg"].update(kv_cache_dtype="fp8"),
            "kv_cache_dtype=auto",
        ),
        (
            lambda _p, g: g["vllm_kwargs"].update(quantization="modelopt"),
            "vllm_kwargs.quantization",
        ),
        (
            lambda _p, g: g["vllm_kwargs"].update(speculative_config={}),
            "speculative decoding",
        ),
    ],
)
def test_configure_real_quant_rejects_unsupported_runtime_options(mutate, match):
    policy, generation, descriptor = _valid_configs()
    mutate(policy, generation)

    with pytest.raises(ValueError, match=match):
        configure_modelopt_real_quant_generation(policy, generation, descriptor)


def test_real_quant_worker_uses_policy_descriptor_without_custom_quant_class(
    monkeypatch,
):
    _, generation, descriptor = _valid_configs()
    llm_kwargs = {"hf_overrides": {"quantization_config": descriptor}}
    monkeypatch.setattr(
        "nemo_rl.modelopt.models.generation.vllm_quant_worker."
        "checkpoint_engine_refit_config",
        lambda _cfg: None,
    )
    monkeypatch.delenv("VLLM_MODELOPT_REAL_QUANT", raising=False)

    _configure_quant_engine_kwargs(generation, llm_kwargs)

    assert os.environ["VLLM_MODELOPT_REAL_QUANT"] == "1"
    assert llm_kwargs["hf_overrides"]["quantization_config"] is descriptor
    assert "quantization" not in llm_kwargs
    assert "worker_cls" not in llm_kwargs
    assert llm_kwargs["worker_extension_cls"].endswith(
        ".VllmQuantInternalWorkerExtension"
    )


def test_real_quant_selects_modelopt_vllm_worker_without_generation_quant_cfg():
    default = "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker"

    resolved = resolve_generation_worker_cls(
        default,
        {"quant_cfg": None, "real_quant": True},
    )

    assert resolved.endswith(".VllmQuantGenerationWorker")


def test_real_quant_uses_vllm_full_model_reload_lifecycle(monkeypatch):
    extension = object.__new__(VllmQuantInternalWorkerExtension)
    model = torch.nn.Linear(2, 2)
    vllm_config = object()
    model_config = object()
    extension.model_runner = types.SimpleNamespace(
        model=model,
        vllm_config=vllm_config,
    )
    extension.model_config = model_config
    extension.device = torch.device("cpu")
    monkeypatch.setattr(extension, "_is_real_quant_model", lambda: True)

    events = []

    @contextmanager
    def set_current(config):
        events.append(("enter", config))
        yield
        events.append(("exit", config))

    monkeypatch.setattr("vllm.config.set_current_vllm_config", set_current)
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.reload.initialize_layerwise_reload",
        lambda target: events.append(("initialize", target)),
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.reload.finalize_layerwise_reload",
        lambda target, config: events.append(("finalize", target, config)),
    )
    monkeypatch.setattr(
        torch.accelerator,
        "synchronize",
        lambda: events.append(("synchronize",)),
    )

    with extension._weight_update_lifecycle("collective") as finalize:
        finalize()

    assert ("initialize", model) in events
    assert ("finalize", model, model_config) in events
    assert events.count(("synchronize",)) == 1


def _real_quant_extension(monkeypatch):
    extension = object.__new__(VllmQuantInternalWorkerExtension)
    extension.model_runner = types.SimpleNamespace(
        model=torch.nn.Linear(2, 2),
        vllm_config=object(),
    )
    extension.model_config = object()
    extension.device = torch.device("cpu")
    monkeypatch.setenv("VLLM_MODELOPT_REAL_QUANT", "1")

    @contextmanager
    def set_current(_config):
        yield

    monkeypatch.setattr(
        "vllm.config.set_current_vllm_config",
        set_current,
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.reload.initialize_layerwise_reload",
        lambda _model: None,
    )
    return extension


def test_real_quant_ipc_load_failure_acks_and_propagates(monkeypatch):
    from nemo_rl.models.generation.vllm import vllm_backend

    class FakeSocket:
        def __init__(self):
            self.received = iter(
                [
                    (object(), ["model.weight"], 4),
                    IPCProtocol.COMPLETE,
                ]
            )
            self.sent = []

        def recv_pyobj(self):
            return next(self.received)

        def send(self, payload):
            self.sent.append(payload)

    extension = _real_quant_extension(monkeypatch)
    extension.state_dict_info = {"model.weight": (torch.Size([1]), torch.float32)}
    extension.zmq_socket = FakeSocket()
    extension.maybe_init_zmq = lambda: None
    extension._synchronize_before_ipc_data_ack = lambda: None

    def fail_load(_weights):
        raise RuntimeError("load failed")

    extension._load_weights = fail_load
    monkeypatch.setattr(
        vllm_backend,
        "rebuild_cuda_tensor_from_ipc",
        lambda _handle, _device_index: torch.zeros(4, dtype=torch.uint8),
    )
    monkeypatch.setattr(vllm_backend, "calculate_aligned_size", lambda size: size)

    with pytest.raises(RuntimeError, match="refit rejected"):
        extension.update_weights_via_ipc_zmq()

    assert extension.zmq_socket.sent == [
        IPCProtocol.ACK.value.encode(),
        IPCProtocol.ACK.value.encode(),
    ]


def test_real_quant_collective_load_failure_propagates(monkeypatch):
    from nemo_rl.models.generation.vllm import vllm_backend

    extension = _real_quant_extension(monkeypatch)
    extension.state_dict_info = {"model.weight": object()}
    extension.model_update_group = object()

    def fail_load(_weights):
        raise RuntimeError("load failed")

    extension._load_weights = fail_load

    def consume(**kwargs):
        kwargs["post_unpack_func"]([("model.weight", torch.ones(1))])

    monkeypatch.setattr(
        vllm_backend,
        "packed_broadcast_consumer",
        consume,
    )

    with pytest.raises(RuntimeError, match="collective refit failed"):
        extension.update_weights_from_collective()


def test_real_quant_load_preserves_names_and_owns_transport_tensors(monkeypatch):
    extension = object.__new__(VllmQuantInternalWorkerExtension)
    extension.device = torch.device("cpu")
    monkeypatch.setattr(extension, "_is_real_quant_model", lambda: True)
    received = []
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_backend."
        "VllmInternalWorkerExtension._load_weights",
        lambda _self, weights: received.extend(weights) or "loaded",
    )
    source = torch.arange(4)

    result = extension._load_weights([("model.layers.0.weight", source)])

    assert result == "loaded"
    assert received[0][0] == "model.layers.0.weight"
    assert torch.equal(received[0][1], source)
    assert received[0][1].data_ptr() != source.data_ptr()
