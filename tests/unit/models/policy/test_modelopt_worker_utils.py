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

import importlib
import os
import sys
import types
from functools import partial

import pytest
import torch
from torch.utils.data import DataLoader


def _ensure_package(name):
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    parent_name, _, child_name = name.rpartition(".")
    if parent_name:
        parent = _ensure_package(parent_name)
        setattr(parent, child_name, module)
    return module


def _ensure_module(name):
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    parent_name, _, child_name = name.rpartition(".")
    if parent_name:
        parent = _ensure_package(parent_name)
        setattr(parent, child_name, module)
    return module


def _install_optional_dependency_stubs():
    modelopt_quant = _ensure_package("modelopt.torch.quantization")
    modelopt_quant.quantize = lambda model, cfg, forward_loop: model
    modelopt_quant.print_quant_summary = lambda model: None

    modelopt_config = _ensure_module("modelopt.torch.quantization.config")
    modelopt_config.need_calibration = lambda cfg: False

    dataset_utils = _ensure_module("modelopt.torch.utils.dataset_utils")
    dataset_utils.create_forward_loop = lambda dataloader: (
        lambda model: [model(batch) for batch in dataloader]
    )
    dataset_utils.get_dataset_dataloader = lambda **kwargs: DataLoader(
        _SimpleDataset({"input_ids": torch.ones(1, 1, dtype=torch.long)}),
        batch_size=1,
    )

    plugins = _ensure_module("modelopt.torch.utils.plugins")
    plugins.megatron_prefill = lambda model, input_ids, skip_return_logits: None

    gpt_provider = _ensure_module("megatron.bridge.models.gpt_provider")
    gpt_provider.transformer_engine_layer_spec = object()

    mamba_provider = _ensure_module("megatron.bridge.models.mamba.mamba_provider")
    mamba_provider.modelopt_mamba_stack_spec = object()
    mamba_provider.transformer_engine_mamba_stack_spec = object()

    model_specs = _ensure_module("megatron.core.post_training.modelopt.gpt.model_specs")

    def fake_gpt_modelopt_spec(**kwargs):
        return kwargs

    model_specs.get_gpt_modelopt_spec = fake_gpt_modelopt_spec

    parallel_state = _ensure_module("megatron.core.parallel_state")
    parallel_state.get_context_parallel_world_size = lambda: 1


class _SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.data.items()}

    def __len__(self):
        return len(next(iter(self.data.values())))


try:
    worker_utils = importlib.import_module(
        "nemo_rl.modelopt.models.policy.workers.utils"
    )
except ImportError:
    _install_optional_dependency_stubs()
    sys.modules.pop("nemo_rl.modelopt.models.policy.workers.utils", None)
    worker_utils = importlib.import_module(
        "nemo_rl.modelopt.models.policy.workers.utils"
    )


def test_get_tokenizer_applies_modelopt_calibration_defaults(monkeypatch):
    tokenizer = types.SimpleNamespace(padding_side="right", model_max_length=0)
    monkeypatch.setattr(worker_utils, "_base_get_tokenizer", lambda cfg: tokenizer)

    result = worker_utils.get_tokenizer("checkpoint-path", max_seq_len=128)

    assert result is tokenizer
    assert tokenizer.padding_side == "left"
    assert tokenizer.model_max_length == 128


def test_megatron_forward_loop_prefills_batch_input_ids(monkeypatch):
    seen = []
    dataloader = DataLoader(
        worker_utils._DictDataset({"input_ids": torch.tensor([[1, 2, 3]])}),
        batch_size=1,
    )
    monkeypatch.setattr(
        worker_utils,
        "megatron_prefill",
        lambda model, input_ids, skip_return_logits: seen.append(
            (model, input_ids.clone(), skip_return_logits)
        ),
    )

    loop = worker_utils.get_forward_loop_func(True, dataloader)
    loop("model")

    assert seen[0][0] == "model"
    torch.testing.assert_close(seen[0][1], torch.tensor([[1, 2, 3]]))
    assert seen[0][2] is True


def test_quantize_model_skips_forward_loop_for_weight_only_config(monkeypatch):
    model = torch.nn.Linear(1, 1)
    calls = []

    monkeypatch.setattr(
        worker_utils,
        "resolve_quant_cfg",
        lambda quant_cfg: {"quant_cfg": [{"name": quant_cfg}]},
    )
    monkeypatch.setattr(worker_utils, "need_calibration", lambda cfg: False)
    monkeypatch.setattr(
        worker_utils.mtq,
        "quantize",
        lambda model_arg, cfg, forward_loop: calls.append(
            (model_arg, cfg, forward_loop)
        )
        or model_arg,
    )
    monkeypatch.setattr(worker_utils.mtq, "print_quant_summary", lambda model: None)

    worker_utils.quantize_model(
        model,
        "NVFP4_DEFAULT_CFG",
        tokenizer=None,
        calib_size=8,
    )

    assert calls == [
        (model, {"quant_cfg": [{"name": "NVFP4_DEFAULT_CFG"}]}, None),
    ]


def test_quantize_model_requires_calibration_data(monkeypatch):
    model = torch.nn.Linear(1, 1)
    monkeypatch.setattr(worker_utils, "resolve_quant_cfg", lambda quant_cfg: {})
    monkeypatch.setattr(worker_utils, "need_calibration", lambda cfg: True)

    with pytest.raises(ValueError, match="policy.quant_calib_data"):
        worker_utils.quantize_model(
            model,
            "activation-cfg",
            tokenizer=None,
            calib_size=8,
            data=None,
        )


def test_quantize_model_uses_random_calibration_loop(monkeypatch):
    model = torch.nn.Linear(1, 1)
    calls = []

    monkeypatch.setattr(worker_utils, "resolve_quant_cfg", lambda quant_cfg: {})
    monkeypatch.setattr(worker_utils, "need_calibration", lambda cfg: True)
    monkeypatch.setattr(
        worker_utils,
        "get_forward_loop_func",
        lambda is_megatron, dataloader: (
            "loop",
            is_megatron,
            len(dataloader.dataset),
        ),
    )
    monkeypatch.setattr(
        worker_utils.mtq,
        "quantize",
        lambda model_arg, cfg, forward_loop: calls.append(forward_loop) or model_arg,
    )
    monkeypatch.setattr(worker_utils.mtq, "print_quant_summary", lambda model: None)

    worker_utils.quantize_model(
        model,
        "activation-cfg",
        tokenizer=None,
        calib_size=8,
        is_megatron=True,
        data="random",
    )

    assert calls == [("loop", True, 1)]


def test_quantize_model_uses_named_calibration_dataset(monkeypatch):
    model = torch.nn.Linear(1, 1)
    dataset = worker_utils._DictDataset(
        {"input_ids": torch.ones(2, 3, dtype=torch.long)}
    )
    dataloader = DataLoader(dataset, batch_size=2)
    calls = []

    monkeypatch.setattr(worker_utils, "resolve_quant_cfg", lambda quant_cfg: {})
    monkeypatch.setattr(worker_utils, "need_calibration", lambda cfg: True)
    monkeypatch.setattr(
        worker_utils,
        "get_dataset_dataloader",
        lambda **kwargs: calls.append(("dataset", kwargs)) or dataloader,
    )
    monkeypatch.setattr(
        worker_utils,
        "get_forward_loop_func",
        lambda is_megatron, calib_dataloader: (
            "loop",
            is_megatron,
            calib_dataloader,
        ),
    )
    monkeypatch.setattr(
        worker_utils.mtq,
        "quantize",
        lambda model_arg, cfg, forward_loop: calls.append(
            ("quantize", model_arg, cfg, forward_loop)
        )
        or model_arg,
    )
    monkeypatch.setattr(worker_utils.mtq, "print_quant_summary", lambda model: None)

    worker_utils.quantize_model(
        model,
        "activation-cfg",
        tokenizer="tokenizer",
        calib_size=8,
        batch_size=4,
        data="cnn_dailymail",
        max_sample_length=16,
    )

    assert calls[0] == (
        "dataset",
        {
            "dataset_name": "cnn_dailymail",
            "tokenizer": "tokenizer",
            "batch_size": 4,
            "num_samples": 8,
            "device": model.weight.device,
            "include_labels": False,
            "max_sample_length": 16,
        },
    )
    assert calls[1] == ("quantize", model, {}, ("loop", False, dataloader))


def test_get_modelopt_checkpoint_dir_env_precedence(monkeypatch):
    monkeypatch.setenv("NRL_MODELOPT_CHECKPOINT_DIR", "/custom/modelopt")
    monkeypatch.setenv("HF_HOME", "/hf/home")

    assert worker_utils.get_modelopt_checkpoint_dir() == "/custom/modelopt"


def test_get_modelopt_checkpoint_dir_uses_hf_home(monkeypatch):
    monkeypatch.delenv("NRL_MODELOPT_CHECKPOINT_DIR", raising=False)
    monkeypatch.setenv("HF_HOME", "/hf/home")

    assert worker_utils.get_modelopt_checkpoint_dir() == os.path.join(
        "/hf/home", "nemo_rl"
    )


def test_get_modelopt_checkpoint_dir_falls_back_to_home(monkeypatch):
    monkeypatch.delenv("NRL_MODELOPT_CHECKPOINT_DIR", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setattr(worker_utils.os.path, "expanduser", lambda path: "/home/test")

    assert worker_utils.get_modelopt_checkpoint_dir() == os.path.join(
        "/home/test", ".cache", "huggingface", "nemo_rl"
    )


def test_get_quantization_layer_spec_can_be_disabled():
    assert worker_utils.get_quantization_layer_spec(True) is (
        worker_utils.transformer_engine_layer_spec
    )


def test_get_quantization_layer_spec_returns_modelopt_partial():
    layer_spec = worker_utils.get_quantization_layer_spec(False)

    assert isinstance(layer_spec, partial)
    assert layer_spec.func is worker_utils.get_gpt_modelopt_spec
    assert layer_spec.keywords["real_quant_cfg"] == "None"
