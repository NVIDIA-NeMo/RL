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

"""The native policy facade uses the backbone's parsed configuration."""

import json
import math
from types import SimpleNamespace

import pytest
import torch
from omegaconf import DictConfig
from transformers import AutoConfig, NemotronHConfig

pytest.importorskip("nemo.collections.speechlm2")
pytest.importorskip("nemo_automodel.components.utils.model_utils")

from nemo.collections.speechlm2.models.salm_automodel import SALMAutomodel  # noqa: E402

from nemo_rl.models.automodel.speechlm import SpeechLMPolicy  # noqa: E402


def native_config(**kwargs):
    return {
        "trainable_names": ["llm.proj.weight"],
        "expected_trainable_tensors": 1,
        "expected_trainable_scalars": 1,
        "max_expanded_sequence_length": 128,
        **kwargs,
    }


@pytest.mark.parametrize("immediate", [False, True])
def test_native_facade_uses_loaded_config_and_preserves_overrides(
    tmp_path, monkeypatch, immediate
):
    backbone = NemotronHConfig(
        architectures=["NemotronHForCausalLM"],
        time_step_limit=[0.0, float("inf")],
        num_nextn_predict_layers=1,
    )
    backbone.save_pretrained(tmp_path)
    serialized = json.loads((tmp_path / "config.json").read_text())
    assert serialized["time_step_limit"][1] == {"__float__": "Infinity"}
    loaded = AutoConfig.from_pretrained(tmp_path)
    assert math.isinf(loaded.time_step_limit[1])

    def initialize(self, cfg):
        self.cfg = DictConfig(cfg)
        self.llm = None
        if cfg["init_configure_model"]:
            self.configure_model()

    def configure(self, **kwargs):
        self.llm = SimpleNamespace(config=loaded)

    # Bound the test to configuration: production SpeechLMPolicy methods run,
    # while native module/weight construction is outside this fixture's scope.
    monkeypatch.setattr(SALMAutomodel, "__init__", initialize)
    monkeypatch.setattr(SALMAutomodel, "configure_model", configure)
    cfg = {
        "pretrained_llm": str(tmp_path),
        "llm_config": serialized,
        "audio_locator_tag": "<|audio|>",
        "pretrained_asr": "fixture-asr",
        "prompt_format": "qwen",
        "pretrained_weights": False,
        "init_configure_model": immediate,
        "encoder_chunk_size_seconds": 10.0,
    }
    model = SpeechLMPolicy(
        cfg,
        native_model=native_config(
            model_config_overrides={"encoder_chunk_size_seconds": 20.0}
        ),
    )
    if not immediate:
        assert model.llm is None
        model.configure_model()
    assert math.isinf(model.config.text_config.time_step_limit[1])
    assert model.config.text_config.num_nextn_predict_layers == 1
    assert model.config.encoder_chunk_size_seconds == 20.0
    assert model.config.retain_native_mtp is True
    assert model.cfg.mtp.enabled is False
    assert model.config.mtp is None
    assert cfg["encoder_chunk_size_seconds"] == 10.0
    assert cfg["llm_config"]["time_step_limit"][1] == {"__float__": "Infinity"}


@pytest.mark.parametrize("dtype", ["torch.float32", "float32", torch.float32])
def test_native_loader_uses_fp32_storage_at_speech_boundary(monkeypatch, dtype):
    observed = {}
    model = SimpleNamespace(
        training=True, validate_update_surface=lambda: None, train=lambda mode: None
    )

    def load(cls, model_name, **kwargs):
        observed.update(kwargs)
        return model

    monkeypatch.setattr(SALMAutomodel, "from_pretrained", classmethod(load))
    result = SpeechLMPolicy.from_native_pretrained(
        "fixture",
        native_model=native_config(),
        torch_dtype=dtype,
    )
    assert result is model
    assert observed["torch_dtype"] == torch.float32
    resolved = (
        getattr(torch, observed["torch_dtype"])
        if isinstance(observed["torch_dtype"], str)
        else observed["torch_dtype"]
    )
    assert resolved == torch.float32
    assert "parameter_config" not in observed


def test_native_perception_selection_precedes_activation_checkpoint_wrapping(
    monkeypatch,
):
    from nemo.collections.speechlm2.models import salm_automodel
    from torch import nn
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
    )

    class Perception(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Module()
            self.encoder.asr_encoder = nn.Module()
            self.encoder.asr_encoder.layers = nn.ModuleList(
                [nn.Linear(2, 2) for _ in range(32)]
            )
            self.proj = nn.Linear(2, 2)
            self.checkpointing_enabled = False

        def set_activation_checkpointing(self, enabled):
            self.checkpointing_enabled = enabled
            if enabled:
                layers = self.encoder.asr_encoder.layers
                for index, layer in enumerate(layers):
                    layers[index] = checkpoint_wrapper(layer)

    perception = Perception()
    selected = tuple(
        f"perception.encoder.asr_encoder.layers.{index}.{role}"
        for index in (30, 31)
        for role in ("weight", "bias")
    ) + ("perception.proj.weight", "perception.proj.bias")
    model = SimpleNamespace(
        _trainer=None,
        cfg=DictConfig(
            {
                "pretrained_llm": "fixture",
                "torch_dtype": "float32",
                "pretrained_weights": False,
                "perception_freeze_config": {
                    "freeze_modules": [{"glob": "*"}],
                    "unfreeze_parameters": [
                        name.removeprefix("perception.") for name in selected
                    ],
                },
            }
        ),
        _mtp_enabled=False,
        setup_moe_options=lambda: None,
        _apply_mtp_training_mode=lambda mode: None,
    )
    monkeypatch.setattr(
        salm_automodel,
        "load_pretrained_automodel_llm",
        lambda *args, **kwargs: SimpleNamespace(
            config=SimpleNamespace(num_nextn_predict_layers=0)
        ),
    )
    monkeypatch.setattr(
        salm_automodel,
        "setup_speech_encoder",
        lambda model, **kwargs: setattr(model, "perception", perception),
    )
    monkeypatch.setattr(
        salm_automodel, "update_perception_output_dim", lambda model: None
    )
    monkeypatch.setattr(
        salm_automodel, "maybe_load_pretrained_models", lambda model: None
    )
    # Run the production configure path and real selection/casting/wrappers;
    # only full backbone/ASR weight construction is replaced by tiny modules.
    SALMAutomodel.configure_model(
        model,
        activation_checkpointing_perception=True,
        perception_mixed_precision=SimpleNamespace(param_dtype=torch.bfloat16),
    )
    assert perception.checkpointing_enabled
    parameters = {
        name.replace("._checkpoint_wrapped_module", ""): value
        for name, value in perception.named_parameters()
    }
    assert {
        "perception." + name
        for name, value in parameters.items()
        if value.requires_grad
    } == set(selected)
    assert all(
        value.dtype
        == (torch.float32 if "perception." + name in selected else torch.bfloat16)
        for name, value in parameters.items()
    )
    assert all(
        hasattr(layer, "_checkpoint_wrapped_module")
        for layer in perception.encoder.asr_encoder.layers
    )
    loss = sum(
        layer(torch.ones(1, 2)).sum()
        for layer in perception.encoder.asr_encoder.layers[30:]
    )
    loss.backward()
    assert all(
        perception.encoder.asr_encoder.layers[index].weight.grad is not None
        for index in (30, 31)
    )
    assert perception.encoder.asr_encoder.layers[0].weight.grad is None


@pytest.mark.parametrize("explicit_selection", [False, True])
def test_moe_options_preserve_explicit_parameter_selection(explicit_selection):
    from nemo_automodel.components.moe.config import MoEConfig
    from nemo_automodel.components.moe.layers import Gate
    from nemo_automodel.components.utils.model_utils import (
        apply_parameter_freezing,
    )
    from torch import nn

    config = MoEConfig(
        n_routed_experts=2,
        n_shared_experts=0,
        n_activated_experts=1,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=False,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=2,
        inter_dim=2,
        moe_inter_dim=2,
        norm_topk_prob=True,
        router_bias=True,
        dtype=torch.float32,
    )
    backbone = nn.Module()
    backbone.proj = nn.Linear(2, 2)
    backbone.frozen_gate = Gate(config)
    backbone.selected_gate = Gate(config)
    backbone.mtp = nn.Module()
    backbone.mtp.gate = Gate(config)
    selected = ("proj.weight", "selected_gate.weight", "selected_gate.bias")
    freeze_config = (
        {"freeze_modules": [{"glob": "*"}], "unfreeze_parameters": list(selected)}
        if explicit_selection
        else None
    )
    if freeze_config is not None:
        apply_parameter_freezing(backbone, freeze_config)
    model = SimpleNamespace(
        llm=backbone,
        cfg=DictConfig(
            {
                "train_gate": True,
                "aux_loss_coeff": 0.2,
                "automodel_freeze_config": freeze_config,
            }
        ),
    )
    # Real Gate modules and real selection followed by the production MoE
    # override, in the same order as native model construction.
    SALMAutomodel.setup_moe_options(model)
    gates = (backbone.frozen_gate, backbone.selected_gate, backbone.mtp.gate)
    assert all(gate.train_gate and gate.aux_loss_coeff == 0.2 for gate in gates)
    if explicit_selection:
        assert {
            name for name, value in backbone.named_parameters() if value.requires_grad
        } == set(selected)
        assert not backbone.frozen_gate.weight.requires_grad
        assert not backbone.frozen_gate.bias.requires_grad
        assert not backbone.mtp.gate.weight.requires_grad
        assert not backbone.mtp.gate.bias.requires_grad
    else:
        assert all(
            gate.weight.requires_grad and gate.bias.requires_grad for gate in gates
        )


def test_native_loader_preserves_frozen_frontend_and_fp32_selected_weights(monkeypatch):
    from nemo.collections.asr.modules.audio_preprocessing import (
        AudioToMelSpectrogramPreprocessor,
    )
    from nemo.collections.speechlm2.models import salm_automodel
    from torch import nn

    class Perception(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(2, 2)
            self.preprocessor = AudioToMelSpectrogramPreprocessor(
                features=4,
                n_fft=16,
                window_size=None,
                window_stride=None,
                n_window_size=16,
                n_window_stride=8,
                dither=0.0,
                pad_to=0,
                normalize=None,
            )
            self.encoder = nn.Module()
            self.encoder.diarization_model = nn.Module()
            self.encoder.diarization_model.preprocessor = (
                AudioToMelSpectrogramPreprocessor(
                    features=4,
                    n_fft=16,
                    window_size=None,
                    window_stride=None,
                    n_window_size=16,
                    n_window_stride=8,
                    dither=0.0,
                    pad_to=0,
                    normalize=None,
                )
            )

        def set_activation_checkpointing(self, enabled):
            pass

    perception = Perception()
    # Exact seed storage, including a selected FP32 value that BF16 cannot retain.
    seed = {
        name: value.to(torch.bfloat16).clone()
        for name, value in perception.state_dict().items()
    }
    seed["proj.weight"] = torch.full_like(perception.proj.weight, 1.000123)
    observed = {}

    def load(cls, model_name, **kwargs):
        observed.update(kwargs)
        dtype = kwargs["torch_dtype"]
        dtype_name = str(dtype).removeprefix("torch.")
        model = SimpleNamespace(
            _trainer=None,
            cfg=DictConfig(
                {
                    "pretrained_llm": "fixture",
                    "torch_dtype": dtype_name,
                    "pretrained_weights": False,
                    "perception_freeze_config": {
                        "freeze_modules": [{"glob": "*"}],
                        "unfreeze_parameters": ["proj.weight"],
                    },
                }
            ),
            _mtp_enabled=False,
            setup_moe_options=lambda: None,
            _apply_mtp_training_mode=lambda mode: None,
            training=True,
            validate_update_surface=lambda: None,
            train=lambda mode: None,
        )
        SALMAutomodel.configure_model(
            model,
            perception_mixed_precision=SimpleNamespace(param_dtype=torch.bfloat16),
        )
        model.perception.load_state_dict(seed, strict=True)
        return model

    monkeypatch.setattr(SALMAutomodel, "from_pretrained", classmethod(load))
    monkeypatch.setattr(
        salm_automodel,
        "load_pretrained_automodel_llm",
        lambda *args, **kwargs: SimpleNamespace(
            config=SimpleNamespace(num_nextn_predict_layers=0)
        ),
    )
    monkeypatch.setattr(
        salm_automodel,
        "setup_speech_encoder",
        lambda model, **kwargs: setattr(model, "perception", perception),
    )
    monkeypatch.setattr(
        salm_automodel, "update_perception_output_dim", lambda model: None
    )
    monkeypatch.setattr(
        salm_automodel, "maybe_load_pretrained_models", lambda model: None
    )
    SpeechLMPolicy.from_native_pretrained(
        "fixture",
        native_model=native_config(trainable_names=["perception.proj.weight"]),
        torch_dtype="torch.float32",
    )
    assert perception.proj.weight.dtype == torch.float32
    assert perception.proj.weight.requires_grad
    assert torch.equal(perception.proj.weight, seed["proj.weight"])
    assert perception.proj.bias.dtype == torch.bfloat16
    assert not perception.proj.bias.requires_grad
    for name, value in perception.state_dict().items():
        if name != "proj.weight":
            assert value.dtype == seed[name].dtype, name
            assert torch.equal(value, seed[name]), name
    for frontend in (
        perception.preprocessor,
        perception.encoder.diarization_model.preprocessor,
    ):
        assert frontend.dtype_sentinel_tensor.dtype == torch.bfloat16
        assert "dtype_sentinel_tensor" not in frontend.state_dict()
        features, _ = frontend(
            input_signal=torch.ones(1, 64, dtype=torch.float32),
            length=torch.tensor([64]),
        )
        assert features.dtype == torch.bfloat16


@pytest.fixture
def update_surface_model():
    """Small registered model for the real policy's update-surface validation."""
    model = torch.nn.Module()
    model.llm = torch.nn.Module()
    model.llm.proj = torch.nn.Linear(1, 1, bias=False, dtype=torch.float32)
    model.llm.mtp = torch.nn.Linear(1, 1, bias=False, dtype=torch.bfloat16)
    model.llm.mtp.requires_grad_(False)
    model.ctc_timestamp = torch.nn.Module()
    model.ctc_timestamp.decoder = torch.nn.Linear(1, 1, bias=False, dtype=torch.float32)
    model.ctc_timestamp.requires_grad_(False)
    model.cfg = DictConfig(
        {"ctc_timestamp_artifact": {"tensor_dtypes": {"decoder.weight": "float32"}}}
    )
    model.native_model = native_config()
    return model


@pytest.mark.parametrize(
    "ctc_dtype", [None, torch.float32, torch.float16, torch.bfloat16]
)
def test_update_surface_accepts_optional_frozen_ctc(update_surface_model, ctc_dtype):
    model = update_surface_model
    if ctc_dtype is None:
        model.ctc_timestamp = None
    else:
        model.ctc_timestamp.to(dtype=ctc_dtype)
        model.cfg.ctc_timestamp_artifact.tensor_dtypes["decoder.weight"] = str(
            ctc_dtype
        ).removeprefix("torch.")
    SpeechLMPolicy.validate_update_surface(model)
    assert SpeechLMPolicy.refit_parameter_names(model) == ("llm.proj.weight",)


@pytest.mark.parametrize(
    "defect", ["ctc_dtype", "ctc_unfrozen", "ctc_selected", "core_dtype"]
)
def test_update_surface_rejects_ctc_or_core_mutation(update_surface_model, defect):
    model = update_surface_model
    if defect == "ctc_dtype":
        model.ctc_timestamp.to(dtype=torch.bfloat16)
        message = "CTC parameters"
    elif defect in ("ctc_unfrozen", "ctc_selected"):
        model.ctc_timestamp.requires_grad_(True)
        message = "surface mismatch"
        if defect == "ctc_selected":
            model.native_model["trainable_names"].append("ctc_timestamp.decoder.weight")
            model.native_model["expected_trainable_tensors"] = 2
            model.native_model["expected_trainable_scalars"] = 2
            message = "CTC parameters"
    else:
        model.llm.mtp.to(dtype=torch.float32)
        message = "Frozen native parameter storage"
    with pytest.raises(ValueError, match=message):
        SpeechLMPolicy.validate_update_surface(model)
