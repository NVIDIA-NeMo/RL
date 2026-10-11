"""The native policy facade uses the backbone's parsed configuration."""
import json
import math
from types import SimpleNamespace

import pytest
import torch
from omegaconf import DictConfig
from transformers import AutoConfig, NemotronHConfig

from nemo.collections.speechlm2.models.salm_automodel import SALMAutomodel
from nemo_rl.models.automodel.speechlm import SpeechLMPolicy


@pytest.mark.parametrize("immediate", [False, True])
def test_native_facade_uses_loaded_config_and_preserves_overrides(tmp_path, monkeypatch, immediate):
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

    def initialize(self, cfg, *, parameter_config):
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
        parameter_config=None,
        native_model={"model_config_overrides": {"encoder_chunk_size_seconds": 20.0}},
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
def test_native_loader_preserves_dtype_at_speech_boundary(monkeypatch, dtype):
    observed = {}
    model = SimpleNamespace(training=True, validate_update_surface=lambda: None, train=lambda mode: None)
    def load(cls, model_name, **kwargs):
        observed.update(kwargs)
        return model
    monkeypatch.setattr(SALMAutomodel, "from_pretrained", classmethod(load))
    result = SpeechLMPolicy.from_pretrained(
        "fixture", native_model={"trainable_names": ["llm.proj.weight"]}, torch_dtype=dtype,
    )
    assert result is model
    assert observed["torch_dtype"] == ("float32" if isinstance(dtype, str) else dtype)
    resolved = getattr(torch, observed["torch_dtype"]) if isinstance(observed["torch_dtype"], str) else observed["torch_dtype"]
    assert resolved == torch.float32
    assert observed["parameter_config"].trainable_dtype == torch.float32
    assert observed["parameter_config"].frozen_dtype == torch.bfloat16


def test_native_perception_selection_precedes_activation_checkpoint_wrapping(monkeypatch):
    from torch import nn
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
    from nemo_automodel.components.utils.parameter_config import ParameterConfig
    from nemo.collections.speechlm2.models import salm_automodel

    class Perception(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Module()
            self.encoder.asr_encoder = nn.Module()
            self.encoder.asr_encoder.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(32)])
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
        for index in (30, 31) for role in ("weight", "bias")
    ) + ("perception.proj.weight", "perception.proj.bias")
    model = SimpleNamespace(
        _trainer=None,
        cfg=DictConfig({"pretrained_llm": "fixture", "torch_dtype": "float32", "pretrained_weights": False}),
        parameter_config=ParameterConfig(selected, torch.float32, torch.bfloat16),
        _mtp_enabled=False,
        setup_moe_options=lambda: None,
        _apply_mtp_training_mode=lambda mode: None,
    )
    monkeypatch.setattr(salm_automodel, "load_pretrained_automodel_llm", lambda *args, **kwargs: SimpleNamespace(config=SimpleNamespace(num_nextn_predict_layers=0)))
    monkeypatch.setattr(salm_automodel, "setup_speech_encoder", lambda model, **kwargs: setattr(model, "perception", perception))
    monkeypatch.setattr(salm_automodel, "update_perception_output_dim", lambda model: None)
    monkeypatch.setattr(salm_automodel, "maybe_load_pretrained_models", lambda model: None)
    # Run the production configure path and real selection/casting/wrappers;
    # only full backbone/ASR weight construction is replaced by tiny modules.
    SALMAutomodel.configure_model(model, activation_checkpointing_perception=True)
    assert perception.checkpointing_enabled
    parameters = {name.replace("._checkpoint_wrapped_module", ""): value for name, value in perception.named_parameters()}
    assert {"perception." + name for name, value in parameters.items() if value.requires_grad} == set(selected)
    assert all(value.dtype == (torch.float32 if "perception." + name in selected else torch.bfloat16) for name, value in parameters.items())
    assert all(hasattr(layer, "_checkpoint_wrapped_module") for layer in perception.encoder.asr_encoder.layers)
    loss = sum(layer(torch.ones(1, 2)).sum() for layer in perception.encoder.asr_encoder.layers[30:])
    loss.backward()
    assert all(perception.encoder.asr_encoder.layers[index].weight.grad is not None for index in (30, 31))
    assert perception.encoder.asr_encoder.layers[0].weight.grad is None


@pytest.mark.parametrize("explicit_selection", [False, True])
def test_moe_options_preserve_explicit_parameter_selection(explicit_selection):
    from torch import nn
    from nemo_automodel.components.moe.config import MoEConfig
    from nemo_automodel.components.moe.layers import Gate
    from nemo_automodel.components.utils.parameter_config import ParameterConfig, apply_parameter_config

    config = MoEConfig(
        n_routed_experts=2, n_shared_experts=0, n_activated_experts=1,
        n_expert_groups=1, n_limited_groups=1, train_gate=False,
        gate_bias_update_factor=0.0, aux_loss_coeff=0.0, score_func="softmax",
        route_scale=1.0, dim=2, inter_dim=2, moe_inter_dim=2,
        norm_topk_prob=True, router_bias=True, dtype=torch.float32,
    )
    backbone = nn.Module()
    backbone.proj = nn.Linear(2, 2)
    backbone.frozen_gate = Gate(config)
    backbone.selected_gate = Gate(config)
    backbone.mtp = nn.Module()
    backbone.mtp.gate = Gate(config)
    selected = ("proj.weight", "selected_gate.weight", "selected_gate.bias")
    parameter_config = ParameterConfig(selected) if explicit_selection else None
    if parameter_config is not None:
        apply_parameter_config(backbone, parameter_config)
    model = SimpleNamespace(
        llm=backbone, parameter_config=parameter_config,
        cfg=DictConfig({"train_gate": True, "aux_loss_coeff": 0.2}),
    )
    # Real Gate modules and real selection followed by the production MoE
    # override, in the same order as native model construction.
    SALMAutomodel.setup_moe_options(model)
    gates = (backbone.frozen_gate, backbone.selected_gate, backbone.mtp.gate)
    assert all(gate.train_gate and gate.aux_loss_coeff == 0.2 for gate in gates)
    if explicit_selection:
        assert {name for name, value in backbone.named_parameters() if value.requires_grad} == set(selected)
        assert not backbone.frozen_gate.weight.requires_grad
        assert not backbone.frozen_gate.bias.requires_grad
        assert not backbone.mtp.gate.weight.requires_grad
        assert not backbone.mtp.gate.bias.requires_grad
    else:
        assert all(gate.weight.requires_grad and gate.bias.requires_grad for gate in gates)
