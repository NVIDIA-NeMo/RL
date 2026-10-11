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
