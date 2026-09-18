# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-specific CPU/config checks inside the actual learner and vLLM envs.

These checks do not load the full checkpoint or certify distributed training.
"""

import hashlib
import inspect
import json
import os
from copy import deepcopy
from pathlib import Path


def check_model_runtime(role: str, checkpoint: Path) -> dict:
    """Verify the selected Super format and its non-optional vision normalization."""
    raw_config = (checkpoint / "config.json").read_bytes()
    config = json.loads(raw_config)
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    expected_architecture = "NemotronH_Omni_Reasoning_V3"
    assert config["architectures"] == [expected_architecture], config["architectures"]
    norm_names = {
        f"vision_projector.vision_final_layernorm.{kind}" for kind in ("weight", "bias")
    }
    assert norm_names <= index["weight_map"].keys()
    result = {
        "architecture": expected_architecture,
        "config_sha256": hashlib.sha256(raw_config).hexdigest(),
    }

    # Heavy model libraries are imported only in their own qualified interpreter.
    if role == "learner":
        from megatron.bridge import AutoBridge
        from megatron.bridge.models.nemotron_omni.nemotron_omni_bridge import (
            Nemotron35SuperVLBridge,
        )
        from megatron.core.models.hybrid.hybrid_layer_allocation import (
            get_hybrid_total_layer_count,
            parse_hybrid_pattern,
        )
        from omegaconf import OmegaConf
        from transformers import AutoConfig

        from nemo_rl.models.megatron.setup import (
            _apply_mtp_config,
            _patch_hf_config_double_instantiation,
            load_model_config,
        )
        from nemo_rl.utils.config import load_config

        hf_config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
        bridge = AutoBridge.from_hf_config(hf_config)
        provider = bridge.to_megatron_provider(load_weights=False)
        assert isinstance(bridge._model_bridge, Nemotron35SuperVLBridge)
        assert provider.vision_final_layernorm is True
        root = Path(os.environ["PROJECT_ROOT"])
        recipe = (
            root
            / "examples/configs/recipes/vlm/vlm_grpo-nemotron-super-omni-120ba12b-image-tools-8n4g-megatron-tp8ep16cp2-async.v1.yaml"
        )
        policy = OmegaConf.to_container(load_config(recipe).policy, resolve=False)
        assert policy["megatron_cfg"]["mtp_num_layers"] == 0
        # Exercise the *actual* override on fresh and finalized providers. The
        # cached-checkpoint path does not call finalize() after applying MTP=0.
        finalized = bridge.to_megatron_provider(load_weights=False)
        finalized.finalize()
        assert parse_hybrid_pattern(finalized.hybrid_layer_pattern).mtp_num_depths > 0
        for settings in (
            {},
            {"mtp_num_layers": 1},
            {"mtp_num_layers": 0, "_provider_override_allowlist": []},
        ):
            retained = deepcopy(finalized)
            _apply_mtp_config(retained, {"megatron_cfg": settings})
            assert retained.hybrid_layer_pattern == finalized.hybrid_layer_pattern
            assert retained.mtp_num_layers == finalized.mtp_num_layers
        providers = {"fresh": provider, "finalized": finalized}
        cached_config = os.environ.get("IMAGE_TOOLS_CACHED_MODEL_CONFIG")
        if cached_config:
            _patch_hf_config_double_instantiation()
            cached, _ = load_model_config(str(Path(cached_config).parent))
            providers["cached"] = cached
        for lifecycle, candidate in providers.items():
            _apply_mtp_config(candidate, policy)
            parsed = parse_hybrid_pattern(candidate.hybrid_layer_pattern)
            assert parsed.mtp_pattern is None and parsed.mtp_num_depths == 0, lifecycle
            assert candidate.vision_final_layernorm is True, lifecycle
            candidate.finalize()
            assert (
                parse_hybrid_pattern(candidate.hybrid_layer_pattern).mtp_num_depths == 0
            )
            assert candidate._build_vision_config(candidate).mtp_num_layers is not None
        provider.finalize()
        assert provider.num_layers == hf_config.llm_config.num_hidden_layers, (
            provider.num_layers,
            hf_config.llm_config.num_hidden_layers,
        )
        assert (
            get_hybrid_total_layer_count(provider.hybrid_layer_pattern)
            == provider.num_layers
        )
        vision_config = provider._build_vision_config(provider)
        assert vision_config.mtp_num_layers is not None
        mappings = bridge._model_bridge.mapping_registry().mappings
        for kind in ("weight", "bias"):
            assert any(
                mapping.megatron_param == f"vision_model.decoder.final_layernorm.{kind}"
                and mapping.hf_param
                == f"vision_projector.vision_final_layernorm.{kind}"
                for mapping in mappings
            )
        assert bridge._model_bridge._mtp_hf_prefix() == "language_model."
        result.update(
            bridge=type(bridge._model_bridge).__name__,
            layers=provider.num_layers,
            vision_final_layernorm=provider.vision_final_layernorm,
            language_mtp=provider.mtp_num_layers,
            mtp_disabled_lifecycles=list(providers),
        )
    elif role == "generation":
        import torch
        from vllm.model_executor.models.nano_nemotron_vl import NemotronH_Nano_VL_V2

        model_class = NemotronH_Nano_VL_V2
        assert hasattr(model_class, "_apply_vision_final_layernorm"), (
            "Bundled vLLM lacks the Super final-LayerNorm fix"
        )
        load_source = inspect.getsource(model_class.load_weights)
        assert "vision_projector.vision_final_layernorm." in load_source
        assert "_vision_final_layernorm_enabled" in load_source
        for method in (
            model_class.extract_feature,
            model_class.extract_feature_dynamic,
        ):
            assert "_apply_vision_final_layernorm" in inspect.getsource(method)
        # Exercise the actual installed normalization method on a tiny tensor;
        # no full-model construction or checkpoint allocation is involved.
        model = object.__new__(model_class)
        norm = torch.nn.LayerNorm(4, eps=1e-6).float()
        object.__setattr__(model, "vision_final_layernorm", norm)
        object.__setattr__(model, "_vision_final_layernorm_enabled", True)
        inputs = torch.tensor([[[1.0, 2.0, 4.0, 8.0]]], dtype=torch.bfloat16)
        torch.testing.assert_close(
            model._apply_vision_final_layernorm(inputs),
            norm(inputs.float()).to(inputs.dtype),
        )
        result.update(
            model_class=model_class.__name__,
            vision_final_layernorm=True,
            module_sha256=hashlib.sha256(
                Path(inspect.getfile(model_class)).read_bytes()
            ).hexdigest(),
        )
    else:
        raise ValueError(f"Unexpected runtime role: {role}")
    return result
