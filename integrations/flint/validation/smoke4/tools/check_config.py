"""Validate the concrete run config before constructing Ray workers."""

import json
from pathlib import Path

from omegaconf import OmegaConf

from nemo_automodel._transformers import registry
from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
from nemo_rl.utils.outdated_config_checks import check_outdated_config
from transformers import AutoConfig

register_omegaconf_resolvers()
path = "examples/configs/recipes/llm/grpo-flint-4layer-1n4g-automodel-sglang-quick.yaml"
raw = OmegaConf.to_container(load_config(path), resolve=True)
check_outdated_config(raw)
config = MasterConfig(**raw)
hf = AutoConfig.from_pretrained(config.policy["model_name"])
assert hf.num_hidden_layers == 4
assert hf.layer_types == ["full_attention", "full_attention", "sliding_attention", "sliding_attention"]
assert hf.no_rope_layer_indices == [0, 1]
assert hf.router_aux_loss_coef == 0
assert not (Path(config.policy["model_name"]) / ".PROXY_BUILD_INCOMPLETE").exists()
print(json.dumps({"status": "passed", "native_registered": "FlintForCausalLM" in registry.MODEL_ARCH_MAPPING,
                  "config_class": type(hf).__module__ + "." + type(hf).__qualname__,
                  "layers": hf.num_hidden_layers, "layer_types": hf.layer_types,
                  "train_global_batch_size": config.policy["train_global_batch_size"],
                  "backend": config.policy["generation"]["backend"]}, indent=2))
