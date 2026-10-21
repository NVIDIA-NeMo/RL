"""Validate the concrete run config before constructing Ray workers."""

import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

from nemo_automodel._transformers import registry
from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
from nemo_rl.utils.outdated_config_checks import check_outdated_config
from transformers import AutoConfig

register_omegaconf_resolvers()
path = sys.argv[1] if len(sys.argv) > 1 else "examples/configs/recipes/llm/grpo-flint-49layer-1n4g-automodel-sglang-actckpt-quick.yaml"
raw = OmegaConf.to_container(load_config(path), resolve=True)
check_outdated_config(raw)
config = MasterConfig(**raw)
hf = AutoConfig.from_pretrained(config.policy["model_name"], **config.policy["hf_config_overrides"])
assert hf.num_hidden_layers == 49
assert len(hf.layer_types) == 49
assert hf.layer_types[:4] == ["full_attention", "full_attention", "sliding_attention", "sliding_attention"]
assert hf.router_aux_loss_coef == 0
assert config.policy["automodel_cfg"]["activation_checkpointing"]
assert config.policy["generation"]["sglang_cfg"]["sglang_server_config"]["num_gpus"] == 1
assert not config.policy["optimizer"]["kwargs"]["foreach"]
assert config.policy["model_name"] == "/flint-model"
assert config.grpo.num_prompts_per_step == 32
assert config.grpo.num_generations_per_prompt == 8
assert config.policy["train_global_batch_size"] == 128
assert config.policy["generation"]["sglang_cfg"]["max_total_tokens"] == 4096
assert not (Path(config.policy["model_name"]) / ".PROXY_BUILD_INCOMPLETE").exists()
print(json.dumps({"status": "passed", "native_registered": "FlintForCausalLM" in registry.MODEL_ARCH_MAPPING,
                  "config_class": type(hf).__module__ + "." + type(hf).__qualname__,
                  "layers": hf.num_hidden_layers, "layer_types": hf.layer_types,
                  "prompts": config.grpo.num_prompts_per_step,
                  "responses_per_prompt": config.grpo.num_generations_per_prompt,
                  "train_global_batch_size": config.policy["train_global_batch_size"],
                  "serving_max_total_tokens": config.policy["generation"]["sglang_cfg"]["max_total_tokens"],
                  "backend": config.policy["generation"]["backend"]}, indent=2))
