#!/usr/bin/env python3

from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf

from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

MOE_8G_RECIPES = (
    "dapo-deepseek-v3-64n8g.v2.yaml",
    "grpo-deepseek-v3-32n8g.yaml",
    "grpo-deepseek-v3-64n8g.yaml",
    "grpo-deepseek-v3-64n8g-async-1off.yaml",
    "grpo-deepseek-v3-64n8g-fp8-async-1off.yaml",
    "grpo-nemotron3-super-120BA12B-32n8g.yaml",
    "grpo-nemotron3-super-120BA12B-32n8g-async-1off.yaml",
    "grpo-qwen3-235b-16n8g.yaml",
    "grpo-qwen3-235b-32n8g.yaml",
    "grpo-qwen3-235b-32n8g-async-1off.yaml",
    "grpo-qwen3-30ba3b-4n8g.yaml",
    "grpo-qwen3-30ba3b-4n8g-40K.yaml",
    "grpo-qwen3-30ba3b-4n8g-async-1off.yaml",
    "grpo-qwen3-30ba3b-24n8g-async-8off.yaml",
)

DENSE_8G_RECIPES = (
    "grpo-llama3.1-8b-instruct-2n8g.yaml",
    "grpo-llama3.1-8b-instruct-2n8g-async-1off.yaml",
    "grpo-llama3.1-8b-instruct-2n8g-fp8-async-1off.yaml",
    "grpo-qwen3-32b-4n8g.yaml",
    "grpo-qwen3-32b-8n8g-async-1off.yaml",
)

PREPAD_8G_RECIPES = {
    "grpo-nemotron3-super-120BA12B-32n8g.yaml",
    "grpo-nemotron3-super-120BA12B-32n8g-async-1off.yaml",
    "grpo-qwen3-30ba3b-4n8g.yaml",
    "grpo-qwen3-30ba3b-4n8g-40K.yaml",
    "grpo-qwen3-30ba3b-24n8g-async-8off.yaml",
}

HYBRIDEP_ENVIRONMENT = {
    "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": "8",
    "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": "128",
    "NVLINK_DOMAIN_SIZE": "8",
    "USE_MNNVL": "0",
}


def _recipe_dir() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "configs"
        / "recipes"
        / "llm"
        / "performance"
    )


def _megatron_config(recipe_name: str) -> dict[str, Any]:
    resolved = OmegaConf.to_container(
        load_config(_recipe_dir() / recipe_name), resolve=True
    )
    assert isinstance(resolved, dict), recipe_name
    policy = resolved["policy"]
    assert isinstance(policy, dict), recipe_name
    megatron_cfg = policy["megatron_cfg"]
    assert isinstance(megatron_cfg, dict), recipe_name
    return cast(dict[str, Any], megatron_cfg)


def main() -> None:
    register_omegaconf_resolvers()

    for recipe_name in MOE_8G_RECIPES:
        config = _megatron_config(recipe_name)
        environment = config.get("env_vars", {})
        assert isinstance(environment, dict), recipe_name
        assert config["moe_token_dispatcher_type"] == "flex", recipe_name
        assert config["moe_flex_dispatcher_backend"] == "hybridep", recipe_name
        assert config["moe_hybridep_num_sms"] == 32, recipe_name
        assert HYBRIDEP_ENVIRONMENT.items() <= environment.items(), recipe_name
        assert bool(config.get("moe_hybridep_prepad_packed_inputs", False)) is (
            recipe_name in PREPAD_8G_RECIPES
        ), recipe_name

    for recipe_name in DENSE_8G_RECIPES:
        config = _megatron_config(recipe_name)
        assert config["moe_token_dispatcher_type"] == "alltoall", recipe_name
        assert "moe_flex_dispatcher_backend" not in config, recipe_name

    print(
        f"Validated {len(MOE_8G_RECIPES)} HybridEP MoE recipes and "
        f"{len(DENSE_8G_RECIPES)} unchanged dense recipes."
    )


if __name__ == "__main__":
    main()
