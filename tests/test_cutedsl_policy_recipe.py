from pathlib import Path

from omegaconf import OmegaConf

from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECIPE_NAME = "grpo-qwen3-30ba3b-1n4g-megatron-mxfp8-cutedsl"
RECIPE_PATH = (
    PROJECT_ROOT / "examples/configs/recipes/llm/performance" / f"{RECIPE_NAME}.yaml"
)

register_omegaconf_resolvers()


def test_cutedsl_policy_recipe_contract() -> None:
    config = OmegaConf.to_container(load_config(RECIPE_PATH), resolve=True)
    assert isinstance(config, dict)

    policy = config["policy"]
    megatron_cfg = policy["megatron_cfg"]
    fp8_cfg = megatron_cfg["fp8_cfg"]

    assert policy["model_name"] == "Qwen/Qwen3-30B-A3B"
    assert config["cluster"]["num_nodes"] == 1
    assert config["cluster"]["gpus_per_node"] == 4
    assert megatron_cfg["tensor_model_parallel_size"] == 1
    assert megatron_cfg["pipeline_model_parallel_size"] == 1
    assert megatron_cfg["context_parallel_size"] == 1
    assert megatron_cfg["expert_tensor_parallel_size"] == 1
    assert megatron_cfg["expert_model_parallel_size"] == 4
    assert fp8_cfg["enabled"] is True
    assert fp8_cfg["fp8"] == "e4m3"
    assert fp8_cfg["fp8_recipe"] == "mxfp8"
    assert fp8_cfg["fp8_param"] is False
    assert megatron_cfg["moe_grouped_gemm"] is True
    assert megatron_cfg["use_transformer_engine_op_fuser"] is True
    assert megatron_cfg["moe_mlp_glu_interleave_size"] == 32
    assert megatron_cfg["env_vars"]["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] == "1"
    assert policy["sequence_packing"]["enabled"] is False
    assert policy["dynamic_batching"]["enabled"] is False
    assert policy["train_global_batch_size"] == 4
    assert policy["train_micro_batch_size"] == 1
    assert policy["train_global_batch_size"] // policy["train_micro_batch_size"] == 4
    assert policy["max_total_sequence_length"] == 1024
    assert config["grpo"]["num_prompts_per_step"] == 2
    assert config["grpo"]["num_generations_per_prompt"] == 2
    assert config["grpo"]["max_num_steps"] == 3
    assert RECIPE_NAME in config["checkpointing"]["checkpoint_dir"]
    assert RECIPE_NAME in config["logger"]["log_dir"]
    assert config["logger"]["wandb"]["name"] == RECIPE_NAME
    assert policy["generation"]["backend"] == "vllm"
    assert policy["generation"]["colocated"]["enabled"] is True
    assert policy["generation"]["vllm_cfg"]["precision"] == "fp8"
    assert policy["generation"]["vllm_cfg"]["is_mx"] is True
