# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Refitted Policy Comparison Script.

This script compares logprobs between a Megatron policy and a vLLM policy
after performing model weight refitting. It demonstrates the workflow for
getting consistent logprobs across different inference backends.

Usage:
    uv run --extra mcore python3 tools/refit_verifier.py --model_name /path/to/model


Example Output:

--- Comparing Logprobs ---

Input prompt: The following are multiple choice questions (with answers) about world religions.

When was the first Buddhist temple constructed in Japan?
A. 325 CE
B. 119 CE
C. 451 CE
D. 596 CE
Answer:
Input tokens: tensor([200000,    954,   2182,    583,   6146,   9031,   5808,    330,   5992,
      8860,     21,   1509,   3817,  99867,   1574,   7022,    812,    290,
      1660, 120819,  55594,  24043,    310,  11197,   1044,     45,     26,
       220,  23325,  13607,    198,     46,     26,    220,  12860,  13607,
       198,     47,     26,    220,  34518,  13607,    198,     48,     26,
       220,  43145,  13607,    198,   4984,     38])

Comparing 10 generated tokens (from position 51 to 60):
vLLM generated logprobs: tensor([-7.0227, -7.1559, -6.4603, -6.7419, -6.3026, -6.8391, -6.3128, -6.6454,
    -7.1514, -6.8304])
Megatron generated logprobs: tensor([-7.0225, -7.1873, -6.4600, -6.7418, -6.3027, -6.8704, -6.2502, -6.6453,
    -7.1518, -6.8304])
Absolute difference: tensor([2.0981e-04, 3.1348e-02, 2.6035e-04, 1.6689e-04, 1.4973e-04, 3.1272e-02,
    6.2590e-02, 1.7643e-04, 3.2902e-04, 4.1485e-05])
Mean absolute difference: 0.012654399499297142
Max absolute difference: 0.06259012222290039

--- Token-by-Token Comparison (Generated Tokens Only) ---
Token           Token ID   Position   vLLM         Megatron     Diff
---------------------------------------------------------------------------
tok_51          pos_51     51         -7.022674    -7.022464    0.000210
tok_52          pos_52     52         -7.155923    -7.187271    0.031348
tok_53          pos_53     53         -6.460307    -6.460047    0.000260
tok_54          pos_54     54         -6.741926    -6.741759    0.000167
tok_55          pos_55     55         -6.302569    -6.302719    0.000150
tok_56          pos_56     56         -6.839099    -6.870371    0.031272
tok_57          pos_57     57         -6.312774    -6.250184    0.062590
tok_58          pos_58     58         -6.645445    -6.645269    0.000176
tok_59          pos_59     59         -7.151441    -7.151770    0.000329
tok_60          pos_60     60         -6.830355    -6.830397    0.000041
"""

import argparse
import copy
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import ray
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from nemo_rl.algorithms.grpo import refit_policy_generation
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.dynamo import DynamoGeneration
from nemo_rl.models.generation.sglang.sglang_generation import SGLangGeneration
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare Megatron and vLLM policy logprobs after refitting"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="/root/checkpoints/llama4-scout-custom-init",
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=1,
        help="Tensor parallelism size (TP) for Megatron",
    )
    parser.add_argument(
        "--ep_size",
        type=int,
        default=1,
        help="Expert parallelism size (EP) for Megatron",
    )
    parser.add_argument(
        "--pp_size",
        type=int,
        default=1,
        help="Pipeline parallelism size (PP) for Megatron",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=10,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=256,
        help="Maximum total sequence length",
    )
    parser.add_argument(
        "--refit_buffer_size_gb", type=int, default=4, help="Refit buffer size in GB"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Here is a short introduction to me:",
        help="Input prompt for generation",
    )

    return parser.parse_args()


def setup_configs(args, tokenizer):
    """Setup configuration dictionaries for Megatron and vLLM.

    Args:
        args: Parsed command line arguments
        tokenizer: HuggingFace tokenizer

    Returns:
        tuple: (megatron_config, vllm_config)
    """
    # Megatron Configuration
    megatron_config = {
        "model_name": args.model_name,
        "training_backend": "megatron",
        "train_global_batch_size": 1,
        "train_micro_batch_size": 1,
        "generation_batch_size": 2,
        "learning_rate": 0.0001,
        "logprob_batch_size": 1,
        "generation": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": None,
            "max_total_sequence_length": args.max_sequence_length,
            "max_new_tokens": args.max_sequence_length,
            "do_sample": False,
            "pad_token_id": tokenizer.eos_token_id,
            "colocated": {
                "enabled": True,
                "resources": {
                    "gpus_per_node": None,
                    "num_nodes": None,
                },
            },
        },
        "precision": "bfloat16",
        "offload_optimizer_for_logprob": False,
        "pipeline_dtype": "bfloat16",
        "parallel_output": True,
        "max_total_sequence_length": args.max_sequence_length,
        "fsdp_offload_enabled": False,
        "max_grad_norm": 1.0,
        "refit_buffer_size_gb": args.refit_buffer_size_gb,
        "make_sequence_length_divisible_by": args.tp_size,
        "optimizer": {
            "type": "adam",
            "kwargs": {
                "lr": 0.0001,
                "weight_decay": 0.0,
                "eps": 1e-8,
            },
        },
        "dtensor_cfg": {
            "enabled": False,
        },
        "dynamic_batching": {
            "enabled": False,
            "train_mb_tokens": 256,
            "logprob_mb_tokens": 256,
            "sequence_length_round": 64,
        },
        "sequence_packing": {
            "enabled": False,
        },
        "megatron_cfg": {
            "enabled": True,
            "empty_unused_memory_level": 1,
            "tensor_model_parallel_size": args.tp_size,
            "sequence_parallel": False,
            "expert_tensor_parallel_size": args.tp_size,
            "expert_model_parallel_size": args.ep_size,
            "pipeline_model_parallel_size": args.pp_size,
            "context_parallel_size": 1,
            "num_layers_in_first_pipeline_stage": None,
            "num_layers_in_last_pipeline_stage": None,
            "activation_checkpointing": False,
            "moe_router_dtype": "fp64",
            "moe_router_load_balancing_type": "none",
            "moe_router_bias_update_rate": 0.0,
            "moe_permute_fusion": False,
            "pipeline_dtype": "bfloat16",
            "train_iters": 1,
            "bias_activation_fusion": False,
            "moe_per_layer_logging": False,
            "freeze_moe_router": False,
            "apply_rope_fusion": False,
            "gradient_accumulation_fusion": False,
            "use_fused_weighted_squared_relu": False,
            "optimizer": {
                "optimizer": "adam",
                "lr": 5.0e-6,
                "min_lr": 5.0e-7,
                "weight_decay": 0.01,
                "bf16": False,
                "fp16": False,
                "params_dtype": "float32",
                # Adam optimizer settings
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "adam_eps": 1e-8,
                # SGD optimizer settings
                "sgd_momentum": 0.9,
                # Distributed optimizer settings
                "use_distributed_optimizer": True,
                "use_precision_aware_optimizer": True,
                "clip_grad": 1.0,
                # Optimizer CPU offload settings
                "optimizer_cpu_offload": False,
                "optimizer_offload_fraction": 0.0,
            },
            "scheduler": {
                "start_weight_decay": 0.01,
                "end_weight_decay": 0.01,
                "weight_decay_incr_style": "constant",
                "lr_decay_style": "constant",
                "lr_decay_iters": None,
                "lr_warmup_iters": 50,
                "lr_warmup_init": 5.0e-7,
            },
            "distributed_data_parallel_config": {
                "grad_reduce_in_fp32": False,
                "overlap_grad_reduce": False,
                "overlap_param_gather": False,
                "use_custom_fsdp": False,
                "data_parallel_sharding_strategy": "optim_grads_params",
            },
        },
    }

    # vLLM Configuration (match new VllmGeneration expectations: TP/PP/EP provided separately)
    vllm_config = {
        "backend": "vllm",
        "model_name": args.model_name,
        "tokenizer": {
            "name": args.model_name,
        },
        "dtype": "bfloat16",
        "max_new_tokens": args.max_new_tokens,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": None,
        "stop_token_ids": None,
        "stop_strings": None,
        "vllm_cfg": {
            "tensor_parallel_size": args.tp_size,
            "pipeline_parallel_size": args.pp_size,
            "expert_parallel_size": args.ep_size,
            "gpu_memory_utilization": 0.6,
            "max_model_len": args.max_sequence_length,
            "precision": "bfloat16",
            "async_engine": False,
            "skip_tokenizer_init": False,
            "load_format": "dummy",
            "enforce_eager": "False",
        },
        "colocated": {
            "enabled": True,
            "resources": {
                "gpus_per_node": None,
                "num_nodes": None,
            },
        },
        "vllm_kwargs": {},
    }

    # Configure vLLM with tokenizer
    vllm_config = configure_generation_config(vllm_config, tokenizer)

    return megatron_config, vllm_config


def setup_clusters_and_policies(args, megatron_config, vllm_config, tokenizer):
    """Setup Ray clusters and initialize policies.

    Args:
        args: Parsed command line arguments
        megatron_config: Megatron configuration dictionary
        vllm_config: vLLM configuration dictionary
        tokenizer: HuggingFace tokenizer

    Returns:
        tuple: (megatron_cluster, policy, vllm_inference_policy)
    """
    gpus_per_node = args.tp_size * args.ep_size * args.pp_size
    print(f"Setting up Megatron Cluster with TP={gpus_per_node}")
    megatron_cluster = RayVirtualCluster(
        name="megatron_cluster",
        bundle_ct_per_node_list=[gpus_per_node],
        use_gpus=True,
        num_gpus_per_node=gpus_per_node,
        max_colocated_worker_groups=2,
    )

    print("Instantiating Policy with Megatron backend...")
    policy = Policy(
        cluster=megatron_cluster,
        config=megatron_config,
        tokenizer=tokenizer,
        init_reference_model=False,
        init_optimizer=False,
    )

    # Create vLLM inference configuration with limited generation
    vllm_inference_config = vllm_config.copy()
    vllm_inference_config["max_new_tokens"] = args.max_new_tokens
    vllm_inference_config = configure_generation_config(
        vllm_inference_config, tokenizer
    )

    # Create vLLM policy for inference-only logprobs
    vllm_inference_policy = VllmGeneration(
        cluster=megatron_cluster, config=vllm_inference_config
    )

    return megatron_cluster, policy, vllm_inference_policy


def prepare_input_data(prompt, tokenizer):
    """Tokenize the input prompt and prepare generation data.

    Args:
        prompt: Input text prompt
        tokenizer: HuggingFace tokenizer

    Returns:
        BatchedDataDict: Prepared input data
    """
    print("Preparing input data...")

    # Tokenize the prompt
    tokenized = tokenizer(
        [prompt],
        padding=True,
        truncation=True,
        return_tensors="pt",
        padding_side="right",
    )

    # Calculate input lengths from attention mask
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    input_lengths = attention_mask.sum(dim=1).to(torch.int32)

    generation_data = BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
        }
    )

    return generation_data


def run_model_refitting(policy, vllm_inference_policy, refit_buffer_size_gb):
    """Perform model weight refitting between Megatron and vLLM policies.

    Args:
        policy: Megatron policy
        vllm_inference_policy: vLLM inference policy
        refit_buffer_size_gb: Buffer size for refitting in GB
    """
    print("\n--- Performing Model Refitting ---")

    # Perform the refitting between policies using GRPO's refit function
    # Note: colocated_inference=True since we're using the same cluster
    refit_policy_generation(
        policy,
        vllm_inference_policy,
        colocated_inference=True,
        _refit_buffer_size_gb=refit_buffer_size_gb,
    )
    print("Model refitting completed")


def generate_and_compare_logprobs(policy, vllm_inference_policy, generation_data):
    """Generate outputs and compare logprobs between vLLM and Megatron policies.

    Args:
        policy: Megatron policy
        vllm_inference_policy: vLLM inference policy
        generation_data: Input data for generation

    Returns:
        tuple: (vllm_logprobs_data, megatron_generation_data)
    """
    # Generate with vLLM for logprobs
    print("\n--- Getting vLLM Policy Logprobs ---")
    vllm_logprobs_data = vllm_inference_policy.generate(generation_data, greedy=True)
    print(f"vLLM Logprobs shape: {vllm_logprobs_data['logprobs'].shape}")
    print(f"vLLM Logprobs sample: {vllm_logprobs_data['logprobs'][0, -10:]}")

    # Generate with Megatron policy
    print("\n--- Getting Megatron Generation ---")
    policy.prepare_for_generation()

    # Prepare input data for Megatron using vLLM outputs
    megatron_input_data = copy.deepcopy(generation_data)
    print("=" * 100)
    print(megatron_input_data)
    print(vllm_logprobs_data)
    output_ids = vllm_logprobs_data["output_ids"]
    output_lengths = vllm_logprobs_data["unpadded_sequence_lengths"]
    megatron_input_data["input_ids"] = output_ids
    megatron_input_data["input_lengths"] = output_lengths
    positions = torch.arange(output_ids.shape[1]).unsqueeze(0)
    megatron_input_data["token_mask"] = positions < output_lengths.unsqueeze(1)
    megatron_input_data["sample_mask"] = torch.ones(
        output_ids.shape[0], dtype=torch.bool
    )

    # Get logprobs from Megatron
    policy.prepare_for_lp_inference()
    megatron_generation_data = policy.get_logprobs(megatron_input_data)
    print(f"Megatron Generation shape: {megatron_generation_data['logprobs'].shape}")
    print(
        f"Megatron Generation sample: {megatron_generation_data['logprobs'][0, -10:]}"
    )

    return vllm_logprobs_data, megatron_generation_data


def analyze_logprob_differences(
    vllm_logprobs_data, megatron_generation_data, generation_data, tokenizer, prompt
):
    """Analyze and display differences between vLLM and Megatron logprobs.

    Args:
        vllm_logprobs_data: vLLM generation results
        megatron_generation_data: Megatron generation results
        generation_data: Original input data
        tokenizer: HuggingFace tokenizer
        prompt: Original input prompt
    """
    print("\n--- Comparing Logprobs ---")
    print(f"Input prompt: {prompt}")
    print(
        f"Input tokens: {generation_data['input_ids'][0, : generation_data['input_lengths'][0]]}"
    )

    # Extract generation parameters
    input_length = generation_data["input_lengths"][0].item()
    total_length = vllm_logprobs_data["logprobs"].shape[1]
    generated_length = vllm_logprobs_data["generation_lengths"][0].item()

    if generated_length > 0:
        print(
            f"\nComparing {generated_length} generated tokens (from position {input_length} to {total_length - 1}):"
        )

        # Extract generated logprobs
        vllm_gen_logprobs = vllm_logprobs_data["logprobs"][0, input_length:total_length]
        megatron_gen_logprobs = megatron_generation_data["logprobs"][
            0, input_length:total_length
        ]

        print(f"vLLM generated logprobs: {vllm_gen_logprobs}")
        print(f"Megatron generated logprobs: {megatron_gen_logprobs}")

        # Calculate and display differences
        abs_diff = torch.abs(vllm_gen_logprobs - megatron_gen_logprobs)
        print(f"Absolute difference: {abs_diff}")
        print(f"Mean absolute difference: {torch.mean(abs_diff)}")
        print(f"Max absolute difference: {torch.max(abs_diff)}")

        # Detailed token-by-token comparison
        _detailed_token_comparison(
            vllm_gen_logprobs,
            megatron_gen_logprobs,
            vllm_logprobs_data,
            input_length,
            total_length,
            tokenizer,
        )
    else:
        print(
            f"No generated tokens to compare (input_length: {input_length}, total_length: {total_length})"
        )


def _detailed_token_comparison(
    vllm_logprobs,
    megatron_logprobs,
    vllm_logprobs_data,
    input_length,
    total_length,
    tokenizer,
):
    """Display detailed token-by-token comparison of logprobs.

    Args:
        vllm_logprobs: vLLM logprobs for generated tokens
        megatron_logprobs: Megatron logprobs for generated tokens
        vllm_logprobs_data: Vllm generation data
        input_length: Length of input sequence
        total_length: Total sequence length
        tokenizer: HuggingFace tokenizer
    """
    print("\n--- Token-by-Token Comparison (Generated Tokens Only) ---")

    if total_length > input_length:
        # Get generated tokens if available
        if "output_ids" in vllm_logprobs_data:
            generated_tokens = vllm_logprobs_data["output_ids"][
                0, input_length:total_length
            ]
        else:
            generated_tokens = torch.arange(input_length, total_length)

        # Display header
        print(
            f"{'Token':<15} {'Token ID':<10} {'Position':<10} {'vLLM':<12} {'Megatron':<12} {'Diff':<12}"
        )
        print("-" * 75)

        # Display each token comparison
        for i, pos in enumerate(range(input_length, total_length)):
            if "output_ids" in vllm_logprobs_data:
                token_id = generated_tokens[i].item()
                token_text = tokenizer.decode([token_id])
            else:
                token_id = f"pos_{pos}"
                token_text = f"tok_{pos}"

            vllm_lp = vllm_logprobs[i].item()
            megatron_lp = megatron_logprobs[i].item()
            diff = abs(vllm_lp - megatron_lp)

            print(
                f"{token_text:<15} {token_id:<10} {pos:<10} {vllm_lp:<12.6f} {megatron_lp:<12.6f} {diff:<12.6f}"
            )
    else:
        print("No generated tokens to compare in detail.")


def cleanup_resources(vllm_inference_policy):
    """Clean up resources and shutdown policies.

    Args:
        vllm_inference_policy: vLLM policy to shutdown
    """
    print("\n--- Cleaning up ---")
    vllm_inference_policy.shutdown()
    print("Cleanup completed successfully!")


def init_sglang(inference_cluster, generation_config):
    """Initialize SGLang generation workers and snapshot/reset weights for verification.

    Args:
        inference_cluster: Ray virtual cluster for inference workers.
        generation_config: SGLang generation config (typed as SGLangConfig).

    Returns:
        Tuple of (SGLangGeneration instance, initialization wall time in seconds).
    """
    t0 = time.perf_counter()
    pg = SGLangGeneration(
        cluster=inference_cluster,
        sglang_cfg=generation_config,
    )
    pg.check_weights(action="snapshot")
    pg.check_weights(action="reset")
    pg.finish_generation()
    return pg, time.perf_counter() - t0


def initialize_generation_with_policy(
    init_generation_fn: Callable,
    init_policy_fn: Callable,
    init_time_key: str,
    colocated_inference: bool,
    worker_init_timing_metrics: dict,
    policy: Policy | None = None,
):
    """Initialize SGLang generation + policy, then run the weight-equality check.

    Verifier-only variant: after both sides are up and refit metadata is
    exchanged, this always refits the policy weights into SGLang and calls
    `check_weights(action="compare")` against the snapshot taken in
    `init_sglang`. Production GRPO uses its own copy in `grpo.setup`.

    Args:
        init_generation_fn: Callable returning (engine, init_time_s).
        init_policy_fn: Callable returning (policy, init_time_s).
        init_time_key: Metrics key for generation init time.
        colocated_inference: Whether inference is colocated with training.
        worker_init_timing_metrics: Dict populated with init/parallel timing.
        policy: Optional pre-initialized policy; if set, init_policy_fn is skipped.

    Returns:
        Tuple of (policy_generation, policy).
    """
    use_parallel_init = not colocated_inference and policy is None

    if use_parallel_init:
        print(
            "  ⚡ Using parallel worker initialization (non-colocated mode)",
            flush=True,
        )
        parallel_start_time = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as executor:
            generation_future = executor.submit(init_generation_fn)
            policy_future = executor.submit(init_policy_fn)
            policy_generation, generation_time = generation_future.result()
            policy, policy_time = policy_future.result()
        parallel_wall_time = time.perf_counter() - parallel_start_time

        worker_init_timing_metrics[init_time_key] = generation_time
        worker_init_timing_metrics["policy_init_time_s"] = policy_time
        worker_init_timing_metrics["parallel_wall_time_s"] = parallel_wall_time
        worker_init_timing_metrics["parallel_init_enabled"] = True
    else:
        print(
            "  ⚙️  Using sequential worker initialization (colocated mode)",
            flush=True,
        )
        policy_generation, generation_time = init_generation_fn()
        worker_init_timing_metrics[init_time_key] = generation_time

        if policy is None:
            policy, policy_time = init_policy_fn()
            worker_init_timing_metrics["policy_init_time_s"] = policy_time
        worker_init_timing_metrics["parallel_init_enabled"] = 0.0

    state_dict_info = policy.prepare_refit_info()
    policy_generation.prepare_refit_info(state_dict_info)

    refit_policy_generation(
        policy=policy,
        policy_generation=policy_generation,
        colocated_inference=colocated_inference,
    )
    policy_generation.check_weights(action="compare")
    policy_generation.finish_generation()
    policy.prepare_for_training()

    return policy_generation, policy


def parse_sglang_args():
    """Parse args for the sglang weight-check flow: --config + hydra-style overrides."""
    parser = argparse.ArgumentParser(
        description="SGLang weight-update verification via the same YAML config GRPO uses"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config file (same file used by run_grpo.py).",
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def parse_dynamo_args():
    """Parse the managed Dynamo verifier config and Hydra-style overrides."""
    parser = argparse.ArgumentParser(
        description="Managed Dynamo native NCCL refit and logprob verification"
    )
    parser.add_argument(
        "--dynamo-config",
        type=str,
        required=True,
        help="GRPO YAML using backend=dynamo and dynamo_cfg.deployment=ray.",
    )
    parser.add_argument(
        "--max-logprob-diff",
        type=float,
        default=0.12,
        help=(
            "Maximum accepted absolute generated-token logprob difference "
            "between BF16 DTensor and vLLM attention implementations."
        ),
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Here is a short introduction to me:",
    )
    parser.add_argument(
        "--initial-load-format",
        choices=("dummy", "auto"),
        default="dummy",
        help=(
            "Initial Dynamo/vLLM load format. The acceptance path uses dummy; "
            "auto is available to diagnose direct HF loading before refit."
        ),
    )
    parser.add_argument(
        "--compare-before-refit",
        action="store_true",
        help=(
            "Compare directly loaded Dynamo/vLLM weights before native NCCL refit. "
            "Requires --initial-load-format=auto."
        ),
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def _prepare_dynamo_verifier_policy_config(policy_config: dict) -> None:
    """Fill training-only Megatron fields required during model construction.

    The GRPO setup path derives ``train_iters`` from the dataloader and run
    limits before constructing ``Policy``.  The refit verifier intentionally
    has no dataloader or optimizer, but Megatron still validates that the field
    exists while building the inference-capable policy.  One scheduler step is
    sufficient for this inference-only verifier; preserve an explicit value.
    """
    megatron_cfg = policy_config.get("megatron_cfg", {})
    if megatron_cfg.get("enabled", False):
        megatron_cfg.setdefault("train_iters", 1)


def _validate_dynamo_verifier_diagnostic_mode(
    initial_load_format: str, compare_before_refit: bool
) -> None:
    """Reject a meaningless comparison against intentionally dummy weights."""
    if compare_before_refit and initial_load_format != "auto":
        raise ValueError("--compare-before-refit requires --initial-load-format=auto.")


def _max_generated_logprob_diff(
    generation_data: BatchedDataDict,
    generation_output: BatchedDataDict,
    policy_output: BatchedDataDict,
) -> float:
    """Return the maximum generated-token logprob difference."""
    input_length = int(generation_data["input_lengths"][0].item())
    total_length = generation_output["logprobs"].shape[1]
    if total_length <= input_length:
        raise AssertionError("Managed Dynamo verifier generated no output tokens.")
    return torch.max(
        torch.abs(
            generation_output["logprobs"][0, input_length:total_length]
            - policy_output["logprobs"][0, input_length:total_length]
        )
    ).item()


def main_dynamo():
    """Run managed Dynamo from dummy weights, refit, and compare logprobs."""
    args, overrides = parse_dynamo_args()
    _validate_dynamo_verifier_diagnostic_mode(
        args.initial_load_format, args.compare_before_refit
    )
    register_omegaconf_resolvers()
    config = load_config(args.dynamo_config)
    if overrides:
        config = parse_hydra_overrides(config, overrides)
    master_config = OmegaConf.to_container(config, resolve=True)

    init_ray()
    policy_config = master_config["policy"]
    _prepare_dynamo_verifier_policy_config(policy_config)
    generation_config = policy_config["generation"]
    cluster_config = master_config["cluster"]
    assert generation_config["backend"] == "dynamo", (
        "--dynamo-config requires policy.generation.backend=dynamo"
    )
    assert generation_config["dynamo_cfg"].get("deployment") == "ray", (
        "Dynamo refit verification requires dynamo_cfg.deployment=ray"
    )
    assert not generation_config["colocated"]["enabled"], (
        "Managed Dynamo refit verification requires colocated.enabled=false"
    )

    # Compare raw model logprobs, not backend-specific renormalization after
    # temperature/top-k/top-p filtering. Keep the deterministic decode short so
    # the verifier tests refit correctness rather than sampling behavior.
    generation_config["temperature"] = 1.0
    generation_config["top_p"] = 1.0
    generation_config["top_k"] = None
    generation_config["max_new_tokens"] = min(
        int(generation_config["max_new_tokens"]), 10
    )

    tokenizer = get_tokenizer(policy_config["tokenizer"])
    generation_config = configure_generation_config(generation_config, tokenizer)
    generation_config["model_name"] = policy_config["model_name"]
    # The acceptance path starts from dummy weights, so a successful post-refit
    # comparison proves that native NCCL replaced them. Direct HF loading is a
    # diagnostic control for separating transfer errors from backend/model
    # numerical differences.
    generation_config["vllm_cfg"]["load_format"] = args.initial_load_format

    inference_resources = generation_config["colocated"]["resources"]
    inference_gpus_per_node = int(inference_resources["gpus_per_node"])
    total_nodes = int(cluster_config["num_nodes"])
    total_gpus_per_node = int(cluster_config["gpus_per_node"])
    if total_nodes == 1:
        inference_nodes = 1
        train_nodes = 1
        train_gpus_per_node = total_gpus_per_node - inference_gpus_per_node
    else:
        inference_nodes = int(inference_resources["num_nodes"])
        train_nodes = total_nodes - inference_nodes
        train_gpus_per_node = total_gpus_per_node
    if train_nodes <= 0 or train_gpus_per_node <= 0:
        raise ValueError(
            "Dynamo verifier requires non-empty train and inference GPU pools."
        )

    train_cluster = RayVirtualCluster(
        name="dynamo_refit_train_cluster",
        bundle_ct_per_node_list=[train_gpus_per_node] * train_nodes,
        use_gpus=True,
        num_gpus_per_node=train_gpus_per_node,
        max_colocated_worker_groups=1,
    )
    inference_cluster = RayVirtualCluster(
        name="dynamo_refit_inference_cluster",
        bundle_ct_per_node_list=[inference_gpus_per_node] * inference_nodes,
        use_gpus=True,
        num_gpus_per_node=inference_gpus_per_node,
        max_colocated_worker_groups=1,
    )

    policy = None
    policy_generation = None
    try:
        policy = Policy(
            cluster=train_cluster,
            config=policy_config,
            tokenizer=tokenizer,
            init_reference_model=False,
            init_optimizer=False,
        )
        policy_generation = DynamoGeneration(
            cluster=inference_cluster,
            config=generation_config,
            tokenizer=tokenizer,
            tokenizer_config=policy_config.get("tokenizer"),
        )

        generation_data = prepare_input_data(args.prompt, tokenizer)
        if args.compare_before_refit:
            direct_data, direct_policy_data = generate_and_compare_logprobs(
                policy, policy_generation, generation_data
            )
            analyze_logprob_differences(
                direct_data,
                direct_policy_data,
                generation_data,
                tokenizer,
                args.prompt,
            )
            direct_diff = _max_generated_logprob_diff(
                generation_data, direct_data, direct_policy_data
            )
            print(f"Managed Dynamo direct-load max logprob difference: {direct_diff}")

        ip, port = train_cluster.get_master_address_and_port()
        train_world_size = train_cluster.world_size()
        inference_world_size = policy_generation.get_inference_world_size()
        world_size = train_world_size + inference_world_size
        train_refs = policy.init_collective(
            ip,
            port,
            world_size,
            train_world_size=train_world_size,
            nccl_peer="vllm",
        )
        inference_refs = policy_generation.init_collective(
            ip,
            port,
            world_size,
            train_world_size=train_world_size,
        )
        ray.get(train_refs + inference_refs)

        state_dict_info = policy.prepare_refit_info()
        policy_generation.prepare_refit_info(state_dict_info)
        refit_policy_generation(
            policy,
            policy_generation,
            colocated_inference=False,
        )
        policy_generation.invalidate_kv_cache()

        dynamo_data, policy_data = generate_and_compare_logprobs(
            policy, policy_generation, generation_data
        )
        analyze_logprob_differences(
            dynamo_data,
            policy_data,
            generation_data,
            tokenizer,
            args.prompt,
        )
        max_diff = _max_generated_logprob_diff(
            generation_data, dynamo_data, policy_data
        )
        if max_diff > args.max_logprob_diff:
            raise AssertionError(
                f"Post-refit max logprob difference {max_diff} exceeds "
                f"{args.max_logprob_diff}."
            )
        print(f"Managed Dynamo post-refit max logprob difference: {max_diff}")
    finally:
        if policy_generation is not None:
            policy_generation.shutdown()
        if policy is not None:
            policy.shutdown()
        ray.shutdown()


def main_sglang():
    """Load the GRPO YAML config and run the SGLang weight-equality check.

    Mirrors run_grpo.py's config-loading pipeline and grpo.setup's colocated
    cluster bootstrap, but skips dataset/loss/logger/checkpointer/grpo_state
    since the verifier only needs to refit weights once and diff against the
    pre-reset snapshot. Generation backend must be sglang.
    """
    args, overrides = parse_sglang_args()

    register_omegaconf_resolvers()
    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")
    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)
    master_config = OmegaConf.to_container(config, resolve=True)

    init_ray()

    tokenizer = get_tokenizer(master_config["policy"]["tokenizer"])

    policy_config = master_config["policy"]
    generation_config = master_config["policy"]["generation"]
    cluster_config = master_config["cluster"]
    assert generation_config is not None, "policy.generation must be set in the config"
    assert generation_config["backend"] == "sglang", (
        f"refit_verifier sglang mode requires backend=sglang, got "
        f"{generation_config['backend']!r}"
    )
    assert generation_config["colocated"]["enabled"], (
        "refit_verifier sglang mode currently only supports colocated inference"
    )

    generation_config = configure_generation_config(generation_config, tokenizer)
    generation_config["model_name"] = policy_config["model_name"]
    generation_config["sglang_cfg"].setdefault(
        "model_path", policy_config["model_name"]
    )

    policy_nodes = cluster_config["num_nodes"]
    policy_gpus_per_node = cluster_config["gpus_per_node"]
    cluster = RayVirtualCluster(
        name="refit_verifier_cluster",
        bundle_ct_per_node_list=[policy_gpus_per_node] * policy_nodes,
        use_gpus=True,
        num_gpus_per_node=policy_gpus_per_node,
        max_colocated_worker_groups=2,
    )
    print(
        f"  ✓ Ray cluster initialized with {policy_nodes} nodes "
        f"× {policy_gpus_per_node} GPUs/node",
        flush=True,
    )

    def init_policy_fn():
        t0 = time.perf_counter()
        p = Policy(
            cluster=cluster,
            config=policy_config,
            tokenizer=tokenizer,
            init_reference_model=False,
            init_optimizer=False,
        )
        return p, time.perf_counter() - t0

    worker_init_timing_metrics: dict = {}
    policy_generation, _ = initialize_generation_with_policy(
        init_generation_fn=lambda: init_sglang(cluster, generation_config),
        init_policy_fn=init_policy_fn,
        init_time_key="sglang_init_time_s",
        colocated_inference=True,
        worker_init_timing_metrics=worker_init_timing_metrics,
    )

    print("\n--- SGLang weight-check timing ---")
    for k, v in worker_init_timing_metrics.items():
        print(f"  {k}: {v}")

    policy_generation.shutdown()
    print("SGLang weight-check completed successfully!")


def _is_sglang_mode() -> bool:
    """Detect the sglang flow by the presence of --config in argv.

    The vLLM path has no --config argument, so its presence unambiguously
    signals the YAML-driven sglang flow. Inspecting sys.argv here keeps
    the existing vLLM path byte-identical.
    """
    import sys

    for tok in sys.argv[1:]:
        if tok == "--config" or tok.startswith("--config="):
            return True
    return False


def _is_dynamo_mode() -> bool:
    import sys

    return any(
        token == "--dynamo-config" or token.startswith("--dynamo-config=")
        for token in sys.argv[1:]
    )


def main_vllm():
    """VLLM weight-check flow."""
    # Parse command line arguments
    args = parse_args()

    # Initialize Ray
    ray.init()

    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Setup configurations
    megatron_config, vllm_config = setup_configs(args, tokenizer)

    # Setup clusters and policies
    megatron_cluster, policy, vllm_inference_policy = setup_clusters_and_policies(
        args, megatron_config, vllm_config, tokenizer
    )

    # Prepare input data
    generation_data = prepare_input_data(args.prompt, tokenizer)

    # prepare refit info
    state_dict_info = policy.prepare_refit_info()
    vllm_inference_policy.prepare_refit_info(state_dict_info)

    # Perform model refitting
    run_model_refitting(policy, vllm_inference_policy, args.refit_buffer_size_gb)

    # Generate and compare logprobs
    vllm_logprobs_data, megatron_generation_data = generate_and_compare_logprobs(
        policy, vllm_inference_policy, generation_data
    )

    # Analyze differences
    analyze_logprob_differences(
        vllm_logprobs_data,
        megatron_generation_data,
        generation_data,
        tokenizer,
        args.prompt,
    )

    # Cleanup
    cleanup_resources(vllm_inference_policy)


def main():
    """Dispatch to the Dynamo, SGLang, or vLLM weight-check flow."""
    if _is_dynamo_mode():
        main_dynamo()
    elif _is_sglang_mode():
        main_sglang()
    else:
        main_vllm()
    print("Script completed successfully!")


if __name__ == "__main__":
    main()
