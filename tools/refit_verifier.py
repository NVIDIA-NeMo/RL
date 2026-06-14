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

import ray
import torch
from transformers import AutoTokenizer

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation import configure_generation_config, create_generation
from nemo_rl.models.generation.constants import VLLM_BACKEND
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.weight_sync import create_weight_synchronizer


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
        "--vllm_tp_size",
        type=int,
        default=None,
        help="Tensor parallelism size (TP) for vLLM. Defaults to --tp_size.",
    )
    parser.add_argument(
        "--vllm_ep_size",
        type=int,
        default=None,
        help="Expert parallelism size (EP) for vLLM. Defaults to --ep_size.",
    )
    parser.add_argument(
        "--vllm_pp_size",
        type=int,
        default=None,
        help="Pipeline parallelism size (PP) for vLLM. Defaults to --pp_size.",
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
        "--pretrained_checkpoint_path",
        type=str,
        default=None,
        help=(
            "Optional Megatron checkpoint root or iter directory to load instead "
            "of converting model_name from Hugging Face format."
        ),
    )
    parser.add_argument(
        "--pretrained_checkpoint_format",
        type=str,
        choices=("megatron_bridge", "megatron_lm"),
        default="megatron_bridge",
        help="Format for --pretrained_checkpoint_path.",
    )
    parser.add_argument(
        "--num_refits",
        type=int,
        default=1,
        help="Number of refit passes to run before generation.",
    )
    parser.add_argument(
        "--non_colocated",
        action="store_true",
        help="Use separate policy and generation clusters with collective refit.",
    )
    parser.add_argument(
        "--policy_num_nodes",
        type=int,
        default=1,
        help="Number of nodes to reserve for the policy cluster.",
    )
    parser.add_argument(
        "--generation_num_nodes",
        type=int,
        default=1,
        help="Number of nodes to reserve for the generation cluster.",
    )
    parser.add_argument(
        "--policy_gpus_per_node",
        type=int,
        default=None,
        help="Policy GPUs per node. Defaults to TP * EP * PP.",
    )
    parser.add_argument(
        "--generation_gpus_per_node",
        type=int,
        default=None,
        help="Generation GPUs per node. Defaults to max(TP, EP) * PP.",
    )
    parser.add_argument(
        "--enable_delta_compression",
        action="store_true",
        help="Enable delta-compressed collective refit in non-colocated mode.",
    )
    parser.add_argument(
        "--delta_full_sync_interval",
        type=int,
        default=20,
        help="Successful delta refits between full baseline refreshes.",
    )
    parser.add_argument(
        "--delta_sparse_bucket_size_bytes",
        type=int,
        default=5 * 1024**3,
        help="Sparse payload bytes to bucket before broadcasting.",
    )
    parser.add_argument(
        "--delta_load_batch_size_bytes",
        type=int,
        default=512 * 1024**2,
        help="Decoded delta tensor bytes to batch before vLLM load_weights.",
    )
    parser.add_argument(
        "--vllm_enforce_eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run vLLM in eager mode. This avoids long compile/cudagraph startup in verifier runs.",
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.6,
        help="GPU memory utilization passed to vLLM.",
    )
    parser.add_argument(
        "--megatron_attention_backend",
        choices=("auto", "flash", "fused", "unfused", "local"),
        default=None,
        help="Optional Megatron attention backend override for verifier runs.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Here is a short introduction to me:",
        help="Input prompt for generation",
    )
    return parser.parse_args()


def resolve_cluster_sizes(args) -> None:
    """Fill cluster-size defaults derived from the requested model parallelism."""
    if args.vllm_tp_size is None:
        args.vllm_tp_size = args.tp_size
    if args.vllm_ep_size is None:
        args.vllm_ep_size = args.ep_size
    if args.vllm_pp_size is None:
        args.vllm_pp_size = args.pp_size

    if args.policy_gpus_per_node is None:
        args.policy_gpus_per_node = args.tp_size * args.ep_size * args.pp_size
    if args.generation_gpus_per_node is None:
        args.generation_gpus_per_node = (
            max(args.vllm_tp_size, args.vllm_ep_size) * args.vllm_pp_size
        )


def setup_configs(args, tokenizer):
    """Setup configuration dictionaries for Megatron and vLLM.

    Args:
        args: Parsed command line arguments
        tokenizer: HuggingFace tokenizer

    Returns:
        tuple: (megatron_config, vllm_config)
    """
    colocated = not args.non_colocated
    tensor_pipeline_size = args.tp_size * args.pp_size
    policy_world_size = args.policy_num_nodes * args.policy_gpus_per_node
    if policy_world_size % tensor_pipeline_size != 0:
        raise ValueError(
            "Policy world size must be divisible by TP * PP: "
            f"policy_world_size={policy_world_size}, TP={args.tp_size}, "
            f"PP={args.pp_size}"
        )
    train_global_batch_size = policy_world_size // tensor_pipeline_size

    megatron_config = {
        "model_name": args.model_name,
        "training_backend": "megatron",
        "train_global_batch_size": train_global_batch_size,
        "train_micro_batch_size": 1,
        "generation_batch_size": 2,
        "learning_rate": 0.0001,
        "logprob_batch_size": 1,
        "generation": {
            "backend": VLLM_BACKEND,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": None,
            "max_total_sequence_length": args.max_sequence_length,
            "max_new_tokens": args.max_sequence_length,
            "do_sample": False,
            "pad_token_id": tokenizer.eos_token_id,
            "colocated": {
                "enabled": colocated,
                "resources": {
                    "gpus_per_node": args.generation_gpus_per_node,
                    "num_nodes": args.generation_num_nodes,
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
            "moe_enable_deepep": False,
            "moe_token_dispatcher_type": "allgather",
            "moe_shared_expert_overlap": False,
            "moe_grouped_gemm": True,
            "pipeline_dtype": "bfloat16",
            "train_iters": 1,
            "bias_activation_fusion": False,
            "moe_per_layer_logging": False,
            "freeze_moe_router": False,
            "apply_rope_fusion": False,
            "gradient_accumulation_fusion": False,
            "optimizer": {
                "optimizer": "adam",
                "lr": 5.0e-6,
                "min_lr": 5.0e-7,
                "weight_decay": 0.01,
                "bf16": True,
                "fp16": False,
                "params_dtype": "float32",
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "adam_eps": 1e-8,
                "sgd_momentum": 0.9,
                "use_distributed_optimizer": True,
                "use_precision_aware_optimizer": True,
                "clip_grad": 1.0,
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
    if args.megatron_attention_backend is not None:
        megatron_config["megatron_cfg"]["attention_backend"] = (
            args.megatron_attention_backend
        )
    if args.pretrained_checkpoint_path is not None:
        megatron_config["pretrained_checkpoint"] = {
            "path": args.pretrained_checkpoint_path,
            "format": args.pretrained_checkpoint_format,
        }

    vllm_config = {
        "backend": VLLM_BACKEND,
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
            "tensor_parallel_size": args.vllm_tp_size,
            "pipeline_parallel_size": args.vllm_pp_size,
            "expert_parallel_size": args.vllm_ep_size,
            "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
            "max_model_len": args.max_sequence_length,
            "precision": "bfloat16",
            "async_engine": False,
            "skip_tokenizer_init": False,
            "load_format": "dummy",
            "enforce_eager": args.vllm_enforce_eager,
        },
        "colocated": {
            "enabled": colocated,
            "resources": {
                "gpus_per_node": args.generation_gpus_per_node,
                "num_nodes": args.generation_num_nodes,
            },
        },
        "vllm_kwargs": {},
    }
    if args.enable_delta_compression:
        if colocated:
            raise ValueError(
                "--enable_delta_compression requires --non_colocated because "
                "delta compression is only supported for collective refit."
            )
        delta_compression_config = {
            "enabled": True,
            "dtype": "bfloat16",
            "full_sync_interval": args.delta_full_sync_interval,
            "sparse_bucket_size_bytes": args.delta_sparse_bucket_size_bytes,
            "delta_load_batch_size_bytes": args.delta_load_batch_size_bytes,
        }
        vllm_config["delta_compression"] = delta_compression_config
        megatron_config["generation"]["delta_compression"] = delta_compression_config
        megatron_config["generation"]["vllm_cfg"] = {
            "precision": vllm_config["vllm_cfg"]["precision"]
        }

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
        tuple: (megatron_cluster, generation_cluster, policy, vllm_inference_policy)
    """
    print(
        f"Setting up Megatron Cluster with {args.policy_num_nodes} node(s), "
        f"{args.policy_gpus_per_node} GPU(s)/node"
    )
    megatron_cluster = RayVirtualCluster(
        name="megatron_cluster",
        bundle_ct_per_node_list=[args.policy_gpus_per_node] * args.policy_num_nodes,
        use_gpus=True,
        num_gpus_per_node=args.policy_gpus_per_node,
        max_colocated_worker_groups=1 if args.non_colocated else 2,
    )

    print("Instantiating Policy with Megatron backend...")
    policy = Policy(
        cluster=megatron_cluster,
        config=megatron_config,
        tokenizer=tokenizer,
        init_reference_model=False,
        init_optimizer=False,
    )

    generation_cluster = megatron_cluster
    if args.non_colocated:
        print(
            f"Setting up vLLM Cluster with {args.generation_num_nodes} node(s), "
            f"{args.generation_gpus_per_node} GPU(s)/node"
        )
        generation_cluster = RayVirtualCluster(
            name="vllm_cluster",
            bundle_ct_per_node_list=[args.generation_gpus_per_node]
            * args.generation_num_nodes,
            use_gpus=True,
            num_gpus_per_node=args.generation_gpus_per_node,
            max_colocated_worker_groups=1,
            placement_group_strategy="PACK",
        )

    vllm_inference_policy = create_generation(
        backend=vllm_config["backend"],
        cluster=generation_cluster,
        config=vllm_config,
    )

    return megatron_cluster, generation_cluster, policy, vllm_inference_policy


def prepare_input_data(prompt, tokenizer):
    """Tokenize the input prompt and prepare generation data.

    Args:
        prompt: Input text prompt
        tokenizer: HuggingFace tokenizer

    Returns:
        BatchedDataDict: Prepared input data
    """
    print("Preparing input data...")

    tokenized = tokenizer(
        [prompt],
        padding=True,
        truncation=True,
        return_tensors="pt",
        padding_side="right",
    )

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


def run_model_refitting(
    weight_sync,
    num_refits,
):
    """Perform model weight refitting between Megatron and vLLM policies.

    Args:
        weight_sync: Weight synchronizer for the policy/generation pair
        num_refits: Number of refit passes to run
    """
    print("\n--- Performing Model Refitting ---")

    if num_refits < 1:
        raise ValueError("--num_refits must be >= 1")

    for refit_idx in range(num_refits):
        print(f"Refit pass {refit_idx + 1}/{num_refits}")
        weight_sync.sync_weights()
        if refit_idx + 1 < num_refits:
            weight_sync.mark_stale()
    print("Model refitting completed")


def _repeat_batch_to_multiple(data, multiple):
    """Repeat batch-aligned tensors so Megatron DP sharding can split them evenly."""
    input_ids = data["input_ids"]
    batch_size = input_ids.shape[0]
    if batch_size % multiple == 0:
        return

    target_batch_size = ((batch_size + multiple - 1) // multiple) * multiple
    repeat_factor = (target_batch_size + batch_size - 1) // batch_size
    for key, value in data.items():
        if isinstance(value, torch.Tensor) and value.shape[:1] == (batch_size,):
            repeats = (repeat_factor,) + (1,) * (value.ndim - 1)
            data[key] = value.repeat(repeats)[:target_batch_size]

    print(
        "Repeated Megatron logprob comparison batch "
        f"from {batch_size} to {target_batch_size} for DP={multiple}"
    )


def generate_and_compare_logprobs(policy, vllm_inference_policy, generation_data):
    """Generate outputs and compare logprobs between vLLM and Megatron policies.

    Args:
        policy: Megatron policy
        vllm_inference_policy: vLLM inference policy
        generation_data: Input data for generation

    Returns:
        tuple: (vllm_logprobs_data, megatron_generation_data)
    """
    print("\n--- Getting vLLM Policy Logprobs ---")
    vllm_logprobs_data = vllm_inference_policy.generate(generation_data, greedy=True)
    print(f"vLLM Logprobs shape: {vllm_logprobs_data['logprobs'].shape}")
    print(f"vLLM Logprobs sample: {vllm_logprobs_data['logprobs'][0, -10:]}")

    print("\n--- Getting Megatron Generation ---")
    policy.prepare_for_generation()

    megatron_input_data = BatchedDataDict(generation_data)
    megatron_input_data["input_ids"] = vllm_logprobs_data["output_ids"]
    megatron_input_data["input_lengths"] = vllm_logprobs_data[
        "unpadded_sequence_lengths"
    ]
    dp_size = policy.sharding_annotations.get_axis_size("data_parallel")
    _repeat_batch_to_multiple(megatron_input_data, dp_size)

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

    input_length = generation_data["input_lengths"][0].item()
    total_length = vllm_logprobs_data["logprobs"].shape[1]
    generated_length = vllm_logprobs_data["generation_lengths"][0].item()

    if generated_length > 0:
        print(
            f"\nComparing {generated_length} generated tokens (from position {input_length} to {total_length - 1}):"
        )

        vllm_gen_logprobs = vllm_logprobs_data["logprobs"][0, input_length:total_length]
        megatron_gen_logprobs = megatron_generation_data["logprobs"][
            0, input_length:total_length
        ]

        print(f"vLLM generated logprobs: {vllm_gen_logprobs}")
        print(f"Megatron generated logprobs: {megatron_gen_logprobs}")

        abs_diff = torch.abs(vllm_gen_logprobs - megatron_gen_logprobs)
        print(f"Absolute difference: {abs_diff}")
        print(f"Mean absolute difference: {torch.mean(abs_diff)}")
        print(f"Max absolute difference: {torch.max(abs_diff)}")

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
        if "output_ids" in vllm_logprobs_data:
            generated_tokens = vllm_logprobs_data["output_ids"][
                0, input_length:total_length
            ]
        else:
            generated_tokens = torch.arange(input_length, total_length)

        print(
            f"{'Token':<15} {'Token ID':<10} {'Position':<10} {'vLLM':<12} {'Megatron':<12} {'Diff':<12}"
        )
        print("-" * 75)

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


def cleanup_resources(
    policy,
    vllm_inference_policy,
    megatron_cluster=None,
    generation_cluster=None,
):
    """Clean up resources and shutdown policies.

    Args:
        policy: Megatron policy to shutdown
        vllm_inference_policy: vLLM policy to shutdown
        megatron_cluster: Policy cluster to shutdown
        generation_cluster: Optional separate generation cluster to shutdown
    """
    print("\n--- Cleaning up ---")

    resources = [
        vllm_inference_policy,
        policy,
    ]
    if generation_cluster is not None and generation_cluster is not megatron_cluster:
        resources.append(generation_cluster)
    resources.append(megatron_cluster)

    for resource in resources:
        if resource is None:
            continue
        resource.shutdown()

    print("Cleanup completed successfully!")


def main():
    """Main execution function."""
    args = parse_args()
    resolve_cluster_sizes(args)

    ray.init()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    megatron_cluster = None
    generation_cluster = None
    policy = None
    vllm_inference_policy = None
    try:
        megatron_config, vllm_config = setup_configs(args, tokenizer)
        (
            megatron_cluster,
            generation_cluster,
            policy,
            vllm_inference_policy,
        ) = setup_clusters_and_policies(args, megatron_config, vllm_config, tokenizer)

        print("\n--- Initializing weight synchronizer ---")
        weight_sync = create_weight_synchronizer(
            policy=policy,
            generation=vllm_inference_policy,
            generation_backend=VLLM_BACKEND,
            colocated=not args.non_colocated,
            train_cluster=megatron_cluster,
            inference_cluster=generation_cluster,
            refit_buffer_size_gb=args.refit_buffer_size_gb,
        )
        weight_sync.init_communicator()
        if args.non_colocated:
            print(
                "Collective refit initialized "
                f"(train_world_size={megatron_cluster.world_size()}, "
                f"inference_world_size={generation_cluster.world_size()})"
            )

        generation_data = prepare_input_data(args.prompt, tokenizer)
        run_model_refitting(
            weight_sync,
            num_refits=args.num_refits,
        )

        vllm_logprobs_data, megatron_generation_data = generate_and_compare_logprobs(
            policy, vllm_inference_policy, generation_data
        )

        analyze_logprob_differences(
            vllm_logprobs_data,
            megatron_generation_data,
            generation_data,
            tokenizer,
            args.prompt,
        )

        print("Script completed successfully!")
    finally:
        cleanup_resources(
            policy,
            vllm_inference_policy,
            megatron_cluster,
            generation_cluster,
        )
        ray.shutdown()


if __name__ == "__main__":
    main()
