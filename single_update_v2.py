#!/usr/bin/env python3
"""Minimal single-update demonstration script.

What it does:
 1) Sets up a RayVirtualCluster
 2) Initializes VllmGeneration
 3) Initializes LM Policy
 4) Trains on a tiny synthetic batch (global batch size = 2) with NLLLoss
 5) Refits the generation engine with the latest policy weights
 6) Optionally repeats the train→refit cycle in a short loop

Notes:
- The configuration is defined entirely in this file, inspired by examples/configs/grpo_math_1B.yaml
- Uses vLLM for generation and a small model for demonstration
- Uses simple NLL loss for brevity
"""

import os
from dataclasses import dataclass, fields
import argparse

import torch

from nemo_rl.algorithms.grpo import refit_policy_generation
from nemo_rl.algorithms.loss_functions import NLLLoss
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.models.policy.lm_policy import Policy


@dataclass
class Config:
    """Configuration for single update demonstration."""
    model: str = "Qwen/Qwen2.5-0.5B"
    num_gpus_per_node: int = 1
    num_nodes: int = 1

    @classmethod
    def parse_args(cls):
        parser = argparse.ArgumentParser()
        
        # Automatically create arguments from dataclass fields
        for field in fields(cls):
            parser.add_argument(
                f"--{field.name}", 
                type=field.type, 
                default=field.default
            )
        
        return cls(**vars(parser.parse_args()))


def build_policy_config(config: Config):
    """Create a minimal policy config inline, inspired by grpo_math_1B.yaml.

    Keep it small and simple for demonstration. Adjust model_name as desired.
    """
    model_name = config.model

    # Build tokenizer early to set pad/eos-dependent fields
    tokenizer_cfg = {"name": model_name}
    tokenizer = get_tokenizer(tokenizer_cfg)

    policy_precision = "bfloat16"
    max_total_seq_len = 64

    policy_config = {
        "model_name": model_name,
        "tokenizer": tokenizer_cfg,
        "train_global_batch_size": config.num_nodes * config.num_gpus_per_node,
        "train_micro_batch_size": 1,
        "generation_batch_size": 1,
        "logprob_batch_size": 1,
        "max_total_sequence_length": max_total_seq_len,
        "make_sequence_length_divisible_by": 1,
        "precision": policy_precision,

        "megatron_cfg": {
            "enabled": True,
            "empty_unused_memory_level": 1,
            "activation_checkpointing": False,
            "tensor_model_parallel_size": 2,
            "expert_tensor_parallel_size": 1,
            "expert_model_parallel_size": 1,
            "pipeline_model_parallel_size": 2,
            "context_parallel_size": 1,
            "pipeline_dtype": policy_precision,
            "num_layers_in_first_pipeline_stage": None,
            "num_layers_in_last_pipeline_stage": None,
            "sequence_parallel": False,
            "freeze_moe_router": False,
            "moe_router_dtype": None,
            "moe_router_load_balancing_type": "aux_loss",
            "moe_router_bias_update_rate": 1e-3,
            "apply_rope_fusion": True,

            
            "optimizer": {
                "optimizer": "adam",
                "lr": 2.0e-5,
                "min_lr": 1.99999e-5,
                "weight_decay": 0.01,
                "bf16": True,
                "fp16": False,
                "params_dtype": "float32",
                "adam_beta1": 0.9,
                "adam_beta2": 0.98,
                "adam_eps": 1e-5,

                "sgd_momentum": 0.9,

                "use_distributed_optimizer": True,
                "use_precision_aware_optimizer": True,

                "clip_grad": 1.0,
            },



            "scheduler": {
                "start_weight_decay": 0.01,
                "end_weight_decay": 0.01,
                "weight_decay_incr_style": "constant",
                "lr_decay_style": "constant",
                "lr_decay_iters": None,
                "lr_warmup_iters": 50,
                "lr_warmup_init": 1.9999e-65,
            },

            "distributed_data_parallel_config": {
                "grad_reduce_in_fp32": False,
                "overlap_grad_reduce": True,
                "overlap_param_gather": True,
                "average_in_collective": True,
                "data_parallel_sharding_strategy": "optim_grads_params",
            },
        },
        # Keep dynamic batching and sequence packing disabled for clarity
        "dynamic_batching": {
            "enabled": False,
        },
        "sequence_packing": {
            "enabled": False,
        },
        # Dtensor stuff
        # Optimizer and scheduler kept minimal
        # # Keep DTensor enabled but trivial (tp=cp=1) for simplicity
        # "dtensor_cfg": {
        #     "enabled": True,
        #     "cpu_offload": False,
        #     "sequence_parallel": False,
        #     "activation_checkpointing": False,
        #     "tensor_parallel_size": 1,
        #     "context_parallel_size": 1,
        #     "custom_parallel_plan": None,
        # },
        # "optimizer": {
        #     "name": "torch.optim.AdamW",
        #     "kwargs": {
        #         "lr": 8e-6,  # 5.0e-6,
        #         "weight_decay": 0.01,
        #         "betas": [0.9, 0.999],
        #         "eps": 1.0e-8,
        #         "foreach": False,
        #         "fused": False,
        #     },
        # },
        # "scheduler": {
        #     "name": "torch.optim.lr_scheduler.ConstantLR",
        #     "kwargs": {"factor": 1.0, "total_iters": 1},
        # },
        # "max_grad_norm": 1.0,
    }
    # Define generation config like grpo_math_1B.yaml (let helper fill extras)
    generation_cfg = {
        "backend": "vllm",
        "max_new_tokens": 64,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": None,
        "stop_token_ids": None,
        "stop_strings": None,
        "model_name": model_name,
        "colocated": {"enabled": True},
        "vllm_cfg": {
            "async_engine": False,
            "precision": policy_precision,
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "gpu_memory_utilization": 0.6,
            "max_model_len": max_total_seq_len,
            "enforce_eager": True,
        },
    }
    policy_config["generation"] = configure_generation_config(generation_cfg, tokenizer)
    return policy_config, tokenizer


def create_batch_from(tokenizer, sentences: list[str]) -> BatchedDataDict:
    """Create a tiny batch from raw sentences (no chat templates)."""
    enc = tokenizer(
        sentences,
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"].to(torch.float32)
    input_lengths = attention_mask.sum(dim=1).to(torch.int32)
    sample_mask = torch.ones(input_ids.size(0), dtype=torch.float32)

    # For simple NLL training, use the attention mask as token_mask
    # (loss will be applied to positions 1..len-1 via NLLLoss)
    token_mask = torch.ones_like(input_ids)

    return BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "token_mask": token_mask,
            "sample_mask": sample_mask,
        }
    )


def main(config: Config) -> None:
    init_ray()

    # 0) Config
    policy_config, tokenizer = build_policy_config(config)

    # 1) Set up compute cluster (single GPU for demo)
    print("\n▶ Setting up compute cluster...")
    cluster = RayVirtualCluster(
        name="single_update_cluster",
        bundle_ct_per_node_list=[config.num_gpus_per_node] * config.num_nodes,
        use_gpus=True,
        num_gpus_per_node=config.num_gpus_per_node,
        max_colocated_worker_groups=2,
    )

    # 2) Initialize vLLM generation first for a clean GPU environment
    print("\n▶ Initializing vLLM generation...")
    # Initialize vLLM directly from config
    policy_generation = VllmGeneration(
        cluster=cluster, config=policy_config["generation"]
    )
    # Pre-initialize workers to avoid contention later
    policy_generation.finish_generation()
    print("  ✓ vLLM generation ready")

    # 3) Initialize policy (LM)
    print("\n▶ Initializing LM Policy...")
    policy = Policy(
        cluster=cluster,
        config=policy_config,
        tokenizer=tokenizer,
        init_reference_model=False,
    )
    print("  ✓ Policy created")

    # Prepare refit info once before first refit
    state_dict_info = policy.prepare_refit_info()
    policy_generation.prepare_refit_info(state_dict_info or {})

    # 4) Create tiny numeric batch and train with NLLLoss
    print("\n▶ Creating tiny numeric batch and training with NLLLoss...")
    train_sentences = ["a b c d e hello", "a d f world"] * config.num_nodes * config.num_gpus_per_node
    generation_prompts = [
        "Have you heard of NVIDIA?",
        "What is calligraphy?",
        "What is the capital of France?",
        "What is the capital of the United States?",
        "What is the capital of the United Kingdom?",
        "What is the capital of the Philippines?",
        "What is the capital of the China?",
        "What is the capital of the Japan?",
        "What is the capital of the Korea?",
        "What is the capital of the India?",
        "What is the capital of the Pakistan?",
        "What is the capital of the Bangladesh?",
        "What is the capital of the Nepal?",
    ]
    data = create_batch_from(tokenizer, sentences=train_sentences)
    loss_fn = NLLLoss()

    # Optionally repeat the train→refit cycle
    num_iters = int(os.environ.get("SINGLE_UPDATE_ITERS", "10"))

    for step in range(num_iters):
        print(f"\n===== Iteration {step + 1}/{num_iters} =====")
        # Generate before training using predefined prompts
        gen_inputs = create_batch_from(tokenizer, sentences=generation_prompts)
        gen_data = BatchedDataDict(
            {
                "input_ids": gen_inputs["input_ids"],
                "input_lengths": gen_inputs["input_lengths"],
            }
        )
        policy_generation.prepare_for_generation()
        gen_outputs = policy_generation.generate(
            gen_data, greedy=True
        )  # greedy for demonstration
        policy_generation.finish_generation()
        decoded = tokenizer.batch_decode(
            gen_outputs["output_ids"].tolist(), skip_special_tokens=True
        )
        print(
            "  • Pre-train generations (first turn would be gibberish b/c vllm dummy weights; at around loss <0.3 you should see memorization):"
        )
        for i, out_text in enumerate(decoded):
            print(f"    - prompt: '{generation_prompts[i]}' -> '{out_text}'")
        policy.prepare_for_training()
        results = policy.train(data, loss_fn)
        loss_tensor = results["loss"]
        print(f"  • Training loss: {loss_tensor}")

        # 5) Refit policy_generation with latest policy weights
        print("  • Refit generation with latest policy weights...")
        refit_policy_generation(
            policy=policy,
            policy_generation=policy_generation,
            colocated_inference=policy_config["generation"]["colocated"]["enabled"],
        )
        print("  ✓ Refit complete")

    print("\nAll done.")

    policy.shutdown()
    policy_generation.shutdown()
    cluster.shutdown()


if __name__ == "__main__":
    config = Config.parse_args()
    main(config)
