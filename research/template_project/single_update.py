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

import argparse
import os

from omegaconf import OmegaConf
from template_project.data_utils import create_batch_from

from nemo_rl.algorithms.grpo import MasterConfig, refit_policy_generation
from nemo_rl.algorithms.loss_functions import NLLLoss
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.config import load_config, parse_hydra_overrides

OmegaConf.register_new_resolver("mul", lambda a, b: a * b)


def main(config: MasterConfig) -> None:
    # 0) Config
    policy_config = config["policy"]
    tokenizer = get_tokenizer(policy_config["tokenizer"])
    policy_config["generation"] = configure_generation_config(
        policy_config["generation"], tokenizer
    )

    # 1) Set up compute cluster (single GPU for demo)
    print("\n▶ Setting up compute cluster...")
    init_ray()
    cluster = RayVirtualCluster(
        name="single_update_cluster",
        bundle_ct_per_node_list=[config["cluster"]["gpus_per_node"]]
        * config["cluster"]["num_nodes"],
        use_gpus=True,
        num_gpus_per_node=config["cluster"]["gpus_per_node"],
        max_colocated_worker_groups=1
        if policy_config["generation"]["backend"] == "megatron"
        else 2,
    )

    # 2) Initialize vLLM generation first for a clean GPU environment
    print("\n▶ Initializing vLLM generation...")
    # Initialize vLLM directly from config
    policy_config["generation"]["model_name"] = policy_config["model_name"]
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
    train_sentences = ["a b c d e hello", "a d f world"] * config["policy"][
        "train_global_batch_size"
    ]
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

        print("  • Refit generation with latest policy weights...")
        refit_policy_generation(
            policy=policy,
            policy_generation=policy_generation,
            colocated_inference=policy_config["generation"]["colocated"]["enabled"],
        )
        print("  ✓ Refit complete")

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

    print("\nAll done.")

    policy.shutdown()
    policy_generation.shutdown()
    cluster.shutdown()


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GRPO training with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Parse known args for the script
    args, overrides = parser.parse_known_args()

    return args, overrides


if __name__ == "__main__":
    # Parse arguments
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "grpo_math_1B.yaml"
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")
    from rich.pretty import pprint

    pprint(config)

    main(config)
