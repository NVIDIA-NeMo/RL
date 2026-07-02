# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""End-to-end Kimi K2.6 vLLM generation smoke.

This diagnostic exercises the path that the Kimi GRPO recipe depends on without
running rewards, logprobs, or policy training:

1. Load the Kimi recipe and Hydra overrides.
2. Start a colocated 16x4 Ray virtual cluster.
3. Initialize vLLM with dummy weights.
4. Initialize the Megatron policy.
5. Refit live Megatron weights into vLLM.
6. Generate text for a small prompt batch and validate that outputs are present.
"""

import argparse
import json
import string
import time
from pathlib import Path
from typing import Any, Sequence


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "examples/configs/recipes/llm/grpo-kimi-k2.6-16n4g-tp8pp8ep8-megatron.yaml"
)
DEFAULT_PROMPTS = [
    "Write one clear sentence about why the moon appears bright at night.",
    "Solve 2 + 3 and answer with a short sentence.",
]
KIMI_FLASHINFER_MLA_MAX_MODEL_LEN_MULTIPLE = 128


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run a Kimi K2.6 Megatron-to-vLLM live-refit generation smoke."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Kimi recipe YAML to load before applying Hydra overrides.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        default=None,
        help="Prompt to generate from. May be passed more than once.",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy generation instead of the sampling settings in the recipe.",
    )
    parser.add_argument(
        "--min-output-chars",
        type=int,
        default=4,
        help="Minimum stripped output length required for each prompt.",
    )
    parser.add_argument(
        "--ray-log-dir",
        type=str,
        default=None,
        help="Optional Ray log/temp directory.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Render the key smoke configuration fields and exit before Ray init.",
    )
    return parser.parse_known_args()


def _load_master_config(config_path: str, overrides: list[str]) -> dict[str, Any]:
    from omegaconf import OmegaConf

    from nemo_rl.utils.config import (
        load_config,
        parse_hydra_overrides,
        register_omegaconf_resolvers,
    )

    register_omegaconf_resolvers()
    config = load_config(config_path)
    if overrides:
        config = parse_hydra_overrides(config, overrides)
    return OmegaConf.to_container(config, resolve=True)


def _preflight_summary(master_config: dict[str, Any]) -> dict[str, Any]:
    policy = master_config["policy"]
    generation = policy["generation"]
    megatron = policy.get("megatron_cfg", {})
    vllm = generation.get("vllm_cfg", {})
    return {
        "model_name": policy["model_name"],
        "backend": generation["backend"],
        "cluster": master_config["cluster"],
        "policy_parallelism": {
            "tp": megatron.get("tensor_model_parallel_size"),
            "pp": megatron.get("pipeline_model_parallel_size"),
            "ep": megatron.get("expert_model_parallel_size"),
            "etp": megatron.get("expert_tensor_parallel_size"),
        },
        "vllm_parallelism": {
            "tp": vllm.get("tensor_parallel_size"),
            "pp": vllm.get("pipeline_parallel_size"),
            "ep": vllm.get("expert_parallel_size"),
        },
        "vllm_load_format": vllm.get("load_format"),
        "ipc_refit_metadata_in_payload": vllm.get("ipc_refit_metadata_in_payload"),
        "refit_buffer_size_gb": policy.get("refit_buffer_size_gb"),
        "refit_buffer_memory_ratio": policy.get("refit_buffer_memory_ratio"),
        "refit_offload_max_workers_per_node": policy.get(
            "refit_offload_max_workers_per_node"
        ),
        "max_new_tokens": generation.get("max_new_tokens"),
        "max_model_len": vllm.get("max_model_len"),
        "max_num_batched_tokens": generation.get("vllm_kwargs", {}).get(
            "max_num_batched_tokens"
        ),
    }


def _validate_preflight(summary: dict[str, Any]) -> None:
    expected = {
        "backend": "vllm",
        "vllm_load_format": "dummy",
        "ipc_refit_metadata_in_payload": True,
    }
    for key, expected_value in expected.items():
        if summary.get(key) != expected_value:
            raise ValueError(
                f"Expected {key}={expected_value!r}, got {summary.get(key)!r}"
            )

    model_name = str(summary.get("model_name", ""))
    if model_name != "moonshotai/Kimi-K2.6" and "Kimi-K2.6" not in model_name:
        raise ValueError(
            "Expected model_name to be moonshotai/Kimi-K2.6 or a local "
            f"Kimi-K2.6 snapshot path, got {model_name!r}"
        )

    if summary["cluster"].get("num_nodes") != 16:
        raise ValueError(f"Expected 16 nodes, got {summary['cluster'].get('num_nodes')}")
    if summary["cluster"].get("gpus_per_node") != 4:
        raise ValueError(
            f"Expected 4 GPUs per node, got {summary['cluster'].get('gpus_per_node')}"
        )
    if summary["policy_parallelism"] != {"tp": 8, "pp": 8, "ep": 8, "etp": 1}:
        raise ValueError(
            f"Unexpected policy parallelism: {summary['policy_parallelism']}"
        )
    if summary["vllm_parallelism"] != {"tp": 8, "pp": 1, "ep": 64}:
        raise ValueError(f"Unexpected vLLM parallelism: {summary['vllm_parallelism']}")

    max_model_len = int(summary["max_model_len"])
    if max_model_len % KIMI_FLASHINFER_MLA_MAX_MODEL_LEN_MULTIPLE != 0:
        raise ValueError(
            "Kimi K2.6 vLLM FlashInfer MLA requires max_model_len to be a "
            f"multiple of {KIMI_FLASHINFER_MLA_MAX_MODEL_LEN_MULTIPLE}; "
            f"got {max_model_len}. Round smoke sequence caps up, e.g. "
            "704 input + 512 output -> 1280."
        )


def validate_generated_texts(
    prompts: Sequence[str],
    texts: Sequence[str],
    *,
    min_output_chars: int,
) -> None:
    if len(texts) != len(prompts):
        raise ValueError(f"Expected {len(prompts)} outputs, got {len(texts)}")
    if min_output_chars <= 0:
        raise ValueError("min_output_chars must be > 0")

    printable = set(string.printable)
    placeholder_values = {"dummy", "none", "null", "nan", "undefined", "<dummy>"}
    failures: list[str] = []
    for idx, text in enumerate(texts):
        normalized = " ".join(str(text).split())
        if len(normalized) < min_output_chars:
            failures.append(
                f"output {idx} is too short: {len(normalized)} chars < {min_output_chars}"
            )
            continue
        if normalized.lower() in placeholder_values:
            failures.append(f"output {idx} looks like a placeholder: {normalized!r}")
            continue
        printable_ratio = sum(ch in printable for ch in normalized) / len(normalized)
        if printable_ratio < 0.85:
            failures.append(
                f"output {idx} has low printable ratio: {printable_ratio:.2f}"
            )
            continue
        if len(set(normalized)) < 4 and len(normalized) >= min_output_chars:
            failures.append(f"output {idx} has too little character diversity")

    if failures:
        raise ValueError("Generated text validation failed: " + "; ".join(failures))


def main() -> None:
    args, overrides = parse_args()
    prompts = args.prompts if args.prompts else DEFAULT_PROMPTS
    master_config = _load_master_config(args.config, overrides)
    summary = _preflight_summary(master_config)
    print("KIMI_GENERATION_SMOKE_PREFLIGHT")
    print(json.dumps(summary, indent=2, sort_keys=True))
    _validate_preflight(summary)
    if args.preflight_only:
        print("KIMI_GENERATION_SMOKE_PREFLIGHT_OK")
        return

    from nemo_rl.algorithms.grpo import refit_policy_generation
    from nemo_rl.algorithms.utils import get_tokenizer
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.distributed.virtual_cluster import (
        RayVirtualCluster,
        init_ray,
        prepare_segment_topology,
    )
    from nemo_rl.models.generation import configure_generation_config
    from nemo_rl.models.generation.vllm import VllmGeneration
    from nemo_rl.models.policy.lm_policy import Policy

    init_ray(log_dir=args.ray_log_dir)

    policy_config = master_config["policy"]
    generation_config = policy_config["generation"]
    cluster_config = master_config["cluster"]
    if not generation_config["colocated"]["enabled"]:
        raise ValueError("This smoke expects colocated generation.")

    tokenizer = get_tokenizer(policy_config["tokenizer"])
    generation_config = configure_generation_config(generation_config, tokenizer)
    generation_config["model_name"] = policy_config["model_name"]
    generation_config.setdefault("vllm_kwargs", {})["hf_overrides"] = policy_config.get(
        "hf_config_overrides", {}
    )
    policy_config["generation"] = generation_config
    if policy_config.get("megatron_cfg", {}).get("enabled", False):
        policy_config["megatron_cfg"]["train_iters"] = max(
            1, int(master_config.get("grpo", {}).get("max_num_steps", 1))
        )

    policy_nodes = int(cluster_config["num_nodes"])
    gpus_per_node = int(cluster_config["gpus_per_node"])
    segment_size = cluster_config.get("segment_size")
    node_resource_constraints, _, _ = prepare_segment_topology(
        segment_size,
        policy_nodes,
    )
    cluster = RayVirtualCluster(
        name="kimi_generation_smoke_cluster",
        bundle_ct_per_node_list=[gpus_per_node] * policy_nodes,
        use_gpus=True,
        num_gpus_per_node=gpus_per_node,
        max_colocated_worker_groups=2,
        port_range_low=cluster_config.get("master_port_range_low"),
        port_range_high=cluster_config.get("master_port_range_high"),
        segment_size=segment_size,
        node_resource_constraints=node_resource_constraints,
    )
    print(
        f"KIMI_GENERATION_SMOKE_CLUSTER nodes={policy_nodes} gpus_per_node={gpus_per_node}",
        flush=True,
    )

    policy = None
    policy_generation = None
    try:
        start = time.perf_counter()
        policy_generation = VllmGeneration(cluster=cluster, config=generation_config)
        policy_generation.finish_generation()
        print(
            f"KIMI_GENERATION_SMOKE_VLLM_READY seconds={time.perf_counter() - start:.1f}",
            flush=True,
        )

        start = time.perf_counter()
        policy = Policy(
            cluster=cluster,
            config=policy_config,
            tokenizer=tokenizer,
            init_optimizer=False,
            init_reference_model=False,
        )
        print(
            f"KIMI_GENERATION_SMOKE_POLICY_READY seconds={time.perf_counter() - start:.1f}",
            flush=True,
        )

        start = time.perf_counter()
        state_dict_info = policy.prepare_refit_info()
        policy_generation.prepare_refit_info(state_dict_info)
        refit_policy_generation(
            policy,
            policy_generation,
            colocated_inference=True,
            _refit_buffer_size_gb=policy_config.get("refit_buffer_size_gb"),
            _refit_buffer_memory_ratio=policy_config.get("refit_buffer_memory_ratio"),
        )
        print(
            f"KIMI_GENERATION_SMOKE_REFIT_OK seconds={time.perf_counter() - start:.1f}",
            flush=True,
        )

        outputs = policy_generation.generate_text(
            BatchedDataDict({"prompts": list(prompts)}),
            greedy=args.greedy,
        )
        texts = [str(text) for text in outputs["texts"]]
        validate_generated_texts(
            prompts,
            texts,
            min_output_chars=args.min_output_chars,
        )
        print("KIMI_GENERATION_SMOKE_OUTPUTS")
        print(json.dumps({"prompts": list(prompts), "texts": texts}, indent=2))
        print("KIMI_GENERATION_SMOKE_PASS", flush=True)
    finally:
        if policy_generation is not None:
            policy_generation.shutdown()
        if policy is not None:
            policy.shutdown()
        cluster.shutdown()


if __name__ == "__main__":
    main()
