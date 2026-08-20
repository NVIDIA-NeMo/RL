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

"""Validate Nano-v3 generation through NeMo-RL Megatron Inference."""

import argparse
import gc
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import ray
import torch
from omegaconf import OmegaConf

from nemo_rl.algorithms.grpo import refit_policy_generation
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.megatron import MegatronGeneration
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
from nemo_rl.weight_sync.factory import create_weight_synchronizer
from nemo_rl.weight_sync.nccl_reshard_utils import (
    check_nccl_reshard_refit_support,
)


def parse_args() -> argparse.Namespace:
    """Parse smoke-test arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--precision", choices=("bf16", "mxfp8"), required=True)
    parser.add_argument("--source-precision", choices=("bf16", "mxfp8"), required=True)
    parser.add_argument(
        "--refit-impl",
        choices=("bridge", "mcore"),
        help="Override the recipe's Megatron refit implementation.",
    )
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        help=(
            "Use the prompt tokens from a saved train_data_step JSONL row "
            "instead of tokenizing --prompt."
        ),
    )
    parser.add_argument("--input-row", type=int, default=0)
    parser.add_argument("--input-repeats", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument(
        "--logprob-sample-mode",
        choices=("greedy", "stochastic"),
        default="greedy",
    )
    parser.add_argument(
        "--execution-mode",
        choices=("cuda-graphs", "eager"),
        default="cuda-graphs",
    )
    parser.add_argument("--ray-log-dir", type=Path, required=True)
    parser.add_argument("--perf-batch-size", type=int, default=4)
    parser.add_argument("--perf-warmup-iterations", type=int, default=3)
    parser.add_argument("--perf-iterations", type=int, default=10)
    parser.add_argument("--skip-perf", action="store_true")
    parser.add_argument("--reuse-smoke-for-logprob", action="store_true")
    parser.add_argument("--check-inference-batch-variance", action="store_true")
    parser.add_argument("--refit-iterations", type=int, default=1)
    parser.add_argument("--max-prob-mult-error", type=float, default=1.05)
    parser.add_argument("--logprob-window-size", type=int, default=32)
    args = parser.parse_args()
    if args.refit_iterations < 1:
        parser.error("--refit-iterations must be at least 1")
    if not args.skip_perf and args.perf_warmup_iterations < 1:
        parser.error("--perf-warmup-iterations must be at least 1")
    if not args.skip_perf and args.perf_iterations < 1:
        parser.error("--perf-iterations must be at least 1")
    if args.reuse_smoke_for_logprob and args.logprob_sample_mode != "stochastic":
        parser.error(
            "--reuse-smoke-for-logprob requires --logprob-sample-mode=stochastic"
        )
    if args.input_row < 0:
        parser.error("--input-row must be nonnegative")
    if args.input_repeats < 1:
        parser.error("--input-repeats must be at least 1")
    if args.check_inference_batch_variance and args.input_repeats < 2:
        parser.error("--check-inference-batch-variance requires --input-repeats >= 2")
    if args.max_new_tokens is not None and args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    if args.logprob_window_size < 1:
        parser.error("--logprob-window-size must be at least 1")
    return args


def load_policy_config(config_path: Path) -> dict[str, Any]:
    """Load and resolve the recipe's policy configuration."""
    register_omegaconf_resolvers()
    config = OmegaConf.to_container(load_config(config_path), resolve=True)
    assert isinstance(config, dict)
    policy_config = config["policy"]
    assert isinstance(policy_config, dict)
    return policy_config


def make_input(tokenizer: Any, prompt: str, batch_size: int = 1) -> BatchedDataDict:
    """Tokenize a batch of identical right-padded prompts for generation."""
    encodings = tokenizer(
        [prompt] * batch_size,
        padding="max_length",
        max_length=64,
        truncation=True,
        return_tensors="pt",
        padding_side="right",
    )
    return BatchedDataDict(
        {
            "input_ids": encodings["input_ids"],
            "input_lengths": encodings["attention_mask"].sum(dim=1).to(torch.int32),
        }
    )


def load_saved_input(path: Path, row_index: int, repeats: int = 1) -> BatchedDataDict:
    """Load the unpadded prompt prefix from a saved training-data row."""
    rows = path.read_text().splitlines()
    if row_index >= len(rows):
        raise RuntimeError(
            f"Requested row {row_index} from {path}, which has {len(rows)} rows."
        )
    row = json.loads(rows[row_index])
    token_ids = row["token_ids"][0]
    token_loss_mask = row["token_loss_mask"][0]
    try:
        prompt_length = next(
            index for index, is_generated in enumerate(token_loss_mask) if is_generated
        )
    except StopIteration as error:
        raise RuntimeError(
            f"No generated tokens found in row {row_index} of {path}."
        ) from error
    if prompt_length == 0:
        raise RuntimeError(f"Row {row_index} of {path} has an empty prompt.")
    return BatchedDataDict(
        {
            "input_ids": torch.tensor(
                [token_ids[:prompt_length]] * repeats, dtype=torch.long
            ),
            "input_lengths": torch.full((repeats,), prompt_length, dtype=torch.int32),
        }
    )


def main() -> None:
    """Refit Nano weights into Megatron Inference and validate generation."""
    args = parse_args()
    policy_config = load_policy_config(args.config)
    generation_config = policy_config["generation"]
    if args.max_new_tokens is not None:
        generation_config["max_new_tokens"] = args.max_new_tokens
    mcore_generation_config = generation_config["mcore_generation_config"]
    if args.refit_impl is not None:
        mcore_generation_config["refit_impl"] = args.refit_impl
        if args.refit_impl == "mcore":
            generation_config["refit_transport"] = None
    if args.execution_mode == "eager":
        mcore_generation_config["cuda_graph_impl"] = "none"
        mcore_generation_config["inference_cuda_graph_scope"] = "none"
        mcore_generation_config["num_cuda_graphs"] = None
        mcore_generation_config["use_cuda_graphs_for_non_decode_steps"] = False
    assert generation_config["backend"] == "megatron"
    assert not generation_config["colocated"]["enabled"]
    assert mcore_generation_config["transformer_impl"] == "inference_optimized"
    if args.execution_mode == "cuda-graphs":
        assert mcore_generation_config["cuda_graph_impl"] == "local"
        assert mcore_generation_config["inference_cuda_graph_scope"] in (
            "layer",
            "block",
        )
        assert mcore_generation_config["num_cuda_graphs"] is not None
    else:
        assert mcore_generation_config["cuda_graph_impl"] == "none"
        assert mcore_generation_config["inference_cuda_graph_scope"] == "none"
        assert mcore_generation_config["num_cuda_graphs"] is None
    assert mcore_generation_config["logprobs_mode"] == "raw_logprobs"
    fp8_config = mcore_generation_config["fp8_cfg"]
    if args.precision == "mxfp8":
        assert fp8_config == {
            "enabled": True,
            "fp8": "e4m3",
            "fp8_recipe": "mxfp8",
            "fp8_param": True,
        }
    else:
        fp8_config["enabled"] = False
        fp8_config["fp8_param"] = False
        assert not fp8_config["enabled"]
        assert not fp8_config["fp8_param"]

    use_nccl_reshard = generation_config.get("refit_transport") == "nccl_reshard"
    if use_nccl_reshard:
        check_nccl_reshard_refit_support(SimpleNamespace(policy=policy_config))

    marker_prefix = f"NRL_NANOV3_MEGATRON_{args.precision.upper()}"

    init_ray(log_dir=str(args.ray_log_dir), num_cpus=8)
    tokenizer = get_tokenizer(policy_config["tokenizer"])
    policy_config["generation"] = configure_generation_config(
        generation_config,
        tokenizer,
        has_refit_draft_weights=False,
        trains_mtp=False,
    )

    source_cluster = RayVirtualCluster(
        bundle_ct_per_node_list=[2],
        use_gpus=True,
        max_colocated_worker_groups=1,
        num_gpus_per_node=2,
        name="nanov3-mxfp8-source-cluster",
    )
    generation_cluster = RayVirtualCluster(
        bundle_ct_per_node_list=[2],
        use_gpus=True,
        max_colocated_worker_groups=1,
        num_gpus_per_node=2,
        name="nanov3-mxfp8-generation-cluster",
    )
    policy = None
    generation = None
    try:
        policy = Policy(
            cluster=source_cluster,
            config=policy_config,
            tokenizer=tokenizer,
            init_optimizer=False,
            init_reference_model=False,
            name_prefix="nanov3_mxfp8_source",
        )
        # Actor handles are returned before Ray finishes their constructors.
        # Force the one-time HF-to-Megatron import to complete before creating
        # another policy group that targets the same converted checkpoint.
        if not all(policy.run_all_workers_single_data("is_alive")):
            raise RuntimeError("A source Megatron policy worker failed readiness.")
        source_precision_info = policy.run_all_workers_single_data(
            "get_runtime_precision_info"
        )
        for rank, info in enumerate(source_precision_info):
            source_is_mxfp8 = info["fp8_enabled"] and info["fp8_recipe"] == "mxfp8"
            if source_is_mxfp8 != (args.source_precision == "mxfp8"):
                raise RuntimeError(
                    f"Source rank {rank} precision mismatch: expected "
                    f"{args.source_precision}, got {info}."
                )
            print(
                f"NRL_NANOV3_SOURCE_{args.source_precision.upper()}_VERIFY: PASS "
                f"rank={rank} runtime={info}",
                flush=True,
            )
        generation = MegatronGeneration(
            cluster=generation_cluster,
            config=policy_config,
            tokenizer=tokenizer,
            skip_weight_load=True,
            name_prefix="nanov3_mxfp8_generation",
        )

        if use_nccl_reshard:
            generation.weight_synchronizer = create_weight_synchronizer(
                policy=policy,
                generation=generation,
                generation_backend="megatron",
                colocated=False,
                train_cluster=source_cluster,
                inference_cluster=generation_cluster,
            )
            generation.weight_synchronizer.init_communicator()
        else:
            ip, port = source_cluster.get_master_address_and_port()
            source_world_size = source_cluster.world_size()
            world_size = source_world_size + generation_cluster.world_size()
            if generation.uses_native_refit:
                source_futures = policy.init_collective_mcore_generation(
                    ip,
                    port,
                    world_size,
                    rank_offset=0,
                    refit_backend=policy_config["generation"][
                        "mcore_generation_config"
                    ]["refit_backend"],
                )
            else:
                source_futures = policy.init_collective(
                    ip,
                    port,
                    world_size,
                    train_world_size=source_world_size,
                )
            generation_futures = generation.init_collective(
                ip,
                port,
                world_size,
                train_world_size=source_world_size,
            )
            ray.get(source_futures + generation_futures)
            if not generation.uses_native_refit:
                state_dict_info = policy.prepare_refit_info()
                generation.prepare_refit_info(state_dict_info)
        for refit_iteration in range(1, args.refit_iterations + 1):
            refit_policy_generation(policy, generation, colocated_inference=False)
            print(
                f"{marker_prefix}_REFIT: PASS iteration={refit_iteration}",
                flush=True,
            )

        inputs = (
            load_saved_input(args.input_jsonl, args.input_row, args.input_repeats)
            if args.input_jsonl is not None
            else make_input(tokenizer, args.prompt)
        )
        outputs = generation.generate(inputs, greedy=False)
        if not bool((outputs["generation_lengths"] > 0).all()):
            raise RuntimeError("Megatron Inference generated zero tokens.")
        decoded = tokenizer.batch_decode(
            outputs["output_ids"], skip_special_tokens=True
        )
        print(
            f"{marker_prefix}_SMOKE: PASS "
            f"generated_tokens={outputs['generation_lengths'].tolist()} "
            f"text={decoded!r}",
            flush=True,
        )

        batch_variance_outputs = None
        if args.check_inference_batch_variance:
            singleton_inputs = BatchedDataDict(
                {
                    "input_ids": inputs["input_ids"][:1],
                    "input_lengths": inputs["input_lengths"][:1],
                }
            )
            singleton_outputs = generation.generate(singleton_inputs, greedy=True)
            batched_outputs = generation.generate(inputs, greedy=True)
            batch_variance_outputs = batched_outputs
            singleton_length = int(singleton_outputs["unpadded_sequence_lengths"][0])
            batched_length = int(batched_outputs["unpadded_sequence_lengths"][0])
            common_length = min(singleton_length, batched_length)
            singleton_ids = singleton_outputs["output_ids"][0, :common_length]
            batched_ids = batched_outputs["output_ids"][0, :common_length]
            matching_ids = singleton_ids == batched_ids
            first_token_mismatch = (
                int((~matching_ids).nonzero()[0]) if not bool(matching_ids.all()) else None
            )
            prompt_length = int(singleton_inputs["input_lengths"][0])
            logprob_end = min(singleton_length, batched_length)
            batch_logprob_diff = (
                singleton_outputs["logprobs"][0, prompt_length:logprob_end]
                - batched_outputs["logprobs"][0, prompt_length:logprob_end]
            ).abs()
            print(
                f"{marker_prefix}_BATCH_VARIANCE: PASS "
                f"singleton_length={singleton_length} "
                f"batched_length={batched_length} "
                f"first_token_mismatch={first_token_mismatch} "
                f"compared_logprobs={batch_logprob_diff.numel()} "
                f"mean_abs_logprob_diff={batch_logprob_diff.mean().item():.8f} "
                f"max_abs_logprob_diff={batch_logprob_diff.max().item():.8f}",
                flush=True,
            )

        perf_inputs = (
            inputs
            if args.input_jsonl is not None
            else make_input(tokenizer, args.prompt, args.perf_batch_size)
        )
        if not args.skip_perf:
            for _ in range(args.perf_warmup_iterations):
                generation.generate(perf_inputs, greedy=False)
            latencies = []
            generated_token_count = 0
            for _ in range(args.perf_iterations):
                start_time = time.perf_counter()
                perf_outputs = generation.generate(perf_inputs, greedy=False)
                latencies.append(time.perf_counter() - start_time)
                generated_token_count += int(perf_outputs["generation_lengths"].sum())
            total_perf_time = sum(latencies)
            print(
                f"{marker_prefix}_PERF: PASS "
                f"batch_size={len(perf_inputs['input_ids'])} "
                f"warmup_iterations={args.perf_warmup_iterations} "
                f"iterations={args.perf_iterations} "
                f"generated_tokens={generated_token_count} "
                f"tokens_per_second={generated_token_count / total_perf_time:.4f} "
                f"mean_latency_seconds={statistics.mean(latencies):.4f} "
                f"p50_latency_seconds={statistics.median(latencies):.4f} "
                f"max_latency_seconds={max(latencies):.4f}",
                flush=True,
            )

        # Keep the numerical gate deterministic.  Reusing the final stochastic
        # performance sample made this check depend on which four continuations
        # happened to be drawn, and only compared 32 tokens.  Greedy decoding
        # gives us a stable, repeatable cross-backend comparison while still
        # exercising CUDA-graph prefill and decode replay.
        if args.reuse_smoke_for_logprob:
            logprob_outputs = outputs
        elif (
            batch_variance_outputs is not None
            and args.logprob_sample_mode == "greedy"
        ):
            logprob_outputs = batch_variance_outputs
        else:
            logprob_outputs = generation.generate(
                perf_inputs, greedy=args.logprob_sample_mode == "greedy"
            )
        fprop_data = BatchedDataDict(
            {
                "input_ids": logprob_outputs["output_ids"],
                "input_lengths": logprob_outputs["unpadded_sequence_lengths"],
            }
        )
        policy.prepare_for_lp_inference()
        train_logprobs = policy.get_logprobs(fprop_data)["logprobs"]
        generated_token_mask = torch.zeros_like(
            logprob_outputs["logprobs"], dtype=torch.bool
        )
        for row, (start, end) in enumerate(
            zip(
                perf_inputs["input_lengths"],
                logprob_outputs["unpadded_sequence_lengths"],
            )
        ):
            generated_token_mask[row, start:end] = True
        all_abs_diff = (logprob_outputs["logprobs"] - train_logprobs).abs()
        abs_diff = all_abs_diff.masked_select(generated_token_mask)
        signed_logprob_diff = (
            train_logprobs - logprob_outputs["logprobs"]
        ).masked_select(generated_token_mask)
        if abs_diff.numel() == 0:
            raise RuntimeError("No generated-token log probabilities were compared.")
        prob_mult_errors = torch.exp(abs_diff)
        importance_ratios = torch.exp(signed_logprob_diff)
        avg_prob_mult_error = prob_mult_errors.mean().item()
        max_prob_mult_error = prob_mult_errors.max().item()
        quantiles = torch.quantile(
            prob_mult_errors.float(),
            torch.tensor([0.5, 0.9, 0.99], device=prob_mult_errors.device),
        ).tolist()
        threshold_fraction = (
            prob_mult_errors > args.max_prob_mult_error
        ).float().mean().item()
        logprob_metrics = (
            f"sample_mode={args.logprob_sample_mode} "
            f"refit_iterations={args.refit_iterations} "
            f"tokens={abs_diff.numel()} mean_abs_diff={abs_diff.mean().item():.6f} "
            f"max_abs_diff={abs_diff.max().item():.6f} "
            f"avg_prob_mult_error={avg_prob_mult_error:.6f} "
            f"max_prob_mult_error={max_prob_mult_error:.6f} "
            f"p50_prob_mult_error={quantiles[0]:.6f} "
            f"p90_prob_mult_error={quantiles[1]:.6f} "
            f"p99_prob_mult_error={quantiles[2]:.6f} "
            f"fraction_above_threshold={threshold_fraction:.6f} "
            f"mean_signed_logprob_diff={signed_logprob_diff.mean().item():.6f} "
            f"mean_importance_ratio={importance_ratios.mean().item():.6f} "
            f"fraction_importance_ratio_outside_0.8_1.2="
            f"{((importance_ratios < 0.8) | (importance_ratios > 1.2)).float().mean().item():.6f}"
        )
        generated_abs_diff_rows = []
        for row, (start, end) in enumerate(
            zip(
                perf_inputs["input_lengths"],
                logprob_outputs["unpadded_sequence_lengths"],
            )
        ):
            generated_abs_diff_rows.append(
                all_abs_diff[row, int(start.item()) : int(end.item())]
            )
        max_generated_length = max(row.numel() for row in generated_abs_diff_rows)
        window_metrics = []
        for window_start in range(0, max_generated_length, args.logprob_window_size):
            window_end = min(
                window_start + args.logprob_window_size, max_generated_length
            )
            window_values = torch.cat(
                [
                    row[window_start:window_end]
                    for row in generated_abs_diff_rows
                    if row.numel() > window_start
                ]
            )
            window_multipliers = torch.exp(window_values)
            window_metrics.append(
                f"{window_start}:{window_end}="
                f"avg{window_multipliers.mean().item():.6f}/"
                f"p90{torch.quantile(window_multipliers.float(), 0.9).item():.6f}/"
                f"max{window_multipliers.max().item():.6f}"
            )
        print(
            f"{marker_prefix}_LOGPROB_WINDOWS: " + " ".join(window_metrics),
            flush=True,
        )
        runtime_info = generation.get_inference_runtime_info()
        for rank, info in enumerate(runtime_info):
            if not info["initialized"]:
                raise RuntimeError(f"Inference engine rank {rank} is not initialized.")
            if args.execution_mode == "cuda-graphs":
                if info["captured_graph_count"] <= 0 or info["capture_stats"] is None:
                    raise RuntimeError(
                        f"Inference engine rank {rank} did not capture CUDA graphs: "
                        f"{info}."
                    )
                execution_marker = "CUDA_GRAPH"
            else:
                if (
                    info["captured_graph_count"] != 0
                    or info["capture_stats"] is not None
                ):
                    raise RuntimeError(
                        f"Eager inference rank {rank} unexpectedly used CUDA graphs: "
                        f"{info}."
                    )
                execution_marker = "EAGER"
            print(
                f"{marker_prefix}_{execution_marker}: PASS rank={rank} runtime={info}",
                flush=True,
            )
        if avg_prob_mult_error > args.max_prob_mult_error:
            print(
                f"{marker_prefix}_LOGPROB: FAIL {logprob_metrics} "
                f"threshold={args.max_prob_mult_error:.6f}",
                flush=True,
            )
            raise RuntimeError(
                f"Megatron {args.precision.upper()} generation log probabilities "
                f"diverged from the {args.source_precision.upper()} training policy: "
                f"{avg_prob_mult_error:.6f} > "
                f"{args.max_prob_mult_error:.6f}."
            )
        print(
            f"{marker_prefix}_LOGPROB: PASS {logprob_metrics}",
            flush=True,
        )
    finally:
        if generation is not None:
            generation.shutdown()
        if policy is not None:
            policy.shutdown()
        generation_cluster.shutdown()
        source_cluster.shutdown()
        if ray.is_initialized():
            ray.shutdown()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
