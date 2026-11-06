# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Real-rollout replay using the existing fixed-input parity harness helpers.

All cases restore the exact initial model, raw/public optimizer, scheduler and
RNG state. Numerical capture is separate from uninstrumented timing repeats.
No result from this diagnostic changes the original numerical acceptance gates.
"""

import argparse
import copy
import hashlib
import json
import os
import statistics
import time
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import ray
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from pydantic import TypeAdapter
from torch.distributed.fsdp import FSDPModule

from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn
from nemo_rl.algorithms.loss.interfaces import LossType, MetricNormalizer
from nemo_rl.algorithms.single_controller_utils.utils import aggregate_step_metrics
from nemo_rl.algorithms.utils import calculate_kl
from nemo_rl.data_plane.adapters.local import local_batch_to_tensordict
from nemo_rl.data_plane.interfaces import KVBatchMeta, LocalDataPlaneConfig
from nemo_rl.data_plane.preshard import shard_meta_for_dp
from nemo_rl.data_plane.schema import GLOBAL_FORWARD_PAD_SEQLEN
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.tq_policy import _aggregate_train_results
from nemo_rl.models.policy.workers import automodel_policy_worker as worker_module
from nemo_rl.models.policy.workers.automodel_policy_worker import (
    AutomodelPolicyWorkerImpl,
)
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
from tests.functional import automodel_sc_full_batch_parity as parity


@dataclass(frozen=True)
class Case:
    name: str
    split: bool = False
    production_sharding: bool = False
    numerator: bool = False
    recompute_prev: bool = False


CASES = (
    Case("full"),
    Case("full_repeat"),
    Case("numerator_full", numerator=True),
    Case("split_fixed_shards", split=True),
    Case("split_production_shards", split=True, production_sharding=True),
    Case("full_recompute", recompute_prev=True),
    Case(
        "split_production_recompute",
        split=True,
        production_sharding=True,
        recompute_prev=True,
    ),
)


def exact_comparison(actual: Any, expected: Any) -> dict[str, Any]:
    try:
        parity.compare_tree(actual, expected, rtol=0, atol=0)
    except AssertionError as error:
        return {"exact": False, "first_difference": str(error)[:600]}
    return {"exact": True}


def gradient_comparison(
    actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]
) -> dict[str, Any]:
    report = parity.bf16_gradient_report(actual, expected)
    sufficient = torch.tensor(
        list(report["sufficient_statistics"].values()),
        dtype=torch.float64,
        device="cuda",
    )
    dist.all_reduce(sufficient)
    difference_sq, actual_sq, expected_sq, dot = sufficient.tolist()
    return {
        "relative_l2": (difference_sq / expected_sq) ** 0.5
        if expected_sq
        else difference_sq**0.5,
        "cosine": dot / (actual_sq * expected_sq) ** 0.5
        if actual_sq and expected_sq
        else float(actual_sq == expected_sq),
        "norm_relative_error": abs(actual_sq**0.5 - expected_sq**0.5) / expected_sq**0.5
        if expected_sq
        else actual_sq**0.5,
        "local_max_abs": max(report["tensor_max_abs"].values(), default=0.0),
        "exact": difference_sq == 0,
    }


def direct_kl(batch: BatchedDataDict[Any]) -> float:
    mask = batch["token_mask"][:, 1:] * batch["sample_mask"][:, None]
    valid = mask.bool()
    gen = torch.where(valid, batch["generation_logprobs"][:, 1:], 0.0)
    prev = torch.where(valid, batch["prev_logprobs"][:, 1:], 0.0)
    values = calculate_kl(
        gen, prev, kl_type="k3", input_clamp_value=None, output_clamp_value=None
    )
    return float((values.double() * mask.double()).sum() / mask.double().sum())


def run(args: argparse.Namespace) -> None:
    if os.environ["USER"] != "root" or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Replay must run inside an allocated container")
    rank, world, local_rank = (
        int(os.environ["RANK"]),
        int(os.environ["WORLD_SIZE"]),
        int(os.environ["LOCAL_RANK"]),
    )
    torch.cuda.set_device(local_rank)
    register_omegaconf_resolvers()
    cfg = load_config(args.recipe)
    packets = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in sorted(Path(args.capture).glob("chunk-*.pt"))
    ]
    if not packets:
        raise ValueError(f"No rollout packets in {args.capture}")
    ids = [sample for packet in packets for sample in packet["ids"]]
    versions = {packet["trainer_version"] for packet in packets}
    if len(ids) != len(set(ids)) or len(versions) != 1:
        raise ValueError("Replay requires unique sample IDs from one weight version")
    batch = BatchedDataDict.from_batches(
        [BatchedDataDict(packet["data"]) for packet in packets]
    )
    if batch.size != 32 or batch.size % (world * 2):
        raise ValueError(
            f"Expected 32 real rollout responses, divisible by two dispatches and DP={world}"
        )
    if args.checkpoint and versions != {210}:
        raise ValueError(f"Checkpoint210 must match rollout versions, got {versions}")
    sample_versions = {
        tag["weight_version"] for packet in packets for tag in packet["tags"]
    }
    if sample_versions != versions:
        raise ValueError(
            f"Rollout weight metadata does not match checkpoint: {sample_versions} vs {versions}"
        )
    if args.checkpoint and torch.count_nonzero(batch["advantages"]).item() == 0:
        raise ValueError(
            "Large-model update replay needs a batch with nonzero advantages"
        )
    cfg.policy.train_global_batch_size = batch.size
    cfg.policy.train_micro_batch_size = 1
    cfg.policy.logprob_batch_size = 1
    cfg.policy.generation.colocated.enabled = False
    cfg.policy.automodel_cfg.deterministic = True
    policy_cfg = TypeAdapter(PolicyConfig).validate_python(
        OmegaConf.to_container(cfg.policy, resolve=True)
    )
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    with patch.object(ray, "get_gpu_ids", return_value=[local_rank]):
        worker = AutomodelPolicyWorkerImpl(
            policy_cfg,
            weights_path=str(checkpoint / "policy/weights") if checkpoint else None,
            optimizer_path=str(checkpoint / "policy/optimizer") if checkpoint else None,
            init_reference_model=False,
        )
    worker.setup_data_plane(LocalDataPlaneConfig(max_partitions=16))
    client = worker._require_dp_client()
    loss = ClippedPGLossFn(
        ClippedPGLossConfig(**OmegaConf.to_container(cfg.loss_fn, resolve=True))
    )
    if (
        loss.force_on_policy_ratio
        or loss.reference_policy_kl_penalty
        or loss.loss_type != LossType.TOKEN_LEVEL
    ):
        raise ValueError(
            "This replay is the actual token-level GRPO/TIS2/zero-reference-KL recipe"
        )
    if not torch.isfinite(batch["prev_logprobs"]).all():
        raise ValueError("Captured policy logprobs must be finite")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for module in worker.model.modules():
        if isinstance(module, FSDPModule):
            module.reshard()
    initial_model = parity.snapshot_model(worker)
    initial_optimizer = parity.snapshot_optimizer(worker)
    initial_raw = parity.snapshot_raw_optimizer(worker)
    initial_scheduler = copy.deepcopy(worker.scheduler.state_dict())
    cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()

    def reset() -> None:
        # Inference may leave the FSDP root unsharded. Restore into the same
        # local shard layout as the initial snapshot, using the public API.
        for module in worker.model.modules():
            if isinstance(module, FSDPModule):
                module.reshard()
        parity.restore_model(worker, initial_model)
        parity.restore_optimizer(worker, initial_optimizer, initial_raw)
        worker.scheduler.load_state_dict(initial_scheduler)
        worker.optimizer.zero_grad()
        worker._train_step_state = None
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(gpu_rng)
        worker.prepare_for_training()
        dist.barrier()

    def metas_for(case: Case) -> list[KVBatchMeta]:
        partition = f"replay-{case.name}"
        client.register_partition(
            partition_id=partition,
            fields=list(batch),
            num_samples=batch.size,
            consumer_tasks=["train"],
        )
        meta = client.put_samples(
            sample_ids=ids,
            partition_id=partition,
            fields=local_batch_to_tensordict(dict(batch), batch_size=batch.size),
        )
        meta = replace(
            meta,
            task_name="train",
            extra_info={
                **meta.extra_info,
                GLOBAL_FORWARD_PAD_SEQLEN: batch["input_ids"].shape[1],
            },
        )
        if case.production_sharding:
            return [
                shard_meta_for_dp(
                    meta.slice(start, start + batch.size // 2),
                    dp_world=world,
                    batch_size=batch.size // 2,
                )[0][rank]
                for start in (0, batch.size // 2)
            ]
        shards, _ = shard_meta_for_dp(meta, dp_world=world, batch_size=batch.size)
        shard = shards[rank]
        if case.split:
            local_half = batch.size // world // 2
            return [shard.slice(0, local_half), shard.slice(local_half, local_half * 2)]
        return [shard]

    case_metas = {case.name: metas_for(case) for case in CASES}
    original_process, original_clip = (
        worker_module.process_global_batch,
        worker_module.scale_grads_and_clip_grad_norm,
    )

    def execute(
        case: Case, *, capture: bool
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor], list[dict[str, Any]]]:
        gradients: dict[str, torch.Tensor] = {}
        lp_rows = []
        denominator = 1.0
        global_seqs = 1.0

        def process(*values: Any, **kwargs: Any) -> dict[str, Any]:
            nonlocal denominator, global_seqs
            result = original_process(*values, **kwargs)
            denominator, global_seqs = (
                result["global_valid_toks"].item(),
                result["global_valid_seqs"].item(),
            )
            if case.numerator:
                result["global_valid_toks"] = torch.ones_like(
                    result["global_valid_toks"]
                )
                result["global_valid_seqs"] = torch.ones_like(
                    result["global_valid_seqs"]
                )
            return result

        def clip(*values: Any, **kwargs: Any) -> Any:
            for name, parameter in worker.model.named_parameters():
                if parameter.grad is not None:
                    if case.numerator:
                        parameter.grad.mul_(1.0 / denominator)
                    if capture:
                        gradients[name] = (
                            parity.local(parameter.grad).detach().cpu().clone()
                        )
            return original_clip(*values, **kwargs)

        opened = False
        with (
            patch.object(worker_module, "process_global_batch", side_effect=process)
            if case.numerator
            else nullcontext(),
            patch.object(
                worker_module, "scale_grads_and_clip_grad_norm", side_effect=clip
            )
            if capture or case.numerator
            else nullcontext(),
        ):
            for meta in case_metas[case.name]:
                worker.prepare_for_lp_inference(keep_train_buffers=opened)
                data = worker._fetch(meta)
                lp = worker.get_logprobs(data)["logprobs"].cpu()
                if capture:
                    lp_rows.append({"ids": meta.sample_ids, "logprobs": lp.clone()})
                if case.recompute_prev:
                    # The process-local adapter intentionally permits one put
                    # per partition. Change only this materialized input; real
                    # TQ delta writes were already exercised during capture.
                    data["prev_logprobs"] = lp.to(data["input_ids"].device)
                worker.prepare_for_training()
                if case.numerator:
                    result = worker.train(data, loss)
                else:
                    if not opened:
                        worker.begin_train_step_presharded(loss)
                        opened = True
                    if case.recompute_prev:
                        worker.train_microbatch(data)
                    else:
                        worker.train_microbatch_presharded(meta)
            if not case.numerator:
                result = worker.finish_train_step_presharded()
        if case.numerator:
            # The independent full-input reference moves only normalization.
            for key, values in result["all_mb_metrics"].items():
                kind = loss.metric_normalizations.get(key)
                if kind == MetricNormalizer.TOKENS:
                    result["all_mb_metrics"][key] = [
                        value / denominator for value in values
                    ]
                elif kind == MetricNormalizer.SEQUENCES:
                    result["all_mb_metrics"][key] = [
                        value / global_seqs for value in values
                    ]
                elif key == "global_valid_toks":
                    result["all_mb_metrics"][key] = [denominator for value in values]
                elif key == "global_valid_seqs":
                    result["all_mb_metrics"][key] = [global_seqs for value in values]
            result["global_loss"] /= denominator
            result["is_replica_leader"] = True
        return result, gradients, lp_rows

    report: dict[str, Any] = {
        "source_sha": args.source_sha,
        "kind": "real_rollout_diagnostic_only",
        "releases_training": False,
        "capture_version": next(iter(versions)),
        "sample_count": batch.size,
        "sample_ids": ids,
        "input_hash": hashlib.sha256(batch["input_ids"].numpy().tobytes()).hexdigest(),
        "world": world,
        "ep": cfg.policy.automodel_cfg.get("expert_parallel_size", 1),
        "direct_captured_gen_kl": direct_kl(batch),
        "cases": {},
        "timing": {},
    }
    reference_gradient = None
    reference_model = reference_optimizer = reference_raw = reference_scheduler = None
    numerator_gradient = numerator_model = numerator_optimizer = numerator_raw = None
    pipeline_gradient = pipeline_model = pipeline_optimizer = pipeline_raw = None
    all_lp: dict[str, dict[str, torch.Tensor]] = {}
    reference_post_lp: dict[str, torch.Tensor] | None = None
    valid_tokens = (batch["token_mask"] * batch["sample_mask"][:, None]).bool()
    valid_tokens[:, 0] = False
    try:
        for case in CASES:
            reset()
            result, gradient, lp_rows = execute(case, capture=True)
            gathered_results = [None] * world
            gathered_lp = [None] * world
            dist.all_gather_object(gathered_results, result)
            dist.all_gather_object(gathered_lp, lp_rows)
            metrics = aggregate_step_metrics(_aggregate_train_results(gathered_results))
            global_lp = {
                sample: row
                for rank_rows in gathered_lp
                for packet in rank_rows
                for sample, row in zip(packet["ids"], packet["logprobs"])
            }
            if set(global_lp) != set(ids):
                raise ValueError("LP replay lost or duplicated samples")
            all_lp[case.name] = global_lp
            case_report: dict[str, Any] = {
                "metrics": {
                    key: metrics.get(key)
                    for key in (
                        "gen_kl_error",
                        "policy_kl_error",
                        "grad_norm",
                        "loss",
                        "global_valid_toks",
                        "global_valid_seqs",
                    )
                }
            }
            if reference_gradient is None:
                reference_gradient = gradient
                reference_model = parity.snapshot_model(worker)
                reference_optimizer = parity.snapshot_optimizer(worker)
                reference_raw = parity.snapshot_raw_optimizer(worker)
                reference_scheduler = copy.deepcopy(worker.scheduler.state_dict())
            case_report["gradient_vs_full"] = gradient_comparison(
                gradient, reference_gradient
            )
            case_report["model_vs_full"] = exact_comparison(
                parity.snapshot_model(worker), reference_model
            )
            case_report["optimizer_vs_full"] = exact_comparison(
                parity.snapshot_optimizer(worker), reference_optimizer
            )
            case_report["raw_optimizer_vs_full"] = exact_comparison(
                parity.snapshot_raw_optimizer(worker), reference_raw
            )
            case_report["scheduler_vs_full"] = exact_comparison(
                worker.scheduler.state_dict(), reference_scheduler
            )
            case_report["eval_lp_max_abs_vs_full"] = float(
                torch.stack(
                    [global_lp[sample] - all_lp["full"][sample] for sample in ids]
                )[valid_tokens]
                .abs()
                .max()
            )
            if case.numerator:
                numerator_gradient = gradient
                numerator_model = parity.snapshot_model(worker)
                numerator_optimizer = parity.snapshot_optimizer(worker)
                numerator_raw = parity.snapshot_raw_optimizer(worker)
            if case.name == "split_fixed_shards":
                case_report["gradient_vs_numerator"] = gradient_comparison(
                    gradient, numerator_gradient
                )
                case_report["model_vs_numerator"] = exact_comparison(
                    parity.snapshot_model(worker), numerator_model
                )
                case_report["optimizer_vs_numerator"] = exact_comparison(
                    parity.snapshot_optimizer(worker), numerator_optimizer
                )
                case_report["raw_optimizer_vs_numerator"] = exact_comparison(
                    parity.snapshot_raw_optimizer(worker), numerator_raw
                )
            if case.name == "full_recompute":
                pipeline_gradient = gradient
                pipeline_model = parity.snapshot_model(worker)
                pipeline_optimizer = parity.snapshot_optimizer(worker)
                pipeline_raw = parity.snapshot_raw_optimizer(worker)
            if case.name == "split_production_recompute":
                case_report["gradient_vs_full_recompute"] = gradient_comparison(
                    gradient, pipeline_gradient
                )
                case_report["model_vs_full_recompute"] = exact_comparison(
                    parity.snapshot_model(worker), pipeline_model
                )
                case_report["optimizer_vs_full_recompute"] = exact_comparison(
                    parity.snapshot_optimizer(worker), pipeline_optimizer
                )
                case_report["raw_optimizer_vs_full_recompute"] = exact_comparison(
                    parity.snapshot_raw_optimizer(worker), pipeline_raw
                )
            # Evaluate every updated model on the same full-batch DP layout.
            # This separates changed weights from changed LP batch placement.
            worker.prepare_for_lp_inference()
            common_meta = case_metas["full"][0]
            post_lp = worker.get_logprobs(worker._fetch(common_meta))["logprobs"].cpu()
            gathered_post = [None] * world
            dist.all_gather_object(
                gathered_post, {"ids": common_meta.sample_ids, "lp": post_lp}
            )
            post_values = {
                sample: row
                for packet in gathered_post
                for sample, row in zip(packet["ids"], packet["lp"])
            }
            if reference_post_lp is None:
                reference_post_lp = post_values
            case_report["post_update_eval_lp_max_abs_vs_full"] = float(
                torch.stack(
                    [post_values[sample] - reference_post_lp[sample] for sample in ids]
                )[valid_tokens]
                .abs()
                .max()
            )
            rank_reports = [None] * world
            dist.all_gather_object(rank_reports, case_report)
            case_report["all_ranks_exact"] = {
                key: all(row[key]["exact"] for row in rank_reports)
                for key, value in case_report.items()
                if isinstance(value, dict) and "exact" in value
            }
            (output / f"{case.name}-rank{rank}.json").write_text(
                json.dumps(case_report, indent=2) + "\n"
            )
            if rank == 0:
                torch.save(global_lp, output / f"{case.name}-eval-lp.pt")
                recomputed = BatchedDataDict(dict(batch))
                recomputed["prev_logprobs"] = torch.stack(
                    [global_lp[sample] for sample in ids]
                )
                case_report["gen_kl_with_recomputed_eval_lp"] = direct_kl(recomputed)
                expected_kl = (
                    case_report["gen_kl_with_recomputed_eval_lp"]
                    if case.recompute_prev
                    else report["direct_captured_gen_kl"]
                )
                case_report["logged_gen_kl_abs_error_vs_direct"] = abs(
                    metrics["gen_kl_error"] - expected_kl
                )
                recomputed["prev_logprobs"] = torch.stack(
                    [post_values[sample] for sample in ids]
                )
                case_report["one_update_kl_on_fixed_old_rollout"] = direct_kl(
                    recomputed
                )
                torch.save(post_values, output / f"{case.name}-post-update-lp.pt")
                report["cases"][case.name] = case_report
                (output / "diagnostic.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
                print(
                    f"REPLAY_CASE {case.name} gen_kl={metrics['gen_kl_error']} grad={case_report['gradient_vs_full']}",
                    flush=True,
                )

        # No gradient/state copies, disk capture or all_gather_object in these
        # timed regions. Restore and rank barriers happen before each repeat.
        for repetition in range(args.repeats):
            names = (
                ("full", "split_fixed_shards")
                if repetition % 2 == 0
                else ("split_fixed_shards", "full")
            )
            for name in names:
                case = next(value for value in CASES if value.name == name)
                reset()
                torch.cuda.synchronize()
                start = time.perf_counter()
                execute(case, capture=False)
                torch.cuda.synchronize()
                duration = torch.tensor(
                    time.perf_counter() - start, device="cuda", dtype=torch.float64
                )
                dist.all_reduce(duration, op=dist.ReduceOp.MAX)
                report["timing"].setdefault(name, []).append(duration.item())
        report["timing_medians_seconds"] = {
            name: statistics.median(values) for name, values in report["timing"].items()
        }
        if rank == 0:
            report["complete"] = True
            report["note"] = (
                "Fixed worker workload includes LP/preparation/backward/commit; no rollout GPU or TQ/Ray dispatch latency. Not end-to-end throughput."
            )
            (output / "diagnostic.json").write_text(json.dumps(report, indent=2) + "\n")
            print(
                "REAL_ROLLOUT_REPLAY_COMPLETE",
                report["timing_medians_seconds"],
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    run(parser.parse_args())
