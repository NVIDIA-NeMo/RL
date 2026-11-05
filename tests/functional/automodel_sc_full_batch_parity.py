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

"""Fixed-data parity through AutoModel's full-batch SC metadata lifecycle.

Run inside the AutoModel environment with torchrun and enough GPUs for the model.
The L0 wrapper uses a tiny Llama on two GPUs; larger models use this same harness.
The local data plane exercises real metadata sharding/fetch/materialization;
network TransferQueue and live vLLM are covered by the separate SC smoke run.
"""

import argparse
import copy
import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import ray
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from pydantic import TypeAdapter
from torch.distributed.tensor import DTensor
from torch.utils._pytree import tree_map
from transformer_engine.pytorch.optimizers.fused_adam import FusedAdam

from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn
from nemo_rl.data_plane.adapters.local import local_batch_to_tensordict
from nemo_rl.data_plane.interfaces import LocalDataPlaneConfig
from nemo_rl.data_plane.preshard import shard_meta_for_dp
from nemo_rl.data_plane.schema import GLOBAL_FORWARD_PAD_SEQLEN
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.workers import automodel_policy_worker as worker_module
from nemo_rl.models.policy.workers.automodel_policy_worker import (
    AutomodelPolicyWorkerImpl,
)
from nemo_automodel.components.moe.layers import Gate
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers


def local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def snapshot_model(worker: AutomodelPolicyWorkerImpl) -> dict[str, torch.Tensor]:
    return {
        name: local(tensor).detach().cpu().clone()
        for name, tensor in list(worker.model.named_parameters())
        + list(worker.model.named_buffers())
    }


@torch.no_grad()
def restore_model(
    worker: AutomodelPolicyWorkerImpl, state: dict[str, torch.Tensor]
) -> None:
    for name, tensor in list(worker.model.named_parameters()) + list(
        worker.model.named_buffers()
    ):
        local(tensor).copy_(state[name])


def snapshot_optimizer(worker: AutomodelPolicyWorkerImpl) -> dict[str, Any]:
    return tree_map(
        lambda x: (
            x.detach().cpu().clone()
            if isinstance(x, torch.Tensor)
            else copy.deepcopy(x)
        ),
        worker.optimizer.state_dict(),
    )


def snapshot_raw_optimizer(worker: AutomodelPolicyWorkerImpl) -> dict[str, Any]:
    """Include TE's compressed buffers and scales, which state_dict dequantizes."""

    def clone(value: Any) -> Any:
        return (
            local(value).detach().cpu().clone()
            if isinstance(value, torch.Tensor)
            else copy.deepcopy(value)
        )

    names = dict(worker.model.named_parameters())
    state = {
        name: tree_map(clone, worker.optimizer.state[parameter])
        for name, parameter in names.items()
        if parameter in worker.optimizer.state
    }
    scales = (
        {
            name: tree_map(clone, worker.optimizer._scales[parameter])
            for name, parameter in names.items()
            if parameter in worker.optimizer._scales
        }
        if isinstance(worker.optimizer, FusedAdam)
        else None
    )
    return {"state": state, "scales": scales}


def restore_optimizer(
    worker: AutomodelPolicyWorkerImpl,
    public: dict[str, Any],
    raw: dict[str, Any],
) -> None:
    """Restore logical state and the exact compressed representation for replay."""
    worker.optimizer.load_state_dict(public)
    names = dict(worker.model.named_parameters())
    for name, state in raw["state"].items():
        for key, value in state.items():
            destination = worker.optimizer.state[names[name]][key]
            if isinstance(value, torch.Tensor):
                local(destination).copy_(value)
            else:
                worker.optimizer.state[names[name]][key] = copy.deepcopy(value)
    if isinstance(worker.optimizer, FusedAdam):
        worker.optimizer._scales = {
            names[name]: {
                key: value.to(names[name].device) for key, value in scales.items()
            }
            for name, scales in raw["scales"].items()
        }


def compare_tree(actual: Any, expected: Any, *, rtol: float, atol: float) -> float:
    """Compare all leaves, returning the largest absolute tensor/scalar error."""
    if isinstance(expected, torch.Tensor):
        a, b = local(actual).detach().cpu(), local(expected).detach().cpu()
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
        return float((a.double() - b.double()).abs().max()) if a.numel() else 0.0
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        return max(
            (
                compare_tree(actual[k], v, rtol=rtol, atol=atol)
                for k, v in expected.items()
            ),
            default=0.0,
        )
    if isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        return max(
            (
                compare_tree(a, b, rtol=rtol, atol=atol)
                for a, b in zip(actual, expected)
            ),
            default=0.0,
        )
    if isinstance(expected, float):
        torch.testing.assert_close(
            torch.tensor(actual, dtype=torch.float64),
            torch.tensor(expected, dtype=torch.float64),
            rtol=rtol,
            atol=atol,
        )
        return abs(actual - expected)
    assert actual == expected, (actual, expected)
    return 0.0


def bf16_gradient_report(
    actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]
) -> dict[str, Any]:
    """Measure gradient differences without changing the declared pass limits."""
    assert actual.keys() == expected.keys()
    difference_squared = actual_squared = expected_squared = dot_product = 0.0
    tensor_errors = {}
    for name in expected:
        actual_tensor = actual[name].double()
        expected_tensor = expected[name].double()
        difference = actual_tensor - expected_tensor
        tensor_errors[name] = (
            float(difference.abs().max()) if difference.numel() else 0.0
        )
        difference_squared += float(difference.square().sum())
        actual_squared += float(actual_tensor.square().sum())
        expected_squared += float(expected_tensor.square().sum())
        dot_product += float((actual_tensor * expected_tensor).sum())
    relative_l2 = (
        (difference_squared / expected_squared) ** 0.5
        if expected_squared
        else difference_squared**0.5
    )
    cosine = (
        dot_product / (actual_squared * expected_squared) ** 0.5
        if actual_squared and expected_squared
        else float(actual_squared == expected_squared)
    )
    norm_error = (
        abs(actual_squared**0.5 - expected_squared**0.5) / expected_squared**0.5
        if expected_squared
        else actual_squared**0.5
    )
    return {
        "relative_l2": relative_l2,
        "cosine": cosine,
        "norm_relative_error": norm_error,
        "tensor_max_abs": tensor_errors,
        "sufficient_statistics": {
            "difference_squared": difference_squared,
            "actual_squared": actual_squared,
            "expected_squared": expected_squared,
            "dot_product": dot_product,
        },
    }


def compare_bf16_gradients(
    actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]
) -> float:
    """Apply the original, unchanged BF16 gradient contract."""
    report = bf16_gradient_report(actual, expected)
    print("GRADIENT_CONTRACT " + json.dumps(report), flush=True)
    assert report["relative_l2"] <= 0.01, report
    assert report["cosine"] >= 0.9999, report
    assert report["norm_relative_error"] <= 0.01, report
    return max(report["tensor_max_abs"].values(), default=0.0)


def compare_bf16_model(
    actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]
) -> float:
    """Check FP32 master values and BF16 parameter ULPs without broad rtol."""
    assert actual.keys() == expected.keys()
    maximum = 0.0
    for name, expected_tensor in expected.items():
        actual_tensor = actual[name]
        if expected_tensor.dtype == torch.bfloat16:
            ulps = (
                actual_tensor.view(torch.int16).int()
                - expected_tensor.view(torch.int16).int()
            ).abs()
            assert not ulps.numel() or int(ulps.max()) <= 1, (name, int(ulps.max()))
        else:
            torch.testing.assert_close(
                actual_tensor, expected_tensor, rtol=0, atol=1e-7
            )
        if expected_tensor.numel():
            maximum = max(
                maximum,
                float((actual_tensor.double() - expected_tensor.double()).abs().max()),
            )
    return maximum


def measure_tree(actual: Any, expected: Any) -> float:
    """Measure differences without treating a diagnostic as numerical parity."""
    if isinstance(expected, torch.Tensor):
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
        if expected.ndim == 0:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        return (
            float((actual.double() - expected.double()).abs().max())
            if expected.numel()
            else 0.0
        )
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        return max(
            (measure_tree(actual[key], value) for key, value in expected.items()),
            default=0.0,
        )
    if isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        return max(
            (
                measure_tree(value, reference)
                for value, reference in zip(actual, expected)
            ),
            default=0.0,
        )
    assert actual == expected
    return 0.0


def optimizer_oracle(
    worker: AutomodelPolicyWorkerImpl,
    before: dict[str, torch.Tensor],
    optim_before: dict[str, Any],
    raw_optim_before: dict[str, Any],
    sched_before: dict[str, Any],
    gradients: dict[str, torch.Tensor],
    gate_loads: list[tuple[Gate, torch.Tensor | None]],
    buffers: dict[str, torch.Tensor],
    clip: Callable[..., Any],
) -> None:
    """Apply a single reference update to the captured streaming gradients."""
    actual_model = snapshot_model(worker)
    actual_optimizer = snapshot_optimizer(worker)
    actual_raw_optimizer = snapshot_raw_optimizer(worker)
    actual_scheduler = copy.deepcopy(worker.scheduler.state_dict())
    restore_model(worker, before)
    restore_optimizer(worker, optim_before, raw_optim_before)
    worker.scheduler.load_state_dict(sched_before)
    for name, buffer in worker.model.named_buffers():
        local(buffer).copy_(buffers[name])
    for gate, load in gate_loads:
        gate._cumulative_expert_load = load
    for name, parameter in worker.model.named_parameters():
        if name in gradients:
            parameter.grad = torch.zeros_like(parameter)
            local(parameter.grad).copy_(gradients[name])
    clip(
        worker.max_grad_norm,
        [worker.model],
        norm_type=2.0,
        pp_enabled=False,
        device_mesh=worker.device_mesh,
        moe_mesh=worker.moe_mesh,
        ep_axis_name="ep" if worker.moe_mesh is not None else None,
        pp_axis_name=None,
        foreach=True,
        num_label_tokens=1,
        dp_group_size=worker.dp_size * worker.cp_size,
    )
    worker.optimizer.step()
    worker._update_moe_gate_bias_if_supported()
    worker.scheduler.step()
    worker.optimizer.zero_grad()
    compare_tree(snapshot_model(worker), actual_model, rtol=0, atol=0)
    compare_tree(snapshot_optimizer(worker), actual_optimizer, rtol=0, atol=0)
    compare_tree(snapshot_raw_optimizer(worker), actual_raw_optimizer, rtol=0, atol=0)
    compare_tree(worker.scheduler.state_dict(), actual_scheduler, rtol=0, atol=0)


def make_batch(gbs: int, seq_len: int) -> BatchedDataDict[Any]:
    gen = torch.Generator().manual_seed(731)
    tokens = torch.randint(100, 1000, (gbs, seq_len), generator=gen)
    lengths = torch.tensor(
        [seq_len - 3 * (i % 4) for i in range(gbs)], dtype=torch.int32
    )
    attention = (torch.arange(seq_len)[None, :] < lengths[:, None]).float()
    token_mask = attention.clone()
    token_mask[:, :8] = 0
    sample_mask = torch.ones(gbs)
    sample_mask[-1] = 0
    return BatchedDataDict(
        {
            "input_ids": tokens,
            "input_lengths": lengths,
            "attention_mask": attention,
            "token_mask": token_mask,
            "sample_mask": sample_mask,
            "advantages": torch.randn(gbs, seq_len, generator=gen),
        }
    )


def normalizer_placement_control(
    worker: AutomodelPolicyWorkerImpl,
    args: argparse.Namespace,
    dispatch_sizes: list[int],
    output: Path,
) -> None:
    """Diagnose rounding with an independent full-input numerator backward.

    Compare three paths from identical state: ordinary V1, ordinary V1 with
    only normalization moved to clip entry, and SC streaming. The last two
    must match exactly. A passing diagnostic does not pass or relax the
    original BF16 full-batch gradient gate and cannot release a long run.
    """
    rank, world = dist.get_rank(), dist.get_world_size()
    records = []
    original_process = worker_module.process_global_batch
    original_clip = worker_module.scale_grads_and_clip_grad_norm
    client = worker._require_dp_client()
    for token_level in (True, False):
        loss_fn = ClippedPGLossFn(
            ClippedPGLossConfig(
                token_level_loss=token_level,
                reference_policy_kl_penalty=0.01,
                force_on_policy_ratio=True,
                use_importance_sampling_correction=True,
            )
        )
        for step in range(args.steps):
            batch = make_batch(world * args.local_batch_size, args.seq_len)
            shard = batch.get_batch(rank, args.local_batch_size)
            worker.prepare_for_lp_inference()
            lp = worker.get_logprobs(shard)["logprobs"].cpu()
            gathered = [torch.empty_like(lp, device="cuda") for _ in range(world)]
            dist.all_gather(gathered, lp.cuda())
            full_lp = torch.cat(gathered).cpu()
            batch["prev_logprobs"] = full_lp
            batch["generation_logprobs"] = full_lp - 0.02
            batch["reference_policy_logprobs"] = full_lp + 0.01
            shard = batch.get_batch(rank, args.local_batch_size)
            worker.prepare_for_training()
            before = snapshot_model(worker)
            optim_before = snapshot_optimizer(worker)
            raw_before = snapshot_raw_optimizer(worker)
            sched_before = copy.deepcopy(worker.scheduler.state_dict())
            cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()

            def reset() -> None:
                restore_model(worker, before)
                restore_optimizer(worker, optim_before, raw_before)
                worker.scheduler.load_state_dict(sched_before)
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state(gpu_rng)

            def full_input(numerator: bool) -> dict[str, torch.Tensor]:
                gradients = {}
                denominator = 1.0

                def process(*values: Any, **kwargs: Any) -> dict[str, Any]:
                    nonlocal denominator
                    result = original_process(*values, **kwargs)
                    key = "global_valid_toks" if token_level else "global_valid_seqs"
                    denominator = result[key].item()
                    if numerator:
                        result["global_valid_toks"] = torch.ones_like(
                            result["global_valid_toks"]
                        )
                        result["global_valid_seqs"] = torch.ones_like(
                            result["global_valid_seqs"]
                        )
                    return result

                def capture(*values: Any, **kwargs: Any) -> Any:
                    for name, parameter in worker.model.named_parameters():
                        if parameter.grad is not None:
                            if numerator:
                                parameter.grad.mul_(1.0 / denominator)
                            gradients[name] = (
                                local(parameter.grad).detach().cpu().clone()
                            )
                    return original_clip(*values, **kwargs)

                with (
                    patch.object(
                        worker_module, "process_global_batch", side_effect=process
                    ),
                    patch.object(
                        worker_module,
                        "scale_grads_and_clip_grad_norm",
                        side_effect=capture,
                    ),
                ):
                    worker.train(shard, loss_fn)
                return gradients

            ordinary_gradients = full_input(False)
            reset()
            numerator_gradients = full_input(True)
            original_report = bf16_gradient_report(
                numerator_gradients, ordinary_gradients
            )
            sufficient = torch.tensor(
                list(original_report["sufficient_statistics"].values()),
                dtype=torch.float64,
                device="cuda",
            )
            dist.all_reduce(sufficient)
            difference_sq, actual_sq, expected_sq, dot = sufficient.tolist()
            global_report = {
                "relative_l2": (difference_sq / expected_sq) ** 0.5
                if expected_sq
                else difference_sq**0.5,
                "cosine": dot / (actual_sq * expected_sq) ** 0.5
                if actual_sq and expected_sq
                else float(actual_sq == expected_sq),
                "norm_relative_error": abs(actual_sq**0.5 - expected_sq**0.5)
                / expected_sq**0.5
                if expected_sq
                else actual_sq**0.5,
            }
            reference_model = snapshot_model(worker)
            reference_optimizer = snapshot_optimizer(worker)
            reference_raw = snapshot_raw_optimizer(worker)
            reference_scheduler = copy.deepcopy(worker.scheduler.state_dict())
            reset()
            partition = f"normalizer-{token_level}-{step}"
            client.register_partition(
                partition_id=partition,
                fields=list(batch),
                num_samples=batch.size,
                consumer_tasks=["train"],
            )
            meta = client.put_samples(
                sample_ids=[str(index) for index in range(batch.size)],
                partition_id=partition,
                fields=local_batch_to_tensordict(dict(batch), batch_size=batch.size),
            )
            meta = replace(
                meta,
                task_name="train",
                extra_info={
                    **meta.extra_info,
                    GLOBAL_FORWARD_PAD_SEQLEN: args.seq_len,
                },
            )
            metas, _ = shard_meta_for_dp(meta, dp_world=world)
            worker.begin_train_step_presharded(loss_fn)
            offset = 0
            for size in dispatch_sizes:
                worker.train_microbatch_presharded(
                    metas[rank].slice(offset, offset + size)
                )
                offset += size
                lp_cpu, lp_gpu = torch.get_rng_state(), torch.cuda.get_rng_state()
                worker.prepare_for_lp_inference(keep_train_buffers=True)
                worker.get_logprobs(shard)
                worker.prepare_for_training()
                torch.set_rng_state(lp_cpu)
                torch.cuda.set_rng_state(lp_gpu)

            def check_gradient(*values: Any, **kwargs: Any) -> Any:
                gradients = {
                    name: local(parameter.grad).detach().cpu()
                    for name, parameter in worker.model.named_parameters()
                    if parameter.grad is not None
                }
                compare_tree(gradients, numerator_gradients, rtol=0, atol=0)
                return original_clip(*values, **kwargs)

            with patch.object(
                worker_module,
                "scale_grads_and_clip_grad_norm",
                side_effect=check_gradient,
            ):
                worker.finish_train_step_presharded()
            compare_tree(snapshot_model(worker), reference_model, rtol=0, atol=0)
            compare_tree(
                snapshot_optimizer(worker), reference_optimizer, rtol=0, atol=0
            )
            compare_tree(snapshot_raw_optimizer(worker), reference_raw, rtol=0, atol=0)
            compare_tree(
                worker.scheduler.state_dict(), reference_scheduler, rtol=0, atol=0
            )
            record = {
                "token_level": token_level,
                "step": step,
                "numerator_reference_exact": True,
                "original_bf16_gradient_gate_passed": (
                    original_report["relative_l2"] <= 0.01
                    and original_report["cosine"] >= 0.9999
                    and original_report["norm_relative_error"] <= 0.01
                ),
                "normalizer_only_difference": original_report,
                "global_sharded_vector_difference": global_report,
            }
            records.append(record)
            (output / f"normalizer-control-rank{rank}.json").write_text(
                json.dumps(records, indent=2) + "\n"
            )
            print(
                f"NORMALIZER_CONTROL rank={rank} token={token_level} step={step} exact=True original_l2={original_report['relative_l2']}",
                flush=True,
            )
    original_gates = torch.tensor(
        [record["original_bf16_gradient_gate_passed"] for record in records],
        dtype=torch.int64,
        device="cuda",
    )
    dist.all_reduce(original_gates, op=dist.ReduceOp.MIN)
    dist.barrier()
    if rank == 0:
        (output / "normalizer-control.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "contract": "normalizer_placement_diagnostic_only",
                    "releases_training": False,
                    "source_sha": args.source_sha,
                    "source_file_sha256": {
                        str(
                            path.relative_to(Path(__file__).resolve().parents[2])
                        ): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in (
                            Path(__file__).resolve(),
                            Path(worker_module.__file__).resolve(),
                        )
                    },
                    "original_bf16_gradient_gate_passed": bool(
                        original_gates.all().item()
                    ),
                    "global_sharded_vector_differences": [
                        record["global_sharded_vector_difference"] for record in records
                    ],
                    "rank_reports": [
                        f"normalizer-control-rank{index}.json" for index in range(world)
                    ],
                },
                indent=2,
            )
            + "\n"
        )


def run(args: argparse.Namespace) -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    torch.manual_seed(731)
    register_omegaconf_resolvers()
    cfg = load_config(args.recipe)
    cfg.policy.model_name = args.model
    cfg.policy.tokenizer.name = args.model
    if args.precision is not None:
        cfg.policy.precision = args.precision
        if args.precision == "float32":
            cfg.policy.automodel_cfg.automodel_kwargs.attn_implementation = "sdpa"
    dispatch_sizes = (
        [int(size) for size in args.dispatch_sizes.split(",")]
        if args.dispatch_sizes is not None
        else [args.local_batch_size]
    )
    if min(dispatch_sizes) < 1 or sum(dispatch_sizes) != args.local_batch_size:
        raise ValueError("dispatch sizes must be positive and sum to local batch size")
    cfg.policy.train_global_batch_size = world * args.local_batch_size
    cfg.policy.train_micro_batch_size = 1
    cfg.policy.logprob_batch_size = 1
    cfg.policy.dynamic_batching.enabled = False
    cfg.policy.sequence_packing.enabled = False
    cfg.policy.automodel_cfg.deterministic = True
    if args.force_hf:
        cfg.policy.automodel_cfg.automodel_kwargs.force_hf = True
        cfg.policy.automodel_cfg.automodel_kwargs.pop("backend", None)
    elif (
        cfg.policy.automodel_cfg.expert_parallel_size in (None, 1)
        and "backend" in cfg.policy.automodel_cfg.automodel_kwargs
    ):
        cfg.policy.automodel_cfg.automodel_kwargs.backend.dispatcher = "torch"
    cfg.policy.generation.colocated.enabled = False
    # Keep the checkpoint's full-precision parameters and use a simple reference
    # optimizer whose complete state can be compared, including its step counter.
    if not args.recipe_optimizer:
        cfg.policy.optimizer = {
            "name": "torch.optim.AdamW",
            "kwargs": {
                "lr": 2e-6,
                "weight_decay": 0.1,
                "foreach": False,
                "fused": False,
            },
        }
        cfg.policy.scheduler = [
            {
                "name": "torch.optim.lr_scheduler.StepLR",
                "kwargs": {"step_size": 1, "gamma": 0.9},
            },
            {"milestones": []},
        ]
    policy_cfg = TypeAdapter(PolicyConfig).validate_python(
        OmegaConf.to_container(cfg.policy, resolve=True)
    )
    # torchrun supplies rank/device ownership instead of a Ray actor for this
    # numerical test; production initialization is otherwise unchanged.
    with patch.object(ray, "get_gpu_ids", return_value=[local_rank]):
        worker = AutomodelPolicyWorkerImpl(policy_cfg, init_reference_model=False)
    expert_parallel_size = cfg.policy.automodel_cfg.expert_parallel_size
    if expert_parallel_size is not None and expert_parallel_size > 1:
        assert worker.moe_mesh is not None
        assert worker.moe_mesh["ep"].size() == expert_parallel_size
        assert worker.dp_size == world
    if args.strict_determinism:
        # Worker initialization selects warn-only determinism. Numerical
        # regression tests require deterministic kernels or an explicit failure.
        torch.use_deterministic_algorithms(True, warn_only=False)
    worker.setup_data_plane(LocalDataPlaneConfig(max_partitions=2))
    client = worker._require_dp_client()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.normalizer_control:
        try:
            normalizer_placement_control(worker, args, dispatch_sizes, output)
        finally:
            dist.destroy_process_group()
        return
    run_log = None
    if rank == 0 and args.wandb_project:
        import wandb

        run_log = wandb.init(
            project=args.wandb_project,
            name=args.name,
            config={
                "model": args.model,
                "force_hf": args.force_hf,
                "dp": world,
                "ep": cfg.policy.automodel_cfg.expert_parallel_size,
                "gbs": world * args.local_batch_size,
                "dispatch_sizes": dispatch_sizes,
                "steps_per_loss_type": args.steps,
                "rtol": args.rtol,
                "atol": args.atol,
                "metadata_transport": "local",
                "source_sha": args.source_sha,
            },
        )
        print("WANDB_URL=" + run_log.url, flush=True)
    records = []
    passed = False
    try:
        for token_level in (True, False):
            loss_fn = ClippedPGLossFn(
                ClippedPGLossConfig(
                    token_level_loss=token_level,
                    reference_policy_kl_penalty=0.01,
                    force_on_policy_ratio=True,
                    use_importance_sampling_correction=True,
                )
            )
            for step in range(args.steps):
                batch = make_batch(world * args.local_batch_size, args.seq_len)
                if args.mask_rank is not None:
                    start = args.mask_rank * args.local_batch_size
                    batch["sample_mask"][start : start + args.local_batch_size] = 0
                shard = batch.get_batch(rank, args.local_batch_size)
                worker.prepare_for_lp_inference()
                logprobs = worker.get_logprobs(shard)["logprobs"].cpu()
                gathered = [
                    torch.empty_like(logprobs, device="cuda") for _ in range(world)
                ]
                dist.all_gather(gathered, logprobs.cuda())
                full_lp = torch.cat(gathered).cpu()
                batch["prev_logprobs"] = full_lp
                batch["generation_logprobs"] = full_lp - 0.02
                batch["reference_policy_logprobs"] = full_lp + 0.01
                shard = batch.get_batch(rank, args.local_batch_size)
                worker.prepare_for_training()
                before = snapshot_model(worker)
                optim_before = snapshot_optimizer(worker)
                raw_optim_before = snapshot_raw_optimizer(worker)
                sched_before = copy.deepcopy(worker.scheduler.state_dict())
                cpu_rng = torch.get_rng_state()
                gpu_rng = torch.cuda.get_rng_state()
                captured = {}
                original_clip = worker_module.scale_grads_and_clip_grad_norm

                def capture(*clip_args, **clip_kwargs):
                    captured.update(
                        {
                            n: local(p.grad).detach().cpu().clone()
                            for n, p in worker.model.named_parameters()
                            if p.grad is not None
                        }
                    )
                    return original_clip(*clip_args, **clip_kwargs)

                with patch.object(
                    worker_module, "scale_grads_and_clip_grad_norm", side_effect=capture
                ):
                    sync_metrics = worker.train(shard, loss_fn)
                expected = snapshot_model(worker)
                expected_optim = snapshot_optimizer(worker)
                expected_raw_optim = snapshot_raw_optimizer(worker)
                expected_sched = copy.deepcopy(worker.scheduler.state_dict())
                restore_model(worker, before)
                restore_optimizer(worker, optim_before, raw_optim_before)
                worker.scheduler.load_state_dict(sched_before)
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state(gpu_rng)
                partition = f"parity-{token_level}-{step}"
                client.register_partition(
                    partition_id=partition,
                    fields=list(batch),
                    num_samples=batch.size,
                    consumer_tasks=["train"],
                )
                meta = client.put_samples(
                    sample_ids=[str(i) for i in range(batch.size)],
                    partition_id=partition,
                    fields=local_batch_to_tensordict(
                        dict(batch), batch_size=batch.size
                    ),
                )
                meta = replace(
                    meta,
                    task_name="train",
                    extra_info={
                        **meta.extra_info,
                        GLOBAL_FORWARD_PAD_SEQLEN: args.seq_len,
                    },
                )
                metas, _ = shard_meta_for_dp(meta, dp_world=world)
                worker.begin_train_step_presharded(loss_fn)
                offset = 0
                for size in dispatch_sizes:
                    worker.train_microbatch_presharded(
                        metas[rank].slice(offset, offset + size)
                    )
                    offset += size
                    retained_grad = {
                        name: local(parameter.grad).detach().cpu().clone()
                        for name, parameter in worker.model.named_parameters()
                        if parameter.grad is not None
                    }
                    lp_cpu_rng = torch.get_rng_state()
                    lp_gpu_rng = torch.cuda.get_rng_state()
                    worker.prepare_for_lp_inference(keep_train_buffers=True)
                    worker.get_logprobs(shard)
                    worker.prepare_for_training()
                    torch.set_rng_state(lp_cpu_rng)
                    torch.cuda.set_rng_state(lp_gpu_rng)
                    compare_tree(
                        {
                            name: local(parameter.grad)
                            for name, parameter in worker.model.named_parameters()
                            if parameter.grad is not None
                        },
                        retained_grad,
                        rtol=0,
                        atol=0,
                    )
                split_grad = {}
                gate_loads = [
                    (
                        gate,
                        gate._cumulative_expert_load.detach().clone()
                        if gate._cumulative_expert_load is not None
                        else None,
                    )
                    for gate in worker.model.modules()
                    if isinstance(gate, Gate)
                ]
                backward_buffers = {
                    name: local(buffer).detach().clone()
                    for name, buffer in worker.model.named_buffers()
                }

                def capture_split(*clip_args, **clip_kwargs):
                    split_grad.update(
                        {
                            name: local(parameter.grad).detach().cpu().clone()
                            for name, parameter in worker.model.named_parameters()
                            if parameter.grad is not None
                        }
                    )
                    return original_clip(*clip_args, **clip_kwargs)

                with patch.object(
                    worker_module,
                    "scale_grads_and_clip_grad_norm",
                    side_effect=capture_split,
                ):
                    split_metrics = worker.finish_train_step_presharded()
                errors = {
                    "gradient_max_abs": compare_bf16_gradients(split_grad, captured)
                    if args.bf16_contract
                    else compare_tree(
                        split_grad, captured, rtol=args.rtol, atol=args.atol
                    )
                }
                split_metrics.pop("is_replica_leader")
                split_metrics.pop("route_fallback_counts")
                if args.bf16_contract:
                    torch.testing.assert_close(
                        split_metrics["grad_norm"],
                        sync_metrics["grad_norm"],
                        rtol=0.01,
                        atol=0,
                    )
                    split_metrics = dict(split_metrics)
                    sync_metrics = dict(sync_metrics)
                    split_metrics.pop("grad_norm")
                    expected_grad_norm = sync_metrics.pop("grad_norm")
                else:
                    expected_grad_norm = sync_metrics["grad_norm"]
                errors["metrics_max_abs"] = compare_tree(
                    split_metrics, sync_metrics, rtol=args.rtol, atol=args.atol
                )
                if args.optimizer_oracle:
                    actual_model = snapshot_model(worker)
                    actual_optim = snapshot_optimizer(worker)
                    worst_name = next(iter(captured))
                    worst_difference = -1.0
                    for parameter_name in captured:
                        difference = float(
                            (actual_model[parameter_name] - expected[parameter_name])
                            .abs()
                            .max()
                        )
                        if difference > worst_difference:
                            worst_name = parameter_name
                            worst_difference = difference
                    worst_index = int(
                        (actual_model[worst_name] - expected[worst_name])
                        .abs()
                        .reshape(-1)
                        .argmax()
                    )
                    print(
                        "UPDATE_DIAGNOSTIC "
                        + json.dumps(
                            {
                                "parameter": worst_name,
                                "flat_index": worst_index,
                                "full_batch_gradient": float(
                                    captured[worst_name].reshape(-1)[worst_index]
                                ),
                                "streaming_gradient": float(
                                    split_grad[worst_name].reshape(-1)[worst_index]
                                ),
                                "full_batch_parameter": float(
                                    expected[worst_name].reshape(-1)[worst_index]
                                ),
                                "streaming_parameter": float(
                                    actual_model[worst_name].reshape(-1)[worst_index]
                                ),
                                "lr": worker.optimizer.param_groups[0]["lr"],
                            }
                        ),
                        flush=True,
                    )
                    errors["parameter_max_abs"] = measure_tree(actual_model, expected)
                    errors["optimizer_max_abs"] = measure_tree(
                        actual_optim, expected_optim
                    )
                    errors["raw_optimizer_max_abs"] = measure_tree(
                        snapshot_raw_optimizer(worker), expected_raw_optim
                    )
                    try:
                        compare_bf16_model(actual_model, expected)
                    except AssertionError:
                        v1_parameter_passed = False
                    else:
                        v1_parameter_passed = True
                    try:
                        compare_tree(actual_optim, expected_optim, rtol=1e-2, atol=1e-6)
                    except AssertionError:
                        v1_optimizer_passed = False
                    else:
                        v1_optimizer_passed = True
                    optimizer_oracle(
                        worker,
                        before,
                        optim_before,
                        raw_optim_before,
                        sched_before,
                        split_grad,
                        gate_loads,
                        backward_buffers,
                        original_clip,
                    )
                    print(
                        "OPTIMIZER_ORACLE "
                        + json.dumps(
                            {
                                "passed": True,
                                "v1_parameter_gate_passed": v1_parameter_passed,
                                "v1_optimizer_gate_passed": v1_optimizer_passed,
                            }
                        ),
                        flush=True,
                    )
                else:
                    errors["parameter_max_abs"] = (
                        compare_bf16_model(snapshot_model(worker), expected)
                        if args.bf16_contract
                        else compare_tree(
                            snapshot_model(worker),
                            expected,
                            rtol=args.rtol,
                            atol=args.atol,
                        )
                    )
                    errors["optimizer_max_abs"] = compare_tree(
                        snapshot_optimizer(worker),
                        expected_optim,
                        rtol=1e-2 if args.bf16_contract else args.rtol,
                        atol=1e-6 if args.bf16_contract else args.atol,
                    )
                    errors["raw_optimizer_max_abs"] = compare_tree(
                        snapshot_raw_optimizer(worker),
                        expected_raw_optim,
                        rtol=1e-2 if args.bf16_contract else args.rtol,
                        atol=1e-6 if args.bf16_contract else args.atol,
                    )
                compare_tree(
                    worker.scheduler.state_dict(), expected_sched, rtol=0, atol=0
                )
                errors = {k: float(v) for k, v in errors.items()}
                maxima = torch.tensor(
                    list(errors.values()), device="cuda", dtype=torch.float64
                )
                dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
                errors = dict(zip(errors, maxima.cpu().tolist()))
                if args.optimizer_oracle:
                    gates = torch.tensor(
                        [v1_parameter_passed, v1_optimizer_passed],
                        device="cuda",
                        dtype=torch.int64,
                    )
                    dist.all_reduce(gates, op=dist.ReduceOp.MIN)
                    v1_parameter_passed, v1_optimizer_passed = map(bool, gates.tolist())
                record = {
                    "token_level": token_level,
                    "step": step,
                    "dispatch_sizes": dispatch_sizes,
                    "optimizer_oracle": args.optimizer_oracle,
                    "v1_parameter_gate_passed": v1_parameter_passed
                    if args.optimizer_oracle
                    else True,
                    "v1_optimizer_gate_passed": v1_optimizer_passed
                    if args.optimizer_oracle
                    else True,
                    **errors,
                    "loss": float(sync_metrics["global_loss"].sum()),
                    "grad_norm": float(expected_grad_norm),
                }
                records.append(record)
                if rank == 0:
                    print("PARITY " + json.dumps(record), flush=True)
                    if run_log:
                        run_log.log(record)
                client.clear_samples(None, partition)
                del expected, expected_optim, before, optim_before
                split_grad.clear()
                captured.clear()
                # Abort after real backward must preserve parameters and schedule.
                abort_before = snapshot_model(worker)
                abort_optimizer = snapshot_optimizer(worker)
                worker.begin_train_step(loss_fn)
                worker.train_microbatch(shard.slice(0, dispatch_sizes[0]))
                worker.abort_train_step()
                compare_tree(snapshot_model(worker), abort_before, rtol=0, atol=0)
                compare_tree(
                    snapshot_optimizer(worker), abort_optimizer, rtol=0, atol=0
                )
                compare_tree(
                    worker.scheduler.state_dict(), expected_sched, rtol=0, atol=0
                )
                assert all(p.grad is None for p in worker.model.parameters())
                del abort_before, abort_optimizer
        passed = True
        if run_log:
            run_log.summary["parity_passed"] = True
        if rank == 0:
            (output / "parity.json").write_text(
                json.dumps(
                    {
                        "passed": True,
                        "source_sha": args.source_sha,
                        "source_file_sha256": {
                            str(
                                path.relative_to(Path(__file__).resolve().parents[2])
                            ): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in (
                                Path(__file__).resolve(),
                                Path(worker_module.__file__).resolve(),
                            )
                        },
                        "recipe_optimizer": args.recipe_optimizer,
                        "optimizer_class": type(worker.optimizer).__qualname__,
                        "model_parameter_dtypes": sorted(
                            {
                                str(parameter.dtype)
                                for parameter in worker.model.parameters()
                            }
                        ),
                        "raw_optimizer_tensor_dtypes": sorted(
                            {
                                str(value.dtype)
                                for state in snapshot_raw_optimizer(worker)[
                                    "state"
                                ].values()
                                for value in state.values()
                                if isinstance(value, torch.Tensor)
                            }
                        ),
                        "activation_checkpointing": cfg.policy.automodel_cfg.activation_checkpointing,
                        "contract": "bf16_gradient_and_optimizer_oracle_v2"
                        if args.optimizer_oracle
                        else "full_batch_state_parity",
                        "full_batch_parameter_gate_passed": all(
                            record["v1_parameter_gate_passed"] for record in records
                        ),
                        "full_batch_optimizer_gate_passed": all(
                            record["v1_optimizer_gate_passed"] for record in records
                        ),
                        "records": records,
                        "wandb_url": run_log.url if run_log else None,
                    },
                    indent=2,
                )
            )
            print("PARITY_PASSED", flush=True)
    finally:
        if run_log:
            run_log.finish(exit_code=0 if passed else 1)
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--force-hf", action="store_true")
    parser.add_argument("--strict-determinism", action="store_true")
    parser.add_argument(
        "--recipe",
        default="examples/configs/recipes/llm/dapo-gemma4-e2b-it-1n8g-automodel.yaml",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--local-batch-size", type=int, default=2)
    parser.add_argument("--dispatch-sizes")
    parser.add_argument("--precision", choices=("float32", "bfloat16"))
    parser.add_argument("--bf16-contract", action="store_true")
    parser.add_argument("--optimizer-oracle", action="store_true")
    parser.add_argument("--recipe-optimizer", action="store_true")
    parser.add_argument("--mask-rank", type=int)
    parser.add_argument("--normalizer-control", action="store_true")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--wandb-project")
    parser.add_argument("--name", default="gemma4-e2b-automodel-sc-full-batch-parity")
    parser.add_argument("--source-sha", required=True)
    run(parser.parse_args())
