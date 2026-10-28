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

"""Fixed-data Gemma parity through AutoModel's SC metadata lifecycle.

Run inside the AutoModel environment with torchrun --nproc-per-node=4 (80 GB GPUs).
The local data plane exercises real metadata sharding/fetch/materialization;
network TransferQueue and live vLLM are covered by the separate SC smoke run.
"""

import argparse
import copy
import json
import os
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import ray
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from pydantic import TypeAdapter
from torch.distributed.tensor import DTensor
from torch.utils._pytree import tree_map

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
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers


def local(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def snapshot_model(worker):
    return {
        name: local(tensor).detach().cpu().clone()
        for name, tensor in list(worker.model.named_parameters())
        + list(worker.model.named_buffers())
    }


@torch.no_grad()
def restore_model(worker, state):
    for name, tensor in list(worker.model.named_parameters()) + list(
        worker.model.named_buffers()
    ):
        local(tensor).copy_(state[name])


def snapshot_optimizer(worker):
    return tree_map(
        lambda x: (
            x.detach().cpu().clone()
            if isinstance(x, torch.Tensor)
            else copy.deepcopy(x)
        ),
        worker.optimizer.state_dict(),
    )


def compare_tree(actual, expected, *, rtol, atol):
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


def make_batch(gbs, seq_len):
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


def run(args):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    torch.manual_seed(731)
    register_omegaconf_resolvers()
    cfg = load_config(args.recipe)
    cfg.policy.model_name = args.model
    cfg.policy.tokenizer.name = args.model
    cfg.policy.train_global_batch_size = world * 2
    cfg.policy.train_micro_batch_size = 1
    cfg.policy.logprob_batch_size = 1
    cfg.policy.dynamic_batching.enabled = False
    cfg.policy.sequence_packing.enabled = False
    cfg.policy.automodel_cfg.deterministic = True
    cfg.policy.automodel_cfg.expert_parallel_size = 1
    if "backend" in cfg.policy.automodel_cfg.automodel_kwargs:
        cfg.policy.automodel_cfg.automodel_kwargs.backend.dispatcher = "torch"
    cfg.policy.generation.colocated.enabled = False
    # Keep the checkpoint's full-precision parameters and use a simple reference
    # optimizer whose complete state can be compared, including its step counter.
    cfg.policy.optimizer = {
        "name": "torch.optim.AdamW",
        "kwargs": {"lr": 2e-6, "weight_decay": 0.1, "foreach": False, "fused": False},
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
    worker.setup_data_plane(LocalDataPlaneConfig(max_partitions=2))
    client = worker._require_dp_client()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    run_log = None
    if rank == 0 and args.wandb_project:
        import wandb

        run_log = wandb.init(
            project=args.wandb_project,
            name=args.name,
            config={
                "model": args.model,
                "dp": world,
                "gbs": world * 2,
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
                batch = make_batch(world * 2, args.seq_len)
                shard = batch.get_batch(rank, 2)
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
                shard = batch.get_batch(rank, 2)
                worker.prepare_for_training()
                before = snapshot_model(worker)
                optim_before = snapshot_optimizer(worker)
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
                expected_sched = copy.deepcopy(worker.scheduler.state_dict())
                restore_model(worker, before)
                worker.optimizer.load_state_dict(optim_before)
                worker.scheduler.load_state_dict(sched_before)
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state(gpu_rng)
                del before, optim_before
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
                worker.train_microbatch_presharded(metas[rank])
                split_grad = {
                    n: local(p.grad)
                    for n, p in worker.model.named_parameters()
                    if p.grad is not None
                }
                errors = {
                    "gradient_max_abs": compare_tree(
                        split_grad, captured, rtol=args.rtol, atol=args.atol
                    )
                }
                # Exercise the existing LP transition with accumulated gradients.
                worker.prepare_for_lp_inference(keep_train_buffers=True)
                worker.get_logprobs(shard)
                worker.prepare_for_training()
                compare_tree(
                    {
                        n: local(p.grad)
                        for n, p in worker.model.named_parameters()
                        if p.grad is not None
                    },
                    captured,
                    rtol=args.rtol,
                    atol=args.atol,
                )
                split_metrics = worker.finish_train_step_presharded()
                split_metrics.pop("is_replica_leader")
                split_metrics.pop("route_fallback_counts")
                errors["metrics_max_abs"] = compare_tree(
                    split_metrics, sync_metrics, rtol=args.rtol, atol=args.atol
                )
                errors["parameter_max_abs"] = compare_tree(
                    snapshot_model(worker), expected, rtol=args.rtol, atol=args.atol
                )
                errors["optimizer_max_abs"] = compare_tree(
                    snapshot_optimizer(worker),
                    expected_optim,
                    rtol=args.rtol,
                    atol=args.atol,
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
                record = {
                    "token_level": token_level,
                    "step": step,
                    **errors,
                    "loss": float(sync_metrics["global_loss"].sum()),
                    "grad_norm": float(sync_metrics["grad_norm"]),
                }
                records.append(record)
                if rank == 0:
                    print("PARITY " + json.dumps(record), flush=True)
                    if run_log:
                        run_log.log(record)
                client.clear_samples(None, partition)
                del expected, expected_optim, split_grad
                captured.clear()
                # Abort after real backward must preserve parameters and schedule.
                abort_before = snapshot_model(worker)
                abort_optimizer = snapshot_optimizer(worker)
                worker.begin_train_step(loss_fn)
                worker.train_microbatch(shard)
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
    parser.add_argument(
        "--recipe",
        default="examples/configs/recipes/llm/dapo-gemma4-e2b-it-1n8g-automodel.yaml",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--wandb-project")
    parser.add_argument("--name", default="gemma4-e2b-automodel-sc-stage1-parity")
    parser.add_argument("--source-sha", required=True)
    run(parser.parse_args())
