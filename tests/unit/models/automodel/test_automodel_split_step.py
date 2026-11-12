# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real FSDP gradients and optimizer boundaries for Automodel streaming steps."""

import copy
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

pytest.importorskip("nemo_automodel")
from nemo_automodel.shared.torch_patches import (
    patch_fsdp_accumulated_grad_guard,
    patch_fsdp_uniform_reduce_dtype,
)

from nemo_rl.algorithms.loss.interfaces import LossInputType, LossType, MetricNormalizer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy.workers.automodel_policy_worker import (
    AutomodelPolicyWorkerImpl,
)
from nemo_rl.utils.timer import Timer

pytestmark = pytest.mark.gpu


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.head = nn.Linear(8, 3, bias=False)

    def forward(self, input_ids, **kwargs):
        return self.head(self.embedding(input_ids))


class _SquaredLoss:
    input_type = LossInputType.LOGIT

    def __init__(self, loss_type):
        self.loss_type = loss_type
        self.metric_normalizations = {
            "loss": MetricNormalizer.TOKENS
            if loss_type is LossType.TOKEN_LEVEL
            else MetricNormalizer.SEQUENCES,
            "token_metric": MetricNormalizer.TOKENS,
            "sequence_metric": MetricNormalizer.SEQUENCES,
            "num_valid_samples": MetricNormalizer.NONE,
            "value_max": MetricNormalizer.NONE,
        }

    def __call__(self, *, logits, data, global_valid_seqs, global_valid_toks):
        squared = (logits[:, :-1, 0].float() - data["advantages"][:, 1:]).square()
        token_mask = data["token_mask"][:, 1:]
        mask = token_mask * data["sample_mask"].unsqueeze(-1)
        token_sum = (squared * mask).sum()
        sequence_sum = (
            (squared * token_mask).sum(-1)
            / token_mask.sum(-1).clamp(min=1)
            * data["sample_mask"]
        ).sum()
        loss = (
            token_sum / global_valid_toks
            if self.loss_type is LossType.TOKEN_LEVEL
            else sequence_sum / global_valid_seqs
        )
        return loss, {
            "loss": loss.detach().item(),
            "token_metric": (token_sum / global_valid_toks).detach().item(),
            "sequence_metric": (sequence_sum / global_valid_seqs).detach().item(),
            "num_valid_samples": data["sample_mask"].sum().item(),
            "value_max": squared.detach().max().item(),
        }


def _data():
    ids = torch.arange(30, device="cuda").reshape(6, 5) % 16
    tokens = torch.ones_like(ids)
    tokens[:, 0] = 0
    tokens[0, 3:] = 0
    tokens[4, 2:] = 0
    return BatchedDataDict(
        {
            "input_ids": ids,
            "input_lengths": torch.full((6,), 5, device="cuda", dtype=torch.int64),
            "sample_mask": torch.tensor([1, 0, 1, 1, 1, 1], device="cuda"),
            "token_mask": tokens,
            "advantages": torch.cos(torch.arange(30, device="cuda").float()).reshape(
                6, 5
            ),
        }
    )


def _worker(model, mesh, dtype):
    worker = object.__new__(AutomodelPolicyWorkerImpl)
    worker.model = model
    worker.cfg = {
        "train_global_batch_size": 6,
        "train_micro_batch_size": 1,
        "automodel_cfg": {"sequence_parallel": False},
        "sequence_packing": {"enabled": False},
        "dynamic_batching": {"enabled": False},
    }
    worker.dp_size, worker.tp_size, worker.cp_size = 2, 1, 1
    worker.device_mesh = mesh
    worker.dp_mesh, worker.tp_mesh, worker.cp_mesh = mesh["dp"], mesh["tp"], mesh["cp"]
    worker.moe_mesh = None
    worker.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2)
    worker.draft_runtime = None
    worker.draft_model = None
    worker.enable_seq_packing = False
    worker.sampling_params = None
    worker.allow_flash_attn_args = False
    worker.autocast_enabled = False
    worker.dtype = dtype
    worker._router_replay_gates = None
    worker._train_step_state = None
    worker.max_grad_norm = 0.2
    worker.timer = Timer()
    worker.optimizer = torch.optim.SGD(
        model.parameters(), lr=0.2, momentum=0.9, foreach=False
    )
    worker.scheduler = torch.optim.lr_scheduler.ConstantLR(
        worker.optimizer, factor=1.0, total_iters=100
    )
    return worker


def _full_weights(model):
    return [p.full_tensor().detach().clone() for p in model.parameters()]


def _exercise(rank, rendezvous, dtype, loss_type):
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    dist.init_process_group(
        "nccl",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        patch_fsdp_accumulated_grad_guard()
        patch_fsdp_uniform_reduce_dtype()
        mesh = init_device_mesh("cuda", (2, 1, 1), mesh_dim_names=("dp", "cp", "tp"))
        torch.manual_seed(7)
        original = _Model().to(device="cuda", dtype=dtype)
        reference = copy.deepcopy(original)
        whole_model, split_model = copy.deepcopy(original), copy.deepcopy(original)
        precision = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=torch.float32)
        fully_shard(whole_model, mesh=mesh["dp"], mp_policy=precision)
        fully_shard(split_model, mesh=mesh["dp"], mp_policy=precision)
        whole, split = (
            _worker(whole_model, mesh, dtype),
            _worker(split_model, mesh, dtype),
        )
        ref_optimizer = torch.optim.SGD(
            reference.parameters(), lr=0.2, momentum=0.9, foreach=False
        )
        loss_fn = _SquaredLoss(loss_type)
        global_data = _data()
        data = global_data.get_batch(batch_idx=rank, batch_size=3)
        tolerance = (
            {"atol": 3e-3, "rtol": 3e-2}
            if dtype is torch.bfloat16
            else {"atol": 2e-6, "rtol": 2e-6}
        )
        for step in range(2):
            clip = 0.2 if step == 0 else 1000.0
            whole.max_grad_norm = split.max_grad_norm = clip
            ref_optimizer.zero_grad()
            seqs = global_data["sample_mask"].sum()
            toks = (
                global_data["token_mask"][:, 1:]
                * global_data["sample_mask"].unsqueeze(-1)
            ).sum()
            expected_loss, _ = loss_fn(
                logits=reference(global_data["input_ids"]),
                data=global_data,
                global_valid_seqs=seqs,
                global_valid_toks=toks,
            )
            expected_loss.backward()
            expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), clip)
            ref_optimizer.step()
            normal = whole.train(data, loss_fn=loss_fn)
            before = _full_weights(split.model)
            split.begin_train_step(loss_fn)
            with pytest.raises(RuntimeError, match="already open"):
                split.begin_train_step(loss_fn)
            split.train_microbatch(data.get_batch(0, 1))
            split.model.eval()
            split.train_microbatch(data.slice(1, 3))
            for old, current in zip(before, _full_weights(split.model)):
                torch.testing.assert_close(current, old, rtol=0, atol=0)
            actual = split.finish_train_step()
            torch.testing.assert_close(
                actual["global_loss"], normal["global_loss"], **tolerance
            )
            torch.testing.assert_close(
                actual["global_loss"][0],
                expected_loss.detach().float().cpu(),
                **tolerance,
            )
            torch.testing.assert_close(
                actual["grad_norm"], normal["grad_norm"], **tolerance
            )
            torch.testing.assert_close(
                actual["grad_norm"], expected_norm.float().cpu(), **tolerance
            )
            for key in [
                "loss",
                "token_metric",
                "sequence_metric",
                "num_valid_samples",
                "value_max",
            ]:
                torch.testing.assert_close(
                    torch.tensor(actual["all_mb_metrics"][key]),
                    torch.tensor(normal["all_mb_metrics"][key]),
                    **tolerance,
                )
            for expected, normal_weight, split_weight in zip(
                reference.parameters(),
                _full_weights(whole.model),
                _full_weights(split.model),
            ):
                torch.testing.assert_close(normal_weight, expected, **tolerance)
                torch.testing.assert_close(split_weight, expected, **tolerance)
            assert whole.scheduler.last_epoch == split.scheduler.last_epoch == step + 1
            assert split._train_step_state is None
        before = _full_weights(split.model)
        epoch = split.scheduler.last_epoch
        split.begin_train_step(loss_fn)
        with pytest.raises(RuntimeError, match="empty"):
            split.finish_train_step()
        split.train_microbatch(data.get_batch(0, 1))
        split.abort_train_step()
        split.abort_train_step()
        assert split.scheduler.last_epoch == epoch
        assert all(p.grad is None for p in split.model.parameters())
        for old, current in zip(before, _full_weights(split.model)):
            torch.testing.assert_close(current, old, rtol=0, atol=0)
        with pytest.raises(RuntimeError, match="open a step"):
            split.finish_train_step()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("loss_type", [LossType.TOKEN_LEVEL, LossType.SEQUENCE_LEVEL])
def test_split_step_matches_global_loss_and_one_shot_fsdp(tmp_path, dtype, loss_type):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    mp.spawn(
        _exercise,
        args=(str(tmp_path / "rendezvous"), dtype, loss_type),
        nprocs=2,
        join=True,
    )
