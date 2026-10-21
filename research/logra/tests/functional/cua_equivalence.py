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

"""Four-GPU CUA checks: native route replay, FSDP trial rollback and FD KL."""

import copy
import math
from contextlib import nullcontext
from types import SimpleNamespace

import torch
import torch.distributed as dist
from logra.config import LoGRAConfig
from logra.cua import CuaConfig, CuaUpdate, make_cua_loss
from logra.optimizer import LoGRAOptimizer
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import Gate
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.utils.checkpoint import checkpoint

from nemo_rl.algorithms.logits_sampling_utils import TrainingSamplingParams
from nemo_rl.algorithms.loss.loss_functions import ClippedPGLossConfig
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.automodel.data import ProcessedInputs, ProcessedMicrobatch


def batch_for(ids, mask, routes=None):
    data = BatchedDataDict(
        {
            "input_ids": ids,
            "input_lengths": torch.full(
                (ids.shape[0],), ids.shape[1], device=ids.device
            ),
            "token_mask": mask,
            "sample_mask": torch.ones(1, device=ids.device),
        }
    )
    if routes is not None:
        data["routed_experts"] = routes
    return ProcessedMicrobatch(
        data, ProcessedInputs(ids, ids.shape[1]), 1, ids.shape[1]
    )


def check_routing():
    cfg = MoEConfig(
        dim=8,
        inter_dim=8,
        moe_inter_dim=8,
        n_routed_experts=4,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=False,
        enable_routing_replay=True,
        gate_bias_update_factor=0,
        aux_loss_coeff=0,
        score_func="softmax",
        route_scale=1,
        norm_topk_prob=True,
        dtype=torch.float32,
    )
    net = nn.Module()
    net.layers = nn.ModuleList(
        [nn.Identity(), Gate(cfg), nn.Identity(), Gate(cfg)]
    ).cuda()
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.normal_(0, 0.2)
    update = CuaUpdate(SimpleNamespace(model=net), CuaConfig())
    ids = torch.zeros(1, 3, dtype=torch.long, device="cuda")
    for compact in (False, True):
        routes = torch.full(
            (1, 3, 2 if compact else 4, 2), -1, device="cuda", dtype=torch.int16
        )
        routes[:, 1:, 0 if compact else 1] = torch.tensor([2, 3], device="cuda")
        routes[:, 1:, 1 if compact else 3] = torch.tensor([0, 1], device="cuda")
        batch = batch_for(ids, torch.ones_like(ids), routes)
        x = torch.randn(3, 8, device="cuda", requires_grad=True)
        live = {
            layer_idx: gate(x, torch.ones(3, device="cuda", dtype=torch.bool), None)[1]
            for layer_idx, gate in ((1, net.layers[1]), (3, net.layers[3]))
        }
        with update.replay(batch):
            for layer_idx, gate in ((1, net.layers[1]), (3, net.layers[3])):
                replayed = gate(
                    x, torch.ones(3, device="cuda", dtype=torch.bool), None
                )[1]
                torch.testing.assert_close(replayed[0], live[layer_idx][0])
                expected = [2, 3] if layer_idx == 1 else [0, 1]
                torch.testing.assert_close(
                    replayed[1:], torch.tensor([expected, expected], device="cuda")
                )
        calls = []

        def forward(value):
            calls.append(
                all(gate.handle._nrl_replay is not None for gate in update.gates)
            )
            return sum(
                gate(value, torch.ones(3, device="cuda", dtype=torch.bool), None)[0]
                .square()
                .sum()
                for gate in (net.layers[1], net.layers[3])
            )

        with update.training_batches(iter([batch])) as batches:
            for _ in batches:
                checkpoint(forward, x, use_reentrant=False).backward()
        assert len(calls) >= 2 and all(calls), calls
        assert all(gate.handle._nrl_replay is None for gate in update.gates)
        try:
            with update.training_batches(iter([batch])) as batches:
                next(batches)
                raise ValueError("consumer failed")
        except ValueError as error:
            assert str(error) == "consumer failed"
        assert all(gate.handle._nrl_replay is None for gate in update.gates)
    if dist.get_rank() == 0:
        print("CUA_ROUTING_CHECKPOINT_AND_CLEANUP_OK", flush=True)


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(17, 8)
        self.proj = nn.Linear(8, 17, bias=False)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids, logits_to_keep=0, **kwargs):
        hidden = self.embed(input_ids)
        if isinstance(logits_to_keep, torch.Tensor):
            hidden = hidden[:, logits_to_keep]
        return self.proj(hidden)


def check_fd():
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(41)
    model = TinyPolicy().cuda()
    reference = copy.deepcopy(model)
    mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("dp",))
    fully_shard(model, mesh=mesh)
    optimizer = LoGRAOptimizer(
        model,
        torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0),
        LoGRAConfig(
            rank=3, optimizer="sgd", target_modules=["proj"], freeze_non_target=True
        ),
    )
    config = CuaConfig(
        routing_replay=False,
        max_update_ratio=0.03,
        kl_budget=1e-4,
        probe_sequences=1,
        probe_tokens=3,
    )
    worker = SimpleNamespace(
        model=model,
        dp_mesh=mesh,
        device_mesh=mesh,
        tokenizer=SimpleNamespace(pad_token_id=0),
        allow_flash_attn_args=False,
        sampling_params=TrainingSamplingParams(),
        _autocast_context=nullcontext,
    )
    update = CuaUpdate(worker, config)
    ids = torch.tensor([[1, 2, 3, 4, 5]], device="cuda")
    mask = torch.tensor([[0, 0, 1, 1, 1]], device="cuda")
    if rank == world - 1:
        mask.zero_()  # zero probe tokens still participate in FSDP forwards
    batch = batch_for(ids, mask)
    with update.training_batches(iter([batch])) as batches:
        for _ in batches:
            model(input_ids=ids).square().mean().backward()
    norm = optimizer.synchronize_and_clip(dist.group.WORLD, None)
    layer = optimizer.layers[0]
    direction = layer.sketch @ layer.projection
    target_norm = config.max_update_ratio * reference.proj.weight.norm()
    scale = target_norm / direction.norm()
    with torch.no_grad():
        lp0 = reference(ids)[0, 1:4].log_softmax(-1)
        reference.proj.weight.sub_(scale * direction)
        lp1 = reference(ids)[0, 1:4].log_softmax(-1)
        q = (lp0.exp() * (lp0 - lp1)).sum(-1).mean().item()
        alpha = min(config.alpha_max, math.sqrt(config.kl_budget / q))
    metrics = update.step(optimizer, norm)
    torch.testing.assert_close(
        torch.tensor(metrics["kl_proposal"]), torch.tensor(q), rtol=0.005, atol=1e-7
    )
    assert metrics["kl_probe_tokens"] == 3 * (world - 1)
    expected = reference.proj.weight + scale * (1 - alpha) * direction
    torch.testing.assert_close(
        model.proj.weight.full_tensor(), expected, rtol=2e-5, atol=2e-6
    )
    assert model.training and not update.probes
    if rank == 0:
        print(f"CUA_FSDP_FD_UPDATE_OK {metrics}", flush=True)


def check_loss():
    config = CuaConfig()
    loss = make_cua_loss(
        config,
        ClippedPGLossConfig(
            disable_ppo_ratio=True,
            reference_policy_kl_penalty=0,
            use_importance_sampling_correction=True,
            truncated_importance_sampling_type="tis",
            truncated_importance_sampling_ratio_min=0.5,
            truncated_importance_sampling_ratio=2,
        ),
    )
    current = torch.tensor(
        [[-0.2, -1.0, -2.0], [-1.0, -0.1, -0.5]], device="cuda", requires_grad=True
    )
    rollout = torch.tensor(
        [[0.0, -1.0, -0.2, -2.0], [0.0, -0.8, -1.0, -1.0]], device="cuda"
    )
    masks = torch.tensor([[0.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 0.0]], device="cuda")
    advantages = torch.tensor([[0.7] * 4, [-0.4] * 4], device="cuda")
    data = BatchedDataDict(
        {
            "input_ids": torch.zeros(2, 4, dtype=torch.long, device="cuda"),
            "token_mask": masks,
            "sample_mask": torch.ones(2, device="cuda"),
            "advantages": advantages,
            "generation_logprobs": rollout,
        }
    )
    actual, _ = loss(
        next_token_logprobs=current,
        data=data,
        global_valid_seqs=torch.tensor(2.0, device="cuda"),
        global_valid_toks=masks.sum(),
    )
    ratio = (current.detach() - rollout[:, 1:]).exp().clamp(0.5, 2)
    expected = (
        -(
            advantages[:, 1:]
            * current
            * ratio
            * masks[:, 1:]
            * (1024 / masks.sum(-1, keepdim=True))
        ).sum()
        / masks.sum()
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, current, retain_graph=True)[0],
        torch.autograd.grad(expected, current)[0],
    )
    assert "prev_logprobs" not in data
    torch.testing.assert_close(data["advantages"], advantages)
    if dist.get_rank() == 0:
        print("CUA_NATIVE_REINFORCE_TIS_EQUIVALENCE_OK", flush=True)


def check_chunked_loss():
    """The long-context path must preserve native loss and logits gradients."""
    from logra.worker import CuaLossPostProcessor

    from nemo_rl.models.automodel.train import LossPostProcessor

    mesh = init_device_mesh(
        "cuda", (dist.get_world_size(), 1), mesh_dim_names=("dp", "tp")
    )
    torch.manual_seed(23 + dist.get_rank())
    ids = torch.randint(0, 37, (1, 773), device="cuda")
    data = BatchedDataDict(
        {
            "input_ids": ids,
            "input_lengths": torch.full(
                (ids.shape[0],), ids.shape[1], device=ids.device
            ),
            "token_mask": (torch.rand_like(ids, dtype=torch.float32) > 0.4).float(),
            "sample_mask": torch.ones(1, device="cuda"),
            "advantages": torch.full_like(ids, 0.7, dtype=torch.float32),
            "generation_logprobs": torch.full_like(ids, -3.5, dtype=torch.float32),
        }
    )
    loss = make_cua_loss(
        CuaConfig(),
        ClippedPGLossConfig(
            disable_ppo_ratio=True,
            reference_policy_kl_penalty=0,
            use_importance_sampling_correction=True,
            truncated_importance_sampling_type="tis",
            truncated_importance_sampling_ratio_min=0.5,
            truncated_importance_sampling_ratio=2,
        ),
    )
    kwargs = dict(
        loss_fn=loss,
        cfg={},
        cp_mesh=None,
        cp_size=1,
        dp_size=dist.get_world_size(),
        sampling_params=TrainingSamplingParams(temperature=1.0),
    )
    chunked = CuaLossPostProcessor(vocab_group=mesh["tp"].get_group(), **kwargs)
    native = LossPostProcessor(**kwargs)
    for dtype in (torch.float32, torch.bfloat16):
        for valid in (1.0, 0.0):
            data["sample_mask"].fill_(valid)
            original = torch.randn(
                1, 773, 37, device="cuda", dtype=dtype, requires_grad=True
            )
            copy_logits = original.detach().clone().requires_grad_()
            args = dict(
                data_dict=data,
                processed_inputs=ProcessedInputs(ids, ids.shape[1]),
                global_valid_seqs=torch.tensor(1.0, device="cuda"),
                global_valid_toks=data["token_mask"].sum(),
                cp_sharder=None,
            )
            expected, _ = native(logits=original, **args)
            actual, _ = chunked(logits=copy_logits, **args)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
            actual_grad = torch.autograd.grad(actual, copy_logits)[0]
            reference_grad = torch.autograd.grad(expected, original)[0]
            torch.testing.assert_close(
                actual_grad,
                reference_grad,
                rtol=0.01 if dtype == torch.bfloat16 else 1e-5,
                atol=1e-7,
            )
    if dist.get_rank() == 0:
        print("CUA_CHUNKED_LOSS_AND_GRADIENT_EQUIVALENCE_OK", flush=True)


def main():
    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank())
    check_routing()
    check_loss()
    check_chunked_loss()
    check_fd()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
