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

"""SC ownership tests with real CPU gradients, optimizer, and mutable MoE state."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch
from torch.utils.checkpoint import checkpoint

pytest.importorskip("nemo_automodel")
from nemo_automodel.components.moe.layers import Gate, MoE
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.parallelizer import (
    _preserve_gate_load_during_recompute,
)

from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy.workers import automodel_policy_worker as implementation
from nemo_rl.utils.timer import Timer

pytestmark = pytest.mark.automodel


@pytest.fixture
def worker(monkeypatch):
    worker = object.__new__(implementation.AutomodelPolicyWorkerImpl)
    worker._train_step_state = None
    worker._train_step_commit_failed = False
    worker.cfg = {
        "train_global_batch_size": 4,
        "train_micro_batch_size": 1,
        "dynamic_batching": {"enabled": False},
        "automodel_cfg": {},
    }
    worker.dp_size = 2
    worker.tp_size = worker.cp_size = 1
    worker.cpu_offload = worker.is_vlm = worker.is_moe_model = False
    worker.lora_enabled = worker.enable_seq_packing = False
    worker.draft_model = None
    worker.model = torch.nn.Linear(2, 1, bias=False)
    worker.model.register_buffer("counter", torch.tensor(0))
    worker.optimizer = torch.optim.SGD(worker.model.parameters(), lr=0.1, momentum=0.9)
    worker.scheduler = torch.optim.lr_scheduler.StepLR(worker.optimizer, 1, gamma=0.5)
    worker.max_grad_norm = 1.0
    worker.device_mesh = worker.moe_mesh = worker.cp_mesh = None
    worker.dp_mesh = Mock()
    worker.timer = Timer()
    worker.tokenizer = SimpleNamespace(pad_token_id=0)
    worker.sampling_params = worker._router_replay_gates = None
    worker._is_reward_model = False
    worker.allow_flash_attn_args = False
    worker.dtype = torch.float32
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *a, **kw: None)
    tensor_factory = torch.tensor

    def cpu_tensor(*args, **kwargs):
        kwargs.pop("device", None)
        return tensor_factory(*args, **kwargs)

    monkeypatch.setattr(torch, "tensor", cpu_tensor)
    monkeypatch.setattr(implementation, "check_sequence_dim", lambda data: (1, 2))
    monkeypatch.setattr(implementation, "LossPostProcessor", Mock())
    monkeypatch.setattr(
        implementation,
        "process_global_batch",
        lambda data, *a, **kw: {
            "batch": data,
            "global_valid_seqs": torch.tensor(4),
            "global_valid_toks": torch.tensor(8),
        },
    )
    monkeypatch.setattr(
        implementation, "get_microbatch_iterator", lambda *a, **kw: (None, 2)
    )

    def backward(**kwargs):
        for _ in range(2):
            loss = worker.model(torch.ones(1, 2)).sum() / 2
            loss.backward()
            yield loss, {"num_valid_samples": 1, "loss": loss.item()}

    monkeypatch.setattr(implementation, "automodel_forward_backward", backward)
    monkeypatch.setattr(
        implementation,
        "aggregate_training_statistics",
        lambda **kw: {
            "global_loss": torch.tensor(kw["losses"]),
            "grad_norm": None,
        },
    )
    monkeypatch.setattr(
        implementation,
        "scale_grads_and_clip_grad_norm",
        lambda max_norm, models, **kw: torch.nn.utils.clip_grad_norm_(
            models[0].parameters(), max_norm
        ),
    )
    return worker


def dispatch(worker):
    data = Mock(size=2)
    data.to.return_value = data
    worker.train_microbatch(data)


def test_single_commit_and_retained_gradients(worker):
    before = worker.model.weight.detach().clone()
    worker.begin_train_step(Mock())
    with pytest.raises(RuntimeError, match="already open"):
        worker.begin_train_step(Mock())
    with pytest.raises(RuntimeError, match="successfully completed"):
        worker.finish_train_step()
    dispatch(worker)
    with pytest.raises(RuntimeError, match="exactly one"):
        dispatch(worker)
    torch.testing.assert_close(before, worker.model.weight)
    torch.testing.assert_close(worker.model.weight.grad, torch.ones_like(before))
    assert worker.scheduler.last_epoch == 0
    worker.model.eval()
    result = worker.finish_train_step()
    torch.testing.assert_close(result["grad_norm"], torch.tensor(2.0).sqrt())
    assert worker.model.training
    assert worker.scheduler.last_epoch == 1
    assert worker.optimizer.param_groups[0]["lr"] == 0.05
    assert worker._train_step_state is None
    assert worker.model.weight.grad is None
    assert not torch.equal(before, worker.model.weight)
    with pytest.raises(RuntimeError, match="successfully completed"):
        worker.finish_train_step()


@pytest.mark.parametrize("expert_parallel_size", [2, 16])
def test_expert_parallel_commit_and_abort(worker, monkeypatch, expert_parallel_size):
    worker.cfg["automodel_cfg"]["expert_parallel_size"] = expert_parallel_size
    worker.moe_mesh = SimpleNamespace(mesh_dim_names=("ep", "ep_shard"))
    clip = Mock(wraps=implementation.scale_grads_and_clip_grad_norm)
    cleanup = Mock()
    monkeypatch.setattr(implementation, "scale_grads_and_clip_grad_norm", clip)
    monkeypatch.setattr(implementation, "release_moe_dispatcher_state", cleanup)
    worker.begin_train_step(Mock())
    dispatch(worker)
    worker.finish_train_step()
    assert clip.call_args.kwargs["moe_mesh"] is worker.moe_mesh
    assert clip.call_args.kwargs["ep_axis_name"] == "ep"
    cleanup.assert_called_once_with(worker.model)
    assert worker.scheduler.last_epoch == 1
    committed = worker.model.weight.detach().clone()
    worker.begin_train_step(Mock())
    dispatch(worker)
    worker.abort_train_step()
    torch.testing.assert_close(worker.model.weight, committed)
    assert worker.model.weight.grad is None
    assert worker.scheduler.last_epoch == 1
    assert cleanup.call_count == 2


@pytest.mark.parametrize("backward", [False, True])
def test_abort_restores_buffers_and_does_not_update(worker, backward):
    before = worker.model.weight.detach().clone()
    worker.begin_train_step(Mock())
    if backward:
        dispatch(worker)
    worker.model.counter.add_(1)
    worker.abort_train_step()
    worker.abort_train_step()
    torch.testing.assert_close(before, worker.model.weight)
    assert worker.model.counter == 0
    assert worker.model.weight.grad is None
    assert not worker.optimizer.state
    assert worker.scheduler.last_epoch == 0
    worker.begin_train_step(Mock())
    worker.abort_train_step()


def test_moe_abort_and_gate_update_once(worker):
    gate = object.__new__(Gate)
    torch.nn.Module.__init__(gate)
    gate._cumulative_expert_load = torch.tensor([2, 3])
    worker.model.add_module("gate", gate)
    worker.is_moe_model = True
    worker.begin_train_step(Mock())
    gate._cumulative_expert_load.add_(10)
    dispatch(worker)
    worker.abort_train_step()
    torch.testing.assert_close(gate._cumulative_expert_load, torch.tensor([2, 3]))
    calls = []

    def update_bias():
        assert worker.model.training
        assert worker.optimizer.state  # The optimizer committed before the hook.
        calls.append(gate._cumulative_expert_load.clone())
        gate._cumulative_expert_load = None

    worker.model.update_moe_gate_bias = update_bias
    worker.begin_train_step(Mock())
    gate._cumulative_expert_load.add_(1)
    dispatch(worker)
    worker.model.eval()
    worker.finish_train_step()
    assert len(calls) == 1
    torch.testing.assert_close(calls[0], torch.tensor([3, 4]))


@pytest.mark.parametrize(
    "failure", ["clip", "optimizer", "gate_bias", "scheduler", "dispatcher"]
)
def test_failed_commit_requires_recovery(worker, monkeypatch, failure):
    worker.begin_train_step(Mock())
    dispatch(worker)
    failing_call = Mock(side_effect=RuntimeError("commit failed"))
    if failure == "clip":
        monkeypatch.setattr(
            implementation, "scale_grads_and_clip_grad_norm", failing_call
        )
    elif failure == "optimizer":
        monkeypatch.setattr(worker.optimizer, "step", failing_call)
    elif failure == "gate_bias":
        monkeypatch.setattr(worker, "_update_moe_gate_bias_if_supported", failing_call)
    elif failure == "scheduler":
        monkeypatch.setattr(worker.scheduler, "step", failing_call)
    else:
        monkeypatch.setattr(
            implementation, "release_moe_dispatcher_state", failing_call
        )
    with pytest.raises(RuntimeError, match="commit failed"):
        worker.finish_train_step()
    assert worker._train_step_state is None
    assert worker.model.weight.grad is None
    worker.abort_train_step()
    with pytest.raises(RuntimeError, match="recover"):
        worker.begin_train_step(Mock())


def test_cleanup_failure_after_update_requires_recovery(worker, monkeypatch):
    worker.begin_train_step(Mock())
    dispatch(worker)
    monkeypatch.setattr(
        worker.optimizer, "zero_grad", Mock(side_effect=RuntimeError("cleanup failed"))
    )
    with pytest.raises(RuntimeError, match="cleanup failed"):
        worker.finish_train_step()
    assert worker.scheduler.last_epoch == 1
    assert worker._train_step_state is None
    worker.abort_train_step()
    with pytest.raises(RuntimeError, match="recover"):
        worker.begin_train_step(Mock())


@pytest.mark.parametrize("gbs,mbs", [(0, 1), (4, 0), (3, 1), (4, 3)])
def test_invalid_sizes(worker, gbs, mbs):
    with pytest.raises(ValueError, match="divisible"):
        worker.begin_train_step(Mock(), gbs=gbs, mbs=mbs)
    assert worker._train_step_state is None


@pytest.mark.parametrize(
    "flag", ["cpu_offload", "is_vlm", "lora_enabled", "enable_seq_packing"]
)
def test_unsupported_worker_modes(worker, flag):
    setattr(worker, flag, True)
    with pytest.raises(ValueError, match="AutoModel SC"):
        worker.begin_train_step(Mock())


def test_dispatch_requires_begin(worker):
    with pytest.raises(RuntimeError, match="begin_train_step"):
        dispatch(worker)


@pytest.mark.parametrize("minimum,maximum", [(1, 2), (2, 3), (0, 0)])
def test_rejects_incomplete_or_unequal_dp_batch(worker, monkeypatch, minimum, maximum):
    def reduce_sizes(sizes, **kwargs):
        sizes.copy_(torch.tensor([minimum, -maximum]))

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce_sizes)
    worker.begin_train_step(Mock())
    with pytest.raises(ValueError, match="every DP rank"):
        dispatch(worker)
    with pytest.raises(RuntimeError, match="successfully completed"):
        worker.finish_train_step()
    worker.abort_train_step()


def test_failed_backward_can_abort_and_start_again(worker, monkeypatch):
    original = implementation.automodel_forward_backward

    def fail_after_backward(**kwargs):
        worker.model(torch.ones(1, 2)).sum().backward()
        raise RuntimeError("backward failed")

    monkeypatch.setattr(
        implementation, "automodel_forward_backward", fail_after_backward
    )
    worker.begin_train_step(Mock())
    with pytest.raises(RuntimeError, match="backward failed"):
        dispatch(worker)
    with pytest.raises(RuntimeError, match="failed"):
        dispatch(worker)
    worker.abort_train_step()
    monkeypatch.setattr(implementation, "automodel_forward_backward", original)
    worker.begin_train_step(Mock())
    dispatch(worker)
    torch.testing.assert_close(
        worker.model.weight.grad, torch.ones_like(worker.model.weight)
    )
    worker.finish_train_step()


def partial_batch(worker: Any, *, masked: bool = False) -> None:
    batch = BatchedDataDict(
        {
            "input_ids": torch.ones((1, 3), dtype=torch.int64),
            "sample_mask": torch.tensor([0.0 if masked else 1.0]),
            "token_mask": torch.tensor([[0.0, 1.0, 1.0]]),
        }
    )
    batch.to = lambda *args: batch
    worker.train_microbatch(batch)


@pytest.mark.parametrize("token_level", [True, False])
def test_streaming_retains_gradients_until_one_update(
    worker: Any, monkeypatch: pytest.MonkeyPatch, token_level: bool
) -> None:
    worker.begin_train_step(
        ClippedPGLossFn(ClippedPGLossConfig(token_level_loss=token_level))
    )
    before = worker.model.weight.detach().clone()
    partial_batch(worker)
    first_grad = worker.model.weight.grad.clone()
    partial_batch(worker)
    torch.testing.assert_close(worker.model.weight.grad, first_grad * 2)
    torch.testing.assert_close(worker.model.weight, before)
    assert not worker.optimizer.state
    assert worker.scheduler.last_epoch == 0
    seen = []
    original_clip = implementation.scale_grads_and_clip_grad_norm

    def capture(*args, **kwargs):
        seen.append(worker.model.weight.grad.clone())
        return original_clip(*args, **kwargs)

    monkeypatch.setattr(implementation, "scale_grads_and_clip_grad_norm", capture)
    worker.finish_train_step()
    torch.testing.assert_close(seen[0], first_grad * 2 / (4 if token_level else 2))
    assert worker.scheduler.last_epoch == 1
    assert worker.model.weight.grad is None


def test_streaming_zero_tokens_requires_abort(worker: Any) -> None:
    worker.begin_train_step(ClippedPGLossFn(ClippedPGLossConfig()))
    partial_batch(worker, masked=True)
    with pytest.raises(ValueError, match="no valid response"):
        worker.finish_train_step()
    assert not worker.optimizer.state
    assert worker.scheduler.last_epoch == 0
    with pytest.raises(RuntimeError, match="failed"):
        partial_batch(worker)
    worker.abort_train_step()
    assert worker.model.weight.grad is None


def test_streaming_failure_after_successful_chunk_cannot_commit(
    worker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker.cfg["train_global_batch_size"] = 6
    worker.begin_train_step(ClippedPGLossFn(ClippedPGLossConfig()))
    partial_batch(worker)
    monkeypatch.setattr(
        implementation,
        "automodel_forward_backward",
        Mock(side_effect=RuntimeError("chunk failed")),
    )
    with pytest.raises(RuntimeError, match="chunk failed"):
        partial_batch(worker)
    with pytest.raises(RuntimeError, match="no failure"):
        worker.finish_train_step()
    worker.abort_train_step()
    assert not worker.optimizer.state
    assert worker.model.weight.grad is None


def test_streaming_rejects_aux_before_forward(
    worker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = object.__new__(Gate)
    torch.nn.Module.__init__(gate)
    gate._cumulative_expert_load = None
    gate.aux_loss_coeff = 0.01
    worker.model.add_module("gate", gate)
    forward = Mock()
    monkeypatch.setattr(implementation, "automodel_forward_backward", forward)
    worker.begin_train_step(ClippedPGLossFn(ClippedPGLossConfig()))
    with pytest.raises(ValueError, match="auxiliary loss"):
        partial_batch(worker)
    forward.assert_not_called()
    worker.abort_train_step()


def test_streaming_abort_restores_gate_diagnostics(worker):
    gate = object.__new__(Gate)
    torch.nn.Module.__init__(gate)
    gate._cumulative_expert_load = torch.tensor([2, 3])
    gate._last_expert_load = torch.tensor([3, 4])
    gate._last_aux_loss = torch.tensor(0.5)
    gate.aux_loss_coeff = 0.0
    worker.model.add_module("gate", gate)
    worker.begin_train_step(ClippedPGLossFn(ClippedPGLossConfig()))
    partial_batch(worker)
    gate._cumulative_expert_load.add_(10)
    gate._last_expert_load = None
    gate._last_aux_loss = None
    worker.abort_train_step()
    torch.testing.assert_close(gate._cumulative_expert_load, torch.tensor([2, 3]))
    torch.testing.assert_close(gate._last_expert_load, torch.tensor([3, 4]))
    torch.testing.assert_close(gate._last_aux_loss, torch.tensor(0.5))


@pytest.mark.parametrize("activation_checkpointing", [False, True])
def test_streaming_real_gate_bias_commits_once_and_abort_restores(
    worker: Any, monkeypatch: pytest.MonkeyPatch, activation_checkpointing: bool
) -> None:
    gate = Gate(
        MoEConfig(
            n_routed_experts=2,
            n_shared_experts=0,
            n_activated_experts=1,
            n_expert_groups=0,
            n_limited_groups=0,
            train_gate=True,
            gate_bias_update_factor=0.01,
            aux_loss_coeff=0.0,
            score_func="sigmoid",
            route_scale=1.0,
            dim=2,
            inter_dim=4,
            moe_inter_dim=4,
            norm_topk_prob=True,
            dtype=torch.float32,
        )
    )
    with torch.no_grad():
        gate.weight.copy_(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
    worker.model.add_module("gate", gate)
    block = torch.nn.Module()
    block.moe = object.__new__(MoE)
    torch.nn.Module.__init__(block.moe)
    block.moe.gate = gate
    update = Mock(wraps=gate.update_bias)
    worker.model.update_moe_gate_bias = update

    def backward(**kwargs: Any) -> list[tuple[torch.Tensor, dict[str, Any]]]:
        def forward(inputs: torch.Tensor) -> torch.Tensor:
            weights, _, _ = gate(inputs, torch.ones(1, dtype=torch.bool), None)
            return worker.model(inputs).sum() + weights.sum()

        inputs = torch.ones(1, 2, requires_grad=True)
        if activation_checkpointing:
            loss = checkpoint(
                forward,
                inputs,
                use_reentrant=False,
                context_fn=_preserve_gate_load_during_recompute(block),
            )
        else:
            loss = forward(inputs)
        loss.backward()
        return [(loss, {"num_valid_samples": 1, "loss": loss.item()})]

    monkeypatch.setattr(implementation, "automodel_forward_backward", backward)
    worker.begin_train_step(ClippedPGLossFn(ClippedPGLossConfig()))
    partial_batch(worker)
    gate.eval()
    with torch.no_grad():
        gate(torch.ones(1, 2), torch.ones(1, dtype=torch.bool), None)
    torch.testing.assert_close(gate._cumulative_expert_load, torch.tensor([1, 0]))
    partial_batch(worker)
    torch.testing.assert_close(gate._cumulative_expert_load, torch.tensor([2, 0]))
    torch.testing.assert_close(gate.e_score_correction_bias, torch.zeros(2))
    worker.finish_train_step()
    update.assert_called_once()
    torch.testing.assert_close(
        gate.e_score_correction_bias, torch.tensor([-0.01, 0.01])
    )
    assert gate._cumulative_expert_load is None
    worker.begin_train_step(ClippedPGLossFn(ClippedPGLossConfig()))
    partial_batch(worker)
    worker.abort_train_step()
    torch.testing.assert_close(
        gate.e_score_correction_bias, torch.tensor([-0.01, 0.01])
    )
    assert gate._cumulative_expert_load is None
    update.assert_called_once()
