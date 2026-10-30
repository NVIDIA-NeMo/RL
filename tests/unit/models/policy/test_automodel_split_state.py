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
from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("nemo_automodel")
from nemo_automodel.components.moe.layers import Gate

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


def test_failed_commit_requires_recovery(worker, monkeypatch):
    worker.begin_train_step(Mock())
    dispatch(worker)
    monkeypatch.setattr(
        worker.optimizer, "step", Mock(side_effect=RuntimeError("optimizer failed"))
    )
    with pytest.raises(RuntimeError, match="optimizer failed"):
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
    with pytest.raises(RuntimeError, match="exactly one"):
        dispatch(worker)
    worker.abort_train_step()
    monkeypatch.setattr(implementation, "automodel_forward_backward", original)
    worker.begin_train_step(Mock())
    dispatch(worker)
    torch.testing.assert_close(
        worker.model.weight.grad, torch.ones_like(worker.model.weight)
    )
    worker.finish_train_step()
