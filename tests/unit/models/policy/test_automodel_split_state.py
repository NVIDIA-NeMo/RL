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
from nemo_automodel.components.moe.config import MoEConfig
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
        "automodel_cfg": {},
    }
    worker.dp_size = 2
    worker.cp_size = 1
    worker.is_moe_model = False
    worker.enable_seq_packing = False
    worker.model = torch.nn.Linear(2, 1, bias=False)
    worker.optimizer = torch.optim.SGD(worker.model.parameters(), lr=0.1, momentum=0.9)
    worker.scheduler = torch.optim.lr_scheduler.StepLR(worker.optimizer, 1, gamma=0.5)
    worker.max_grad_norm = 1.0
    worker.device_mesh = worker.moe_mesh = worker.cp_mesh = worker.tp_mesh = None
    worker.pipeline = None
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


def test_expert_parallel_commit_and_abort(worker, monkeypatch):
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
def test_abort_does_not_update(worker, backward):
    before = worker.model.weight.detach().clone()
    worker.begin_train_step(Mock())
    if backward:
        dispatch(worker)
    worker.abort_train_step()
    worker.abort_train_step()
    torch.testing.assert_close(before, worker.model.weight)
    assert worker.model.weight.grad is None
    assert not worker.optimizer.state
    assert worker.scheduler.last_epoch == 0
    worker.begin_train_step(Mock())
    worker.abort_train_step()


def _real_gate() -> Gate:
    config = MoEConfig(
        n_routed_experts=4,
        n_shared_experts=0,
        n_activated_experts=1,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=1e-3,
        aux_loss_coeff=0.0,
        score_func="sigmoid",
        route_scale=1.0,
        dim=2,
        inter_dim=4,
        moe_inter_dim=4,
        norm_topk_prob=False,
        dtype=torch.float32,
    )
    gate = Gate(config)
    with torch.no_grad():
        # Every token routes to expert 0, so the load and the bias update are fixed.
        gate.weight.copy_(
            torch.tensor([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        )
    return gate


def route(gate: Gate, tokens: int) -> None:
    gate(torch.ones(tokens, 2), torch.ones(tokens, dtype=torch.bool), None)


def test_moe_abort_restores_router_load_and_finish_updates_bias_once(worker):
    gate = _real_gate()
    worker.model.add_module("gate", gate)
    worker.model.update_moe_gate_bias = gate.update_bias
    worker.is_moe_model = True
    worker.model.train()

    # No load at begin: abort drops the load the step's forwards accumulated.
    worker.begin_train_step(Mock())
    assert gate._cumulative_expert_load is None
    route(gate, 3)
    assert gate._cumulative_expert_load is not None
    dispatch(worker)
    worker.abort_train_step()
    assert gate._cumulative_expert_load is None
    torch.testing.assert_close(gate.e_score_correction_bias, torch.zeros(4))

    # Finish sums the step's forwards and applies one sign(avg - load) update.
    worker.begin_train_step(Mock())
    route(gate, 3)
    route(gate, 5)
    torch.testing.assert_close(gate._cumulative_expert_load, torch.tensor([8, 0, 0, 0]))
    dispatch(worker)
    worker.model.eval()
    worker.finish_train_step()
    assert worker.optimizer.state  # The optimizer committed before the hook.
    assert gate._cumulative_expert_load is None
    committed = torch.tensor([-1.0, 1.0, 1.0, 1.0]) * 1e-3
    torch.testing.assert_close(gate.e_score_correction_bias, committed)

    # A load present at begin, and the committed bias, both survive an abort.
    route(gate, 2)
    worker.begin_train_step(Mock())
    route(gate, 4)
    dispatch(worker)
    worker.abort_train_step()
    torch.testing.assert_close(gate._cumulative_expert_load, torch.tensor([2, 0, 0, 0]))
    torch.testing.assert_close(gate.e_score_correction_bias, committed)


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
    with pytest.raises(RuntimeError, match="restart the run"):
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
    with pytest.raises(RuntimeError, match="restart the run"):
        worker.begin_train_step(Mock())


@pytest.mark.parametrize("gbs,mbs", [(0, 1), (4, 0), (3, 1), (4, 3)])
def test_invalid_sizes(worker, gbs, mbs):
    with pytest.raises(ValueError, match="divisible"):
        worker.begin_train_step(Mock(), gbs=gbs, mbs=mbs)
    assert worker._train_step_state is None


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
