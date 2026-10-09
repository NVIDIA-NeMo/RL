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

"""CPU contract and gradient tests for the actual value split methods.

Load the worker methods without importing the CUDA-only Megatron stack. A small
PyTorch model substitutes for the pipeline schedule; the real value loss and
value-output alignment run unchanged. Distributed GPU parity is a separate test.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from nemo_rl.algorithms.loss.interfaces import LossType, MetricNormalizer
from nemo_rl.algorithms.loss.loss_functions import MseValueLossConfig, MseValueLossFn
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


@pytest.fixture
def worker_type(monkeypatch):
    source = (
        Path(__file__).parents[4]
        / "nemo_rl/models/value/workers/megatron_value_worker.py"
    )
    tree = ast.parse(source.read_text())
    names = {
        "begin_train_step",
        "train_microbatch",
        "finish_train_step",
        "abort_train_step",
        "_assert_train_step_open",
        "_restore_train_hooks",
    }
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in {
            "_ValueTrainStep",
            "_value_loss_prepare_fn",
        }:
            nodes.append(node)
        if isinstance(node, ast.ClassDef) and node.name == "MegatronValueWorkerImpl":
            node.bases = []
            node.body = [
                method
                for method in node.body
                if isinstance(method, ast.FunctionDef) and method.name in names
            ]
            for method in node.body:
                method.decorator_list = []
            nodes.append(node)
    # Keep CUDA device selection out of this CPU-only harness.
    tensor, zeros = torch.tensor, torch.zeros
    monkeypatch.setattr(
        torch,
        "tensor",
        lambda *a, **kw: tensor(*a, **{k: v for k, v in kw.items() if k != "device"}),
    )
    monkeypatch.setattr(
        torch,
        "zeros",
        lambda *a, **kw: zeros(*a, **{k: v for k, v in kw.items() if k != "device"}),
    )
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *a, **kw: None)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    namespace = {
        "__name__": __name__,
        "torch": torch,
        "dataclass": dataclass,
        "field": field,
        "nullcontext": nullcontext,
        "defaultdict": defaultdict,
        "LossType": LossType,
        "MetricNormalizer": MetricNormalizer,
        "parallel_state": SimpleNamespace(get_data_parallel_group=lambda: None),
        "get_pg_collection": lambda model: SimpleNamespace(mp=None),
        "reduce_max_stat_across_model_parallel_group": lambda x, **kw: x,
        "is_pipeline_last_stage": lambda **kw: True,
        "broadcast_loss_metrics_from_last_stage": lambda x: x,
        "get_microbatch_iterator": lambda data, cfg, mbs, **kw: (
            iter([data]),
            1,
            mbs,
            4,
            4,
        ),
        "LossPostProcessor": lambda **kw: SimpleNamespace(**kw),
        "get_rerun_state_machine": lambda: SimpleNamespace(
            should_run_forward_backward=MagicMock(side_effect=[True, False])
        ),
    }
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)

    def forward_backward(**kw):
        model = kw["model"]
        assert model.sync_depth == 1
        assert model.config.finalize_model_grads_func is None
        data = next(kw["data_iterator"])
        logits, data = namespace["_value_loss_prepare_fn"](model(data), data)
        loss, metrics = kw["post_processing_fn"].loss_fn(
            **logits,
            data=data,
            global_valid_seqs=kw["global_valid_seqs"],
            global_valid_toks=kw["global_valid_toks"],
        )
        loss.backward()
        return [metrics]

    namespace["megatron_forward_backward"] = forward_backward
    return namespace["MegatronValueWorkerImpl"], namespace


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.4))
        self.sync_depth = 0
        self.finalize = MagicMock()
        self.config = SimpleNamespace(
            grad_sync_func="sync",
            no_sync_func=nullcontext,
            finalize_model_grads_func=self.finalize,
            num_moe_experts=None,
        )
        self.start_grad_sync = MagicMock()

    def forward(self, data):
        return self.weight * data["input_ids"].float()

    def zero_grad_buffer(self):
        self.zero_grad()

    def scale_gradients(self, scale):
        self.weight.grad.mul_(scale)

    def no_sync(self):
        model = self

        class Context:
            def __enter__(self):
                model.sync_depth += 1

            def __exit__(self, *args):
                model.sync_depth -= 1

        return Context()


class Optimizer:
    def __init__(self, model):
        self.inner = torch.optim.SGD(model.parameters(), lr=0.05)
        self.param_groups = self.inner.param_groups
        self.steps = 0
        self.model = model

    def zero_grad(self):
        self.inner.zero_grad()

    def step(self):
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.7).item()
        self.inner.step()
        self.steps += 1
        return True, norm, 0


def make_worker(cls):
    w = cls()
    w._train_step_state = None
    w.model = Model()
    w.optimizer = Optimizer(w.model)
    w.scheduler = MagicMock()
    w.scheduler.get_lr.return_value = 0.05
    w.scheduler.get_wd.return_value = 0.0
    w.cfg = {
        "train_global_batch_size": 4,
        "train_micro_batch_size": 1,
        "megatron_cfg": {
            "empty_unused_memory_level": 0,
            "distributed_data_parallel_config": {"overlap_grad_reduce": True},
        },
    }
    w.dp_size = 1
    w._policy_like_cfg = {}
    w.mcore_state = SimpleNamespace(straggler_timer=None)
    w.defer_fp32_logits = False
    return w


def batch():
    return BatchedDataDict(
        {
            "input_ids": torch.tensor(
                [[1, 2, 3, 4], [2, 1, 2, 3], [3, 4, 2, 1], [1, 3, 2, 4]]
            ),
            "token_mask": torch.tensor(
                [[0.0, 1, 1, 1], [0, 0, 1, 0], [0, 1, 1, 0], [0, 1, 1, 1]]
            ),
            "sample_mask": torch.tensor([1.0, 1, 1, 0]),
            "returns": torch.ones(4, 4),
            "values": torch.zeros(4, 4),
        }
    )


@pytest.mark.parametrize("cliprange", [None, 0.2])
@pytest.mark.parametrize("splits", [[4], [1, 3], [1, 1, 2]])
def test_split_matches_full_batch_gradients_and_updates(worker_type, cliprange, splits):
    cls, namespace = worker_type
    worker = make_worker(cls)
    reference = Model()
    optimizer = Optimizer(reference)
    loss_fn = MseValueLossFn(MseValueLossConfig(cliprange=cliprange))
    data = batch()
    for _ in range(2):
        optimizer.zero_grad()
        logits, _ = namespace["_value_loss_prepare_fn"](reference(data), data)
        loss, expected_metrics = loss_fn(
            **logits,
            data=data,
            global_valid_seqs=data["sample_mask"].sum(),
            global_valid_toks=(
                data["token_mask"][:, 1:] * data["sample_mask"].unsqueeze(-1)
            ).sum(),
        )
        loss.backward()
        _, expected_norm, _ = optimizer.step()
        worker.begin_train_step(loss_fn)
        offset = 0
        for size in splits:
            chunk = BatchedDataDict(
                {k: v[offset : offset + size] for k, v in data.items()}
            )
            worker.train_microbatch(chunk)
            offset += size
        result = worker.finish_train_step()
        torch.testing.assert_close(worker.model.weight, reference.weight)
        assert result["global_loss"].item() == pytest.approx(loss.item(), rel=1e-6)
        assert result["grad_norm"].item() == pytest.approx(
            expected_norm, rel=1e-6, abs=1e-7
        )
        for key in ["values_mean", "returns_mean", "vf_clipfrac", "num_valid_samples"]:
            assert sum(result["all_mb_metrics"][key]) == pytest.approx(
                expected_metrics[key], rel=1e-6
            )
    assert worker.optimizer.steps == 2
    assert worker.model.finalize.call_count == 2
    assert worker.model.start_grad_sync.call_count == 2
    assert worker.scheduler.step.call_count == 2
    assert worker.model.config.grad_sync_func == "sync"


def test_abort_and_invalid_transitions(worker_type):
    cls, _ = worker_type
    worker = make_worker(cls)
    loss_fn = MseValueLossFn(MseValueLossConfig())
    with pytest.raises(RuntimeError, match="no value train step"):
        worker.finish_train_step()
    worker.begin_train_step(loss_fn)
    with pytest.raises(RuntimeError, match="already open"):
        worker.begin_train_step(loss_fn)
    with pytest.raises(RuntimeError, match="empty"):
        worker.finish_train_step()
    with pytest.raises(RuntimeError, match="failed"):
        worker.train_microbatch(batch())
    worker.abort_train_step()
    worker.abort_train_step()
    worker.begin_train_step(loss_fn)
    worker.train_microbatch(batch())
    worker.abort_train_step()
    assert worker.model.weight.item() == pytest.approx(0.4)
    assert worker.optimizer.steps == 0
    assert worker.model.weight.grad is None
    worker.begin_train_step(loss_fn)
    worker.train_microbatch(batch())
    worker.finish_train_step()
    assert worker.optimizer.steps == 1


def test_incomplete_batch_cannot_commit(worker_type):
    cls, _ = worker_type
    worker = make_worker(cls)
    worker.begin_train_step(MseValueLossFn(MseValueLossConfig()), gbs=8)
    worker.train_microbatch(batch())
    with pytest.raises(ValueError, match="expected 8"):
        worker.finish_train_step()
    assert worker.optimizer.steps == 0
    worker.abort_train_step()


def test_backward_error_restores_hooks_and_requires_abort(worker_type):
    cls, namespace = worker_type
    worker = make_worker(cls)
    worker.begin_train_step(MseValueLossFn(MseValueLossConfig()))

    def fail(**kwargs):
        raise RuntimeError("backward failed")

    namespace["megatron_forward_backward"] = fail
    with pytest.raises(RuntimeError, match="backward failed"):
        worker.train_microbatch(batch())
    assert worker.model.config.grad_sync_func == "sync"
    assert worker.model.config.finalize_model_grads_func is worker.model.finalize
    with pytest.raises(RuntimeError, match="failed"):
        worker.finish_train_step()
    worker.abort_train_step()
    assert worker.optimizer.steps == 0


def test_all_masked_step_does_not_update(worker_type):
    cls, _ = worker_type
    worker = make_worker(cls)
    data = batch()
    data["sample_mask"].zero_()
    worker.begin_train_step(MseValueLossFn(MseValueLossConfig()))
    worker.train_microbatch(data)
    with pytest.raises(ValueError, match="no valid"):
        worker.finish_train_step()
    assert worker.optimizer.steps == 0
    worker.abort_train_step()
