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
"""Real-collective normalization tests for split Megatron draft training."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("megatron.bridge")

from nemo_rl.algorithms.loss.draft import DraftLossStats  # noqa: E402
from nemo_rl.algorithms.loss.interfaces import LossType  # noqa: E402
from nemo_rl.models.megatron.draft.step_state import (  # noqa: E402
    DRAFT_STEP_PAYLOAD_KEY,
    DraftStepState,
)
from tests.unit.models.policy.test_megatron_split_state import (  # noqa: E402
    WORKER_MOD,
    _make_worker,
)

pytestmark = pytest.mark.mcore


def _run_context_parallel_finish_parity(rank: int, world_size: int) -> None:
    singleton_groups = [
        torch.distributed.new_group(ranks=[group_rank])
        for group_rank in range(world_size)
    ]
    dp_group = singleton_groups[rank]
    cp_group = torch.distributed.group.WORLD
    device = torch.device("cuda")

    local_numerators, local_counts = {
        1: ([40.0], [10.0]),
        2: ([0.0, 40.0], [0.0, 10.0]),
        4: ([0.0, 10.0, 0.0, 30.0], [0.0, 2.0, 0.0, 8.0]),
    }[world_size]
    local_numerator = local_numerators[rank]
    local_count = local_counts[rank]

    draft_step_state = DraftStepState()
    payload = DraftStepState.metric_payload(
        DraftLossStats(
            numerators=torch.tensor([local_numerator], device=device),
            counts=torch.tensor([local_count], device=device),
            weights=torch.ones(1, device=device),
        )
    )
    draft_step_state.accumulate(payload)

    worker = _make_worker(LossType.TOKEN_LEVEL)
    worker.rank = rank
    worker.cfg["draft"] = SimpleNamespace(enabled=True, loss_weight=1.0)
    draft_param = torch.nn.Parameter(torch.tensor(1.0, device=device))
    draft_param.grad_norm_group = "draft"
    # MCore's CP loss scaling supplies 1/CP before this finish-time correction.
    draft_param.main_grad = torch.tensor(3.0 / world_size, device=device)
    worker.model.parameters.side_effect = lambda: iter([draft_param])
    worker.optimizer.grad_norms_by_group = {"draft": 0.25}

    # Start from the production initializer so keys added to the split-step
    # state (e.g. MTP flags) stay in sync with finish_train_step.
    state = worker._split_step_state_init(
        loss_fn=SimpleNamespace(loss_type=LossType.TOKEN_LEVEL), gbs=32, mbs=None
    )
    state.update(
        {
            "draft_step_state": draft_step_state,
            "local_valid_seqs": torch.tensor(8.0, device=device),
            "local_valid_toks": torch.tensor(2048.0, device=device),
            "saved_finalize_model_grads_func": lambda _models, _tokens: None,
            "num_chunks": 1,
            "total_num_microbatches": 1,
            "all_mb_metrics": [
                {
                    "loss": 2048.0,
                    "draft_loss": torch.tensor(local_numerator, device=device),
                    DRAFT_STEP_PAYLOAD_KEY: payload,
                }
            ],
            "mb_losses": [2048.0],
        }
    )

    def data_parallel_group(
        *, with_context_parallel: bool = False
    ) -> torch.distributed.ProcessGroup:
        return cp_group if with_context_parallel else dp_group

    def aggregate_training_statistics(
        *,
        all_mb_metrics: list[dict[str, Any]],
        losses: list[float],
        **_kwargs: Any,
    ) -> tuple[dict[str, list[Any]], torch.Tensor]:
        return (
            {
                "loss": [all_mb_metrics[0]["loss"]],
                "draft_loss": [all_mb_metrics[0]["draft_loss"]],
            },
            torch.tensor(sum(losses), device=device),
        )

    with (
        patch(
            f"{WORKER_MOD}.parallel_state.get_data_parallel_group",
            side_effect=data_parallel_group,
        ),
        patch(
            f"{WORKER_MOD}.parallel_state.get_context_parallel_group",
            return_value=cp_group,
        ),
        patch(
            f"{WORKER_MOD}.parallel_state.get_context_parallel_world_size",
            return_value=world_size,
        ),
        patch(
            f"{WORKER_MOD}.get_pg_collection",
            return_value=SimpleNamespace(mp=cp_group),
        ),
        patch(
            f"{WORKER_MOD}.logical_and_across_model_parallel_group",
            side_effect=lambda value, **_kwargs: value,
        ),
        patch(
            f"{WORKER_MOD}.reduce_max_stat_across_model_parallel_group",
            side_effect=lambda value, **_kwargs: value,
        ),
        patch(
            f"{WORKER_MOD}.aggregate_training_statistics",
            side_effect=aggregate_training_statistics,
        ),
    ):
        metrics = worker._finish_train_step_body(state)

    actual_grad = draft_param.main_grad.item()
    actual_metric = metrics["all_mb_metrics"]["draft_loss"][0].item()
    assert actual_grad == pytest.approx(
        614.4, rel=1e-5
    ) and actual_metric == pytest.approx(4.0), (
        f"CP{world_size} normalization mismatch: grad={actual_grad}, metric={actual_metric}"
    )
    assert metrics["all_mb_metrics"]["loss"][0] == pytest.approx(1.0)


@pytest.mark.parametrize("context_parallel_size", [1, 2, 4])
def test_context_parallel_finish_metric_and_draft_grad_parity(
    distributed_test_runner,
    context_parallel_size: int,
) -> None:
    distributed_test_runner(
        _run_context_parallel_finish_parity,
        world_size=context_parallel_size,
    )


def _run_monolithic_context_parallel_draft_denominator(
    rank: int, world_size: int
) -> None:
    """The async/non-TQ train() path must give CP draft grads the CP1 value.

    Each CP rank owns different draft windows (two microbatches here). A draft
    parameter scales every owned numerator, so with a correct denominator the
    CP-summed gradient of ``normalized(counts) / cp`` -- the loss train() feeds
    backward -- equals the CP1 gradient total_numerator / total_count.
    """
    from nemo_rl.models.policy.workers.megatron_policy_worker import (
        MegatronPolicyWorkerImpl,
    )

    singleton_groups = [
        torch.distributed.new_group(ranks=[group_rank])
        for group_rank in range(world_size)
    ]
    dp_group = singleton_groups[rank]
    cp_group = torch.distributed.group.WORLD
    device = torch.device("cuda")

    # Owned (numerator, count) per microbatch for each CP rank; two slot bins.
    owned = {
        1: [[([30.0, 10.0], [6.0, 4.0]), ([20.0, 0.0], [5.0, 0.0])]],
        2: [
            [([30.0, 10.0], [6.0, 4.0]), ([0.0, 0.0], [0.0, 0.0])],
            [([20.0, 0.0], [5.0, 0.0]), ([0.0, 0.0], [0.0, 0.0])],
        ],
        4: [
            [([30.0, 0.0], [6.0, 0.0]), ([0.0, 0.0], [0.0, 0.0])],
            [([0.0, 10.0], [0.0, 4.0]), ([0.0, 0.0], [0.0, 0.0])],
            [([20.0, 0.0], [5.0, 0.0]), ([0.0, 0.0], [0.0, 0.0])],
            [([0.0, 0.0], [0.0, 0.0]), ([0.0, 0.0], [0.0, 0.0])],
        ],
    }[world_size][rank]
    total_numerator = 60.0
    total_count = 15.0

    microbatches = [
        SimpleNamespace(
            data_dict={"index": index},
            draft_sequence_layout=SimpleNamespace(cp_rank=rank),
        )
        for index in range(len(owned))
    ]
    provider = SimpleNamespace(
        normalization_counts=lambda data, *, optimizer_step, sequence_layout: (
            torch.tensor(owned[data["index"]][1], device=device)
        )
    )
    worker = object.__new__(MegatronPolicyWorkerImpl)
    worker.cfg = {}
    worker.mcore_state = SimpleNamespace(straggler_timer=None)
    worker.delegate_pack_to_model = False
    worker.delegate_mtp_loss_mask_to_model = False
    worker.model_slices_context_parallel_inputs = False
    worker.mtp_enabled = False
    worker.create_router_padding_mask = False
    worker.allow_unmasked_chunkwise_cp = False
    worker.scheduler = SimpleNamespace(num_steps=3)

    def data_parallel_group(*, with_context_parallel: bool = False):
        return cp_group if with_context_parallel else dp_group

    with (
        patch(
            f"{WORKER_MOD}.get_microbatch_iterator",
            return_value=(iter(microbatches), len(microbatches), 1, 8, 8),
        ),
        patch(
            f"{WORKER_MOD}.parallel_state.get_data_parallel_group",
            side_effect=data_parallel_group,
        ),
    ):
        counts = worker._context_parallel_draft_normalization_counts(
            object(), draft_provider=provider, mbs=1, cp_size=world_size
        )

    assert counts is not None
    assert counts.sum().item() == pytest.approx(total_count / world_size)

    draft_param = torch.nn.Parameter(torch.tensor(1.0, device=device))
    weights = torch.ones(2, device=device)
    loss = torch.zeros((), device=device)
    for numerators, mb_counts in owned:
        stats = DraftLossStats(
            numerators=torch.tensor(numerators, device=device) * draft_param,
            counts=torch.tensor(mb_counts, device=device),
            weights=weights,
        )
        # train(): wrapper normalizes, then cp_normalize divides by cp_size.
        loss = loss + stats.normalized(normalization_counts=counts) / world_size
    loss.backward()
    grad = draft_param.grad.detach().clone()
    torch.distributed.all_reduce(grad, group=cp_group)
    assert grad.item() == pytest.approx(total_numerator / total_count, rel=1e-5), (
        f"CP{world_size} monolithic draft gradient {grad.item()} != CP1 "
        f"{total_numerator / total_count}"
    )


@pytest.mark.parametrize("context_parallel_size", [1, 2, 4])
def test_monolithic_train_context_parallel_draft_denominator(
    distributed_test_runner,
    context_parallel_size: int,
) -> None:
    distributed_test_runner(
        _run_monolithic_context_parallel_draft_denominator,
        world_size=context_parallel_size,
    )


def test_monolithic_context_parallel_draft_requires_packed_layout() -> None:
    from nemo_rl.models.policy.workers.megatron_policy_worker import (
        MegatronPolicyWorkerImpl,
    )

    worker = object.__new__(MegatronPolicyWorkerImpl)
    worker.cfg = {}
    worker.mcore_state = SimpleNamespace(straggler_timer=None)
    worker.delegate_pack_to_model = False
    worker.delegate_mtp_loss_mask_to_model = False
    worker.model_slices_context_parallel_inputs = False
    worker.mtp_enabled = False
    worker.create_router_padding_mask = False
    worker.allow_unmasked_chunkwise_cp = False
    worker.scheduler = SimpleNamespace(num_steps=0)
    unpacked = SimpleNamespace(data_dict={}, draft_sequence_layout=None)
    with patch(
        f"{WORKER_MOD}.get_microbatch_iterator",
        return_value=(iter([unpacked]), 1, 1, 8, 8),
    ):
        with pytest.raises(RuntimeError, match="packed draft sequence layout"):
            worker._context_parallel_draft_normalization_counts(
                object(), draft_provider=SimpleNamespace(), mbs=1, cp_size=2
            )
