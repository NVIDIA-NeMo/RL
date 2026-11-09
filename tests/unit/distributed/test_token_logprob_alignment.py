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
"""Same-position diffusion and next-token AR scoring share the core kernels."""

import pytest
import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from nemo_rl.algorithms.logits_sampling_utils import TrainingSamplingParams
from nemo_rl.distributed.model_utils import get_next_token_logprobs_from_logits


def reference(logits, targets, shift_labels):
    if shift_labels:
        logits, targets = logits[:, :-1], targets[:, 1:]
    return -torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
    ).reshape(targets.shape)


@pytest.mark.parametrize("shift_labels", [True, False])
@pytest.mark.parametrize("length", [0, 1, 7])
def test_local_values_and_gradients(shift_labels, length):
    torch.manual_seed(18)
    logits = torch.randn(2, length, 12, requires_grad=True)
    targets = torch.randint(12, (2, length))
    actual = get_next_token_logprobs_from_logits(
        targets, logits, shift_labels=shift_labels
    )
    expected = reference(logits, targets, shift_labels)
    torch.testing.assert_close(actual, expected)
    weights = torch.randn_like(actual)
    actual_grad = torch.autograd.grad((actual * weights).sum(), logits)[0]
    expected_grad = torch.autograd.grad((expected * weights).sum(), logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    if shift_labels:
        torch.testing.assert_close(
            get_next_token_logprobs_from_logits(targets, logits), expected
        )


@pytest.mark.parametrize("shift_labels", [True, False])
def test_sampling_filter_preserves_alignment(shift_labels):
    logits = torch.tensor([[[0.0, 2.0, 1.0], [3.0, 0.0, 1.0], [0.0, 1.0, 4.0]]])
    targets = torch.tensor([[1, 0, 2]])
    actual = get_next_token_logprobs_from_logits(
        targets,
        logits.clone(),
        shift_labels=shift_labels,
        sampling_params=TrainingSamplingParams(top_k=1),
    )
    expected = torch.full((1, 2), -torch.inf) if shift_labels else torch.zeros(1, 3)
    torch.testing.assert_close(actual, expected)


class ContiguousCPLayout:
    """Two-rank layout with padding, exercising the sharder API used by AutoModel."""

    def shard_token_tensor(self, tensor, *, seq_dim, fill):
        self.length = tensor.shape[seq_dim]
        tensor = torch.nn.functional.pad(tensor, (0, self.length % 2), value=fill)
        return tensor.chunk(2, dim=seq_dim)[dist.get_rank()].contiguous()

    def gather_token_tensor(self, tensor, *, seq_dim, trim):
        gathered = torch.cat(dist_nn.all_gather(tensor), dim=seq_dim)
        return gathered[:, : self.length] if trim else gathered


def distributed_alignment(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2
    )
    try:
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
        torch.manual_seed(91)
        full = torch.randn(2, 7, 12)
        targets = torch.randint(12, (2, 7))
        for shift_labels in (True, False):
            for backend in ("vocab", "dtensor", "cp"):
                for chunk_size in (None, 2):
                    dense = full.clone().requires_grad_(True)
                    expected = reference(dense, targets, shift_labels)
                    weights = (
                        torch.arange(expected.numel(), dtype=torch.float32).reshape_as(
                            expected
                        )
                        / 10
                    )
                    expected_grad = torch.autograd.grad(
                        (expected * weights).sum(), dense
                    )[0]
                    kwargs = {}
                    if backend == "cp":
                        local = (
                            torch.nn.functional.pad(full, (0, 0, 0, 1))
                            .chunk(2, dim=1)[rank]
                            .clone()
                            .requires_grad_(True)
                        )
                        kwargs["cp_sharder"] = ContiguousCPLayout()
                        model_logits = local
                        expected_local_grad = torch.nn.functional.pad(
                            expected_grad, (0, 0, 0, 1)
                        ).chunk(2, dim=1)[rank]
                    else:
                        local = full.chunk(2, dim=-1)[rank].clone().requires_grad_(True)
                        expected_local_grad = expected_grad.chunk(2, dim=-1)[rank]
                        if backend == "dtensor":
                            model_logits = DTensor.from_local(local, mesh, [Shard(-1)])
                        else:
                            model_logits = local
                            kwargs.update(
                                vocab_parallel_group=dist.group.WORLD,
                                vocab_parallel_rank=rank,
                            )
                    actual = get_next_token_logprobs_from_logits(
                        targets,
                        model_logits,
                        shift_labels=shift_labels,
                        chunk_size=chunk_size,
                        **kwargs,
                    )
                    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
                    loss = (actual * weights).sum()
                    # The CP gather backward sums the replicated losses across ranks.
                    if backend == "cp":
                        loss = loss / 2
                    loss.backward()
                    torch.testing.assert_close(
                        local.grad, expected_local_grad, atol=1e-6, rtol=1e-5
                    )
    finally:
        dist.destroy_process_group()


def test_distributed_values_and_gradients(tmp_path):
    mp.spawn(
        distributed_alignment, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True
    )
