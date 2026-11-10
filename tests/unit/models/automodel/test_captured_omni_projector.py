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

"""Real FSDP projector ordering and gradient behavior for mixed captured rows."""

import copy
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

pytest.importorskip("nemo_automodel")
from nemo_automodel.components.models.nemotron_omni.model import VisionProjector
from nemo_automodel.shared.torch_patches import (
    patch_fsdp_accumulated_grad_guard,
    patch_fsdp_uniform_reduce_dtype,
)

from nemo_rl.models.automodel.train import _captured_omni_projector_context


class _ToyLanguage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 16)

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding


class _ToyNativeOmni(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(architectures=["NemotronH_Omni_Reasoning_V3"])
        self.language_model = _ToyLanguage()
        self.vision_projector = VisionProjector(4, 16, 16)
        self.head = nn.Linear(16, 8, bias=False)

    def forward(
        self, input_ids: torch.Tensor, pixel_values: torch.Tensor | None = None
    ) -> torch.Tensor:
        output = self.language_model.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            output = output + self.vision_projector(pixel_values)
        return self.head(output)


class _ToyNativeVisionOmni(_ToyNativeOmni):
    def __init__(self) -> None:
        super().__init__()
        self.vision_model = nn.Linear(16, 16, bias=False)
        self.force_image_size = 4

    def extract_feature_dynamic(self, pixels, sizes):
        features = [self.vision_model(image.reshape(1, -1, 16)) for image in pixels]
        return self.vision_projector(torch.cat(features, dim=1))

    def forward(self, input_ids, pixel_values=None):
        output = self.language_model.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            output = output + self.extract_feature_dynamic(pixel_values, None).mean(
                1, keepdim=True
            )
        return self.head(output)


def _check_mixed_projector(
    rank: int, dtype: torch.dtype, *, trainable_vision: bool = False
) -> None:
    mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("dp",))
    patch_fsdp_accumulated_grad_guard()
    patch_fsdp_uniform_reduce_dtype()
    torch.manual_seed(7)
    model = (_ToyNativeVisionOmni() if trainable_vision else _ToyNativeOmni()).to(
        device="cuda", dtype=dtype
    )
    reference = copy.deepcopy(model)
    tokens = torch.tensor([[1, 2]], device="cuda")
    pixels = torch.randn(1, 2, 16, device="cuda", dtype=dtype)
    with torch.no_grad():
        reference(tokens, pixels)
    torch.cuda.synchronize()
    policy = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=torch.float32)
    if trainable_vision:
        fully_shard(
            model.vision_model, mesh=mesh, mp_policy=policy, reshard_after_forward=True
        )
    for module in [
        model.language_model.embedding,
        model.vision_projector,
        model.head,
        model,
    ]:
        fully_shard(module, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
    actual_optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.1)
    reference_optimizer = torch.optim.SGD(
        reference.parameters(), lr=0.01, weight_decay=0.1
    )
    image_counts = [(2, 0), (1, 2), (0, 0)] if trainable_vision else [(1, 0), (0, 0)]
    for counts in image_counts:
        actual_optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        batch = {"input_ids": tokens}
        if counts[rank]:
            batch["pixel_values"] = pixels.expand(counts[rank], -1, -1)
        expected = reference(**batch)
        expected.sum().backward()
        for parameter in reference.parameters():
            present = torch.tensor(int(parameter.grad is not None), device="cuda")
            dist.all_reduce(present, op=dist.ReduceOp.MAX)
            if present.item():
                gradient = (
                    parameter.grad
                    if parameter.grad is not None
                    else torch.zeros_like(parameter)
                )
                gradient = gradient.float()
                dist.all_reduce(gradient)
                parameter.grad = (gradient / 2).to(dtype)
        original_hook_count = len(model.language_model.embedding._forward_hooks)
        with _captured_omni_projector_context(model, batch, mesh):
            actual = model(**batch)
            torch.testing.assert_close(
                actual,
                expected,
                rtol=0.02 if dtype == torch.bfloat16 else 1e-5,
                atol=0.003 if dtype == torch.bfloat16 else 1e-5,
            )
            actual.sum().backward()
        assert len(model.language_model.embedding._forward_hooks) == original_hook_count
        for (name, actual_parameter), (reference_name, expected_parameter) in zip(
            model.named_parameters(), reference.named_parameters(), strict=True
        ):
            assert name == reference_name
            if expected_parameter.grad is None:
                assert actual_parameter.grad is None, name
            else:
                assert actual_parameter.grad is not None, name
                torch.testing.assert_close(
                    actual_parameter.grad.full_tensor(),
                    expected_parameter.grad,
                    rtol=0.03 if dtype == torch.bfloat16 else 2e-5,
                    atol=0.003 if dtype == torch.bfloat16 else 2e-5,
                )
        prior_projector = {
            n: p.full_tensor().clone()
            for n, p in model.vision_projector.named_parameters()
        }
        actual_optimizer.step()
        reference_optimizer.step()
        for (name, actual_parameter), (reference_name, expected_parameter) in zip(
            model.named_parameters(), reference.named_parameters(), strict=True
        ):
            torch.testing.assert_close(
                actual_parameter.full_tensor(),
                expected_parameter,
                rtol=0.03 if dtype == torch.bfloat16 else 2e-5,
                atol=0.003 if dtype == torch.bfloat16 else 2e-5,
            )
        if not any(counts):
            for name, parameter in model.vision_projector.named_parameters():
                assert parameter.grad is None
                assert torch.equal(parameter.full_tensor(), prior_projector[name]), name
    del actual_optimizer, reference_optimizer, model, reference
    torch.cuda.empty_cache()


def _projector_worker(rank: int, rendezvous: str, dtype: torch.dtype) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=2,
        init_method="file://" + rendezvous,
        timeout=timedelta(seconds=120),
    )
    try:
        _check_mixed_projector(rank, dtype)
    finally:
        dist.destroy_process_group()


@pytest.mark.automodel
@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA devices required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mixed_projector_matches_outputs_gradients_and_all_text_updates(
    tmp_path: Path, dtype: torch.dtype
) -> None:
    mp.spawn(
        _projector_worker,
        args=(str(tmp_path / "rendezvous"), dtype),
        nprocs=2,
        join=True,
    )


def _vision_worker(rank: int, rendezvous: str, dtype: torch.dtype) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=2,
        init_method="file://" + rendezvous,
        timeout=timedelta(seconds=90),
    )
    try:
        _check_mixed_projector(rank, dtype, trainable_vision=True)
    finally:
        dist.destroy_process_group()


@pytest.mark.automodel
@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA devices required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_trainable_vision_uneven_images_preserves_gradients_and_all_text_updates(
    tmp_path: Path, dtype: torch.dtype
) -> None:
    mp.spawn(
        _vision_worker,
        args=(str(tmp_path / "vision-rendezvous"), dtype),
        nprocs=2,
        join=True,
    )
