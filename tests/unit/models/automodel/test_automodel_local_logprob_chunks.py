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

"""Qualify local-vocabulary loss chunking through the AutoModel loss boundary."""

import os
from typing import Any, Optional

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

try:
    import nemo_automodel  # noqa: F401
except ImportError:
    pytest.skip("nemo_automodel not available", allow_module_level=True)

import nemo_rl.distributed.model_utils as model_utils
from nemo_rl.algorithms.logits_sampling_utils import TrainingSamplingParams
from nemo_rl.algorithms.loss.interfaces import LossInputType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.automodel.train import LossPostProcessor

BATCH, SEQ, VOCAB = 2, 37, 257


class WeightedTokenLoss:
    """Expose token logprobs and signed, nonuniform gradients for parity checks."""

    input_type = LossInputType.LOGPROB

    def __call__(
        self, *, data: Any, next_token_logprobs: torch.Tensor, **kwargs: Any
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        return (
            (next_token_logprobs * data["weights"]).sum(),
            {"selected": next_token_logprobs.detach()},
        )


@pytest.fixture(scope="module")
def singleton_tp_mesh() -> DeviceMesh:
    """A real singleton TP mesh, matching each FSDP rank's TP=1 mesh."""
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29519")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(backend="nccl", rank=0, world_size=1)
    return DeviceMesh("cuda", [0], mesh_dim_names=("tp",))


def _run(
    tp_mesh: DeviceMesh,
    chunk_size: Optional[int],
    dtype: torch.dtype = torch.float32,
    sampling_params: Optional[TrainingSamplingParams] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drive one forward/backward through the loss boundary; return loss/logprobs/grad."""
    torch.manual_seed(42)
    logits = torch.randn(BATCH, SEQ, VOCAB, device="cuda", dtype=dtype).requires_grad_()
    data = BatchedDataDict(
        {
            "input_ids": torch.randint(VOCAB, (BATCH, SEQ), device="cuda"),
            "weights": torch.randn(BATCH, SEQ - 1, device="cuda"),
            # Only read by the top-k/top-p branch of prepare_loss_input.
            "token_mask": torch.ones(BATCH, SEQ, device="cuda"),
            "sample_mask": torch.ones(BATCH, device="cuda"),
        }
    )
    processor = LossPostProcessor(
        loss_fn=WeightedTokenLoss(),
        cfg={"logprob_chunk_size": chunk_size},
        cp_mesh=None,
        cp_size=1,
        dp_size=1,
        sampling_params=sampling_params,
        tp_mesh=tp_mesh,
    )
    loss, metrics = processor(
        logits,
        data,
        None,
        torch.tensor(BATCH, device="cuda"),
        torch.tensor(BATCH * (SEQ - 1), device="cuda"),
        cp_sharder=None,
    )
    loss.backward()
    return loss.detach(), metrics["selected"], logits.grad


@pytest.mark.automodel
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("chunk_size", [1, 7, 37])
def test_chunked_local_loss_preserves_selected_values_and_gradients(
    singleton_tp_mesh: DeviceMesh, dtype: torch.dtype, chunk_size: int
) -> None:
    """Cover ragged final chunks, the exact-fit boundary, signed weights, and both dtypes."""
    unchunked = _run(singleton_tp_mesh, chunk_size=None, dtype=dtype)
    chunked = _run(singleton_tp_mesh, chunk_size=chunk_size, dtype=dtype)
    torch.testing.assert_close(chunked[0], unchunked[0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(chunked[1], unchunked[1], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        chunked[2],
        unchunked[2],
        rtol=0.01 if dtype == torch.bfloat16 else 1e-5,
        atol=5e-5 if dtype == torch.bfloat16 else 1e-7,
    )


@pytest.mark.automodel
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_loss_chunking_splits_the_sequence_into_chunks(
    singleton_tp_mesh: DeviceMesh, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The training loss must actually chunk, not silently fall back to one pass."""
    seen: list[int] = []
    original = model_utils._compute_distributed_selected_logprobs

    def spy(logits: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        seen.append(int(logits.shape[1]))
        return original(logits, **kwargs)

    monkeypatch.setattr(model_utils, "_compute_distributed_selected_logprobs", spy)
    _run(singleton_tp_mesh, chunk_size=7)
    assert seen == [7, 7, 7, 7, 7, 2], seen


@pytest.mark.automodel
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_top_k_filtering_keeps_the_local_unchunked_path(
    singleton_tp_mesh: DeviceMesh, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sampling cannot chunk, so it must not be routed through the vocab-parallel kernel."""
    calls: list[int] = []
    original = model_utils.from_parallel_logits_to_logprobs

    def spy(*args: Any, **kwargs: Any) -> torch.Tensor:
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(model_utils, "from_parallel_logits_to_logprobs", spy)
    _run(
        singleton_tp_mesh,
        chunk_size=7,
        sampling_params=TrainingSamplingParams(top_k=5),
    )
    assert calls == []


@pytest.mark.automodel
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("chunk_size", [0, -1])
def test_nonpositive_chunk_size_is_rejected(
    singleton_tp_mesh: DeviceMesh, chunk_size: int
) -> None:
    """A chunk size straight from YAML is unvalidated upstream, so reject it here."""
    with pytest.raises(ValueError, match="logprob_chunk_size must be positive"):
        _run(singleton_tp_mesh, chunk_size=chunk_size)
