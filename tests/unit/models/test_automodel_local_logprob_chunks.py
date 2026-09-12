"""Qualify local-vocabulary loss chunking through the AutoModel loss boundary."""

from collections.abc import Iterator
from typing import Any

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from nemo_rl.algorithms.loss.interfaces import LossInputType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.automodel.train import LossPostProcessor


class WeightedTokenLoss:
    """Expose token logprobs and signed, nonuniform gradients for parity checks."""

    input_type = LossInputType.LOGPROB

    def __call__(self, *, data: Any, next_token_logprobs: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        return (next_token_logprobs * data['weights']).sum(), {'selected': next_token_logprobs.detach()}


@pytest.fixture(scope='module')
def singleton_tp_mesh(tmp_path_factory: Any) -> Iterator[DeviceMesh]:
    """Use a real singleton NCCL group, matching each FSDP rank's TP=1 mesh."""
    assert torch.cuda.is_available(), 'this qualification requires a GPU'
    store = tmp_path_factory.mktemp('local-logprob') / 'store'
    dist.init_process_group('nccl', init_method=f'file://{store}', rank=0, world_size=1)
    try:
        yield DeviceMesh('cuda', [0], mesh_dim_names=('tp',))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('chunk_size', [1, 7, 64])
def test_chunked_local_loss_preserves_selected_values_and_gradients(singleton_tp_mesh: DeviceMesh, dtype: torch.dtype, chunk_size: int) -> None:
    """Cover ragged final chunks, oversized chunks, signed weights, and both dtypes."""
    torch.manual_seed(42)
    original = torch.randn(2, 37, 257, device='cuda', dtype=dtype)
    data = BatchedDataDict({'input_ids': torch.randint(257, (2, 37), device='cuda'), 'weights': torch.randn(2, 36, device='cuda') / 72})
    results = []
    for selected_chunk in (None, chunk_size):
        logits = original.clone().requires_grad_()
        processor = LossPostProcessor(loss_fn=WeightedTokenLoss(), cfg={'logprob_chunk_size': selected_chunk}, cp_mesh=None, cp_size=1, dp_size=1, tp_mesh=singleton_tp_mesh)
        loss, metrics = processor(logits, data, None, torch.tensor(2, device='cuda'), torch.tensor(72, device='cuda'), cp_sharder=None)
        loss.backward()
        results.append((loss.detach(), metrics['selected'], logits.grad))
    torch.testing.assert_close(results[1][0], results[0][0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(results[1][1], results[0][1], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(results[1][2], results[0][2], rtol=.01 if dtype == torch.bfloat16 else 1e-5, atol=5e-5 if dtype == torch.bfloat16 else 1e-7)
