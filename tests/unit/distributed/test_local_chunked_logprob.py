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
"""CPU-only tests for the full-local-vocabulary (tensor-parallel-size 1) logprob path.

Covers ``LocalChunkedLogprob`` numerics against a plain ``log_softmax`` + gather
reference, the ``_tp_target_logprobs`` fallback chunk size, and the pre-cast
decision in ``get_next_token_logprobs_from_logits``.
"""

import contextlib
import os
import tempfile

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from nemo_rl.algorithms.logits_sampling_utils import (
    TrainingSamplingParams,
    apply_top_k_top_p,
)
from nemo_rl.distributed import model_utils
from nemo_rl.distributed.model_utils import (
    DEFAULT_LOCAL_LOGPROB_CHUNK_SIZE,
    LocalChunkedLogprob,
    _tp_target_logprobs,
    get_next_token_logprobs_from_logits,
)

# Chunking only reassociates the reduction, so the deviation from an unchunked
# log_softmax + gather is pure float32 rounding. Measured at the small shapes
# used below, over 40 seeds x the chunk sizes below: max 9.5e-07 forward and
# 2.4e-07 backward for float32 logits; for bfloat16 logits the gradients happen
# to round identically at this shape, but that is shape-dependent and larger
# vocabularies drift up to ~1.5e-05, which is what BACKWARD_TOL covers.
FORWARD_TOL = 2e-6
BACKWARD_TOL = 2e-5


def _make_inputs(
    batch: int,
    seq_len: int,
    vocab: int,
    dtype: torch.dtype,
    seed: int = 1234,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (logits, target, grad_output) with a reproducible spread of values."""
    generator = torch.Generator().manual_seed(seed)
    logits = (
        torch.randn(batch, seq_len, vocab, generator=generator, dtype=torch.float32)
        * 3.0
    ).to(dtype)
    target = torch.randint(0, vocab, (batch, seq_len), generator=generator)
    grad_output = torch.randn(batch, seq_len, generator=generator, dtype=torch.float32)
    return logits, target, grad_output


def _reference_logprobs(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Unchunked log_softmax + gather, i.e. what the chunked function must match."""
    log_probs = torch.nn.functional.log_softmax(logits.to(torch.float32), dim=-1)
    return log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 7, 8, 64])
def test_forward_matches_log_softmax_gather(dtype, chunk_size):
    """Forward equals the reference for chunk sizes that do not divide the sequence."""
    logits, target, _ = _make_inputs(2, 7, 11, dtype)

    actual = LocalChunkedLogprob.apply(logits, target, chunk_size, False)
    expected = _reference_logprobs(logits, target)

    assert actual.shape == (2, 7)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=FORWARD_TOL, atol=FORWARD_TOL)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("chunk_size", [1, 3, 5, 7, 8, 64])
def test_backward_matches_reference_gradient(dtype, chunk_size):
    """Backward equals autograd through the reference, in the logits' own dtype."""
    logits, target, grad_output = _make_inputs(2, 7, 11, dtype)

    chunked_logits = logits.clone().requires_grad_(True)
    LocalChunkedLogprob.apply(chunked_logits, target, chunk_size, False).backward(
        grad_output
    )

    reference_logits = logits.clone().requires_grad_(True)
    _reference_logprobs(reference_logits, target).backward(grad_output)

    assert chunked_logits.grad is not None
    assert reference_logits.grad is not None
    # The gradient buffer must stay in the logits' dtype: a float32 buffer here
    # is the second ~10 GiB allocation this path exists to avoid.
    assert chunked_logits.grad.dtype == dtype

    torch.testing.assert_close(
        chunked_logits.grad.to(torch.float32),
        reference_logits.grad.to(torch.float32),
        rtol=BACKWARD_TOL,
        atol=BACKWARD_TOL,
    )


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_non_positive_chunk_size_is_rejected(chunk_size):
    """A non-positive chunk would skip the loop and return uninitialized memory."""
    logits, target, _ = _make_inputs(2, 7, 11, torch.float32)

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        LocalChunkedLogprob.apply(logits, target, chunk_size, False)


@pytest.mark.parametrize("chunk_size", [None, 0])
def test_tp_target_logprobs_normalizes_unusable_chunk_size(chunk_size):
    """``None`` and ``0`` both fall back to the default instead of reaching the kernel."""
    logits, target, _ = _make_inputs(2, 7, 11, torch.float32)

    actual = _tp_target_logprobs(
        logits,
        target,
        vocab_start_index=0,
        vocab_end_index=11,
        tp_group=None,
        chunk_size=chunk_size,
    )

    torch.testing.assert_close(
        actual,
        _reference_logprobs(logits, target),
        rtol=FORWARD_TOL,
        atol=FORWARD_TOL,
    )


@pytest.mark.parametrize(
    "sampling_params", [None, TrainingSamplingParams(top_k=3, top_p=1.0)]
)
def test_empty_sequence_returns_empty_logprobs(sampling_params):
    """Both branches handle a zero-length sequence without indexing an empty list."""
    logits = torch.randn(2, 0, 11)
    target = torch.zeros((2, 0), dtype=torch.long)

    actual = _tp_target_logprobs(
        logits,
        target,
        vocab_start_index=0,
        vocab_end_index=11,
        tp_group=None,
        chunk_size=4,
        sampling_params=sampling_params,
    )

    assert actual.shape == (2, 0)
    assert actual.dtype == torch.float32


def test_backward_saves_only_inputs():
    """Nothing of size [B, S, V] is materialized for backward."""
    logits, target, _ = _make_inputs(2, 7, 11, torch.bfloat16)
    logits = logits.requires_grad_(True)

    out = LocalChunkedLogprob.apply(logits, target, 3, False)

    saved = out.grad_fn.saved_tensors
    assert len(saved) == 2
    # The saved logits are the input itself, still in its original dtype.
    assert saved[0].dtype == torch.bfloat16
    assert saved[0].data_ptr() == logits.data_ptr()
    assert saved[1].data_ptr() == target.data_ptr()


def test_inference_only_saves_nothing():
    """``inference_only=True`` keeps the forward result but saves no tensors."""
    logits, target, _ = _make_inputs(2, 7, 11, torch.float32)
    logits = logits.requires_grad_(True)

    out = LocalChunkedLogprob.apply(logits, target, 3, True)

    torch.testing.assert_close(
        out,
        _reference_logprobs(logits.detach(), target),
        rtol=FORWARD_TOL,
        atol=FORWARD_TOL,
    )
    assert out.grad_fn.saved_tensors == ()


def test_no_grad_matches_reference():
    """The inference path used by ``get_logprobs`` produces the same numbers."""
    logits, target, _ = _make_inputs(2, 7, 11, torch.bfloat16)

    with torch.no_grad():
        out = LocalChunkedLogprob.apply(logits, target, 3, True)

    assert not out.requires_grad
    torch.testing.assert_close(
        out,
        _reference_logprobs(logits, target),
        rtol=FORWARD_TOL,
        atol=FORWARD_TOL,
    )


@pytest.mark.parametrize("chunk_size", [None, 4])
def test_tp_target_logprobs_uses_chunked_function_without_tensor_parallel(chunk_size):
    """``tp_group=None`` routes through the chunked function, chunked even for None."""
    logits, target, grad_output = _make_inputs(2, 7, 11, torch.bfloat16)
    logits = logits.requires_grad_(True)

    out = _tp_target_logprobs(
        logits,
        target,
        vocab_start_index=0,
        vocab_end_index=11,
        tp_group=None,
        chunk_size=chunk_size,
    )

    assert type(out.grad_fn).__name__.startswith("LocalChunkedLogprob")
    assert out.grad_fn.saved_tensors[0].data_ptr() == logits.data_ptr()

    torch.testing.assert_close(
        out,
        _reference_logprobs(logits.detach(), target),
        rtol=FORWARD_TOL,
        atol=FORWARD_TOL,
    )

    out.backward(grad_output)
    assert logits.grad is not None
    assert logits.grad.dtype == torch.bfloat16


def test_default_chunk_size_splits_long_sequences():
    """With ``chunk_size=None`` the sequence is still split, on an odd boundary."""
    assert DEFAULT_LOCAL_LOGPROB_CHUNK_SIZE == 1024

    seq_len = DEFAULT_LOCAL_LOGPROB_CHUNK_SIZE + 7
    logits, target, grad_output = _make_inputs(1, seq_len, 9, torch.float32)

    chunked_logits = logits.clone().requires_grad_(True)
    actual = _tp_target_logprobs(
        chunked_logits,
        target,
        vocab_start_index=0,
        vocab_end_index=9,
        tp_group=None,
        chunk_size=None,
    )
    actual.backward(grad_output)

    reference_logits = logits.clone().requires_grad_(True)
    expected = _reference_logprobs(reference_logits, target)
    expected.backward(grad_output)

    torch.testing.assert_close(
        actual.detach(), expected.detach(), rtol=FORWARD_TOL, atol=FORWARD_TOL
    )
    torch.testing.assert_close(
        chunked_logits.grad,
        reference_logits.grad,
        rtol=BACKWARD_TOL,
        atol=BACKWARD_TOL,
    )


def test_tp_target_logprobs_top_k_path_is_unchanged():
    """Top-k/top-p filtering keeps the plain autograd branch and its numbers."""
    logits, target, _ = _make_inputs(2, 7, 11, torch.float32)
    sampling_params = TrainingSamplingParams(top_k=3, top_p=1.0)

    actual = _tp_target_logprobs(
        logits.clone(),
        target,
        vocab_start_index=0,
        vocab_end_index=11,
        tp_group=None,
        chunk_size=2,
        sampling_params=sampling_params,
    )

    filtered, _ = apply_top_k_top_p(logits.clone(), top_k=3, top_p=1.0)
    expected = _reference_logprobs(filtered, target)

    torch.testing.assert_close(actual, expected, rtol=FORWARD_TOL, atol=FORWARD_TOL)


class _RecordingCpSharder:
    """Stand-in for ``ContextParallelSharder``; the dispatcher only passes it through."""


@pytest.mark.parametrize("chunk_size", [None, 4])
def test_context_parallel_path_is_not_pre_cast_to_float32(monkeypatch, chunk_size):
    """The CP path must receive the logits in their own dtype.

    Its kernels cast per chunk, so an up-front cast of the whole tensor would
    materialize the float32 [B, S, V] activation (and its float32 gradient in
    backward) that chunking exists to avoid.
    """
    seen: dict[str, torch.dtype] = {}

    def fake_get_cp_sharded_next_token_logprobs(
        logits, input_ids, cp_sharder, *, chunk_size=None, sampling_params=None
    ):
        seen["dtype"] = logits.dtype
        return torch.zeros(
            input_ids.shape[0], input_ids.shape[1] - 1, dtype=torch.float32
        )

    monkeypatch.setattr(
        model_utils,
        "get_cp_sharded_next_token_logprobs",
        fake_get_cp_sharded_next_token_logprobs,
    )

    logits, _, _ = _make_inputs(2, 7, 11, torch.bfloat16)
    input_ids = torch.randint(0, 11, (2, 7))

    get_next_token_logprobs_from_logits(
        input_ids=input_ids,
        next_token_logits=logits,
        chunk_size=chunk_size,
        cp_sharder=_RecordingCpSharder(),
    )

    assert seen["dtype"] == torch.bfloat16


def test_context_parallel_path_is_pre_cast_when_filtering(monkeypatch):
    """Top-k/top-p filtering has no chunked kernel, so the pre-cast still applies."""
    seen: dict[str, torch.dtype] = {}

    def fake_get_cp_sharded_next_token_logprobs(
        logits, input_ids, cp_sharder, *, chunk_size=None, sampling_params=None
    ):
        seen["dtype"] = logits.dtype
        return torch.zeros(
            input_ids.shape[0], input_ids.shape[1] - 1, dtype=torch.float32
        )

    monkeypatch.setattr(
        model_utils,
        "get_cp_sharded_next_token_logprobs",
        fake_get_cp_sharded_next_token_logprobs,
    )

    logits, _, _ = _make_inputs(2, 7, 11, torch.bfloat16)
    input_ids = torch.randint(0, 11, (2, 7))

    get_next_token_logprobs_from_logits(
        input_ids=input_ids,
        next_token_logits=logits,
        chunk_size=None,
        sampling_params=TrainingSamplingParams(top_k=3, top_p=1.0),
        cp_sharder=_RecordingCpSharder(),
    )

    assert seen["dtype"] == torch.float32


@contextlib.contextmanager
def _single_rank_gloo_group():
    """Bring up a 1-rank gloo group so DTensor construction works on CPU."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        torch.distributed.init_process_group(
            backend="gloo",
            init_method=f"file://{os.path.join(tmp_dir, 'store')}",
            world_size=1,
            rank=0,
        )
        try:
            yield
        finally:
            torch.distributed.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="requires torch.distributed with the gloo backend",
)
@pytest.mark.parametrize(
    "chunk_size,expected_dtype", [(None, torch.float32), (4, torch.bfloat16)]
)
def test_context_parallel_dtensor_precast_follows_chunk_size(
    monkeypatch, chunk_size, expected_dtype
):
    """With a vocabulary shard the CP path only skips the pre-cast when chunking.

    ``chunk_size=None`` reaches the unchunked vocabulary-parallel kernel, which
    materializes float32 itself, so the pre-cast stays. Either way the logits
    must still arrive as a DTensor.
    """
    seen: dict[str, object] = {}

    def fake_get_cp_sharded_next_token_logprobs(
        logits, input_ids, cp_sharder, *, chunk_size=None, sampling_params=None
    ):
        seen["dtype"] = logits.dtype
        seen["is_dtensor"] = isinstance(logits, DTensor)
        return torch.zeros(
            input_ids.shape[0], input_ids.shape[1] - 1, dtype=torch.float32
        )

    monkeypatch.setattr(
        model_utils,
        "get_cp_sharded_next_token_logprobs",
        fake_get_cp_sharded_next_token_logprobs,
    )

    local_logits, _, _ = _make_inputs(2, 7, 11, torch.bfloat16)
    input_ids = torch.randint(0, 11, (2, 7))

    with _single_rank_gloo_group():
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("tp",))
        logits = DTensor.from_local(local_logits, mesh, [Shard(-1)])

        get_next_token_logprobs_from_logits(
            input_ids=input_ids,
            next_token_logits=logits,
            chunk_size=chunk_size,
            cp_sharder=_RecordingCpSharder(),
        )

    assert seen["dtype"] == expected_dtype
    assert seen["is_dtensor"] is True
