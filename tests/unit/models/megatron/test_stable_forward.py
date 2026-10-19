# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native scheduler restoration and cross-thread recomputation controls."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import transformer_engine.pytorch.cpp_extensions as cpp
import transformer_engine.pytorch.cpp_extensions.gemm as gemm
import transformer_engine.pytorch.module.grouped_linear as grouped
import transformer_engine.pytorch.module.linear as linear
import transformer_engine.pytorch.ops.basic.rmsnorm as rms
from megatron.core.transformer.custom_layers import batch_invariant_kernels as kernels

import nemo_rl.models.megatron.stable_forward as stable
import nemo_rl.models.megatron.train as train

pytestmark = pytest.mark.mcore

def aliases() -> tuple:
    return (
        linear._linear_forward_impl,
        linear.reduce_scatter_along_first_dim,
        cpp.rmsnorm_fwd,
        rms.rmsnorm_fwd,
        cpp.rmsnorm_bwd,
        rms.rmsnorm_bwd,
        grouped.general_grouped_gemm,
        gemm.general_grouped_gemm,
    )


@pytest.mark.parametrize("forward_only", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_native_scheduler_scope_and_exception_restoration(
    forward_only: bool, enabled: bool, failure: bool
) -> None:
    """Real helper setup/restore surrounds the scheduler in both native modes."""
    before = aliases()
    model = SimpleNamespace(
        config=SimpleNamespace(fine_grained_activation_offloading=False)
    )
    iterator = iter([])
    processor = object()
    result = object()
    calls = []

    def scheduler(**kwargs: object) -> object:
        now = aliases()
        assert kwargs["forward_only"] is forward_only
        assert kwargs["model"] is model and kwargs["data_iterator"] is iterator
        assert kwargs["seq_length"] == kwargs["decoder_seq_length"] == 262144
        assert kwargs["num_microbatches"] == 4
        assert kwargs["micro_batch_size"] == 1
        partial = kwargs["forward_step_func"]
        assert partial.func is train.forward_with_post_processing_fn
        assert partial.keywords["post_processing_fn"] is processor
        assert not kernels.is_batch_invariant_mode_enabled()
        # Both backward entry points and the generic C++ GEMM alias stay native.
        assert now[4] is before[4] and now[5] is before[5]
        assert now[7] is before[7]
        for index in (0, 1, 2, 3, 6):
            assert (now[index] is not before[index]) is enabled
        calls.append(True)
        if failure:
            raise RuntimeError("intentional scheduler exception")
        return result

    with patch.object(train, "get_forward_backward_func", return_value=scheduler):
        args = dict(
            model=model,
            data_iterator=iterator,
            num_microbatches=4,
            seq_length=262144,
            mbs=1,
            post_processing_fn=processor,
            forward_only=forward_only,
            stable_policy_forward=enabled,
        )
        if failure:
            with pytest.raises(RuntimeError, match="intentional scheduler exception"):
                train.megatron_forward_backward(**args)
        else:
            assert train.megatron_forward_backward(**args) is result
    assert calls == [True]
    assert all(a is b for a, b in zip(aliases(), before, strict=True))
    assert not torch.cuda.is_initialized()


def test_bad_flag_rejected() -> None:
    with pytest.raises(ValueError, match="must be a boolean"):
        with stable.stabilized_forward(enabled="true"):
            raise AssertionError("Invalid flag was accepted")


def test_native_source_change_rejected_before_installation() -> None:
    before = aliases()
    with patch.dict(stable._TP_SOURCES, linear_source_sha256="0" * 64):
        with pytest.raises(ValueError, match="source identity changed"):
            with stable.stabilized_forward(enabled=True):
                raise AssertionError("Changed source was accepted")
    assert all(a is b for a, b in zip(aliases(), before, strict=True))


def test_expert_setup_failure_restores_prior_tp_and_norm_installation() -> None:
    before = aliases()
    with patch.object(
        stable.StableExpertGemm,
        "__enter__",
        side_effect=RuntimeError("intentional expert setup rejection"),
    ):
        with pytest.raises(RuntimeError, match="intentional expert setup rejection"):
            with stable.stabilized_forward(enabled=True):
                raise AssertionError("Rejected expert setup was accepted")
    assert all(a is b for a, b in zip(aliases(), before, strict=True))


@pytest.mark.parametrize(
    ("all_threads", "grad", "expect_native"),
    [(False, False, True), (True, False, False), (True, True, True)],
)
def test_expert_recomputation_thread_scope_preserves_native_backward(
    all_threads: bool, grad: bool, expect_native: bool
) -> None:
    """Use the real installed wrapper; reject an invalid forward before CUDA."""
    before = aliases()
    original_calls = []
    with stable.StableExpertGemm(
        stable._EXPERT_SOURCES, all_threads=all_threads
    ) as expert:
        # Instrument only the saved delegate; no kernel, model or backward runs.
        def native_delegate(*args: object, **kwargs: object) -> str:
            original_calls.append(kwargs["grad"])
            return "native delegate observed"

        expert._original = native_delegate

        def other_thread() -> None:
            assert not expert._active.get()
            if expect_native:
                assert (
                    grouped.general_grouped_gemm(
                        [], [], [], [], torch.bfloat16, grad=grad
                    )
                    == "native delegate observed"
                )
            else:
                with pytest.raises(
                    RuntimeError, match="Unqualified expert forward GEMM ABI"
                ):
                    grouped.general_grouped_gemm(
                        [], [], [], [], torch.bfloat16, grad=grad
                    )

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(other_thread).result(timeout=10)
        finally:
            # Restore the actual delegate before the real alias cleanup.
            expert._original = before[6]
    assert original_calls == ([grad] if expect_native else [])
    assert all(a is b for a, b in zip(aliases(), before, strict=True))
    assert not torch.cuda.is_initialized()
