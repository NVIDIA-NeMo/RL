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

"""Source-pinned BF16 policy forwards with native TE backward operations.

This opt-in path stabilizes forward TP sums, RMS statistics and expert GEMMs.
The pinned implementations were qualified on the Super EP64 and EP32 layouts.
Full policy backward, optimizer updates and rollout behavior require separate
qualification. Importing this module does not import optional TE dependencies.
"""

from __future__ import annotations

import ast
import contextvars
import hashlib
import importlib
import inspect
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, Iterator, cast

import torch


def forward_argument_names(source: str) -> tuple[str, ...]:
    functions = [
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "_linear_forward_impl"
    ]
    if len(functions) != 1:
        raise ValueError("Expected one native TE linear forward implementation")
    assignments = [
        n
        for n in ast.walk(functions[0])
        if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Name)
        and n.value.id == "non_tensor_args"
        and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Tuple)
    ]
    if len(assignments) != 1 or not all(
        isinstance(n, ast.Name) for n in assignments[0].targets[0].elts
    ):
        raise ValueError("Native TE non-tensor argument layout changed")
    names = tuple(n.id for n in assignments[0].targets[0].elts)
    required = {
        "fp8",
        "tp_size",
        "sequence_parallel",
        "activation_dtype",
        "parallel_mode",
        "ub_overlap_rs_fprop",
    }
    if len(set(names)) != len(names) or not required.issubset(names):
        raise ValueError("Native TE forward flags are missing or duplicated")
    return names


class ForwardFP32TPReduction:
    def __init__(self, source_contract, *, linear_module=None, distributed_module=None):
        self.linear = linear_module or importlib.import_module(
            "transformer_engine.pytorch.module.linear"
        )
        self.distributed = distributed_module or importlib.import_module(
            "transformer_engine.pytorch.distributed"
        )
        linear_source = Path(cast(str, self.linear.__file__)).read_bytes()
        self.source_info = {
            "linear_source_sha256": hashlib.sha256(linear_source).hexdigest(),
            "distributed_source_sha256": hashlib.sha256(
                Path(cast(str, self.distributed.__file__)).read_bytes()
            ).hexdigest(),
        }
        for name, digest in self.source_info.items():
            if source_contract[name] != digest:
                raise ValueError(f"Exact native TE source identity changed: {name}")
        self.argument_names = forward_argument_names(linear_source.decode())
        self.original_forward = self.linear._linear_forward_impl
        self.original_reduce = self.linear.reduce_scatter_along_first_dim
        if self.original_reduce is not self.distributed.reduce_scatter_along_first_dim:
            raise ValueError("Native reduce-scatter alias was already replaced")
        if self.original_forward.__name__ != "_linear_forward_impl":
            raise ValueError("Native TE forward was already wrapped")
        self.signature = inspect.signature(self.original_forward)
        if "non_tensor_args" not in self.signature.parameters:
            raise ValueError("Native TE forward signature changed")
        self.active = ContextVar("frozen_candidate_fp32_tp_forward", default=False)
        self.installed = False
        self.stats = {
            "linear_forward_calls": 0,
            "forward_fp32_reduce_scatters": 0,
            "unchanged_native_reduce_scatters": 0,
        }

    def __enter__(self):
        if self.installed:
            raise ValueError("FP32 forward-reduction candidate already installed")
        if (
            self.linear._linear_forward_impl is not self.original_forward
            or self.linear.reduce_scatter_along_first_dim is not self.original_reduce
        ):
            raise ValueError(
                "Native TE functions changed before candidate installation"
            )

        def linear_forward(*args, **kwargs):
            bound = self.signature.bind(*args, **kwargs)
            values = bound.arguments["non_tensor_args"]
            if len(values) != len(self.argument_names):
                raise ValueError("Native TE forward argument count changed")
            flags = dict(zip(self.argument_names, values, strict=True))
            row_sp = (
                flags["parallel_mode"] == "row"
                and flags["sequence_parallel"]
                and flags["tp_size"] > 1
            )
            if row_sp and (flags["fp8"] or flags["ub_overlap_rs_fprop"]):
                raise ValueError(
                    "This frozen BF16 candidate does not support FP8 or fused user-buffer reductions"
                )
            eligible = row_sp and flags["activation_dtype"] == torch.bfloat16
            token = self.active.set(eligible)
            self.stats["linear_forward_calls"] += 1
            try:
                return self.original_forward(*args, **kwargs)
            finally:
                self.active.reset(token)

        def reduce_scatter(inp, tp_group, async_op=False):
            if not self.active.get():
                self.stats["unchanged_native_reduce_scatters"] += 1
                return self.original_reduce(inp, tp_group, async_op=async_op)
            if async_op or inp.dtype != torch.bfloat16:
                raise ValueError(
                    "Unexpected asynchronous or non-BF16 forward reduction"
                )
            output, handle = self.original_reduce(
                inp.to(dtype=torch.float32), tp_group, async_op=False
            )
            if handle is not None:
                raise ValueError(
                    "Synchronous forward reduction returned an async work handle"
                )
            self.stats["forward_fp32_reduce_scatters"] += 1
            return output.to(dtype=inp.dtype), None

        self.linear._linear_forward_impl = linear_forward
        self.linear.reduce_scatter_along_first_dim = reduce_scatter
        self.installed = True
        return self

    def __exit__(self, _kind, _error, _traceback):
        if self.installed:
            self.linear._linear_forward_impl = self.original_forward
            self.linear.reduce_scatter_along_first_dim = self.original_reduce
        self.installed = False
        return False


def ordered_reduce_scatter(inp, group, *, collectives):
    """Redistribute BF16 contributions, then add in fixed source-group-rank order."""
    world = collectives.get_world_size(group)
    if world < 2 or inp.dtype != torch.bfloat16 or inp.shape[0] % world:
        raise ValueError("Ordered forward reduction requires divisible BF16 TP input")
    received = torch.empty_like(inp)
    work = collectives.all_to_all_single(
        received, inp.contiguous(), group=group, async_op=False
    )
    if work is not None:
        raise ValueError("Synchronous all-to-all returned a work handle")
    parts = received.view(world, inp.shape[0] // world, *inp.shape[1:])
    # Separate elementwise additions preserve this source order at every token,
    # independently of which TP rank owns the token after sequence sharding.
    output = parts[0].float()
    for rank in range(1, world):
        output.add_(parts[rank].float())
    return output.to(dtype=inp.dtype), None


class StablePolicyForward(ForwardFP32TPReduction):
    def __init__(
        self,
        tp_source_contract,
        norm_source_contract=None,
        *,
        stable_norm=False,
        collectives=None,
        linear_module=None,
        distributed_module=None,
        cpp_module=None,
        ops_norm_module=None,
        extension_module=None,
        constants_module=None,
        mean_module=None,
    ):
        super().__init__(
            tp_source_contract,
            linear_module=linear_module,
            distributed_module=distributed_module,
        )
        self.collectives = collectives or torch.distributed
        self.stable_norm = stable_norm
        self.norm_installed = False
        self.cpp = self.ops_norm = self.extension = self.mean_module = cast(Any, None)
        self.original_rms_fwd = None
        self.original_cpp_bwd = self.original_ops_bwd = None
        self.dtype_by_enum = {}
        self.stats.update(forward_ordered_reduce_scatters=0, stable_rmsnorm_forwards=0)
        if stable_norm:
            self.cpp = cpp_module or importlib.import_module(
                "transformer_engine.pytorch.cpp_extensions"
            )
            self.ops_norm = ops_norm_module or importlib.import_module(
                "transformer_engine.pytorch.ops.basic.rmsnorm"
            )
            self.extension = extension_module or importlib.import_module(
                "transformer_engine_torch"
            )
            self.mean_module = mean_module or importlib.import_module(
                "megatron.core.transformer.custom_layers.batch_invariant_kernels"
            )
            constants = constants_module or importlib.import_module(
                "transformer_engine.pytorch.constants"
            )
            for key, module in (
                ("cpp_extensions_init_sha256", self.cpp),
                ("ops_rmsnorm_sha256", self.ops_norm),
                ("mean_module_sha256", self.mean_module),
            ):
                digest = hashlib.sha256(
                    Path(cast(str, module.__file__)).read_bytes()
                ).hexdigest()
                if cast(dict[str, str], norm_source_contract)[key] != digest:
                    raise ValueError(f"Native normalization source changed: {key}")
                self.source_info[key] = digest
            self.original_rms_fwd = self.cpp.rmsnorm_fwd
            if not (
                self.original_rms_fwd is self.ops_norm.rmsnorm_fwd
                and self.original_rms_fwd is self.extension.rmsnorm_fwd
            ):
                raise ValueError("Native RMSNorm forward aliases were already replaced")
            self.original_cpp_bwd = self.cpp.rmsnorm_bwd
            self.original_ops_bwd = self.ops_norm.rmsnorm_bwd
            if not (
                self.original_cpp_bwd is self.original_ops_bwd
                and self.original_cpp_bwd is self.extension.rmsnorm_bwd
            ):
                raise ValueError(
                    "Native RMSNorm backward aliases were already replaced"
                )
            self.dtype_by_enum = {
                constants.TE_DType[dtype]: dtype
                for dtype in (torch.float16, torch.bfloat16, torch.float32)
            }

    @torch.no_grad()
    def rmsnorm_forward(
        self,
        inp,
        weight,
        eps,
        out,
        quantizer,
        output_enum,
        sm_margin,
        zero_centered_gamma,
    ):
        dtype = self.dtype_by_enum.get(output_enum)
        if (
            quantizer is not None
            or dtype is None
            or inp.ndim != 2
            or inp.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or weight.shape != (inp.shape[-1],)
        ):
            raise ValueError("Unsupported native RMSNorm forward ABI/path")
        if out is not None and (out.shape != inp.shape or out.dtype != dtype):
            raise ValueError("Native RMSNorm output buffer changed shape/dtype")
        x = inp.float()
        w = weight.float()
        if zero_centered_gamma:
            w = w + 1.0
        mean_square = self.mean_module.mean_dim(x * x, dim=-1, keepdim=True)
        reciprocal_std = torch.rsqrt(mean_square + eps)
        values = (x * reciprocal_std) * w
        if out is None:
            out = values.to(dtype=dtype)
        else:
            out.copy_(values)
        self.stats["stable_rmsnorm_forwards"] += 1
        # Both fused TE modules and the BasicOperation RMSNorm save this native
        # FP32 statistic for their unchanged C++ backward implementation.
        return out, None, reciprocal_std.squeeze(-1).contiguous()

    def __enter__(self):
        if self.stable_norm and (
            self.cpp.rmsnorm_fwd is not self.original_rms_fwd
            or self.ops_norm.rmsnorm_fwd is not self.original_rms_fwd
        ):
            raise ValueError("RMSNorm functions changed before candidate installation")
        super().__enter__()

        def reduction(inp, tp_group, async_op=False):
            if not self.active.get():
                self.stats["unchanged_native_reduce_scatters"] += 1
                return self.original_reduce(inp, tp_group, async_op=async_op)
            if async_op:
                raise ValueError("Ordered forward reduction cannot be asynchronous")
            result = ordered_reduce_scatter(inp, tp_group, collectives=self.collectives)
            self.stats["forward_fp32_reduce_scatters"] += 1
            self.stats["forward_ordered_reduce_scatters"] += 1
            return result

        self.linear.reduce_scatter_along_first_dim = reduction
        if self.stable_norm:
            self.cpp.rmsnorm_fwd = self.ops_norm.rmsnorm_fwd = self.rmsnorm_forward
            self.norm_installed = True
        return self

    def __exit__(self, kind, error, traceback):
        if self.norm_installed:
            self.cpp.rmsnorm_fwd = self.ops_norm.rmsnorm_fwd = self.original_rms_fwd
        self.norm_installed = False
        return super().__exit__(kind, error, traceback)


def write_groups(
    weights: list[torch.Tensor],
    inputs: list[torch.Tensor],
    destination: torch.Tensor,
    splits: list[int],
    *,
    kernel: Callable,
    max_output_bytes: int = 16 * 1024 * 1024,
) -> int:
    """Write every group using bounded native-kernel output temporaries.

    The native kernel returns an allocated output even though TE has already
    provided the destination. Process row blocks to bound that extra allocation
    without changing any dot product or the native GroupedLinear backward.
    """
    if not weights or len(weights) != len(inputs) or len(weights) != len(splits):
        raise ValueError("Expected equally sized, nonempty weight/input/split lists")
    features, hidden = weights[0].shape
    row_bytes = features * destination.element_size()
    if type(max_output_bytes) is not int or max_output_bytes <= 0:
        raise ValueError("Output temporary budget must be a positive integer")
    if row_bytes <= 0 or max_output_bytes < row_bytes:
        raise ValueError("Output temporary budget must fit at least one output row")
    block_rows = max_output_bytes // row_bytes
    if destination.shape != (sum(splits), features):
        raise ValueError("Output shape does not match original group sizes")
    for weight, values, rows in zip(weights, inputs, splits, strict=True):
        if type(rows) is not int or rows < 0:
            raise ValueError("Group row counts must be nonnegative integers")
        if weight.shape != (features, hidden) or values.shape != (rows, hidden):
            raise ValueError("Incompatible expert weight/input dimensions")
        if weight.dtype != torch.bfloat16 or values.dtype != torch.bfloat16:
            raise ValueError("Only unquantized BF16 expert inputs are qualified")
        if weight.device != destination.device or values.device != destination.device:
            raise ValueError("All tensors must be on the same device")
    if destination.dtype != torch.bfloat16 or not destination.is_contiguous():
        raise ValueError("Expected native contiguous BF16 output")
    offset = 0
    calls = 0
    for weight, values, rows in zip(weights, inputs, splits, strict=True):
        for start in range(0, rows, block_rows):
            stop = min(start + block_rows, rows)
            destination[offset + start : offset + stop].copy_(
                kernel(values[start:stop], weight.t())
            )
            calls += 1
        offset += rows
    return calls


class StableExpertGemm:
    """Install only the native GroupedLinear call alias and restore it on exit.

    The exact native forward ABI is required. Backward and calls outside this
    context delegate verbatim to the saved TE implementation. This does not
    enable MCore's global batch-invariant mode or its DeepGEMM backend.

    all_threads keeps checkpoint recomputation covered when autograd invokes a
    forward on another thread. The alias is scoped to this process and restored
    on exit; grad=True GEMMs still delegate to the original implementation.
    """

    def __init__(self, sources: dict[str, str], *, all_threads: bool = False) -> None:
        self.sources = sources
        self.all_threads = all_threads
        self.installed = False
        self.stats = {
            "forward_calls": 0,
            "group_kernels": 0,
            "empty_groups": 0,
            "native_backward_calls": 0,
            "native_outside_context_calls": 0,
        }
        self._active = contextvars.ContextVar(
            "stable_expert_gemm_active", default=False
        )
        self._token = cast(Any, None)
        self._grouped = cast(Any, None)
        self._gemm = cast(Any, None)
        self._original = cast(Any, None)
        self._wrapper = cast(Any, None)
        self._kernel = cast(Any, None)

    def __enter__(self) -> "StableExpertGemm":
        if self.installed:
            raise RuntimeError("Stable expert GEMM is already installed")
        # Native CUDA/TE imports are optional for CPU unit tests of write_groups.
        modules = {name: importlib.import_module(name) for name in self.sources}
        for name, module in modules.items():
            if (
                hashlib.sha256(
                    Path(cast(str, module.__file__)).read_bytes()
                ).hexdigest()
                != self.sources[name]
            ):
                raise RuntimeError(f"Unqualified native source: {name}")
        self._grouped = modules["transformer_engine.pytorch.module.grouped_linear"]
        self._gemm = modules["transformer_engine.pytorch.cpp_extensions.gemm"]
        kernels = modules[
            "megatron.core.transformer.custom_layers.batch_invariant_kernels"
        ]
        if not kernels.HAVE_TRITON or kernels.is_batch_invariant_mode_enabled():
            raise RuntimeError(
                "Expected native Triton with global batch-invariant mode disabled"
            )
        if self._grouped.general_grouped_gemm is not self._gemm.general_grouped_gemm:
            raise RuntimeError("Native grouped GEMM alias has already been changed")
        self._original = self._grouped.general_grouped_gemm
        self._kernel = kernels.matmul_persistent

        def forward(
            A,
            B,
            out,
            quantization_params,
            out_dtype,
            layout="TN",
            m_splits=None,
            gelu=False,
            grad=False,
            accumulate=False,
            bias=None,
            use_bias=False,
            use_split_accumulator=False,
            D_dtype=None,
            single_output=False,
        ):
            if grad or not (
                self._active.get() or (self.all_threads and self.installed)
            ):
                self.stats[
                    "native_backward_calls" if grad else "native_outside_context_calls"
                ] += 1
                return self._original(
                    A,
                    B,
                    out,
                    quantization_params,
                    out_dtype,
                    layout=layout,
                    m_splits=m_splits,
                    gelu=gelu,
                    grad=grad,
                    accumulate=accumulate,
                    bias=bias,
                    use_bias=use_bias,
                    use_split_accumulator=use_split_accumulator,
                    D_dtype=D_dtype,
                    single_output=single_output,
                )
            if (
                layout != "TN"
                or not single_output
                or gelu
                or accumulate
                or use_bias
                or D_dtype is not None
                or out_dtype != torch.bfloat16
            ):
                raise RuntimeError("Unqualified expert forward GEMM ABI")
            if (
                not isinstance(quantization_params, (list, tuple))
                or len(quantization_params) != len(A)
                or any(item is not None for item in quantization_params)
            ):
                raise RuntimeError("Quantized expert GEMM is not qualified")
            if len(out) != 1 or out[0].device.type != "cuda":
                raise RuntimeError("Expected one native CUDA expert output")
            if bias is not None and any(
                item is not None and item.numel() for item in bias
            ):
                raise RuntimeError("Unexpected native expert bias")
            calls = write_groups(
                A, B, out[0], cast(list[int], m_splits), kernel=self._kernel
            )
            self.stats["forward_calls"] += 1
            self.stats["group_kernels"] += calls
            self.stats["empty_groups"] += sum(
                rows == 0 for rows in cast(list[int], m_splits)
            )
            empty = self._gemm._empty_tensor()
            return out, [empty] * len(A), [empty] * len(A)

        self._wrapper = forward
        self._token = self._active.set(True)
        self._grouped.general_grouped_gemm = self._wrapper
        self.installed = True
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if not self.installed:
            return
        changed = self._grouped.general_grouped_gemm is not self._wrapper
        self._grouped.general_grouped_gemm = self._original
        self._active.reset(self._token)
        self.installed = False
        if changed and exc_type is None:
            raise RuntimeError(
                "Grouped GEMM alias changed inside the diagnostic context"
            )


_TP_SOURCES = {
    "linear_source_sha256": "0a4a0f81e80cab6d62d5a51f19df3493e4848e9b6b9cf485eb58547274765434",
    "distributed_source_sha256": "afa020a375b4133c97b0a45841faa069c40adae7bc19c121e00985bad9f3433f",
}

_NORM_SOURCES = {
    "cpp_extensions_init_sha256": "7f327d55b6e010829989b77911affa7fc70415677bf6a96d375a1809f4a2ff4c",
    "ops_rmsnorm_sha256": "cefd8751948ac5797dcf2c0c89152f40494f95d2b1b9883cd37b353486b0fa9e",
    "mean_module_sha256": "09d9f77b61241f8e05896c5baa2156217d317f9bea51c8a2c3d92e85e852f3bb",
}

_EXPERT_SOURCES = {
    "transformer_engine.pytorch.module.grouped_linear": "d76296a6914eb23c221a417d94f8a646b93e949c27aa3987f0d253003fe565f5",
    "transformer_engine.pytorch.cpp_extensions.gemm": "975a46a79c5a80b9c03aa3eca063df1d76959e11cd88783bb559d772d8c7ab86",
    "megatron.core.transformer.custom_layers.batch_invariant_kernels": "09d9f77b61241f8e05896c5baa2156217d317f9bea51c8a2c3d92e85e852f3bb",
}


@contextmanager
def stabilized_forward(*, enabled: bool) -> Iterator[None]:
    """Scope the qualified math across the native scheduler and recomputation.

    Args:
        enabled: Explicit opt-in for the pinned unquantized BF16 implementation.

    Raises:
        ValueError: The flag or native implementation does not match its contract.
        RuntimeError: An unsupported expert forward path is encountered.
    """
    if type(enabled) is not bool:
        raise ValueError("megatron_cfg.stable_policy_forward must be a boolean")
    if not enabled:
        yield
        return
    # ExitStack also restores the TP/RMS aliases if expert setup rejects its ABI.
    with ExitStack() as stack:
        stack.enter_context(
            StablePolicyForward(_TP_SOURCES, _NORM_SOURCES, stable_norm=True)
        )
        stack.enter_context(StableExpertGemm(_EXPERT_SOURCES, all_threads=True))
        yield
