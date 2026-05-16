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
"""DTensor-aware Muon optimizer.

Wraps emerging_optimizers.OrthogonalizedOptimizer with a Newton-Schulz
orthogonalization routine that is intended to understand DTensor placements.
This commit lands the single-GPU surface only; the TP-aware path that maps
DTensor placements to ``newton_schulz_tp(partition_dim=...)`` is added in a
follow-up commit so each phase has its own verification gate.
"""

from __future__ import annotations

from typing import Any, Callable, Literal

import torch
from torch.distributed.tensor import DTensor, Shard
from torch.optim.optimizer import ParamsT

try:
    from emerging_optimizers.orthogonalized_optimizers import (
        OrthogonalizedOptimizer,
        get_muon_scale_factor,
    )
    from emerging_optimizers.orthogonalized_optimizers.muon_utils import (
        newton_schulz,
        newton_schulz_tp,
    )

    HAVE_EMERGING_OPTIMIZERS = True
except ImportError:  # pragma: no cover - exercised only when extras missing
    HAVE_EMERGING_OPTIMIZERS = False
    OrthogonalizedOptimizer = object  # type: ignore[assignment,misc]


_MISSING_DEP_MSG = (
    "DTensorMuon requires the 'emerging_optimizers' package. Install the NeMo-RL "
    "'automodel' extras for the DTensor backend (`uv sync --extra automodel`) "
    "or the 'mcore' extras for the Megatron backend."
)

WeightDecayMethodT = Literal["decoupled", "independent", "l2"]
FP32MatmulPrecT = Literal["highest", "high", "medium"]
NSCoeffT = Literal["simple", "quintic", "polar_express"]
MuonScaleT = Literal["shape_scaling", "spectral", "unit_rms_norm"]
MuonTpModeT = Literal["blockwise", "duplicated", "distributed"]


class DTensorMuon(OrthogonalizedOptimizer):  # type: ignore[misc]
    """Muon optimizer for parameters owned by NeMo-RL's DTensor policy worker.

    Single-GPU semantics in this commit: each gradient is treated as fully
    replicated and the orthogonalize step delegates to the same primitives
    that ``emerging_optimizers.Muon`` uses, which makes the per-step update
    bit-identical to vanilla Muon for the same input gradient and state.
    The TP-aware branch (``newton_schulz_tp`` with a real ``partition_dim``)
    is wired in a later commit so its verification stays separate.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        *,
        nesterov: bool = False,
        weight_decay_method: WeightDecayMethodT = "decoupled",
        fp32_matmul_prec: FP32MatmulPrecT = "medium",
        coefficient_type: NSCoeffT = "quintic",
        num_ns_steps: int = 5,
        scale_mode: MuonScaleT = "spectral",
        extra_scale_factor: float = 1.0,
        tp_mode: MuonTpModeT = "duplicated",
        split_qkv: bool = False,
        is_qkv_fn: Callable[[torch.Tensor], bool] | None = None,
        qkv_split_shapes: tuple[int, int, int] | None = None,
    ) -> None:
        if not HAVE_EMERGING_OPTIMIZERS:
            raise ImportError(_MISSING_DEP_MSG)
        if num_ns_steps < 1:
            raise ValueError(f"num_ns_steps must be at least 1, got {num_ns_steps}")
        if split_qkv and (is_qkv_fn is None or qkv_split_shapes is None):
            raise ValueError(
                "split_qkv=True requires both is_qkv_fn and qkv_split_shapes"
            )

        self._coefficient_type: NSCoeffT = coefficient_type
        self._num_ns_steps = num_ns_steps
        self._scale_mode: MuonScaleT = scale_mode
        self._extra_scale_factor = extra_scale_factor
        self._tp_mode: MuonTpModeT = tp_mode
        self._split_qkv = split_qkv
        self._is_qkv_fn = is_qkv_fn
        self._qkv_split_shapes = qkv_split_shapes

        def scaled_orthogonalize_fn(grad: torch.Tensor) -> torch.Tensor:
            orth = newton_schulz(
                grad,
                steps=num_ns_steps,
                coefficient_type=coefficient_type,
            )
            scale = get_muon_scale_factor(
                grad.size(-2), grad.size(-1), mode=scale_mode
            )
            return orth * scale * extra_scale_factor

        super().__init__(
            params,
            lr,
            momentum,
            weight_decay,
            nesterov=nesterov,
            weight_decay_method=weight_decay_method,
            fp32_matmul_prec=fp32_matmul_prec,
            scaled_orthogonalize_fn=scaled_orthogonalize_fn,
        )

    def orthogonalize(  # type: ignore[override]
        self, p: torch.Tensor, grad: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """Route the orthogonalize step based on the parameter's layout.

        * Plain tensors and DTensors with all-Replicate placements run the
          single-GPU primitives that ``emerging_optimizers.Muon`` uses.
        * DTensors with a ``Shard`` placement on a TP mesh dim feed the local
          shard and the sub-mesh process group to ``newton_schulz_tp`` and
          rebuild the result as a DTensor with the same placements so the
          base class's ``p.add_(orth_grad, alpha=-lr)`` stays type-consistent.
        * Fused QKV weights identified by ``is_qkv_fn`` go through a per-head
          split-orthogonalize-concat path that mirrors Megatron's
          ``TensorParallelMuon.orthogonalize`` at
          ``3rdparty/Megatron-LM-workspace/Megatron-LM/megatron/core/optimizer/muon.py:136-159``.
        """
        if self._split_qkv and self._is_qkv_fn is not None and self._is_qkv_fn(p):
            return self._qkv_split_orthogonalize(grad)
        if isinstance(grad, DTensor):
            return self._dtensor_orthogonalize(grad)
        return self.scaled_orthogonalize_fn(grad)

    def _qkv_split_orthogonalize(
        self, grad: torch.Tensor | DTensor
    ) -> torch.Tensor | DTensor:
        """Apply orthogonalize independently to Q, K and V slices of a fused
        ``linear_qkv.weight`` parameter.

        Mirrors the per-head split / orthogonalize / concat sequence in
        Megatron's ``TensorParallelMuon.orthogonalize``. ``qkv_split_shapes``
        is the per-head ``(q, k, v)`` tuple as derived in
        ``3rdparty/Megatron-LM-workspace/Megatron-LM/megatron/core/optimizer/muon.py:234-238``.
        """
        assert self._qkv_split_shapes is not None  # checked in __init__
        # We always operate on the local tensor: the QKV split is a row-wise
        # reshape that does not interact with sharding metadata, so for DTensor
        # we unwrap, run the split, then re-wrap with the original placements.
        is_dtensor = isinstance(grad, DTensor)
        if is_dtensor:
            mesh = grad.device_mesh
            placements = grad.placements
            local_grad = grad.to_local()
        else:
            local_grad = grad

        grad_shape = local_grad.shape
        per_head_total = sum(self._qkv_split_shapes)
        if grad_shape[0] % per_head_total != 0:
            raise ValueError(
                f"Fused QKV grad row count {grad_shape[0]} is not divisible by "
                f"sum(qkv_split_shapes)={per_head_total}"
            )
        num_query_groups = grad_shape[0] // per_head_total
        qkv_grads = torch.split(
            local_grad.view(num_query_groups, per_head_total, -1),
            list(self._qkv_split_shapes),
            dim=1,
        )
        qkv_grads = [g.reshape(-1, grad_shape[-1]) for g in qkv_grads]
        # Each slice goes through the same orthogonalize logic the rest of the
        # path uses, including the DTensor TP routing if the original grad was
        # a DTensor sharded on row dim 0.
        if is_dtensor:
            sliced_dts = [
                DTensor.from_local(g, mesh, placements, run_check=False)
                for g in qkv_grads
            ]
            orth_slices = [
                self._dtensor_orthogonalize(g_dt).to_local() for g_dt in sliced_dts
            ]
        else:
            orth_slices = [self.scaled_orthogonalize_fn(g) for g in qkv_grads]

        orth_slices = [
            s.view(num_query_groups, -1, grad_shape[-1]) for s in orth_slices
        ]
        local_out = torch.cat(orth_slices, dim=1).view(grad_shape)
        if is_dtensor:
            return DTensor.from_local(local_out, mesh, placements, run_check=False)
        return local_out

    def _dtensor_orthogonalize(self, grad: DTensor) -> DTensor:
        mesh = grad.device_mesh
        placements = grad.placements

        shard_axes = [
            (axis, p.dim) for axis, p in enumerate(placements) if isinstance(p, Shard)
        ]
        # All-Replicate fast path (or weight that hasn't been sharded along any
        # mesh axis): treat as single-GPU. Wrap the orthogonalized local tensor
        # back into a DTensor with the original placements so the base class's
        # in-place add stays consistent.
        if not shard_axes:
            local = grad.to_local()
            orth_local = self.scaled_orthogonalize_fn(local)
            return DTensor.from_local(orth_local, mesh, placements, run_check=False)

        # When the gradient is sharded across more than one mesh dim (e.g. 2D
        # TP), fall back to the all-gather path on the first shard dim and
        # ignore the others; emerging_optimizers' newton_schulz_tp only models
        # one TP axis. Multi-axis TP for Muon is out of scope for this commit.
        shard_axis, partition_dim = shard_axes[0]
        tp_group = mesh.get_group(shard_axis)
        local = grad.to_local()

        # ``blockwise`` aliases to ``duplicated`` in newton_schulz_tp (the
        # local-only path runs an all-gather first), matching Megatron's
        # TensorParallelMuon at muon.py:87.
        ns_mode: Literal["duplicated", "distributed"] = (
            "duplicated" if self._tp_mode in ("blockwise", "duplicated") else "distributed"
        )
        orth_local = newton_schulz_tp(
            local,
            steps=self._num_ns_steps,
            coefficient_type=self._coefficient_type,
            tp_group=tp_group,
            partition_dim=partition_dim,
            tp_mode=ns_mode,
        )
        # The scale factor in get_muon_scale_factor is computed on the
        # **global** matrix shape, so undo the partition before asking for it.
        full_size = [local.size(-2), local.size(-1)]
        full_size[partition_dim] *= tp_group.size()
        scale = get_muon_scale_factor(
            full_size[0], full_size[1], mode=self._scale_mode
        )
        orth_local = orth_local * scale * self._extra_scale_factor
        return DTensor.from_local(orth_local, mesh, placements, run_check=False)
