"""MultiLinearLoRA — N LoRA adapters in one nn.Linear.

Same shape and conventions as nemo_automodel's LinearLoRA (see
`nemo_automodel/components/_peft/lora.py`) — we subclass `nn.Linear` directly,
keep base weight/bias at their stock FQN, and expose `_init_adapter` as a
staticmethod so the same logic works for both __init__ and monkey-patching.

Difference from LinearLoRA: the N adapters' A/B matrices are stored as two
stacked Parameters,

    lora_A:  [N, dim, in_features]
    lora_B:  [N, out_features, dim]

Per-row routing is a LongTensor of length B indexing into N. Forward gathers
the per-row A/B slices and runs two bmms — base output uses the shared frozen
linear, exactly like LinearLoRA. When no routing is set, MultiLinearLoRA is
bit-equivalent to the stock frozen base nn.Linear (LoRA path skipped).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLinearLoRA(nn.Linear):
    """N-adapter LoRA, drop-in for nn.Linear at the same FQN."""

    def __init__(
        self,
        orig_linear: nn.Linear,
        n_adapters: int,
        dim: int = 8,
        alpha: int = 32,
        lora_A_init_method: str = "xavier",
        lora_dtype: Optional[torch.dtype] = None,
        adapter_names: Optional[list[str]] = None,
    ):
        assert isinstance(orig_linear, nn.Linear)
        assert n_adapters >= 1
        super().__init__(
            in_features=orig_linear.in_features,
            out_features=orig_linear.out_features,
            bias=orig_linear.bias is not None,
            device=orig_linear.weight.device,
            dtype=orig_linear.weight.dtype,
        )
        self.weight.data.copy_(orig_linear.weight.data)
        if orig_linear.bias is not None:
            self.bias.data.copy_(orig_linear.bias.data)
        MultiLinearLoRA._init_adapter(
            self,
            n_adapters=n_adapters,
            dim=dim,
            alpha=alpha,
            lora_A_init_method=lora_A_init_method,
            lora_dtype=lora_dtype,
        )
        # Store adapter names for routing/diag. Optional — when None, adapters
        # are addressed by integer index 0..n_adapters-1.
        self.adapter_names = adapter_names

    @torch.no_grad
    @staticmethod
    def _init_adapter(
        obj,
        n_adapters: int,
        dim: int = 8,
        alpha: int = 32,
        lora_A_init_method: str = "xavier",
        lora_dtype: Optional[torch.dtype] = None,
    ):
        """Adds N stacked LoRA weights to obj. obj is a MultiLinearLoRA or an
        nn.Linear (when monkey-patching).
        """
        obj.n_adapters = int(n_adapters)
        obj.dim = int(dim)
        obj.scale = alpha / dim

        # Freeze base
        obj.weight.requires_grad = False
        if obj.bias is not None:
            obj.bias.requires_grad = False

        device = obj.weight.device
        dtype = lora_dtype or obj.weight.dtype
        in_f, out_f = obj.in_features, obj.out_features

        # Stacked Parameters: [N, dim, in_f] and [N, out_f, dim].
        obj.lora_A = nn.Parameter(torch.empty(n_adapters, dim, in_f, device=device, dtype=dtype))
        obj.lora_B = nn.Parameter(torch.zeros(n_adapters, out_f, dim, device=device, dtype=dtype))
        # FSDP2 shard-dim hint: shard on dim=1 (the "r" dimension) so each
        # adapter slot's local shard matches the layout of an independently-
        # sharded single-adapter [r, in_f] / [out_f, r] parameter. Without this,
        # FSDP defaults to dim-0 sharding (sharding n_adapters), which breaks
        # the adapter slot layout and deadlocks backward.
        obj.lora_A._fsdp_shard_dim = 1
        obj.lora_B._fsdp_shard_dim = 1
        MultiLinearLoRA.init_lora_weights(obj, lora_A_init_method)

        # Routing: LongTensor[B] indexing into [0, N). None = LoRA path skipped.
        obj.adapter_ids: Optional[torch.Tensor] = None

    @torch.no_grad
    def init_lora_weights(self, init_method: str = "xavier"):
        """Initialize each adapter's lora_A slice; lora_B is explicitly zeroed.

        Two problems this has to solve simultaneously:

        1) Stock-style ``nn.init.xavier_normal_(self.lora_A.data[i])`` is broken
           when ``self.lora_A`` is a DTensor. ``_init_peft_adapters`` runs AFTER
           ``moe_parallelize_model`` wraps experts with ``ExpertParallel`` (see
           ``nemo_automodel/components/moe/parallelizer.py:60-63``), which calls
           ``distribute_tensor(param, ep_mesh, [Shard(0)])``. With ``EP=8`` and
           ``n_adapters=4``, dim-0 sharding means ranks 0-3 each own a local
           ``[1, dim, in_f]`` slice and ranks 4-7 own empty ``[0, ...]``.

           The stock loop ``for i in range(self.n_adapters)`` walks indices
           0..3, but on a sharded DTensor, ``self.lora_A.data[i]`` returns a
           view that either out-of-bounds (caught by the silent try/except in
           ``_init_peft_adapters``, see ``checkpointing.py:995``) or returns a
           DTensor view that ``normal_()`` doesn't actually write to local
           memory through. Diagnosed empirically 2026-06-01 on multi run
           152329 — 100% NaN in lora_A across every expert across every rank
           with a non-empty shard.

        2) ``_init_adapter`` allocates ``lora_B = torch.zeros(...)`` but
           ``to_empty_parameters_only(model, device=device)`` calls
           ``torch.empty_like`` on every parameter post-construction, replacing
           the zero storage with uninitialized memory. We must explicitly
           ``fill_(0)`` here. (Same reason stock ``LinearLoRA.init_lora_weights``
           ends with the same line.)

        Strategy: write directly to the LOCAL shard via ``to_local()``, use the
        full unsharded shape to compute fan_in/fan_out (so the init distribution
        is correct regardless of how dim 0 was sharded), and iterate over the
        LOCAL adapter count, not the global one.
        """
        # 1) lora_B: zero the local shard. This is safe whether lora_B is a
        #    plain tensor or a DTensor — DTensor.to_local() returns the local
        #    shard view, fill_(0) writes through to storage.
        lora_B_local = (
            self.lora_B.data.to_local() if hasattr(self.lora_B.data, "to_local")
            else self.lora_B.data
        )
        lora_B_local.fill_(0)

        # 2) lora_A: init the local shard using FULL-shape fan_in/fan_out.
        lora_A_local = (
            self.lora_A.data.to_local() if hasattr(self.lora_A.data, "to_local")
            else self.lora_A.data
        )

        # Number of adapters present on THIS rank (1 or 0 for EP=8 / n_adapters=4).
        # Skip ranks that own no slice (lora_A_local.shape[0] == 0).
        local_n_adapters = lora_A_local.shape[0]
        if local_n_adapters == 0:
            return

        # Full-shape fans match LinearLoRA.init_lora_weights (each adapter's
        # A is logically a (dim, in_features) Linear). We do NOT use the
        # sharded local shape to compute fan, because in this codebase
        # nothing shards lora_A's dim-1 or dim-2 -- only dim-0 (adapter dim)
        # is sharded by EP. So in_features / dim are full and correct on every
        # rank; the per-slice 2D init shape is [dim, in_features].
        fan_out = self.dim         # output features of A
        fan_in = self.in_features  # input features of A
        if init_method == "xavier":
            std = math.sqrt(2.0 / (fan_in + fan_out))
            for i in range(local_n_adapters):
                lora_A_local[i].normal_(0.0, std)
        else:
            # Kaiming uniform with a = sqrt(5) (matches PyTorch default).
            a = math.sqrt(5)
            std = math.sqrt(2.0 / ((1 + a * a) * fan_in))
            bound = math.sqrt(3.0) * std
            for i in range(local_n_adapters):
                lora_A_local[i].uniform_(-bound, bound)

    def set_routing(self, adapter_ids: torch.Tensor):
        """Set per-row adapter ids for the next forward.

        Args:
            adapter_ids: LongTensor of shape [B], values in [0, n_adapters).
        """
        if not (isinstance(adapter_ids, torch.Tensor) and adapter_ids.dtype == torch.long):
            raise TypeError(f"adapter_ids must be a LongTensor; got {adapter_ids!r}")
        if adapter_ids.numel() > 0:
            lo, hi = int(adapter_ids.min()), int(adapter_ids.max())
            if lo < 0 or hi >= self.n_adapters:
                raise ValueError(
                    f"adapter_ids out of range [0, {self.n_adapters}); "
                    f"got min={lo} max={hi}"
                )
        self.adapter_ids = adapter_ids

    def clear_routing(self):
        self.adapter_ids = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Output = frozen base + per-row LoRA pathway.

        Shapes:
            x: [B, S, in_f] or [B, in_f]
            adapter_ids: [B] or None

        With no routing set, returns the frozen base output (no LoRA).

        MoE-expert fallback: when ``x.shape[0]`` doesn't match the routing
        buffer length, an MoE router has scattered tokens to this expert and
        the per-row id↔row alignment is invalid. Fall back to canonical
        single-adapter (adapter 0) for all rows — matches rollback's bug-11
        guard, which preserves bit-equivalence with single-LoRA on expert
        MLPs (where multi-LoRA can't meaningfully apply per-row anyway).
        """
        res = F.linear(x, self.weight, self.bias)

        ids = self.adapter_ids
        if ids is None:
            return res

        # MoE/empty-expert guard: row counts must match for per-row gather.
        if x.shape[0] != ids.shape[0]:
            # Canonical-adapter (slot 0) path for all rows.
            A0 = self.lora_A[0]  # [dim, in_f]
            B0 = self.lora_B[0]  # [out_f, dim]
            Ax = F.linear(x, A0) * self.scale
            return res + F.linear(Ax, B0)

        # Whenever a routed tensor is adapter-homogeneous, execute the exact
        # standalone LinearLoRA kernel chain: no per-row gather/bmm and no
        # index_select/index_copy. This occurs for homogeneous microbatches
        # (including mbs=1) and for homogeneous expert-local token subsets.
        # Besides being cheaper, it preserves backward byte parity with a
        # true single. Mixed-id tensors continue through the grouped/bmm paths.
        first_id = ids[0]
        if bool(torch.all(ids == first_id)):
            k = int(first_id.item())
            Ax = F.linear(x, self.lora_A[k]) * self.scale
            return res + F.linear(Ax, self.lora_B[k])

        # MoE group-by-adapter path. Expert token ids may be non-contiguous;
        # group with index_select/index_copy to stay memory-flat without a
        # Python/GPU synchronization for every token.
        if getattr(self, "_nousnet_route_group_mode", False):
            lora_out = torch.zeros_like(res)
            for k in torch.unique(ids).tolist():
                idx = (ids == k).nonzero(as_tuple=True)[0]
                xk = x.index_select(0, idx)
                Axk = F.linear(xk, self.lora_A[k]) * self.scale
                yk = F.linear(Axk, self.lora_B[k])
                lora_out.index_copy_(0, idx, yk.to(lora_out.dtype))
            return res + lora_out

        # Dense mixed-adapter path. Packed/striped batches keep adapter rows
        # block-contiguous, so each block can execute the exact standalone
        # two-F.linear graph. This avoids the gathered bmm kernel whose forward
        # and backward reduction order differs from LinearLoRA.
        if x.dim() not in (2, 3):
            raise ValueError(
                f"MultiLinearLoRA.forward expects 2D or 3D input; got {x.dim()}D"
            )
        lora_out = torch.zeros_like(res)
        start = 0
        while start < ids.numel():
            k = int(ids[start].item())
            end = start + 1
            while end < ids.numel() and int(ids[end].item()) == k:
                end += 1
            xk = x.narrow(0, start, end - start)
            Axk = F.linear(xk, self.lora_A[k]) * self.scale
            yk = F.linear(Axk, self.lora_B[k])
            lora_out.narrow(0, start, end - start).copy_(yk.to(lora_out.dtype))
            start = end
        return res + lora_out

        # Per-row gather: A:[B,dim,in_f], B:[B,out_f,dim].
        # Scale is applied between A and B to match upstream
        # `LinearLoRA.forward`: `lora_B(lora_A(x) * scale)`. In fp32 the
        # order is mathematically equivalent to `scale * lora_B(lora_A(x))`,
        # but in bf16 it changes the GEMM accumulation order and breaks
        # bit-equivalence with single-LoRA. See worklog 2026-05-13/01.
        A = self.lora_A[ids]
        B = self.lora_B[ids]

        if x.dim() == 3:
            # x:[B,S,in_f] -> A@x.T :[B,dim,S] -> scale -> B@(...) :[B,out_f,S] -> .T :[B,S,out_f]
            Ax_scaled = torch.bmm(A, x.transpose(-1, -2)) * self.scale
            BAx = torch.bmm(B, Ax_scaled).transpose(-1, -2)
        elif x.dim() == 2:
            # x:[B,in_f] -> A@x.unsqueeze(-1) :[B,dim,1] -> scale -> B@(...) :[B,out_f,1]
            Ax_scaled = torch.bmm(A, x.unsqueeze(-1)) * self.scale
            BAx = torch.bmm(B, Ax_scaled).squeeze(-1)
        else:
            raise ValueError(f"MultiLinearLoRA.forward expects 2D or 3D input; got {x.dim()}D")

        return res + BAx



