# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelExpress v2 refit receiver, as a vLLM v1 ``worker_extension_cls``.

Registered into the vLLM ``Worker`` class via ``parallel_config.worker_extension_cls``
so its methods become callable through ``AsyncLLM.collective_rpc``. Each method
runs **inside the worker process**, where ``self.model_runner.model`` and
``self.device`` are available.

Lifetime model (mirrors ``VllmInternalWorkerExtension``):

  * ``prepare_refit_info(state_dict_info)`` is called once per worker before the
    first refit. Stores per-tensor (shape, dtype) for asserts and FP8 paths.
  * ``update_weights_via_mx(version, mx_config)`` is called every refit cycle.
    Lazy-initializes an :class:`modelexpress.MxV2RefitReceiver` on first call,
    registers ``model.named_parameters()`` as NIXL receive buffers, then for
    every subsequent cycle: discover same-rank source → RDMA receive → call
    ``_load_weights`` → optionally republish as inference_replica for tree
    fan-out.

The Dynamo handler in ``components/src/dynamo/vllm/handlers.py`` invokes this
via ``await self.engine_client.collective_rpc("update_weights_via_mx", kwargs=...)``.
"""

from __future__ import annotations

import gc
import logging
import os
import time as _time
import traceback
from dataclasses import dataclass, field
from typing import Any

import torch

logger = logging.getLogger(__name__)


# =============================================================================
# MxConfig — wire-compatible with nemo_rl.distributed.mx_helpers.MxConfig
# =============================================================================


@dataclass
class MxConfig:
    """Subset of nemo-rl's MxConfig needed on the receiver side.

    The trainer sends this as a dict over the Dynamo Endpoint RPC; we parse it
    with :meth:`from_dict`. Field names and defaults must stay in sync with
    nemo_rl.distributed.mx_helpers.MxConfig.
    """

    enabled: bool = True
    mx_server_url: str = "modelexpress-server:8001"
    timeout_seconds: float = 300.0
    same_rank_only: bool = True
    tree_scale_out: bool = True
    moe_expert_filter: bool = True
    register_self_buffers: list[str] = field(default_factory=list)
    nic_pin: str = "auto"
    retain_latest_k: int = 1

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "MxConfig":
        if not d:
            return cls()
        return cls(
            enabled=bool(d.get("enabled", True)),
            mx_server_url=str(d.get("mx_server_url", "modelexpress-server:8001")),
            timeout_seconds=float(d.get("timeout_seconds", 300.0)),
            same_rank_only=bool(d.get("same_rank_only", True)),
            tree_scale_out=bool(d.get("tree_scale_out", True)),
            moe_expert_filter=bool(d.get("moe_expert_filter", True)),
            register_self_buffers=list(d.get("register_self_buffers", []) or []),
            nic_pin=str(d.get("nic_pin", "auto")),
            retain_latest_k=int(d.get("retain_latest_k", 1)),
        )


# =============================================================================
# NIC pinning (port of nemo_rl.distributed.mx_helpers.pin_local_nic)
# =============================================================================


def _pin_local_nic(*, device_id: int, mode: str = "auto") -> None:
    """Best-effort NUMA-local NIC pinning before NIXL initializes.

    On multi-NIC RDMA fabrics (e.g. GB200/GCP four-subnet RoCE) each rank's
    NIXL agent must bind to the NIC NUMA-closest to its GPU; cross-NIC writes
    are unrouted. Delegated to modelexpress's helper.
    """
    if mode == "off":
        return
    try:
        from modelexpress.ucx_utils import apply_nic_pin_for_device

        if mode == "auto":
            apply_nic_pin_for_device(device_id=device_id)
            logger.info("[mx] pinned NIC for device %d (auto)", device_id)
        else:
            os.environ["UCX_NET_DEVICES"] = mode
            os.environ["MX_RDMA_NIC_PIN"] = "off"
            logger.info("[mx] pinned NIC explicitly: %s", mode)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[mx] NIC pin failed (mode=%s): %s", mode, exc)


# =============================================================================
# Worker extension class — injected into vLLM Worker via worker_extension_cls
# =============================================================================


class MxRefitWorkerExtension:
    """Methods added to vLLM's ``Worker`` class via ``worker_extension_cls``.

    Has no ``__init__``: vLLM merges this class's methods into the existing
    ``Worker`` via ``__bases__``, and any state we need is stashed on ``self``
    lazily inside the methods themselves (``self._mx_receiver``,
    ``self._mx_recv_buffers``, ``self._mx_state_dict_info``). No conflict
    checks fire because all our attribute names use the ``_mx_`` prefix.
    """

    # ------------------------------------------------------------------ #
    # Refit-info preparation
    # ------------------------------------------------------------------ #
    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        """Record per-tensor (shape, dtype) info from the trainer.

        Called once by the trainer driver before the first refit cycle.
        Mirrors :py:meth:`VllmInternalWorkerExtension.prepare_refit_info` on
        the NeMo-RL side — assigns an instance attribute used by the FP8 path
        and for assertion checks. No-op for now if the worker class already
        had this attribute (the IPC-ZMQ path also stores it).
        """
        self._mx_state_dict_info = state_dict_info  # noqa: SLF001
        if not hasattr(self, "state_dict_info"):
            self.state_dict_info = state_dict_info

    # ------------------------------------------------------------------ #
    # Weight loading (minimal — adds GPT-OSS / FP8 / draft handling later)
    # ------------------------------------------------------------------ #
    # Stacked-param groups vLLM fuses on axis 0. Each entry is
    # (fused_suffix, member_suffix, order) — ``order`` is the position
    # of the member within the fused param, used to accumulate byte
    # offsets. Matches vLLM's Qwen3 ``stacked_params_mapping``.
    _MX_STACKED_GROUPS = (
        ("qkv_proj", "q_proj", 0),
        ("qkv_proj", "k_proj", 1),
        ("qkv_proj", "v_proj", 2),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    )

    def _mx_build_fused_dest_map(
        self,
        weights: list[tuple[str, torch.Tensor]],
    ) -> None:
        """Precompute, per HF tensor, its destination in the vLLM model.

        MDL (Mapped Direct Load): rather than lean on vLLM's stock
        ``load_weights`` (fragile per-arch traversal; the Qwen3-MoE
        fused-layout bug we hit on 2026-07-03 lives there), we resolve
        each received HF tensor to exactly ONE of three destinations,
        computed ONCE. This is our own eager, destination-mapped
        in-place write — inspired by RDT's "don't re-run the loader
        each refit" goal, but NOT RDT: no lazy tensors, no deferred
        narrow(), no dependency on the RDT API. We eagerly RDMA-pull
        then copy into the params' final slots, because NeMo-RL already
        knows its target layout (declared TargetTpLayout) so the dynamic
        layout discovery RDT's laziness buys is unnecessary here.

          * ``direct``  — ``hf_name`` matches a vLLM param 1:1 by shape.
            Warm cycle does ``param.data.copy_(tensor)``.
          * ``fused``   — ``hf_name`` is a member of a stacked param
            (q/k/v -> qkv_proj, gate/up -> gate_up_proj). Warm cycle
            does ``param.data.narrow(0, offset, size).copy_(tensor)``.
            Offsets are derived from the ACTUAL member tensor shapes
            (not model config), so this is version-robust: whatever
            order/size vLLM allocated, we match it by summing member
            rows in canonical (q,k,v / gate,up) order.
          * ``expert``  — per-expert MoE tensor (gate/up/down_proj)
            resolved via vLLM's ``get_expert_mapping()`` to a slot in
            the stacked ``w13_weight`` / ``w2_weight`` param. Warm cycle
            does ``param.data[expert_id].narrow(axis, offset, size).
            copy_(tensor)``. Requires the standard 3D layout
            (``--moe-backend triton``); the swizzled 4D ``auto`` layout
            routes to fallback (and the swizzle guard errors first).
          * ``fallback`` — anything else, or MoE experts under a
            swizzled backend. Warm cycle routes these through vLLM's
            stock loader. On dense + triton-MoE this set is EMPTY.

        Built after cycle 1's stock load so ``named_parameters`` is
        populated. Idempotent — only rebuilds if not present.
        """
        params = self._mx_param_cache
        direct: dict[str, "torch.Tensor"] = {}
        # fused: hf_name -> (fused_param, axis, offset, size)
        fused: dict[str, tuple] = {}
        fallback_names: set[str] = set()

        # expert dest: hf_name -> (fused_param, expert_id, axis, offset, size)
        # for MoE per-expert tensors written into the stacked w13/w2 params.
        expert: dict[str, tuple] = {}

        # Build the MoE expert-name → (fused_param, expert_id, shard_id)
        # lookup from vLLM's own authoritative mapping table, so we don't
        # hardcode name transforms. Empty on dense models / when the
        # backend is swizzled (see _mx_check_moe_swizzle — that guard
        # fires first). shard_id: w1=gate, w3=up, w2=down.
        expert_lookup: dict[str, tuple] = {}  # weight_suffix -> (param_suffix, expert_id, shard_id)
        try:
            model = self.model_runner.model
            if hasattr(model, "get_expert_mapping"):
                for param_suffix, weight_suffix, expert_id, shard_id in model.get_expert_mapping():
                    expert_lookup[weight_suffix] = (param_suffix, int(expert_id), shard_id)
        except Exception as exc:  # noqa: BLE001
            logger.info("[mx-mdl] no expert mapping (dense or unavailable): %s", exc)

        # First, bucket the stacked-group members by their fused param
        # name so we can accumulate offsets in canonical order.
        # group_key = fused_param_name; value = list of
        # (order, hf_name, member_rows).
        groups: dict[str, list[tuple[int, str, int]]] = {}
        name_to_shape = {n: tuple(t.shape) for n, t in weights}

        for hf_name in name_to_shape:
            param = params.get(hf_name)
            if param is not None and tuple(param.shape) == name_to_shape[hf_name]:
                direct[hf_name] = param
                continue
            # MoE per-expert tensor? Resolve via vLLM's expert mapping.
            if ".experts." in hf_name and expert_lookup:
                dest = self._mx_resolve_expert_dest(
                    hf_name, name_to_shape[hf_name], expert_lookup, params,
                )
                if dest is not None:
                    expert[hf_name] = dest
                    continue
            # Try stacked-group membership.
            matched = False
            for fused_suffix, member_suffix, order in self._MX_STACKED_GROUPS:
                if member_suffix + "." in hf_name or hf_name.endswith(member_suffix + ".weight"):
                    fused_name = hf_name.replace(member_suffix, fused_suffix)
                    fused_param = params.get(fused_name)
                    if fused_param is None:
                        continue
                    member_rows = name_to_shape[hf_name][0]
                    groups.setdefault(fused_name, []).append(
                        (order, hf_name, member_rows)
                    )
                    matched = True
                    break
            if not matched:
                fallback_names.add(hf_name)

        # Resolve offsets within each fused group (canonical order).
        for fused_name, members in groups.items():
            fused_param = params[fused_name]
            members.sort(key=lambda m: m[0])
            offset = 0
            for _order, hf_name, member_rows in members:
                fused[hf_name] = (fused_param, 0, offset, member_rows)
                offset += member_rows
            # Sanity: total should equal the fused param's axis-0 size.
            if offset != int(fused_param.shape[0]):
                logger.warning(
                    "[mx-mdl] fused group %s: member rows sum to %d but "
                    "param axis-0 is %d; routing group to fallback",
                    fused_name, offset, int(fused_param.shape[0]),
                )
                for _o, hf_name, _r in members:
                    fused.pop(hf_name, None)
                    fallback_names.add(hf_name)

        self._mx_mdl_direct = direct
        self._mx_mdl_fused = fused
        self._mx_mdl_expert = expert
        self._mx_mdl_fallback = fallback_names
        logger.info(
            "[mx-mdl] dest map built: %d direct, %d fused-slice, "
            "%d expert-slice, %d fallback",
            len(direct), len(fused), len(expert), len(fallback_names),
        )

    def _mx_resolve_expert_dest(
        self,
        hf_name: str,
        hf_shape: tuple,
        expert_lookup: dict,
        params: dict,
    ) -> tuple | None:
        """Resolve a per-expert HF tensor to its slot in the stacked w13/w2 param.

        Returns ``(fused_param, local_expert_idx, axis, offset, size)``
        for the warm-cycle write ``fused_param.data[local_expert_idx].
        narrow(axis, offset, size).copy_(tensor)``, or ``None`` to route
        to fallback.

        Layout (vLLM standard / --moe-backend triton):
          * w1 (gate): w13_weight[E] rows [0, inter)      → axis 0, offset 0
          * w3 (up):   w13_weight[E] rows [inter, 2*inter) → axis 0, offset inter
          * w2 (down): w2_weight[E] full                  → axis 0, offset 0, full

        EP>1 correctness: ``get_expert_mapping()`` yields a GLOBAL expert
        id, but under expert-parallel the vLLM param only holds this
        rank's LOCAL experts, so the param index must be the local slot.
        We map global→local via the owning FusedMoE module's
        ``_map_global_expert_id_to_local_expert_id`` (mirrors vLLM's own
        weight_loader). At EP=1 that map is identity, so this is a no-op
        for the validated single-rank path. A global id not local to
        this rank maps to -1 → routed to fallback/skipped (the EP filter
        should have pruned it from the pull upstream anyway).

        Guards: only accepts if the resolved fused param exists, is 3D
        (standard stacked, NOT the swizzled 4D layout — that routes to
        fallback and the swizzle guard errors), the local index is in
        range, and the destination slice shape matches the received
        tensor exactly (protects the TP assumption; TP>1 per-expert
        sharding would mismatch and route to fallback).
        """
        # Match the received name against vLLM's weight suffixes.
        for weight_suffix, (param_suffix, expert_id, shard_id) in expert_lookup.items():
            if weight_suffix not in hf_name:
                continue
            fused_name = hf_name.replace(weight_suffix, param_suffix)
            fused_param = params.get(fused_name)
            if fused_param is None or fused_param.ndim != 3:
                return None  # swizzled/absent → fallback

            # global → local expert index (identity at EP=1).
            local_idx = self._mx_map_global_to_local_expert(fused_name, expert_id)
            if local_idx is None or local_idx < 0 or local_idx >= int(fused_param.shape[0]):
                return None  # not local to this rank → fallback/skip

            per_expert = fused_param.shape[1]  # axis-0 size of param.data[E]
            rows = hf_shape[0]
            if shard_id == "w1":       # gate → first half
                axis, offset, size = 0, 0, rows
            elif shard_id == "w3":     # up → second half
                axis, offset, size = 0, rows, rows
            else:                       # w2 (down) → full slot
                axis, offset, size = 0, 0, int(per_expert)
            # Shape sanity: the narrowed slot must equal the received tensor.
            if size != rows and shard_id in ("w1", "w3"):
                return None
            if shard_id == "w2" and int(per_expert) != rows:
                return None
            return (fused_param, local_idx, axis, offset, size)
        return None

    def _mx_map_global_to_local_expert(
        self, fused_param_name: str, global_expert_id: int
    ) -> int | None:
        """Map a global expert id to this rank's local slot index.

        Resolves the owning FusedMoE module from the fused param name
        (e.g. ``model.layers.3.mlp.experts.w13_weight`` -> the
        ``...mlp.experts`` module) and calls its
        ``_map_global_expert_id_to_local_expert_id``. Returns the local
        index, ``-1`` if the expert isn't on this rank, or the global id
        unchanged if the module/method can't be resolved (EP=1 identity
        fallback). Cached per fused-param module to avoid re-walking.
        """
        cache = getattr(self, "_mx_moe_module_cache", None)
        if cache is None:
            cache = {}
            self._mx_moe_module_cache = cache
        module = cache.get(fused_param_name, "MISS")
        if module == "MISS":
            mod_path = fused_param_name.rsplit(".", 1)[0]
            obj = self.model_runner.model
            for part in mod_path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            module = obj
            cache[fused_param_name] = module
        if module is not None and hasattr(
            module, "_map_global_expert_id_to_local_expert_id"
        ):
            try:
                return int(
                    module._map_global_expert_id_to_local_expert_id(global_expert_id)
                )
            except Exception:  # noqa: BLE001
                return global_expert_id
        # No EP map available (EP=1 / dense-style): identity.
        return global_expert_id

    def _mx_load_weights(self, weights: list[tuple[str, torch.Tensor]]) -> None:
        """Push refitted weights into the running vLLM model.

        MX_LOAD_MODE=direct enables MDL (Mapped Direct Load): cycle 1
        runs vLLM's stock ``load_weights`` (correct cold start) and
        builds a destination map (see ``_mx_build_fused_dest_map``).
        Cycle 2+ writes every tensor to its precomputed destination —
        ``param.data.copy_`` for 1:1 params, ``param.narrow(...).copy_``
        for stacked members (q/k/v -> qkv_proj, gate/up -> gate_up_proj)
        — with ZERO calls into vLLM's stock loader for anything the
        dest map covers. On dense models the map covers 100% of
        tensors; per-expert MoE tensors (packed N-D vLLM params) route
        to a stock fallback pass.

        MDL is our own eager, destination-mapped in-place write — NOT
        Anyscale RDT (no lazy tensors / deferred narrow / RDT API dep).

        Default OFF (MX_LOAD_MODE unset -> stock loader every cycle).
        """
        mode = os.environ.get("MX_LOAD_MODE", "stock").lower()

        if not hasattr(self, "_mx_param_cache"):
            self._mx_param_cache = None
            self._mx_direct_load_cycles = 0

        # Cycle 1 (or stock mode): stock load, then build cache + map.
        if mode != "direct" or self._mx_param_cache is None:
            _t0 = _time.perf_counter()
            self.model_runner.model.load_weights(weights=weights)
            _t_stock = _time.perf_counter() - _t0

            if mode == "direct":
                self._mx_param_cache = dict(
                    self.model_runner.model.named_parameters()
                )
                self._mx_build_fused_dest_map(weights)
                logger.info(
                    "[mx-mdl] cold-cycle stock load_weights: %.2fs; "
                    "cached %d params",
                    _t_stock, len(self._mx_param_cache),
                )
            return

        # Warm cycle: write each tensor to its precomputed destination.
        direct_hits = 0
        fused_hits = 0
        fallback = []
        _t0 = _time.perf_counter()
        expert_hits = 0
        with torch.no_grad():
            for hf_name, tensor in weights:
                dest = self._mx_mdl_fused.get(hf_name)
                if dest is not None:
                    param, axis, offset, size = dest
                    param.data.narrow(axis, offset, size).copy_(
                        tensor, non_blocking=True
                    )
                    fused_hits += 1
                    continue
                edest = self._mx_mdl_expert.get(hf_name)
                if edest is not None:
                    param, expert_id, axis, offset, size = edest
                    param.data[expert_id].narrow(axis, offset, size).copy_(
                        tensor, non_blocking=True
                    )
                    expert_hits += 1
                    continue
                param = self._mx_mdl_direct.get(hf_name)
                if param is not None and tuple(param.shape) == tuple(tensor.shape):
                    param.data.copy_(tensor, non_blocking=True)
                    direct_hits += 1
                    continue
                fallback.append((hf_name, tensor))

        _t_direct = _time.perf_counter() - _t0

        _t_fb = 0.0
        if fallback:
            _t0 = _time.perf_counter()
            self.model_runner.model.load_weights(weights=fallback)
            _t_fb = _time.perf_counter() - _t0

        self._mx_direct_load_cycles += 1
        logger.info(
            "[mx-mdl] warm-cycle: %d direct + %d fused-slice + %d expert-slice "
            "in %.3fs, %d fallback via stock in %.3fs (cycle %d)",
            direct_hits, fused_hits, expert_hits, _t_direct,
            len(fallback), _t_fb, self._mx_direct_load_cycles,
        )

    def _mx_check_moe_swizzle(self) -> None:
        """Fail loud if a MoE expert param is in a swizzled >3D layout.

        vLLM's ``--moe-backend auto`` selects a batched/packed backend on
        some GPUs (observed on GB200 for Qwen3-MoE) whose
        ``process_weights_after_loading`` repacks ``w13_weight`` into a
        4D kernel layout ``(num_experts, tile, 2*inter, hidden_tile)``.
        The incremental refit ``load_weights`` path can't write raw
        per-expert HF weights back into that swizzled param — it fails
        deep in ``_load_w13`` with an opaque
        ``shard_dim=0 is not a valid data dimension for a 3D tensor``.

        We detect the swizzle up-front (any ``experts...w13_weight`` /
        ``w2_weight`` param with ``ndim > 3``) and raise a directive to
        launch with ``--moe-backend triton`` (standard un-swizzled
        layout), which is our contained fix until vLLM ships a
        refit-aware reload path. See
        pensieve/RL/NemoRL/JulyAlignment/NemoRL_MegaMX_Design.md §3-§5.

        No-op on dense models and on correctly-configured MoE.
        """
        for name, param in self.model_runner.model.named_parameters():
            if ("experts" in name and name.endswith(("w13_weight", "w2_weight"))):
                if param.ndim > 3:
                    raise RuntimeError(
                        f"[mx-megatron] MoE expert param {name!r} is in a "
                        f"swizzled {param.ndim}D layout {tuple(param.shape)}; "
                        f"vLLM's refit load_weights cannot write raw HF "
                        f"weights into it. Relaunch the vLLM worker with "
                        f"'--moe-backend triton' to keep the standard "
                        f"(num_experts, 2*inter, hidden) layout. See "
                        f"NemoRL_MegaMX_Design.md §5 (contained fix)."
                    )
                # First expert param checked is representative; done.
                return

    def _mx_apply_receiver_ep_filter(
        self,
        receive_specs: dict,
    ) -> None:
        """Rewrite ``role_descriptor['local_expert_ids']`` on expert-role
        specs to reflect THIS receiver's EP layout.

        Wired to vLLM's parallel_config (§4.5 of the MX-RL design doc):
        when ``enable_expert_parallel=True``, only the experts routed to
        this inference rank need to be pulled — the planner's
        ``_plan_per_expert`` filters to that set. When EP is disabled
        (default), every rank owns every expert and this method still
        writes the full set (identity result; no filter applied at
        planner time).

        Uses ``modelexpress.rl_expert_layout.compute_local_expert_ids``
        so the placement math (linear vs round_robin) stays in one
        place and matches what a rank-to-rank publisher would advertise
        under the same layout.
        """
        pc = self.model_runner.vllm_config.parallel_config
        # vLLM's EP is enabled via ``enable_expert_parallel``. When on,
        # the effective EP world size equals the TP*PP*DP group's total
        # devices (via get_ep_group); when off, EP is a no-op mesh of 1.
        ep_enabled = bool(getattr(pc, "enable_expert_parallel", False))
        if ep_enabled:
            try:
                from vllm.distributed import parallel_state as _ps
                _ep = _ps.get_ep_group()
                ep_world_size = int(_ep.world_size)
                ep_rank = int(_ep.rank_in_group)
            except Exception:
                # If EP is enabled in config but group isn't up yet,
                # fall back to no filter.
                ep_world_size, ep_rank = 1, 0
        else:
            ep_world_size, ep_rank = 1, 0

        # num_experts from HF config. Present on all MoE architectures
        # under different names; try the common ones. Skip filter if
        # this isn't an MoE model at all.
        hf_cfg = self.model_runner.model_config.hf_config
        num_experts = (
            getattr(hf_cfg, "num_local_experts", None)
            or getattr(hf_cfg, "num_experts", None)
            or getattr(hf_cfg, "n_routed_experts", None)
        )
        if not num_experts:
            return  # not MoE — nothing to filter

        placement = getattr(pc, "expert_placement_strategy", "linear")
        # Only "linear" and "round_robin" are recognised by
        # compute_local_expert_ids; anything else defaults to linear.
        if placement not in ("linear", "round_robin"):
            placement = "linear"

        from modelexpress.rl_expert_layout import compute_local_expert_ids
        local = compute_local_expert_ids(
            ep_rank=ep_rank,
            ep_world_size=ep_world_size,
            num_experts=int(num_experts),
            placement=placement,
        )
        local_str = ",".join(str(e) for e in local)

        # Walk expert-role specs, overwrite the local_expert_ids hint.
        # Non-expert specs are unchanged. This is a receiver-side
        # override — the trainer's own hint (its EP-owned set) stays in
        # the sidecar but the planner uses what we set here.
        touched = 0
        for spec in receive_specs.values():
            if not spec.role.startswith("expert_"):
                continue
            rd = dict(spec.role_descriptor or {})
            rd["local_expert_ids"] = local_str
            spec.role_descriptor = rd
            touched += 1

        logger.info(
            "[mx-megatron] EP filter: ep_enabled=%s ep_rank=%d ep_size=%d "
            "num_experts=%d placement=%s local=%d experts (%s...) "
            "applied to %d expert-role specs",
            ep_enabled, ep_rank, ep_world_size, num_experts, placement,
            len(local), local_str[:60], touched,
        )

    def _mx_verify_byte_identity(
        self,
        weights: list[tuple[str, torch.Tensor]],
        *,
        gt_path: str,
    ) -> None:
        """Compare received HF tensors bitwise against a Bridge ground truth.

        Loads ``gt_path`` (produced by ``bridge.export_hf_weights`` on
        the trainer) as ``{"hf_weights": {name: tensor}, ...}`` OR a bare
        state-dict, and does a per-tensor ``torch.equal`` for every name
        in ``weights``. Logs a summary line

            [mx-verify] byte-identity: N/M tensors match (X mismatches)

        which is grep-able across cycles + workers. On mismatch, logs
        the first-N offending tensor names + shape/dtype/max-abs-diff.

        Only invoked when the ``MX_VERIFY_BYTE_IDENTITY`` env var is set,
        so there's no overhead in the production path. Runs inside the
        vLLM worker process; ``gt_path`` must be visible via a mounted
        volume (e.g. the shared PVC on ``/mnt/rl-workspace``).
        """
        t0 = _time.perf_counter()
        # Use mmap so tensors are backed by the file on disk rather than
        # copied into anonymous RAM up-front. Critical at MoE scale where
        # the GT file is ~60 GB — copying it all into the pod's 128 GB
        # memory limit on top of the pinned-CPU buffer cache (another
        # ~60 GB) OOM-kills the container. With mmap=True torch keeps
        # only page-cache pressure, not per-tensor RSS.
        try:
            gt_blob = torch.load(
                gt_path, map_location="cpu", weights_only=False, mmap=True,
            )
            _mmap_ok = True
        except (TypeError, RuntimeError) as exc:
            # Older torch or non-mmap-compatible pickle format. Fall
            # back to non-mmap load and warn.
            logger.warning(
                "[mx-verify] mmap=True load failed (%s); falling back "
                "to eager load (may OOM on large GTs)",
                exc,
            )
            gt_blob = torch.load(gt_path, map_location="cpu", weights_only=False)
            _mmap_ok = False
        # Accept both the pensieve wrapper ({"hf_weights": {...}, ...})
        # and a bare state-dict.
        if isinstance(gt_blob, dict) and "hf_weights" in gt_blob:
            gt = gt_blob["hf_weights"]
        else:
            gt = gt_blob
        assert isinstance(gt, dict), (
            f"GT file {gt_path!r} did not resolve to a tensor dict "
            f"(got {type(gt).__name__})"
        )
        logger.info(
            "[mx-verify] loaded GT (mmap=%s, %d tensors) in %.2fs",
            _mmap_ok, len(gt), _time.perf_counter() - t0,
        )

        match = 0
        missing_in_gt = 0
        shape_dtype_mismatch = 0
        value_mismatch = 0
        mismatch_examples: list[str] = []
        # Stream compare: pop each GT tensor as we consume it so the
        # (small) receive tensor plus the (streamed) GT page is the
        # only extra live memory. On mmap-backed tensors ``del`` +
        # ``gt.pop`` release the mmap ref immediately.
        for name, recv in weights:
            gt_t = gt.pop(name, None)
            if gt_t is None:
                missing_in_gt += 1
                if len(mismatch_examples) < 5:
                    mismatch_examples.append(f"missing-in-gt: {name}")
                continue
            recv_cpu = recv.detach().to("cpu") if recv.device.type != "cpu" else recv
            if tuple(recv_cpu.shape) != tuple(gt_t.shape) or recv_cpu.dtype != gt_t.dtype:
                shape_dtype_mismatch += 1
                if len(mismatch_examples) < 5:
                    mismatch_examples.append(
                        f"shape/dtype: {name} recv={tuple(recv_cpu.shape)}/{recv_cpu.dtype} "
                        f"gt={tuple(gt_t.shape)}/{gt_t.dtype}"
                    )
                del gt_t
                continue
            if torch.equal(recv_cpu, gt_t):
                match += 1
            else:
                value_mismatch += 1
                if len(mismatch_examples) < 5:
                    diff = (recv_cpu.to(torch.float32) - gt_t.to(torch.float32)).abs()
                    mismatch_examples.append(
                        f"value: {name} shape={tuple(recv_cpu.shape)} "
                        f"max_abs_diff={diff.max().item():.4e} "
                        f"mean_abs_diff={diff.mean().item():.4e}"
                    )
            del gt_t
        total = len(weights)
        mismatches = missing_in_gt + shape_dtype_mismatch + value_mismatch
        elapsed = _time.perf_counter() - t0
        logger.info(
            "[mx-verify] byte-identity: %d/%d tensors match "
            "(%d mismatches: %d missing-in-gt, %d shape/dtype, %d value) "
            "in %.2fs against gt=%s",
            match, total, mismatches, missing_in_gt, shape_dtype_mismatch,
            value_mismatch, elapsed, gt_path,
        )
        if mismatch_examples:
            for line in mismatch_examples:
                logger.info("[mx-verify]   %s", line)

    def _mx_maybe_process_fp8_kv_cache(self) -> None:
        """If the model uses FP8 KV cache, re-run vLLM's weight-loading hook.

        Static FP8 KV scales are computed in ``process_weights_after_loading``;
        they need to be recomputed after every refit so the scales match the
        new weights. Skipped silently for non-FP8 KV cache configurations.
        """
        use_fp8_kv_cache = False
        if hasattr(self.model_runner.vllm_config, "cache_config"):
            kv_cache_dtype = getattr(
                self.model_runner.vllm_config.cache_config, "cache_dtype", None
            )
            use_fp8_kv_cache = (
                kv_cache_dtype is not None and "fp8" in str(kv_cache_dtype).lower()
            )

        if not use_fp8_kv_cache:
            return

        from vllm.model_executor.model_loader.utils import (
            process_weights_after_loading,
        )

        target_device = next(self.model_runner.model.parameters()).device
        process_weights_after_loading(
            self.model_runner.model,
            self.model_runner.model_config,
            target_device,
        )

    # ------------------------------------------------------------------ #
    # The refit RPC entry point
    # ------------------------------------------------------------------ #
    def update_weights_via_mx(
        self,
        *,
        version: int,
        mx_config: Any = None,
    ) -> bool:
        """Receive weights via NIXL RDMA from the MX server (v2 path).

        Mirrors ``VllmInternalWorkerExtension.update_weights_via_mx`` in
        NeMo-RL. The lazy-init path runs on first call per worker; subsequent
        calls reuse the registered NIXL buffers.

        Args:
            version: monotonically-increasing training-step counter; the
                receiver picks sources whose ``training_step >= version``.
            mx_config: an ``MxConfig`` instance or a dict matching its fields.
                The trainer typically sends a dict over the Dynamo Endpoint
                RPC and we parse it here.

        Returns:
            True on successful refit, False on recoverable failures
            (no source found, source doesn't cover required experts).
            Unrecoverable errors are logged with a traceback and return False
            so the caller can decide whether to retry.
        """
        try:
            # Allow dict input — the handler may pass through unparsed JSON
            if not isinstance(mx_config, MxConfig):
                mx_config = MxConfig.from_dict(mx_config or {})

            # ---- Lazy-init receiver (no pre-registered buffers; scratch path) ----
            # We use ``MxRefitReceiver.receive_weights_scratch`` rather than the
            # pre-registered-buffer path because the trainer publishes HF
            # state_dict names (``q_proj``, ``k_proj``, ``v_proj``) but vLLM's
            # internal params are fused (``qkv_proj``). Registering vLLM's
            # ``named_parameters()`` as receive buffers gives a name mismatch
            # that breaks ``model.load_weights`` (it does the HF→fused merge
            # itself, and if you feed it ``qkv_proj`` it produces ``qkqkv_proj``
            # via its stacked_params_mapping). The scratch path allocates temp
            # CUDA buffers sized to the publisher's tensor list, RDMA-pulls
            # into them, and yields ``(hf_name, tensor)`` pairs that
            # ``load_weights`` consumes correctly. Extra GPU memory cost:
            # ~1× model size briefly per refit, freed at end.
            if not getattr(self, "_mx_receiver", None):
                # Import here so workers that never refit via MX don't pay
                # the modelexpress import cost.
                from modelexpress import MxV2RefitReceiver

                rank = (
                    torch.distributed.get_rank()
                    if torch.distributed.is_initialized()
                    else 0
                )
                _pin_local_nic(
                    device_id=self.device.index, mode=mx_config.nic_pin
                )
                self._mx_receiver = MxV2RefitReceiver(  # noqa: SLF001
                    agent_name=f"dynamo-vllm-r{rank}",
                    device_id=self.device.index,
                    mx_server_url=mx_config.mx_server_url,
                    worker_rank=rank,
                )
                self._mx_receiver.initialize(model_tensors=None)
                logger.info(
                    "[mx] receiver initialized (scratch path): rank=%d device=%d",
                    rank,
                    self.device.index,
                )

            # ---- Discover, pick source, RDMA pull ----
            model_name = getattr(
                self.model_runner.vllm_config.model_config, "model", "unknown"
            )
            candidates = self._mx_receiver.discover_v2_sources(
                model_name=model_name,
                min_version=int(version),
                same_rank_only=mx_config.same_rank_only,
                include_replicas=mx_config.tree_scale_out,
            )
            if not candidates:
                logger.warning(
                    "[mx] no v2 source available for version>=%d on rank %d",
                    version,
                    self._mx_receiver.worker_rank,
                )
                return False

            # ---- Megatron-MX dispatch ----
            # Sources whose ``megatron_meta`` is populated come from a
            # Megatron-Core publisher. The HF state-dict layout requires
            # role-aware translation (QKV un-interleave, gated-MLP split,
            # per-expert grouped split) rather than the DTensor path's
            # bulk-pull-of-HF-tensors. Route to the Megatron handler.
            if any(c.megatron_meta is not None for c in candidates):
                return self._update_weights_via_mx_megatron(
                    candidates=candidates,
                    version=int(version),
                    mx_config=mx_config,
                    model_name=model_name,
                )

            chosen = self._mx_receiver.pick_best_source(candidates)
            if chosen is None:
                logger.warning(
                    "[mx] no candidate covers required experts on rank %d",
                    self._mx_receiver.worker_rank,
                )
                return False
            logger.info(
                "[mx] rank=%d chosen role=%s src_rank=%d version=%s",
                self._mx_receiver.worker_rank,
                chosen.role,
                chosen.worker_rank,
                chosen.ref.training_step,
            )

            # Scratch path: allocate temp buffers, RDMA-pull, collect HF-named
            # tensors. ``receive_weights_scratch`` lives on the inner
            # ``MxRefitReceiver``; ``MxV2RefitReceiver.receive_from`` wraps the
            # non-scratch ``receive_weights`` which assumes name parity.
            #
            # Build a tensor_shapes dict from the v2 candidate's registry so
            # the yielded tensors come back with their original shape (not
            # flat 1D). vLLM's ``load_weights`` calls ``.copy_(t)`` into the
            # model param, so shape must match (or .view() must succeed).
            tensor_shapes: dict[str, tuple[int, ...]] = {}
            registry = getattr(chosen, "registry", None)
            if registry:
                for td in registry.get("tensors", []):
                    tensor_shapes[td.name] = tuple(int(s) for s in td.global_shape)
            weights: list[tuple[str, torch.Tensor]] = list(
                self._mx_receiver._receiver.receive_weights_scratch(
                    chosen.ref,
                    timeout_seconds=mx_config.timeout_seconds,
                    tensor_shapes=tensor_shapes or None,
                )
            )

            # ---- vLLM's load_weights handles HF→fused merge ----
            self._mx_load_weights(weights)
            torch.cuda.current_stream().synchronize()
            self._mx_maybe_process_fp8_kv_cache()

            # ---- Tree fan-out: republish self as inference_replica ----
            if mx_config.tree_scale_out:
                try:
                    self._mx_receiver.publish_self_as_source(
                        version=int(version),
                        model_name=model_name,
                    )
                except Exception as exc:  # noqa: BLE001
                    # Non-fatal: refit succeeded, we just can't serve as a
                    # source for downstream receivers this cycle.
                    logger.warning(
                        "[mx] tree-scale-out republish failed: %s", exc
                    )

            gc.collect()
            torch.cuda.empty_cache()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[mx] update_weights_via_mx failed on rank=%d: %s\n%s",
                getattr(
                    getattr(self, "_mx_receiver", None), "worker_rank", -1
                ),
                exc,
                traceback.format_exc(),
            )
            return False

    # ------------------------------------------------------------------ #
    # Megatron-MX path (cluster-validated 2026-06-10 on Qwen3-MoE-30B-A3B:
    # 18 867 / 18 867 HF tensors byte-identical against bridge ground truth)
    # ------------------------------------------------------------------ #
    def _update_weights_via_mx_megatron(
        self,
        *,
        candidates: list,
        version: int,
        mx_config: Any,
        model_name: str,
    ) -> bool:
        """Megatron-MX path of :meth:`update_weights_via_mx`.

        Megatron-Core trainers publish per-rank native shards (no allgather)
        — column-parallel rows / row-parallel cols / fused QKV / fused
        gated-MLP / per-expert grouped tensors / etc. The receiver-side
        translator (``modelexpress.megatron_translator``) assembles those
        shards into HF-shaped tensors via vendored Bridge helpers,
        without taking a Bridge dependency in this worker image.

        Two paths depending on the trainer's TP layout:

          * **matched-TP** (source_tp == target_tp): one source per
            tp_rank; bulk receive_from into pre-allocated dest buffers
            registered with NIXL; translator walks the filled buffers
            directly (no host-side assembly).

          * **mixed-TP** (source_tp != target_tp): per-source sliced
            pull. Each plan's per-source contribution lands directly in
            the planner's pre-narrowed dest view via the v1
            ``MxRefitReceiver.pull_to`` primitive (one combined NIXL
            transfer with N descriptor pairs per source). Row-parallel
            (axis-1 narrows, non-contiguous in memory) falls back to
            v0 scratch + host copy.

        Cluster-validated:
          * Qwen3-4B-Thinking-2507 matched-TP: 398 / 398 byte-identical
          * Qwen3-MoE-30B-A3B-Instruct-2507 matched-TP: 18 867 / 18 867
          * synthetic TP=2 → TP=1 mixed-TP target-narrower: 8 / 8
          * synthetic TP=1 → TP=2 mixed-TP target-wider (v1 sliced-pull): 16 / 16
        """
        import time as _time
        from modelexpress.megatron_translator import (
            MegatronReceiverContext, ReceiveSpec,
            assemble_into_destination, discover_megatron_context,
            run_refit_cycle, translate_megatron_to_hf,
        )
        from modelexpress.nemo_rl_v2 import MegatronTensorSpec, TargetTpLayout

        megatron_cands = [c for c in candidates if c.megatron_meta is not None]
        if not megatron_cands:
            return False

        # Sidecar (transformer_config + Megatron→HF name map).
        sidecar_cfg, name_map = discover_megatron_context(megatron_cands)
        if sidecar_cfg is None:
            logger.warning(
                "[mx-megatron] sources advertise Megatron but no "
                "transformer_config sidecar found; aborting refit"
            )
            return False

        # Receiver's target layout: vLLM TP world × rank.
        target_tp = getattr(
            self.model_runner.vllm_config.parallel_config,
            "tensor_parallel_size", 1,
        )
        target_tp_rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_initialized() else 0
        )
        layout = TargetTpLayout(tp_size=target_tp, tp_rank=target_tp_rank)

        # Build ReceiveSpecs from candidate registries (union — replicated
        # tensors may only be published by rank 0).
        SHARD_AXIS_BY_ROLE = {
            "column": 0, "qkv_column": 0, "gated_mlp_column": 0,
            "vocab_parallel": 0, "row": 1,
            "expert_column": 0, "expert_row": 0, "replicated": 0,
        }
        receive_specs: dict[str, ReceiveSpec] = {}
        source_tp_size = max(
            c.megatron_meta.tp_size for c in megatron_cands
            if c.megatron_meta.tp_size > 0
        )
        for c in megatron_cands:
            for td in (c.registry.get("tensors", []) if c.registry else []):
                if not td.megatron_role or td.name in receive_specs:
                    continue
                role = td.megatron_role
                shard_axis = SHARD_AXIS_BY_ROLE.get(role, int(td.shard_axis))
                per_rank_shape = list(td.global_shape)
                global_shape = list(per_rank_shape)
                if role != "replicated":
                    global_shape[shard_axis] = (
                        per_rank_shape[shard_axis] * source_tp_size
                    )
                lookup_name = (
                    td.name[len("module."):]
                    if td.name.startswith("module.") else td.name
                )
                hf_names = name_map.get(
                    lookup_name, name_map.get(td.name, [td.name])
                )
                receive_specs[td.name] = ReceiveSpec(
                    megatron_name=td.name,
                    hf_names=list(hf_names),
                    role=role,
                    target_shape=tuple(int(s) for s in global_shape),
                    target_dtype=td.dtype or "bfloat16",
                    shard_axis=shard_axis,
                    pp_rank=c.megatron_meta.pp_rank,
                    role_descriptor=dict(td.megatron_extras or {}),
                )

        logger.info(
            "[mx-megatron] %d ReceiveSpecs built; source_tp=%d target_tp=%d",
            len(receive_specs), source_tp_size, target_tp,
        )

        # Inference-side EP filter wire-up (§4.5 of the design doc).
        # For expert-role specs, overwrite ``role_descriptor['local_expert_ids']``
        # with the RECEIVER's local expert set — derived from vLLM's
        # ``parallel_config``. The trainer's advertised set (via
        # ``td.megatron_extras``) reflects trainer-side ownership; we
        # need receiver-side ownership so ``pick_megatron_slice_plans``
        # only asks for experts THIS inference rank will actually route
        # to (see ``_plan_per_expert`` — filters ``wanted`` to a subset
        # of every source's ``owned_experts_per_layer``).
        #
        # No-op when ``enable_expert_parallel=False`` (EP=1) — every
        # inference rank owns every expert, receiver == trainer set.
        # Meaningful only when EP > 1.
        try:
            self._mx_apply_receiver_ep_filter(receive_specs)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[mx-megatron] receiver EP filter skipped (%s); "
                "falling back to trainer-published local_expert_ids",
                exc,
            )

        # Fail loud (with the fix in the message) if a MoE expert param
        # is swizzled into a layout refit can't write into. Contained
        # fix: --moe-backend triton. See NemoRL_MegaMX_Design.md §5.
        self._mx_check_moe_swizzle()

        # Matched-TP fast path: bulk receive_from into pre-allocated buffers.
        matched = next(
            (c for c in megatron_cands
             if c.megatron_meta.tp_rank == layout.tp_rank
             and c.megatron_meta.tp_size == layout.tp_size),
            None,
        )
        is_matched_tp = matched is not None

        device = self.device
        dt_map = {
            "bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32,
        }

        # MX_MEGATRON_BUFFER_LOC picks where the persistent NIXL-registered
        # cache lives (Istvan Phase 0.5):
        #   "device" (default): allocate on self.device (HBM). Historical
        #     behavior. Cache size ≈ model shard, competes with vLLM
        #     weights + KV cache for HBM. On 190 GB HBM this OOMs at
        #     ~30B MoE and above.
        #   "host": allocate pinned CPU. NIXL registers as DRAM, RDMA
        #     writes into pinned host, then _load_weights ->
        #     param.copy_(non_blocking=True) triggers async H2D copies
        #     overlapping with the rest of the pipeline. Frees the HBM
        #     that the cache would have occupied.
        _buffer_loc = os.environ.get("MX_MEGATRON_BUFFER_LOC", "device").lower()
        if _buffer_loc == "host":
            _alloc_kwargs: dict = {"pin_memory": True}
        elif _buffer_loc == "device":
            _alloc_kwargs = {"device": device}
        else:
            raise ValueError(
                f"MX_MEGATRON_BUFFER_LOC={_buffer_loc!r} not recognized; "
                f"expected 'device' or 'host'"
            )

        weights: list[tuple[str, torch.Tensor]] = []

        if is_matched_tp:
            # Pre-allocate + NIXL-register the per-rank buffers ONCE per
            # worker lifetime, not once per refit cycle. Cluster-validated
            # 2026-06-23: on Qwen3-4B-Thinking, register_tensors costs
            # ~0.15s per cycle (paid by every refit); caching it drops
            # warm-cycle total from 0.39s to 0.21s (-45%). On Llama 3.1 8B
            # at 16 GB the savings scales with the registration cost
            # (~0.3-0.5s per cycle saved).
            #
            # The receive_specs are deterministic given the source's TP
            # layout, so cycle-N allocations would be identical to
            # cycle-1's. Cache the dict + reuse for receive_from.
            buffers = getattr(self, "_mx_megatron_buffers", None)
            if buffers is None:
                buffers = {}
                for spec in receive_specs.values():
                    dt = dt_map.get(spec.target_dtype, torch.bfloat16)
                    # Receiver's per-rank window — for matched-TP that's the
                    # source's natural shard along the role's shard axis.
                    full_shape = list(spec.target_shape)
                    if spec.role != "replicated" and target_tp > 1:
                        axis_extent = full_shape[spec.shard_axis]
                        per_rank = axis_extent // target_tp
                        full_shape[spec.shard_axis] = (
                            axis_extent if layout.tp_rank == target_tp - 1
                            else per_rank
                        )
                    if spec.role.startswith("expert_"):
                        # Grouped-MoE per-expert tensors are passthrough — the
                        # source-side per-expert shape IS the target.
                        pass
                    buffers[spec.megatron_name] = torch.empty(
                        full_shape, dtype=dt, **_alloc_kwargs,
                    )
                self._mx_receiver._receiver._nixl.register_tensors(buffers)
                self._mx_megatron_buffers = buffers
                logger.info(
                    "[mx-megatron] matched-TP: ALLOCATED + registered %d buffers (%.2f GB, loc=%s) "
                    "[first cycle; cached for subsequent refits]",
                    len(buffers),
                    sum(b.numel() * b.element_size() for b in buffers.values()) / 1e9,
                    _buffer_loc,
                )
            else:
                # Safety: rebind before transfer in case another code path
                # (e.g. mixed-TP v0 scratch on a prior worker call) swapped
                # self._tensors underneath us. rebind_tensors is a no-op
                # if the NIXL agent is already pointed at these buffers.
                self._mx_receiver._receiver._nixl.rebind_tensors(buffers)
                logger.info(
                    "[mx-megatron] matched-TP: reusing %d cached buffers (%.2f GB, loc=%s)",
                    len(buffers),
                    sum(b.numel() * b.element_size() for b in buffers.values()) / 1e9,
                    _buffer_loc,
                )
            t0 = _time.perf_counter()
            for _name, _t in self._mx_receiver.receive_from(
                matched, timeout_seconds=mx_config.timeout_seconds,
            ):
                pass
            elapsed = _time.perf_counter() - t0
            logger.info(
                "[mx-megatron] matched-TP bulk receive_from: %.2fs", elapsed,
            )

            ctx = MegatronReceiverContext(
                target_tp_layout=layout,
                transformer_config=sidecar_cfg,
                hf_name_map=name_map,
                receive_specs=receive_specs,
            )
            for hf_name, hf_tensor in run_refit_cycle(
                self._mx_receiver,
                candidates=megatron_cands,
                context=ctx,
                pull=lambda src, dest: None,
                device=device,
                pre_assembled_buffers=buffers,
            ):
                weights.append((hf_name, hf_tensor))
        else:
            # Mixed-TP: v1 sliced-pull where dest narrow is contiguous,
            # v0 scratch+copy otherwise. Mirrors NeMo-RL's
            # vllm_backend.py::_update_weights_via_mx_megatron mixed-TP
            # branch.
            target_specs = {
                m_name: MegatronTensorSpec(
                    role=rs.role, target_shape=rs.target_shape,
                    target_dtype=rs.target_dtype, shard_axis=rs.shard_axis,
                    pp_rank=rs.pp_rank,
                    role_descriptor=dict(rs.role_descriptor or {}),
                )
                for m_name, rs in receive_specs.items()
            }
            plans = self._mx_receiver.pick_megatron_slice_plans(
                megatron_cands, target_tp_layout=layout,
                target_tensor_specs=target_specs,
            )

            # Cache plan_dests across refit cycles. Plan shapes are
            # deterministic for a fixed source TP layout + target TP
            # layout; re-allocating + re-registering NIXL buffers every
            # cycle is the bug surfaced by John's 16-receiver Llama 3.1
            # benchmark (2026-06-22). v1 sliced-pull writes directly into
            # these dest views, so cached buffers stay live across pulls.
            cached_plan_dests: dict[str, torch.Tensor] | None = getattr(
                self, "_mx_megatron_plan_dests", None,
            )
            plan_dests: dict[str, torch.Tensor] = cached_plan_dests or {}
            v1_batches: dict[str, list] = {c.ref.mx_source_id: [] for c in megatron_cands}
            v0_plans: list = []
            newly_allocated_this_cycle = 0

            for plan in plans:
                if not plan.sources:
                    continue
                rs = receive_specs[plan.tensor_name]
                if plan.assembly == "per_expert":
                    v0_plans.append(plan)
                    continue
                dt = dt_map.get(rs.target_dtype, torch.bfloat16)
                if plan.tensor_name in plan_dests:
                    dest = plan_dests[plan.tensor_name]
                else:
                    dest = torch.empty(
                        plan.target_shape, dtype=dt, **_alloc_kwargs,
                    )
                    plan_dests[plan.tensor_name] = dest
                    newly_allocated_this_cycle += 1
                axis = 1 if plan.assembly == "concat_dim1" else 0
                routed_v1 = True
                for src in plan.sources:
                    target_lo, target_hi = src.target_local_range
                    dest_view = dest.narrow(axis, target_lo, target_hi - target_lo)
                    if not dest_view.is_contiguous():
                        routed_v1 = False
                        break
                    v1_batches[src.mx_source_id].append(
                        (plan.tensor_name, src.source_subslice, dest_view)
                    )
                if not routed_v1:
                    # Don't drop cached entries — they may be valid for
                    # other plans; just route this plan to v0.
                    if cached_plan_dests is None:
                        plan_dests.pop(plan.tensor_name, None)
                    for sid in v1_batches:
                        v1_batches[sid] = [
                            r for r in v1_batches[sid] if r[0] != plan.tensor_name
                        ]
                    v0_plans.append(plan)

            # Only register NIXL if we have NEW allocations. If everything
            # is cached, skip the register call entirely.
            if newly_allocated_this_cycle > 0 and plan_dests:
                self._mx_receiver._receiver._nixl.register_tensors(plan_dests)
                self._mx_megatron_plan_dests = plan_dests
                logger.info(
                    "[mx-megatron] mixed-TP: registered %d plan_dests "
                    "(%d newly allocated this cycle, loc=%s)",
                    len(plan_dests), newly_allocated_this_cycle, _buffer_loc,
                )
            elif plan_dests:
                # Cache hit — no register call this cycle. Rebind so
                # pull_to's local descriptors resolve against the cached
                # buffers, and self._local_mem_type stays correct for
                # DRAM/CUDA distinction on the transfer's local side.
                self._mx_receiver._receiver._nixl.rebind_tensors(plan_dests)
            n_v1_slices = sum(len(b) for b in v1_batches.values())
            logger.info(
                "[mx-megatron] mixed-TP: %d v1 slices across %d sources "
                "(plans: %d v1, %d v0)",
                n_v1_slices, sum(1 for b in v1_batches.values() if b),
                len(plan_dests), len(v0_plans),
            )

            for cand in megatron_cands:
                batch = v1_batches[cand.ref.mx_source_id]
                if not batch:
                    continue
                self._mx_receiver._receiver.pull_to(
                    cand.ref, batch,
                    timeout_seconds=mx_config.timeout_seconds,
                )

            scratch: dict[str, dict[str, torch.Tensor]] = {}
            if v0_plans:
                v0_names_by_source: dict[str, set[str]] = {}
                for plan in v0_plans:
                    for src in plan.sources:
                        v0_names_by_source.setdefault(src.mx_source_id, set()).add(
                            plan.tensor_name
                        )
                scratch_bytes = 0
                for cand in [
                    c
                    for c in megatron_cands
                    if c.ref.mx_source_id in v0_names_by_source
                ]:
                    include_names = v0_names_by_source[cand.ref.mx_source_id]
                    tensor_shapes = {
                        td.name: tuple(int(dim) for dim in td.global_shape)
                        for td in (
                            cand.registry.get("tensors", [])
                            if cand.registry
                            else []
                        )
                        if td.name in include_names and tuple(td.global_shape)
                    }
                    buf_dict: dict[str, torch.Tensor] = {}
                    for name, t in self._mx_receiver._receiver.receive_weights_scratch(
                        cand.ref, timeout_seconds=mx_config.timeout_seconds,
                        tensor_shapes=tensor_shapes or None,
                        include_names=include_names,
                    ):
                        buf_dict[name] = t
                        scratch_bytes += t.numel() * t.element_size()
                    scratch[cand.ref.mx_source_id] = buf_dict
                logger.info(
                    "[mx-megatron] mixed-TP v0 fallback: %d plans, %d "
                    "source-tensor ranges, %d wire bytes across %d sources",
                    len(v0_plans),
                    sum(len(names) for names in v0_names_by_source.values()),
                    scratch_bytes,
                    len(v0_names_by_source),
                )

            ctx = MegatronReceiverContext(
                target_tp_layout=layout,
                transformer_config=sidecar_cfg,
                hf_name_map=name_map,
                receive_specs=receive_specs,
            )
            for plan in plans:
                if not plan.sources:
                    continue
                rs = receive_specs[plan.tensor_name]
                if plan.tensor_name in plan_dests:
                    assembled = plan_dests[plan.tensor_name]
                else:
                    def _pull_factory(name=plan.tensor_name, assembly=plan.assembly):
                        def _pull(src, dest):
                            full = scratch.get(src.mx_source_id, {}).get(name)
                            if full is None:
                                raise RuntimeError(
                                    f"mixed-TP v0: scratch missing {name!r} from "
                                    f"source {src.mx_source_id}"
                                )
                            axis = 1 if assembly == "concat_dim1" else 0
                            if src.source_subslice is not None:
                                slo, shi = src.source_subslice
                                slice_src = full.narrow(axis, slo, shi - slo)
                            else:
                                slice_src = full
                            dest.copy_(slice_src, non_blocking=True)
                        return _pull
                    assembled = assemble_into_destination(
                        plan, pull=_pull_factory(), device=device,
                    )
                for hf_name, hf_tensor in translate_megatron_to_hf(
                    plan, assembled,
                    transformer_config=ctx.transformer_config,
                    hf_names=list(rs.hf_names),
                ):
                    weights.append((hf_name, hf_tensor))

        if not weights:
            logger.warning("[mx-megatron] cycle yielded 0 tensors; refit aborted")
            return False
        logger.info(
            "[mx-megatron] yielded %d HF tensors; calling vLLM load_weights",
            len(weights),
        )

        # Phase 1 byte-identity verifier hook. Opt-in via
        # MX_VERIFY_BYTE_IDENTITY=<path-to-gt-pt>. Loads the ground-truth
        # dict of {hf_name: hf_tensor} (as produced by Bridge's
        # ``export_hf_weights``) and compares every tensor received in
        # this cycle against it BEFORE handing to vLLM's load_weights.
        # This is the correctness gate for the Megatron -> HF conversion
        # path; it isolates any bug in the receiver-side translation
        # from downstream vLLM quantization/reshaping effects.
        #
        # Prints a match/total summary to the extension log so callers
        # can grep for it. Non-fatal on mismatch — the refit still
        # completes so we can compare match rates across configurations.
        _gt_path = os.environ.get("MX_VERIFY_BYTE_IDENTITY")
        if _gt_path:
            try:
                self._mx_verify_byte_identity(weights, gt_path=_gt_path)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "[mx-verify] byte-identity check failed with exception: %s",
                    exc,
                )

        self._mx_load_weights(weights)
        torch.cuda.current_stream().synchronize()
        self._mx_maybe_process_fp8_kv_cache()

        if mx_config.tree_scale_out:
            try:
                self._mx_receiver.publish_self_as_source(
                    version=version, model_name=model_name,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[mx-megatron] tree-scale-out republish failed: %s", exc,
                )

        gc.collect()
        torch.cuda.empty_cache()
        return True

    # ------------------------------------------------------------------ #
    # Receiver-side polling — the trainer publishes to MX without sending
    # any trigger RPC, so the worker watches the MX server itself and
    # refits whenever a newer version appears for its model_name.
    # ------------------------------------------------------------------ #
    def start_mx_refit_poller(
        self,
        *,
        mx_config: Any = None,
        poll_interval_s: float = 5.0,
    ) -> bool:
        """Spawn a background thread that watches MX for new versions.

        Called once per worker at startup (or on first publish-detection),
        from the dynamo handler. The thread:
          1. Builds an MxConfig (defaults are fine for the smoke).
          2. Loops on ``discover_v2_sources(min_version=last_seen+1)``.
          3. When a new version appears, calls ``update_weights_via_mx``
             with the new version. That method's lazy-init flow registers
             NIXL buffers on first call.
          4. Sleeps ``poll_interval_s`` between polls.

        Returns True if the thread was started (or was already running).
        Idempotent — repeated calls are no-ops.
        """
        if getattr(self, "_mx_poller_thread", None) is not None:
            return True

        import threading

        cfg = (
            mx_config
            if isinstance(mx_config, MxConfig)
            else MxConfig.from_dict(mx_config or {})
        )
        self._mx_poller_stop = threading.Event()
        self._mx_poller_last_version: int = 0
        self._mx_poller_cfg = cfg
        self._mx_poller_interval = float(poll_interval_s)

        def _poll_loop() -> None:
            from modelexpress import MxV2RefitReceiver

            model_name = getattr(
                self.model_runner.vllm_config.model_config, "model", "unknown"
            )
            rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_initialized()
                else 0
            )
            logger.info(
                "[mx-poller] started: rank=%d model=%s interval=%.1fs",
                rank,
                model_name,
                self._mx_poller_interval,
            )
            # Lazy receiver just for discovery — the refit path lazy-inits
            # its own receiver against the same MX server. We call
            # ``initialize(model_tensors=None)`` to wire the NIXL agent +
            # gRPC client without registering receive buffers (we don't
            # use this receiver to pull; only to poll for new versions).
            discover_only = MxV2RefitReceiver(
                agent_name=f"dynamo-vllm-poller-r{rank}",
                device_id=self.device.index,
                mx_server_url=cfg.mx_server_url,
                worker_rank=rank,
            )
            try:
                discover_only.initialize(model_tensors=None)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "[mx-poller] discover-receiver initialize() failed: %s; "
                    "polling thread will not start", exc,
                )
                return
            while not self._mx_poller_stop.is_set():
                try:
                    candidates = discover_only.discover_v2_sources(
                        model_name=model_name,
                        min_version=int(self._mx_poller_last_version) + 1,
                        same_rank_only=cfg.same_rank_only,
                        include_replicas=cfg.tree_scale_out,
                    )
                    if candidates:
                        latest = max(
                            int(c.ref.training_step) for c in candidates
                        )
                        logger.info(
                            "[mx-poller] new version detected: %d (last=%d)",
                            latest,
                            self._mx_poller_last_version,
                        )
                        ok = self.update_weights_via_mx(
                            version=latest, mx_config=cfg
                        )
                        if ok:
                            self._mx_poller_last_version = latest
                            logger.info(
                                "[mx-poller] refit OK to version %d", latest
                            )
                        else:
                            logger.warning(
                                "[mx-poller] refit failed for version %d; will retry",
                                latest,
                            )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[mx-poller] poll error: %s", exc)
                self._mx_poller_stop.wait(self._mx_poller_interval)
            logger.info("[mx-poller] stopped on rank=%d", rank)

        self._mx_poller_thread = threading.Thread(
            target=_poll_loop, name="mx-refit-poller", daemon=True
        )
        self._mx_poller_thread.start()
        return True
