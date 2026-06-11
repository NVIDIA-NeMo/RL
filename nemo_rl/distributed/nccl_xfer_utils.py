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

"""Refit metadata builders and lightweight wrapper types for nccl_xfer.

This module provides:
- MeshInfo: lightweight DeviceMesh-compatible wrapper that doesn't require a
  shared torch.distributed process group (needed for cross-world transfers)
- Placement rules: mapping param names to TP/EP sharding strategies
- build_nccl_xfer_refit_info: compute per-layer param metadata for refit
- restore_refit_info_placements: undo msgspec dict-flattening of placements
  and meshes on the receiving side

The transfer kernel (``xferdtensor``) and its ``DTensorRef`` src/dst wrapper
live in ``nemo_rl/distributed/xferdtensor.py`` — import both from there.
"""

import re
from collections import OrderedDict
from typing import Any, Optional

import torch
from torch.distributed._tensor import Shard
from torch.distributed.tensor.placement_types import Replicate

# =========================================================================
# MeshInfo — lightweight mesh wrapper type
# =========================================================================


class MeshInfo:
    """Lightweight mesh metadata compatible with xferdtensor.

    Provides the same .mesh / ._mesh interface as DeviceMesh but without
    requiring torch.distributed process groups -- allowing xferdtensor
    to read mesh topology across separate torch.distributed worlds.
    """

    def __init__(self, rank_tensor: torch.Tensor):
        self.mesh = rank_tensor
        self._mesh = rank_tensor

    @property
    def ndim(self):
        return self.mesh.ndim


# =========================================================================
# Placement rules (from xferdtensor/src/placement_rules.py)
# =========================================================================

# Column-parallel suffixes: TP shards along dim 0 (output dimension).
# FP8 ``_scale_inv`` siblings are NOT listed here — they take the misc
# refit path (see ``is_misc_param``), so vLLM's load_weights handles the
# Parameter-specific FP8 blockwise quant layout.
COLUMN_PARALLEL_SUFFIXES = [
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "gate_proj.weight",
    "up_proj.weight",
    # DeepSeek MLA up-projections (column-parallel on output: num_heads * head_dim)
    "q_b_proj.weight",
    "kv_b_proj.weight",
    # NOTE: grouped MoE expert gate_proj/up_proj are column-parallel too; they
    # match the gate_proj.weight/up_proj.weight suffixes above via
    # _get_expert_tp_shard_dim (which, unlike get_tp_shard_dim, does not skip
    # .experts. names).  The vLLM-specific w13_weight fusion happens gen-side.
]

# DeepSeek MLA down-projections.  vLLM creates these as ReplicatedLinear (or
# merges them into ``fused_qkv_a_proj`` with ``disable_tp=True``), so the full
# tensor lives on every TP rank.  Megatron with TP=1 also keeps them
# unsharded, so leaving them out of COLUMN_PARALLEL_SUFFIXES gives the right
# "all Replicate" placement on both sides.

# Row-parallel suffixes: TP shards along dim 1 (input dimension).
ROW_PARALLEL_SUFFIXES = [
    "o_proj.weight",
    "down_proj.weight",  # also grouped MoE expert down_proj (row-parallel, gen-side -> w2_weight)
    # NemotronH Mamba2 output projection
    "out_proj.weight",
]

# Vocabulary-parallel: TP shards along dim 0 (vocab dimension).
VOCAB_PARALLEL_NAMES = [
    "embed_tokens.weight",
    "lm_head.weight",
    # NemotronH HF token embedding (vLLM: model.embed_tokens.weight).
    "embeddings.weight",
]


def get_tp_shard_dim(param_name: str) -> Optional[int]:
    """Return the tensor dimension to shard for TP, or None if replicated."""
    # MoE expert params use EP, not TP
    if ".experts." in param_name:
        return None
    # MoE router gate is always replicated
    if (
        param_name.endswith("mlp.gate.weight")
        or "e_score_correction_bias" in param_name
    ):
        return None

    for suffix in COLUMN_PARALLEL_SUFFIXES:
        if param_name.endswith(suffix):
            return 0
    for suffix in ROW_PARALLEL_SUFFIXES:
        if param_name.endswith(suffix):
            return 1
    for name in VOCAB_PARALLEL_NAMES:
        if param_name.endswith(name):
            return 0
    return None


def is_expert_param(param_name: str) -> bool:
    """Return True if the parameter is a MoE expert weight (sharded by EP)."""
    return ".experts." in param_name


def is_misc_param(param_name: str, qkv_to_misc: bool = False) -> bool:
    """Return True if the parameter should be routed to the misc refit path.

    Misc params are all-gathered to their full HF shape on the train side and
    loaded via vLLM's ``model.load_weights`` on the gen side, letting vLLM
    handle param-specific quirks (FP8 blockwise layout, Mamba/MoE sharding,
    ``A_log -> A``) that xferdtensor's uniform column/row sharding can't
    reproduce.  Everything else takes the xferdtensor bulk fast path.

    The same classifier runs on both the gen-side HF names and the train-side
    Megatron names (Bridge preserves these suffixes through ``megatron_to_hf``).
    It MUST classify each param identically on both sides, or the bulk/misc
    split desyncs and packed_broadcast deadlocks.

    ``qkv_to_misc`` routes attention QKV to the misc path too.  The bulk path
    splits the fused QKV on each TP-local shard (split_qkv_weights), which needs
    every rank to own whole K/V heads; when ``tp_size > num_query_groups`` (GQA
    with few KV heads) Megatron channel-splits the KV heads and that local split
    is impossible.
    """
    if qkv_to_misc and (
        # Megatron fused QKV *weight* only — not the fused-in input layernorm
        # (``linear_qkv.layer_norm_weight``), which maps to an HF norm and must
        # stay on whichever path its HF name selects.
        param_name.endswith("linear_qkv.weight")
        or param_name.endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight"))
    ):
        return True
    # FP8 blockwise scale siblings + KV-cache/activation scales.
    if param_name.endswith("_scale_inv"):
        return True
    if param_name.endswith(
        (".k_scale", ".v_scale", ".q_scale", ".weight_scale", ".input_scale")
    ):
        return True
    # MoE router/gate Linear: vLLM blockwise-FP8 quantizes it to fp8 (+scale)
    # while Megatron keeps it bf16 — that dtype mismatch would deadlock the
    # xferdtensor collective, so let vLLM's load_weights quantize it.  Tiny and
    # not an expert weight, so misc costs negligible bandwidth.  Match the HF
    # forms (``mlp.gate.weight``, NemotronH ``mixer.gate.weight``) and the
    # Megatron form (``mlp.router.weight``) so both sides classify it the same.
    if param_name.endswith(
        ("mlp.gate.weight", "mlp.router.weight", "mixer.gate.weight")
    ):
        return True
    # NemotronH Mamba2 mixer params:
    if param_name.endswith(
        (
            "mixer.in_proj.weight",
            "mixer.conv1d.weight",
            "mixer.conv1d.bias",
            "mixer.A_log",
            "mixer.A",
            "mixer.D",
            "mixer.dt_bias",
            "mixer.norm.weight",
        )
    ):
        return True
    return False


def _get_expert_tp_shard_dim(param_name: str) -> Optional[int]:
    """Like get_tp_shard_dim but does NOT skip .experts. params."""
    for suffix in COLUMN_PARALLEL_SUFFIXES:
        if param_name.endswith(suffix):
            return 0
    for suffix in ROW_PARALLEL_SUFFIXES:
        if param_name.endswith(suffix):
            return 1
    return None


# =========================================================================
# Dtype lookup (used when receiving msgspec'd dtype names over the wire)
# =========================================================================

_STR_TO_DTYPE = {
    "torch.bfloat16": torch.bfloat16,
    "torch.float16": torch.float16,
    "torch.float32": torch.float32,
    "torch.float8_e4m3fn": torch.float8_e4m3fn,
    "torch.float8_e5m2": torch.float8_e5m2,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
}


# =========================================================================
# Placement restoration (undo msgspec dict-serialization on the wire)
# =========================================================================


def _restore_placement(p):
    """Reconstruct a single placement object after msgspec round-trip.

    vLLM's collective_rpc serializes ``Shard(N)`` to ``{"dim": N}`` and
    ``Replicate()`` to ``{}``; this restores them to real instances. No-op
    if the input is already a Shard/Replicate.
    """
    if isinstance(p, (Shard, Replicate)):
        return p
    if isinstance(p, dict):
        if "dim" in p:
            return Shard(p["dim"])
        return Replicate()
    return Replicate()


def restore_refit_info_placements(refit_info: dict) -> dict:
    """Restore placements and meshes in ``refit_info`` after msgspec transit.

    vLLM's collective_rpc encodes ``Shard``/``Replicate`` as plain dicts and
    ``MeshInfo`` as a dict with a nested mesh list.  This rebuilds the
    original Python objects in place so that the canonical
    ``xferdtensor`` (which relies on ``isinstance(p, Shard)``) works
    correctly.  Idempotent — safe to call on already-restored ``refit_info``.
    """
    for layer_name in refit_info.get("layer_names", []):
        for param_info in refit_info.get("per_layer_params", {}).get(layer_name, []):
            param_info["src_placements"] = [
                _restore_placement(p) for p in param_info["src_placements"]
            ]
            param_info["dst_placements"] = [
                _restore_placement(p) for p in param_info["dst_placements"]
            ]
            # Reconstruct MeshInfo if serialized to dict
            for key in ("src_mesh_info", "dst_mesh_info"):
                mesh = param_info.get(key)
                if mesh is not None and not isinstance(mesh, MeshInfo):
                    if isinstance(mesh, dict) and "mesh" in mesh:
                        mesh_tensor = mesh["mesh"]
                        if not isinstance(mesh_tensor, torch.Tensor):
                            mesh_tensor = torch.tensor(mesh_tensor)
                        param_info[key] = MeshInfo(mesh_tensor)
    return refit_info


# =========================================================================
# Mesh and placement construction
# =========================================================================


def build_mesh_info(
    num_gpus: int,
    rank_offset: int,
    tp_size: int = 1,
    ep_size: int = 1,
    pp_size: int = 1,
) -> tuple:
    """Build a ``MeshInfo`` and *dim_map* from a parallelism config.

    Dims are emitted in the order ``(tp, ep, dp, pp)``, size-1 dims are
    dropped, and the survivors are reversed into the row-major rank tensor
    (outer->inner).  So the *first* surviving dim in that order becomes the
    **innermost** (rightmost, fastest-varying) axis — consecutive global ranks
    differ in it.

    Callers never activate EP and TP in the same mesh:
    ``_build_train_src_meshes`` builds a separate non-expert mesh
    (``ep_size=1``) and expert mesh (``tp_size=1``).  So the innermost active
    dim — and the coord a modulo recovers — is:

      * **TP** in the non-expert mesh (EP dropped): ``global_rank % tp_size``
        recovers the TP coord, the standard Megatron non-expert layout.
      * **EP** in the expert mesh (TP dropped): ``global_rank % ep_size``
        recovers the EP coord, Megatron-Core's MoE rank layout.

    (Were EP and TP ever active together, the emit order would make TP — not
    EP — innermost; that case does not arise today.)

    Returns:
        ``(MeshInfo, dim_map)`` where *dim_map* maps ``"tp"``/``"ep"``/``"dp"``/``"pp"``
        to the corresponding mesh-tensor axis index.
    """
    dp_size = num_gpus // (tp_size * ep_size * pp_size)
    assert dp_size * tp_size * ep_size * pp_size == num_gpus, (
        f"Cannot divide {num_gpus} GPUs into TP={tp_size} EP={ep_size} PP={pp_size} DP={dp_size}"
    )

    dim_sizes = {"tp": tp_size, "ep": ep_size, "dp": dp_size, "pp": pp_size}
    active_dims = [
        (n, dim_sizes[n]) for n in ("tp", "ep", "dp", "pp") if dim_sizes[n] > 1
    ]

    if not active_dims:
        return MeshInfo(torch.arange(rank_offset, rank_offset + num_gpus)), {}

    # Reverse to outer->inner for the row-major rank tensor
    active_dims_rev = list(reversed(active_dims))
    mesh_shape = [s for _, s in active_dims_rev]
    dim_map = {name: i for i, (name, _) in enumerate(active_dims_rev)}
    ranks = torch.arange(rank_offset, rank_offset + num_gpus).reshape(mesh_shape)
    return MeshInfo(ranks), dim_map


def get_placements(param_name: str, dim_map: dict, ndim: int) -> list:
    """Determine DTensor placements for a parameter given a *dim_map*.

    1-D params (layernorm, bias) are always fully replicated.
    Expert params shard dim 0 on EP; their TP shard dims are shifted by +1.
    """
    num_mesh_dims = len(dim_map) or 1
    placements = [Replicate() for _ in range(num_mesh_dims)]

    if ndim < 2:
        # For 1-D params, it assumes that it is always fully replicated.
        return placements

    if is_expert_param(param_name):
        if "ep" in dim_map:
            placements[dim_map["ep"]] = Shard(0)
        else:
            # We currently never shards both EP and TP for the expert params
            # so far. If MCore etp is enabled, the logic should be changed
            tp_dim = _get_expert_tp_shard_dim(param_name)
            if tp_dim is not None and "tp" in dim_map:
                placements[dim_map["tp"]] = Shard(tp_dim + 1)
    else:
        # for non-expert params
        tp_dim = get_tp_shard_dim(param_name)
        if tp_dim is not None and "tp" in dim_map:
            placements[dim_map["tp"]] = Shard(tp_dim)

    return placements


# =========================================================================
# MoE expert param fusion
# =========================================================================

# Matches individual expert params: model.layers.X.mlp.experts.Y.proj.weight
# Anchored with ``$`` so it doesn't prefix-match FP8 ``_scale_inv`` siblings
# (scale_inv siblings take the misc refit path via ``is_misc_param`` and
# must not appear in the per-expert weight fusion groups).
_INDIVIDUAL_EXPERT_RE = re.compile(
    r"(.+\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)


def group_expert_params_in_metadata(
    state_dict_metadata: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Group per-expert MoE params into backend-agnostic grouped HF entries.

    For each MoE projection, stack every expert's HF param into ONE entry along
    a new leading expert dim, keeping the *HF* projection name (no backend
    fusion, no w13/w2, no gen-TP / interleave):
      - gate_proj : [E, intermediate, hidden]
      - up_proj   : [E, intermediate, hidden]
      - down_proj : [E, hidden, intermediate]
    Each grouped entry is tagged ``grouped_expert_proj`` ("gate_proj" /
    "up_proj" / "down_proj").  The train side stacks the matching per-expert
    tensors (``_group_experts``); the gen backend maps these grouped HF params
    into ITS fused layout entirely on the gen side (for vLLM: gate/up ->
    w13_weight halves, down -> w2_weight; see
    ``vllm_backend._build_hf_to_vllm_mapping``).

    Grouping is universal — every gen backend needs experts grouped for an
    efficient reshard (per-expert would be hundreds of tiny collectives) — so
    the builder applies this unconditionally; only the gen-side fusion (the
    actual ``w13``/``w2`` names + gate/up interleave) is backend-specific and
    lives in the gen backend.

    The input ``state_dict_metadata`` must be EP-gathered (enumerate every
    expert globally).  Non-expert params are passed through unchanged.
    """
    # Group individual expert params by (prefix, proj_type)
    expert_groups: dict[tuple[str, str], list[tuple[str, dict]]] = {}
    grouped_metadata: dict[str, dict[str, Any]] = OrderedDict()

    for name, meta in state_dict_metadata.items():
        m = _INDIVIDUAL_EXPERT_RE.match(name)
        if m:
            prefix = m.group(1)  # e.g., "model.layers.0.mlp.experts"
            proj = m.group(3)  # "gate_proj", "up_proj", "down_proj"
            expert_groups.setdefault((prefix, proj), []).append((name, meta))
        else:
            grouped_metadata[name] = meta

    if not expert_groups:
        return state_dict_metadata

    # Stack each (prefix, proj) group into one [E, *per_expert_shape] grouped
    # HF entry.  Prepending the expert dim is layout-agnostic — it works for
    # gate/up ([inter, hidden] -> [E, inter, hidden]) and down ([hidden, inter]
    # -> [E, hidden, inter]) uniformly.  The reshard ships it Shard(0) on EP
    # (src) -> Shard(intermediate-axis) on gen TP (dst); see get_placements.
    for (prefix, proj), entries in expert_groups.items():
        num_experts_global = len(entries)
        per_expert_shape = list(entries[0][1]["shape"])
        grouped_metadata[f"{prefix}.{proj}.weight"] = {
            "shape": [num_experts_global, *per_expert_shape],
            "dtype": entries[0][1]["dtype"],
            "grouped_expert_proj": proj,
        }

    return grouped_metadata


# =========================================================================
# Layer grouping and refit-info construction
# =========================================================================

# Match the per-layer prefix and the top-level module group for both the
# Llama/Qwen HF convention (``model.layers.N`` / ``model.embed_tokens`` /
# ``model.norm``) and the NemotronH convention (``backbone.layers.N`` /
# ``backbone.embeddings`` / ``backbone.norm_f``).  Keeping these naming-agnostic
# lets _extract_layer_name produce a stable per-layer key that matches the keys
# _build_layer_to_pp_stage emits, so PP-stage filtering works for both families.
_LAYER_RE = re.compile(r"((?:model|backbone)\.layers\.\d+)\.")
_MODEL_PREFIX_RE = re.compile(r"((?:model|backbone)\.\w+)\.")


def _extract_layer_name(param_name: str) -> str:
    """Extract the layer group name from a parameter name.

    Examples:
        ``model.layers.0.self_attn.q_proj.weight`` -> ``model.layers.0``
        ``model.embed_tokens.weight`` -> ``model.embed_tokens``
        ``lm_head.weight`` -> ``lm_head``
    """
    m = _LAYER_RE.match(param_name)
    if m:
        return m.group(1)
    m = _MODEL_PREFIX_RE.match(param_name)
    if m:
        return m.group(1)
    return param_name.split(".")[0]


def check_nccl_xfer_refit_support(master_config: dict) -> None:
    """Validate ``master_config`` against every precondition of nccl_xfer_refit.

    Collects all violations and raises a single ``ValueError`` listing them, so
    a user fixing their config can address everything in one pass rather than
    re-running after each individual failure.  No-op on success.

    Conditions checked here are everything that can be decided from
    ``master_config`` alone (no model loading, no GPU work).  The following
    additional constraint cannot be checked from config and is enforced at
    runtime by the MoE fusion regex:

      - MoE experts must use the ``...experts.N.{up_proj,down_proj}.weight``
        naming, optionally with a ``gate_proj`` sibling.  Both gated SwiGLU
        (gate_proj + up_proj) and non-gated ReLU^2 (up_proj only) are fused;
        an expert naming the fusion regex doesn't recognize falls through and
        vLLM's ``w13_weight`` / ``w2_weight`` consumers will then reject it.

    Raises:
        ValueError: if any precondition is violated.
    """
    policy = master_config.get("policy", {}) or {}
    generation = policy.get("generation", {}) or {}
    megatron_cfg = policy.get("megatron_cfg", {}) or {}
    dtensor_cfg = policy.get("dtensor_cfg", {}) or {}
    vllm_cfg = generation.get("vllm_cfg", {}) or {}

    violations: list[str] = []

    # Non-colocated — refit happens cross-world; colocated path uses IPC.
    if generation.get("colocated", {}).get("enabled", False):
        violations.append(
            "policy.generation.colocated.enabled must be False "
            "(nccl_xfer_refit is only for disaggregated train/gen)."
        )

    # Gen backend = vLLM — only backend with prepare_nccl_xfer_refit_info.
    backend = generation.get("backend")
    if backend != "vllm":
        violations.append(
            f"policy.generation.backend must be 'vllm' (got {backend!r})."
        )

    # This initial version supports only the Megatron train + vLLM gen
    # combination; the DTensor train backend refit path is intentionally
    # dropped (Megatron is the path validated end-to-end).
    megatron_enabled = megatron_cfg.get("enabled", False)
    dtensor_enabled = dtensor_cfg.get("enabled", False)
    if not megatron_enabled:
        violations.append(
            "policy.megatron_cfg.enabled must be True "
            "(this initial version supports the Megatron train backend only)."
        )
    if dtensor_enabled:
        violations.append(
            "policy.dtensor_cfg.enabled must be False "
            "(this initial version supports the Megatron train backend only)."
        )

    if megatron_enabled:
        etp = megatron_cfg.get("expert_tensor_parallel_size", 1)

        # ETP is not supported yet.
        if etp not in (1, None):
            violations.append(
                f"Megatron expert_tensor_parallel_size is not supported yet "
                f"(got etp={etp})."
            )

        # PP-layout knobs that _build_layer_to_pp_stage doesn't yet handle.
        if megatron_cfg.get("pipeline_model_parallel_layout") is not None:
            violations.append(
                "policy.megatron_cfg.pipeline_model_parallel_layout must be unset."
            )
        vpp = megatron_cfg.get("virtual_pipeline_model_parallel_size")
        if vpp not in (None, 1):
            violations.append(
                "policy.megatron_cfg.virtual_pipeline_model_parallel_size must be "
                f"None or 1 (got {vpp})."
            )
        if megatron_cfg.get("account_for_embedding_in_pipeline_split", False):
            violations.append(
                "policy.megatron_cfg.account_for_embedding_in_pipeline_split must be False."
            )
        if megatron_cfg.get("account_for_loss_in_pipeline_split", False):
            violations.append(
                "policy.megatron_cfg.account_for_loss_in_pipeline_split must be False."
            )

        # Precision compatibility (train ↔ gen).  Supported combinations:
        #   BF16 train  ↔ BF16 gen   (default, tested)
        #   FP8  train  ↔ FP8  gen   (fp8_param=True + blockwise + vllm precision=fp8)
        # BF16→FP8 (train-side quant on the fly) is not implemented; FP8→BF16
        # has no consumer (vLLM doesn't accept FP8 bytes into a BF16 param).
        fp8_cfg = megatron_cfg.get("fp8_cfg", {}) or {}
        fp8_param = fp8_cfg.get("fp8_param", False)
        fp8_recipe = fp8_cfg.get("fp8_recipe", None)
        gen_precision = vllm_cfg.get("precision", None)

        # The refit byte-copies weights train -> gen, so gen dtype must match
        # train: BF16 (unset / "auto" / "bf16" / "bfloat16") or FP8 ("fp8").  A
        # value like "float16"/"float32" would silently mismatch the bf16 train
        # bytes and deadlock/corrupt the bulk collective; reject anything outside
        # the supported set up front (this also catches typos such as
        # "fp8_e4m3" that would otherwise skip the FP8 checks below).
        if gen_precision not in (None, "auto", "bf16", "bfloat16", "fp8"):
            violations.append(
                f"policy.generation.vllm_cfg.precision={gen_precision!r} is not "
                "supported by nccl_xfer_refit (use 'bf16'/'bfloat16', 'fp8', "
                "'auto', or leave unset); the refit byte-copies weights, so the "
                "gen dtype must match the train dtype."
            )

        if gen_precision == "fp8":
            if not fp8_param:
                violations.append(
                    "policy.generation.vllm_cfg.precision='fp8' requires "
                    "policy.megatron_cfg.fp8_cfg.fp8_param=True "
                    "(BF16→FP8 train-side quantization is not implemented yet)."
                )
            elif fp8_recipe != "blockwise":
                violations.append(
                    "policy.megatron_cfg.fp8_cfg.fp8_recipe must be 'blockwise' "
                    f"when fp8_param=True (got {fp8_recipe!r}); other recipes "
                    "don't produce export-ready scale_inv tensors."
                )
        elif fp8_param:
            violations.append(
                "policy.megatron_cfg.fp8_cfg.fp8_param=True requires "
                "policy.generation.vllm_cfg.precision='fp8' "
                "(FP8 storage on train side has no BF16 gen consumer)."
            )

    # vllm related restricions. NCCL reshard only support TP & DP for the vllm-side
    if generation.get("backend") == "vllm":
        gen_ep = vllm_cfg.get("expert_parallel_size", 1)
        gen_pp = vllm_cfg.get("pipeline_parallel_size", 1)
        if gen_ep != 1:
            violations.append(
                f"policy.generation.vllm_cfg.expert_parallel_size must be 1 (got {gen_ep})."
            )
        if gen_pp != 1:
            violations.append(
                f"policy.generation.vllm_cfg.pipeline_parallel_size must be 1 (got {gen_pp})."
            )

    if violations:
        raise ValueError(
            "nccl_xfer_refit cannot be enabled with the current config:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )


def build_nccl_xfer_refit_info(
    state_dict_metadata: dict[str, dict[str, Any]],
    train_parallelism: dict[str, int],
    gen_parallelism: dict[str, int],
    train_world_size: int,
    gen_world_size: int,
    layer_to_pp_stage: Optional[dict[str, int]] = None,
) -> dict[str, Any]:
    """Build per-layer parameter info for nccl_xfer-based refit.

    The input ``state_dict_metadata`` must be EP-gathered (enumerate every
    expert globally, not per-rank).  Both backends meet this naturally:
    Megatron's metadata pipeline routes through ``export_hf_weights`` (which
    does the EP all-gather) and DTensor doesn't use EP at all.

    Args:
        state_dict_metadata: ``{hf_param_name: {"shape": list, "dtype": str}}``
        train_parallelism / gen_parallelism: ``{"tp_size", "ep_size", "pp_size"}``
        train_world_size / gen_world_size: number of GPUs per side
        layer_to_pp_stage: optional mapping from layer name to PP stage index.
            When provided (PP>1), per-stage meshes are built so each PP stage's
            train ranks + all gen ranks form an independent sub-group.

    Returns:
        ``{"layer_names": [...], "per_layer_params": {layer: [param_info, ...]},
           "pp_size": int}``
    """
    # state_dict_metadata is already pre-filtered to exclude misc params
    # (caller separates misc into a parallel dict for the
    # packed_broadcast-based misc refit path).
    # Group per-expert MoE params into backend-agnostic grouped HF entries
    # (gate_proj/up_proj/down_proj, each [E, ...]).  Grouping is universal; the
    # gen backend maps these into its own fused layout (w13/w2) gen-side.  No-op
    # for dense (non-MoE) models (early return when there are no expert params).
    state_dict_metadata = group_expert_params_in_metadata(state_dict_metadata)

    pp_size = train_parallelism.get("pp_size", 1)
    tp_size = train_parallelism.get("tp_size", 1)
    ep_size = train_parallelism.get("ep_size", 1)
    use_per_stage = pp_size > 1
    if use_per_stage:
        assert layer_to_pp_stage is not None, (
            "layer_to_pp_stage must be provided when pp_size > 1"
        )

    # Megatron-Core with ETP=1 gives non-expert and expert params *different*
    # rank-to-coord layouts on the same physical ranks: for non-expert params,
    # ranks are partitioned (tp, dp); for expert params, ranks are partitioned
    # (ep, edp).  TP and EP are not jointly indexable on a single rectangular
    # mesh (the coords aren't independent — e.g. EP4×TP2 has tp=r%2 AND
    # ep=r%4 on the same rank).  We therefore build two src meshes per stage
    # and pick per-param based on whether the param is a MoE expert weight.
    def _build_train_src_meshes(num_gpus: int, rank_offset: int, stage_pp: int):
        non_expert_mesh, non_expert_dim_map = build_mesh_info(
            num_gpus,
            rank_offset=rank_offset,
            tp_size=tp_size,
            ep_size=1,
            pp_size=stage_pp,
        )
        expert_mesh, expert_dim_map = build_mesh_info(
            num_gpus,
            rank_offset=rank_offset,
            tp_size=1,
            ep_size=ep_size,
            pp_size=stage_pp,
        )
        return (non_expert_mesh, non_expert_dim_map), (expert_mesh, expert_dim_map)

    if use_per_stage:
        # Per-PP-stage meshes: within each sub-group, train ranks are
        # 0..train_ranks_per_stage-1 and gen ranks follow immediately after.
        train_ranks_per_stage = train_world_size // pp_size
        per_stage_src_nonexpert = {}
        per_stage_src_expert = {}
        for s in range(pp_size):
            (
                per_stage_src_nonexpert[s],
                per_stage_src_expert[s],
            ) = _build_train_src_meshes(
                train_ranks_per_stage, rank_offset=0, stage_pp=1
            )

        # dst mesh: gen ranks start at train_ranks_per_stage within each sub-group
        dst_mesh, dst_dim_map = build_mesh_info(
            gen_world_size,
            rank_offset=train_ranks_per_stage,
            **{k: gen_parallelism.get(k, 1) for k in ("tp_size", "ep_size", "pp_size")},
        )
    else:
        # Single global mesh pair (PP=1 or no per-stage mapping)
        (non_expert_mesh, non_expert_dim_map), (expert_mesh, expert_dim_map) = (
            _build_train_src_meshes(train_world_size, rank_offset=0, stage_pp=pp_size)
        )
        dst_mesh, dst_dim_map = build_mesh_info(
            gen_world_size,
            rank_offset=train_world_size,
            **{k: gen_parallelism.get(k, 1) for k in ("tp_size", "ep_size", "pp_size")},
        )

    per_layer_params: dict[str, list] = OrderedDict()
    for name, meta in state_dict_metadata.items():
        layer = _extract_layer_name(name)
        ndim = len(meta["shape"])
        expert = is_expert_param(name)

        if use_per_stage:
            stage = layer_to_pp_stage.get(layer, 0)
            stage_src_mesh, stage_src_dim_map = (
                per_stage_src_expert[stage]
                if expert
                else per_stage_src_nonexpert[stage]
            )
            info = {
                "name": name,
                "global_shape": tuple(meta["shape"]),
                "dtype": meta["dtype"],
                "pp_stage": stage,
                "src_mesh_info": stage_src_mesh,
                "src_placements": get_placements(name, stage_src_dim_map, ndim),
                "dst_mesh_info": dst_mesh,
                "dst_placements": get_placements(name, dst_dim_map, ndim),
            }
        else:
            this_src_mesh, this_src_dim_map = (
                (expert_mesh, expert_dim_map)
                if expert
                else (non_expert_mesh, non_expert_dim_map)
            )
            info = {
                "name": name,
                "global_shape": tuple(meta["shape"]),
                "dtype": meta["dtype"],
                "src_mesh_info": this_src_mesh,
                "src_placements": get_placements(name, this_src_dim_map, ndim),
                "dst_mesh_info": dst_mesh,
                "dst_placements": get_placements(name, dst_dim_map, ndim),
            }

        # Propagate the grouped-expert projection tag (gate_proj/up_proj/
        # down_proj) so the train side stacks the matching per-expert tensors
        # (_group_experts) and the gen backend maps the grouped HF param into
        # its own fused layout (w13/w2).
        if "grouped_expert_proj" in meta:
            info["grouped_expert_proj"] = meta["grouped_expert_proj"]

        per_layer_params.setdefault(layer, []).append(info)

    return {
        "layer_names": list(per_layer_params.keys()),
        "per_layer_params": per_layer_params,
        "train_world_size": train_world_size,
        "gen_world_size": gen_world_size,
        "pp_size": pp_size,
        "gen_tp_size": gen_parallelism.get("tp_size", 1),
    }
