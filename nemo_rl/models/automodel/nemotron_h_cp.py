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
"""Context parallelism for the HuggingFace (``force_hf``) NemotronH implementation.

AUTOMODEL-WORKAROUND(nemotron-h-cp): Automodel's ``NemotronHParallelizationStrategy``
selects on the class name ``NemotronHForCausalLM``, which both implementations share,
but its body mixes the two:

* it reads ``model.backbone.layers`` -- only the HF remote-code model has
  ``.backbone`` (the custom ``nemotron_v3`` model uses ``.model.layers``), and
* its ``context_parallel_size > 1`` branch reads ``layer.mixer.attn_module`` and
  sets ``mixer.cp`` -- both exist only on the custom model. On the HF model the
  attribute read raises ``AttributeError`` during model construction, and even
  with that guarded the assigned ``mixer.cp`` would be ignored because the HF
  Mamba mixer's ``forward`` never consults it. The Mamba layers would then run on
  each rank's sequence shard with no cross-rank state, silently producing wrong
  activations.

A third gap sits in the checkpoint's own remote code and fires first: NeMo-RL and
Automodel both request ``attn_implementation="sdpa"`` once ``context_parallel_size
> 1`` (ring attention needs torch SDPA, not flash/TE), but
``NemotronHPreTrainedModel`` never sets ``_supports_sdpa``, so transformers >= 5
rejects the request outright:

    ValueError: NemotronHForCausalLM does not support an attention implementation
    through torch.nn.functional.scaled_dot_product_attention yet.

The remote code does ship an SDPA path (``NEMOTRONH_ATTENTION_CLASSES["sdpa"] ->
NemotronHSdpaAttention``, and even the "eager" class calls
``F.scaled_dot_product_attention``); only the capability flag is missing.

This module closes all three gaps for the HF model, without touching the 3rdparty
submodule or the checkpoint:

* Attention: the HF mixer is torch ``scaled_dot_product_attention`` (NeMo-RL
  forces ``attn_implementation="sdpa"`` whenever ``context_parallel_size > 1``),
  which torch's ``context_parallel()`` context manager already ring-patches. The
  TE ``set_context_parallel_group`` hook is therefore not needed -- an
  ``attn_module = None`` marker makes upstream's ``isinstance`` check skip it.
* Mamba: ``NemotronHMamba2Mixer.forward`` is rebound per instance to a CP-aware
  variant that routes the fused ``mamba_split_conv1d_scan_combined`` call through
  :class:`MambaContextParallel` (all-to-all to a hidden-parallel layout so each
  rank sees the full sequence for a slice of the heads), mirroring the custom
  ``nemotron_v3`` mixer's CP path.

Call :func:`apply_nemotron_h_cp_fixes` once per worker process before the model
is built; it is idempotent and a no-op for the custom implementation.
"""

from __future__ import annotations

import sys
import types
from typing import Any, Optional

import torch


def _module_of(obj: Any) -> Any:
    """Return the (dynamically loaded) python module a HF remote-code object came from."""
    return sys.modules[type(obj).__module__]


def _cp_aware_mamba_forward(
    self: Any,
    hidden_states: torch.Tensor,
    cache_params: Optional[Any] = None,
    cache_position: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """CP-aware replacement for ``NemotronHMamba2Mixer.forward``.

    Delegates to the stock forward unless a :class:`MambaContextParallel` helper is
    attached and active. The CP path mirrors Automodel's custom ``nemotron_v3``
    mixer: redistribute the ``in_proj`` output from sequence-sharded to
    hidden-sharded, run the fused conv+SSM kernel over the *full* sequence with
    per-rank head/group slices of ``conv1d`` / ``dt_bias`` / ``A_log`` / ``D``,
    redistribute back, then apply the gated RMSNorm and ``out_proj`` (both of
    which the single-GPU path fuses into the kernel and which must run on the
    full hidden dimension).
    """
    cp = getattr(self, "cp", None)
    if cp is None or cp.cp_size == 1:
        return self._nrl_orig_forward(
            hidden_states, cache_params, cache_position, attention_mask
        )
    if cache_params is not None:
        raise NotImplementedError(
            "NemotronH Mamba context parallelism supports the training forward only; "
            "cached/step decoding under CP is not implemented."
        )

    # Nemotron-H supplies this optional runtime dependency; other models should
    # not have to install it merely to import NeMo RL.
    from mamba_ssm.ops.triton.ssd_combined import (  # pyrefly: ignore[import-error]
        mamba_split_conv1d_scan_combined,
    )

    hf = _module_of(self)
    hidden_states = hf.apply_mask_to_padding_states(hidden_states, attention_mask)
    projected_states = self.in_proj(hidden_states)

    dt_limit_kwargs = (
        {}
        if self.time_step_limit == (0.0, float("inf"))
        else {"dt_limit": self.time_step_limit}
    )

    projected_states = cp.pre_conv_ssm(projected_states)
    A = -torch.exp(cp.get_A_log().float())
    out = mamba_split_conv1d_scan_combined(
        projected_states,
        cp.get_conv1d_weight(),
        cp.get_conv1d_bias(),
        cp.get_dt_bias(),
        A,
        D=cp.get_D(),
        chunk_size=self.chunk_size,
        seq_idx=None,
        activation=self.activation,
        # rmsnorm/outproj are deliberately NOT fused here: they operate on the
        # full hidden dimension, which only exists after post_conv_ssm. With
        # rmsnorm_weight=None the kernel still applies the silu gate inside the
        # scan, so the composition below stays RMSNorm(x_ssm * silu(z)) -- the
        # same expression the fused single-GPU call computes.
        rmsnorm_weight=None,
        outproj_weight=None,
        headdim=self.head_dim,
        ngroups=cp.n_groups_local,
        norm_before_gate=False,
        return_final_states=False,
        **dt_limit_kwargs,
    )
    if out.ndim == 4:
        out = out.reshape(out.shape[0], out.shape[1], -1)
    out = cp.post_conv_ssm(out)
    out = self.norm(out, gate=None)
    return self.out_proj(out)


def _bind_cp_aware_mamba_forward(mixer: torch.nn.Module) -> None:
    """Rebind ``mixer.forward`` to the CP-aware variant (idempotent)."""
    if hasattr(mixer, "_nrl_orig_forward"):
        return
    object.__setattr__(mixer, "_nrl_orig_forward", mixer.forward)
    mixer.forward = types.MethodType(_cp_aware_mamba_forward, mixer)
    if not hasattr(mixer, "cp"):
        object.__setattr__(mixer, "cp", None)


def prepare_hf_nemotron_h_for_cp(model: torch.nn.Module) -> int:
    """Make an HF remote-code NemotronH survive and honor Automodel's CP branch.

    Returns the number of Mamba mixers rebound (0 for the custom implementation,
    which Automodel's MoE parallelizer handles natively).
    """
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return 0

    rebound = 0
    for layer in backbone.layers:
        block_type = getattr(layer, "block_type", None)
        mixer = getattr(layer, "mixer", None)
        if mixer is None:
            continue
        if block_type == "attention" and not hasattr(mixer, "attn_module"):
            # Upstream does `isinstance(layer.mixer.attn_module, DotProductAttention)`;
            # the HF mixer has no such attribute. None makes that check False, so the
            # TE-only hook is skipped -- correct here, because torch's
            # context_parallel() already ring-patches the SDPA this mixer calls.
            mixer.attn_module = None
        elif block_type == "mamba":
            _bind_cp_aware_mamba_forward(mixer)
            rebound += 1
    return rebound


def _declare_sdpa_support(cls: Any) -> bool:
    """Set ``_supports_sdpa`` on a dynamically loaded NemotronH class.

    Returns True if the flag was flipped. Only applied when the loaded module
    actually maps an SDPA attention class, so this never claims a capability the
    checkpoint's code does not implement.
    """
    module = sys.modules.get(getattr(cls, "__module__", ""), None)
    attention_classes = getattr(module, "NEMOTRONH_ATTENTION_CLASSES", None)
    if not attention_classes or "sdpa" not in attention_classes:
        return False
    flipped = False
    for klass in cls.__mro__:
        if klass.__name__ == "NemotronHPreTrainedModel" and not getattr(
            klass, "_supports_sdpa", False
        ):
            klass._supports_sdpa = True
            flipped = True
    return flipped


def _patch_dynamic_module_sdpa_flag() -> None:
    """Declare SDPA support on NemotronH classes as transformers loads them.

    ``get_class_in_module`` is where ``trust_remote_code`` classes materialize, and
    it is reached from every entry point (``AutoModel``, Automodel's
    ``_from_pretrained_parent_class``). Patching it means the flag is set before
    ``_sdpa_can_dispatch`` runs, without NeMo-RL having to resolve the remote class
    itself.
    """
    import transformers.dynamic_module_utils as dynamic_module_utils

    if getattr(dynamic_module_utils, "_nrl_nemotron_h_sdpa_patched", False):
        return

    original = dynamic_module_utils.get_class_in_module

    def _get_class_in_module(class_name, *args, **kwargs):
        cls = original(class_name, *args, **kwargs)
        if isinstance(cls, type) and class_name.startswith("NemotronH"):
            if _declare_sdpa_support(cls):
                print(
                    "[nemo-rl] NemotronH (HF): declared _supports_sdpa=True "
                    "(remote code implements SDPA but omits the flag; "
                    "context parallelism requires attn_implementation='sdpa')",
                    flush=True,
                )
        return cls

    dynamic_module_utils.get_class_in_module = _get_class_in_module
    dynamic_module_utils._nrl_nemotron_h_sdpa_patched = True


def apply_nemotron_h_cp_fixes() -> None:
    """Install the CP-corrected NemotronH parallelization strategy (idempotent).

    Wraps Automodel's strategy rather than replacing it: the pre-pass runs on the
    unparallelized model, then the upstream body (TP plan, ``MambaContextParallel``
    attachment, activation checkpointing, FSDP) runs unchanged, so this stays
    correct if upstream's plan evolves.
    """
    from nemo_automodel.components.distributed import parallelizer as _p

    _patch_dynamic_module_sdpa_flag()

    if getattr(_p, "_nrl_nemotron_h_cp_applied", False):
        return

    upstream_cls = type(_p.PARALLELIZATION_STRATEGIES["NemotronHForCausalLM"])

    class _NemotronHCPFixStrategy(upstream_cls):  # type: ignore[misc,valid-type]
        """NemotronH strategy that also supports the HF remote-code model under CP."""

        def parallelize(self, model, device_mesh, **kwargs):  # type: ignore[override]
            mesh_names = device_mesh.mesh_dim_names or ()
            cp_mesh = device_mesh["cp"] if "cp" in mesh_names else None
            if cp_mesh is not None and cp_mesh.size() > 1:
                rebound = prepare_hf_nemotron_h_for_cp(model)
                if rebound:
                    print(
                        f"[nemo-rl] NemotronH (HF): enabled Mamba context parallelism "
                        f"on {rebound} mixers (cp={cp_mesh.size()})",
                        flush=True,
                    )
            return super().parallelize(model, device_mesh, **kwargs)

    _p.PARALLELIZATION_STRATEGIES["NemotronHForCausalLM"] = _NemotronHCPFixStrategy()
    _p._nrl_nemotron_h_cp_applied = True
