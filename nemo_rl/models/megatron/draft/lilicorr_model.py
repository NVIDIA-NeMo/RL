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

"""Assemble a LiLiCorr draft model: DFlash backbone + reranker head.

Separate from ``lilicorr.py`` (which holds the head and its objective) so the head stays
importable without pulling in Megatron's model-provider machinery -- the same split the
other drafters use, and it keeps the head unit-testable on CPU.
"""

from __future__ import annotations

import json
import os

import torch
from torch import Tensor
from typing import TYPE_CHECKING, Any

from megatron.core.transformer.module import MegatronModule

from nemo_rl.models.megatron.draft.dflash import DFlashBody, DFlashBodyConfig
from nemo_rl.models.megatron.draft.lilicorr import (
    LiLiCorrDraftOutput,
    LiLiCorrHeadConfig,
    build_lilicorr_head,
)

if TYPE_CHECKING:
    from megatron.core.process_groups_config import ProcessGroupCollection


def _load_draft_checkpoint_config(model_name: str | None) -> dict[str, Any]:
    """Read the drafter's config.json.

    Refuses to proceed without it: NeMo-RL does not support from-scratch draft init, and
    a LiLiCorr built from defaults would load the released weights into a differently
    shaped reranker or silently skip them.
    """
    if not model_name:
        raise ValueError(
            "policy.draft.model_name is required for speculator_type=lilicorr "
            "(from-scratch draft init is not supported)"
        )
    path = model_name
    if os.path.isdir(path):
        path = os.path.join(path, "config.json")
    if not os.path.isfile(path):
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(model_name, trust_remote_code=True).to_dict()
    with open(path) as fh:
        return json.load(fh)


# ModelOpt builds feature_mlp as nn.Sequential, so its parameters are positional
# (feature_mlp.0.weight). Released drafters carry the serving layout
# (feature_mlp.up_proj.weight) -- the same 14-tensor mapping as
# NVIDIA/Model-Optimizer#3061, applied here in reverse so the checkpoint loads into the
# module as ModelOpt defines it.
_SERVING_TO_POSITIONAL = {
    "feature_norm.weight": "feature_mlp.0.weight",
    "feature_norm.bias": "feature_mlp.0.bias",
    "feature_mlp.up_proj.weight": "feature_mlp.1.weight",
    "feature_mlp.up_proj.bias": "feature_mlp.1.bias",
    "feature_mlp.down_proj.weight": "feature_mlp.3.weight",
    "feature_mlp.down_proj.bias": "feature_mlp.3.bias",
}


def _from_positional_name(name: str) -> str:
    """Inverse of :func:`_to_serving_name`: ModelOpt module names -> serving names.

    Export has to undo exactly what load did. Without this the exported head carries
    positional names (``feature_mlp.0.weight``) while every consumer of a released
    LiLiCorr expects serving names (``feature_norm.weight``), so the checkpoint loads
    with those tensors missing -- silently, if the consumer is not strict.
    """
    for serving, positional in _SERVING_TO_POSITIONAL.items():
        if name == positional:
            return serving
    for serving, positional in (("mlp.up_proj", "mlp.0"), ("mlp.down_proj", "mlp.2")):
        if f".{positional}." in name:
            return name.replace(f".{positional}.", f".{serving}.")
    return name


def _to_serving_name(name: str) -> str:
    """Map a released checkpoint name onto ModelOpt's positional module names."""
    if name in _SERVING_TO_POSITIONAL:
        return _SERVING_TO_POSITIONAL[name]
    # layers.N.mlp.{up,down}_proj -> layers.N.mlp.{0,2}
    for serving, positional in (("mlp.up_proj", "mlp.0"), ("mlp.down_proj", "mlp.2")):
        if f".{serving}." in name:
            return name.replace(f".{serving}.", f".{positional}.")
    return name


def load_lilicorr_weights(head: Any, model_name: str) -> None:
    """Load ``lilicorr.*`` tensors from the drafter checkpoint into the head.

    Strict by construction: every head parameter must be covered and every checkpoint
    tensor consumed. Quantized drafters carry ``weight_scale``/``weight_scale_2`` leaves
    that belong to the packed weights, not to the head's float parameters, so they are
    skipped explicitly rather than silently tolerated by a loose load.
    """
    import glob

    from safetensors.torch import load_file

    root = model_name if os.path.isdir(model_name) else os.path.dirname(model_name)
    shards = sorted(glob.glob(os.path.join(root, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"no .safetensors under {root}")

    prefix = "lilicorr."
    found: dict[str, Any] = {}
    packed = []
    for shard in shards:
        shard_sd = load_file(shard)
        for key, tensor in shard_sd.items():
            if not key.startswith(prefix):
                continue
            if key.endswith(("weight_scale", "weight_scale_2", "input_scale")):
                continue
            # A packed NVFP4 weight is uint8 with two 4-bit values per byte, so its stored
            # shape is half the logical one. Loading it into a float parameter would either
            # fail on shape or, worse, succeed after a reshape and feed the reranker
            # garbage. Point model_name at a dequantized drafter instead.
            if tensor.dtype == torch.uint8 and f"{key}_scale" in shard_sd:
                packed.append(key)
                continue
            found[_to_serving_name(key[len(prefix) :])] = tensor

    if packed:
        raise ValueError(
            f"{len(packed)} LiLiCorr tensors in {root} are NVFP4-packed (e.g. "
            f"{packed[:2]}). Co-training needs gradients, so point "
            "policy.draft.model_name at a dequantized copy of the drafter."
        )

    missing, unexpected = head.load_state_dict(found, strict=False)
    missing = [m for m in missing if "rotary" not in m]
    if missing or unexpected:
        raise ValueError(
            f"LiLiCorr head did not load cleanly from {root}: "
            f"{len(missing)} missing (e.g. {missing[:3]}), "
            f"{len(unexpected)} unexpected (e.g. {list(unexpected)[:3]}). "
            "A partial load leaves the reranker at its init values and collapses "
            "acceptance without erroring."
        )
    print(f"  LiLiCorr head: loaded {len(found)} tensors from {root}")


class LiLiCorrDraftModel(MegatronModule):
    """DFlash backbone with a LiLiCorr reranker on top."""

    def __init__(
        self,
        *,
        body: DFlashBody,
        head: Any,
        config: Any,
    ) -> None:
        super().__init__(config=body.config if hasattr(body, "config") else None)
        self.body = body
        self.lilicorr = head
        self.draft_config = config

    def forward(
        self,
        *,
        target_taps: Any,
        block_embeddings: Any,
        plan: Any,
        sequence_layout: Any = None,
        context_parallel_group: Any = None,
    ) -> Tensor:
        """Run the DFlash backbone and return its block hidden states.

        Deliberately the SAME signature and return type as :class:`DFlashBody`, so the
        Megatron train step and DFlashSpeculator.forward drive this drafter unchanged.
        The reranker is a separate call (:meth:`score_candidates`) rather than part of
        this forward: folding it in here would mean returning a tuple that every DFlash
        call site would have to special-case.
        """
        return self.body(
            target_taps=target_taps,
            block_embeddings=block_embeddings,
            plan=plan,
            sequence_layout=sequence_layout,
            context_parallel_group=context_parallel_group,
        )

    def score_candidates(
        self,
        *,
        block_hidden: Tensor,
        output_weight: Tensor,
        embed_tokens: Any,
        anchor_hidden: Tensor,
        anchor_valid: Any = None,
    ) -> "LiLiCorrDraftOutput":
        """Rescore the backbone's own top-k proposals with the candidate lattice.

        Candidates come from the backbone's logits and are DETACHED: they index the
        lattice the reranker scores, and a gradient through topk is meaningless.
        """
        student_logits = block_hidden @ output_weight.T

        # Slot 0 is the anchor -- a committed token, not a prediction. The reranker
        # scores only the mask slots (num_candidate_slots = block_size - 1), so the
        # lattice inputs drop it. Passing the full block makes the head reject the
        # shape, and silently trimming the wrong end would score the lattice against
        # positions the drafter never predicted.
        mask_hidden = block_hidden[:, 1:]
        mask_logits = student_logits[:, 1:]
        topk = self.lilicorr.candidate_topk
        cand_logprobs, cand_ids = torch.log_softmax(
            mask_logits.detach().float(), dim=-1
        ).topk(topk, dim=-1)

        # The head takes [batch, blocks, slots, k]; Megatron hands us rows that already
        # flatten batch x anchors into one dimension. Add a singleton batch rather than
        # guessing a split: the head reshapes to (bsz * n_blocks, ...) immediately, so
        # only the product matters and 1 x rows is the faithful mapping.
        start_scores, pair_scores = self.lilicorr(
            candidate_token_ids=cand_ids.unsqueeze(0),
            candidate_log_probs=cand_logprobs.unsqueeze(0),
            pass_hidden=mask_hidden.unsqueeze(0),
            embed_tokens=embed_tokens,
            anchor_hidden=anchor_hidden.unsqueeze(0),
            anchor_valid=None if anchor_valid is None else anchor_valid.unsqueeze(0),
        )
        return LiLiCorrDraftOutput(
            student_logits=student_logits,
            start_scores=start_scores,
            pair_scores=pair_scores,
            candidate_ids=cand_ids,
        )


def build_lilicorr_draft_model(
    *,
    model_provider: Any,
    draft_config: Any,
    pg_collection: "ProcessGroupCollection",
    policy_model_chunk: MegatronModule,
) -> MegatronModule | None:
    """Build the LiLiCorr draft model for a policy chunk."""
    if not draft_config.enabled:
        return None

    ckpt_cfg = _load_draft_checkpoint_config(getattr(draft_config, "model_name", None))
    dflash_cfg = ckpt_cfg.get("dflash_config", {})

    projector = dflash_cfg.get("projector_type")
    if projector != "lilicorr":
        raise ValueError(
            f"draft checkpoint at {draft_config.model_name} has "
            f"projector_type={projector!r}, not 'lilicorr'. Set "
            "policy.draft.speculator_type to match the checkpoint."
        )

    head_cfg = LiLiCorrHeadConfig.from_checkpoint_config(ckpt_cfg)

    # The backbone runs at the TARGET's width, not the reranker's. head_cfg.hidden_size
    # (1024) is the reranker's internal width; the body consumes and emits the target's
    # hidden (4096), which is also what the head's pass_hidden_proj expects. Building the
    # body at 1024 fails with "mat1 and mat2 shapes cannot be multiplied (Nx1024 and
    # 4096x1024)" the moment the head sees its output.
    # A DEDICATED parallel config, not the policy's. DFlashBody is context-parallel
    # unaware and rejects context_parallel_size > 1 outright; passing the policy's
    # provider straight through therefore makes the drafter unusable on any recipe that
    # runs CP, which is most 120B recipes. The body always runs CP=1 and the target's
    # CP-sharded hidden states are gathered before they reach it, so copying only the
    # fields the body actually needs -- and leaving context_parallel_size at its default
    # of 1 -- is what lets the policy keep CP > 1. Mirrors the DFlash provider.
    from megatron.core.model_parallel_config import ModelParallelConfig

    parallel_config = ModelParallelConfig(
        tensor_model_parallel_size=model_provider.tensor_model_parallel_size,
        use_cpu_initialization=model_provider.use_cpu_initialization,
        fp16=model_provider.fp16,
        bf16=model_provider.bf16,
        params_dtype=model_provider.params_dtype,
        sequence_parallel=False,
    )
    # Body geometry comes from the DRAFTER CHECKPOINT's top-level config, not from
    # head_cfg and not from DFlashBodyConfig's defaults.
    #
    # Two bugs lived here. head_cfg.num_heads is the RERANKER's head count (8 for Super
    # 3.5) while the body has 32 -- that one fails loudly at the strict weight load. The
    # other does not: DFlashBodyConfig's defaults pin the public Qwen3-8B-DFlash-b16
    # artifact, whose rms_norm_eps is 1e-6, while this checkpoint trained at 1e-5. The
    # weights load cleanly either way and the body just computes something slightly
    # different for the rest of training.
    #
    # Anything the checkpoint does not name falls back to the dataclass default, so a
    # drafter that genuinely matches the public artifact still works unchanged.
    body_defaults = DFlashBodyConfig()
    body_config = DFlashBodyConfig(
        hidden_size=int(ckpt_cfg.get("hidden_size", head_cfg.model_hidden_size)),
        intermediate_size=int(
            ckpt_cfg.get("intermediate_size", body_defaults.intermediate_size)
        ),
        num_attention_heads=int(
            ckpt_cfg.get("num_attention_heads", body_defaults.num_attention_heads)
        ),
        num_key_value_heads=int(
            ckpt_cfg.get("num_key_value_heads", body_defaults.num_key_value_heads)
        ),
        head_dim=int(ckpt_cfg.get("head_dim", body_defaults.head_dim)),
        num_hidden_layers=int(
            ckpt_cfg.get("num_hidden_layers", body_defaults.num_hidden_layers)
        ),
        num_target_taps=len(dflash_cfg.get("target_layer_ids", []))
        or body_defaults.num_target_taps,
        rope_theta=float(ckpt_cfg.get("rope_theta", body_defaults.rope_theta)),
        rms_norm_eps=float(ckpt_cfg.get("rms_norm_eps", body_defaults.rms_norm_eps)),
    )
    body = DFlashBody(
        config=body_config,
        parallel_config=parallel_config,
        tp_group=getattr(pg_collection, "tp", None),
    )

    head = build_lilicorr_head(head_cfg)

    # Load the released reranker weights STRICTLY. A silent partial load is the failure
    # this project keeps meeting: the module initialises, training runs, and acceptance is
    # quietly near zero because the head is still at its init values.
    load_lilicorr_weights(head, draft_config.model_name)

    model = LiLiCorrDraftModel(body=body, head=head, config=draft_config)
    if model_provider.params_dtype is not None:
        model = model.to(model_provider.params_dtype)
    return model


def export_lilicorr_weights_to_hf(model: Any) -> list[tuple[Tensor, Tensor]]:
    """Export a LiLiCorr drafter: DFlash body plus the reranker head.

    The body goes through DFlash's exporter, which gathers its tensor-parallel shards
    into logical full weights. The reranker is REPLICATED rather than sharded -- it is
    built from plain nn.Linear, so every tensor-parallel rank holds an identical copy --
    so its tensors are emitted as-is under the ``lilicorr.`` prefix that the serving
    checkpoint uses.
    """
    from nemo_rl.models.megatron.draft.utils import (
        export_dflash_weights_to_hf,
        unwrap_model,
    )

    module = unwrap_model(model)
    body = getattr(module, "body", None)
    head = getattr(module, "lilicorr", None)
    if body is None or head is None:
        raise RuntimeError(
            "export_lilicorr_weights_to_hf expects a LiLiCorrDraftModel carrying both a "
            "DFlash body and a reranker head."
        )

    exported = list(export_dflash_weights_to_hf(body))
    exported.extend(
        (f"lilicorr.{_from_positional_name(name)}", tensor.detach())
        for name, tensor in head.state_dict().items()
    )
    return exported

