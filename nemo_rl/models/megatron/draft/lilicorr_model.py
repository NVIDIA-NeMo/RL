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

import dataclasses
import json
import os
from types import SimpleNamespace

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


# Checkpoint tensors that are not part of the DFlash body: the reranker head (loaded
# by load_lilicorr_weights) and the drafter-owned LM head (load_lilicorr_lm_head).
_NON_BODY_PREFIXES = ("lilicorr.", "lm_head.")


def load_lilicorr_body_weights(body: DFlashBody, model_name: str) -> None:
    """Load the DFlash backbone, convolutions included, STRICTLY from the checkpoint.

    Without this the body trains from random init while the head is pretrained: every
    loss stays finite and nothing errors, but the served drafter is untrained.
    """
    from nemo_rl.models.megatron.draft.utils import (
        _load_checkpoint_state,
        _load_normalized_hf_weights_to_dflash,
        _normalize_draft_state_dict,
    )

    state = _normalize_draft_state_dict(_load_checkpoint_state(model_name))
    body_state = {
        name: tensor
        for name, tensor in state.items()
        if not name.startswith(_NON_BODY_PREFIXES)
    }
    missing, unexpected = _load_normalized_hf_weights_to_dflash(body, body_state)
    if missing or unexpected:
        raise ValueError(
            f"LiLiCorr body did not load cleanly from {model_name}: "
            f"missing={sorted(missing)[:6]}, unexpected={sorted(unexpected)[:6]}"
        )
    print(f"  LiLiCorr body: loaded {len(body_state)} tensors from {model_name}")


def load_lilicorr_lm_head_shard(
    model_name: str, *, policy_lm_head_shard: Tensor, tp_rank: int
) -> Tensor:
    """Load the drafter-owned LM head as the same vocab shard the policy head holds.

    Checkpoints with ``has_own_lm_head`` are served with THEIR head (vLLM LiLiCorr), so
    training must project draft hidden states through it too. Sharding it exactly like
    the policy head lets the existing vocab-parallel losses use it unchanged.
    """
    from nemo_rl.models.megatron.draft.utils import (
        _load_checkpoint_state,
        _normalize_draft_state_dict,
    )

    state = _normalize_draft_state_dict(_load_checkpoint_state(model_name))
    if "lm_head.weight" not in state:
        raise ValueError(
            f"{model_name} sets has_own_lm_head but carries no lm_head.weight"
        )
    full = state["lm_head.weight"]
    rows, hidden = policy_lm_head_shard.shape
    if full.shape[1] != hidden:
        raise ValueError(
            f"drafter lm_head hidden {full.shape[1]} != policy hidden {hidden}"
        )
    shard = torch.zeros(
        rows,
        hidden,
        dtype=policy_lm_head_shard.dtype,
        device=policy_lm_head_shard.device,
    )
    start = tp_rank * rows
    # Megatron pads the vocab; padded rows stay zero, like the policy's.
    valid = max(0, min(rows, full.shape[0] - start))
    if valid:
        shard[:valid] = full[start : start + valid].to(shard)
    return shard


def vocab_parallel_topk_logprobs(
    local_logits: Tensor,
    k: int,
    *,
    vocab_start: int,
    vocab_size: int,
    tp_group: Any,
) -> tuple[Tensor, Tensor]:
    """Global top-k token ids and log-probs from vocab-parallel logits.

    Each tensor-parallel rank holds logits for one contiguous vocab range. A local
    log_softmax/top-k would normalise over a fraction of the vocab and return LOCAL
    indices, so every rank would score different, wrongly-identified candidates.
    """
    local_logits = local_logits.float()
    local_size = local_logits.shape[-1]
    # Megatron pads the vocab; padded columns are never real tokens.
    real = vocab_size - vocab_start
    if real < local_size:
        local_logits = local_logits.clone()
        local_logits[..., max(real, 0) :] = float("-inf")
    world = 1 if tp_group is None else torch.distributed.get_world_size(tp_group)
    max_logit = local_logits.amax(dim=-1, keepdim=True)
    if world > 1:
        torch.distributed.all_reduce(
            max_logit, op=torch.distributed.ReduceOp.MAX, group=tp_group
        )
    sum_exp = (local_logits - max_logit).exp().sum(dim=-1, keepdim=True)
    if world > 1:
        torch.distributed.all_reduce(sum_exp, group=tp_group)
    log_normalizer = max_logit + sum_exp.log()

    values, ids = local_logits.topk(k, dim=-1)
    ids = ids + vocab_start
    if world > 1:
        gathered_values = [torch.empty_like(values) for _ in range(world)]
        gathered_ids = [torch.empty_like(ids) for _ in range(world)]
        torch.distributed.all_gather(gathered_values, values, group=tp_group)
        torch.distributed.all_gather(gathered_ids, ids, group=tp_group)
        values = torch.cat(gathered_values, dim=-1)
        ids = torch.cat(gathered_ids, dim=-1)
        values, order = values.topk(k, dim=-1)
        ids = ids.gather(-1, order)
    return values - log_normalizer, ids


class LiLiCorrDraftModel(MegatronModule):
    """DFlash backbone with a LiLiCorr reranker on top."""

    def __init__(
        self,
        *,
        body: DFlashBody,
        head: Any,
        config: Any,
    ) -> None:
        # ModelOpt's Megatron plugin ASSIGNS onto the `.config` of every MegatronModule
        # it walks (hetereogenous_dist_checkpoint, modelopt_output_layer_untied). The
        # body's DFlashBodyConfig is frozen AND slots=True, so it rejects both: frozen
        # blocks assignment, and slots blocks the new attribute even were it mutable.
        #
        #   FrozenInstanceError: cannot assign to field 'hetereogenous_dist_checkpoint'
        #
        # The failure surfaces deep inside ModelOpt state restore and names neither the
        # drafter nor this class. Hand MegatronModule a mutable structural copy instead;
        # readers of `.config` (draft_vocab_size in megatron/draft/utils.py) see the same
        # values, and the body keeps its frozen config untouched.
        body_config = getattr(body, "config", None)
        if dataclasses.is_dataclass(body_config):
            config_for_megatron = SimpleNamespace(
                **{
                    field.name: getattr(body_config, field.name)
                    for field in dataclasses.fields(body_config)
                }
            )
        else:
            config_for_megatron = body_config
        super().__init__(config=config_for_megatron)
        self.body = body
        self.lilicorr = head
        self.draft_config = config
        # Drafter-owned LM head as a TRAINABLE vocab-parallel shard (laid out like the
        # policy head), or None to use the policy's head.
        self.register_parameter("lm_head_weight", None)
        # Real (unpadded) vocab and this rank's vocab offset for global top-k ids.
        self.vocab_size: int | None = None
        self.vocab_start: int = 0
        self.vocab_tp_group: Any = None

    def set_vocab_parallel(
        self,
        *,
        vocab_size: int,
        vocab_start: int,
        tp_group: Any,
        own_lm_head: Tensor | None = None,
    ) -> None:
        if own_lm_head is not None:
            from megatron.core.tensor_parallel.layers import (
                set_tensor_model_parallel_attributes,
            )

            weight = torch.nn.Parameter(own_lm_head, requires_grad=True)
            # Mark it vocab-sharded so grad norm, the distributed optimizer and
            # checkpoints treat each rank's rows as distinct, not as replicas.
            set_tensor_model_parallel_attributes(
                weight, is_parallel=True, dim=0, stride=1
            )
            self.lm_head_weight = weight
        self.vocab_size = int(vocab_size)
        self.vocab_start = int(vocab_start)
        self.vocab_tp_group = tp_group

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: dict | None = None,
    ) -> Any:
        """Checkpoint the owned LM head as a vocab (dim 0) TP shard, not a replica."""
        from megatron.core.transformer.utils import (
            ensure_metadata_has_dp_cp_group,
            make_sharded_tensors_for_checkpoint,
            sharded_state_dict_default,
        )

        metadata = ensure_metadata_has_dp_cp_group(metadata)
        sharded = {}
        tp_group = getattr(self.body, "tp_group", None)
        for name, module in self.named_children():
            sharded.update(
                sharded_state_dict_default(
                    module,
                    f"{prefix}{name}.",
                    sharded_offsets,
                    metadata,
                    tp_group=tp_group,
                )
            )
        if self.lm_head_weight is not None:
            sharded.update(
                make_sharded_tensors_for_checkpoint(
                    {"lm_head_weight": self.lm_head_weight},
                    prefix,
                    {"lm_head_weight": 0},
                    sharded_offsets,
                    tp_group=self.vocab_tp_group,
                    dp_cp_group=metadata.get("dp_cp_group"),
                )
            )
        return sharded

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
        # output_weight is a vocab-parallel shard: take the GLOBAL top-k.
        cand_logprobs, cand_ids = vocab_parallel_topk_logprobs(
            mask_logits.detach(),
            topk,
            vocab_start=self.vocab_start,
            vocab_size=(
                self.vocab_size
                if self.vocab_size is not None
                else self.vocab_start + mask_logits.shape[-1]
            ),
            tp_group=self.vocab_tp_group,
        )

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
        # Grouped-convolution checkpoints are served with the convolutions (vLLM
        # LiLiCorr builds DFlash2 layers), so the trained body must run them too.
        conv_kernel_size=int(dflash_cfg.get("conv_kernel_size", 0) or 0),
        conv_group_size=int(dflash_cfg.get("conv_group_size", 0) or 0),
    )
    tp_group = getattr(pg_collection, "tp", None)
    body = DFlashBody(
        config=body_config,
        parallel_config=parallel_config,
        tp_group=tp_group,
    )

    head = build_lilicorr_head(head_cfg)

    # Load the released weights STRICTLY -- body and reranker. A silent partial load is
    # the failure this project keeps meeting: the module initialises, training runs, and
    # acceptance is quietly near zero because part of the drafter is at init values.
    load_lilicorr_body_weights(body, draft_config.model_name)
    load_lilicorr_weights(head, draft_config.model_name)

    model = LiLiCorrDraftModel(body=body, head=head, config=draft_config)

    from nemo_rl.models.megatron.draft.utils import (
        DRAFT_GRAD_NORM_GROUP,
        get_policy_lm_head_weight,
        register_draft_grad_norm_group,
    )

    policy_head = get_policy_lm_head_weight(policy_model_chunk)
    tp_rank = 0 if tp_group is None else torch.distributed.get_rank(tp_group)
    own_lm_head = None
    if ckpt_cfg.get("has_own_lm_head"):
        own_lm_head = load_lilicorr_lm_head_shard(
            draft_config.model_name, policy_lm_head_shard=policy_head, tp_rank=tp_rank
        )
    model.set_vocab_parallel(
        vocab_size=int(ckpt_cfg.get("draft_vocab_size", ckpt_cfg["vocab_size"])),
        vocab_start=tp_rank * policy_head.shape[0],
        tp_group=tp_group,
        own_lm_head=own_lm_head,
    )
    if model_provider.params_dtype is not None:
        model = model.to(model_provider.params_dtype)
    # Report the drafter's gradient norm separately from the policy's, as the DFlash
    # provider does for its body.
    register_draft_grad_norm_group()
    for parameter in model.parameters():
        parameter.grad_norm_group = DRAFT_GRAD_NORM_GROUP
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
    own_lm_head = getattr(module, "lm_head_weight", None)
    if own_lm_head is not None:
        # vLLM starts the drafter from dummy weights and fills it by refit, so the
        # drafter-owned head must be streamed even though it is frozen here.
        shard = own_lm_head.detach()
        group = module.vocab_tp_group
        world = 1 if group is None else torch.distributed.get_world_size(group)
        if world > 1:
            parts = [torch.empty_like(shard) for _ in range(world)]
            torch.distributed.all_gather(parts, shard, group=group)
            shard = torch.cat(parts, dim=0)
        exported.append(("lm_head.weight", shard[: module.vocab_size]))
    return exported
