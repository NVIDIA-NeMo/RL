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
"""DSpark draft co-training integration for the DTensor-v2 policy worker.

Bridges the vendored DSpark modules with nemo-rl's automodel training loop:
building and sharding the draft from a pretrained checkpoint, capturing the
policy's hidden states and raw logits during the training forward, computing
the DSpark loss on rollout data, and checkpointing the draft alongside the
policy with a validated metadata record.
"""

import contextlib
import json
import os
import re
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor

from nemo_rl.models.automodel.draft.common import DSparkForwardOutput
from nemo_rl.models.automodel.draft.draft_qwen3 import Qwen3DSparkModel
from nemo_rl.models.automodel.draft.eagle3_llama import Eagle3DraftModel
from nemo_rl.models.automodel.draft.loss import compute_dspark_loss
from nemo_rl.models.policy import Eagle3DraftOptions

DSPARK_REQUIRED_CONFIG_FIELDS = ("block_size", "target_layer_ids", "mask_token_id")
DRAFT_CHECKPOINT_DIRNAME = "draft"
DSPARK_META_FILENAME = "dspark_meta.json"

# eagle3 checkpoints exported by SGLang SpecForge (e.g.
# lmsys/SGLang-EAGLE3-*): a plain HF-style flat config (no speculators
# wrapper), one decoder layer under the "midlayer." key prefix instead of
# the vendored model's "layers.0.", and no embed_tokens of their own (the
# drafter shares the target model's embedding table). See
# _adapt_native_flat_eagle3_config / _load_native_eagle3_weights.
NATIVE_FLAT_EAGLE3_ARCHS = ("LlamaForCausalLMEagle3",)


def load_draft_hf_config(
    model_name: str,
    algo: str = "dspark",
    target_num_hidden_layers: Optional[int] = None,
) -> Any:
    """Load a draft config for ``algo``, adapting speculators-format checkpoints.

    Two checkpoint families exist:
    - Flat qwen3-style configs (e.g. deepseek-ai/dspark_qwen3_8b_block7 and
      its dflash sibling): loadable via AutoConfig, drafter fields at the top
      level. Their ``target_layer_ids`` already use the trainer's
      output-of-layer convention and their blocks use the next-token
      supervision layout (they were produced by the trainer this code is
      vendored from), so no convention shifts apply.
    - Speculators-format configs (e.g. RedHatAI/*-speculator.*): no top-level
      model_type (AutoConfig fails), the draft transformer fields nest under
      ``transformer_layer_config``, and layer/block conventions follow
      vLLM/speculators indexing. These are adapted to the flat layout the
      vendored models expect.
    """
    from transformers import AutoConfig, PretrainedConfig

    if algo not in ("dspark", "dflash", "eagle3"):
        raise ValueError(f"Unknown draft algo {algo!r} for checkpoint {model_name}.")

    config_dict, _ = PretrainedConfig.get_config_dict(model_name)
    spec_type = config_dict.get("speculators_model_type")
    if spec_type is not None or "transformer_layer_config" in config_dict:
        spec_type = spec_type or "dspark"
        if spec_type != algo:
            raise ValueError(
                f"Draft checkpoint {model_name} is a speculators "
                f"{spec_type!r} model but policy.draft.algo={algo!r}."
            )
        if algo == "eagle3":
            return _adapt_speculators_eagle3_config(
                config_dict, model_name, target_num_hidden_layers
            )
        return _adapt_speculators_dspark_config(config_dict, algo=algo)
    if algo == "eagle3":
        architectures = config_dict.get("architectures") or []
        if not any(arch in architectures for arch in NATIVE_FLAT_EAGLE3_ARCHS):
            raise ValueError(
                f"Draft checkpoint {model_name} is not speculators-format and "
                f"does not declare one of {list(NATIVE_FLAT_EAGLE3_ARCHS)}; only "
                "speculators-format and SGLang SpecForge-style eagle3 "
                "checkpoints are supported on the DTensor-v2 path."
            )
        return _adapt_native_flat_eagle3_config(
            config_dict, model_name, target_num_hidden_layers
        )
    flat_config = AutoConfig.from_pretrained(model_name)
    if not hasattr(flat_config, "sample_from_anchor"):
        # Flat checkpoints come from the vendored trainer itself, whose blocks
        # always use the next-token supervision layout.
        flat_config.sample_from_anchor = True
    return flat_config


def _adapt_speculators_dspark_config(
    config_dict: dict[str, Any], algo: str = "dspark"
) -> Any:
    """Map a speculators-format dspark/dflash config onto the vendored layout."""
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    layer_cfg = dict(config_dict["transformer_layer_config"])
    model_type = layer_cfg.pop("model_type", "qwen3")
    if model_type != "qwen3":
        raise ValueError(
            f"Speculators {algo} adapter only supports qwen3 draft transformers, "
            f"got transformer_layer_config.model_type={model_type!r}."
        )
    adapted = Qwen3Config(**layer_cfg)
    adapted.architectures = list(
        config_dict.get("architectures")
        or (["DFlashDraftModel"] if algo == "dflash" else ["Qwen3DSparkModel"])
    )
    if "block_size" not in config_dict or "mask_token_id" not in config_dict:
        raise ValueError(
            f"Speculators {algo} checkpoint config is missing block_size or "
            f"mask_token_id (checkpoint architectures: {adapted.architectures})."
        )
    # The speculators block_size counts SLOTS (anchor + mask positions), the
    # same meaning as the vendored trainer's block_size, so it passes through
    # unchanged. What differs between the families is the supervision layout:
    # dspark (sample_from_anchor=True) supervises every slot on the NEXT
    # token, dflash (False) leaves the anchor slot unsupervised and each mask
    # slot predicts the token AT its own position — matching vLLM's dflash
    # speculator (1 + N query slots, ``sample_pos = query_pos``).
    sample_from_anchor = bool(config_dict.get("sample_from_anchor", algo == "dspark"))
    adapted.sample_from_anchor = sample_from_anchor
    adapted.block_size = int(config_dict["block_size"])
    adapted.mask_token_id = int(config_dict["mask_token_id"])
    # Convention shift: speculators/vLLM aux_hidden_state_layer_ids follow
    # vLLM's capture indexing, where the hidden recorded as id j is appended
    # AFTER layer j-1 runs (`_maybe_add_hidden_state(aux, idx + 1, ...)`),
    # i.e. j = output of decoder layer j-1 and j=0 = embedding output. The
    # vendored capture's target_layer_ids mean "output of layer i" with -1
    # for the embedding, so shift by -1 to feed the drafter the same
    # features it was pretrained on.
    adapted.target_layer_ids = [
        int(i) - 1 for i in config_dict["aux_hidden_state_layer_ids"]
    ]
    adapted.enable_confidence_head = bool(
        config_dict.get("enable_confidence_head", False)
    )
    if adapted.enable_confidence_head:
        adapted.confidence_head_with_markov = bool(
            config_dict["confidence_head_with_markov"]
        )
    adapted.markov_rank = int(config_dict.get("markov_rank", 0))
    if adapted.markov_rank > 0:
        adapted.markov_head_type = str(config_dict["markov_head_type"])
    if config_dict.get("draft_vocab_size"):
        adapted.draft_vocab_size = int(config_dict["draft_vocab_size"])
    return adapted


def default_eagle3_aux_layer_ids_vllm(target_num_hidden_layers: int) -> list[int]:
    """Default vLLM EAGLE3 aux layers for a target, in vLLM capture indexing.

    Mirrors ``SupportsEagle3.get_eagle3_default_aux_hidden_state_layers``:
    (2, N // 2, N - 3). The trainer capture uses these minus 1 (output-of-layer
    convention).
    """
    n = int(target_num_hidden_layers)
    return [2, n // 2, n - 3]


def _adapt_speculators_eagle3_config(
    config_dict: dict[str, Any],
    model_name: str,
    target_num_hidden_layers: Optional[int],
) -> Any:
    """Map a speculators-format eagle3 config onto automodel's LlamaEagle3DraftModel."""
    from transformers.models.llama.configuration_llama import LlamaConfig

    layer_cfg = dict(config_dict["transformer_layer_config"])
    model_type = layer_cfg.pop("model_type", "llama")
    # automodel's LlamaEagle3DraftModel (nemo_automodel.components.speculative
    # .eagle.draft_llama) only builds llama-style layers (no q/k norms). This
    # matches every published eagle3 checkpoint seen so far: speculators
    # publishes qwen3-target eagle3 drafts under the LLAMA model_type anyway
    # (e.g. RedHatAI/Qwen3-8B-speculator.eagle3), and both speculators
    # training and the pinned vLLM's serving drafter build llama-style
    # layers for them regardless of the target family.
    if model_type != "llama":
        raise ValueError(
            "Speculators eagle3 adapter only supports llama-family draft "
            f"transformers (automodel's LlamaEagle3DraftModel), got "
            f"transformer_layer_config.model_type={model_type!r} for {model_name}."
        )
    adapted = LlamaConfig(**layer_cfg)
    # Eagle3LlamaAttention reads the plain `attn_implementation` field, not
    # HF's private `_attn_implementation` (owned by PreTrainedModel).
    # flash_attention_2, not eager: eager attention materializes a dense
    # [B, H, T, T] fp32 softmax intermediate (H=32 heads here), which is a
    # ~12 GiB single allocation at a 10k-token packed row -- OOMs at full
    # 4n8g training scale. FlashAttention-2's varlen kernel (driven via
    # Eagle3DraftModel.forward's seq_lens) never materializes it.
    adapted.attn_implementation = "flash_attention_2"
    adapted.architectures = list(
        config_dict.get("architectures") or ["Eagle3Speculator"]
    )
    if not config_dict.get("draft_vocab_size"):
        raise ValueError(
            f"Speculators eagle3 checkpoint {model_name} config is missing "
            "draft_vocab_size."
        )
    adapted.draft_vocab_size = int(config_dict["draft_vocab_size"])
    if "norm_before_residual" not in config_dict:
        raise ValueError(
            f"Speculators eagle3 checkpoint {model_name} config is missing "
            "norm_before_residual; refusing to guess the first-layer residual "
            "convention."
        )
    adapted.norm_before_residual = bool(config_dict["norm_before_residual"])

    # Aux capture layers: pinned ids from the checkpoint when present, else
    # the same default selection vLLM applies for this target at serving
    # time; either way convert from vLLM indexing (id j = output of layer
    # j-1, 0 = embedding) to the trainer's output-of-layer convention.
    aux_ids = config_dict.get("eagle_aux_hidden_state_layer_ids") or config_dict.get(
        "aux_hidden_state_layer_ids"
    )
    if not aux_ids:
        if target_num_hidden_layers is None:
            raise ValueError(
                f"Speculators eagle3 checkpoint {model_name} pins no aux layer "
                "ids and no target layer count was provided to derive vLLM's "
                "default selection."
            )
        aux_ids = default_eagle3_aux_layer_ids_vllm(target_num_hidden_layers)
    adapted.target_layer_ids = [int(i) - 1 for i in aux_ids]
    adapted.num_aux_hidden_states = len(adapted.target_layer_ids)
    return adapted


def _adapt_native_flat_eagle3_config(
    config_dict: dict[str, Any],
    model_name: str,
    target_num_hidden_layers: Optional[int],
) -> Any:
    """Adapt a native (non-speculators) flat eagle3 config, e.g. SpecForge.

    Unlike the speculators family, these configs load through AutoConfig
    directly (draft_vocab_size etc. sit at the top level, model_type is a
    real HF value); what they don't record is the aux capture layer ids or
    the first-layer residual convention, so both are filled in the same way
    as the speculators adapter does when a checkpoint pins neither.
    """
    from transformers import AutoConfig

    adapted = AutoConfig.from_pretrained(model_name)
    # Eagle3LlamaAttention reads the plain `attn_implementation` field, not
    # HF's private `_attn_implementation` (owned by PreTrainedModel).
    # flash_attention_2, not eager: eager attention materializes a dense
    # [B, H, T, T] fp32 softmax intermediate (H=32 heads here), which is a
    # ~12 GiB single allocation at a 10k-token packed row -- OOMs at full
    # 4n8g training scale. FlashAttention-2's varlen kernel (driven via
    # Eagle3DraftModel.forward's seq_lens) never materializes it.
    adapted.attn_implementation = "flash_attention_2"
    if not getattr(adapted, "draft_vocab_size", None):
        raise ValueError(
            f"eagle3 draft checkpoint {model_name} config is missing draft_vocab_size."
        )
    aux_ids = config_dict.get("eagle_aux_hidden_state_layer_ids") or config_dict.get(
        "aux_hidden_state_layer_ids"
    )
    if not aux_ids:
        if target_num_hidden_layers is None:
            raise ValueError(
                f"eagle3 draft checkpoint {model_name} pins no aux layer ids "
                "and no target layer count was provided to derive vLLM's "
                "default selection."
            )
        aux_ids = default_eagle3_aux_layer_ids_vllm(target_num_hidden_layers)
    adapted.target_layer_ids = [int(i) - 1 for i in aux_ids]
    adapted.num_aux_hidden_states = len(adapted.target_layer_ids)
    if not hasattr(adapted, "norm_before_residual"):
        # Not recorded in the checkpoint and not independently verified
        # against the SpecForge training source -- confirm empirically
        # (does draft_loss/accept_rate converge sanely?) before trusting a
        # run's numbers, and flip this if it turns out wrong.
        adapted.norm_before_residual = False
    return adapted


class PolicyWithDraft(nn.Module):
    """Composite module pairing the policy and draft for optimizer state I/O.

    The Automodel checkpointer pairs one model with one optimizer for optimizer
    state save/load; the single optimizer here owns param groups from both the
    policy and the draft, so this composite is the module handed to the
    checkpointer's optimizer paths. Training, refit, and inference keep
    referencing the policy module directly.
    """

    def __init__(self, policy: nn.Module, draft: nn.Module):
        super().__init__()
        self.policy = policy
        self.draft = draft

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.policy(*args, **kwargs)


def validate_dspark_draft_config(
    draft_hf_config: Any, dspark_options: dict[str, Any], algo: str = "dspark"
) -> None:
    """Validate a dspark/dflash checkpoint's config against the training options.

    DFlash is the markov-free, confidence-free subset of DSpark: the same
    vendored model runs both, but a dflash run must not silently pick up
    markov/confidence behavior (or vice versa expect heads that don't exist).
    """
    missing = [
        field
        for field in DSPARK_REQUIRED_CONFIG_FIELDS
        if getattr(draft_hf_config, field, None) is None
    ]
    if missing:
        raise ValueError(
            f"{algo} draft checkpoint config.json is missing required fields: {missing}. "
            "Expected a checkpoint produced by DSpark/DFlash training (e.g. "
            "deepseek-ai/dspark_qwen3_8b_block7)."
        )
    architectures = getattr(draft_hf_config, "architectures", None) or []
    accepted_archs = {
        # deepseek exports both families under the DSpark class name.
        "dspark": ("Qwen3DSparkModel",),
        "dflash": ("DFlashDraftModel", "Qwen3DSparkModel"),
    }[algo]
    if not any(arch in architectures for arch in accepted_archs):
        raise ValueError(
            f"{algo} draft checkpoint must declare one of {list(accepted_archs)} "
            f"architectures, got {architectures}. Only Qwen3-family drafts are "
            "supported."
        )
    sample_from_anchor = bool(getattr(draft_hf_config, "sample_from_anchor", True))
    if algo == "dspark" and not sample_from_anchor:
        raise ValueError(
            "policy.draft.algo=dspark requires the next-token block layout "
            "(sample_from_anchor=true), but the checkpoint uses the dflash "
            "bonus-anchor layout; run it as algo=dflash instead."
        )
    if algo == "dflash" and sample_from_anchor:
        raise ValueError(
            "policy.draft.algo=dflash requires the bonus-anchor block layout "
            "(sample_from_anchor=false), but the checkpoint uses the "
            "next-token layout; run it as algo=dspark instead."
        )
    confidence_alpha = float(dspark_options["confidence_loss_alpha"])
    if algo == "dflash":
        if int(getattr(draft_hf_config, "markov_rank", 0) or 0) > 0:
            raise ValueError(
                "policy.draft.algo=dflash but the checkpoint carries a Markov "
                f"head (markov_rank={draft_hf_config.markov_rank}); run it as "
                "algo=dspark instead."
            )
        if bool(getattr(draft_hf_config, "enable_confidence_head", False)):
            raise ValueError(
                "policy.draft.algo=dflash but the checkpoint carries a "
                "confidence head; run it as algo=dspark instead."
            )
        if confidence_alpha != 0:
            raise ValueError(
                "policy.draft.algo=dflash requires "
                "policy.draft.dspark.confidence_loss_alpha=0 (DFlash has no "
                f"confidence head), got {confidence_alpha}."
            )
    if confidence_alpha > 0 and not bool(
        getattr(draft_hf_config, "enable_confidence_head", False)
    ):
        raise ValueError(
            "policy.draft.dspark.confidence_loss_alpha > 0 but the draft checkpoint "
            "has no confidence head (enable_confidence_head is false in config.json). "
            "Set confidence_loss_alpha: 0.0 or use a checkpoint with a confidence head."
        )
    l1_alpha = float(dspark_options["l1_loss_alpha"])
    if (
        l1_alpha < 0
        or confidence_alpha < 0
        or float(dspark_options["ce_loss_alpha"]) < 0
    ):
        raise ValueError("DSpark loss alphas must be non-negative.")


def build_dspark_draft_model(
    model_name: str,
    dspark_options: dict[str, Any],
    torch_dtype: torch.dtype,
    mesh: Any,
    algo: str = "dspark",
) -> Qwen3DSparkModel:
    """Build, validate, and FSDP2-shard a dspark/dflash draft from a checkpoint."""
    draft_hf_config = load_draft_hf_config(model_name, algo=algo)
    validate_dspark_draft_config(draft_hf_config, dspark_options, algo=algo)
    # num_anchors is a training-time knob, not checkpoint architecture; the
    # flex-attention implementation is required by the training block mask.
    draft_hf_config.num_anchors = int(dspark_options["num_anchors"])
    draft_hf_config._attn_implementation = "flex_attention"

    draft_model = Qwen3DSparkModel.from_pretrained(
        model_name,
        config=draft_hf_config,
        torch_dtype=torch_dtype,
    )
    draft_model = draft_model.to("cuda")

    draft_model.requires_grad_(True)
    train_embed_and_head = bool(dspark_options["train_embed_and_head"])
    draft_model.set_embedding_head_trainable(train_embed_and_head)

    from torch.distributed.fsdp import fully_shard

    for layer in draft_model.layers:
        fully_shard(layer, mesh=mesh)
    fully_shard(draft_model, mesh=mesh)
    return draft_model


def build_eagle3_draft_model(
    model_name: str,
    eagle3_options: "Eagle3DraftOptions",
    torch_dtype: torch.dtype,
    mesh: Any,
    target_num_hidden_layers: int,
    policy_model: Optional[nn.Module] = None,
) -> Eagle3DraftModel:
    """Build, validate, and FSDP2-shard an EAGLE3 draft from a checkpoint.

    ``policy_model`` is only required for native-flat (SpecForge-style)
    checkpoints, which ship no ``embed_tokens`` of their own -- see
    ``_load_eagle3_weights``.
    """
    draft_hf_config = load_draft_hf_config(
        model_name, algo="eagle3", target_num_hidden_layers=target_num_hidden_layers
    )
    # "Eagle3DraftModel" is the current speculators class name (e.g.
    # RedHatAI/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3);
    # "Eagle3Speculator" is an older speculators naming that some earlier
    # checkpoints still carry; NATIVE_FLAT_EAGLE3_ARCHS covers non-speculators
    # exports (e.g. SGLang SpecForge).
    accepted_archs = (
        "Eagle3Speculator",
        "Eagle3DraftModel",
        *NATIVE_FLAT_EAGLE3_ARCHS,
    )
    architectures = getattr(draft_hf_config, "architectures", None) or []
    if not any(arch in architectures for arch in accepted_archs):
        raise ValueError(
            f"eagle3 draft checkpoint must declare one of {list(accepted_archs)} "
            f"architectures, got {architectures}."
        )
    if int(eagle3_options.ttt_steps) < 1:
        raise ValueError(
            f"policy.draft.eagle3.ttt_steps must be >= 1, got "
            f"{eagle3_options.ttt_steps}."
        )

    # LlamaRotaryEmbedding sizes its cos/sin cache from config.torch_dtype,
    # defaulting to fp32 when unset (neither speculators-format checkpoints'
    # transformer_layer_config nor a bare adapted config carry this field).
    # An fp32 cache promotes q/k to fp32 through apply_rotary_pos_emb while
    # the cached V (untouched by RoPE) stays in torch_dtype, so eager
    # attention's attn_probs @ v0 mismatches dtypes; set it explicitly so
    # RoPE runs in the training dtype throughout, matching every other
    # tensor in the TTT forward.
    draft_hf_config.torch_dtype = torch_dtype
    draft_model = Eagle3DraftModel(draft_hf_config)
    _load_eagle3_weights(
        draft_model,
        model_name,
        policy_model,
        needs_embed_tokens_from_policy=any(
            arch in architectures for arch in NATIVE_FLAT_EAGLE3_ARCHS
        ),
    )
    draft_model = draft_model.to(torch_dtype).to("cuda")

    draft_model.requires_grad_(True)
    draft_model.set_embedding_head_trainable(bool(eagle3_options.train_embed_and_head))

    from torch.distributed.fsdp import fully_shard

    for layer in draft_model.model.layers:
        fully_shard(layer, mesh=mesh)
    fully_shard(draft_model, mesh=mesh)
    return draft_model


def _load_eagle3_weights(
    draft_model: Eagle3DraftModel,
    model_name: str,
    policy_model: Optional[nn.Module],
    needs_embed_tokens_from_policy: bool,
) -> None:
    """Load an eagle3 checkpoint's weights into automodel's LlamaEagle3DraftModel.

    Checkpoints publish a FLAT layout (no ``model.`` prefix -- the vendored
    single-file trainer this replaces used the same flat layout) with the
    lone decoder layer under either ``layers.0.`` (speculators/RedHatAI) or
    ``midlayer.`` (SGLang SpecForge/lmsys); both remap onto automodel's
    ``model.layers.0.`` (``lm_head``/``d2t``/``t2d`` stay top-level in both
    layouts, matching automodel's).

    Some checkpoints (SpecForge) ship no ``embed_tokens`` at all -- the
    drafter is meant to share the target model's embedding table. True
    weight tying (aliasing the same ``nn.Parameter``) isn't possible here:
    the draft and policy are ``fully_shard``-ed on different device meshes,
    so ``needs_embed_tokens_from_policy`` copies the policy's embedding
    VALUES once at init instead (matching how the megatron eagle path
    backfills a checkpoint's missing lm_head from the policy) and lets the
    copy train independently thereafter, governed by ``train_embed_and_head``
    like the rest of the embedding/head.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    weights_path = (
        os.path.join(model_name, "model.safetensors")
        if os.path.isdir(model_name)
        else hf_hub_download(model_name, "model.safetensors")
    )

    def _remap(key: str) -> str:
        if key in ("lm_head.weight", "d2t", "t2d"):
            return key
        if key.startswith("midlayer."):
            return "model.layers.0." + key[len("midlayer.") :]
        if key.startswith("layers.0."):
            return "model.layers.0." + key[len("layers.0.") :]
        return "model." + key

    state_dict = {_remap(key): value for key, value in load_file(weights_path).items()}

    if needs_embed_tokens_from_policy:
        if policy_model is None:
            raise ValueError(
                f"eagle3 draft checkpoint {model_name} ships no embed_tokens "
                "weight (it expects to share the target model's embedding "
                "table), but no policy model was provided to copy it from."
            )
        policy_base, _ = _resolve_layers(policy_model)
        policy_embed = getattr(policy_base, "embed_tokens", None)
        if policy_embed is None:
            raise ValueError(
                f"Cannot initialize eagle3 draft embed_tokens for {model_name}: "
                "the policy model has no `.embed_tokens`."
            )
        embed_weight = policy_embed.weight.detach()
        if isinstance(embed_weight, DTensor):
            embed_weight = embed_weight.full_tensor()
        state_dict["model.embed_tokens.weight"] = embed_weight.to(
            draft_model.model.embed_tokens.weight.dtype
        )

    missing, unexpected = draft_model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"eagle3 checkpoint {model_name} state dict mismatch after the "
            f"flat->model.-prefixed remap: missing={missing}, "
            f"unexpected={unexpected}."
        )


def _resolve_layers(
    policy_model: nn.Module,
) -> tuple[nn.Module, Any]:
    """Locate the decoder backbone (embedding owner) and its layer container.

    The container is an nn.ModuleList for HF-style stacks or an nn.ModuleDict
    keyed by stringified layer index for Automodel custom backbones (e.g.
    Qwen3_5Moe); use ``_layer_module`` to index either uniformly.
    """
    candidates = [getattr(policy_model, "model", None), policy_model]
    for base in candidates:
        if base is None:
            continue
        layers = getattr(base, "layers", None)
        if layers is not None:
            return base, layers
    raise ValueError(
        "DSpark hidden capture could not locate `.layers` on the policy "
        f"model (searched {type(policy_model).__name__}). Only HF-style "
        "decoder stacks are supported for DSpark co-training."
    )


def _layer_module(layers: Any, layer_id: int) -> nn.Module:
    if isinstance(layers, nn.ModuleDict):
        return layers[str(layer_id)]
    return layers[layer_id]


def _hook_output_tensor(output: Any) -> torch.Tensor:
    hidden = output[0] if isinstance(output, tuple) else output
    if isinstance(hidden, DTensor):
        # Under TP the layer output can be sharded or partial; materialize the
        # replicated full tensor (a no-op collective when already replicated).
        if any(not p.is_replicate() for p in hidden.placements):
            hidden = hidden.full_tensor()
        else:
            hidden = hidden.to_local()
    return hidden.detach()


class DSparkHiddenCapture:
    """Forward hooks capturing the policy's per-layer hiddens for the draft.

    Captured tensors are detached: the DSpark loss never backprops into the
    policy trunk. The hooks are "armed" by a pre-forward hook on the policy
    root and disarmed by ``collect()``: activation checkpointing replays layer
    forwards during backward (after the loss consumed the capture), and the
    disarmed hooks turn those replays into no-ops instead of re-running
    capture work (and, under TP, its collectives).
    """

    def __init__(self, policy_model: nn.Module, target_layer_ids: list[int]):
        self.target_layer_ids = [int(i) for i in target_layer_ids]
        self._policy_root = policy_model
        base, layers = _resolve_layers(policy_model)
        num_layers = len(layers)
        for layer_id in self.target_layer_ids:
            if layer_id != -1 and not (0 <= layer_id < num_layers):
                raise ValueError(
                    f"target_layer_id {layer_id} out of range for a policy with "
                    f"{num_layers} decoder layers."
                )
        self._modules_by_id: dict[int, nn.Module] = {}
        for layer_id in self.target_layer_ids:
            if layer_id == -1:
                embed = getattr(base, "embed_tokens", None)
                if embed is None:
                    raise ValueError(
                        "target_layer_ids includes -1 (embedding output) but the "
                        "policy model has no `.embed_tokens`."
                    )
                self._modules_by_id[layer_id] = embed
            else:
                self._modules_by_id[layer_id] = _layer_module(layers, layer_id)
        self._handles: list[Any] = []
        self._captured: dict[int, torch.Tensor] = {}
        self._armed = False

    @property
    def active(self) -> bool:
        return bool(self._handles)

    def activate(self) -> None:
        if self._handles:
            return

        def arm_hook(_module: nn.Module, _args: Any, _kwargs: Any) -> None:
            self._armed = True

        def make_layer_hook(layer_id: int):
            def hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
                if self._armed:
                    self._captured[layer_id] = _hook_output_tensor(output)

            return hook

        # The root pre-hook re-arms capture at each microbatch forward; the
        # checkpointed backward replay calls only layer forwards, so capture
        # stays disarmed there.
        self._handles.append(
            self._policy_root.register_forward_pre_hook(arm_hook, with_kwargs=True)
        )
        for layer_id, module in self._modules_by_id.items():
            self._handles.append(
                module.register_forward_hook(make_layer_hook(layer_id))
            )

    def deactivate(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self.clear()

    def clear(self) -> None:
        self._captured = {}
        self._armed = False

    def collect(self) -> torch.Tensor:
        """Concatenated target hidden states captured by the layer hooks."""
        missing = [i for i in self.target_layer_ids if i not in self._captured]
        if missing:
            raise RuntimeError(
                "DSpark hidden capture did not observe the policy forward "
                f"(missing layer ids {missing}). The capture hooks must be "
                "active during the training forward."
            )
        self._armed = False
        return torch.cat([self._captured[i] for i in self.target_layer_ids], dim=-1)


def anchor_sampling_seed(
    dp_rank: int, global_batch_index: int, microbatch_index: int
) -> int:
    """Deterministic seed for the draft's anchor sampling.

    A pure function of the DP rank and the training progress counters: every
    TP/CP peer of the same DP slice derives the identical seed (the draft is
    replicated across them, so they must sample identical anchors or the
    replicas drift apart), while DP ranks, global batches, and microbatches
    all draw differently. Independent of each rank's ambient RNG state.
    """
    seed = 0x5EED_D5BA
    for value in (dp_rank, global_batch_index, microbatch_index):
        # Splitmix64-style mixing keeps nearby counter values decorrelated.
        seed = (seed ^ (value + 0x9E3779B97F4A7C15)) * 0xBF58476D1CE4E5B9
        seed &= (1 << 63) - 1
    return seed


class _DraftRuntimeBase:
    """Shared per-worker draft co-training state and distributed plumbing.

    Subclasses supply the loss (``compute_loss``) and the layer-id source for
    the hidden capture (``_capture_layer_ids``); everything here is common to
    every drafter family: the option/group state, the microbatch-slot
    normalization, and the teacher-logits stash.
    """

    # Algo family name used in error messages.
    _algo_label = "Draft"

    def __init__(
        self,
        draft_model: nn.Module,
        options: Any,
        loss_weight: float,
        dp_group: Optional[dist.ProcessGroup],
        tp_group: Optional[dist.ProcessGroup] = None,
        cp_group: Optional[dist.ProcessGroup] = None,
    ):
        self.draft_model = draft_model
        self.options = options
        self.loss_weight = float(loss_weight)
        self.dp_group = dp_group
        self.tp_group = tp_group
        # Under context parallelism the captured hiddens, teacher logits, and
        # input_ids are sequence-sharded (load-balanced); compute_loss gathers
        # them to the full sequence and every CP peer runs the identical
        # full-sequence draft forward (grads stay consistent through the
        # draft's dp_cp FSDP mesh, like the TP-replicated case).
        self.cp_group = cp_group
        self.capture: Optional[DSparkHiddenCapture] = None
        self._teacher_logits: Optional[torch.Tensor] = None
        self._num_microbatch_slots: int = 1
        self._dp_rank: int = dist.get_rank(dp_group) if dp_group is not None else 0
        self._global_batch_index: int = 0
        self._microbatch_index: int = 0

    @property
    def _cp_size(self) -> int:
        return dist.get_world_size(self.cp_group) if self.cp_group is not None else 1

    def begin_global_batch(self, num_microbatch_slots: int) -> None:
        """Record this global batch's microbatch-slot count (identical on all ranks).

        The training loop sums per-microbatch losses across gradient
        accumulation, while each draft microbatch loss is a mean over that
        microbatch's (DP-all-reduced) supervised tokens. Dividing every
        microbatch term by the slot count turns the sum into an average of
        per-slot global means, keeping the draft gradient scale independent of
        gbs/mbs — the same effective scale as the policy loss, which normalizes
        each microbatch by the whole-global-batch token denominator instead.
        """
        self._num_microbatch_slots = max(int(num_microbatch_slots), 1)
        self._global_batch_index += 1
        self._microbatch_index = 0

    def _capture_layer_ids(self) -> list[int]:
        raise NotImplementedError

    def attach_capture(self, policy_model: nn.Module) -> None:
        if self.capture is None:
            self.capture = DSparkHiddenCapture(policy_model, self._capture_layer_ids())

    def stash_teacher_logits(
        self, logits: torch.Tensor, will_scale_inplace: bool
    ) -> None:
        """Record the policy's raw logits before any in-place temperature scaling.

        With temperature 1.0 no scaling happens and a detached view is exact and
        free; otherwise the in-place div would corrupt the view, so clone first.

        Under TP the logits arrive as a vocab-sharded DTensor; it is stashed
        as-is and gathered to the full vocab lazily in ``compute_loss`` so the
        full-size tensor does not coexist with the policy-loss peak.
        """
        if not isinstance(logits, DTensor) and self.tp_group is not None:
            if dist.get_world_size(self.tp_group) > 1:
                raise RuntimeError(
                    f"{self._algo_label} co-training with tensor_parallel_size "
                    "> 1 expects the policy logits as a vocab-sharded DTensor, "
                    f"but got a plain {type(logits).__name__}; a bare local "
                    "shard cannot be gathered safely."
                )
        raw = logits.detach()
        self._teacher_logits = raw.clone() if will_scale_inplace else raw

    def _require_capture_and_teacher(self) -> torch.Tensor:
        """Common compute_loss preamble: guards plus the stashed teacher."""
        if self.capture is None or not self.capture.active:
            raise RuntimeError(
                f"{self._algo_label} loss requested but hidden capture is not "
                "active; the worker must activate capture around the training "
                "forward."
            )
        if self._teacher_logits is None:
            raise RuntimeError(
                f"{self._algo_label} loss requested but no teacher logits were "
                "stashed for this microbatch."
            )
        return self._teacher_logits


class DSparkRuntime(_DraftRuntimeBase):
    """Per-worker DSpark co-training state: draft model, capture, and loss."""

    _algo_label = "DSpark"

    def __init__(
        self,
        draft_model: Qwen3DSparkModel,
        dspark_options: dict[str, Any],
        loss_weight: float,
        dp_group: Optional[dist.ProcessGroup],
        tp_group: Optional[dist.ProcessGroup] = None,
        cp_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__(
            draft_model=draft_model,
            options=dspark_options,
            loss_weight=loss_weight,
            dp_group=dp_group,
            tp_group=tp_group,
            cp_group=cp_group,
        )

    def _capture_layer_ids(self) -> list[int]:
        return list(self.draft_model.config.target_layer_ids)

    def compute_loss(self, data_dict: Any) -> tuple[torch.Tensor, dict[str, float]]:
        """Run the draft forward on the captured hiddens and compute the DSpark loss."""
        teacher_logits = self._require_capture_and_teacher()
        if isinstance(teacher_logits, DTensor):
            # Vocab-sharded under TP; the draft's teacher gather needs the full
            # vocab dimension.
            teacher_logits = teacher_logits.full_tensor()
        if getattr(self.draft_model, "draft_vocab_size", None) is not None:
            # Reduced-vocab drafts compare distributions over the draft vocab;
            # mapping the teacher through d2t BEFORE the CP allgather shrinks
            # the gathered tensor by vocab_ratio (e.g. 248320 -> 32000). Note
            # d2t stores offsets (target_id = draft_idx + d2t[draft_idx]).
            teacher_logits = teacher_logits.index_select(
                -1, self.draft_model._get_d2t_target_ids().to(teacher_logits.device)
            )
        target_hidden_states = self.capture.collect()

        input_ids = data_dict["input_ids"]
        if isinstance(input_ids, DTensor):
            # Under CP the loss prep wraps the load-balanced LOCAL shard as a
            # DTensor with a (nominal) contiguous Shard(1) placement, so
            # full_tensor() would return a mis-ordered sequence. Take the
            # local shard; the CP branch below restores the true order via
            # allgather_cp_sharded_tensor.
            input_ids = input_ids.to_local()
        loss_mask = data_dict["token_mask"].float()
        if "sample_mask" in data_dict:
            loss_mask = loss_mask * data_dict["sample_mask"].float().unsqueeze(-1)

        if self._cp_size > 1:
            from nemo_rl.distributed.model_utils import allgather_cp_sharded_tensor

            # Captured tensors are sequence-local (load-balanced CP shards);
            # restore the full contiguous sequence on every CP peer. loss_mask
            # comes from the un-sharded data dict and is already full-length.
            teacher_logits = allgather_cp_sharded_tensor(
                teacher_logits, self.cp_group, seq_dim=1
            )
            target_hidden_states = allgather_cp_sharded_tensor(
                target_hidden_states, self.cp_group, seq_dim=1
            )
            if input_ids.size(1) != loss_mask.size(1):
                input_ids = allgather_cp_sharded_tensor(
                    input_ids, self.cp_group, seq_dim=1
                )
            if input_ids.size(1) != loss_mask.size(1) or target_hidden_states.size(
                1
            ) != loss_mask.size(1):
                raise RuntimeError(
                    "DSpark CP gather produced inconsistent sequence lengths: "
                    f"input_ids={input_ids.size(1)}, "
                    f"hiddens={target_hidden_states.size(1)}, "
                    f"loss_mask={loss_mask.size(1)}."
                )

        self._microbatch_index += 1
        anchor_generator = torch.Generator(device=input_ids.device)
        anchor_generator.manual_seed(
            anchor_sampling_seed(
                self._dp_rank, self._global_batch_index, self._microbatch_index
            )
        )
        outputs: DSparkForwardOutput = self.draft_model(
            input_ids=input_ids,
            target_hidden_states=target_hidden_states,
            loss_mask=loss_mask,
            teacher_logits=teacher_logits,
            anchor_generator=anchor_generator,
        )
        loss, terms = compute_dspark_loss(
            outputs=outputs,
            loss_decay_gamma=float(self.options["loss_decay_gamma"]),
            ce_loss_alpha=float(self.options["ce_loss_alpha"]),
            l1_loss_alpha=float(self.options["l1_loss_alpha"]),
            confidence_head_alpha=float(self.options["confidence_loss_alpha"]),
            process_group=self.dp_group,
            return_terms=True,
        )
        metrics = self._terms_to_metrics(terms)
        # See begin_global_batch: average across microbatch slots so the summed
        # backward matches the policy loss's global-mean gradient scale.
        loss = loss / self._num_microbatch_slots

        # Free per-microbatch stash; the next forward re-stashes.
        self._teacher_logits = None
        self.capture.clear()
        return loss, metrics

    @staticmethod
    def _terms_to_metrics(terms: dict[str, torch.Tensor]) -> dict[str, float]:
        """Flatten the loss terms into shape-stable scalar metrics.

        Ratios (tau, accept_rate) are emitted as raw ``*_num``/``*_den`` pairs,
        NOT pre-divided: the training-loop metric aggregation
        (``grpo.py``'s per-key reduction) sums every metric not on its small
        mean-reduction allowlist across all microbatches and DP ranks, so a
        pre-divided per-microbatch ratio would be summed into a meaningless
        value (see ``finalize_draft_ratio_metrics``, which the training loop
        calls after that reduction to turn the summed num/den pairs back into
        the correct token-weighted global ratio).
        """
        metrics: dict[str, float] = {
            "draft_loss": float(terms["loss"].item()),
            "draft_ce_loss": float(terms["ce_loss"].item()),
            "draft_tv_loss": float(terms["l1_loss"].item()),
            "draft_conf_loss": float(terms["confidence_loss"].item()),
            "draft_tau_num": float(terms["tau_num"].item()),
            "draft_tau_den": float(terms["tau_den"].item()),
        }
        # One host transfer per vector instead of one sync per position.
        pos_nums = terms["accept_rate_per_pos_num"].tolist()
        pos_dens = terms["accept_rate_per_pos_den"].tolist()
        for k, (num_k, den_k) in enumerate(zip(pos_nums, pos_dens)):
            metrics[f"draft_accept_rate_num@{k + 1}"] = num_k
            metrics[f"draft_accept_rate_den@{k + 1}"] = den_k
        return metrics


@contextlib.contextmanager
def draft_capture_ctx(runtime: Any):
    """Keep the hidden-capture hooks active for the duration of a train call.

    Works for any draft runtime exposing ``capture`` (dspark/dflash/eagle3).
    """
    assert runtime.capture is not None, "attach_capture must run before training."
    runtime.capture.activate()
    try:
        yield
    finally:
        runtime.capture.deactivate()


def next_token_position_mask(token_mask: torch.Tensor) -> torch.Tensor:
    """Shift a per-TOKEN mask onto the POSITIONS whose logits predict it.

    Rollout ``token_mask[t]`` marks token t as supervised (a response token),
    but logits at position t predict token t+1 (the policy loss paths apply
    ``token_mask[:, 1:]`` for the same reason). The eagle3 TTT forward gates
    loss at logit/teacher positions, so its mask must be
    ``mask[t] = token_mask[t + 1]`` with a zero tail: the final position has
    no next-token label, and the last-prompt-token position (whose label is
    the FIRST response token — exactly where drafting starts) is supervised.
    """
    shifted = torch.zeros_like(token_mask)
    shifted[:, :-1] = token_mask[:, 1:]
    return shifted


class Eagle3Runtime(_DraftRuntimeBase):
    """Per-worker EAGLE3 co-training state: draft model, capture, TTT loss.

    Exposes the same runtime protocol as DSparkRuntime so the worker and
    loss wrapper treat all draft algos uniformly.
    """

    _algo_label = "EAGLE3"

    def __init__(
        self,
        draft_model: Eagle3DraftModel,
        eagle3_options: Eagle3DraftOptions,
        loss_weight: float,
        dp_group: Optional[dist.ProcessGroup],
        tp_group: Optional[dist.ProcessGroup] = None,
        cp_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__(
            draft_model=draft_model,
            options=eagle3_options,
            loss_weight=loss_weight,
            dp_group=dp_group,
            tp_group=tp_group,
            cp_group=cp_group,
        )

    def _capture_layer_ids(self) -> list[int]:
        return list(self.draft_model.target_layer_ids)

    def compute_loss(self, data_dict: Any) -> tuple[torch.Tensor, dict[str, float]]:
        """Run the TTT draft forward on captured hiddens and compute the loss."""
        teacher_logits = self._require_capture_and_teacher()
        if isinstance(teacher_logits, DTensor):
            teacher_logits = teacher_logits.full_tensor()
        # Map the teacher into draft-vocab order before any CP gather so the
        # gathered tensor is draft_vocab wide, not target_vocab wide.
        teacher_logits = teacher_logits.index_select(
            -1, self.draft_model.get_d2t_target_ids().to(teacher_logits.device)
        )

        fused_hidden = self.capture.collect()

        input_ids = data_dict["input_ids"]
        if isinstance(input_ids, DTensor):
            # CP loss prep wraps the load-balanced local shard with a nominal
            # contiguous placement; take the local and restore order below.
            input_ids = input_ids.to_local()
        if "input_lengths" not in data_dict:
            raise RuntimeError(
                "EAGLE3 co-training requires input_lengths in the microbatch to "
                "mark padding for the packed-row document mask."
            )
        input_lengths = data_dict["input_lengths"].to(fused_hidden.device)
        # See next_token_position_mask: the TTT forward gates loss at logit
        # positions (position t supervises token t + 1), not at token indices.
        loss_mask = next_token_position_mask(data_dict["token_mask"].float())
        if "sample_mask" in data_dict:
            loss_mask = loss_mask * data_dict["sample_mask"].float().unsqueeze(-1)

        if self._cp_size > 1:
            from nemo_rl.distributed.model_utils import allgather_cp_sharded_tensor

            teacher_logits = allgather_cp_sharded_tensor(
                teacher_logits, self.cp_group, seq_dim=1
            )
            fused_hidden = allgather_cp_sharded_tensor(
                fused_hidden, self.cp_group, seq_dim=1
            )
            if input_ids.size(1) != loss_mask.size(1):
                input_ids = allgather_cp_sharded_tensor(
                    input_ids, self.cp_group, seq_dim=1
                )

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Flatten the padded batch into one packed row: document ids separate
        # the sequences in the attention mask (padding marked -1) and
        # per-sequence position ids restart at 1, matching the speculators
        # packed-row layout.
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        valid = positions < input_lengths.unsqueeze(1)
        document_ids = torch.where(
            valid,
            torch.arange(batch_size, device=device).unsqueeze(1),
            torch.full_like(positions.expand(batch_size, -1), -1),
        )
        position_ids = (1 + positions).expand(batch_size, -1)

        total = batch_size * seq_len
        # Every sample occupies a fixed seq_len-wide slot in the packed row
        # (padding inside the slot, at its tail); for the flash_attention_2
        # path each slot is declared as one FlashAttention-varlen "document"
        # of that fixed width -- see Eagle3DraftModel.forward's seq_lens
        # docstring for why that's safe despite the internal padding. Built
        # unconditionally; Eagle3DraftModel ignores it on the eager path.
        seq_lens = torch.full((1, batch_size), seq_len, dtype=torch.long, device=device)

        # Called as self.draft_model(...) (nn.Module.__call__), not a bare
        # method: FSDP2's fully_shard unshard/reshard hooks fire on
        # __call__, and this is the only entry point that runs the WHOLE
        # TTT unroll under a single top-level call (see Eagle3DraftModel).
        terms = self.draft_model(
            fused_hidden_states=fused_hidden.reshape(1, total, -1),
            input_ids=input_ids.reshape(1, total),
            document_ids=document_ids.reshape(1, total),
            loss_mask=(loss_mask > 0.5).reshape(1, total),
            teacher_logits=teacher_logits.reshape(1, total, -1),
            position_ids=position_ids.reshape(1, total),
            ttt_steps=int(self.options.ttt_steps),
            seq_lens=seq_lens,
        )

        decay = float(self.options.ttt_step_loss_decay)
        # One all-reduce over the stacked per-step denominators instead of one
        # tiny collective per TTT step (elementwise sum-reduce is identical).
        dens_global = torch.stack([den.detach() for den in terms.loss_dens])
        if self.dp_group is not None and dist.is_initialized():
            dist.all_reduce(dens_global, group=self.dp_group)
        loss = fused_hidden.new_zeros((), dtype=torch.float32)
        for step, num in enumerate(terms.loss_nums):
            loss = loss + (decay**step) * num / dens_global[step]

        # Metrics: one host transfer for all per-step scalars instead of ~5
        # device syncs per step (see DSparkRuntime._terms_to_metrics).
        stats = (
            torch.stack(
                [
                    torch.stack(terms.loss_nums).detach(),
                    torch.stack(terms.loss_dens).detach(),
                    torch.stack(terms.full_acc_nums),
                    torch.stack(terms.full_acc_dens),
                    torch.stack(terms.cond_acc_nums),
                    torch.stack(terms.cond_acc_dens),
                ]
            )
            .float()
            .tolist()
        )
        loss_nums, loss_dens, full_nums, full_dens, cond_nums, cond_dens = stats
        # Raw num/den pairs, NOT pre-divided: see
        # DSparkRuntime._terms_to_metrics / finalize_draft_ratio_metrics for
        # why (grpo.py sums per-microbatch metrics across microbatches and DP
        # ranks by default; a pre-divided ratio would be summed into a
        # meaningless value).
        metrics: dict[str, float] = {}
        for step in range(len(loss_nums)):
            metrics[f"draft_ttt_loss_num@{step}"] = loss_nums[step]
            metrics[f"draft_ttt_loss_den@{step}"] = loss_dens[step]
            metrics[f"draft_full_acc_num@{step}"] = full_nums[step]
            metrics[f"draft_full_acc_den@{step}"] = full_dens[step]
            metrics[f"draft_cond_acc_num@{step}"] = cond_nums[step]
            metrics[f"draft_cond_acc_den@{step}"] = cond_dens[step]
        metrics["draft_loss"] = float(loss.item())
        # See DSparkRuntime.begin_global_batch: average across microbatch
        # slots so the summed backward keeps a global-mean gradient scale.
        loss = loss / self._num_microbatch_slots

        self._teacher_logits = None
        self.capture.clear()
        return loss, metrics


_RATIO_NUM_RE = re.compile(r"^(.*)_num(@\d+)?$")


def finalize_draft_ratio_metrics(metrics: dict[str, Any]) -> None:
    """Turn summed ``*_num``/``*_den`` metric pairs back into ratios, in place.

    DSparkRuntime/Eagle3Runtime emit per-microbatch accuracy/rate metrics as
    raw num/den pairs instead of pre-divided ratios, because the training
    loop's default per-key metric reduction (``grpo.py``) sums every metric
    not on its small mean-reduction allowlist across all microbatches and DP
    ranks -- summing a pre-divided per-microbatch ratio would produce a
    meaningless value (e.g. a "draft_full_acc@0" of 240 instead of a rate in
    [0, 1]). Call this AFTER that per-key reduction has summed the raw num/den
    pairs into global totals, so the ratio it computes is the correct
    token-weighted global rate.
    """
    for key in list(metrics.keys()):
        match = _RATIO_NUM_RE.match(key)
        if match is None:
            continue
        base, suffix = match.group(1), match.group(2) or ""
        den_key = f"{base}_den{suffix}"
        if den_key not in metrics:
            continue
        num = metrics.pop(key)
        den = metrics.pop(den_key)
        metrics[f"{base}{suffix}"] = num / den if den > 0 else 0.0


DSPARK_OPTIMIZER_GROUP_NAMES = ("policy", "draft")


def optimizer_layout_record(
    optimizer: torch.optim.Optimizer,
) -> list[dict[str, Any]]:
    """Record named param-group layout for resume validation.

    A DSpark co-training optimizer must consist of exactly the named groups
    ["policy", "draft"] in that order; anything else means the optimizer was
    not built by the dspark setup path and its state cannot be paired safely.
    """
    names = [group.get("name") for group in optimizer.param_groups]
    if tuple(names) != DSPARK_OPTIMIZER_GROUP_NAMES:
        raise ValueError(
            "DSpark co-training expects optimizer param groups named "
            f"{list(DSPARK_OPTIMIZER_GROUP_NAMES)} in that order, got {names}. "
            "The optimizer was not constructed by the dspark setup path."
        )
    return [
        {"name": group["name"], "num_params": len(group["params"])}
        for group in optimizer.param_groups
    ]


DRAFT_META_VERSION = 1


def draft_meta_record(
    draft_model: nn.Module,
    model_name: str,
    optimizer: Optional[torch.optim.Optimizer],
    algo: str = "dspark",
    train_embed_and_head: Optional[bool] = None,
    ttt_steps: Optional[int] = None,
) -> dict[str, Any]:
    """Versioned per-algo draft checkpoint metadata.

    Legacy dspark checkpoints predate versioning (no meta_version/algo); the
    validator treats those as dspark/v0. ``ttt_steps`` is part of the eagle3
    training contract (the unroll depth the drafter was trained for) and is
    required there; the block-drafter algos ignore it.
    """
    config = draft_model.config
    record: dict[str, Any] = {
        "meta_version": DRAFT_META_VERSION,
        "algo": algo,
        "model_name": model_name,
        "train_embed_and_head": train_embed_and_head,
        "optimizer_layout": optimizer_layout_record(optimizer) if optimizer else None,
    }
    if algo in ("dspark", "dflash"):
        record.update(
            {
                "block_size": int(config.block_size),
                "mask_token_id": int(config.mask_token_id),
                "target_layer_ids": [int(i) for i in config.target_layer_ids],
                "draft_vocab_size": getattr(config, "draft_vocab_size", None),
                "sample_from_anchor": bool(getattr(config, "sample_from_anchor", True)),
            }
        )
    elif algo == "eagle3":
        if ttt_steps is None:
            raise ValueError(
                "eagle3 draft checkpoint metadata requires ttt_steps (the TTT "
                "unroll depth is part of the training contract)."
            )
        record.update(
            {
                "aux_layer_ids": [int(i) for i in config.target_layer_ids],
                "draft_vocab_size": int(config.draft_vocab_size),
                "ttt_steps": int(ttt_steps),
            }
        )
    else:
        raise ValueError(f"Unknown draft algo {algo!r} for checkpoint metadata.")
    return record


def draft_checkpoint_dir(weights_path: str) -> str:
    """The draft's DCP directory: a SIBLING of the policy weights directory.

    The draft must live outside the policy weight tree because
    ``detect_checkpoint_format(weights_path)`` walks it recursively and would
    mis-detect the safetensors policy checkpoint as DCP after seeing the
    draft's ``.distcp`` files.
    """
    # abspath normalizes trailing slashes before dirname takes the parent.
    normalized = os.path.abspath(weights_path)
    # The Automodel checkpoint loader accepts a weights_path that points
    # directly at the ``.../weights/model`` subdirectory; the draft sibling
    # lives next to ``weights`` either way, so strip that trailing component
    # before deriving the parent.
    if os.path.basename(normalized) == "model":
        normalized = os.path.dirname(normalized)
    return os.path.join(os.path.dirname(normalized), DRAFT_CHECKPOINT_DIRNAME)


def save_draft_checkpoint(
    draft_model: nn.Module,
    weights_path: str,
    meta: dict[str, Any],
) -> None:
    """Save the draft's sharded weights (DCP) plus the dspark metadata record."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_model_state_dict

    draft_dir = draft_checkpoint_dir(weights_path)
    state_dict = get_model_state_dict(draft_model)
    dcp.save(state_dict, checkpoint_id=draft_dir)
    if not dist.is_initialized() or dist.get_rank() == 0:
        with open(os.path.join(draft_dir, DSPARK_META_FILENAME), "w") as f:
            json.dump(meta, f, indent=2)
    if dist.is_initialized():
        dist.barrier()


def validate_dspark_checkpoint_meta(
    saved_meta: dict[str, Any], expected_meta: dict[str, Any]
) -> None:
    """Hard-error on inconsistent dspark checkpoint metadata.

    Architecture fields are always compared. The optimizer layout is compared
    only when the current run restores optimizer state (expected layout is
    non-None): weights-only loads (init_optimizer=False, e.g. eval or
    logprob-only workers) legitimately carry no layout while training
    checkpoints do.
    """
    # Legacy dspark metadata predates versioning: missing algo means dspark,
    # missing meta_version means v0. Never resume across algos.
    saved_algo = saved_meta.get("algo", "dspark")
    expected_algo = expected_meta.get("algo", "dspark")
    if saved_algo != expected_algo:
        raise ValueError(
            f"Draft checkpoint metadata algo mismatch: checkpoint has "
            f"{saved_algo!r}, current run expects {expected_algo!r}. "
            "Refusing to resume with an inconsistent draft configuration."
        )
    if expected_algo == "eagle3":
        keys = ["aux_layer_ids", "draft_vocab_size", "ttt_steps"]
    else:
        keys = ["block_size", "mask_token_id", "target_layer_ids"]
        # draft_vocab_size and sample_from_anchor were added with versioning;
        # only compare when the checkpoint recorded them.
        for versioned_key in ("draft_vocab_size", "sample_from_anchor"):
            if versioned_key in saved_meta:
                keys.append(versioned_key)
    if expected_meta.get("optimizer_layout") is not None:
        keys.append("optimizer_layout")
    for key in keys:
        if saved_meta.get(key) != expected_meta.get(key):
            raise ValueError(
                f"Draft checkpoint metadata mismatch for '{key}': checkpoint has "
                f"{saved_meta.get(key)!r}, current run expects {expected_meta.get(key)!r}. "
                "Refusing to resume with an inconsistent draft configuration."
            )


def load_draft_checkpoint(
    draft_model: nn.Module,
    weights_path: str,
    expected_meta: dict[str, Any],
) -> None:
    """Load the draft's weights from a checkpoint, validating the metadata record."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        get_model_state_dict,
        set_model_state_dict,
    )

    draft_dir = draft_checkpoint_dir(weights_path)
    meta_path = os.path.join(draft_dir, DSPARK_META_FILENAME)
    if not os.path.isdir(draft_dir) or not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"DSpark draft entry not found in checkpoint at {draft_dir}. A dspark "
            "run cannot resume from a checkpoint saved without draft state."
        )
    with open(meta_path) as f:
        saved_meta = json.load(f)
    validate_dspark_checkpoint_meta(saved_meta, expected_meta)
    state_dict = get_model_state_dict(draft_model)
    dcp.load(state_dict, checkpoint_id=draft_dir)
    set_model_state_dict(draft_model, state_dict)


def load_dspark_checkpoint(
    checkpoint_manager: Any,
    model: nn.Module,
    draft_model: nn.Module,
    composite_model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Any,
    weights_path: str,
    optimizer_path: Optional[str],
    model_name: str,
    algo: str = "dspark",
    ttt_steps: Optional[int] = None,
) -> None:
    """Load a dspark checkpoint with the composite-optimizer pairing rule.

    Policy weights load exactly as without a draft; draft weights load from
    the sibling draft entry with validated metadata; optimizer state pairs
    with the composite module whose param groups span policy + draft. This is
    the single home of that pairing invariant for both resume-at-setup and
    mid-run checkpoint loads.
    """
    checkpoint_manager.load_checkpoint(model=model, weights_path=weights_path)
    # Enforce the optimizer-layout record only when optimizer state is actually
    # restored: weights-only loads (no optimizer_path) must accept checkpoints
    # saved without a layout record or with a different optimizer grouping.
    restoring_optimizer = bool(optimizer_path) and optimizer is not None
    load_draft_checkpoint(
        draft_model,
        weights_path,
        expected_meta=draft_meta_record(
            draft_model,
            model_name,
            optimizer if restoring_optimizer else None,
            algo=algo,
            ttt_steps=ttt_steps,
        ),
    )
    if optimizer_path and optimizer is not None:
        checkpoint_manager.checkpointer.load_optimizer(
            optimizer=optimizer,
            model=composite_model,
            weights_path=optimizer_path,
            scheduler=scheduler,
        )
