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
"""Building the draft model and its optimizer for DTensor-v2 co-training.

Adapts a dspark/dflash/eagle3 checkpoint's config onto the vendored draft
models, validates it against the training options, and FSDP2-shards it. Also
builds the composite policy+draft optimizer used for checkpoint I/O.
"""

import os
from typing import Any, Optional

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from nemo_rl.models.automodel.draft.checkpoint import DSPARK_OPTIMIZER_GROUP_NAMES
from nemo_rl.models.automodel.draft.draft_qwen3 import Qwen3DSparkModel
from nemo_rl.models.automodel.draft.eagle3_llama import Eagle3DraftModel
from nemo_rl.models.automodel.draft.hidden_capture import _resolve_layers
from nemo_rl.models.policy import Eagle3DraftConfig

DSPARK_REQUIRED_CONFIG_FIELDS = ("block_size", "target_layer_ids", "mask_token_id")

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
                f"{spec_type!r} model but policy.draft.speculator_type={algo!r}."
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
            "policy.draft.speculator_type=dspark requires the next-token block layout "
            "(sample_from_anchor=true), but the checkpoint uses the dflash "
            "bonus-anchor layout; run it as algo=dflash instead."
        )
    if algo == "dflash" and sample_from_anchor:
        raise ValueError(
            "policy.draft.speculator_type=dflash requires the bonus-anchor block layout "
            "(sample_from_anchor=false), but the checkpoint uses the "
            "next-token layout; run it as algo=dspark instead."
        )
    confidence_alpha = float(dspark_options["confidence_loss_alpha"])
    if algo == "dflash":
        if int(getattr(draft_hf_config, "markov_rank", 0) or 0) > 0:
            raise ValueError(
                "policy.draft.speculator_type=dflash but the checkpoint carries a Markov "
                f"head (markov_rank={draft_hf_config.markov_rank}); run it as "
                "algo=dspark instead."
            )
        if bool(getattr(draft_hf_config, "enable_confidence_head", False)):
            raise ValueError(
                "policy.draft.speculator_type=dflash but the checkpoint carries a "
                "confidence head; run it as algo=dspark instead."
            )
        if confidence_alpha != 0:
            raise ValueError(
                "policy.draft.speculator_type=dflash requires "
                "policy.draft.confidence_loss_alpha=0 (DFlash has no "
                f"confidence head), got {confidence_alpha}."
            )
    if confidence_alpha > 0 and not bool(
        getattr(draft_hf_config, "enable_confidence_head", False)
    ):
        raise ValueError(
            "policy.draft.confidence_loss_alpha > 0 but the draft checkpoint "
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
    eagle3_options: "Eagle3DraftConfig",
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
            f"policy.draft.ttt_steps must be >= 1, got {eagle3_options.ttt_steps}."
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


def build_policy_and_draft_optimizer(
    optimizer_cls: Any,
    optimizer_kwargs: dict[str, Any],
    model: nn.Module,
    draft_model: nn.Module,
    draft_config: Any,
) -> torch.optim.Optimizer:
    """Build the optimizer with named [policy, draft] param groups; the draft group uses its own lr."""
    policy_name, draft_name = DSPARK_OPTIMIZER_GROUP_NAMES
    return optimizer_cls(
        [
            {
                "name": policy_name,
                "params": [p for p in model.parameters() if p.requires_grad],
            },
            {
                "name": draft_name,
                "params": [p for p in draft_model.parameters() if p.requires_grad],
                "lr": float(draft_config.learning_rate),
            },
        ],
        **optimizer_kwargs,
    )
