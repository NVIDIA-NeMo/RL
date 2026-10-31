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

"""Draft (speculative-decoding) co-training configuration.

The ``speculator_type`` field discriminates the drafter family; each concrete
config carries only the fields that family uses (no more nested
``dspark:``/``eagle3:`` sub-blocks). This mirrors the config shape adopted by
the Megatron-side draft co-training work (NVIDIA-NeMo/RL#3701) so both
backends read ``policy.draft`` the same way.
"""

import difflib
from collections.abc import Mapping
from typing import Annotated, Any, ClassVar, Literal, Self, Union

from pydantic import BaseModel, Discriminator, Field, Tag, TypeAdapter, model_validator


class DraftOptimizerConfig(BaseModel, extra="forbid"):
    """Optional optimizer schedule for draft-model parameters."""

    lr: Annotated[float, Field(gt=0)]
    min_lr: Annotated[float, Field(ge=0)] | None = None
    weight_decay: Annotated[float, Field(ge=0)] | None = None

    @model_validator(mode="after")
    def validate_lr_range(self) -> Self:
        """Require the draft minimum learning rate to fit its schedule."""
        if self.min_lr is not None and self.min_lr > self.lr:
            raise ValueError("draft optimizer min_lr must not exceed lr")
        return self


class _DraftConfigBase(BaseModel, extra="allow"):
    """Shared near-miss-key guard for every concrete draft config."""

    @model_validator(mode="after")
    def _reject_near_miss_extra_keys(self) -> "_DraftConfigBase":
        # extra="allow" preserves genuinely novel legacy keys, but a typo of a
        # declared field (e.g. "enalbed") would otherwise silently no-op the
        # real field's default. Reject extras that look like misspellings.
        declared = set(type(self).model_fields)
        for key in self.model_extra or {}:
            close = difflib.get_close_matches(key, declared, n=1, cutoff=0.8)
            if close:
                raise ValueError(
                    f"unknown draft config key {key!r}; did you mean {close[0]!r}?"
                )
        return self


class Eagle3DraftConfig(_DraftConfigBase):
    """Configuration for EAGLE-3 draft-model co-training with the policy.

    Runs on the Megatron backend (single-step distillation) or the Automodel
    backend (TTT training). num_layers/aux_layer_indices apply to the
    Megatron path only; architecture fields for the Automodel path (hidden
    sizes, draft vocab, d2t/t2d maps) are read from the draft checkpoint's
    config.json instead.
    """

    supports_context_parallel: ClassVar[bool] = False
    # Packed EAGLE-3 still requires pipeline_model_parallel_size=1 (see the
    # guard in lm_policy.py).
    supports_sequence_packing: ClassVar[bool] = True
    supports_target_sequence_parallel: ClassVar[bool] = False

    speculator_type: Literal["eagle3"] = "eagle3"
    enabled: bool = False
    model_name: str | None = None
    loss_weight: float = 0.1
    num_layers: int | None = None
    aux_layer_indices: list[int] | None = None
    # Tokens per FP32 vocab tile in the streaming soft-CE (Megatron path).
    # Mirrors DEFAULT_DRAFT_TOKEN_CHUNK_SIZE in nemo_rl/algorithms/loss/draft.py,
    # which is not imported here: the loss package pulls in nemo_rl.models.* and
    # would make this module circular.
    token_chunk_size: Annotated[int, Field(gt=0)] = 4096
    # Megatron-path-only override of the draft optimizer's lr/min_lr/
    # weight_decay schedule (NVIDIA-NeMo/RL#3707); unused by the Automodel
    # TTT path below, which always uses learning_rate.
    optimizer: DraftOptimizerConfig | None = None
    # Learning rate for the draft's optimizer param group; the draft needs a
    # much higher rate than the policy's RL lr to track the policy's
    # distribution drift.
    learning_rate: float = 1.0e-4
    # Test-time-training unroll depth: the draft re-consumes its own context
    # for this many steps per training forward (speculators default).
    ttt_steps: int = 3
    # Per-unroll-step loss decay factor (1.0 keeps all steps equally
    # weighted, matching the speculators default).
    ttt_step_loss_decay: float = 1.0
    # Train the draft's embed_tokens/lm_head (streamed on every refit)
    # instead of keeping the checkpoint copies frozen.
    train_embed_and_head: bool = True


class _BlockDraftConfig(_DraftConfigBase):
    """Shared fields for the Automodel-only block drafters (DSpark/DFlash).

    Architecture fields (block_size, target_layer_ids, mask_token_id, markov
    and confidence head layout) are read from the draft checkpoint's
    config.json and are intentionally not configurable here.
    """

    enabled: bool = False
    model_name: str | None = None
    loss_weight: float = 0.1
    # Anchor blocks sampled per sequence each training forward (capped by the
    # number of valid response positions). Draft-side transient memory scales
    # with num_anchors * block_size * vocab (draft logits + markov bias +
    # teacher gather); co-training shares the GPU with full policy training,
    # so headroom is tight -- shipped recipes use 32-64, not this default's
    # pretraining-scale value.
    num_anchors: int = 64
    # Learning rate for the draft's optimizer param group. The draft needs a
    # much higher rate than the policy's RL lr to track the policy's
    # distribution drift (dspark pretraining used 6e-4; the policy trains at
    # ~1e-6).
    learning_rate: float = 1.0e-4
    # Cross-entropy weight against the rollout tokens.
    ce_loss_alpha: float = 0.1
    # Total-variation distillation weight against the policy's raw logits.
    l1_loss_alpha: float = 0.9
    # Confidence-head BCE weight; requires the checkpoint's confidence head
    # (dflash checkpoints have none, so dflash overrides this to 0.0).
    confidence_loss_alpha: float = 1.0
    # Exponential per-block-position decay exp(-k / gamma) on the loss mask.
    loss_decay_gamma: float = 4.0
    # Train the draft's embed_tokens/lm_head (streamed on every refit) instead
    # of keeping the checkpoint copies frozen.
    train_embed_and_head: bool = True


class DSparkDraftConfig(_BlockDraftConfig):
    """Training options for DSpark draft co-training (Automodel backend only)."""

    speculator_type: Literal["dspark"] = "dspark"


class DFlashDraftConfig(_BlockDraftConfig):
    """Training options for DFlash draft co-training (Automodel backend only).

    DFlash is the markov-free/confidence-free subset of DSpark, so its
    checkpoints carry no confidence head.
    """

    speculator_type: Literal["dflash"] = "dflash"
    confidence_loss_alpha: float = 0.0


class _MegatronBlockDraftConfigBase(BaseModel, extra="forbid"):
    """Strict base for the Megatron block drafters."""

    @model_validator(mode="before")
    @classmethod
    def _drop_inherited_eagle3_defaults(cls, data: Any) -> Any:
        # Recipes deep-merge policy.draft from grpo_math_1B.yaml, and a nested
        # override cannot delete its Eagle3 knobs. Drop such a key only while it
        # still holds the Eagle3 default; any other value stays extra_forbidden.
        if not isinstance(data, Mapping):
            return data
        eagle3_fields = Eagle3DraftConfig.model_fields
        return {
            key: value
            for key, value in data.items()
            if key in cls.model_fields
            or key not in eagle3_fields
            or value != eagle3_fields[key].default
        }


class MegatronDFlashDraftConfig(_MegatronBlockDraftConfigBase):
    """Body-only DFlash co-training with a live target (Megatron backend only).

    Selected over ``DFlashDraftConfig`` when ``target_hidden_state_layer_ids``
    is set. The draft architecture comes from this config, not a checkpoint.
    """

    supports_context_parallel: ClassVar[bool] = True
    supports_sequence_packing: ClassVar[bool] = True
    supports_target_sequence_parallel: ClassVar[bool] = True

    speculator_type: Literal["dflash"] = "dflash"
    enabled: bool = False
    model_name: str | None = None
    model_revision: str | None = None
    aux_layer_indices: None = Field(default=None, exclude=True, repr=False)
    loss_weight: Annotated[float, Field(gt=0)] = 0.1
    gamma: Annotated[int, Field(gt=0)]
    anchors_per_sample: Annotated[int, Field(gt=0)]
    mask_token_id: Annotated[int, Field(ge=0)]
    target_hidden_state_layer_ids: Annotated[list[int], Field(min_length=1)]
    num_layers: Annotated[int, Field(gt=0)] = 5
    seed: int = 0
    vocab_tile_size: Annotated[int, Field(gt=0)] = 256
    position_decay: Annotated[float, Field(gt=0, le=1)] = 1.0
    max_cp_boundary_exclusion_fraction: Annotated[
        float,
        Field(ge=0, le=1),
    ] = 0.25
    optimizer: DraftOptimizerConfig | None = None
    update_probe_enabled: bool = False

    @model_validator(mode="after")
    def validate_target_taps(self) -> Self:
        """Reject ambiguous and out-of-range layer taps before model creation."""
        if any(layer_id < 0 for layer_id in self.target_hidden_state_layer_ids):
            raise ValueError("target hidden-state layer IDs must be non-negative")
        if len(set(self.target_hidden_state_layer_ids)) != len(
            self.target_hidden_state_layer_ids
        ):
            raise ValueError("target hidden-state layer IDs must be unique")
        return self


class MegatronDSparkDraftConfig(_MegatronBlockDraftConfigBase):
    """DSpark co-training with target-owned embeddings/head (Megatron backend only).

    Selected over ``DSparkDraftConfig`` when ``target_hidden_state_layer_ids``
    is set. The draft architecture comes from this config, not a checkpoint.
    """

    supports_context_parallel: ClassVar[bool] = True
    supports_sequence_packing: ClassVar[bool] = True
    supports_target_sequence_parallel: ClassVar[bool] = True

    speculator_type: Literal["dspark"] = "dspark"
    enabled: bool = False
    model_name: str | None = None
    model_revision: str | None = None
    loss_weight: Annotated[float, Field(gt=0)] = 0.1
    aux_layer_indices: None = Field(default=None, exclude=True, repr=False)
    block_size: Annotated[int, Field(gt=1)]
    anchors_per_sample: Annotated[int, Field(gt=0)]
    mask_token_id: Annotated[int, Field(ge=0)]
    target_hidden_state_layer_ids: Annotated[list[int], Field(min_length=1)]
    num_layers: Annotated[int, Field(gt=0)] = 5
    draft_vocab_size: Annotated[int, Field(gt=0)] | None = None
    markov_rank: Annotated[int, Field(gt=0)] = 256
    markov_head_type: Literal["vanilla"] = "vanilla"
    confidence_enabled: bool = True
    confidence_with_markov: bool = True
    ce_loss_weight: Annotated[float, Field(ge=0)] = 0.1
    tv_loss_weight: Annotated[float, Field(ge=0)] = 0.9
    confidence_loss_weight: Annotated[float, Field(ge=0)] = 1.0
    loss_decay_gamma: Annotated[float, Field(gt=0)] = 4.0
    seed: int = 0
    vocab_tile_size: Annotated[int, Field(gt=0)] = 256
    max_cp_boundary_exclusion_fraction: Annotated[
        float,
        Field(ge=0, le=1),
    ] = 0.25
    optimizer: DraftOptimizerConfig | None = None
    update_probe_enabled: bool = False

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        """Reject invalid taps and confidence dependencies before model creation."""
        if any(layer_id < 0 for layer_id in self.target_hidden_state_layer_ids):
            raise ValueError("target hidden-state layer IDs must be non-negative")
        if len(set(self.target_hidden_state_layer_ids)) != len(
            self.target_hidden_state_layer_ids
        ):
            raise ValueError("target hidden-state layer IDs must be unique")
        if self.confidence_with_markov and not self.confidence_enabled:
            raise ValueError("confidence_with_markov requires confidence_enabled")
        if not any(
            weight > 0
            for weight in (
                self.ce_loss_weight,
                self.tv_loss_weight,
                self.confidence_loss_weight,
            )
        ):
            raise ValueError("at least one DSpark loss weight must be positive")
        return self

    def resolve_draft_vocab_size(self, *, target_vocab_size: int) -> int:
        """Require the online draft to share the live target vocabulary."""
        if target_vocab_size <= 0:
            raise ValueError("live target vocabulary size must be positive")
        if (
            self.draft_vocab_size is not None
            and self.draft_vocab_size != target_vocab_size
        ):
            raise ValueError(
                "DSpark draft vocabulary must match the live target vocabulary"
            )
        return target_vocab_size


MEGATRON_BLOCK_DRAFT_ALGOS = ("dflash", "dspark")


def _speculator_type(v: Any) -> str:
    """Discriminator callback: default a missing tag to eagle3.

    A plain ``speculator_type: Literal[...]`` discriminator requires every
    ``policy.draft`` block to carry the tag, even a disabled one that never
    cared which family it would have been (``MasterConfig`` validation then
    fails with ``union_tag_not_found``). Mirrors the pre-union implicit-
    eagle3-only default.
    """
    if isinstance(v, (MegatronDFlashDraftConfig, MegatronDSparkDraftConfig)):
        return f"megatron_{v.speculator_type}"
    if isinstance(v, Mapping):
        tag = v.get("speculator_type", "eagle3")
        has_taps = "target_hidden_state_layer_ids" in v
    else:
        tag = getattr(v, "speculator_type", "eagle3")
        has_taps = False
    # The Megatron block drafters build their architecture from config and
    # always declare target taps; the Automodel ones read it from a checkpoint.
    if tag in MEGATRON_BLOCK_DRAFT_ALGOS and has_taps:
        return f"megatron_{tag}"
    return tag


DraftConfig = Annotated[
    Union[
        Annotated[Eagle3DraftConfig, Tag("eagle3")],
        Annotated[DSparkDraftConfig, Tag("dspark")],
        Annotated[DFlashDraftConfig, Tag("dflash")],
        Annotated[MegatronDFlashDraftConfig, Tag("megatron_dflash")],
        Annotated[MegatronDSparkDraftConfig, Tag("megatron_dspark")],
    ],
    Discriminator(_speculator_type),
]

AnyDraftConfig = (
    Eagle3DraftConfig
    | DSparkDraftConfig
    | DFlashDraftConfig
    | MegatronDFlashDraftConfig
    | MegatronDSparkDraftConfig
)

_DRAFT_CONFIG_ADAPTER: TypeAdapter[Any] = TypeAdapter(DraftConfig)


def coerce_draft_config(
    config: "AnyDraftConfig | Mapping[str, Any] | None",
) -> AnyDraftConfig | None:
    """Accept either a validated model or a raw mapping at API boundaries.

    ``MasterConfig`` validation normally produces the model, but ``PolicyConfig``
    is a TypedDict, so callers that assemble one by hand still pass a plain dict.
    """
    if config is None or isinstance(
        config,
        (
            Eagle3DraftConfig,
            DSparkDraftConfig,
            DFlashDraftConfig,
            MegatronDFlashDraftConfig,
            MegatronDSparkDraftConfig,
        ),
    ):
        return config
    return _DRAFT_CONFIG_ADAPTER.validate_python(config)


def draft_refit_enabled(
    config: "AnyDraftConfig | Mapping[str, Any] | None",
) -> bool:
    """Return whether generation must accept refitted draft weights."""
    coerced = coerce_draft_config(config)
    return coerced is not None and coerced.enabled
