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

from collections.abc import Mapping
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter


class Eagle3DraftConfig(BaseModel, extra="allow"):
    """Configuration for EAGLE-3 draft-model co-training with the policy.

    Runs on the Megatron backend (single-step distillation) or the DTensor v2
    backend (TTT training). num_layers/aux_layer_indices apply to the
    Megatron path only; architecture fields for the DTensor v2 path (hidden
    sizes, draft vocab, d2t/t2d maps) are read from the draft checkpoint's
    config.json instead.
    """

    speculator_type: Literal["eagle3"] = "eagle3"
    enabled: bool = False
    model_name: str | None = None
    loss_weight: float = 0.1
    num_layers: int | None = None
    aux_layer_indices: list[int] | None = None
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


class _BlockDraftConfig(BaseModel, extra="allow"):
    """Shared fields for the DTensor-v2-only block drafters (DSpark/DFlash).

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
    """Training options for DSpark draft co-training (DTensor-v2 backend only)."""

    speculator_type: Literal["dspark"] = "dspark"


class DFlashDraftConfig(_BlockDraftConfig):
    """Training options for DFlash draft co-training (DTensor-v2 backend only).

    DFlash is the markov-free/confidence-free subset of DSpark, so its
    checkpoints carry no confidence head.
    """

    speculator_type: Literal["dflash"] = "dflash"
    confidence_loss_alpha: float = 0.0


DraftConfig = Annotated[
    Union[Eagle3DraftConfig, DSparkDraftConfig, DFlashDraftConfig],
    Field(discriminator="speculator_type"),
]

_DRAFT_CONFIG_ADAPTER: TypeAdapter[Any] = TypeAdapter(DraftConfig)


def coerce_draft_config(
    config: "Eagle3DraftConfig | DSparkDraftConfig | DFlashDraftConfig | Mapping[str, Any] | None",
) -> Eagle3DraftConfig | DSparkDraftConfig | DFlashDraftConfig | None:
    """Accept either a validated model or a raw mapping at API boundaries.

    ``MasterConfig`` validation normally produces the model, but ``PolicyConfig``
    is a TypedDict, so callers that assemble one by hand still pass a plain dict.
    """
    if config is None or isinstance(
        config, (Eagle3DraftConfig, DSparkDraftConfig, DFlashDraftConfig)
    ):
        return config
    return _DRAFT_CONFIG_ADAPTER.validate_python(config)


def draft_refit_enabled(
    config: "Eagle3DraftConfig | DSparkDraftConfig | DFlashDraftConfig | Mapping[str, Any] | None",
) -> bool:
    """Return whether generation must accept refitted draft weights."""
    coerced = coerce_draft_config(config)
    return coerced is not None and coerced.enabled
