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
"""Draft checkpoint save/load: optimizer-layout record and the DCP sibling entry."""

import json
import os
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch import nn

DRAFT_CHECKPOINT_DIRNAME = "draft"
DSPARK_META_FILENAME = "dspark_meta.json"

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


def _draft_meta_from_config(
    draft_model: nn.Module,
    draft_config: Any,
    optimizer: Optional[torch.optim.Optimizer],
) -> dict[str, Any]:
    """``draft_meta_record`` fed directly from a validated ``policy.draft`` config.

    Both ``Eagle3DraftConfig`` and the block-drafter configs carry
    ``model_name``/``speculator_type``/``train_embed_and_head``; only
    ``Eagle3DraftConfig`` carries ``ttt_steps``.
    """
    algo = draft_config.speculator_type
    return draft_meta_record(
        draft_model,
        draft_config.model_name,
        optimizer,
        algo=algo,
        train_embed_and_head=bool(draft_config.train_embed_and_head),
        ttt_steps=int(draft_config.ttt_steps) if algo == "eagle3" else None,
    )


def load_checkpoint_with_draft(
    checkpoint_manager: Any,
    model: nn.Module,
    draft_model: nn.Module,
    composite_model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Any,
    weights_path: str,
    optimizer_path: Optional[str],
    draft_config: Any,
) -> None:
    """Load a checkpoint with the composite-optimizer pairing rule.

    Policy weights load exactly as without a draft; draft weights load from
    the sibling draft entry with validated metadata; optimizer state pairs
    with the composite module whose param groups span policy + draft. This is
    the single home of that pairing invariant for both resume-at-setup and
    mid-run checkpoint loads. Works for eagle3 as well as dspark/dflash --
    all three validate the same ``policy.draft`` config shape.
    """
    checkpoint_manager.load_checkpoint(model=model, weights_path=weights_path)
    # Enforce the optimizer-layout record only when optimizer state is actually
    # restored: weights-only loads (no optimizer_path) must accept checkpoints
    # saved without a layout record or with a different optimizer grouping.
    restoring_optimizer = bool(optimizer_path) and optimizer is not None
    load_draft_checkpoint(
        draft_model,
        weights_path,
        expected_meta=_draft_meta_from_config(
            draft_model,
            draft_config,
            optimizer if restoring_optimizer else None,
        ),
    )
    if optimizer_path and optimizer is not None:
        checkpoint_manager.checkpointer.load_optimizer(
            optimizer=optimizer,
            model=composite_model,
            weights_path=optimizer_path,
            scheduler=scheduler,
        )


def save_checkpoint_with_draft(
    checkpoint_manager: Any,
    model: nn.Module,
    draft_model: nn.Module,
    composite_model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Any,
    weights_path: str,
    optimizer_path: Optional[str],
    tokenizer: Any,
    tokenizer_path: Optional[str],
    is_final_checkpoint: bool,
    peft_config: Any,
    draft_config: Any,
) -> None:
    """Save a checkpoint with the composite-optimizer pairing rule.

    Mirrors ``load_checkpoint_with_draft``: policy weights save without an
    optimizer, the optimizer (spanning policy + draft param groups) pairs
    with ``composite_model``, then the draft's own DCP entry and metadata
    record save last.
    """
    checkpoint_manager.save_checkpoint(
        model=model,
        weights_path=weights_path,
        optimizer=None,
        optimizer_path=None,
        scheduler=None,
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
        is_final_checkpoint=is_final_checkpoint,
        peft_config=peft_config,
    )
    if optimizer_path and optimizer is not None:
        checkpoint_manager.checkpointer.save_optimizer(
            optimizer=optimizer,
            model=composite_model,
            weights_path=optimizer_path,
            scheduler=scheduler,
        )
    save_draft_checkpoint(
        draft_model,
        weights_path,
        meta=_draft_meta_from_config(draft_model, draft_config, optimizer),
    )
