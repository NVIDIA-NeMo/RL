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
"""Offline critic pretraining on stored rollouts (stage B, decoupled PPO).

Trains ONLY the PPO value model on prompt-group shards written by
:mod:`nemo_rl.algorithms.rollout_collection` — no policy worker, no generation
engines, no gym. Each step mirrors the async PPO loop's critic path exactly
(reward/mask processing -> fresh value forward -> GAE returns -> value train),
so with the production SWE config (``gae_lambda_value=1``, ``gae_gamma=1``,
KL=0) one offline epoch is the same optimization as the online critic warmup.

Turn-level mode (``ppo.adv_estimator.name=turn_gae``) changes what a "position"
means here: the critic is supervised at ONE anchor per assistant turn (the turn's
first token, where the right-shifted value head reads ``V(s_k)``) with turn-level
GAE returns, instead of at all ~45k response tokens. Stage C must run with the
same setting — a token-level critic and a turn-level one are not interchangeable
warm starts. See research/ppo/turn_level_critic_plan.md.

Extras the online warmup cannot provide:
  * a held-out shard split (``dataset_idx % heldout_mod == 0``) with
    explained-variance / calibration / terminal-AUC eval on unseen rollouts;
  * checkpoints in the standard layout with ONLY a ``value/`` dir — the PPO
    resume path explicitly tolerates a missing ``policy/``, which is how
    stage C warm-starts from these checkpoints (scripts/swe/ppo/
    prep_warm_start.sh).

The train-file ORDER is frozen in the first checkpoint
(``critic_pretrain_files.json``) so resume replays the identical stream even if
new shards appear later. ``critic_pretrain.num_epochs`` (default 1) sets how
many passes over the train split the stream contains; epoch ``e`` is shuffled
with ``Random(seed + e)``, so epoch 0 is byte-identical to the original
one-epoch order and raising ``num_epochs`` on a finished run EXTENDS its frozen
stream (resume continues) rather than rewriting it.
"""

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from nemo_rl.algorithms.rollout_collection import load_group, parse_group_index
from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

FILE_LIST_NAME = "critic_pretrain_files.json"


# ===============================================================================
# Pure helpers (unit-tested, no heavy deps)
# ===============================================================================
def list_group_files(shards_dir: str | Path) -> list[Path]:
    """All group files under ``shards_dir`` (searches shard_*/ and the dir itself)."""
    shards_dir = Path(shards_dir)
    files = sorted(shards_dir.glob("shard_*/group_*.pt")) + sorted(
        shards_dir.glob("group_*.pt")
    )
    return [f for f in files if parse_group_index(f.name) is not None]


def split_heldout(files: list[Path], heldout_mod: int) -> tuple[list[Path], list[Path]]:
    """Deterministic train/held-out split by dataset index.

    ``dataset_idx % heldout_mod == 0`` goes to held-out; ``heldout_mod <= 0``
    disables the split (everything trains).
    """
    if heldout_mod <= 0:
        return list(files), []
    train, heldout = [], []
    for f in files:
        idx = parse_group_index(f.name)
        (heldout if idx % heldout_mod == 0 else train).append(f)
    return train, heldout


def build_epoch_stream(
    base_files: list[Path], num_epochs: int, seed: int
) -> list[Path]:
    """Concatenate ``num_epochs`` independent shuffles of ``base_files``.

    Epoch ``e`` is shuffled with ``Random(seed + e)``, which makes two
    properties hold and both matter:

      * epoch 0 reproduces the original single-epoch order EXACTLY (that order
        was ``Random(seed).shuffle(train_files)``), so raising ``num_epochs`` on
        an existing run extends the stream instead of rewriting it;
      * any prefix of the stream is a pure function of (base set, seed,
        epoch index), so resume replays the consumed prefix identically.

    Each epoch is a fresh permutation rather than a repeat of the same order, so
    the model does not see the same batch composition twice.
    """
    stream: list[Path] = []
    for e in range(num_epochs):
        epoch = list(base_files)
        random.Random(seed + e).shuffle(epoch)
        stream += epoch
    return stream


def terminal_value_reward_auc(
    values: torch.Tensor,
    rewards: torch.Tensor,
    token_mask: torch.Tensor,
    positive_threshold: float = 0.5,
) -> float:
    """AUC of the LAST response token's value as a predictor of success.

    Rank-based (Mann-Whitney) AUC with tie correction; returns nan when the
    batch has a single outcome class. This is the "end-verification" critic
    quality signal from the privileged-critic analyses.
    """
    mask = token_mask.bool()
    has_response = mask.any(dim=1)
    if int(has_response.sum()) < 2:
        return float("nan")
    last_idx = mask.shape[1] - 1 - mask.fliplr().float().argmax(dim=1)
    v = values[has_response, last_idx[has_response]].float()
    y = (rewards[has_response].float() >= positive_threshold).float()
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(v)
    ranks = torch.empty_like(v)
    ranks[order] = torch.arange(1, v.numel() + 1, dtype=v.dtype)
    # midranks for ties
    for val in torch.unique(v):
        tie = v == val
        if int(tie.sum()) > 1:
            ranks[tie] = ranks[tie].mean()
    auc = (ranks[y.bool()].sum().item() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def _rank_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Mann-Whitney AUC with tie correction; nan on a single-class input."""
    n_pos, n_neg = int(labels.sum()), int((1 - labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty_like(scores)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=scores.dtype)
    for val in torch.unique(scores):
        tie = scores == val
        if int(tie.sum()) > 1:
            ranks[tie] = ranks[tie].mean()
    return float(
        (ranks[labels.bool()].sum().item() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    )


def within_group_auc(
    values: torch.Tensor,
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    token_mask: torch.Tensor,
    n_buckets: int = 4,
    positive_threshold: float = 0.5,
) -> dict[str, float]:
    """Does the critic rank a group's WINNING siblings above its losing ones?

    ``terminal_value_reward_auc`` pools every trajectory in the batch, so it is
    dominated by between-task variation — a critic that only knows "this issue is
    hopeless" scores well on it while carrying no within-task information at all.
    That is exactly the failure mode the residual target is meant to remove, so
    the go/no-go diagnostic has to hold the task fixed.

    Restricted to MIXED-outcome groups (homogeneous ones are undefined: no
    positive or no negative sibling). On the pi0 SWE pool only 43.5% of groups
    qualify, and that fraction — not the dataset size — bounds what any critic
    trained on terminal reward can learn about the within-task component.

    Scores are per-trajectory means of ``values`` over each progress bucket, so
    this is calibration-free: it survives the scale/offset errors that depress
    explained variance.

    Returns per-bucket mean AUC plus ``n_mixed_groups``. Buckets run earliest to
    latest by relative position within each trajectory's own response.
    """
    mask = token_mask.bool()
    rel = (torch.cumsum(mask.long(), dim=1) - 1).float() / mask.sum(
        dim=1, keepdim=True
    ).clamp(min=1).float()
    labels = (rewards.float() >= positive_threshold).float()

    out: dict[str, float] = {}
    n_mixed = 0
    for b in range(n_buckets):
        lo, hi = b / n_buckets, (b + 1) / n_buckets
        upper = (rel < hi) if b < n_buckets - 1 else (rel <= 1.0)
        bmask = mask & (rel >= lo) & upper
        counts = bmask.sum(dim=1)
        # Per-trajectory mean value inside this progress bucket.
        scores = (values * bmask).sum(dim=1) / counts.clamp(min=1)
        aucs = []
        for gid in torch.unique(group_ids):
            sel = (group_ids == gid) & (counts > 0)
            if int(sel.sum()) < 2:
                continue
            y = labels[sel]
            if int(y.sum()) == 0 or int((1 - y).sum()) == 0:
                continue  # homogeneous group: within-group AUC undefined
            auc = _rank_auc(scores[sel].float(), y)
            if auc == auc:  # not nan
                aucs.append(auc)
        if b == 0:
            n_mixed = len(aucs)
        out[f"critic/within_group_auc_q{b + 1}"] = (
            sum(aucs) / len(aucs) if aucs else float("nan")
        )
    out["critic/n_mixed_groups"] = float(n_mixed)
    return out


def verify_shard_meta(
    shards_dir: str | Path, master_config: Any, tokenizer: Any
) -> None:
    """Assert stored-shard provenance matches this run's model/tokenizer.

    Shards are token-id level: a different base model, tokenizer/chat template,
    or max sequence length silently invalidates them. Checks every shard
    meta.json written by stage A; missing meta files only warn (older shards).
    """
    from nemo_rl.algorithms.rollout_collection import _sha256

    metas = sorted(Path(shards_dir).glob("shard_*/meta.json"))
    if not metas:
        print(
            f"⚠️ No shard meta.json found under {shards_dir}; skipping provenance check."
        )
        return
    expected = {
        "model_name": master_config.policy["model_name"],
        "chat_template_sha256": _sha256(
            getattr(tokenizer, "chat_template", None) or ""
        ),
        "max_total_sequence_length": master_config.policy["max_total_sequence_length"],
    }
    for meta_path in metas:
        with open(meta_path) as f:
            meta = json.load(f)
        for key, want in expected.items():
            got = meta.get(key)
            assert got == want, (
                f"Shard provenance mismatch in {meta_path}: {key}={got!r} but this "
                f"run expects {want!r}. Shards are token-id level and are only "
                "valid for the exact model/tokenizer/max-length they were "
                "generated with."
            )
    print(f"  ✓ Shard provenance verified ({len(metas)} shard meta files)")


def resolve_critic_pretrain_config(
    raw: Optional[dict[str, Any]], ppo_config: dict[str, Any]
) -> dict[str, Any]:
    """Fill defaults for the ``critic_pretrain:`` config block."""
    cfg = dict(raw or {})
    assert cfg.get("shards_dir"), (
        "critic_pretrain.shards_dir is required "
        "(pass ++critic_pretrain.shards_dir=<stage A out_dir>)"
    )
    cfg.setdefault("groups_per_step", ppo_config["num_prompts_per_step"])
    cfg.setdefault("heldout_mod", 16)
    cfg.setdefault("eval_period", 10)
    cfg.setdefault("heldout_max_groups", cfg["groups_per_step"])
    cfg.setdefault("max_steps", None)
    # Passes over the train split. 1 = the original one-epoch behaviour; epoch e
    # is a fresh permutation seeded with (seed + e), so raising this on a
    # finished run EXTENDS its frozen stream rather than rewriting it.
    cfg.setdefault("num_epochs", 1)
    cfg.setdefault("seed", ppo_config.get("seed", 42))
    # Eval/dump mode: no training — load a specific checkpoint, score the
    # held-out groups, and dump per-token values aligned to message spans for
    # offline value-vs-behavior analysis.
    cfg.setdefault("eval_only", False)
    cfg.setdefault("eval_checkpoint_path", None)
    cfg.setdefault("dump_dir", None)  # default: <checkpoint_dir>/value_dumps
    cfg.setdefault("dump_text_groups", 8)  # decode message text for first N groups
    # per-token strings (for the token-level HTML heatmap) are ~35k/sample, so
    # store them only for a bounded, contrastful subset of samples per text group
    cfg.setdefault("dump_token_samples", 4)
    for key in (
        "groups_per_step",
        "heldout_mod",
        "eval_period",
        "heldout_max_groups",
        "num_epochs",
    ):
        cfg[key] = int(cfg[key])
    assert cfg["num_epochs"] >= 1, (
        f"critic_pretrain.num_epochs must be >= 1, got {cfg['num_epochs']}"
    )
    return cfg


def message_spans(message_log: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-message (role, start, len) spans within a sample's flattened tokens.

    Mirrors ``batched_message_log_to_flat_message``'s concatenation order, so
    span positions index directly into the flat per-token value/mask tensors.
    """
    spans = []
    pos = 0
    for m in message_log:
        n = len(m["token_ids"])
        spans.append({"role": m["role"], "start": pos, "len": n})
        pos += n
    return spans


# ===============================================================================
# Batch construction (mirrors the async PPO loop's steps 2-3 minus logprobs)
# ===============================================================================
def build_value_train_data(
    groups: list[dict[str, Any]],
    tokenizer: Any,
    master_config: Any,
) -> tuple[BatchedDataDict, BatchedDataDict]:
    """Assemble (train_data, repeated_batch) from loaded group payloads.

    Follows async_ppo_train's reward-processing + inline loss-mask block
    verbatim (overlong filtering, env-flagged sample masking, unmask ALL
    assistant messages) so offline critic batches are bit-identical in shape
    and masking to what the online warmup trains on. Policy/reference logprobs
    are deliberately absent: the critic path never consumes them (KL-in-reward
    handles logprobs=None) and computing them is the warmup's main train-node
    waste.
    """
    per_prompt_batches = [g["batch"] for g in groups]
    repeated_batch = BatchedDataDict.from_batches(per_prompt_batches)

    use_overlong_filtering = master_config.ppo["overlong_filtering"]
    if use_overlong_filtering:
        loss_multiplier = repeated_batch["loss_multiplier"].clone()
        truncated = repeated_batch["truncated"]
        if isinstance(truncated, list):
            truncated = torch.tensor(truncated, dtype=torch.bool)
        loss_multiplier[truncated] = 0
        repeated_batch["loss_multiplier"] = loss_multiplier

    if "mask_sample" in repeated_batch:
        loss_multiplier = repeated_batch["loss_multiplier"].clone()
        mask_sample = repeated_batch["mask_sample"]
        if isinstance(mask_sample, list):
            mask_sample = torch.tensor(mask_sample, dtype=torch.bool)
        loss_multiplier[mask_sample.bool()] = 0
        repeated_batch["loss_multiplier"] = loss_multiplier

    # PPO's inline loss-mask setup: unmask all assistant messages.
    for message_log in repeated_batch["message_log"]:
        for message in message_log:
            if message["role"] == "assistant":
                message["token_loss_mask"] = torch.ones_like(message["token_ids"])
            else:
                message["token_loss_mask"] = torch.zeros_like(message["token_ids"])
            if "generation_logprobs" not in message:
                message["generation_logprobs"] = torch.zeros_like(
                    message["token_ids"], dtype=torch.float32
                )

    flat_messages, input_lengths = batched_message_log_to_flat_message(
        repeated_batch["message_log"],
        pad_value_dict={"token_ids": tokenizer.pad_token_id},
        make_sequence_length_divisible_by=master_config.policy[
            "make_sequence_length_divisible_by"
        ],
    )

    train_data = BatchedDataDict(
        {
            "input_ids": flat_messages["token_ids"],
            "input_lengths": input_lengths,
            "rewards": repeated_batch["total_reward"],
            "token_mask": flat_messages["token_loss_mask"],
            "sample_mask": repeated_batch["loss_multiplier"],
        }
    )
    train_data.to("cpu")
    return train_data, repeated_batch


# ===============================================================================
# Value forward + returns (shared by train and held-out eval)
# ===============================================================================
def _forward_values_and_returns(
    value_model: Any,
    adv_estimator: Any,
    train_data: BatchedDataDict,
    repeated_batch: BatchedDataDict,
    tokenizer: Any,
    master_config: Any,

) -> tuple[Optional[BatchedDataDict], Optional[Any]]:
    """Populate train_data['values'/'returns'] in place.

    Returns ``(critic_batch, turn_spans)``: the batch the critic should actually
    train on when that differs from ``train_data`` (the privileged critic's
    answer-augmented batch, or the turn-level anchor batch) else None, and the
    turn structure (None on the token-level path) so callers can score metrics
    at the positions the critic is actually supervised at.

    Mirrors async_ppo_train steps 3 (value inference, incl. the privileged
    answer-conditioned remap) and 6 (GAE returns; logprobs=None is valid for
    the critic path since KL-in-reward is the only consumer of logprobs).
    """
    from nemo_rl.algorithms.grpo import extract_initial_prompt_messages
    from nemo_rl.algorithms.ppo import build_turn_spans_for_batch
    from nemo_rl.algorithms.turn_level import build_turn_value_batch
    from nemo_rl.algorithms.privileged_critic import (
        build_privileged_value_inputs,
        remap_by_response_mask,
    )


    privileged_critic_cfg = master_config.value.get("privileged_critic")
    if privileged_critic_cfg is not None and not privileged_critic_cfg.get("enabled"):
        privileged_critic_cfg = None

    critic_batch = None

    # Turn structure (None on the token-level path). Stage B must build this the
    # same way stage C does, or the pretrained critic is supervised at positions
    # PPO never reads.
    turn_spans = build_turn_spans_for_batch(master_config, repeated_batch, train_data)

    value_model.prepare_for_inference()
    if privileged_critic_cfg is not None:
        critic_batch = build_privileged_value_inputs(
            repeated_batch,
            tokenizer,
            privileged_critic_cfg,
            make_seq_len_divisible_by=master_config.policy[
                "make_sequence_length_divisible_by"
            ],
        )

        vals_aug = value_model.get_values(critic_batch)["values"].squeeze(-1)
        critic_batch["values"] = vals_aug
        train_data["values"] = remap_by_response_mask(
            vals_aug,
            critic_batch["token_mask"],
            train_data["token_mask"],
        )
    else:
        train_data["values"] = value_model.get_values(train_data)["values"].squeeze(-1)
    value_model.finish_inference()

    initial_prompt_message_logs = extract_initial_prompt_messages(
        repeated_batch["message_log"],
        repeated_batch["length"],
    )
    prompt_batched_flat, _ = batched_message_log_to_flat_message(
        initial_prompt_message_logs,
        pad_value_dict={"token_ids": tokenizer.pad_token_id},
    )
    adv_kwargs = dict(
        prompt_ids=prompt_batched_flat["token_ids"],
        rewards=train_data["rewards"],
        mask=train_data["token_mask"],
        values=train_data["values"],
        reference_logprobs=None,
        logprobs=None,
        sample_mask=train_data["sample_mask"],
    )
    if turn_spans is not None:
        adv_kwargs["turn_spans"] = turn_spans
    advantages, returns = adv_estimator.compute_advantage(**adv_kwargs)
    del advantages  # critic pretraining has no actor; only returns are used
    train_data["returns"] = returns
    if turn_spans is not None:
        # One supervised position per turn, equally weighted (swapping token_mask
        # for the anchor mask is what makes MseValueLossFn a per-turn mean).
        critic_batch = build_turn_value_batch(train_data, turn_spans)
    elif critic_batch is not None:
        critic_batch["returns"] = remap_by_response_mask(
            returns,
            train_data["token_mask"],
            critic_batch["token_mask"],
        )
        critic_batch["sample_mask"] = train_data["sample_mask"]
    return critic_batch, turn_spans


def _heldout_metrics(
    value_model: Any,
    adv_estimator: Any,
    heldout_files: list[Path],
    tokenizer: Any,
    master_config: Any,
) -> dict[str, float]:
    """Critic quality on held-out rollouts: EV, positional EV/ECE, terminal AUC.

    On the turn-level path every metric is scored at the positions the critic is
    actually supervised at (turn anchors) — scoring an anchor-layout return
    tensor over the full response mask would average each real target against
    ~270 structural zeros — and the per-turn metrics from the estimator are
    merged in.
    """
    from nemo_rl.algorithms.ppo import (
        _mixed_group_mask,
        _mixed_group_value_metrics,
        _positional_value_metrics,
    )

    groups = [load_group(p) for p in heldout_files]
    train_data, repeated_batch = build_value_train_data(
        groups, tokenizer, master_config
    )

    _, turn_spans = _forward_values_and_returns(
        value_model,
        adv_estimator,
        train_data,
        repeated_batch,
        tokenizer,
        master_config,

    )
    values, returns = train_data["values"], train_data["returns"]
    scored_mask = (
        turn_spans.anchor_mask if turn_spans is not None else train_data["token_mask"]
    )
    mask = scored_mask.bool()
    metrics: dict[str, float] = {}

    # Per-sample offsets into each return space (both zero without a residual
    # estimator, i.e. exactly today's numbers).
    zeros = torch.zeros(returns.shape[0], device=returns.device)
    raw_to_abs = getattr(adv_estimator, "last_returns_to_abs", None)
    raw_to_res = getattr(adv_estimator, "last_returns_to_res", None)
    to_abs = zeros if raw_to_abs is None else raw_to_abs.to(returns.device)
    to_res = zeros if raw_to_res is None else raw_to_res.to(returns.device)

    if int(mask.sum()) >= 2:
        v, r = values[mask].float(), returns[mask].float()
        # Both explained variances, on the same convention _compute_critic_metrics
        # uses in PPO: critic/explained_var is ALWAYS absolute-space and
        # critic/ev_res ALWAYS residual-space, whichever space `returns` is in.
        # The prediction error is shared (R - (B+C) == (R-B) - C); only the
        # denominator changes. Held-out ev_res is the go/no-go number.
        err_var = (r - v).var(unbiased=False)
        for key, offset in (("explained_var", to_abs), ("ev_res", to_res)):
            target = (returns + offset.unsqueeze(-1).to(returns.dtype))[mask].float()
            var_t = target.var(unbiased=False)
            metrics[f"critic/{key}"] = (
                (1.0 - err_var / var_t).item() if var_t > 1e-8 else 0.0
            )
        metrics["critic/mse"] = ((r - v) ** 2).mean().item()
    metrics.update(
        _positional_value_metrics(
            values,
            returns,
            scored_mask,
            returns_to_abs=raw_to_abs,
            returns_to_res=raw_to_res,
        )
    )
    # Residual EV restricted to mixed-outcome groups. critic/ev_res stays the
    # whole-batch go/no-go number; this says whether a near-zero ev_res means
    # "no within-task signal" or "signal, taxed by the ~58% homogeneous groups
    # where Y = 0 and any prediction is a pure penalty".
    metrics.update(
        _mixed_group_value_metrics(
            values,
            returns,
            scored_mask,
            _mixed_group_mask(adv_estimator),
            returns_to_res=raw_to_res,
        )
    )
    # Scored on `scored_mask`: in turn mode the last RESPONSE token carries an
    # untrained value, while the last anchor is the supervised V(s_K).
    #
    # Scored on ABSOLUTE values (V~ = C + B_LOO), like every other metric here.
    # This AUC pools all trajectories, so it is largely a between-task ranking;
    # in residual space the values are C with E[C | X] = 0, which strips exactly
    # that component out and would read as a large regression versus the
    # absolute arm when nothing regressed.
    abs_values = values + to_abs.unsqueeze(-1).to(values.dtype)
    metrics["critic/terminal_auc"] = terminal_value_reward_auc(
        abs_values, train_data["rewards"], scored_mask
    )
    # Sibling ranking with the task held fixed — the calibration-free go/no-go
    # complement to explained variance, and the only AUC that is not confounded
    # by between-task difficulty.
    #
    # Deliberately scored on RAW values, unlike terminal_auc above. B_LOO is
    # leave-one-out, so it is NOT constant within a group: adding it would fold
    # each sibling's own reward into that sibling's score and leak the label,
    # inflating this AUC. Raw values are already the right quantity in both arms
    # (C in residual mode, V in absolute mode), since the task-level component is
    # common to the group and cancels from a within-group ranking either way.
    group_ids = getattr(adv_estimator, "last_group_ids", None)
    if group_ids is not None:
        metrics.update(
            within_group_auc(
                values, train_data["rewards"], group_ids.cpu(), scored_mask
            )
        )
    metrics.update(getattr(adv_estimator, "last_metrics", {}) or {})

    metrics["reward"] = train_data["rewards"].float().mean().item()
    metrics["num_heldout_samples"] = float(train_data["input_ids"].shape[0])
    return metrics


def _dump_heldout_values(
    value_model: Any,
    adv_estimator: Any,
    heldout_files: list[Path],
    tokenizer: Any,
    master_config: Any,
    dump_dir: Path,
    dump_text_groups: int,
    dump_token_samples: int = 4,
) -> None:
    """Score held-out groups with the loaded critic and dump per-token values.

    Values/returns are packed over response tokens; per-message spans (with
    decoded text for the first ``dump_text_groups`` groups) let offline
    analysis align value movements to agent/tool behavior in the trajectory.

    In turn-level mode ``returns`` is an ANCHOR-layout tensor: it is the turn
    return at each turn's first token and structurally 0 at the other ~270
    tokens of the turn. Averaging it over all stored tokens is meaningless, so
    the payload carries ``credit_level`` and a per-token ``is_anchor`` flag
    (format_version 3) and consumers must filter on it. Values are per-token in
    both modes.
    """
    dump_dir.mkdir(parents=True, exist_ok=True)
    for gi, path in enumerate(heldout_files):
        g = load_group(path)
        train_data, repeated_batch = build_value_train_data(
            [g], tokenizer, master_config
        )
        _, turn_spans = _forward_values_and_returns(
            value_model,
            adv_estimator,
            train_data,
            repeated_batch,
            tokenizer,
            master_config,
        )
        mask = train_data["token_mask"].bool()
        coords = mask.nonzero(as_tuple=False)
        with_text = gi < dump_text_groups
        # per-token strings power the token-level HTML heatmap but cost a decode
        # per token (~35k/sample), so carry them only for a bounded, contrastful
        # subset: successes first, then fails, capped at dump_token_samples.
        render_samples = []
        if with_text and dump_token_samples > 0:
            rew = train_data["rewards"].float().tolist()
            succ = [i for i in range(len(rew)) if rew[i] > 0.5]
            fail = [i for i in range(len(rew)) if rew[i] <= 0.5]
            half = max(1, dump_token_samples // 2)
            render_samples = succ[:half] + fail[: dump_token_samples - len(succ[:half])]
            render_samples = sorted(render_samples[:dump_token_samples])
        render_set = set(render_samples)
        samples_msgs = []
        for si, ml in enumerate(repeated_batch["message_log"]):
            spans = message_spans(ml)
            if with_text:
                want_toks = si in render_set
                for m, s in zip(ml, spans):
                    s["text"] = tokenizer.decode(m["token_ids"])
                    if want_toks and m["role"] == "assistant":
                        s["toks"] = [tokenizer.decode([int(t)]) for t in m["token_ids"]]
            samples_msgs.append(spans)
        anchor_mask = turn_spans.anchor_mask if turn_spans is not None else None
        payload = {
            "format_version": 3,
            "dataset_idx": g["dataset_idx"],
            "source_file": str(path),
            # "token": returns are per-token. "turn": returns are the turn
            # return at anchors and structurally 0 elsewhere — filter on
            # is_anchor before averaging or computing EV.
            "credit_level": "turn" if anchor_mask is not None else "token",
            "rewards": train_data["rewards"].float().cpu(),
            "sample_mask": train_data["sample_mask"].float().cpu(),
            "token_sample_index": coords[:, 0].to(torch.int32),
            "token_position": coords[:, 1].to(torch.int32),
            "values": train_data["values"][mask].to(torch.float16).cpu(),
            "returns": train_data["returns"][mask].to(torch.float16).cpu(),
            "messages": samples_msgs,
            "has_text": with_text,
            "render_samples": render_samples,
        }
        if anchor_mask is not None:
            payload["is_anchor"] = anchor_mask[mask].bool().cpu()
        out = dump_dir / f"valuedump_{g['dataset_idx']:08d}.pt"
        torch.save(payload, out)
        if (gi + 1) % 10 == 0 or gi + 1 == len(heldout_files):
            print(f"  💾 dumped {gi + 1}/{len(heldout_files)} groups", flush=True)


# ===============================================================================
# Main entry point
# ===============================================================================
def critic_pretrain(master_config: Any, tokenizer: Any) -> None:
    """Set up the value model and run offline critic pretraining.

    Heavy setup (Ray cluster, Megatron value workers, checkpointing) lives here
    rather than in a separate setup() so the driver stays thin; the module-level
    helpers above stay importable without Ray/Megatron for unit tests.
    """
    from pathlib import Path as _Path

    from nemo_rl.algorithms.loss.loss_functions import MseValueLossFn
    from nemo_rl.algorithms.ppo import (
        _compute_critic_metrics,
        _create_advantage_estimator,
        _mixed_group_mask,
        _mixed_group_value_metrics,
        _positional_value_metrics,
        _prepare_value_train_batch,
        _resolve_resume_optimizer_path,
    )
    from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
    from nemo_rl.models.value.lm_value import Value
    from nemo_rl.utils.checkpoint import CheckpointManager
    from nemo_rl.utils.logger import Logger

    cp_config = resolve_critic_pretrain_config(
        getattr(master_config, "critic_pretrain", None), master_config.ppo
    )
    value_config = master_config.value
    cluster_config = master_config.cluster

    # Known, inherent divergence from the online warmup: the seq-level
    # train/inference logprob-error masking (ppo.seq_logprob_error_threshold)
    # needs policy-engine logprobs, which a value-only job cannot compute. The
    # online loop zeroes sample_mask for badly mismatched sequences; offline
    # those sequences stay in the critic loss.
    if master_config.ppo.get("seq_logprob_error_threshold") is not None:
        print(
            "⚠️ ppo.seq_logprob_error_threshold is set, but offline critic "
            "pretraining cannot apply seq-logprob-error masking (no policy "
            "worker). Sequences the online warmup would mask are trained on."
        )

    # Privileged critic scores [prompt + answer + response]: raise the value
    # model's sequence/packing budgets exactly as ppo.setup() does, so
    # answer-augmented near-max-length samples fit the packing bins.
    _privileged_critic = value_config.get("privileged_critic")
    if _privileged_critic is not None and _privileged_critic.get("enabled"):
        _needed = (
            master_config.policy["max_total_sequence_length"]
            + int(_privileged_critic.get("max_answer_tokens", 256) or 0)
            + 128  # grader-note template + chat re-render slack
        )
        if value_config["max_total_sequence_length"] < _needed:
            print(
                "  ↑ privileged critic: raising value.max_total_sequence_length "
                f"{value_config['max_total_sequence_length']} -> {_needed}",
                flush=True,
            )
            value_config["max_total_sequence_length"] = _needed
        for _bcfg_key in ("sequence_packing", "dynamic_batching"):
            _bcfg = value_config.get(_bcfg_key) or {}
            if not _bcfg.get("enabled"):
                continue
            for _tok_key in ("train_mb_tokens", "logprob_mb_tokens"):
                if _bcfg.get(_tok_key) is not None and _bcfg[_tok_key] < _needed:
                    print(
                        f"  ↑ privileged critic: raising value.{_bcfg_key}."
                        f"{_tok_key} {_bcfg[_tok_key]} -> {_needed}",
                        flush=True,
                    )
                    _bcfg[_tok_key] = _needed

    logger = Logger(master_config.logger)
    logger.log_hyperparams(master_config.model_dump())

    checkpointer = CheckpointManager(master_config.checkpointing)
    last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
    save_state = checkpointer.load_training_info(last_checkpoint_path) or {
        "total_steps": 0,
        "groups_consumed": 0,
        "consumed_samples": 0,
    }
    step = int(save_state["total_steps"])

    # ------------------------------------------------------------------
    # Shard discovery + frozen multi-epoch order (replayed exactly on resume).
    # ------------------------------------------------------------------
    all_files = list_group_files(cp_config["shards_dir"])
    assert all_files, f"No group files found under {cp_config['shards_dir']}"
    verify_shard_meta(cp_config["shards_dir"], master_config, tokenizer)
    num_epochs = cp_config["num_epochs"]
    frozen = None
    if last_checkpoint_path is not None:
        file_list_path = os.path.join(last_checkpoint_path, FILE_LIST_NAME)
        if os.path.exists(file_list_path):
            with open(file_list_path) as f:
                frozen = json.load(f)
    if frozen is not None:
        heldout_files = [_Path(p) for p in frozen["heldout"]]
        frozen_train = [_Path(p) for p in frozen["train"]]
        # The frozen stream is `num_epochs_then` shuffles of a base set; recover
        # the base in its pre-shuffle order (list_group_files sorts by path) and
        # regenerate the stream for the num_epochs asked for NOW.
        base_train = [_Path(p) for p in sorted({str(p) for p in frozen_train})]
        train_files = build_epoch_stream(base_train, num_epochs, cp_config["seed"])
        n_frozen = len(frozen_train)
        # Fail loud rather than train on a different sequence than the
        # checkpoint recorded: the regenerated stream MUST reproduce the frozen
        # one as a prefix, otherwise resume would silently replay other data.
        if len(train_files) < n_frozen or train_files[:n_frozen] != frozen_train:
            raise ValueError(
                f"Cannot reconcile critic_pretrain.num_epochs={num_epochs} with the "
                f"frozen stream in {last_checkpoint_path}: the regenerated order does "
                f"not reproduce its {n_frozen} entries as a prefix, so resuming would "
                "train on a different sequence than the checkpoint recorded. Check "
                "that critic_pretrain.seed / heldout_mod / shards_dir are unchanged "
                f"(seed={cp_config['seed']}, heldout_mod={cp_config['heldout_mod']}), "
                f"and that num_epochs is not below the {n_frozen // max(len(base_train), 1)} "
                "epoch(s) already frozen."
            )
        if len(train_files) > n_frozen:
            print(
                f"↻ Extending the frozen stream: {n_frozen} -> {len(train_files)} "
                f"groups ({num_epochs} epochs x {len(base_train)} train groups)."
            )
        newly_seen = len(all_files) - len(base_train) - len(heldout_files)
        if newly_seen > 0:
            print(
                f"ℹ️ {newly_seen} group files appeared after the file list was "
                "frozen; they are ignored this run (frozen-dataset semantics)."
            )
    else:
        base_train, heldout_files = split_heldout(all_files, cp_config["heldout_mod"])
        train_files = build_epoch_stream(base_train, num_epochs, cp_config["seed"])
    missing = [
        p for p in set(train_files) | set(heldout_files) if not os.path.exists(p)
    ]
    assert not missing, (
        f"{len(missing)} frozen group files are missing, e.g. {missing[:3]}"
    )

    groups_per_step = cp_config["groups_per_step"]
    # drop-last within the whole multi-epoch stream
    planned_steps = len(train_files) // groups_per_step
    if cp_config["max_steps"] is not None:
        planned_steps = min(planned_steps, int(cp_config["max_steps"]))
    assert planned_steps > 0, (
        f"Not enough train groups ({len(train_files)}) for one step of "
        f"{groups_per_step} groups."
    )
    print(
        f"📚 {len(base_train)} train groups x {num_epochs} epoch(s) = "
        f"{len(train_files)}, {len(heldout_files)} held-out groups "
        f"-> {planned_steps} steps of {groups_per_step} groups (resuming at {step})"
    )
    # A finished run relaunched unchanged used to fall straight through the
    # train loop and exit having done nothing. Say so instead of exiting silently.
    if step >= planned_steps:
        print(
            f"\n✅ Nothing to do: {step} steps already completed and this config "
            f"plans {planned_steps} (num_epochs={num_epochs}, "
            f"max_steps={cp_config['max_steps']}). Raise "
            "++critic_pretrain.num_epochs to train further.\n"
        )
        return

    # Scheduler budget: one tick per train() call, one call per step.
    if value_config.get("megatron_cfg", {}).get("enabled", False):
        value_config["megatron_cfg"]["train_iters"] = planned_steps

    # ------------------------------------------------------------------
    # Cluster + value model (mirrors ppo.setup()'s init_value resume probe).
    # ------------------------------------------------------------------
    cluster = RayVirtualCluster(
        name="critic_pretrain_cluster",
        bundle_ct_per_node_list=[cluster_config["gpus_per_node"]]
        * cluster_config["num_nodes"],
        use_gpus=True,
        num_gpus_per_node=cluster_config["gpus_per_node"],
        max_colocated_worker_groups=1,
        port_range_low=cluster_config.get("master_port_range_low"),
        port_range_high=cluster_config.get("master_port_range_high"),
    )
    print(
        f"  ✓ Ray cluster: {cluster_config['num_nodes']} nodes x "
        f"{cluster_config['gpus_per_node']} GPUs (value model only)"
    )

    eval_only = bool(cp_config.get("eval_only"))
    if eval_only and cp_config.get("eval_checkpoint_path"):
        # Eval/dump mode scores with an EXPLICIT checkpoint (e.g. .../step_10),
        # independent of this dir's latest; no optimizer needed.
        _value_weights = _Path(cp_config["eval_checkpoint_path"]) / "value" / "weights"
        assert _value_weights.exists(), (
            f"eval_checkpoint_path has no value/weights: {_value_weights}"
        )
        value_weights_path = _value_weights
        value_optimizer_path = None
        print(f"  ✓ Eval mode: loading critic from {value_weights_path}")
    elif last_checkpoint_path:
        _value_weights = _Path(last_checkpoint_path) / "value" / "weights"
        _value_optim = _Path(last_checkpoint_path) / "value" / "optimizer"
        value_weights_path = _value_weights if _value_weights.exists() else None
        value_optimizer_path = _resolve_resume_optimizer_path(
            _value_optim, value_weights_path, value_config
        )
        if value_weights_path is not None:
            print(f"  ✓ Resuming value model from: {value_weights_path}")
    else:
        value_weights_path = None
        value_optimizer_path = None

    value_model = Value(
        cluster=cluster,
        config=value_config,
        tokenizer=tokenizer,
        name_prefix="lm_value",
        weights_path=value_weights_path,
        optimizer_path=value_optimizer_path,
        init_optimizer=not eval_only,
    )
    value_model.finish_training()  # block init, offload until first use
    print("  ✓ Value model initialized")

    value_loss_fn = MseValueLossFn(master_config.value_loss_fn)
    adv_estimator = _create_advantage_estimator(master_config)

    if eval_only:
        dump_dir = _Path(
            cp_config.get("dump_dir")
            or os.path.join(
                master_config.checkpointing["checkpoint_dir"], "value_dumps"
            )
        )
        print(
            f"🔍 Eval-only: scoring {len(heldout_files)} held-out groups -> {dump_dir}"
        )
        val_metrics = _heldout_metrics(
            value_model, adv_estimator, heldout_files, tokenizer, master_config
        )
        logger.log_metrics(val_metrics, 0, prefix="validation")
        print("  heldout metrics:", {k: round(v, 4) for k, v in val_metrics.items()})
        _dump_heldout_values(
            value_model,
            adv_estimator,
            heldout_files,
            tokenizer,
            master_config,
            dump_dir,
            int(cp_config["dump_text_groups"]),
            int(cp_config["dump_token_samples"]),
        )
        with open(dump_dir / "summary.json", "w") as f:
            json.dump(
                {
                    "checkpoint": str(value_weights_path),
                    "num_groups": len(heldout_files),
                    "metrics": val_metrics,
                },
                f,
                indent=2,
            )
        print(f"🏁 Eval dump complete: {dump_dir}")
        return

    expected_gbs = value_config["train_global_batch_size"]
    save_period = master_config.checkpointing["save_period"]
    checkpointing_enabled = master_config.checkpointing["enabled"]
    eval_period = cp_config["eval_period"]
    heldout_eval_files = heldout_files[: cp_config["heldout_max_groups"]]

    # ------------------------------------------------------------------
    # Train loop: one pass over the frozen file order.
    # ------------------------------------------------------------------
    while step < planned_steps:
        step_start = time.perf_counter()
        print(f"\n{'=' * 25} Critic step {step + 1}/{planned_steps} {'=' * 25}")

        step_files = train_files[step * groups_per_step : (step + 1) * groups_per_step]
        groups = [load_group(p) for p in step_files]
        train_data, repeated_batch = build_value_train_data(
            groups, tokenizer, master_config
        )
        if train_data["input_ids"].shape[0] != expected_gbs:
            raise ValueError(
                f"Step batch has {train_data['input_ids'].shape[0]} samples but "
                f"value.train_global_batch_size={expected_gbs}. Override "
                "value.train_global_batch_size (and critic_pretrain."
                "groups_per_step) to match groups_per_step * gens_per_prompt "
                "of the stored shards."
            )

        print("▶ Computing values...")

        critic_batch, turn_spans = _forward_values_and_returns(
            value_model,
            adv_estimator,
            train_data,
            repeated_batch,
            tokenizer,
            master_config,

        )

        print("▶ Training critic...")
        value_model.prepare_for_training()
        value_train_batch = critic_batch if critic_batch is not None else train_data
        # Same residual bookkeeping the PPO loops apply: without it the value
        # loss sees no return-space offsets and critic/ev_res silently
        # duplicates critic/explained_var, and homogeneous_group_weight would be
        # a no-op that the launcher nonetheless advertises.
        value_train_batch = _prepare_value_train_batch(
            value_train_batch, adv_estimator, master_config
        )
        value_results = value_model.train(value_train_batch, value_loss_fn)
        value_model.finish_training()

        # ---- Metrics ----
        metrics = _compute_critic_metrics(value_results)
        # critic/loss and critic/grad_norm come back as numpy arrays; the async
        # loop scalarizes ndarray metrics before printing/logging — mirror that.
        for k, v in metrics.items():
            if isinstance(v, (np.ndarray, list)):
                metrics[k] = np.sum(v).item()
        metrics.update(
            _positional_value_metrics(
                train_data["values"],
                train_data["returns"],
                turn_spans.anchor_mask
                if turn_spans is not None
                else train_data["token_mask"],
                returns_to_abs=getattr(adv_estimator, "last_returns_to_abs", None),
                returns_to_res=getattr(adv_estimator, "last_returns_to_res", None),
            )
        )
        metrics.update(
            _mixed_group_value_metrics(
                train_data["values"],
                train_data["returns"],
                turn_spans.anchor_mask
                if turn_spans is not None
                else train_data["token_mask"],
                _mixed_group_mask(adv_estimator),
                returns_to_res=getattr(adv_estimator, "last_returns_to_res", None),
            )
        )
        metrics.update(getattr(adv_estimator, "last_metrics", {}) or {})

        metrics["reward"] = train_data["rewards"].float().mean().item()
        metrics["num_samples"] = float(train_data["input_ids"].shape[0])
        metrics["total_step_time"] = time.perf_counter() - step_start
        logger.log_metrics(metrics, step + 1, prefix="train")
        print(
            f"  step {step + 1}: loss={metrics.get('critic/loss'):.6f} "
            f"ev={metrics.get('critic/explained_var'):.4f} "
            f"reward={metrics['reward']:.3f} "
            f"({metrics['total_step_time']:.1f}s)"
        )

        # ---- Held-out eval ----
        is_last_step = step + 1 == planned_steps
        if heldout_eval_files and (
            (eval_period > 0 and (step + 1) % eval_period == 0) or is_last_step
        ):
            print("🔍 Held-out eval...")
            val_metrics = _heldout_metrics(
                value_model,
                adv_estimator,
                heldout_eval_files,
                tokenizer,
                master_config,
            )
            logger.log_metrics(val_metrics, step + 1, prefix="validation")
            print(
                f"  heldout: ev={val_metrics.get('critic/explained_var', float('nan')):.4f} "
                f"terminal_auc={val_metrics.get('critic/terminal_auc', float('nan')):.4f}"
            )

        # ---- Checkpoint (value/ only — stage C's warm-start seed layout) ----
        step += 1
        save_state["total_steps"] = step
        save_state["groups_consumed"] = step * groups_per_step
        save_state["consumed_samples"] = save_state.get("consumed_samples", 0) + int(
            train_data["input_ids"].shape[0]
        )
        if checkpointing_enabled and (is_last_step or step % save_period == 0):
            print(f"💾 Saving checkpoint for step {step}...")
            checkpoint_path = checkpointer.init_tmp_checkpoint(
                step, save_state, master_config
            )
            value_model.prepare_for_training()
            value_model.save_checkpoint(
                weights_path=os.path.join(checkpoint_path, "value", "weights"),
                optimizer_path=os.path.join(checkpoint_path, "value", "optimizer"),
                tokenizer_path=os.path.join(checkpoint_path, "value", "tokenizer"),
                checkpointing_cfg=master_config.checkpointing,
            )
            value_model.finish_training()
            with open(os.path.join(checkpoint_path, FILE_LIST_NAME), "w") as f:
                json.dump(
                    {
                        "train": [str(p) for p in train_files],
                        "heldout": [str(p) for p in heldout_files],
                    },
                    f,
                )
            checkpointer.finalize_checkpoint(checkpoint_path)
            print(f"  ✓ Checkpoint saved: step_{step}")

    print(
        f"\n🏁 Critic pretraining complete: {step} steps, "
        f"{save_state['consumed_samples']} samples. Latest checkpoint: "
        f"{checkpointer.get_latest_checkpoint_path()}"
    )
