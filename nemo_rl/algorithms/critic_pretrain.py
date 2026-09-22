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
from nemo_rl.utils.timer import TimeoutChecker

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


def pick_response_donors(
    base_files: list[Path], donors_per_step: int, full_per_step: int, seed: int
) -> set[Path]:
    """Choose which prompts donate their tail responses to the response-eval set.

    Sized so both pools run out together: a step consumes ``donors_per_step``
    donor groups and ``full_per_step`` full ones, so the donor share of the
    corpus must be ``a / (a + b)``. Any other split exhausts one pool early and
    silently shortens the epoch to whichever runs dry first.

    Deterministic in ``seed`` and in the (sorted) base set, so resume
    regenerates exactly the same donors — a re-draw would leak previously
    held-out responses into training.
    """
    if donors_per_step <= 0:
        return set()
    ordered = sorted(base_files, key=str)
    n_donors = round(len(ordered) * donors_per_step / (donors_per_step + full_per_step))
    n_donors = max(0, min(len(ordered), n_donors))
    return set(random.Random(seed).sample(ordered, n_donors))


RESPONSE_SPLIT_NAME = "response_split.json"


def load_or_create_response_split(
    shards_dir: str | Path,
    base_files: list[Path],
    donors_per_step: int,
    full_per_step: int,
    heldout_responses_per_group: int,
    heldout_mod: int,
    seed: int,
    redraw: bool = False,
) -> set[Path]:
    """The donor assignment as a shards-level sidecar, shared by every run.

    ``pick_response_donors`` is deterministic in (sorted base set, seed) — but
    the base set is a moving target while a collection fills, and a
    different-length base gives a completely DIFFERENT draw, not a superset.
    Two runs are therefore only comparable if they launch on the identical file
    set. This sidecar freezes the assignment once, in SHARDS_DIR itself: the
    first run writes it, every later run loads and VERIFIES it, so all arms of
    an A/B score the same donors' tail responses no matter when they launch.

    File names are stored RELATIVE to ``shards_dir`` so the /lustre and
    /scratch aliases of the same directory read one split. Any drift — base
    set, K, a/b ratio, heldout_mod, seed — fails loud. ``redraw=True``
    (``critic_pretrain.response_split_redraw``) renames the old sidecar aside
    and draws a fresh split: the explicit, logged way to accept that eval
    comparability with earlier runs on this collection is broken.

    Concurrent first launches write identical content (same base, same seed;
    atomic replace), so the race is benign; a launch during collection followed
    by one after it is exactly the drift case the verification rejects.
    """
    shards_dir = Path(shards_dir)
    split_path = shards_dir / RESPONSE_SPLIT_NAME
    rel_of = {p: os.path.relpath(str(p), str(shards_dir)) for p in base_files}
    base_rel = sorted(rel_of.values())
    expected = {
        "heldout_responses_per_group": heldout_responses_per_group,
        "donors_per_step": donors_per_step,
        "full_per_step": full_per_step,
        "heldout_mod": heldout_mod,
        "seed": seed,
    }
    if split_path.exists() and not redraw:
        with open(split_path) as f:
            saved = json.load(f)
        for key, want in expected.items():
            got = saved.get(key)
            if got != want:
                raise ValueError(
                    f"{split_path} was drawn with {key}={got!r} but this run uses "
                    f"{want!r}. The donor assignment is only meaningful for the "
                    "parameters it was drawn with — relaunch with the recorded "
                    "parameters, or pass ++critic_pretrain.response_split_redraw="
                    "true to draw a fresh split (breaks eval comparability with "
                    "every earlier run on this collection)."
                )
        saved_base = saved["base_files"]
        if saved_base != base_rel:
            gone = sorted(set(saved_base) - set(base_rel))
            new = sorted(set(base_rel) - set(saved_base))
            raise ValueError(
                f"{split_path} was drawn over {len(saved_base)} base groups but "
                f"this run sees {len(base_rel)} ({len(new)} new, {len(gone)} "
                f"missing; e.g. new={new[:3]} missing={gone[:3]}). A split drawn "
                "on a different base is a different experiment — if the "
                "collection finished since the split was drawn (pilot -> "
                "production), pass ++critic_pretrain.response_split_redraw=true; "
                "if files vanished, restore them."
            )
        donors_rel = set(saved["donors"])
        abs_of = {r: p for p, r in rel_of.items()}
        print(
            f"  ✓ Response split loaded from {split_path}: "
            f"{len(donors_rel)}/{len(base_rel)} donors "
            f"(drawn {saved.get('created', '?')})"
        )
        return {abs_of[r] for r in donors_rel}

    donors = pick_response_donors(base_files, donors_per_step, full_per_step, seed)
    payload = {
        **expected,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_base": len(base_files),
        "n_donors": len(donors),
        "donors": sorted(rel_of[p] for p in donors),
        "base_files": base_rel,
    }
    if split_path.exists():  # redraw: keep the old assignment for the record
        backup = split_path.with_name(
            f"{RESPONSE_SPLIT_NAME}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
        )
        os.replace(split_path, backup)
        print(f"  ⚠️ response_split_redraw: previous split moved to {backup}")
    tmp = split_path.with_name(f"{RESPONSE_SPLIT_NAME}.tmp-{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp, split_path)
    print(
        f"  ✓ Response split drawn and frozen to {split_path}: "
        f"{len(donors)}/{len(base_files)} donors (K="
        f"{heldout_responses_per_group}, a={donors_per_step}, b={full_per_step})"
    )
    return donors


def build_packed_epoch_stream(
    donors: list[Path],
    fulls: list[Path],
    donors_per_step: int,
    full_per_step: int,
    num_epochs: int,
    seed: int,
) -> list[Path]:
    """Multi-epoch stream where every ``a + b`` slice is a exactly-GBS step.

    Donor groups contribute ``gpp - K`` samples and full groups ``gpp``, so a
    step is only a fixed sample count if it mixes a FIXED number of each:
    ``a`` donors + ``b`` fulls. Emitting them pre-interleaved in step order
    keeps the flat-list contract the rest of the loop (and the whole frozen /
    resume path) is built on — ``train_files[step * gps : (step + 1) * gps]``
    stays correct with ``gps = a + b``.

    Epoch ``e`` shuffles each pool with ``Random(seed + e)`` independently, so
    the same prefix-replay guarantee as ``build_epoch_stream`` holds.
    """
    stream: list[Path] = []
    for epoch in range(num_epochs):
        d = sorted(donors, key=str)
        f = sorted(fulls, key=str)
        random.Random(seed + epoch).shuffle(d)
        random.Random(seed + epoch + 10_000).shuffle(f)
        steps = min(len(d) // donors_per_step, len(f) // full_per_step)
        for s in range(steps):
            stream.extend(d[s * donors_per_step : (s + 1) * donors_per_step])
            stream.extend(f[s * full_per_step : (s + 1) * full_per_step])
    return stream


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


def _same_model_dir(a: Any, b: Any) -> bool:
    """True if two model paths resolve to the same directory.

    Only claims equality when BOTH paths exist and resolve to one real
    directory — realpath() passes a nonexistent path through unchanged, which
    would silently equate two genuinely different (and missing) checkpoints.
    """
    if not isinstance(a, str) or not isinstance(b, str):
        return False
    if not (os.path.isdir(a) and os.path.isdir(b)):
        return False
    return os.path.realpath(a) == os.path.realpath(b)


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
    alias_noted = False
    for meta_path in metas:
        with open(meta_path) as f:
            meta = json.load(f)
        for key, want in expected.items():
            got = meta.get(key)
            # A "<ckpt>_copy" symlink alias is the normal way to give a second
            # run its own Megatron import cache under HF_HOME (the cache dir is
            # keyed on the literal path). Same target => same weights and same
            # tokenizer, so the shards are valid; chat_template_sha256 below is
            # what actually guards token-id validity.
            if key == "model_name" and got != want and _same_model_dir(got, want):
                if not alias_noted:
                    print(
                        f"  ✓ model_name differs by path alias only "
                        f"({got!r} -> {os.path.realpath(str(got))}); accepting shards."
                    )
                    alias_noted = True
                continue
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
    # Train the critic on env-flagged (mask_sample) rollouts instead of dropping
    # them. The policy masks these because a wall-clock timeout makes the reward
    # unattributable to the actions; the critic's job is different — it predicts
    # the return, and for a timed-out trajectory 0 IS the realised return. On the
    # SWE shards this is ~13% more data whose label is specifically "this
    # trajectory went nowhere", which is the within-trajectory signal a
    # terminal-reward critic is otherwise starved of.
    #
    # TRAIN ONLY. The held-out batches keep masking regardless (see
    # build_value_train_data's apply_env_mask), so critic/explained_var stays on
    # the same yardstick as every earlier run and this stays a clean A/B.
    cfg.setdefault("train_on_env_masked", False)
    # Train the critic on TRUNCATED rollouts too (spec: the 196k budget is part
    # of the environment; a truncated rollout's evaluated reward IS a legitimate
    # MC sample of the return, and excluding them makes V optimistic exactly on
    # long-horizon states). TRAIN ONLY — held-out batches keep the overlong
    # mask so critic/explained_var stays on the historical yardstick.
    cfg.setdefault("train_on_truncated", False)
    cfg["train_on_env_masked"] = bool(cfg["train_on_env_masked"])
    # ---- Response-level held-out split (see split_group_responses) ----
    # heldout_mod holds out whole PROMPTS, which measures generalisation to an
    # unseen task. That is not the regime stage C runs in: PPO trains over the
    # same prompt set this critic was pretrained on, so at serve time the critic
    # is scoring FRESH RESPONSES to prompts it has already memorised. K > 0
    # reserves the last K responses of every training group for a second eval
    # set that reproduces exactly that condition.
    #
    # Both splits are reported. The prompt-level one keeps its existing metric
    # names, so critic/explained_var stays comparable to every earlier run.
    cfg.setdefault("heldout_responses_per_group", 0)
    cfg["heldout_responses_per_group"] = int(cfg["heldout_responses_per_group"])
    # Donor groups per step. Only a SUBSET of prompts gives up responses, so the
    # cost is (donor fraction x K/gpp) of the corpus rather than K/gpp of it:
    # at gpp=16, K=4, GBS=512 a step packs a donors x 12 + b fulls x 16 = 512,
    # e.g. a=16/b=20 loses 11% of training data where holding K out of EVERY
    # group would lose 25%. b is derived and the tiling is asserted exact.
    cfg.setdefault("heldout_response_donor_groups_per_step", 0)
    cfg["heldout_response_donor_groups_per_step"] = int(
        cfg["heldout_response_donor_groups_per_step"]
    )
    # Scoring the response-eval set over every training prompt would cost more
    # than a train step; cap it at a FIXED prefix of the (sorted) train files so
    # the metric is comparable across steps, runs and resumes.
    cfg.setdefault("heldout_response_max_groups", 512)
    cfg["heldout_response_max_groups"] = int(cfg["heldout_response_max_groups"])
    # Redraw the shards-level donor assignment (response_split.json) instead of
    # loading it. Breaks eval comparability with every earlier run on the same
    # collection — the sidecar exists precisely to prevent that happening by
    # accident — so this must be asked for explicitly (pilot -> production).
    cfg.setdefault("response_split_redraw", False)
    cfg["response_split_redraw"] = bool(cfg["response_split_redraw"])
    # ---- Stage-1 soft targets (prior consolidation) ----
    # For the first N epochs every sample regresses to its GROUP's mean reward
    # instead of its own 0/1 outcome. The target is identical across a task's
    # trajectories, so per-trajectory features have nothing left to explain and
    # the only zero-loss solution is task-keyed — replay epochs cannot buy
    # anything by memorising individual trajectories (which is what ate the
    # gpp8 run: train EV 0.53 > the 0.44 task-prior ceiling). Hard 0/1 targets
    # resume from epoch N (stage 2). BOTH held-out evals always score against
    # the REAL rewards, so validation_response/critic/explained_var reads the
    # same quantity in every phase.
    cfg.setdefault("soft_target_epochs", 0)
    cfg["soft_target_epochs"] = int(cfg["soft_target_epochs"])
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


def split_group_responses(
    group: dict[str, Any], start: int, end: int
) -> dict[str, Any]:
    """A copy of a loaded group holding only samples ``[start, end)``.

    The unit of the shard pipeline is a whole group (one prompt x gpp
    responses), so a response-level train/eval split has to cut INSIDE the
    payload. ``BatchedDataDict.slice`` does the per-field work — tensors,
    ``message_log`` and ``extra_env_info`` are all per-sample and slice
    together, which is what keeps a sliced group a valid input to
    ``build_value_train_data``.

    Slicing by POSITION (not a shuffle) is deliberate: generation order carries
    no meaning here — the gym fans out gpp independent rollouts of one prompt —
    so a positional cut is an unbiased split, and being deterministic it
    survives resume and reproduces across runs.
    """
    batch = group["batch"]
    size = batch.size
    if "rollout_local_idx" in batch:
        # Multi-trace shards: rows are TRACES, several per rollout, stored
        # contiguously per rollout. `start`/`end` are ROLLOUT indices; cutting
        # by raw row position would split a rollout's root and subagent traces
        # across train/heldout (label leakage: siblings share the reward).
        rid = batch["rollout_local_idx"].tolist()
        assert all(a <= b for a, b in zip(rid, rid[1:])), (
            "rollout_local_idx is not non-decreasing within a group — the "
            "flatten order changed; response-level splitting cannot cut at "
            "rollout boundaries."
        )
        n_rollouts = len(set(rid))
        assert 0 <= start < end <= n_rollouts, (
            f"response slice [{start}, {end}) out of range for a group of "
            f"{n_rollouts} rollouts ({size} trace rows); check critic_pretrain."
            "heldout_responses_per_group against the shards' gens_per_prompt"
        )
        row_start = next((i for i, r in enumerate(rid) if r >= start), size)
        row_end = next((i for i, r in enumerate(rid) if r >= end), size)
        out = dict(group)
        out["batch"] = batch.slice(row_start, row_end)
        return out
    assert 0 <= start < end <= size, (
        f"response slice [{start}, {end}) out of range for a group of {size} "
        "samples; check critic_pretrain.heldout_responses_per_group against "
        "the shards' gens_per_prompt"
    )
    out = dict(group)
    out["batch"] = batch.slice(start, end)
    return out


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
    apply_env_mask: bool = True,
    apply_overlong_mask: bool = True,
) -> tuple[BatchedDataDict, BatchedDataDict]:
    """Assemble (train_data, repeated_batch) from loaded group payloads.

    ``apply_env_mask=False`` keeps env-flagged (``mask_sample``) rollouts in the
    batch — see ``critic_pretrain.train_on_env_masked``. Overlong filtering is
    unaffected either way. Callers building HELD-OUT batches must leave this
    True so the eval yardstick never moves.

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
    # Explicit per-row group identity (one shard group = one prompt). Under
    # multi-trace, sibling traces have different prompts, so prompt-token
    # grouping cannot recover this — carry it positionally instead.
    repeated_batch["shard_group_id"] = torch.cat(
        [
            torch.full((b.size,), gi, dtype=torch.int64)
            for gi, b in enumerate(per_prompt_batches)
        ]
    )

    use_overlong_filtering = master_config.ppo["overlong_filtering"]
    if use_overlong_filtering and apply_overlong_mask:
        loss_multiplier = repeated_batch["loss_multiplier"].clone()
        truncated = repeated_batch["truncated"]
        if isinstance(truncated, list):
            truncated = torch.tensor(truncated, dtype=torch.bool)
        loss_multiplier[truncated] = 0
        repeated_batch["loss_multiplier"] = loss_multiplier

    if "mask_sample" in repeated_batch:
        mask_sample = repeated_batch["mask_sample"]
        if isinstance(mask_sample, list):
            mask_sample = torch.tensor(mask_sample, dtype=torch.bool)
        mask_sample = mask_sample.bool()
        if apply_env_mask:
            loss_multiplier = repeated_batch["loss_multiplier"].clone()
            loss_multiplier[mask_sample] = 0
            repeated_batch["loss_multiplier"] = loss_multiplier
        elif int(mask_sample.sum()):
            # Say what is being kept: this is the whole point of the flag, and
            # silence here would make an env-mask A/B unfalsifiable from a log.
            print(
                f"  📊 train_on_env_masked: KEEPING "
                f"{int(mask_sample.sum())}/{len(mask_sample)} env-flagged samples"
            )

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


def _pad_rows_for_value(
    data: BatchedDataDict, multiple: int
) -> tuple[BatchedDataDict, int]:
    """Row-pad every per-row tensor to a multiple of the value model's sharding.

    Multi-trace batches have a variable row count; the value workers shard rows
    by data-parallel rank. Padding duplicates row 0 with sample_mask zeroed so
    padded rows contribute no loss and no metric. Returns (padded_batch,
    n_unpadded); the input is returned unchanged when already divisible.
    """
    n = data["input_ids"].shape[0]
    pad = (-n) % max(int(multiple), 1)
    if pad == 0:
        return data, n
    out = type(data)()
    idx = torch.zeros(pad, dtype=torch.long)
    for key, value in data.items():
        if torch.is_tensor(value) and value.dim() >= 1 and value.shape[0] == n:
            out[key] = torch.cat([value, value[idx]], dim=0)
        elif isinstance(value, list) and len(value) == n:
            out[key] = list(value) + [value[0]] * pad
        else:
            out[key] = value
    out["sample_mask"] = out["sample_mask"].clone()
    out["sample_mask"][n:] = 0
    return out, n


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
    metrics_out: Optional[dict[str, float]] = None,
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
    from nemo_rl.algorithms.swe_privileged_critic import (
        build_swe_privileged_value_inputs,
        build_turn_value_batch_augmented,
    )
    from nemo_rl.algorithms.swe_privileged_critic import (
        resolve_config as swe_privileged_resolve_config,
    )

    privileged_critic_cfg = master_config.value.get("privileged_critic")
    if privileged_critic_cfg is not None and not privileged_critic_cfg.get("enabled"):
        privileged_critic_cfg = None
    swe_privileged_cfg = swe_privileged_resolve_config(master_config)
    critic_batch = None

    # Turn structure (None on the token-level path). Stage B must build this the
    # same way stage C does, or the pretrained critic is supervised at positions
    # PPO never reads.
    turn_spans = build_turn_spans_for_batch(master_config, repeated_batch, train_data)

    value_model.prepare_for_inference()
    if swe_privileged_cfg is not None:
        critic_batch = build_swe_privileged_value_inputs(
            repeated_batch,
            tokenizer,
            swe_privileged_cfg,
            make_seq_len_divisible_by=master_config.policy[
                "make_sequence_length_divisible_by"
            ],
            metrics_out=metrics_out,
        )
    elif privileged_critic_cfg is not None:
        critic_batch = build_privileged_value_inputs(
            repeated_batch,
            tokenizer,
            privileged_critic_cfg,
            make_seq_len_divisible_by=master_config.policy[
                "make_sequence_length_divisible_by"
            ],
        )
    if critic_batch is not None:
        _is_multi_trace = "trace_in_rollout_idx" in repeated_batch and bool(
            (repeated_batch["trace_in_rollout_idx"] != 0).any()
        )
        if _is_multi_trace and turn_spans is not None:
            # Token-level privilege IS multi-trace safe (see below), but the
            # turn-level anchor remap has not been verified against sibling
            # traces — fail loud rather than supervise at wrong positions.
            raise NotImplementedError(
                "Privileged + TURN-LEVEL critic pretraining is not supported on "
                "multi-trace shards (the anchor remap into the augmented batch "
                "assumes one trace per rollout). Use the token-level estimator."
            )
        # Token level: both the builder and remap_by_response_mask are strictly
        # PER-ROW — the block is prefixed to each row's own message log and
        # values are carried back row i -> row i by response-token count. A
        # subagent trace is just another row whose extra_env_info names the same
        # instance, so it receives the same reference block as its root (verified
        # on real shards: 45/45 rows incl. 13 subagent traces resolve the golden
        # patch, one instance_id per group). The within-group no-confound
        # argument is likewise unaffected: the block stays byte-identical across
        # every trace of every sibling rollout.
        vals_aug = value_model.get_values(critic_batch)["values"].squeeze(-1)
        critic_batch["values"] = vals_aug
        train_data["values"] = remap_by_response_mask(
            vals_aug,
            critic_batch["token_mask"],
            train_data["token_mask"],
        )
    else:
        _value_dp = value_model.sharding_annotations.get_axis_size("data_parallel")
        _padded, _n_rows = _pad_rows_for_value(train_data, _value_dp)
        train_data["values"] = value_model.get_values(_padded)["values"].squeeze(-1)[
            :_n_rows
        ]
    value_model.finish_inference()

    if "rollout_local_idx" in repeated_batch:
        # Sibling traces have different prompts — prompt-token grouping would
        # put each trace in its own group and corrupt the residual/B_LOO
        # diagnostics. Use the explicit per-row shard group ids instead
        # (stamped by build_value_train_data; one shard group = one prompt).
        prompt_ids_for_adv = repeated_batch["shard_group_id"].unsqueeze(-1)
    else:
        initial_prompt_message_logs = extract_initial_prompt_messages(
            repeated_batch["message_log"],
            repeated_batch["length"],
        )
        prompt_batched_flat, _ = batched_message_log_to_flat_message(
            initial_prompt_message_logs,
            pad_value_dict={"token_ids": tokenizer.pad_token_id},
        )
        prompt_ids_for_adv = prompt_batched_flat["token_ids"]
    adv_kwargs = dict(
        prompt_ids=prompt_ids_for_adv,
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
    if turn_spans is not None and critic_batch is not None:
        # Privileged AND turn-level: anchors must be remapped into the augmented
        # layout, else the critic trains on policy-layout sequences while its
        # values came from privileged ones.
        critic_batch = build_turn_value_batch_augmented(
            critic_batch, train_data, turn_spans
        )
    elif turn_spans is not None:
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
    response_slice: Optional[tuple[int, int]] = None,
) -> dict[str, float]:
    """Critic quality on held-out rollouts: EV, positional EV/ECE, terminal AUC.

    ``response_slice=(start, end)`` scores only those response positions of each
    group, which is how the response-level eval set is drawn from prompts that
    are ALSO in training (see ``critic_pretrain.heldout_responses_per_group``).
    Left None the whole group is scored, i.e. the prompt-level held-out set.

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
    if response_slice is not None:
        groups = [split_group_responses(g, *response_slice) for g in groups]
    train_data, repeated_batch = build_value_train_data(
        groups, tokenizer, master_config
    )
    priv_metrics: dict[str, float] = {}
    _, turn_spans = _forward_values_and_returns(
        value_model,
        adv_estimator,
        train_data,
        repeated_batch,
        tokenizer,
        master_config,
        metrics_out=priv_metrics,
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
    # Trajectory-level AUCs count each ROLLOUT once: sibling traces repeat the
    # rollout reward, and a per-row AUC would be delegation-weighted.
    if "trace_in_rollout_idx" in repeated_batch:
        _ft = (repeated_batch["trace_in_rollout_idx"] == 0).cpu()
    else:
        _ft = torch.ones(values.shape[0], dtype=torch.bool)
    metrics["critic/terminal_auc"] = terminal_value_reward_auc(
        abs_values[_ft], train_data["rewards"][_ft], scored_mask[_ft]
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
                values[_ft],
                train_data["rewards"][_ft],
                group_ids.cpu()[_ft],
                scored_mask[_ft],
            )
        )
    metrics.update(getattr(adv_estimator, "last_metrics", {}) or {})
    metrics.update(priv_metrics)
    metrics["reward"] = train_data["rewards"][_ft].float().mean().item()
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
    _swe_privileged = value_config.get("swe_privileged_critic")
    if _swe_privileged is not None and _swe_privileged.get("enabled"):
        from nemo_rl.algorithms.swe_privileged_critic import privilege_budget_tokens

        _needed = master_config.policy[
            "max_total_sequence_length"
        ] + privilege_budget_tokens(_swe_privileged)
        if value_config["max_total_sequence_length"] < _needed:
            print(
                "  ↑ SWE privileged critic: raising value.max_total_sequence_length "
                f"{value_config['max_total_sequence_length']} -> {_needed}",
                flush=True,
            )
            value_config["max_total_sequence_length"] = _needed
        # The packing/dynamic-batching token budgets are OmegaConf interpolations
        # of max_total_sequence_length, but the runner resolves the config
        # (OmegaConf.to_container(resolve=True)) BEFORE setup() runs, so raising
        # the length above does not propagate to them. Without this the packer
        # raises "Sequence length N exceeds bin capacity" on the long tail --
        # minutes-to-hours into the run, not at startup.
        # Required field of ValueConfig; a call-site fallback here would
        # silently under-size the packing bins to _needed * 1 and resurface as
        # "Sequence length N exceeds bin capacity" deep into a run.
        _mbs = int(value_config["train_micro_batch_size"])
        for _bcfg_key in ("sequence_packing", "dynamic_batching"):
            _bcfg = value_config.get(_bcfg_key) or {}
            if not _bcfg.get("enabled"):
                continue
            for _tok_key in ("train_mb_tokens", "logprob_mb_tokens"):
                _want = _needed * _mbs
                if _bcfg.get(_tok_key) is not None and _bcfg[_tok_key] < _want:
                    print(
                        f"  ↑ SWE privileged critic: raising value.{_bcfg_key}.{_tok_key} "
                        f"{_bcfg[_tok_key]} -> {_want}",
                        flush=True,
                    )
                    _bcfg[_tok_key] = _want

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

    # ---- Response-level split: same prompts as training, unseen responses ----
    # heldout_mod holds out whole PROMPTS, i.e. it measures generalisation to an
    # unseen task. That is NOT the regime stage C runs in: PPO trains over the
    # same prompt set this critic was pretrained on, so at serve time the critic
    # scores fresh responses to prompts it has already memorised. K > 0 reserves
    # the last K responses of a SUBSET of prompts (the donors) for a second eval
    # set reproducing exactly that condition. Both splits are reported and the
    # prompt-level one keeps its metric names, so critic/explained_var stays
    # comparable to every earlier run.
    n_resp_heldout = cp_config["heldout_responses_per_group"]
    gpp = int(master_config.ppo["num_generations_per_prompt"])
    donors_per_step = cp_config["heldout_response_donor_groups_per_step"]
    full_per_step = 0
    donor_set: set[Path] = set()
    if n_resp_heldout > 0:
        assert 0 < n_resp_heldout < gpp, (
            f"critic_pretrain.heldout_responses_per_group={n_resp_heldout} must be "
            f"in (0, num_generations_per_prompt={gpp})"
        )
        assert donors_per_step > 0, (
            "critic_pretrain.heldout_responses_per_group > 0 requires "
            "critic_pretrain.heldout_response_donor_groups_per_step > 0 (how many "
            "reduced-size groups each step packs)."
        )
        gbs = int(value_config["train_global_batch_size"])
        donor_samples = donors_per_step * (gpp - n_resp_heldout)
        rest = gbs - donor_samples
        # Exact tiling or nothing: a step that does not total GBS would trip the
        # batch-size check every step, and rounding it would silently drop data.
        assert rest >= 0 and rest % gpp == 0, (
            f"No exact packing: {donors_per_step} donor groups x "
            f"({gpp} - {n_resp_heldout}) = {donor_samples} samples leaves "
            f"{rest} of value.train_global_batch_size={gbs}, which is not a "
            f"multiple of gpp={gpp}. Pick a donors_per_step where "
            f"(GBS - a*(gpp-K)) % gpp == 0."
        )
        full_per_step = rest // gpp

    def _make_stream(base: list[Path]) -> list[Path]:
        """The multi-epoch train stream, packed when a response split is on.

        Both the fresh and the resume path go through here so they cannot build
        different orders — the resume path's prefix check compares the two.
        """
        nonlocal donor_set
        if n_resp_heldout <= 0:
            return build_epoch_stream(base, num_epochs, cp_config["seed"])
        donor_set = load_or_create_response_split(
            cp_config["shards_dir"],
            base,
            donors_per_step,
            full_per_step,
            n_resp_heldout,
            cp_config["heldout_mod"],
            cp_config["seed"],
            redraw=cp_config["response_split_redraw"],
        )
        return build_packed_epoch_stream(
            [p for p in base if p in donor_set],
            [p for p in base if p not in donor_set],
            donors_per_step,
            full_per_step,
            num_epochs,
            cp_config["seed"],
        )

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
        train_files = _make_stream(base_train)
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
        train_files = _make_stream(base_train)
    missing = [
        p for p in set(train_files) | set(heldout_files) if not os.path.exists(p)
    ]
    assert not missing, (
        f"{len(missing)} frozen group files are missing, e.g. {missing[:3]}"
    )

    groups_per_step = cp_config["groups_per_step"]
    if n_resp_heldout > 0:
        # A packed step is a+b groups by construction, and the stream is emitted
        # in that layout — so groups_per_step is DERIVED, not configured. Take it
        # over rather than letting a stale launcher value slice across step
        # boundaries and mix a donor tail into the next step's batch.
        packed_gps = donors_per_step + full_per_step
        if groups_per_step != packed_gps:
            print(
                f"  ℹ️ groups_per_step {groups_per_step} -> {packed_gps} "
                f"({donors_per_step} donor x {gpp - n_resp_heldout} + "
                f"{full_per_step} full x {gpp} = "
                f"{value_config['train_global_batch_size']} samples/step)"
            )
        groups_per_step = packed_gps
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

    # The response-eval set is drawn ONLY from donor prompts — those are the
    # ones whose tail responses were withheld from training. A fixed prefix of
    # the sorted donors (not the shuffled stream) so the same prompts are scored
    # at every eval and the curve moves only when the critic does.
    response_eval_files: list[Path] = []
    if n_resp_heldout > 0:
        response_eval_files = sorted(donor_set, key=str)[
            : cp_config["heldout_response_max_groups"]
        ]
        n_donor, n_full = len(donor_set), len(base_train) - len(donor_set)
        lost = len(donor_set) * n_resp_heldout
        total = len(base_train) * gpp
        print(
            f"  ✓ Response-level split: {n_donor} donor prompts train on responses "
            f"[0,{gpp - n_resp_heldout}) and are evaluated on "
            f"[{gpp - n_resp_heldout},{gpp}); {n_full} prompts train on all {gpp}. "
            f"Withheld {lost}/{total} samples ({100 * lost / max(total, 1):.1f}%). "
            f"Response eval scores {len(response_eval_files)} prompts."
        )

    # Walltime-bounded save. planned_steps is many 4h windows long, so without
    # this a window that never lands on a save_period multiple is lost outright
    # (is_last_step only fires at the very end of the whole multi-epoch stream).
    # fit_last_save_time makes it fire one iteration EARLY, using the running
    # mean step time, so the save itself fits inside the budget.
    timeout = TimeoutChecker(
        timeout=master_config.checkpointing["checkpoint_must_save_by"],
        fit_last_save_time=True,
    )
    timeout.start_iterations()

    # ------------------------------------------------------------------
    # Train loop: one pass over the frozen file order.
    # ------------------------------------------------------------------
    while step < planned_steps:
        step_start = time.perf_counter()
        print(f"\n{'=' * 25} Critic step {step + 1}/{planned_steps} {'=' * 25}")

        step_files = train_files[step * groups_per_step : (step + 1) * groups_per_step]
        groups = [load_group(p) for p in step_files]
        if n_resp_heldout > 0:
            # Per FILE, not per position: donor membership is a fixed property
            # of a prompt, so its withheld tail stays withheld in every epoch.
            groups = [
                split_group_responses(g, 0, gpp - n_resp_heldout)
                if p in donor_set
                else g
                for g, p in zip(groups, step_files)
            ]
        # Train-only: held-out batches (evaluate_heldout / the value dump) always
        # apply the env mask, so critic/explained_var keeps comparing like with
        # like across runs regardless of this flag.
        train_data, repeated_batch = build_value_train_data(
            groups,
            tokenizer,
            master_config,
            apply_env_mask=not cp_config["train_on_env_masked"],
            apply_overlong_mask=not cp_config["train_on_truncated"],
        )
        num_step_rows = train_data["input_ids"].shape[0]
        if "trace_in_rollout_idx" in repeated_batch:
            num_step_rollouts = int(
                (repeated_batch["trace_in_rollout_idx"] == 0).sum()
            )
        else:
            num_step_rollouts = num_step_rows
        if num_step_rollouts != expected_gbs:
            raise ValueError(
                f"Step batch has {num_step_rollouts} rollouts ({num_step_rows} trace rows) but "
                f"value.train_global_batch_size={expected_gbs}. Override "
                "value.train_global_batch_size (and critic_pretrain."
                "groups_per_step) to match groups_per_step * gens_per_prompt "
                "of the stored shards"
                + (
                    f", MINUS the {n_resp_heldout} responses/group reserved for "
                    f"the response-level eval: expected "
                    f"{groups_per_step} * ({gpp} - {n_resp_heldout}) = "
                    f"{groups_per_step * (gpp - n_resp_heldout)}."
                    if n_resp_heldout > 0
                    else "."
                )
            )

        # ---- Stage-1 soft targets: regress to the group-mean reward ----
        # Applied BEFORE returns are computed, so GAE broadcasts the soft target
        # to every token exactly like a real reward. Donor groups were already
        # sliced above, so a donor's mean is over its TRAINING responses only —
        # the withheld eval responses never leak into the target. The epoch
        # boundary is derived from the frozen stream, so it is stable across
        # resume and across NUM_EPOCHS extensions (stage 2 = relaunch with a
        # larger num_epochs and the same soft_target_epochs).
        soft_target_epochs = cp_config["soft_target_epochs"]
        steps_per_epoch = max(1, len(train_files) // num_epochs // groups_per_step)
        in_soft_phase = (step // steps_per_epoch) < soft_target_epochs
        if soft_target_epochs > 0 and step % steps_per_epoch == 0:
            print(
                f"  🎯 epoch {step // steps_per_epoch}: "
                f"{'SOFT group-mean' if in_soft_phase else 'HARD per-trajectory'} "
                f"value targets (soft_target_epochs={soft_target_epochs}, "
                f"steps_per_epoch={steps_per_epoch})"
            )
        if in_soft_phase:
            # Clone: train_data["rewards"] aliases repeated_batch["total_reward"],
            # and the real rewards must keep flowing to logging/metrics.
            soft_rewards = train_data["rewards"].float().clone()
            sample_mask = train_data["sample_mask"].float()
            # Multi-trace: the group mean must be over ROLLOUTS — sibling
            # traces repeat the rollout reward, so a row mean would overweight
            # delegation-heavy rollouts. The soft target still broadcasts to
            # every trace row of the group (same semantics as a real reward).
            first_trace_all = (
                (repeated_batch["trace_in_rollout_idx"] == 0)
                if "trace_in_rollout_idx" in repeated_batch
                else torch.ones_like(sample_mask, dtype=torch.bool)
            )
            off = 0
            for g in groups:
                n = int(g["batch"].size)
                sl = slice(off, off + n)
                kept = (sample_mask[sl] > 0) & first_trace_all[sl]
                # An all-masked group keeps its raw rewards: its loss is fully
                # masked anyway, so no target is ever trained on.
                if bool(kept.any()):
                    soft_rewards[sl] = soft_rewards[sl][kept].mean()
                off += n
            assert off == soft_rewards.shape[0], (
                f"group sizes sum to {off} but the batch has "
                f"{soft_rewards.shape[0]} samples"
            )
            train_data["rewards"] = soft_rewards

        print("▶ Computing values...")
        priv_metrics: dict[str, float] = {}
        critic_batch, turn_spans = _forward_values_and_returns(
            value_model,
            adv_estimator,
            train_data,
            repeated_batch,
            tokenizer,
            master_config,
            metrics_out=priv_metrics,
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
        _value_dp = value_model.sharding_annotations.get_axis_size("data_parallel")
        _pad_mult = _value_dp * max(
            int(master_config.value["train_micro_batch_size"]), 1
        )
        value_train_batch, _ = _pad_rows_for_value(value_train_batch, _pad_mult)
        value_results = value_model.train(
            value_train_batch,
            value_loss_fn,
            gbs=value_train_batch["input_ids"].shape[0],
        )
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
        metrics.update(priv_metrics)
        # Real rewards, not the (possibly soft) value targets: train/reward must
        # stay comparable across phases and runs.
        _real_rewards = repeated_batch["total_reward"]
        if isinstance(_real_rewards, list):
            _real_rewards = torch.tensor(_real_rewards)
        metrics["reward"] = _real_rewards.float().mean().item()
        metrics["critic/soft_targets"] = 1.0 if in_soft_phase else 0.0
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
        # The two eval sets are gated on ONE schedule but independently on their
        # own file lists: nesting the response-level eval under
        # `if heldout_eval_files` would silently disable it whenever
        # heldout_mod <= 0, which is exactly the config someone picks when they
        # want the response split to be the only eval.
        is_last_step = step + 1 == planned_steps
        do_eval = (eval_period > 0 and (step + 1) % eval_period == 0) or is_last_step

        if do_eval and heldout_eval_files:
            print("🔍 Held-out eval (unseen prompts)...")
            val_metrics = _heldout_metrics(
                value_model,
                adv_estimator,
                heldout_eval_files,
                tokenizer,
                master_config,
            )
            logger.log_metrics(val_metrics, step + 1, prefix="validation")
            print(
                f"  heldout (unseen prompt): "
                f"ev={val_metrics.get('critic/explained_var', float('nan')):.4f} "
                f"terminal_auc={val_metrics.get('critic/terminal_auc', float('nan')):.4f}"
            )

        # Same prompts as training, responses the critic has never seen — the
        # stage C condition. Its own prefix, so it never collides with the
        # unseen-prompt numbers above.
        if do_eval and response_eval_files:
            print("🔍 Response-level eval (seen prompts, unseen responses)...")
            resp_metrics = _heldout_metrics(
                value_model,
                adv_estimator,
                response_eval_files,
                tokenizer,
                master_config,
                response_slice=(gpp - n_resp_heldout, gpp),
            )
            logger.log_metrics(resp_metrics, step + 1, prefix="validation_response")
            print(
                f"  heldout (unseen response): "
                f"ev={resp_metrics.get('critic/explained_var', float('nan')):.4f} "
                f"terminal_auc={resp_metrics.get('critic/terminal_auc', float('nan')):.4f} "
                f"within_group_auc_q4="
                f"{resp_metrics.get('critic/within_group_auc_q4', float('nan')):.4f}"
            )

        # ---- Checkpoint (value/ only — stage C's warm-start seed layout) ----
        step += 1
        save_state["total_steps"] = step
        save_state["groups_consumed"] = step * groups_per_step
        save_state["consumed_samples"] = save_state.get("consumed_samples", 0) + int(
            train_data["input_ids"].shape[0]
        )
        timeout.mark_iteration()
        should_save_by_step = is_last_step or step % save_period == 0
        # Latches: fires once, then returns False for the rest of the window.
        should_save_by_timeout = timeout.check_save()
        if should_save_by_timeout and not should_save_by_step:
            print(
                f"⏰ checkpoint_must_save_by budget reached — saving at step {step} "
                "before the walltime."
            )
        if checkpointing_enabled and (should_save_by_step or should_save_by_timeout):
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
