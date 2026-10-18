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
"""Offline PPO critic pretraining on SingleController rollout banks (stage B).

The legacy decoupled pipeline's stage B, on the SingleController's critic and
data plane. A ``rollout_collection`` job set (stage A: generation only, one small
job per data shard) banks finalized prompt groups -- the token-capture rows PPO
trains on, their raw flags, and each prompt's ``extra_env_info`` -- and this
driver fits the PPO critic to them, value model only. Per step:

1. ``value.train_global_batch_size`` rollouts' worth of banked groups go into a
   data-plane partition, padded to the critic's data-parallel multiple.
2. The critic scores them (``TQValue.get_values_from_meta``); with
   ``value.swe_privileged_critic`` the reference block is rebuilt from the banked
   env info and prefixed to every row.
3. GAE returns on those predictions, under legacy stage B's masks: the
   finalizer's ``sample_mask``, the env flag unless ``train_on_env_masked``, the
   overlong filter unless ``train_on_truncated``. There is no sequence-logprob
   gate: the bank holds no trainer logprobs, and legacy had none either.
4. One ``train_from_meta`` update, clipped against the step's own predictions.

Held out, as legacy: whole tasks (``heldout_mod``) and, with K =
``heldout_responses_per_group`` > 0, the last K responses of donor prompts that
otherwise train (``validation_response/*``: fresh responses to a seen prompt, the
PPO-time condition). The donor draw is frozen in ``response_split.json`` beside
the banks. Held-out batches always apply the env and overlong masks.

The output ``step_N/value/weights`` + ``config.yaml`` is what
``ppo.warm_start_value_checkpoint`` loads.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import re
import shutil
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import numpy as np
import torch
from pydantic import BaseModel, Field, model_validator
from tensordict import TensorDict

from nemo_rl.algorithms.advantage_estimator import GeneralizedAdvantageEstimator
from nemo_rl.algorithms.single_controller_utils.rollout_bank import (
    BANK_FIELDS,
    BankGroup,
    BankGroupRef,
    batch_fields,
    find_banks,
    list_bank_groups,
    load_bank_group,
    read_bank_meta,
)
from nemo_rl.algorithms.single_controller_utils.utils import (
    fields_for_put,
    squeeze_trailing_unit_dim,
    tensor_field,
)
from nemo_rl.algorithms.swe_privileged_critic import PRIVILEGE_KEY_TAG
from nemo_rl.data_plane import KVBatchMeta

FILE_LIST_NAME = "critic_pretrain_files.json"
RESPONSE_SPLIT_NAME = "response_split.json"
_PAD_SUFFIX = "_pad"
_ROLLOUT_IN_ID = re.compile(r"_g(\d+)(?:_t\d+)?$")


class CriticPretrainConfig(BaseModel, extra="forbid"):
    """``critic_pretrain``: offline critic fitting on rollout banks.

    The step size is ``value.train_global_batch_size`` ROLLOUTS (the legacy
    unit): with segment rows a rollout is ~1.4 rows, and the critic's actual
    global batch is every row of the step.
    """

    # Banks, or collection roots holding one bank per shard job.
    bank_dirs: list[str]
    # Passes over the training groups; epoch e is a fresh permutation seeded with
    # seed + e, so raising this on a finished run extends its stream.
    num_epochs: int = Field(default=1, gt=0)
    # Optional cap on the planned steps.
    max_steps: Optional[int] = Field(default=None, gt=0)
    # Hold out whole tasks whose key hashes to 0 mod this; 0 disables it.
    heldout_mod: int = Field(default=16, ge=0)
    # Held-out groups scored per evaluation (a fixed prefix); None scores all.
    heldout_max_groups: Optional[int] = Field(default=None, ge=0)
    # Evaluate every this many steps (and after the last); 0: only after the last.
    eval_period: int = Field(default=10, ge=0)
    seed: int = 42
    # Train on env-flagged (mask_sample) rollouts: for a timed-out trajectory 0
    # is the realised return. Train only; held-out batches keep the mask.
    train_on_env_masked: bool = False
    # Train on truncated rollouts: the length budget is part of the environment.
    # Train only; held-out batches keep the overlong mask.
    train_on_truncated: bool = False
    # Response-level held-out: K responses withheld from each donor prompt.
    heldout_responses_per_group: int = Field(default=0, ge=0)
    # Donor groups packed into every step (with K > 0).
    heldout_response_donor_groups_per_step: int = Field(default=0, ge=0)
    # Donor groups scored by the response-level eval (a fixed sorted prefix).
    heldout_response_max_groups: int = Field(default=512, ge=0)
    # Redraw response_split.json instead of loading it (breaks comparability
    # with earlier runs on the same banks).
    response_split_redraw: bool = False
    # Data-plane partition the driver stages its rows in.
    partition_id: str = "critic_pretrain"
    # Where the critic's optimizer state lives. Megatron embeds it in value/weights,
    # and for an Adam critic it is ~5/6 of a checkpoint. "every_checkpoint": every
    # step_N is a full checkpoint. "latest_only": step_N holds weights only (what a
    # PPO warm start loads) and one rolling full checkpoint, rewritten at every save
    # under <checkpoint_dir>/resume_state/current, resumes the run.
    optimizer_state: Literal["every_checkpoint", "latest_only"] = "every_checkpoint"

    @model_validator(mode="after")
    def validate_settings(self) -> "CriticPretrainConfig":
        """Require banks; the two response-split knobs go together."""
        if not self.bank_dirs:
            raise ValueError("critic_pretrain.bank_dirs is empty")
        if (self.heldout_responses_per_group > 0) != (
            self.heldout_response_donor_groups_per_step > 0
        ):
            raise ValueError(
                "critic_pretrain.heldout_responses_per_group and "
                "heldout_response_donor_groups_per_step must both be set (> 0) to "
                "enable the response-level held-out split, or both left at 0."
            )
        return self


# ── rolling resume checkpoint (optimizer_state="latest_only") ─────────────────

RESUME_STATE_DIRNAME = "resume_state"
RESUME_STEP_FILENAME = "step.json"


def resume_state_dirs(checkpoint_dir: str | os.PathLike) -> tuple[Path, Path]:
    """(current, tmp): the rolling full checkpoint and the one being written."""
    root = Path(checkpoint_dir) / RESUME_STATE_DIRNAME
    return root / "current", root / "tmp"


def rolling_resume_step(checkpoint_dir: str | os.PathLike) -> Optional[int]:
    """The step the rolling full checkpoint was saved at, or None if there is none."""
    current, _ = resume_state_dirs(checkpoint_dir)
    try:
        return int(json.loads((current / RESUME_STEP_FILENAME).read_text())["step"])
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return None


def promote_rolling_resume(checkpoint_dir: str | os.PathLike, step: int) -> None:
    """Make the just-written resume_state/tmp the rolling full checkpoint for ``step``.

    Called after step_N is finalized, so a crash in between leaves the previous
    rolling checkpoint, which no longer matches the latest step; resuming then
    falls back to step_N's weights with a fresh optimizer.
    """
    current, tmp = resume_state_dirs(checkpoint_dir)
    (tmp / RESUME_STEP_FILENAME).write_text(json.dumps({"step": step}))
    old = current.with_name("old")
    shutil.rmtree(old, ignore_errors=True)
    if current.exists():
        os.replace(current, old)
    os.replace(tmp, current)
    shutil.rmtree(old, ignore_errors=True)


def critic_resume_paths(
    checkpointer: Any,
    last_checkpoint_path: Optional[str | os.PathLike],
    last_step: int,
    optimizer_state: str,
) -> tuple[Optional[Path], Optional[Path], Optional[str]]:
    """(weights_path, optimizer_path, source) to resume the critic from."""
    if last_checkpoint_path is None:
        return None, None, None
    if optimizer_state == "latest_only":
        current, _ = resume_state_dirs(checkpointer.checkpoint_dir)
        rolling = rolling_resume_step(checkpointer.checkpoint_dir)
        if rolling == last_step:
            weights, optimizer = checkpointer.get_resume_paths(
                current, model_component="value"
            )
            return weights, optimizer, str(current)
        print(
            f"⚠️ No rolling full checkpoint for step {last_step} (found {rolling}); "
            f"resuming from {last_checkpoint_path}'s weights with a fresh optimizer.",
            flush=True,
        )
    weights, optimizer = checkpointer.get_resume_paths(
        last_checkpoint_path, model_component="value"
    )
    return weights, optimizer, str(last_checkpoint_path)


# ── split / order (pure, unit-tested) ───────────────────────────────────────


def is_heldout(task_key: str, heldout_mod: int) -> bool:
    """Whether a task is held out: a stable hash of its key, 0 mod ``heldout_mod``."""
    if heldout_mod <= 0:
        return False
    digest = hashlib.sha1(task_key.encode()).hexdigest()
    return int(digest, 16) % heldout_mod == 0


def split_heldout(
    refs: Sequence[BankGroupRef], heldout_mod: int
) -> tuple[list[BankGroupRef], list[BankGroupRef]]:
    """(train, held-out) groups, split by task so repeats never straddle the split."""
    train: list[BankGroupRef] = []
    heldout: list[BankGroupRef] = []
    for ref in refs:
        (heldout if is_heldout(ref.task_key, heldout_mod) else train).append(ref)
    return train, heldout


def step_layout(
    rollouts_per_step: int, gpp: int, withheld: int, donors_per_step: int
) -> tuple[int, int]:
    """(donor, full) groups per step so a step holds exactly ``rollouts_per_step``.

    A donor group trains on ``gpp - withheld`` rollouts and a full one on
    ``gpp``: ``a * (gpp - K) + b * gpp`` must equal the step size.
    """
    if withheld == 0:
        if rollouts_per_step % gpp:
            raise ValueError(
                f"value.train_global_batch_size={rollouts_per_step} rollouts is not "
                f"a whole number of {gpp}-rollout groups"
            )
        return 0, rollouts_per_step // gpp
    if not 0 < withheld < gpp:
        raise ValueError(
            f"heldout_responses_per_group={withheld} must be in (0, {gpp}), the "
            "banks' generations per prompt"
        )
    rest = rollouts_per_step - donors_per_step * (gpp - withheld)
    if rest < 0 or rest % gpp:
        raise ValueError(
            f"{donors_per_step} donor groups x {gpp - withheld} + b full groups x "
            f"{gpp} cannot make value.train_global_batch_size={rollouts_per_step} "
            "rollouts; adjust the donor count, K or the batch size"
        )
    return donors_per_step, rest // gpp


def pick_response_donors(
    base: Sequence[str], donors_per_step: int, full_per_step: int, seed: int
) -> set[str]:
    """Donor prompts, a share ``a / (a + b)`` of the base so both pools end together."""
    if donors_per_step <= 0:
        return set()
    ordered = sorted(base)
    n_donors = round(len(ordered) * donors_per_step / (donors_per_step + full_per_step))
    n_donors = max(0, min(len(ordered), n_donors))
    return set(random.Random(seed).sample(ordered, n_donors))


def load_or_create_response_split(
    root: Path,
    base: Sequence[str],
    *,
    withheld: int,
    donors_per_step: int,
    full_per_step: int,
    heldout_mod: int,
    seed: int,
    redraw: bool,
) -> set[str]:
    """The donor draw, frozen in ``root/response_split.json`` and shared by every run.

    Stored relative to ``root``. A later run must see the same base set and
    parameters; ``redraw`` moves the old split aside and draws again.
    """
    split_path = root / RESPONSE_SPLIT_NAME
    rel = {path: os.path.relpath(path, root) for path in base}
    base_rel = sorted(rel.values())
    expected = {
        "heldout_responses_per_group": withheld,
        "donors_per_step": donors_per_step,
        "full_per_step": full_per_step,
        "heldout_mod": heldout_mod,
        "seed": seed,
    }
    if split_path.exists() and not redraw:
        saved = json.loads(split_path.read_text())
        changed = [k for k, v in expected.items() if saved.get(k) != v]
        if changed or saved["base_files"] != base_rel:
            raise ValueError(
                f"{split_path} was drawn with different settings {changed} or a "
                "different base set; relaunch with its settings, or set "
                "critic_pretrain.response_split_redraw=true (breaks eval "
                "comparability with earlier runs on these banks)."
            )
        absolute = {r: p for p, r in rel.items()}
        return {absolute[r] for r in saved["donors"]}
    donors = pick_response_donors(base, donors_per_step, full_per_step, seed)
    if split_path.exists():
        backup = split_path.with_name(
            f"{RESPONSE_SPLIT_NAME}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
        )
        os.replace(split_path, backup)
        print(f"  ⚠️ response_split_redraw: previous split moved to {backup}")
    tmp = split_path.with_name(f".{RESPONSE_SPLIT_NAME}.tmp{os.getpid()}")
    tmp.write_text(
        json.dumps(
            {
                **expected,
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "donors": sorted(rel[p] for p in donors),
                "base_files": base_rel,
            },
            indent=1,
        )
    )
    os.replace(tmp, split_path)
    return donors


def build_epoch_stream(base: Sequence[str], num_epochs: int, seed: int) -> list[str]:
    """``num_epochs`` permutations of ``base``, epoch e seeded with seed + e."""
    stream: list[str] = []
    for epoch in range(num_epochs):
        order = list(base)
        random.Random(seed + epoch).shuffle(order)
        stream.extend(order)
    return stream


def build_packed_epoch_stream(
    donors: Sequence[str],
    fulls: Sequence[str],
    donors_per_step: int,
    full_per_step: int,
    num_epochs: int,
    seed: int,
) -> list[str]:
    """Epoch streams laid out step by step: ``a`` donors then ``b`` full groups."""
    stream: list[str] = []
    for epoch in range(num_epochs):
        d, f = sorted(donors), sorted(fulls)
        random.Random(seed + epoch).shuffle(d)
        random.Random(seed + epoch + 10_000).shuffle(f)
        steps = min(len(d) // donors_per_step, len(f) // full_per_step)
        for s in range(steps):
            stream.extend(d[s * donors_per_step : (s + 1) * donors_per_step])
            stream.extend(f[s * full_per_step : (s + 1) * full_per_step])
    return stream


def rollout_of_row(sample_id: str, tag: dict[str, Any]) -> int:
    """A row's rollout index within its group (the ``rollout_local_idx`` tag)."""
    if "rollout_local_idx" in tag:
        return int(tag["rollout_local_idx"])
    match = _ROLLOUT_IN_ID.search(sample_id)
    if match is None:
        raise ValueError(f"cannot tell which rollout row {sample_id!r} belongs to")
    return int(match.group(1))


def slice_rollouts(group: BankGroup, start: int, end: int) -> BankGroup:
    """The group's rows of rollouts ``[start, end)``, subagent rows included.

    By position, as legacy: the gym fans out independent rollouts of one prompt,
    so a positional cut is an unbiased split and is stable across resumes.
    """
    keep = [
        i
        for i, (sid, tag) in enumerate(zip(group.sample_ids, group.tags))
        if start <= rollout_of_row(sid, tag) < end
    ]
    return replace(
        group,
        sample_ids=[group.sample_ids[i] for i in keep],
        tags=[group.tags[i] for i in keep],
        sequence_lengths=[group.sequence_lengths[i] for i in keep],
        fields={name: [rows[i] for i in keep] for name, rows in group.fields.items()},
    )


def is_canonical(tag: dict[str, Any]) -> bool:
    """A rollout's canonical row (its first trace); segment rows repeat its reward."""
    return int(tag.get("trace_in_rollout_idx", 0)) == 0


@dataclass
class StagedRows:
    """A step's rows, ready for a data-plane put."""

    sample_ids: list[str]
    tags: list[dict[str, Any]]
    sequence_lengths: list[int]
    columns: dict[str, list[torch.Tensor]]
    # Index into the step's groups of each row's group (a pad row: its source's).
    group_of_row: list[int]
    is_pad: list[bool]

    @property
    def num_rows(self) -> int:
        return len(self.sample_ids)

    def real_rows(self) -> torch.Tensor:
        """``(B,)`` bool: rows that are not DP padding."""
        return ~torch.tensor(self.is_pad, dtype=torch.bool)

    def canonical_rows(self) -> torch.Tensor:
        """``(B,)`` bool: real rows that are a rollout's canonical row."""
        return self.real_rows() & torch.tensor(
            [is_canonical(tag) for tag in self.tags], dtype=torch.bool
        )


def stage_rows(groups: Sequence[BankGroup], *, id_prefix: str) -> StagedRows:
    """Concatenate groups' rows under step-unique ids.

    A group can come back in a later epoch, so ids are prefixed per step slot
    (``{prefix}{slot}-{original}``); the ``{group}_g{i}[_t{j}]`` grammar the
    row-identity and privilege code parse is kept.
    """
    fields = [name for name in BANK_FIELDS if name in groups[0].fields]
    for group in groups:
        missing = [name for name in fields if name not in group.fields]
        if missing:
            raise ValueError(
                f"bank group {group.group_id} lacks {missing}; banks with "
                "different fields cannot share a step"
            )
    staged = StagedRows([], [], [], {name: [] for name in fields}, [], [])
    for slot, group in enumerate(groups):
        for row, sample_id in enumerate(group.sample_ids):
            staged.sample_ids.append(f"{id_prefix}{slot}-{sample_id}")
            staged.tags.append(dict(group.tags[row]))
            staged.sequence_lengths.append(int(group.sequence_lengths[row]))
            staged.group_of_row.append(slot)
            staged.is_pad.append(False)
            for name in fields:
                staged.columns[name].append(group.fields[name][row])
    return staged


def pad_rows(staged: StagedRows, multiple: int) -> StagedRows:
    """Pad to a multiple of ``multiple`` with zero-weight copies of real rows.

    Same shape as the controller's DP pad rows (``{sid}_pad{i}``, cyclic
    sources, ``sample_mask`` zeroed): they shard and pack like real rows and
    carry no loss.
    """
    num_pad = (-staged.num_rows) % multiple
    if num_pad == 0:
        return staged
    sources = [i % staged.num_rows for i in range(num_pad)]
    return StagedRows(
        sample_ids=staged.sample_ids
        + [f"{staged.sample_ids[s]}{_PAD_SUFFIX}{i}" for i, s in enumerate(sources)],
        tags=staged.tags + [dict(staged.tags[s]) for s in sources],
        sequence_lengths=staged.sequence_lengths
        + [staged.sequence_lengths[s] for s in sources],
        columns={
            name: rows
            + [
                torch.zeros_like(rows[s]) if name == "sample_mask" else rows[s]
                for s in sources
            ]
            for name, rows in staged.columns.items()
        },
        group_of_row=staged.group_of_row + [staged.group_of_row[s] for s in sources],
        is_pad=staged.is_pad + [True] * num_pad,
    )


# ── masks / targets / metrics (pure, unit-tested) ───────────────────────────


def loss_sample_mask(
    sample_mask: torch.Tensor,
    mask_sample: torch.Tensor,
    truncated: torch.Tensor,
    *,
    apply_env_mask: bool,
    apply_overlong_mask: bool,
) -> torch.Tensor:
    """Legacy stage B's per-row loss weight (no sequence-logprob gate)."""
    mask = sample_mask.float()
    if apply_env_mask:
        mask = mask * (~mask_sample.bool()).float()
    if apply_overlong_mask:
        mask = mask * (~truncated.bool()).float()
    return mask


def gae_returns(
    estimator: GeneralizedAdvantageEstimator,
    *,
    token_mask: torch.Tensor,
    sample_mask: torch.Tensor,
    rewards: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """The critic's regression target, as the PPO advantage stage builds it."""
    mask = token_mask * sample_mask.unsqueeze(-1)
    if not bool(mask[:, 1:].bool().any()):
        return torch.zeros_like(mask)
    estimator_mask = token_mask if estimator.normalize_over == "all_rows" else mask
    _, returns = estimator.compute_advantage(
        prompt_ids=torch.zeros(rewards.shape[0], 1, dtype=torch.long),
        rewards=rewards,
        mask=estimator_mask,
        values=values,
        repeated_batch={"total_reward": rewards},
        valid_mask=sample_mask,
    )
    return returns


def _rank_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Mann-Whitney AUC with tie correction; NaN unless both classes are present."""
    labels = labels.bool()
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    scores = scores.double()
    order = torch.argsort(scores)
    ranks = torch.empty_like(scores)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=scores.dtype)
    for value in torch.unique(scores):
        tie = scores == value
        if int(tie.sum()) > 1:
            ranks[tie] = ranks[tie].mean()
    return float(
        (ranks[labels].sum().item() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    )


_BUCKETS = 4


@dataclass
class HeldoutStats:
    """Legacy held-out critic metrics, accumulated over chunks of whole groups.

    Token metrics pool every response token (``critic/explained_var``, the
    legacy yardstick, which ignores ``sample_mask``) and the trained ones
    (``critic/explained_var_masked``). Trajectory metrics use each rollout's
    canonical row once: ``terminal_auc`` (last response token's value vs
    success) and ``within_group_auc_q1..q4`` (sibling ranking by mean value per
    progress quarter, mixed-outcome groups only).
    """

    sums: dict[str, float] = field(default_factory=dict)
    last_values: list[float] = field(default_factory=list)
    labels: list[float] = field(default_factory=list)
    bucket_values: list[list[float]] = field(default_factory=list)
    bucket_present: list[list[bool]] = field(default_factory=list)
    group_ids: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    rows: int = 0
    _next_group: int = 0

    def _pool(self, prefix: str, r: torch.Tensor, v: torch.Tensor) -> None:
        e = r - v
        for name, value in (
            ("n", float(r.numel())),
            ("r", float(r.sum())),
            ("r2", float((r * r).sum())),
            ("e", float(e.sum())),
            ("e2", float((e * e).sum())),
        ):
            key = f"{prefix}{name}"
            self.sums[key] = self.sums.get(key, 0.0) + value

    def add(
        self,
        *,
        values: torch.Tensor,
        returns: torch.Tensor,
        token_mask: torch.Tensor,
        sample_mask: torch.Tensor,
        rewards: torch.Tensor,
        canonical: torch.Tensor,
        group_of_row: Sequence[int],
    ) -> None:
        """Accumulate one chunk's real rows (``group_of_row`` local to the chunk)."""
        tokens = token_mask.bool()
        trained = tokens & (sample_mask.unsqueeze(-1) > 0)
        values, returns = values.double(), returns.double()
        self._pool("all_", returns[tokens], values[tokens])
        self._pool("masked_", returns[trained], values[trained])
        self.rows += int(values.shape[0])
        rel = (torch.cumsum(tokens.long(), dim=1) - 1).double() / tokens.sum(
            dim=1, keepdim=True
        ).clamp(min=1).double()
        has_response = tokens.any(dim=1)
        last = tokens.shape[1] - 1 - tokens.fliplr().double().argmax(dim=1)
        offset = self._next_group
        for row in torch.nonzero(canonical & has_response).reshape(-1).tolist():
            self.last_values.append(float(values[row, last[row]]))
            self.labels.append(float(rewards[row] >= 0.5))
            self.rewards.append(float(rewards[row]))
            self.group_ids.append(offset + int(group_of_row[row]))
            means, present = [], []
            for b in range(_BUCKETS):
                lo, hi = b / _BUCKETS, (b + 1) / _BUCKETS
                upper = rel[row] < hi if b < _BUCKETS - 1 else rel[row] <= 1.0
                sel = tokens[row] & (rel[row] >= lo) & upper
                count = int(sel.sum())
                present.append(count > 0)
                means.append(float(values[row][sel].mean()) if count else 0.0)
            self.bucket_values.append(means)
            self.bucket_present.append(present)
        self._next_group = offset + max(group_of_row, default=-1) + 1

    def _explained_var(self, prefix: str) -> Optional[float]:
        n = self.sums.get(f"{prefix}n", 0.0)
        if n < 2:
            return None
        r_mean = self.sums[f"{prefix}r"] / n
        e_mean = self.sums[f"{prefix}e"] / n
        var_r = self.sums[f"{prefix}r2"] / n - r_mean**2
        var_e = self.sums[f"{prefix}e2"] / n - e_mean**2
        return 1.0 - var_e / var_r if var_r > 1e-8 else 0.0

    def metrics(self) -> dict[str, float]:
        """The legacy ``critic/*`` held-out metrics plus reward and row counts."""
        out: dict[str, float] = {"num_heldout_samples": float(self.rows)}
        for key, prefix in (
            ("critic/explained_var", "all_"),
            ("critic/explained_var_masked", "masked_"),
        ):
            ev = self._explained_var(prefix)
            if ev is not None:
                out[key] = ev
        n = self.sums.get("all_n", 0.0)
        if n:
            out["critic/mse"] = self.sums["all_e2"] / n
        if not self.labels:
            return out
        labels = torch.tensor(self.labels)
        out["critic/terminal_auc"] = _rank_auc(torch.tensor(self.last_values), labels)
        out["reward"] = float(torch.tensor(self.rewards).mean())
        groups = torch.tensor(self.group_ids)
        buckets = torch.tensor(self.bucket_values, dtype=torch.float64)
        present = torch.tensor(self.bucket_present, dtype=torch.bool)
        n_mixed = 0
        for b in range(_BUCKETS):
            aucs = []
            for gid in torch.unique(groups):
                sel = (groups == gid) & present[:, b]
                if int(sel.sum()) < 2:
                    continue
                auc = _rank_auc(buckets[sel, b], labels[sel])
                if auc == auc:
                    aucs.append(auc)
            if b == 0:
                n_mixed = len(aucs)
            out[f"critic/within_group_auc_q{b + 1}"] = (
                sum(aucs) / len(aucs) if aucs else float("nan")
            )
        out["critic/n_mixed_groups"] = float(n_mixed)
        return out


def _fit_width(t: torch.Tensor, width: int) -> torch.Tensor:
    """Right-pad or cut a ``(B, S)`` tensor to ``width`` columns."""
    if t.shape[1] == width:
        return t
    if t.shape[1] > width:
        return t[:, :width]
    return torch.nn.functional.pad(t, (0, width - t.shape[1]))


# ── driver ──────────────────────────────────────────────────────────────────


class _CriticPretrainer:
    """Holds the critic, the data plane and the privilege cache for the loop."""

    def __init__(
        self,
        master_config: Any,
        cfg: CriticPretrainConfig,
        *,
        value: Any,
        dp_client: Any,
        estimator: GeneralizedAdvantageEstimator,
        loss_fn: Any,
        privilege_store: Any,
        groups_per_step: int,
    ) -> None:
        self.cfg = cfg
        self.groups_per_step = groups_per_step
        self.value = value
        self.dp_client = dp_client
        self.estimator = estimator
        self.loss_fn = loss_fn
        self.privilege_store = privilege_store
        self.overlong_filtering = bool(master_config.ppo.overlong_filtering)
        self.row_multiple = value.sharding_annotations.get_axis_size(
            "data_parallel"
        ) * int(master_config.value["train_micro_batch_size"])
        self._loop = asyncio.new_event_loop()

    def close(self) -> None:
        self._loop.close()

    # -- staging --

    def _privileged_meta(
        self, meta: KVBatchMeta, staged: StagedRows, groups: Sequence[BankGroup]
    ) -> KVBatchMeta:
        """Stamp every row with its group's reference-block prefix tag."""
        if self.privilege_store is None:
            return meta
        tags = list(meta.tags or [])
        for slot, group in enumerate(groups):
            # Pad rows carry their source's prefix, like the controller's.
            rows = [i for i, g in enumerate(staged.group_of_row) if g == slot]
            group_meta = KVBatchMeta(
                partition_id=meta.partition_id,
                task_name=None,
                sample_ids=[meta.sample_ids[i] for i in rows],
                tags=[tags[i] for i in rows],
            )
            enriched = self._loop.run_until_complete(
                self.privilege_store.enrich_from_env_info(
                    group_meta,
                    group.extra_env_info,
                    int(group.prompt_idx) if group.prompt_idx is not None else -1,
                )
            )
            key = enriched.tags[0][PRIVILEGE_KEY_TAG]
            if group.instance_id and key != group.instance_id:
                raise ValueError(
                    f"bank group {group.group_id}: banked instance "
                    f"{group.instance_id!r} but its env info resolves to {key!r}"
                )
            for i, tag in zip(rows, enriched.tags or []):
                tags[i] = tag
        return replace(meta, tags=tags)

    def _put(
        self, groups: Sequence[BankGroup], *, id_prefix: str
    ) -> tuple[KVBatchMeta, StagedRows]:
        staged = pad_rows(stage_rows(groups, id_prefix=id_prefix), self.row_multiple)
        meta = KVBatchMeta(
            partition_id=self.cfg.partition_id,
            task_name=None,
            sample_ids=list(staged.sample_ids),
            fields=list(staged.columns),
            sequence_lengths=list(staged.sequence_lengths),
            extra_info={},
            tags=list(staged.tags),
        )
        meta = self._privileged_meta(meta, staged, groups)
        self.dp_client.put_samples(
            sample_ids=meta.sample_ids,
            partition_id=meta.partition_id,
            fields=TensorDict(
                batch_fields(staged.columns), batch_size=[staged.num_rows]
            ),
            tags=meta.tags,
        )
        return meta, staged

    def _privilege_kwargs(self, meta: KVBatchMeta) -> dict[str, Any]:
        if self.privilege_store is None:
            return {}
        return {"privilege_prefixes": self.privilege_store.prefixes_for(meta)}

    def _score(self, meta: KVBatchMeta, *, train: bool) -> dict[str, torch.Tensor]:
        """Critic forward, then the masks and GAE returns for the rows."""
        self.value.prepare_for_inference()
        self.value.get_values_from_meta(meta, **self._privilege_kwargs(meta))
        self.value.finish_inference()
        data = self.dp_client.get_samples(
            sample_ids=meta.sample_ids,
            partition_id=meta.partition_id,
            select_fields=[
                "token_mask",
                "sample_mask",
                "mask_sample",
                "truncated",
                "total_reward",
                "values",
            ],
        )
        token_mask = tensor_field(data, "token_mask").float()
        values = _fit_width(tensor_field(data, "values").float(), token_mask.shape[1])
        rewards = squeeze_trailing_unit_dim(tensor_field(data, "total_reward")).float()
        flags = {
            name: squeeze_trailing_unit_dim(tensor_field(data, name))
            for name in ("sample_mask", "mask_sample", "truncated")
        }
        sample_mask = loss_sample_mask(
            flags["sample_mask"],
            flags["mask_sample"],
            flags["truncated"],
            apply_env_mask=not (train and self.cfg.train_on_env_masked),
            apply_overlong_mask=self.overlong_filtering
            and not (train and self.cfg.train_on_truncated),
        )
        returns = gae_returns(
            self.estimator,
            token_mask=token_mask,
            sample_mask=sample_mask,
            rewards=rewards,
            values=values,
        )
        return {
            "token_mask": token_mask,
            "values": values,
            "returns": returns,
            "rewards": rewards,
            "sample_mask": sample_mask,
            "mask_sample": flags["mask_sample"].bool(),
            "truncated": flags["truncated"].bool(),
            "finalizer_mask": flags["sample_mask"].float(),
        }

    def _cleanup(self, meta: KVBatchMeta) -> None:
        self.dp_client.clear_samples(
            sample_ids=meta.sample_ids, partition_id=meta.partition_id
        )
        if self.privilege_store is not None:
            self.privilege_store.release([meta])

    # -- step / eval --

    def train_step(self, groups: Sequence[BankGroup]) -> dict[str, float]:
        """One critic optimizer step on ``groups``."""
        meta, staged = self._put(groups, id_prefix="s")
        try:
            scored = self._score(meta, train=True)
            self.dp_client.put_samples(
                sample_ids=meta.sample_ids,
                partition_id=meta.partition_id,
                fields=fields_for_put(
                    meta,
                    {
                        "returns": scored["returns"],
                        "sample_mask": scored["sample_mask"],
                    },
                ),
            )
            self.value.prepare_for_training()
            result = self.value.train_from_meta(
                meta,
                self.loss_fn,
                gbs=staged.num_rows,
                **self._privilege_kwargs(meta),
            )
            self.value.finish_training()
            # Deferred: ppo pulls in the legacy training stack.
            from nemo_rl.algorithms.legacy_ppo_diagnostics import critic_gnorm_metrics
            from nemo_rl.algorithms.ppo import _compute_critic_metrics

            metrics = {
                key: float(np.sum(value)) if isinstance(value, np.ndarray) else value
                for key, value in _compute_critic_metrics(result).items()
            }
            # Pre-clip gradient norm per network part (attention / mamba / moe / mlp /
            # value_head / embedding / other) and its share of the total norm^2, as PPO
            # logs them: critic/gnorm/* and critic/gnorm_frac/*.
            metrics.update(critic_gnorm_metrics(result.get("grad_norm_groups")))
            if self.privilege_store is not None:
                metrics.update(self.privilege_store.step_metrics([meta]))
            real = staged.real_rows()
            canonical = staged.canonical_rows()
            valid = real & (scored["finalizer_mask"] > 0)
            metrics["reward"] = float(scored["rewards"][canonical].mean())
            metrics["num_samples"] = float(real.sum())
            metrics["num_rollouts"] = float(canonical.sum())
            metrics["pad_rows"] = float(staged.num_rows - int(real.sum()))
            metrics["trained_rows"] = float((real & (scored["sample_mask"] > 0)).sum())
            metrics["env_masked_rows"] = float((valid & scored["mask_sample"]).sum())
            metrics["truncated_rows"] = float((valid & scored["truncated"]).sum())
            return metrics
        finally:
            self._cleanup(meta)

    def evaluate(self, groups: Sequence[BankGroup]) -> dict[str, float]:
        """Forward-only held-out metrics, ``groups_per_step`` groups at a time."""
        stats = HeldoutStats()
        for start in range(0, len(groups), self.groups_per_step):
            chunk = groups[start : start + self.groups_per_step]
            meta, staged = self._put(chunk, id_prefix="h")
            try:
                scored = self._score(meta, train=False)
                real = staged.real_rows()
                stats.add(
                    values=scored["values"][real],
                    returns=scored["returns"][real],
                    token_mask=scored["token_mask"][real],
                    sample_mask=scored["sample_mask"][real],
                    rewards=scored["rewards"][real],
                    canonical=staged.canonical_rows()[real],
                    group_of_row=[
                        g
                        for g, pad in zip(staged.group_of_row, staged.is_pad)
                        if not pad
                    ],
                )
            finally:
                self._cleanup(meta)
        return stats.metrics()


def _bank_settings(banks: Sequence[Path], master_config: Any) -> int:
    """Check the banks share one tokenizer (this run's) and return their gpp."""
    tokenizer = dict(master_config.policy["tokenizer"])
    gpps = set()
    for bank in banks:
        meta = read_bank_meta(bank)
        if meta.get("tokenizer") != tokenizer:
            raise ValueError(
                f"rollout bank {bank} was collected with tokenizer "
                f"{meta.get('tokenizer')!r}, this run uses {tokenizer!r}"
            )
        if meta.get("policy_model_name") != master_config.policy["model_name"]:
            print(
                f"  ⚠️ rollout bank {bank} was generated by "
                f"{meta.get('policy_model_name')!r}, not this run's "
                f"policy.model_name {master_config.policy['model_name']!r}",
                flush=True,
            )
        gpps.add(int(meta["num_generations_per_prompt"]))
    if len(gpps) != 1:
        raise ValueError(f"banks mix generations per prompt: {sorted(gpps)}")
    return gpps.pop()


@dataclass
class _Plan:
    stream: list[str]
    heldout: list[str]
    donors: set[str]
    groups_per_step: int
    donors_per_step: int
    gpp: int


def _plan(
    cfg: CriticPretrainConfig,
    refs: Sequence[BankGroupRef],
    *,
    rollouts_per_step: int,
    gpp: int,
    last_checkpoint_path: Optional[str],
) -> tuple[_Plan, dict[str, Any]]:
    """Split, donors and the multi-epoch stream; frozen at step 0, replayed on resume."""
    k = cfg.heldout_responses_per_group
    a, b = step_layout(
        rollouts_per_step, gpp, k, cfg.heldout_response_donor_groups_per_step
    )
    settings = {
        "seed": cfg.seed,
        "heldout_mod": cfg.heldout_mod,
        "heldout_responses_per_group": k,
        "donors_per_step": a,
        "full_per_step": b,
        "gpp": gpp,
    }
    paths = {str(ref.path) for ref in refs}
    if last_checkpoint_path is not None:
        frozen_path = Path(last_checkpoint_path) / FILE_LIST_NAME
        frozen = json.loads(frozen_path.read_text())
        changed = [
            key for key, v in settings.items() if frozen["settings"].get(key) != v
        ]
        if changed:
            raise ValueError(
                f"critic_pretrain settings {changed} differ from those {frozen_path} "
                "was frozen with; restore them or start a new checkpoint_dir."
            )
        missing = [p for p in frozen["base"] + frozen["heldout"] if p not in paths]
        if missing:
            raise ValueError(
                f"{len(missing)} frozen bank groups are gone: {missing[:3]}"
            )
        base, heldout = list(frozen["base"]), list(frozen["heldout"])
        donors = set(frozen["donors"])
        if len(paths) > len(base) + len(heldout):
            print(
                f"ℹ️ {len(paths) - len(base) - len(heldout)} bank groups were "
                "committed after the split was frozen; this run ignores them.",
                flush=True,
            )
    else:
        train_refs, heldout_refs = split_heldout(refs, cfg.heldout_mod)
        if not train_refs:
            raise ValueError("every banked task fell in the held-out split")
        base = [str(ref.path) for ref in train_refs]
        heldout = [str(ref.path) for ref in heldout_refs]
        donors = (
            load_or_create_response_split(
                Path(cfg.bank_dirs[0]),
                base,
                withheld=k,
                donors_per_step=a,
                full_per_step=b,
                heldout_mod=cfg.heldout_mod,
                seed=cfg.seed,
                redraw=cfg.response_split_redraw,
            )
            if k > 0
            else set()
        )
    if k > 0:
        stream = build_packed_epoch_stream(
            sorted(donors),
            sorted(set(base) - donors),
            a,
            b,
            cfg.num_epochs,
            cfg.seed,
        )
    else:
        stream = build_epoch_stream(base, cfg.num_epochs, cfg.seed)
    record = {
        "settings": settings,
        "base": base,
        "heldout": heldout,
        "donors": sorted(donors),
    }
    return (
        _Plan(
            stream=stream,
            heldout=heldout,
            donors=donors,
            groups_per_step=a + b,
            donors_per_step=a,
            gpp=gpp,
        ),
        record,
    )


def _load_step_groups(
    paths: Sequence[str], plan: _Plan, withheld: int
) -> list[BankGroup]:
    """Load a step's groups; donors keep only their first ``gpp - K`` rollouts."""
    groups = []
    for path in paths:
        group = load_bank_group(path)
        if path in plan.donors:
            group = slice_rollouts(group, 0, plan.gpp - withheld)
        groups.append(group)
    return groups


def critic_pretrain(master_config: Any, tokenizer: Any) -> None:
    """Fit the PPO critic on rollout banks; see the module docstring."""
    from nemo_rl.algorithms.loss.loss_functions import MseValueLossFn
    from nemo_rl.algorithms.single_controller_utils.config import (
        _validate_swe_privileged_critic,
        is_ppo_run,
    )
    from nemo_rl.algorithms.single_controller_utils.setup import (
        _build_advantage_estimator,
    )
    from nemo_rl.algorithms.swe_privileged_critic import (
        SwePrivilegePrefixStore,
        resolve_config,
    )
    from nemo_rl.data_plane import build_data_plane_client
    from nemo_rl.distributed.virtual_cluster import (
        RayVirtualCluster,
        prepare_segment_topology,
    )
    from nemo_rl.models.value.tq_value import TQValue
    from nemo_rl.utils.checkpoint import (
        CheckpointManager,
        validate_warm_start_checkpoint,
    )
    from nemo_rl.utils.logger import Logger
    from nemo_rl.utils.timer import TimeoutChecker

    if not is_ppo_run(master_config):
        raise ValueError("critic pretraining needs the `ppo` and `value` blocks")
    cfg = CriticPretrainConfig.model_validate(master_config.critic_pretrain)
    value_config = master_config.value
    if not value_config.get("megatron_cfg", {}).get("enabled"):
        raise ValueError("the TQ critic needs value.megatron_cfg.enabled=true")
    _validate_swe_privileged_critic(master_config)
    estimator = _build_advantage_estimator(master_config)
    if not isinstance(estimator, GeneralizedAdvantageEstimator):
        raise ValueError(
            "critic pretraining fits GAE returns; set ppo.adv_estimator.name=gae"
        )
    if estimator.use_kl_in_reward and estimator.kl_coef > 0:
        raise ValueError(
            "the bank has no policy or reference logprobs, so KL-in-reward returns "
            "cannot be rebuilt; set loss_fn.use_kl_in_reward=false"
        )
    banks = find_banks(cfg.bank_dirs)
    gpp = _bank_settings(banks, master_config)
    refs = list_bank_groups(banks)
    if not refs:
        raise ValueError(f"no committed groups in {cfg.bank_dirs}")
    rollouts_per_step = int(value_config["train_global_batch_size"])
    # Megatron checks this at worker init; fail before the cluster comes up.
    megatron = value_config["megatron_cfg"]
    value_dp = (
        master_config.cluster.num_nodes
        * master_config.cluster.gpus_per_node
        // (
            megatron["tensor_model_parallel_size"]
            * megatron["context_parallel_size"]
            * megatron["pipeline_model_parallel_size"]
        )
    )
    dp_multiple = value_dp * int(value_config["train_micro_batch_size"])
    if rollouts_per_step % dp_multiple:
        raise ValueError(
            f"value.train_global_batch_size={rollouts_per_step} (rollouts per step) "
            f"must be a multiple of the critic's data-parallel size x micro batch "
            f"({value_dp} x {value_config['train_micro_batch_size']}): Megatron "
            "requires it at init."
        )

    checkpointer = CheckpointManager(master_config.checkpointing)
    last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
    save_state = checkpointer.load_training_info(last_checkpoint_path) or {
        "total_steps": 0,
        "consumed_samples": 0,
    }
    step = int(save_state["total_steps"])
    plan, files_record = _plan(
        cfg,
        refs,
        rollouts_per_step=rollouts_per_step,
        gpp=gpp,
        last_checkpoint_path=last_checkpoint_path,
    )
    gps = plan.groups_per_step
    planned_steps = len(plan.stream) // gps
    if cfg.max_steps is not None:
        planned_steps = min(planned_steps, cfg.max_steps)
    if planned_steps <= 0:
        raise ValueError(
            f"{len(plan.stream)} training groups make no step of {gps} groups"
        )
    heldout_eval = (
        plan.heldout
        if cfg.heldout_max_groups is None
        else plan.heldout[: cfg.heldout_max_groups]
    )
    response_eval = sorted(plan.donors)[: cfg.heldout_response_max_groups]
    k = cfg.heldout_responses_per_group
    print(
        f"📚 {len(banks)} bank(s), {len(refs)} groups x {gpp} rollouts: "
        f"{len(files_record['base'])} train ({len(plan.donors)} donors), "
        f"{len(plan.heldout)} held out -> {planned_steps} steps of {gps} groups "
        f"({plan.donors_per_step} donors x {gpp - k} + {gps - plan.donors_per_step} "
        f"x {gpp} = {rollouts_per_step} rollouts), resuming at {step}",
        flush=True,
    )
    if step >= planned_steps:
        print(
            f"✅ Nothing to do: {step}/{planned_steps} steps done. Raise "
            "critic_pretrain.num_epochs to train further.",
            flush=True,
        )
        return
    # One scheduler tick per step.
    value_config["megatron_cfg"]["train_iters"] = planned_steps

    if last_checkpoint_path is not None:
        weights_path, optimizer_path, source = critic_resume_paths(
            checkpointer, last_checkpoint_path, step, cfg.optimizer_state
        )
        print(
            f"♻️ Resuming the critic from {source} "
            f"({'with' if optimizer_path is not None else 'without'} optimizer state)",
            flush=True,
        )
    elif master_config.ppo.warm_start_value_checkpoint is not None:
        warm_start = master_config.ppo.warm_start_value_checkpoint
        validate_warm_start_checkpoint(warm_start)
        weights_path, _ = checkpointer.get_resume_paths(
            warm_start, model_component="value"
        )
        optimizer_path = None
        print(f"🔥 Initializing the critic from {warm_start} (weights only)")
    else:
        weights_path, optimizer_path = None, None

    dp_cfg = master_config.data_plane
    dp_client = build_data_plane_client(dp_cfg, bootstrap=True)
    cluster_cfg = master_config.cluster
    constraints, _, _ = prepare_segment_topology(
        cluster_cfg.segment_size, cluster_cfg.num_nodes, role="training"
    )
    cluster = RayVirtualCluster(
        name="critic_pretrain_cluster",
        bundle_ct_per_node_list=[cluster_cfg.gpus_per_node] * cluster_cfg.num_nodes,
        use_gpus=True,
        num_gpus_per_node=cluster_cfg.gpus_per_node,
        max_colocated_worker_groups=1,
        port_range_low=cluster_cfg.master_port_range_low,
        port_range_high=cluster_cfg.master_port_range_high,
        segment_size=cluster_cfg.segment_size,
        node_resource_constraints=constraints,
    )
    value = TQValue(
        cluster=cluster,
        config=value_config,
        tokenizer=tokenizer,
        name_prefix="lm_value",
        weights_path=weights_path,
        optimizer_path=optimizer_path,
        init_optimizer=True,
        dp_cfg=dp_cfg,
    )
    value.finish_training()
    dp_client.register_partition(
        partition_id=cfg.partition_id,
        fields=[*BANK_FIELDS, "values", "returns"],
        # Schema warm-up only; the TQ adapter reads neither of these.
        num_samples=gps,
        consumer_tasks=["value_fwd", "value_train"],
    )
    privilege_cfg = resolve_config(value_config)
    trainer = _CriticPretrainer(
        master_config,
        cfg,
        value=value,
        dp_client=dp_client,
        estimator=estimator,
        loss_fn=MseValueLossFn(master_config.value_loss_fn),
        privilege_store=(
            SwePrivilegePrefixStore(tokenizer, privilege_cfg)
            if privilege_cfg is not None
            else None
        ),
        groups_per_step=gps,
    )
    logger = Logger(master_config.logger)
    logger.log_hyperparams(master_config.model_dump())
    timeout = TimeoutChecker(
        timeout=master_config.checkpointing["checkpoint_must_save_by"],
        fit_last_save_time=True,
    )
    timeout.start_iterations()

    try:
        while step < planned_steps:
            started = time.perf_counter()
            groups = _load_step_groups(
                plan.stream[step * gps : (step + 1) * gps], plan, k
            )
            rollouts = sum(is_canonical(tag) for g in groups for tag in g.tags)
            if rollouts != rollouts_per_step:
                raise ValueError(
                    f"step {step + 1} holds {rollouts} rollouts, not "
                    f"value.train_global_batch_size={rollouts_per_step}: a bank group "
                    f"has other than {gpp} rollouts"
                )
            metrics = trainer.train_step(groups)
            step += 1
            metrics["total_step_time"] = time.perf_counter() - started
            logger.log_metrics(metrics, step, prefix="train")
            print(
                f"  step {step}/{planned_steps}: loss={metrics['critic/loss']:.5f} "
                f"ev={metrics['critic/explained_var']:.4f} "
                f"reward={metrics['reward']:.3f} rows={int(metrics['num_samples'])} "
                f"({metrics['total_step_time']:.0f}s)",
                flush=True,
            )
            is_last = step == planned_steps
            if is_last or (cfg.eval_period > 0 and step % cfg.eval_period == 0):
                for prefix, paths, start, end in (
                    ("validation", heldout_eval, 0, gpp),
                    ("validation_response", response_eval, gpp - k, gpp),
                ):
                    if not paths:
                        continue
                    eval_groups = [
                        slice_rollouts(load_bank_group(p), start, end)
                        if prefix == "validation_response"
                        else load_bank_group(p)
                        for p in paths
                    ]
                    eval_metrics = trainer.evaluate(eval_groups)
                    logger.log_metrics(eval_metrics, step, prefix=prefix)
                    print(
                        f"  {prefix}: "
                        f"ev={eval_metrics.get('critic/explained_var', float('nan')):.4f} "
                        f"terminal_auc={eval_metrics.get('critic/terminal_auc', float('nan')):.4f} "
                        f"q1={eval_metrics.get('critic/within_group_auc_q1', float('nan')):.4f} "
                        f"q4={eval_metrics.get('critic/within_group_auc_q4', float('nan')):.4f}",
                        flush=True,
                    )

            save_state["total_steps"] = step
            save_state["consumed_samples"] = int(
                save_state.get("consumed_samples", 0)
            ) + int(metrics["num_rollouts"])
            timeout.mark_iteration()
            save_by_step = (
                is_last or step % master_config.checkpointing["save_period"] == 0
            )
            save_by_timeout = timeout.check_save()
            if master_config.checkpointing["enabled"] and (
                save_by_step or save_by_timeout
            ):
                print(f"💾 Saving critic checkpoint for step {step}...", flush=True)
                checkpoint_path = checkpointer.init_tmp_checkpoint(
                    step, save_state, master_config
                )
                rolling = (
                    cfg.optimizer_state == "latest_only" and checkpointer.save_optimizer
                )
                value.prepare_for_training()
                value.save_checkpoint(
                    weights_path=os.path.join(checkpoint_path, "value", "weights"),
                    optimizer_path=os.path.join(checkpoint_path, "value", "optimizer")
                    if checkpointer.save_optimizer and not rolling
                    else None,
                    tokenizer_path=os.path.join(checkpoint_path, "value", "tokenizer"),
                    is_final_checkpoint=is_last,
                )
                if rolling:
                    _, resume_tmp = resume_state_dirs(checkpointer.checkpoint_dir)
                    shutil.rmtree(resume_tmp, ignore_errors=True)
                    value.save_checkpoint(
                        weights_path=str(resume_tmp / "value" / "weights"),
                        optimizer_path=str(resume_tmp / "value" / "optimizer"),
                        tokenizer_path=str(resume_tmp / "value" / "tokenizer"),
                        is_final_checkpoint=is_last,
                    )
                value.finish_training()
                with open(os.path.join(checkpoint_path, FILE_LIST_NAME), "w") as f:
                    json.dump(files_record, f)
                checkpointer.finalize_checkpoint(checkpoint_path)
                if rolling:
                    promote_rolling_resume(checkpointer.checkpoint_dir, step)
            if save_by_timeout:
                print(
                    "⏰ Walltime budget reached; stopping after the save.", flush=True
                )
                break
    finally:
        trainer.close()
        value.shutdown()
    print(f"🏁 Critic pretraining stopped at step {step}/{planned_steps}.", flush=True)
