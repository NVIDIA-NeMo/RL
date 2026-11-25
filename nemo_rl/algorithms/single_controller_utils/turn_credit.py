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
"""Turn credit: redistribute a rollout's GRPO advantage over its turns from turn-judge labels.

``grpo.turn_credit``: a turn judge (turn_judge_coordinator.py; an agentic judge on a separate service) labels the
model calls of a group's rollouts where a decision was made: +1 good, -1 harmful, 0 for every other call. The
advantage stage, right after the estimator (one value ``A`` per rollout), moves advantage between the rollout's
turns::

    j_t        = label of the call that generated token t (0 if unlabeled)
    jbar       = sum_t w_t j_t / sum_t w_t                 (w = trained-token mask; token-weighted mean)
    lambda_eff = min(lambda, kappa / max_t |j_t - jbar|)    (per rollout)
    a_t        = A + lambda_eff * |A| * (j_t - jbar)

- Exactly zero-sum per rollout: ``sum_t w_t (a_t - A) = 0``, so the rollout's total credit (its sequence-level
  GRPO signal) is unchanged; it is only redistributed between turns.
- No sign flip: ``|a_t - A| <= kappa |A| < |A|``. A capped rollout scales all of its shifts together (a per-token
  clip would break the zero-sum).
- ``|A|`` scale: no shaping where the group has no outcome contrast (A = 0); unlabeled rollouts are untouched.

Turn k of a row is its k-th run of generated tokens (``token_mask == 1``), the k-th model call of the trained
chain (one canonical row per rollout: token_capture.segment_rows off, no compaction). The judge counts calls on
the rollout's ``response.output`` (a call starts at item 0 and at every reasoning item). A label on a wrong turn is
worse than no label, so a rollout keeps ``A`` (a mapping drop) unless the mapping is certain:
- the judge service proved its calls are OpenCode's model calls one to one (``aligned``; service call_alignment.py);
- the row's run count equals the judge's call count, or exceeds it by at most the number of empty model calls the
  service proved at the end of the session (``trailing_empty_calls``: a reply with no text or tool left no item in
  ``response.output`` but generated a few tokens), and then only if the row's turns tag agrees and the advantage
  stage found no tool-call token in those extra trailing runs (``trailing_ok``); the extra runs carry no label;
- the row's ``rollout_debug`` turns tag (its model-call count), when present, equals its run count.

Only advantages change (masks untouched). ``apply: false`` (shadow mode) computes everything, logs it, and
returns the advantages unchanged. ``shuffle_within_rollout`` (the control arm) moves each label to another call of the
same rollout, keeping the rollout's label count and mix. ``shuffle_mode: size_matched`` (default) draws the new call
among the rollout's calls of similar size (trained tokens within ``shuffle_size_ratio``, never a really labeled call
when another exists): the judge labels the large calls (median ~4.6k trained tokens vs ~375 for all calls, replay
080), so a uniform draw would move ~6x less advantage than the real arm and confound position with mass.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import torch
from pydantic import BaseModel, Field, model_validator

GOOD = 1
HARMFUL = -1
# An extra trailing run (an empty model reply) longer than this is not accepted as one: the row keeps A.
MAX_TRAILING_RUN_TOKENS = 256


class TurnJudgeClientConfig(BaseModel, extra="forbid"):
    """How the controller reaches the turn-judge service (turn_judge_coordinator.py)."""

    # File with ``export TURN_JUDGE_URL=http://host:port`` written by the judge service at start; re-read
    # whenever a request fails, so a restarted service on another node is picked up.
    endpoint_file: str
    # Agent names (env.nemo_gym agent_ref names) whose groups are judged; None = every agent.
    agents: Optional[list[str]] = None
    # Per-session caps the service applies (user, 2026-11-14): stage 1 (group analysis) once, no retry; each
    # pair session capped (a capped pair is a missing judgment).
    stage1_timeout_s: float = 900.0
    pair_timeout_s: float = 1200.0
    # Total wait from the group's seal; past it the group trains as plain GRPO. stage1 + pair caps + margin
    # for sandbox start and queueing.
    budget_s: float = 2400.0
    poll_interval_s: float = 15.0
    request_timeout_s: float = 60.0
    # Consecutive transport failures that open the breaker, and how long it stays open (groups sealed while
    # it is open are not judged).
    breaker_failures: int = 5
    breaker_cooloff_s: float = 600.0
    # Passed through to the service (judge variant, merge options, seed, ...).
    request_overrides: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> "TurnJudgeClientConfig":
        if self.budget_s < self.stage1_timeout_s + self.pair_timeout_s:
            raise ValueError(
                "turn_credit.judge.budget_s must cover stage1_timeout_s + pair_timeout_s, got "
                f"{self.budget_s} < {self.stage1_timeout_s} + {self.pair_timeout_s}"
            )
        return self


class TurnCreditConfig(BaseModel, extra="forbid"):
    enabled: bool = False
    # False = shadow mode: judge online, labels and would-be advantages logged, advantages unchanged.
    apply: bool = True
    lam: float = 0.5
    kappa: float = 0.8
    # Optional: harmful labels weigh lam_harm / lam of a good label (j = -lam_harm / lam); zero-sum is kept.
    lam_harm: Optional[float] = None
    # Control arm: labels moved to random distinct calls of the same rollout (seeded per rollout).
    shuffle_within_rollout: bool = False
    shuffle_seed: int = 0
    shuffle_mode: Literal["size_matched", "uniform"] = "size_matched"
    shuffle_size_ratio: float = 1.5
    # Which rollouts are shaped, by the sign of their GRPO advantage: all; passing (A > 0: credit moves toward the
    # good decision turns, failing rollouts keep plain GRPO); failing (A < 0 only).
    shape_rollouts: Literal["all", "passing", "failing"] = "all"
    # Shape a rollout only if each of its row's generated-token runs has exactly the token count of the same model
    # call in the export the judge read (the service's step_output_tokens); no counts -> no labels.
    require_token_match: bool = True
    judge: Optional[TurnJudgeClientConfig] = None
    # Restart state is saved in every full checkpoint. Optional: an append-only event log (one file per Slurm job)
    # in this directory, which also receives the side log (per labeled rollout: A, lambda_eff, labeled calls);
    # None = off (no writes outside the checkpoint).
    state_dir: Optional[str] = None
    side_log: bool = True

    @model_validator(mode="after")
    def _check(self) -> "TurnCreditConfig":
        if not self.lam > 0:
            raise ValueError(f"turn_credit.lam must be > 0, got {self.lam}")
        if not 0 < self.kappa < 1:
            raise ValueError(
                f"turn_credit.kappa must be in (0, 1) (it bounds |a_t - A| / |A|), got {self.kappa}"
            )
        if not self.shuffle_size_ratio >= 1:
            raise ValueError(
                f"turn_credit.shuffle_size_ratio must be >= 1, got {self.shuffle_size_ratio}"
            )
        if self.lam_harm is not None and not self.lam_harm > 0:
            raise ValueError(f"turn_credit.lam_harm must be > 0, got {self.lam_harm}")
        if self.enabled and self.judge is None:
            raise ValueError(
                "turn_credit.enabled needs turn_credit.judge (the judge service)"
            )
        return self


def turn_credit_config(algo_cfg: object) -> Optional[TurnCreditConfig]:
    """The algorithm block's turn_credit when enabled, else None (dicts from ``++`` overrides are parsed)."""
    raw = getattr(algo_cfg, "turn_credit", None)
    if raw is None:
        return None
    cfg = (
        raw
        if isinstance(raw, TurnCreditConfig)
        else TurnCreditConfig.model_validate(dict(raw))
    )
    return cfg if cfg.enabled else None


@dataclass(frozen=True)
class RolloutTurnLabels:
    """The judge's labels for one rollout, as the advantage stage receives them (metadata only)."""

    group_id: str
    generation_index: int
    # Model calls the judge counted on the rollout's response.output.
    n_calls: int
    # (call ordinal, GOOD | HARMFUL), at most one per call after the coordinator's merge.
    labels: tuple[tuple[int, int], ...]
    # The service's proof that calls 0..n_calls-1 are OpenCode's model calls 0..n_calls-1 (None: an older service
    # that did not check), and the number of empty model calls that end the session (only when aligned).
    aligned: Optional[bool] = None
    trailing_empty_calls: int = 0
    # Generated tokens of each OpenCode model call in the export the judge read (all steps, trailing empty ones
    # included; an entry is None for a final step that never finished); None from a service without it. Checked
    # against the row's runs (tokens_match, TurnCreditConfig.require_token_match).
    step_tokens: Optional[tuple[Optional[int], ...]] = None


MIN_MATCHED_RUNS = 2


def tokens_match(
    run_lengths: list[int], step_tokens: tuple[Optional[int], ...]
) -> bool:
    """Whether a row's generated-token runs are the export's model calls, run by run.

    Run k must have exactly step k's token count. A None count (a final step that never finished, so its usage was
    never recorded) may only be the export's last step and is not compared; it is not a tolerance on a recorded
    count. Steps past the row's runs must have produced nothing (0 or None), and at least ``MIN_MATCHED_RUNS`` runs
    must match exactly. Validated on 1,140 labeled rows (LOG 095): all healthy rows pass; every known mix-up fails at
    its first run, and 0 of 23,912 same-group cross pairs pass.
    """
    if len(run_lengths) > len(step_tokens) or any(t is None for t in step_tokens[:-1]):
        return False
    matched = 0
    for length, tokens in zip(run_lengths, step_tokens):
        if tokens is None:
            continue
        if length != tokens:
            return False
        matched += 1
    if any(t not in (0, None) for t in step_tokens[len(run_lengths) :]):
        return False
    return matched >= MIN_MATCHED_RUNS


def call_of(call_starts: list[int], item_index: int) -> int:
    """The call ordinal that owns response.output item ``item_index`` (calls start at ``call_starts``)."""
    k = -1
    for k_, start in enumerate(call_starts):
        if start > item_index:
            break
        k = k_
    if k < 0:
        raise ValueError(
            f"item {item_index} precedes the first call start {call_starts[:1]}"
        )
    return k


def call_starts_of(output: list[Any]) -> list[int]:
    """Indices of the items that start a model call: item 0 and every reasoning item (the judge's rule)."""
    return [
        i
        for i, item in enumerate(output)
        if i == 0 or (isinstance(item, dict) and item.get("type") == "reasoning")
    ]


def token_runs(token_mask_row: torch.Tensor) -> list[tuple[int, int]]:
    """``[(start, end)]`` of every run of generated tokens (``token_mask > 0``) in one row, in order."""
    g = (token_mask_row.detach().reshape(-1) > 0).to(torch.int8).cpu()
    edges = torch.diff(torch.nn.functional.pad(g, (1, 1)))
    starts = torch.nonzero(edges == 1, as_tuple=False).reshape(-1).tolist()
    ends = torch.nonzero(edges == -1, as_tuple=False).reshape(-1).tolist()
    return list(zip(starts, ends))


def _run_diagnostics(
    record: dict[str, Any], runs: list[tuple[int, int]], trained_tokens: int
) -> None:
    """Where a dropped row's runs split or merge: run lengths at both ends, tiny runs (driver-log diagnostics)."""
    lengths = [e - s for s, e in runs]
    record["run_lengths_head"] = lengths[:4]
    record["run_lengths_tail"] = lengths[-4:]
    record["runs_le2_tokens"] = sum(1 for n in lengths if n <= 2)
    record["trained_tokens"] = trained_tokens


def trailing_extra_runs(
    lab: RolloutTurnLabels, n_runs: int, turns: Optional[int], content_ok: bool
) -> bool:
    """Whether a row with more runs than the judge counted may still be shaped, its extra runs being trailing empty
    model calls: the service proved the alignment and k trailing empty calls, the row has 1..k extra runs, its turns
    tag says the same, and the advantage stage checked the extra runs' tokens (``content_ok``)."""
    extra = n_runs - lab.n_calls
    return bool(
        lab.aligned is True
        and 0 < extra <= lab.trailing_empty_calls
        and turns is not None
        and turns == n_runs
        and content_ok
    )


def shuffled_labels(
    labels: tuple[tuple[int, int], ...],
    n_calls: int,
    *,
    seed: int,
    group_id: str,
    generation_index: int,
    call_tokens: Optional[list[float]] = None,
    size_ratio: float = 1.5,
) -> tuple[tuple[int, int], ...]:
    """The same label values on other distinct calls of the rollout (deterministic per rollout).

    Without ``call_tokens``: calls drawn uniformly without replacement. With them (size-matched): each label, largest
    call first, moves to a call whose size is within ``size_ratio`` of its own, drawn uniformly among those; calls
    that carry a real label are excluded while any other call is free; with no call in range, the nearest in log size.
    """
    digest = hashlib.sha256(f"{seed}:{group_id}:{generation_index}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "little"))
    if call_tokens is None:
        calls = rng.sample(range(n_calls), len(labels))
        return tuple(sorted((k, value) for k, (_, value) in zip(calls, labels)))
    real = {k for k, _ in labels}
    taken: set[int] = set()
    out = []
    counts = call_tokens
    for k, value in sorted(labels, key=lambda kv: (-counts[kv[0]], kv[0])):
        free = [c for c in range(n_calls) if c not in taken]
        if len(free) > len(
            real - taken
        ):  # keep real positions out while other calls remain
            free = [c for c in free if c not in real] or free
        n = call_tokens[k]
        in_range = [
            c for c in free if n / size_ratio <= call_tokens[c] <= n * size_ratio
        ]
        if not in_range:
            gap = {c: abs(math.log1p(call_tokens[c]) - math.log1p(n)) for c in free}
            best = min(gap.values())
            in_range = [c for c in free if gap[c] == best]
        c = rng.choice(in_range)
        taken.add(c)
        out.append((c, value))
    return tuple(sorted(out))


@dataclass
class TurnCreditStats:
    """Additive per-step sums; ``to_metrics`` turns them into shares and means."""

    applied: bool = True
    trained_rows: int = 0
    labeled_rows: int = 0  # trained rows that arrived with labels
    shaped_rows: int = 0
    drops: dict[str, int] = field(default_factory=dict)
    noops: dict[str, int] = field(default_factory=dict)
    good_calls: int = 0
    harmful_calls: int = 0
    trained_tokens: float = 0.0
    shaped_row_tokens: float = 0.0
    good_tokens: float = 0.0
    harmful_tokens: float = 0.0
    lambda_eff_sum: float = 0.0
    lambda_capped_rows: int = 0
    abs_adv_mass: float = 0.0  # sum over shaped rows' trained tokens of |A|
    moved_mass: float = 0.0  # ... of |a_t - A|
    max_shift_over_abs_adv: float = 0.0
    zero_sum_max_err: float = (
        0.0  # max over rows of |sum_t w_t (a_t - A)| / (|A| sum_t w_t), after the cast
    )
    sign_flips: int = 0
    trailing_extra_accepted: int = (
        0  # rows shaped with extra trailing empty-call runs (proven; unlabeled)
    )
    # Token sums / counts of the shaped advantages by (label class, sign of A): good|harmful|unlabeled x pos|neg.
    adv_sum: dict[str, float] = field(default_factory=dict)
    adv_count: dict[str, float] = field(default_factory=dict)

    def merge(self, other: "TurnCreditStats") -> None:
        self.applied = self.applied and other.applied
        for name in (
            "trained_rows",
            "labeled_rows",
            "shaped_rows",
            "good_calls",
            "harmful_calls",
            "trained_tokens",
            "shaped_row_tokens",
            "good_tokens",
            "harmful_tokens",
            "lambda_eff_sum",
            "lambda_capped_rows",
            "abs_adv_mass",
            "moved_mass",
            "sign_flips",
            "trailing_extra_accepted",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.max_shift_over_abs_adv = max(
            self.max_shift_over_abs_adv, other.max_shift_over_abs_adv
        )
        self.zero_sum_max_err = max(self.zero_sum_max_err, other.zero_sum_max_err)
        for mine, theirs in (
            (self.drops, other.drops),
            (self.noops, other.noops),
        ):
            for key, value in theirs.items():
                mine[key] = mine.get(key, 0) + value
        for key, value in other.adv_sum.items():
            self.adv_sum[key] = self.adv_sum.get(key, 0.0) + value
        for key, count in other.adv_count.items():
            self.adv_count[key] = self.adv_count.get(key, 0.0) + count

    def to_metrics(self) -> dict[str, float]:
        if not self.trained_rows:
            return {}
        m = {
            "turn_credit/applied": float(self.applied),
            "turn_credit/labeled_row_share": self.labeled_rows / self.trained_rows,
            "turn_credit/shaped_row_share": self.shaped_rows / self.trained_rows,
            "turn_credit/shaped_rows": float(self.shaped_rows),
            "turn_credit/good_calls": float(self.good_calls),
            "turn_credit/harmful_calls": float(self.harmful_calls),
            "turn_credit/labeled_token_share": (
                (self.good_tokens + self.harmful_tokens) / self.trained_tokens
                if self.trained_tokens
                else 0.0
            ),
            "turn_credit/labeled_token_share_in_shaped_rows": (
                (self.good_tokens + self.harmful_tokens) / self.shaped_row_tokens
                if self.shaped_row_tokens
                else 0.0
            ),
            "turn_credit/lambda_eff_mean": (
                self.lambda_eff_sum / self.shaped_rows if self.shaped_rows else 0.0
            ),
            "turn_credit/lambda_capped_share": (
                self.lambda_capped_rows / self.shaped_rows if self.shaped_rows else 0.0
            ),
            "turn_credit/mass_moved_share": (
                self.moved_mass / self.abs_adv_mass if self.abs_adv_mass else 0.0
            ),
            "turn_credit/max_shift_over_abs_adv": self.max_shift_over_abs_adv,
            "turn_credit/zero_sum_max_err": self.zero_sum_max_err,
            "turn_credit/sign_flips": float(self.sign_flips),
            "turn_credit/trailing_extra_accepted": float(self.trailing_extra_accepted),
        }
        for key, value in self.drops.items():
            m[f"turn_credit/drop/{key}"] = float(value)
        for key, value in self.noops.items():
            m[f"turn_credit/noop/{key}"] = float(value)
        for key, count in self.adv_count.items():
            if count:
                m[f"turn_credit/adv_mean/{key}"] = self.adv_sum.get(key, 0.0) / count
        return m

    def _drop(self, reason: str) -> None:
        self.drops[reason] = self.drops.get(reason, 0) + 1

    def _noop(self, reason: str) -> None:
        self.noops[reason] = self.noops.get(reason, 0) + 1

    def _add_class(self, key: str, values: torch.Tensor) -> None:
        if values.numel():
            self.adv_sum[key] = self.adv_sum.get(key, 0.0) + float(values.sum())
            self.adv_count[key] = self.adv_count.get(key, 0.0) + float(values.numel())


def shape_turn_credit(
    advantages: torch.Tensor,
    *,
    train_mask: torch.Tensor,
    token_mask: torch.Tensor,
    row_labels: dict[int, RolloutTurnLabels],
    config: TurnCreditConfig,
    sample_ids: Optional[list[str]] = None,
    row_turns: Optional[dict[int, Optional[int]]] = None,
    trailing_ok: Optional[dict[int, bool]] = None,
) -> tuple[torch.Tensor, TurnCreditStats, list[dict[str, Any]]]:
    """Shape the rows in ``row_labels`` (see the module docstring).

    Args:
        advantages: ``(B, T)`` the estimator's advantages, one value per row on its trained tokens (may be a
            stride-0 expanded view; it is never written to).
        train_mask: ``(B, T)`` > 0 on trained tokens (``token_mask * final_sample_mask``).
        token_mask: ``(B, T)`` the generated-token mask (turn k = the k-th run of ones).
        row_labels: row index -> that rollout's labels.
        config: the turn_credit block.
        sample_ids: row sample ids, for the side-log records.
        row_turns: row index -> its ``rollout_debug`` turns tag (model calls of the trained chain), when known.
        row_turns / trailing_ok: see ``trailing_extra_runs``; a row absent from ``trailing_ok`` gets no extra runs.

    Returns:
        The advantages to write back (a new tensor when any row was shaped and ``config.apply``; the input
        object otherwise), the step statistics, and one side-log record per labeled trained row.
    """
    if train_mask.shape != advantages.shape or token_mask.shape != advantages.shape:
        raise ValueError(
            "turn_credit needs advantages, train_mask and token_mask of one shape, got "
            f"{tuple(advantages.shape)}, {tuple(train_mask.shape)}, {tuple(token_mask.shape)}"
        )
    stats = TurnCreditStats(applied=config.apply)
    trained = train_mask.detach().cpu() > 0
    trained_counts = trained.sum(dim=-1)
    stats.trained_rows = int((trained_counts > 0).sum())
    stats.trained_tokens = float(trained_counts.sum())
    harmful_value = (
        -(config.lam_harm / config.lam)
        if config.lam_harm is not None
        else float(HARMFUL)
    )
    adv_cpu = advantages.detach().cpu()
    shaped: dict[int, torch.Tensor] = {}
    records: list[dict[str, Any]] = []
    for r, lab in sorted(row_labels.items()):
        w = trained[r]
        n = int(w.sum())
        if n == 0:
            stats._noop("untrained_row")
            continue
        stats.labeled_rows += 1
        record: dict[str, Any] = {
            "sample_id": sample_ids[r] if sample_ids is not None else None,
            "group_id": lab.group_id,
            "generation_index": lab.generation_index,
            "n_calls": lab.n_calls,
            "labels": [list(x) for x in lab.labels],
        }
        records.append(record)
        runs = token_runs(token_mask[r])
        record["n_runs"] = len(runs)
        turns = (row_turns or {}).get(r)
        record["turns_tag"] = turns
        if lab.aligned is False:
            # The coordinator does not pass unproven rollouts; a guard in case one arrives.
            stats._drop("alignment_not_proven")
            record["status"] = "drop:alignment_not_proven"
            continue
        if turns is not None and turns != len(runs):
            stats._drop("turns_tag_mismatch")
            record["status"] = "drop:turns_tag_mismatch"
            _run_diagnostics(record, runs, n)
            continue
        extra = len(runs) - lab.n_calls
        trailing_accepted = False
        if extra != 0 and trailing_extra_runs(
            lab, len(runs), turns, (trailing_ok or {}).get(r, False)
        ):
            trailing_accepted = True
            record["trailing_extra_runs"] = extra
            _run_diagnostics(record, runs, n)
        elif extra != 0:
            stats._drop("n_runs_mismatch")
            record["status"] = "drop:n_runs_mismatch"
            _run_diagnostics(record, runs, n)
            continue
        if config.require_token_match:
            # Identity: the row's generated-token runs are the OpenCode model calls of the export the judge read.
            # Equal call counts alone are not proof: a Gym row re-dispatched under the same rollout id can train
            # another session's tokens (LOG 094).
            if lab.step_tokens is None:
                stats._drop("no_token_fingerprint")
                record["status"] = "drop:no_token_fingerprint"
                continue
            if not tokens_match([e - s for s, e in runs], lab.step_tokens):
                stats._drop("token_mismatch")
                record["status"] = "drop:token_mismatch"
                record["step_tokens_head"] = list(lab.step_tokens[:4])
                _run_diagnostics(record, runs, n)
                continue
        if not lab.labels:
            stats._noop("no_labels")
            record["status"] = "noop:no_labels"
            continue
        if any(not 0 <= k < lab.n_calls for k, _ in lab.labels) or any(
            value not in (GOOD, HARMFUL) for _, value in lab.labels
        ):
            stats._drop("bad_label")
            record["status"] = "drop:bad_label"
            continue
        if len({k for k, _ in lab.labels}) != len(lab.labels):
            stats._drop("duplicate_call_label")
            record["status"] = "drop:duplicate_call_label"
            continue
        row = adv_cpu[r].double()
        trained_values = row[w]
        a = float(trained_values[0])
        if float((trained_values - a).abs().max()) > 1e-6 * max(1.0, abs(a)):
            stats._drop("non_constant_advantage")
            record["status"] = "drop:non_constant_advantage"
            continue
        record["A"] = a
        if a == 0.0:
            stats._noop("zero_advantage")
            record["status"] = "noop:zero_advantage"
            continue
        if (config.shape_rollouts == "passing" and a < 0) or (
            config.shape_rollouts == "failing" and a > 0
        ):
            stats._noop(f"not_{config.shape_rollouts}")
            record["status"] = f"noop:not_{config.shape_rollouts}"
            continue
        applied_labels = lab.labels
        if config.shuffle_within_rollout:
            applied_labels = shuffled_labels(
                lab.labels,
                lab.n_calls,
                seed=config.shuffle_seed,
                group_id=lab.group_id,
                generation_index=lab.generation_index,
                call_tokens=(
                    [float(w[s:e].sum()) for s, e in runs]
                    if config.shuffle_mode == "size_matched"
                    else None
                ),
                size_ratio=config.shuffle_size_ratio,
            )
            record["applied_labels"] = [list(x) for x in applied_labels]
        wd = w.double()
        j = torch.zeros_like(wd)
        good_tokens = harmful_tokens = 0.0
        for k, value in applied_labels:
            s, e = runs[k]
            jk = 1.0 if value == GOOD else harmful_value
            j[s:e] = jk
            n_t = float(wd[s:e].sum())
            if value == GOOD:
                stats.good_calls += 1
                good_tokens += n_t
            else:
                stats.harmful_calls += 1
                harmful_tokens += n_t
        jbar = float((wd * j).sum()) / n
        dev = (j - jbar) * wd
        m = float(dev.abs().max())
        if m == 0.0:
            stats._noop("uniform_labels")
            record["status"] = "noop:uniform_labels"
            continue
        lam_eff = min(config.lam, config.kappa / m)
        shift = lam_eff * abs(a) * dev
        new_row = torch.where(w, row + shift, row).to(advantages.dtype)
        # Invariants on the values the loss will see (after the cast back).
        moved = new_row.double()[w] - a
        zero_sum_err = abs(float(moved.sum())) / (abs(a) * n)
        flips = int((torch.sign(new_row.double()[w]) != math.copysign(1.0, a)).sum())
        stats.shaped_rows += 1
        stats.shaped_row_tokens += n
        stats.good_tokens += good_tokens
        stats.harmful_tokens += harmful_tokens
        stats.lambda_eff_sum += lam_eff
        stats.lambda_capped_rows += int(lam_eff < config.lam)
        stats.abs_adv_mass += abs(a) * n
        stats.moved_mass += float(moved.abs().sum())
        stats.max_shift_over_abs_adv = max(
            stats.max_shift_over_abs_adv, float(moved.abs().max()) / abs(a)
        )
        stats.zero_sum_max_err = max(stats.zero_sum_max_err, zero_sum_err)
        stats.sign_flips += flips
        sign = "pos" if a > 0 else "neg"
        values = new_row.double()
        stats._add_class(f"good_{sign}", values[w & (j > 0)])
        stats._add_class(f"harmful_{sign}", values[w & (j < 0)])
        stats._add_class(f"unlabeled_{sign}", values[w & (j == 0)])
        shaped[r] = new_row
        stats.trailing_extra_accepted += int(trailing_accepted)
        record.update(
            status="shaped",
            lambda_eff=lam_eff,
            jbar=jbar,
            calls=[
                [
                    k,
                    value,
                    float(wd[runs[k][0] : runs[k][1]].sum()),
                    float(new_row[runs[k][0]]),
                ]
                for k, value in applied_labels
            ],
            other_adv=float(a + lam_eff * abs(a) * (0.0 - jbar)),
        )
    if not shaped or not config.apply:
        return advantages, stats, records
    out = advantages.detach().clone(memory_format=torch.contiguous_format)
    for r, new_row in shaped.items():
        out[r] = new_row.to(device=out.device)
    return out, stats, records
