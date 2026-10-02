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
"""Legacy async-PPO step diagnostics for the SingleController PPO path.

One :class:`LegacyPPODiagnostics` per controller. The controller calls it at
three points of a PPO step (one chunk per step):

1. :meth:`on_advantage_stage` at the end of ``_advantage_stage``, with the
   tensors that stage already fetched or computed. Everything that legacy
   derived from ``train_data`` / ``repeated_batch`` is computed here: critic
   EV/positional/mixed-group metrics, ``adv_raw/*``, ``residual/*``,
   multi-trace and log-prob-error breakdowns, trajectory ages, and the
   ``rollout_debug`` rows.
2. :meth:`step_metrics` while the step's metrics are assembled, with the critic
   train result (and, when enabled, the post-update critic pass).
3. :meth:`write_rollout_debug` once the step is closed, which writes
   ``rollout_debug_step{N}.jsonl`` to the logger's log dir.

Per-row provenance (rollout_info / trace_metadata / rollout and trace indices)
rides the TransferQueue row tags as one JSON string written at commit time
(:func:`nemo_rl.experience.legacy_rollout_metrics.rollout_debug_tags`), so it
follows the rows through selection, padding and the critic stage.

Pad rows (``_pad_rows_to_dp_multiple``: ``{sample_id}_pad{i}``, sample_mask 0)
are treated exactly like legacy's DP padding rows: excluded from every per-trace
statistic and from the jsonl, included in the critic metrics that legacy scored
over the padded ``train_data``.
"""

from __future__ import annotations

import functools
import inspect
import os
import re
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Optional, TypeVar, overload

import numpy as np
import torch

from nemo_rl.algorithms.legacy_ppo_diagnostics import (
    critic_gnorm_metrics,
    mixed_group_value_metrics,
    multi_trace_composition_metrics,
    pooled_explained_var,
    positional_value_metrics,
    residual_baseline_diagnostics,
    return_space_stats,
    trajectory_age_metrics,
)
from nemo_rl.algorithms.multi_trace_metrics import compute_multi_trace_diagnostics
from nemo_rl.experience.legacy_rollout_metrics import iter_rollout_debug_rows

_PAD_ROW_SUFFIX = re.compile(r"_pad\d+$")

_F = TypeVar("_F", bound=Callable[..., Any])


def _diagnostic(default: Callable[[], Any]) -> Callable[[_F], _F]:
    """Diagnostics must never end a training run.

    Outside ``strict`` mode (unit tests set it), an exception is printed with its
    traceback and the method returns ``default()`` instead of propagating.
    """

    def wrap(method: _F) -> _F:
        @functools.wraps(method)
        def inner(self: "LegacyPPODiagnostics", *args: Any, **kwargs: Any) -> Any:
            try:
                return method(self, *args, **kwargs)
            except Exception:
                if self.strict:
                    raise
                print(
                    f"WARNING: legacy diagnostics {method.__name__} failed; "
                    "skipping this step's legacy metrics:\n" + traceback.format_exc(),
                    flush=True,
                )
                return default()

        return inner  # type: ignore[return-value]

    return wrap


def is_pad_row(sample_id: str) -> bool:
    """Whether ``sample_id`` names a multi-trace DP pad row."""
    return bool(_PAD_ROW_SUFFIX.search(sample_id))


def group_key(sample_id: str) -> str:
    """Prompt-group id of a row: ``{group_id}_g{i}`` (pad rows: of their source row)."""
    base = _PAD_ROW_SUFFIX.sub("", sample_id)
    return base.rsplit("_g", 1)[0] if "_g" in base else base


class LegacyPPODiagnostics:
    """Stateful per-step collector of the legacy async-PPO diagnostics."""

    #: Re-raise instead of logging and skipping (unit tests).
    strict: bool = False

    def __init__(
        self,
        *,
        num_generations_per_prompt: int,
        policy_training_start_step: int,
        log_rollout_debug: bool,
        log_critic_diagnostics: bool,
        log_post_update_critic_metrics: bool,
        log_rollout_dump: bool = False,
        rollout_dump_period: int = 1,
    ) -> None:
        if rollout_dump_period < 1:
            raise ValueError(
                f"ppo.rollout_dump_period must be >= 1, got {rollout_dump_period}."
            )
        self.log_rollout_dump = bool(log_rollout_dump)
        self.rollout_dump_period = rollout_dump_period
        self._dump: Optional[dict[str, Any]] = None
        self._tokenizer: Any = None
        self.num_generations_per_prompt = int(num_generations_per_prompt)
        self.policy_training_start_step = int(policy_training_start_step)
        self.log_rollout_debug = bool(log_rollout_debug)
        self.log_critic_diagnostics = bool(log_critic_diagnostics)
        self.post_update_enabled = bool(log_post_update_critic_metrics)
        #: Set by the controller after the last critic update when
        #: ``post_update_enabled``: the forward-only (eval_mode) critic pass.
        self.post_value_result: Optional[dict[str, Any]] = None
        self._metrics: dict[str, Any] = {}
        self._rollout_debug: Optional[dict[str, list[Any]]] = None
        self._critic_overrides: dict[str, float] = {}

    @classmethod
    def from_algo_config(cls, algo_cfg: Any) -> "LegacyPPODiagnostics":
        return cls(
            num_generations_per_prompt=algo_cfg.num_generations_per_prompt,
            policy_training_start_step=algo_cfg.policy_training_start_step,
            log_rollout_debug=algo_cfg.log_rollout_debug,
            log_critic_diagnostics=algo_cfg.log_critic_diagnostics,
            log_post_update_critic_metrics=algo_cfg.log_post_update_critic_metrics,
            log_rollout_dump=algo_cfg.log_rollout_dump,
            rollout_dump_period=algo_cfg.rollout_dump_period,
        )

    def rollout_dump_due(self, step: int) -> bool:
        """Whether 1-based ``step`` writes ``ppo_rollout_dump_step{step}.pt``."""
        return self.log_rollout_dump and step % self.rollout_dump_period == 0

    # ------------------------------------------------------------------ stage 1
    @_diagnostic(lambda: None)
    def on_advantage_stage(
        self,
        *,
        sample_ids: Sequence[str],
        tags: Optional[Sequence[Mapping[str, Any]]],
        sequence_lengths: Optional[Sequence[int]],
        trainer_version: int,
        rewards: torch.Tensor,
        token_mask: torch.Tensor,
        mask_sample: torch.Tensor,
        final_sample_mask: torch.Tensor,
        seq_error_tensors: Mapping[str, torch.Tensor],
        generation_logprobs: Optional[torch.Tensor],
        policy_logprobs: Optional[torch.Tensor],
        values: Optional[torch.Tensor],
        advantages: torch.Tensor,
        returns: Optional[torch.Tensor],
        estimator_mask: torch.Tensor,
        advantage_estimator: Any,
        input_ids: Optional[torch.Tensor] = None,
        reference_logprobs: Optional[torch.Tensor] = None,
        truncated: Optional[torch.Tensor] = None,
    ) -> None:
        """Compute this step's advantage-stage diagnostics (see module doc)."""
        # A failure part-way must not leak a previous step's numbers.
        self._metrics, self._rollout_debug, self._critic_overrides = {}, None, {}
        self._dump = None
        num_rows = len(sample_ids)
        tags = list(tags) if tags is not None else [{} for _ in range(num_rows)]
        real_rows = [i for i, sid in enumerate(sample_ids) if not is_pad_row(sid)]
        n = len(real_rows)
        real_lengths = (
            [int(sequence_lengths[i]) for i in real_rows]
            if sequence_lengths is not None
            else None
        )
        # SC appends pad rows after the real ones; keep views when it does.
        real_index = (
            None
            if real_rows == list(range(n))
            else torch.tensor(real_rows, dtype=torch.long)
        )

        @overload
        def _real(t: torch.Tensor) -> torch.Tensor: ...
        @overload
        def _real(t: None) -> None: ...
        def _real(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if t is None:
                return None
            return t[:n] if real_index is None else t[real_index]

        debug_rows = iter_rollout_debug_rows(tags[i] for i in real_rows)
        group_order: dict[str, int] = {}
        rows_seen_in_group: dict[int, int] = {}
        trace_group_ids: list[int] = []
        trace_rollout_ids: list[int] = []
        for i, row in zip(real_rows, debug_rows):
            gid = group_order.setdefault(group_key(sample_ids[i]), len(group_order))
            position = rows_seen_in_group.get(gid, 0)
            rows_seen_in_group[gid] = position + 1
            trace_group_ids.append(gid)
            # Rows without provenance (non-Gym) are one rollout each.
            local = int(row.get("rollout_local_idx", position))
            row.setdefault("rollout_local_idx", local)
            trace_rollout_ids.append(gid * self.num_generations_per_prompt + local)
        trace_in_rollout_idx = [
            int(r.get("trace_in_rollout_idx", 0)) for r in debug_rows
        ]
        is_empty_rollout = [bool(r.get("is_empty_rollout", False)) for r in debug_rows]
        trace_kinds = [
            (r.get("trace_metadata") or {}).get("kind", "unknown") for r in debug_rows
        ]

        pre_gate = seq_error_tensors.get("pre_seq_error_sample_loss_mask")
        if pre_gate is None:
            pre_gate = final_sample_mask.detach().clone()
        seq_err = seq_error_tensors.get("seq_mult_prob_error")
        if seq_err is None:
            seq_err = torch.zeros_like(final_sample_mask, dtype=torch.float32)
        masked_by_gate = seq_error_tensors.get("masked_by_seq_logprob_error")
        if masked_by_gate is None:
            masked_by_gate = torch.zeros_like(final_sample_mask, dtype=torch.bool)

        metrics: dict[str, Any] = {}

        # ---- multi-trace + per-trace log-prob-error breakdowns (legacy: every
        # Gym step; buckets are per trace index / segment kind).
        if n > 0:
            if generation_logprobs is not None and policy_logprobs is not None:
                metrics.update(
                    compute_multi_trace_diagnostics(
                        seq_mult_prob_error=_real(seq_err),
                        masked_by_seq_logprob_error=_real(masked_by_gate),
                        pre_seq_error_sample_loss_mask=_real(pre_gate),
                        mask_sample=_real(mask_sample),
                        is_empty_rollout=is_empty_rollout,
                        trace_in_rollout_idx=trace_in_rollout_idx,
                        trace_kinds=trace_kinds,
                        token_mask=_real(token_mask),
                        sample_mask=_real(final_sample_mask),
                        generation_logprobs=_real(generation_logprobs),
                        prev_logprobs=_real(policy_logprobs),
                        num_unpadded_traces=n,
                    )
                )
            lengths = (
                torch.tensor(real_lengths, dtype=torch.long)
                if real_lengths is not None
                else _real(token_mask).new_full((n,), token_mask.shape[1])
            )
            metrics.update(
                multi_trace_composition_metrics(
                    trace_rollout_ids=trace_rollout_ids,
                    sample_mask=_real(final_sample_mask),
                    trace_lengths=lengths,
                    num_rows=num_rows,
                )
            )
            # Legacy reported these over the unpadded traces (SC's
            # total_num_tokens also counts the DP pad rows).
            if real_lengths is not None:
                metrics["total_num_tokens"] = float(lengths.sum().item())
            prompt_tokens = [
                (r.get("trace_metadata") or {}).get("prompt_tokens") for r in debug_rows
            ]
            if all(p is not None for p in prompt_tokens):
                metrics["mean_prompt_length"] = float(np.mean(prompt_tokens))
            # Legacy `reward`: one value per ROLLOUT (its first trace), every
            # unpadded rollout including env-masked ones. SC's own `reward`
            # averages trained rows (sample_mask > 0, every trace).
            first_trace = torch.tensor(
                [t == 0 for t in trace_in_rollout_idx], dtype=torch.bool
            )
            if bool(first_trace.any()):
                metrics["reward"] = float(_real(rewards).float()[first_trace].mean())
            # Legacy `advantages/*`: over every unpadded row's response tokens.
            response_advantages = _real(advantages)[_real(token_mask).bool()]
            if response_advantages.numel() > 0:
                metrics["advantages/mean"] = float(response_advantages.mean())
                metrics["advantages/max"] = float(response_advantages.max())
                metrics["advantages/min"] = float(response_advantages.min())

            # ---- trajectory ages: one generation version per prompt group.
            gen_versions: dict[str, int] = {}
            for i in real_rows:
                version = (tags[i] or {}).get("weight_version")
                if version is not None:
                    gen_versions.setdefault(group_key(sample_ids[i]), int(version))
            metrics.update(
                trajectory_age_metrics(
                    list(gen_versions.values()),
                    trainer_version,
                    self.policy_training_start_step,
                )
            )

        # ---- adv_raw/* (pre-whitening scale, captured inside GAE).
        metrics.update(getattr(advantage_estimator, "last_metrics", None) or {})

        # ---- critic diagnostics + residual/* (B_LOO for metrics only).
        if self.log_critic_diagnostics and values is not None and returns is not None:
            all_group_order: dict[str, int] = {}
            group_ids = torch.tensor(
                [
                    all_group_order.setdefault(group_key(sid), len(all_group_order))
                    for sid in sample_ids
                ],
                dtype=torch.long,
            )
            residual_metrics, to_res, homogeneous = residual_baseline_diagnostics(
                group_ids,
                rewards,
                returns,
                estimator_mask,
                final_sample_mask,
            )
            metrics.update(residual_metrics)
            values_f = values.float()
            returns_f = returns.float()
            ev_abs, ev_res = pooled_explained_var(
                values_f, returns_f, token_mask, final_sample_mask, to_res
            )
            # Overwrite the loss-derived (training-pass) EV, as legacy did.
            self._critic_overrides = {
                "critic/explained_var": ev_abs,
                "critic/ev_res": ev_res,
            }
            metrics.update(
                return_space_stats(
                    returns_f, token_mask * final_sample_mask.unsqueeze(-1), to_res
                )
            )
            metrics.update(
                positional_value_metrics(
                    values_f, returns_f, token_mask, returns_to_res=to_res
                )
            )
            mixed = 1.0 - homogeneous if homogeneous.numel() else None
            metrics.update(
                mixed_group_value_metrics(
                    values_f, returns_f, token_mask, mixed, returns_to_res=to_res
                )
            )

        self._metrics = metrics

        # ---- packed per-token rollout dump (ppo.log_rollout_dump steps only).
        if input_ids is not None and n > 0:
            self._dump = _pack_rollout_dump(
                num_generations_per_prompt=self.num_generations_per_prompt,
                token_mask=_real(token_mask),
                input_ids=_real(input_ids),
                rewards=_real(rewards),
                sample_mask=_real(final_sample_mask),
                input_lengths=(
                    real_lengths
                    if real_lengths is not None
                    else [int(token_mask.shape[1])] * n
                ),
                prompt_lengths=[
                    int((r.get("trace_metadata") or {}).get("prompt_tokens", 0))
                    for r in debug_rows
                ],
                values=_real(values),
                advantages=_real(advantages),
                generation_logprobs=_real(generation_logprobs),
                prev_logprobs=_real(policy_logprobs),
                returns=_real(returns),
                reference_logprobs=_real(reference_logprobs),
                adv_raw_metrics=getattr(advantage_estimator, "last_metrics", None),
                trace_group_ids=trace_group_ids,
                trace_rollout_ids=trace_rollout_ids,
                idx=[(tags[i] or {}).get("prompt_idx") for i in real_rows],
                truncated=_real(truncated),
            )

        # ---- rollout_debug rows (written once the step closes).
        if self.log_rollout_debug and n > 0:
            self._rollout_debug = self._build_rollout_debug(
                debug_rows=debug_rows,
                trace_group_ids=trace_group_ids,
                trace_rollout_ids=trace_rollout_ids,
                trace_in_rollout_idx=trace_in_rollout_idx,
                is_empty_rollout=is_empty_rollout,
                rewards=_real(rewards),
                final_sample_mask=_real(final_sample_mask),
                pre_gate=_real(pre_gate),
                seq_err=_real(seq_err),
                masked_by_gate=_real(masked_by_gate),
                advantages=_real(advantages),
                token_mask=_real(token_mask),
                values=_real(values),
                total_tokens=real_lengths if real_lengths is not None else [None] * n,
            )

    @staticmethod
    def _build_rollout_debug(
        *,
        debug_rows: list[dict[str, Any]],
        trace_group_ids: list[int],
        trace_rollout_ids: list[int],
        trace_in_rollout_idx: list[int],
        is_empty_rollout: list[bool],
        rewards: torch.Tensor,
        final_sample_mask: torch.Tensor,
        pre_gate: torch.Tensor,
        seq_err: torch.Tensor,
        masked_by_gate: torch.Tensor,
        advantages: torch.Tensor,
        token_mask: torch.Tensor,
        values: Optional[torch.Tensor],
        total_tokens: Sequence[Optional[int]],
    ) -> dict[str, list[Any]]:
        """Legacy ``rollout_debug_step*.jsonl`` columns (ppo.py async loop)."""
        n = len(debug_rows)
        adv = advantages.detach().cpu().float()
        tm = token_mask.detach().cpu().bool()
        has_tokens = tm.any(dim=-1)
        first_tok = tm.float().argmax(dim=-1)
        adv_first = adv.gather(1, first_tok.unsqueeze(1)).squeeze(1)
        adv_first = torch.where(has_tokens, adv_first, torch.zeros_like(adv_first))
        adv_min = (
            torch.where(tm, adv, torch.full_like(adv, float("inf"))).min(dim=-1).values
        )
        adv_max = (
            torch.where(tm, adv, torch.full_like(adv, float("-inf"))).max(dim=-1).values
        )
        adv_min = torch.where(has_tokens, adv_min, torch.zeros_like(adv_min))
        adv_max = torch.where(has_tokens, adv_max, torch.zeros_like(adv_max))
        if values is not None:
            vals = values.detach().cpu().float()
            value_first = torch.where(
                has_tokens,
                vals.gather(1, first_tok.unsqueeze(1)).squeeze(1),
                torch.zeros(n),
            )
        else:
            value_first = torch.zeros(n)
        return {
            "rollout_info": [r.get("rollout_info", {}) for r in debug_rows],
            "trace_metadata": [r.get("trace_metadata", {}) for r in debug_rows],
            "rollout_local_idx": [
                int(r.get("rollout_local_idx", 0)) for r in debug_rows
            ],
            "trace_in_rollout_idx": trace_in_rollout_idx,
            "is_empty_rollout": is_empty_rollout,
            "trace_group_id": trace_group_ids,
            "trace_rollout_id": trace_rollout_ids,
            "reward": rewards.detach().cpu().float().tolist(),
            "sample_loss_mask": final_sample_mask.detach().cpu().float().tolist(),
            "pre_seq_error_sample_loss_mask": pre_gate.detach().cpu().float().tolist(),
            "seq_mult_prob_error": seq_err.detach().cpu().float().tolist(),
            "masked_by_seq_logprob_error": masked_by_gate.detach()
            .cpu()
            .bool()
            .tolist(),
            "advantage": adv_first.tolist(),
            "advantage_min": adv_min.tolist(),
            "advantage_max": adv_max.tolist(),
            "value_first_token": value_first.tolist(),
            "num_generated_tokens": tm.sum(dim=-1).tolist(),
            "total_tokens": total_tokens,
        }

    # ------------------------------------------------------------------ stage 2
    @_diagnostic(dict)
    def step_metrics(
        self,
        *,
        value_result: Optional[Mapping[str, Any]],
        buffer_size: Optional[int] = None,
        sc_reward: Optional[float] = None,
    ) -> dict[str, Any]:
        """This step's legacy metrics, for ``step_metrics.update(...)``.

        Must be merged AFTER ``_compute_critic_metrics(value_result)`` and the
        other SC step metrics: it overwrites ``critic/explained_var`` /
        ``critic/ev_res`` with the pooled pre-update values and
        ``total_num_tokens`` with the unpadded count, as legacy reported them.
        """
        out = dict(self._metrics)
        if "reward" in out and sc_reward is not None:
            # Keep SC's trained-rows reward next to the legacy per-rollout one.
            out["reward_trained_rows"] = float(sc_reward)
        if value_result is not None:
            out.update(critic_gnorm_metrics(value_result.get("grad_norm_groups")))
            out.update(self._critic_overrides)
        if self.post_value_result is not None:
            from nemo_rl.algorithms.ppo import _compute_critic_metrics

            post = _compute_critic_metrics(dict(self.post_value_result))
            out["critic/explained_var_post_update"] = post["critic/explained_var"]
            out["critic/loss_post_update"] = float(np.mean(post["critic/loss"]))
        if buffer_size is not None:
            out["buffer_size"] = buffer_size
        self._metrics = {}
        self._critic_overrides = {}
        self.post_value_result = None
        return out

    # ------------------------------------------------------------------ stage 3
    @_diagnostic(lambda: None)
    def write_rollout_dump(
        self, log_dir: str, *, step: int, tokenizer_factory: Callable[[], Any]
    ) -> Optional[str]:
        """Write ``ppo_rollout_dump_step{step}.pt`` (if this step packed one)."""
        dump, self._dump = self._dump, None
        if dump is None:
            return None
        if self._tokenizer is None:
            self._tokenizer = tokenizer_factory()
        tokenizer = self._tokenizer
        row_ids: list[torch.Tensor] = dump.pop("_row_input_ids")
        dump["step"] = int(step)
        dump["content"] = [
            tokenizer.decode(ids.tolist(), skip_special_tokens=False) for ids in row_ids
        ]
        dump["token_text"] = tokenizer.convert_ids_to_tokens(dump["token_ids"].tolist())
        path = os.path.join(log_dir, f"ppo_rollout_dump_step{step}.pt")
        torch.save(dump, path)
        print(f"  Dumped rollout data to {path}", flush=True)
        return path

    def pop_rollout_debug(
        self, trainer_weight_version: int
    ) -> Optional[dict[str, list[Any]]]:
        """The step's jsonl columns (or None), consuming them."""
        rows, self._rollout_debug = self._rollout_debug, None
        if rows is None:
            return None
        rows["trainer_weight_version"] = [int(trainer_weight_version)] * len(
            rows["reward"]
        )
        return rows

    @_diagnostic(lambda: None)
    def write_rollout_debug(
        self, logger: Any, *, step: int, trainer_weight_version: int
    ) -> None:
        """Write ``rollout_debug_step{step}.jsonl`` under the logger's log dir."""
        rows = self.pop_rollout_debug(trainer_weight_version)
        if rows is not None:
            logger.log_batched_dict_as_jsonl(rows, f"rollout_debug_step{step}.jsonl")


# ---------------------------------------------------------------------------
# timing/train/*, efficiency/*, performance/*, timing/setup/* parity
# ---------------------------------------------------------------------------

#: SC sub-timer name -> the legacy (lm_policy / lm_value) name for the same span.
LEGACY_TIMING_ALIASES: dict[str, str] = {
    "get_logprobs/shard_meta": "get_logprobs/shard_data",
    "get_logprobs/submit_futures": "get_logprobs/submit_logprob_futures",
    "policy_training/shard_meta": "policy_training/sharding_data",
    "policy_training/submit_microbatch_futures": (
        "policy_training/submit_training_futures"
    ),
    "value_training/shard_meta": "value_training/sharding_data",
}


def add_legacy_timing_aliases(timing_metrics: dict[str, float]) -> dict[str, float]:
    """Also report SC sub-timers under their legacy ``timing/train/*`` names.

    Additive: SC names stay. A legacy name SC already emits itself (e.g.
    ``policy_training/submit_training_futures`` on the whole-batch path) is summed
    with the aliased span, since both are the same phase of one step.
    """
    for sc_name, legacy_name in LEGACY_TIMING_ALIASES.items():
        if sc_name in timing_metrics:
            timing_metrics[legacy_name] = timing_metrics.get(legacy_name, 0.0) + float(
                timing_metrics[sc_name]
            )
    return timing_metrics


def legacy_setup_timing_metrics(
    setup_metrics: Mapping[str, float], generation_backend: str
) -> dict[str, float]:
    """``timing/setup/*`` plus legacy's ``vllm_init_time_s`` for a vLLM run.

    SC reports the generation fleet's reserve + load time as
    ``generation_init_time_s``; legacy named the same span ``vllm_init_time_s``.
    """
    out = dict(setup_metrics)
    if generation_backend == "vllm" and "generation_init_time_s" in out:
        out.setdefault("vllm_init_time_s", out["generation_init_time_s"])
    return out


WALL_CLOCK_EFFICIENCY_CATEGORIES = (
    "init/total",
    "idle/buffer_starvation",
    "idle/refit_bubble",
    "idle/validation",
)
THREAD_ACCUMULATED_EFFICIENCY_CATEGORIES = (
    "idle/buffer_full_backoff",
    "idle/generation_limit_pause",
    "idle/refit_event_wait",
    "wasted/failed_trajectory",
)


def legacy_efficiency_summary(
    efficiency_metrics: Mapping[str, float],
    total_wall_time_s: float,
) -> dict[str, float]:
    """Legacy ``print_efficiency_summary`` (RL jiaqiz/ppo-dev utils.py), verbatim maths.

    Wall-clock categories get ``_s`` and ``_pct`` of the cumulative wall time;
    collector categories are thread-seconds (``_s`` only). Waste is the sum of the
    wall-clock categories (init included, clamped to the wall time); productive
    time is the rest. Like legacy, the driver categories are the CURRENT step's
    totals while the denominator is cumulative since training began.
    """
    loggable: dict[str, float] = {}
    for category in WALL_CLOCK_EFFICIENCY_CATEGORIES:
        duration = float(efficiency_metrics.get(category, 0.0))
        pct = (duration / total_wall_time_s * 100) if total_wall_time_s > 0 else 0.0
        loggable[f"efficiency/{category}_s"] = duration
        loggable[f"efficiency/{category}_pct"] = pct
    thread_seconds_total = 0.0
    for category in THREAD_ACCUMULATED_EFFICIENCY_CATEGORIES:
        duration = float(efficiency_metrics.get(category, 0.0))
        thread_seconds_total += duration
        loggable[f"efficiency/{category}_s"] = duration
    wall_waste = sum(
        float(efficiency_metrics.get(cat, 0.0))
        for cat in WALL_CLOCK_EFFICIENCY_CATEGORIES
    )
    if total_wall_time_s > 0 and wall_waste > total_wall_time_s:
        wall_waste = total_wall_time_s
    productive = max(0.0, total_wall_time_s - wall_waste)
    efficiency_pct = (
        (productive / total_wall_time_s * 100) if total_wall_time_s > 0 else 100.0
    )
    efficiency_pct = min(100.0, max(0.0, efficiency_pct))
    loggable["efficiency/thread_seconds_total_s"] = thread_seconds_total
    loggable["efficiency/total_waste_s"] = wall_waste
    loggable["efficiency/productive_time_s"] = productive
    loggable["efficiency/efficiency_pct"] = efficiency_pct
    loggable["efficiency/total_wall_time_s"] = total_wall_time_s
    return loggable


class LegacyEfficiencyClock:
    """Run-level clocks behind legacy ``efficiency/*`` and the starvation timers.

    * Buffer starvation: :meth:`starved` on every empty poll of the train pump,
      :meth:`fed` once a batch is selected (or the wait ends). The first wait of
      the process is legacy's ``init/total`` (the pre-loop buffer fill, logged at
      the job's first step); later waits are ``idle/buffer_starvation``.
    * Collector thread-seconds, cumulative since the controller started (legacy's
      collector timer was never reset): ``idle/generation_limit_pause`` = time
      the rollout pump was blocked before dispatch (buffer capacity, in-flight
      slots, weight-sync pause -- legacy paused its single collector loop for
      the same limits); ``wasted/failed_trajectory`` = wall time of rollout
      attempts that raised (fed from the rollout manager's stats).
      ``idle/buffer_full_backoff`` is structurally 0 on SC (the capacity permit
      is taken before dispatch, so a finished group never waits to enqueue) and
      ``idle/refit_event_wait`` is folded into the pre-dispatch wait above.
    """

    def __init__(self) -> None:
        self.wall_start = time.perf_counter()
        self._starved_since: Optional[float] = None
        self._initial_fill_done = False
        self.collector_seconds: dict[str, float] = {
            category: 0.0 for category in THREAD_ACCUMULATED_EFFICIENCY_CATEGORIES
        }

    def starved(self) -> None:
        if self._starved_since is None:
            self._starved_since = time.perf_counter()

    def fed(self, timer: Any) -> None:
        """Close an open starvation wait into ``timer`` (no-op when not starved)."""
        if self._starved_since is None:
            self._initial_fill_done = True
            return
        elapsed = time.perf_counter() - self._starved_since
        self._starved_since = None
        label = "idle/buffer_starvation" if self._initial_fill_done else "init/total"
        self._initial_fill_done = True
        timer.record(label, elapsed)

    def record_pre_dispatch_wait(self, seconds: float) -> None:
        self.collector_seconds["idle/generation_limit_pause"] += max(0.0, seconds)

    def summary(
        self, timer: Any, *, failed_trajectory_s: Optional[float] = None
    ) -> dict[str, float]:
        """``efficiency/*`` for the current step (call before ``timer.reset()``)."""
        if failed_trajectory_s is not None:
            self.collector_seconds["wasted/failed_trajectory"] = float(
                failed_trajectory_s
            )
        driver = {
            category: timer.reduce(category, "sum")
            for category in WALL_CLOCK_EFFICIENCY_CATEGORIES
            if category in getattr(timer, "_timers", {})
        }
        merged = {**driver}
        for category, seconds in self.collector_seconds.items():
            merged[category] = merged.get(category, 0.0) + seconds
        return legacy_efficiency_summary(merged, time.perf_counter() - self.wall_start)


def legacy_performance_metrics(
    *,
    train_result: Optional[Mapping[str, Any]],
    step_metrics: Mapping[str, Any],
    timing_metrics: Mapping[str, float],
    master_config: Any,
    num_prompts_per_step: int,
    num_generations_per_prompt: int,
) -> dict[str, float]:
    """Legacy ``performance/*`` via the shared ``print_performance_metrics``.

    Works on copies: the helper deletes ``per_worker_token_counts`` from the
    metrics it is given and must not see the SC-only ``vllm/*`` dicts.
    """
    from nemo_rl.algorithms.utils import print_performance_metrics

    if "total_step_time" not in timing_metrics:
        return {}
    return print_performance_metrics(
        dict(train_result or {}),
        dict(step_metrics),
        dict(timing_metrics),
        master_config,
        num_prompts_per_step=num_prompts_per_step,
        num_generations_per_prompt=num_generations_per_prompt,
        is_async_rl=True,
    )


class _NullEfficiencyClock(LegacyEfficiencyClock):
    """No-op clock for controllers built without ``__init__`` (unit tests)."""

    def starved(self) -> None:
        return None

    def fed(self, timer: Any) -> None:
        return None

    def record_pre_dispatch_wait(self, seconds: float) -> None:
        return None


NULL_EFFICIENCY_CLOCK: LegacyEfficiencyClock = _NullEfficiencyClock()


def fetch_generation_logger_metrics(generation: Any) -> dict[str, Any]:
    """Legacy ``generation_logger_metrics``: the per-engine vLLM logger samples.

    ``{metric: {dp_idx: [samples]}}`` since the last call, which then clears
    the workers' buffers (legacy fetched before each refit and cleared after it;
    without the clear the worker-side histories grow for the whole run). W&B
    flattens it to ``train/generation_logger_metrics.<metric>.<dp_idx>``.
    Empty when the backend has no logger (non-vLLM, or the logger disabled).
    """
    try:
        metrics = generation.get_logger_metrics()
        generation.clear_logger_metrics()
    except Exception as error:  # noqa: BLE001 - diagnostics only
        print(f"Skipping generation_logger_metrics: {error!r}", flush=True)
        return {}
    return {"generation_logger_metrics": metrics} if metrics else {}


def _pack_rollout_dump(
    *,
    num_generations_per_prompt: int,
    token_mask: torch.Tensor,
    input_ids: torch.Tensor,
    rewards: torch.Tensor,
    sample_mask: torch.Tensor,
    input_lengths: Sequence[int],
    prompt_lengths: Sequence[int],
    values: Optional[torch.Tensor],
    advantages: torch.Tensor,
    generation_logprobs: Optional[torch.Tensor],
    prev_logprobs: Optional[torch.Tensor],
    returns: Optional[torch.Tensor],
    reference_logprobs: Optional[torch.Tensor],
    adv_raw_metrics: Optional[Mapping[str, float]],
    trace_group_ids: Sequence[int],
    trace_rollout_ids: Sequence[int],
    idx: Sequence[Any],
    truncated: Optional[torch.Tensor],
) -> dict[str, Any]:
    """Legacy ``_build_ppo_rollout_dump_payload`` (format_version 2), unpadded rows.

    Response tokens (``token_mask == 1``) are packed into flat tensors;
    ``token_sample_index`` / ``token_sequence_position`` map each packed token
    back to its row and position. Everything reflects the state entering the
    update loop (pre-update critic values, pi_old logprobs). ``content`` and
    ``token_text`` are filled in at write time from ``_row_input_ids``.
    Difference from legacy: DP pad rows are not dumped (legacy dumped them with
    sample_loss_mask 0 while its prompt_length/content arrays stopped short of
    them), and ``content`` is the decoded token sequence (chat template
    included) rather than the concatenated message texts SC no longer keeps.
    """
    tm = token_mask.detach().bool().cpu()
    batch_size = tm.shape[0]
    token_count = tm.sum(dim=-1)
    coords = tm.nonzero(as_tuple=False)
    token_sample_index = coords[:, 0].to(torch.int32)
    token_sequence_position = coords[:, 1].to(torch.int32)
    if token_sample_index.numel() > 0:
        token_response_position = torch.cat(
            [torch.arange(int(c), dtype=torch.int32) for c in token_count.tolist()]
        )
    else:
        token_response_position = torch.empty(0, dtype=torch.int32)

    def _pack(t: torch.Tensor) -> torch.Tensor:
        return t.detach().float().cpu()[tm]

    ids = input_ids.detach().cpu()
    sample_indices = torch.arange(batch_size, dtype=torch.int32)
    payload: dict[str, Any] = {
        "format_version": 2,
        "credit_level": "token",
        "description": (
            "Packed response-token PPO rollout dump. Per-token tensors are "
            "flattened over token_mask; entry i belongs to sample "
            "token_sample_index[i] at sequence position "
            "token_sequence_position[i] and describes token token_ids[i]. "
            "prev_logprobs are the policy logprobs before any update this "
            "step (the PPO-ratio denominator, pi_old); generation_logprobs "
            "come from the rollout backend; values are the critic estimates "
            "GAE consumed; returns are the critic regression targets."
        ),
        "num_generations_per_prompt": int(num_generations_per_prompt),
        "sample_index": sample_indices,
        "reward": rewards.detach().float().cpu(),
        "sample_loss_mask": sample_mask.detach().float().cpu(),
        "input_length": torch.tensor(list(input_lengths), dtype=torch.int64),
        "prompt_length": torch.tensor(list(prompt_lengths), dtype=torch.int64),
        "num_response_tokens": token_count,
        "token_sample_index": token_sample_index,
        "token_sequence_position": token_sequence_position,
        "token_response_position": token_response_position,
        "token_ids": ids[tm],
        "advantages": _pack(advantages),
        "_row_input_ids": [
            ids[i, : int(input_lengths[i])].clone() for i in range(batch_size)
        ],
    }
    for name, tensor in (
        ("values", values),
        ("generation_logprobs", generation_logprobs),
        ("prev_logprobs", prev_logprobs),
        ("returns", returns),
        ("reference_policy_logprobs", reference_logprobs),
    ):
        if tensor is not None:
            payload[name] = _pack(tensor)
    if adv_raw_metrics:
        for key in ("mean", "std"):
            value = adv_raw_metrics.get(f"adv_raw/{key}")
            if value is not None:
                payload[f"adv_raw_{key}"] = float(value)
    gpp = max(int(num_generations_per_prompt), 1)
    payload["prompt_group_index"] = torch.tensor(
        list(trace_group_ids), dtype=torch.int32
    )
    payload["generation_index"] = (
        torch.tensor(list(trace_rollout_ids), dtype=torch.int64) % gpp
    ).to(torch.int32)
    if all(i is not None for i in idx):
        payload["idx"] = list(idx)
    if truncated is not None:
        payload["truncated"] = truncated.detach().cpu()
    return payload


def timer_kwargs(method: Callable[..., Any], timer: Any) -> dict[str, Any]:
    """``{"timer": timer}`` when ``method`` declares a ``timer`` parameter, else ``{}``.

    TQPolicy / TQValue time their shard and submit phases under the timer they
    are handed (legacy's ``timing/train/*/sharding_data`` etc.); trainer doubles
    and other backends without the argument are called exactly as before.
    """
    try:
        params = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return {}
    # An explicit parameter only: a bare **kwargs (mocks, wrappers) is not a
    # promise to time anything.
    return {"timer": timer} if "timer" in params else {}
