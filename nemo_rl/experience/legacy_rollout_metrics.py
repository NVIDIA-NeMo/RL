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
"""Legacy async-PPO rollout diagnostics, ported to the SingleController path.

Ports of the per-prompt-group rollout metrics and per-trace provenance that
legacy ``nemo_rl/experience/rollouts.py::run_async_nemo_gym_rollout`` computed
(branch jiaqiz/ppo-dev), so existing W&B panels and the offline
``rollout_debug_step*.jsonl`` tooling keep working:

* :func:`legacy_rollout_group_metrics` -- one prompt group's rollout metrics
  (``termination/*``, ``mask_sample/by_kind/*``, ``reward/by_*``,
  ``resolved/by_delegation*``, ``turns_per_trace/*``, ``traces_per_sample/*``,
  ``compaction*``, ``subagent_*_per_sample/*``, ``delegation_rate``,
  ``mask_sample_rate``, ``empty_rollout_count``,
  ``format/think_tag_violation/*``). Exact-aggregation ``X/sum`` / ``X/count``
  pairs are summed across groups and finalized by
  :func:`aggregate_rollout_metrics_with_sum_counts`.
* :func:`rollout_debug_tags` -- the per-row provenance (``rollout_info``,
  ``trace_metadata``, rollout/trace indices) that legacy carried on the train
  batch; on SC it rides the TransferQueue row ``tags`` as one JSON string.

Function bodies are kept line-for-line with legacy where possible; the only
SC-specific code is :func:`rollout_traces`, which views an SC NeMo-Gym result
(``session_traces`` for multi-trace, ``full_result.is_empty_rollout`` for the
masked placeholder) as legacy's list of per-trace dicts.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import nullcontext
from typing import Any, Optional

import torch

from nemo_rl.algorithms.multi_trace_metrics import (
    finalize_sum_count_metrics,
    trace_index_bucket,
)
from nemo_rl.experience.interfaces import TRACE_METADATA_KEY
from nemo_rl.experience.metric_utils import calculate_single_metric, pct

#: Tag key carrying one row's JSON-encoded provenance (see rollout_debug_tags).
ROLLOUT_DEBUG_TAG = "rollout_debug"

# One-hot terminal reason of a rollout (see termination_kind). Fixed order so
# every group emits the same key set (absent kinds emit 0).
TERMINATION_KINDS: tuple[str, ...] = (
    "completed",
    "max_iteration",
    "context_window",
    "stuck_in_loop",
    "other_error",
    "agent_timeout",
    "agent_oom",
    "eval_timeout",
    "eval_oom",
)
_NAMED_AGENT_ERROR_KINDS = frozenset(
    {"max_iteration", "context_window", "stuck_in_loop"}
)

# Trace kinds whose per-trace turn count is comparable to `agent_max_turns`
# (the compaction summary is a single forced call; the empty dummy has none).
TURN_COUNT_TRACE_KINDS: tuple[str, ...] = (
    "uncompacted",
    "pre_compaction",
    "post_compaction",
    "subagent",
)


def rollout_debug_info(full_result: Mapping[str, Any]) -> dict[str, Any]:
    """Compact, JSON-serializable rollout-level record for offline debugging.

    Built from the env's ``full_result`` (NeMo Gym SWE agent
    ``SWEBenchVerifyResponse``); every field is optional so other Gym agents
    degrade to a mostly-empty record. Only the fixed fields below are copied --
    never the request/response payloads, so nothing from the agent's environment
    (credentials included) can leak into the jsonl.
    """
    instance_config = full_result.get("instance_config") or {}
    problem_info = instance_config.get("problem_info") or {}
    responses = full_result.get("responses") or []
    segment_metas = [dict((r or {}).get("metadata") or {}) for r in responses]
    num_compactions = sum(
        1 for m in segment_metas if m.get("segment_boundary_reason") == "compaction"
    )
    subagent_sessions = {
        m.get("session_id") for m in segment_metas if m.get("parent_session_id")
    }
    # Root vs subagent split. Classify per SESSION, not per segment: a
    # subagent's own compaction-summary segment is dumped without
    # parent_session_id (the fork's compaction call does not forward it), so
    # judging that segment by its own parent field would count it as root.
    # A session is a subagent if ANY of its segments carries a parent id.
    sessions: dict[str, list[dict]] = {}
    for m in segment_metas:
        sessions.setdefault(str(m.get("session_id") or ""), []).append(m)

    def _session_compactions(segs: list[dict]) -> int:
        return sum(1 for s in segs if s.get("segment_boundary_reason") == "compaction")

    num_root_compactions = sum(
        _session_compactions(segs)
        for segs in sessions.values()
        if not any(s.get("parent_session_id") for s in segs)
    )
    num_subagent_compactions = num_compactions - num_root_compactions
    if "opencode_finished" in full_result:
        return _sandboxed_agent_debug_info(full_result, instance_config)
    return {
        "instance_id": problem_info.get("instance_id") or instance_config.get("name"),
        "dataset_name": problem_info.get("dataset_name"),
        "reward": full_result.get("reward"),
        "resolved": full_result.get("resolved"),
        "patch_exists": full_result.get("patch_exists"),
        "mask_sample": bool(instance_config.get("mask_sample", False)),
        "agent_error_kind": full_result.get("agent_error_kind"),
        "agent_timed_out": full_result.get("agent_timed_out"),
        "eval_timed_out": full_result.get("eval_timed_out"),
        "oom_killed": full_result.get("oom_killed"),
        "eval_oom_killed": full_result.get("eval_oom_killed"),
        "openhands_run_time": full_result.get("openhands_run_time"),
        "final_eval_time": full_result.get("final_eval_time"),
        "num_segments": len(responses),
        "num_compactions": num_compactions,
        "num_root_compactions": num_root_compactions,
        "num_subagent_compactions": num_subagent_compactions,
        "num_subagent_sessions": len(subagent_sessions),
        "segments": [
            {
                "session_id": m.get("session_id"),
                "parent_session_id": m.get("parent_session_id") or "",
                "segment_index": m.get("segment_index"),
                "segment_boundary_reason": m.get("segment_boundary_reason") or "",
            }
            for m in segment_metas
        ],
    }


def _sandboxed_agent_debug_info(
    full_result: Mapping[str, Any], instance_config: Mapping[str, Any]
) -> dict[str, Any]:
    """:func:`rollout_debug_info` for a Gym opencode_sandboxed_agent result.

    Maps the sandbox agent's and SWE servers' fields onto the legacy record so
    the legacy termination / delegation / dataset breakdowns read the same:
    an exec timeout is ``agent_timeout``, exit 137 ``agent_oom``, any other
    failed OpenCode run (including a context-length stop, which legacy's fork
    reported as ``max_compaction``) ``other_error``; an evaluation the server
    reports incomplete because it timed out / was killed is ``eval_timeout`` /
    ``eval_oom``. Session segments arrive later as finalizer rows, so the
    segment fields are empty and subagent sessions are the ``task`` calls.
    """
    error_type = full_result.get("opencode_error_type")
    agent_timed_out = error_type in ("timeout", "TimeoutError")
    oom_killed = full_result.get("opencode_exit_code") == 137
    agent_error_kind = None
    if not (agent_timed_out or oom_killed) and full_result.get("opencode_failed"):
        agent_error_kind = str(error_type or "opencode_failed")
    eval_error = str(full_result.get("error") or "")
    eval_incomplete = full_result.get("evaluation_completed") is False
    return {
        "instance_id": full_result.get("instance_id"),
        "dataset_name": full_result.get("dataset_name"),
        "reward": full_result.get("reward"),
        "resolved": full_result.get("resolved"),
        "patch_exists": (full_result.get("model_patch_bytes") or 0) > 0,
        "mask_sample": bool(instance_config.get("mask_sample", False)),
        "agent_error_kind": agent_error_kind,
        "agent_timed_out": agent_timed_out,
        "eval_timed_out": eval_incomplete and "timed out" in eval_error,
        "oom_killed": oom_killed,
        "eval_oom_killed": eval_incomplete and "exit 137" in eval_error,
        "openhands_run_time": full_result.get("opencode_run_time_taken"),
        "final_eval_time": full_result.get("patch_verification_time_taken"),
        "num_segments": 0,
        "num_compactions": 0,
        "num_root_compactions": 0,
        "num_subagent_compactions": 0,
        "num_subagent_sessions": int(full_result.get("opencode_num_task_calls") or 0),
        "segments": [],
    }


def termination_kind(info: Mapping[str, Any]) -> str:
    """One-hot terminal reason of a rollout from its :func:`rollout_debug_info` record.

    Priority (first match wins): oom_killed -> ``agent_oom``; agent_timed_out ->
    ``agent_timeout``; agent_error_kind in {max_iteration, context_window,
    stuck_in_loop} -> that string; any other non-empty agent_error_kind ->
    ``other_error``; eval_oom_killed -> ``eval_oom``; eval_timed_out ->
    ``eval_timeout``; else ``completed``.
    """
    if info.get("oom_killed"):
        return "agent_oom"
    if info.get("agent_timed_out"):
        return "agent_timeout"
    error_kind = info.get("agent_error_kind") or ""
    if not isinstance(error_kind, str):
        error_kind = str(error_kind)
    if error_kind in _NAMED_AGENT_ERROR_KINDS:
        return error_kind
    if error_kind:
        return "other_error"
    if info.get("eval_oom_killed"):
        return "eval_oom"
    if info.get("eval_timed_out"):
        return "eval_timeout"
    return "completed"


def trace_kind(
    trace_metadata: Mapping[str, Any] | None,
    rollout_info: Mapping[str, Any] | None,
    is_empty_rollout: bool = False,
) -> str:
    """Segment kind of one trace from its env metadata and its rollout's info.

    ``subagent`` (parent_session_id set) > ``compaction_summary``
    (segment_boundary_reason == "compaction") > ``post_compaction``
    (== "post_compaction") > ``pre_compaction`` (segment_index "0" of a rollout
    that compacted at least once) > ``uncompacted``. The empty-rollout dummy
    trace is ``empty``.
    """
    if is_empty_rollout:
        return "empty"
    md = trace_metadata or {}
    if md.get("parent_session_id"):
        return "subagent"
    reason = md.get("segment_boundary_reason") or ""
    if reason == "compaction":
        return "compaction_summary"
    if reason == "post_compaction":
        return "post_compaction"
    num_compactions = (rollout_info or {}).get("num_compactions") or 0
    if str(md.get("segment_index", "")) == "0" and num_compactions > 0:
        return "pre_compaction"
    return "uncompacted"


def compaction_rollout_metrics(
    rollout_infos: Sequence[Mapping[str, Any]], rewards: Sequence[float]
) -> dict[str, float]:
    """Per-prompt-group rollout metrics as exact ``/sum`` + ``/count`` pairs.

    Emitted as sums/counts (not means) so the trainer can aggregate the prompt
    groups exactly (sum, then divide) instead of averaging per-group means over
    unequal bucket sizes; :func:`finalize_sum_count_metrics` derives ``/mean``
    and ``/rate``.

    Keys: ``rollouts/count``; ``reward/by_num_compactions/{0,1,2plus}/{sum,count}``;
    ``termination/{kind}/count`` and ``reward/by_termination/{kind}/{sum,count}``
    for every kind in ``TERMINATION_KINDS``; ``mask_sample/by_kind/{kind}/count``
    (termination kind of env-masked rollouts); ``reward/{masked,unmasked}/{sum,count}``;
    ``reward/masked_resolved/count`` (env-masked AND resolved). Delegation split
    (``delegated`` = the rollout spawned >=1 subagent session):
    ``{reward,resolved}/by_delegation/{delegated,none}/{sum,count}``, the same
    pair under ``by_delegation_unmasked``, and
    ``mask_sample/by_delegation/{bucket}/count``.
    """
    assert len(rollout_infos) == len(rewards), (
        f"{len(rollout_infos)} rollout infos vs {len(rewards)} rewards"
    )
    rewards = [float(r) for r in rewards]
    metrics: dict[str, float] = {"rollouts/count": len(rollout_infos)}

    by_num_compactions: dict[str, list[float]] = {"0": [], "1": [], "2plus": []}
    by_termination: dict[str, list[float]] = {k: [] for k in TERMINATION_KINDS}
    masked_by_kind: dict[str, int] = {k: 0 for k in TERMINATION_KINDS}
    masked_rewards: list[float] = []
    unmasked_rewards: list[float] = []
    masked_resolved = 0
    delegation_buckets = ("delegated", "none")
    by_delegation: dict[str, list[float]] = {b: [] for b in delegation_buckets}
    resolved_by_delegation: dict[str, int] = {b: 0 for b in delegation_buckets}
    masked_by_delegation: dict[str, int] = {b: 0 for b in delegation_buckets}
    unmasked_by_delegation: dict[str, list[float]] = {b: [] for b in delegation_buckets}
    unmasked_resolved_by_delegation: dict[str, int] = {b: 0 for b in delegation_buckets}
    for info, reward in zip(rollout_infos, rewards):
        num_compactions = info.get("num_compactions") or 0
        by_num_compactions[
            "0" if num_compactions == 0 else ("1" if num_compactions == 1 else "2plus")
        ].append(reward)
        kind = termination_kind(info)
        by_termination[kind].append(reward)
        delegation = (
            "delegated" if (info.get("num_subagent_sessions") or 0) > 0 else "none"
        )
        resolved = bool(info.get("resolved"))
        by_delegation[delegation].append(reward)
        resolved_by_delegation[delegation] += resolved
        if info.get("mask_sample"):
            masked_by_kind[kind] += 1
            masked_by_delegation[delegation] += 1
            masked_rewards.append(reward)
            if resolved:
                masked_resolved += 1
        else:
            unmasked_rewards.append(reward)
            unmasked_by_delegation[delegation].append(reward)
            unmasked_resolved_by_delegation[delegation] += resolved

    for bucket, values in by_num_compactions.items():
        metrics[f"reward/by_num_compactions/{bucket}/sum"] = sum(values)
        metrics[f"reward/by_num_compactions/{bucket}/count"] = len(values)
    for kind in TERMINATION_KINDS:
        values = by_termination[kind]
        metrics[f"termination/{kind}/count"] = len(values)
        metrics[f"reward/by_termination/{kind}/sum"] = sum(values)
        metrics[f"reward/by_termination/{kind}/count"] = len(values)
        metrics[f"mask_sample/by_kind/{kind}/count"] = masked_by_kind[kind]
    for bucket in delegation_buckets:
        values = by_delegation[bucket]
        unmasked_values = unmasked_by_delegation[bucket]
        metrics[f"reward/by_delegation/{bucket}/sum"] = sum(values)
        metrics[f"reward/by_delegation/{bucket}/count"] = len(values)
        metrics[f"resolved/by_delegation/{bucket}/sum"] = resolved_by_delegation[bucket]
        metrics[f"resolved/by_delegation/{bucket}/count"] = len(values)
        metrics[f"mask_sample/by_delegation/{bucket}/count"] = masked_by_delegation[
            bucket
        ]
        metrics[f"reward/by_delegation_unmasked/{bucket}/sum"] = sum(unmasked_values)
        metrics[f"reward/by_delegation_unmasked/{bucket}/count"] = len(unmasked_values)
        metrics[f"resolved/by_delegation_unmasked/{bucket}/sum"] = (
            unmasked_resolved_by_delegation[bucket]
        )
        metrics[f"resolved/by_delegation_unmasked/{bucket}/count"] = len(
            unmasked_values
        )
    metrics["reward/masked/sum"] = sum(masked_rewards)
    metrics["reward/masked/count"] = len(masked_rewards)
    metrics["reward/unmasked/sum"] = sum(unmasked_rewards)
    metrics["reward/unmasked/count"] = len(unmasked_rewards)
    metrics["reward/masked_resolved/count"] = masked_resolved
    return metrics


def trace_kind_metrics(
    kinds: Sequence[str],
    turns_per_trace: Sequence[int],
    prompt_tokens: Sequence[int],
    gen_tokens: Sequence[int],
) -> dict[str, float]:
    """Per-trace size metrics split by segment kind, as ``/sum``, ``/count``, ``/max``.

    ``compaction/summary_gen_tokens/*``: generated tokens of compaction-summary
    traces. ``compaction/trigger_prompt_tokens/*``: prompt tokens of
    compaction-summary traces (= context size when compaction fired).
    ``compaction/post_compaction_prompt_tokens/*``: prompt size of the first
    post-compaction call. ``turns_per_trace/by_kind/{kind}/*`` for kinds in
    ``TURN_COUNT_TRACE_KINDS``. Keys are always present (0 when the kind is
    absent) so the key set is stable across groups.
    """
    assert len(kinds) == len(turns_per_trace) == len(prompt_tokens) == len(gen_tokens)

    def _sum_count_max(values: list[int], key: str) -> dict[str, float]:
        return {
            f"{key}/sum": sum(values),
            f"{key}/count": len(values),
            f"{key}/max": max(values, default=0),
        }

    summary_rows = [i for i, k in enumerate(kinds) if k == "compaction_summary"]
    post_rows = [i for i, k in enumerate(kinds) if k == "post_compaction"]
    metrics: dict[str, float] = {
        **_sum_count_max(
            [gen_tokens[i] for i in summary_rows], "compaction/summary_gen_tokens"
        ),
        **_sum_count_max(
            [prompt_tokens[i] for i in summary_rows],
            "compaction/trigger_prompt_tokens",
        ),
        **_sum_count_max(
            [prompt_tokens[i] for i in post_rows],
            "compaction/post_compaction_prompt_tokens",
        ),
    }
    for kind in TURN_COUNT_TRACE_KINDS:
        metrics.update(
            _sum_count_max(
                [t for t, k in zip(turns_per_trace, kinds) if k == kind],
                f"turns_per_trace/by_kind/{kind}",
            )
        )
    return metrics


def _count_token(token_ids: Any, token_id: int) -> int:
    if torch.is_tensor(token_ids):
        return int((token_ids == token_id).sum().item())
    return sum(1 for t in token_ids if t == token_id)


def trace_has_think_tag_token_violation(
    message_log: Sequence[Mapping[str, Any]],
    think_open_token_id: int,
    think_close_token_id: int,
) -> bool:
    """Token-ID think-tag check over one trace's (user, assistant) pairs.

    Thinking mode is inferred from the prompt: open == close means
    enable_thinking=False (expect 0 open / 0 close in the generation);
    open == close + 1 (trailing ``<think>``) means enable_thinking=True (expect
    0 open / 1 close). Any other prompt pattern or generation count is a violation.
    """
    for prev, msg in zip(message_log, message_log[1:]):
        if prev.get("role") != "user" or msg.get("role") != "assistant":
            continue
        prompt_open = _count_token(prev["token_ids"], think_open_token_id)
        prompt_close = _count_token(prev["token_ids"], think_close_token_id)
        if prompt_open == prompt_close:
            expected_open, expected_close = 0, 0
        elif prompt_open == prompt_close + 1:
            expected_open, expected_close = 0, 1
        else:
            return True
        if (
            _count_token(msg["token_ids"], think_open_token_id) != expected_open
            or _count_token(msg["token_ids"], think_close_token_id) != expected_close
        ):
            return True
    return False


def rollout_has_think_tag_string_violation(full_result: Mapping[str, Any]) -> bool:
    """String think-tag check on the decoded ``generation_str`` of every output item.

    Catches ``<think>`` / ``</think>`` spelled with regular tokens. Reads the
    per-segment ``responses`` when present, else ``response``. ``<think>`` must
    never be generated; ``</think>`` at most once per item.
    """
    decoded_responses = full_result.get("responses") or [
        full_result.get("response") or {}
    ]
    for response in decoded_responses:
        for item in (response or {}).get("output", []) or []:
            gen_str = item.get("generation_str", "") if isinstance(item, dict) else ""
            if gen_str and (
                gen_str.count("<think>") > 0 or gen_str.count("</think>") > 1
            ):
                return True
    return False


def think_tag_violation_metrics(
    rollout_results: Sequence[Sequence[Mapping[str, Any]]],
    token_ids_cfg: Mapping[str, Any] | None,
) -> dict[str, int]:
    """Count-only think-tag violations (always on; independent of the penalty flag).

    ``format/think_tag_violation/count``: rollouts with a violation in any trace
    (same predicate as the ``penalize_malformed_think_tag`` penalty).
    ``format/think_tag_violation/by_trace_in_rollout_idx/{0,1,2plus}/count``:
    traces failing the token-ID check, by index within their rollout.
    """
    cfg = token_ids_cfg or {}
    think_open = cfg.get("think_open", 12)
    think_close = cfg.get("think_close", 13)
    rollout_violations = 0
    trace_violations = {"0": 0, "1": 0, "2plus": 0}
    for traces in rollout_results:
        real_traces = [
            (t_idx, trace)
            for t_idx, trace in enumerate(traces)
            if not trace.get("is_empty_rollout", False)
        ]
        if not real_traces:
            continue
        token_violations = [
            trace_has_think_tag_token_violation(
                trace["message_log"], think_open, think_close
            )
            for _, trace in real_traces
        ]
        for (t_idx, _), violated in zip(real_traces, token_violations):
            if violated:
                trace_violations[trace_index_bucket(t_idx)] += 1
        # The string check reads the shared full_result, so run it once.
        if any(token_violations) or rollout_has_think_tag_string_violation(
            real_traces[0][1]["full_result"]
        ):
            rollout_violations += 1
    return {
        "format/think_tag_violation/count": rollout_violations,
        **{
            f"format/think_tag_violation/by_trace_in_rollout_idx/{bucket}/count": n
            for bucket, n in trace_violations.items()
        },
    }


def _legacy_trace_metadata(md: Mapping[str, Any] | None) -> dict[str, Any]:
    """SC trace metadata minus the SC-only index (legacy stored it as its own column)."""
    return {k: v for k, v in (md or {}).items() if k != "trace_in_rollout_idx"}


def _turns(message_log: Sequence[Mapping[str, Any]]) -> int:
    return sum(1 for m in message_log if m.get("role") == "user")


def _prompt_tokens(message_log: Sequence[Mapping[str, Any]]) -> int:
    return len(message_log[0]["token_ids"]) if message_log else 0


def _gen_tokens(message_log: Sequence[Mapping[str, Any]]) -> int:
    return sum(len(m["token_ids"]) for m in message_log if m.get("role") == "assistant")


def rollout_traces(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """View one SC NeMo-Gym rollout result as legacy's list of per-trace dicts.

    Each entry has ``message_log``, ``full_result`` (shared), ``trace_metadata``
    (legacy shape, i.e. without ``trace_in_rollout_idx``) and
    ``is_empty_rollout``. SC marks the masked no-generation placeholder on
    ``full_result`` (legacy marked the trace itself).
    """
    full_result = result["full_result"]
    if "session_traces" in result:
        return [
            {
                "message_log": trace["message_log"],
                "full_result": full_result,
                "trace_metadata": _legacy_trace_metadata(trace.get("trace_metadata")),
                "is_empty_rollout": False,
            }
            for trace in result["session_traces"]
        ]
    return [
        {
            "message_log": result.get("message_log") or [],
            "full_result": full_result,
            "trace_metadata": None,
            "is_empty_rollout": bool(
                result.get("is_empty_rollout") or full_result.get("is_empty_rollout")
            ),
        }
    ]


def legacy_rollout_group_metrics(
    results: Sequence[Mapping[str, Any]],
    reward_penalty_config: Mapping[str, Any] | None,
    *,
    timer: Optional[Any] = None,
    timer_prefix: str = "timing/rollout",
) -> dict[str, Any]:
    """One prompt group's legacy rollout metrics, from SC NeMo-Gym results.

    Must run after reward penalties were applied (``full_result.reward`` is the
    post-penalty reward legacy logged) and after the message logs were
    tensorized. Per-rollout quantities count every trace of the rollout, as
    legacy did; per-trace ones (``turns_per_trace``) count every trace row.
    With ``timer``, the two halves are timed under legacy's
    ``{timer_prefix}/prepare_for_metrics_calculation`` and
    ``{timer_prefix}/aggregate_metrics``.
    """

    def _timed(name: str) -> Any:
        return (
            timer.time(f"{timer_prefix}/{name}") if timer is not None else nullcontext()
        )

    with _timed("prepare_for_metrics_calculation"):
        prepared = _prepare_group(results)
    if prepared is None:
        return {}
    with _timed("aggregate_metrics"):
        return _aggregate_group(prepared, reward_penalty_config)


def _prepare_group(results: Sequence[Mapping[str, Any]]) -> Optional[dict[str, Any]]:
    rollout_results = [rollout_traces(r) for r in results]
    batch_size = len(rollout_results)
    if batch_size == 0:
        return None
    traces = [t for rollout in rollout_results for t in rollout]
    trace_rollout_local_idx = [
        i for i, rollout in enumerate(rollout_results) for _ in rollout
    ]
    turns_per_trace = [_turns(t["message_log"]) for t in traces]
    rollout_infos = [
        rollout_debug_info(rollout[0]["full_result"]) for rollout in rollout_results
    ]
    trace_kinds = [
        trace_kind(
            t["trace_metadata"],
            rollout_infos[trace_rollout_local_idx[i]],
            is_empty_rollout=t["is_empty_rollout"],
        )
        for i, t in enumerate(traces)
    ]
    # Subagent TRACES per rollout (rows), vs `num_subagent_sessions` (sessions):
    # a subagent that compacts contributes several traces from one session.
    subagent_traces_per_rollout = [0] * batch_size
    for i, kind in enumerate(trace_kinds):
        if kind == "subagent":
            subagent_traces_per_rollout[trace_rollout_local_idx[i]] += 1
    trace_prompt_tokens = [_prompt_tokens(t["message_log"]) for t in traces]
    trace_gen_tokens = [_gen_tokens(t["message_log"]) for t in traces]
    total_rewards = [
        float(rollout[0]["full_result"].get("reward") or 0.0)
        for rollout in rollout_results
    ]
    # Legacy per-ROLLOUT sizes sum every trace of the rollout (SC's own
    # turns/tokens_per_sample read the main trace only). Token-capture receipts
    # carry no message logs, so SC's manifest-based numbers stand there.
    per_rollout_sizes = None
    if not any("receipt" in r for r in results):
        per_rollout_sizes = {
            "turns": [
                sum(_turns(t["message_log"]) for t in ro) for ro in rollout_results
            ],
            "gen_tokens": [
                sum(_gen_tokens(t["message_log"]) for t in ro) for ro in rollout_results
            ],
            "total_tokens": [
                sum(sum(len(m["token_ids"]) for m in t["message_log"]) for t in ro)
                for ro in rollout_results
            ],
        }
    return {
        "capture": any("receipt" in r for r in results),
        "per_rollout_sizes": per_rollout_sizes,
        "rollout_results": rollout_results,
        "traces": traces,
        "turns_per_trace": turns_per_trace,
        "rollout_infos": rollout_infos,
        "trace_kinds": trace_kinds,
        "subagent_traces_per_rollout": subagent_traces_per_rollout,
        "trace_prompt_tokens": trace_prompt_tokens,
        "trace_gen_tokens": trace_gen_tokens,
        "total_rewards": total_rewards,
    }


def _aggregate_group(
    prepared: Mapping[str, Any], reward_penalty_config: Mapping[str, Any] | None
) -> dict[str, Any]:
    rollout_results = prepared["rollout_results"]
    traces = prepared["traces"]
    turns_per_trace = prepared["turns_per_trace"]
    rollout_infos = prepared["rollout_infos"]
    trace_kinds = prepared["trace_kinds"]
    subagent_traces_per_rollout = prepared["subagent_traces_per_rollout"]
    batch_size = len(rollout_results)
    metrics: dict[str, Any] = {
        **calculate_single_metric(
            turns_per_trace, len(turns_per_trace), "turns_per_trace"
        ),
        "turns_per_trace/p95": pct(turns_per_trace, 95),
        "turns_per_trace/p99": pct(turns_per_trace, 99),
        **calculate_single_metric(
            [len(rollout) for rollout in rollout_results],
            batch_size,
            "traces_per_sample",
        ),
        **calculate_single_metric(
            [info["num_compactions"] for info in rollout_infos],
            batch_size,
            "compactions_per_sample",
        ),
        "compaction_rate": sum(
            1 for info in rollout_infos if info["num_compactions"] > 0
        )
        / batch_size,
        **calculate_single_metric(
            [info.get("num_root_compactions", 0) for info in rollout_infos],
            batch_size,
            "compactions_per_sample/root",
        ),
        **calculate_single_metric(
            [info.get("num_subagent_compactions", 0) for info in rollout_infos],
            batch_size,
            "compactions_per_sample/subagent",
        ),
        **calculate_single_metric(
            [info["num_subagent_sessions"] for info in rollout_infos],
            batch_size,
            "subagent_sessions_per_sample",
        ),
        **calculate_single_metric(
            subagent_traces_per_rollout,
            batch_size,
            "subagent_traces_per_sample",
        ),
        "delegation_rate": sum(1 for c in subagent_traces_per_rollout if c > 0)
        / batch_size,
        "mask_sample_rate": sum(1 for info in rollout_infos if info["mask_sample"])
        / batch_size,
        **compaction_rollout_metrics(rollout_infos, prepared["total_rewards"]),
        **trace_kind_metrics(
            trace_kinds,
            turns_per_trace,
            prepared["trace_prompt_tokens"],
            prepared["trace_gen_tokens"],
        ),
        # Rollouts that produced no generation data (masked from gradient).
        "empty_rollout_count": sum(1 for t in traces if t["is_empty_rollout"]),
    }
    sizes = prepared["per_rollout_sizes"]
    if sizes is not None:
        metrics.update(
            {
                **calculate_single_metric(
                    sizes["turns"], batch_size, "turns_per_sample"
                ),
                "turns_per_sample/p95": pct(sizes["turns"], 95),
                "turns_per_sample/p99": pct(sizes["turns"], 99),
                **calculate_single_metric(
                    sizes["total_tokens"], batch_size, "total_tokens_per_sample"
                ),
                **calculate_single_metric(
                    sizes["gen_tokens"], batch_size, "gen_tokens_per_sample"
                ),
            }
        )
        metrics["mean_gen_tokens_per_sample"] = metrics["gen_tokens_per_sample/mean"]
    if prepared.get("capture"):
        # Token-capture receipts carry no message logs: the per-trace panels
        # come from the finalizer's rows (finalize/*), and delegation from the
        # agent's subagent launches.
        metrics = {
            k: v
            for k, v in metrics.items()
            if not k.startswith(_PER_TRACE_METRIC_PREFIXES)
        }
        metrics["delegation_rate"] = (
            sum(1 for info in rollout_infos if info["num_subagent_sessions"] > 0)
            / batch_size
        )
        return metrics
    if reward_penalty_config and "token_ids" in reward_penalty_config:
        metrics.update(
            think_tag_violation_metrics(
                rollout_results, reward_penalty_config["token_ids"]
            )
        )
    return metrics


# Legacy panels built from session-trace message logs, which token-capture
# receipts do not have.
_PER_TRACE_METRIC_PREFIXES = (
    "turns_per_trace",
    "traces_per_sample",
    "subagent_traces_per_sample",
    "compaction/",
    "empty_rollout_count",
)


def _is_nemo_gym_extras(extras: Mapping[str, Any]) -> bool:
    return (
        TRACE_METADATA_KEY in extras
        or "responses_create_params" in extras
        or "instance_config" in extras
    )


def rollout_debug_tags(completions: Sequence[Any]) -> list[str] | None:
    """Per-row JSON provenance for ``rollout_debug_step*.jsonl``, or None.

    One string per completion (= per TQ row), in the record's row order, holding
    the legacy per-trace columns: ``rollout_info`` (the row's rollout, see
    :func:`rollout_debug_info`), ``trace_metadata`` (env segment metadata plus
    ``kind`` / ``turns`` / ``prompt_tokens`` / ``gen_tokens``),
    ``rollout_local_idx``, ``trace_in_rollout_idx`` and ``is_empty_rollout``.
    A JSON string rather than a nested dict because row tags are documented as
    per-sample primitives. None for non-NeMo-Gym records (legacy only logged the
    Gym path).

    Completions arrive grouped by rollout, traces in order (main session
    first), so a rollout starts at every ``trace_in_rollout_idx == 0`` row.
    """
    extras = [dict(c.env_extras or {}) for c in completions]
    if not extras or not any(_is_nemo_gym_extras(e) for e in extras):
        return None
    rows: list[str] = []
    rollout_local_idx = -1
    info: dict[str, Any] | None = None
    for completion, extra in zip(completions, extras):
        sc_metadata = extra.get(TRACE_METADATA_KEY)
        t_idx = int((sc_metadata or {}).get("trace_in_rollout_idx", 0))
        if t_idx == 0 or info is None:
            rollout_local_idx += 1
            info = rollout_debug_info(extra)
        is_empty = bool(extra.get("is_empty_rollout", False))
        message_log = completion.message_log or []
        legacy_md = _legacy_trace_metadata(sc_metadata)
        trace_md = {
            **legacy_md,
            "kind": trace_kind(legacy_md, info, is_empty_rollout=is_empty),
            "turns": _turns(message_log),
            "prompt_tokens": _prompt_tokens(message_log),
            "gen_tokens": _gen_tokens(message_log),
        }
        rows.append(
            json.dumps(
                {
                    "rollout_info": info,
                    "trace_metadata": trace_md,
                    "rollout_local_idx": rollout_local_idx,
                    "trace_in_rollout_idx": t_idx,
                    "is_empty_rollout": is_empty,
                },
                default=str,
            )
        )
    return rows


def capture_rollout_debug_tag(
    rollout_info: Mapping[str, Any],
    *,
    finalizer_kind: str,
    rollout_compacted: bool,
    segment_index: int,
    turns: int,
    prompt_tokens: int,
    gen_tokens: int,
    rollout_local_idx: int,
    trace_in_rollout_idx: int,
    is_empty_rollout: bool,
) -> str:
    """:func:`rollout_debug_tags` row for a token-capture finalizer row.

    Finalizer kinds map onto the legacy names: placeholder -> ``empty``,
    ``subagent`` and ``compaction_summary`` unchanged, ``compaction_segment``
    -> ``pre_compaction``, and the terminal row -> ``post_compaction`` when its
    rollout compacted, else ``uncompacted``.
    """
    if is_empty_rollout or finalizer_kind == "placeholder":
        kind = "empty"
    elif finalizer_kind == "compaction_segment":
        kind = "pre_compaction"
    elif finalizer_kind in ("subagent", "compaction_summary"):
        kind = finalizer_kind
    else:
        kind = "post_compaction" if rollout_compacted else "uncompacted"
    return json.dumps(
        {
            "rollout_info": dict(rollout_info),
            "trace_metadata": {
                "segment_index": segment_index,
                "kind": kind,
                "turns": turns,
                "prompt_tokens": prompt_tokens,
                "gen_tokens": gen_tokens,
            },
            "rollout_local_idx": rollout_local_idx,
            "trace_in_rollout_idx": trace_in_rollout_idx,
            "is_empty_rollout": is_empty_rollout,
        },
        default=str,
    )


def capture_generation_size_metrics(
    gen_tokens: Sequence[int], max_gen_tokens_per_turn: Sequence[int]
) -> dict[str, Any]:
    """Legacy-named generation sizes of a token-capture group, from its finalized rows.

    At rollout time a receipt only has its manifest's per-call deltas (every token
    new since the previous call, tool output included), so the rollout-time
    ``gen_tokens_per_sample`` / ``max_gen_tokens_per_turn`` overcount generation.
    These replace them with the model-generated token counts (all of a rollout's
    rows, subagents included; 0 for a rejected rollout), as legacy logged them.
    """
    if not gen_tokens:
        return {}
    n = len(gen_tokens)
    metrics = {
        **calculate_single_metric(list(gen_tokens), n, "gen_tokens_per_sample"),
        **calculate_single_metric(
            list(max_gen_tokens_per_turn), n, "max_gen_tokens_per_turn"
        ),
        "max_gen_tokens_per_turn/p95": pct(list(max_gen_tokens_per_turn), 95),
    }
    metrics["mean_gen_tokens_per_sample"] = metrics["gen_tokens_per_sample/mean"]
    return metrics


def decode_rollout_debug_tag(tag: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Inverse of :func:`rollout_debug_tags` for one row's ``tags`` dict."""
    raw = (tag or {}).get(ROLLOUT_DEBUG_TAG)
    if not raw:
        return None
    return json.loads(raw) if isinstance(raw, str) else dict(raw)


def _is_sum_count_key(key: str) -> bool:
    return (
        key.endswith("/sum") or key.endswith("/count") or key == "empty_rollout_count"
    )


def aggregate_rollout_metrics_with_sum_counts(
    per_group_metrics: Mapping[str, list[Any]],
    aggregate_fn: Callable[[dict[str, list[Any]]], dict[str, Any]],
) -> dict[str, Any]:
    """Cross-group aggregation with legacy's exact ``/sum`` + ``/count`` handling.

    ``aggregate_fn`` (grpo.aggregate_rollout_metrics) averages every numeric key
    across groups. Legacy summed ``X/sum``, ``X/count`` and
    ``empty_rollout_count`` instead and then derived ``X/mean`` / ``X/rate``
    (:func:`finalize_sum_count_metrics`); this applies exactly that to those keys
    and leaves every other key to ``aggregate_fn``.
    """
    summed: dict[str, Any] = {}
    rest: dict[str, list[Any]] = {}
    for key, values in per_group_metrics.items():
        if _is_sum_count_key(key) and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in values
        ):
            summed[key] = sum(values)
        else:
            rest[key] = list(values)
    aggregated = aggregate_fn(rest) if rest else {}
    if not summed:
        return aggregated
    finalized = finalize_sum_count_metrics(summed)
    return {**aggregated, **finalized}


def iter_rollout_debug_rows(tags: Iterable[Mapping[str, Any] | None]) -> list[dict]:
    """Decoded per-row provenance, ``{}`` for rows that carry none."""
    return [decode_rollout_debug_tag(tag) or {} for tag in tags]


#: Legacy nemo_gym.run_rollouts timer suffixes SC reports per Gym instance.
_ENV_TIMING_SUFFIXES = (
    "await_results",
    "postprocess_results",
    "postprocess_results_pct",
)


def legacy_rollout_timing_aliases(
    env_timing_metrics: Mapping[str, Any], instance_timer_prefix: str, timer_prefix: str
) -> dict[str, Any]:
    """Legacy ``timing/rollout/{await_results,postprocess_results,...}`` names.

    SC times the same NeMo-Gym await / postprocess spans per Gym instance
    (``timing/rollout/shard/<instance>/...``); legacy had one instance and no
    shard segment. Copies, so the SC names stay.
    """
    return {
        f"{timer_prefix}/{suffix}": env_timing_metrics[
            f"{instance_timer_prefix}/{suffix}"
        ]
        for suffix in _ENV_TIMING_SUFFIXES
        if f"{instance_timer_prefix}/{suffix}" in env_timing_metrics
    }
