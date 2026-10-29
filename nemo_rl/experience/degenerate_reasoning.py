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
"""Degenerate-reasoning ("loop") penalty on NeMo-Gym rollout results.

``reward_penalties.penalize_degenerate_reasoning`` scores each arriving rollout's thinking with the vendored loop
detector (``nemo_rl.experience.loop_detector``): one text per model turn, the reasoning before each tool call. A
flagged rollout gets reward 0 and is trained on instead of masked for running out of context: it is exempt from the
mask rules named in ``reward_penalties.degenerate_reasoning_unmask_rules`` (default ``context_limited``), and its rows
that reach ``max_seq_len`` are not marked truncated, so ``overlong_filtering`` keeps them (token capture carries this
per rollout through the recovery ledger to the finalizer as ``keep_truncated``). Every other mask still applies to
it (agent timeout / OOM, incomplete evaluation, empty rollouts, NaN logprobs, the sequence-logprob gate), and every
rollout the detector does not flag is masked exactly as before. Every scored result carries
``degenerate_reasoning`` (bool) and ``degenerate_reasoning_score``, which the per-agent metrics log as
``<agent>/degenerate_reasoning/mean``.

Scoring runs once per rollout, when the result arrives (before effort shaping and any recovery record), in a small
spawned process pool so the pure-Python detector does not hold the controller's event loop.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from traceback import format_exc
from typing import Any, Optional

FLAG_KEY = "degenerate_reasoning"
SCORE_KEY = "degenerate_reasoning_score"
REWARD_BEFORE_KEY = "reward_before_degenerate_penalty"
_DEFAULT_UNMASK_RULES = ("context_limited",)
_POOL: Optional[ProcessPoolExecutor] = None
_POOL_WORKERS = 4


def _cfg(reward_penalty_config: Any, key: str, default: Any = None) -> Any:
    if reward_penalty_config is None:
        return default
    if isinstance(reward_penalty_config, dict):
        value = reward_penalty_config.get(key, default)
    else:
        value = getattr(reward_penalty_config, key, default)
    return default if value is None else value


def enabled(reward_penalty_config: Any) -> bool:
    return bool(_cfg(reward_penalty_config, "penalize_degenerate_reasoning", False))


def applies_to(reward_penalty_config: Any, agent_ref: Any, full_result: dict[str, Any]) -> bool:
    """Whether this rollout is scored: the penalty is on, the agent is allowed, and the rollout is not empty."""
    if not enabled(reward_penalty_config) or full_result.get("is_empty_rollout"):
        return False
    allowed = _cfg(reward_penalty_config, "degenerate_reasoning_agents")
    if not allowed:
        return True
    name = agent_ref.get("name") if isinstance(agent_ref, dict) else agent_ref
    return name in set(allowed)


def reasoning_turns(output: Any) -> list[str]:
    """Reasoning text before each tool call (one entry per model turn) from a Responses output list, split and joined
    exactly as in the turns the detector was fitted on."""
    turns: list[str] = []
    current: list[str] = []
    for item in output or []:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "reasoning":
            parts = list(item.get("summary") or []) + list(item.get("content") or [])
            current.append("\n".join(p.get("text") or "" for p in parts if isinstance(p, dict)))
        elif kind == "function_call":
            turns.append("".join(current))
            current = []
    turns.append("".join(current))
    return [t for t in turns if t]


def score_turns(turns: list[str]) -> tuple[Optional[float], bool]:
    """Detector verdict for one rollout; imported lazily so pool workers load the detector themselves."""
    from nemo_rl.experience import loop_detector

    score, flagged = loop_detector.score_rollout(turns)
    return (None if score is None else float(score)), bool(flagged)


def apply_verdict(
    full_result: dict[str, Any],
    score: Optional[float],
    flagged: bool,
    reward_penalty_config: Any,
) -> bool:
    """Record the verdict on the result and zero its reward when flagged. Returns whether the reward was zeroed."""
    threshold = _cfg(reward_penalty_config, "degenerate_reasoning_threshold")
    if threshold is not None:
        flagged = score is not None and score >= float(threshold)
    full_result[SCORE_KEY] = score
    full_result[FLAG_KEY] = bool(flagged)
    if not flagged:
        return False
    full_result[REWARD_BEFORE_KEY] = full_result.get("reward")
    full_result["reward"] = 0.0
    return True


def unmask_rule_names(reward_penalty_config: Any) -> frozenset[str]:
    return frozenset(_cfg(reward_penalty_config, "degenerate_reasoning_unmask_rules", _DEFAULT_UNMASK_RULES))


def rules_for_result(full_result: dict[str, Any], rules: Sequence[Any], unmask: frozenset[str]) -> Sequence[Any]:
    """The mask rules to apply to one result: all of them, minus ``unmask`` for a rollout this penalty flagged."""
    if not unmask or not full_result.get(FLAG_KEY):
        return rules
    return [rule for rule in rules if getattr(rule, "name", None) not in unmask]


def keeps_truncated(full_result: Optional[dict[str, Any]]) -> bool:
    """Whether this result's rows that reach ``max_seq_len`` still train (the penalty flagged it)."""
    return bool((full_result or {}).get(FLAG_KEY))


def _pool() -> ProcessPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ProcessPoolExecutor(max_workers=_POOL_WORKERS, mp_context=multiprocessing.get_context("spawn"))
    return _POOL


async def check_result(full_result: dict[str, Any], agent_ref: Any, reward_penalty_config: Any) -> None:
    """Score one arriving result and apply the penalty. A detector failure leaves the result unchanged."""
    if not applies_to(reward_penalty_config, agent_ref, full_result):
        return
    turns = reasoning_turns((full_result.get("response") or {}).get("output"))
    try:
        loop = asyncio.get_running_loop()
        score, flagged = await loop.run_in_executor(_pool(), score_turns, turns)
    except Exception:
        print(f"[degenerate_reasoning] detector failed: {format_exc()}", file=sys.stderr)
        return
    apply_verdict(full_result, score, flagged, reward_penalty_config)
