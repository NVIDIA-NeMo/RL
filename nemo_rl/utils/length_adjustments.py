# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Per-prompt-group length bonuses/penalties for rollout rewards.

Rewards conciseness among high-quality generations by applying:
1. A flat bonus to the shortest generation among top scorers in each prompt group.
2. Optional flat penalties on the longest reasoning / longest answer among
   top-percentile scorers (with at least two eligible rollouts to compare).
3. Independent zero-centered penalties for reasoning and answer length.
4. Optional Iglewicz–Hoaglin modified-Z (MAD) high-side outliers among positive
   scorers. Config keys: ``reasoning_zmad_threshold``, ``reasoning_zmad_penalty``,
   ``answer_zmad_threshold``, ``answer_zmad_penalty``.
   If ``reasoning_zmad_threshold`` or ``answer_zmad_threshold`` is ≤ 0, that
   channel is off and its penalty is ignored (no flagging). If threshold > 0
   but the matching penalty is 0, nothing is subtracted.

Supports per-agent filtering and parameter overrides via config.
"""

from __future__ import annotations

import logging
import statistics
from typing import Any

logger = logging.getLogger(__name__)

# MAD/median floor for zMAD (fixed; matches ``flag_reasoning_length_outliers`` default).
_ZMAD_MIN_MAD_REL = 0.015

_PARAM_KEYS = (
    "enabled",
    "reasoning_bonus",
    "answer_bonus",
    "total_bonus",
    "longest_reasoning_penalty",
    "longest_answer_penalty",
    "longest_total_penalty",
    "top_percentile",
    "group_reasoning_length_penalty_coeff",
    "group_answer_length_penalty_coeff",
    "group_total_length_penalty_coeff",
    "length_type",
    "reasoning_zmad_threshold",
    "reasoning_zmad_penalty",
    "answer_zmad_threshold",
    "answer_zmad_penalty",
    "total_zmad_threshold",
    "total_zmad_penalty",
    "profiled_length_penalty",
    "profiled_length_n_std",
    "profiled_length_min_samples",
    "profile_band_total",
    "profile_band_reasoning",
    "profile_band_answer",
    "group_length_penalty_profile_gate",
    "group_length_penalty_profile_gate_channel",
    "group_length_penalty_profile_gate_field",
    "group_length_penalty_profile_gate_positive_only",
)

# Param keys that should be merged as bools rather than floats.
_BOOL_PARAM_KEYS = frozenset({
    "enabled",
    "profile_band_total",
    "profile_band_reasoning",
    "profile_band_answer",
    "group_length_penalty_profile_gate",
    "group_length_penalty_profile_gate_positive_only",
})

_STR_PARAM_KEYS = frozenset({
    "length_type",
    "group_length_penalty_profile_gate_channel",
    "group_length_penalty_profile_gate_field",
})

def _extract_reasoning_and_answer_text(result: dict[str, Any]) -> tuple[str, str]:
    """Extract reasoning and answer text from the Response API output items."""
    fr = result.get("full_result", {})
    response_obj = fr.get("response", {})
    output_items = (
        response_obj.get("output", [])
        if isinstance(response_obj, dict)
        else getattr(response_obj, "output", [])
    )

    reasoning_text = ""
    answer_text = ""
    for item in output_items:
        item_type = item.get("type", "") if isinstance(item, dict) else getattr(item, "type", "")
        if item_type == "reasoning":
            summaries = item.get("summary", []) if isinstance(item, dict) else getattr(item, "summary", [])
            for s in summaries:
                t = s.get("text", "") if isinstance(s, dict) else getattr(s, "text", "")
                reasoning_text += t
        elif item_type == "message":
            content = item.get("content", []) if isinstance(item, dict) else getattr(item, "content", [])
            if isinstance(content, list):
                for c in content:
                    t = c.get("text", "") if isinstance(c, dict) else getattr(c, "text", "")
                    answer_text += t
            elif isinstance(content, str):
                answer_text += content

    return reasoning_text, answer_text


def apply_group_length_adjustments(
    results: list[dict[str, Any]],
    master_config: dict[str, Any],
    tokenizer: Any = None,
) -> None:
    """Apply per-prompt-group length bonuses/penalties.

    Reads ``grpo.length_bonus`` for configuration and mutates
    ``full_result["reward"]`` in place. No-ops when no length-adjustment
    feature is enabled.

    Args:
        results: List of per-generation result dicts.
        master_config: Full training config dict.
        tokenizer: Tokenizer for computing reasoning/answer token counts.
    """
    grpo_config = master_config.get("grpo", {})
    length_cfg = dict(grpo_config.get("length_bonus", {}) or {})
    if not length_cfg:
        return

    default_cfg = length_cfg.get("default", {})
    agents_cfg = length_cfg.get("agent_overrides")
    global_band = _resolve_global_profile_band(length_cfg.get("profile_band"))
    verbose = bool(length_cfg.get("verbose", False))
    if not default_cfg.get("enabled", False) and not agents_cfg and not global_band:
        return

    num_gens = master_config["grpo"]["num_generations_per_prompt"]
    defaults: dict[str, Any] = {}
    for k in _PARAM_KEYS:
        if k == "length_type":
            defaults[k] = default_cfg.get(k, "tokens")
        elif k == "group_length_penalty_profile_gate_channel":
            defaults[k] = default_cfg.get(k, "total")
        elif k == "group_length_penalty_profile_gate_field":
            defaults[k] = default_cfg.get(k, "a")
        elif k == "enabled":
            defaults[k] = default_cfg.get(k, True)
        elif k == "group_length_penalty_profile_gate_positive_only":
            defaults[k] = default_cfg.get(k, True)
        elif k in _BOOL_PARAM_KEYS:
            defaults[k] = default_cfg.get(k, False)
        elif k == "profiled_length_min_samples":
            defaults[k] = default_cfg.get(k, 2)
        elif k == "profiled_length_n_std":
            defaults[k] = default_cfg.get(k, 1.0)
        else:
            defaults[k] = default_cfg.get(k, 0.0)
    defaults.setdefault("top_percentile", 0.2)
    # Channels listed under length_bonus.profile_band.defaults are implicitly
    # enabled; per-agent overrides can still disable them.
    for _ch in global_band:
        defaults[f"profile_band_{_ch}"] = True

    n = len(results)
    original_rewards = [r["full_result"]["reward"] for r in results]
    agent_names = [r["agent_ref"]["name"] for r in results]

    # Extract text once; lengths computed per-group based on resolved length_type
    texts: list[tuple[str, str]] = []
    for r in results:
        texts.append(_extract_reasoning_and_answer_text(r))

    # Phase 1: calculate all adjustments per-group
    all_adjustments = [0.0] * n
    all_reasoning_adj = [0.0] * n
    all_answer_adj = [0.0] * n
    all_total_adj = [0.0] * n
    all_reasoning_bonus = [0.0] * n
    all_answer_bonus = [0.0] * n
    all_total_bonus = [0.0] * n
    all_reasoning_longest_pen = [0.0] * n
    all_answer_longest_pen = [0.0] * n
    all_total_longest_pen = [0.0] * n
    all_zmad_reasoning_adj = [0.0] * n
    all_zmad_answer_adj = [0.0] * n
    all_zmad_total_adj = [0.0] * n
    all_profiled_length_adj = [0.0] * n
    reasoning_lengths = [0] * n
    answer_lengths = [0] * n
    total_lengths = [0] * n
    groups_adjusted = 0
    group_gate_infos: dict[int, dict[str, Any]] = {}

    for g in range(0, n, num_gens):
        agent_name = agent_names[g]
        group_size = min(num_gens, n - g)
        if any(results[g + k].get("low_effort_applied") for k in range(group_size)):
            continue
        params = _resolve_agent_params(agent_name, agents_cfg, defaults)
        if params is None:
            continue
        if not params.pop("enabled", True):
            continue

        group_lt = params.pop("length_type", "tokens")
        use_tokens = group_lt == "tokens"

        for k in range(group_size):
            idx = g + k
            r_text, a_text = texts[idx]
            if use_tokens and tokenizer is not None:
                reasoning_lengths[idx] = len(tokenizer.encode(r_text, add_special_tokens=False)) if r_text else 0
                answer_lengths[idx] = len(tokenizer.encode(a_text, add_special_tokens=False)) if a_text else 0
            else:
                reasoning_lengths[idx] = len(r_text)
                answer_lengths[idx] = len(a_text)

        group_reasoning = reasoning_lengths[g : g + num_gens]
        group_answer = answer_lengths[g : g + num_gens]
        group_total = [r + a for r, a in zip(group_reasoning, group_answer)]
        total_lengths[g : g + group_size] = group_total[:group_size]
        group_rewards = original_rewards[g : g + num_gens]
        gate_info = _group_length_profile_gate_info(
            band=_merged_profile_band(results[g].get("profile_band"), global_band),
            params=params,
            rewards=group_rewards[:group_size],
            reasoning_lengths=group_reasoning[:group_size],
            answer_lengths=group_answer[:group_size],
            total_lengths=group_total[:group_size],
        )
        group_gate_infos[g] = gate_info
        if gate_info["enabled"] and not gate_info["open"]:
            params["group_reasoning_length_penalty_coeff"] = 0.0
            params["group_answer_length_penalty_coeff"] = 0.0
            params["group_total_length_penalty_coeff"] = 0.0
        (
            _,
            adjustments,
            reasoning_adjs,
            answer_adjs,
            total_adjs,
            r_bonus,
            a_bonus,
            t_bonus,
            r_lpen,
            a_lpen,
            t_lpen,
            zmad_r_adj,
            zmad_a_adj,
            zmad_t_adj,
        ) = _apply_length_bonuses_and_penalties(
            group_rewards, group_reasoning, group_answer, group_total, **params
        )

        for k in range(len(adjustments)):
            all_adjustments[g + k] = adjustments[k]
            all_reasoning_adj[g + k] = reasoning_adjs[k]
            all_answer_adj[g + k] = answer_adjs[k]
            all_total_adj[g + k] = total_adjs[k]
            all_reasoning_bonus[g + k] = r_bonus[k]
            all_answer_bonus[g + k] = a_bonus[k]
            all_total_bonus[g + k] = t_bonus[k]
            all_reasoning_longest_pen[g + k] = r_lpen[k]
            all_answer_longest_pen[g + k] = a_lpen[k]
            all_total_longest_pen[g + k] = t_lpen[k]
            all_zmad_reasoning_adj[g + k] = zmad_r_adj[k]
            all_zmad_answer_adj[g + k] = zmad_a_adj[k]
            all_zmad_total_adj[g + k] = zmad_t_adj[k]

        if any(a != 0.0 for a in adjustments):
            groups_adjusted += 1

        # Profiled length penalty: penalize rollouts longer than mean + n_std of
        # passing profiled lengths for this prompt. If fewer than min_samples
        # profiled rollouts passed, the profiled model found the problem hard
        # and its lengths carry no budget signal — apply no penalty at all.
        plp = params.get("profiled_length_penalty", 0.0)
        if plp > 0.0:
            p_rewards = results[g].get("profiled_rewards")
            p_lengths = results[g].get("profiled_output_lengths")
            if p_rewards is not None and p_lengths is not None:
                min_samples = int(params.get("profiled_length_min_samples", 2))
                passing = [l for r, l in zip(p_rewards, p_lengths) if r > 0]
                if len(passing) >= min_samples:
                    mean_l = statistics.mean(passing)
                    std_l = statistics.stdev(passing) if len(passing) >= 2 else 0.0
                    n_std = float(params.get("profiled_length_n_std", 1.0))
                    threshold = mean_l + n_std * std_l
                    for k in range(group_size):
                        idx = g + k
                        if total_lengths[idx] >= threshold:
                            all_profiled_length_adj[idx] = -plp

    # Phase 2: debug print (only when verbose flag is set)
    if verbose:
        num_groups = n // num_gens if num_gens > 0 else 0
        print(f"\n{'=' * 70}", flush=True)
        print(
            f"[Rollout] {n} samples, {num_groups} groups, {groups_adjusted} adjusted"
            f" default longest_reasoning_penalty={defaults['longest_reasoning_penalty']}"
            f" longest_answer_penalty={defaults['longest_answer_penalty']}",
            flush=True,
        )

        for g in range(0, n, num_gens):
            agent_name = agent_names[g]
            group_size = min(num_gens, n - g)
            low_effort = any(results[g + k].get("low_effort_applied") for k in range(group_size))
            params = _resolve_agent_params(agent_name, agents_cfg, defaults)
            skipped = params is None
            disabled = params is not None and not params.get("enabled", True)

            if low_effort:
                print(
                    f"\n  group {g // num_gens} agent={agent_name} [low_effort — skipped]",
                    flush=True,
                )
            elif skipped:
                print(
                    f"\n  group {g // num_gens} agent={agent_name} [skipped]"
                    f" (default longest_reasoning_penalty={defaults['longest_reasoning_penalty']}"
                    f" longest_answer_penalty={defaults['longest_answer_penalty']})",
                    flush=True,
                )
            elif disabled:
                print(
                    f"\n  group {g // num_gens} agent={agent_name} [disabled]"
                    f" longest_reasoning_penalty={params['longest_reasoning_penalty']}"
                    f" longest_answer_penalty={params['longest_answer_penalty']}",
                    flush=True,
                )
            else:
                lt = params.get("length_type", "tokens")
                unit = "tok" if lt == "tokens" else "chr"
                print(
                    f"\n  group {g // num_gens} agent={agent_name}"
                    f" length_type={unit}"
                    f" reasoning_bonus={params['reasoning_bonus']} answer_bonus={params['answer_bonus']}"
                    f" total_bonus={params['total_bonus']}"
                    f" longest_reasoning_penalty={params['longest_reasoning_penalty']}"
                    f" longest_answer_penalty={params['longest_answer_penalty']}"
                    f" longest_total_penalty={params['longest_total_penalty']}"
                    f" top_pct={params['top_percentile']}"
                    f" reasoning_coeff={params['group_reasoning_length_penalty_coeff']}"
                    f" answer_coeff={params['group_answer_length_penalty_coeff']}"
                    f" total_coeff={params['group_total_length_penalty_coeff']}"
                    f" reasoning_zmad_threshold={params['reasoning_zmad_threshold']}"
                    f" reasoning_zmad_penalty={params['reasoning_zmad_penalty']}"
                    f" answer_zmad_threshold={params['answer_zmad_threshold']}"
                    f" answer_zmad_penalty={params['answer_zmad_penalty']}"
                    f" total_zmad_threshold={params['total_zmad_threshold']}"
                    f" total_zmad_penalty={params['total_zmad_penalty']}"
                    f" profiled_length_penalty={params['profiled_length_penalty']}"
                    f" profiled_length_n_std={params['profiled_length_n_std']}"
                    f" profiled_length_min_samples={params['profiled_length_min_samples']}",
                    flush=True,
                )
                gate = group_gate_infos.get(g)
                if gate and gate["enabled"]:
                    print(
                        f"    profile_gate channel={gate['channel']} field={gate['field']}"
                        f" positive_only={gate['positive_only']}"
                        f" mean={gate['mean']}"
                        f" limit={gate['limit']}"
                        f" open={gate['open']}"
                        f" reason={gate['reason']}",
                        flush=True,
                    )
            for k in range(group_size):
                idx = g + k
                orig = original_rewards[idx]
                profiled_adj = all_profiled_length_adj[idx] if all_adjustments[idx] >= 0 else 0.0
                final = orig + all_adjustments[idx] + profiled_adj
                print(
                    f"    [{k}] reward={orig:.4f}"
                    f" reasoning_len={reasoning_lengths[idx]}"
                    f" reasoning_adj={all_reasoning_adj[idx]:+.4f}"
                    f" reasoning_bonus={all_reasoning_bonus[idx]:+.4f}"
                    f" longest_reasoning_penalty_adj={all_reasoning_longest_pen[idx]:+.4f}"
                    f" answer_len={answer_lengths[idx]}"
                    f" answer_adj={all_answer_adj[idx]:+.4f}"
                    f" answer_bonus={all_answer_bonus[idx]:+.4f}"
                    f" longest_answer_penalty_adj={all_answer_longest_pen[idx]:+.4f}"
                    f" total_len={total_lengths[idx]}"
                    f" total_adj={all_total_adj[idx]:+.4f}"
                    f" total_bonus={all_total_bonus[idx]:+.4f}"
                    f" longest_total_penalty_adj={all_total_longest_pen[idx]:+.4f}"
                    f" zmad_r={all_zmad_reasoning_adj[idx]:+.4f}"
                    f" zmad_a={all_zmad_answer_adj[idx]:+.4f}"
                    f" zmad_t={all_zmad_total_adj[idx]:+.4f}"
                    f" profiled_len_adj={all_profiled_length_adj[idx]:+.4f}"
                    f" final_reward={final:.4f}",
                    flush=True,
                )

        print(f"{'=' * 70}\n", flush=True)

    # Phase 3: apply additive adjustments
    additive_base_rewards = [0.0] * n
    for i, r in enumerate(results):
        # The profiled-length penalty stacks only on rollouts whose group
        # adjustments are non-negative: a rollout already penalized by the
        # group-relative channels should not be double-penalized for the same
        # excess length.
        profiled_adj = all_profiled_length_adj[i] if all_adjustments[i] >= 0 else 0.0
        additive_delta = all_adjustments[i] + profiled_adj
        additive_base_rewards[i] = original_rewards[i] + additive_delta
        r["full_result"]["reward"] = additive_base_rewards[i]

    # Phase 4: apply per-prompt profile_band multipliers (correct rollouts only).
    _apply_profile_band_multipliers(
        results=results,
        original_rewards=original_rewards,
        base_rewards=additive_base_rewards,
        total_lengths=total_lengths,
        reasoning_lengths=reasoning_lengths,
        answer_lengths=answer_lengths,
        agent_names=agent_names,
        agents_cfg=agents_cfg,
        defaults=defaults,
        num_gens=num_gens,
        global_band=global_band,
    )



def _resolve_global_profile_band(pb_cfg: Any) -> dict[str, dict[str, Any]]:
    """Parse ``length_bonus.profile_band`` into per-channel {a, b, f} blocks.

    Returns only channels ("total", "reasoning", "answer") present under
    ``defaults`` with a complete, well-formed block. Empty dict when the
    section is absent or disabled.
    """
    if not isinstance(pb_cfg, dict) or not pb_cfg.get("enabled", False):
        return {}
    pb_defaults = pb_cfg.get("defaults")
    if not isinstance(pb_defaults, dict):
        return {}
    band: dict[str, dict[str, Any]] = {}
    for ch in ("total", "reasoning", "answer"):
        ch_cfg = pb_defaults.get(ch)
        if not isinstance(ch_cfg, dict):
            continue
        a, b, f = ch_cfg.get("a"), ch_cfg.get("b"), ch_cfg.get("f")
        if a is None or b is None or f is None or b <= a:
            logger.warning(
                f"length_bonus.profile_band.defaults.{ch} is malformed "
                f"(a={a}, b={b}, f={f}); ignoring this channel"
            )
            continue
        band[ch] = {"a": a, "b": b, "f": f}
    return band


def _merged_profile_band(
    row_band: dict[str, Any] | None,
    global_band: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Merge per-row profile_band over global defaults (row channel wins)."""
    if not global_band:
        return row_band
    if not row_band:
        return dict(global_band)
    merged: dict[str, Any] = dict(global_band)
    merged.update(row_band)
    return merged


def _apply_profile_band_multipliers(
    results: list[dict[str, Any]],
    original_rewards: list[float],
    base_rewards: list[float],
    total_lengths: list[int],
    reasoning_lengths: list[int],
    answer_lengths: list[int],
    agent_names: list[str],
    agents_cfg: dict[str, Any] | None,
    defaults: dict[str, Any],
    num_gens: int,
    global_band: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Apply per-channel profile_band multipliers to correct rollouts.

    Each enabled channel contributes a multiplier in [0.0, 1.0] derived from the
    per-row {a, b, f} block, falling back to ``length_bonus.profile_band.defaults``
    for channels the row does not provide. Mutates scalar rewards in place.

    Skips any group where the low-effort bypass already replaced the reward
    (parity with Phase 1 of ``apply_group_length_adjustments``).
    """
    n = len(results)
    global_band = global_band or {}
    for g in range(0, n, num_gens):
        agent_name = agent_names[g]
        group_size = min(num_gens, n - g)
        if any(results[g + k].get("low_effort_applied") for k in range(group_size)):
            continue
        params = _resolve_agent_params(agent_name, agents_cfg, defaults)
        if params is None:
            continue
        use_total = bool(params.get("profile_band_total", False))
        use_rsn = bool(params.get("profile_band_reasoning", False))
        use_ans = bool(params.get("profile_band_answer", False))
        if not (use_total or use_rsn or use_ans):
            continue
        band = _merged_profile_band(results[g].get("profile_band"), global_band)
        if not band:
            continue
        ch_total = band.get("total") if use_total else None
        ch_rsn = band.get("reasoning") if use_rsn else None
        ch_ans = band.get("answer") if use_ans else None
        for k in range(group_size):
            idx = g + k
            # Gate on the env reward (correct rollouts only).
            if original_rewards[idx] <= 0:
                continue
            current_reward = base_rewards[idx]

            total_m = _band_multiplier(total_lengths[idx], ch_total)
            total_delta = current_reward * total_m - current_reward
            current_reward += total_delta

            reasoning_m = _band_multiplier(reasoning_lengths[idx], ch_rsn)
            reasoning_delta = current_reward * reasoning_m - current_reward
            current_reward += reasoning_delta

            answer_m = _band_multiplier(answer_lengths[idx], ch_ans)
            answer_delta = current_reward * answer_m - current_reward
            current_reward += answer_delta

            results[idx]["full_result"]["reward"] = current_reward


def _band_multiplier(rl: int, ch: dict[str, Any] | None) -> float:
    """Per-channel profile_band reward multiplier.

    Returns 1.0 if the channel block is missing or malformed (no-op).
    Otherwise:
        rl <= a    -> 1.0
        a < rl < b -> linear interpolation from 1.0 down to f
        rl >= b    -> f
    """
    if not ch:
        return 1.0
    a = ch.get("a")
    b = ch.get("b")
    f = ch.get("f")
    if a is None or b is None or f is None or b <= a:
        return 1.0
    if rl <= a:
        return 1.0
    if rl >= b:
        return float(f)
    return 1.0 - (rl - a) / (b - a) * (1.0 - float(f))


def _group_length_profile_gate_info(
    *,
    band: dict[str, Any] | None,
    params: dict[str, Any],
    rewards: list[float],
    reasoning_lengths: list[int],
    answer_lengths: list[int],
    total_lengths: list[int],
) -> dict[str, Any]:
    """Prompt-level gate for group-relative length penalties.

    When enabled, group-relative coefficients are applied only if the mean
    rollout length exceeds a prompt-specific threshold from ``profile_band``.
    """
    enabled = bool(params.get("group_length_penalty_profile_gate", False))
    channel = str(params.get("group_length_penalty_profile_gate_channel", "total"))
    field = str(params.get("group_length_penalty_profile_gate_field", "a"))
    positive_only = bool(params.get("group_length_penalty_profile_gate_positive_only", True))
    info = {
        "enabled": enabled,
        "open": True,
        "channel": channel,
        "field": field,
        "positive_only": positive_only,
        "mean": None,
        "limit": None,
        "reason": "disabled",
    }
    if not enabled:
        return info

    limit = _profile_band_numeric_value(band, channel, field)
    info["limit"] = limit
    if limit is None:
        info["open"] = False
        info["reason"] = "missing_profile_limit"
        return info

    length_by_channel = {
        "reasoning": reasoning_lengths,
        "answer": answer_lengths,
        "total": total_lengths,
    }
    candidate_lengths = length_by_channel.get(channel)
    if candidate_lengths is None:
        info["open"] = False
        info["reason"] = "unknown_channel"
        return info

    if positive_only:
        lengths = [l for r, l in zip(rewards, candidate_lengths) if r > 0]
    else:
        lengths = list(candidate_lengths)
    if not lengths:
        info["open"] = False
        info["reason"] = "no_lengths"
        return info

    mean_length = float(statistics.mean(lengths))
    info["mean"] = mean_length
    info["open"] = mean_length > limit
    info["reason"] = "mean_gt_limit" if info["open"] else "mean_le_limit"
    return info


def _profile_band_numeric_value(
    band: dict[str, Any] | None, channel: str, field: str
) -> float | None:
    if not isinstance(band, dict):
        return None
    channel_block = band.get(channel)
    if not isinstance(channel_block, dict):
        return None
    value = channel_block.get(field)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _resolve_agent_params(
    agent_name: str,
    agents_cfg: dict[str, Any] | None,
    defaults: dict[str, Any],
) -> dict[str, Any] | None:
    """Resolve length bonus parameters for a given agent."""
    if agents_cfg is None:
        return dict(defaults)

    if agent_name not in agents_cfg:
        print(
            f"[length_adjustments] WARNING: agent '{agent_name}' not found in "
            f"agent_overrides, falling back to defaults",
            flush=True,
        )
        return dict(defaults)

    overrides = agents_cfg[agent_name]
    if overrides is None:
        return dict(defaults)

    merged = dict(defaults)
    for key in _PARAM_KEYS:
        if key in overrides:
            if key in _STR_PARAM_KEYS:
                merged[key] = overrides[key]
            elif key in _BOOL_PARAM_KEYS:
                merged[key] = bool(overrides[key])
            else:
                merged[key] = float(overrides[key])
    return merged


def _zmad_local_outliers(
    lengths: list[int], z_thresh: float, min_mad_rel: float
) -> set[int]:
    """Indices into ``lengths`` with Iglewicz–Hoaglin modified Z (MAD) > ``z_thresh``."""
    if len(lengths) < 2:
        return set()
    med = statistics.median(lengths)
    devs = [abs(x - med) for x in lengths]
    mad = statistics.median(devs)
    if mad == 0:
        return set()
    if min_mad_rel > 0 and mad / max(med, 1e-9) < min_mad_rel:
        return set()
    out: set[int] = set()
    for k, x in enumerate(lengths):
        mz = 0.6745 * (x - med) / mad
        if mz > z_thresh:
            out.add(k)
    return out


def _apply_length_bonuses_and_penalties(
    rewards: list[float],
    reasoning_lengths: list[int],
    answer_lengths: list[int],
    total_lengths: list[int],
    reasoning_bonus: float,
    answer_bonus: float,
    total_bonus: float,
    longest_reasoning_penalty: float,
    longest_answer_penalty: float,
    longest_total_penalty: float,
    top_percentile: float,
    group_reasoning_length_penalty_coeff: float,
    group_answer_length_penalty_coeff: float,
    group_total_length_penalty_coeff: float,
    reasoning_zmad_threshold: float = 0.0,
    reasoning_zmad_penalty: float = 0.0,
    answer_zmad_threshold: float = 0.0,
    answer_zmad_penalty: float = 0.0,
    total_zmad_threshold: float = 0.0,
    total_zmad_penalty: float = 0.0,
    **_kwargs,
) -> tuple[
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
]:
    """Apply length-based bonuses/penalties to a single prompt group.

    Only samples with reward > 0 participate. Samples with reward <= 0
    are left untouched and excluded from weight computation.

    1. Reasoning bonus: shortest non-empty reasoning among positive scorers; awarded only if that
       sample satisfies ``reward >= top_threshold``.
    2. Answer bonus: same pattern for shortest non-empty answer.
    3. Total bonus: same pattern for shortest combined (reasoning + answer) length.
    4. Longest penalties: subtract from longest non-empty reasoning / answer / total among
       top-percentile scorers; needs at least two eligible rollouts to compare.
    5. Independent zero-centered penalties for reasoning, answer, and total lengths.
    6. Optional MAD modified-Z outliers among positives for reasoning, answer, and total lengths.
       Each channel runs only if its threshold is > 0; otherwise that channel is disabled and
       its penalty is ignored.
    """
    n = len(rewards)
    zeros = [0.0] * n
    if n < 2:
        return (
            list(rewards),
            list(zeros),
            list(zeros),
            list(zeros),
            list(zeros),
            list(zeros),
            list(zeros),
            list(zeros),
            list(zeros),
            list(zeros),
            list(zeros),
            list(zeros),
            list(zeros),
            list(zeros),
        )

    positive_indices = [i for i in range(n) if rewards[i] > 0]
    if len(positive_indices) < 2:
        return (
            list(rewards),
            list(zeros),
            list(zeros),
            list(zeros),
            list(zeros),
            list(zeros),
            list(zeros),
            list(zeros),
            list(zeros),
            list(zeros),
            list(zeros),
            list(zeros),
            list(zeros),
            list(zeros),
        )

    adjusted = list(rewards)
    adjustments = [0.0] * n
    reasoning_adjs = [0.0] * n
    answer_adjs = [0.0] * n
    total_adjs = [0.0] * n
    r_bonus_per = [0.0] * n
    a_bonus_per = [0.0] * n
    t_bonus_per = [0.0] * n
    r_longest_pen_per = [0.0] * n
    a_longest_pen_per = [0.0] * n
    t_longest_pen_per = [0.0] * n
    zmad_reasoning_adj = [0.0] * n
    zmad_answer_adj = [0.0] * n
    zmad_total_adj = [0.0] * n

    pos_reasoning = [reasoning_lengths[i] for i in positive_indices]
    pos_answer = [answer_lengths[i] for i in positive_indices]
    pos_total = [total_lengths[i] for i in positive_indices]
    pos_rewards = [rewards[i] for i in positive_indices]

    sorted_scores = sorted(pos_rewards, reverse=True)
    threshold_idx = max(0, int(len(pos_rewards) * top_percentile) - 1)
    top_threshold = sorted_scores[threshold_idx]
    top_scorer_indices = [i for i in positive_indices if rewards[i] >= top_threshold]

    # Reasoning bonus: shortest non-empty reasoning among top scorers
    if reasoning_bonus > 0:
        valid = [(pi, pos_reasoning[k]) for k, pi in enumerate(positive_indices) if pos_reasoning[k] > 0]
        if valid:
            shortest_pi, _ = min(valid, key=lambda x: x[1])
            if adjusted[shortest_pi] >= top_threshold:
                adjusted[shortest_pi] += reasoning_bonus
                adjustments[shortest_pi] += reasoning_bonus
                r_bonus_per[shortest_pi] = reasoning_bonus

    # Answer bonus: shortest non-empty answer among top scorers
    if answer_bonus > 0:
        valid = [(pi, pos_answer[k]) for k, pi in enumerate(positive_indices) if pos_answer[k] > 0]
        if valid:
            shortest_pi, _ = min(valid, key=lambda x: x[1])
            if adjusted[shortest_pi] >= top_threshold:
                adjusted[shortest_pi] += answer_bonus
                adjustments[shortest_pi] += answer_bonus
                a_bonus_per[shortest_pi] = answer_bonus

    # Total bonus: shortest combined (reasoning + answer) among top scorers
    if total_bonus > 0:
        valid = [(pi, pos_total[k]) for k, pi in enumerate(positive_indices) if pos_total[k] > 0]
        if valid:
            shortest_pi, _ = min(valid, key=lambda x: x[1])
            if adjusted[shortest_pi] >= top_threshold:
                adjusted[shortest_pi] += total_bonus
                adjustments[shortest_pi] += total_bonus
                t_bonus_per[shortest_pi] = total_bonus

    # Longest reasoning penalty: longest among top-percentile scorers only
    if longest_reasoning_penalty > 0:
        valid = [(pi, reasoning_lengths[pi]) for pi in top_scorer_indices if reasoning_lengths[pi] > 0]
        if len(valid) >= 2:
            longest_pi, _ = max(valid, key=lambda x: x[1])
            pen = -longest_reasoning_penalty
            adjusted[longest_pi] += pen
            adjustments[longest_pi] += pen
            r_longest_pen_per[longest_pi] = pen

    # Longest answer penalty: longest among top-percentile scorers only
    if longest_answer_penalty > 0:
        valid = [(pi, answer_lengths[pi]) for pi in top_scorer_indices if answer_lengths[pi] > 0]
        if len(valid) >= 2:
            longest_pi, _ = max(valid, key=lambda x: x[1])
            pen = -longest_answer_penalty
            adjusted[longest_pi] += pen
            adjustments[longest_pi] += pen
            a_longest_pen_per[longest_pi] = pen

    # Longest total penalty: longest combined length among top-percentile scorers only
    if longest_total_penalty > 0:
        valid = [(pi, total_lengths[pi]) for pi in top_scorer_indices if total_lengths[pi] > 0]
        if len(valid) >= 2:
            longest_pi, _ = max(valid, key=lambda x: x[1])
            pen = -longest_total_penalty
            adjusted[longest_pi] += pen
            adjustments[longest_pi] += pen
            t_longest_pen_per[longest_pi] = pen

    # Independent reasoning, answer, and total length penalties (zero-centered)
    if group_reasoning_length_penalty_coeff > 0 or group_answer_length_penalty_coeff > 0 or group_total_length_penalty_coeff > 0:
        reasoning_weights = _compute_length_weights(pos_reasoning)
        answer_weights = _compute_length_weights(pos_answer)
        total_weights = _compute_length_weights(pos_total)

        for k, i in enumerate(positive_indices):
            r_adj = reasoning_weights[k] * group_reasoning_length_penalty_coeff
            a_adj = answer_weights[k] * group_answer_length_penalty_coeff
            t_adj = total_weights[k] * group_total_length_penalty_coeff
            combined_adj = r_adj + a_adj + t_adj
            reasoning_adjs[i] = r_adj
            answer_adjs[i] = a_adj
            total_adjs[i] = t_adj
            if combined_adj != 0:
                adjusted[i] += combined_adj
                adjustments[i] += combined_adj

    zm = _ZMAD_MIN_MAD_REL
    ztr = float(reasoning_zmad_threshold)
    zpr = float(reasoning_zmad_penalty)
    zta = float(answer_zmad_threshold)
    zpa = float(answer_zmad_penalty)
    ztt = float(total_zmad_threshold)
    zpt = float(total_zmad_penalty)

    if len(positive_indices) >= 2:
        if ztr > 0.0:
            if zpr != 0.0:
                for local_k in _zmad_local_outliers(pos_reasoning, ztr, zm):
                    gi = positive_indices[local_k]
                    adjusted[gi] -= zpr
                    adjustments[gi] -= zpr
                    zmad_reasoning_adj[gi] -= zpr
        if zta > 0.0:
            if zpa != 0.0:
                for local_k in _zmad_local_outliers(pos_answer, zta, zm):
                    gi = positive_indices[local_k]
                    adjusted[gi] -= zpa
                    adjustments[gi] -= zpa
                    zmad_answer_adj[gi] -= zpa
        if ztt > 0.0:
            if zpt != 0.0:
                for local_k in _zmad_local_outliers(pos_total, ztt, zm):
                    gi = positive_indices[local_k]
                    adjusted[gi] -= zpt
                    adjustments[gi] -= zpt
                    zmad_total_adj[gi] -= zpt

    return (
        adjusted,
        adjustments,
        reasoning_adjs,
        answer_adjs,
        total_adjs,
        r_bonus_per,
        a_bonus_per,
        t_bonus_per,
        r_longest_pen_per,
        a_longest_pen_per,
        t_longest_pen_per,
        zmad_reasoning_adj,
        zmad_answer_adj,
        zmad_total_adj,
    )


def _compute_length_weights(lengths: list[int]) -> list[float]:
    """Compute zero-centered weights where shorter = higher weight.

    Returns all zeros if all lengths are equal.
    """
    max_len = max(lengths)
    min_len = min(lengths)

    if max_len == min_len:
        return [0.0] * len(lengths)

    span = max_len - min_len
    raw_weights = [1.0 - ((length - min_len) / span) for length in lengths]
    mean_weight = sum(raw_weights) / len(raw_weights)
    return [w - mean_weight for w in raw_weights]
