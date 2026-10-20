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
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

# MAD/median floor for zMAD (fixed; matches ``flag_reasoning_length_outliers`` default).
_ZMAD_MIN_MAD_REL = 0.015

ProfileBandChannelName = Literal["total", "reasoning", "answer"]
_PROFILE_BAND_CHANNELS: tuple[ProfileBandChannelName, ...] = (
    "total",
    "reasoning",
    "answer",
)


class LengthPenaltyParams(BaseModel, extra="forbid"):
    """Length-adjustment parameters for one agent.

    Used for ``grpo.length_penalty.default`` and for each entry under
    ``grpo.length_penalty.agent_overrides``. Only the keys a user sets in an
    override replace the corresponding ``default`` value. Unknown keys are
    forbidden because a misspelled parameter would silently leave a penalty
    off. See docs/guides/length-penalty.md for the per-key semantics.
    """

    enabled: bool = True
    length_type: Literal["tokens", "chars"] = "tokens"
    top_percentile: float = 0.5
    reasoning_bonus: float = 0.0
    answer_bonus: float = 0.0
    total_bonus: float = 0.0
    longest_reasoning_penalty: float = 0.0
    longest_answer_penalty: float = 0.0
    longest_total_penalty: float = 0.0
    group_reasoning_length_penalty_coeff: float = 0.0
    group_answer_length_penalty_coeff: float = 0.0
    group_total_length_penalty_coeff: float = 0.0
    reasoning_zmad_threshold: float = 0.0
    reasoning_zmad_penalty: float = 0.0
    answer_zmad_threshold: float = 0.0
    answer_zmad_penalty: float = 0.0
    total_zmad_threshold: float = 0.0
    total_zmad_penalty: float = 0.0
    profiled_length_penalty: float = 0.0
    profiled_length_n_std: float = 1.0
    profiled_length_min_samples: int = 2
    pass_rate_length_penalty_weight: float = 0.0
    profile_band_total: bool = False
    profile_band_reasoning: bool = False
    profile_band_answer: bool = False
    group_length_penalty_profile_gate: bool = False
    group_length_penalty_profile_gate_channel: ProfileBandChannelName = "total"
    group_length_penalty_profile_gate_field: Literal["a", "b", "f"] = "a"
    group_length_penalty_profile_gate_positive_only: bool = True


class ProfileBandChannel(BaseModel, extra="forbid"):
    """One ``{a, b, f}`` band: multiplier 1.0 up to ``a``, linear to ``f`` at ``b``.

    ``f`` is bounded to [0, 1]: a value outside it would make rewards negative
    or larger for longer rollouts.
    """

    a: float
    b: float
    f: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_b_gt_a(self) -> "ProfileBandChannel":
        if self.b <= self.a:
            raise ValueError(f"profile band requires b > a, got a={self.a}, b={self.b}")
        return self


class ProfileBandDefaults(BaseModel, extra="forbid"):
    total: Optional[ProfileBandChannel] = None
    reasoning: Optional[ProfileBandChannel] = None
    answer: Optional[ProfileBandChannel] = None


class ProfileBandConfig(BaseModel, extra="forbid"):
    """Global ``{a, b, f}`` bands for datasets without per-row ``profile_band``."""

    enabled: bool = False
    defaults: ProfileBandDefaults = Field(default_factory=ProfileBandDefaults)


class LengthPenaltyConfig(BaseModel, extra="forbid"):
    """``grpo.length_penalty``: per-prompt-group length penalties for NeMo-Gym rollouts.

    ``agent_overrides.<agent>`` may be ``null`` to keep ``default`` for that
    agent. Channels listed under ``profile_band.defaults`` are implicitly
    enabled unless ``default`` sets the matching ``profile_band_*`` flag.
    """

    verbose: bool = False
    default: LengthPenaltyParams = Field(default_factory=LengthPenaltyParams)
    agent_overrides: Optional[dict[str, Optional[LengthPenaltyParams]]] = None
    profile_band: Optional[ProfileBandConfig] = None


# Length adjustments are defined for binary (0/1) single-reward env rewards
# only. Agents already warned about non-binary or multi-component rewards
# (warn once per agent, then skip their prompt groups).
_NON_BINARY_WARNED_AGENTS: set[str] = set()
_BINARY_REWARD_TOL = 1e-6


def _is_binary_reward(value: float) -> bool:
    v = float(value)
    return abs(v) <= _BINARY_REWARD_TOL or abs(v - 1.0) <= _BINARY_REWARD_TOL


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
        item_type = (
            item.get("type", "")
            if isinstance(item, dict)
            else getattr(item, "type", "")
        )
        if item_type == "reasoning":
            summaries = (
                item.get("summary", [])
                if isinstance(item, dict)
                else getattr(item, "summary", [])
            )
            for s in summaries:
                t = s.get("text", "") if isinstance(s, dict) else getattr(s, "text", "")
                reasoning_text += t
        elif item_type == "message":
            role = (
                item.get("role")
                if isinstance(item, dict)
                else getattr(item, "role", None)
            )
            # Some agents (e.g. Gym's gymnasium_agent) append env observations
            # to response.output as user-role messages; count model output only.
            if role not in (None, "assistant"):
                continue
            content = (
                item.get("content", [])
                if isinstance(item, dict)
                else getattr(item, "content", [])
            )
            if isinstance(content, list):
                for c in content:
                    t = (
                        c.get("text", "")
                        if isinstance(c, dict)
                        else getattr(c, "text", "")
                    )
                    answer_text += t
            elif isinstance(content, str):
                answer_text += content

    return reasoning_text, answer_text


def apply_group_length_penalties(
    results: list[dict[str, Any]],
    length_penalty_config: LengthPenaltyConfig | dict[str, Any],
    group_size: int,
    tokenizer: Any = None,
) -> dict[str, float]:
    """Apply per-prompt-group length bonuses/penalties.

    Mutates ``full_result["reward"]`` in place. No-ops when no
    length-adjustment feature is enabled.

    Args:
        results: List of per-generation result dicts, ``group_size`` contiguous
            rows per prompt group.
        length_penalty_config: The ``grpo.length_penalty`` block (a dict is
            validated into :class:`LengthPenaltyConfig`).
        group_size: Number of contiguous rows per prompt group.
        tokenizer: Tokenizer for computing reasoning/answer token counts.

    Returns:
        ``length_penalty/*`` rollout metrics, all per-row means over
        ``results`` (pre-adjustment env reward, reward delta, rollouts whose
        correct reward was wiped to 0, rollouts adjusted, rollouts in groups
        skipped for non-binary or multi-component rewards). Per-row so the
        async path, which calls this once per prompt group and averages the
        per-group values, reports the same numbers as the batched paths.
        Empty when the block enables nothing.
    """
    cfg = (
        length_penalty_config
        if isinstance(length_penalty_config, LengthPenaltyConfig)
        else LengthPenaltyConfig.model_validate(length_penalty_config)
    )
    agents_cfg = cfg.agent_overrides
    global_band = _resolve_global_profile_band(cfg.profile_band)
    verbose = cfg.verbose
    # `enabled` defaults True here AND in the per-group param resolution: a
    # configured `default:` block is intent-to-enable; omitting `enabled` must
    # not silently no-op (and must not depend on unrelated keys being present).
    if not cfg.default.enabled and not agents_cfg and not global_band:
        return {}
    if not results:
        return {}
    if group_size <= 0:
        raise ValueError("group_size must be greater than zero")

    num_gens = group_size
    defaults: dict[str, Any] = cfg.default.model_dump()
    # Channels listed under length_penalty.profile_band.defaults are implicitly
    # enabled — unless the user explicitly configured the channel flag, which
    # always wins (e.g. profile_band_total: false stays false).
    for _ch in global_band:
        if f"profile_band_{_ch}" not in cfg.default.model_fields_set:
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
    all_pass_rate_len_adj = [0.0] * n
    reasoning_lengths = [0] * n
    answer_lengths = [0] * n
    total_lengths = [0] * n
    groups_adjusted = 0
    group_gate_infos: dict[int, dict[str, Any]] = {}
    # Rows whose group passed the binary-rewards check; all other rows are
    # left completely untouched (no adjustment, no clamp, no reward writeback).
    binary_ok = [False] * n
    skipped_non_binary_rows = 0

    for g in range(0, n, num_gens):
        agent_name = agent_names[g]
        group_size = min(num_gens, n - g)
        # Resolve `enabled` first so a disabled agent is skipped silently.
        params = _resolve_agent_params(agent_name, agents_cfg, defaults)
        if params is None:
            continue
        if not params.pop("enabled", True):
            continue
        # Length adjustments are defined for binary (0/1) single-reward env
        # rewards only: every algorithm and the phase-3 clamp assume it, and a
        # multi-reward row must keep reward == sum(reward_components). Skip
        # (and warn once per agent) on graded, negative, or component rewards.
        if any(
            not _is_binary_reward(original_rewards[g + k])
            or results[g + k]["full_result"].get("reward_components")
            for k in range(group_size)
        ):
            skipped_non_binary_rows += group_size
            if agent_name not in _NON_BINARY_WARNED_AGENTS:
                _NON_BINARY_WARNED_AGENTS.add(agent_name)
                logger.warning(
                    f"length penalties require binary (0/1) single-reward env "
                    f"rewards; agent {agent_name} produced non-binary or "
                    f"multi-component rewards — skipping length penalties for "
                    f"its prompt groups"
                )
            continue
        for k in range(group_size):
            binary_ok[g + k] = True
        if any(results[g + k].get("low_effort_applied") for k in range(group_size)):
            continue

        group_lt = params.pop("length_type", "tokens")
        use_tokens = group_lt == "tokens"

        for k in range(group_size):
            idx = g + k
            r_text, a_text = texts[idx]
            if use_tokens and tokenizer is not None:
                reasoning_lengths[idx] = (
                    len(tokenizer.encode(r_text, add_special_tokens=False))
                    if r_text
                    else 0
                )
                answer_lengths[idx] = (
                    len(tokenizer.encode(a_text, add_special_tokens=False))
                    if a_text
                    else 0
                )
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
        ) = _apply_length_penaltyes_and_penalties(
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

        # Pass-rate-scaled length penalty (MAI): -w * rho_q * |y_i| / l_max on
        # correct rollouts, where rho_q is the group's pass rate. Easy prompts
        # (high pass rate) get strong shortening pressure; hard prompts get
        # little; all-wrong groups (rho_q = 0) get exactly none.
        prlp_w = params.get("pass_rate_length_penalty_weight", 0.0)
        if prlp_w > 0.0:
            pass_rate = sum(
                1 for k in range(group_size) if original_rewards[g + k] > 0
            ) / float(group_size)
            if pass_rate > 0.0:
                # l_max is the longest CORRECT rollout, so the penalty is
                # self-normalizing over the set it applies to: the longest
                # correct rollout loses exactly w * rho_q, shorter ones
                # proportionally less — decoupled from wrong-rollout lengths
                # (long wrong rambles must not dilute the pressure).
                l_max = float(
                    max(
                        total_lengths[g + k]
                        for k in range(group_size)
                        if original_rewards[g + k] > 0
                    )
                )
                if l_max > 0.0:
                    for k in range(group_size):
                        idx = g + k
                        if original_rewards[idx] <= 0:
                            continue
                        all_pass_rate_len_adj[idx] = (
                            -prlp_w * pass_rate * total_lengths[idx] / l_max
                        )

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
            low_effort = any(
                results[g + k].get("low_effort_applied") for k in range(group_size)
            )
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
                profiled_adj = (
                    all_profiled_length_adj[idx] if all_adjustments[idx] >= 0 else 0.0
                )
                final = max(
                    0.0,
                    orig
                    + all_adjustments[idx]
                    + profiled_adj
                    + all_pass_rate_len_adj[idx],
                )
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
                    f" pass_rate_len_adj={all_pass_rate_len_adj[idx]:+.4f}"
                    f" final_reward={final:.4f}",
                    flush=True,
                )

        print(f"{'=' * 70}\n", flush=True)

    # Phase 3: apply additive adjustments (binary-verified rows only)
    additive_base_rewards = [0.0] * n
    for i, r in enumerate(results):
        if not binary_ok[i]:
            continue
        # The profiled-length penalty stacks only on rollouts whose group
        # adjustments are non-negative: a rollout already penalized by the
        # group-relative channels should not be double-penalized for the same
        # excess length.
        profiled_adj = all_profiled_length_adj[i] if all_adjustments[i] >= 0 else 0.0
        additive_delta = all_adjustments[i] + profiled_adj + all_pass_rate_len_adj[i]
        additive_base_rewards[i] = original_rewards[i] + additive_delta
        # Rewards here are binary, so any negative value is penalty-created.
        # Length penalties may wipe a reward out but never flip its sign; this
        # also keeps all-wrong groups variance-free (no gradient from a group
        # with no correctness signal).
        if additive_base_rewards[i] < 0:
            additive_base_rewards[i] = 0.0
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
        binary_ok=binary_ok,
    )

    # Rollout metrics: the env reward is overwritten above, so these are the
    # only record of the pass rate and of how far the rewards moved.
    final_rewards = [r["full_result"]["reward"] for r in results]
    return {
        "length_penalty/env_reward_mean": sum(original_rewards) / n,
        "length_penalty/reward_delta_mean": sum(
            f - o for f, o in zip(final_rewards, original_rewards)
        )
        / n,
        "length_penalty/wiped_correct_frac": sum(
            1
            for i in range(n)
            if binary_ok[i] and original_rewards[i] > 0 and final_rewards[i] <= 0.0
        )
        / n,
        "length_penalty/adjusted_frac": sum(
            1 for f, o in zip(final_rewards, original_rewards) if f != o
        )
        / n,
        "length_penalty/skipped_non_binary_frac": skipped_non_binary_rows / n,
    }


def _resolve_global_profile_band(
    pb_cfg: ProfileBandConfig | None,
) -> dict[str, dict[str, Any]]:
    """Parse ``length_penalty.profile_band`` into per-channel {a, b, f} blocks.

    Returns only the channels ("total", "reasoning", "answer") present under
    ``defaults``. Empty dict when the section is absent or disabled.
    """
    if pb_cfg is None or not pb_cfg.enabled:
        return {}
    return {
        ch: block.model_dump()
        for ch in _PROFILE_BAND_CHANNELS
        if (block := getattr(pb_cfg.defaults, ch)) is not None
    }


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
    agents_cfg: dict[str, LengthPenaltyParams | None] | None,
    defaults: dict[str, Any],
    num_gens: int,
    global_band: dict[str, dict[str, Any]] | None = None,
    binary_ok: list[bool] | None = None,
) -> None:
    """Apply per-channel profile_band multipliers to correct rollouts.

    Each enabled channel contributes a multiplier in [0.0, 1.0] derived from the
    per-row {a, b, f} block, falling back to ``length_penalty.profile_band.defaults``
    for channels the row does not provide. Mutates scalar rewards in place.

    Skips any group where the low-effort bypass already replaced the reward
    (parity with Phase 1 of ``apply_group_length_penalties``).
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
            # Only rows whose group passed the binary-rewards check.
            if binary_ok is not None and not binary_ok[idx]:
                continue
            # Gate on the env reward (correct rollouts only).
            if original_rewards[idx] <= 0:
                continue
            # The phase-3 clamp floors the base at 0; multiplying a negative
            # base by m < 1 would RAISE the reward for longer rollouts, so
            # scale only strictly-positive bases.
            if base_rewards[idx] <= 0:
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
    positive_only = bool(
        params.get("group_length_penalty_profile_gate_positive_only", True)
    )
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
    agents_cfg: dict[str, LengthPenaltyParams | None] | None,
    defaults: dict[str, Any],
) -> dict[str, Any] | None:
    """Resolve length bonus parameters for a given agent."""
    if agents_cfg is None:
        return dict(defaults)

    if agent_name not in agents_cfg:
        print(
            f"[length_penalty] WARNING: agent '{agent_name}' not found in "
            f"agent_overrides, falling back to defaults",
            flush=True,
        )
        return dict(defaults)

    overrides = agents_cfg[agent_name]
    if overrides is None:
        return dict(defaults)

    # Only keys the user set under the override replace ``default``.
    merged = dict(defaults)
    merged.update(overrides.model_dump(exclude_unset=True))
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


def _apply_length_penaltyes_and_penalties(
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
        valid = [
            (pi, pos_reasoning[k])
            for k, pi in enumerate(positive_indices)
            if pos_reasoning[k] > 0
        ]
        if valid:
            shortest_pi, _ = min(valid, key=lambda x: x[1])
            if adjusted[shortest_pi] >= top_threshold:
                adjusted[shortest_pi] += reasoning_bonus
                adjustments[shortest_pi] += reasoning_bonus
                r_bonus_per[shortest_pi] = reasoning_bonus

    # Answer bonus: shortest non-empty answer among top scorers
    if answer_bonus > 0:
        valid = [
            (pi, pos_answer[k])
            for k, pi in enumerate(positive_indices)
            if pos_answer[k] > 0
        ]
        if valid:
            shortest_pi, _ = min(valid, key=lambda x: x[1])
            if adjusted[shortest_pi] >= top_threshold:
                adjusted[shortest_pi] += answer_bonus
                adjustments[shortest_pi] += answer_bonus
                a_bonus_per[shortest_pi] = answer_bonus

    # Total bonus: shortest combined (reasoning + answer) among top scorers
    if total_bonus > 0:
        valid = [
            (pi, pos_total[k])
            for k, pi in enumerate(positive_indices)
            if pos_total[k] > 0
        ]
        if valid:
            shortest_pi, _ = min(valid, key=lambda x: x[1])
            if adjusted[shortest_pi] >= top_threshold:
                adjusted[shortest_pi] += total_bonus
                adjustments[shortest_pi] += total_bonus
                t_bonus_per[shortest_pi] = total_bonus

    # Longest reasoning penalty: longest among top-percentile scorers only
    if longest_reasoning_penalty > 0:
        valid = [
            (pi, reasoning_lengths[pi])
            for pi in top_scorer_indices
            if reasoning_lengths[pi] > 0
        ]
        if len(valid) >= 2:
            longest_pi, _ = max(valid, key=lambda x: x[1])
            pen = -longest_reasoning_penalty
            adjusted[longest_pi] += pen
            adjustments[longest_pi] += pen
            r_longest_pen_per[longest_pi] = pen

    # Longest answer penalty: longest among top-percentile scorers only
    if longest_answer_penalty > 0:
        valid = [
            (pi, answer_lengths[pi])
            for pi in top_scorer_indices
            if answer_lengths[pi] > 0
        ]
        if len(valid) >= 2:
            longest_pi, _ = max(valid, key=lambda x: x[1])
            pen = -longest_answer_penalty
            adjusted[longest_pi] += pen
            adjustments[longest_pi] += pen
            a_longest_pen_per[longest_pi] = pen

    # Longest total penalty: longest combined length among top-percentile scorers only
    if longest_total_penalty > 0:
        valid = [
            (pi, total_lengths[pi])
            for pi in top_scorer_indices
            if total_lengths[pi] > 0
        ]
        if len(valid) >= 2:
            longest_pi, _ = max(valid, key=lambda x: x[1])
            pen = -longest_total_penalty
            adjusted[longest_pi] += pen
            adjustments[longest_pi] += pen
            t_longest_pen_per[longest_pi] = pen

    # Independent reasoning, answer, and total length penalties (zero-centered)
    if (
        group_reasoning_length_penalty_coeff > 0
        or group_answer_length_penalty_coeff > 0
        or group_total_length_penalty_coeff > 0
    ):
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
