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
"""Controller-side reward penalty rules evaluated on the Gym ``/run`` response.

A rollout can be degenerate in ways its task reward does not see: a model call
that ran into the per-call ``max_tokens`` cap (a text loop), or a session that
issues the same tool call over and over. Such a rollout is usually masked when
it fails (context-limited, agent timeout), so it teaches nothing, and its loop is
rewarded when it succeeds. These rules give it a fixed reward and keep it in the
loss::

    env:
      reward_penalty_rules:
        - name: capped_call            # a call cut at the per-call max_tokens
          field: opencode_max_call_output_tokens
          at_least: 65000
          reward: -1.0
          unmask_rules: [context_limited, agent_timeout, agent_timeout_exception]
          agents: [swe_rebench_opencode_sandboxed_agent]   # optional; absent = every agent

A rule matches when the rollout's agent is listed (or ``agents`` is absent) and
the dotted ``field`` exists in the Gym response as a number (not a bool) of at
least ``at_least``; a missing field never matches. The first matching rule, in
list order, replaces the reward with its ``reward``. A matched rollout that is
masked only by ``env.mask_sample_rules`` named in the rule's ``unmask_rules`` is
unmasked, so the penalty trains. A rollout the environment itself masked, or
one masked by any other rule (an OOM, an unfinished verifier), keeps its mask
and its reward: an infrastructure failure is never penalized. Apply the rules
after every other reward change so their reward is the one that trains. An
empty list, the default, changes nothing.

A rule may match on ``masked_by`` instead of ``field`` / ``at_least``: the
rollout matches when one of the named ``env.mask_sample_rules`` masked it. With
the same names in ``unmask_rules``, a rollout the policy did not finish trains
instead of being masked; ``reward: null`` keeps the reward the verifier gave it::

        - name: unfinished_trains
          masked_by: [context_limited, agent_timeout, agent_timeout_exception]
          reward: null
          unmask_rules: [context_limited, agent_timeout, agent_timeout_exception]

Step metrics: with rules, the per-agent metrics of
:mod:`nemo_rl.experience.agent_step_metrics` add ``<agent>/<rule>_frac``
(penalized), ``<agent>/<rule>_kept_masked_frac`` (matched, left masked),
``<agent>/raw_reward_mean`` (before the rules) and the max / mean of every rule
field.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

from nemo_rl.data_plane.schema import MASK_SAMPLE
from nemo_rl.experience.mask_sample_rules import _lookup

ENV_REWARD_PENALTY_RULES_KEY = "reward_penalty_rules"
# The reward a rollout had before a penalty replaced it; the streamed per-row and
# the group conversion see the same result, so the second must not read -1 as raw.
RAW_REWARD_KEY = "reward_before_penalty_rules"
_RULE_KEYS = {
    "name",
    "field",
    "at_least",
    "masked_by",
    "reward",
    "unmask_rules",
    "agents",
}


@dataclass(frozen=True)
class RewardPenaltyRule:
    """Set the reward to ``reward`` (None keeps the task reward) for a rollout the rule matches.

    It matches when ``field`` in the Gym response is >= ``at_least`` or, for a
    ``masked_by`` rule, when one of those mask rules masked the rollout.
    """

    name: str
    reward: Optional[float]
    unmask_rules: frozenset[str]
    field: Optional[str] = None
    at_least: Optional[float] = None
    masked_by: frozenset[str] = frozenset()
    agents: Optional[frozenset[str]] = None

    def applies_to(self, agent_name: str) -> bool:
        return self.agents is None or agent_name in self.agents

    def matches(
        self,
        full_result: Mapping[str, Any],
        agent_name: str,
        matched_mask_rules: Sequence[str] = (),
    ) -> bool:
        if not self.applies_to(agent_name):
            return False
        if self.masked_by:
            return not self.masked_by.isdisjoint(matched_mask_rules)
        found, value = _lookup(full_result, self.field)
        return (
            found
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value >= self.at_least
        )


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a number, got {value!r}")
    return float(value)


def _names(value: Any, where: str) -> frozenset[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{where} must be a list of names")
    if not all(isinstance(name, str) and name.strip() for name in value):
        raise ValueError(f"{where} must contain non-empty strings")
    return frozenset(value)


def parse_reward_penalty_rules(
    env_config: Mapping[str, Any] | None,
    *,
    mask_rule_names: Sequence[str] = (),
) -> tuple[RewardPenaltyRule, ...]:
    """Read ``env.reward_penalty_rules``; absent, ``None`` or ``[]`` means no rules.

    ``mask_rule_names`` are the configured ``env.mask_sample_rules`` names; an
    ``unmask_rules`` entry must be one of them, so a typo cannot silently leave
    every penalized rollout masked.
    """
    raw = (env_config or {}).get(ENV_REWARD_PENALTY_RULES_KEY)
    if raw is None:
        return ()
    key = f"env.{ENV_REWARD_PENALTY_RULES_KEY}"
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError(f"{key} must be a list of rules, got {type(raw).__name__}")
    known_mask_rules = set(mask_rule_names)
    rules: list[RewardPenaltyRule] = []
    for index, entry in enumerate(raw):
        where = f"{key}[{index}]"
        if not isinstance(entry, Mapping):
            raise ValueError(f"{where} must be a mapping, got {type(entry).__name__}")
        unknown = set(entry) - _RULE_KEYS
        if unknown:
            raise ValueError(f"{where} has unknown keys: {sorted(unknown)}")
        by_mask = "masked_by" in entry
        required = {"name", "reward"} | (set() if by_mask else {"field", "at_least"})
        missing = required - set(entry)
        if missing:
            raise ValueError(f"{where} needs {sorted(missing)}")
        if by_mask and {"field", "at_least"} & set(entry):
            raise ValueError(
                f"{where} matches on either masked_by or field / at_least, not both"
            )
        name = entry["name"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{where}.name must be a non-empty string")
        field = None if by_mask else entry["field"]
        if not by_mask and (
            not isinstance(field, str) or not field.strip() or field != field.strip()
        ):
            raise ValueError(f"{where}.field must be a non-empty dotted path")
        masked_by = (
            _names(entry["masked_by"], f"{where}.masked_by") if by_mask else frozenset()
        )
        if by_mask and not masked_by:
            raise ValueError(f"{where}.masked_by must name at least one mask rule")
        unmask_rules = _names(entry.get("unmask_rules") or [], f"{where}.unmask_rules")
        for list_key, rule_names in (
            ("masked_by", masked_by),
            ("unmask_rules", unmask_rules),
        ):
            unknown_mask_rules = rule_names - known_mask_rules
            if unknown_mask_rules:
                raise ValueError(
                    f"{where}.{list_key} names rules env.mask_sample_rules does not "
                    f"define: {sorted(unknown_mask_rules)}"
                )
        agents = entry.get("agents")
        rules.append(
            RewardPenaltyRule(
                name=name,
                reward=None
                if entry["reward"] is None
                else _number(entry["reward"], f"{where}.reward"),
                unmask_rules=unmask_rules,
                field=field,
                at_least=None
                if by_mask
                else _number(entry["at_least"], f"{where}.at_least"),
                masked_by=masked_by,
                agents=None if agents is None else _names(agents, f"{where}.agents"),
            )
        )
    names = [rule.name for rule in rules]
    if len(set(names)) != len(names):
        raise ValueError(f"{key} has duplicate rule names: {names}")
    return tuple(rules)


def is_env_masked(full_result: Mapping[str, Any]) -> bool:
    """Whether the environment itself masked the rollout (before any operator rule)."""
    instance_config = full_result.get("instance_config")
    return bool(full_result.get(MASK_SAMPLE)) or (
        isinstance(instance_config, Mapping) and bool(instance_config.get(MASK_SAMPLE))
    )


@dataclass(frozen=True)
class PenaltyOutcome:
    """The first matching rule (None when none matched) and whether it was applied."""

    rule: Optional[str] = None
    applied: bool = False


def apply_reward_penalty_rules(
    full_result: dict[str, Any],
    rules: Sequence[RewardPenaltyRule],
    *,
    agent_name: str,
    env_masked: bool,
    matched_mask_rules: Sequence[str],
) -> PenaltyOutcome:
    """Apply the first matching rule to ``full_result`` in place.

    ``env_masked`` is :func:`is_env_masked` read before ``env.mask_sample_rules``
    ran; ``matched_mask_rules`` are the names those rules matched. Idempotent: a
    second call on the same result applies the same reward and mask.
    """
    rule = next(
        (r for r in rules if r.matches(full_result, agent_name, matched_mask_rules)),
        None,
    )
    if rule is None:
        return PenaltyOutcome()
    if env_masked or not set(matched_mask_rules) <= rule.unmask_rules:
        return PenaltyOutcome(rule=rule.name, applied=False)
    if rule.reward is not None:
        full_result.setdefault(RAW_REWARD_KEY, float(full_result.get("reward") or 0.0))
        full_result["reward"] = rule.reward
    instance_config = full_result.get("instance_config")
    if isinstance(instance_config, dict) and instance_config.get(MASK_SAMPLE):
        instance_config[MASK_SAMPLE] = False
    return PenaltyOutcome(rule=rule.name, applied=True)
