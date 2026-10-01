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
"""Controller-side ``mask_sample`` rules evaluated on the Gym ``/run`` response.

Environments can already ask GRPO to drop a sample from the loss by setting
``instance_config.mask_sample`` in their response. Many environments never
do, even when the response says the sample is not a clean policy outcome
(the verifier did not finish, the agent harness was cut off). These rules let
the RL side derive that flag from fields the response already carries, with
no environment change::

    env:
      mask_sample_rules:
        - name: eval_incomplete        # verifier timed out / errored -> reward is not a policy signal
          field: evaluation_completed
          equals: false
        - name: harness_unfinished     # agent harness hit sandbox_timeout or an exec error
          field: opencode_finished
          equals: false
        - name: too_many_compactions   # optional: drop rollouts that compacted more than 3 times
          field: opencode_num_compactions
          gt: 3

A rule carries a dotted ``field`` and exactly one operator key:

* ``equals`` matches when the field's value equals the operand. This is
  type-strict for booleans, so ``equals: false`` does not match ``0``, ``""``
  or ``null``.
* ``gt`` / ``ge`` / ``lt`` / ``le`` compare numerically. They match only when
  the field's value is an ``int`` or ``float`` that is not a ``bool`` (``true``
  never satisfies ``gt: 0``); a string, ``null``, list or mapping in the
  field never matches. The operand itself must be a non-bool number.

A rule with zero or several operator keys, or any other key besides ``name``
and ``field``, is a config error. A missing field never matches, whatever the
operator. A match sets the same ``instance_config.mask_sample`` flag an
environment would, so everything downstream is unchanged: the finalizer's
``mask_sample`` column, the advantage stage's ``final_sample_mask``, the
baseline's ``valid_mask`` and ``train/num_mask_sample_filtered``. Per-group
match rates are logged as ``mask_rules/<name>_rate`` (with
``mask_rules/<name>_reward_mean`` and ``mask_rules/any_rate``); the metric
names do not depend on the operator.

The ``too_many_compactions`` example above is a policy choice, not a
requirement: in the v1 compaction runs Gerald kept rollouts that hit the
compaction cap (the fork's ``max_compaction`` termination) in the loss with
their verifier reward, i.e. scored rather than masked them, so that running
out of compaction budget was a trained outcome rather than a free one. Use a
comparator rule only when you want the opposite policy; the numeric fields
(``opencode_num_compactions``, ``opencode_num_compaction_attempts``,
``opencode_num_model_calls``, ...) are reported by the agent either way.

The rules run after the ``env.should_mask_flagged_samples`` gate: that gate
drops the environment's own (possibly too coarse) flags, while these rules
are explicit operator choices and are honored regardless. An empty list, the
default, leaves every code path exactly as before.
"""

from __future__ import annotations

import operator as _op
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from nemo_rl.data_plane.schema import MASK_SAMPLE

ENV_MASK_SAMPLE_RULES_KEY = "mask_sample_rules"
_SCALAR_TYPES = (bool, int, float, str, type(None))

EQUALS = "equals"
# Operator key -> numeric predicate (field value, operand). ``equals`` is handled
# separately because it is type-strict rather than numeric.
_COMPARATORS: dict[str, Callable[[Any, Any], bool]] = {
    "gt": _op.gt,
    "ge": _op.ge,
    "lt": _op.lt,
    "le": _op.le,
}
OPERATORS: tuple[str, ...] = (EQUALS, *_COMPARATORS)
_RULE_KEYS = frozenset({"name", "field", *OPERATORS})


def _is_number(value: Any) -> bool:
    """True for int/float operands; bools are numbers in Python but never here."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass(frozen=True)
class MaskSampleRule:
    """Mask a sample when ``field`` in the Gym response satisfies the rule.

    ``operator`` is one of :data:`OPERATORS`; ``equals`` holds the operand for
    every operator (the value to equal, or the numeric threshold to compare
    against). The field is named for the original, equality-only form so
    existing positional constructors ``MaskSampleRule(name, field, value)``
    keep meaning "equals".
    """

    name: str
    field: str
    equals: Any
    operator: str = EQUALS

    def __post_init__(self) -> None:
        if self.operator not in OPERATORS:
            raise ValueError(
                f"MaskSampleRule {self.name!r}: unknown operator {self.operator!r}; "
                f"expected one of {list(OPERATORS)}"
            )
        if self.operator != EQUALS and not _is_number(self.equals):
            raise ValueError(
                f"MaskSampleRule {self.name!r}: `{self.operator}` needs a numeric "
                f"operand, got {type(self.equals).__name__}"
            )

    @property
    def operand(self) -> Any:
        """The value ``field`` is compared against (alias of ``equals``)."""
        return self.equals

    @property
    def metric_key(self) -> str:
        return f"mask_rules/{self.name}_rate"


def parse_mask_sample_rules(
    env_config: Mapping[str, Any] | None,
) -> tuple[MaskSampleRule, ...]:
    """Read ``env.mask_sample_rules``; absent, ``None`` or ``[]`` means no rules."""
    raw = (env_config or {}).get(ENV_MASK_SAMPLE_RULES_KEY)
    if raw is None:
        return ()
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError(
            f"env.{ENV_MASK_SAMPLE_RULES_KEY} must be a list of rules, got {type(raw).__name__}"
        )
    rules: list[MaskSampleRule] = []
    for index, entry in enumerate(raw):
        prefix = f"env.{ENV_MASK_SAMPLE_RULES_KEY}[{index}]"
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"{prefix} must be a mapping, got {type(entry).__name__}"
            )
        field = entry.get("field")
        if not isinstance(field, str) or not field.strip() or field != field.strip():
            raise ValueError(f"{prefix}.field must be a non-empty dotted path")
        unknown = set(entry) - _RULE_KEYS
        if unknown:
            raise ValueError(f"{prefix} has unknown keys: {sorted(unknown)}")
        present = [key for key in OPERATORS if key in entry]
        if len(present) != 1:
            raise ValueError(
                f"{prefix} needs exactly one of {list(OPERATORS)}, got {present or 'none'}"
            )
        op = present[0]
        operand = entry[op]
        if op == EQUALS:
            if not isinstance(operand, _SCALAR_TYPES):
                raise ValueError(
                    f"{prefix}.equals must be a scalar, got {type(operand).__name__}"
                )
        elif not _is_number(operand):
            raise ValueError(
                f"{prefix}.{op} must be a number (int or float, not bool), "
                f"got {type(operand).__name__}"
            )
        name = entry.get("name") or field.replace(".", "_")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{prefix}.name must be a non-empty string")
        rules.append(MaskSampleRule(name=name, field=field, equals=operand, operator=op))
    names = [rule.name for rule in rules]
    if len(set(names)) != len(names):
        raise ValueError(
            f"env.{ENV_MASK_SAMPLE_RULES_KEY} has duplicate rule names: {names}"
        )
    return tuple(rules)


def _lookup(result: Mapping[str, Any], dotted: str) -> tuple[bool, Any]:
    """Follow a dotted path through nested mappings; (found, value)."""
    node: Any = result
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return False, None
        node = node[part]
    return True, node


def _equals(value: Any, equals: Any) -> bool:
    # Type-strict for booleans: `equals: false` must not match a 0 or an empty string.
    if isinstance(value, bool) or isinstance(equals, bool):
        return isinstance(value, bool) and isinstance(equals, bool) and value == equals
    return value == equals


def _matches(value: Any, rule: MaskSampleRule) -> bool:
    if rule.operator == EQUALS:
        return _equals(value, rule.equals)
    # Comparators only look at real numbers: a bool, str, None, list or mapping
    # in the field is "not comparable" and never matches (NaN compares False too).
    if not _is_number(value):
        return False
    return bool(_COMPARATORS[rule.operator](value, rule.equals))


def matching_rules(
    full_result: Mapping[str, Any], rules: Sequence[MaskSampleRule]
) -> list[str]:
    """Names of the rules the Gym response satisfies (no mutation)."""
    matched: list[str] = []
    for rule in rules:
        found, value = _lookup(full_result, rule.field)
        if found and _matches(value, rule):
            matched.append(rule.name)
    return matched


def apply_mask_sample_rules(
    full_result: dict[str, Any], rules: Sequence[MaskSampleRule]
) -> list[str]:
    """Set ``instance_config.mask_sample`` on ``full_result`` when a rule matches.

    Returns the names of the matching rules (empty when none match or ``rules``
    is empty, in which case ``full_result`` is not touched).
    """
    matched = matching_rules(full_result, rules)
    if matched:
        instance_config = full_result.get("instance_config")
        if not isinstance(instance_config, dict):
            instance_config = {}
            full_result["instance_config"] = instance_config
        instance_config[MASK_SAMPLE] = True
    return matched


def mask_rule_step_metrics(
    counts: Mapping[str, int],
    rules: Sequence[MaskSampleRule],
    *,
    reward_sums: Mapping[str, float],
    any_count: int,
    rollouts_seen: int,
) -> dict[str, float]:
    """Step-level rule metrics, logged by the controller under ``train/``.

    ``mask_rules/rollouts_seen``, and per rule ``mask_rules/<name>_count``,
    ``mask_rules/<name>_frac`` (of rollouts seen) and ``mask_rules/<name>_reward_mean``
    (when it matched), plus ``mask_rules/any_count`` / ``mask_rules/any_frac``.
    Empty when no rules are configured or no rollout was seen.
    """
    if not rules or rollouts_seen <= 0:
        return {}
    out: dict[str, float] = {"mask_rules/rollouts_seen": float(rollouts_seen)}
    for rule in rules:
        count = counts.get(rule.name, 0)
        out[f"mask_rules/{rule.name}_count"] = float(count)
        out[f"mask_rules/{rule.name}_frac"] = count / rollouts_seen
        if count and rule.name in reward_sums:
            out[f"mask_rules/{rule.name}_reward_mean"] = reward_sums[rule.name] / count
    out["mask_rules/any_count"] = float(any_count)
    out["mask_rules/any_frac"] = any_count / rollouts_seen
    return out


def mask_rule_metrics(
    counts: Mapping[str, int],
    rules: Sequence[MaskSampleRule],
    num_results: int,
    *,
    reward_sums: Mapping[str, float] | None = None,
    any_count: int = 0,
) -> dict[str, float]:
    """Per-group rule metrics attached to the group's rollout_metrics.

    ``mask_rules/<name>_rate`` for every configured rule (0.0 when unmatched),
    ``mask_rules/<name>_reward_mean`` when the rule matched (mean reward of the
    rows it masked), and ``mask_rules/any_rate`` (rows masked by at least one rule).
    """
    if not rules or num_results <= 0:
        return {}
    out: dict[str, float] = {}
    for rule in rules:
        count = counts.get(rule.name, 0)
        out[rule.metric_key] = count / num_results
        if count and reward_sums is not None and rule.name in reward_sums:
            out[f"mask_rules/{rule.name}_reward_mean"] = reward_sums[rule.name] / count
    out["mask_rules/any_rate"] = any_count / num_results
    return out
