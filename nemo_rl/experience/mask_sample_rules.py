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
        - name: too_many_compactions   # drop rollouts that compacted more than 3 times
          field: opencode_num_compactions
          gt: 3

A rule carries a dotted ``field`` and exactly one operator key:

* ``equals`` matches when the field's value equals the operand, type-strict
  for booleans (``equals: false`` does not match ``0``, ``""`` or ``null``).
* ``gt`` / ``ge`` / ``lt`` / ``le`` compare numerically. They match only an
  ``int`` or ``float`` value that is not a ``bool``; any other value never
  matches. The operand must be a non-bool number.

A rule with zero or several operator keys, or any key besides ``name`` and
``field``, is a config error. A missing field never matches. A match sets the
same ``instance_config.mask_sample`` flag an environment would, so everything
downstream treats the row as env-flagged: the finalizer's ``mask_sample``
column and the advantage stage's ``final_sample_mask``. Per-group match rates
are logged as ``mask_rules/<name>_rate`` (with ``mask_rules/<name>_reward_mean``
and ``mask_rules/any_rate``), and step totals as ``mask_rules/<name>_count``.

The ``env.should_mask_flagged_samples`` gate drops the environment's own
(possibly too coarse) flags, but not a flag one of these rules set: the rules
are explicit operator choices and are honored regardless. An empty list, the
default, leaves every code path exactly as before.
"""

from __future__ import annotations

import operator as _op
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Optional

from pydantic import (
    BaseModel,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from nemo_rl.data_plane.schema import MASK_SAMPLE

ENV_MASK_SAMPLE_RULES_KEY = "mask_sample_rules"

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

# Strict types: YAML ``"3"`` or ``true`` is never coerced into a number.
_Number = StrictInt | StrictFloat


def _is_number(value: Any) -> bool:
    """True for int/float values; bools are numbers in Python but never here."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class MaskSampleRule(BaseModel, extra="forbid", frozen=True):
    """One ``env.mask_sample_rules`` entry: mask a sample when ``field`` matches.

    Exactly one operator key is set. Which one is decided by the keys the
    rule was written with, so ``equals: null`` is a valid rule.
    """

    # Metric suffix; defaults to ``field`` with dots replaced by underscores.
    name: StrictStr
    # Dotted path into the Gym response, e.g. ``stats.duration_s``.
    field: StrictStr
    equals: Optional[StrictBool | StrictInt | StrictFloat | StrictStr] = None
    gt: Optional[_Number] = None
    ge: Optional[_Number] = None
    lt: Optional[_Number] = None
    le: Optional[_Number] = None

    @model_validator(mode="before")
    @classmethod
    def _default_name(cls, data: Any) -> Any:
        if isinstance(data, Mapping) and not data.get("name"):
            field = data.get("field")
            if isinstance(field, str):
                return {**data, "name": field.replace(".", "_")}
        return data

    @field_validator("field")
    @classmethod
    def _check_field(cls, field: str) -> str:
        if not field.strip() or field != field.strip():
            raise ValueError("field must be a non-empty dotted path")
        return field

    @field_validator("name")
    @classmethod
    def _check_name(cls, name: str) -> str:
        if not name.strip():
            raise ValueError("name must be a non-empty string")
        return name

    @model_validator(mode="after")
    def _check_one_operator(self) -> "MaskSampleRule":
        present = [op for op in OPERATORS if op in self.model_fields_set]
        if len(present) != 1:
            raise ValueError(
                f"needs exactly one of {list(OPERATORS)}, got {present or 'none'}"
            )
        if present[0] != EQUALS and getattr(self, present[0]) is None:
            raise ValueError(f"{present[0]} must be a number, got null")
        return self

    @property
    def operator(self) -> str:
        """The rule's operator key, one of :data:`OPERATORS`."""
        return next(op for op in OPERATORS if op in self.model_fields_set)

    @property
    def operand(self) -> Any:
        """The value ``field`` is compared against."""
        return getattr(self, self.operator)

    @property
    def metric_key(self) -> str:
        return f"mask_rules/{self.name}_rate"


_RULES_ADAPTER = TypeAdapter(list[MaskSampleRule])


def parse_mask_sample_rules(
    env_config: Mapping[str, Any] | None,
) -> tuple[MaskSampleRule, ...]:
    """Read ``env.mask_sample_rules``; absent, ``None`` or ``[]`` means no rules."""
    raw = (env_config or {}).get(ENV_MASK_SAMPLE_RULES_KEY)
    if raw is None:
        return ()
    try:
        rules = _RULES_ADAPTER.validate_python(raw)
    except ValidationError as err:
        raise ValueError(f"env.{ENV_MASK_SAMPLE_RULES_KEY} is invalid: {err}") from err
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
    return bool(_COMPARATORS[rule.operator](value, rule.operand))


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
