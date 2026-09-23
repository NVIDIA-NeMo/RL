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

A rule matches when the dotted ``field`` exists in the rollout's Gym response
and equals ``equals`` (type-strict for booleans, so ``false`` does not match
``0``). A match sets the same ``instance_config.mask_sample`` flag an
environment would, so everything downstream is unchanged: the finalizer's
``mask_sample`` column, the advantage stage's ``final_sample_mask``, the
baseline's ``valid_mask`` and ``train/num_mask_sample_filtered``. A missing
field never matches. Per-group match rates are logged as
``mask_rules/<name>_rate``.

The rules run after the ``env.should_mask_flagged_samples`` gate: that gate
drops the environment's own (possibly too coarse) flags, while these rules
are explicit operator choices and are honored regardless. An empty list, the
default, leaves every code path exactly as before.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from nemo_rl.data_plane.schema import MASK_SAMPLE

ENV_MASK_SAMPLE_RULES_KEY = "mask_sample_rules"
_SCALAR_TYPES = (bool, int, float, str, type(None))


@dataclass(frozen=True)
class MaskSampleRule:
    """Mask a sample when ``field`` in the Gym response equals ``equals``."""

    name: str
    field: str
    equals: Any

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
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"env.{ENV_MASK_SAMPLE_RULES_KEY}[{index}] must be a mapping, got {type(entry).__name__}"
            )
        field = entry.get("field")
        if not isinstance(field, str) or not field.strip() or field != field.strip():
            raise ValueError(
                f"env.{ENV_MASK_SAMPLE_RULES_KEY}[{index}].field must be a non-empty dotted path"
            )
        if "equals" not in entry:
            raise ValueError(
                f"env.{ENV_MASK_SAMPLE_RULES_KEY}[{index}] needs an `equals` value"
            )
        equals = entry["equals"]
        if not isinstance(equals, _SCALAR_TYPES):
            raise ValueError(
                f"env.{ENV_MASK_SAMPLE_RULES_KEY}[{index}].equals must be a scalar, got {type(equals).__name__}"
            )
        unknown = set(entry) - {"name", "field", "equals"}
        if unknown:
            raise ValueError(
                f"env.{ENV_MASK_SAMPLE_RULES_KEY}[{index}] has unknown keys: {sorted(unknown)}"
            )
        name = entry.get("name") or field.replace(".", "_")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"env.{ENV_MASK_SAMPLE_RULES_KEY}[{index}].name must be a non-empty string"
            )
        rules.append(MaskSampleRule(name=name, field=field, equals=equals))
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


def _matches(value: Any, equals: Any) -> bool:
    # Type-strict for booleans: `equals: false` must not match a 0 or an empty string.
    if isinstance(value, bool) or isinstance(equals, bool):
        return isinstance(value, bool) and isinstance(equals, bool) and value == equals
    return value == equals


def matching_rules(
    full_result: Mapping[str, Any], rules: Sequence[MaskSampleRule]
) -> list[str]:
    """Names of the rules the Gym response satisfies (no mutation)."""
    matched: list[str] = []
    for rule in rules:
        found, value = _lookup(full_result, rule.field)
        if found and _matches(value, rule.equals):
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


def mask_rule_metrics(
    counts: Mapping[str, int], rules: Sequence[MaskSampleRule], num_results: int
) -> dict[str, float]:
    """Per-group ``mask_rules/<name>_rate`` for every configured rule (0.0 when unmatched)."""
    if not rules or num_results <= 0:
        return {}
    return {rule.metric_key: counts.get(rule.name, 0) / num_results for rule in rules}
