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
"""Step-level rollout metrics per Gym agent, named ``<agent>/<metric>``.

The per-group rollout metrics do not reach the controller's logger on the
token-capture path, so the rollout manager records every finished rollout here
and the controller drains one step's worth with :meth:`AgentStepStats.pop_metrics`
(logged as ``train/<agent>/<metric>``). ``<agent>`` is the row's
``agent_ref.name``, NeMo-RL's namespace for per-agent rollout metrics. Per agent:

- ``rollouts``: rollouts finished since the last step;
- ``reward_mean``: their reward after every reward change;
- ``env_masked_frac``: the share masked by the environment or by
  ``env.mask_sample_rules`` (overlong and logprob-error masking come later and
  are not counted), and ``unmasked_reward_mean``: the reward of the others;
- ``<field>_max`` / ``<field>_mean`` of the numeric result fields in
  :data:`AGENT_RESULT_FIELDS` and of every ``env.reward_penalty_rules`` field,
  for the agents whose results carry them;
- with ``env.reward_penalty_rules`` only: ``raw_reward_mean`` (the reward before
  the rules), ``<rule>_frac`` (penalized) and ``<rule>_kept_masked_frac``
  (matched, left masked).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from nemo_rl.data_plane.schema import MASK_SAMPLE
from nemo_rl.experience.mask_sample_rules import _lookup
from nemo_rl.experience.reward_penalty_rules import (
    RAW_REWARD_KEY,
    PenaltyOutcome,
    RewardPenaltyRule,
)

# Per-session fields of the Gym OpenCode agent's result: the longest model call
# (output + reasoning tokens), the most uses of one identical tool call, and the
# longest run of one identical call in a row. Other agents do not carry them.
AGENT_RESULT_FIELDS = (
    "opencode_max_call_output_tokens",
    "opencode_max_tool_call_repeat",
    "opencode_max_tool_call_streak",
)


def _is_masked(full_result: Mapping[str, Any]) -> bool:
    """The flag batch building reads: ``instance_config.mask_sample``."""
    instance_config = full_result.get("instance_config")
    return isinstance(instance_config, Mapping) and bool(
        instance_config.get(MASK_SAMPLE)
    )


class AgentStepStats:
    """Step-level, per-agent accumulator for the ``<agent>/*`` metrics."""

    def __init__(
        self,
        penalty_rules: Sequence[RewardPenaltyRule] = (),
        fields: Iterable[str] = AGENT_RESULT_FIELDS,
    ) -> None:
        self._rules = tuple(penalty_rules)
        self._fields = tuple(
            dict.fromkeys(
                [
                    *fields,
                    *(rule.field for rule in self._rules if rule.field is not None),
                ]
            )
        )
        self._reset()

    def _reset(self) -> None:
        self._rollouts: dict[str, int] = {}
        self._masked: dict[str, int] = {}
        self._reward_sum: dict[str, float] = {}
        self._unmasked_reward_sum: dict[str, float] = {}
        self._raw_reward_sum: dict[str, float] = {}
        self._applied: dict[tuple[str, str], int] = {}
        self._kept_masked: dict[tuple[str, str], int] = {}
        self._field_max: dict[tuple[str, str], float] = {}
        self._field_sum: dict[tuple[str, str], float] = {}
        self._field_count: dict[tuple[str, str], int] = {}

    def record(
        self,
        agent_name: str,
        full_result: Mapping[str, Any],
        outcome: PenaltyOutcome = PenaltyOutcome(),
    ) -> None:
        """Count one rollout after the mask rules and the penalty rules ran on it."""
        self._rollouts[agent_name] = self._rollouts.get(agent_name, 0) + 1
        reward = float(full_result.get("reward") or 0.0)
        self._reward_sum[agent_name] = self._reward_sum.get(agent_name, 0.0) + reward
        if _is_masked(full_result):
            self._masked[agent_name] = self._masked.get(agent_name, 0) + 1
        else:
            self._unmasked_reward_sum[agent_name] = (
                self._unmasked_reward_sum.get(agent_name, 0.0) + reward
            )
        if self._rules:
            raw_reward = float(full_result.get(RAW_REWARD_KEY, reward) or 0.0)
            self._raw_reward_sum[agent_name] = (
                self._raw_reward_sum.get(agent_name, 0.0) + raw_reward
            )
            if outcome.rule is not None:
                counts = self._applied if outcome.applied else self._kept_masked
                key = (agent_name, outcome.rule)
                counts[key] = counts.get(key, 0) + 1
        for field in self._fields:
            found, value = _lookup(full_result, field)
            if (
                not found
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
            ):
                continue
            key = (agent_name, field)
            self._field_max[key] = max(self._field_max.get(key, value), value)
            self._field_sum[key] = self._field_sum.get(key, 0.0) + value
            self._field_count[key] = self._field_count.get(key, 0) + 1

    def pop_metrics(self) -> dict[str, float]:
        """Metrics since the last call (empty without rollouts); resets."""
        out: dict[str, float] = {}
        for agent, rollouts in self._rollouts.items():
            masked = self._masked.get(agent, 0)
            out[f"{agent}/rollouts"] = float(rollouts)
            out[f"{agent}/reward_mean"] = self._reward_sum[agent] / rollouts
            out[f"{agent}/env_masked_frac"] = masked / rollouts
            if rollouts > masked:
                out[f"{agent}/unmasked_reward_mean"] = self._unmasked_reward_sum[
                    agent
                ] / (rollouts - masked)
            if self._rules:
                out[f"{agent}/raw_reward_mean"] = self._raw_reward_sum[agent] / rollouts
                for rule in self._rules:
                    if not rule.applies_to(agent):
                        continue
                    key = (agent, rule.name)
                    out[f"{agent}/{rule.name}_frac"] = (
                        self._applied.get(key, 0) / rollouts
                    )
                    out[f"{agent}/{rule.name}_kept_masked_frac"] = (
                        self._kept_masked.get(key, 0) / rollouts
                    )
            for field in self._fields:
                key = (agent, field)
                if key in self._field_count:
                    metric = field.replace(".", "_")
                    out[f"{agent}/{metric}_max"] = self._field_max[key]
                    out[f"{agent}/{metric}_mean"] = (
                        self._field_sum[key] / self._field_count[key]
                    )
        self._reset()
        return out
