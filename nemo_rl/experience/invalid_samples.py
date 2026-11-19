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
"""Invalid samples: masked rollouts whose reward is not a measurement of the policy.

``env.invalid_sample_failure_kinds`` names ``failure_kind`` values (the Gym response's own, or the
one an ``env.mask_sample_rules`` entry writes) that mark an infrastructure failure: a lost
sandbox, a verifier or judge that did not finish, a harness killed for memory::

    env:
      invalid_sample_failure_kinds: [agent_run_error, judge_failed, verifier_error, provider_oom_killed]

A masked rollout whose ``failure_kind`` is listed is published as a finalizer placeholder: its
token-capture receipt is poisoned, so its rows carry ``sample_mask`` 0 -- no gradient for the
policy or the critic, and no reward in the group baseline/std. Every other masked rollout keeps
its behaviour: no gradient, and its reward stays in the baseline under
``<algo>.masked_sample_rewards_in_baseline`` (agent timeouts, context limits). An absent or empty
list changes nothing. Token capture only: the placeholder is the finalizer's.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Any, Optional

from nemo_rl.data_plane.schema import MASK_SAMPLE
from nemo_rl.experience.mask_sample_rules import FAILURE_KIND_KEY

ENV_INVALID_SAMPLE_FAILURE_KINDS_KEY = "invalid_sample_failure_kinds"
# Receipt failure_reason prefix, so the finalizer's rejection names why.
INVALID_SAMPLE_REASON = "invalid_sample"


def parse_invalid_sample_failure_kinds(
    env_config: Mapping[str, Any] | None,
) -> frozenset[str]:
    """Read ``env.invalid_sample_failure_kinds``; absent or ``None`` means none."""
    raw = (env_config or {}).get(ENV_INVALID_SAMPLE_FAILURE_KINDS_KEY)
    if raw is None:
        return frozenset()
    if (
        isinstance(raw, (str, bytes))
        or not isinstance(raw, Sequence)
        or not all(isinstance(kind, str) and kind.strip() for kind in raw)
    ):
        raise ValueError(
            f"env.{ENV_INVALID_SAMPLE_FAILURE_KINDS_KEY} must be a list of failure_kind "
            f"names, got {raw!r}"
        )
    return frozenset(raw)


def invalid_sample_kind(
    full_result: Mapping[str, Any], kinds: Collection[str]
) -> Optional[str]:
    """The ``failure_kind`` of a masked rollout when it is one of ``kinds``, else None."""
    instance_config = full_result.get("instance_config")
    masked = isinstance(instance_config, Mapping) and bool(
        instance_config.get(MASK_SAMPLE)
    )
    kind = full_result.get(FAILURE_KIND_KEY)
    return kind if masked and kind in kinds else None


def poisoned_receipt(receipt: Mapping[str, Any], kind: str) -> dict[str, Any]:
    """``receipt`` marked failed, so the finalizer publishes its rollout as a placeholder.

    A receipt Gym already poisoned keeps its own reason.
    """
    if receipt.get("capture_poisoned"):
        return dict(receipt)
    return {
        **receipt,
        "capture_poisoned": True,
        "failure_reason": f"{INVALID_SAMPLE_REASON}:{kind}",
    }
