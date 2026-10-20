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
"""The ``env.nemo_gym`` settings the rollout loop reads, and the metrics hook one of them names.

``_build_gym_actor_config`` in ``nemo_rl.environments.nemo_gym`` keeps three keys
of the ``env.nemo_gym`` mapping out of Gym's global configuration and stores
them as top-level fields of every Gym actor's configuration
(``DRIVER_ROLLOUT_KEYS``): ``health_check_interval_seconds``,
``max_infra_attempts_per_rollout`` and ``rollout_metrics_hook``. An actor hands
them back through ``NemoGym.rollout_config``; ``_nemo_gym_rollout_config``
fetches them, and the ``get_nemo_gym_*`` readers turn them into the values the
rollout loop applies. Both rollout paths import this module: the driver loop in
``nemo_rl.experience.rollouts`` and the single-controller path in
``nemo_rl.experience.rollout_manager``. The hook machinery
(``RolloutMetricsHook``, ``get_nemo_gym_rollout_metrics_hook``,
``apply_nemo_gym_rollout_metrics_hook``) resolves the callable
``rollout_metrics_hook`` names and merges its metrics into a prompt group's
rollout metrics.
"""

import asyncio
import importlib
import logging
from collections.abc import Callable, Mapping
from typing import Any, Optional, cast

from nemo_rl.environments.nemo_gym import (
    NEMO_GYM_HEALTH_CHECK_INTERVAL_KEY,
    NEMO_GYM_MAX_INFRA_ATTEMPTS_KEY,
    NEMO_GYM_ROLLOUT_METRICS_HOOK_KEY,
    as_nemo_gym_shard_set,
)
from nemo_rl.experience.failures import GymTransportError
from nemo_rl.experience.metric_utils import is_histogram_metric

logger = logging.getLogger(__name__)


# A row is dispatched once and never again after an infrastructure failure unless
# env.nemo_gym.max_infra_attempts_per_rollout raises the budget.
DEFAULT_MAX_INFRA_ATTEMPTS_PER_ROLLOUT = 1


def get_nemo_gym_health_check_interval(env_config: dict[str, Any]) -> Optional[float]:
    """Seconds between health checks of the Gym actor while rollouts are awaited.

    Reads ``env.nemo_gym.health_check_interval_seconds``. None, the value when the
    key is absent or null, turns the polling off; a set value must be positive.
    """
    nemo_gym_config = env_config.get("nemo_gym")
    value = (
        nemo_gym_config.get(NEMO_GYM_HEALTH_CHECK_INTERVAL_KEY)
        if isinstance(nemo_gym_config, dict)
        else None
    )
    if value is None:
        return None
    interval = float(value)
    if interval <= 0:
        raise ValueError(
            f"env.nemo_gym.health_check_interval_seconds must be positive, got {value}"
        )
    return interval


def get_nemo_gym_max_infra_attempts(env_config: dict[str, Any]) -> int:
    """How many times one row may be dispatched before an infrastructure failure ends the step.

    Reads ``env.nemo_gym.max_infra_attempts_per_rollout``; an absent or null key
    means ``DEFAULT_MAX_INFRA_ATTEMPTS_PER_ROLLOUT``. The value must be at least 1.

    This budget governs every caller of ``run_async_nemo_gym_rollout``
    (synchronous GRPO, the asynchronous trajectory collector, PPO,
    distillation). The single-controller path
    (``rollout_manager.AsyncNemoGymRolloutImpl``) re-dispatches a
    failure-marker row through its own attempt loop, budgeted by
    ``RolloutRetryPolicy.max_gym_row_attempts``, and whole-attempt failures
    there are budgeted by its ``max_attempts_per_prompt``; the three are
    deliberately separate knobs.
    """
    nemo_gym_config = env_config.get("nemo_gym")
    value = (
        nemo_gym_config.get(NEMO_GYM_MAX_INFRA_ATTEMPTS_KEY)
        if isinstance(nemo_gym_config, dict)
        else None
    )
    if value is None:
        return DEFAULT_MAX_INFRA_ATTEMPTS_PER_ROLLOUT
    # bool is an int subclass and int() truncates a float silently; both would
    # hide a misconfiguration behind a budget the operator never wrote.
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            "env.nemo_gym.max_infra_attempts_per_rollout must be an integer of "
            f"at least 1, got {value!r}"
        )
    return value


# A Gym actor answers rollout_config from its event loop, so an actor whose loop
# is blocked never answers. The read is the first thing a rollout awaits, before
# the health check polls, and the asynchronous trajectory collector retries a
# failed batch against the same actors, so without a bound a probe timeout would
# be followed by a retry that hangs in this read.
NEMO_GYM_ROLLOUT_CONFIG_TIMEOUT_SECONDS = 60.0


# A step-metrics hook: takes one rollout group's input rows and, aligned with
# them, each rollout's ``full_result`` mapping, and returns metrics to merge
# into the group's rollout metrics after every built-in metric of the group.
RolloutMetricsHook = Callable[
    [list[dict[str, Any]], list[dict[str, Any]]], dict[str, float]
]

# Rollout-metrics hook conditions already reported in this process, as
# ``(hook qualified name, condition)`` pairs. Each is reported the first time
# only, since it would otherwise repeat for every prompt group of every step.
_ROLLOUT_METRICS_HOOK_WARNED: set[tuple[str, str]] = set()


def _warn_rollout_metrics_hook_once(
    hook_name: str,
    condition: str,
    message: str,
    *args: Any,
    exc_info: bool = False,
) -> None:
    """Log a hook condition the first time it occurs for ``hook_name`` in this process.

    ``message`` is a logging format string whose first placeholder is the hook
    name; ``args`` fill the remaining placeholders.
    """
    if (hook_name, condition) in _ROLLOUT_METRICS_HOOK_WARNED:
        return
    _ROLLOUT_METRICS_HOOK_WARNED.add((hook_name, condition))
    logger.warning(message, hook_name, *args, exc_info=exc_info)


def _is_loggable_hook_value(key: str, value: Any) -> bool:
    """Whether the step aggregation can average ``value`` under ``key``.

    A histogram key (``is_histogram_metric``) takes a sequence of numbers, the
    raw observations; any other key takes one number.
    """
    if is_histogram_metric(key):
        return isinstance(value, (list, tuple)) and all(
            isinstance(item, (bool, int, float)) for item in value
        )
    return isinstance(value, (bool, int, float))


def apply_nemo_gym_rollout_metrics_hook(
    hook: Optional[RolloutMetricsHook],
    rows: list[dict[str, Any]],
    full_results: list[dict[str, Any]],
    rollout_metrics: dict[str, Any],
) -> None:
    """Merge the hook's metrics for one rollout group into ``rollout_metrics``.

    Called after every built-in metric writer of the group on both rollout
    paths (the timing and routing keys are added afterwards), so on a key
    collision the hook's value wins and the collision is warned once per hook
    rather than silently reabsorbed. A hook that raises or returns no mapping
    is warned once and contributes nothing to the group; a value the step
    aggregation cannot average (anything but a number, or a sequence of
    numbers under a histogram key) is dropped and warned once. The rollout
    never fails because of the hook.
    """
    if hook is None:
        return
    hook_name = getattr(hook, "__qualname__", repr(hook))
    try:
        hook_metrics = hook(rows, full_results)
        if not isinstance(hook_metrics, Mapping):
            raise TypeError(
                f"returned {type(hook_metrics).__name__}, expected a mapping of "
                "metric name to number"
            )
    except Exception:
        _warn_rollout_metrics_hook_once(
            hook_name,
            "failed",
            "NeMo-Gym rollout metrics hook %s failed; its metrics are skipped for "
            "this prompt group (later failures of this hook are not reported)",
            exc_info=True,
        )
        return
    accepted = {
        key: value
        for key, value in hook_metrics.items()
        if _is_loggable_hook_value(key, value)
    }
    if len(accepted) != len(hook_metrics):
        _warn_rollout_metrics_hook_once(
            hook_name,
            "values",
            "NeMo-Gym rollout metrics hook %s returned values the step aggregation "
            "cannot average under %s; those keys are dropped (reported once). "
            "Return a number per key, a sequence of numbers under a histogram "
            "key, and omit a key instead of returning None",
            sorted(set(hook_metrics) - set(accepted)),
        )
    collisions = rollout_metrics.keys() & accepted.keys()
    if collisions:
        _warn_rollout_metrics_hook_once(
            hook_name,
            "collision",
            "NeMo-Gym rollout metrics hook %s overrides built-in metric key(s) "
            "%s (reported once)",
            sorted(collisions),
        )
    rollout_metrics.update(accepted)


def get_nemo_gym_rollout_metrics_hook(
    env_config: dict[str, Any],
) -> Optional[RolloutMetricsHook]:
    """Import the hook ``env.nemo_gym.rollout_metrics_hook`` names, or None when unset.

    The value is the dotted import path of a callable,
    ``package.module.function``, imported in the process that runs the rollout
    loop (under asynchronous GRPO that is the trajectory collector actor, so
    the module must be importable there). The hook is called once per rollout
    group as ``hook(rows, full_results)`` and returns a mapping merged into
    the group's rollout metrics after every built-in metric of the group
    (``apply_nemo_gym_rollout_metrics_hook``; the timing and routing keys are
    added afterwards). Facts a hook author needs:

    - ``rows`` are the group's input rows in rollout order; each carries
      ``agent_ref["name"]`` and the ``_ng_group_id`` prompt-group key, which
      is how a hook recovers prompt boundaries and agent identity.
    - ``full_results`` align with ``rows``: each is that rollout's own
      ``full_result`` mapping, which later stages read too, so the hook treats
      it as read-only. Its ``reward`` is the trained value, already rewritten
      by effort shaping and reward penalties, except on the single-controller
      path under token capture (receipt mode), where it is the verifier's
      reward and shaping and penalties are applied at capture finalization.
    - ``full_results`` include environment-masked rollouts (``mask_sample`` on
      the result or under ``instance_config``), whose reward and scores are
      placeholders; the built-in per-agent means leave them out while
      ``env.should_mask_flagged_samples`` is on, so a hook that wants the same
      view skips them itself
      (``nemo_rl.experience.rollouts.is_environment_masked``).
    - The group differs by caller: one prompt's rollouts under the
      asynchronous collector and the single-controller path, but the whole
      step batch (mixed prompts, possibly mixed agents) under
      ``run_nemo_gym_rollout_sync``. The hook also runs on validation
      batches, whose keys are logged under ``val/``.
    - Values are numbers, or sequences of numbers under a key that ends in
      ``/histogram`` or starts with ``histogram/``; anything else is dropped
      with one warning, so a key the group cannot compute is omitted rather
      than returned as None. Under asynchronous GRPO,
      ``aggregate_rollout_metrics`` averages a returned key over prompt groups
      with equal weight (``/min``, ``/max`` and histogram keys aside), so a
      per-group value aggregates to a mean of group values.
    - The hook runs synchronously on the rollout loop, so it is kept to cheap
      pure-Python work over the group.

    The per-agent aggregation (``nemo_rl.experience.rollouts.per_agent_result_metrics``)
    already logs the mean of every numeric result field; the hook is where an
    environment derives what that cannot express, such as pass-at-k over a
    prompt group.

    Raises:
        ValueError: The path has no module part or does not name a callable.
        ImportError: The module part cannot be imported (``ModuleNotFoundError``
            for a missing module).
    """
    nemo_gym_config = env_config.get("nemo_gym")
    path = (
        nemo_gym_config.get(NEMO_GYM_ROLLOUT_METRICS_HOOK_KEY)
        if isinstance(nemo_gym_config, dict)
        else None
    )
    if not path:
        return None
    module_name, _, attribute = str(path).rpartition(".")
    if not module_name:
        raise ValueError(
            f"env.nemo_gym.rollout_metrics_hook must be a dotted import path "
            f"(package.module.function), got {path!r}"
        )
    hook = getattr(importlib.import_module(module_name), attribute, None)
    if not callable(hook):
        raise ValueError(
            f"env.nemo_gym.rollout_metrics_hook {path!r} does not name a callable"
        )
    return cast(RolloutMetricsHook, hook)


async def _nemo_gym_rollout_config(nemo_gym_environment: Any) -> dict[str, Any]:
    """Fetch the ``env.nemo_gym`` keys the rollout loop reads from the Gym actors.

    ``_build_gym_actor_config`` keeps those keys (``DRIVER_ROLLOUT_KEYS`` in
    ``nemo_rl.environments.nemo_gym``) as top-level fields of every actor's
    configuration and out of Gym's global configuration, and an actor hands
    them back through ``rollout_config``. The environment is unwrapped through
    ``as_nemo_gym_shard_set`` and one actor is asked: every shard is built from
    the same env mapping, so the driver keys agree across the set. The mapping
    is returned in the ``master_config.env`` layout the ``get_nemo_gym_*``
    readers take, so every caller of the rollout function, synchronous or not,
    applies the same configured values without forwarding them.

    The call is bounded by ``NEMO_GYM_ROLLOUT_CONFIG_TIMEOUT_SECONDS`` and a
    missed bound is ``GymTransportError``, the class the rest of the rollout
    raises for an actor that stopped answering.
    """
    handle = as_nemo_gym_shard_set(nemo_gym_environment).all_handles[0]
    try:
        settings = await asyncio.wait_for(
            handle.rollout_config.remote(),
            timeout=NEMO_GYM_ROLLOUT_CONFIG_TIMEOUT_SECONDS,
        )
    except TimeoutError as error:
        raise GymTransportError(
            "NeMo-Gym actor did not answer rollout_config within "
            f"{NEMO_GYM_ROLLOUT_CONFIG_TIMEOUT_SECONDS}s; its event loop may be blocked"
        ) from error
    return {"nemo_gym": dict(settings)}
