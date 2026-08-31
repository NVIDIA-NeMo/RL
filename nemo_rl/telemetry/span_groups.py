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

"""NeMo-RL specific span groups."""

from typing import ClassVar, Final

try:
    from nemo.lens.groups import SpanGroup
except ImportError:  # pragma: no cover — exercised only in lens-free envs
    # Structural stand-in for ``nemo.lens.groups.SpanGroup`` so this module
    # (and everything importing ``RLSpanGroup``) stays importable in a venv
    # without nemo-lens. Constants and ``resolve`` mirror the lens class;
    # telemetry itself cannot be enabled without lens, so nothing consumes the
    # resolved sets in that environment.
    class SpanGroup:  # type: ignore[no-redef]
        """Fallback span-group base when nemo-lens is not installed."""

        JOB = "job"
        CHECKPOINT = "checkpoint"
        EVALUATE = "evaluate"
        MODEL_INIT = "model_init"
        LOAD_CHECKPOINT = "load_checkpoint"
        STEP = "step"
        FORWARD_BACKWARD = "forward_backward"
        OPTIMIZER = "optimizer"

        ALL_GROUPS: Final[frozenset] = frozenset(
            [
                JOB,
                CHECKPOINT,
                EVALUATE,
                MODEL_INIT,
                LOAD_CHECKPOINT,
                STEP,
                FORWARD_BACKWARD,
                OPTIMIZER,
            ]
        )

        _PRESETS: ClassVar[dict] = {
            "default": frozenset([JOB, CHECKPOINT, EVALUATE]),
            "per_step": frozenset(
                [
                    JOB,
                    CHECKPOINT,
                    EVALUATE,
                    MODEL_INIT,
                    LOAD_CHECKPOINT,
                    STEP,
                    FORWARD_BACKWARD,
                    OPTIMIZER,
                ]
            ),
            "all": ALL_GROUPS,
        }

        @classmethod
        def resolve(cls, spec: str) -> frozenset:
            """Resolve a span-group spec string to a frozenset of group names."""
            result: set = set()
            for part in (p.strip().lower() for p in spec.split(",") if p.strip()):
                if part in cls._PRESETS:
                    result |= cls._PRESETS[part]
                elif part in cls.ALL_GROUPS:
                    result.add(part)
                else:
                    valid = sorted(cls.ALL_GROUPS | set(cls._PRESETS))
                    raise ValueError(
                        f"Unknown span group or preset: {part!r}. Valid options: {valid}"
                    )
            return frozenset(result)


class RLSpanGroup(SpanGroup):
    """Span groups for NeMo-RL instrumentation."""

    # ------------------------------------------------------------------ #
    # RL-specific groups
    # ------------------------------------------------------------------ #

    ROLLOUT = "rollout"
    """Rollout collection spans."""

    GENERATION = "generation"
    """Text generation spans."""

    LOGPROB = "logprob"
    """Log-probability computation spans."""

    REWARD = "reward"
    """Reward computation spans."""

    ADVANTAGE = "advantage"
    """Advantage computation spans."""

    POLICY_UPDATE = "policy_update"
    """Policy gradient update spans."""

    REFERENCE_POLICY = "reference_policy"
    """Reference policy log-prob computation spans."""

    DATA_PROCESSING = "data_processing"
    """Data processing / batching spans."""

    EFFICIENCY = "efficiency"
    """Async efficiency phases (idle / wasted accounting).

    Unlike the other leaf groups these do not have one fixed bucket — the
    ``rl.bucket`` comes from the category, so emit them via
    ``instrumentation.efficiency_span``.
    """

    # ------------------------------------------------------------------ #
    # All groups and presets
    # ------------------------------------------------------------------ #

    ALL_GROUPS: Final[frozenset] = SpanGroup.ALL_GROUPS | frozenset(
        [
            ROLLOUT,
            GENERATION,
            LOGPROB,
            REWARD,
            ADVANTAGE,
            POLICY_UPDATE,
            REFERENCE_POLICY,
            DATA_PROCESSING,
            EFFICIENCY,
        ]
    )

    _PRESETS: ClassVar[dict] = {
        "default": frozenset(
            [
                SpanGroup.JOB,
                SpanGroup.CHECKPOINT,
                SpanGroup.EVALUATE,
            ]
        ),
        # NOTE: ``per_step`` deliberately omits ``JOB`` so each training step is
        # its own root trace (bounded size). ``JOB`` — which wraps the whole run
        # and would nest every step under one giant trace — lives in ``default``
        # (coarse: job + checkpoint + evaluate) and ``all``.
        "per_step": frozenset(
            [
                SpanGroup.CHECKPOINT,
                SpanGroup.EVALUATE,
                # rl.vllm.load_model is the only span in this group, and it was
                # otherwise reachable from "all" alone -- so the one phase that
                # explains a slow start was invisible in both presets a user is
                # likely to pick.
                SpanGroup.MODEL_INIT,
                SpanGroup.STEP,
                ROLLOUT,
                GENERATION,
                LOGPROB,
                REWARD,
                ADVANTAGE,
                POLICY_UPDATE,
                REFERENCE_POLICY,
                DATA_PROCESSING,
                # Included here because idle time is what makes a per-step
                # goodput breakdown add up to the step duration.
                EFFICIENCY,
            ]
        ),
        "all": ALL_GROUPS,
    }
