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

    _USES_SPAN_REGISTRY = False
except ImportError:
    from nemo.lens.groups import SpanRegistry

    _USES_SPAN_REGISTRY = True

    class SpanGroup:
        """Compatibility surface for nemo-lens's registry-based API."""

        JOB = "job"
        STEP = "step"
        CHECKPOINT = "checkpoint"
        LOAD_CHECKPOINT = "load_checkpoint"
        EVALUATE = "evaluate"
        MODEL_INIT = "model_init"
        FORWARD_BACKWARD = "forward_backward"
        OPTIMIZER = "optimizer"

        ALL_GROUPS: Final[frozenset[str]] = frozenset(
            {
                JOB,
                STEP,
                CHECKPOINT,
                LOAD_CHECKPOINT,
                EVALUATE,
                MODEL_INIT,
                FORWARD_BACKWARD,
                OPTIMIZER,
            }
        )

        @classmethod
        def resolve(cls, spec: str) -> frozenset[str]:
            resolved: set[str] = set()
            for item in (part.strip().lower() for part in spec.split(",")):
                if item in cls._PRESETS:
                    resolved.update(cls._PRESETS[item])
                elif item in cls.ALL_GROUPS:
                    resolved.add(item)
                else:
                    raise ValueError(f"Unknown span group or preset: {item!r}")
            return frozenset(resolved)


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


if _USES_SPAN_REGISTRY:
    SpanRegistry.register(
        "nemo_rl",
        RLSpanGroup.ALL_GROUPS,
        {
            name: members
            for name, members in RLSpanGroup._PRESETS.items()
            if name != "all"
        },
        allow_override=True,
    )
