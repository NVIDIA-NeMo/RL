# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NeMo-RL specific span groups.

Tries to import the real ``SpanGroup`` from ``nemo.lens``; falls back to a
minimal stub so that NeMo-RL works without nemo-lens installed.
"""

from typing import ClassVar, Final

try:
    from nemo.lens.groups import SpanGroup
except ImportError:
    # TODO(ahmadki): SpanGroups will move from nemo-lens to downstream,
    # so this stub will be removed in the future.
    class SpanGroup:  # type: ignore[no-redef]
        """Minimal stub used when nemo-lens is not installed."""

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
            raise RuntimeError(
                "SpanGroup.resolve() requires nemo-lens. "
                "Install it with: uv sync --extra telemetry"
            )


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
                SpanGroup.STEP,
                ROLLOUT,
                GENERATION,
                LOGPROB,
                REWARD,
                ADVANTAGE,
                POLICY_UPDATE,
                REFERENCE_POLICY,
                DATA_PROCESSING,
            ]
        ),
        "all": ALL_GROUPS,
    }
