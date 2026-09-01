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

from nemo.lens.groups import SpanGroup


class RLSpanGroup(SpanGroup):
    """Span groups for NeMo-RL instrumentation."""

    # ------------------------------------------------------------------ #
    # RL-specific groups
    # ------------------------------------------------------------------ #

    SETUP = "setup"
    """Driver-side startup phases: Ray init, worker builds, collective init.

    Distinct from the base ``MODEL_INIT`` group, which covers the model load
    itself inside a worker. These are the driver's view of the same startup —
    the phases between process start and the first training step.
    """

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

    DATA_PLANE = "data_plane"
    """Transfer-queue / data-plane operations (put, claim, get, clear).

    Its own group rather than part of ``DATA_PROCESSING`` because it is far
    finer-grained: one span per data-plane RPC, several per step, so a user who
    finds the volume noisy can drop these without losing the coarse phases.
    """

    PER_PROMPT = "per_prompt"
    """Spans emitted once per prompt rather than once per step or batch.

    A cardinality axis, unlike every other group here, which names a phase. The
    two axes are independent: ``rl.sc.generate_and_push`` and the rollout path's
    ``rl.data_plane.put`` are a rollout span and a data-plane span respectively,
    but what governs whether a user wants them is neither of those things — it
    is that their count scales with the prompt count, so a 10k-prompt rollout
    emits ~20k spans where the phase groups emit a fixed handful per step.

    Kept out of the ``per_step`` preset for that reason: ``per_step`` is meant
    to be the detailed-but-usable choice, and its span count should scale with
    steps, not with dataset size. Reach for these with
    ``span_groups: "per_step,per_prompt"`` when debugging an individual
    rollout, or ``all``.

    An umbrella group, so its spans carry no ``rl.bucket`` — see
    ``instrumentation.UMBRELLA_GROUPS`` for why concurrency forces that.
    """

    EFFICIENCY = "efficiency"
    """Async efficiency phases (idle / wasted accounting).

    Unlike the other leaf groups these do not have one fixed bucket — the
    ``rl.bucket`` comes from the category, so emit them via
    ``instrumentation.efficiency_span``.
    """

    # ------------------------------------------------------------------ #
    # Umbrella aliases
    # ------------------------------------------------------------------ #
    #
    # Every group either carries an ``rl.bucket`` or does not, and the call site
    # cannot see which -- ``GENERATION`` and ``ROLLOUT`` read as interchangeable
    # choices for a generation span, but one is ``productive`` and the other is
    # not counted at all. Picking the bucketed one for a span that overlaps
    # itself inflates that bucket by the concurrency factor and looks identical
    # to correct code. These aliases put the answer in the name.
    #
    # Aliases, not new groups: the value is the same string, so presets, the
    # ``span_groups`` spec and every config are unaffected, and the two
    # spellings are interchangeable at runtime. What makes the convention hold
    # is the pairing with ``instrumentation.umbrella_span`` /
    # ``umbrella_trace_fn`` plus a drift test that rejects the unprefixed
    # spelling at a span call site.

    U_JOB = SpanGroup.JOB
    U_STEP = SpanGroup.STEP
    U_MODEL_INIT = SpanGroup.MODEL_INIT
    U_EVALUATE = SpanGroup.EVALUATE
    U_ROLLOUT = ROLLOUT
    U_SETUP = SETUP
    U_PER_PROMPT = PER_PROMPT

    # ------------------------------------------------------------------ #
    # All groups and presets
    # ------------------------------------------------------------------ #

    ALL_GROUPS: Final[frozenset] = SpanGroup.ALL_GROUPS | frozenset(
        [
            SETUP,
            ROLLOUT,
            GENERATION,
            LOGPROB,
            REWARD,
            ADVANTAGE,
            POLICY_UPDATE,
            REFERENCE_POLICY,
            DATA_PROCESSING,
            DATA_PLANE,
            PER_PROMPT,
            EFFICIENCY,
        ]
    )

    _PRESETS: ClassVar[dict] = {
        # Startup is in here, and in per_step, for the reason MODEL_INIT was
        # added to per_step: "why was the first step so late" is one of the
        # questions a coarse preset most needs to answer, and both groups emit
        # a fixed handful of spans once per run, so neither grows with step
        # count. They travel together — SETUP without MODEL_INIT would show the
        # worker-build phase as one opaque block with the model load, usually
        # the largest part, missing from inside it.
        "default": frozenset(
            [
                SpanGroup.JOB,
                SpanGroup.CHECKPOINT,
                SpanGroup.EVALUATE,
                SpanGroup.MODEL_INIT,
                SETUP,
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
                SETUP,
                SpanGroup.STEP,
                ROLLOUT,
                GENERATION,
                LOGPROB,
                REWARD,
                ADVANTAGE,
                POLICY_UPDATE,
                REFERENCE_POLICY,
                DATA_PROCESSING,
                # Included rather than left to "all" so the single-controller
                # step, whose phases are largely transfer-queue traffic, is
                # legible under the preset a user is most likely to pick.
                DATA_PLANE,
                # Included here because idle time is what makes a per-step
                # goodput breakdown add up to the step duration.
                EFFICIENCY,
                # NOTE: PER_PROMPT is deliberately absent. Every group above
                # emits a bounded number of spans per step, so this preset's
                # cost scales with steps; per-prompt spans would make it scale
                # with dataset size instead (~2 per prompt). Ask for them
                # explicitly with "per_step,per_prompt", or take "all".
            ]
        ),
        "all": ALL_GROUPS,
    }
