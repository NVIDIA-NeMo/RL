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
"""PolicyTrainerActor: Ray actor wrapper around :class:`TQPolicy`.

SingleController is a CPU-only Ray actor that drives the RL training loop via
``.remote(...)`` calls on a trainer handle. Production ``TQPolicy`` is a
driver-side controller, not a Ray actor — so SC cannot call it directly.

``PolicyTrainerActor`` is the boundary. It owns a ``TQPolicy`` instance inside
its own actor process (CPU-only — the GPUs live on the worker_group actors
that ``TQPolicy`` fans out to). It exposes only the SC-facing surface:

  - ``train_from_meta(meta)``                — full-step training (sync proxy).
  - ``prepare_logprobs_from_meta(meta, *, refresh_policy_logprobs=False,
    refresh_reference_logprobs=False)`` — refresh prev_lp / ref_lp into TQ.
  - ``begin_train_step``                     — open a split-API step.
  - ``train_microbatch_from_meta``           — one microbatch worth of fwd+bwd.
  - ``finish_train_step``                    — close the step, opt.step.
  - ``abort_train_step``                     — drop partial state.
"""

from __future__ import annotations

from typing import Any, Optional

import ray

from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.models.policy.tq_policy import TQPolicy


@ray.remote(num_cpus=1, num_gpus=0)  # pragma: no cover
class PolicyTrainerActor:
    """Ray actor that owns a ``TQPolicy`` and exposes the SC-facing API.

    Construction args mirror ``TQPolicy.__init__`` (``dp_cfg`` is required;
    everything else is forwarded). ``loss_fn`` / ``train_global_batch_size``
    / ``train_micro_batch_size`` are stored on the actor and used by every
    ``train_from_meta`` invocation, so SC does not need to know them.

    ``trainer_version`` advances by one on every successful ``train_from_meta``
    OR ``finish_train_step``. SC reads it to drive its weight-sync barrier.
    """

    def __init__(
        self,
        *policy_args: Any,
        dp_cfg: dict[str, Any],
        loss_fn: LossFunction,
        train_global_batch_size: Optional[int] = None,
        train_micro_batch_size: Optional[int] = None,
        tq_partition_id: str = "train",
        **policy_kwargs: Any,
    ) -> None:
        self._policy: TQPolicy = TQPolicy(
            *policy_args,
            dp_cfg=dp_cfg,
            tq_partition_id=tq_partition_id,
            **policy_kwargs,
        )
        self._loss_fn = loss_fn
        self._train_global_batch_size = train_global_batch_size
        self._train_micro_batch_size = train_micro_batch_size
        self._trainer_version: int = 0

    # ── lifecycle ──────────────────────────────────────────────────────────

    def shutdown(self) -> bool:
        return self._policy.shutdown()

    def ping(self) -> dict[str, Any]:
        return {
            "alive": True,
            "trainer_version": self._trainer_version,
        }

    # ── logprob preparation ────────────────────────────────────────────────

    def prepare_logprobs_from_meta(
        self,
        meta: KVBatchMeta,
        *,
        refresh_policy_logprobs: bool = False,
        refresh_reference_logprobs: bool = False,
    ) -> None:
        """Refresh policy and/or reference logprobs and commit them to TQ.

        SC decides which fields to refresh based on its own config
        (``advantage_policy_logprobs_field`` / ``advantage_reference_logprobs_field``).
        Workers write the per-token tensors back through TQ directly (no Ray
        return payload); this just dispatches and waits. No-op if both flags
        are False.

        NOTE: inlined here rather than delegating to
        ``TQPolicy.prepare_logprobs_from_meta`` so this PR is self-contained.
        Once the TQPolicy helper from #2700 lands together with this PR, this
        can collapse to a one-line delegate.
        """
        if refresh_policy_logprobs:
            self._policy.get_logprobs_from_meta(meta)
        if refresh_reference_logprobs:
            self._policy.get_reference_policy_logprobs_from_meta(meta)

    # ── current full-step training ─────────────────────────────────────────

    def train_from_meta(self, meta: KVBatchMeta) -> dict[str, Any]:
        """Full-step training proxy. Returns SC-shape dict.

        Calls ``TQPolicy.train_from_meta`` with the actor-bound ``loss_fn`` /
        ``gbs`` / ``mbs``, then folds the aggregated result into the shape SC
        expects (``loss``, ``trainer_version``, ``clear_samples``). Every
        successful call advances ``trainer_version`` by one.
        """
        result = self._policy.train_from_meta(
            meta,
            loss_fn=self._loss_fn,
            gbs=self._train_global_batch_size,
            mbs=self._train_micro_batch_size,
        )
        self._trainer_version += 1
        return {
            **result,
            "trainer_version": self._trainer_version,
            "clear_samples": True,
        }

    # ── split API ──────────────────────────────────────────────────────────
    #
    # ``trainer_version`` only advances at :meth:`finish_train_step`. The
    # backend state machine lives on the workers (see ``TQWorkerMixin`` and
    # the per-backend ``begin/microbatch/finish/abort_train_step`` methods).
    # This actor just forwards to ``TQPolicy``.

    def begin_train_step(
        self,
        step_id: str,
        loss_fn: Optional[LossFunction] = None,
        train_global_batch_size: Optional[int] = None,
        train_micro_batch_size: Optional[int] = None,
    ) -> None:
        self._policy.begin_train_step(
            step_id=step_id,
            loss_fn=loss_fn or self._loss_fn,
            gbs=train_global_batch_size or self._train_global_batch_size,
            mbs=train_micro_batch_size or self._train_micro_batch_size,
        )

    def train_microbatch_from_meta(
        self,
        step_id: str,
        meta: KVBatchMeta,
    ) -> dict[str, Any]:
        return self._policy.train_microbatch_from_meta(
            step_id=step_id,
            meta=meta,
        )

    def finish_train_step(self, step_id: str) -> dict[str, Any]:
        result = self._policy.finish_train_step(step_id=step_id)
        self._trainer_version += 1
        return {
            **result,
            "trainer_version": self._trainer_version,
            "clear_samples": True,
        }

    def abort_train_step(self, step_id: str) -> None:
        self._policy.abort_train_step(step_id=step_id)
