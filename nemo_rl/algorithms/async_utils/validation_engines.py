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

"""Lend a few vLLM engines to background validation in async GRPO.

Used by ``grpo.async_grpo.validation_engines``. Training rollouts reach the
engines through a router; validation has its own NeMo-Gym stack pointed at the
last ``num_engines`` engines. Between validations those engines serve training
through the router. While a validation runs they keep the weights of the step it
started at, so its score measures one policy version. With
``serve_training_steps`` they also keep serving training on those weights until
the policy is that many steps ahead; otherwise they serve only the validation.

A validation is queued on the lent engines all at once, so training requests
sent there would wait it out. The engines therefore run with vLLM priority
scheduling and the router sends training requests ahead of validation.

Refit is a collective broadcast that every engine must join, so a validating
engine cannot sit a refit out. It joins and drops the weights instead
(``VllmGeneration.set_discard_refits``).
"""

import enum
from typing import Any

import ray

from nemo_rl.models.generation.vllm import VllmGeneration


class EngineState(enum.Enum):
    """Where the lent engines are in the lend, freeze, return cycle."""

    # Serving training through the router and taking every refit.
    TRAINING = "training"
    # Serving validation, and training too within serve_training_steps; dropping
    # refits.
    VALIDATING = "validating"
    # Validation done; taking refits again but out of the router until one
    # lands, so no training request runs on weights older than its stamp.
    CATCHING_UP = "catching_up"


def split_validation_urls(
    urls: list[str], num_engines: int
) -> tuple[list[str], list[str]]:
    """Split engine URLs into the ones kept for training and the lent ones.

    The lent engines are the last ``num_engines``. Placement bundles are node
    ordered, so they share the last generation node.

    Returns:
        ``(training_urls, validation_urls)``.
    """
    if not 1 <= num_engines < len(urls):
        raise ValueError(
            f"grpo.async_grpo.validation_engines.num_engines={num_engines} must be "
            f"between 1 and {len(urls) - 1}: training keeps at least one of the "
            f"{len(urls)} vLLM engines."
        )
    return urls[: len(urls) - num_engines], urls[len(urls) - num_engines :]


class ValidationEnginePool:
    """Moves the lent engines between training and validation.

    Driven from the training thread only: ``lend_to_validation`` right after a
    refit when a validation starts, ``before_refit`` and ``after_refit`` around
    every refit.
    """

    def __init__(
        self,
        *,
        generation: VllmGeneration,
        router: ray.actor.ActorHandle,
        nemo_gym: Any,
        num_engines: int,
        serve_training_steps: int,
    ) -> None:
        """Initialize the pool.

        Args:
            generation: The vLLM generation every engine belongs to.
            router: Router that NeMo-Gym's training stack sends rollouts to.
            nemo_gym: NeMo-Gym stack that sends validation rollouts straight
                to the lent engines.
            num_engines: How many engines, taken from the end, are lent.
            serve_training_steps: How many steps past their frozen weights the
                lent engines keep serving training. 0 takes them out of
                training for the whole validation.
        """
        urls = list(generation.dp_openai_server_base_urls)
        self.training_urls, _ = split_validation_urls(urls, num_engines)
        self.generation = generation
        self.router = router
        self.nemo_gym = nemo_gym
        self.shard_indices = list(range(len(urls) - num_engines, len(urls)))
        self.all_urls = urls
        self.serve_training_steps = serve_training_steps
        self.state = EngineState.TRAINING
        # Weight version the lent engines hold while validating.
        self.frozen_version: int | None = None
        # Whether the router sends training requests to the lent engines.
        self.lent_in_router = True

    @property
    def serving_training(self) -> bool:
        """Whether the engines can be lent: serving training on current weights."""
        return self.state is EngineState.TRAINING

    def lend_to_validation(self, weight_version: int) -> None:
        """Freeze the engines' weights for validation.

        Call right after a refit, so the frozen weights are the step the
        validation is labelled with. Training requests already running on them
        finish on those weights, which are no newer than their stamp.

        Args:
            weight_version: Weight version the engines hold, which they keep
                until the validation is done.
        """
        if self.state is not EngineState.TRAINING:
            raise RuntimeError(
                f"validation engines are {self.state.value}, not serving training"
            )
        self.frozen_version = weight_version
        if self.serve_training_steps == 0:
            self._remove_from_router()
        self.generation.set_discard_refits(self.shard_indices, True)
        self.state = EngineState.VALIDATING

    def _remove_from_router(self) -> None:
        ray.get(self.router.set_serving_backends.remote(self.training_urls))
        self.lent_in_router = False

    def before_refit(self, *, validation_running: bool) -> None:
        """Let the coming refit update the engines once their validation is done."""
        if self.state is EngineState.VALIDATING and not validation_running:
            self.generation.set_discard_refits(self.shard_indices, False)
            self.state = EngineState.CATCHING_UP

    def after_refit(self, weight_version: int) -> None:
        """Return caught-up engines to training, or retire frozen ones from it.

        Training requests are stamped with the current weight version, so one
        sent to a frozen engine runs on weights up to ``serve_training_steps``
        older than its stamp. Past that the engines leave training until their
        validation is done.

        Args:
            weight_version: Weight version the refit just published.
        """
        if self.state is EngineState.CATCHING_UP:
            if not self.lent_in_router:
                ray.get(self.router.set_serving_backends.remote(self.all_urls))
                self.lent_in_router = True
            self.frozen_version = None
            self.state = EngineState.TRAINING
        elif (
            self.state is EngineState.VALIDATING
            and self.lent_in_router
            and weight_version - self.frozen_version >= self.serve_training_steps
        ):
            self._remove_from_router()

    def shutdown(self) -> None:
        """Stop the router.

        The validation NeMo-Gym stack is shut down with the other environments.
        """
        ray.kill(self.router)
