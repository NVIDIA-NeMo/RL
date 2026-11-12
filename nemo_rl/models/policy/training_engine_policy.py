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

"""Policy whose workers run the model through Megatron-Bridge's TrainingEngine."""

import os
from contextlib import nullcontext
from typing import Any, Iterable, Optional

import ray
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.loss import ClippedPGLossFn
from nemo_rl.algorithms.loss.interfaces import LossFunction, LossType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
)
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation.interfaces import GenerationDatumSpec
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import (
    LogprobOutputSpec,
    ReferenceLogprobOutputSpec,
)
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.models.policy.training_engine_adapter.config import (
    validate_training_engine_config,
)
from nemo_rl.utils.timer import Timer
from nemo_rl.utils.venvs import DEFAULT_VENV_DIR

TRAINING_ENGINE_WORKER_FQN = "nemo_rl.models.policy.workers.training_engine_policy_worker.TrainingEnginePolicyWorker"
_MEGATRON_WORKER_FQN = (
    "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker"
)


def _register_worker_environment() -> None:
    """[WAR] Workaround: run the engine worker with the Megatron worker's interpreter.

    The standard path is an entry in ACTOR_ENVIRONMENTS, which builds an environment
    for the worker from the lock file. The lock file pins a Transformer Engine release
    the engine recipes cannot use, so the worker borrows the Megatron worker's
    environment instead. That environment must already exist in the image and hold a
    Transformer Engine the recipe supports.
    TODO: once the lock file carries such a Transformer Engine, add the worker to
    ACTOR_ENVIRONMENTS and delete this function. See ENGINE_TODOS.md.
    """
    venv_dir = os.environ.get("NEMO_RL_VENV_DIR", DEFAULT_VENV_DIR)
    ACTOR_ENVIRONMENT_REGISTRY.setdefault(
        TRAINING_ENGINE_WORKER_FQN,
        os.path.join(venv_dir, _MEGATRON_WORKER_FQN, "bin", "python"),
    )


def _check_supported_loss(loss_fn: LossFunction) -> None:
    """Reject losses the engine's per-rank token sums cannot reproduce."""
    if not isinstance(loss_fn, ClippedPGLossFn):
        raise NotImplementedError(
            "The TrainingEngine policy supports ClippedPGLossFn only, got "
            f"{type(loss_fn).__name__}."
        )
    unsupported = {
        "token_level_loss=false": loss_fn.loss_type != LossType.TOKEN_LEVEL,
        "sequence_level_importance_ratios": loss_fn.sequence_level_importance_ratios,
        "seq_logprob_error_in_loss": loss_fn.seq_logprob_error_in_loss,
        "truncated_importance_sampling_type=seq-mask-tis": (
            loss_fn.truncated_importance_sampling_type == "seq-mask-tis"
        ),
        "score_centering": loss_fn.score_centering,
        "opd_full": loss_fn.opd_full is not None,
        "positive_example_nll_weight": loss_fn.positive_example_nll_weight > 0,
    }
    enabled = [name for name, on in unsupported.items() if on]
    if enabled:
        # TODO: these need whole sequences or their own denominators; the engine hands
        # the loss one context-parallel slice of each sequence.
        raise NotImplementedError(
            f"The TrainingEngine policy does not support loss_fn options: {', '.join(enabled)}."
        )


class TrainingEnginePolicy(Policy):
    """Policy that sends the whole batch to every worker.

    The native policy sorts, packs and shards a batch per data-parallel replica before
    dispatch. The engine does that itself on the workers, so this policy replaces the
    scoring and training calls with a global dispatch and reuses the rest.
    """

    def __init__(
        self,
        cluster: RayVirtualCluster,
        config: PolicyConfig,
        tokenizer: PreTrainedTokenizerBase,
        **kwargs: Any,
    ):
        validate_training_engine_config(config)
        _register_worker_environment()
        super().__init__(
            cluster,
            config,
            tokenizer,
            worker_extension_cls_fqn=TRAINING_ENGINE_WORKER_FQN,
            **kwargs,
        )

    def _run_on_all_workers(
        self, method_name: str, timer: Optional[Timer], **kwargs: Any
    ) -> Any:
        """Run a call that every worker must enter and return rank 0's result."""
        with timer.time(f"{method_name}/all_workers") if timer else nullcontext():
            results = ray.get(
                self.worker_group.run_all_workers_single_data(method_name, **kwargs)
            )
        return next(result for result in results if result is not None)

    def get_logprobs(
        self,
        data: BatchedDataDict[GenerationDatumSpec],
        timer: Optional[Timer] = None,
    ) -> BatchedDataDict[LogprobOutputSpec]:
        """Get the logprobs of the model for a data dict.

        Returns: as Policy.get_logprobs.
        """
        return self._run_on_all_workers("get_logprobs", timer, data=data)

    def get_reference_policy_logprobs(
        self,
        data: BatchedDataDict[GenerationDatumSpec],
        micro_batch_size: Optional[int] = None,
        timer: Optional[Timer] = None,
    ) -> BatchedDataDict[ReferenceLogprobOutputSpec]:
        """Get the logprobs of the reference policy for a data dict.

        Returns: as Policy.get_reference_policy_logprobs.
        """
        if micro_batch_size is not None:
            raise ValueError(
                "micro_batch_size has no effect with policy.training_engine: the "
                "engine packs by policy.training_engine.seq_length."
            )
        return self._run_on_all_workers(
            "get_reference_policy_logprobs", timer, data=data
        )

    def train(
        self,
        data: BatchedDataDict[Any],
        loss_fn: LossFunction,
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
        timer: Optional[Timer] = None,
        check_dim_skip_keys: Optional[Iterable[str]] = None,
    ) -> dict[str, Any]:
        """Train the policy on a batch of data with a given loss function.

        Runs one optimizer update per global batch of samples.
        """
        if eval_mode:
            # TODO: a forward with the loss and no gradient accumulation.
            raise NotImplementedError(
                "eval_mode is not implemented for the TrainingEngine policy yet."
            )
        if mbs is not None or check_dim_skip_keys is not None:
            raise ValueError(
                "mbs and check_dim_skip_keys have no effect with "
                "policy.training_engine: the engine packs by "
                "policy.training_engine.seq_length."
            )
        _check_supported_loss(loss_fn)
        result = self._run_on_all_workers(
            "train",
            timer,
            data=data,
            loss_fn=loss_fn,
            gbs=gbs or self.cfg["train_global_batch_size"],
        )
        return {
            "loss": result["global_loss"],
            "grad_norm": result["grad_norm"],
            "all_mb_metrics": result["all_mb_metrics"],
        }
