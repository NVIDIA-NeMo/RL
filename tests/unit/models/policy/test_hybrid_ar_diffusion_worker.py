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

"""Guards on the hybrid AR + diffusion policy worker.

The worker itself cannot be instantiated without a GPU and a Megatron model, so
these cover the three seams that break silently without one: the actor-registry
entry ``lm_policy`` validates before any worker spawns, the estimator config
accessors (the config arrives as a dict on some paths and as a validated
pydantic model on others), and the base-class method signatures the Policy layer
calls into.
"""

import inspect

import pytest

from nemo_rl.algorithms.hybrid_ar_diffusion import (
    HYBRID_AR_DIFFUSION_WORKER_FQN,
    get_hybrid_ar_diffusion_cfg,
)
from nemo_rl.models.policy import HybridARDiffusionLogprobEstimationConfig

MINIMAL_ESTIMATION_CFG = {
    "type": "hybrid_ar_diffusion",
    "mask_token_id": 3,
    "ce_loss_weight": 0.1,
}


class TestActorRegistration:
    """``lm_policy`` calls ``get_actor_python_env(extension_fqn)`` eagerly.

    An unregistered FQN aborts the run before any placement group is allocated,
    with a message that reads like a config error rather than a missing code
    change.
    """

    def test_worker_fqn_resolves_to_an_actor_environment(self):
        from nemo_rl.distributed.ray_actor_environment_registry import (
            get_actor_python_env,
        )

        assert get_actor_python_env(HYBRID_AR_DIFFUSION_WORKER_FQN)

    def test_worker_fqn_shares_the_megatron_worker_venv(self):
        """Venvs are cached by actor class name, so a subclass must match its base."""
        from nemo_rl.distributed.actor_environments import ACTOR_ENVIRONMENTS

        assert (
            ACTOR_ENVIRONMENTS[HYBRID_AR_DIFFUSION_WORKER_FQN]
            == (
                ACTOR_ENVIRONMENTS[
                    "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker"
                ]
            )
        )


class TestEstimationConfigAccess:
    def test_exclude_mask_token_from_logits_defaults_to_true(self):
        """Pinned: the default changes the CE term's normalization."""
        cfg = HybridARDiffusionLogprobEstimationConfig.model_validate(
            MINIMAL_ESTIMATION_CFG
        )
        assert cfg.exclude_mask_token_from_logits is True

    def test_exclude_mask_token_from_logits_is_a_declared_field(self):
        """Not ``model_extra``: an undeclared key is silently unreachable."""
        assert (
            "exclude_mask_token_from_logits"
            in HybridARDiffusionLogprobEstimationConfig.model_fields
        )
        cfg = HybridARDiffusionLogprobEstimationConfig.model_validate(
            {**MINIMAL_ESTIMATION_CFG, "exclude_mask_token_from_logits": False}
        )
        assert cfg.model_extra == {}
        assert cfg.exclude_mask_token_from_logits is False

    @pytest.mark.parametrize("already_validated", [False, True])
    def test_accessor_takes_both_a_dict_and_a_validated_model(self, already_validated):
        """``MasterConfig`` coerces ``policy.logprob_estimation`` into a model.

        Hand-built policy configs (and every unit test) still pass a plain dict,
        so the worker must not subscript it and must not assume ``.get``.
        """
        estimation = dict(MINIMAL_ESTIMATION_CFG)
        if already_validated:
            estimation = HybridARDiffusionLogprobEstimationConfig.model_validate(
                estimation
            )
        cfg = get_hybrid_ar_diffusion_cfg({"logprob_estimation": estimation})
        assert isinstance(cfg, HybridARDiffusionLogprobEstimationConfig)
        assert cfg.mask_token_id == 3
        assert cfg.noisy_tail_mode == "mask"


@pytest.mark.mcore
class TestWorkerChain:
    def test_the_whole_chain_imports(self):
        """hybrid worker -> diffu_grpo worker -> diffusion worker -> train hooks."""
        from nemo_rl.models.policy.workers.diffu_grpo_megatron_policy_worker import (
            DiffuGRPOMegatronPolicyWorkerImpl,
        )
        from nemo_rl.models.policy.workers.diffusion_megatron_policy_worker import (
            DiffusionMegatronPolicyWorkerImpl,
        )
        from nemo_rl.models.policy.workers.hybrid_ar_diffusion_megatron_policy_worker import (
            HybridARDiffusionMegatronPolicyWorkerImpl,
        )
        from nemo_rl.models.policy.workers.megatron_policy_worker import (
            MegatronPolicyWorkerImpl,
        )

        mro = HybridARDiffusionMegatronPolicyWorkerImpl.__mro__
        assert mro[:4] == (
            HybridARDiffusionMegatronPolicyWorkerImpl,
            DiffuGRPOMegatronPolicyWorkerImpl,
            DiffusionMegatronPolicyWorkerImpl,
            MegatronPolicyWorkerImpl,
        )

    def test_the_leaf_is_concrete(self):
        """Every abstract diffusion hook is implemented, so Ray can construct it."""
        from nemo_rl.models.policy.workers.hybrid_ar_diffusion_megatron_policy_worker import (
            HybridARDiffusionMegatronPolicyWorkerImpl,
        )

        assert not getattr(
            HybridARDiffusionMegatronPolicyWorkerImpl,
            "__abstractmethods__",
            frozenset(),
        )

    def test_train_accepts_check_dim_skip_keys(self):
        """``lm_policy`` passes it unconditionally on every train call."""
        from nemo_rl.models.policy.workers.diffusion_megatron_policy_worker import (
            DiffusionMegatronPolicyWorkerImpl,
        )
        from nemo_rl.models.policy.workers.megatron_policy_worker import (
            MegatronPolicyWorkerImpl,
        )

        assert list(
            inspect.signature(DiffusionMegatronPolicyWorkerImpl.train).parameters
        ) == list(inspect.signature(MegatronPolicyWorkerImpl.train).parameters)

    def test_get_logprobs_accepts_require_router_replay(self):
        """``get_reference_policy_logprobs`` passes it explicitly."""
        from nemo_rl.models.policy.workers.diffusion_megatron_policy_worker import (
            DiffusionMegatronPolicyWorkerImpl,
        )
        from nemo_rl.models.policy.workers.megatron_policy_worker import (
            MegatronPolicyWorkerImpl,
        )

        assert list(
            inspect.signature(DiffusionMegatronPolicyWorkerImpl.get_logprobs).parameters
        ) == list(inspect.signature(MegatronPolicyWorkerImpl.get_logprobs).parameters)

    def test_configure_worker_is_inherited_not_overridden(self):
        """The fork's override returned a 3-tuple and pinned num_gpus=0.

        ``worker_groups`` unpacks four values, and ``__init__`` opens with
        ``int(ray.get_gpu_ids()[0])`` -- which IndexErrors inside every actor
        when num_gpus is 0. Inheriting is the fix; re-adapting is not.
        """
        from nemo_rl.models.policy.workers.diffusion_megatron_policy_worker import (
            DiffusionMegatronPolicyWorkerImpl,
        )
        from nemo_rl.models.policy.workers.megatron_policy_worker import (
            MegatronPolicyWorkerImpl,
        )

        assert (
            DiffusionMegatronPolicyWorkerImpl.configure_worker
            is MegatronPolicyWorkerImpl.configure_worker
        )

    def test_no_sglang_surface(self):
        """The refit-over-HTTP override was the stack's only SGLang reach."""
        from nemo_rl.models.policy.workers import diffusion_megatron_policy_worker

        assert not hasattr(
            diffusion_megatron_policy_worker.DiffusionMegatronPolicyWorkerImpl,
            "stream_weights_via_http",
        )
