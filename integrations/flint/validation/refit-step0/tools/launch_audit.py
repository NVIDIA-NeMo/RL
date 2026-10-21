# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Use the completed four-layer launcher's registration and logging unchanged."""
import os
import runpy
import sys
from pathlib import Path

os.environ['NEMO_RL_PY_EXECUTABLES_SYSTEM']='1'
from nemo_automodel._transformers import registry
from nemo_rl.distributed.ray_actor_environment_registry import ACTOR_ENVIRONMENT_REGISTRY
from nemo_rl.environments.utils import register_env
assert 'FlintForCausalLM' in registry.MODEL_ARCH_MAPPING
from omegaconf import OmegaConf
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.utils.checkpoint import CheckpointManager
register_omegaconf_resolvers()
checked=MasterConfig(**OmegaConf.to_container(load_config('/flint-audit/recipe.yaml'),resolve=True))
CheckpointManager(checked.checkpointing)
assert checked.policy['automodel_cfg']['save_consolidated']=='false'
for actor,profile in {
    'nemo_rl.experience.sync_rollout_actor.SyncRolloutActor':'driver',
    'nemo_rl.models.policy.workers.automodel_policy_worker.AutomodelPolicyWorker':'training',
    'nemo_rl.models.value.workers.automodel_value_worker.AutomodelValueWorker':'training',
    'nemo_rl.models.generation.sglang.sglang_worker.SGLangGenerationWorker':'audit',
    'nemo_rl.models.generation.sglang.sglang_router.RouterActor':'audit',
    'flint_smoke_environment.FlintSmokeEnvironment':'driver',
}.items():
    ACTOR_ENVIRONMENT_REGISTRY[actor]='/flint-audit/tools/python-sglang-audit.sh' if profile=='audit' else f'/opt/flint/bin/python-{profile}'
register_env('flint_smoke','flint_smoke_environment.FlintSmokeEnvironment')
original=runpy.run_path('/flint-work/tools/launch_grpo.py')
original['capture_metrics']('/flint-audit/evidence/training.metrics.jsonl')
sys.argv=['examples/run_grpo.py','--config','/flint-audit/recipe.yaml']
try:
    runpy.run_path('examples/run_grpo.py',run_name='__main__')
finally:
    import ray
    ray.shutdown()
