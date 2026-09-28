# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run production setup and control flow with test-owned CPU construction."""

from __future__ import annotations

import asyncio
import os
import shlex
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import ray
import torch
from nemo_gym.token_id_capture.staging.records import StagedCallBaseSnapshot
from omegaconf import OmegaConf
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.algorithms.single_controller_utils import MasterConfig
from nemo_rl.algorithms.single_controller_utils import setup as setup_module
from nemo_rl.data_plane import build_data_plane_client
from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
)
from nemo_rl.environments.nemo_gym import (
    NemoGym,
    NemoGymShardSet,
    build_nemo_gym_config,
)
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
from nemo_rl.weight_sync.interfaces import WeightSynchronizer
from tests.mock_stack.config import Scenario
from tests.mock_stack.runtime import GenerationHandle, TrainablePolicy, Trainer
from tests.mock_stack.servers import Call, GenerationServer, TurnGenerator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GYM = ROOT / "3rdparty/Gym-workspace/Gym"


@dataclass
class Run:
    result: dict
    batches: list[tuple[list[str], dict[str, torch.Tensor]]]
    calls: list[Call]
    restored_checkpoint: str | None
    restored_calls: list[StagedCallBaseSnapshot]


@ray.remote(num_cpus=1, num_gpus=0)
class CpuGym(NemoGym.__ray_metadata__.modified_class):
    def _spinup(self):
        os.chdir(HERE / "gym")

        def environment(directory, *_):
            return f"cd {shlex.quote(str(directory))} && export PATH={shlex.quote(str(Path(sys.executable).parent))}:$PATH"

        with patch("nemo_gym.cli.env.setup_env_command", side_effect=environment):
            super()._spinup()


def make_config(scenario: Scenario, output: Path) -> MasterConfig:
    register_omegaconf_resolvers()
    config = load_config(HERE / "controller.yaml")
    config.grpo.num_generations_per_prompt = scenario.siblings_per_prompt
    config.grpo.max_num_steps = len(scenario.prompts)
    config.policy.train_global_batch_size = scenario.siblings_per_prompt
    config.policy.train_micro_batch_size = scenario.siblings_per_prompt
    config.logger.log_dir = str(output / "logs")
    config.checkpointing.checkpoint_dir = str(output / "checkpoints")
    config.token_capture.capture_dir = str(output / "capture" / uuid4().hex)
    gym_config = {
        "config_paths": [
            str(GYM / "responses_api_models/vllm_model/configs/vllm_model.yaml")
        ],
        "counter": {
            "resources_servers": {
                "counter": {
                    "entrypoint": "app.py",
                    "domain": "agent",
                    "verified": False,
                    "description": "CPU checkpoint counter",
                }
            }
        },
    }
    for prompt in scenario.prompts:
        gym_config[f"{prompt.id}_agent"] = {
            "responses_api_agents": {
                "simple_agent": {
                    "entrypoint": "app.py",
                    "max_steps": prompt.turns,
                    "resources_server": {
                        "type": "resources_servers",
                        "name": "counter",
                    },
                    "model_server": {
                        "type": "responses_api_models",
                        "name": "policy_model",
                    },
                }
            }
        }
    config.env = {"should_use_nemo_gym": True, "nemo_gym": gym_config}
    return MasterConfig(**OmegaConf.to_container(config, resolve=True))


def dataset(scenario: Scenario) -> list[dict]:
    rows = []
    for index, prompt in enumerate(scenario.prompts):
        rows.append(
            {
                "idx": index,
                "length": 1,
                "loss_multiplier": 1.0,
                "task_name": "nemo_gym",
                "message_log": [
                    {
                        "role": "user",
                        "content": prompt.id,
                        "token_ids": torch.tensor([1]),
                    }
                ],
                "extra_env_info": {
                    "agent_ref": {
                        "type": "responses_api_agents",
                        "name": f"{prompt.id}_agent",
                    },
                    "initial_count": 0,
                    "expected_count": prompt.turns,
                    "responses_create_params": {
                        "input": [{"role": "user", "content": prompt.id}],
                        "tools": [
                            {
                                "type": "function",
                                "name": "increment_counter",
                                "strict": True,
                                "description": "Increment the saved counter",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"count": {"type": "integer"}},
                                    "required": ["count"],
                                    "additionalProperties": False,
                                },
                            }
                        ],
                    },
                },
            }
        )
    return rows


def run(scenario: Scenario, output: Path) -> Run:
    """Run once. Existing checkpoint discovery decides whether this is a restore."""
    config = make_config(scenario, output)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"[PAD]": 0, "[UNK]": 1}, unk_token="[UNK]")
        ),
        pad_token="[PAD]",
        unk_token="[UNK]",
        eos_token="[PAD]",
    )
    policy = scenario.policy.build()
    generation = scenario.generation.build()
    refit = scenario.refit.build(policy=policy, generation=generation)
    for name, component, contract in (
        ("policy", policy, TrainablePolicy),
        ("generation", generation, TurnGenerator),
        ("refit", refit, WeightSynchronizer),
    ):
        if not isinstance(component, contract):
            raise TypeError(f"{name} must implement {contract.__name__}")
    server = GenerationServer(generation, scenario.prompts)
    handle = GenerationHandle(server)
    trainer = None
    shards = None

    def build_trainer(*args, weights_path, **kwargs):
        nonlocal trainer
        if weights_path is not None:
            policy.load_checkpoint(Path(weights_path))
        plane = build_data_plane_client(
            config.data_plane, bootstrap=True, checkpointing=True
        )
        trainer = Trainer(policy, plane)
        return trainer, 0.0

    def spinup(master_config, base_urls, tokenizer):
        nonlocal shards
        gym_cfg = build_nemo_gym_config(
            master_config.env,
            base_urls=base_urls,
            model_name="cpu",
            enable_router_replay=False,
            use_fastokens=False,
            token_capture=master_config.token_capture.model_dump(),
        )
        actor = CpuGym.remote(gym_cfg)
        shards = NemoGymShardSet({"cpu": [actor]})
        ray.get(actor._spinup.remote())
        ray.get(actor.set_tokenizer.remote(tokenizer))
        return shards, 0.0

    try:
        with ExitStack() as stack:
            stack.enter_context(
                patch.dict(
                    ACTOR_ENVIRONMENT_REGISTRY,
                    {
                        "nemo_rl.experience.rollout_reassembler_actor.RolloutReassemblerActor": sys.executable,
                    },
                )
            )
            replacements = {
                "_build_clusters": lambda *_: (None, None, None),
                "_build_generation": lambda *a, **kw: (handle, 0.0),
                "_build_trainer": build_trainer,
                "_spinup_gym": spinup,
                "create_weight_synchronizer": lambda **kw: refit,
                "setup_response_data": lambda *a, **kw: (dataset(scenario), None),
            }
            for name, replacement in replacements.items():
                stack.enter_context(patch.object(setup_module, name, replacement))
            args, timing = setup_module.setup_single_controller(config, tokenizer)
        restored_calls = (
            server.source.fetch(list(args.gym_checkpoint_staging_keys))
            if args.gym_checkpoint_staging_keys
            else []
        )
        controller = SingleControllerActor.__ray_metadata__.modified_class(
            config, args, timing
        )
        result = asyncio.run(asyncio.wait_for(controller.run(), timeout=300))
        return Run(
            result,
            trainer.batches,
            server.calls,
            args.last_checkpoint_path,
            restored_calls,
        )
    finally:
        if shards is not None:
            shards.shutdown()
        handle.close()
        if trainer is not None:
            trainer.plane.close()
