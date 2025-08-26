# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Any, Dict, List, TypedDict

from pathlib import Path

from omegaconf import open_dict

import ray

from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.interfaces import EnvironmentInterface


class PenguinConfig(TypedDict):
    model_name: str
    base_urls: List[str]
    initial_global_config_dict: Dict[str, Any]


@ray.remote
def start_penguin(cfg: PenguinConfig, nemo_rl_openai_base_url: str):
    # TODO we should probably rename this somehow to penguin. But that is a lot of work...
    from nemo_gym.cli import run

    RELATIVE_PATH = "nemo_rl/environments/penguin.py"
    assert __file__.endswith(RELATIVE_PATH)

    initial_global_config_dict = cfg["initial_global_config_dict"]
    with open_dict(initial_global_config_dict):
        initial_global_config_dict["policy_model_name"] = cfg["model_name"]
        initial_global_config_dict["policy_api_key"] = "dummy_key"  # No key necessary for training.
        initial_global_config_dict["policy_base_url"] = nemo_rl_openai_base_url

    run(
        dotenv_path=Path(__file__.removesuffix(RELATIVE_PATH)).absolute() / "env.yaml",
        initial_global_config_dict=initial_global_config_dict,
    )


@ray.remote(max_restarts=-1, max_task_retries=-1)  # pragma: no cover
class Penguin(EnvironmentInterface):
    """
    This environment class isn't really used for training. It's really meant as an integration wrapper around Penguin that hooks into the existing NeMo RL resource management via ray.
    So there is still one source of truth for resource management in NeMo RL.
    """
    def __init__(self, cfg: PenguinConfig):
        self.cfg = cfg

        self.workers = [
            start_penguin.options(  # type: ignore # (decorated with @ray.remote)
                runtime_env={"py_executable": PY_EXECUTABLES.PENGUIN}
            ).remote(self.cfg)
            for _ in range(self.cfg["base_urls"])
        ]

    def shutdown(self) -> None:
        # shutdown all workers
        for worker in self.workers:
            ray.kill(worker)

    def step(self, message_log_batch, metadata):
        # This is not used since NeMo Gym will handle the rollouts entirely.
        raise NotImplementedError

    def global_post_process_and_metrics(self, batch):
        # Similar to the step function, this is not used.
        raise NotImplementedError
