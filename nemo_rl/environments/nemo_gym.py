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

from omegaconf import OmegaConf, open_dict

import ray

from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.interfaces import EnvironmentInterface


class NeMoGymConfig(TypedDict):
    model_name: str
    base_urls: List[str]
    initial_global_config_dict: Dict[str, Any]


@ray.remote
def start_nemo_gym(cfg: NeMoGymConfig, nemo_rl_openai_base_url: str):
    from nemo_gym.cli import run

    RELATIVE_PATH = "nemo_rl/environments/nemo_gym.py"
    assert __file__.endswith(RELATIVE_PATH)

    initial_global_config_dict = cfg["initial_global_config_dict"]
    with open_dict(initial_global_config_dict):
        initial_global_config_dict["nemo_rl_openai_model_name"] = cfg["model_name"]
        initial_global_config_dict["nemo_rl_openai_base_url"] = nemo_rl_openai_base_url

    run(
        dotenv_path=Path(__file__.removesuffix(RELATIVE_PATH)).absolute(),
        initial_global_config_dict=initial_global_config_dict,
    )


@ray.remote(max_restarts=-1, max_task_retries=-1)  # pragma: no cover
class NeMoGym(EnvironmentInterface):
    """
    This environment class isn't really used for training. It's really meant as an integration wrapper around NeMo Gym that hooks into the existing NeMo RL resource management via ray.
    So there is still one source of truth for resource management in NeMo RL.
    """
    def __init__(self, cfg: NeMoGymConfig):
        self.cfg = cfg

        self.workers = [
            start_nemo_gym.options(  # type: ignore # (decorated with @ray.remote)
                runtime_env={"py_executable": PY_EXECUTABLES.NEMO_GYM}
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
