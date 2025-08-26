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

from tqdm.auto import tqdm

from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.interfaces import EnvironmentInterface

from nemo_rl.distributed.virtual_cluster import _get_node_ip_and_free_port


class PenguinConfig(TypedDict):
    model_name: str
    base_urls: List[str]
    initial_global_config_dict: Dict[str, Any]


@ray.remote
class PenguinWorker:
    def __init__(self, cfg: PenguinConfig, nemo_rl_openai_base_url: str):
        self.cfg = cfg

        self.nemo_rl_openai_base_url = nemo_rl_openai_base_url
        self.node_ip, self.head_server_port = _get_node_ip_and_free_port()

    def get_node_ip_header_server_port(self) -> tuple[str, int]:
        return self.node_ip, self.head_server_port

    def start(self) -> None:
        # TODO we should probably rename this somehow to penguin. But that is a lot of work...
        from nemo_gym.cli import run
        from nemo_gym.server_utils import HEAD_SERVER_KEY_NAME

        RELATIVE_PATH = "nemo_rl/environments/penguin.py"
        assert __file__.endswith(RELATIVE_PATH)

        initial_global_config_dict = self.cfg["initial_global_config_dict"]
        with open_dict(initial_global_config_dict):
            # Policy information
            initial_global_config_dict["policy_model_name"] = self.cfg["model_name"]
            initial_global_config_dict["policy_api_key"] = "dummy_key"  # No key necessary for training.
            initial_global_config_dict["policy_base_url"] = self.nemo_rl_openai_base_url

            # Head server
            initial_global_config_dict[HEAD_SERVER_KEY_NAME] = {
                "host": "0.0.0.0",
                "port": self.head_server_port,
            }

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

        from nemo_gym.server_utils import BaseServerConfig

        self.head_server_configs: list[BaseServerConfig] = []
        self.start_tasks = []
        for worker in self.workers:
            node_ip, head_server_port = ray.get(worker.get_node_ip_header_server_port.remote())
            self.head_server_configs.append(BaseServerConfig(host=node_ip, port=head_server_port))

            self.start_tasks.append(worker.start.remote())

    async def run_rollouts(self, penguin_examples: list[dict]) -> list[dict]:
        from nemo_gym.server_utils import ServerClient

        # For now, we enforce that the total number of examples in the batch is divisible by the number of workers.
        # This just makes the batching logic below easier.
        assert len(penguin_examples) % len(self.workers) == 0

        batch_size = len(penguin_examples) // len(self.workers)

        tasks = []
        for start_idx, head_server_config in zip(range(0, len(penguin_examples), batch_size), self.head_server_configs):
            this_batch_penguin_examples = penguin_examples[start_idx: start_idx + batch_size]

            this_batch_server_client = ServerClient.load_from_global_config(head_server_config)

            this_batch_tasks = [
                this_batch_server_client.post(
                    server_name=row.pop("agent_ref")["name"], url_path="/run", json=row
                )
                for row in this_batch_penguin_examples
            ]
            tasks.extend(this_batch_tasks)

        results = tqdm.gather(*tasks, desc="Collecting Penguin rollouts")
        return [r.json() for r in results]

    def shutdown(self) -> None:
        for start_task in self.start_tasks:
            ray.cancel(start_task)

        # shutdown all workers
        for worker in self.workers:
            ray.kill(worker)

    def step(self, message_log_batch, metadata):
        # This is not used since NeMo Gym will handle the rollouts entirely.
        raise NotImplementedError

    def global_post_process_and_metrics(self, batch):
        # Similar to the step function, this is not used.
        raise NotImplementedError
