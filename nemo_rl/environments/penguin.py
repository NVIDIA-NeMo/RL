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

from nemo_rl.distributed.virtual_cluster import _get_node_ip_local, _get_free_port_local


class PenguinConfig(TypedDict):
    model_name: str
    base_urls: List[str]
    initial_global_config_dict: Dict[str, Any]


@ray.remote
class PenguinWorker:
    def __init__(self, cfg: PenguinConfig, nemo_rl_openai_base_url: str):
        self.cfg = cfg

        self.nemo_rl_openai_base_url = nemo_rl_openai_base_url
        self.node_ip = _get_node_ip_local()
        self.head_server_port = _get_free_port_local()

        # TODO we should probably rename this somehow to penguin. But that is a lot of work...
        from omegaconf import DictConfig
        from nemo_gym.cli import RunHelper
        from nemo_gym.server_utils import HEAD_SERVER_KEY_NAME

        RELATIVE_PATH = "nemo_rl/environments/penguin.py"
        assert __file__.endswith(RELATIVE_PATH)

        initial_global_config_dict = self.cfg["initial_global_config_dict"]
        # Policy information
        initial_global_config_dict["policy_model_name"] = self.cfg["model_name"]
        initial_global_config_dict["policy_api_key"] = "dummy_key"  # No key necessary for training.
        initial_global_config_dict["policy_base_url"] = self.nemo_rl_openai_base_url

        # Head server
        initial_global_config_dict[HEAD_SERVER_KEY_NAME] = {
            "host": "0.0.0.0",
            "port": self.head_server_port,
        }

        self.rh = RunHelper()
        self.rh.start(
            dotenv_path=Path(__file__.removesuffix(RELATIVE_PATH)).absolute() / "env.yaml",
            initial_global_config_dict=DictConfig(initial_global_config_dict),
        )

    async def _call_penguin_for_rollouts(self, examples: list[dict]) -> list[dict]:
        from nemo_gym.server_utils import ServerClient, BaseServerConfig

        head_server_config = BaseServerConfig(
            host=self.node_ip,
            port=self.head_server_port,
        )
        server_client = ServerClient.load_from_global_config(head_server_config)

        tasks = [
            server_client.post(
                server_name=row.pop("agent_ref")["name"], url_path="/run", json=row
            )
            for row in examples
        ]

        results = await tqdm.gather(*tasks, desc="Collecting Penguin rollouts")
        return [r.json() for r in results]

    def _postprocess_penguin_to_nemo_rl_result(self, penguin_result: dict) -> dict:
        from nemo_gym.openai_utils import NeMoGymResponse

        # Check if it is indeed what we expect to receive here.
        NeMoGymResponse.model_validate(penguin_result["response"])

        nemo_rl_message_log = []
        seen_token_ids: List[int] = []
        for output_item_dict in penguin_result["response"]["output"]:
            # Nemo RL really only has two types of messages: assistant and not assistant since that is all that it is concerned with (i.e. to train or not to train)
            # Here we map all the trainable messages to assistant and all the non-trainable messages to user.
            # Eventually we can maybe be smarter about this, but this is functional for now.

            # Note that Penguin will only return token ids on "assistant" messages and not other message types.
            if "generation_token_ids" not in output_item_dict:
                continue

            assert seen_token_ids == output_item_dict["prompt_token_ids"][:seen_token_ids], "Non-contiguous messages found! This may be a tokenization issue where certain tokens are combined when messages are concatenated."

            nemo_rl_message_log.append({
                    "role": "user",
                    "content": "",
                    "token_ids": output_item_dict["prompt_token_ids"][seen_token_ids:],
                }
            )
            nemo_rl_message_log.append({
                    "role": "assistant",
                    "content": "",
                    "token_ids": output_item_dict["generation_token_ids"],
                    "generation_logprobs": output_item_dict["generation_logprobs"],
                }
            )

            seen_token_ids.extend(nemo_rl_message_log[-2]["token_ids"])
            seen_token_ids.extend(nemo_rl_message_log[-1]["token_ids"])

        return {
            "message_log": nemo_rl_message_log[1:],
            "input_message_log": nemo_rl_message_log[:1],
        }

    async def run_rollouts(self, examples: list[dict]) -> list[dict]:
        penguin_results = await self._call_penguin_for_rollouts(examples)

        nemo_rl_results = list(map(self._postprocess_penguin_to_nemo_rl_result, penguin_results))
        return nemo_rl_results


@ray.remote(max_restarts=-1, max_task_retries=-1)  # pragma: no cover
class Penguin(EnvironmentInterface):
    """
    This environment class isn't really used for training. It's really meant as an integration wrapper around Penguin that hooks into the existing NeMo RL resource management via ray.
    So there is still one source of truth for resource management in NeMo RL.
    """
    def __init__(self, cfg: PenguinConfig):
        self.cfg = cfg

        self.workers = [
            PenguinWorker.options(  # type: ignore # (decorated with @ray.remote)
                runtime_env={"py_executable": PY_EXECUTABLES.PENGUIN}
            ).remote(self.cfg, base_url)
            for base_url in self.cfg["base_urls"]
        ]

    async def run_rollouts(self, penguin_examples: list[dict]) -> list[dict]:
        # For now, we enforce that the total number of examples in the batch is divisible by the number of workers.
        # This just makes the batching logic below easier.
        assert len(penguin_examples) % len(self.workers) == 0

        batch_size = len(penguin_examples) // len(self.workers)

        futures = []
        for start_idx, worker in zip(range(0, len(penguin_examples), batch_size), self.workers):
            this_batch_penguin_examples = penguin_examples[start_idx: start_idx + batch_size]
            future = worker.run_rollouts.remote(this_batch_penguin_examples)
            futures.append(future)

        results = []
        for result in ray.get(futures):
            results.extend(result)

        return results

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
