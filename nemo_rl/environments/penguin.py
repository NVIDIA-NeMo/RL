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

import ray
import torch

from tqdm.auto import tqdm

from nemo_rl.data.datasets import DatumSpec
from nemo_rl.environments.interfaces import EnvironmentInterface

from nemo_rl.distributed.virtual_cluster import _get_node_ip_local, _get_free_port_local


class PenguinConfig(TypedDict):
    model_name: str
    base_urls: List[str]
    initial_global_config_dict: Dict[str, Any]
    total_num_rollouts: int


@ray.remote(max_restarts=-1, max_task_retries=-1)  # pragma: no cover
class Penguin(EnvironmentInterface):
    """
    This environment class isn't really used for training. It's really meant as an integration wrapper around Penguin that hooks into the existing NeMo RL resource management via ray.
    So there is still one source of truth for resource management in NeMo RL.
    """
    def __init__(self, cfg: PenguinConfig):
        self.cfg = cfg

        self.node_ip = _get_node_ip_local()
        self.head_server_port = _get_free_port_local()

        from omegaconf import DictConfig
        from penguin.cli import RunHelper, GlobalConfigDictParserConfig
        from penguin.server_utils import HEAD_SERVER_KEY_NAME

        RELATIVE_PATH = "nemo_rl/environments/penguin.py"
        assert __file__.endswith(RELATIVE_PATH)

        initial_global_config_dict = self.cfg["initial_global_config_dict"]
        # Policy information
        initial_global_config_dict["policy_model_name"] = self.cfg["model_name"]
        initial_global_config_dict["policy_api_key"] = "dummy_key"  # No key necessary for training.
        initial_global_config_dict["policy_base_url"] = self.cfg["base_urls"]

        # Set the connection configuration since we know the full batch size ahead-of-time.
        initial_global_config_dict["global_aiohttp_connector_limit"] = self.cfg["total_num_rollouts"]
        assert self.cfg["total_num_rollouts"] % len(self.cfg["base_urls"]), f"Total number of rollouts ({self.cfg['total_num_rollouts']}) must be divisible by the number of data-parallel vLLM worker instances ({len(self.cfg['base_urls'])})"
        initial_global_config_dict["global_aiohttp_connector_limit_per_host"] = self.cfg["total_num_rollouts"] // len(self.cfg["base_urls"])
        print(
            f"""Penguin was configured with max rollouts {self.cfg['total_num_rollouts']}, so the `global_aiohttp_connector_limit` has been set to the same {self.cfg['total_num_rollouts']}.
Since there are {len(self.cfg['base_urls'])} data-parallel vLLM worker instances, and each instance will receive {self.cfg['total_num_rollouts']} // {len(self.cfg['base_urls'])} = {initial_global_config_dict['global_aiohttp_connector_limit_per_host']} examples. `global_aiohttp_connector_limit_per_host` has been set to the same {initial_global_config_dict['global_aiohttp_connector_limit_per_host']}."""
        )

        # Head server
        initial_global_config_dict[HEAD_SERVER_KEY_NAME] = {
            "host": "0.0.0.0",
            "port": self.head_server_port,
        }

        self.rh = RunHelper()
        self.rh.start(
            global_config_dict_parser_config=GlobalConfigDictParserConfig(
                dotenv_path=Path(__file__.removesuffix(RELATIVE_PATH)).absolute() / "penguin_env.yaml",
                initial_global_config_dict=DictConfig(initial_global_config_dict),
                skip_load_from_cli=True,
            )
        )

    def health_check(self) -> bool:
        return True

    async def run_rollouts(self, penguin_examples: list[dict]) -> list[dict]:
        penguin_results = await self._call_penguin_for_rollouts(penguin_examples)

        nemo_rl_results = list(map(self._postprocess_penguin_to_nemo_rl_result, penguin_results))
        return nemo_rl_results

    async def _call_penguin_for_rollouts(self, examples: list[dict]) -> list[dict]:
        from penguin.server_utils import ServerClient, BaseServerConfig

        head_server_config = BaseServerConfig(
            host=self.node_ip,
            port=self.head_server_port,
        )
        server_client = ServerClient.load_from_global_config(head_server_config)

        async def _post_subroutine(row: dict) -> dict:
            res = await server_client.post(
                server_name=row.pop("agent_ref")["name"], url_path="/run", json=row
            )
            return await res.json()

        return await tqdm.gather(
            *map(_post_subroutine, examples), desc="Collecting Penguin rollouts"
        )

    def _postprocess_penguin_to_nemo_rl_result(self, penguin_result: dict) -> dict:
        nemo_rl_message_log = []
        seen_token_ids: List[int] = []
        for output_item_dict in penguin_result["response"]["output"]:
            # Nemo RL really only has two types of messages: assistant and not assistant since that is all that it is concerned with (i.e. to train or not to train)
            # Here we map all the trainable messages to assistant and all the non-trainable messages to user.
            # Eventually we can maybe be smarter about this, but this is functional for now.

            # Note that Penguin will only return token ids on "assistant" messages and not other message types.
            if "generation_token_ids" not in output_item_dict:
                continue

            assert seen_token_ids == output_item_dict["prompt_token_ids"][:len(seen_token_ids)], f"""Non-contiguous messages found! This may be a tokenization issue where certain tokens are combined when messages are concatenated, or it may be due to part of the chat history being truncated (like if super long history is truncated or if reasoning is stripped out).
Seen token IDs: {seen_token_ids}
Output prompt token IDs: {output_item_dict["prompt_token_ids"]}
"""

            nemo_rl_message_log.append({
                    "role": "user",
                    "content": "",
                    "token_ids": output_item_dict["prompt_token_ids"][len(seen_token_ids):],
                }
            )
            nemo_rl_message_log.append({
                    "role": "assistant",
                    "content": "",
                    "token_ids": output_item_dict["generation_token_ids"],
                    "generation_logprobs": output_item_dict["generation_log_probs"],
                }
            )

            seen_token_ids.extend(nemo_rl_message_log[-2]["token_ids"])
            seen_token_ids.extend(nemo_rl_message_log[-1]["token_ids"])

        return {
            "message_log": nemo_rl_message_log,
            "input_message_log": nemo_rl_message_log[:1],
            "full_result": penguin_result,
        }

    def shutdown(self) -> None:
        self.rh.shutdown()

    def step(self, message_log_batch, metadata):
        # This is not used since Penguin will handle the rollouts entirely.
        raise NotImplementedError

    def global_post_process_and_metrics(self, batch):
        # Similar to the step function, this is not used.
        raise NotImplementedError


########################################
# Global config utils
########################################

def setup_qwen3_penguin_config(config, tokenizer):
    generation_config = config["policy"]["generation"]

    generation_config["vllm_cfg"]["http_server_serving_chat_kwargs"] = {
        "enable_auto_tools": True,
        "tool_parser": "hermes",
    }

    # For Qwen 3 models we need to disable thinking truncation over steps and turns. Here, we modify the chat template to do so.
    chat_template = tokenizer.chat_template
    to_replace = r"""        {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}"""
    if to_replace not in chat_template:
        print("WARNING: If you are using a Qwen3 Thinking model, please note that the reasoning truncation over multiple turns is NOT removed from the chat template. If you are not using a Qwen3 Thinking model, you can disregard this message.")
        return

    chat_template = chat_template.replace(
        to_replace,
        r"""        {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}""",
    )
    tokenizer.chat_template = chat_template
    generation_config["vllm_cfg"]["http_server_serving_chat_kwargs"]["chat_template"] = tokenizer.chat_template


def setup_penguin_config(config, tokenizer) -> None:
    generation_config = config["policy"]["generation"]

    # Enable the http server. Requires both async engine and the expose_http_server flag
    generation_config["vllm_cfg"]["async_engine"] = True
    generation_config["vllm_cfg"]["expose_http_server"] = True

    # Stop strings or token ids are not supported
    generation_config["stop_strings"] = None
    generation_config["stop_token_ids"] = None


########################################
# Data utils
########################################

# We do some light preprocessing here to make our data format compatible with nemo rl format
def penguin_example_to_nemo_rl_datum_spec(penguin_example: dict, idx: int) -> DatumSpec:
    return DatumSpec(
        message_log=[{"role": "user", "content": "", "token_ids": torch.tensor([])}],  # Fake message
        length=0,
        extra_env_info=penguin_example,
        loss_multiplier=1.0,  # Fix to 1.0 to backprop on all examples
        idx=idx,
        task_name="penguin",
        stop_strings=None,
        # Extra vars
        token_ids=[],  # Just need this empty key to be compatible with the current NeMo RL GRPO impl
    )
