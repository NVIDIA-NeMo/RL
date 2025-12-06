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
from pathlib import Path
from typing import Any, Dict, List, TypedDict

import ray
import torch
from transformers import PreTrainedTokenizerBase

from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.distributed.virtual_cluster import _get_free_port_local, _get_node_ip_local
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.utils.timer import Timer


class NemoGymConfig(TypedDict):
    model_name: str
    base_urls: List[str]
    initial_global_config_dict: Dict[str, Any]


@ray.remote(max_restarts=-1, max_task_retries=-1)  # pragma: no cover
class NemoGym(EnvironmentInterface):
    """This environment class isn't really used for training. It's really meant as an integration wrapper around NeMo-Gym that hooks into the existing NeMo RL resource management via ray. So there is still one source of truth for resource management in NeMo RL."""

    def __init__(self, cfg: NemoGymConfig):
        self.cfg = cfg

        self.node_ip = _get_node_ip_local()
        self.head_server_port = _get_free_port_local()

        from nemo_gym.cli import GlobalConfigDictParserConfig, RunHelper
        from nemo_gym.rollout_collection import RolloutCollectionHelper
        from nemo_gym.server_utils import HEAD_SERVER_KEY_NAME, BaseServerConfig
        from omegaconf import DictConfig

        RELATIVE_PATH = "nemo_rl/environments/nemo_gym.py"
        assert __file__.endswith(RELATIVE_PATH)

        initial_global_config_dict = (
            self.cfg.get("initial_global_config_dict") or dict()
        )
        # Policy information
        initial_global_config_dict["policy_model_name"] = self.cfg["model_name"]
        initial_global_config_dict["policy_api_key"] = (
            "dummy_key"  # No key necessary for training.
        )
        initial_global_config_dict["policy_base_url"] = self.cfg["base_urls"]

        initial_global_config_dict.setdefault(
            "global_aiohttp_connector_limit_per_host", 16_384
        )
        initial_global_config_dict.setdefault("global_aiohttp_connector_limit", 65_536)
        print(
            f"""Set global_aiohttp_connector_limit_per_host={initial_global_config_dict["global_aiohttp_connector_limit_per_host"]} and global_aiohttp_connector_limit={initial_global_config_dict["global_aiohttp_connector_limit"]}.
Depending on your data shape, you may want to change these values."""
        )

        # Get Ray head node address if Ray is initialized
        assert ray.is_initialized(), (
            "Ray must be initialized before using NeMo-Gym environment"
        )
        ray_context = ray.get_runtime_context()
        assert ray_context.gcs_address, "Ray must have a GCS address"

        initial_global_config_dict["ray_head_node_address"] = ray_context.gcs_address
        print(f"Ray head node address: {ray_context.gcs_address}")

        # Head server
        initial_global_config_dict[HEAD_SERVER_KEY_NAME] = {
            "host": "0.0.0.0",
            "port": self.head_server_port,
        }

        self.rh = RunHelper()
        self.rh.start(
            global_config_dict_parser_config=GlobalConfigDictParserConfig(
                dotenv_path=Path(__file__.removesuffix(RELATIVE_PATH)).absolute()
                / "nemo_gym_env.yaml",
                initial_global_config_dict=DictConfig(initial_global_config_dict),
                skip_load_from_cli=True,
            )
        )

        # Setup for rollout collection
        self.head_server_config = BaseServerConfig(
            host=self.node_ip,
            port=self.head_server_port,
        )
        self.rch = RolloutCollectionHelper()

    def health_check(self) -> bool:
        return True

    async def run_rollouts(
        self,
        nemo_gym_examples: list[dict],
        tokenizer: PreTrainedTokenizerBase,
        timer_prefix: str,
    ) -> list[dict]:
        timer = Timer()

        nemo_gym_result_iterator = self.rch.run_examples(
            examples=nemo_gym_examples, head_server_config=self.head_server_config
        )

        timer.start("_run_rollouts_total")
        nemo_rl_results = []
        for task in nemo_gym_result_iterator:
            with timer.time(label=f"{timer_prefix}/await_results"):
                nemo_gym_result = await task

            with timer.time(label=f"{timer_prefix}/postprocess_results"):
                # Check if this result contains transitions
                contains_transitions = nemo_gym_result["response"].get(
                    "contains_transitions", False
                )
                if contains_transitions:
                    # Postprocess as transitions - expands to multiple training samples
                    nemo_rl_result_list = (
                        self._postprocess_nemo_gym_transitions_to_nemo_rl_results(
                            nemo_gym_result, tokenizer
                        )
                    )
                    nemo_rl_results.extend(nemo_rl_result_list)
                else:
                    # Standard single-trajectory postprocessing
                    nemo_rl_result = self._postprocess_nemo_gym_to_nemo_rl_result(
                        nemo_gym_result, tokenizer
                    )
                    nemo_rl_results.append(nemo_rl_result)

        timer.stop("_run_rollouts_total")
        timing_metrics = timer.get_timing_metrics("sum")
        total_time = timing_metrics.pop("_run_rollouts_total")
        timing_metrics[f"{timer_prefix}/postprocess_results_pct"] = (
            100 * timing_metrics[f"{timer_prefix}/postprocess_results"] / total_time
        )

        return nemo_rl_results, timing_metrics

    def _postprocess_nemo_gym_to_nemo_rl_result(
        self, nemo_gym_result: dict, tokenizer: PreTrainedTokenizerBase
    ) -> dict:
        nemo_rl_message_log = []
        seen_token_ids: List[int] = []
        for output_item_dict in nemo_gym_result["response"]["output"]:
            # Nemo RL really only has two types of messages: assistant and not assistant since that is all that it is concerned with (i.e. to train or not to train)
            # Here we map all the trainable messages to assistant and all the non-trainable messages to user.
            # Eventually we can maybe be smarter about this, but this is functional for now.

            # Note that NeMo-Gym will only return token ids on "assistant" messages and not other message types.
            if "generation_token_ids" not in output_item_dict:
                continue

            assert (
                seen_token_ids
                == output_item_dict["prompt_token_ids"][: len(seen_token_ids)]
            ), f"""Non-contiguous messages found! This may be a tokenization issue where certain tokens are combined when messages are concatenated, or it may be due to part of the chat history being truncated (like if super long history is truncated or if reasoning is stripped out).
Seen token IDs: {seen_token_ids}
Output prompt token IDs: {output_item_dict["prompt_token_ids"]}
"""

            nemo_rl_message_log.append(
                {
                    "role": "user",
                    "content": "",
                    "token_ids": torch.tensor(
                        output_item_dict["prompt_token_ids"][len(seen_token_ids) :]
                    ),
                }
            )
            nemo_rl_message_log.append(
                {
                    "role": "assistant",
                    "content": "",
                    "token_ids": torch.tensor(output_item_dict["generation_token_ids"]),
                    "generation_logprobs": torch.tensor(
                        output_item_dict["generation_log_probs"]
                    ),
                }
            )

            seen_token_ids.extend(nemo_rl_message_log[-2]["token_ids"])
            seen_token_ids.extend(nemo_rl_message_log[-1]["token_ids"])

            # We pop to remove larger tensors from logging.
            output_item_dict["prompt_str"] = tokenizer.decode(
                output_item_dict.pop("prompt_token_ids")
            )
            output_item_dict["generation_str"] = tokenizer.decode(
                output_item_dict.pop("generation_token_ids")
            )
            output_item_dict.pop("generation_log_probs")

        return {
            "message_log": nemo_rl_message_log,
            "input_message_log": nemo_rl_message_log[:1],
            "full_result": nemo_gym_result,
        }

    def _postprocess_nemo_gym_transition_to_nemo_rl_result(
        self, nemo_gym_result: dict, tokenizer: PreTrainedTokenizerBase
    ) -> dict:
        """Postprocess a single transition to NeMo RL format.

        When operating on transitions, we don't guarantee monotonicity, which _postprocess_nemo_gym_to_nemo_rl_result
        requires. Without monotonicity, it becomes trickier to split up token IDs into messages, although
        it would still be possible to iterate through messages in *reverse* order. But really, we only need
        to separate the last (trainable) assistant message from the rest of the (untrainable) tokens. This
        function puts all those untrainable tokens into a single user message.
        """
        nemo_rl_message_log = []
        for output_item_dict in reversed(nemo_gym_result["response"]["output"]):
            # Note that NeMo-Gym will only return token ids on "assistant" messages and not other message types.
            if "generation_token_ids" not in output_item_dict:
                continue

            nemo_rl_message_log.append(
                {
                    "role": "user",
                    "content": "",
                    "token_ids": torch.tensor(output_item_dict["prompt_token_ids"]),
                }
            )
            nemo_rl_message_log.append(
                {
                    "role": "assistant",
                    "content": "",
                    "token_ids": torch.tensor(output_item_dict["generation_token_ids"]),
                    "generation_logprobs": torch.tensor(
                        output_item_dict["generation_log_probs"]
                    ),
                }
            )

            # We pop to remove larger tensors from logging.
            output_item_dict["prompt_str"] = tokenizer.decode(
                output_item_dict.pop("prompt_token_ids")
            )
            output_item_dict["generation_str"] = tokenizer.decode(
                output_item_dict.pop("generation_token_ids")
            )
            output_item_dict.pop("generation_log_probs")

            break

        return {
            "message_log": nemo_rl_message_log,
            "input_message_log": nemo_rl_message_log[:1],
            "full_result": nemo_gym_result,
        }

    def _postprocess_nemo_gym_transitions_to_nemo_rl_results(
        self, nemo_gym_result: dict, tokenizer: PreTrainedTokenizerBase
    ) -> list[dict]:
        """Postprocess a list of transitions to NeMo RL format.

        Should be called if nemo_gym_result contains a list of transitions,
        not one monotonic message history.
        """
        transitions = nemo_gym_result["response"]["output"]
        total_length = len(transitions)

        output: list[dict] = []

        for timestep, transition in enumerate(transitions, start=1):
            # Match the data structure expected by _postprocess_nemo_gym_transition_to_nemo_rl_result,
            # while preserving as much of the original nemo_gym_result as possible.
            # NOTE: This is where we broadcast the reward to all transitions. In particular,
            # we are assuming that the rewards are assigned per-trajectory (well, it's
            # really an assumption coming from NeMo-Gym).
            # NOTE: deepcopy was too expensive, so do it like this.
            wrapped_transition = nemo_gym_result | {
                "response": nemo_gym_result["response"] | {"output": transition}
            }
            output.append(
                {
                    **self._postprocess_nemo_gym_transition_to_nemo_rl_result(
                        wrapped_transition, tokenizer
                    ),
                    "metadata": {
                        "contains_transitions": True,
                        "timestep": timestep,
                        "trajectory_length": total_length,
                        "is_last": timestep == total_length,
                        "group_id": wrapped_transition["response"]["group_id"],
                    },
                }
            )

        return output

    def shutdown(self) -> None:
        self.rh.shutdown()

    def step(self, message_log_batch, metadata):
        # This is not used since NeMo-Gym will handle the rollouts entirely.
        raise NotImplementedError

    def global_post_process_and_metrics(self, batch):
        # Similar to the step function, this is not used.
        raise NotImplementedError


########################################
# Global config utils
########################################


def setup_nemo_gym_config(config, tokenizer) -> None:
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
def nemo_gym_example_to_nemo_rl_datum_spec(
    nemo_gym_example: dict, idx: int
) -> DatumSpec:
    return DatumSpec(
        message_log=[
            {"role": "user", "content": "", "token_ids": torch.tensor([])}
        ],  # Fake message
        length=0,
        extra_env_info=nemo_gym_example,
        loss_multiplier=1.0,  # Fix to 1.0 to backprop on all examples
        idx=idx,
        task_name="nemo_gym",
        stop_strings=None,
        # Extra vars
        token_ids=[],  # Just need this empty key to be compatible with the current NeMo RL GRPO impl
    )
