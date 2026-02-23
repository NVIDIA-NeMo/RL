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
from typing import Any, Dict, List, Optional, TypedDict

import ray
import torch
from transformers import PreTrainedTokenizerBase

from nemo_rl.distributed.virtual_cluster import _get_free_port_local, _get_node_ip_local
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.utils.timer import Timer


class NemoGymConfig(TypedDict):
    model_name: str
    base_urls: List[str]
    ray_gpu_nodes: List[str]
    ray_gpu_pgs: List
    ray_num_gpus_per_node: Optional[int]
    ray_namespace: Optional[str]
    initial_global_config_dict: Dict[str, Any]
    invalid_tool_call_patterns: Optional[List[str]]  # Substrings in assistant text content that indicate an invalid tool call (default: ["<tool_call>", "</tool_call>", "<function_call>", "</function_call>"])
    thinking_tags: Optional[List[str]]  # Thinking tags to check for malformed usage (default: ["<think>", "</think>"])


class GenRMCompareConfig(TypedDict, total=False):
    """Configuration for GenRM batch comparison."""

    enabled: bool
    agent_names: List[str]
    server_name: str
    num_generations_per_prompt: int
    policy_model_server_name: str


@ray.remote(max_restarts=-1, max_task_retries=-1)  # pragma: no cover
class NemoGym(EnvironmentInterface):
    """This environment class isn't really used for training. It's really meant as an integration wrapper around NeMo-Gym that hooks into the existing NeMo RL resource management via ray. So there is still one source of truth for resource management in NeMo RL."""

    def __init__(self, cfg: NemoGymConfig):
        self.cfg = cfg

    def _spinup(self) -> None:
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

        ray_namespace = self.cfg.get("ray_namespace", None)
        if ray_namespace is not None:
            initial_global_config_dict["ray_namespace"] = ray_namespace
            print(f"Ray namespace: {ray_namespace}")

        initial_global_config_dict["ray_gpu_nodes"] = self.cfg["ray_gpu_nodes"]
        # ray_gpu_pgs are Ray PlacementGroup objects — can't go through OmegaConf.
        # They are passed separately to the scheduling helper via set_gpu_pgs().
        initial_global_config_dict["ray_num_gpus_per_node"] = self.cfg[
            "ray_num_gpus_per_node"
        ]
        print(
            f"Ray reserved GPU nodes: {len(initial_global_config_dict['ray_gpu_nodes'])}"
        )
        print(
            f"Ray num GPUs per node: {initial_global_config_dict['ray_num_gpus_per_node']}"
        )

        # Head server
        initial_global_config_dict[HEAD_SERVER_KEY_NAME] = {
            "host": "0.0.0.0",
            "port": self.head_server_port,
        }

        self.rollout_max_attempts_to_avoid_lp_nan = initial_global_config_dict.pop(
            "rollout_max_attempts_to_avoid_lp_nan", 1
        )

        assert self.rollout_max_attempts_to_avoid_lp_nan >= 1, (
            "`rollout_max_attempts_to_avoid_lp_nan` must be at least 1"
        )

        self.rh = RunHelper()
        self.rh.start(
            global_config_dict_parser_config=GlobalConfigDictParserConfig(
                dotenv_path=Path(__file__.removesuffix(RELATIVE_PATH)).absolute()
                / "nemo_gym_env.yaml",
                initial_global_config_dict=DictConfig(initial_global_config_dict),
                skip_load_from_cli=True,
            ),
            ray_gpu_pgs=self.cfg["ray_gpu_pgs"],
            ray_gpu_nodes=self.cfg["ray_gpu_nodes"],
        )

        # Setup for rollout collection
        self.head_server_config = BaseServerConfig(
            host=self.node_ip,
            port=self.head_server_port,
        )
        self.rch = RolloutCollectionHelper()

    async def run_rollouts(
        self,
        nemo_gym_examples: list[dict],
        tokenizer: PreTrainedTokenizerBase,
        timer_prefix: str,
        genrm_config: Optional[GenRMCompareConfig] = None,
    ) -> list[dict]:
        timer = Timer()

        # Build comparison strategy if GenRM is enabled
        comparison_strategy = None
        if genrm_config and genrm_config.get("enabled", False):
            from nemo_gym.comparison_strategies import (
                GenRMStrategy,
                GenRMStrategyConfig,
            )

            comparison_strategy = GenRMStrategy(
                GenRMStrategyConfig(
                    agent_names=genrm_config.get("agent_names", ["genrm_simple_agent"]),
                    genrm_compare_server_name=genrm_config.get(
                        "server_name", "genrm_compare"
                    ),
                    policy_model_server_name=genrm_config.get("policy_model_server_name", "policy_model"),
                    num_generations_per_prompt=genrm_config.get(
                        "num_generations_per_prompt", 16
                    ),
                )
            )

        timer.start("_run_rollouts_total")
        max_attempts, trial = self.rollout_max_attempts_to_avoid_lp_nan, 0
        while trial < max_attempts:
            nemo_gym_num_rows = len(nemo_gym_examples)
            nemo_gym_result_iterator = self.rch.run_examples(
                examples=nemo_gym_examples,
                head_server_config=self.head_server_config,
                comparison_strategy=comparison_strategy,
            )

            nemo_rl_rowidxs = []
            nemo_rl_results = []
            for task in nemo_gym_result_iterator:
                with timer.time(label=f"{timer_prefix}/await_results"):
                    nemo_gym_row, nemo_gym_result = await task

                with timer.time(label=f"{timer_prefix}/postprocess_results"):
                    nemo_rl_result = self._postprocess_nemo_gym_to_nemo_rl_result(
                        nemo_gym_result, tokenizer
                    )

                nemo_rl_rowidxs.append(nemo_gym_row["_rowidx"])
                nemo_rl_results.append(nemo_rl_result)

            # determine if generation_logprobs contain NaN; if not, break;
            logprob_contains_nan = False
            for nemo_rl_result in nemo_rl_results:
                for message in nemo_rl_result["message_log"]:
                    if (
                        "generation_logprobs" in message
                        and message["generation_logprobs"] is not None
                    ):
                        if torch.isnan(message["generation_logprobs"]).any():
                            logprob_contains_nan = True
                            break
            if logprob_contains_nan:
                trial += 1
                print(
                    f"Generation logprobs contain NaN; retrying... (trial {trial}/{max_attempts})"
                )
                continue
            else:
                break

        nemo_rl_sort_results = [None] * nemo_gym_num_rows
        for rowidx, result in zip(nemo_rl_rowidxs, nemo_rl_results):
            nemo_rl_sort_results[rowidx] = result
        nemo_rl_results = nemo_rl_sort_results

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
        assert isinstance(nemo_gym_result, dict), (
            f"Hit a non-successful response when querying NeMo Gym for rollouts: {nemo_gym_result}"
        )

        nemo_rl_message_log = []
        seen_token_ids = torch.tensor([])

        batch_decode_items = []  # Collect (output_item_dict, prompt_token_ids, generation_token_ids) for batch decode
        for output_item_dict in nemo_gym_result["response"]["output"]:
            # Nemo RL really only has two types of messages: assistant and not assistant since that is all that it is concerned with (i.e. to train or not to train)
            # Here we map all the trainable messages to assistant and all the non-trainable messages to user.
            # Eventually we can maybe be smarter about this, but this is functional for now.

            # Note that NeMo-Gym will only return token ids on "assistant" messages and not other message types.
            if "generation_token_ids" not in output_item_dict:
                continue

            prompt_token_ids_tensor = torch.tensor(output_item_dict["prompt_token_ids"])
            n_seen = len(seen_token_ids)
            if n_seen > 0:
                assert torch.equal(
                    seen_token_ids, prompt_token_ids_tensor[:n_seen]
                ), f"""Non-contiguous messages found! This may be a tokenization issue where certain tokens are combined when messages are concatenated, or it may be due to part of the chat history being truncated (like if super long history is truncated or if reasoning is stripped out).
Seen token IDs: {seen_token_ids.tolist()}
Output prompt token IDs: {output_item_dict["prompt_token_ids"]}
"""

            n_seen = len(seen_token_ids)

            # Create tensors for new tokens
            new_prompt_token_ids = torch.tensor(
                output_item_dict["prompt_token_ids"][n_seen:]
            )
            generation_token_ids = torch.tensor(
                output_item_dict["generation_token_ids"]
            )
            generation_logprobs = torch.tensor(output_item_dict["generation_log_probs"])


            nemo_rl_message_log.append(
                {
                    "role": "user",
                    "content": "",
                    "token_ids": new_prompt_token_ids,
                }
            )
            # Valid tool calls go through the structured API (tool_calls field) and get
            # executed by NeMo-Gym. If tool call patterns appear in the text content instead,
            # the call was invalid and never executed — flag it so training can penalize it.
            invalid_tool_call_patterns = self.cfg.get("invalid_tool_call_patterns") or ["<tool_call>", "</tool_call>", "<function_call>", "</function_call>"]
            thinking_tags = self.cfg.get("thinking_tags") or ["<think>", "</think>"]
            is_invalid_tool_call = False

            # NeMo-Gym only attaches generation_token_ids to the last output item of a
            # model call (see vllm_model/app.py postprocess_chat_response). So this item
            # is guaranteed to be the final thing the model produced for this turn.
            # If it's a reasoning item, the model output only reasoning (no content/tool calls).
            is_output_message = "content" in output_item_dict and len(output_item_dict["content"]) > 0 and "text" in output_item_dict["content"][0]
            is_reasoning_message = output_item_dict.get("type") == "reasoning" and len(output_item_dict["summary"]) > 0 and "text" in output_item_dict["summary"][0]

            # Penalize malformed thinking tags: more than one of any thinking tag in
            # reasoning, or any thinking tag leaking into the final answer content.
            has_malformed_thinking = False

            if is_output_message:
                assistant_message_content = output_item_dict["content"][0]["text"]
                if any(pattern in assistant_message_content for pattern in invalid_tool_call_patterns):
                    is_invalid_tool_call = True
                if any(tag in assistant_message_content for tag in thinking_tags):
                    has_malformed_thinking = True
            elif is_reasoning_message:
                assistant_message_content = output_item_dict["summary"][0]["text"]
                if any(pattern in assistant_message_content for pattern in invalid_tool_call_patterns):
                    is_invalid_tool_call = True
                if any(assistant_message_content.count(tag) > 1 for tag in thinking_tags):
                    has_malformed_thinking = True

            nemo_rl_message_log.append(
                {
                    "role": "assistant",
                    "content": "",
                    "token_ids": generation_token_ids,
                    "generation_logprobs": generation_logprobs,
                    "is_invalid_tool_call": is_invalid_tool_call,
                    "has_malformed_thinking": has_malformed_thinking,
                }
            )

            seen_token_ids = torch.cat(
                [seen_token_ids, new_prompt_token_ids, generation_token_ids]
            )

            # We pop to remove larger tensors from logging.
            prompt_token_ids_for_decode = output_item_dict.pop("prompt_token_ids")
            generation_token_ids_for_decode = output_item_dict.pop(
                "generation_token_ids"
            )

            output_item_dict.pop("generation_log_probs")

            batch_decode_items.append(
                (
                    output_item_dict,
                    prompt_token_ids_for_decode,
                    generation_token_ids_for_decode,
                )
            )

        if batch_decode_items:
            prompt_token_ids_batch = [item[1] for item in batch_decode_items]
            generation_token_ids_batch = [item[2] for item in batch_decode_items]

            prompt_strs = tokenizer.batch_decode(prompt_token_ids_batch)
            generation_strs = tokenizer.batch_decode(generation_token_ids_batch)

            for (output_item_dict, _, _), prompt_str, generation_str in zip(
                batch_decode_items, prompt_strs, generation_strs
            ):
                output_item_dict["prompt_str"] = prompt_str
                output_item_dict["generation_str"] = generation_str

        if not nemo_rl_message_log:
            input_messages = nemo_gym_result["responses_create_params"]["input"]
            prompt_token_ids = tokenizer.apply_chat_template(
                input_messages, tokenize=True
            )
            raise ValueError(
                f"NeMo Gym returned a result with no generation data. "
                f"This typically means the prompt for the first turn already exceeds the vLLM max_model_len, "
                f"so vLLM rejected the request before any tokens could be generated.\n"
                f"  Prompt length: {len(prompt_token_ids)} tokens.\n"
                f"  → Fix: increase `policy.max_total_sequence_length` and `policy.generation.vllm_cfg.max_model_len` "
                f"to a value larger than {len(prompt_token_ids)}."
            )

        return {
            "message_log": nemo_rl_message_log,
            "input_message_log": nemo_rl_message_log[:1],
            "full_result": nemo_gym_result,
        }

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
