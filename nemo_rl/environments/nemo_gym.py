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
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, TypedDict

import ray
import torch
from transformers import PreTrainedTokenizerBase

from nemo_rl.distributed.virtual_cluster import _get_free_port_local, _get_node_ip_local
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.utils.timer import Timer


class NemoGymConfig(TypedDict):
    model_name: str
    base_urls: List[str]
    initial_global_config_dict: Dict[str, Any]


def _truncate_error_value(value: Any, max_len: int = 256) -> str:
    text = str(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _summarize_nemo_gym_output_item(output_item: Any) -> dict[str, Any]:
    if not isinstance(output_item, dict):
        return {
            "python_type": type(output_item).__name__,
            "repr": _truncate_error_value(output_item),
        }

    summary: dict[str, Any] = {
        "keys": sorted(output_item.keys()),
    }

    for key in (
        "type",
        "role",
        "status",
        "finish_reason",
        "stop_reason",
        "id",
    ):
        if key in output_item:
            summary[key] = output_item[key]

    for key, summary_key in (
        ("prompt_token_ids", "prompt_token_count"),
        ("generation_token_ids", "generation_token_count"),
        ("generation_log_probs", "generation_logprob_count"),
    ):
        value = output_item.get(key)
        if isinstance(value, list):
            summary[summary_key] = len(value)

    content = output_item.get("content")
    if isinstance(content, list):
        summary["content_len"] = len(content)
        summary["content_types"] = [
            item.get("type", type(item).__name__) if isinstance(item, dict) else type(item).__name__
            for item in content[:8]
        ]
    elif content is not None:
        summary["content_type"] = type(content).__name__
        summary["content_preview"] = _truncate_error_value(content)

    if "refusal" in output_item and output_item["refusal"] is not None:
        summary["refusal"] = _truncate_error_value(output_item["refusal"])

    return summary


def _summarize_nemo_gym_empty_generation_result(nemo_gym_result: dict[str, Any]) -> dict[str, Any]:
    response = nemo_gym_result.get("response")
    if not isinstance(response, dict):
        return {
            "response_python_type": type(response).__name__,
            "response_repr": _truncate_error_value(response),
        }

    output_items = response.get("output")
    if isinstance(output_items, list):
        output_summary = [
            _summarize_nemo_gym_output_item(item) for item in output_items[:8]
        ]
        output_count = len(output_items)
    else:
        output_summary = [
            {
                "python_type": type(output_items).__name__,
                "repr": _truncate_error_value(output_items),
            }
        ]
        output_count = None

    summary = {
        "response_keys": sorted(response.keys()),
        "response_status": response.get("status"),
        "response_finish_reason": response.get("finish_reason"),
        "response_incomplete_details": response.get("incomplete_details"),
        "response_error": response.get("error"),
        "usage": response.get("usage"),
        "output_count": output_count,
        "output_summary": output_summary,
    }

    if "id" in response:
        summary["response_id"] = response["id"]
    if "model" in response:
        summary["response_model"] = response["model"]

    return summary


def _summarize_nemo_gym_row(row: Any) -> dict[str, Any]:
    if not isinstance(row, dict):
        return {"python_type": type(row).__name__, "repr": _truncate_error_value(row)}

    summary: dict[str, Any] = {
        "rowidx": row.get("_rowidx"),
        "keys": sorted(row.keys()),
    }
    for key in ("instance_id", "task_id", "repo", "problem_id"):
        if key in row:
            summary[key] = _truncate_error_value(row[key])

    agent_ref = row.get("agent_ref")
    if isinstance(agent_ref, dict):
        summary["agent"] = agent_ref.get("name")

    responses_create_params = row.get("responses_create_params")
    if isinstance(responses_create_params, dict):
        summary["model"] = responses_create_params.get("model")
        input_messages = responses_create_params.get("input")
        if isinstance(input_messages, list):
            summary["input_messages"] = len(input_messages)
        if "max_output_tokens" in responses_create_params:
            summary["max_output_tokens"] = responses_create_params["max_output_tokens"]

    return summary


async def _await_nemo_gym_task_with_debug(
    task: Any,
    *,
    trial: int,
    max_attempts: int,
    attempt_start: float,
    completed_count: int,
    total_count: int,
) -> Any:
    """Await the next completed Gym rollout while emitting periodic stall context."""
    pending_task = asyncio.ensure_future(task)
    while True:
        try:
            return await asyncio.wait_for(asyncio.shield(pending_task), timeout=30)
        except asyncio.TimeoutError:
            print(
                "[NEMO_GYM_DEBUG] run_rollouts_waiting_for_result "
                f"trial={trial}/{max_attempts} "
                f"elapsed_s={time.perf_counter() - attempt_start:.1f} "
                f"completed={completed_count}/{total_count}"
            )


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
        # In multinode runs, Gym-managed service configs must advertise a real node IP
        # rather than falling back to localhost, or remote workers will connect to
        # their own loopback interface instead of the actor-hosted service.
        initial_global_config_dict.setdefault("default_host", self.node_ip)

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

        timer.start("_run_rollouts_total")
        max_attempts, trial = self.rollout_max_attempts_to_avoid_lp_nan, 0
        while trial < max_attempts:
            nemo_gym_num_rows = len(nemo_gym_examples)
            attempt_start = time.perf_counter()
            print(
                "[NEMO_GYM_DEBUG] run_rollouts_start "
                f"trial={trial + 1}/{max_attempts} rows={nemo_gym_num_rows} "
                "row_summaries="
                f"{json.dumps([_summarize_nemo_gym_row(row) for row in nemo_gym_examples[:16]], default=str)}"
            )
            nemo_gym_result_iterator = self.rch.run_examples(
                examples=nemo_gym_examples, head_server_config=self.head_server_config
            )

            nemo_rl_rowidxs = []
            nemo_rl_results = []
            for task in nemo_gym_result_iterator:
                with timer.time(label=f"{timer_prefix}/await_results"):
                    nemo_gym_row, nemo_gym_result = await _await_nemo_gym_task_with_debug(
                        task,
                        trial=trial + 1,
                        max_attempts=max_attempts,
                        attempt_start=attempt_start,
                        completed_count=len(nemo_rl_results),
                        total_count=nemo_gym_num_rows,
                    )

                with timer.time(label=f"{timer_prefix}/postprocess_results"):
                    row_summary = _summarize_nemo_gym_row(nemo_gym_row)
                    response = (
                        nemo_gym_result.get("response")
                        if isinstance(nemo_gym_result, dict)
                        else None
                    )
                    output = response.get("output") if isinstance(response, dict) else None
                    output_count = len(output) if isinstance(output, list) else None
                    print(
                        "[NEMO_GYM_DEBUG] run_rollouts_result_received "
                        f"trial={trial + 1}/{max_attempts} "
                        f"elapsed_s={time.perf_counter() - attempt_start:.1f} "
                        f"row={json.dumps(row_summary, default=str)} "
                        f"response_status={response.get('status') if isinstance(response, dict) else None} "
                        f"output_count={output_count}"
                    )
                    try:
                        nemo_rl_result = self._postprocess_nemo_gym_to_nemo_rl_result(
                            nemo_gym_result, tokenizer
                        )
                    except Exception:
                        result_summary = (
                            _summarize_nemo_gym_empty_generation_result(nemo_gym_result)
                            if isinstance(nemo_gym_result, dict)
                            else {
                                "python_type": type(nemo_gym_result).__name__,
                                "repr": _truncate_error_value(nemo_gym_result),
                            }
                        )
                        print(
                            "[NEMO_GYM_DEBUG] run_rollouts_postprocess_failed "
                            f"trial={trial + 1}/{max_attempts} "
                            f"row={json.dumps(row_summary, default=str)} "
                            f"result_summary={json.dumps(result_summary, default=str)}"
                        )
                        raise

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
                print(
                    "[NEMO_GYM_DEBUG] run_rollouts_attempt_complete "
                    f"trial={trial + 1}/{max_attempts} rows={nemo_gym_num_rows} "
                    f"elapsed_s={time.perf_counter() - attempt_start:.1f}"
                )
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
        seen_token_ids: List[int] = []
        for output_item_dict in nemo_gym_result["response"]["output"]:
            # Nemo RL really only has two types of messages: assistant and not assistant since that is all that it is concerned with (i.e. to train or not to train)
            # Here we map all the trainable messages to assistant and all the non-trainable messages to user.
            # Eventually we can maybe be smarter about this, but this is functional for now.

            # Note that NeMo-Gym will only return token ids on "assistant" messages and not other message types.
            if "generation_token_ids" not in output_item_dict:
                continue

            nemo_rl_message_log.append(
                {
                    "role": "user",
                    "content": "",
                    "token_ids": torch.tensor(
                        output_item_dict["prompt_token_ids"][len(seen_token_ids) :],
                        dtype=torch.long,
                    ),
                }
            )
            nemo_rl_message_log.append(
                {
                    "role": "assistant",
                    "content": "",
                    "token_ids": torch.tensor(
                        output_item_dict["generation_token_ids"], dtype=torch.long
                    ),
                    "generation_logprobs": torch.tensor(
                        output_item_dict["generation_log_probs"], dtype=torch.float32
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

        if not nemo_rl_message_log:
            input_messages = nemo_gym_result["responses_create_params"]["input"]
            prompt_token_ids = tokenizer.apply_chat_template(
                input_messages, tokenize=True
            )
            response_summary = _summarize_nemo_gym_empty_generation_result(
                nemo_gym_result
            )
            raise ValueError(
                "NeMo Gym returned a result with no generation data.\n"
                f"  Prompt length: {len(prompt_token_ids)} tokens.\n"
                "  Response summary:\n"
                f"  {json.dumps(response_summary, indent=2, sort_keys=True, default=str)}"
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
