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
from pathlib import Path
from typing import Any, Dict, List, TypedDict

import aiohttp
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
            nemo_gym_result_iterator = self.rch.run_examples(
                examples=nemo_gym_examples, head_server_config=self.head_server_config
            )

            nemo_rl_rowidxs = []
            nemo_rl_results = []
            for task in nemo_gym_result_iterator:
                try:
                    with timer.time(label=f"{timer_prefix}/await_results"):
                        nemo_gym_row, nemo_gym_result = await task
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    # A single rollout's gym /run failed at the transport/HTTP layer (a flaky instance
                    # returned 500, a connection dropped, or it timed out). Don't abort the whole batch
                    # over one instance -- the failed row is left as an empty slot in the rowidx-sorted
                    # array below and back-filled there with a zero-reward degenerate trajectory (same
                    # path as the no-generation-data case). We can't recover the row's _rowidx from the
                    # failed task, so the back-fill works off the empty slots instead. Logged loudly so
                    # a SYSTEMIC failure (many rows failing -> ~0 mean reward) stays visible.
                    print(
                        f"  [nemo_gym] WARNING: rollout failed ({type(e).__name__}: {e}); "
                        f"will back-fill as a zero-reward trajectory instead of aborting the batch.",
                        flush=True,
                    )
                    continue

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
        # Back-fill rows whose rollout failed above (left as None) with a zero-reward degenerate
        # trajectory, so batch/group counts stay intact (force_on_policy_ratio needs the full
        # num_prompts*num_generations batch). We synthesize an empty gym result and run it through
        # the same postprocess path the no-generation-data case uses (builds a fresh, masked,
        # reward-0 trajectory per slot -- no aliasing, since downstream mutates message_log in place).
        num_failed = sum(1 for r in nemo_rl_sort_results if r is None)
        if num_failed:
            print(
                f"  [nemo_gym] WARNING: back-filling {num_failed}/{nemo_gym_num_rows} failed "
                f"rollout(s) as zero-reward trajectories (batch counts preserved).",
                flush=True,
            )
            for i in range(nemo_gym_num_rows):
                if nemo_rl_sort_results[i] is None:
                    nemo_rl_sort_results[i] = (
                        self._postprocess_nemo_gym_to_nemo_rl_result(
                            {"response": {"output": []}}, tokenizer
                        )
                    )
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
            # Flag a dropped/unparseable tool call so GRPO can apply
            # grpo.invalid_tool_call_strategy. When the tool parser (e.g. hermes) successfully
            # parses an assistant tool call, vLLM strips the <tool_call>...</tool_call> tags out
            # of the message `content` and moves the call into the structured tool_calls field. If
            # those raw tags survive in `content`, the parser failed (e.g. the model stuffed an
            # un-escaped code patch into the JSON args -- the hermes-on-Instruct failure mode) and
            # the call was dropped. The marker is hermes-specific; XML parsers (qwen3_coder) never
            # emit these tags, so it stays False there and the strategy is inert.
            is_invalid_tool_call = False
            content = output_item_dict.get("content")
            if content and isinstance(content[0], dict) and content[0].get("text"):
                assistant_text = content[0]["text"]
                if "<tool_call>" in assistant_text or "</tool_call>" in assistant_text:
                    is_invalid_tool_call = True

            nemo_rl_message_log.append(
                {
                    "role": "assistant",
                    "content": "",
                    "token_ids": torch.tensor(output_item_dict["generation_token_ids"]),
                    "generation_logprobs": torch.tensor(
                        output_item_dict["generation_log_probs"]
                    ),
                    "is_invalid_tool_call": is_invalid_tool_call,
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

        n_invalid_tool_calls = sum(
            1 for m in nemo_rl_message_log if m.get("is_invalid_tool_call")
        )
        if n_invalid_tool_calls:
            print(
                f"  [invalid-tool-call] rollout produced {n_invalid_tool_calls} "
                f"unparseable tool-call turn(s) (<tool_call> tags survived in content; "
                f"flagged for grpo.invalid_tool_call_strategy)",
                flush=True,
            )

        if not nemo_rl_message_log:
            # A rollout came back with no assistant generation (the agent/container failed to produce
            # a usable turn -- empirically ~0.3% of rollouts). This used to raise and abort the ENTIRE
            # batch, which is fatal during validation where one flaky instance among many kills the run
            # (and produces no val score). Instead, emit a degenerate ZERO-REWARD trajectory so the
            # batch/group counts stay intact (num_gen per prompt-group; full val set) and the instance
            # simply scores 0. The single placeholder assistant token carries NO generation_logprobs,
            # so downstream GRPO masking (add_grpo_token_loss_masks_and_generation_logprobs) marks it
            # non-trainable -> it cannot pollute the gradient.
            # The degenerate trajectory's content is irrelevant (it's a masked, zero-reward sample),
            # so use a minimal placeholder token rather than re-tokenizing the prompt -- apply_chat_template
            # can return a tokenizers.Encoding that torch.tensor() can't consume.
            fallback_token = (
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else (tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0)
            )
            print(
                "  [nemo_gym] WARNING: returned no generation data for a rollout; "
                "treating as a zero-reward trajectory instead of aborting the batch.",
                flush=True,
            )
            nemo_rl_message_log = [
                {
                    "role": "user",
                    "content": "",
                    "token_ids": torch.tensor([fallback_token, fallback_token]),
                },
                {
                    "role": "assistant",
                    "content": "",
                    "token_ids": torch.tensor([fallback_token]),
                },
            ]
            nemo_gym_result["reward"] = 0.0

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
