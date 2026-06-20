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
import os
import sys
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import ray
import torch
from transformers import PreTrainedTokenizerBase

try:
    from nemo_rl.distributed.virtual_cluster import (
        DEFAULT_PORT_RANGE_HIGH,
        DEFAULT_PORT_RANGE_LOW,
        _get_free_port_local,
        _get_node_ip_local,
    )
except ImportError:
    from nemo_rl.distributed.virtual_cluster import (
        _get_free_port_local as _get_free_port_local_unranged,
        _get_node_ip_local,
    )

    DEFAULT_PORT_RANGE_LOW = 11001
    DEFAULT_PORT_RANGE_HIGH = 15000

    def _get_free_port_local(port_range_low=None, port_range_high=None):
        return _get_free_port_local_unranged()
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.utils.timer import Timer


def get_nemo_gym_uv_cache_dir() -> Optional[str]:
    """Return the uv cache directory inside a container, or None outside one.

    Inside a container (NRL_CONTAINER=1), returns the uv cache location so Gym
    stores its caches in the expected shared path. Returns None outside a
    container, meaning the caller should omit this arg and let Gym create the
    cache locally (the default when you may not be able to write to /opt).
    """
    if not os.environ.get("NRL_CONTAINER"):
        return None
    return subprocess.check_output(["uv", "cache", "dir"]).decode().strip()


def get_nemo_gym_venv_dir() -> Optional[str]:
    """Return the NeMo Gym venv directory from NEMO_GYM_VENV_DIR, or None.

    Returns the value of NEMO_GYM_VENV_DIR if set, otherwise None. When None
    the caller should omit this arg and let Gym create venvs locally (the
    default when a container is not used since you may not be able to write
    to /opt).
    """
    return os.environ.get("NEMO_GYM_VENV_DIR")


class NemoGymConfig(TypedDict):
    model_name: str
    base_urls: List[str]
    ray_num_gpus_per_node: Optional[int]
    ray_namespace: Optional[str]
    initial_global_config_dict: Dict[str, Any]
    invalid_tool_call_patterns: Optional[List[str]]  # Substrings in assistant text content that indicate an invalid tool call (default: ["<tool_call>", "</tool_call>", "<function_call>", "</function_call>"])
    thinking_tags: Optional[List[str]]  # Thinking tags to check for malformed usage (default: ["<think>", "</think>"])
    allow_noncontiguous_message_tokens: Optional[bool]  # Qwen reasoning/tool parsing can make template-retokenized history differ from returned prefix IDs.
    diagnose_noncontiguous_message_tokens: Optional[bool]  # Emit decoded token windows around the first mismatch for temporary debugging.
    noncontiguous_message_diagnostic_window: Optional[int]  # Number of tokens to show on each side of the mismatch.


def _summarize_token_ids(token_ids: Any, limit: int = 32) -> str:
    """Return a bounded token-id summary for assertion diagnostics."""
    if isinstance(token_ids, torch.Tensor):
        values = token_ids.tolist()
    else:
        values = list(token_ids)

    length = len(values)
    if length <= 2 * limit:
        return f"len={length}, ids={values}"
    return f"len={length}, head={values[:limit]}, tail={values[-limit:]}"


def _first_token_mismatch(left: torch.Tensor, right: torch.Tensor) -> Optional[int]:
    compare_len = min(len(left), len(right))
    for idx in range(compare_len):
        if left[idx].item() != right[idx].item():
            return idx
    if len(left) != len(right):
        return compare_len
    return None


def _decode_token_ids(tokenizer: PreTrainedTokenizerBase, token_ids: Any) -> str:
    if isinstance(token_ids, torch.Tensor):
        values = token_ids.tolist()
    else:
        values = list(token_ids)
    try:
        return tokenizer.decode(
            values,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(values, skip_special_tokens=False)
    except Exception as exc:
        return f"<decode failed: {exc!r}>"


def _noncontiguous_token_diagnostic(
    tokenizer: PreTrainedTokenizerBase,
    seen_token_ids: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    mismatch_idx: Optional[int],
    window: int,
) -> str:
    if mismatch_idx is None:
        mismatch_idx = min(len(seen_token_ids), len(prompt_token_ids))

    start = max(0, mismatch_idx - window)
    end = min(max(len(seen_token_ids), len(prompt_token_ids)), mismatch_idx + window + 1)
    seen_window = seen_token_ids[start : min(end, len(seen_token_ids))]
    prompt_window = prompt_token_ids[start : min(end, len(prompt_token_ids))]

    return "\n".join(
        [
            "Non-contiguous message token diagnostic:",
            f"  mismatch_idx={mismatch_idx}, window=[{start}:{end}), seen_len={len(seen_token_ids)}, prompt_len={len(prompt_token_ids)}",
            f"  seen_ids={seen_window.tolist()}",
            f"  prompt_ids={prompt_window.tolist()}",
            f"  seen_text={_decode_token_ids(tokenizer, seen_window)!r}",
            f"  prompt_text={_decode_token_ids(tokenizer, prompt_window)!r}",
        ]
    )


def _bool_flag(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def _timer_with_optional_context(context: dict[str, Any]) -> Timer:
    try:
        return Timer(context=context)
    except TypeError as exc:
        if "context" not in str(exc):
            raise
        return Timer()


def _timer_time(timer: Timer, label: str, should_log: bool = False):
    try:
        return timer.time(label=label, should_log=should_log)
    except TypeError as exc:
        if "should_log" not in str(exc):
            raise
        return timer.time(label=label)


@ray.remote(max_restarts=-1, max_task_retries=-1)  # pragma: no cover
class NemoGym(EnvironmentInterface):
    """This environment class isn't really used for training. It's really meant as an integration wrapper around NeMo-Gym that hooks into the existing NeMo RL resource management via ray. So there is still one source of truth for resource management in NeMo RL."""

    def __init__(self, cfg: NemoGymConfig):
        self.cfg = cfg

    def _spinup(self) -> None:
        self.node_ip = _get_node_ip_local()
        port_range_low = self.cfg.get("port_range_low", DEFAULT_PORT_RANGE_LOW)
        port_range_high = self.cfg.get("port_range_high", DEFAULT_PORT_RANGE_HIGH)
        self.head_server_port = _get_free_port_local(port_range_low, port_range_high)

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

        # Gym servers default to 5000-5999, below the OS ephemeral floor (9000
        # on OCI-HSG).  See ray.sub port layout comment for the full map.
        _gym_port_low = self.cfg.get("port_range_low", 5000)
        _gym_port_high = self.cfg.get("port_range_high", 5999)
        if _gym_port_low < 5000 or _gym_port_high > 5999:
            print(
                f"WARNING: Gym port range [{_gym_port_low}, {_gym_port_high}) is outside "
                f"the expected 5000-5999 band. Check ray.sub port layout for conflicts."
            )
        initial_global_config_dict["port_range_low"] = _gym_port_low
        initial_global_config_dict["port_range_high"] = _gym_port_high

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

        ray_num_gpus_per_node = self.cfg.get("ray_num_gpus_per_node")
        if ray_num_gpus_per_node is not None:
            initial_global_config_dict["ray_num_gpus_per_node"] = ray_num_gpus_per_node
            print(f"Ray num GPUs per node: {ray_num_gpus_per_node}")

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
        )

        # Setup for rollout collection
        self.head_server_config = BaseServerConfig(
            host=self.node_ip,
            port=self.head_server_port,
        )
        self.rch = RolloutCollectionHelper()

    def health_check(self) -> bool:
        if not hasattr(self, "rh"):
            self._spinup()
        return True

    async def run_rollouts(
        self,
        nemo_gym_examples: list[dict],
        tokenizer: PreTrainedTokenizerBase,
        timer_prefix: str,
    ) -> tuple[list[dict], dict]:
        import sys
        from collections import Counter

        try:
            from nemo_rl.utils.fastokens import maybe_patch_fastokens
        except ImportError:
            def maybe_patch_fastokens():
                return None

        maybe_patch_fastokens()

        if not hasattr(self, "rch"):
            self._spinup()

        timer = _timer_with_optional_context({"worker": "nemo_gym"})

        counts_left = Counter(r["agent_ref"]["name"] for r in nemo_gym_examples)

        timer.start("_run_rollouts_total")
        nemo_gym_result_iterator = self.rch.run_examples(
            examples=nemo_gym_examples,
            head_server_config=self.head_server_config,
        )

        num_results = 0
        nemo_rl_rowidxs = []
        nemo_rl_results = []
        timing_metrics = {}
        for task in nemo_gym_result_iterator:
            with _timer_time(timer, label=f"{timer_prefix}/await_results", should_log=False):
                try:
                    nemo_gym_row, nemo_gym_result = await task
                except Exception as e:
                    # This response content comes from https://github.com/NVIDIA-NeMo/Gym/blob/30f498b1e994679cebcfacfa4ac630190d0e171f/nemo_gym/server_utils.py#L233
                    if hasattr(e, "response_content"):
                        print("EXCEPTION RESULT", e.response_content, file=sys.stderr)
                    raise e

            with _timer_time(timer, label=f"{timer_prefix}/postprocess_results", should_log=False):
                nemo_rl_result = self._postprocess_nemo_gym_to_nemo_rl_result(
                    nemo_gym_result, tokenizer
                )
                for message in nemo_rl_result["message_log"]:
                    if (
                        "generation_logprobs" in message
                        and message["generation_logprobs"] is not None
                    ):
                        if torch.isnan(message["generation_logprobs"]).any():
                            raise RuntimeError(
                                f"Generation logprobs contain NaN! Failing loudly"
                            )

            num_results += 1
            if num_results == len(nemo_gym_examples):
                timer.stop("_run_rollouts_total")
                timing_metrics = timer.get_timing_metrics("sum")
                total_time = timing_metrics.pop("_run_rollouts_total")
                timing_metrics[f"{timer_prefix}/postprocess_results_pct"] = (
                    100 * timing_metrics[f"{timer_prefix}/postprocess_results"] / total_time
                )

            nemo_rl_rowidxs.append(nemo_gym_row["_rowidx"])
            nemo_rl_results.append(nemo_rl_result)

            counts_left[nemo_gym_row["agent_ref"]["name"]] -= 1
            if counts_left[nemo_gym_row["agent_ref"]["name"]] <= 0:
                counts_left.pop(nemo_gym_row["agent_ref"]["name"])
            # Print every 10 rollouts
            if num_results % 10 == 0:
                # Only print top 5
                top_left = counts_left.most_common(5)
                top_left_str = "\n".join(f"{i + 1}. {k}: {v}" for i, (k, v) in enumerate(top_left))
                print(f"Top 5 NeMo Gym agent refs left in this rollout batch: {top_left_str}", file=sys.stderr)

        nemo_rl_sort_results = [None] * len(nemo_gym_examples)
        for rowidx, result in zip(nemo_rl_rowidxs, nemo_rl_results):
            nemo_rl_sort_results[rowidx] = result

        return nemo_rl_sort_results, timing_metrics

    def _postprocess_nemo_gym_to_nemo_rl_result(
        self, nemo_gym_result: dict, tokenizer: PreTrainedTokenizerBase
    ) -> dict:
        assert isinstance(nemo_gym_result, dict), (
            f"Hit a non-successful response when querying NeMo Gym for rollouts: {nemo_gym_result}"
        )

        nemo_rl_message_log = []
        seen_token_ids = torch.tensor([], dtype=torch.int64)

        batch_decode_items = []  # Collect (output_item_dict, prompt_token_ids, generation_token_ids) for batch decode
        for output_item_dict in nemo_gym_result["response"]["output"]:
            # Nemo RL really only has two types of messages: assistant and not assistant since that is all that it is concerned with (i.e. to train or not to train)
            # Here we map all the trainable messages to assistant and all the non-trainable messages to user.
            # Eventually we can maybe be smarter about this, but this is functional for now.

            # Note that NeMo-Gym will only return token ids on "assistant" messages and not other message types.
            if "generation_token_ids" not in output_item_dict:
                continue

            prompt_token_ids_tensor = torch.tensor(
                output_item_dict["prompt_token_ids"], dtype=torch.int64
            )
            n_seen = len(seen_token_ids)
            if n_seen > 0:
                is_contiguous = torch.equal(
                    seen_token_ids, prompt_token_ids_tensor[:n_seen]
                )
                if not is_contiguous:
                    mismatch_idx = _first_token_mismatch(
                        seen_token_ids, prompt_token_ids_tensor[:n_seen]
                    )
                    message = f"""Non-contiguous messages found! This may be a tokenization issue where certain tokens are combined when messages are concatenated, or it may be due to part of the chat history being truncated (like if super long history is truncated or if reasoning is stripped out).
First mismatch index: {mismatch_idx}
Seen token IDs summary: {_summarize_token_ids(seen_token_ids)}
Output prompt token IDs summary: {_summarize_token_ids(output_item_dict["prompt_token_ids"])}
"""
                    allow_noncontiguous = self.cfg.get(
                        "allow_noncontiguous_message_tokens"
                    )
                    if allow_noncontiguous is None:
                        allow_noncontiguous = os.environ.get(
                            "NEMO_RL_ALLOW_NONCONTIGUOUS_MESSAGE_TOKENS"
                        )
                    if not _bool_flag(allow_noncontiguous, default=True):
                        raise AssertionError(message)

                    print(
                        "WARNING: " + message.strip().replace("\n", " ")
                        + " Collapsing this trajectory to the current turn.",
                        file=sys.stderr,
                    )

                    diagnose_noncontiguous = self.cfg.get(
                        "diagnose_noncontiguous_message_tokens"
                    )
                    if diagnose_noncontiguous is None:
                        diagnose_noncontiguous = os.environ.get(
                            "NEMO_RL_DIAGNOSE_NONCONTIGUOUS_MESSAGE_TOKENS"
                        )
                    if _bool_flag(diagnose_noncontiguous, default=False):
                        diagnostic_window = self.cfg.get(
                            "noncontiguous_message_diagnostic_window"
                        )
                        if diagnostic_window is None:
                            diagnostic_window = os.environ.get(
                                "NEMO_RL_NONCONTIGUOUS_MESSAGE_DIAGNOSTIC_WINDOW",
                                48,
                            )
                        try:
                            diagnostic_window = int(diagnostic_window)
                        except (TypeError, ValueError):
                            diagnostic_window = 48
                        print(
                            _noncontiguous_token_diagnostic(
                                tokenizer=tokenizer,
                                seen_token_ids=seen_token_ids,
                                prompt_token_ids=prompt_token_ids_tensor,
                                mismatch_idx=mismatch_idx,
                                window=max(0, diagnostic_window),
                            ),
                            file=sys.stderr,
                        )

                    nemo_rl_message_log = []
                    seen_token_ids = torch.tensor([], dtype=torch.int64)
                    n_seen = 0

            n_seen = len(seen_token_ids)

            # Create tensors for new tokens
            new_prompt_token_ids = torch.tensor(
                output_item_dict["prompt_token_ids"][n_seen:], dtype=torch.int64
            )
            generation_token_ids = torch.tensor(
                output_item_dict["generation_token_ids"], dtype=torch.int64
            )
            generation_logprobs = torch.tensor(
                output_item_dict["generation_log_probs"], dtype=torch.float32
            )


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
            is_invalid_tool_call = False

            # NeMo-Gym only attaches generation_token_ids to the last output item of a
            # model call (see vllm_model/app.py postprocess_chat_response). So this item
            # is guaranteed to be the final thing the model produced for this turn.
            # If it's a reasoning item, the model output only reasoning (no content/tool calls).
            is_output_message = "content" in output_item_dict and len(output_item_dict["content"]) > 0 and "text" in output_item_dict["content"][0]
            is_reasoning_message = output_item_dict.get("type") == "reasoning" and len(output_item_dict["summary"]) > 0 and "text" in output_item_dict["summary"][0]

            if is_output_message:
                assistant_message_content = output_item_dict["content"][0]["text"]
                if any(pattern in assistant_message_content for pattern in invalid_tool_call_patterns):
                    is_invalid_tool_call = True
            elif is_reasoning_message:
                assistant_message_content = output_item_dict["summary"][0]["text"]
                if any(pattern in assistant_message_content for pattern in invalid_tool_call_patterns):
                    is_invalid_tool_call = True

            nemo_rl_message_log.append(
                {
                    "role": "assistant",
                    "content": "",
                    "token_ids": generation_token_ids,
                    "generation_logprobs": generation_logprobs,
                    "is_invalid_tool_call": is_invalid_tool_call,
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
            try:
                prompt_token_ids = tokenizer.apply_chat_template(
                    input_messages, tokenize=True
                )
                prompt_len_str = f"{len(prompt_token_ids)} tokens"
            except Exception as e:
                prompt_len_str = (
                    f"<unknown — apply_chat_template failed: {type(e).__name__}: {e}>"
                )
            output_item_types = [
                o.get("type") for o in nemo_gym_result["response"]["output"]
            ]
            raise ValueError(
                f"NeMo Gym returned a result with no generation data. "
                f"Possible causes: (1) the prompt for the first turn already exceeds the vLLM max_model_len, "
                f"so vLLM rejected the request before any tokens could be generated; "
                f"(2) all response output items were reasoning/tool-call items with no assistant generation.\n"
                f"  Prompt length: {prompt_len_str}.\n"
                f"  response.output item types ({len(output_item_types)} items): {output_item_types}.\n"
                f"  → If (1): increase `policy.max_total_sequence_length` and `policy.generation.vllm_cfg.max_model_len` "
                f"above the prompt length above.\n"
                f"  → If (2): inspect why no assistant content was produced for this rollout."
            )

        return {
            "message_log": nemo_rl_message_log,
            "input_message_log": nemo_rl_message_log[:1],
            "full_result": nemo_gym_result,
        }

    def shutdown(self) -> None:
        if hasattr(self, "rh"):
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
