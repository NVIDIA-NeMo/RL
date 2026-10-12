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
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import ray
import torch
from transformers import PreTrainedTokenizerBase

from nemo_rl.distributed.virtual_cluster import (
    DEFAULT_PORT_RANGE_HIGH,
    DEFAULT_PORT_RANGE_LOW,
    _get_free_port_local,
    _get_node_ip_local,
)
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.utils.timer import Timer


def _sublist_index(needle: List[int], haystack: List[int]) -> Optional[int]:
    n = len(needle)
    if not n or len(haystack) < n:
        return None
    for i in range(len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return i
    return None


def _strip_injected_token_position_metadata(
    tokenizer: PreTrainedTokenizerBase,
    gen_ids: List[int],
    log_probs: List[float],
) -> tuple[List[int], List[float], Optional[List[int]]]:
    """If present, remove trailing <|injected_token_positions|> + JSON tail; return positions list or None.

    The LP (see custom_logit_processors/v1/base_logit_processor.py::_maybe_suppress_eod)
    intercepts the model's natural EOD when it becomes the argmax and replaces it with:

        <answer_text> <|injected_token_positions|>{metadata_json}<EOD>

    The single EOD token (e.g. <|im_end|>) sits at the very end of gen_ids, AFTER the
    metadata block. We strip from the marker onwards but re-attach that trailing EOD
    (with the model's natural log-prob, captured at intercept-time and embedded in
    the metadata as ``end_logp``) so the assistant turn keeps its terminator and the
    PPO/GRPO importance ratio for that token uses the right log-prob.

    Metadata format compatibility:
      - new (preferred): ``{"injected_positions": [...], "end_logp": -0.232}``
        Returns the list and uses ``end_logp`` for the re-attached EOD.
      - old (bare list): ``[1, 2, 3]``
        Returns the list and falls back to ``log_probs[-1]`` for the EOD log-prob,
        which is the LP-forced ~0 value (slightly wrong but better than dropping
        the EOD entirely; matches pre-fix behavior on legacy data).
    """
    marker = tokenizer(" <|injected_token_positions|>", add_special_tokens=False)[
        "input_ids"
    ]
    start = _sublist_index(marker, gen_ids)
    if start is None:
        return gen_ids, log_probs, None
    tail = gen_ids[start:]
    meta = (
        tokenizer.decode(tail, skip_special_tokens=True)
        .strip()
        .removeprefix("<|injected_token_positions|>")
        .removesuffix("<|im_end|>")
        .strip()
    )

    positions: Optional[List[int]] = None
    end_logp: Optional[float] = None
    try:
        parsed = json.loads(meta)
    except json.JSONDecodeError:
        print(f"WARNING: Failed to parse injected_tokens_str as JSON (truncated?): {meta!r}")
        parsed = None
    if isinstance(parsed, list):
        positions = parsed  # legacy bare-list format
    elif isinstance(parsed, dict):
        positions = parsed.get("injected_positions")
        raw_logp = parsed.get("end_logp")
        if raw_logp is not None:
            try:
                end_logp = float(raw_logp)
            except (TypeError, ValueError):
                end_logp = None

    head_ids = list(gen_ids[:start])
    head_lps = list(log_probs[:start])

    # Preserve the LP-appended trailing EOD (the model's intended turn-end token,
    # which the LP held back while it injected the metadata block).
    eod_ids = set()
    if tokenizer.eos_token_id is not None:
        eod_ids.add(tokenizer.eos_token_id)
    for special in ("<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(special)
        if isinstance(tid, int) and tid != tokenizer.unk_token_id:
            eod_ids.add(tid)
    # Only re-attach if the trailing token actually is an EOD (handles rollouts
    # that hit max_seq_len mid-metadata-injection — those won't end with EOD).
    if gen_ids and gen_ids[-1] in eod_ids:
        head_ids.append(gen_ids[-1])
        # Prefer the LP-captured natural log-prob; fall back to log_probs[-1]
        # (the LP-forced value, ~0) for legacy rollouts without end_logp.
        head_lps.append(end_logp if end_logp is not None else log_probs[-1])

    return head_ids, head_lps, positions


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

        initial_global_config_dict["ray_num_gpus_per_node"] = self.cfg[
            "ray_num_gpus_per_node"
        ]
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

        self.max_concurrent_rollout_requests = initial_global_config_dict.pop(
            "max_concurrent_rollout_requests", None
        )
        if self.max_concurrent_rollout_requests is not None:
            print(
                f"Capping concurrent NeMo-Gym rollout requests at {self.max_concurrent_rollout_requests}"
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

    async def run_rollouts(
        self,
        nemo_gym_examples: list[dict],
        tokenizer: PreTrainedTokenizerBase,
        timer_prefix: str,
    ) -> list[dict]:
        from nemo_rl.utils.fastokens import maybe_patch_fastokens

        maybe_patch_fastokens()

        timer = Timer(context={"worker": "nemo_gym"})


        timer.start("_run_rollouts_total")
        max_attempts, trial = self.rollout_max_attempts_to_avoid_lp_nan, 0
        rollout_semaphore = (
            asyncio.Semaphore(self.max_concurrent_rollout_requests)
            if self.max_concurrent_rollout_requests
            else None
        )
        while trial < max_attempts:
            nemo_gym_num_rows = len(nemo_gym_examples)
            nemo_gym_result_iterator = self.rch.run_examples(
                examples=nemo_gym_examples,
                head_server_config=self.head_server_config,
                semaphore=rollout_semaphore,
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
                assert torch.equal(
                    seen_token_ids, prompt_token_ids_tensor[:n_seen]
                ), f"""Non-contiguous messages found! This may be a tokenization issue where certain tokens are combined when messages are concatenated, or it may be due to part of the chat history being truncated (like if super long history is truncated or if reasoning is stripped out).
Seen token IDs: {seen_token_ids.tolist()}
Output prompt token IDs: {output_item_dict["prompt_token_ids"]}
"""

            n_seen = len(seen_token_ids)

            gen_list = list(output_item_dict["generation_token_ids"])
            logp_list = list(output_item_dict["generation_log_probs"])
            if gen_list and isinstance(gen_list[0], str):
                gen_list = [int(x) for x in gen_list]
            gen_list, logp_list, inj_positions = _strip_injected_token_position_metadata(
                tokenizer, gen_list, logp_list
            )
            output_item_dict["generation_token_ids"] = gen_list
            output_item_dict["generation_log_probs"] = logp_list

            new_prompt_token_ids = torch.tensor(
                output_item_dict["prompt_token_ids"][n_seen:], dtype=torch.int64
            )
            generation_token_ids = torch.tensor(gen_list, dtype=torch.int64)
            generation_logprobs = torch.tensor(logp_list, dtype=torch.float32)


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

            # Optional: act on the first injected position so the post-inject
            # tail of the assistant turn doesn't contribute to the loss.
            #   action=mask                  → expand inj_positions to [first..end) so
            #                                  the existing token_loss_mask logic in
            #                                  grpo.py zeros that whole range (tensor
            #                                  unchanged).
            #   action=truncate              → slice token_ids/logprobs to length=first;
            #                                  drops inj_positions since nothing to mark.
            #   action=mask_if_malformed     → like mask, but only when the assistant turn
            #                                  has >1 </think> token id (= the LP-injected
            #                                  one plus a model-emitted "evasion" one).
            #                                  Requires env.nemo_gym.think_close_id.
            #   action=truncate_if_malformed → like truncate, conditional on the same
            #                                  >1-</think> check.
            # Configured via env.nemo_gym.injection_first_pos_action; default "none".
            if inj_positions:
                env_cfg = self.cfg.get("initial_global_config_dict") or {}
                action = env_cfg.get("injection_first_pos_action", "none")

                # For *_if_malformed actions, only fire when the assistant turn has
                # more than one </think> token (counted across the whole gen, not
                # just outside inj_positions — simpler rule, matches the malformed-
                # think reward penalty's check).
                conditional = action in ("mask_if_malformed", "truncate_if_malformed")
                trigger = True
                if conditional:
                    think_close_id = env_cfg.get("think_close_id")
                    if think_close_id is None:
                        raise ValueError(
                            f"injection_first_pos_action={action} requires "
                            "env.nemo_gym.think_close_id to be set in the YAML"
                        )
                    n_close = sum(
                        1 for t in generation_token_ids.tolist() if t == think_close_id
                    )
                    trigger = n_close > 1

                if trigger and action in ("truncate", "truncate_if_malformed"):
                    first = min(inj_positions)
                    generation_token_ids = generation_token_ids[:first]
                    generation_logprobs = generation_logprobs[:first]
                    inj_positions = None
                elif trigger and action in ("mask", "mask_if_malformed"):
                    first = min(inj_positions)
                    inj_positions = list(range(first, len(generation_token_ids)))

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": "",
                "token_ids": generation_token_ids,
                "generation_logprobs": generation_logprobs,
                "is_invalid_tool_call": is_invalid_tool_call,
            }
            if inj_positions is not None:
                assistant_msg["injected_token_positions"] = inj_positions
            nemo_rl_message_log.append(assistant_msg)

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
