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
import subprocess
from pathlib import Path
from typing import Any, Dict, List, NotRequired, Optional, TypedDict

import ray
import torch
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from transformers import PreTrainedTokenizerBase

from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
from nemo_rl.distributed.virtual_cluster import (
    DEFAULT_GYM_PORT_RANGE_HIGH,
    DEFAULT_GYM_PORT_RANGE_LOW,
    _get_free_port_local,
    _get_node_ip_local,
)
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.utils.timer import Timer
from nemo_rl.utils.venvs import create_local_venv_on_each_node

DEFAULT_INVALID_TOOL_CALL_PATTERNS = [
    "<tool_call>",
    "</tool_call>",
    "<function_call>",
    "</function_call>",
]
DEFAULT_THINKING_TAGS = ["<think>", "</think>"]


def get_nemo_gym_uv_cache_dir() -> str | None:
    """Return the uv cache directory inside a container, or None outside one.

    Inside a container (NRL_CONTAINER=1), returns the uv cache location so Gym
    stores its caches in the expected shared path. Returns None outside a
    container, meaning the caller should omit this arg and let Gym create the
    cache locally (the default when you may not be able to write to /opt).
    """
    if not os.environ.get("NRL_CONTAINER"):
        return None
    return subprocess.check_output(["uv", "cache", "dir"]).decode().strip()


def get_nemo_gym_venv_dir() -> str | None:
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
    initial_global_config_dict: Dict[str, Any]
    # Port range for Gym HTTP servers (head server + subprocess servers).
    # Defaults to DEFAULT_GYM_PORT_RANGE_LOW/HIGH (15001-20000) from
    # nemo_rl.distributed.virtual_cluster.  See the port layout there.
    port_range_low: NotRequired[int]
    port_range_high: NotRequired[int]
    invalid_tool_call_patterns: NotRequired[
        List[str] | None
    ]  # Substrings in assistant text content that indicate an invalid tool call
    thinking_tags: NotRequired[
        List[str] | None
    ]  # Thinking tags to check for malformed usage
    # Gate-authoritative token capture (token_capture.enabled): the dumped
    # TokenCaptureConfig. Turns on the gate in Gym's policy model server,
    # switches run_rollouts to receipt mode, and adds the register/seal/fail
    # control-plane helpers. None/absent = legacy token-echo path.
    token_capture: NotRequired[Dict[str, Any] | None]


# Gym control-plane server name (the model server hosting the gate) and the
# metadata key rollout ids ride on (defined in Gym's token_id_capture.gate).
_POLICY_SERVER_NAME = "policy_model"
_NG_ROLLOUT_ID_METADATA_KEY = "ng_rollout_id"


def _detect_invalid_tool_call_and_malformed_thinking(
    output_item_dict: dict[str, Any],
    invalid_tool_call_patterns: list[str] | None = None,
    thinking_tags: list[str] | None = None,
) -> tuple[bool, bool]:
    """Flag a NeMo-Gym output item as an invalid tool call / malformed thinking.

    Inspects the final output item of a model turn. For a final *content*
    message, any thinking tag is malformed (thinking should never leak into the
    answer); for a *reasoning* summary, only a repeated tag (count > 1) is
    malformed (a single pair is expected). A textual tool-call pattern in either
    indicates an invalid (unexecuted) tool call.

    Returns:
        (is_invalid_tool_call, has_malformed_thinking).
    """
    invalid_tool_call_patterns = (
        invalid_tool_call_patterns or DEFAULT_INVALID_TOOL_CALL_PATTERNS
    )
    thinking_tags = thinking_tags or DEFAULT_THINKING_TAGS

    is_output_message = (
        "content" in output_item_dict
        and len(output_item_dict["content"]) > 0
        and "text" in output_item_dict["content"][0]
    )
    # NeMo-Gym only attaches generation_token_ids to the last output item of a
    # model call (see vllm_model/app.py postprocess_chat_response). So this item
    # is guaranteed to be the final thing the model produced for this turn.
    # If it's a reasoning item, the model output only reasoning (no content/tool calls).
    is_reasoning_message = (
        output_item_dict.get("type") == "reasoning"
        and len(output_item_dict.get("summary", [])) > 0
        and "text" in output_item_dict["summary"][0]
    )

    is_invalid_tool_call = False
    has_malformed_thinking = False
    if is_output_message:
        assistant_message_content = output_item_dict["content"][0]["text"]
        if any(
            pattern in assistant_message_content
            for pattern in invalid_tool_call_patterns
        ):
            is_invalid_tool_call = True
        if any(tag in assistant_message_content for tag in thinking_tags):
            has_malformed_thinking = True
    elif is_reasoning_message:
        assistant_message_content = output_item_dict["summary"][0]["text"]
        if any(
            pattern in assistant_message_content
            for pattern in invalid_tool_call_patterns
        ):
            is_invalid_tool_call = True
        if any(assistant_message_content.count(tag) > 1 for tag in thinking_tags):
            has_malformed_thinking = True

    return is_invalid_tool_call, has_malformed_thinking


@ray.remote(max_restarts=-1, max_task_retries=-1)  # pragma: no cover
class NemoGym(EnvironmentInterface):
    """This environment class isn't really used for training. It's really meant as an integration wrapper around NeMo-Gym that hooks into the existing NeMo RL resource management via ray. So there is still one source of truth for resource management in NeMo RL."""

    def __init__(self, cfg: NemoGymConfig):
        self.cfg = cfg

    def _spinup(self) -> None:
        """Start the NeMo-Gym head server and rollout collection helper.

        Deferred from __init__ so the actor can be created cheaply (and
        scheduled onto reserved nodes) and spun up explicitly once the vLLM
        server URLs are available, overlapping with vLLM model loading.
        """
        self.node_ip = _get_node_ip_local()
        _gym_port_low = self.cfg.get("port_range_low", DEFAULT_GYM_PORT_RANGE_LOW)
        _gym_port_high = self.cfg.get("port_range_high", DEFAULT_GYM_PORT_RANGE_HIGH)
        self.head_server_port = _get_free_port_local(_gym_port_low, _gym_port_high)

        from nemo_gym.cli import GlobalConfigDictParserConfig, RunHelper
        from nemo_gym.rollout_collection import RolloutCollectionHelper
        from nemo_gym.server_utils import HEAD_SERVER_KEY_NAME, BaseServerConfig
        from omegaconf import DictConfig

        RELATIVE_PATH = "nemo_rl/environments/nemo_gym.py"
        assert __file__.endswith(RELATIVE_PATH)

        # Make a shallow copy so that NeMo-RL-side keys we pop or add below
        # do not mutate the caller's config dict (config.env["nemo_gym"]).
        initial_global_config_dict = dict(
            self.cfg.get("initial_global_config_dict") or {}
        )
        # Strip NeMo-RL-only training knobs that must not be forwarded to the
        # NeMo-Gym server (same pattern as the pops in run_grpo_nemo_gym.py).
        initial_global_config_dict.pop("effort_levels", None)
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

        _gym_port_low = self.cfg.get("port_range_low", DEFAULT_GYM_PORT_RANGE_LOW)
        _gym_port_high = self.cfg.get("port_range_high", DEFAULT_GYM_PORT_RANGE_HIGH)
        if (
            _gym_port_low < DEFAULT_GYM_PORT_RANGE_LOW
            or _gym_port_high > DEFAULT_GYM_PORT_RANGE_HIGH
        ):
            print(
                f"WARNING: Gym port range [{_gym_port_low}, {_gym_port_high}) is outside "
                f"the default [{DEFAULT_GYM_PORT_RANGE_LOW}, {DEFAULT_GYM_PORT_RANGE_HIGH}). "
                f"Check the port layout in virtual_cluster.py for conflicts."
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

        # Gate-authoritative token capture: turn on the gate in the policy
        # model server (via the policy_model global-config override block the
        # env yamls already use) and disable the legacy token echo. Receipt
        # mode is incompatible with re-dispatching a batch under the same
        # rollout ids, so the NaN retry must be exactly 1 (create-only
        # registration would fail the retry loudly anyway; fail at setup).
        token_capture = self.cfg.get("token_capture") or None
        self._token_capture_enabled = bool(
            token_capture and token_capture.get("enabled")
        )
        self._server_client = None
        if self._token_capture_enabled:
            if self.rollout_max_attempts_to_avoid_lp_nan != 1:
                raise ValueError(
                    "token_capture.enabled requires "
                    "rollout_max_attempts_to_avoid_lp_nan == 1: a NaN retry "
                    "would re-register create-only rollout ids at the gate"
                )
            policy_overrides = (
                initial_global_config_dict.setdefault("policy_model", {})
                .setdefault("responses_api_models", {})
                .setdefault("vllm_model", {})
            )
            policy_overrides["return_token_id_information"] = False
            policy_overrides["token_capture_gate"] = {
                "enabled": True,
                "registration_ttl_s": token_capture["registration_ttl_s"],
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

    # ── gate control plane (token-capture mode) ─────────────────────────────

    def _control_client(self):
        """Gym ServerClient resolving servers by name from the head server."""
        if self._server_client is None:
            from nemo_gym.server_utils import ServerClient

            self._server_client = ServerClient.load_from_global_config(
                self.head_server_config
            )
        return self._server_client

    async def _control(self, method: str, path: str, **kwargs: Any) -> dict:
        response = await self._control_client().request(
            server_name=_POLICY_SERVER_NAME, url_path=path, method=method, **kwargs
        )
        if response.status != 200:
            raise RuntimeError(
                f"gate control call {method} {path} failed: "
                f"HTTP {response.status} {await response.text()}"
            )
        return await response.json()

    async def register_rollouts(self, rollout_ids: list[str]) -> None:
        """Create-only registration before dispatch (§ 3.1)."""
        for rollout_id in rollout_ids:
            await self._control("PUT", f"/ng-control/rollouts/{rollout_id}")

    async def fail_rollouts(self, rollout_ids: list[str], *, reason: str) -> None:
        """Best-effort gate cleanup for a cancelled/failed dispatch (§ 7)."""
        for rollout_id in rollout_ids:
            try:
                await self._control(
                    "POST",
                    f"/ng-control/rollouts/{rollout_id}/fail",
                    json={"reason": reason},
                )
            except (RuntimeError, OSError) as error:
                # The registration TTL is the backstop when the gate is
                # unreachable during teardown.
                print(f"fail_rollout({rollout_id}) failed: {error}", flush=True)

    async def gate_metrics(self) -> dict:
        return await self._control("GET", "/ng-control/metrics")

    async def run_rollouts(
        self,
        nemo_gym_examples: list[dict],
        tokenizer: PreTrainedTokenizerBase,
        timer_prefix: str,
    ) -> list[dict]:
        timer = Timer()

        if self._token_capture_enabled:
            # Receipt mode: register the gate-registered ids riding each
            # row's metadata before dispatch; seal at completion.
            rollout_ids = [
                example["responses_create_params"]["metadata"][
                    _NG_ROLLOUT_ID_METADATA_KEY
                ]
                for example in nemo_gym_examples
            ]
            await self.register_rollouts(rollout_ids)

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
                with timer.time(label=f"{timer_prefix}/await_results"):
                    nemo_gym_row, nemo_gym_result = await task

                with timer.time(label=f"{timer_prefix}/postprocess_results"):
                    if self._token_capture_enabled:
                        nemo_rl_result = await self._postprocess_receipt_mode(
                            nemo_gym_row, nemo_gym_result
                        )
                    else:
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

    async def _postprocess_receipt_mode(
        self, nemo_gym_row: dict, nemo_gym_result: dict
    ) -> dict:
        """Seal the rollout and return a token-free result (§ 9.1).

        The legacy token walk (and its contiguity assert) does not run: the
        gate owns lineage now, output items carry no token arrays, and the
        canonical row is rebuilt by the finalizer from staged deltas. The
        Ray return carries only the receipt (~100 B/call) beside the
        agent-level result.
        """
        assert isinstance(nemo_gym_result, dict), (
            f"Hit a non-successful response when querying NeMo Gym for rollouts: {nemo_gym_result}"
        )
        rollout_id = nemo_gym_row["responses_create_params"]["metadata"][
            _NG_ROLLOUT_ID_METADATA_KEY
        ]
        try:
            receipt = await self._control(
                "POST",
                f"/ng-control/rollouts/{rollout_id}/seal",
                json={"reward": float(nemo_gym_result.get("reward") or 0.0)},
            )
        except (RuntimeError, OSError) as error:
            # An unsealable rollout finalizes as a placeholder; the gate's
            # registration TTL sweeps its state.
            print(f"seal({rollout_id}) failed: {error}", flush=True)
            receipt = None
        return {
            "message_log": [],
            "input_message_log": [],
            "full_result": nemo_gym_result,
            "rollout_id": rollout_id,
            "receipt": receipt,
        }

    def _postprocess_nemo_gym_to_nemo_rl_result(
        self, nemo_gym_result: dict, tokenizer: PreTrainedTokenizerBase
    ) -> dict:
        assert isinstance(nemo_gym_result, dict), (
            f"Hit a non-successful response when querying NeMo Gym for rollouts: {nemo_gym_result}"
        )

        nemo_rl_message_log = []
        seen_token_ids: List[int] = []
        batch_decode_items = []
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

            prompt_token_ids = output_item_dict.pop("prompt_token_ids")
            generation_token_ids = output_item_dict.pop("generation_token_ids")
            generation_log_probs = output_item_dict.pop("generation_log_probs")
            new_prompt_token_ids = prompt_token_ids[len(seen_token_ids) :]

            nemo_rl_message_log.append(
                {
                    "role": "user",
                    "content": "",
                    "token_ids": torch.tensor(new_prompt_token_ids),
                }
            )
            # Valid tool calls go through the structured API (tool_calls field) and get
            # executed by NeMo-Gym. If tool call patterns appear in the text content instead,
            # the call was invalid and never executed — flag it so training can penalize it.
            is_invalid_tool_call, has_malformed_thinking = (
                _detect_invalid_tool_call_and_malformed_thinking(
                    output_item_dict,
                    invalid_tool_call_patterns=self.cfg.get(
                        "invalid_tool_call_patterns"
                    ),
                    thinking_tags=self.cfg.get("thinking_tags"),
                )
            )

            nemo_rl_message_log.append(
                {
                    "role": "assistant",
                    "content": "",
                    "token_ids": torch.tensor(generation_token_ids),
                    "generation_logprobs": torch.tensor(generation_log_probs),
                    "is_invalid_tool_call": is_invalid_tool_call,
                    "has_malformed_thinking": has_malformed_thinking,
                }
            )

            seen_token_ids.extend(new_prompt_token_ids)
            seen_token_ids.extend(generation_token_ids)

            # We pop to remove larger tensors from logging.
            batch_decode_items.append(
                (output_item_dict, prompt_token_ids, generation_token_ids)
            )

        if batch_decode_items:
            prompt_strs = tokenizer.batch_decode(
                [item[1] for item in batch_decode_items]
            )
            generation_strs = tokenizer.batch_decode(
                [item[2] for item in batch_decode_items]
            )

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
    generation_config = config.policy["generation"]

    # Enable the http server. Requires both async engine and the expose_http_server flag
    generation_config["vllm_cfg"]["async_engine"] = True
    generation_config["vllm_cfg"]["expose_http_server"] = True

    # Stop strings or token ids are not supported
    generation_config["stop_strings"] = None
    generation_config["stop_token_ids"] = None


def spinup_nemo_gym_actor(
    env_configs: dict[str, Any],
    base_urls: list[Optional[str]],
    model_name: str,
    token_capture: Optional[dict[str, Any]] = None,
) -> Any:
    """Spin up the NeMo-Gym actor against the given generation server URLs.

    When ``env_configs["nemo_gym"]["num_gpu_nodes"] > 0``, the actor is
    scheduled with soft NodeAffinity to the current Ray node so its colocated
    GPU resources land where the caller expects.
    """
    nemo_gym_py_exec = get_actor_python_env("nemo_rl.environments.nemo_gym.NemoGym")
    if nemo_gym_py_exec.startswith("uv"):
        nemo_gym_py_exec = create_local_venv_on_each_node(
            nemo_gym_py_exec, "nemo_rl.environments.nemo_gym.NemoGym"
        )

    nemo_gym_dict = env_configs["nemo_gym"]
    # NeMo-RL-side detection knobs are top-level NemoGymConfig fields
    # (where the detector reads them), not part of Gym's global config.
    invalid_tool_call_patterns = nemo_gym_dict.pop("invalid_tool_call_patterns", None)
    thinking_tags = nemo_gym_dict.pop("thinking_tags", None)
    uv_cache_dir = get_nemo_gym_uv_cache_dir()
    if uv_cache_dir is not None:
        nemo_gym_dict.setdefault("uv_cache_dir", uv_cache_dir)
    uv_venv_dir = get_nemo_gym_venv_dir()
    if uv_venv_dir is not None:
        nemo_gym_dict.setdefault("uv_venv_dir", uv_venv_dir)

    nemo_gym_cfg = NemoGymConfig(
        model_name=model_name,
        base_urls=base_urls,
        invalid_tool_call_patterns=invalid_tool_call_patterns,
        thinking_tags=thinking_tags,
        initial_global_config_dict=nemo_gym_dict,
        token_capture=token_capture,
    )

    nemo_gym_opts: dict[str, Any] = {}
    if nemo_gym_dict.get("num_gpu_nodes", 0):
        nemo_gym_opts["scheduling_strategy"] = NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().get_node_id(),
            soft=True,
        )
    nemo_gym_opts["runtime_env"] = {
        "py_executable": nemo_gym_py_exec,
        "env_vars": {
            **os.environ,
            "VIRTUAL_ENV": nemo_gym_py_exec,
            "UV_PROJECT_ENVIRONMENT": nemo_gym_py_exec,
        },
    }

    actor = NemoGym.options(**nemo_gym_opts).remote(nemo_gym_cfg)
    ray.get(actor._spinup.remote())
    return actor
