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
import sys
from pathlib import Path
from typing import Any, Dict, List, NotRequired, TypedDict

import ray
import torch
from transformers import PreTrainedTokenizerBase

from nemo_rl.distributed.virtual_cluster import (
    DEFAULT_GYM_PORT_RANGE_HIGH,
    DEFAULT_GYM_PORT_RANGE_LOW,
    _get_free_port_local,
    _get_node_ip_local,
)
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.utils.timer import Timer

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
    require_routed_experts: NotRequired[
        bool
    ]  # Require Gym output items to carry R3 routed_experts
    # Train on ALL session traces of a rollout (root + subagent sessions +
    # compaction segments) when the env returns a plural `responses` list.
    # False (default): keep only the root session's traces — matches legacy
    # single-trace behavior even against a plural-responses Gym.
    train_on_all_session_traces: NotRequired[bool]


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
                with timer.time(label=f"{timer_prefix}/await_results"):
                    nemo_gym_row, nemo_gym_result = await task

                with timer.time(label=f"{timer_prefix}/postprocess_results"):
                    # One rollout may yield multiple traces (subagent sessions,
                    # compaction segments) — keep them grouped per row so the
                    # rowidx re-sort below stays aligned with the input batch.
                    nemo_rl_row_traces = self._postprocess_nemo_gym_to_nemo_rl_result(
                        nemo_gym_result, tokenizer
                    )

                nemo_rl_rowidxs.append(nemo_gym_row["_rowidx"])
                nemo_rl_results.append(nemo_rl_row_traces)

            # determine if generation_logprobs contain NaN; if not, break;
            logprob_contains_nan = False
            for nemo_rl_row_traces in nemo_rl_results:
                for nemo_rl_result in nemo_rl_row_traces:
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

        # Re-sort rows to input order; each row holds a LIST of traces.
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
    ) -> list[dict]:
        """Convert one NeMo-Gym rollout result into a list of NeMo RL trace results.

        Environments may return multiple trainable traces per rollout via a
        `responses` list (e.g. opencode subagent sessions and compaction
        segments, one response per on-policy-contiguous segment). Legacy
        environments return a single `response`; that case yields one trace.

        Every trace shares the same `full_result` dict (and therefore the same
        rollout-level reward) so downstream reward mutation (penalties, reward
        shaping) applies to all traces of the rollout at once.
        """
        assert isinstance(nemo_gym_result, dict), (
            f"Hit a non-successful response when querying NeMo Gym for rollouts: {nemo_gym_result}"
        )

        responses = nemo_gym_result.get("responses") or []
        if responses:
            # The legacy aggregate `response` duplicates the per-segment data.
            # Drop its bulky token arrays so they don't ride along in
            # full_result (replay buffer / logging).
            for output_item_dict in (nemo_gym_result.get("response") or {}).get(
                "output", []
            ) or []:
                output_item_dict.pop("prompt_token_ids", None)
                output_item_dict.pop("generation_token_ids", None)
                output_item_dict.pop("generation_log_probs", None)
        elif nemo_gym_result.get("response") is not None:
            responses = [nemo_gym_result["response"]]
        # else: no responses at all (crashed agent, contract-violating env) —
        # fall through with zero traces so the empty-rollout masking below
        # handles it instead of crashing the whole rollout batch.

        if len(responses) > 1 and not self.cfg.get(
            "train_on_all_session_traces", False
        ):
            # Opt-in gate: keep only the ROOT session's segments (empty
            # parent_session_id) so a plural-responses Gym reproduces legacy
            # single-session training exactly. NOTE: without compaction a
            # subagent's every segment carries its parent id; with compaction
            # a subagent's own summary segment is dumped without one (classify
            # per session before relying on this filter under compaction).
            root_responses = [
                r
                for r in responses
                if not ((r.get("metadata") or {}).get("parent_session_id"))
            ]
            if root_responses:
                responses = root_responses

        trace_results = []
        for response in responses:
            # Per-segment provenance from the env (for the SWE agent:
            # session_id, parent_session_id, segment_index,
            # segment_boundary_reason). Carried on every trace so the trainer
            # can write per-segment-kind debug rows (rollout_debug_step*.jsonl).
            trace_metadata = dict(response.get("metadata") or {})
            # A response yields at most one message log (segments must be
            # prefix-contiguous — the builder hard-asserts on mid-segment
            # history rewrites). Segments with no trainable generations yield
            # nothing and are skipped; we only mask the rollout if it has no
            # generations at all.
            for nemo_rl_message_log in self._build_trace_message_logs(
                response, tokenizer
            ):
                if trace_metadata.get("segment_boundary_reason") == "compaction":
                    # This segment's generation IS the summary the next segment
                    # conditions on; keep the decoded text (attached by the
                    # builder above as generation_str) so summary quality can
                    # be inspected offline.
                    trace_metadata["generation_text"] = "".join(
                        item.get("generation_str", "")
                        for item in (response.get("output") or [])
                        if isinstance(item, dict)
                    )
                trace_results.append(
                    {
                        "message_log": nemo_rl_message_log,
                        "input_message_log": nemo_rl_message_log[:1],
                        "full_result": nemo_gym_result,
                        "trace_metadata": trace_metadata,
                    }
                )

        if not trace_results:
            # A rollout can legitimately come back with no generation data at
            # scale (agent crashed before any model call, apptainer OOM, prompt
            # exceeded max_model_len, ...). Raising here would kill the whole
            # rollout batch worker and restart every rollout in it, so instead
            # mark the rollout failed: emit one minimal masked trace with no
            # trainable tokens. Its reward (usually 0) still counts toward the
            # group's advantage baseline, and `is_empty_rollout` makes the
            # downstream batch mask it from the gradient.
            input_messages = (nemo_gym_result.get("responses_create_params") or {}).get("input") or []
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
                o.get("type") for o in (nemo_gym_result.get("response") or {}).get("output", []) or []
            ]
            instance_config = nemo_gym_result.get("instance_config") or {}
            print(
                f"WARNING: NeMo Gym returned a result with no generation data in any trace — "
                f"masking this rollout (reward {nemo_gym_result.get('reward')} still counts toward the group baseline). "
                f"Possible causes: (1) the first-turn prompt exceeds the vLLM max_model_len; "
                f"(2) the agent produced no assistant generation (agent error/timeout/OOM before any model call).\n"
                f"  Instance: {instance_config.get('name')} agent_error_kind={nemo_gym_result.get('agent_error_kind')} "
                f"agent_timed_out={nemo_gym_result.get('agent_timed_out')} oom_killed={nemo_gym_result.get('oom_killed')}.\n"
                f"  Number of traces: {len(responses)}.\n"
                f"  Prompt length: {prompt_len_str}.\n"
                f"  response.output item types ({len(output_item_types)} items): {output_item_types}.",
                file=sys.stderr,
            )
            dummy_token_id = tokenizer.pad_token_id
            if dummy_token_id is None:
                dummy_token_id = tokenizer.eos_token_id or 0
            dummy_message_log = [
                {
                    "role": "user",
                    "content": "",
                    "token_ids": torch.tensor([dummy_token_id], dtype=torch.int64),
                }
            ]
            trace_results.append(
                {
                    "message_log": dummy_message_log,
                    "input_message_log": dummy_message_log[:1],
                    "full_result": nemo_gym_result,
                    "is_empty_rollout": True,
                    "trace_metadata": {},
                }
            )

        # trace_idx==0 marks the rollout's representative trace; reward
        # penalties use it to evaluate rollout-scoped checks exactly once.
        for trace_idx, trace_result in enumerate(trace_results):
            trace_result["trace_idx"] = trace_idx

        return trace_results

    def _build_trace_message_logs(
        self, response: dict, tokenizer: PreTrainedTokenizerBase
    ) -> list[list[dict]]:
        """Build NeMo RL message logs from a single response.

        Every model call's prompt must extend the previously-seen token prefix:
        segments are on-policy contiguous by construction (compaction starts a
        NEW segment), so a prefix mismatch inside one response means the
        harness rewrote history mid-segment — a harness bug we want to fail
        loudly on rather than train through.
        """
        message_logs: list[list[dict]] = []
        nemo_rl_message_log = []
        seen_token_ids: List[int] = []
        batch_decode_items = []
        for output_item_dict in response["output"]:
            # Nemo RL really only has two types of messages: assistant and not assistant since that is all that it is concerned with (i.e. to train or not to train)
            # Here we map all the trainable messages to assistant and all the non-trainable messages to user.
            # Eventually we can maybe be smarter about this, but this is functional for now.

            # Note that NeMo-Gym will only return token ids on "assistant" messages and not other message types.
            if "generation_token_ids" not in output_item_dict:
                continue

            n_seen = len(seen_token_ids)
            assert (
                len(output_item_dict["prompt_token_ids"]) >= n_seen
                and seen_token_ids == output_item_dict["prompt_token_ids"][:n_seen]
            ), f"""Non-contiguous messages found! This may be a tokenization issue where certain tokens are combined when messages are concatenated, or it may be due to part of the chat history being truncated (like if super long history is truncated or if reasoning is stripped out). Segments are on-policy contiguous by construction, so this indicates a harness bug.
Response id: {response.get("id")}, metadata: {response.get("metadata")}
n_seen={n_seen}, new_prompt_len={len(output_item_dict["prompt_token_ids"])}
Seen token IDs (tail): {seen_token_ids[-32:]}
Output prompt token IDs (at divergence region): {output_item_dict["prompt_token_ids"][max(0, n_seen - 32):n_seen + 32]}
"""

            prompt_token_ids = output_item_dict.pop("prompt_token_ids")
            generation_token_ids = output_item_dict.pop("generation_token_ids")
            generation_log_probs = output_item_dict.pop("generation_log_probs")
            routed_experts_raw = output_item_dict.pop("routed_experts", None)
            new_prompt_token_ids = prompt_token_ids[len(seen_token_ids) :]

            routed_experts = None
            if routed_experts_raw is not None:
                routed_experts = torch.as_tensor(routed_experts_raw, dtype=torch.int32)
                if routed_experts.dim() != 3:
                    raise ValueError(
                        "NeMo Gym returned routed_experts with invalid shape. "
                        "Expected [tokens, num_moe_layers, topk], got "
                        f"{tuple(routed_experts.shape)}."
                    )
                expected_tokens = len(prompt_token_ids) + len(generation_token_ids)
                if routed_experts.shape[0] < expected_tokens:
                    raise ValueError(
                        "NeMo Gym returned too few routed_experts rows for a "
                        "trainable output item: "
                        f"routes={routed_experts.shape[0]}, expected_at_least="
                        f"{expected_tokens}."
                    )
            elif self.cfg.get("require_routed_experts", False):
                raise ValueError(
                    "policy.router_replay.enabled=true requires NeMo Gym output "
                    "items to include routed_experts, but the field was missing. "
                    "Make sure the Gym repo includes routed_experts propagation "
                    "and the NeMo-RL vLLM OpenAI-compatible server is configured "
                    "with enable_return_routed_experts."
                )

            prompt_start = len(seen_token_ids)
            prompt_end = len(prompt_token_ids)
            generation_start = prompt_end
            generation_end = prompt_end + len(generation_token_ids)

            user_message = {
                "role": "user",
                "content": "",
                "token_ids": torch.tensor(new_prompt_token_ids),
            }
            if routed_experts is not None:
                user_message["routed_experts"] = routed_experts[prompt_start:prompt_end]
            nemo_rl_message_log.append(user_message)
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

            assistant_message = {
                "role": "assistant",
                "content": "",
                "token_ids": torch.tensor(generation_token_ids),
                "generation_logprobs": torch.tensor(generation_log_probs),
                "is_invalid_tool_call": is_invalid_tool_call,
                "has_malformed_thinking": has_malformed_thinking,
            }
            if routed_experts is not None:
                assistant_message["routed_experts"] = routed_experts[
                    generation_start:generation_end
                ]
            nemo_rl_message_log.append(assistant_message)

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

        # A response with no trainable generations yields no message log; the
        # rollout-level empty case (no generations in ANY trace) is handled in
        # _postprocess_nemo_gym_to_nemo_rl_result with a loud warning + masked
        # dummy trace. NOTE: this replaces the previous single-trace behavior of
        # raising ValueError on an empty rollout — with multi-trace, a raise
        # would kill the whole rollout batch worker over one crashed agent.
        if nemo_rl_message_log:
            message_logs.append(nemo_rl_message_log)

        return message_logs

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
