# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
from collections import Counter
from collections.abc import AsyncGenerator
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, NotRequired, Optional, TypedDict

import ray
import torch
from PIL import Image
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from transformers import PreTrainedTokenizerBase

from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
from nemo_rl.data.multimodal_utils import (
    PackedTensor,
    encode_images_in_examples,
    get_dim_to_pack_along,
    get_multimodal_keys_from_processor,
    resolve_to_image,
    uses_image_placeholder,
)

from nemo_rl.distributed.virtual_cluster import (
    DEFAULT_GYM_PORT_RANGE_HIGH,
    DEFAULT_GYM_PORT_RANGE_LOW,
    _get_free_port_local,
    _get_node_ip_local,
)
from nemo_rl.environments.generation_contract import (
    bind_runtime_generation_contract,
    build_training_admission_contract,
    canonical_digest,
    validate_runtime_generation_contract,
)
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.environments.nemo_gym_trace import build_rollout_trace_bundle
from nemo_rl.models.policy import TokenizerConfig
from nemo_rl.utils.routed_experts_codec import decode_routed_experts
from nemo_rl.utils.timer import Timer
from nemo_rl.utils.venvs import create_local_venv_on_each_node

# Kept local (not imported from models.generation) so the gym actor stays free of
# generation-module imports. Must cover every name resolve_routed_experts_dtype
# can produce.
_ROUTED_EXPERTS_DTYPES = {
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
}

DEFAULT_INVALID_TOOL_CALL_PATTERNS = [
    "<tool_call>",
    "</tool_call>",
    "<function_call>",
    "</function_call>",
]
DEFAULT_THINKING_TAGS = ["<think>", "</think>"]

_EXACT_TRACE_RESPONSE_PROJECTION_FIELDS = (
    "id",
    "status",
    "error",
    "incomplete_details",
    "usage",
    "output",
    "context_compaction_contract",
    "chunk_records",
    "boundary_events",
    "guard_records",
)
_MEDIA_PART_TYPES = frozenset({"input_image", "image", "image_url"})


def _has_nan_generation_logprobs(result: dict) -> bool:
    """Return whether a postprocessed rollout contains NaN policy logprobs."""
    return any(
        message.get("generation_logprobs") is not None
        and torch.isnan(message["generation_logprobs"]).any()
        for message in result["message_log"]
    )


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
    # Defaults to DEFAULT_GYM_PORT_RANGE_LOW/HIGH (5000-5999) from
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
    routed_experts_dtype: NotRequired[
        str
    ]  # Carry dtype name for routed_experts tensors ("int8"/"int16"/"int32"), resolved from the model's expert count
    # Forwarded from policy.tokenizer.use_fastokens so rollout actors patch their
    # tokenizer consistently with the driver. Defaults to off when absent.
    use_fastokens: NotRequired[bool]
    # Multimodal fields (populated by `setup_nemo_gym_config` when VLM is enabled).
    tokenizer_config: NotRequired[
        Optional[TokenizerConfig]
    ]  # For processor reconstruction inside the actor
    context_compaction_runtime_contract: NotRequired[
        Optional[Dict[str, Any]]
    ]  # Launcher-owned model/tokenizer/template/processor identity


def _compact_json_size(value: Any) -> int:
    return len(
        json.dumps(
            value,
            separators=(",", ":"),
            ensure_ascii=False,
            default=repr,
        ).encode("utf-8")
    )


def _actor_peak_rss_gib() -> float:
    """Return this Linux actor process's lifetime peak resident set size."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)


def _project_semantic_value(value: Any) -> Any:
    """Remove raw media and generation-private bulk from a semantic log value."""
    if isinstance(value, list):
        projected = []
        for item in value:
            if isinstance(item, dict) and item.get("type") in _MEDIA_PART_TYPES:
                continue
            projected.append(_project_semantic_value(item))
        return projected
    if isinstance(value, dict):
        return {
            key: _project_semantic_value(item)
            for key, item in value.items()
            if key
            not in {
                "encrypted_content",
                "prompt_str",
                "prompt_token_ids",
                "generation_token_ids",
                "generation_log_probs",
                "routed_experts",
            }
            and not (
                key in {"image", "image_url", "url"}
                and isinstance(item, str)
                and item.startswith("data:image")
            )
        }
    return value


def _build_exact_trace_full_result_projection(
    nemo_gym_result: dict[str, Any],
    *,
    trace_bundle: dict[str, Any],
    generation_only: bool,
) -> dict[str, Any]:
    """Build the bounded Ray/logging projection after exact trace factoring.

    The complete Gym HTTP result is required until tokens and media have been
    independently validated and materialized. After that point, training
    consumers need only scalar environment results and a small semantic
    projection used by reward penalties. Exact token/media authority lives in
    ``trace_bundle`` and the physical message logs, not in ``full_result``.
    """
    gym_http_bytes = _compact_json_size(nemo_gym_result)
    response = nemo_gym_result["response"]
    response_projection = {
        key: (
            _project_semantic_value(response[key])
            if key == "output"
            else deepcopy(response[key])
        )
        for key in _EXACT_TRACE_RESPONSE_PROJECTION_FIELDS
        if key in response and response[key] is not None
    }
    projection = {
        key: deepcopy(value)
        for key, value in nemo_gym_result.items()
        if key
        not in {
            "response",
            "responses_create_params",
            "nemo_rl_trace_bundle",
        }
    }
    projection["response"] = response_projection

    projection["context_compaction_gym_http_bytes"] = gym_http_bytes
    projection["context_compaction_ray_env_extras_bytes"] = 0
    projection["context_compaction_transport_reduction_ratio"] = 0.0
    # These metrics contribute a few bytes to the object they measure. Iterate
    # to a fixed point so the reported projection size includes the final
    # integer and ratio rather than their zero placeholders.
    for _ in range(8):
        ray_env_extras_bytes = _compact_json_size(projection)
        reduction_ratio = (
            1.0 - (ray_env_extras_bytes / gym_http_bytes) if gym_http_bytes else 0.0
        )
        current = (
            projection["context_compaction_ray_env_extras_bytes"],
            projection["context_compaction_transport_reduction_ratio"],
        )
        updated = (ray_env_extras_bytes, reduction_ratio)
        projection["context_compaction_ray_env_extras_bytes"] = ray_env_extras_bytes
        projection["context_compaction_transport_reduction_ratio"] = reduction_ratio
        if current == updated:
            break
    if generation_only:
        # Trajectory collection intentionally persists the canonical trace
        # bundle. Training already carries it through the dedicated
        # rollout_trace_bundle field and must not duplicate it in env_extras.
        projection["nemo_rl_trace_bundle"] = trace_bundle
        projection["context_compaction_trajectory_record_bytes"] = 0
        for _ in range(8):
            trajectory_record_bytes = _compact_json_size(projection)
            if (
                projection["context_compaction_trajectory_record_bytes"]
                == trajectory_record_bytes
            ):
                break
            projection["context_compaction_trajectory_record_bytes"] = (
                trajectory_record_bytes
            )
    return projection


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


########################################
# Multimodal helpers
########################################


def _extract_input_images_from_message(item: dict) -> list[Image.Image]:
    """Pull PIL images out of a non-assistant Responses-API item.

    Handles both content-list items (user / tool messages carrying
    ``input_image``/``image``/``image_url`` parts) and ``function_call_output``
    items whose ``output`` field is an image data URL.
    """
    images: list[Image.Image] = []
    if item.get("type") == "function_call_output":
        src = item.get("output")
        if isinstance(src, str):
            images.append(resolve_to_image(src))
        return images
    content = item.get("content") or []
    if not isinstance(content, list):
        return images
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") not in ("input_image", "image", "image_url"):
            continue
        src = part.get("image") or part.get("image_url") or part.get("url")
        if src is None:
            continue
        if isinstance(src, dict):
            src = src.get("url")
        if src is None:
            continue
        images.append(resolve_to_image(src))
    return images


def _index_per_turn_images(
    initial_input: list[dict], seed_obs: list[dict], output: list[dict]
) -> list[list[Image.Image]]:
    """Bin server-returned images by the trainable turn that saw them.

    Walks the Responses-API items in order and flushes ``pending`` into a
    per-turn bucket each time it hits an item carrying truthy
    ``generation_token_ids`` — matching the exact gate that
    ``_postprocess_nemo_gym_to_nemo_rl_result`` uses to decide which items
    become trainable turns. Every other item (user turns, tool messages,
    ``function_call_output``, non-trainable reasoning) contributes its images
    to ``pending`` for the next trainable turn. This ensures the returned list
    has one entry per trainable turn, aligned with the postprocess loop's
    ``turn_idx`` even when the trainable item's role is not ``assistant``
    (e.g. a reasoning-only response, or a ``function_call``).
    """
    per_turn: list[list[Image.Image]] = []
    pending: list[Image.Image] = []
    for item in [*(initial_input or []), *(seed_obs or []), *output]:
        if item.get(
            "generation_token_ids"
        ):  # empty generations are not trainable and must not consume a bucket
            per_turn.append(pending)
            pending = []
        elif item.get("role") != "assistant":
            pending.extend(_extract_input_images_from_message(item))
    return per_turn


def _resolve_images_by_media_id(
    media_assets: dict[str, dict[str, Any]],
    media_ids: list[str],
) -> list[Image.Image]:
    """Resolve ordered media occurrences from the rollout-owned media arena."""
    images: list[Image.Image] = []
    for media_id in media_ids:
        try:
            asset = media_assets[media_id]
        except KeyError as exc:
            raise ValueError(
                f"Completion evidence references unknown media ID {media_id!r}"
            ) from exc
        part = asset.get("source_part", asset) if isinstance(asset, dict) else asset
        if not isinstance(part, dict):
            raise TypeError(f"Media asset {media_id!r} source_part must be a mapping")
        src = part.get("image") or part.get("image_url") or part.get("url")
        if isinstance(src, dict):
            src = src.get("url")
        if src is None:
            raise ValueError(
                f"Media asset {media_id!r} does not contain an image source"
            )
        images.append(resolve_to_image(src))
    return images


def _attach_multimodal_data_to_user_message(
    user_message: dict,
    *,
    images: list[Image.Image],
    processor: Any,
) -> None:
    """Attach per-turn multimodal tensors to ``user_message``.

    The processor is only invoked to extract multimodal tensors (pixel_values,
    imgs_sizes, num_patches, etc.); its text output is discarded — vLLM's
    tokens remain the trajectory. We therefore feed it the minimal placeholder
    text it needs to count image regions: one ``processor.image_token`` per
    image. Passing the vLLM-decoded text does not work because that text
    already contains expanded ``<img>...<image>*N...</img>`` regions, and the
    processor would try to re-expand every embedded ``<image>``.
    """
    if not images or processor is None:
        return
    image_token = getattr(processor, "image_token", "<image>")
    processed = processor(
        text=image_token * len(images),
        images=images,
        return_tensors="pt",
    )
    uses_placeholder = uses_image_placeholder(processor)
    multimodal_keys = list(get_multimodal_keys_from_processor(processor))
    # Historical checkpoints may emit dynamic image tiles without imgs_sizes.
    # Mirror the media-metadata handling in vlm_hf_data_processor.
    if (
        uses_placeholder
        and "pixel_values" in processed
        and "imgs_sizes" not in processed
        and processed["pixel_values"].ndim == 4
    ):
        pixel_values = processed["pixel_values"]
        num_tiles, _, height, width = pixel_values.shape
        processed["imgs_sizes"] = torch.tensor(
            [[height, width]] * num_tiles, dtype=torch.long
        )

    # imgs_sizes / num_frames are not always declared in model_input_names by
    # bundled image processors. RADIO uses temporal patching even for still
    # images and requires one num_frames=1 entry per image/tile.
    if "imgs_sizes" in processed and "imgs_sizes" not in multimodal_keys:
        multimodal_keys.append("imgs_sizes")
    if "imgs_sizes" in processed and "num_frames" not in processed:
        processed["num_frames"] = torch.ones(
            len(processed["imgs_sizes"]), dtype=torch.long
        )
    if "num_frames" in processed and "num_frames" not in multimodal_keys:
        multimodal_keys.append("num_frames")
    for key in multimodal_keys:
        if key not in processed:
            continue
        value = processed[key]
        if key == "imgs_sizes":
            value = value.to(dtype=torch.int32)
        user_message[key] = PackedTensor(
            value,
            dim_to_pack=get_dim_to_pack_along(processor, key),
            pad_to_max_shape=uses_placeholder and key == "pixel_values",
        )


def _stamp_context_compaction_rollout_ids(
    rows: list[dict[str, Any]],
    *,
    rollout_batch_index: int,
    runtime_contract: Optional[Dict[str, Any]] = None,
) -> None:
    """Stamp caller-owned logical rollout IDs before Gym dispatch."""
    if runtime_contract is not None:
        validate_runtime_generation_contract(runtime_contract)
    stamped_ids: set[str] = set()
    for row in rows:
        contract_version = row.get("context_compaction_contract_version")
        if contract_version is None:
            if runtime_contract is not None:
                raise ValueError(
                    "Context-compaction training rows require "
                    "context_compaction_contract_version"
                )
            continue
        if contract_version not in {1, 2}:
            raise ValueError(
                "Unsupported context compaction row contract version: "
                f"{contract_version!r}"
            )
        row_index = row.get("_rowidx")
        group_id = row.get("context_compaction_group_id")
        if (
            not isinstance(row_index, int)
            or not isinstance(group_id, str)
            or not group_id
        ):
            raise ValueError(
                "Context compaction rows require an integer _rowidx and a "
                "non-empty context_compaction_group_id"
            )
        if contract_version == 1:
            rollout_id = (
                f"{group_id}:batch-{rollout_batch_index:06d}:row-{row_index:06d}"
            )
        else:
            task_id = row.get("context_compaction_task_id")
            rollout_index = row.get("context_compaction_rollout_index")
            attempt_index = row.get("context_compaction_attempt_index")
            if (
                not isinstance(task_id, str)
                or not task_id
                or not isinstance(rollout_index, int)
                or rollout_index < 0
                or not isinstance(attempt_index, int)
                or attempt_index < 0
            ):
                raise ValueError(
                    "Version 2 context compaction rows require a non-empty "
                    "context_compaction_task_id and non-negative integer "
                    "context_compaction_rollout_index and "
                    "context_compaction_attempt_index"
                )
            identity = json.dumps(
                {
                    "task_id": task_id,
                    "group_id": group_id,
                    "rollout_index": rollout_index,
                    "attempt_index": attempt_index,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            rollout_id = f"rollout-{digest[:24]}"
        if rollout_id in stamped_ids:
            raise ValueError(f"Duplicate context compaction rollout ID {rollout_id!r}")
        stamped_ids.add(rollout_id)
        row["context_compaction_rollout_id"] = rollout_id
        if runtime_contract is not None:
            row["context_compaction_runtime_contract"] = runtime_contract


@ray.remote(max_restarts=-1, max_task_retries=-1)  # pragma: no cover
class NemoGym(EnvironmentInterface):
    """This environment class isn't really used for training. It's really meant as an integration wrapper around NeMo-Gym that hooks into the existing NeMo RL resource management via ray. So there is still one source of truth for resource management in NeMo RL."""

    def __init__(self, cfg: NemoGymConfig):
        self.cfg = cfg
        # Reconstruct the processor inside the actor (rather than serializing it
        # per rollout call) for full-trajectory multimodal postprocessing.
        self._processor: Optional[Any] = None
        self._rollout_batch_index = 0
        self._context_compaction_runtime_contract = cfg.get(
            "context_compaction_runtime_contract"
        )
        if self._context_compaction_runtime_contract is not None:
            validate_runtime_generation_contract(
                self._context_compaction_runtime_contract
            )
        tokenizer_config = cfg.get("tokenizer_config")
        if tokenizer_config:
            from nemo_rl.algorithms.utils import get_tokenizer

            self._processor = get_tokenizer(tokenizer_config, get_processor=True)

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
        generation_only: bool = False,
        generation_policy_version: Optional[str] = None,
    ) -> AsyncGenerator[tuple[int, dict, dict | None], None]:
        """Stream postprocessed rollouts as NeMo-Gym tasks complete."""
        if not nemo_gym_examples:
            raise ValueError("NeMo-Gym rollout batch must not be empty")

        from nemo_rl.utils.fastokens import maybe_patch_fastokens

        maybe_patch_fastokens(bool(self.cfg.get("use_fastokens")))

        timer = Timer()
        counts_left = Counter(row["agent_ref"]["name"] for row in nemo_gym_examples)

        # For multimodal runs, replace local filesystem image paths in the
        # examples with base64 data URLs before shipping to vLLM. No-op when
        # examples carry no `input_image` items (text-only case).
        encode_images_in_examples(nemo_gym_examples)
        rollout_batch_index = self._rollout_batch_index
        self._rollout_batch_index += 1
        runtime_contract = None
        if (
            self._context_compaction_runtime_contract is not None
            and not generation_only
        ):
            if generation_policy_version is None:
                raise ValueError(
                    "Context-compaction training requires a synchronized "
                    "generation_policy_version"
                )
            runtime_contract = bind_runtime_generation_contract(
                self._context_compaction_runtime_contract,
                generation_policy_version=generation_policy_version,
            )
        _stamp_context_compaction_rollout_ids(
            nemo_gym_examples,
            rollout_batch_index=rollout_batch_index,
            runtime_contract=runtime_contract,
        )

        timer.start("_run_rollouts_total")
        nemo_gym_result_iterator = self.rch.run_examples(
            examples=nemo_gym_examples, head_server_config=self.head_server_config
        )

        num_results = 0
        for task in nemo_gym_result_iterator:
            with timer.time(label=f"{timer_prefix}/await_results"):
                try:
                    nemo_gym_row, nemo_gym_result = await task
                except Exception as error:
                    if hasattr(error, "response_content"):
                        print(
                            "EXCEPTION RESULT",
                            error.response_content,
                            file=sys.stderr,
                        )
                    raise

            with timer.time(label=f"{timer_prefix}/postprocess_results"):
                nemo_rl_result = self._postprocess_nemo_gym_to_nemo_rl_result(
                    nemo_gym_row,
                    nemo_gym_result,
                    tokenizer,
                    generation_only=generation_only,
                )
                if _has_nan_generation_logprobs(nemo_rl_result):
                    raise RuntimeError("Generation logprobs contain NaN")

            num_results += 1
            timing_metrics = None
            if num_results == len(nemo_gym_examples):
                timer.stop("_run_rollouts_total")
                timing_metrics = timer.get_timing_metrics("sum")
                total_time = timing_metrics.pop("_run_rollouts_total")
                timing_metrics[f"{timer_prefix}/postprocess_results_pct"] = (
                    100
                    * timing_metrics[f"{timer_prefix}/postprocess_results"]
                    / total_time
                )
                timing_metrics[f"{timer_prefix}/nemo_gym_actor_peak_rss_gib"] = (
                    _actor_peak_rss_gib()
                )

            agent_name = nemo_gym_row["agent_ref"]["name"]
            counts_left[agent_name] -= 1
            if counts_left[agent_name] <= 0:
                counts_left.pop(agent_name)
            if num_results % 10 == 0 and counts_left:
                top_left = counts_left.most_common(5)
                top_left_str = "\n".join(
                    f"{index + 1}. {name}: {count}"
                    for index, (name, count) in enumerate(top_left)
                )
                print(
                    "Top 5 NeMo Gym agent refs left in this rollout batch: "
                    f"{top_left_str}",
                    file=sys.stderr,
                )

            yield nemo_gym_row["_rowidx"], nemo_rl_result, timing_metrics

    def _postprocess_nemo_gym_to_nemo_rl_result(
        self,
        nemo_gym_row: dict,
        nemo_gym_result: dict,
        tokenizer: PreTrainedTokenizerBase,
        *,
        generation_only: bool = False,
    ) -> dict:
        assert isinstance(nemo_gym_result, dict), (
            f"Hit a non-successful response when querying NeMo Gym for rollouts: {nemo_gym_result}"
        )

        processor = getattr(self, "_processor", None)
        response = nemo_gym_result["response"]
        contract = response.get("context_compaction_contract")
        exact_trace_authority = contract is not None
        if exact_trace_authority:
            if not isinstance(contract, dict):
                raise TypeError("context_compaction_contract must be a mapping")
            contract_version = contract.get("schema_version")
            if (
                contract_version not in {2, 3}
                or contract.get("mode") != "exact_trace_authority"
            ):
                raise ValueError(
                    f"Unsupported context compaction response contract: {contract!r}"
                )
            expected_rollout_id = nemo_gym_row.get("context_compaction_rollout_id")
            if not expected_rollout_id:
                raise ValueError(
                    "An exact-trace response requires a caller-stamped "
                    "context_compaction_rollout_id"
                )
            if contract.get("rollout_id") != expected_rollout_id:
                raise ValueError(
                    "Gym returned evidence for the wrong logical rollout: "
                    f"expected={expected_rollout_id!r}, "
                    f"observed={contract.get('rollout_id')!r}"
                )
            expected_group_id = nemo_gym_row.get("context_compaction_group_id")
            if contract.get("group_id") != expected_group_id:
                raise ValueError(
                    "Gym returned the wrong context compaction group ID: "
                    f"expected={expected_group_id!r}, "
                    f"observed={contract.get('group_id')!r}"
                )
            for contract_field, row_field in (
                ("task_id", "context_compaction_task_id"),
                ("rollout_index", "context_compaction_rollout_index"),
                ("attempt_index", "context_compaction_attempt_index"),
            ):
                if contract.get(contract_field) != nemo_gym_row.get(row_field):
                    raise ValueError(
                        "Gym returned the wrong context compaction "
                        f"{contract_field}: expected="
                        f"{nemo_gym_row.get(row_field)!r}, observed="
                        f"{contract.get(contract_field)!r}"
                    )
            if not isinstance(contract.get("generation_contract"), dict):
                raise ValueError(
                    "Version 2 exact-trace response is missing its generation contract"
                )
            runtime_contract = nemo_gym_row.get("context_compaction_runtime_contract")
            training_admission = (
                build_training_admission_contract(
                    contract["generation_contract"],
                    runtime_contract,
                )
                if runtime_contract is not None
                else None
            )
        else:
            training_admission = None

        initial_input = response.get("agent_input")
        if initial_input is None:
            initial_input = nemo_gym_row.get("responses_create_params", {}).get(
                "input", []
            )
        per_turn_images = _index_per_turn_images(
            initial_input,
            response.get("seed_obs") or [],
            response["output"],
        )
        trainable_output_items = [
            item
            for item in response["output"]
            if "generation_token_ids" in item and item["generation_token_ids"]
        ]
        evidence_field = (
            "model_call_metadata"
            if exact_trace_authority and contract["schema_version"] == 3
            else "completion_evidence"
        )
        completion_evidence = response.get(evidence_field) or []
        if exact_trace_authority and not completion_evidence:
            raise ValueError(
                f"Exact-trace authority response is missing {evidence_field}"
            )
        if (exact_trace_authority or completion_evidence) and len(
            completion_evidence
        ) != len(trainable_output_items):
            raise ValueError(
                "Completion evidence count does not match trainable model calls: "
                f"evidence={len(completion_evidence)} "
                f"calls={len(trainable_output_items)}"
            )

        trace_calls = []
        for call_index, output_item in enumerate(trainable_output_items):
            evidence = completion_evidence[call_index] if completion_evidence else None
            if evidence is not None:
                canonical_arrays = {
                    "prompt_token_ids": output_item["prompt_token_ids"],
                    "sampled_token_ids": output_item["generation_token_ids"],
                    "sampled_logprobs": output_item["generation_log_probs"],
                }
                if contract["schema_version"] == 2:
                    evidence_arrays = {
                        "prompt_token_ids": evidence["prompt_token_ids"],
                        "sampled_token_ids": evidence["sampled_token_ids"],
                        "sampled_logprobs": evidence["sampled_logprobs"],
                    }
                    if evidence_arrays != canonical_arrays:
                        raise ValueError(
                            "Gym completion evidence does not exactly match the "
                            f"generation response at call {call_index}: "
                            f"expected={canonical_arrays}, actual={evidence_arrays}"
                        )
                elif evidence.get("generation_evidence_digest") != canonical_digest(
                    canonical_arrays
                ):
                    raise ValueError(
                        "Gym model-call metadata digest does not match the "
                        f"canonical generation arrays at call {call_index}"
                    )
                turn_id = evidence["turn_id"]
                completion_id = evidence["completion_id"]
                media_ids = evidence.get("media_ids") or []
            else:
                turn_id = call_index + 1
                completion_id = output_item.get("id") or f"completion-{turn_id:06d}"
                media_ids = []
            trace_calls.append(
                {
                    "turn_id": turn_id,
                    "completion_id": completion_id,
                    "prompt_token_ids": output_item["prompt_token_ids"],
                    "sampled_token_ids": output_item["generation_token_ids"],
                    "sampled_logprobs": output_item["generation_log_probs"],
                    "media_ids": media_ids,
                    "prepared_request_id": (
                        evidence.get("prepared_request_id")
                        if evidence is not None
                        else None
                    ),
                    "request_id": (
                        evidence.get("request_id") if evidence is not None else None
                    ),
                    "context_epoch": (
                        evidence.get("context_epoch") if evidence is not None else None
                    ),
                    "segment_index": (
                        evidence.get("segment_index") if evidence is not None else None
                    ),
                    "segment_id": (
                        evidence.get("segment_id") if evidence is not None else None
                    ),
                    "expected_append_compatible": (
                        evidence.get("expected_append_compatible")
                        if evidence is not None
                        else None
                    ),
                    "compaction_event_id": (
                        evidence.get("compaction_event_id")
                        if evidence is not None
                        else None
                    ),
                    "rollout_id": (
                        evidence.get("rollout_id") if evidence is not None else None
                    ),
                    "action_id": (
                        evidence.get("action_id") if evidence is not None else None
                    ),
                    "finish_reason": (
                        evidence.get("finish_reason") if evidence is not None else None
                    ),
                    "policy_decision": (
                        evidence.get("policy_decision")
                        if evidence is not None
                        else None
                    ),
                    "processor_fingerprint": (
                        evidence.get("processor_fingerprint")
                        if evidence is not None
                        else None
                    ),
                    "generation_contract_id": (
                        evidence.get("generation_contract_id")
                        if evidence is not None
                        else None
                    ),
                    "policy_output_spans": (
                        evidence.get("policy_output_spans")
                        if evidence is not None
                        else None
                    ),
                    "media_occurrences": (
                        evidence.get("media_occurrences")
                        if evidence is not None
                        else None
                    ),
                    "eligible": (
                        evidence.get("eligible", True) if evidence is not None else True
                    ),
                    "evidence_source": (
                        evidence.get("evidence_source")
                        if evidence is not None
                        else None
                    ),
                }
            )

        rollout_id = (
            contract["rollout_id"]
            if exact_trace_authority
            else completion_evidence[0]["rollout_id"]
            if completion_evidence
            else f"nemo-gym-row-{nemo_gym_row.get('_rowidx', 0)}"
        )
        policy_name = None
        if completion_evidence:
            policy_name = (
                completion_evidence[0].get("policy_decision", {}).get("policy_name")
            )
        media_assets = response.get("media_assets")
        if exact_trace_authority and media_assets is None:
            raise ValueError(
                "Exact-trace authority response is missing its media asset arena"
            )
        media_assets = media_assets or {}
        trace_bundle = build_rollout_trace_bundle(
            rollout_id=rollout_id,
            calls=trace_calls,
            # Evaluation does not train from physical trace boundaries. Infer
            # discontinuities from the calls instead, so a terminal boundary
            # emitted after the final admitted call cannot abort a whole shard.
            boundary_events=(
                [] if generation_only else response.get("boundary_events") or []
            ),
            policy_name=policy_name,
            group_id=(contract.get("group_id") if exact_trace_authority else None),
            source_row_index=nemo_gym_row.get("_rowidx"),
            reward=nemo_gym_result.get("reward"),
            media_assets=media_assets,
            generation_contract=(
                contract.get("generation_contract") if exact_trace_authority else None
            ),
            training_admission=training_admission,
            final_policy_decision=response.get("final_policy_decision"),
            lineage_deltas=response.get("lineage_deltas"),
            # Validation only needs rewards and bounded projections. Keep exact
            # trace construction best-effort there so a terminal/guard boundary
            # cannot abort an entire Eval shard; training remains strict.
            strict=exact_trace_authority and not generation_only,
        )
        trace_model_calls = trace_bundle["model_calls"]
        turn_idx = 0

        nemo_rl_message_log = []
        physical_message_logs: list[list[dict[str, Any]]] = []
        current_physical_message_log: list[dict[str, Any]] | None = None
        seen_token_ids: List[int] = []
        batch_decode_items = []
        for output_item_dict in nemo_gym_result["response"]["output"]:
            # Nemo RL really only has two types of messages: assistant and not assistant since that is all that it is concerned with (i.e. to train or not to train)
            # Here we map all the trainable messages to assistant and all the non-trainable messages to user.
            # Eventually we can maybe be smarter about this, but this is functional for now.

            # Note that NeMo-Gym will only return token ids on "assistant" messages and not other message types.
            # Also skip if generation_token_ids is present but empty, e.g. all-EOS generation stripped to [] — torch.tensor([]) defaults to float32 and breaks batch dtype consistency.
            if (
                "generation_token_ids" not in output_item_dict
                or not output_item_dict["generation_token_ids"]
            ):
                continue

            trace_call = trace_model_calls[turn_idx]
            if trace_call["starts_physical_trace"]:
                current_physical_message_log = []
                physical_message_logs.append(current_physical_message_log)
                if turn_idx > 0 and (exact_trace_authority or generation_only):
                    # The trace planner has already independently verified the
                    # declared token/media rewrite for exact-authority
                    # responses. Generation-only collection also preserves
                    # inspectable rewrites, while legacy training remains
                    # fail-closed below.
                    seen_token_ids = []

            prompt_is_contiguous = (
                seen_token_ids
                == output_item_dict["prompt_token_ids"][: len(seen_token_ids)]
            )
            if not prompt_is_contiguous and not generation_only:
                raise AssertionError(
                    f"""Non-contiguous messages found! This may be a tokenization issue where certain tokens are combined when messages are concatenated, or it may be due to part of the chat history being truncated (like if super long history is truncated or if reasoning is stripped out).
Seen token IDs: {seen_token_ids}
Output prompt token IDs: {output_item_dict["prompt_token_ids"]}
output prompt token ids till seen: {output_item_dict["prompt_token_ids"][: len(seen_token_ids)]}
"""
                )

            if not trace_call["starts_physical_trace"] and not prompt_is_contiguous:
                raise AssertionError(
                    "Trace planner and NeMo-Gym postprocessor disagree about "
                    f"prefix continuity at turn {turn_idx + 1}"
                )

            prompt_token_ids = output_item_dict.pop("prompt_token_ids")
            generation_token_ids = output_item_dict.pop("generation_token_ids")
            generation_log_probs = output_item_dict.pop("generation_log_probs")
            routed_experts_raw = output_item_dict.pop("routed_experts", None)
            new_prompt_token_ids = prompt_token_ids[len(seen_token_ids) :]

            routed_experts = None
            if routed_experts_raw is not None:
                routed_experts_dtype = _ROUTED_EXPERTS_DTYPES[
                    self.cfg.get("routed_experts_dtype", "int16")
                ]
                routed_experts = decode_routed_experts(
                    routed_experts_raw, dtype=routed_experts_dtype
                )
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
            assert current_physical_message_log is not None
            current_physical_message_log.append(user_message)

            if processor is not None:
                if completion_evidence:
                    images_this_turn = _resolve_images_by_media_id(
                        media_assets,
                        trace_call["new_media_ids"],
                    )
                else:
                    images_this_turn = (
                        per_turn_images[turn_idx]
                        if turn_idx < len(per_turn_images)
                        else []
                    )
                _attach_multimodal_data_to_user_message(
                    user_message,
                    images=images_this_turn,
                    processor=processor,
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
            current_physical_message_log.append(assistant_message)

            seen_token_ids.extend(new_prompt_token_ids)
            seen_token_ids.extend(generation_token_ids)

            # We pop to remove larger tensors from logging.
            batch_decode_items.append(
                (output_item_dict, prompt_token_ids, generation_token_ids)
            )
            turn_idx += 1

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

        if exact_trace_authority:
            full_result = _build_exact_trace_full_result_projection(
                nemo_gym_result,
                trace_bundle=trace_bundle,
                generation_only=generation_only,
            )
        else:
            nemo_gym_result["nemo_rl_trace_bundle"] = trace_bundle
            full_result = nemo_gym_result
        return {
            "message_log": nemo_rl_message_log,
            "input_message_log": nemo_rl_message_log[:1],
            "physical_message_logs": physical_message_logs,
            "rollout_trace_bundle": trace_bundle,
            "full_result": full_result,
        }

    def shutdown(self) -> None:
        self.rh.shutdown()

    def step(self, message_log_batch, metadata):
        # This is not used since NeMo-Gym will handle the rollouts entirely.
        raise NotImplementedError

    def global_post_process_and_metrics(self, batch):
        # Similar to the step function, this is not used.
        raise NotImplementedError


def extract_reward_components(nemo_gym_result: dict) -> Dict[str, float] | None:
    """Return per-component rewards from a NeMo Gym verify result, or None.

    Single-reward NeMo Gym environments return only a scalar ``reward``. Multi-reward
    environments additionally return ``reward_components``: a mapping of
    component-name -> score. These are surfaced as ``reward/<name>`` batch keys and
    consumed by GDPO (see ``nemo_rl.algorithms.advantage_estimator.GDPOAdvantageEstimator``).

    Returns ``None`` when the environment is single-reward (no ``reward_components``),
    so callers fall back to the scalar ``reward`` path unchanged.
    """
    components = nemo_gym_result.get("reward_components")
    if not components:
        return None
    return {str(name): float(score) for name, score in components.items()}


def build_reward_component_columns(
    component_dicts: List[Dict[str, float] | None],
) -> Dict[str, torch.Tensor]:
    """Build ``reward/<name>`` batch columns from per-sample reward-component dicts.

    Takes the union of component names across the batch in sorted (deterministic) order
    and, for each, emits a ``reward/<name>`` tensor with one entry per sample. A
    component absent on a given sample is filled with ``0.0`` so every column covers all
    samples (the per-prompt baseline requires each component present for all responses).

    Keys are prefixed ``reward/`` so they are exactly what
    ``nemo_rl.algorithms.utils.get_gdpo_reward_component_keys`` selects (it matches
    ``startswith("reward/")`` and sorts by name); the name carries the component identity,
    so no positional index is needed. Returns an empty dict when no sample has components.
    """
    component_names = sorted(
        {name for c in component_dicts if c is not None for name in c}
    )
    return {
        f"reward/{name}": torch.tensor(
            [c[name] if c is not None and name in c else 0.0 for c in component_dicts]
        )
        for name in component_names
    }


def validate_reward_components_match_scalar(nemo_gym_results: List[dict]) -> None:
    """Assert each multi-reward result sets ``reward == sum(reward_components)``.

    A multi-reward verifier must set the scalar ``reward`` to the sum of its
    ``reward_components`` so single-reward (GRPO) consumers and GDPO read the same
    aggregate. We keep the verifier's scalar ``reward`` as ``total_reward`` rather than
    silently overwriting it with the component sum, so a verifier that violates this
    contract must be surfaced here instead of masked.

    Raises ``ValueError`` on the first violating result. A no-op for single-reward
    results (those without ``reward_components``).
    """
    for idx, result in enumerate(nemo_gym_results):
        components = extract_reward_components(result)
        if components is None:
            continue
        scalar_reward = float(result["reward"])
        component_sum = sum(components.values())
        if not math.isclose(scalar_reward, component_sum, rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError(
                f"NeMo Gym verify result {idx} has reward={scalar_reward} but its "
                f"reward_components sum to {component_sum} ({components}). A multi-reward "
                "verifier must set reward = sum(reward_components.values()) so single-reward "
                "(GRPO) consumers and GDPO read the same aggregate."
            )


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

    # For VLM runs, plumb the tokenizer config into the gym env config so the
    # NemoGym actor can reconstruct the processor inside itself (needed for
    # multi-turn multimodal postprocessing).
    if config.policy.get("is_vlm"):
        env_cfg = config.env.setdefault("nemo_gym", {})
        env_cfg.setdefault("tokenizer_config", dict(config.policy["tokenizer"]))


def spinup_nemo_gym_actor(
    env_configs: dict[str, Any],
    base_urls: list[Optional[str]],
    model_name: str,
    *,
    enable_router_replay: bool,
    routed_experts_dtype: str,
    use_fastokens: bool,
    context_compaction_runtime_contract: Optional[Dict[str, Any]] = None,
) -> Any:
    """Spin up the NeMo-Gym actor against the given generation server URLs.

    When env_configs["nemo_gym"]["num_gpu_nodes"] > 0, the actor is scheduled
    with soft NodeAffinity to the current Ray node so its colocated GPU
    resources land where the caller expects.

    Args:
        env_configs: The master_config.env mapping; env_configs["nemo_gym"] supplies
            the Gym global config plus NeMo-RL detection knobs (invalid_tool_call_patterns,
            thinking_tags, num_gpu_nodes).
        base_urls: Per-DP-rank OpenAI-compatible server base URLs from the generation backend.
        model_name: Served model name the Gym rollouts should target.
        enable_router_replay: Sets require_routed_experts on the NemoGymConfig.
        routed_experts_dtype: Dtype name for R3 routed_experts tensors ("int8"/"int16"/"int32"),
            resolved by the caller from the model's expert count.
        use_fastokens: Forwarded from policy.tokenizer.use_fastokens so the rollout actor
            patches its tokenizer consistently with the driver.
        context_compaction_runtime_contract: Launcher-owned generation identity
            bound to a synchronized policy version for exact-trace training.

    Returns:
        The spun-up NemoGym Ray actor handle (_spinup already awaited).
    """
    nemo_gym_dict = dict(env_configs["nemo_gym"])

    # NeMo-RL-side detection knobs are top-level NemoGymConfig fields
    # (where the detector reads them), not part of Gym's global config.
    invalid_tool_call_patterns = nemo_gym_dict.pop("invalid_tool_call_patterns", None)
    thinking_tags = nemo_gym_dict.pop("thinking_tags", None)
    tokenizer_config = nemo_gym_dict.pop("tokenizer_config", None)

    # Pass prebuilt cache + venv dirs through the global config so the gym reuses
    # image-baked venvs instead of rebuilding them.
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
        tokenizer_config=tokenizer_config,
        require_routed_experts=enable_router_replay,
        routed_experts_dtype=routed_experts_dtype,
        use_fastokens=use_fastokens,
        context_compaction_runtime_contract=context_compaction_runtime_contract,
        initial_global_config_dict=nemo_gym_dict,
    )

    nemo_gym_py_exec = get_actor_python_env("nemo_rl.environments.nemo_gym.NemoGym")
    if nemo_gym_py_exec.startswith("uv"):
        nemo_gym_py_exec = create_local_venv_on_each_node(
            nemo_gym_py_exec, "nemo_rl.environments.nemo_gym.NemoGym"
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
