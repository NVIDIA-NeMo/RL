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
import base64
import copy
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import ray
import requests
import torch
from PIL import Image
from transformers import PreTrainedTokenizerBase

from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data.multimodal_utils import (
    PackedTensor,
    get_dim_to_pack_along,
    get_multimodal_keys_from_processor,
)
from nemo_rl.distributed.virtual_cluster import _get_free_port_local, _get_node_ip_local
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.utils.timer import Timer


def resolve_to_image(image_path_or_image: str | Image.Image) -> Image.Image:
    """Resolve the image path to a PIL.Image object.

    image_path can be either:
    - path to local file
    - url to image
    - base64 encoded image
    """
    if isinstance(image_path_or_image, Image.Image):
        return image_path_or_image

    if image_path_or_image.startswith(("http://", "https://")):
        response = requests.get(image_path_or_image)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    elif image_path_or_image.startswith("data:"):
        header, encoded = image_path_or_image.split(",", 1)
        image_data = base64.b64decode(encoded)
        return Image.open(io.BytesIO(image_data)).convert("RGB")
    else:
        return Image.open(image_path_or_image).convert("RGB")


def image_to_data_url(image: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL Image as a base64 data URL."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = f"image/{fmt.lower()}"
    return f"data:{mime};base64,{encoded}"


def encode_images_in_examples(nemo_gym_examples: list[dict]) -> list[dict]:
    """Walk examples and replace local image paths with base64 data URLs.

    Operates in-place on each example's
    responses_create_params.input[].content[] items of type 'input_image'.
    """
    for example in nemo_gym_examples:
        input_items = (
            example.get("responses_create_params", {}).get("input", [])
        )
        for item in input_items:
            for part in item.get("content", []):
                if not isinstance(part, dict) or part.get("type") != "input_image":
                    continue
                url = part.get("image_url", "")
                if url.startswith(("http://", "https://", "data:", "file://")):
                    continue
                # Local filesystem path — encode as data URL
                pil_image = resolve_to_image(url)
                part["image_url"] = image_to_data_url(pil_image)
    return nemo_gym_examples


def _strip_image_placeholders(text: str) -> str:
    for token in ("<image>", "<img>", "</img>"):
        text = text.replace(token, "")
    return text


def _example_has_image_payload(nemo_gym_example: dict) -> bool:
    input_items = nemo_gym_example.get("responses_create_params", {}).get("input", [])
    for item in input_items:
        content = item.get("content", "")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type in ("input_image", "image", "image_url"):
                return True
            if part.get("image_url") is not None or part.get("image") is not None:
                return True
    return False


def sanitize_text_only_nemo_gym_example(nemo_gym_example: dict) -> dict:
    """Remove image placeholder text markers only for text-only rows.

    This prevents placeholder tokens from entering vLLM prompts when a row has no
    actual image payload.
    """
    if _example_has_image_payload(nemo_gym_example):
        return nemo_gym_example

    sanitized = copy.deepcopy(nemo_gym_example)
    input_items = sanitized.get("responses_create_params", {}).get("input", [])
    for item in input_items:
        content = item.get("content", "")
        if isinstance(content, str):
            item["content"] = _strip_image_placeholders(content)
            continue
        if not isinstance(content, list):
            continue
        for idx, part in enumerate(content):
            if isinstance(part, str):
                content[idx] = _strip_image_placeholders(part)
                continue
            if not isinstance(part, dict):
                continue
            text_value = part.get("text")
            if isinstance(text_value, str):
                part["text"] = _strip_image_placeholders(text_value)
    return sanitized


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

        # Get Ray head node address if Ray is initialized
        assert ray.is_initialized(), (
            "Ray must be initialized before using NeMo-Gym environment"
        )
        ray_context = ray.get_runtime_context()
        assert ray_context.gcs_address, "Ray must have a GCS address"

        initial_global_config_dict["ray_head_node_address"] = ray_context.gcs_address

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
        original_message_logs: list[list[dict]] = None,
    ) -> list[dict]:
        timer = Timer()

        nemo_gym_num_rows = len(nemo_gym_examples)
        encode_images_in_examples(nemo_gym_examples)
        nemo_gym_result_iterator = self.rch.run_examples(
            examples=nemo_gym_examples, head_server_config=self.head_server_config
        )

        timer.start("_run_rollouts_total")
        nemo_rl_rowidxs = []
        nemo_rl_results = []
        for task in nemo_gym_result_iterator:
            with timer.time(label=f"{timer_prefix}/await_results"):
                nemo_gym_row, nemo_gym_result = await task

            with timer.time(label=f"{timer_prefix}/postprocess_results"):
                # Get original message_log with pixel_values for this row
                rowidx = nemo_gym_row["_rowidx"]
                original_message_log = original_message_logs[rowidx] if original_message_logs else None

                nemo_rl_result = self._postprocess_nemo_gym_to_nemo_rl_result(
                    nemo_gym_result, tokenizer, original_message_log
                )

            nemo_rl_rowidxs.append(rowidx)
            nemo_rl_results.append(nemo_rl_result)

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
        self, nemo_gym_result: dict, tokenizer: PreTrainedTokenizerBase, original_message_log: list[dict] = None
    ) -> dict:
        nemo_rl_message_log = []
        seen_token_ids: List[int] = []
        response_metadata = nemo_gym_result["response"].get("metadata") or {}
        context_length_exceeded = (
            response_metadata.get("context_length_exceeded") == "true"
        )

        # Extract multimodal data (pixel_values, imgs_sizes, etc.) from the original message_log.
        # The original message_log was created by the HF processor and contains pixel_values
        # that are not available in the vLLM nemo_gym response.
        multimodal_data = {}
        if original_message_log:
            for msg in original_message_log:
                if msg["role"] == "user":
                    for key in list(msg.keys()):
                        if key not in ["role", "content", "token_ids"]:
                            multimodal_data[key] = msg[key]

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

            prompt_tokens_original = output_item_dict["prompt_token_ids"][len(seen_token_ids):]

            if original_message_log and multimodal_data and not nemo_rl_message_log:
                # First turn, multimodal: use HF processor token_ids
                hf_token_ids = torch.cat([msg["token_ids"] for msg in original_message_log], dim=0)
                user_token_ids = hf_token_ids
            else:
                user_token_ids = torch.tensor(prompt_tokens_original, dtype=torch.long)

            user_message = {
                "role": "user",
                "content": tokenizer.decode(user_token_ids.tolist()),
                "token_ids": user_token_ids,
            }
            # Add pixel_values and other multimodal data from the original message_log.
            # Only on the first turn: subsequent turns use vLLM prompt_tokens_original which
            # don't contain the <img><image>×N</img> wrapper tokens, so imgs_sizes/pixel_values
            # must not be added or they'd mismatch the <image> token count in input_ids.
            if not nemo_rl_message_log:
                user_message.update(multimodal_data)

            nemo_rl_message_log.append(user_message)
            assistant_token_ids_raw = torch.tensor(
                output_item_dict["generation_token_ids"], dtype=torch.long
            )
            assistant_logprobs = torch.tensor(
                output_item_dict["generation_log_probs"], dtype=torch.float32
            )

            # Prompt overflow is represented as a prompt-only user turn with no
            # assistant tokens, rather than an empty assistant completion.
            if context_length_exceeded and assistant_token_ids_raw.numel() == 0:
                seen_token_ids.extend(prompt_tokens_original)
                # `seen_token_ids` now equals the full rendered prompt that
                # triggered the overflow (prior turns + current turn delta).
                full_prompt_token_ids = list(seen_token_ids)
                output_item_dict["prompt_str"] = tokenizer.decode(
                    output_item_dict.pop("prompt_token_ids")
                )
                output_item_dict["generation_str"] = ""
                output_item_dict.pop("generation_token_ids")
                output_item_dict.pop("generation_log_probs")

                # Multi-turn rollouts accumulate tokens across prior turns, so
                # the cumulative message_log (not just this turn's delta) can
                # exceed policy.max_total_sequence_length once the agent loops
                # enough times. Replace the entire message_log with a single
                # user turn truncated to policy_max_input_tokens so downstream
                # sequence packing accepts the sample. The full prompt is
                # preserved in `prompt_str` for logging, and the sample is
                # excluded from loss/reward via loss_multiplier=0 (set in
                # run_async_nemo_gym_rollout) and the reward valid mask.
                max_input_tokens = self.cfg.get(
                    "initial_global_config_dict", {}
                ).get("policy_max_input_tokens")
                if max_input_tokens is not None:
                    truncated_token_ids = torch.tensor(
                        full_prompt_token_ids[:max_input_tokens], dtype=torch.long
                    )
                    nemo_rl_message_log[:] = [
                        {
                            "role": "user",
                            "content": tokenizer.decode(
                                truncated_token_ids.tolist()
                            ),
                            "token_ids": truncated_token_ids,
                        }
                    ]
                continue

            assistant_token_ids = assistant_token_ids_raw
            nemo_rl_message_log.append(
                {
                    "role": "assistant",
                    "content": tokenizer.decode(assistant_token_ids.tolist()),
                    "token_ids": assistant_token_ids,
                    "generation_logprobs": assistant_logprobs,
                }
            )

            # Track seen tokens using the ORIGINAL vLLM ids (without 21/22 wrappers) so that
            # the prefix assertion remains valid for multi-turn conversations.
            seen_token_ids.extend(prompt_tokens_original)
            seen_token_ids.extend(assistant_token_ids_raw.tolist())

            # We pop to remove larger tensors from logging.
            output_item_dict["prompt_str"] = tokenizer.decode(
                output_item_dict.pop("prompt_token_ids")
            )
            output_item_dict["generation_str"] = tokenizer.decode(
                output_item_dict.pop("generation_token_ids")
            )
            output_item_dict.pop("generation_log_probs")

        input_message_log = nemo_rl_message_log[:1]
        if (
            context_length_exceeded
            and nemo_rl_message_log
            and nemo_rl_message_log[-1]["role"] == "user"
        ):
            input_message_log = [nemo_rl_message_log[-1]]

        return {
            "message_log": nemo_rl_message_log,
            "input_message_log": input_message_log,
            "full_result": nemo_gym_result,
            "context_length_exceeded": context_length_exceeded,
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

    max_input_tokens = generation_config["vllm_cfg"].get("max_model_len")
    if max_input_tokens is not None:
        config.setdefault("env", {}).setdefault("nemo_gym", {})[
            "policy_max_input_tokens"
        ] = max_input_tokens


########################################
# Data utils
########################################


# We do some light preprocessing here to make our data format compatible with nemo rl format
def nemo_gym_example_to_nemo_rl_datum_spec(
    nemo_gym_example: dict, idx: int, processor: Optional[Any] = None
) -> DatumSpec:
    nemo_gym_example = sanitize_text_only_nemo_gym_example(nemo_gym_example)

    if processor is None:
        return DatumSpec(
            message_log=[
                {"role": "user", "content": "", "token_ids": torch.tensor([])}
            ],  # Fake message
            length=0,
            extra_env_info=nemo_gym_example,
            loss_multiplier=1.0,
            idx=idx,
            task_name="nemo_gym",
            stop_strings=None,
            token_ids=[],
        )

    # Extract messages from nemo_gym format
    input_messages = nemo_gym_example.get("responses_create_params", {}).get("input", [])

    # Build user message only (no system message — NeMo-Gym sends empty system to vLLM)
    user_message = {"role": "user", "content": []}

    for msg in input_messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            if isinstance(content, str):
                user_message["content"].append({"type": "text", "text": content})
            elif isinstance(content, list):
                user_message["content"] = content

    # Build user_message with PIL.Image objects for the HF processor
    user_message_with_images = {"role": "user", "content": []}
    for item in user_message["content"]:
        if not isinstance(item, dict):
            # Defensive fallback for legacy plain-string content entries.
            if isinstance(item, str):
                user_message_with_images["content"].append({"type": "text", "text": item})
            continue

        if item.get("type") in ("input_image", "image_url"):
            url = item.get("image_url", "")
            if isinstance(url, dict):
                url = url.get("url", "")
            if url:
                pil_image = resolve_to_image(url)
                user_message_with_images["content"].append({"type": "image", "image": pil_image})
        elif item.get("type") in ("input_text", "text"):
            user_message_with_images["content"].append({"type": "text", "text": item.get("text", "")})
        else:
            user_message_with_images["content"].append(item.copy())

    # Process user message with images
    message_both = processor.apply_chat_template(
        [user_message_with_images],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    # Keep the full HF processor token layout (with <img>/<image>×N/</img>) so that
    # collapse_multimodal_tokens() in Megatron can identify image regions and preserve
    # pixel_values for proper image embedding. Stripping wrappers here causes Megatron
    # to drop pixel_values and either crash or compute logprobs without image context.
    user_message["token_ids"] = message_both["input_ids"][0]

    # Extract multimodal keys (pixel_values, etc.)
    multimodal_keys = get_multimodal_keys_from_processor(processor)
    for key in multimodal_keys:
        if key in message_both:
            user_message[key] = PackedTensor(
                message_both[key],
                dim_to_pack=get_dim_to_pack_along(processor, key)
            )

    if "imgs_sizes" in message_both:
        user_message["imgs_sizes"] = PackedTensor(message_both["imgs_sizes"], dim_to_pack=0)

    if "token_type_ids" in message_both:
        user_message["token_type_ids"] = message_both["token_type_ids"][0]

    message_log = [user_message]
    length = sum(len(m["token_ids"]) for m in message_log)

    return DatumSpec(
        message_log=message_log,
        length=length,
        extra_env_info=nemo_gym_example,
        loss_multiplier=1.0,
        idx=idx,
        task_name="nemo_gym",
        stop_strings=None,
        token_ids=[],
    )
