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

import argparse
import base64
import os
import pprint
from collections import defaultdict
from io import BytesIO
from typing import Any, Optional

import requests
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoProcessor

from nemo_rl.algorithms.mpo import MasterConfig, mpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.datasets.response_datasets.mmpr import (
    MMPRDataset,
    format_mmpr_dataset,
)
from nemo_rl.data.interfaces import (
    DatumSpec,
    LLMMessageLogType,
    TaskDataProcessFnCallable,
    TaskDataSpec,
)
from nemo_rl.data.llm_message_utils import (
    get_formatted_message_log,
    strip_image_tokens_from_text,
)
from nemo_rl.data.multimodal_utils import (
    PackedTensor,
    get_dim_to_pack_along,
    get_multimodal_keys_from_processor,
)
from nemo_rl.distributed.ray_actor_environment_registry import (
    get_actor_python_env,
)
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.environments.vlm_environment import VLMEnvironment
from nemo_rl.models import nemotron_h_nano_vl
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir

OmegaConf.register_new_resolver("mul", lambda a, b: a * b)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run VLM DPO training with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    # Parse known args for the script
    args, overrides = parser.parse_known_args()
    return args, overrides


# ===============================================================================
#                             VLM DPO Data Processor
# ===============================================================================


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
        # Handle URL
        response = requests.get(image_path_or_image)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    elif image_path_or_image.startswith("data:"):
        # Handle base64 encoded image
        # Format: data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwA/vAAEEAQMCBAIGBgYFBwkICgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwA/v
        header, encoded = image_path_or_image.split(",", 1)
        image_data = base64.b64decode(encoded)
        return Image.open(BytesIO(image_data)).convert("RGB")
    else:
        # Handle local file path
        return Image.open(image_path_or_image).convert("RGB")


def process_multimodal_message(
    message: list[dict],
    task_data_spec: TaskDataSpec,
    processor: AutoProcessor,
) -> tuple[dict, list[Image.Image]]:
    """Process a multimodal message for DPO training.

    Args:
        message: Message dictionary containing multimodal content
        task_data_spec: Task specification
        processor: AutoProcessor for tokenization

    Returns:
        Tuple of (processed_message, images)
    """
    images = []
    content = message["content"]
    # Handle multimodal content
    if isinstance(content, list):
        processed_content = []
        for item in content:
            if item["type"] == "image":
                images.append(resolve_to_image(item["image"]))
                processed_content.append({"type": "image", "image": resolve_to_image(item["image"])})
            elif item["type"] == "text":
                text = strip_image_tokens_from_text(item["text"])
                processed_content.append({
                    "type": "text",
                    "text": task_data_spec.prompt.format(text)
                    if task_data_spec.prompt
                    else text,
                })
            else:
                processed_content.append(item)
    else:
        # Text-only content
        text = strip_image_tokens_from_text(content)
        processed_content = (
            task_data_spec.prompt.format(text) if task_data_spec.prompt else text
        )

    message["content"] = processed_content
    return message, images


def vlm_mpo_preprocessor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    processor: AutoProcessor,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process a datum dictionary for VLM DPO training.

    Expected format:
        >>> # context can also contain multiple turns
        >>> datum = {
        ...     "context": [{"role": "user", "content": "I have a question."}, {"role": "assistant", "content": "Sure!"}, {"role": "user", "content": "What is 2+2?"}],
        ...     "completions": [
        ...         {"rank": 0, "completion": [{"role": "assistant", "content": "4"}]},
        ...         {"rank": 1, "completion": [{"role": "assistant", "content": "5"}]}
        ...     ]
        ... }
    """
    # Format the data based on task type
    if task_data_spec.task_name == "mmpr":
        datum_dict = format_mmpr_dataset(datum_dict)
    else:
        raise ValueError(f"No data processor for task {task_data_spec.task_name}")

    assert len(datum_dict["completions"]) == 2, (
        "MPO training supports only two completions"
    )
    # Lower rank is preferred
    if datum_dict["completions"][0]["rank"] < datum_dict["completions"][1]["rank"]:
        chosen_completion = datum_dict["completions"][0]
        rejected_completion = datum_dict["completions"][1]
    elif datum_dict["completions"][0]["rank"] > datum_dict["completions"][1]["rank"]:
        chosen_completion = datum_dict["completions"][1]
        rejected_completion = datum_dict["completions"][0]
    else:
        raise NotImplementedError(
            "Ties are not supported yet. You can use the following command to filter out ties: `cat <LocalPathToPreferenceDataset> | jq 'select(.completions[0].rank != .completions[1].rank)'`."
        )
    context_messages = datum_dict["context"]
    messages_chosen = context_messages + chosen_completion["completion"]
    messages_rejected = context_messages + rejected_completion["completion"]
    
    # Process multimodal content in context
    processed_context = []
    all_images = []
    for msg in context_messages:
        processed_msg, images = process_multimodal_message(msg, task_data_spec, processor)
        processed_context.append(processed_msg)
        all_images.extend(images)

    # Process chosen completion
    processed_chosen = []
    for msg in chosen_completion["completion"]:
        processed_msg, _ = process_multimodal_message(
            msg, task_data_spec, processor
        )
        processed_chosen.append(processed_msg)

    # Process rejected completion
    processed_rejected = []
    for msg in rejected_completion["completion"]:
        processed_msg, _ = process_multimodal_message(
            msg, task_data_spec, processor
        )
        processed_rejected.append(processed_msg)

    # Create message logs
    messages_chosen = processed_context + processed_chosen
    messages_rejected = processed_context + processed_rejected

    # Format messages using chat template
    message_log_chosen = get_formatted_message_log(
        messages_chosen, processor, task_data_spec
    )
    message_log_rejected = get_formatted_message_log(
        messages_rejected, processor, task_data_spec
    )

    # Calculate lengths
    length_chosen = sum(len(m["token_ids"]) for m in message_log_chosen)
    length_rejected = sum(len(m["token_ids"]) for m in message_log_rejected)

    # Discard overlong samples by zeroing the loss multiplier. Token stubs and
    # empty PackedTensors are kept as structural placeholders for batching.
    loss_multiplier = 1.0
    if max(length_chosen, length_rejected) > max_seq_length:
        print(f"Discarding overlong sample: chosen={length_chosen}, rejected={length_rejected}, max={max_seq_length}")
        tokenizer = processor.tokenizer
        image_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in ("<img>", "<image>", "</img>")]
        for message in message_log_chosen:
            token_ids = message["token_ids"][
                : min(4, max_seq_length // len(message_log_chosen))
            ]
            # Filter out all image tokens (<img>, <image>, </img>) since we're discarding images
            for img_token_id in image_token_ids:
                token_ids = token_ids[token_ids != img_token_id]
            message["token_ids"] = token_ids
            for key, value in message.items():
                if isinstance(value, PackedTensor):
                    message[key] = PackedTensor.empty_like(value)
        for message in message_log_rejected:
            token_ids = message["token_ids"][
                : min(4, max_seq_length // len(message_log_rejected))
            ]
            # Filter out all image tokens (<img>, <image>, </img>) since we're discarding images
            for img_token_id in image_token_ids:
                token_ids = token_ids[token_ids != img_token_id]
            message["token_ids"] = token_ids
            for key, value in message.items():
                if isinstance(value, PackedTensor):
                    message[key] = PackedTensor.empty_like(value)
        loss_multiplier = 0.0
        length_chosen = sum(len(m["token_ids"]) for m in message_log_chosen)
        length_rejected = sum(len(m["token_ids"]) for m in message_log_rejected)

    output = {
        "message_log_chosen": message_log_chosen,
        "length_chosen": length_chosen,
        "message_log_rejected": message_log_rejected,
        "length_rejected": length_rejected,
        "extra_env_info": None,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
        "task_name": task_data_spec.task_name,
    }
    return output


def setup_data(
    processor: AutoProcessor,
    data_config: DataConfig,
) -> tuple[
    AllTaskProcessedDataset,
    Optional[AllTaskProcessedDataset]
]:
    """This function will create a TaskSpec, DatumSpec, and connect the two for VLM DPO.

    task_spec contains the task name as well as prompt and system prompt modifiers that can be used by data processor
    """
    print("\n▶ Setting up VLM DPO data...")

    # Load appropriate VLM dataset
    if data_config["dataset_name"] == "mmpr":
        data: Any = MMPRDataset(
            data_path=data_config["data_path"],
            split=data_config["split"]
        )
    else:
        raise ValueError(f"No processor for VLM dataset {data_config['dataset_name']}.")

    mpo_task_spec = data.task_name
    vlm_task_spec = TaskDataSpec(
        task_name=mpo_task_spec,
        prompt_file=data_config["prompt_file"],
        system_prompt_file=data_config["system_prompt_file"],
    )

    # Create datasets
    train_dataset = AllTaskProcessedDataset(
        data.formatted_ds["train"],
        processor,
        vlm_task_spec,
        vlm_mpo_preprocessor,
        max_seq_length=data_config["max_input_seq_length"],
    )

    val_dataset: Optional[AllTaskProcessedDataset] = None
    if data.formatted_ds.get("validation"):
        val_dataset = AllTaskProcessedDataset(
            data.formatted_ds["validation"],
            processor,
            vlm_task_spec,
            vlm_mpo_preprocessor,
            max_seq_length=data_config["max_input_seq_length"],
        )
    if not isinstance(val_dataset, dict):
        val_dataset = {} if val_dataset is None else {"default": val_dataset}
    # Set up task-to-environment mapping
    # task_to_env: dict[str, EnvironmentInterface] = defaultdict(lambda: vlm_env)
    # task_to_env[task_name] = vlm_env

    return train_dataset, val_dataset


def main() -> None:
    """Main entry point for VLM DPO training."""
    nemotron_h_nano_vl.register()

    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "vlm_mpo_mmpr.yaml"
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    # Print config
    print("Final config:")
    pprint.pprint(config)

    # Get the next experiment directory with incremented ID
    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"📊 Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        print(
            f"📊 Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}"
        )

    init_ray()

    # Initialize processor for multimodal processing
    processor = get_tokenizer(config["policy"]["tokenizer"], get_processor=True)
    tokenizer = processor.tokenizer

    # Configure generation if specified
    if config["policy"].get("generation"):
        config["policy"]["generation"] = configure_generation_config(
            config["policy"]["generation"], processor.tokenizer
        )

    # Setup data with multimodal processing
    (
        train_dataset,
        val_dataset,
    ) = setup_data(processor, config["data"])

    # Setup DPO training
    (
        policy,
        cluster,
        train_dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        mpo_save_state,
        master_config,
    ) = setup(config, tokenizer, train_dataset, val_dataset)

    # Run DPO training
    mpo_train(
        policy,
        train_dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        master_config,
        logger,
        checkpointer,
        mpo_save_state,
    )


if __name__ == "__main__":
    main()
