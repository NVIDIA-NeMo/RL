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
import torch
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoProcessor

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset, load_response_dataset
from nemo_rl.data.datasets.response_datasets.clevr import format_clevr_cogent_dataset
from nemo_rl.data.datasets.response_datasets.geometry3k import format_geometry3k_dataset
from nemo_rl.data.datasets.response_datasets.refcoco import format_refcoco_dataset
from nemo_rl.data.interfaces import (
    DatumSpec,
    LLMMessageLogType,
    TaskDataProcessFnCallable,
    TaskDataSpec,
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
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir

OmegaConf.register_new_resolver("mul", lambda a, b: a * b)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GRPO training with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    # Parse known args for the script
    args, overrides = parser.parse_known_args()
    return args, overrides


# ===============================================================================
#                             VLM Data Processor
# ===============================================================================


def adjust_image_tokens(
    input_ids,
    num_tiles,
    img_start_token_id,
    img_end_token_id,
):
    """Ensures the input_ids tensor contains the correct number of <image> tokens as specified by num_tiles.

    This adjustment is necessary to bridge the gap between from HF processor to Megatron LLaVAModel.

    Example:
        input_ids decoded may look like this
        System: ...
        User:...
        Image 1: <img><image>...<image></img>  # adjust number of <image> tokens to be num_tiles[0]
        Image 2: <img><image>...<image></img>  # adjust number of <image> tokens to be num_tiles[1]
        ...
        etc
    Args:
        input_ids: The input_ids tensor (output of HF processor)
            or a dictionary of tensors, one of the keys of which must be "input_ids",
            and other tensors must have the same shape as input_ids
        num_tiles: The number of <image> tokens to ensure, either a single int or a list of ints
        img_start_token_id: The token id of <img>
        img_end_token_id: The token id of </img>
    Returns:
        The input_ids tensor with the correct number of <image> tokens
        or a dictionary of tensors each with the same shape as input_ids
    """
    if isinstance(num_tiles, int):
        num_tiles = [num_tiles]
    if isinstance(input_ids, dict):
        assert "input_ids" in input_ids, (
            "input_ids must be a dictionary with 'input_ids' as one of the keys"
        )
        other_tensors = {
            key: value for key, value in input_ids.items() if key != "input_ids"
        }
        input_ids = input_ids["input_ids"]
    else:
        other_tensors = None

    for i, num_tile in enumerate(num_tiles):
        image_start_pos = (
            (input_ids[0] == img_start_token_id).nonzero(as_tuple=True)[0][i].item()
        )
        image_end_pos = (
            (input_ids[0] == img_end_token_id).nonzero(as_tuple=True)[0][i].item()
        )
        media_token_id = input_ids[
            0, image_start_pos + 1
        ]  # this can be <image> or <video> token
        existing = image_end_pos - image_start_pos + 1

        if num_tile > existing:
            # Need to add tokens
            repeat = num_tile + 2 - existing  # +2 for <img> and </img> tokens
            repeat_tokens = torch.full(
                (1, repeat),
                media_token_id,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            if other_tensors is not None:
                for key, tensor in other_tensors.items():
                    assert other_tensors[key].shape == input_ids.shape, (
                        f"Tensor {key} has shape {other_tensors[key].shape} but input_ids has shape {input_ids.shape}"
                    )
                    other_tensors[key] = torch.cat(
                        [
                            tensor[:, : image_start_pos + 1],
                            repeat_tokens,
                            tensor[:, image_start_pos + 1 :],
                        ],
                        dim=1,
                    )
            input_ids = torch.cat(
                [
                    input_ids[:, : image_start_pos + 1],
                    repeat_tokens,
                    input_ids[:, image_start_pos + 1 :],
                ],
                dim=1,
            )

        elif num_tile < existing:
            # Need to remove tokens (keep only the first `num_tile` occurrences)
            keep_tokens_mask = torch.ones_like(input_ids, dtype=torch.bool)
            positions = (
                input_ids[0][image_start_pos : image_end_pos + 1] == media_token_id
            ).nonzero(as_tuple=True)[0] + image_start_pos
            # positions to drop are after the first num_tile occurrences
            drop_positions = positions[num_tile:].tolist()
            keep_tokens_mask[0, drop_positions] = False
            if other_tensors is not None:
                for key, tensor in other_tensors.items():
                    assert other_tensors[key].shape == input_ids.shape, (
                        f"Tensor {key} has shape {other_tensors[key].shape} but input_ids has shape {input_ids.shape}"
                    )
                    other_tensors[key] = tensor[keep_tokens_mask].unsqueeze(0)
            input_ids = input_ids[keep_tokens_mask].unsqueeze(0)

    if other_tensors is not None:
        return {
            "input_ids": input_ids,
            **other_tensors,
        }
    else:
        return input_ids


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
        # Format: data:image/jpeg;base64,/9j/4AAQSkZJRg...
        header, encoded = image_path_or_image.split(",", 1)
        image_data = base64.b64decode(encoded)
        return Image.open(BytesIO(image_data)).convert("RGB")
    else:
        # Handle local file path
        return Image.open(image_path_or_image).convert("RGB")


def hf_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    processor: AutoProcessor,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process a datum dictionary (directly loaded from response_datasets/<dataset_name>.py) into a DatumSpec for the VLM Environment."""
    # depending on the task, format the data differently
    if task_data_spec.task_name == "clevr-cogent":
        datum_dict = format_clevr_cogent_dataset(datum_dict)
    elif task_data_spec.task_name == "refcoco":
        datum_dict = format_refcoco_dataset(datum_dict)
    elif task_data_spec.task_name == "geometry3k":
        datum_dict = format_geometry3k_dataset(datum_dict)
    else:
        raise ValueError(f"No data processor for task {task_data_spec.task_name}")

    user_message = datum_dict["messages"]
    problem = user_message[0]["content"]
    extra_env_info = {"ground_truth": user_message[1]["content"]}

    message_log: LLMMessageLogType = []
    ### only one round of interaction is assumed, this can easily be extended to a conversational setting
    user_message = {"role": "user", "content": []}
    #
    images = []
    if isinstance(problem, list):
        for content in problem:
            # for image, video, just append it
            # for text, format the prompt to the problem
            if content["type"] != "text":
                user_message["content"].append(content)
                if content["type"] == "image":
                    images.append(content["image"])
                else:
                    raise ValueError(f"Unsupported content type: {content['type']}")
            elif content["type"] == "text":
                user_message["content"].append(
                    {
                        "type": "text",
                        "text": task_data_spec.prompt.format(content["text"])
                        if task_data_spec.prompt
                        else content["text"],
                    }
                )
    else:
        # conversation consists of a text-only message
        user_message["content"] = task_data_spec.prompt.format(problem)

    images = [resolve_to_image(image) for image in images]

    # get formatted user message
    if hasattr(processor, "conversation_preprocessor"):
        user_message_for_chat_template = processor.conversation_preprocessor(
            user_message
        )
    else:
        user_message_for_chat_template = user_message

    # this is the string-tokenized conversation template for the generation policy (for vllm)
    string_formatted_dialog = processor.apply_chat_template(
        [user_message_for_chat_template],
        tokenize=False,
        add_generation_prompt=True,
    )

    # this is the id-tokenized and image processed conversation template for the policy
    message: dict = processor.apply_chat_template(
        [user_message],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    # import pdb
    # pdb.set_trace()

    # add this for backward compatibility
    user_message["token_ids"] = adjust_image_tokens(
        message["input_ids"], message["num_patches"], 131073, 131074
    )[0]
    # add all keys and values to the user message, and the list of keys
    multimodal_keys = get_multimodal_keys_from_processor(processor)
    for key in multimodal_keys:
        if key in message:
            user_message[key] = PackedTensor(
                message[key], dim_to_pack=get_dim_to_pack_along(processor, key)
            )

    # specifically for gemma, we need to add token_type_ids to the user message as a sequence-type value
    if "token_type_ids" in message:
        user_message["token_type_ids"] = message["token_type_ids"][0]

    ### append to user message
    message_log.append(user_message)

    length = sum(len(m["token_ids"]) for m in message_log)
    loss_multiplier = 1.0
    if length >= max_seq_length:
        # Treat truncated messages as text only
        vllm_kwargs = {
            "vllm_content": None,
            "vllm_images": [],
        }

        # make smaller and mask out
        for chat_message in message_log:
            chat_message["token_ids"] = chat_message["token_ids"][
                : min(4, max_seq_length // len(message_log))
            ]
            for key, value in chat_message.items():
                if isinstance(value, PackedTensor):
                    chat_message[key] = PackedTensor.empty_like(value)
        loss_multiplier = 0.0
    else:
        # get the prompt content! (use this for vllm-backend that needs formatted dialog and list of images) for the entire conversation
        # add images for vllm serving
        vllm_kwargs = {
            "vllm_content": string_formatted_dialog,
            "vllm_images": images,
        }

    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
        "task_name": task_data_spec.task_name,
        **vllm_kwargs,
    }
    return output


def setup_data(
    processor: AutoProcessor,
    data_config: DataConfig,
    env_configs: dict[str, Any],
    seed: int,
) -> tuple[
    AllTaskProcessedDataset,
    Optional[AllTaskProcessedDataset],
    dict[str, EnvironmentInterface],
    dict[str, EnvironmentInterface],
]:
    """This function will create a TaskSpec, DatumSpec, and connect the two.

    task_spec contains the task name as well as prompt and system prompt modifiers that can be used by data processor
    """
    print("\n▶ Setting up data...")

    # load dataset
    # TODO @yukih: currently seed is not used for vlm datasets
    data: Any = load_response_dataset(data_config, seed)

    task_name = data.task_name
    vlm_task_spec = TaskDataSpec(
        task_name=task_name,
        prompt_file=data_config["prompt_file"],
        system_prompt_file=data_config["system_prompt_file"],
    )

    # add data processor for different tasks
    task_data_processors: dict[str, tuple[TaskDataSpec, TaskDataProcessFnCallable]] = (
        defaultdict(lambda: (vlm_task_spec, hf_data_processor))
    )
    task_data_processors[task_name] = (vlm_task_spec, hf_data_processor)

    env_name = data_config["env_name"]
    vlm_env = VLMEnvironment.options(  # type: ignore # it's wrapped with ray.remote
        runtime_env={
            "py_executable": get_actor_python_env(
                "nemo_rl.environments.vlm_environment.VLMEnvironment"
            ),
            "env_vars": dict(os.environ),  # Pass thru all user environment variables
        }
    ).remote(env_configs[env_name])

    dataset = AllTaskProcessedDataset(
        data.formatted_ds["train"],
        processor,
        vlm_task_spec,
        task_data_processors,
        max_seq_length=data_config["max_input_seq_length"],
    )

    val_dataset: Optional[AllTaskProcessedDataset] = None
    if data.formatted_ds["validation"]:
        val_dataset = AllTaskProcessedDataset(
            data.formatted_ds["validation"],
            processor,
            vlm_task_spec,
            task_data_processors,
            max_seq_length=data_config["max_input_seq_length"],
        )
    else:
        val_dataset = None

    task_to_env: dict[str, EnvironmentInterface] = defaultdict(lambda: vlm_env)
    task_to_env[task_name] = vlm_env
    return dataset, val_dataset, task_to_env, task_to_env


def main() -> None:
    """Main entry point."""
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "vlm_grpo_3B.yaml"
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

    # init processor
    processor = get_tokenizer(config["policy"]["tokenizer"], get_processor=True)
    tokenizer = processor.tokenizer

    assert config["policy"]["generation"] is not None, (
        "A generation config is required for GRPO"
    )
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], processor.tokenizer
    )
    if "vllm_cfg" in config["policy"]["generation"]:
        assert (
            config["policy"]["generation"]["vllm_cfg"]["skip_tokenizer_init"] == False
        ), (
            "VLMs require tokenizer to be initialized before generation, so skip_tokenizer_init must be set to False."
        )

    # setup data
    # this function is local to this script, and can be extended to other VLM datasets
    (
        dataset,
        val_dataset,
        task_to_env,
        val_task_to_env,
    ) = setup_data(processor, config["data"], config["env"], config["grpo"]["seed"])

    (
        policy,
        policy_generation,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    ) = setup(config, tokenizer, dataset, val_dataset, processor=processor)

    grpo_train(
        policy,
        policy_generation,
        dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        task_to_env,
        val_task_to_env,
        logger,
        checkpointer,
        grpo_state,
        master_config,
        processor,
    )


if __name__ == "__main__":
    main()
