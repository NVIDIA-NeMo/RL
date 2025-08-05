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
import os
import pprint
from collections import defaultdict
from typing import Any, Optional

import torch
from omegaconf import OmegaConf
from transformers import AutoProcessor
from PIL import Image
import requests
from io import BytesIO
import base64

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.environments.vlm_environment import VLMEnvironment
from nemo_rl.data.hf_datasets.clevr import CLEVRCoGenTDataset, format_clevr_cogent_dataset
from nemo_rl.data.hf_datasets.geometry3k import Geometry3KDataset, format_geometry3k_dataset
from nemo_rl.data.hf_datasets.refcoco import RefCOCODataset, format_refcoco_dataset
from nemo_rl.data.interfaces import (
    DatumSpec,
    LLMMessageLogType,
    TaskDataProcessFnCallable,
    TaskDataSpec,
)
from nemo_rl.distributed.ray_actor_environment_registry import (
    get_actor_python_env,
)
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir
from nemo_rl.data.multimodal_utils import PackedMultimodalDataItem, \
    get_multimodal_keys_from_processor,  \
    reroute_processor_model_name_patch, \
    get_dim_to_pack_along

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

def resolve_to_image(image_path_or_image: str | Image.Image) -> Image.Image:
    """ Resolve the image path to a PIL.Image object. 
    
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
    """Process a datum dictionary (directly loaded from data/hf_datasets/<dataset_name>.py) into a DatumSpec for the VLM Environment."""

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
    user_message = {
        "role": "user",
        "content": []
    }
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
                user_message["content"].append({
                    "type": "text",
                    "text": task_data_spec.prompt.format(content["text"]) if task_data_spec.prompt else content["text"],
                })
    else:
        # conversation consists of a text-only message
        user_message["content"] = task_data_spec.prompt.format(problem)
    
    images = [resolve_to_image(image) for image in images]

    # get formatted user message
    if hasattr(processor, 'conversation_preprocessor'):
        user_message_for_chat_template = processor.conversation_preprocessor(user_message)
    else:
        user_message_for_chat_template = user_message

    # this is the string-tokenized conversation template for the generation policy (for vllm)
    string_formatted_dialog = processor.apply_chat_template(
        [user_message_for_chat_template],
        tokenize=False,
        add_generation_prompt=True,
    )
    if isinstance(string_formatted_dialog, (list, tuple)):
        string_formatted_dialog = string_formatted_dialog[0]
    
    try:
        # this is the id-tokenized and image processed conversation template for the policy
        message: dict = processor.apply_chat_template(
            [user_message],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    except Exception as e:
        # some processors (Phi-4) do not support apply_chat_template with tokenize=True, in these cases we have to use the 
        # __call__ function directly
        # on need to add generation prompt here, string formatted dialog is already the prompt
        message: dict = processor(text=string_formatted_dialog, images=images, return_tensors="pt")

    # add this for backward compatibility
    user_message['token_ids'] = message['input_ids'][0]
    # gemma
    if 'token_type_ids' in message:
        user_message['token_type_ids'] = message['token_type_ids'][0]

    # add all keys and values to the user message, and the list of keys
    multimodal_keys = get_multimodal_keys_from_processor(processor)
    for key in multimodal_keys:
        if key in message:
            user_message[key] = PackedMultimodalDataItem(message[key], dim_to_pack=get_dim_to_pack_along(processor, key))

    ### append to user message
    message_log.append(user_message)

    length = sum(len(m["token_ids"]) for m in message_log)
    loss_multiplier = 1.0
    if length > max_seq_length:
        # make smaller and mask out
        for chat_message in message_log:
            chat_message["token_ids"] = chat_message["token_ids"][
                : min(4, max_seq_length // len(message_log))
            ]
        loss_multiplier = 0.0

    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
        "task_name": task_data_spec.task_name,
        # get the prompt content! (use this for vllm-backend that needs formatted dialog and list of images) for the entire conversation
        # add images for vllm serving
        'vllm_content': string_formatted_dialog,
        'vllm_images': images,
    }
    return output


def setup_data(
    processor: AutoProcessor,
    data_config: DataConfig,
    env_configs: dict[str, Any],
) -> tuple[
    AllTaskProcessedDataset,
    Optional[AllTaskProcessedDataset],
    dict[str, EnvironmentInterface],
    dict[str, EnvironmentInterface],
]:
    '''
    This function will create a TaskSpec, DatumSpec, and connect the two 
    
    task_spec contains the task name as well as prompt and system prompt modifiers that can be used by data processor

    '''
    print("\n▶ Setting up data...")
    # define task name and use it (make it as generic as possible)
    task_name = data_config['task_name']
    vlm_task_spec = TaskDataSpec(
        task_name=task_name,
        prompt_file=data_config["prompt_file"],
        system_prompt_file=data_config["system_prompt_file"],
    )

    # Load CLEVR-CoGenT dataset using nemo rl datasets
    # other VLM datasets can be added here
    if data_config['dataset_name'] == 'clevr-cogent':
        data: Any = CLEVRCoGenTDataset(split=data_config['split'], 
                                       seed=data_config['seed'], task_name=data_config['task_name'])
    elif data_config['dataset_name'] == 'refcoco':
        data: Any = RefCOCODataset(split=data_config['split'], seed=data_config['seed'], task_name=data_config['task_name'], path_to_coco_images=data_config.get('path_to_coco_images', None))
    elif data_config['dataset_name'] == 'geometry3k':
        data: Any = Geometry3KDataset(split=data_config['split'], task_name=data_config['task_name'])
    else:
        raise ValueError(f"No processor for dataset {data_config['dataset_name']}.")

    # add data processor for different tasks
    task_data_processors: dict[str, tuple[TaskDataSpec, TaskDataProcessFnCallable]] = (
        defaultdict(lambda: (vlm_task_spec, hf_data_processor))
    )
    task_data_processors[task_name] = (vlm_task_spec, hf_data_processor)

    vlm_env = VLMEnvironment.options(  # type: ignore # it's wrapped with ray.remote
        runtime_env={
            "py_executable": get_actor_python_env(
                "nemo_rl.environments.vlm_environment.VLMEnvironment"
            ),
            "env_vars": dict(os.environ),  # Pass thru all user environment variables
        }
    ).remote(env_configs[task_name])

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
            os.path.dirname(__file__), "configs", "grpo_clevr_cogent_trainA.yaml"
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

    # setup tokenizer
    processor = AutoProcessor.from_pretrained(reroute_processor_model_name_patch(config["policy"]["model_name"]), trust_remote_code=True)
    tokenizer = processor.tokenizer
    # inherit tokenizer's special tokens
    processor.pad_token = processor.tokenizer.pad_token
    processor.eos_token = processor.tokenizer.eos_token
    processor.bos_token = processor.tokenizer.bos_token
    processor.pad_token_id = processor.tokenizer.pad_token_id
    processor.eos_token_id = processor.tokenizer.eos_token_id
    processor.bos_token_id = processor.tokenizer.bos_token_id


    assert config["policy"]["generation"] is not None, (
        "A generation config is required for GRPO"
    )
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], processor.tokenizer
    )

    # setup data
    # this function is local to this script, and can be extended to other VLM datasets
    (
        dataset,
        val_dataset,
        task_to_env,
        val_task_to_env,
    ) = setup_data(processor, config["data"], config["env"])

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
