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
import copy
import os
import re
import pprint
from typing import Any, Optional
from collections import defaultdict

from omegaconf import OmegaConf
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.dwrl import dwrl_train, setup
from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.interfaces import DatumSpec, TaskDataSpec, TaskDataProcessFnCallable
#from nemo_rl.data.utils import setup_response_data
from nemo_rl.distributed.ray_actor_environment_registry import (
    get_actor_python_env,
)
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir
from nemo_rl.data.datasets.preference_datasets.reward_benchmarks import HelpSteer3LocalDataset
#from nemo_rl.environments.binary_genrm_environment import (
#    GenRMEnvironment,
#    format_scoring_stage_prompt,
#)
from nemo_rl.environments.genrm_environment import GenRMEnvironment

TokenizerType = PreTrainedTokenizerBase


DWRL_PROMPT_TEMPLATE = """You are an expert evaluation judge specializing in the assessment of LLM responses. You are impartial, rigorous, and consistent. Given the conversation context and a single response to the user's latest query, you will follow the evaluation plan and scoring guidelines exactly as written below.

#### Conversation Context ####
{context}

#### Response to be Scored ####
[The Start of Response]
{response}
[The End of Response]

Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user prompt. Begin your evaluation by generating your own answer to the prompt. You must provide your answer before judging any answers. When evaluating the assistant's response, compare the assistant's response with your answer. You must identify and correct any mistakes or inaccurate information. Then consider if the assistant's response is helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when the user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user rather than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive. Then consider the creativity and novelty of the assistant's response when needed. Finally, identify any missing important information in the assistant's response which would be beneficial to include when responding to the user prompt.

#### Scoring Guidelines ####
Based on the evaluation plan above, assign scores using these scales:
**Individual Helpfulness Scores (1-5):**
- 5: Extremely Helpful - Completely aligned with what the user was asking for
- 4: Mostly Helpful - Generally useful with minor room for improvement
- 3: Partially Helpful - Misses the overall goal in some way
- 2: Borderline Unhelpful - Mostly doesn't capture what the user wanted
- 1: Not Helpful - Completely missed the essence of the request

#### Output Format ####
Analyze step by step following the evaluation plan, then provide your judgment as JSON:
```json
{{
    "response_analysis": "Your detailed analysis of the Response based on the evaluation plan",
    "score": <1-5>
}}
```"""


def flatten_to_single_turn(message_log):
    ret = ""
    for idx, message in enumerate(message_log):
        if message["role"] == "system":
            ret += "System: " + message["content"].strip() + "\n\n"
        elif message["role"] == "user":
            ret += "User: " + message["content"].strip() + "\n\n"
        elif message["role"] == "assistant":
            resp_no_thinking1 = re.sub(r"(?i)(?s)(\<think\>)(.*?)(\<\/think\>)", "", message["content"]).strip()
            resp_no_thinking2 = re.sub(r"(?i)(?s)(.*?)(\<\/think\>)", "", resp_no_thinking1).strip()
            ret += "Assistant: " + resp_no_thinking2 + "\n\n"
    
    return ret.strip()

def dwrl_preference_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process HelpSteer3 data for DWRL GenRM training."""
    def get_message_log(prompt_str):
        # Create message log
        message_log = []
        user_message = {
            "role": "user",
            "content": prompt_str,
        }
        message = tokenizer.apply_chat_template(  # type: ignore
            [user_message],
            tokenize=False,
            add_generation_prompt=True,
            add_special_tokens=False,
            #enable_thinking=False,
        )

        user_message["token_ids"] = tokenizer(message, return_tensors="pt")["input_ids"][0]
        user_message["content"] = message  
        message_log.append(user_message)
        
        return message_log
    
    # Extract the prompt (which contains the full formatted prompt with responses)
    context = datum_dict.get("context", "")
    resp1 = datum_dict.get("response1", "")
    resp2 = datum_dict.get("response2", "")
    pref_binary = datum_dict["preference"]
    if pref_binary == 0:
        chosen = resp1
        reject = resp2
    else:
        chosen = resp2
        reject = resp1
    
    prompt_chosen = DWRL_PROMPT_TEMPLATE.format(context=context, response=chosen)
    prompt_reject = DWRL_PROMPT_TEMPLATE.format(context=context, response=reject)

    # Prepare metadata for environment
    metadata = copy.deepcopy(datum_dict)
    #metadata['pair_idx'] = pair_idx
    metadata['num_responses'] = 1

    # Extract system prompt if present
    # system_prompt = datum_dict.get("system_prompt", "")
    
    message_log_chosen = get_message_log(prompt_chosen)
    message_log_reject = get_message_log(prompt_reject)
    
    # Calculate total length
    total_length_chosen = sum(len(msg["token_ids"]) for msg in message_log_chosen)
    total_length_reject = sum(len(msg["token_ids"]) for msg in message_log_reject)
    
    # Check if we need to truncate
    loss_multiplier = 1.0
    if total_length_chosen > max_seq_length or total_length_reject > max_seq_length:
        # Truncate messages
        for msg in message_log_chosen:
            msg["token_ids"] = msg["token_ids"][: min(4, max_seq_length // len(message_log_chosen)) ]
        for msg in message_log_reject:
            msg["token_ids"] = msg["token_ids"][: min(4, max_seq_length // len(message_log_reject)) ]
        loss_multiplier = 0.0

    return DatumSpec(
        message_log_chosen=message_log_chosen,
        message_log_rejected=message_log_reject,
        length_chosen=total_length_chosen,
        length_rejected=total_length_reject,
        extra_env_info=metadata,
        loss_multiplier=loss_multiplier,
        idx=idx,
        task_name=task_data_spec.task_name,
    )

def setup_data(
    tokenizer: TokenizerType,
    data_config: DataConfig,
    env_configs: dict[str, Any],
    seed: int,
) -> tuple[
    AllTaskProcessedDataset,
    Optional[AllTaskProcessedDataset],
    dict[str, EnvironmentInterface],
    dict[str, EnvironmentInterface],
]:
    print("\n▶ Setting up DWRL data...")

    # Create task spec for GenRM RLHF
    genrm_dwrl_task_spec = TaskDataSpec(
        task_name="genrm_dwrl",
        prompt_file=data_config.get("prompt_file"),
        system_prompt_file=data_config.get("system_prompt_file"),
    )

    # Load dataset
    train_data_path = data_config.get("train_data_path")
    val_data_path = data_config.get("val_data_path")
    train_dataset = HelpSteer3LocalDataset(train_data_path, task_name="genrm_dwrl", shuffle_seed=seed, split='train', double_swap=False)
    val_dataset = HelpSteer3LocalDataset(val_data_path, task_name="genrm_dwrl", shuffle_seed=seed, split='validation', double_swap=False) if val_data_path else None

    # Set up data processor
    task_data_processors: dict[str, tuple[TaskDataSpec, TaskDataProcessFnCallable]] = (
        defaultdict(lambda: (genrm_dwrl_task_spec, dwrl_preference_data_processor))
    )
    task_data_processors["genrm_dwrl"] = (
        genrm_dwrl_task_spec,
        dwrl_preference_data_processor,
    )
    # Also register under alternate name for backward compatibility
    task_data_processors["dwrl_genrm"] = (
        genrm_dwrl_task_spec,
        dwrl_preference_data_processor,
    )

    # Setup GenRM RLHF environment
    #genrm_rlhf_env = GenRMRLHFEnvironment.options(  # type: ignore # it's wrapped with ray.remote
    #    runtime_env={
    #        "py_executable": get_actor_python_env(
    #            "nemo_rl.environments.genrm_rlhf_environment.GenRMRLHFEnvironment"
    #        ),
    #        "env_vars": dict(os.environ),  # Pass thru all user environment variables
    #    }
    #).remote(env_configs["genrm_rlhf"])
    
    # Initialize GenRM environment
    genrm_env = GenRMEnvironment.options(
        runtime_env={
            "py_executable": get_actor_python_env(
                "nemo_rl.environments.genrm_environment.GenRMEnvironment"
            ),
            "env_vars": dict(os.environ),
        }
    ).remote(env_configs.get("genrm_dwrl", {}))

    # Create training dataset
    processed_train_dataset = AllTaskProcessedDataset(
        train_dataset,
        tokenizer,
        genrm_dwrl_task_spec,  # default_task_data_spec
        task_data_processors,  # task_data_processors dict
        max_seq_length=data_config["max_input_seq_length"],
    )

    # Create validation dataset if available
    processed_val_dataset: Optional[AllTaskProcessedDataset] = None
    if val_dataset:
        processed_val_dataset = AllTaskProcessedDataset(
            val_dataset,
            tokenizer,
            genrm_dwrl_task_spec,  # default_task_data_spec
            task_data_processors,  # task_data_processors dict
            max_seq_length=data_config["max_input_seq_length"],
        )

    # Map task to environment
    task_to_env: dict[str, EnvironmentInterface] = defaultdict(lambda: genrm_env)
    task_to_env["genrm_dwrl"] = genrm_env
    # Also register under alternate name for backward compatibility
    task_to_env["dwrl_genrm"] = genrm_env

    return processed_train_dataset, processed_val_dataset, task_to_env, task_to_env


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GRPO training with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Parse known args for the script
    args, overrides = parser.parse_known_args()

    return args, overrides


def main() -> None:
    """Main entry point."""
    # Parse arguments
    register_omegaconf_resolvers()
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "grpo_genrm_dwrl_1B.yaml"
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
    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    if "Qwen3" in config["policy"]["tokenizer"]["name"]:
        print("****** Qwen3 tokenizer detected! Swapping in fixed jinja template! ******", flush=True)
        #import requests
        #response = requests.get('https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/raw/main/chat_template.jinja')
        #tokenizer.chat_template = response.text
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen3_fixed_template.jinja"), "r", encoding="utf_8") as fr:
            qwen3_fixed_template = fr.read()
        tokenizer.chat_template = qwen3_fixed_template
    
    assert len(tokenizer.encode(config["grpo"]["dwrl"]["score_token"])) == 1, f'`grpo.dwrl.score_token` of [ {config["grpo"]["dwrl"]["score_token"]} ] does not tokenize to a single token, so DWRL will not work correctly. Pick a different token.'
    assert len(tokenizer.encode(config["grpo"]["dwrl"]["opposite_token"])) == 1, f'`grpo.dwrl.opposite_token` of [ {config["grpo"]["dwrl"]["opposite_token"]} ] does not tokenize to a single token, so DWRL will not work correctly. Pick a different token.'
    
    assert config["policy"]["generation"] is not None, (
        "A generation config is required for DWRL"
    )
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer
    )

    # setup data
    (
        dataset,
        val_dataset,
        task_to_env,
        val_task_to_env,
    ) = setup_data(tokenizer, config["data"], config["env"], config["grpo"]["seed"])

    (
        policy,
        policy_generation,
        _nemo_gym_actor,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    ) = setup(config, tokenizer, dataset, val_dataset)

    # Check if async mode is enabled
    if "async_grpo" in config["grpo"] and config["grpo"]["async_grpo"]["enabled"]:
        return RuntimeError("async grpo is not yet supported for DWRL")
        # Async GRPO does not support dynamic sampling, reward scaling, or reward shaping (DAPO features)
        unsupported_features = [
            "use_dynamic_sampling",
            "reward_scaling",
            "reward_shaping",
        ]

        for feature in unsupported_features:
            if feature not in config["grpo"]:
                continue

            if feature == "use_dynamic_sampling":
                if config["grpo"][feature]:
                    raise NotImplementedError(
                        f"{feature} is not supported with async GRPO"
                    )
            else:
                if config["grpo"][feature]["enabled"]:
                    raise NotImplementedError(
                        f"{feature} is not supported with async GRPO"
                    )

        from nemo_rl.algorithms.dwrl import async_dwrl_train

        print("🚀 Running async GRPO training")

        async_config = config["grpo"]["async_grpo"]
        # Run async GRPO training
        async_dwrl_train(
            policy=policy,
            policy_generation=policy_generation,
            dataloader=dataloader,
            val_dataloader=val_dataloader,
            tokenizer=tokenizer,
            loss_fn=loss_fn,
            task_to_env=task_to_env,
            val_task_to_env=val_task_to_env,
            logger=logger,
            checkpointer=checkpointer,
            grpo_save_state=grpo_state,
            master_config=master_config,
            max_trajectory_age_steps=async_config["max_trajectory_age_steps"],
        )
    else:
        print("🚀 Running synchronous GRPO training")

        # Run standard GRPO training
        dwrl_train(
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
        )


if __name__ == "__main__":
    main()
