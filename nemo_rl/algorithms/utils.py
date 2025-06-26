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
import random
import warnings
from functools import wraps
from typing import Optional

import numpy as np
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from nemo_rl.data import hf_datasets
from nemo_rl.models.policy import TokenizerConfig


def calculate_kl_penalty_joschu2020(
    logprobs_policy: torch.Tensor, logprobs_reference: torch.Tensor
) -> torch.Tensor:
    """Calculates a per-token estimate of the KL Divergence between two log_probs.

    From Schulman 2020, always positive.

    logprobs_policy:    torch.Tensor (b, s)
    logprobs_reference: torch.Tensor (b, s)
    """
    r = logprobs_reference - logprobs_policy
    return torch.exp(r) - r - 1


def calculate_baseline_and_std_per_prompt(
    prompts: torch.Tensor,
    rewards: torch.Tensor,
    valid_mask: torch.Tensor,
    leave_one_out_baseline: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Function to compute a baseline for each (prompt, response) pair in the batch.

    The same baseline is calculated for each prompt. Samples set to 0 in 'valid_mask'
    are not included in the baseline calculation.

    prompts:    tensor (b, s)     Tensor of prompts the model used. May be on any device
    rewards:    tensor (b,)       Float-valued rewards. May be on any device
    valid_mask: tensor (b,)       Vector of 0/1, where 0 is to ignore and 1 is to keep
    leave_one_out_baseline: bool  Compute an unbiased baseline by leaving out the sample that
                                  the baseline is for (from RLOO https://arxiv.org/abs/2402.14740)

    Returns:
    tensor (b,), tensor (b,) of baselines and std on the same device as 'rewards'
    """
    unique_prompts = torch.unique(prompts, dim=0)

    baseline = torch.zeros_like(rewards)
    sq_baseline = torch.zeros_like(rewards)
    device_ordinal = rewards.get_device()
    if device_ordinal == -1:
        reward_device = torch.device("cpu")
    else:
        reward_device = torch.device(reward_device)

    for i in range(len(unique_prompts)):
        is_matching_prompt = (prompts == unique_prompts[i]).all(1)
        prompt_idx = torch.arange(len(prompts), device=reward_device)[
            is_matching_prompt
        ]

        if leave_one_out_baseline:
            baseline_mask_matrix = (1 - torch.eye(len(prompt_idx))).to(reward_device)
        else:
            baseline_mask_matrix = torch.ones((len(prompt_idx), len(prompt_idx))).to(
                reward_device
            )

        if valid_mask[prompt_idx].sum() <= 1:
            # Ignore sample: there are no valid responses, so set baseline equal to reward
            # to ignore it in the loss computation
            baseline[prompt_idx] = rewards[prompt_idx]
        else:
            num_valid = valid_mask[prompt_idx].float().sum() - int(
                leave_one_out_baseline
            )
            prompt_baseline = (
                torch.matmul(
                    baseline_mask_matrix, rewards[prompt_idx] * valid_mask[prompt_idx]
                )
                / num_valid
            )
            prompt_baseline_square = (
                torch.matmul(
                    baseline_mask_matrix,
                    (rewards[prompt_idx] ** 2) * valid_mask[prompt_idx],
                )
                / num_valid
            )

            baseline[prompt_idx] = prompt_baseline
            sq_baseline[prompt_idx] = prompt_baseline_square

    std = (sq_baseline - baseline.square()).sqrt().nan_to_num(0)
    return baseline, std


def surpress_user_warnings(f):  # type: ignore
    @wraps(f)
    def wrapper(*args, **kwargs):  # type: ignore
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            output = f(*args, **kwargs)
        return output

    return wrapper


def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    dim: Optional[int] = None,
    global_normalization_factor: Optional[torch.Tensor | float] = None,
):
    """Computes the mean of a microbatch, using a global statistic as the normalization factor."""
    normalization_factor = (
        torch.sum(mask, dim=dim)
        if global_normalization_factor is None
        else global_normalization_factor
    )
    return torch.sum(values * mask, dim=dim) / (normalization_factor + 1e-8)


def set_seed(seed: int) -> None:
    """Sets the seed for python, numpy, and pytorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_tokenizer(tokenizer_config: TokenizerConfig) -> PreTrainedTokenizerBase:
    """Get the tokenizer and set pad token to eos token if it is not already set.

    This function initializes a tokenizer from the Hugging Face transformers library
    and configures it with appropriate chat templates and padding tokens.

    Args:
        tokenizer_config: A dictionary containing tokenizer configuration.
            Required keys:
                - name: The name or path of the pretrained tokenizer
            Optional keys:
                - chat_template: The chat template to use. Can be:
                    - None: Uses a passthrough template that just returns message content
                    - "default": Uses the tokenizer's default template
                    - A custom jinja2 template string
                    If not specified, the tokenizer's default template will be used.

    Returns:
        PreTrainedTokenizerBase: The configured tokenizer instance

    Examples:
        ```{doctest}
        >>> from transformers import AutoTokenizer
        >>> from nemo_rl.algorithms.utils import get_tokenizer
        >>> # not specifying a chat template uses the tokenizer's default
        >>> config = {"name": "meta-llama/Llama-3.2-1B-Instruct"}
        >>> tokenizer = get_tokenizer(config)
        No chat template provided, using tokenizer's default
        >>> messages = [
        ...     {"role": "system", "content": "You are a helpful AI assistant."},
        ...     {"role": "user", "content": "Hello!"}
        ... ]
        >>> formatted = tokenizer.apply_chat_template(messages, tokenize=False)
        >>> assert formatted == AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct").apply_chat_template(messages, tokenize=False)

        >>> # Using a passthrough template
        >>> config = {
        ...     "name": "meta-llama/Llama-3.2-1B-Instruct",
        ...     "chat_template": None
        ... }
        >>> tokenizer = get_tokenizer(config)
        Using passthrough chat template
        >>> formatted = tokenizer.apply_chat_template(messages, tokenize=False)
        >>> assert formatted == "".join(msg["content"] for msg in messages)

        >>> # Using a custom template
        >>> config = {
        ...     "name": "meta-llama/Llama-3.2-1B-Instruct",
        ...     "chat_template": "{% for message in messages %}{{ ' START: ' + message['content'] + ' END.' }}{% endfor %}"
        ... }
        >>> tokenizer = get_tokenizer(config)
        Using custom chat template
        >>> formatted = tokenizer.apply_chat_template(messages, tokenize=False)
        >>> assert formatted == " START: You are a helpful AI assistant. END. START: Hello! END."
        ```
    """
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_config["name"], trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if "chat_template" in tokenizer_config:
        if tokenizer_config["chat_template"] is None:
            print("Using passthrough chat template")
            tokenizer.chat_template = (
                hf_datasets.COMMON_CHAT_TEMPLATES.passthrough_prompt_response
            )
        elif tokenizer_config["chat_template"].lower() == "default":
            print("Using tokenizer's default chat template")
        else:
            print("Using custom chat template")
            tokenizer.chat_template = tokenizer_config["chat_template"]
    else:
        print("No chat template provided, using tokenizer's default")

    return tokenizer
