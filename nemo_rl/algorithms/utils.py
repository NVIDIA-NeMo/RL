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
from collections import defaultdict
from functools import wraps
from typing import Optional, Union

import numpy as np
import torch
from transformers import AutoTokenizer

from nemo_rl.data import hf_datasets
from nemo_rl.models.dtensor.parallelize import (
    get_logprobs_from_vocab_parallel_logits,
    token_level_entropy_from_vocab_parallel_logits,
)
from nemo_rl.models.policy import TokenizerConfig


def calculate_kl_penalty_joschu2020(
    logprobs_policy: torch.Tensor, logprobs_reference: torch.Tensor
):
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
    expected_responses_per_prompt: Optional[int] = None,
):
    """Function to compute a baseline for each (prompt, response) pair in the batch.

    The same baseline is calculated for each prompt. Samples set to 0 in 'valid_mask'
    are not included in the baseline calculation.

    prompts:    tensor (b, s)     Tensor of prompts the model used. May be on any device
    rewards:    tensor (b,)       Float-valued rewards. May be on any device
    valid_mask: tensor (b,)       Vector of 0/1, where 0 is to ignore and 1 is to keep
    leave_one_out_baseline: bool  Compute an unbiased baseline by leaving out the sample that
                                  the baseline is for (from RLOO https://arxiv.org/abs/2402.14740)
    expected_responses_per_prompt: int  Expected number of responses per prompt (for validation)

    Returns:
    tensor (b,) of baselines on the same device as 'rewards'
    """
    unique_prompts = torch.unique(prompts, dim=0)

    # ASSERTION 1: Check that each prompt has the expected number of responses
    if expected_responses_per_prompt is not None:
        for i in range(len(unique_prompts)):
            is_matching_prompt = (prompts == unique_prompts[i]).all(1)
            num_responses_this_prompt = is_matching_prompt.sum().item()
            
            if num_responses_this_prompt != expected_responses_per_prompt:
                prompt_repr = unique_prompts[i][:10]  # First 10 tokens for debugging
                raise AssertionError(
                    f"Baseline calculation correctness violation: "
                    f"Prompt {prompt_repr}... has {num_responses_this_prompt} responses "
                    f"but expected {expected_responses_per_prompt}. "
                    f"This indicates that responses for the same prompt are split across DP ranks, "
                    f"which will result in incorrect baseline calculation. "
                    f"Consider using grouped sharding or ensuring num_prompts >= num_dp_ranks."
                )

    baseline = torch.zeros_like(rewards)
    sq_baseline = torch.zeros_like(rewards)
    reward_device = rewards.get_device()
    if reward_device == -1:
        reward_device = torch.device("cpu")

    metrics = defaultdict(list)

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
            rewards_reshaped = rewards[prompt_idx] * valid_mask[prompt_idx]

            metrics["average_reward_per_prompt"].append(rewards_reshaped.mean())
            metrics["average_pass_at_k_per_prompt"].append((rewards_reshaped > 0).any())

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
    return (
        baseline,
        std,
        {
            k: torch.as_tensor(v, dtype=torch.float32).mean().item()
            for k, v in metrics.items()
        },
    )


def surpress_user_warnings(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            output = f(*args, **kwargs)
        return output

    return wrapper


def masked_mean(
    values,
    mask,
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


def set_seed(seed: int):
    """Sets the seed for python, numpy, and pytorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_tokenizer(tokenizer_config: TokenizerConfig) -> AutoTokenizer:
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
        AutoTokenizer: The configured tokenizer instance

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
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_config["name"])
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


def compute_token_level_entropy(
    logits: Union[torch.Tensor, torch.distributed.tensor.DTensor],
):
    """Computes token-level entropy from logits and input ids.

    return B x (S-1)
    """
    if isinstance(logits, torch.distributed.tensor.DTensor):
        token_level_entropy = token_level_entropy_from_vocab_parallel_logits(
            vocab_parallel_logits=logits,
        )
    else:
        next_token_logits_wo_last = logits[:, :-1]  # Remove last position's logits
        next_token_logprobs = torch.nn.functional.log_softmax(
            next_token_logits_wo_last, dim=-1
        )
        token_level_entropy = -(next_token_logprobs.exp() * next_token_logprobs).sum(-1)

    return token_level_entropy


def compute_token_logprobs(
    logits: Union[torch.Tensor, torch.distributed.tensor.DTensor],
    input_ids: torch.Tensor,
    chunk_size: int = 1024,
):
    """Computes token-level logprobs from logits and input ids."""
    if isinstance(logits, torch.distributed.tensor.DTensor):
        curr_logprobs = get_logprobs_from_vocab_parallel_logits(
            vocab_parallel_logits=logits,
            input_ids=input_ids,
            chunk_size=chunk_size,
        )
    else:
        logits = logits.to(torch.float32)
        next_token_logits_wo_last = logits[:, :-1]  # Remove last position's logits
        next_token_logprobs = torch.nn.functional.log_softmax(
            next_token_logits_wo_last, dim=-1
        )
        next_tokens = input_ids[:, 1:].cuda()  # Skip first token
        curr_logprobs = next_token_logprobs.gather(
            dim=-1, index=next_tokens.unsqueeze(-1)
        ).squeeze(-1)

    return curr_logprobs
