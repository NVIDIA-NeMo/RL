from typing import Any, cast

import torch
from transformers import PreTrainedTokenizerBase
from datasets import Dataset, load_dataset

from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType, TaskDataSpec

TokenizerType = PreTrainedTokenizerBase


def format_ether1(data: dict[str, str | float | int]) -> dict[str, list[Any] | str]:
    return {
        "messages": [
            {
                "role": "user",
                "content": data["problem"],
            },
            {
                "role": "assistant",
                "content": data["solution"],
            },
        ],
        "task_name": "ether1",
    }


def prepare_ether1_dataset(
    seed: int = 42, 
    eval_runs: int = 1,
    num_train_epochs: int = 1,
    problem_types: str | list[str] | None = None
) -> dict[str, Dataset | None]:
    """Load and split the DeepScaler dataset into train and test sets.
    
    Args:
        seed: Random seed for shuffling
        eval_runs: Number of times to repeat validation samples
        num_train_epochs: Number of times to repeat training samples
        problem_types: Comma-separated string or list of problem types to filter by.
                      If None, no filtering is applied.
    """
    # Load the dataset for training (loading test benchmark cause its open sourced)
    train_ds = load_dataset("futurehouse/ether0-benchmark", split="test")
    test_ds = load_dataset("futurehouse/ether0-benchmark", split="test")

    # Filter datasets by problem type if specified
    if problem_types is not None:
        if isinstance(problem_types, str):
            problem_types = [pt.strip() for pt in problem_types.split(",")]
        problem_types_set = set(problem_types)
        train_ds = train_ds.filter(
            lambda x: x["problem_type"] in problem_types_set,
            desc=f"Filtering by problem types {problem_types} - train"
        )
        test_ds = test_ds.filter(
            lambda x: x["problem_type"] in problem_types_set,
            desc=f"Filtering by problem types {problem_types} - test"
        )

    # Shuffle the training dataset with the specified seed
    train_ds = train_ds.shuffle(seed=seed)

    # Format the examples, removing original columns
    train_formatted = train_ds.map(format_ether1, remove_columns=train_ds.column_names)
    test_formatted = test_ds.map(format_ether1, remove_columns=test_ds.column_names)

    # Repeat training dataset N times
    train_repeated = []
    for _ in range(num_train_epochs):
        train_repeated.extend(train_formatted)
    train_formatted = train_formatted.from_list(train_repeated)

    # Compute accuracy N times per sample
    test_repeated = []
    for _ in range(eval_runs):
        test_repeated.extend(test_formatted)
    test_formatted = test_formatted.from_list(test_repeated)

    return {
        "train": train_formatted,
        "validation": test_formatted,
    }


class Ether1Dataset:
    def __init__(self, seed: int = 42, eval_runs: int = 1, num_train_epochs: int = 1, problem_types: str | list[str] | None = None) -> None:
        """Initialize the ehter1 dataset with train/test split.

        Args:
            seed: Random seed for reproducible splitting
            eval_runs: Number of times to repeat validation samples
            num_train_epochs: Number of times to repeat training samples
            problem_types: Comma-separated string or list of problem types to filter by.
                          If None, no filtering is applied.
        """
        self.formatted_ds = prepare_ether1_dataset(seed=seed, eval_runs=eval_runs, num_train_epochs=num_train_epochs, problem_types=problem_types)

        self.task_spec = TaskDataSpec(
            task_name="ether1",
        )


def ether1_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process a datum dictionary (directly loaded from data/hf_datasets/openmathinstruct2.py) into a DatumSpec for the Reward Model Environment."""
    user_message = datum_dict["messages"]
    problem = user_message[0]["content"]
    extra_env_info = {"ground_truth": user_message[1]["content"]}

    message_log: LLMMessageLogType = []

    # system prompt
    if task_data_spec.system_prompt:
        sys_prompt: dict[str, str | torch.Tensor] = {
            "role": "system",
            "content": task_data_spec.system_prompt,
        }
        sys = tokenizer.apply_chat_template(
            [cast(dict[str, str], sys_prompt)],
            tokenize=False,
            add_generation_prompt=False,
            add_special_tokens=False,
        )
        sys_prompt["token_ids"] = tokenizer(
            sys, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]
        message_log.append(sys_prompt)

    formatted_content = (
        task_data_spec.prompt.format(problem) if task_data_spec.prompt else problem
    )
    user_message = {
        "role": "user",
        "content": formatted_content,
    }
    message: list[str] = tokenizer.apply_chat_template(  # type: ignore
        [user_message],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )

    user_message["token_ids"] = tokenizer(
        message,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]
    user_message["content"] = message
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
        "task_name": datum_dict["task_name"],
    }
    return output