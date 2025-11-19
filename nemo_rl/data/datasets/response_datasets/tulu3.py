from typing import Any

from datasets import Dataset, load_dataset

from nemo_rl.data.interfaces import TaskDataSpec


def format_sample(sample: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    messages = [
        {
            "role": m["role"],
            "content": m["content"],
        }
        for m in sample["messages"]
    ]

    assert messages[-1]["role"] == "assistant", (
        "This formatter assumes the last message is from the assistant. Only the last message will be trained on."
    )

    return {"messages": messages}


def prepare_tulu3_dataset(test_size: float, seed: int) -> Dataset:
    dataset = load_dataset(
        "allenai/tulu-3-sft-mixture",
        split="train",
    )
    split_ds = dataset.train_test_split(test_size=test_size, seed=seed)

    train_formatted = split_ds["train"].map(
        format_sample,
        remove_columns=split_ds["train"].column_names,
    )
    val_formatted = split_ds["test"].map(
        format_sample,
        remove_columns=split_ds["test"].column_names,
    )

    return {
        "train": train_formatted,
        "validation": val_formatted,
    }


class Tulu3Dataset:
    def __init__(
        self,
        seed: int,
        test_size: float = 0.05,
    ):
        """Initialize the Tulu3 dataset with train/validation split.

        Args:
            seed: Random seed for reproducible splitting
            test_size: Proportion of data to use for validation (0.0-1.0)
        """
        self.formatted_ds = prepare_tulu3_dataset(test_size, seed)

        self.task_spec = TaskDataSpec(
            task_name="Tulu3",
        )
