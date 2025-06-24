"""AIME 2024 dataset."""

from typing import Any, Literal, Optional

from datasets import load_dataset

from nemo_rl.data.interfaces import TaskDataSpec


class AIME2024Dataset:
    def __init__(self,
            prompt_file: Optional[str]=None,
            system_prompt_file: Optional[str]=None,
        ):
        ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
        self.rekeyed_ds = ds.map(self._rekey, remove_columns=ds.column_names)
        self.task_spec = TaskDataSpec(
            task_name="aime2024",
            prompt_file=prompt_file,
            system_prompt_file=system_prompt_file,
        )

    def _rekey(self, data: dict[str, Any]):
        return {
            'problem': data['problem'],
            'expected_answer': data['answer'],
        }
