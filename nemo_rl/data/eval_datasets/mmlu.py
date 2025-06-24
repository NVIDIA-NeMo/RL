"""MMLU dataset and its variants."""

from typing import Any, Optional

from datasets import load_dataset

from nemo_rl.data.interfaces import TaskDataSpec


class MMLUDataset:
    def __init__(self, prompt_file: Optional[str] = None, system_prompt_file: Optional[str] = None):
        ds = load_dataset('csv', data_files="https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv", split='train')
        self.rekeyed_ds = ds.map(self._rekey, remove_columns=ds.column_names)

        self.task_spec = TaskDataSpec(
            task_name='MMLU',
            prompt_file=prompt_file,
            system_prompt_file=system_prompt_file,
        )

    def _rekey(self, data: dict[str, Any]):
        return {
            'question': data['Question'],
            'options': dict(
                A=data['A'],
                B=data['B'],
                C=data['C'],
                D=data['D'],
            ),
            'answer': data['Answer'],
            'subject': data['Subject'],
        }

