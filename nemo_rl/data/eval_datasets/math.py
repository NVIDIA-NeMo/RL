"""Math dataset and its variants."""

from typing import Any, Literal, Optional

from datasets import load_dataset

from nemo_rl.data.interfaces import TaskDataSpec


class MathDataset:
    def __init__(self, 
            variant: Literal["math_test", "math_500_test"] = "math_test",
            prompt_file: Optional[str]=None, 
            system_prompt_file: Optional[str]=None,
        ):
        ds = load_dataset('csv', data_files=f"https://openaipublic.blob.core.windows.net/simple-evals/{variant}.csv", split='train')
        self.rekeyed_ds = ds.map(self._rekey, remove_columns=ds.column_names)
        self.task_spec = TaskDataSpec(
            task_name=f'{variant}',
            prompt_file=prompt_file,
            system_prompt_file=system_prompt_file,
        )
    
    def _rekey(self, data: dict[str, Any]):
        return {
            'problem': data['Question'],
            'expected_answer': data['Answer'],
        }

