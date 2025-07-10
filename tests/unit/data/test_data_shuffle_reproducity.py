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

from collections import defaultdict

from torchdata.stateful_dataloader import StatefulDataLoader

from examples.run_grpo_math import hf_data_processor
from nemo_rl.algorithms.utils import get_tokenizer, set_seed
from nemo_rl.data.datasets import AllTaskProcessedDataset, rl_collate_fn
from nemo_rl.data.hf_datasets.openmathinstruct2 import OpenMathInstruct2Dataset
from nemo_rl.data.interfaces import TaskDataProcessFnCallable, TaskDataSpec
from nemo_rl.models.policy import TokenizerConfig

# Test configuration
TOKENIZER_CONFIG: TokenizerConfig = {
    "name": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "chat_template": "default",
}

SEED = 42
BATCH_SIZE = 4
MAX_SEQ_LENGTH = 128
MAX_BATCHES_TO_TEST = 10


def create_dataloader(seed: int = SEED) -> StatefulDataLoader:
    """Create a dataloader with consistent configuration for testing."""
    set_seed(seed)

    # Initialize dataset
    data = OpenMathInstruct2Dataset(seed=seed)

    # Setup tokenizer
    tokenizer = get_tokenizer(TOKENIZER_CONFIG)

    # Configure task specification
    math_task_spec = TaskDataSpec(
        task_name="math",
        prompt_file="examples/prompts/cot.txt",
        system_prompt_file=None,
    )

    task_data_processors: dict[str, tuple[TaskDataSpec, TaskDataProcessFnCallable]] = (
        defaultdict(lambda: (math_task_spec, hf_data_processor))
    )
    task_data_processors["math"] = (math_task_spec, hf_data_processor)

    dataset = AllTaskProcessedDataset(
        dataset=data.formatted_ds["train"],
        tokenizer=tokenizer,
        default_task_data_spec=math_task_spec,
        task_data_processors=task_data_processors,
        max_seq_length=128,
    )

    # Create dataloader
    return StatefulDataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=rl_collate_fn,
        drop_last=True,
    )


def test_data_shuffle_reproducity():
    """Test that dataloader shuffling is reproducible with the same seed."""
    dataloader0 = create_dataloader()
    dataloader1 = create_dataloader()

    for i, (batch0, batch1) in enumerate(zip(dataloader0, dataloader1)):
        assert str(batch0) == str(batch1), f"Batch {i} is different"
        if i >= MAX_BATCHES_TO_TEST:
            break
