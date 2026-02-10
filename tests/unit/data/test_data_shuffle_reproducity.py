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

import os
import tempfile
from collections import defaultdict

import pytest
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from nemo_rl.algorithms.utils import get_tokenizer, set_seed
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.datasets.response_datasets import OpenMathInstruct2Dataset
from nemo_rl.data.interfaces import TaskDataProcessFnCallable, TaskDataSpec
from nemo_rl.data.processors import math_hf_data_processor
from nemo_rl.models.policy import TokenizerConfig

# Test configuration
TOKENIZER_CONFIG: TokenizerConfig = {
    "name": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "chat_template": "default",
}

MAX_BATCHES_TO_TEST = 10


def create_dataloader(
    seed: int = 42, max_seq_length: int = 128, batch_size: int = 4
) -> StatefulDataLoader:
    """Create a dataloader with consistent configuration for testing."""
    # Initialize dataset
    data = OpenMathInstruct2Dataset(seed=seed)
    task_name = (
        data.task_name if hasattr(data, "task_name") else data.task_spec.task_name
    )

    # Setup tokenizer
    tokenizer = get_tokenizer(TOKENIZER_CONFIG)

    # Configure task specification
    math_task_spec = TaskDataSpec(
        task_name=task_name,
        prompt_file=f"{os.path.dirname(os.path.abspath(__file__))}/../../../examples/prompts/cot.txt",
        system_prompt_file=None,
    )

    task_data_processors: dict[str, tuple[TaskDataSpec, TaskDataProcessFnCallable]] = (
        defaultdict(lambda: (math_task_spec, math_hf_data_processor))
    )
    task_data_processors[task_name] = (math_task_spec, math_hf_data_processor)

    dataset = AllTaskProcessedDataset(
        dataset=data.dataset.select(range(1000)),
        tokenizer=tokenizer,
        default_task_data_spec=math_task_spec,
        task_data_processors=task_data_processors,
        max_seq_length=max_seq_length,
    )

    return StatefulDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=rl_collate_fn,
        drop_last=True,
    )


@pytest.mark.parametrize("seed", [42, 24])
def test_data_shuffle_reproducity_from_start(seed):
    """Test that dataloader shuffling is reproducible with the same seed."""
    # Step 1: Set seed and create initial dataloader
    set_seed(seed)
    original_dataloader = create_dataloader(seed=seed)

    expected_batches = []
    for batch in original_dataloader:
        expected_batches.append(batch)
        if len(expected_batches) >= MAX_BATCHES_TO_TEST:
            break

    # Step 2: to mimic a new experiment:
    #    set original seed and create new dataloader under the same seed environment
    set_seed(seed)
    new_dataloader = create_dataloader(seed=seed)

    for i, (expected_batch, actual_batch) in enumerate(
        zip(expected_batches, new_dataloader)
    ):
        assert str(expected_batch) == str(actual_batch), f"Batch {i} is different"


@pytest.mark.parametrize("save_state_at_batch", [6, 10])
def test_data_shuffle_reproducity_from_continue(save_state_at_batch, seed=42):
    """Test that dataloader state can be saved and restored for continuation."""
    # Step 1: Set seed and create initial dataloader
    set_seed(seed)
    original_dataloader = create_dataloader(seed=seed)

    with tempfile.TemporaryDirectory() as temp_dir:
        expected_batches = []
        for i, batch in enumerate(original_dataloader):
            if (
                i >= save_state_at_batch - 1
            ):  # Stop after consuming exactly save_state_at_batch batches
                if i == save_state_at_batch - 1:
                    # Step 2: Save the state at this point
                    state_file = os.path.join(temp_dir, "dataloader_state.pt")
                    torch.save(original_dataloader.state_dict(), state_file)
                else:
                    # Step 3: Get the expected continuation from original dataloader
                    expected_batches.append(batch)
                    if len(expected_batches) >= MAX_BATCHES_TO_TEST:
                        break

        # step 4: to mimic a continued experiment:
        #    set original seed and create new dataloader under the same seed environment
        #    load the saved state and continue from the saved point
        set_seed(seed)
        continued_dataloader = create_dataloader(seed=seed)

        state_dict = torch.load(state_file)
        continued_dataloader.load_state_dict(state_dict)

        # Step 5: Get batches from the continued dataloader
        actual_batches = []
        for batch in continued_dataloader:
            if len(actual_batches) >= MAX_BATCHES_TO_TEST:
                break
            actual_batches.append(batch)

        assert len(actual_batches) == len(expected_batches)

        # Step 6: Compare the batches - they should be identical
        for i, (actual_batch, expected_batch) in enumerate(
            zip(actual_batches, expected_batches)
        ):
            assert str(actual_batch) == str(expected_batch), (
                f"Batch {i} from continued dataloader doesn't match expected batch\n"
                f"actual_batch['idx']:\t{actual_batch['idx']}\n"
                f"expected_batch['idx']:\t{expected_batch['idx']}"
            )
