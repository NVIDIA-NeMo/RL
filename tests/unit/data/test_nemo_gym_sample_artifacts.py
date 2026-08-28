# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import threading

import pytest
import torch

from nemo_rl.data.nemo_gym_sample_artifacts import (
    NEMO_GYM_TRAINING_SAMPLE_ARTIFACT_FORMAT,
    AsyncNemoGymTrainingSampleWriter,
    save_nemo_gym_training_samples,
)


def test_save_nemo_gym_training_samples_writes_one_atomic_pt(tmp_path):
    samples = [
        {
            "_ng_task_index": 12,
            "_ng_rollout_index": 1,
            "full_result": {
                "response": {
                    "output": [
                        {
                            "generation_token_ids": {
                                "__nemo_rl_popped__": True,
                                "shape": [3],
                            }
                        }
                    ]
                }
            },
        }
    ]

    path = save_nemo_gym_training_samples(
        base_dir=tmp_path,
        step=3,
        samples=samples,
    )

    assert path.name == "sample_batch_data.step_3.pt"
    assert list(tmp_path.iterdir()) == [path]
    payload = torch.load(path, weights_only=False)
    assert payload == {
        "format": NEMO_GYM_TRAINING_SAMPLE_ARTIFACT_FORMAT,
        "step": 3,
        "num_samples": 1,
        "samples": samples,
    }


def test_save_nemo_gym_training_samples_rejects_nested_tensors(tmp_path):
    with pytest.raises(TypeError, match=r"samples\[0\]\.full_result\.tokens"):
        save_nemo_gym_training_samples(
            base_dir=tmp_path,
            step=1,
            samples=[{"full_result": {"tokens": torch.tensor([1, 2])}}],
        )


def test_async_nemo_gym_training_sample_writer_can_be_polled(tmp_path):
    entered = threading.Event()
    release = threading.Event()

    def blocking_save(*, step, samples):
        entered.set()
        assert release.wait(timeout=5)
        return tmp_path / f"step-{step}-{len(samples)}.pt"

    writer = AsyncNemoGymTrainingSampleWriter(blocking_save)
    writer.start(step=2, samples=[{"full_result": {"reward": 1.0}}])

    assert entered.wait(timeout=5)
    assert writer.finish(wait=False) is False
    release.set()
    assert writer.finish(wait=True) is True
