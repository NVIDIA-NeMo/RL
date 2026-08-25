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
import torch

from nemo_rl.models.diffusion.flow_grpo_policy import FlowGRPOPolicy

merge = FlowGRPOPolicy._merge_trajectories


def test_merge_single_shard_is_identity():
    traj = {"prompts": ["a"], "images": torch.rand(1, 3, 4, 4)}
    assert merge([traj]) is traj


def test_merge_concats_tensors_and_lists():
    t1 = {
        "prompts": ["a", "b"],
        "metadata": [{}, {}],
        "images": torch.rand(2, 3, 4, 4),
        "timesteps": torch.arange(8).view(1, 8).expand(2, -1),
    }
    t2 = {
        "prompts": ["c"],
        "metadata": [{"k": 1}],
        "images": torch.rand(1, 3, 4, 4),
        "timesteps": torch.arange(8).view(1, 8),
    }
    out = merge([t1, t2])
    assert out["prompts"] == ["a", "b", "c"]
    assert out["metadata"][2] == {"k": 1}
    assert out["images"].shape == (3, 3, 4, 4)
    assert torch.equal(out["images"][:2], t1["images"])
    assert out["timesteps"].shape == (3, 8)


def test_merge_pads_ragged_sequence_dims():
    # Per-worker prompt embeddings pad only to the local batch max; the
    # merge must right-pad dim 1 to the global max with zeros.
    t1 = {"prompt_embeds": torch.ones(2, 5, 4), "prompt_embeds_mask": torch.ones(2, 5)}
    t2 = {"prompt_embeds": torch.ones(1, 3, 4), "prompt_embeds_mask": torch.ones(1, 3)}
    out = merge([t1, t2])
    assert out["prompt_embeds"].shape == (3, 5, 4)
    assert out["prompt_embeds_mask"].shape == (3, 5)
    assert torch.equal(out["prompt_embeds"][2, 3:], torch.zeros(2, 4))
    assert torch.equal(
        out["prompt_embeds_mask"][2], torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0])
    )
