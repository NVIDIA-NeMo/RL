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

import copy

import pytest
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy.training_engine_adapter.config import (
    validate_training_engine_config,
)


def _policy_config() -> dict:
    return {
        "max_total_sequence_length": 16384,
        "megatron_cfg": {"enabled": True, "pipeline_model_parallel_size": 1},
        "sequence_packing": {
            "enabled": True,
            "train_mb_tokens": 32768,
            "logprob_mb_tokens": 32768,
        },
        "training_engine": {
            "recipe": "some_recipe",
            "recipe_kwargs": {},
            "checkpoint_path": "/ckpt",
            "seq_length": 32768,
            "overrides": {},
        },
    }


def test_valid_training_engine_config_passes():
    validate_training_engine_config(_policy_config())


@pytest.mark.parametrize(
    "path, value, match",
    [
        (("megatron_cfg", "enabled"), False, "megatron_cfg.enabled"),
        (("megatron_cfg", "pipeline_model_parallel_size"), 2, "pipeline"),
        (("sequence_packing", "enabled"), False, "sequence_packing.enabled"),
        (("sequence_packing", "train_mb_tokens"), 16384, "train_mb_tokens"),
        (("sequence_packing", "logprob_mb_tokens"), 16384, "logprob_mb_tokens"),
        (("training_engine", "recipe_kwargs"), {"seq_length": 1}, "recipe_kwargs"),
        (("max_total_sequence_length",), 65536, "max_total_sequence_length"),
    ],
)
def test_training_engine_config_rejects(path, value, match):
    config = copy.deepcopy(_policy_config())
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=match):
        validate_training_engine_config(config)


@pytest.mark.mcore
def test_scoring_datums_are_target_aligned_and_scores_map_back():
    from nemo_rl.models.policy.training_engine_adapter.data import (
        padded_scores,
        scoring_datums,
    )

    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[10, 11, 12, 13, 0], [20, 21, 0, 0, 0]]),
            "input_lengths": torch.tensor([4, 2]),
        }
    )
    datums = scoring_datums(data)

    assert [d.input_ids.tolist() for d in datums] == [[10, 11, 12], [20]]
    assert [d.target_ids.tolist() for d in datums] == [[11, 12, 13], [21]]
    assert [d.loss_mask.tolist() for d in datums] == [[1.0, 1.0, 1.0], [1.0]]

    scores = [torch.tensor([-1.0, -2.0, -3.0]), torch.tensor([-4.0])]
    assert padded_scores(scores, 5).tolist() == [
        [0.0, -1.0, -2.0, -3.0, 0.0],
        [0.0, -4.0, 0.0, 0.0, 0.0],
    ]


def _training_batch() -> BatchedDataDict:
    """Two samples of 6 and 4 tokens; the second row's last two columns are padding."""
    torch.manual_seed(0)
    token_mask = torch.tensor(
        [[0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 0, 0]], dtype=torch.float32
    )
    return BatchedDataDict(
        {
            "input_ids": torch.arange(12).reshape(2, 6),
            "input_lengths": torch.tensor([6, 4]),
            "token_mask": token_mask,
            "sample_mask": torch.ones(2),
            "advantages": torch.randn(2, 6),
            "prev_logprobs": -torch.rand(2, 6),
            "generation_logprobs": -torch.rand(2, 6),
            "reference_policy_logprobs": -torch.rand(2, 6),
        }
    )


@pytest.mark.mcore
def test_training_datums_carry_the_shifted_loss_fields():
    from nemo_rl.models.policy.training_engine_adapter.data import training_datums

    data = _training_batch()
    first, second = training_datums(data)

    assert first.loss_mask.tolist() == [0.0, 1.0, 1.0, 1.0, 1.0]
    assert second.loss_mask.tolist() == [1.0, 1.0, 1.0]
    assert torch.equal(first.advantages, data["advantages"][0, 1:6])
    assert torch.equal(second.old_logprobs, data["prev_logprobs"][1, 1:4])
    assert torch.equal(second.rollout_logprobs, data["generation_logprobs"][1, 1:4])
    assert torch.equal(
        second.reference_logprobs, data["reference_policy_logprobs"][1, 1:4]
    )


@pytest.mark.mcore
def test_training_datums_reject_weighted_masks():
    from nemo_rl.models.policy.training_engine_adapter.data import training_datums

    data = _training_batch()
    data["sample_mask"] = torch.tensor([1.0, 0.5])
    with pytest.raises(ValueError, match="binary"):
        training_datums(data)


@pytest.mark.mcore
def test_clipped_pg_loss_sums_to_the_native_loss():
    """Token sums over the engine's slices, divided by the token count, equal NeMo RL's loss."""
    from types import SimpleNamespace

    from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn
    from nemo_rl.algorithms.loss.loss_functions import KL_PENALTY_KEY
    from nemo_rl.models.policy.training_engine_adapter.data import training_datums
    from nemo_rl.models.policy.training_engine_adapter.losses import (
        NeMoRLTrainingEngine,
    )

    loss_fn = ClippedPGLossFn(
        ClippedPGLossConfig(
            use_importance_sampling_correction=True,
            truncated_importance_sampling_type="tis",
            truncated_importance_sampling_ratio=2,
        )
    )
    data = _training_batch()
    current = -torch.rand(2, 5)
    num_tokens = (data["token_mask"] * data["sample_mask"].unsqueeze(-1)).sum()
    native_loss, native_metrics = loss_fn(
        next_token_logprobs=current,
        data=data,
        global_valid_seqs=torch.tensor(2.0),
        global_valid_toks=num_tokens,
    )

    # One packed row holding both Datums, cut into two slices as context parallelism does.
    datums = training_datums(data)
    packed = {
        name: torch.cat([getattr(d, name) for d in datums]).unsqueeze(0)
        for name in (
            "loss_mask",
            "advantages",
            "old_logprobs",
            "rollout_logprobs",
            "reference_logprobs",
        )
    }
    logprobs = torch.cat([current[0, :5], current[1, :3]]).unsqueeze(0)
    loss_sum = torch.zeros(())
    kl_sum = 0.0
    for columns in (slice(0, 3), slice(3, 8)):
        batch = SimpleNamespace(
            active_mask=packed["loss_mask"][:, columns].bool(),
            **{k: v[:, columns] for k, v in packed.items() if k != "loss_mask"},
        )
        loss, sums = NeMoRLTrainingEngine.clipped_pg_loss(
            None, batch, SimpleNamespace(logprobs=logprobs[:, columns]), loss_fn=loss_fn
        )
        loss_sum = loss_sum + loss
        kl_sum += float(sums[KL_PENALTY_KEY])

    assert loss_sum / num_tokens == pytest.approx(float(native_loss), abs=1e-5)
    assert kl_sum / float(num_tokens) == pytest.approx(
        native_metrics[KL_PENALTY_KEY], abs=1e-5
    )


@pytest.mark.mcore
def test_scoring_datums_reject_a_one_token_sample():
    from nemo_rl.models.policy.training_engine_adapter.data import scoring_datums

    data = BatchedDataDict(
        {"input_ids": torch.tensor([[10, 0]]), "input_lengths": torch.tensor([1])}
    )
    with pytest.raises(ValueError, match="at least 2"):
        scoring_datums(data)
