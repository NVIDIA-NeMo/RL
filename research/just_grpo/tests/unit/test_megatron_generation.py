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


import pytest
import torch
from just_grpo.algorithms.block_just_grpo import BlockJustGRPOSchedule
from just_grpo.config import DiffusionSamplingParams, ScheduleConfig, validate_config
from just_grpo.diffusion.denoising_schedule import aggregate_diffusion_logprobs
from just_grpo.generation.megatron_generation import generate_responses, sample_batch
from test_schedule import TinyDiffusion
from test_sudoku_and_config import BASE, RECIPE, load, load_reference_vllm

from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class TinyDoubled:
    """Model with the native asymmetric block mask and a full logits output."""

    def __init__(self, model):
        self.model = model
        self.calls = []

    def __call__(self, ids, positions):
        self.calls.append((ids.clone(), positions.clone()))
        width = ids.shape[1] // 2
        q = torch.arange(2 * width)[:, None]
        k = torch.arange(2 * width)[None, :]
        mask = (
            ((q < width) & (k < width) & (q // 4 == k // 4))
            | ((q < width) & (k >= width) & (q // 4 > (k - width) // 4))
            | ((q >= width) & (k >= width) & (q >= k))
        )
        x = self.model.embedding(ids) + positions[..., None] / 100
        for layer in self.model.projections:
            q, k, v = layer(x).chunk(3, -1)
            x = (
                x
                + torch.nn.functional.scaled_dot_product_attention(
                    q[:, None], k[:, None], v[:, None], attn_mask=mask
                )[:, 0]
            )
        return self.model.head(x)


def params(**kwargs):
    return DiffusionSamplingParams(
        temperature=0.7,
        block_size=4,
        max_steps=4,
        selection_policy="leftmost",
        threshold=0.9,
        returns_entropy=False,
        returns_reveal_steps=False,
        **kwargs,
    )


@pytest.mark.parametrize("reveal", [1, 2, 4])
def test_sampled_logprobs_match_block_training(reveal):
    torch.manual_seed(7)
    model = TinyDiffusion()
    decoder = TinyDoubled(model)
    sampling = params().model_copy(update={"max_steps": 4 // reveal})
    prompts = torch.tensor([[31, 1, 2, 3], [31, 4, 5, 6]])
    responses = sample_batch(
        decoder,
        prompts,
        sampling=sampling,
        max_new_tokens=8,
        mask_token_id=31,
        stop_token_ids=[],
        max_sequence_length=16,
        generator=torch.Generator().manual_seed(19),
    )
    ids = torch.tensor(
        [p.tolist() + r["token_ids"] for p, r in zip(prompts, responses)]
    )
    mask = torch.zeros_like(ids)
    mask[:, 4:] = 1
    data = BatchedDataDict(
        input_ids=ids,
        input_lengths=torch.tensor([12, 12]),
        token_mask=mask,
        sample_mask=torch.ones(2),
    )
    schedule = BlockJustGRPOSchedule(
        data,
        ScheduleConfig(mask_token_id=31, block_size=4, reveal_tokens_per_step=reveal),
        pad_token_id=0,
    )

    # TinyDiffusion returns logprobs; obtain temperature-scaled logits by scaling its head.
    with torch.no_grad():
        model.head.weight.div_(sampling.temperature)
        model.head.bias.div_(sampling.temperature)
    recomputed = aggregate_diffusion_logprobs(
        schedule.iter_levels(),
        model,
        original_shape=ids.shape,
        device=torch.device("cpu"),
    )
    expected = torch.tensor([r["logprobs"] for r in responses])
    torch.testing.assert_close(recomputed[:, 4:], expected, atol=2e-6, rtol=1e-5)
    first, positions = decoder.calls[0]
    assert first.shape == (2, 32)
    assert (first[:, 4:16] == 31).all()
    torch.testing.assert_close(positions[:, :16], positions[:, 16:])
    for inputs, _ in decoder.calls:
        torch.testing.assert_close(inputs[:, :16], inputs[:, 16:])


@pytest.mark.parametrize("reveal", [1, 2, 4])
def test_independent_eos_and_sampled_mask_tokens(reveal):
    calls = []

    def forward(ids, positions):
        calls.append(ids.clone())
        logits = torch.full((*ids.shape, 8), -100.0)
        logits[0, :, 2] = 0
        logits[1, :, 7] = 0  # MASK is a valid sampled token.
        logits[1, 7, 7] = -100
        logits[1, 7, 2] = 0
        return logits

    result = sample_batch(
        forward,
        torch.tensor([[1] * 4, [3] * 4]),
        sampling=params().model_copy(
            update={"temperature": 0, "max_steps": 4 // reveal}
        ),
        max_new_tokens=8,
        max_sequence_length=12,
        mask_token_id=7,
        stop_token_ids=[2],
        generator=torch.Generator().manual_seed(1),
    )
    assert result[0]["token_ids"] == [2]
    assert result[1]["token_ids"] == [7, 7, 7, 2]
    assert all(r["finish_reason"] == "stop" for r in result)
    assert len(calls) == 4 // reveal
    assert all(len(r["logprobs"]) == len(r["token_ids"]) for r in result)


def test_seed_reproducibility_and_live_weights():
    model = TinyDiffusion()
    forward = TinyDoubled(model)
    kwargs = dict(
        sampling=params(),
        max_new_tokens=8,
        max_sequence_length=12,
        mask_token_id=31,
        stop_token_ids=[],
    )

    def run():
        return sample_batch(
            forward,
            torch.tensor([[1] * 4]),
            generator=torch.Generator().manual_seed(5),
            **kwargs,
        )

    first = run()
    assert first == run()
    assert first[0]["finish_reason"] == "length"
    with torch.no_grad():
        model.head.bias[7] += 100
    assert run()[0]["token_ids"] == [7] * 8


def test_mixed_lengths_microbatching_preserves_order():
    sizes = []

    def forward(ids, positions):
        sizes.append(len(ids))
        logits = torch.full((*ids.shape, 8), -100.0)
        logits.scatter_(-1, ids[:, :1, None].expand(-1, ids.shape[1], 1), 0.0)
        return logits

    prompts = [[1] * 4, [2] * 8, [3] * 4, [4] * 4]
    result = generate_responses(
        forward,
        prompts,
        batch_size=2,
        device=torch.device("cpu"),
        seed=1,
        sampling=params().model_copy(update={"temperature": 0}),
        max_new_tokens=4,
        max_sequence_length=12,
        mask_token_id=7,
        stop_token_ids=[],
    )
    assert [r["token_ids"] for r in result] == [[i] * 4 for i in range(1, 5)]
    assert max(sizes) == 2


@pytest.mark.parametrize("prompt_length,context", [(3, 12), (4, 8), (4, 13)])
def test_invalid_alignment_or_context_fails_before_forward(prompt_length, context):
    def forward(*args):
        pytest.fail("Invalid context must fail before the model runs")

    with pytest.raises(ValueError):
        sample_batch(
            forward,
            torch.ones((1, prompt_length), dtype=torch.long),
            sampling=params(),
            max_new_tokens=8,
            max_sequence_length=context,
            mask_token_id=31,
            stop_token_ids=[],
            generator=torch.Generator(),
        )


@pytest.mark.parametrize("path", [BASE, RECIPE])
def test_backend_configs_preserve_experiment_settings(path):
    vllm = load_reference_vllm(path)
    megatron = load(path)
    validate_config(vllm)
    assert validate_config(megatron).runtime == "megatron"
    assert megatron.just_grpo.generation_python is None
    assert megatron.policy.generation.backend == "megatron"
    assert megatron.policy.megatron_cfg.tensor_model_parallel_size == 1
    for key in ("grpo", "loss_fn", "data", "cluster"):
        assert vllm[key] == megatron[key]
    assert vllm.policy.optimizer == megatron.policy.optimizer
    assert vllm.just_grpo.sampling == megatron.just_grpo.sampling


def test_megatron_generation_rejects_tp2():
    config = load()
    config.policy.megatron_cfg.tensor_model_parallel_size = 2
    with pytest.raises(ValueError, match="TP=1"):
        validate_config(config)
