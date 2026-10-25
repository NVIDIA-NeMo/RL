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

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from nemo_rl.algorithms.grpo import add_grpo_token_loss_masks_and_generation_logprobs
from nemo_rl.algorithms.logits_sampling_utils import TrainingSamplingParams
from nemo_rl.algorithms.loss.clipped_pg import score_centered_actor_objective
from nemo_rl.algorithms.loss.loss_functions import ClippedPGLossConfig, ClippedPGLossFn
from nemo_rl.algorithms.loss.loss_input import prepare_loss_input
from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.nemo_gym import NemoGym
from nemo_rl.models.generation.vllm.vllm_worker import BaseVllmGenerationWorker


@pytest.mark.parametrize("mode", [None, "tis", "icepop"])
def test_score_centering_matches_full_vocabulary_gradient(mode):
    logits = torch.tensor([0.4, -0.2, 0.1, 0.5, -0.5], requires_grad=True)
    logp = logits.log_softmax(-1)
    p = logp.exp().detach()
    q = torch.cat((torch.tensor([0.38, 0.17]), 0.45 * p[2:] / p[2:].sum()))
    logq = q.log()
    sampled = 3
    high, low = 1.2, 0.8

    actual, head_mass, _ = score_centered_actor_objective(
        logp[sampled].reshape(1, 1),
        logp[[0, 1, 0]].reshape(1, 1, 3),
        logq[sampled].reshape(1, 1),
        torch.tensor([[[logq[0], logq[1], float("-inf")]]]),
        torch.ones(1, 1),
        torch.ones(1, 1),
        torch.tensor(1.0),
        use_importance_sampling_correction=mode is not None,
        truncated_importance_sampling_type=mode,
        truncated_importance_sampling_ratio=high,
        truncated_importance_sampling_ratio_min=low,
    )
    actual_grad = torch.autograd.grad(actual, logits, retain_graph=True)[0]

    ratio = (p / q).detach()
    if mode == "tis":
        weight = ratio.clamp(low, high)
    elif mode == "icepop":
        weight = ratio * ((ratio >= low) & (ratio <= high))
    elif mode is None:
        weight = torch.ones_like(ratio)
    expected = -(weight[sampled] * logp[sampled] - (q * weight * logp).sum())
    expected_grad = torch.autograd.grad(expected, logits)[0]

    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(head_mass, q[:2].sum())


def test_score_centering_replayed_support_matches_exact_gradient():
    logits = torch.tensor([0.1, -0.4, 0.5, 0.2, -0.2], requires_grad=True)
    support = torch.tensor([0, 2, 4])
    logp = logits[support].log_softmax(-1)
    p = logp.exp().detach()
    q = torch.tensor([0.6, 0.25, 0.15])
    logq = q.log()
    ratio = p / q
    weights = ratio * ((ratio >= 0.5) & (ratio <= 5.0))

    actual, _, _ = score_centered_actor_objective(
        logp[1].reshape(1, 1),
        logp.reshape(1, 1, 3),
        logq[1].reshape(1, 1),
        logq.reshape(1, 1, 3),
        torch.ones(1, 1),
        torch.ones(1, 1),
        torch.tensor(1.0),
        use_importance_sampling_correction=True,
        truncated_importance_sampling_type="icepop",
        truncated_importance_sampling_ratio=5.0,
        truncated_importance_sampling_ratio_min=0.5,
    )
    actual_grad = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    expected = -(weights[1] * logp[1] - (q * weights * logp).sum())
    expected_grad = torch.autograd.grad(expected, logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("support_replay", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_score_centering_uses_policy_head_logprobs(support_replay):
    logits = torch.randn(1, 3, 8, device="cuda", requires_grad=True)
    ids = torch.tensor([[[0, 0, 0], [2, 5, 6], [1, 4, 7]]], device="cuda")
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[3, 6, 7]], device="cuda"),
            "score_centering_ids": ids,
            "score_centering_logprobs": torch.tensor(
                [[[float("-inf")] * 3, [-0.2, -2.0, -2.2], [-0.3, -1.7, -2.1]]],
                device="cuda",
            ),
        }
    )
    loss_fn = ClippedPGLossFn(
        ClippedPGLossConfig(
            score_centering_top_k=3,
            score_centering_support_replay=support_replay,
            disable_ppo_ratio=True,
            reference_policy_kl_penalty=0,
        )
    )
    _, prepared = prepare_loss_input(logits, data, loss_fn)
    expected = logits.log_softmax(-1)[:, :2].gather(-1, ids[:, 1:])
    if support_replay:
        expected = expected - expected.logsumexp(-1, keepdim=True)
        torch.testing.assert_close(
            prepared["curr_logprobs_unfiltered"],
            logits.log_softmax(-1)[:, :2]
            .gather(-1, data["input_ids"][:, 1:, None])
            .squeeze(-1),
        )
    torch.testing.assert_close(prepared["score_centering_policy_logprobs"], expected)
    prepared["score_centering_policy_logprobs"].sum().backward()
    assert logits.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("metrics_level", ["minimal", "full"])
@pytest.mark.parametrize(
    "backend",
    [
        "loss",
        pytest.param("automodel", marks=pytest.mark.automodel),
        pytest.param("megatron", marks=pytest.mark.mcore),
    ],
)
def test_score_centering_grpo_loss_backpropagates(metrics_level, backend, monkeypatch):
    logits = torch.randn(1, 3, 8, device="cuda", requires_grad=True)
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[3, 6, 7]], device="cuda"),
            "score_centering_ids": torch.tensor(
                [[[0, 0], [2, 6], [1, 7]]], device="cuda"
            ),
            "score_centering_logprobs": torch.tensor(
                [[[float("-inf")] * 2, [-0.4, -1.1], [-0.5, -1.0]]],
                device="cuda",
            ),
            "generation_logprobs": torch.tensor([[0.0, -1.1, -1.0]], device="cuda"),
            "prev_logprobs": torch.tensor([[0.0, -1.2, -1.3]], device="cuda"),
            "reference_policy_logprobs": torch.zeros(1, 3, device="cuda"),
            "advantages": torch.tensor([[0.0, 1.0, -1.0]], device="cuda"),
            "token_mask": torch.tensor([[0.0, 1.0, 1.0]], device="cuda"),
            "sample_mask": torch.ones(1, device="cuda"),
        }
    )
    loss_fn = ClippedPGLossFn(
        ClippedPGLossConfig(
            score_centering_top_k=2,
            disable_ppo_ratio=True,
            reference_policy_kl_penalty=0,
            use_importance_sampling_correction=True,
            truncated_importance_sampling_type="icepop",
            truncated_importance_sampling_ratio_min=0.5,
            truncated_importance_sampling_ratio=5.0,
            metrics_level=metrics_level,
        )
    )
    counts = {
        "global_valid_seqs": torch.tensor(1.0, device="cuda"),
        "global_valid_toks": torch.tensor(2.0, device="cuda"),
    }
    loss_input, prepared = prepare_loss_input(logits, data, loss_fn)
    expected_loss, expected_metrics = loss_fn(**loss_input, data=prepared, **counts)
    expected_grad = torch.autograd.grad(expected_loss, logits, retain_graph=True)[0]
    sampling_params = TrainingSamplingParams(temperature=0.7)
    if backend == "loss":
        loss, metrics = expected_loss, expected_metrics
    elif backend == "automodel":
        # Deferred: each backend has its own optional dependencies and test marker.
        from nemo_rl.models.automodel import train
        from nemo_rl.models.automodel.data import ProcessedInputs, ProcessedMicrobatch

        inputs = ProcessedInputs(input_ids=data["input_ids"], seq_len=3)
        mb = ProcessedMicrobatch(
            data, inputs, original_batch_size=1, original_seq_len=3
        )
        model = torch.nn.Identity()
        monkeypatch.setattr(train, "model_forward", lambda *_: logits)
        monkeypatch.setattr(train, "extract_logits", lambda _, output: output)
        processor = train.LossPostProcessor(
            loss_fn,
            {},
            cp_mesh=None,
            cp_size=1,
            dp_size=1,
            sampling_params=sampling_params,
        )
        loss, metrics, _ = train.forward_with_post_processing_fn(
            model,
            train.prepare_model_forward(
                model,
                inputs,
                device_mesh=None,
                cp_size=1,
                padding_token_id=0,
                is_reward_model=False,
                allow_flash_attn_args=False,
            ),
            processor,
            mb,
            sampling_params=sampling_params,
            **counts,
        )
    else:
        from nemo_rl.models.megatron import train
        from nemo_rl.models.megatron.data import ProcessedMicrobatch

        mb = ProcessedMicrobatch(
            data_dict=data,
            input_ids=data["input_ids"],
            input_ids_cp_sharded=data["input_ids"],
            attention_mask=None,
            position_ids=None,
            packed_seq_params=None,
            cu_seqlens_padded=None,
        )
        monkeypatch.setattr(train, "model_forward", lambda *args, **kwargs: logits)
        monkeypatch.setattr(train, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(train, "get_tensor_model_parallel_group", lambda: None)
        monkeypatch.setattr(train, "get_context_parallel_group", lambda: None)
        monkeypatch.setattr(train, "get_context_parallel_world_size", lambda: 1)
        processor = train.LossPostProcessor(
            loss_fn,
            {"sequence_packing": {"enabled": False}},
            sampling_params=sampling_params,
        )
        output, finish = train.forward_with_post_processing_fn(
            iter([mb]),
            torch.nn.Identity(),
            processor,
            sampling_params=sampling_params,
            **counts,
        )
        loss, metrics = finish(output)

    torch.testing.assert_close(loss, expected_loss)
    assert torch.isfinite(loss)
    assert "score_centering_head_mass" in metrics
    if metrics_level == "full":
        torch.testing.assert_close(
            metrics["probs_ratio_clamped"], metrics["probs_ratio"]
        )
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    torch.testing.assert_close(logits.grad, expected_grad)


def test_score_centering_requires_reinforce_and_logits():
    with pytest.raises(ValueError, match="disable_ppo_ratio"):
        ClippedPGLossFn(ClippedPGLossConfig(score_centering_top_k=2))
    with pytest.raises(ValueError, match="model logits"):
        ClippedPGLossFn(
            ClippedPGLossConfig(score_centering_top_k=2, disable_ppo_ratio=True),
            use_fused_linear_logprobs=True,
        )


def test_score_centering_uses_realized_sampling_support():
    values = [
        {2: SimpleNamespace(logprob=-0.2), 5: SimpleNamespace(logprob=-1.7)},
        {1: SimpleNamespace(logprob=-0.4), 4: SimpleNamespace(logprob=-1.1)},
    ]
    ids, logprobs = BaseVllmGenerationWorker._score_centering_head(
        values,
        2,
        3,
        device=torch.device("cpu"),
        sampling_mask=[[5, 2], [1, 4]],
        sampled_tokens=[2, 4],
    )
    torch.testing.assert_close(ids, torch.tensor([[5, 2, 0], [1, 4, 0]]))
    torch.testing.assert_close(
        logprobs,
        torch.tensor([[-1.7, -0.2, float("-inf")], [-0.4, -1.1, float("-inf")]]),
    )
    with pytest.raises(RuntimeError, match="full sampling support"):
        BaseVllmGenerationWorker._score_centering_head(
            values,
            2,
            3,
            device=torch.device("cpu"),
            sampling_mask=[[5, 2], [1, 7]],
            sampled_tokens=[2, 1],
        )


def test_score_centering_fields_survive_message_flattening():
    message_logs = [
        [
            {"role": "user", "token_ids": torch.tensor([1, 2])},
            {
                "role": "assistant",
                "token_ids": torch.tensor([3, 4]),
                "generation_logprobs": torch.tensor([-0.4, -0.5]),
                "score_centering_ids": torch.tensor([[3, 5], [4, 6]]),
                "score_centering_logprobs": torch.tensor([[-0.4, -1.1], [-0.5, -1.0]]),
            },
        ]
    ]
    add_grpo_token_loss_masks_and_generation_logprobs(message_logs)
    flat, _ = batched_message_log_to_flat_message(message_logs)
    torch.testing.assert_close(
        flat["score_centering_ids"], torch.tensor([[[0, 0], [0, 0], [3, 5], [4, 6]]])
    )
    torch.testing.assert_close(
        flat["score_centering_logprobs"][0, 2:],
        torch.tensor([[-0.4, -1.1], [-0.5, -1.0]]),
    )


def test_score_centering_gym_rollout_fields_survive_flattening():
    class _Tokenizer:
        def batch_decode(self, batch):
            return ["" for _ in batch]

    class _Gym:
        cfg = {"score_centering_top_k": 3}

    result = (
        NemoGym.__ray_metadata__.modified_class._postprocess_nemo_gym_to_nemo_rl_result(
            _Gym(),
            {},
            {
                "responses_create_params": {"input": []},
                "response": {
                    "output": [
                        {
                            "type": "message",
                            "prompt_token_ids": [1],
                            "generation_token_ids": [2],
                            "generation_log_probs": [-0.2],
                            "score_centering_ids": [[2, 3]],
                            "score_centering_logprobs": [[-0.2, -1.7]],
                        }
                    ]
                },
            },
            _Tokenizer(),
        )
    )
    message_logs = [result["message_log"]]
    add_grpo_token_loss_masks_and_generation_logprobs(message_logs)
    flat, _ = batched_message_log_to_flat_message(message_logs)
    torch.testing.assert_close(
        flat["score_centering_ids"][0, 1], torch.tensor([2, 3, 0])
    )
    torch.testing.assert_close(
        flat["score_centering_logprobs"][0, 1],
        torch.tensor([-0.2, -1.7, float("-inf")]),
    )


def _dtensor_score_centering_worker(rank, init_file):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl", init_method=f"file://{init_file}", rank=rank, world_size=2
    )
    try:
        mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("tp",))
        torch.manual_seed(7)
        full_logits = torch.randn(1, 3, 8, device="cuda", requires_grad=True)
        local_logits = (
            full_logits.detach()[..., rank * 4 : (rank + 1) * 4]
            .contiguous()
            .requires_grad_()
        )
        logits = DTensor.from_local(local_logits, mesh, [Shard(2)])
        ids = torch.tensor([[[0, 0], [2, 5], [1, 4]]], device="cuda")
        data = BatchedDataDict(
            {
                "input_ids": torch.tensor([[3, 6, 7]], device="cuda"),
                "score_centering_ids": ids,
                "score_centering_logprobs": torch.full(
                    (1, 3, 2), float("-inf"), device="cuda"
                ),
            }
        )
        loss_fn = ClippedPGLossFn(
            ClippedPGLossConfig(
                score_centering_top_k=2,
                disable_ppo_ratio=True,
                reference_policy_kl_penalty=0,
            )
        )
        loss_input, prepared = prepare_loss_input(logits, data, loss_fn)
        expected_logprobs = full_logits.log_softmax(-1)[:, :2]
        expected_head = expected_logprobs.gather(-1, ids[:, 1:])
        expected_sample = expected_logprobs.gather(
            -1, data["input_ids"][:, 1:, None]
        ).squeeze(-1)
        torch.testing.assert_close(
            prepared["score_centering_policy_logprobs"], expected_head
        )
        torch.testing.assert_close(loss_input["next_token_logprobs"], expected_sample)

        (
            prepared["score_centering_policy_logprobs"].sum()
            + loss_input["next_token_logprobs"].sum()
        ).backward()
        (expected_head.sum() + expected_sample.sum()).backward()
        torch.testing.assert_close(
            local_logits.grad, full_logits.grad[..., rank * 4 : (rank + 1) * 4]
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
def test_score_centering_dtensor_tensor_parallel(tmp_path):
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(
            target=_dtensor_score_centering_worker,
            args=(rank, str(tmp_path / "init")),
        )
        for rank in range(2)
    ]
    for process in procs:
        process.start()
    for process in procs:
        process.join(timeout=120)
        assert process.exitcode == 0
