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
"""Contracts between research adapters and upstream GRPO rollouts."""

from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import torch
from just_grpo.environments.sudoku import SudokuEnvironmentImpl, SudokuResponseDataset
from just_grpo.generation.megatron_generation import (
    MegatronDiffusionGeneration,
    pack_responses,
)
from test_sudoku_and_config import answer, load, load_reference_vllm


def test_sudoku_round_trip_preserves_aligned_prompt_and_reward():
    config = load()
    tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw: [7] * 19)
    dataset = SudokuResponseDataset(config, tokenizer)
    sample = dataset[0]
    assert sample["length"] == 32
    ids = sample["message_log"][0]["token_ids"]
    assert ids[:13].tolist() == [config.just_grpo.schedule.mask_token_id] * 13
    assert ids[13:].tolist() == [7] * 19
    example = sample["extra_env_info"]["example"]
    message = sample["message_log"] + [
        {"role": "assistant", "content": answer(example.solution)}
    ]
    result = SudokuEnvironmentImpl().step([message], [sample["extra_env_info"]])
    assert result.rewards.tolist() == [1.0]
    assert result.terminateds.tolist() == [True]


def test_generation_output_keeps_sampled_logprobs_at_token_positions():
    prompts = [[1, 2, 3], [4]]
    responses = [
        {"token_ids": [8, 9], "logprobs": [-0.2, -0.3]},
        {"token_ids": [7, 0], "logprobs": [-0.4, -0.5]},
    ]
    result = pack_responses(
        prompts, responses, pad_token_id=0, max_new_tokens=2, stop_token_ids=[0]
    )
    assert result["output_ids"].tolist() == [[1, 2, 3, 8, 9], [4, 7, 0, 0, 0]]
    assert result["unpadded_sequence_lengths"].tolist() == [5, 3]
    assert result["generation_lengths"].tolist() == [2, 2]
    assert result["truncated"].tolist() == [True, False]
    torch.testing.assert_close(
        result["logprobs"],
        torch.tensor([[0.0, 0.0, 0.0, -0.2, -0.3], [0.0, -0.4, -0.5, 0.0, 0.0]]),
    )


def test_generation_reuses_policy_dispatch_and_native_offload():
    policy = Mock()
    generation = MegatronDiffusionGeneration(
        policy=policy,
        config=SimpleNamespace(
            policy={
                "generation": {
                    "backend": "megatron",
                    "refit_transport": "mcore",
                    "mcore_generation_config": {
                        "refit_backend": "gloo",
                        "expose_http_server": False,
                    },
                },
                "logprob_batch_size": 2,
            }
        ),
    )
    assert generation.weight_synchronizer is None
    generation.prepare_for_generation()
    policy.prepare_for_lp_inference.assert_called_once()
    batch = Mock()
    assert generation.generate(batch) is policy.generate.return_value
    policy.generate.assert_called_once_with(batch, greedy=False)


@pytest.mark.parametrize("backend", ["vllm", "megatron"])
def test_native_setup_selects_vllm_or_megatron_generation(monkeypatch, backend):
    pytest.importorskip(
        "soundfile", reason="Full GRPO controller needs the NeMo-RL runtime"
    )
    from omegaconf import OmegaConf

    from nemo_rl.algorithms import grpo

    config = load() if backend == "megatron" else load_reference_vllm()
    master = grpo.MasterConfig(**OmegaConf.to_container(config, resolve=True))
    checkpointer = Mock()
    checkpointer.get_latest_checkpoint_path.return_value = None
    checkpointer.load_training_info.return_value = None
    checkpointer.get_resume_paths.return_value = (None, None)
    monkeypatch.setattr(grpo, "CheckpointManager", Mock(return_value=checkpointer))
    monkeypatch.setattr(grpo, "Logger", Mock())
    from unittest.mock import MagicMock

    loader = MagicMock()
    loader.__len__.return_value = 1
    monkeypatch.setattr(grpo, "StatefulDataLoader", Mock(return_value=loader))
    monkeypatch.setattr(grpo, "RayVirtualCluster", Mock())
    engine = Mock()
    monkeypatch.setattr(
        grpo,
        "MegatronGeneration" if backend == "megatron" else "VllmGeneration",
        engine,
    )
    native_policy = Mock()
    monkeypatch.setattr(grpo, "Policy", native_policy)
    result = grpo.setup(master, Mock(), [Mock()], [Mock()])
    native_policy.assert_called_once()
    engine.assert_called_once()
    assert result[0] is native_policy.return_value
    assert result[1] is engine.return_value
    if backend == "megatron":
        assert engine.call_args.kwargs["policy"] is native_policy.return_value
    else:
        engine.return_value.finish_generation.assert_called_once()


def test_native_colocated_sync_uses_live_policy_without_weight_export():
    from nemo_rl.weight_sync.megatron_weight_synchronizer import (
        MegatronWeightSynchronizer,
    )

    policy = Mock()
    generation = MegatronDiffusionGeneration(
        policy=policy,
        config=SimpleNamespace(
            policy={
                "generation": {
                    "backend": "megatron",
                    "refit_transport": "mcore",
                    "mcore_generation_config": {
                        "refit_backend": "gloo",
                        "expose_http_server": False,
                    },
                },
                "logprob_batch_size": 2,
            }
        ),
    )
    sync = MegatronWeightSynchronizer(policy, generation, colocated=True)
    sync.init_communicator()
    assert sync.is_stale
    assert sync.sync_weights() == {}
    assert not sync.is_stale
    policy.offload_before_refit.assert_called_once()
    policy.prepare_for_lp_inference.assert_called_once()
    assert policy.mock_calls == [
        call.offload_before_refit(),
        call.prepare_for_lp_inference(),
    ]
