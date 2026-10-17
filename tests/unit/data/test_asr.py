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

import numpy as np
import pytest
import soundfile as sf
import torch

from nemo_rl.data.asr import asr_data_processor
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.environments.asr_reward import asr_reward, normalize_asr_tokens


class FakeTokenizer:
    def __init__(self):
        self.messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        return messages[0]["content"]

    def __call__(self, prompt, **kwargs):
        return {"input_ids": torch.tensor([[3, 4, 5]])}


def test_asr_reference_never_enters_prompt_and_audio_buffers_are_owned(tmp_path):
    path = tmp_path / "a.wav"
    sf.write(path, np.zeros((160, 2), dtype=np.float32), 16000)
    tokenizer = FakeTokenizer()
    row = {
        "audio_filepath": str(path),
        "instruction": "Transcribe",
        "expected_answer": "PRIVATE_REFERENCE",
        "language": "hi",
        "task": "ordinary_asr",
        "reward_metric": "wer",
        "responses_create_params": {"max_output_tokens": 17},
    }
    result = asr_data_processor(
        row, TaskDataSpec(task_name="asr"), SimpleNamespace(tokenizer=tokenizer), 100, 5
    )
    assert result["vllm_content"] == "Transcribe\n<|audio|>"
    assert tokenizer.messages == [{"role": "user", "content": "Transcribe\n<|audio|>"}]
    assert result["extra_env_info"]["ground_truth"] == "PRIVATE_REFERENCE"
    assert result["extra_env_info"]["max_output_tokens"] == 17
    assert result["extra_env_info"]["language"] == "hi"
    policy_waveform = result["message_log"][0]["audio_samples"].as_tensor()
    rollout_waveform, rate = result["vllm_multi_modal_data"]["audio"][0]
    assert rate == 16000
    rollout_waveform[:] = 1
    assert torch.count_nonzero(policy_waveform) == 0


@pytest.mark.parametrize(
    "reference,hypothesis,reward",
    [
        ("", "", 1),
        ("", "a b c", -2),
        ("spoken words", "", 0),
        ("a b", "a b c", 0.5),
        ("नमस्ते", "नमस्ते", 1),
        ("a", "a b c d", -2),
        ("<spk:2>", "spk 2", 1),
    ],
)
def test_wer_contract(reference, hypothesis, reward):
    assert asr_reward(reference, hypothesis, {}) == reward


def test_unicode_marks_and_no_nfkc():
    assert normalize_asr_tokens("नमस्ते दुनिया") == ["नमस्ते", "दुनिया"]
    assert normalize_asr_tokens("Ａ") != normalize_asr_tokens("A")


def test_mandarin_routing():
    metadata = {
        "language": "cmn",
        "task": "ordinary_asr",
        "reward_metric": "cer",
        "cer_whitespace_mode": "han_adjacent",
    }
    assert asr_reward("今 天 use AI", "今天use AI", metadata) == 1
    with pytest.raises(ValueError):
        asr_reward("你好", "你好", {**metadata, "reward_metric": "wer"})
    with pytest.raises(ValueError):
        asr_reward("你好", "你好", {**metadata, "task": "sot"})
    with pytest.raises(ValueError):
        asr_reward("", "你好", metadata)


def test_incompatible_declared_input_order_is_rejected():
    with pytest.raises(ValueError, match="input_order"):
        asr_data_processor(
            {"input_order": "audio_before_text"},
            TaskDataSpec(task_name="asr"),
            SimpleNamespace(tokenizer=FakeTokenizer()),
            100,
            0,
        )


def test_audio_padding_survives_batch_operations():
    from nemo_rl.data.multimodal_utils import PackedTensor

    audio = PackedTensor(
        [torch.ones(1, 3), torch.ones(1, 5)], 0, preprocess_mode="pad_to_max_shape"
    )
    padded = audio.as_tensor()
    assert padded.shape == (2, 5)
    assert torch.equal(padded[0], torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0]))
    assert audio.slice([0, 1]).preprocess_mode == "pad_to_max_shape"
    assert PackedTensor.empty_like(audio).preprocess_mode == "pad_to_max_shape"
    for combine in (PackedTensor.concat, PackedTensor.flattened_concat):
        result = combine([audio.slice([0]), audio.slice([1])])
        assert result.preprocess_mode == "pad_to_max_shape"
        assert torch.equal(result.as_tensor(), padded)
        with pytest.raises(AssertionError, match="preprocess"):
            combine([audio, PackedTensor(torch.ones(1, 5), 0)])


def test_asr_audio_survives_collation_group_expansion_and_vllm_formatting(tmp_path):
    from nemo_rl.data.collate_fn import rl_collate_fn
    from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.models.generation.vllm.utils import format_prompt_for_vllm_generation

    waveforms = [
        np.linspace(-0.5, 0.5, 160, dtype=np.float32),
        np.linspace(0.25, -0.25, 240, dtype=np.float32),
    ]
    data = []
    for index, waveform in enumerate(waveforms):
        path = tmp_path / f"audio-{index}.wav"
        sf.write(path, waveform, 16000, subtype="FLOAT")
        data.append(
            asr_data_processor(
                {
                    "audio_filepath": str(path),
                    "instruction": f"Transcribe {index}",
                    "expected_answer": f"PRIVATE_REFERENCE_{index}",
                },
                TaskDataSpec(task_name="asr"),
                SimpleNamespace(tokenizer=FakeTokenizer()),
                100,
                index,
            )
        )
    # Use the production batch operations and message flattening used by rollouts.
    batch = rl_collate_fn(data).repeat_interleave(2).select_indices([3, 0, 2, 1])
    flat, lengths = batched_message_log_to_flat_message(
        batch["message_log"], pad_value_dict={"token_ids": 0}
    )
    generation_input = BatchedDataDict(
        input_ids=flat["token_ids"], input_lengths=lengths
    )
    generation_input.update(flat.get_multimodal_dict(as_tensors=False))
    for key in ("vllm_content", "vllm_multi_modal_data"):
        generation_input[key] = batch.get(key)
    prompts = format_prompt_for_vllm_generation(generation_input)
    for position, original_index in enumerate([1, 0, 1, 0]):
        prompt = prompts[position]
        assert prompt["prompt"] == f"Transcribe {original_index}\n<|audio|>"
        assert "PRIVATE_REFERENCE" not in prompt["prompt"]
        assert set(prompt["multi_modal_data"]) == {"audio"}
        ((audio, rate),) = prompt["multi_modal_data"]["audio"]
        assert rate == 16000 and audio.dtype == np.float32
        np.testing.assert_array_equal(audio, waveforms[original_index])
        policy_audio = batch["message_log"][position][0]["audio_samples"].as_tensor()
        assert policy_audio.dtype == torch.float32
        torch.testing.assert_close(policy_audio.squeeze(0), torch.from_numpy(audio))
    assert (
        format_prompt_for_vllm_generation(generation_input, sample_idx=2)["prompt"]
        == prompts[2]["prompt"]
    )
    # Rollout buffers must not alias native-policy audio storage.
    prompts[0]["multi_modal_data"]["audio"][0][0][:] = 0
    assert (
        torch.count_nonzero(data[1]["message_log"][0]["audio_samples"].as_tensor()) > 0
    )
