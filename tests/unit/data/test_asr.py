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
    rollout_waveform, rate = result["vllm_multi_modal_data"]["audio"]
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

    audio = PackedTensor([torch.ones(1, 3), torch.ones(1, 5)], 0, pad_to_max_shape=True)
    padded = audio.as_tensor()
    assert padded.shape == (2, 5)
    assert torch.equal(padded[0], torch.tensor([1., 1., 1., 0., 0.]))
    assert audio.slice([0, 1]).pad_to_max_shape
    assert PackedTensor.empty_like(audio).pad_to_max_shape
    for combine in (PackedTensor.concat, PackedTensor.flattened_concat):
        result = combine([audio.slice([0]), audio.slice([1])])
        assert result.pad_to_max_shape
        assert torch.equal(result.as_tensor(), padded)
        with pytest.raises(ValueError, match="padding mode"):
            combine([audio, PackedTensor(torch.ones(1, 5), 0)])
