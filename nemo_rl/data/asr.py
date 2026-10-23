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

"""ASR Parquet row processing for the standard multimodal rollout transport."""

from typing import Any

import numpy as np
import soundfile as sf
import torch

from nemo_rl.data.interfaces import DatumSpec, TaskDataSpec
from nemo_rl.data.multimodal_utils import PackedTensor


def asr_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    processor: Any,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Render text before audio and keep the reference exclusively in reward metadata."""
    if datum_dict.get("system_prompt") or task_data_spec.system_prompt:
        raise ValueError("SpeechLM ASR expects an empty system prompt")
    if datum_dict.get("input_order", "text_before_audio") != "text_before_audio":
        raise ValueError("SpeechLM ASR requires input_order=text_before_audio")
    audio_path = datum_dict.get("audio_filepath") or datum_dict.get("audio_path")
    if not audio_path:
        raise ValueError("ASR row must contain audio_filepath or audio_path")
    audio, sampling_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    if sampling_rate != 16000 or len(audio) == 0:
        raise ValueError("SpeechLM ASR requires nonempty 16-kHz audio")
    waveform = audio.mean(axis=1)
    instruction = datum_dict.get("instruction") or datum_dict.get("prompt")
    if not isinstance(instruction, str) or not instruction:
        raise ValueError("ASR row must contain a nonempty instruction or prompt")
    reference = datum_dict.get("reference")
    if reference is None:
        reference = datum_dict.get("expected_answer")
    if not isinstance(reference, str):
        raise ValueError(
            "ASR row must contain a separate string reference or expected_answer"
        )
    tokenizer = processor.tokenizer
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction + "\n<|audio|>"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    token_ids = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ][0]
    if len(token_ids) >= max_seq_length:
        raise ValueError(
            f"ASR prompt length {len(token_ids)} exceeds max_input_seq_length={max_seq_length}"
        )
    info: dict[str, Any] = {
        "ground_truth": reference,
        "audio_filepath": str(audio_path),
        "cut_id": str(datum_dict.get("cut_id") or datum_dict.get("id") or idx),
    }
    for key in (
        "language",
        "lang",
        "task",
        "reward_metric",
        "cer_whitespace_mode",
        "reward_normalizer",
    ):
        if datum_dict.get(key) is not None:
            info[key] = datum_dict[key]
    params = datum_dict.get("responses_create_params")
    if params is not None:
        cap = params.get("max_output_tokens")
        if cap is not None:
            if type(cap) is not int or cap < 1:
                raise ValueError("max_output_tokens must be a positive integer")
            info["max_output_tokens"] = cap
    message = {
        "role": "user",
        "content": rendered,
        "token_ids": token_ids,
        "audio_samples": PackedTensor(
            torch.from_numpy(waveform.copy()).unsqueeze(0),
            dim_to_pack=0,
            preprocess_mode="pad_to_max_shape",
        ),
        "audio_lengths": PackedTensor(torch.tensor([len(waveform)]), dim_to_pack=0),
    }
    return {
        "message_log": [message],
        "length": len(token_ids),
        "extra_env_info": info,
        "loss_multiplier": 1.0,
        "idx": idx,
        "task_name": task_data_spec.task_name or "asr",
        "vllm_content": rendered,
        "vllm_multi_modal_data": {
            "audio": [(np.array(waveform, copy=True), sampling_rate)]
        },
    }


class SpeechLMProcessor:
    """Expose a native HF tokenizer without loading the speech encoder."""

    model_input_names = ["input_ids", "audio_samples", "audio_lengths"]

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, name: str, **kwargs: Any) -> "SpeechLMProcessor":
        from transformers import AutoTokenizer  # Tokenizer load is optional at import.

        return cls(AutoTokenizer.from_pretrained(name, **kwargs))
