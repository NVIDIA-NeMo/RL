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

"""Data processors shared by the GenRM / DWRL VLM runners: `dwrl_pairwise_data_processor` is used by both
`examples/run_dwrl_pairwise_vlm.py` and `examples/run_genrm_pairwise_vlm.py`."""

import copy
from typing import Any

from nemo_rl.data.genrm_multimodal_utils import build_user_message, prepare_context_and_media
from nemo_rl.data.genrm_prompts import DWRL_AND_GENRM_PAIRWISE_RUBRIC_TEMPLATE, GENRM_PAIRWISE_DEFAULT_PRINCIPLE
from nemo_rl.data.interfaces import DatumSpec, TaskDataSpec
from nemo_rl.data.multimodal_utils import PackedTensor


def dwrl_pairwise_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process HelpSteer3 data (text or base64 multimodal) for DWRL GenRM training.

    Returns the pairwise judge prompt (both responses) as a one-message message log plus the native vLLM media keys
    (`vllm_content`, `vllm_images`, `vllm_videos`, `vllm_audios`); `rl_collate_fn` batches them as they are, and they are
    the per-row keys that the rollout reads."""
    # Extract the prompt (which contains the full formatted prompt with responses)
    context = datum_dict.get("context", "")
    resp1 = datum_dict.get("response_1", "")
    resp2 = datum_dict.get("response_2", "")
    principle = datum_dict.get("principle", GENRM_PAIRWISE_DEFAULT_PRINCIPLE)

    # Decode the base64 media once per datum: the prompt tokens and the native vLLM payload share it.
    context, images, videos, audios, video_cfg, sound_config = prepare_context_and_media(context, task_data_spec, tokenizer)

    prompt_str = DWRL_AND_GENRM_PAIRWISE_RUBRIC_TEMPLATE.format(context=context, response_1=resp1, response_2=resp2, principle=principle)

    # Prepare metadata for environment. The base64 payload stays out of it: the environment never reads the
    # context, and this dict is copied to every generation of the prompt and shipped to the reward workers.
    metadata = copy.deepcopy({key: value for key, value in datum_dict.items() if key != "context"})
    metadata["context"] = context
    #metadata['score_1'] = metadata['helpfulness_1']
    #metadata['score_2'] = metadata['helpfulness_2']
    #metadata['ranking'] = metadata['preference_ranking']

    # Extract system prompt if present
    # system_prompt = datum_dict.get("system_prompt", "")

    message_raw, vllm_content = build_user_message(prompt_str, images, videos, audios, tokenizer, video_cfg, sound_config)
    message_log = [message_raw]
    vllm_kwargs = {
        "vllm_content": vllm_content,
        "vllm_images": images,
        "vllm_videos": videos,  # [(frames, metadata)]: vLLM's native video item
        "vllm_audios": [(waveform, sound_config["sampling_rate"]) for waveform in audios],  # [(waveform, sample rate)]
    }

    # Calculate total length
    total_length = sum(len(msg["token_ids"]) for msg in message_log)

    # Check if we need to truncate
    loss_multiplier = 1.0
    if total_length > max_seq_length:
        # Truncate messages. Truncated rows are treated as text-only: their media tensors (and native vLLM media)
        # would no longer match the tokens, and the sample is masked out of the loss.
        for msg in message_log:
            msg["token_ids"] = msg["token_ids"][: min(4, max_seq_length // len(message_log)) ]
            for key, value in msg.items():
                if isinstance(value, PackedTensor):
                    msg[key] = PackedTensor.empty_like(value)
        vllm_kwargs = {"vllm_content": None, "vllm_images": [], "vllm_videos": [], "vllm_audios": []}
        loss_multiplier = 0.0
    # The environment only sees extra_env_info: a masked sample gets 0.0 there without being parsed
    metadata["loss_multiplier"] = loss_multiplier

    return DatumSpec(
        message_log=message_log,
        length=total_length,
        extra_env_info=metadata,
        loss_multiplier=loss_multiplier,
        idx=idx,
        task_name=task_data_spec.task_name,
        **vllm_kwargs,  # pyrefly: ignore[bad-unpacking]
    )
