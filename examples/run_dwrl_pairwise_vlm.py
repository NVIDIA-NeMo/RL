# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import base64
import copy
import functools
import io
import math
import os
import re
import pprint
import time
import wave
from typing import Any, Optional
from collections import defaultdict
from PIL import Image
import numpy as np
import torch

from omegaconf import OmegaConf
from transformers import PretrainedConfig, PreTrainedTokenizerBase

from nemo_rl.algorithms.dwrl_pairwise import dwrl_train_pairwise
from nemo_rl.algorithms.grpo import MasterConfig, setup, shutdown_environments
from nemo_rl.algorithms.loss import DWRLPairwiseLossFn
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.interfaces import DatumSpec, TaskDataSpec, TaskDataProcessFnCallable
from nemo_rl.data.multimodal_utils import (
    AUDIO_CONTENT_TYPES,
    IMAGE_CONTENT_TYPES,
    VIDEO_CONTENT_TYPES,
    PackedTensor,
    _materialize_ragged_pixel_values,
    extract_multimodal_model_inputs,
    uses_image_placeholder,
)
from nemo_rl.environments.nemotron_utils import process_nemotron_video_frames
from nemo_rl.models.generation.vllm.video_utils import (
    _build_video_metadata,
    _decode_torchcodec_video,
    _torchcodec_sample_indices,
)
#from nemo_rl.data.utils import setup_response_data
from nemo_rl.distributed.ray_actor_environment_registry import (
    get_actor_python_env,
)
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir, log_container_init_timing
from nemo_rl.utils.timer import Timer
from nemo_rl.data.datasets.preference_datasets.reward_benchmarks import HelpSteer3LocalDataset
#from nemo_rl.environments.binary_genrm_environment import (
#    GenRMEnvironment,
#    format_scoring_stage_prompt,
#)
#from nemo_rl.environments.genrm_environment import GenRMEnvironment
from nemo_rl.environments.vanilla_genrm_environment import VanillaGenRMEnvironment

TokenizerType = PreTrainedTokenizerBase


DWRL_PROMPT_TEMPLATE = """You are an expert evaluation judge specializing in comparative assessment of LLM responses. You are impartial, rigorous, and consistent. Given the conversation context and two assistant responses to the user's latest query, you will follow the evaluation plan and scoring guidelines exactly as written below.

#### Conversation Context ####
{context}

#### Responses to be Scored ####
[The Start of Response 1]
{response_1}
[The End of Response 1]

[The Start of Response 2]
{response_2}
[The End of Response 2]

#### Evaluation Plan ####
{principle}

#### Scoring Guidelines ####
Based on the evaluation plan above, assign scores using these scales:

**Individual Helpfulness Scores (1-5):**
- 5: Extremely Helpful - Completely aligned with what the user was asking for
- 4: Mostly Helpful - Generally useful with minor room for improvement
- 3: Partially Helpful - Misses the overall goal in some way
- 2: Borderline Unhelpful - Mostly doesn't capture what the user wanted
- 1: Not Helpful - Completely missed the essence of the request

**Comparative Ranking (1-6):**
- 1: Response 1 is much better than Response 2
- 2: Response 1 is better than Response 2
- 3: Response 1 is slightly better than Response 2
- 4: Response 2 is slightly better than Response 1
- 5: Response 2 is better than Response 1
- 6: Response 2 is much better than Response 1

#### Output Format ####
Analyze step by step following the evaluation plan, then provide your judgment as JSON:
```json
{{
    "response_1_analysis": "Your detailed analysis of Response 1 based on the evaluation plan",
    "response_2_analysis": "Your detailed analysis of Response 2 based on the evaluation plan",
    "score_1": <1-5>,
    "score_2": <1-5>,
    "ranking": <1-6>
}}
```"""

DEFAULT_PRINCIPLE = "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt. Begin your evaluation by generating your own answer to the prompt. You must provide your answer before judging any answers. When evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information. Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when the user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user rather than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive. Then consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers which would be beneficial to include when responding to the user prompt."

# Placeholders understood by the Super 3.5 VL chat template and processor. A literal marker in the text is
# kept where it is (the template adds none of its own), and the processor expands it to <img><image>*N</img>.
IMAGE_MARKER = "<image>"
VIDEO_MARKER = "<video>"

# Audio settings (only for models with a sound encoder, i.e. Omni). Sampling rate, mel bins and the encoder's subsampling
# factor come from the model's own `sound_config`, not from here.
AUDIO_DEFAULTS = {
    "marker": "<so_embedding>",  # expanded to <so_start> + marker*N + <so_end>
    "start_token": "<so_start>",
    "end_token": "<so_end>",
    "max_duration_s": 30.0,  # audio is cut to this many seconds (Bridge's Omni collator default)
}

# Video settings used when `data.default` doesn't provide them (same names as TaskDataSpec fields; values from the
# Omni/Super NeMo-RL video recipes).
VIDEO_DEFAULTS = {
    "num_frames": 32,
    "video_temporal_patch_size": 2,
    "video_target_num_patches": 1024,
    "video_maintain_aspect_ratio": True,
    "video_sampling_style": "nemotron_vl",
}

def _select_trainer(master_config: MasterConfig):
    """Pick the synchronous trainer based on ``data_plane.enabled``.

    Factored out so test_architecture_invariants can verify dispatch
    without the full setup() path.
    """

    print("ðŸš€ Running synchronous DWRL training (legacy)")
    return dwrl_train_pairwise

def _media_marker(part_type: str) -> str:
    """Inline placeholder for a media content part (kept at the position it had in the conversation)."""
    if part_type in IMAGE_CONTENT_TYPES:
        return IMAGE_MARKER
    if part_type in VIDEO_CONTENT_TYPES:
        return VIDEO_MARKER
    if part_type in AUDIO_CONTENT_TYPES:
        return AUDIO_DEFAULTS["marker"]
    raise ValueError(f"unsupported content part type {part_type!r}")

@functools.lru_cache(maxsize=None)  # keyed on the tokenizer/processor object; reading a hub model's config costs ~100 ms
def get_sound_config(tokenizer) -> Optional[dict]:
    """The runtime audio gate: the model's `sound_config` (a dict for a model with a sound encoder such as Omni), or None
    when it can't consume audio (Super 3.5 VL).

    This is the same test vLLM applies (`supports_audio = hf_config.sound_config is not None`). Note that
    `processor.feature_extractor` can NOT be used for it: neither the Omni nor the Super processor has one. Nor can the
    processor's `audio` argument, which Super's processor also accepts although its model has no sound encoder.
    Like `process_nemotron_video_frames`, the config is looked up through `tokenizer.name_or_path` (HF id or local
    directory); only the plain config.json is read, no remote code is run."""
    chat_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    config_dict, _ = PretrainedConfig.get_config_dict(chat_tokenizer.name_or_path)
    return config_dict.get("sound_config")

def flatten_to_single_turn(message_log):
    """Flatten the conversation context into the text of the judge prompt.

    Media parts become inline `<image>` / `<video>` markers; the raw parts are returned, in order, for
    `extract_all_modals` to decode."""
    def strip_prefix(text: str) -> str:
        if text.lower().startswith("user:"):
            return text[5:].strip()
        if text.lower().startswith("assistant:"):
            return text[10:].strip()
        if text.lower().startswith("system:"):
            return text[7:].strip()

        return text

    ret = ""
    multimodal = []
    for idx, message in enumerate(message_log):
        if message["role"] == "system":
            ret += "System: " + strip_prefix(message["content"]) + "\n\n"
        elif message["role"] == "user":
            if isinstance(message["content"], str):
                ret += "User: " + strip_prefix(message["content"]) + "\n\n"
            else:
                for payload in message["content"]:
                    if payload["type"] == "text":
                        ret += "User: " + strip_prefix(payload["text"]) + "\n\n"
                    else:
                        ret += "User: " + _media_marker(payload["type"]) + "\n\n"
                        multimodal.append(payload)
        elif message["role"] == "assistant":
            resp_no_thinking1 = re.sub(r"(?i)(?s)(\<think\>)(.*?)(\<\/think\>)", "", message["content"]).strip()
            resp_no_thinking2 = re.sub(r"(?i)(?s)(.*?)(\<\/think\>)", "", resp_no_thinking1).strip()
            ret += "Assistant: " + strip_prefix(resp_no_thinking2) + "\n\n"

    return ret.strip(), multimodal

def _split_data_url(url: str) -> tuple[str, str]:
    """'data:<mime>;base64,<payload>' -> (mime, payload). Mirrors vLLM connector.py `_load_data_url`."""
    if not url.startswith("data:"):
        raise ValueError("`decode_data_url` only handles base64 data URLs (vLLM also fetches http/file URLs)")
    header, payload = url[len("data:"):].split(",", 1)
    mime, _, encoding = header.partition(";")
    if encoding != "base64":
        raise NotImplementedError("only base64 data URLs are supported")
    return mime, payload

def decode_image_data_url(url: str) -> Image.Image:
    """vLLM media/image.py:70-80."""
    _, payload = _split_data_url(url)
    img = Image.open(io.BytesIO(base64.b64decode(payload, validate=True)))
    img.load()
    return img.convert("RGB")

def _decode_mp4_pyav(raw: bytes, num_frames: int, temporal_patch_size: int):
    """PyAV fallback for `_decode_torchcodec_video` (same sampling, same outputs). The PyAV wheel bundles its own
    FFmpeg, so no system FFmpeg shared libraries are needed."""
    import av

    with av.open(io.BytesIO(raw)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 0.0)
        duration = float(container.duration / av.time_base) if container.duration else 0.0
        total = int(stream.frames or round(duration * fps))  # stream.frames is 0 for some containers
        if total <= 0 or fps <= 0:
            raise ValueError("video has no frames / invalid fps")
        indices = _torchcodec_sample_indices(
            total_frames=total,
            fps=fps,
            requested_num_frames=num_frames,
            temporal_patch_size=temporal_patch_size,
            sampling_style=VIDEO_DEFAULTS["video_sampling_style"],
        )
        wanted, got = set(indices), {}
        for i, frame in enumerate(container.decode(stream)):
            if i in wanted:
                got[i] = frame.to_ndarray(format="rgb24")
            if i >= max(indices):
                break
    return np.stack([got[i] for i in indices]), fps, total, indices

def decode_data_url(url: str, num_frames: int = VIDEO_DEFAULTS["num_frames"],
                    temporal_patch_size: int = VIDEO_DEFAULTS["video_temporal_patch_size"],
                    target_sr: int = 16000):
    """base64 data URL -> media, in memory.

      * image/*                          -> PIL.Image (RGB)
      * audio/*                          -> mono float32 waveform at `target_sr` (soundfile if installed, else stdlib WAV)
      * video/mp4  (real video file)     -> (frames uint8 [T,H,W,3], metadata dict). Sampled like NeMo-RL's static video
        datum (`nemotron_vl` style: `num_frames` frames, rounded to a multiple of `temporal_patch_size`), decoded with
        torchcodec (NeMo-RL's decoder) and, if its FFmpeg shared libraries can't be loaded, with PyAV.
      * video/jpeg (data:video/jpeg;base64,<jpg1>,<jpg2>,...) -> (frames, metadata); all frames are used, 1 fps.

    The (frames, metadata) pair is also the exact item vLLM's Nemotron VL processor takes in `multi_modal_data["video"]`.
    Whether the model can use audio at all is decided by the caller (`get_sound_config`)."""
    mime, payload = _split_data_url(url)
    kind = mime.lower().split("/")[0]

    if kind == "image":
        return decode_image_data_url(url)

    if kind == "audio":
        raw = base64.b64decode(payload, validate=True)
        try:
            import soundfile as sf
            y, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)  # (n, ch)
        except ImportError:
            with wave.open(io.BytesIO(raw)) as w:
                sr, ch, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
                buf = w.readframes(n)
            if width == 2:
                y = np.frombuffer(buf, dtype="<i2").astype(np.float32) / 32768.0
            elif width == 4:
                y = np.frombuffer(buf, dtype="<i4").astype(np.float32) / 2147483648.0
            elif width == 1:
                y = (np.frombuffer(buf, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            else:
                raise NotImplementedError(f"{8 * width}-bit WAV needs `pip install soundfile`")
            y = y.reshape(-1, ch)
        y = y.mean(axis=1)  # mono by averaging channels
        if sr != target_sr:
            from scipy.signal import resample_poly
            g = math.gcd(sr, target_sr)
            y = resample_poly(y, target_sr // g, sr // g).astype(np.float32)
        return np.ascontiguousarray(y, dtype=np.float32)

    if mime.lower() == "video/jpeg":
        frames = np.stack([np.asarray(decode_image_data_url("data:image/jpeg;base64," + p)) for p in payload.split(",")])
        metadata = _build_video_metadata(
            fps=1.0,
            total_frames=len(frames),
            sampled_indices=list(range(len(frames))),
            backend="jpeg_sequence",
            sampling_style=VIDEO_DEFAULTS["video_sampling_style"],
        )
        return frames, metadata

    if mime.lower() == "video/mp4":
        raw = base64.b64decode(payload, validate=True)
        try:
            frames, fps, total, indices = _decode_torchcodec_video(
                raw,
                requested_num_frames=num_frames,
                temporal_patch_size=temporal_patch_size,
                sampling_style=VIDEO_DEFAULTS["video_sampling_style"],
                source_description="base64 mp4",
            )
            backend = "torchcodec"
        except (ImportError, RuntimeError) as error:
            if isinstance(error, RuntimeError) and "libtorchcodec" not in str(error):
                raise  # a real decode error, not a missing FFmpeg
            print(f"[dwrl_vlm] torchcodec unavailable ({type(error).__name__}); decoding with PyAV", flush=True)
            frames, fps, total, indices = _decode_mp4_pyav(raw, num_frames, temporal_patch_size)
            backend = "pyav"
        metadata = _build_video_metadata(
            fps=fps,
            total_frames=total,
            sampled_indices=indices,
            backend=backend,
            sampling_style=VIDEO_DEFAULTS["video_sampling_style"],
        )
        return frames, metadata

    raise ValueError(f"unsupported mime type {mime}")

def extract_all_modals(messages: list[dict], video_cfg: dict, sound_config: Optional[dict] = None):
    """Decode every media content part, in order.
    -> (images: list[PIL], videos: list[(frames, metadata)], audios: list[mono float32 waveform]).

    `video_cfg` is the video settings dict (VIDEO_DEFAULTS keys); videos are sampled with `num_frames` and
    `video_temporal_patch_size`. Audio is decoded at the model's own sampling rate and cut to AUDIO_DEFAULTS["max_duration_s"].
    `sound_config` is the gate (`get_sound_config`): without it the model can't consume audio, so an audio part is an
    error, not silently dropped."""
    images, videos, audios = [], [], []
    for part in messages:
        # The data URL sits under a different key depending on the part type (image_url, image, input_image, audio_url, ...),
        # either directly or as {"url": ...}.
        value = next((part[key] for key in (part["type"], "url", "image_url", "image", "video_url", "video", "audio_url", "audio")
                      if part.get(key) is not None), None)
        if value is None:
            raise ValueError(f"no media payload found in content part {part['type']!r}")
        url = value["url"] if isinstance(value, dict) else value

        if part["type"] in IMAGE_CONTENT_TYPES:
            images.append(decode_data_url(url))
        elif part["type"] in VIDEO_CONTENT_TYPES:
            videos.append(decode_data_url(
                url,
                num_frames=video_cfg["num_frames"],
                temporal_patch_size=video_cfg["video_temporal_patch_size"],
            ))
        elif part["type"] in AUDIO_CONTENT_TYPES:
            if sound_config is None:
                raise ValueError("this sample has audio, but the model has no sound encoder (`sound_config` is null in its "
                                 "config.json, as for Super 3.5 VL). Use an audio-capable model such as Nemotron Omni, "
                                 "or remove the audio parts from the data.")
            sampling_rate = sound_config["sampling_rate"]
            waveform = decode_data_url(url, target_sr=sampling_rate)
            audios.append(waveform[: int(AUDIO_DEFAULTS["max_duration_s"] * sampling_rate)])

    return images, videos, audios

def compute_audio_features(waveform: np.ndarray, sound_config: dict):
    """One waveform -> (mel [frames, mel_bins] float32, valid_length, num_tokens), as Megatron-Bridge's Omni collator.

    `num_tokens` is how many <so_embedding> the model will turn into features: the encoder's subsampling applied to the
    extractor's VALID frame count. It is NOT the HF processor's count, which comes from the raw waveform length and is one
    too high for some durations (2 s, 10 s, 20 s, 30 s, ...), which would break the model's placeholder/feature match."""
    def mel_extractor():
        from transformers import ParakeetFeatureExtractor  # needs `librosa`

        return ParakeetFeatureExtractor(feature_size=sound_config["num_mel_bins"], sampling_rate=sound_config["sampling_rate"])

    sampling_rate = sound_config["sampling_rate"]
    features = mel_extractor()(waveform, sampling_rate=sampling_rate, return_tensors="pt", return_attention_mask=True)
    mel = features["input_features"].squeeze(0)
    mask = features["attention_mask"][0].bool()
    valid_length = int(mask.sum())
    assert valid_length > 0 and bool(mask[:valid_length].all()), "audio feature mask must be one non-empty valid prefix"
    num_tokens = valid_length
    for _ in range(int(math.log2(sound_config["subsampling_factor"]))):
        num_tokens = (num_tokens + 1) // 2
    return mel, valid_length, max(1, num_tokens)

def expand_audio_placeholders(token_ids: torch.Tensor, num_tokens: list[int], tokenizer) -> torch.Tensor:
    """Each lone <so_embedding> -> <so_start> <so_embedding>*N <so_end>, one N per clip, in order."""
    chat_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    marker = AUDIO_DEFAULTS["marker"]
    marker_id, start_id, end_id = chat_tokenizer.convert_tokens_to_ids(
        [marker, AUDIO_DEFAULTS["start_token"], AUDIO_DEFAULTS["end_token"]])
    expanded, clip = [], 0
    for token in token_ids.tolist():
        if token == marker_id:
            if clip >= len(num_tokens):
                raise ValueError(f"the prompt has more {marker} markers than the {len(num_tokens)} audio clips")
            expanded += [start_id] + [marker_id] * num_tokens[clip] + [end_id]
            clip += 1
        else:
            expanded.append(token)
    if clip != len(num_tokens):
        raise ValueError(f"the prompt has {clip} {marker} markers for {len(num_tokens)} audio clips")
    return torch.tensor(expanded, dtype=token_ids.dtype)

def build_user_message(prompt_str: str, images: list, videos: list, audios: list, tokenizer, video_cfg: dict,
                       sound_config: Optional[dict]):
    """One judge-prompt user message, as the Megatron policy and the vLLM rollout need it.

    -> (user_message, vllm_content)
      * user_message["token_ids"]: the prompt with every media marker expanded (<img><image>*N</img>, and for audio
        <so_start><so_embedding>*N<so_end>).
      * user_message[pixel_values | imgs_sizes | num_frames | sound_clips | sound_length]: `PackedTensor`s, the
        policy-ready media. They ride along in the message log, so `dwrl_train` hands them to the Megatron model
        unchanged (still-image tensors come from the HF processor; video tensors from NeMo-RL's own video preprocessing,
        as in the Gym video datum; audio as mel features + valid lengths, the model's `sound_clips` / `sound_length`).
      * vllm_content: the un-expanded prompt text (markers still in place). vLLM expands the media itself from the
        native `vllm_images` / `vllm_videos` / `vllm_audios` keys, so rollout and training see the same media.
    """
    chat_tokenizer = tokenizer if hasattr(tokenizer, "apply_chat_template") else tokenizer.tokenizer

    def render(content: str) -> str:
        return chat_tokenizer.apply_chat_template(  # type: ignore
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            add_special_tokens=False,
            #enable_thinking=False,
        )

    if (images or videos or audios) and not uses_image_placeholder(tokenizer):
        raise ValueError("multimodal DWRL needs the HF processor of a Nemotron Omni/Super VL model: "
                         "set policy.use_hf_processor=true")

    # The branches below are alternatives, so what they can't combine has to be refused here, loudly: otherwise `elif`
    # would silently drop the video. Audio is independent of the visual path (handled after it) and combines with anything.
    if images and videos:
        raise NotImplementedError("a sample with both still images and a video isn't supported: NeMo-RL's video "
                                  "preprocessing treats every <image> placeholder as a video frame (they share one token id)")
    if len(videos) > 1:
        raise NotImplementedError("one video per sample is supported (NeMo-RL's video preprocessing handles a single video)")

    user_message = {"role": "user"}

    if images:
        text = render(prompt_str)
        assert text.count(IMAGE_MARKER) == len(images), (
            f"prompt has {text.count(IMAGE_MARKER)} {IMAGE_MARKER} markers for {len(images)} images")
        # Images of different sizes come back as a ragged list, which the processor can't stack: ask for plain
        # lists and fold them into one padded tensor (`imgs_sizes` keeps the true sizes). Same as NeMo-RL's
        # `attach_image_model_inputs_to_message`, but with the real prompt text so the tokens are expanded too.
        ragged = len(images) > 1
        processed = dict(tokenizer(text=text, images=images, return_tensors=None if ragged else "pt"))
        if ragged:
            processed = _materialize_ragged_pixel_values(processed, tokenizer)
        token_ids = processed["input_ids"][0]
        user_message.update(extract_multimodal_model_inputs(tokenizer, processed))
    elif videos:
        frames, metadata = videos[0]
        before, _, after = prompt_str.partition(VIDEO_MARKER)
        frame_items = [
            {"type": "image", "image": Image.fromarray(frame), "_is_video_frame": True, "_video_source": "base64 video",
             "_video_frame_index": int(index), "_video_fps": float(metadata["fps"])}
            for frame, index in zip(frames, metadata["frames_indices"])
        ]
        processed = process_nemotron_video_frames(
            tokenizer,
            [{"role": "user", "content": [{"type": "text", "text": before}, *frame_items, {"type": "text", "text": after}]}],
            template_kwargs={},
            temporal_patch_size=video_cfg["video_temporal_patch_size"],
            target_num_patches=video_cfg["video_target_num_patches"],
            maintain_aspect_ratio=video_cfg["video_maintain_aspect_ratio"],
        )
        processed["num_frames"] = torch.tensor([len(frame_items)], dtype=torch.int32)
        token_ids = processed["input_ids"][0]
        user_message.update(extract_multimodal_model_inputs(tokenizer, processed))
        # same "\n"-joined layout as the other paths, with the single <video> marker standing in for the frames
        text = render("\n".join([before, VIDEO_MARKER, after]))
    else:
        text = render(prompt_str)
        token_ids = tokenizer(text=[text], return_tensors="pt")["input_ids"][0]

    if audios:
        # The audio markers are expanded here, not by the processor (see `compute_audio_features`): the count must
        # match what the sound encoder produces from `sound_clips` / `sound_length`.
        marker = AUDIO_DEFAULTS["marker"]
        assert text.count(marker) == len(audios), (
            f"prompt has {text.count(marker)} {marker} markers for {len(audios)} audio clips")
        features = [compute_audio_features(waveform, sound_config) for waveform in audios]
        token_ids = expand_audio_placeholders(token_ids, [num_tokens for _, _, num_tokens in features], tokenizer)
        sound_clips = torch.zeros(len(features), max(mel.shape[0] for mel, _, _ in features), features[0][0].shape[1])
        for clip, (mel, _, _) in enumerate(features):
            sound_clips[clip, : mel.shape[0]] = mel  # zero-padded; sound_length says how much of each clip is real
        user_message["sound_clips"] = PackedTensor(sound_clips, dim_to_pack=0, pad_to_max_shape=True)
        user_message["sound_length"] = PackedTensor(
            torch.tensor([valid_length for _, valid_length, _ in features], dtype=torch.long), dim_to_pack=0
        )

    user_message["token_ids"] = token_ids
    user_message["content"] = text
    return user_message, (text if (images or videos or audios) else None)

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
    resp1 = datum_dict.get("response1", "")
    resp2 = datum_dict.get("response2", "")
    principle = datum_dict.get("principle", DEFAULT_PRINCIPLE)

    if isinstance(context, list):
        context, multimodal = flatten_to_single_turn(context)
    elif isinstance(context, str):
        multimodal = []
    else:
        raise RuntimeError(f"Unexpected context type in dwrl_pairwise_data_processor: {type(context)}")

    # Decode the base64 media once per datum: the prompt tokens and the native vLLM payload share it.
    video_cfg = {
        key: (getattr(task_data_spec, key, None) if getattr(task_data_spec, key, None) is not None else default)
        for key, default in VIDEO_DEFAULTS.items()
    }
    # The audio gate is only consulted for samples that actually contain audio, so image/video-only runs never depend on it.
    has_audio = any(part["type"] in AUDIO_CONTENT_TYPES for part in multimodal)
    sound_config = get_sound_config(tokenizer) if has_audio else None
    images, videos, audios = extract_all_modals(multimodal, video_cfg, sound_config)

    prompt_str = DWRL_PROMPT_TEMPLATE.format(context=context, response_1=resp1, response_2=resp2, principle=principle)

    # Prepare metadata for environment. The base64 payload stays out of it: the environment never reads the
    # context, and this dict is copied to every generation of the prompt and shipped to the reward workers.
    metadata = copy.deepcopy({key: value for key, value in datum_dict.items() if key != "context"})
    metadata["context"] = context
    metadata['score_1'] = metadata['helpfulness_1']
    metadata['score_2'] = metadata['helpfulness_2']
    metadata['ranking'] = metadata['preference_ranking']

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

    return DatumSpec(
        message_log=message_log,
        length=total_length,
        extra_env_info=metadata,
        loss_multiplier=loss_multiplier,
        idx=idx,
        task_name=task_data_spec.task_name,
        **vllm_kwargs,  # pyrefly: ignore[bad-unpacking]
    )

def setup_data(
    tokenizer: TokenizerType,
    data_config: DataConfig,
    env_configs: dict[str, Any],
    seed: int,
) -> tuple[
    AllTaskProcessedDataset,
    Optional[AllTaskProcessedDataset],
    dict[str, EnvironmentInterface],
    dict[str, EnvironmentInterface],
]:
    print("\n▶ Setting up DWRL data...")

    # Create task spec for GenRM RLHF
    # Video settings (num_frames, video_temporal_patch_size, video_target_num_patches, ...) come from `data.default`;
    # anything missing falls back to VIDEO_DEFAULTS inside the data processor.
    default_data_config = data_config.get("default") or {}
    genrm_dwrl_task_spec = TaskDataSpec(
        task_name="genrm_dwrl",
        prompt_file=data_config.get("prompt_file"),
        system_prompt_file=data_config.get("system_prompt_file"),
        **{key: default_data_config[key] for key in VIDEO_DEFAULTS if default_data_config.get(key) is not None},
    )

    # Load dataset
    train_data_path = data_config["train"].get("data_path")
    val_data_path = data_config["validation"].get("data_path")
    train_dataset = HelpSteer3LocalDataset(train_data_path, media_path=data_config["train"].get("media_path"), task_name="genrm_dwrl", shuffle_seed=seed, split='train', double_swap=True)
    val_dataset = HelpSteer3LocalDataset(val_data_path, media_path=data_config["validation"].get("media_path"), task_name="genrm_dwrl", shuffle_seed=seed, split='validation', double_swap=True) if val_data_path else None

    # Set up data processor
    task_data_processors: dict[str, tuple[TaskDataSpec, TaskDataProcessFnCallable]] = (
        defaultdict(lambda: (genrm_dwrl_task_spec, dwrl_pairwise_data_processor))
    )
    task_data_processors["genrm_dwrl"] = (
        genrm_dwrl_task_spec,
        dwrl_pairwise_data_processor,
    )
    # Also register under alternate name for backward compatibility
    task_data_processors["dwrl_genrm"] = (
        genrm_dwrl_task_spec,
        dwrl_pairwise_data_processor,
    )
    
    # Initialize GenRM environment
    '''
    genrm_env = GenRMEnvironment.options(
        runtime_env={
            "py_executable": get_actor_python_env(
                "nemo_rl.environments.genrm_environment.GenRMEnvironment"
            ),
            "env_vars": dict(os.environ),
        }
    ).remote(env_configs.get("genrm_dwrl", {}))
    '''
    genrm_env = VanillaGenRMEnvironment.options(
        runtime_env={
            "py_executable": get_actor_python_env(
                "nemo_rl.environments.vanilla_genrm_environment.VanillaGenRMEnvironment"
            ),
            "env_vars": dict(os.environ),
        }
    ).remote(env_configs.get("genrm_dwrl", {}))
    

    # Create training dataset
    processed_train_dataset = AllTaskProcessedDataset(
        train_dataset,
        tokenizer,
        genrm_dwrl_task_spec,  # default_task_data_spec
        task_data_processors,  # task_data_processors dict
        max_seq_length=data_config["max_input_seq_length"],
    )

    # Create validation dataset if available
    processed_val_dataset: Optional[AllTaskProcessedDataset] = None
    if val_dataset:
        processed_val_dataset = AllTaskProcessedDataset(
            val_dataset,
            tokenizer,
            genrm_dwrl_task_spec,  # default_task_data_spec
            task_data_processors,  # task_data_processors dict
            max_seq_length=data_config["max_input_seq_length"],
        )

    # Map task to environment
    task_to_env: dict[str, EnvironmentInterface] = defaultdict(lambda: genrm_env)
    task_to_env["genrm_dwrl"] = genrm_env
    # Also register under alternate name for backward compatibility
    task_to_env["dwrl_genrm"] = genrm_env

    return processed_train_dataset, processed_val_dataset, task_to_env, task_to_env


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GRPO training with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Parse known args for the script
    args, overrides = parser.parse_known_args()

    return args, overrides


def main() -> None:
    """Main entry point."""
    main_start = time.perf_counter()
    log_container_init_timing()
    rl_init_timer = Timer(context={"worker": "driver"})
    
    # Parse arguments
    register_omegaconf_resolvers()
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "grpo_genrm_dwrl_pairwise_1B.yaml"
        )

    with rl_init_timer.time("config"):
        config = load_config(args.config)
        print(f"Loaded configuration from: {args.config}")
    
        if overrides:
            print(f"Overrides: {overrides}")
            config = parse_hydra_overrides(config, overrides)
    
        config = OmegaConf.to_container(config, resolve=True)
        config = MasterConfig(**config)
        print("Applied CLI overrides")

    # Print config
    print("Final config:")
    pprint.pprint(config)

    # Get the next experiment directory with incremented ID
    config.logger["log_dir"] = get_next_experiment_dir(config.logger["log_dir"])
    print(f"ðŸ“Š Using log directory: {config.logger['log_dir']}")
    if config.checkpointing["enabled"]:
        print(
            f"ðŸ“Š Using checkpoint directory: {config.checkpointing['checkpoint_dir']}"
        )

    init_ray()

    # setup tokenizer
    tokenizer = get_tokenizer(config.policy["tokenizer"], get_processor=config.policy.get("use_hf_processor", False))
    if "Qwen3" in config.policy["tokenizer"]["name"]:
        print("****** Qwen3 tokenizer detected! Swapping in fixed jinja template! ******", flush=True)
        #import requests
        #response = requests.get('https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/raw/main/chat_template.jinja')
        #tokenizer.chat_template = response.text
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen3_fixed_template.jinja"), "r", encoding="utf_8") as fr:
            qwen3_fixed_template = fr.read()
        (tokenizer if hasattr(tokenizer, "chat_template") else tokenizer.tokenizer).chat_template = qwen3_fixed_template
    
    assert hasattr(tokenizer, "image_processor"), (
        "run_dwrl_pairwise_vlm.py needs the multimodal HF processor, not a bare tokenizer: set `policy.use_hf_processor=true`"
    )
    # get_tokenizer applies `policy.tokenizer.chat_template_kwargs` to the inner tokenizer only, but this runner and dwrl.py
    # render prompts with the processor's own apply_chat_template, which would ignore it (while process_nemotron_video_frames,
    # which uses the inner tokenizer, would honor it). Apply it to the processor too so every rendering path agrees.
    if config.policy["tokenizer"].get("chat_template_kwargs"):
        tokenizer.apply_chat_template = functools.partial(tokenizer.apply_chat_template, **config.policy["tokenizer"]["chat_template_kwargs"])
    try:
        print("Audio input:", "supported (model has a sound encoder)" if get_sound_config(tokenizer) else
              "not supported: `sound_config` is null for this model, samples with audio will be rejected", flush=True)
    except Exception as error:  # informational only: the gate is re-evaluated per sample that has audio
        print(f"Audio input: could not read sound_config from the model config ({type(error).__name__}: {error})", flush=True)
    
    assert len((tokenizer if hasattr(tokenizer, "encode") else tokenizer.tokenizer).encode(config.grpo.dwrl["score_token"])) == 1, f'`grpo.dwrl.score_token` of [ {config["grpo"]["dwrl"]["score_token"]} ] does not tokenize to a single token, so DWRL will not work correctly. Pick a different token.'
    assert len((tokenizer if hasattr(tokenizer, "encode") else tokenizer.tokenizer).encode(config.grpo.dwrl["opposite_token"])) == 1, f'`grpo.dwrl.opposite_token` of [ {config["grpo"]["dwrl"]["opposite_token"]} ] does not tokenize to a single token, so DWRL will not work correctly. Pick a different token.'
    
    assert config.policy["generation"] is not None, (
        "A generation config is required for DWRL"
    )
    has_refit_draft_weights = bool(config.policy["draft"]["enabled"])
    megatron_cfg = config.policy.get("megatron_cfg") or {}
    trains_mtp = bool(megatron_cfg.get("mtp_num_layers"))
    config.policy["generation"] = configure_generation_config(
        config.policy["generation"],
        tokenizer,
        has_refit_draft_weights=has_refit_draft_weights,
        trains_mtp=trains_mtp,
    )

    # setup data
    with rl_init_timer.time("data"):
        (
            dataset,
            val_dataset,
            task_to_env,
            val_task_to_env,
        ) = setup_data(tokenizer, config.data, config.env, config.grpo.seed)
    
    with rl_init_timer.time("setup"):
        (
            policy,
            policy_generation,
            _nemo_gym,
            cluster,
            dataloader,
            val_dataloader,
            loss_fn,
            logger,
            checkpointer,
            grpo_state,
            master_config,
            teacher_worker_groups,
            alias_to_group_alias,

        ) = setup(
            config,
            tokenizer,
            dataset,
            val_dataset,
            policy_factory=None,
        )
    
    rl_init_timer.record("total", time.perf_counter() - main_start)

    rl_init_metrics = rl_init_timer.get_timing_metrics(reduction_op="sum")
    print("\n" + "=" * 60)
    print(" " * 14 + "RL INIT TIMING BREAKDOWN")
    for label, value in sorted(rl_init_metrics.items()):
        if isinstance(value, (int, float)):
            print(f"  {label}: {value:.1f}s")
    print("=" * 60 + "\n", flush=True)


    try:
        # Check if async mode is enabled
        if config.grpo.async_grpo.enabled:
            raise RuntimeError("async training is not yet supported for DWRL")

            # Async GRPO does not support dynamic sampling, reward scaling, or reward shaping (DAPO features)
            if config.grpo.use_dynamic_sampling:
                raise NotImplementedError(
                    "use_dynamic_sampling is not supported with async GRPO"
                )
            if config.grpo.reward_scaling.enabled:
                raise NotImplementedError(
                    "reward_scaling is not supported with async GRPO"
                )
            if config.grpo.reward_shaping.enabled:
                raise NotImplementedError(
                    "reward_shaping is not supported with async GRPO"
                )

            # Async GRPO does not support multiple dataloaders
            if config.data["use_multiple_dataloader"]:
                raise NotImplementedError(
                    "use_multiple_dataloader is not supported with async GRPO"
                )

            from nemo_rl.algorithms.dwrl import async_dwrl_train

            print("ðŸš€ Running async DWRL training")

            # Run async GRPO training
            async_dwrl_train(
                policy=policy,
                policy_generation=policy_generation,
                dataloader=dataloader,
                val_dataloader=val_dataloader,
                tokenizer=tokenizer,
                loss_fn=loss_fn,
                task_to_env=task_to_env,
                val_task_to_env=val_task_to_env,
                logger=logger,
                checkpointer=checkpointer,
                grpo_save_state=grpo_state,
                master_config=master_config,
                max_trajectory_age_steps=config.grpo.async_grpo.max_trajectory_age_steps,
                teacher_worker_groups=teacher_worker_groups,
                alias_to_group_alias=alias_to_group_alias,
            )
        else:
            # Two parallel synchronous trainers (verl-style â€” main_ppo.py vs
            # main_ppo_sync.py). data_plane.enabled selects which one runs.
            trainer = _select_trainer(master_config)
            megatron_cfg = config.policy.get("megatron_cfg", {})
            use_fused_linear_logprobs = megatron_cfg.get("enabled", False) and megatron_cfg.get("use_fused_linear_logprobs", False)
            loss_fn = DWRLPairwiseLossFn(config.loss_fn, use_fused_linear_logprobs=use_fused_linear_logprobs)
            # grpo_train_sync defers checkpoint finalization to the checkpointer's
            # background threads; the context manager guarantees they are flushed on
            # exit. (grpo_train also flushes internally; shutdown() is idempotent.)
            with checkpointer:
                trainer(
                    policy,
                    policy_generation,
                    dataloader,
                    val_dataloader,
                    tokenizer,
                    loss_fn,
                    task_to_env,
                    val_task_to_env,
                    logger,
                    checkpointer,
                    grpo_state,
                    master_config,
                )
    finally:
        shutdown_environments(task_to_env, val_task_to_env)
        try:
            policy_generation.shutdown()
        except Exception as error:
            print(f"Error shutting down generation: {error}", flush=True)


if __name__ == "__main__":
    main()
