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

"""Multimodal plumbing shared by the GenRM / DWRL VLM runners (`examples/run_dwrl_vlm.py`,
`examples/run_dwrl_pairwise_vlm.py`, `examples/run_genrm_pairwise_vlm.py`).

Turns a conversation with base64 media parts into a judge-prompt user message: the policy-ready token ids and
`PackedTensor` media for Megatron, plus the un-expanded prompt text and decoded media for vLLM's native multimodal input.
Also the runners' shared tokenizer (HF processor) setup, `setup_vlm_tokenizer`."""

import base64
import functools
import io
import math
import os
import wave
from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import PretrainedConfig

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.genrm_prompts import AUDIO_MARKER, IMAGE_MARKER, VIDEO_MARKER, flatten_to_single_turn, media_part_url
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


# Audio settings (only for models with a sound encoder, i.e. Omni). Sampling rate, mel bins and the encoder's subsampling
# factor come from the model's own `sound_config`, not from here.
AUDIO_DEFAULTS = {
    "marker": AUDIO_MARKER,  # expanded to <so_start> + marker*N + <so_end>
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
            print(f"[genrm_multimodal_vlm] torchcodec unavailable ({type(error).__name__}); decoding with PyAV", flush=True)
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
        url = media_part_url(part)

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

def prepare_context_and_media(context, task_data_spec, tokenizer):
    """A datum's context as judge-prompt text, with its media decoded once (the data processors' shared first step).

    -> (context text with media markers, images, videos, audios, video_cfg, sound_config). `video_cfg` is the task's
    video settings, each falling back to VIDEO_DEFAULTS; `sound_config` is None unless the sample has audio."""
    if isinstance(context, list):
        context, multimodal = flatten_to_single_turn(context)
    elif isinstance(context, str):
        multimodal = []
    else:
        raise RuntimeError(f"Unexpected context type in the GenRM data processor: {type(context)}")

    video_cfg = {
        key: (getattr(task_data_spec, key, None) if getattr(task_data_spec, key, None) is not None else default)
        for key, default in VIDEO_DEFAULTS.items()
    }
    # The audio gate is only consulted for samples that actually contain audio, so image/video-only runs never depend on it.
    has_audio = any(part["type"] in AUDIO_CONTENT_TYPES for part in multimodal)
    sound_config = get_sound_config(tokenizer) if has_audio else None
    images, videos, audios = extract_all_modals(multimodal, video_cfg, sound_config)
    return context, images, videos, audios, video_cfg, sound_config

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

def setup_vlm_tokenizer(config, runner_file: str):
    """The tokenizer (the multimodal HF processor) for a GenRM / DWRL VLM runner.

    `runner_file` is the runner's `__file__`: the fixed Qwen3 chat template is read from the runner's directory, and the
    assertion message names the runner."""
    tokenizer = get_tokenizer(config.policy["tokenizer"], get_processor=config.policy.get("use_hf_processor", False))
    if "Qwen3" in config.policy["tokenizer"]["name"]:
        print("****** Qwen3 tokenizer detected! Swapping in fixed jinja template! ******", flush=True)
        #import requests
        #response = requests.get('https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/raw/main/chat_template.jinja')
        #tokenizer.chat_template = response.text
        with open(os.path.join(os.path.dirname(os.path.abspath(runner_file)), "qwen3_fixed_template.jinja"), "r", encoding="utf_8") as fr:
            qwen3_fixed_template = fr.read()
        (tokenizer if hasattr(tokenizer, "chat_template") else tokenizer.tokenizer).chat_template = qwen3_fixed_template

    assert hasattr(tokenizer, "image_processor"), (
        f"{os.path.basename(runner_file)} needs the multimodal HF processor, not a bare tokenizer: set `policy.use_hf_processor=true`"
    )
    # get_tokenizer applies `policy.tokenizer.chat_template_kwargs` to the inner tokenizer only, but the VLM runners and
    # dwrl.py render prompts with the processor's own apply_chat_template, which would ignore it (while
    # process_nemotron_video_frames, which uses the inner tokenizer, would honor it). Apply it to the processor too so
    # every rendering path agrees.
    if config.policy["tokenizer"].get("chat_template_kwargs"):
        tokenizer.apply_chat_template = functools.partial(tokenizer.apply_chat_template, **config.policy["tokenizer"]["chat_template_kwargs"])
    try:
        print("Audio input:", "supported (model has a sound encoder)" if get_sound_config(tokenizer) else
              "not supported: `sound_config` is null for this model, samples with audio will be rejected", flush=True)
    except Exception as error:  # informational only: the gate is re-evaluated per sample that has audio
        print(f"Audio input: could not read sound_config from the model config ({type(error).__name__}: {error})", flush=True)
    return tokenizer
