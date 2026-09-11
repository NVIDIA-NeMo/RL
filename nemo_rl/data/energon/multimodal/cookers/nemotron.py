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

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import asdict, is_dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image
from megatron.energon import (
    CachePool,
    CrudeSample,
    FileStore,
    SourceInfo,
    basic_sample_keys,
    cooker,
    stateless,
)
from megatron.energon.av import AVDecoder
from megatron.energon.epathlib import EPath

from nemo_rl.data.energon.multimodal.model_families import supports_model_families
from nemo_rl.data.energon.multimodal.types import (
    CanonicalSFTSample,
    FrozenMediaMetadata,
    MediaRef,
    freeze_media_metadata,
)

# Source parsing follows the Apache-2.0 NeMo-RL implementation in
# nemo_rl/data/datasets/response_datasets/general_conversations_dataset.py.
# Tag names, extension fallback, Granary prompt text, and video/audio ordering
# were checked against Megatron-LM 6822175d92a40e0528be905aee50f5930cfa0c98.
_FIELD_ALIASES = {
    "speech": "audio",
    "speeches": "audio",
    "sound": "audio",
    "audios": "audio",
    "images": "image",
    "videos": "video",
    "video-sound": "video-audio",
}
_TAG_ALIASES = {
    **_FIELD_ALIASES,
    "image": "image",
    "video": "video",
    "audio": "audio",
}
_TAG_PATTERN = re.compile(
    "("
    + "|".join(
        f"<{re.escape(tag)}>" for tag in sorted(_TAG_ALIASES, key=len, reverse=True)
    )
    + ")"
)
_DEFAULT_EXTENSIONS = {
    "image": ("png", "jpeg", "jpg", "img"),
    "video": ("mp4",),
    "video-audio": ("mp4",),
    "audio": ("wav", "flac", "mp3"),
}
_ROLE_ALIASES = {"human": "user", "gpt": "assistant", "agent": "assistant"}
_ROLES = frozenset({"system", "user", "assistant", "tool"})
_CONVERSATION_MEDIA_TYPES = frozenset(
    {"image", "video", "video_frame", "audio"}
)

warn_about_slow_media_loading: defaultdict[str, bool] = defaultdict(lambda: True)

EXPLICIT_ASSISTANT_LOSS_COOK = "general_conversations_jsonl_explicit_loss_v1"
EXPLICIT_ASSISTANT_LOSS_MODE = "explicit_assistant_turns"
EXPLICIT_ASSISTANT_LOSS_FIELD = "conversations[*].loss"
_ASSISTANT_ROLES = frozenset({"assistant", "gpt", "agent"})
_EXPLICIT_ASSISTANT_LOSS_INCOMPATIBLE_OPTIONS = (
    "train_only_on_last_assistant_turn",
    "skip_chat_template",
    "tool_response_as_turn_boundary",
    "offline_packed_messages",
)

GRANARY_ENGLISH_PROMPT = (
    "<audio>. \nTranscribe the spoken content to written english text, "
    "with punctuations and capitalizations."
)

# Nano cleanup and the legacy source schemas follow the NVIDIA BSD-3-Clause
# Megatron-LM examples pinned at 6822175d92a40e0528be905aee50f5930cfa0c98:
# examples/multimodal/data_loading/cookers/{conversation,audio_conversation,
# omcat_legacy_audio_conversation}.py.
NO_TOOL_SYSTEM_CONTENT = (
    "<|im_start|>system\n"
    "You are a helpful and harmless assistant.\n\n"
    "You are not allowed to use any tools.<|im_end|>\n"
)
LEGACY_SYSTEM_CONTENT = (
    "<|im_start|>system\nYou are a helpful and harmless assistant.<|im_end|>\n"
)
EMPTY_SYSTEM_CONTENT = "<|im_start|>system\n<|im_end|>\n"

_NANO_ROLE_ALIASES = {"human": "user", "gpt": "assistant", "function": "tool"}
_NANO_ROLES = frozenset({"system", "user", "assistant", "tool"})
_AUDIO_TAG_PATTERN = re.compile(r"<(image|video|sound|video-sound)>")
_OMCAT_TAG_ALIASES = {
    "speech": "sound",
    "speeches": "sound",
    "audio": "sound",
    "audios": "sound",
    "images": "image",
    "videos": "video",
}
_OMCAT_TAG_PATTERN = re.compile(
    r"<(image|video|sound|video-sound|speech|speeches|audio|audios|images|videos)>"
)
_OMCAT_MEMBER_EXTENSIONS = {
    "image": ("png", "jpeg", "jpg", "img"),
    "video": ("mp4",),
    "sound": ("wav", "flac", "mp3"),
}


def _decode_payload(sample: CrudeSample) -> dict[str, Any]:
    value: Any = sample.get("json", sample)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("Nemotron conversation data must decode to an object.")
    return dict(value)


def _descriptor(entry: Any) -> tuple[Any, FrozenMediaMetadata, bool]:
    """Return the media source, frozen metadata, and whether source lookup applies."""
    if not isinstance(entry, dict):
        return entry, (), True

    metadata = freeze_media_metadata(entry.get("metadata"))
    if "value" in entry:
        return entry["value"], metadata, False
    for key in ("member", "path"):
        if key in entry:
            return entry[key], metadata, True
    raise ValueError("Media entries need a value, member, or path.")


def _metadata_values(metadata: object) -> Mapping[str, Any]:
    if metadata is None:
        return {}
    if isinstance(metadata, Mapping):
        return metadata
    if is_dataclass(metadata) and not isinstance(metadata, type):
        return asdict(metadata)
    try:
        return vars(metadata)
    except TypeError as error:
        raise ValueError(
            f"Unsupported Energon media metadata type {type(metadata)!r}."
        ) from error


def _media_metadata(store: FileStore, path: str) -> FrozenMediaMetadata:
    try:
        metadata = store.get_media_metadata(path)
    except Exception as e:
        store_path = str(store.get_path())
        if warn_about_slow_media_loading[store_path]:
            print(
                f"WARNING: Dataset {store_path} not prepared with media metadata, "
                f"slow metadata for {path}: {e!r}"
            )
            warn_about_slow_media_loading[store_path] = False
        return ()
    return freeze_media_metadata(_metadata_values(metadata))


def _open_media(modality: str, value: Any) -> Any:
    if modality == "image":
        if isinstance(value, Image.Image):
            return value
        if isinstance(value, (bytes, bytearray, memoryview)):
            image = Image.open(BytesIO(bytes(value)))
            image.load()
            return image
        if isinstance(value, (str, Path)):
            image = Image.open(value)
            image.load()
            return image
    elif modality in {"video", "video_frame", "audio"}:
        if isinstance(value, AVDecoder):
            return value
        if isinstance(value, (str, Path)):
            value = Path(value).read_bytes()
        if isinstance(value, (bytes, bytearray, memoryview)):
            return AVDecoder(BytesIO(bytes(value)))
    raise ValueError(f"Cannot open {modality} media from {type(value)!r}.")


def _derived_media_metadata(modality: str, value: Any) -> FrozenMediaMetadata:
    if modality == "image":
        image = _open_media(modality, value)
        return freeze_media_metadata(
            {
                "width": image.width,
                "height": image.height,
                "format": image.format,
                "mode": image.mode,
            }
        )
    decoder = _open_media(modality, value)
    return freeze_media_metadata(_metadata_values(decoder.get_metadata()))


def _load_media_metadata(
    store: FileStore,
    path: str,
    *,
    modality: str,
    derive_missing_metadata: bool,
) -> FrozenMediaMetadata:
    metadata = _media_metadata(store, path)
    if metadata or not derive_missing_metadata:
        return metadata

    media_path = EPath(store.get_path()) / path
    with media_path.open("rb") as stream:
        if modality == "image":
            with Image.open(stream) as image:
                return freeze_media_metadata(
                    {
                        "width": image.width,
                        "height": image.height,
                        "format": image.format,
                        "mode": image.mode,
                    }
                )
        return freeze_media_metadata(
            _metadata_values(AVDecoder(stream).get_metadata())
        )


def _source_info(store: FileStore, path: str) -> SourceInfo:
    return SourceInfo(
        dataset_path=store.get_path(),
        index=path,
        shard_name=None,
        file_names=(path,),
    )


def _aux_store_and_path(
    sample: CrudeSample,
    path: str,
    *,
    media_source: FileStore | None,
    media_sources: dict[str, FileStore],
    strip_matched_prefix: bool,
    basename_missing_absolute: bool,
    missing_aux_source_is_error: bool,
) -> tuple[FileStore | None, str]:
    store = media_source
    media_path = path
    if store is None:
        clean_path = re.sub(r"(?:^\./|/\.(?=/))", "", path)
        prefixes = (sample.get("__subflavors__") or {}).get("aux_data_prefixes", {})
        for prefix, aux_key in prefixes.items():
            if not clean_path.startswith(prefix):
                continue
            try:
                store = media_sources[aux_key]
            except KeyError as error:
                if not missing_aux_source_is_error:
                    return None, path
                raise ValueError(
                    f"Auxiliary media source {aux_key!r} is not available for {path!r}."
                ) from error
            media_path = clean_path[len(prefix) :] if strip_matched_prefix else path
            break

    local_path = Path(media_path)
    if (
        store is not None
        and basename_missing_absolute
        and local_path.is_absolute()
        and not local_path.is_file()
    ):
        media_path = PurePosixPath(media_path).name
    return store, media_path


def _aux_media(
    sample: CrudeSample,
    path: str,
    *,
    modality: str,
    metadata: FrozenMediaMetadata = (),
    cache: CachePool | None,
    media_source: FileStore | None,
    media_sources: dict[str, FileStore],
    strip_matched_prefix: bool,
    basename_missing_absolute: bool,
    allow_local: bool,
    derive_missing_metadata: bool,
    missing_aux_source_is_error: bool,
) -> tuple[Any, FrozenMediaMetadata, SourceInfo | None]:
    store, media_path = _aux_store_and_path(
        sample,
        path,
        media_source=media_source,
        media_sources=media_sources,
        strip_matched_prefix=strip_matched_prefix,
        basename_missing_absolute=basename_missing_absolute,
        missing_aux_source_is_error=missing_aux_source_is_error,
    )
    if store is None:
        if not allow_local:
            raise ValueError(
                f"No configured media source matches media path {path!r}."
            )
        local_path = Path(path)
        if not local_path.is_file():
            raise ValueError(f"Cannot find media file {path!r} in configured sources.")
        if cache is None:
            raise ValueError("Local media loading requires an Energon cache pool.")
        opened = _open_media(modality, local_path)
        metadata = metadata or _derived_media_metadata(modality, opened)
        source = SourceInfo(
            dataset_path=local_path.parent,
            index=local_path.name,
            shard_name=None,
            file_names=(local_path.name,),
        )
        cache_key = f"{sample.get('__key__', '')}.{local_path.suffix.lstrip('.')}"
        return cache.to_cache(opened, cache_key), metadata, source
    if cache is None:
        raise ValueError("Auxiliary media loading requires an Energon cache pool.")
    metadata = metadata or _load_media_metadata(
        store,
        media_path,
        modality=modality,
        derive_missing_metadata=derive_missing_metadata,
    )
    return cache.get_lazy(store, media_path), metadata, _source_info(store, media_path)


def _primary_media(
    sample: CrudeSample,
    member: str,
    *,
    modality: str,
    metadata: FrozenMediaMetadata,
    cache: CachePool | None,
    primary: FileStore | None,
) -> tuple[Any, FrozenMediaMetadata]:
    opened = _open_media(modality, sample[member])
    metadata = metadata or (
        () if primary is None else _media_metadata(primary, f".{member}")
    )
    if not metadata:
        metadata = _derived_media_metadata(modality, opened)
    if cache is None:
        return sample[member], metadata
    cache_key = f"{sample.get('__key__', '')}.{member}"
    return cache.to_cache(opened, cache_key), metadata


def _fragment_metadata(
    fragment: Mapping[str, Any], *, modality: str
) -> tuple[FrozenMediaMetadata, FrozenMediaMetadata]:
    metadata = freeze_media_metadata(fragment.get("metadata"))
    timing: dict[str, Any] = {}
    timing_keys: tuple[str, ...] = ()
    if modality == "video":
        timing_keys = ("start_time", "end_time")
    elif modality == "video_frame":
        timing_keys = ("timestamp", "frame_index", "sample_index")
    unexpected_keys = set(fragment) - {"t", "value", "metadata", *timing_keys}
    if unexpected_keys:
        raise ValueError(
            f"Unexpected {modality} fragment fields: {sorted(unexpected_keys)!r}."
        )
    for key in timing_keys:
        if key in fragment:
            timing[key] = fragment[key]
    return metadata, freeze_media_metadata(timing)


def _merge_media_metadata(
    metadata: FrozenMediaMetadata, extra: FrozenMediaMetadata
) -> FrozenMediaMetadata:
    values = dict(metadata)
    values.update(extra)
    return freeze_media_metadata(values)


def _sample_keys_with_dataset(
    sample: CrudeSample, extra_sources: tuple[SourceInfo, ...] = ()
) -> dict[str, Any]:
    sample_keys = basic_sample_keys(sample, extra_sources)
    dataset = _decode_payload(sample).get("dataset")
    if dataset is not None:
        subflavors = dict(sample_keys.get("__subflavors__", {}) or {})
        subflavors["dataset"] = dataset
        sample_keys["__subflavors__"] = subflavors
    return sample_keys


def _validate_explicit_assistant_loss(
    sample: CrudeSample, conversations: list[Any]
) -> None:
    subflavors = sample.get("__subflavors__", {}) or {}
    if subflavors.get("loss_mask_mode") != EXPLICIT_ASSISTANT_LOSS_MODE:
        raise ValueError(
            f"{EXPLICIT_ASSISTANT_LOSS_COOK} requires "
            f"loss_mask_mode={EXPLICIT_ASSISTANT_LOSS_MODE}."
        )
    if subflavors.get("assistant_loss_mask_field") != EXPLICIT_ASSISTANT_LOSS_FIELD:
        raise ValueError(
            f"{EXPLICIT_ASSISTANT_LOSS_COOK} requires "
            f"assistant_loss_mask_field={EXPLICIT_ASSISTANT_LOSS_FIELD}."
        )
    for option in _EXPLICIT_ASSISTANT_LOSS_INCOMPATIBLE_OPTIONS:
        if subflavors.get(option, False):
            raise ValueError(f"Explicit assistant loss is incompatible with {option}.")

    has_trainable_assistant = False
    for index, message in enumerate(conversations):
        if not isinstance(message, Mapping):
            raise ValueError(f"conversations[{index}] must be an object.")
        sender = message.get("from")
        if sender not in _ROLES | _ASSISTANT_ROLES | {"human"}:
            raise ValueError(
                f"conversations[{index}].from has unsupported sender {sender!r}."
            )
        if sender in _ASSISTANT_ROLES:
            if type(message.get("loss")) is not bool:
                raise ValueError(
                    f"conversations[{index}].loss must be a boolean for "
                    f"assistant sender {sender!r}."
                )
            has_trainable_assistant = has_trainable_assistant or message["loss"]
        elif "loss" in message:
            raise ValueError(
                f"conversations[{index}].loss is only valid on assistant turns."
            )
    if not has_trainable_assistant:
        raise ValueError("Explicit assistant loss requires at least one loss=true turn.")


def _validate_loss_mask_subflavors(
    sample: CrudeSample, *, explicit_assistant_loss: bool
) -> None:
    subflavors = sample.get("__subflavors__", {}) or {}
    cook_name = subflavors.get("cook")
    loss_mask_mode = subflavors.get("loss_mask_mode")
    assistant_loss_mask_field = subflavors.get("assistant_loss_mask_field")
    if loss_mask_mode not in (None, "", EXPLICIT_ASSISTANT_LOSS_MODE):
        raise ValueError(f"Unsupported loss_mask_mode={loss_mask_mode!r}.")
    if (
        "assistant_loss_mask_field" in subflavors
        and loss_mask_mode != EXPLICIT_ASSISTANT_LOSS_MODE
    ):
        raise ValueError(
            "assistant_loss_mask_field requires "
            f"loss_mask_mode={EXPLICIT_ASSISTANT_LOSS_MODE}."
        )

    selected_explicit_loss = loss_mask_mode == EXPLICIT_ASSISTANT_LOSS_MODE
    if selected_explicit_loss and cook_name != EXPLICIT_ASSISTANT_LOSS_COOK:
        raise ValueError(
            f"loss_mask_mode={EXPLICIT_ASSISTANT_LOSS_MODE} requires "
            f"cook={EXPLICIT_ASSISTANT_LOSS_COOK}."
        )
    if selected_explicit_loss and assistant_loss_mask_field != EXPLICIT_ASSISTANT_LOSS_FIELD:
        raise ValueError(
            f"loss_mask_mode={EXPLICIT_ASSISTANT_LOSS_MODE} requires "
            f"assistant_loss_mask_field={EXPLICIT_ASSISTANT_LOSS_FIELD}."
        )
    if cook_name == EXPLICIT_ASSISTANT_LOSS_COOK and not selected_explicit_loss:
        raise ValueError(
            f"cook={EXPLICIT_ASSISTANT_LOSS_COOK} requires "
            f"loss_mask_mode={EXPLICIT_ASSISTANT_LOSS_MODE}."
        )
    if selected_explicit_loss != explicit_assistant_loss:
        expected = EXPLICIT_ASSISTANT_LOSS_COOK if explicit_assistant_loss else "a standard cooker"
        raise ValueError(f"Loss-mask configuration does not match {expected}.")


def _apply_last_assistant_mask(
    sample: CrudeSample, messages: list[dict[str, Any]]
) -> None:
    subflavors = sample.get("__subflavors__", {}) or {}
    if not subflavors.get("train_only_on_last_assistant_turn", False):
        return
    assistant_indexes = [
        index for index, message in enumerate(messages) if message["role"] == "assistant"
    ]
    if not assistant_indexes:
        raise ValueError(
            "train_only_on_last_assistant_turn requires an assistant message."
        )
    last_assistant = assistant_indexes[-1]
    for index, message in enumerate(messages):
        message["train_on_message"] = index == last_assistant


@supports_model_families("nemotron")
@stateless
@cooker(need_cache=True)
def cook_nemotron_conversation(
    sample: CrudeSample,
    cache: CachePool | None = None,
    media_source: FileStore | None = None,
    **media_sources: FileStore,
) -> CanonicalSFTSample:
    """Cook the Nemotron ``conversation[].fragments[]`` source schema."""
    payload = _decode_payload(sample)
    conversation = payload.get("conversation")
    if not isinstance(conversation, list) or not conversation:
        raise ValueError("Nemotron fragment conversations require a non-empty list.")

    messages: list[dict[str, Any]] = []
    media: list[MediaRef] = []
    source_info: list[SourceInfo] = []
    for raw_message in conversation:
        if not isinstance(raw_message, dict):
            raise ValueError("Nemotron conversation messages must be objects.")
        role = _ROLE_ALIASES.get(raw_message.get("sender"), raw_message.get("sender"))
        if role not in _ROLES:
            raise ValueError(f"Unknown Nemotron conversation role {role!r}.")
        fragments = raw_message.get("fragments")
        if not isinstance(fragments, list) or not fragments:
            raise ValueError("Nemotron conversation messages require fragments.")

        content: list[dict[str, Any]] = []
        for fragment in fragments:
            if isinstance(fragment, str):
                content.append({"type": "text", "text": fragment})
                continue
            if not isinstance(fragment, dict):
                raise ValueError("Nemotron conversation fragments must be objects.")
            fragment_type = fragment.get("t")
            value = fragment.get("value")
            if fragment_type == "text":
                if not isinstance(value, str):
                    raise ValueError("Text fragments require a string value.")
                content.append({"type": "text", "text": value})
                continue
            if fragment_type not in _CONVERSATION_MEDIA_TYPES:
                raise ValueError(
                    f"Unsupported Nemotron conversation fragment type {fragment_type!r}."
                )
            if not isinstance(value, str) or not value:
                raise ValueError("Media fragments require a non-empty path.")
            metadata, timing_metadata = _fragment_metadata(
                fragment, modality=fragment_type
            )
            media_value, store_metadata, source = _aux_media(
                sample,
                value,
                modality=fragment_type,
                metadata=metadata,
                cache=cache,
                media_source=media_source,
                media_sources=media_sources,
                strip_matched_prefix=True,
                basename_missing_absolute=False,
                allow_local=False,
                derive_missing_metadata=False,
                missing_aux_source_is_error=True,
            )
            metadata = _merge_media_metadata(
                metadata or store_metadata, timing_metadata
            )
            media_index = len(media)
            media.append(
                MediaRef(
                    modality=fragment_type,
                    value=media_value,
                    metadata=metadata,
                )
            )
            content.append({"type": fragment_type, "media_index": media_index})
            if source is not None:
                source_info.append(source)
        message = {"role": role, "content": content}
        if "loss" in raw_message:
            raise ValueError(
                "Fragment conversations do not support explicit loss fields; use "
                f"cook={EXPLICIT_ASSISTANT_LOSS_COOK}."
            )
        messages.append(message)

    _apply_last_assistant_mask(sample, messages)
    return CanonicalSFTSample(
        **_sample_keys_with_dataset(sample, tuple(source_info)),
        messages=messages,
        media=media,
        tools=None,
    )


def _resolve_media_value(
    sample: CrudeSample,
    source: Any,
    *,
    media_tag: str,
    metadata: FrozenMediaMetadata,
    lookup_source: bool,
    require_member: bool,
    cache: CachePool | None,
    primary: FileStore | None,
    media_source: FileStore | None,
    media_sources: dict[str, FileStore],
    tried_default_extensions: set[str],
) -> tuple[Any, FrozenMediaMetadata, SourceInfo | None]:
    if not lookup_source or not isinstance(source, str):
        return source, (), None

    sample_key = str(sample.get("__key__", ""))
    basename = PurePosixPath(source).name
    candidates = [basename.split(".", 1)[1] if "." in basename else basename]
    for candidate in candidates:
        if require_member and candidate and candidate in sample:
            modality = "video" if media_tag == "video-audio" else media_tag
            value, metadata = _primary_media(
                sample,
                candidate,
                modality=modality,
                metadata=metadata,
                cache=cache,
                primary=primary,
            )
            return value, metadata, None
    if require_member:
        for candidate in _DEFAULT_EXTENSIONS[media_tag]:
            if candidate in tried_default_extensions or candidate not in sample:
                continue
            tried_default_extensions.add(candidate)
            modality = "video" if media_tag == "video-audio" else media_tag
            value, metadata = _primary_media(
                sample,
                candidate,
                modality=modality,
                metadata=metadata,
                cache=cache,
                primary=primary,
            )
            return value, metadata, None
    modality = "video" if media_tag == "video-audio" else media_tag
    value, metadata, source_info = _aux_media(
        sample,
        source,
        modality=modality,
        metadata=metadata,
        cache=cache,
        media_source=media_source,
        media_sources=media_sources,
        strip_matched_prefix=False,
        basename_missing_absolute=True,
        allow_local=True,
        derive_missing_metadata=True,
        missing_aux_source_is_error=False,
    )
    if source_info is not None:
        return value, metadata, source_info
    if require_member:
        raise ValueError(
            f"Media member {source!r} is absent from Energon sample {sample_key!r}."
        )
    return source, (), None


def _normalized_media_fields(payload: dict[str, Any]) -> dict[str, list[Any]]:
    fields: dict[str, list[Any]] = {}
    aliases = {
        "audio": ("sound", "speech", "speeches", "audio", "audios"),
        "image": ("image", "images"),
        "video": ("video", "videos"),
        "video-audio": ("video-sound",),
    }
    for canonical, source_fields in aliases.items():
        for field in source_fields:
            if field in payload:
                value = payload[field]
                fields[canonical] = value if isinstance(value, list) else [value]
    return fields


def _append_media_part(
    *,
    sample: CrudeSample,
    entry: Any,
    media_tag: str,
    require_member: bool,
    cache: CachePool | None,
    primary: FileStore | None,
    media_source: FileStore | None,
    media_sources: dict[str, FileStore],
    media: list[MediaRef],
    content: list[dict[str, Any]],
    source_info: list[SourceInfo],
    tried_default_extensions: set[str],
) -> None:
    source, metadata, lookup_source = _descriptor(entry)
    value, store_metadata, resolved_source = _resolve_media_value(
        sample,
        source,
        media_tag=media_tag,
        metadata=metadata,
        lookup_source=lookup_source,
        require_member=require_member,
        cache=cache,
        primary=primary,
        media_source=media_source,
        media_sources=media_sources,
        tried_default_extensions=tried_default_extensions,
    )
    metadata = metadata or store_metadata
    modalities = ("video", "audio") if media_tag == "video-audio" else (media_tag,)
    for modality in modalities:
        if resolved_source is not None:
            source_info.append(resolved_source)
        media_index = len(media)
        media.append(MediaRef(modality=modality, value=value, metadata=metadata))
        content.append({"type": modality, "media_index": media_index})


def _cook_general_payload(
    sample: CrudeSample,
    payload: dict[str, Any],
    *,
    require_member: bool,
    explicit_assistant_loss: bool,
    cache: CachePool | None,
    primary: FileStore | None,
    media_source: FileStore | None,
    media_sources: dict[str, FileStore],
) -> CanonicalSFTSample:
    conversations = payload.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError("Nemotron samples require a non-empty conversations list.")
    _validate_loss_mask_subflavors(
        sample, explicit_assistant_loss=explicit_assistant_loss
    )
    if explicit_assistant_loss:
        _validate_explicit_assistant_loss(sample, conversations)

    media_fields = _normalized_media_fields(payload)
    media_indexes: defaultdict[str, int] = defaultdict(int)
    messages: list[dict[str, Any]] = []
    media: list[MediaRef] = []
    source_info: list[SourceInfo] = []
    tried_default_extensions: set[str] = set()

    for message in conversations:
        if not isinstance(message, dict):
            raise ValueError("Each Nemotron conversation turn must be an object.")
        if "loss" in message and not explicit_assistant_loss:
            raise ValueError(
                "Standard Nemotron conversation cookers do not accept explicit loss fields."
            )
        role = _ROLE_ALIASES.get(message.get("from"), message.get("from"))
        if role not in _ROLES:
            raise ValueError(
                f"Unknown Nemotron conversation role {message.get('from')!r}."
            )
        value = message.get("value")
        if not isinstance(value, str):
            raise ValueError("Each Nemotron conversation value must be a string.")

        content: list[dict[str, Any]] = []
        has_text = False
        for part in re.split(_TAG_PATTERN, value):
            tag_name = (
                part[1:-1] if part.startswith("<") and part.endswith(">") else None
            )
            if tag_name in _TAG_ALIASES:
                media_tag = _TAG_ALIASES[tag_name]
                entries = media_fields.get(media_tag)
                media_index = media_indexes[media_tag]
                if entries is None:
                    raise ValueError(
                        f"Tag <{tag_name}> has no {media_tag!r} media field."
                    )
                if media_index >= len(entries):
                    raise ValueError(f"Tag <{tag_name}> has no remaining media value.")
                _append_media_part(
                    sample=sample,
                    entry=entries[media_index],
                    media_tag=media_tag,
                    require_member=require_member,
                    cache=cache,
                    primary=primary,
                    media_source=media_source,
                    media_sources=media_sources,
                    media=media,
                    content=content,
                    source_info=source_info,
                    tried_default_extensions=tried_default_extensions,
                )
                media_indexes[media_tag] += 1
            elif part.strip():
                content.append({"type": "text", "text": part})
                has_text = True
        if not has_text:
            content.append({"type": "text", "text": " "})
        cooked_message = {"role": role, "content": content}
        if explicit_assistant_loss and role == "assistant":
            cooked_message["train_on_message"] = message["loss"]
        messages.append(cooked_message)

    for media_tag, used in media_indexes.items():
        entries = media_fields[media_tag]
        if used != len(entries):
            raise ValueError(
                f"Retrieved {used}/{len(entries)} {media_tag} media values from sample "
                f"{sample.get('__key__', '')!r}."
            )

    _apply_last_assistant_mask(sample, messages)
    return CanonicalSFTSample(
        **_sample_keys_with_dataset(sample, tuple(source_info)),
        messages=messages,
        media=media,
        tools=None,
    )


@supports_model_families("nemotron")
@stateless
@cooker(need_cache=True, need_primary=True)
def cook_general_conversations_webdataset(
    sample: CrudeSample,
    cache: CachePool | None = None,
    primary: FileStore | None = None,
    media_source: FileStore | None = None,
    **media_sources: FileStore,
) -> CanonicalSFTSample:
    """Cook one monolithic Nemotron conversation without opening its media."""
    return _cook_general_payload(
        sample,
        _decode_payload(sample),
        require_member=True,
        explicit_assistant_loss=False,
        cache=cache,
        primary=primary,
        media_source=media_source,
        media_sources=media_sources,
    )


@supports_model_families("nemotron")
@stateless
@cooker(need_cache=True, need_primary=True)
def cook_general_conversations_jsonl(
    sample: CrudeSample,
    cache: CachePool | None = None,
    primary: FileStore | None = None,
    media_source: FileStore | None = None,
    **media_sources: FileStore,
) -> CanonicalSFTSample:
    """Cook one Nemotron JSONL conversation with lazy media paths."""
    return _cook_general_payload(
        sample,
        _decode_payload(sample),
        require_member=False,
        explicit_assistant_loss=False,
        cache=cache,
        primary=primary,
        media_source=media_source,
        media_sources=media_sources,
    )


@supports_model_families("nemotron")
@stateless
@cooker(need_cache=True, need_primary=True)
def cook_general_conversations_jsonl_explicit_loss_v1(
    sample: CrudeSample,
    cache: CachePool | None = None,
    primary: FileStore | None = None,
    media_source: FileStore | None = None,
    **media_sources: FileStore,
) -> CanonicalSFTSample:
    """Cook versioned JSONL with explicit per-assistant-turn loss flags."""
    return _cook_general_payload(
        sample,
        _decode_payload(sample),
        require_member=False,
        explicit_assistant_loss=True,
        cache=cache,
        primary=primary,
        media_source=media_source,
        media_sources=media_sources,
    )


def _cook_granary(
    sample: CrudeSample,
    *,
    require_member: bool,
    cache: CachePool | None,
    primary: FileStore | None,
    media_source: FileStore | None,
    media_sources: dict[str, FileStore],
) -> CanonicalSFTSample:
    payload = _decode_payload(sample)
    if "audio_filepath" not in payload:
        raise ValueError("Granary samples require audio_filepath.")
    text = payload.get("text")
    if not isinstance(text, str):
        raise ValueError("Granary samples require a string text field.")
    granary_payload = {
        "audio": payload["audio_filepath"],
        "conversations": [
            {"from": "user", "value": GRANARY_ENGLISH_PROMPT},
            {"from": "assistant", "value": text},
        ],
    }
    return _cook_general_payload(
        sample,
        granary_payload,
        require_member=require_member,
        explicit_assistant_loss=False,
        cache=cache,
        primary=primary,
        media_source=media_source,
        media_sources=media_sources,
    )


@supports_model_families("nemotron")
@stateless
@cooker(need_cache=True, need_primary=True)
def cook_granary_english_webdataset(
    sample: CrudeSample,
    cache: CachePool | None = None,
    primary: FileStore | None = None,
    media_source: FileStore | None = None,
    **media_sources: FileStore,
) -> CanonicalSFTSample:
    """Cook one Granary WDS row without opening its audio member."""
    return _cook_granary(
        sample,
        require_member=True,
        cache=cache,
        primary=primary,
        media_source=media_source,
        media_sources=media_sources,
    )


@supports_model_families("nemotron")
@stateless
@cooker(need_cache=True, need_primary=True)
def cook_granary_english_jsonl(
    sample: CrudeSample,
    cache: CachePool | None = None,
    primary: FileStore | None = None,
    media_source: FileStore | None = None,
    **media_sources: FileStore,
) -> CanonicalSFTSample:
    """Cook one Granary JSONL row with a lazy audio path."""
    return _cook_granary(
        sample,
        require_member=False,
        cache=cache,
        primary=primary,
        media_source=media_source,
        media_sources=media_sources,
    )


def _canonical_sample(
    sample: CrudeSample,
    *,
    messages: list[dict[str, Any]],
    media: list[MediaRef] | None = None,
    offline_packed_messages: bool = False,
    extra_sources: tuple[SourceInfo, ...] = (),
) -> CanonicalSFTSample:
    sample_keys = _sample_keys_with_dataset(sample, extra_sources)
    if offline_packed_messages:
        subflavors = dict(sample_keys.get("__subflavors__", {}) or {})
        subflavors["offline_packed_messages"] = True
        sample_keys["__subflavors__"] = subflavors
    return CanonicalSFTSample(
        **sample_keys,
        messages=messages,
        media=[] if media is None else media,
        tools=None,
    )


def _nano_content(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError("Nano message content must be text or a list of text parts.")

    fragments: list[str] = []
    for part in content:
        if isinstance(part, str):
            fragments.append(part)
            continue
        if not isinstance(part, Mapping):
            raise ValueError("Nano message content parts must be strings or objects.")
        part_type = part.get("type", part.get("t"))
        if part_type not in (None, "text"):
            raise ValueError(
                f"Nano text cookers do not support content type {part_type!r}."
            )
        text = part.get("text") or part.get("content") or part.get("value") or ""
        if not isinstance(text, str):
            raise ValueError("Nano text content values must be strings.")
        fragments.append(text)
    return "".join(fragments)


def _normalize_nano_messages(
    raw_messages: list[object],
) -> list[dict[str, Any]]:
    messages: list[dict[str, str]] = []
    for raw_message in raw_messages:
        if not isinstance(raw_message, Mapping):
            raise ValueError("Nano messages entries must be objects.")
        if "loss" in raw_message:
            raise ValueError("Nano text cookers do not accept explicit loss fields.")
        role = raw_message.get("role")
        if not isinstance(role, str):
            raise ValueError("Nano messages require a string role.")
        role = _NANO_ROLE_ALIASES.get(role, role)
        if role not in _NANO_ROLES:
            raise ValueError(f"Unsupported Nano message role {role!r}.")
        messages.append(
            {"role": role, "text": _nano_content(raw_message.get("content"))}
        )

    if not messages:
        raise ValueError("Nano text cookers require a non-empty messages list.")
    if messages[0]["role"] != "system":
        if messages[0]["text"].startswith(EMPTY_SYSTEM_CONTENT):
            messages[0]["text"] = messages[0]["text"].replace(EMPTY_SYSTEM_CONTENT, "")
        messages.insert(0, {"role": "system", "text": EMPTY_SYSTEM_CONTENT})
    elif messages[0]["text"] in (NO_TOOL_SYSTEM_CONTENT, LEGACY_SYSTEM_CONTENT):
        messages[0]["text"] = EMPTY_SYSTEM_CONTENT

    empty_think = "<|im_end|>\n<|im_start|>assistant\n<think></think>\n"
    for message in messages:
        if message["role"] == "tool":
            message["role"] = "user"
        if message["role"] == "user" and empty_think in message["text"]:
            message["text"] = message["text"].replace(
                empty_think, empty_think.removesuffix("\n")
            )
        elif message["role"] == "assistant":
            message["text"] = message["text"].rstrip() + "\n"

    open_think = "<|im_end|>\n<|im_start|>assistant\n<think>\n"
    for index, message in enumerate(messages):
        if message["role"] == "user" and index < len(messages) - 1:
            next_message = messages[index + 1]
            if message["text"].endswith(open_think) and next_message["text"].startswith(
                "\n</think>"
            ):
                message["text"] = message["text"].removesuffix(open_think) + (
                    "<|im_end|>\n<|im_start|>assistant\n<think></think>"
                )
                next_message["text"] = next_message["text"][
                    len("\n</think>") :
                ].lstrip()
        elif (
            message["role"] == "assistant"
            and index > 0
            and message["text"].startswith("\n")
            and messages[index - 1]["text"].endswith("\n")
        ):
            message["text"] = message["text"].lstrip()

    return [
        {
            "role": message["role"],
            "content": [{"type": "text", "text": message["text"]}],
        }
        for message in messages
    ]


def _raw_nano_messages(sample: CrudeSample) -> list[object]:
    messages = _decode_payload(sample).get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Nano text cookers require a non-empty messages list.")
    return messages


@supports_model_families("nemotron")
@stateless
@cooker(need_cache=True)
def cook_nano_openai_messages_jsonl(
    sample: CrudeSample,
    cache: CachePool | None = None,
    media_source: FileStore | None = None,
) -> CanonicalSFTSample:
    """Normalize one Nano OpenAI-style text conversation."""
    messages = _normalize_nano_messages(_raw_nano_messages(sample))
    if any(message["role"] == "system" for message in messages[1:]):
        raise ValueError("Nano JSONL supports only one leading system message.")
    return _canonical_sample(sample, messages=messages)


def _split_nano_messages(raw_messages: list[object]) -> list[list[object]]:
    conversations: list[list[object]] = []
    current: list[object] = []
    for message in raw_messages:
        if not isinstance(message, Mapping):
            raise ValueError("Nano messages entries must be objects.")
        if message.get("role") == "system" and current:
            conversations.append(current)
            current = []
        current.append(message)
    if current:
        conversations.append(current)
    return conversations


@supports_model_families("nemotron")
@stateless
@cooker(need_cache=True)
def cook_nano_openai_messages_offline_packed_jsonl(
    sample: CrudeSample,
    cache: CachePool | None = None,
    media_source: FileStore | None = None,
) -> CanonicalSFTSample:
    """Normalize a Nano row that contains several pre-packed conversations."""
    messages = [
        message
        for conversation in _split_nano_messages(_raw_nano_messages(sample))
        for message in _normalize_nano_messages(conversation)
    ]
    return _canonical_sample(
        sample,
        messages=messages,
        offline_packed_messages=True,
    )


def _media_descriptor(entry: Any) -> tuple[Any, FrozenMediaMetadata]:
    if not isinstance(entry, Mapping):
        return entry, ()
    metadata = freeze_media_metadata(entry.get("metadata"))
    for key in ("value", "path", "member"):
        if key in entry:
            return entry[key], metadata
    raise ValueError("Legacy audio media entries need a value, path, or member.")


def _media_parts(
    *,
    sample: CrudeSample,
    payload: Mapping[str, Any],
    tag: str,
    cache: CachePool | None,
    media_source: FileStore | None,
    media_sources: dict[str, FileStore],
    media: list[MediaRef],
    source_info: list[SourceInfo],
) -> list[dict[str, Any]]:
    prefixes = ("vis_video", "vis_sound") if tag == "video-sound" else (tag,)
    parts: list[dict[str, Any]] = []
    for prefix in prefixes:
        matches = [entry for key, entry in payload.items() if key.startswith(prefix)]
        if not matches:
            raise ValueError(
                f"Tag <{tag}> has no legacy field starting with {prefix!r}."
            )
        modality = "video" if prefix in ("video", "vis_video") else prefix
        modality = "audio" if modality in ("sound", "vis_sound") else modality
        for entry in matches:
            value, metadata = _media_descriptor(entry)
            if isinstance(value, str):
                value, store_metadata, source = _aux_media(
                    sample,
                    value,
                    modality=modality,
                    metadata=metadata,
                    cache=cache,
                    media_source=media_source,
                    media_sources=media_sources,
                    strip_matched_prefix=True,
                    basename_missing_absolute=False,
                    allow_local=False,
                    derive_missing_metadata=True,
                    missing_aux_source_is_error=True,
                )
                metadata = metadata or store_metadata
                if source is not None:
                    source_info.append(source)
            elif not metadata:
                metadata = _derived_media_metadata(modality, value)
            media_index = len(media)
            media.append(MediaRef(modality=modality, value=value, metadata=metadata))
            parts.append({"type": modality, "media_index": media_index})
    return parts


def _legacy_message(
    *,
    raw_message: object,
    tag_pattern: re.Pattern[str],
    normalize_tag: dict[str, str],
    resolve_tag: Callable[[str], list[dict[str, Any]]],
    seen_tags: set[str],
) -> dict[str, Any]:
    if not isinstance(raw_message, Mapping):
        raise ValueError("Legacy audio conversation turns must be objects.")
    sender = raw_message.get("from")
    if not isinstance(sender, str):
        raise ValueError("Legacy audio conversation turns require a string sender.")
    role = {"human": "user", "gpt": "assistant"}.get(sender)
    if role is None:
        raise ValueError(f"Unknown legacy audio sender {sender!r}.")
    value = raw_message.get("value")
    if not isinstance(value, str):
        raise ValueError("Legacy audio conversation values must be strings.")

    content: list[dict[str, Any]] = []
    for index, part in enumerate(re.split(tag_pattern, value)):
        if index % 2 == 1:
            canonical_tag = normalize_tag.get(part, part)
            if canonical_tag in seen_tags:
                raise ValueError(f"Tag <{canonical_tag}> appears more than once.")
            seen_tags.add(canonical_tag)
            content.extend(resolve_tag(canonical_tag))
        elif part.strip():
            content.append({"type": "text", "text": part})
    return {"role": role, "content": content}


@supports_model_families("nemotron")
@stateless
@cooker(need_cache=True, need_primary=True)
def cook_audio_conversation_jsonl(
    sample: CrudeSample,
    cache: CachePool | None = None,
    primary: FileStore | None = None,
    media_source: FileStore | None = None,
    **media_sources: FileStore,
) -> CanonicalSFTSample:
    """Cook the polylithic Nemotron audio-conversation source schema."""
    payload = _decode_payload(sample)
    raw_messages = payload.get("conversations")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError("Audio conversations require a non-empty conversations list.")
    media: list[MediaRef] = []
    source_info: list[SourceInfo] = []
    seen_tags: set[str] = set()
    messages = [
        _legacy_message(
            raw_message=message,
            tag_pattern=_AUDIO_TAG_PATTERN,
            normalize_tag={},
            resolve_tag=lambda tag: _media_parts(
                sample=sample,
                payload=payload,
                tag=tag,
                cache=cache,
                media_source=media_source,
                media_sources=media_sources,
                media=media,
                source_info=source_info,
            ),
            seen_tags=seen_tags,
        )
        for message in raw_messages
    ]
    for tag in ("image", "video", "sound", "video-sound"):
        if tag in payload and tag not in seen_tags:
            raise ValueError(
                f"Legacy media field {tag!r} is not used by a message tag."
            )
    return _canonical_sample(
        sample,
        messages=messages,
        media=media,
        extra_sources=tuple(source_info),
    )


def _omcat_member(sample: CrudeSample, tag: str) -> str:
    if tag == "video-sound":
        raise ValueError(
            "OMCAT video-sound has no defined member mapping in the pinned reference."
        )
    extensions = _OMCAT_MEMBER_EXTENSIONS[tag]
    for extension in extensions:
        if extension in sample:
            return extension
    for member in sample:
        if member.lower() in extensions:
            return member
    raise ValueError(
        f"OMCAT tag <{tag}> needs a member with extension {list(extensions)!r}."
    )


@supports_model_families("nemotron")
@stateless
@cooker(need_cache=True, need_primary=True)
def cook_omcat_legacy_conversation_monolithic(
    sample: CrudeSample,
    cache: CachePool | None = None,
    primary: FileStore | None = None,
) -> CanonicalSFTSample:
    """Cook the OMCAT monolithic extension-keyed source schema."""
    payload = _decode_payload(sample)
    for alias, canonical in _OMCAT_TAG_ALIASES.items():
        if alias not in payload:
            continue
        payload[canonical] = payload.pop(alias)

    raw_messages = payload.get("conversations")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError("OMCAT samples require a non-empty conversations list.")
    media: list[MediaRef] = []
    seen_tags: set[str] = set()

    def resolve_tag(tag: str) -> list[dict[str, Any]]:
        media_index = len(media)
        modality = "audio" if tag == "sound" else tag
        member = _omcat_member(sample, tag)
        value, metadata = _primary_media(
            sample,
            member,
            modality=modality,
            metadata=(),
            cache=cache,
            primary=primary,
        )
        media.append(MediaRef(modality=modality, value=value, metadata=metadata))
        return [{"type": modality, "media_index": media_index}]

    messages = [
        _legacy_message(
            raw_message=message,
            tag_pattern=_OMCAT_TAG_PATTERN,
            normalize_tag=_OMCAT_TAG_ALIASES,
            resolve_tag=resolve_tag,
            seen_tags=seen_tags,
        )
        for message in raw_messages
    ]
    for tag in ("image", "video", "sound", "video-sound"):
        if tag in payload and tag not in seen_tags:
            raise ValueError(f"OMCAT media field {tag!r} is not used by a message tag.")
    return _canonical_sample(sample, messages=messages, media=media)


__all__ = [
    "EMPTY_SYSTEM_CONTENT",
    "GRANARY_ENGLISH_PROMPT",
    "LEGACY_SYSTEM_CONTENT",
    "NO_TOOL_SYSTEM_CONTENT",
    "cook_audio_conversation_jsonl",
    "cook_general_conversations_jsonl",
    "cook_general_conversations_jsonl_explicit_loss_v1",
    "cook_general_conversations_webdataset",
    "cook_granary_english_jsonl",
    "cook_granary_english_webdataset",
    "cook_nano_openai_messages_jsonl",
    "cook_nano_openai_messages_offline_packed_jsonl",
    "cook_nemotron_conversation",
    "cook_omcat_legacy_conversation_monolithic",
]
