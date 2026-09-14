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

import hashlib
import io
import json
import os
import traceback
import urllib.parse
from collections.abc import Callable
from typing import Any, Iterator, Literal, Mapping, Protocol, cast

import torch
from megatron.energon import (
    Cooker,
    CrudeSample,
    FileStoreCachePool,
    SourceInfo,
    WorkerConfig,
    get_savable_loader,
    get_train_dataset,
    get_val_dataset,
)

from nemo_rl.data.energon.config import EnergonLoaderConfig, EnergonSourceConfig
from nemo_rl.data.energon.multimodal.registry import (
    COOKER_REGISTRY,
    TASK_ENCODER_REGISTRY,
    selected_registry_identity,
)
from nemo_rl.data.energon.multimodal.task_encoders.base import BaseSFTTaskEncoder
from nemo_rl.data.energon.multimodal.task_encoders.generic_sft import (
    build_processor_adapter,
)
from nemo_rl.data.energon.multimodal.types import CanonicalSFTSample
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

_V2_STATE_FORMAT_VERSION = 2


def compact_sample_error_handler(
    exception: Exception,
    sample: Any | list[Any],
    sources: list[SourceInfo] | None = None,
) -> None:
    """Log a sample-processing error and let Energon request another sample."""
    if isinstance(exception, AssertionError):
        if sources is None:
            print(f"Assertion error in sample {str(sample)[:100]}: {exception}")
            return

        print(f"Assertion error: {exception}")
        if os.environ.get("NRL_ENERGON_SAMPLE_VIEWER") == "1":
            data = [
                {
                    "dataset_path": str(source.dataset_path),
                    "index": source.index,
                    "shard_name": source.shard_name,
                    "file_names": list(source.file_names),
                }
                for source in sources
            ]
            url = (
                "vscode://nvidia.energon-sample-viewer/open?data="
                f"{urllib.parse.quote(json.dumps(data))}"
            )
            print(f"(Ctrl+)Click to view sample in energon viewer: {url}")
        return

    print("Ignoring error processing sample:")
    traceback.print_exc()


class SFTDataLoader(Protocol):
    """Iterator and state methods consumed by the SFT algorithm."""

    def __iter__(self) -> Iterator[BatchedDataDict[Any]]: ...

    def __len__(self) -> int: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...


def _identity_fingerprint(identity: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _brief(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 80 else f"{text[:77]}..."


def _identity_differences(saved: Any, current: Any, path: str = "") -> list[str]:
    """List the identity entries that changed, as `key old -> new` lines."""
    if isinstance(saved, Mapping) and isinstance(current, Mapping):
        differences: list[str] = []
        for key in sorted(set(saved) | set(current)):
            child = f"{path}.{key}" if path else str(key)
            if key not in saved:
                differences.append(f"{child} added {_brief(current[key])}")
            elif key not in current:
                differences.append(f"{child} removed {_brief(saved[key])}")
            else:
                differences.extend(
                    _identity_differences(saved[key], current[key], child)
                )
        return differences
    if saved != current:
        return [f"{path} {_brief(saved)} -> {_brief(current)}"]
    return []


class EnergonSFTDataLoader:
    """Expose Energon rank state through NeMo-RL's dataloader interface."""

    def __init__(self, loader: Any, *, identity: Mapping[str, Any]) -> None:
        self._loader = loader
        self._identity = dict(identity)
        self._fingerprint = _identity_fingerprint(self._identity)
        self._iteration_started = False

    def __iter__(self) -> Iterator[BatchedDataDict[Any]]:
        self._iteration_started = True
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self._loader)

    def state_dict(self) -> dict[str, Any]:
        buffer = io.BytesIO()
        torch.save(self._loader.save_state_rank(), buffer)
        return {
            "backend": "energon",
            "format_version": _V2_STATE_FORMAT_VERSION,
            "fingerprint": self._fingerprint,
            # The payload the fingerprint hashes. Kept so a rejected restore can
            # name the settings that moved instead of only reporting a hash
            # mismatch. It holds JSON scalars, lists and dicts only, so the
            # outer checkpoint stays weights-only loadable.
            "identity": self._identity,
            # The outer NeMo-RL checkpoint stays compatible with torch.load's
            # weights-only default. Energon classes are decoded after validation.
            "loader_state": torch.frombuffer(
                bytearray(buffer.getvalue()), dtype=torch.uint8
            ).clone(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if self._iteration_started:
            raise RuntimeError(
                "Energon loader state must be restored before iteration."
            )
        if state.get("backend") != "energon":
            raise ValueError("Cannot restore non-Energon state into an Energon loader.")
        if state.get("format_version") != _V2_STATE_FORMAT_VERSION:
            raise ValueError(
                "Unsupported Energon loader state format "
                f"{state.get('format_version')!r}."
            )
        if state.get("fingerprint") != self._fingerprint:
            saved_identity = state.get("identity")
            differences = (
                _identity_differences(saved_identity, self._identity)
                if isinstance(saved_identity, Mapping)
                else []
            )
            # A state whose identity is missing or malformed carries nothing to
            # diff against, so report the mismatch without naming a setting.
            detail = (
                "; ".join(differences)
                if differences
                else "dataset, processor, or loader settings changed"
            )
            raise ValueError(f"Energon loader identity changed: {detail}")
        loader_state = state.get("loader_state")
        if (
            not isinstance(loader_state, torch.Tensor)
            or loader_state.dtype != torch.uint8
        ):
            raise ValueError("Energon loader state payload is invalid.")
        decoded = torch.load(
            io.BytesIO(loader_state.cpu().numpy().tobytes()),
            weights_only=False,
        )
        self._loader.restore_state_rank(decoded)


def _source_config(value: Any, *, name: str) -> EnergonSourceConfig:
    try:
        return EnergonSourceConfig.model_validate(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid Energon {name} source configuration.") from error


def _loader_config(value: Any) -> EnergonLoaderConfig:
    try:
        config = EnergonLoaderConfig.model_validate(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid Energon loader configuration.") from error
    if config.topology_mapper != "default":
        raise ValueError(
            f"Unknown data-loader topology mapper {config.topology_mapper!r}."
        )
    if config.task_encoder.name == "generic_sft" and config.task_encoder.options:
        raise ValueError(
            f"Task encoder {config.task_encoder.name!r} has no configurable options."
        )
    if not config.cookers:
        raise ValueError("At least one Energon cooker must be configured.")
    if any(cooker.options for cooker in config.cookers):
        raise ValueError("The generic conversation cooker has no options.")
    fallback_cookers = [
        index
        for index, cooker in enumerate(config.cookers)
        if cooker.has_subflavors is None
    ]
    if len(fallback_cookers) > 1:
        raise ValueError(
            "At most one Energon cooker can omit has_subflavors when several "
            "cookers are configured."
        )
    if fallback_cookers and fallback_cookers[0] != len(config.cookers) - 1:
        raise ValueError("An Energon fallback cooker must be the final cooker.")
    filters = [
        cooker.has_subflavors
        for cooker in config.cookers
        if cooker.has_subflavors is not None
    ]
    if any(current in filters[:index] for index, current in enumerate(filters)):
        raise ValueError("Energon cooker has_subflavors filters must be unique.")
    TASK_ENCODER_REGISTRY.resolve_for_model_family(
        config.task_encoder.name,
        model_family=config.model_family,
    )
    for cooker in config.cookers:
        COOKER_REGISTRY.resolve_for_model_family(
            cooker.name,
            model_family=config.model_family,
        )
    return config


def _v2_topology(
    *,
    loader_config: EnergonLoaderConfig,
    placement_fingerprint: str,
    logical_rank: int,
    logical_world_size: int,
) -> dict[str, Any]:
    """Describe the DP shard a V2 loader state belongs to.

    This is what makes one DP shard refuse another shard's state, so it is
    hashed into the loader identity by :func:`_loader_identity`.
    """
    return {
        "mapper": loader_config.topology_mapper,
        "placement": placement_fingerprint,
        "logical_rank": logical_rank,
        "logical_world_size": logical_world_size,
    }


def _loader_identity(
    *,
    source: EnergonSourceConfig,
    loader_config: EnergonLoaderConfig,
    adapter_fingerprint: str,
    split_role: str,
    batch_size: int,
    shuffle: bool | None,
    topology: dict[str, Any],
) -> dict[str, Any]:
    """Describe what a restored loader must still agree with."""
    return {
        "source": source.model_dump(mode="json"),
        "loader": loader_config.model_dump(mode="json"),
        "adapter": adapter_fingerprint,
        "split_role": split_role,
        # Energon rescales a restored worker offset only when it can find a
        # BatchDataset, and overriding batch_group_criterion substitutes a
        # sibling GroupBatchDataset, so a changed batch size would otherwise
        # resume mid-stream and silently replay samples.
        "batch_size": batch_size,
        # Toggling data.shuffle reorders shard slices as well as samples, so a
        # restored offset would point into a stream the saved state never saw.
        # None for validation, which is not shuffled either way.
        "shuffle": shuffle,
        # Bumping the format version invalidates saved fingerprints, which is
        # what a change to this payload's shape needs.
        "state_format_version": _V2_STATE_FORMAT_VERSION,
        "registries": selected_registry_identity(
            task_encoder=loader_config.task_encoder.name,
            cookers=[cooker.name for cooker in loader_config.cookers],
        ),
        "topology": topology,
    }


def _worker_config(
    config: EnergonLoaderConfig, *, logical_rank: int, logical_world_size: int
) -> WorkerConfig:
    if logical_world_size <= 0:
        raise ValueError("Logical data world size must be positive.")
    if not 0 <= logical_rank < logical_world_size:
        raise ValueError(
            f"Logical data rank {logical_rank} is outside world size "
            f"{logical_world_size}."
        )
    return WorkerConfig(
        rank=logical_rank,
        world_size=logical_world_size,
        num_workers=config.num_workers,
        seed_offset=config.seed_offset,
        global_error_handler=compact_sample_error_handler,
        restore_error_handler=compact_sample_error_handler,
    )


def _task_encoder(
    *,
    loader_config: EnergonLoaderConfig,
    adapter: Any,
    include_source_ids: bool,
) -> BaseSFTTaskEncoder:
    cooker_functions = [
        Cooker(
            cast(
                Callable[[CrudeSample], CanonicalSFTSample],
                COOKER_REGISTRY.resolve(cooker.name),
            ),
            has_subflavors=cooker.has_subflavors,
        )
        for cooker in loader_config.cookers
    ]
    encoder_type = cast(
        Any, TASK_ENCODER_REGISTRY.resolve(loader_config.task_encoder.name)
    )
    encoder_options: dict[str, Any] = dict(loader_config.task_encoder.options)
    return cast(
        BaseSFTTaskEncoder,
        encoder_type(
            adapter=adapter,
            cooker_functions=cooker_functions,
            include_source_ids=include_source_ids,
            **encoder_options,
        ),
    )


def build_energon_sft_loader(
    *,
    data_config: Mapping[str, Any],
    source: Mapping[str, Any] | EnergonSourceConfig,
    processor: Any,
    batch_size: int,
    max_sequence_length: int,
    split_role: Literal["train", "validation"],
    logical_rank: int,
    logical_world_size: int,
    placement_fingerprint: str,
) -> EnergonSFTDataLoader:
    """Build one loader for an explicit logical data shard and split."""
    if "energon" not in data_config:
        raise ValueError("data.backend=energon requires a data.energon block.")
    if processor is None:
        raise ValueError("data.backend=energon requires a multimodal processor.")
    if batch_size <= 0:
        raise ValueError("Energon SFT batch size must be positive.")
    if not placement_fingerprint:
        raise ValueError("SFTv2 requires a non-empty placement fingerprint.")

    resolved_source = _source_config(source, name=split_role)
    loader_config = _loader_config(data_config["energon"])
    adapter = build_processor_adapter(
        processor_adapter=loader_config.processor_adapter,
        processor=processor,
        max_sequence_length=max_sequence_length,
        add_bos=data_config.get("add_bos", True),
        add_eos=data_config.get("add_eos", True),
        add_generation_prompt=data_config.get("add_generation_prompt", False),
    )
    task_encoder = _task_encoder(
        loader_config=loader_config,
        adapter=adapter,
        include_source_ids=True,
    )
    worker_config = _worker_config(
        loader_config,
        logical_rank=logical_rank,
        logical_world_size=logical_world_size,
    )

    if split_role == "train":
        if resolved_source.virtual_epoch_length <= 0:
            raise ValueError(
                "Energon training requires train.virtual_epoch_length in batches."
            )
        if not data_config["shuffle"]:
            raise ValueError("Energon training requires shuffle=true.")

        dataset = get_train_dataset(
            resolved_source.path,
            split_part=resolved_source.split,
            worker_config=worker_config,
            batch_size=batch_size,
            batch_drop_last=True,
            shuffle_buffer_size=(loader_config.shuffle_buffer_size),
            shuffle_over_epochs_multiplier=1,
            max_samples_per_sequence=None,
            virtual_epoch_length=resolved_source.virtual_epoch_length,
            task_encoder=task_encoder,
        )
    else:
        dataset = get_val_dataset(
            resolved_source.path,
            split_part=resolved_source.split,
            worker_config=worker_config,
            batch_size=batch_size,
            batch_drop_last=False,
            limit=resolved_source.limit,
            task_encoder=task_encoder,
        )

    cache_pool = (
        FileStoreCachePool(method="raw")
        if any(cooker.need_cache for cooker in task_encoder.cookers)
        else None
    )
    loader = get_savable_loader(
        dataset,
        cache_pool=cache_pool,
        checkpoint_every_sec=loader_config.checkpoint_every_sec,
        prefetch_factor=loader_config.prefetch_factor,
        watchdog_timeout_seconds=loader_config.watchdog_timeout_seconds,
        fail_on_timeout=True,
    )
    return EnergonSFTDataLoader(
        loader,
        identity=_loader_identity(
            source=resolved_source,
            loader_config=loader_config,
            adapter_fingerprint=adapter.fingerprint,
            split_role=split_role,
            batch_size=batch_size,
            shuffle=bool(data_config["shuffle"]) if split_role == "train" else None,
            topology=_v2_topology(
                loader_config=loader_config,
                placement_fingerprint=placement_fingerprint,
                logical_rank=logical_rank,
                logical_world_size=logical_world_size,
            ),
        ),
    )


__all__ = [
    "EnergonSFTDataLoader",
    "SFTDataLoader",
    "build_energon_sft_loader",
]
