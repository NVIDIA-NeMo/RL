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

"""Opt-in, row-scoped forensic dumps for sequence TMPE outliers.

The configured output path is also the feature gate: an unset or blank
``NEMO_RL_TMPE_OUTLIER_DUMP_PATH`` leaves the diagnostic completely disabled.
The diagnostic never supplies a training mask.  It only compares the rollout
and freshly-computed previous-policy logprobs, selects rows whose sequence TMPE
exceeds its own diagnostic threshold, and snapshots those rows to atomic
``torch.save`` files.

After successful advantage writeback, a small ``.admission.json`` sidecar can
bind the raw dump to the actual final owner-wide mask. It records admission
only, independently of whether a later optimizer update commits.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from nemo_rl.algorithms.utils import compute_seq_logprob_errors
from nemo_rl.data.multimodal_utils import PackedTensor
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.schema import ROUTED_EXPERTS_FIELD

TMPE_OUTLIER_DUMP_PATH_ENV = "NEMO_RL_TMPE_OUTLIER_DUMP_PATH"
TMPE_OUTLIER_THRESHOLD_ENV = "NEMO_RL_TMPE_OUTLIER_THRESHOLD"
DEFAULT_TMPE_OUTLIER_THRESHOLD = 1.5
TMPE_OUTLIER_DUMP_SCHEMA_VERSION = 1

TMPE_METRIC_FIELDS = (
    "token_mask",
    "generation_logprobs",
    "prev_logprobs",
)
TMPE_OPTIONAL_MEDIA_FIELDS = (
    "pixel_values",
    "imgs_sizes",
    "num_frames",
)
TMPE_OPTIONAL_FIELDS = (*TMPE_OPTIONAL_MEDIA_FIELDS, ROUTED_EXPERTS_FIELD)
TMPE_OUTLIER_DUMP_FIELDS = (
    "input_ids",
    "input_lengths",
    *TMPE_METRIC_FIELDS,
    *TMPE_OPTIONAL_FIELDS,
)


@dataclass(frozen=True)
class TMPEOutlierDumpConfig:
    """Resolved diagnostic-only settings."""

    output_path: Path
    threshold: float


@dataclass(frozen=True)
class TMPEOutlierDumpRef:
    """Bind one selected sample to its unique, immutable raw dump."""

    sample_id: str
    path: Path


def tmpe_outlier_dump_config_from_env(
    environ: Mapping[str, str] | None = None,
) -> TMPEOutlierDumpConfig | None:
    """Resolve the opt-in dump configuration from an environment mapping.

    No output path means disabled.  Keeping the default-off decision at the
    first call site guarantees that normal training does not perform an extra
    data-plane read or tensor comparison.
    """
    env = os.environ if environ is None else environ
    raw_output_path = env.get(TMPE_OUTLIER_DUMP_PATH_ENV, "").strip()
    if not raw_output_path:
        return None

    raw_threshold = env.get(
        TMPE_OUTLIER_THRESHOLD_ENV,
        str(DEFAULT_TMPE_OUTLIER_THRESHOLD),
    ).strip()
    try:
        threshold = float(raw_threshold)
    except ValueError as error:
        raise ValueError(
            f"{TMPE_OUTLIER_THRESHOLD_ENV} must be a finite positive number; "
            f"got {raw_threshold!r}."
        ) from error
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError(
            f"{TMPE_OUTLIER_THRESHOLD_ENV} must be a finite positive number; "
            f"got {raw_threshold!r}."
        )

    return TMPEOutlierDumpConfig(
        output_path=Path(raw_output_path).expanduser(),
        threshold=threshold,
    )


def _dense_tensor(value: Any, field_name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"TMPE outlier diagnostic expected tensor field {field_name!r}; "
            f"got {type(value).__name__}."
        )
    if value.is_nested:
        return torch.nested.to_padded_tensor(value, padding=0)
    return value


@torch.no_grad()
def find_tmpe_outlier_rows(
    data: Mapping[str, Any],
    *,
    threshold: float,
) -> tuple[list[int], torch.Tensor]:
    """Return raw sequence-TMPE outlier indices and all per-row errors.

    The first token is omitted exactly as in the training TMPE metric.  An
    all-ones sample mask deliberately makes this a raw diagnostic: later
    environment, overlong, and TMPE training masks do not influence which
    forensic rows are retained.  Non-finite values are always outliers.
    """
    token_mask = _dense_tensor(data["token_mask"], "token_mask")[:, 1:]
    generation_logprobs = _dense_tensor(
        data["generation_logprobs"], "generation_logprobs"
    )[:, 1:]
    prev_logprobs = _dense_tensor(data["prev_logprobs"], "prev_logprobs")[:, 1:]
    if not (token_mask.shape == generation_logprobs.shape == prev_logprobs.shape):
        raise ValueError(
            "TMPE outlier diagnostic requires aligned token_mask, "
            "generation_logprobs, and prev_logprobs after the first-token shift; "
            f"got {tuple(token_mask.shape)}, {tuple(generation_logprobs.shape)}, "
            f"and {tuple(prev_logprobs.shape)}."
        )

    sequence_errors, valid_rows = compute_seq_logprob_errors(
        policy_logprobs=prev_logprobs,
        generation_logprobs=generation_logprobs,
        token_mask=token_mask,
        sample_mask=torch.ones(
            token_mask.shape[0],
            dtype=token_mask.dtype,
            device=token_mask.device,
        ),
    )
    outlier_mask = valid_rows & (
        ~torch.isfinite(sequence_errors) | (sequence_errors > threshold)
    )
    indices = outlier_mask.nonzero(as_tuple=False).flatten().tolist()
    return indices, sequence_errors.detach().to(device="cpu")


def _snapshot_packed_row(value: PackedTensor, row: int) -> PackedTensor:
    snapshot = copy.deepcopy(value.slice([row]))
    snapshot.tensors = [
        None if tensor is None else tensor.detach().to(device="cpu").clone()
        for tensor in snapshot.tensors
    ]
    return snapshot


def _snapshot_row(value: Any, row: int) -> Any:
    """Detach one logical batch row without mutating its training value."""
    if isinstance(value, PackedTensor):
        return _snapshot_packed_row(value, row)
    if isinstance(value, torch.Tensor):
        selected = value[row]
        return selected.detach().to(device="cpu").clone()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return copy.deepcopy(value[row])
    try:
        return copy.deepcopy(value[row])
    except (IndexError, KeyError, TypeError) as error:
        raise TypeError(
            "TMPE outlier diagnostic cannot select a batch row from "
            f"{type(value).__name__}."
        ) from error


def _atomic_torch_save(payload: dict[str, Any], target: Path) -> None:
    """Write ``payload`` with a same-directory temp file and atomic rename."""
    target.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as output:
            file_descriptor = -1
            torch.save(payload, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, target)
    except BaseException:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        temporary_path.unlink(missing_ok=True)
        raise


def dump_tmpe_outlier_rows(
    *,
    data: Mapping[str, Any],
    meta: KVBatchMeta,
    sequence_errors: torch.Tensor,
    config: TMPEOutlierDumpConfig,
    step: int,
    trainer_version: int,
) -> list[Path]:
    """Atomically dump the supplied outlier-only batch, one file per row."""
    if sequence_errors.ndim != 1 or sequence_errors.numel() != meta.size:
        raise ValueError(
            "TMPE outlier dump errors must align 1:1 with meta.sample_ids; "
            f"got {tuple(sequence_errors.shape)} for {meta.size} samples."
        )

    written: list[Path] = []
    tags = meta.tags or [{} for _ in range(meta.size)]
    for row, (sample_id, sequence_error) in enumerate(
        zip(meta.sample_ids, sequence_errors, strict=True)
    ):
        error_value = float(sequence_error.item())
        if math.isfinite(error_value) and error_value <= config.threshold:
            # Defense in depth: even if a caller passes a wider batch, a
            # non-outlier can never land in this forensic directory.
            continue
        sample_digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:12]
        filename = (
            f"tmpe-outlier-step-{step:08d}-version-{trainer_version:08d}-"
            f"row-{row:06d}-{sample_digest}-{uuid.uuid4().hex}.pt"
        )
        target = config.output_path / filename
        payload: dict[str, Any] = {
            "schema_version": TMPE_OUTLIER_DUMP_SCHEMA_VERSION,
            "step": step,
            "trainer_version": trainer_version,
            "sample_id": sample_id,
            "tags": copy.deepcopy(tags[row]),
            "sequence_length": (
                meta.sequence_lengths[row]
                if meta.sequence_lengths is not None
                else None
            ),
            "sequence_tmpe": error_value,
            "threshold": config.threshold,
        }
        for field_name in TMPE_OUTLIER_DUMP_FIELDS:
            payload[field_name] = (
                _snapshot_row(data[field_name], row) if field_name in data else None
            )
        _atomic_torch_save(payload, target)
        written.append(target)
    return written


def dump_tmpe_outlier_admissions(
    *,
    dumps: Sequence[TMPEOutlierDumpRef],
    meta: KVBatchMeta,
    input_sample_mask: torch.Tensor,
    mask_sample: torch.Tensor,
    truncated: torch.Tensor,
    pre_owner_sample_mask: torch.Tensor,
    final_sample_mask: torch.Tensor,
    token_mask: torch.Tensor,
    overlong_filtering: bool,
    seq_logprob_error_threshold: float | None,
    step: int,
    trainer_version: int,
) -> list[Path]:
    """Record actual admission after advantage writeback, without tensor reloads.

    Each small JSON sidecar names the unique pre-filter raw dump. A sidecar
    proves mask writeback succeeded, not that the later optimizer committed.
    Missing sidecars leave admission unknown; raw outliers remain untouched.
    """
    if not dumps:
        return []
    rows = {sample_id: row for row, sample_id in enumerate(meta.sample_ids)}
    if len(rows) != meta.size:
        raise ValueError("TMPE admission requires unique sample IDs")
    columns = {
        "input_sample_mask": input_sample_mask,
        "mask_sample": mask_sample,
        "truncated": truncated,
        "pre_owner_sample_mask": pre_owner_sample_mask,
        "final_sample_mask": final_sample_mask,
    }
    if any(value.shape != (meta.size,) for value in columns.values()):
        raise ValueError("TMPE admission masks must align 1:1 with sample IDs")
    if token_mask.ndim != 2 or token_mask.shape[0] != meta.size:
        raise ValueError("TMPE admission token mask must align with sample IDs")
    if len({dump.path for dump in dumps}) != len(dumps):
        raise ValueError("TMPE admission requires unique raw dump paths")
    # Only already-computed CPU metadata is copied; no model or media reload.
    snapshots = {name: value.detach().cpu().tolist() for name, value in columns.items()}
    shifted_target_counts = token_mask.detach().cpu()[:, 1:].bool().sum(-1).tolist()
    written: list[Path] = []
    for dump in dumps:
        if dump.sample_id not in rows:
            raise ValueError(f"Unknown TMPE admission sample {dump.sample_id!r}")
        if not dump.path.is_file():
            raise FileNotFoundError(dump.path)
        row = rows[dump.sample_id]
        payload = {
            "schema_version": 1,
            "scope": "post_advantage_writeback; not_optimizer_commit",
            "raw_dump": dump.path.name,
            "step": step,
            "trainer_version": trainer_version,
            "partition_id": meta.partition_id,
            "sample_id": dump.sample_id,
            **{name: values[row] for name, values in snapshots.items()},
            "overlong_filtering": overlong_filtering,
            "seq_logprob_error_threshold": seq_logprob_error_threshold,
            "shifted_target_tokens": shifted_target_counts[row],
            "admitted": bool(
                snapshots["final_sample_mask"][row] and shifted_target_counts[row]
            ),
        }
        target = dump.path.with_suffix(".admission.json")
        descriptor, temporary_name = tempfile.mkstemp(
            dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w") as output:
                descriptor = -1
                json.dump(payload, output, sort_keys=True, allow_nan=False)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_path, target)
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            temporary_path.unlink(missing_ok=True)
            raise
        written.append(target)
    return written


__all__ = [
    "DEFAULT_TMPE_OUTLIER_THRESHOLD",
    "TMPE_METRIC_FIELDS",
    "TMPE_OPTIONAL_FIELDS",
    "TMPE_OPTIONAL_MEDIA_FIELDS",
    "TMPE_OUTLIER_DUMP_FIELDS",
    "TMPE_OUTLIER_DUMP_PATH_ENV",
    "TMPE_OUTLIER_THRESHOLD_ENV",
    "TMPEOutlierDumpConfig",
    "TMPEOutlierDumpRef",
    "dump_tmpe_outlier_admissions",
    "dump_tmpe_outlier_rows",
    "find_tmpe_outlier_rows",
    "tmpe_outlier_dump_config_from_env",
]
