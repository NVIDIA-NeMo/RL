# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded original-call forensic archives, independent of training semantics.

The existing opt-in TMPE dump directory enables this diagnostic. A shared,
nonblocking file lock and write-ahead byte reservations bound all finalizers
together. A failed or interrupted capture can consume a reservation, but can
never make later writers exceed the budget. No existing archive is removed.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, cast

import torch

ORIGINAL_CALL_EVIDENCE_PATH_TAG = "original_call_evidence_path"
ORIGINAL_CALL_EVIDENCE_SHA256_TAG = "original_call_evidence_sha256"
ORIGINAL_CALL_EVIDENCE_SCHEMA_VERSION = 1
ORIGINAL_CALL_EVIDENCE_TOTAL_BYTES = 8 * 1024**3
ORIGINAL_CALL_EVIDENCE_CONTEXT_BYTES = 2 * 1024**3
ORIGINAL_CALL_EVIDENCE_CONTEXTS_PER_WORKER = 2

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class OriginalCallEvidenceConfig:
    """Resolved internal diagnostic limits; absent from training config hashes."""

    output_path: str
    max_total_bytes: int
    max_context_bytes: int
    max_contexts_per_worker: int

    def __post_init__(self) -> None:
        if (
            not self.output_path.strip()
            or min(
                self.max_total_bytes,
                self.max_context_bytes,
                self.max_contexts_per_worker,
            )
            <= 0
        ):
            raise ValueError(
                "Original-call evidence requires a path and positive limits"
            )


def original_call_evidence_config(
    outlier_output_path: Path | None,
) -> OriginalCallEvidenceConfig | None:
    """Derive a bounded archive from the already resolved TMPE diagnostic opt-in."""
    if outlier_output_path is None:
        return None
    return OriginalCallEvidenceConfig(
        output_path=str(outlier_output_path / "original_calls"),
        max_total_bytes=ORIGINAL_CALL_EVIDENCE_TOTAL_BYTES,
        max_context_bytes=ORIGINAL_CALL_EVIDENCE_CONTEXT_BYTES,
        max_contexts_per_worker=ORIGINAL_CALL_EVIDENCE_CONTEXTS_PER_WORKER,
    )


@dataclass(frozen=True)
class OriginalCallEvidenceRef:
    """Only this small identity may accompany a canonical training row."""

    path: str
    sha256: str


@dataclass
class _LimitedWriter:
    destination: IO[bytes]
    limit: int

    def write(self, data: bytes) -> int:
        if self.destination.tell() + len(data) > self.limit:
            raise RuntimeError("Original-call evidence byte reservation exceeded")
        return self.destination.write(data)

    def flush(self) -> None:
        self.destination.flush()


def _atomic_index(index: dict[str, Any], target: Path) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=target.parent, prefix=".index-", delete=False
        ) as destination:
            temporary = Path(destination.name)
            json.dump(index, destination, sort_keys=True)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _receipt_identity(receipt: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class OriginalCallEvidenceWriter:
    """Best-effort writer for already verified selected contexts only.

    Calls are synchronous within each finalizer. Other finalizers skip a busy
    lock instead of waiting for diagnostic I/O. Each selected tensor is cloned
    only after reserving space, avoiding retention of an entire fetched batch's
    backing storage. Training publication and staging cleanup remain unchanged.
    """

    def __init__(self, config: OriginalCallEvidenceConfig) -> None:
        self.config = config
        self._written: set[str] = set()
        self._disabled = False

    @property
    def wants_capture(self) -> bool:
        return (
            not self._disabled
            and len(self._written) < self.config.max_contexts_per_worker
        )

    def reference_for_verified_receipt(
        self, receipt: dict[str, Any]
    ) -> OriginalCallEvidenceRef | None:
        """Reattach existing evidence on retries, even after exhausting the cap."""
        root = Path(self.config.output_path).absolute()
        try:
            index_path = root / "index.json"
            if not index_path.exists():
                return None
            entry = json.loads(index_path.read_text())["entries"].get(
                _receipt_identity(receipt)
            )
            if entry is not None and entry["status"] == "committed":
                target = root / entry["filename"]
                if target.stat().st_size == entry["bytes"]:
                    return OriginalCallEvidenceRef(str(target), entry["sha256"])
        except (OSError, TypeError, ValueError, KeyError) as error:
            self._disabled = True
            _LOGGER.warning("Original-call evidence lookup disabled: %s", error)
        return None

    def write_verified_context(
        self,
        *,
        receipt: dict[str, Any],
        rows: dict[str, dict[str, torch.Tensor]],
        context_compaction: bool,
        capture_media: bool,
        action_flags: list[dict[str, bool]] | None,
    ) -> OriginalCallEvidenceRef | None:
        """Persist original wire tensors before their existing staging release.

        The caller must already have verified the base snapshots, exact selected
        chain, extras/route commitments and media. Only those selected keys may
        appear here. The archive contains neither newly routed tensors nor a
        claim that the generation/refit timeline was independently audited.
        """
        if not self.wants_capture:
            return None
        if set(rows) != {record["staging_key"] for record in receipt["manifest"]}:
            raise ValueError("Evidence rows must exactly cover the verified selection")
        identity = _receipt_identity(receipt)
        root = Path(self.config.output_path).absolute()
        temporary = None
        try:
            root.mkdir(parents=True, exist_ok=True)
            with (root / ".budget.lock").open("ab") as lock:
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return None
                index_path = root / "index.json"
                index = (
                    json.loads(index_path.read_text())
                    if index_path.exists()
                    else {
                        "schema_version": ORIGINAL_CALL_EVIDENCE_SCHEMA_VERSION,
                        "entries": {},
                    }
                )
                entries = index["entries"]
                if (
                    index["schema_version"] != ORIGINAL_CALL_EVIDENCE_SCHEMA_VERSION
                    or not isinstance(entries, dict)
                    or any(
                        type(entry["bytes"]) is not int
                        or entry["bytes"] <= 0
                        or entry["status"] not in {"reserved", "committed"}
                        for entry in entries.values()
                    )
                ):
                    raise ValueError("Invalid original-call evidence budget index")
                existing = entries.get(identity)
                if existing is not None:
                    if existing["status"] == "committed":
                        target = root / existing["filename"]
                        if target.stat().st_size == existing["bytes"]:
                            self._written.add(identity)
                            return OriginalCallEvidenceRef(
                                str(target), existing["sha256"]
                            )
                    return None
                used = sum(entry["bytes"] for entry in entries.values())
                reservation = min(
                    self.config.max_context_bytes, self.config.max_total_bytes - used
                )
                tensor_bytes = sum(
                    value.numel() * value.element_size()
                    for columns in rows.values()
                    for value in columns.values()
                )
                if reservation <= 0:
                    self._disabled = True
                    return None
                if tensor_bytes >= reservation:
                    return None
                filename = f"original-calls-{identity}.pt"
                entries[identity] = {
                    "status": "reserved",
                    "filename": filename,
                    "bytes": reservation,
                }
                # The reservation survives crashes, serialization failures and
                # index update failures. Never infer unused bytes from filenames.
                _atomic_index(index, index_path)
                payload = {
                    "schema_version": ORIGINAL_CALL_EVIDENCE_SCHEMA_VERSION,
                    "receipt": receipt,
                    "rows": {
                        key: {
                            field: value.detach().cpu().clone()
                            for field, value in columns.items()
                        }
                        for key, columns in rows.items()
                    },
                    "context_compaction": context_compaction,
                    "capture_media": capture_media,
                    "action_flags": action_flags,
                    "native_base_and_route_commitments_verified": True,
                    "refit_free_generation_verified": False,
                }
                with tempfile.NamedTemporaryFile(
                    mode="wb", dir=root, prefix=".original-calls-", delete=False
                ) as destination:
                    temporary = Path(destination.name)
                    # torch.save's documented file-like contract is write +
                    # flush; its IO annotation also lists unused read methods.
                    torch.save(
                        payload,
                        cast(IO[bytes], _LimitedWriter(destination, reservation)),
                    )
                    destination.flush()
                    os.fsync(destination.fileno())
                size = temporary.stat().st_size
                with temporary.open("rb") as source:
                    digest = hashlib.file_digest(source, "sha256").hexdigest()
                target = root / filename
                os.replace(temporary, target)
                entries[identity] = {
                    "status": "committed",
                    "filename": filename,
                    "bytes": size,
                    "sha256": digest,
                }
                _atomic_index(index, index_path)
                self._written.add(identity)
                return OriginalCallEvidenceRef(str(target), digest)
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
            # Diagnostic storage or serialization errors must never reject an
            # otherwise sound training row or bypass its normal staging cleanup.
            self._disabled = True
            _LOGGER.warning("Original-call evidence capture disabled: %s", error)
            return None
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError as error:
                    _LOGGER.warning(
                        "Original-call evidence temporary cleanup failed: %s", error
                    )
