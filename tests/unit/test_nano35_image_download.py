# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reject obsolete images and damaged downloads without contacting W&B."""

import hashlib
from pathlib import Path

import pytest
from nano35.fetch_wandb_image import validate_manifest, verify_image


def test_obsolete_artifact_is_rejected_even_with_valid_metadata() -> None:
    manifest = {
        "schema_version": 1,
        "image": {"file_name": "old.sqsh", "size_bytes": 1024, "sha256": "a" * 64},
    }
    with pytest.raises(ValueError, match="requested release SHA-256"):
        validate_manifest(manifest, expected_sha256="b" * 64)


@pytest.mark.parametrize("name", ["../image.sqsh", "/image.sqsh", "image.tar"])
def test_manifest_cannot_choose_an_external_path(name: str) -> None:
    manifest = {
        "schema_version": 1,
        "image": {"file_name": name, "size_bytes": 1024, "sha256": "a" * 64},
    }
    with pytest.raises(ValueError, match="filename"):
        validate_manifest(manifest, expected_sha256="a" * 64)


def test_download_bytes_must_match_release_size_and_checksum(tmp_path: Path) -> None:
    path = tmp_path / "image.sqsh"
    payload = b"verified image bytes"
    digest = hashlib.sha256(payload).hexdigest()
    manifest = {
        "schema_version": 1,
        "image": {"file_name": path.name, "size_bytes": len(payload), "sha256": digest},
    }
    image = validate_manifest(manifest, expected_sha256=digest)
    path.write_bytes(payload)
    verify_image(path, sha256=image["sha256"], size_bytes=image["size_bytes"])
    path.write_bytes(payload[:-1])
    with pytest.raises(ValueError, match="size mismatch"):
        verify_image(path, sha256=digest, size_bytes=len(payload))
    path.write_bytes(b"X" * len(payload))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_image(path, sha256=digest, size_bytes=len(payload))
