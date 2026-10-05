#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download a pinned W&B SQSH artifact and verify its release SHA-256."""

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


def verify_image(path: Path, *, sha256: str, size_bytes: int) -> None:
    """Reject a truncated or modified image before it is made available."""
    if path.stat().st_size != size_bytes:
        raise ValueError(f"Image size mismatch: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != sha256:
        raise ValueError(f"Image SHA-256 mismatch: {path}")


def validate_manifest(
    manifest: dict[str, Any], *, expected_sha256: str
) -> dict[str, Any]:
    """Reject an obsolete or mismatched artifact before downloading its image."""
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported image manifest schema")
    image = manifest["image"]
    name = image["file_name"]
    if Path(name).name != name or not name.endswith(".sqsh"):
        raise ValueError("Invalid image filename in manifest")
    if not re.fullmatch(r"[a-f0-9]{64}", image["sha256"]):
        raise ValueError("Invalid image SHA-256 in manifest")
    if image["sha256"] != expected_sha256:
        raise ValueError("Artifact does not match the requested release SHA-256")
    if type(image["size_bytes"]) is not int or image["size_bytes"] <= 0:
        raise ValueError("Invalid image size in manifest")
    return image


def main() -> None:
    """Fetch one immutable artifact version without creating a W&B run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", help="ENTITY/nano3.5-e2e/ARTIFACT:vN")
    parser.add_argument("output", type=Path, help="New absolute .sqsh path")
    parser.add_argument("--sha256", required=True, help="Expected release SHA-256")
    args = parser.parse_args()
    if not re.fullmatch(r"[^/:]+/[^/:]+/[^/:]+:v[0-9]+", args.artifact):
        parser.error("Use the published ENTITY/PROJECT/ARTIFACT:vN, not a moving alias")
    if not re.fullmatch(r"[a-f0-9]{64}", args.sha256):
        parser.error("--sha256 must be the 64-character release checksum")
    output = args.output
    if not output.is_absolute() or output.suffix != ".sqsh":
        parser.error("The output must be an absolute .sqsh path")
    if os.path.lexists(output):
        parser.error("The output already exists; choose a new filename")
    output.parent.mkdir(parents=True, exist_ok=True)

    # Keep the optional download client separate from the training environment.
    import wandb

    artifact = wandb.Api().artifact(args.artifact, type="container-image")
    with tempfile.TemporaryDirectory(
        prefix=".nano35-download-", dir=output.parent
    ) as tmp:
        manifest_path = artifact.get_entry("image.json").download(root=tmp)
        manifest = json.loads(Path(manifest_path).read_text())
        image = validate_manifest(manifest, expected_sha256=args.sha256)
        name = image["file_name"]
        entry = artifact.get_entry(name)
        if entry.size != image["size_bytes"]:
            raise ValueError("W&B file size disagrees with the release manifest")
        downloaded = Path(entry.download(root=tmp, skip_cache=True))
        verify_image(downloaded, sha256=image["sha256"], size_bytes=image["size_bytes"])
        # Linking within the destination filesystem is atomic and never overwrites.
        os.link(downloaded, output)
    print(f"Image ready: {output}")
    print(f"SHA-256: {image['sha256']}")
    print(f"Source: {args.artifact}")
    print(f"Validation: {manifest['validation']['status']}")


if __name__ == "__main__":
    main()
