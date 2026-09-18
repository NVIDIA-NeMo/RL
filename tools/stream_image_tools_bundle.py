#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stream a prepared image-tool bundle between clusters with verified asset hashes.

Export writes a tar stream to stdout. Import consumes stdin into a NEW directory;
it never extracts arbitrary tar paths, links or permissions. Source data remains
unchanged. No model images are printed to diagnostic logs or staged on the Mac.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile


def hash_stream(stream):
    """Hash sequentially with bounded memory, including older login-node Python."""
    result = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        result.update(block)
    return result.hexdigest()


def safe_asset(path):
    """Allow only fan-out asset filenames, never traversal or absolute paths."""
    parts = PurePosixPath(path).parts
    if (
        len(parts) != 2
        or any(p in (".", "..") for p in parts)
        or PurePosixPath(path).is_absolute()
    ):
        raise ValueError(f"Invalid asset path: {path}")
    return path


def add_json(archive, name, value):
    """Add a small JSON metadata record to the stream."""
    payload = json.dumps(value).encode()
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def export_bundle(source, output):
    """Stream only manifest-listed images, then integrity metadata and split rows."""
    assets = [
        json.loads(line) for line in (source / "assets.jsonl").read_text().splitlines()
    ]
    mapping = {
        a["source_path"]: "images/" + safe_asset(a["relative_path"]) for a in assets
    }
    seen = set()
    integrity = {}
    with tarfile.open(fileobj=output, mode="w|") as archive:
        for asset in assets:
            name = mapping[asset["source_path"]]
            if name in seen:
                continue
            seen.add(name)
            path = Path(asset["source_path"])
            # Hash first, then receiver hashes the transferred bytes. A changing
            # source file cannot be silently accepted as the prepared asset.
            with path.open("rb") as stream:
                sha = hash_stream(stream)
            if path.stat().st_size != asset["bytes"]:
                raise ValueError(f"Source image size changed: {name}")
            integrity[name] = {"sha256": sha, "bytes": asset["bytes"]}
            info = tarfile.TarInfo(name)
            info.size = asset["bytes"]
            with path.open("rb") as stream:
                archive.addfile(info, stream)
        for split in ["train", "validation"]:
            rows = []
            for line in (source / (split + ".jsonl")).read_text().splitlines():
                row = json.loads(line)
                for msg in row["responses_create_params"]["input"]:
                    if not isinstance(msg.get("content"), list):
                        continue
                    for part in msg["content"]:
                        if part.get("type") == "input_image":
                            part["image_url"] = mapping[
                                str(Path(part["image_url"]).resolve(strict=True))
                            ]
                rows.append(row)
            add_json(archive, split + ".json", rows)
        add_json(
            archive, "report.json", json.loads((source / "report.json").read_text())
        )
        add_json(archive, "integrity.json", integrity)


def import_bundle(destination, source):
    """Validate every transferred member; write HSG absolute paths only after verification."""
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    received = {}
    metadata = {}
    with tarfile.open(fileobj=source, mode="r|") as archive:
        for member in archive:
            if (
                not member.isfile()
                or member.name in received
                or member.name in metadata
            ):
                raise ValueError("Non-regular or duplicate archive member")
            stream = archive.extractfile(member)
            if member.name in {
                "train.json",
                "validation.json",
                "report.json",
                "integrity.json",
            }:
                if member.size > 100_000_000:
                    raise ValueError("Unexpectedly large metadata record")
                metadata[member.name] = json.load(stream)
            elif member.name.startswith("images/"):
                safe_asset(member.name[len("images/") :])
                target = destination / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as output:
                    shutil.copyfileobj(stream, output, 1024 * 1024)
                with target.open("rb") as data:
                    sha = hash_stream(data)
                received[member.name] = {"bytes": target.stat().st_size, "sha256": sha}
            else:
                raise ValueError(f"Unexpected archive member: {member.name}")
    if set(metadata) != {
        "train.json",
        "validation.json",
        "report.json",
        "integrity.json",
    }:
        raise ValueError("Incomplete stream metadata")
    if received != metadata["integrity.json"]:
        raise ValueError("Transferred assets failed integrity verification")
    for split in ["train", "validation"]:
        rows = metadata[split + ".json"]
        for row in rows:
            for msg in row["responses_create_params"]["input"]:
                if not isinstance(msg.get("content"), list):
                    continue
                for part in msg["content"]:
                    if part.get("type") == "input_image":
                        if part["image_url"] not in received:
                            raise ValueError("Split row refers to an unverified image")
                        part["image_url"] = str(destination / part["image_url"])
        with (destination / (split + ".jsonl")).open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
    for name in ["report.json", "integrity.json"]:
        (destination / name).write_text(json.dumps(metadata[name], indent=2) + "\n")
    result = {
        "assets": len(received),
        "bytes": sum(a["bytes"] for a in received.values()),
        "train_rows": len(metadata["train.json"]),
        "validation_rows": len(metadata["validation.json"]),
        "asset_root": str(destination / "images"),
        "all_asset_hashes_verified": True,
    }
    (destination / "READY.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), file=sys.stderr, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["export", "import"])
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.mode == "export":
        export_bundle(args.directory, sys.stdout.buffer)
    else:
        import_bundle(args.directory, sys.stdin.buffer)
