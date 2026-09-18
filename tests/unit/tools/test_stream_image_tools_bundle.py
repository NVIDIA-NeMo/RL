# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import io
import json
import tarfile

import pytest

from tools.stream_image_tools_bundle import (
    add_json,
    export_bundle,
    import_bundle,
    safe_asset,
)


def test_transfer_round_trip_and_no_overwrite(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    image = source / "original.png"
    image.write_bytes(b"image fixture")
    asset = {
        "source_path": str(image),
        "relative_path": "ab/abcdef.png",
        "bytes": image.stat().st_size,
    }
    (source / "assets.jsonl").write_text(json.dumps(asset) + "\n")
    row = {
        "expected_answer": "blue",
        "responses_create_params": {
            "input": [{"content": [{"type": "input_image", "image_url": str(image)}]}]
        },
    }
    for split in ["train", "validation"]:
        (source / (split + ".jsonl")).write_text(json.dumps(row) + "\n")
    (source / "report.json").write_text("{}")
    payload = io.BytesIO()
    export_bundle(source, payload)
    target = tmp_path / "target"
    import_bundle(target, io.BytesIO(payload.getvalue()))
    restored = json.loads((target / "train.jsonl").read_text())
    assert restored["expected_answer"] == "blue"
    path = restored["responses_create_params"]["input"][0]["content"][0]["image_url"]
    assert path == str(target / "images/ab/abcdef.png")
    assert (target / "images/ab/abcdef.png").read_bytes() == b"image fixture"
    assert json.loads((target / "READY.json").read_text())["all_asset_hashes_verified"]
    with pytest.raises(FileExistsError):
        import_bundle(target, io.BytesIO(payload.getvalue()))


@pytest.mark.parametrize("name", ["../bad", "/ab/file", "ab/../../file", "flat"])
def test_reject_unsafe_asset_names(name):
    with pytest.raises(ValueError):
        safe_asset(name)


def test_reject_tar_link_without_ready_marker(tmp_path):
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        entry = tarfile.TarInfo("images/ab/link")
        entry.type = tarfile.SYMTYPE
        entry.linkname = "/etc/passwd"
        archive.addfile(entry)
    target = tmp_path / "target"
    with pytest.raises(ValueError, match="Non-regular"):
        import_bundle(target, io.BytesIO(payload.getvalue()))
    assert not (target / "READY.json").exists()


def test_corrupt_asset_never_becomes_ready(tmp_path):
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        entry = tarfile.TarInfo("images/ab/fixture.png")
        entry.size = 3
        archive.addfile(entry, io.BytesIO(b"bad"))
        for split in ["train", "validation"]:
            add_json(archive, split + ".json", [])
        add_json(archive, "report.json", {})
        add_json(
            archive,
            "integrity.json",
            {"images/ab/fixture.png": {"bytes": 3, "sha256": "0" * 64}},
        )
    target = tmp_path / "target"
    with pytest.raises(ValueError, match="integrity verification"):
        import_bundle(target, io.BytesIO(payload.getvalue()))
    assert not (target / "READY.json").exists()
    assert not (target / "train.jsonl").exists()


def test_missing_integrity_metadata_never_becomes_ready(tmp_path):
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        for split in ["train", "validation"]:
            add_json(archive, split + ".json", [])
        add_json(archive, "report.json", {})
    target = tmp_path / "target"
    with pytest.raises(ValueError, match="Incomplete stream metadata"):
        import_bundle(target, io.BytesIO(payload.getvalue()))
    assert not (target / "READY.json").exists()
