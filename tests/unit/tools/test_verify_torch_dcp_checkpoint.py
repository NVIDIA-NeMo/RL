from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp

from tools import verify_torch_dcp_checkpoint as verifier


def _save_checkpoint(checkpoint_dir: Path) -> Path:
    dcp.save(
        {
            "model.weight": torch.arange(32, dtype=torch.float32).reshape(4, 8),
            "optimizer.step": torch.tensor(7),
        },
        checkpoint_id=checkpoint_dir,
    )
    shards = sorted(checkpoint_dir.glob("*.distcp"))
    assert shards
    return shards[0]


def test_verifies_real_dcp_checkpoint_and_full_sha256_manifest(tmp_path) -> None:
    checkpoint_dir = tmp_path / "weights"
    shard = _save_checkpoint(checkpoint_dir)

    report = verifier.verify_checkpoint(checkpoint_dir, sha256=True)

    assert report["status"] == "PASS"
    assert report["metadata"]["state_dict_entries"] == 2
    assert report["metadata"]["storage_records"] == 2
    assert report["referenced_file_count"] == 1
    assert report["referenced_files"][0]["path"] == shard.name
    manifest = report["sha256_manifest"]
    assert [entry["path"] for entry in manifest["files"]] == [
        ".metadata",
        shard.name,
    ]
    assert all(len(entry["sha256"]) == 64 for entry in manifest["files"])
    assert len(manifest["manifest_sha256"]) == 64


def test_rejects_missing_referenced_shard(tmp_path) -> None:
    checkpoint_dir = tmp_path / "weights"
    shard = _save_checkpoint(checkpoint_dir)
    shard.unlink()

    with pytest.raises(
        verifier.CheckpointVerificationError,
        match="referenced storage file is missing",
    ):
        verifier.verify_checkpoint(checkpoint_dir)


def test_rejects_empty_referenced_shard(tmp_path) -> None:
    checkpoint_dir = tmp_path / "weights"
    shard = _save_checkpoint(checkpoint_dir)
    shard.write_bytes(b"")

    with pytest.raises(verifier.CheckpointVerificationError, match="is empty"):
        verifier.verify_checkpoint(checkpoint_dir)


def test_rejects_truncated_referenced_shard(tmp_path) -> None:
    checkpoint_dir = tmp_path / "weights"
    shard = _save_checkpoint(checkpoint_dir)
    shard.write_bytes(shard.read_bytes()[:1])

    with pytest.raises(
        verifier.CheckpointVerificationError,
        match="beyond file size",
    ):
        verifier.verify_checkpoint(checkpoint_dir)


def test_rejects_corrupt_metadata(tmp_path) -> None:
    checkpoint_dir = tmp_path / "weights"
    _save_checkpoint(checkpoint_dir)
    (checkpoint_dir / ".metadata").write_bytes(b"not a Torch DCP metadata file")

    with pytest.raises(
        verifier.CheckpointVerificationError,
        match="FileSystemReader rejected checkpoint metadata",
    ):
        verifier.verify_checkpoint(checkpoint_dir)


def test_rejects_truncated_metadata(tmp_path) -> None:
    checkpoint_dir = tmp_path / "weights"
    _save_checkpoint(checkpoint_dir)
    metadata_path = checkpoint_dir / ".metadata"
    metadata = metadata_path.read_bytes()
    metadata_path.write_bytes(metadata[: len(metadata) // 2])

    with pytest.raises(
        verifier.CheckpointVerificationError,
        match="FileSystemReader rejected checkpoint metadata",
    ):
        verifier.verify_checkpoint(checkpoint_dir)


def test_rejects_arbitrary_pickled_object_as_metadata(tmp_path) -> None:
    checkpoint_dir = tmp_path / "weights"
    checkpoint_dir.mkdir()
    with (checkpoint_dir / ".metadata").open("wb") as handle:
        pickle.dump({"storage_data": {}}, handle)

    with pytest.raises(
        verifier.CheckpointVerificationError,
        match="FileSystemReader rejected checkpoint metadata",
    ):
        verifier.verify_checkpoint(checkpoint_dir)


def test_cli_emits_json_failure_and_nonzero_exit(tmp_path, capsys) -> None:
    missing_checkpoint = tmp_path / "missing"

    exit_code = verifier.main(["--checkpoint-dir", str(missing_checkpoint)])

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert report["status"] == "FAIL"
    assert report["error"]["type"] == "CheckpointVerificationError"
    assert report["checkpoint_dir"] == str(missing_checkpoint)


def test_cli_emits_json_pass(tmp_path, capsys) -> None:
    checkpoint_dir = tmp_path / "weights"
    _save_checkpoint(checkpoint_dir)

    exit_code = verifier.main(["--checkpoint-dir", str(checkpoint_dir)])

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert report["status"] == "PASS"
    assert report["sha256_enabled"] is False


def test_cli_rejects_sha256_manifest_mismatch(tmp_path, capsys) -> None:
    checkpoint_dir = tmp_path / "weights"
    _save_checkpoint(checkpoint_dir)

    exit_code = verifier.main(
        [
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--sha256",
            "--expected-manifest-sha256",
            "0" * 64,
        ]
    )

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert report["status"] == "FAIL"
    assert "manifest SHA256" in report["error"]["message"]
