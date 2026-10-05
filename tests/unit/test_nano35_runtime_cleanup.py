# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime cleanup must remove author metadata without breaking version state."""

import hashlib
import json
from pathlib import Path

import pytest
from nano35.sanitize_runtime import sanitize


def put(root: Path, name: str, data: bytes = b"private record") -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_cleanup_preserves_runtime_and_git_fingerprint(tmp_path):
    put(tmp_path, "opt/nano35-metadata/runtime-ready", b"ready")
    put(
        tmp_path,
        "etc/passwd",
        b"root:x:0:0:root:/root:/bin/bash\n"
        b"builder:x:1000:1000:Build User:/home/builder:/bin/bash\n"
        b"nobody:x:65534:65534:nobody:/nonexistent:/usr/sbin/nologin\n",
    )
    put(tmp_path, "etc/group", b"builder:x:1000:\nsudo:x:27:builder,service\n")
    put(tmp_path, "etc/shadow", b"root:!:1:0:99999:7:::\nbuilder:!:1:0:99999:7:::\n")
    put(tmp_path, "home/builder/.profile")
    put(tmp_path, "root/.cache/huggingface/modules/private-checkpoint/config.py")
    put(tmp_path, "root/.aws/credentials")
    put(tmp_path, "opt/nemo-rl/.git/logs/HEAD")
    put(tmp_path, "opt/nemo-rl/.git/modules/dependency/logs/HEAD")
    put(tmp_path, "opt/nano35-metadata/build-attempt-1.txt")
    record = put(
        tmp_path,
        "opt/nano35-metadata/request-stats-fix.json",
        json.dumps(
            {
                "gpu_refit_job": "12345",
                "base_image": "/private/project/base.sqsh",
                "runtime_patch_sha256": "known-patch-hash",
            }
        ).encode(),
    )
    config = put(
        tmp_path,
        "opt/nemo-rl/.git/config",
        b"[core]\n\tbare = false\n[user]\n\temail = builder@internal\n"
        b'[remote "origin"]\n\turl = https://github.com/NVIDIA-NeMo/RL.git\n',
    )
    preserved = {
        "opt/nemo_rl_container_fingerprint": b"fingerprint",
        "opt/nemo-rl/.git/HEAD": b"ref: refs/heads/main\n",
        "opt/nemo-rl/.git/refs/heads/main": b"commit\n",
        "opt/nemo-rl/.git/modules/dependency/HEAD": b"dependency-commit\n",
        "opt/ray_venvs/inference/lib/runtime.so": b"binary runtime",
        "opt/trtllm_wheels/runtime.whl": b"compiled wheel",
    }
    for name, data in preserved.items():
        put(tmp_path, name, data)
    result = sanitize(tmp_path)
    assert result["removed_entries"] > 0
    assert not (tmp_path / "home/builder").exists()
    assert not (tmp_path / "root/.cache/huggingface").exists()
    assert not (tmp_path / "opt/nemo-rl/.git/logs").exists()
    assert not (tmp_path / "opt/nemo-rl/.git/modules/dependency/logs").exists()
    assert b"builder" not in (tmp_path / "etc/passwd").read_bytes()
    assert b"nobody" in (tmp_path / "etc/passwd").read_bytes()
    assert (tmp_path / "etc/group").read_bytes() == b"sudo:x:27:service\n"
    assert json.loads(record.read_bytes()) == {
        "runtime_patch_sha256": "known-patch-hash"
    }
    assert b"builder" not in config.read_bytes()
    assert b"https://github.com/NVIDIA-NeMo/RL.git" in config.read_bytes()
    for name, data in preserved.items():
        assert (tmp_path / name).read_bytes() == data
    assert sanitize(tmp_path) == {"removed_entries": 0, "rewritten_files": 0}


def test_cleanup_does_not_follow_symlinks(tmp_path):
    root = tmp_path / "image"
    put(root, "opt/nano35-metadata/runtime-ready")
    put(root, "etc/passwd", b"root:x:0:0:root:/root:/bin/bash\n")
    outside = tmp_path / "outside"
    retained = put(outside, ".git/logs/HEAD")
    (root / "opt/external").symlink_to(outside, target_is_directory=True)
    sanitize(root)
    assert retained.read_bytes() == b"private record"


def test_cleanup_rejects_unrecognized_filesystem(tmp_path):
    with pytest.raises(ValueError, match="finalized Nano runtime"):
        sanitize(tmp_path)


def test_tool_caches_and_logging_identity_are_cleaned_with_matching_checksum(tmp_path):
    put(tmp_path, "opt/nano35-metadata/runtime-ready")
    put(tmp_path, "etc/passwd", b"root:x:0:0:root:/root:/bin/bash\n")
    put(tmp_path, "opt/nemo-rl/.ruff_cache/cache")
    marker = put(tmp_path, "usr/local/cuda-13/compat/.driver.internal-host.checked")
    library = put(tmp_path, "usr/local/cuda-13/compat/libcuda.so", b"driver library")
    name = "examples/configs/recipes/llm/grpo-nano3.5-swe-32n4g-tp4cp16-async-trtllm.v1.yaml"
    before = (
        b"grpo:\n  num_prompts_per_step: 32\nlogger:\n  wandb_enabled: false\n"
        b"  wandb:\n    project: private-builder-project\n"
    )
    recipe = put(tmp_path, "opt/nemo-rl/" + name, before)
    checksums = put(
        tmp_path,
        "opt/nano35-metadata/runtime-source-checksums.txt",
        f"{'0' * 64}  {name}\n".encode(),
    )
    sanitize(tmp_path)
    assert not (tmp_path / "opt/nemo-rl/.ruff_cache").exists()
    assert not marker.exists()
    assert library.read_bytes() == b"driver library"
    assert recipe.read_bytes() == before.replace(
        b"private-builder-project", b"nano3.5-e2e"
    )
    assert checksums.read_text() == (
        f"{hashlib.sha256(recipe.read_bytes()).hexdigest()}  {name}\n"
    )
    assert sanitize(tmp_path) == {"removed_entries": 0, "rewritten_files": 0}
