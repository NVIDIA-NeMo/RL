# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json

import pytest

from tools.check_image_tools_runtime import environments
from tools.image_tools_launch_checks import require_online_wandb, validate_qualification


@pytest.mark.parametrize(
    "change",
    [
        {},
        {"WANDB_MODE": "offline"},
        {"WANDB_DISABLED": "true"},
        {"WANDB_API_KEY": ""},
        {"WANDB_RUN_ID": ""},
    ],
)
def test_online_logging_is_required_without_disclosing_credentials(change):
    env = {
        "WANDB_API_KEY": "synthetic-secret",
        "WANDB_RUN_ID": "stable-id",
        "WANDB_MODE": "online",
        "WANDB_DISABLED": "false",
        **change,
    }
    if not change:
        require_online_wandb(env)
    else:
        with pytest.raises(ValueError) as error:
            require_online_wandb(env)
        assert "synthetic-secret" not in str(error.value)


def test_qualification_matches_build_source_and_all_checks(tmp_path):
    lock = tmp_path / "uv.lock"
    lock.write_text("lock fixture")
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    receipt = {
        "source_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "requirements_sha256": "fixture",
    }
    (overlay / "READY.json").write_text(json.dumps(receipt))
    records = [receipt, *({"check": label, "ok": True} for label, *_ in environments())]
    log = tmp_path / "run.log"
    log.write_text("\n".join(map(json.dumps, records)))
    validate_qualification(log, overlay, lock)
    records[-1]["ok"] = False
    log.write_text("\n".join(map(json.dumps, records)))
    with pytest.raises(ValueError, match="every required check"):
        validate_qualification(log, overlay, lock)


def test_qualification_rejects_other_source_or_checkpoint(tmp_path, monkeypatch):
    lock = tmp_path / "uv.lock"
    lock.write_text("lock fixture")
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    receipt = {"source_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest()}
    (overlay / "READY.json").write_text(json.dumps(receipt))
    identity = {
        "qualification_source_sha256": "source-a",
        "model_checkpoint": "/step_120/hf",
    }
    records = [
        receipt,
        identity,
        *({"check": label, "ok": True} for label, *_ in environments()),
    ]
    log = tmp_path / "run.log"
    log.write_text("\n".join(map(json.dumps, records)))
    monkeypatch.setattr(
        "tools.image_tools_launch_checks.source_fingerprint", lambda root: "source-a"
    )
    validate_qualification(
        log, overlay, lock, project_root=tmp_path, model_checkpoint="/step_120/hf"
    )
    with pytest.raises(ValueError, match="source and checkpoint"):
        validate_qualification(
            log, overlay, lock, project_root=tmp_path, model_checkpoint="/other/hf"
        )
    monkeypatch.setattr(
        "tools.image_tools_launch_checks.source_fingerprint", lambda root: "source-b"
    )
    with pytest.raises(ValueError, match="source and checkpoint"):
        validate_qualification(
            log, overlay, lock, project_root=tmp_path, model_checkpoint="/step_120/hf"
        )
    records[-1]["ok"] = True
    log.write_text("\n".join(map(json.dumps, records[1:])))
    with pytest.raises(ValueError, match="exact overlay"):
        validate_qualification(log, overlay, lock)
    log.write_text("\n".join(map(json.dumps, records)))
    lock.write_text("changed lock")
    with pytest.raises(ValueError, match="source lock"):
        validate_qualification(log, overlay, lock)
