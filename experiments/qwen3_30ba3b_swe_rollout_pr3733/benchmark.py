#!/usr/bin/env python3
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

"""Plan and record the PR #3733 Qwen3-30B-A3B SWE rollout benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MANIFEST_PATH = Path(__file__).with_name("benchmark_matrix.json")
SHA40_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_ARMS = (
    ("baseline", None, 0, None),
    ("dflash_k5", "dflash", 5, "dflash"),
    ("dflash_k7", "dflash", 7, "dflash"),
    ("dspark_k5", "dspark", 5, "dspark"),
    ("dspark_k7", "dspark", 7, "dspark"),
)


class ContractError(ValueError):
    """Raised when a benchmark identity or lifecycle contract is violated."""


@dataclass(frozen=True)
class Arm:
    """One method/K point in the matched benchmark matrix."""

    name: str
    method: str | None
    num_speculative_tokens: int
    draft: str | None
    config: str
    config_sha256: str


def load_manifest() -> dict[str, Any]:
    """Load the checked-in manifest and validate its five-arm contract."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ContractError("unsupported benchmark manifest schema")

    observed_arms = tuple(
        (
            item.get("name"),
            item.get("method"),
            item.get("num_speculative_tokens"),
            item.get("draft"),
        )
        for item in manifest.get("arms", [])
    )
    if observed_arms != EXPECTED_ARMS:
        raise ContractError(f"benchmark arm drift: {observed_arms!r}")
    if manifest.get("canary_arms") != ["baseline", "dflash_k5"]:
        raise ContractError("canary must contain baseline and DFlash K5")
    if manifest.get("request_buckets") != [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        raise ContractError("request bucket drift")
    source_files = manifest.get("source_files_sha256")
    if not isinstance(source_files, dict) or not source_files:
        raise ContractError("semantics-critical source identities are missing")
    if any(
        not isinstance(path, str)
        or not isinstance(digest, str)
        or not SHA256_PATTERN.fullmatch(digest)
        for path, digest in source_files.items()
    ):
        raise ContractError("semantics-critical source identities are invalid")
    repo_root = MANIFEST_PATH.parents[2]
    for item in manifest["arms"]:
        config_path = Path(item.get("config", ""))
        config_sha256 = item.get("config_sha256")
        if (
            config_path.is_absolute()
            or not SHA256_PATTERN.fullmatch(str(config_sha256))
            or not (repo_root / config_path).is_file()
        ):
            raise ContractError(f"arm config identity is invalid: {item['name']}")
        _require_equal(
            f"arm config SHA256 {item['name']}",
            _sha256(repo_root / config_path),
            config_sha256,
        )
    return manifest


def _canonical_sha256(value: object) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).hexdigest()


def make_preflight_record(
    *, manifest: dict[str, Any], source: dict[str, Any], artifacts: dict[str, Any]
) -> dict[str, Any]:
    """Bind verified source and artifact evidence to the exact manifest."""
    if source.get("status") != "verified" or artifacts.get("status") != "verified":
        raise ContractError("preflight evidence must be verified")
    body = {
        "status": "verified",
        "manifest_sha256": _canonical_sha256(manifest),
        "source": source,
        "artifacts": artifacts,
    }
    return {**body, "preflight_id": _canonical_sha256(body)}


def _validate_preflight_record(
    *,
    preflight: dict[str, Any],
    manifest: dict[str, Any],
    source_commit: str,
    container: Path,
    container_sha256: str,
) -> None:
    body = {key: value for key, value in preflight.items() if key != "preflight_id"}
    _require_equal("preflight status", preflight.get("status"), "verified")
    _require_equal(
        "preflight source status", preflight.get("source", {}).get("status"), "verified"
    )
    _require_equal(
        "preflight artifact status",
        preflight.get("artifacts", {}).get("status"),
        "verified",
    )
    _require_equal(
        "preflight ID", preflight.get("preflight_id"), _canonical_sha256(body)
    )
    _require_equal(
        "preflight manifest SHA256",
        preflight.get("manifest_sha256"),
        _canonical_sha256(manifest),
    )
    _require_equal(
        "preflight source commit",
        preflight.get("source", {}).get("source_commit"),
        source_commit,
    )
    expected_container = {"path": str(container), "sha256": container_sha256}
    _require_equal(
        "preflight container",
        preflight.get("artifacts", {}).get("container"),
        expected_container,
    )
    expected_target = {
        key: manifest["target"][key]
        for key in (
            "config_sha256",
            "model_index_sha256",
            "weight_files",
            "weight_bytes",
            "weight_sha256",
        )
    }
    expected_drafts = {
        name: {
            key: draft[key]
            for key in (
                "config_sha256",
                "weight_files",
                "weight_bytes",
                "weight_sha256",
            )
        }
        for name, draft in manifest["drafts"].items()
    }
    artifacts = preflight.get("artifacts", {})
    _require_equal("target artifact evidence", artifacts.get("target"), expected_target)
    _require_equal("draft artifact evidence", artifacts.get("drafts"), expected_drafts)
    _require_equal("data artifact evidence", artifacts.get("data"), manifest["data"])
    expected_metadata_paths = {str(container), manifest["data"]["path"]}
    target_path = Path(manifest["target"]["path"])
    expected_metadata_paths.update(
        {
            str(target_path / "config.json"),
            str(target_path / "model.safetensors.index.json"),
            *(str(target_path / name) for name in manifest["target"]["weight_sha256"]),
        }
    )
    for draft in manifest["drafts"].values():
        draft_path = Path(draft["path"])
        expected_metadata_paths.add(str(draft_path / "config.json"))
        expected_metadata_paths.update(
            str(draft_path / name) for name in draft["weight_sha256"]
        )
    metadata = artifacts.get("file_metadata")
    if not isinstance(metadata, dict) or set(metadata) != expected_metadata_paths:
        raise ContractError("artifact evidence has an incomplete tracked-path set")
    if any(
        not isinstance(value, dict)
        or set(value) != {"size", "mtime_ns", "inode"}
        or any(not isinstance(item, int) for item in value.values())
        for value in metadata.values()
    ):
        raise ContractError("artifact evidence has invalid file metadata")
    expected_protected = {
        manifest["pr_head"]: [
            manifest["recipe"],
            "examples/nemo_gym/grpo_qwen3_30ba3b_thinking_swe1.yaml",
            "examples/nemo_gym/run_qwen3_swe_rollout_only.sh",
            manifest["entrypoint"],
        ]
    }
    source = preflight.get("source", {})
    _require_equal(
        "source ancestor evidence",
        source.get("required_ancestors"),
        [manifest["pr_head"]],
    )
    _require_equal(
        "source protected-path evidence",
        source.get("protected_paths_by_head"),
        expected_protected,
    )
    _require_equal(
        "source file evidence",
        source.get("source_files_sha256"),
        manifest["source_files_sha256"],
    )


def _validate_canary_record(
    *, canary: dict[str, Any], manifest: dict[str, Any], output_root: Path
) -> dict[str, Any]:
    expected_path = output_root / "inputs/swebench_verified_first1.jsonl"
    expected = {
        "path": str(expected_path),
        "lines": 1,
        "parent_path": manifest["data"]["path"],
        "parent_sha256": manifest["data"]["sha256"],
        "parent_lines": manifest["data"]["lines"],
        "selection": "first JSONL record",
    }
    for key, value in expected.items():
        _require_equal(f"canary {key}", canary.get(key), value)
    if not expected_path.is_file():
        raise ContractError(f"canonical canary is missing: {expected_path}")
    _require_equal("canary SHA256", _sha256(expected_path), canary.get("sha256"))
    with expected_path.open("rb") as handle:
        lines = handle.readlines()
    if len(lines) != 1:
        raise ContractError("canonical canary must contain exactly one JSONL record")
    try:
        json.loads(lines[0])
    except json.JSONDecodeError as error:
        raise ContractError("canonical canary record is invalid JSON") from error
    return canary


def validate_plan_runtime_files(*, plan: dict[str, Any]) -> None:
    """Fail closed if a planned canary changed after plan construction."""
    if plan.get("profile") != "canary":
        return
    data = plan.get("data", {})
    canary_path = Path(data.get("path", ""))
    parent_path = Path(data.get("parent_path", ""))
    if not canary_path.is_file():
        raise ContractError("canonical canary is missing")
    _require_equal("canary SHA256", _sha256(canary_path), data.get("sha256"))
    if not parent_path.is_file():
        raise ContractError("canonical canary parent is missing")
    with canary_path.open("rb") as canary_handle:
        canary_line = canary_handle.readline()
        if canary_handle.readline():
            raise ContractError("canonical canary must contain exactly one record")
    with parent_path.open("rb") as parent_handle:
        parent_first_line = parent_handle.readline()
    _require_equal("canary first-record bytes", canary_line, parent_first_line)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_equal(label: str, observed: object, expected: object) -> None:
    if observed != expected:
        raise ContractError(
            f"{label} drift: observed {observed!r}, expected {expected!r}"
        )


def _weight_identity(path: Path) -> tuple[int, int]:
    weights = sorted(path.glob("*.safetensors"))
    return len(weights), sum(item.stat().st_size for item in weights)


def _file_metadata(path: Path) -> dict[str, int]:
    stat_result = path.stat()
    return {
        "size": stat_result.st_size,
        "mtime_ns": stat_result.st_mtime_ns,
        "inode": stat_result.st_ino,
    }


def validate_output_root(
    *,
    output_root: Path,
    repo_root: Path | None = None,
    approved_prefix: Path | None = None,
) -> Path:
    """Reject broad, overlapping, or off-allocation experiment outputs."""
    resolved = output_root.resolve(strict=False)
    if not output_root.is_absolute() or resolved == Path("/"):
        raise ContractError(f"unsafe output root: {output_root}")
    if repo_root is not None:
        resolved_repo = repo_root.resolve(strict=False)
        if (
            resolved == resolved_repo
            or resolved in resolved_repo.parents
            or resolved_repo in resolved.parents
        ):
            raise ContractError(f"unsafe output root overlaps source: {output_root}")
    if approved_prefix is not None:
        resolved_prefix = approved_prefix.resolve(strict=False)
        if resolved == resolved_prefix or resolved_prefix not in resolved.parents:
            raise ContractError(
                f"unsafe output root is outside approved prefix: {output_root}"
            )
    return resolved


def validate_artifact_metadata(*, artifacts: dict[str, Any]) -> None:
    """Catch any artifact filesystem drift after the cryptographic preflight."""
    metadata = artifacts.get("file_metadata")
    if not isinstance(metadata, dict) or not metadata:
        raise ContractError("preflight artifact metadata is incomplete")
    for serialized_path, expected in metadata.items():
        path = Path(serialized_path)
        if not path.is_file():
            raise ContractError(f"artifact metadata drift: missing {path}")
        _require_equal(
            f"artifact metadata drift for {path}", _file_metadata(path), expected
        )


def _verify_weights(label: str, path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    weights = sorted(path.glob("*.safetensors"))
    observed_names = [item.name for item in weights]
    expected_hashes = expected.get("weight_sha256")
    if not isinstance(expected_hashes, dict) or any(
        not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value)
        for value in expected_hashes.values()
    ):
        raise ContractError(f"{label} per-file weight SHA256 identity is incomplete")
    _require_equal(
        f"{label} weight names", observed_names, sorted(expected_hashes.keys())
    )
    count, total_bytes = _weight_identity(path)
    _require_equal(f"{label} weight files", count, expected["weight_files"])
    _require_equal(f"{label} weight bytes", total_bytes, expected["weight_bytes"])
    observed_hashes: dict[str, str] = {}
    file_metadata: dict[str, dict[str, int]] = {}
    for weight in weights:
        digest = _sha256(weight)
        _require_equal(
            f"{label} weight SHA256 {weight.name}",
            digest,
            expected_hashes[weight.name],
        )
        observed_hashes[weight.name] = digest
        stat_result = weight.stat()
        file_metadata[weight.name] = {
            "size": stat_result.st_size,
            "mtime_ns": stat_result.st_mtime_ns,
            "inode": stat_result.st_ino,
        }
    return {
        "weight_files": count,
        "weight_bytes": total_bytes,
        "weight_sha256": observed_hashes,
        "file_metadata": file_metadata,
    }


def verify_artifacts(
    *, manifest: dict[str, Any], container: Path, container_sha256: str
) -> dict[str, Any]:
    """Verify byte identities for the container, checkpoints, and SWE JSONL."""
    if not container.is_absolute() or not container.is_file():
        raise ContractError(f"container is not an absolute regular file: {container}")
    tracked_paths = [container]
    _require_equal("container SHA256", _sha256(container), container_sha256)

    target = manifest["target"]
    target_path = Path(target["path"])
    if not target_path.is_absolute() or not target_path.is_dir():
        raise ContractError(
            f"target checkpoint is not an absolute directory: {target_path}"
        )
    _require_equal(
        "target config SHA256",
        _sha256(target_path / "config.json"),
        target["config_sha256"],
    )
    tracked_paths.append(target_path / "config.json")
    _require_equal(
        "target model index SHA256",
        _sha256(target_path / "model.safetensors.index.json"),
        target["model_index_sha256"],
    )
    tracked_paths.append(target_path / "model.safetensors.index.json")
    target_result = _verify_weights("target", target_path, target)
    tracked_paths.extend(sorted(target_path.glob("*.safetensors")))

    draft_results: dict[str, Any] = {}
    for key, label in (("dflash", "DFlash"), ("dspark", "DSpark")):
        draft = manifest["drafts"][key]
        draft_path = Path(draft["path"])
        if not draft_path.is_absolute() or not draft_path.is_dir():
            raise ContractError(
                f"{label} checkpoint is not an absolute directory: {draft_path}"
            )
        _require_equal(
            f"{label} config SHA256",
            _sha256(draft_path / "config.json"),
            draft["config_sha256"],
        )
        tracked_paths.append(draft_path / "config.json")
        draft_results[key] = _verify_weights(label, draft_path, draft)
        tracked_paths.extend(sorted(draft_path.glob("*.safetensors")))

    data = manifest["data"]
    data_path = Path(data["path"])
    if not data_path.is_absolute() or not data_path.is_file():
        raise ContractError(f"SWE data is not an absolute regular file: {data_path}")
    _require_equal("SWE data SHA256", _sha256(data_path), data["sha256"])
    with data_path.open("rb") as handle:
        line_count = sum(1 for _ in handle)
    _require_equal("SWE data lines", line_count, data["lines"])
    tracked_paths.append(data_path)
    return {
        "status": "verified",
        "container": {"path": str(container), "sha256": container_sha256},
        "target": {
            "config_sha256": target["config_sha256"],
            "model_index_sha256": target["model_index_sha256"],
            **{
                key: target_result[key]
                for key in ("weight_files", "weight_bytes", "weight_sha256")
            },
        },
        "drafts": {
            name: {
                "config_sha256": manifest["drafts"][name]["config_sha256"],
                **{
                    key: result[key]
                    for key in ("weight_files", "weight_bytes", "weight_sha256")
                },
            }
            for name, result in draft_results.items()
        },
        "data": {"path": str(data_path), "sha256": data["sha256"], "lines": line_count},
        "file_metadata": {str(path): _file_metadata(path) for path in tracked_paths},
    }


def _git(
    repo_root: Path, *args: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
        check=check,
    )


def verify_source(
    *,
    repo_root: Path,
    source_commit: str,
    required_ancestors: list[str],
    protected_paths_by_head: dict[str, list[Path]],
    source_files_sha256: dict[Path, str] | None = None,
) -> dict[str, Any]:
    """Require a clean descendant of both PR heads with owned files unchanged."""
    if not repo_root.is_absolute() or not (repo_root / ".git").exists():
        raise ContractError(f"source is not an absolute Git checkout: {repo_root}")
    if not SHA40_PATTERN.fullmatch(source_commit) or not required_ancestors:
        raise ContractError("source and PR commits must be full lowercase Git SHAs")
    if any(not SHA40_PATTERN.fullmatch(head) for head in required_ancestors):
        raise ContractError("source and PR commits must be full lowercase Git SHAs")
    if set(required_ancestors) != set(protected_paths_by_head):
        raise ContractError("every required ancestor must own protected paths")
    observed_head = _git(repo_root, "rev-parse", "HEAD").stdout.strip()
    _require_equal("source HEAD", observed_head, source_commit)
    status = _git(
        repo_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignore-submodules=none",
    ).stdout
    if status:
        raise ContractError(f"source checkout is not clean:\n{status.rstrip()}")
    for head in required_ancestors:
        ancestor = _git(
            repo_root, "merge-base", "--is-ancestor", head, source_commit, check=False
        )
        if ancestor.returncode != 0:
            raise ContractError(f"PR head {head} is not an ancestor of {source_commit}")
        for path in protected_paths_by_head[head]:
            diff = _git(
                repo_root,
                "diff",
                "--quiet",
                head,
                source_commit,
                "--",
                str(path),
                check=False,
            )
            if diff.returncode != 0:
                raise ContractError(f"protected PR workload changed: {path}")
    submodules = _git(
        repo_root, "submodule", "status", "--recursive"
    ).stdout.splitlines()
    dirty_submodules = [line for line in submodules if line[:1] in {"-", "+", "U"}]
    if dirty_submodules:
        raise ContractError(
            "recursive submodule identity drift: " + "; ".join(dirty_submodules)
        )
    source_files_sha256 = source_files_sha256 or {}
    for path, expected_sha256 in source_files_sha256.items():
        if path.is_absolute() or not SHA256_PATTERN.fullmatch(expected_sha256):
            raise ContractError(
                "source file identities must use relative paths and SHA256"
            )
        source_path = repo_root / path
        if not source_path.is_file():
            raise ContractError(f"semantics-critical source file is missing: {path}")
        _require_equal(
            f"source file SHA256 {path}", _sha256(source_path), expected_sha256
        )
    return {
        "status": "verified",
        "source_commit": source_commit,
        "required_ancestors": required_ancestors,
        "protected_paths_by_head": {
            head: [str(path) for path in paths]
            for head, paths in protected_paths_by_head.items()
        },
        "source_files_sha256": {
            str(path): digest for path, digest in source_files_sha256.items()
        },
    }


def materialize_canary(
    *, source: Path, source_sha256: str, source_lines: int, destination: Path
) -> dict[str, Any]:
    """Create the deterministic first-record canary without altering the source."""
    if _sha256(source) != source_sha256:
        raise ContractError("canary parent SHA256 drift")
    with source.open("rb") as handle:
        observed_lines = sum(1 for _ in handle)
    _require_equal("canary parent lines", observed_lines, source_lines)
    with source.open("rb") as handle:
        first_line = handle.readline()
    if not first_line.endswith(b"\n"):
        raise ContractError("canary parent first record must end with a newline")
    try:
        json.loads(first_line)
    except json.JSONDecodeError as error:
        raise ContractError("canary parent first record is invalid JSON") from error
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as error:
        raise ContractError(
            f"canary destination already exists: {destination}"
        ) from error
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(first_line)
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "path": str(destination),
        "sha256": _sha256(destination),
        "lines": 1,
        "selection": "first JSONL record",
        "parent_path": str(source),
        "parent_sha256": source_sha256,
        "parent_lines": source_lines,
    }


def _capture_sizes(
    arm: Arm, request_buckets: list[int], drafts: dict[str, Any]
) -> list[int]:
    if arm.method is None:
        return request_buckets
    if arm.draft is None:
        raise ContractError(f"speculative arm {arm.name} is missing a draft")

    target_width = arm.num_speculative_tokens + 1
    widths = {target_width}
    draft_config = drafts[arm.draft]
    if draft_config["draft_query_width"] == "k":
        widths.add(arm.num_speculative_tokens)
    elif draft_config["draft_query_width"] != "k_plus_one":
        raise ContractError(f"unsupported draft query width for {arm.name}")

    return sorted({bucket * width for bucket in request_buckets for width in widths})


def _speculative_override(arm: Arm, drafts: dict[str, Any]) -> str:
    key = "policy.generation.vllm_kwargs.speculative_config"
    if arm.method is None:
        return f"{key}=null"
    if arm.draft is None:
        raise ContractError(f"speculative arm {arm.name} is missing a draft")
    draft_path = drafts[arm.draft]["path"]
    return (
        f"{key}={{method:{arm.method},model:{draft_path},"
        f"num_speculative_tokens:{arm.num_speculative_tokens},"
        "draft_tensor_parallel_size:1}"
    )


def render_plan(
    *,
    profile: str,
    source_commit: str,
    container: Path,
    container_sha256: str,
    output_root: Path,
    preflight: dict[str, Any],
    canary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Render a deterministic plan without touching Slurm or remote state."""
    if profile not in {"canary", "full"}:
        raise ContractError(f"unsupported profile: {profile}")
    if not SHA40_PATTERN.fullmatch(source_commit):
        raise ContractError("source commit must be a full lowercase Git SHA")
    if not SHA256_PATTERN.fullmatch(container_sha256):
        raise ContractError("container SHA256 must be 64 lowercase hex characters")
    if not container.is_absolute():
        raise ContractError("container must be an absolute path")
    output_root = validate_output_root(output_root=output_root)

    manifest = load_manifest()
    _validate_preflight_record(
        preflight=preflight,
        manifest=manifest,
        source_commit=source_commit,
        container=container,
        container_sha256=container_sha256,
    )
    selected_names = (
        set(manifest["canary_arms"])
        if profile == "canary"
        else {item["name"] for item in manifest["arms"]}
    )
    data = dict(manifest["data"])
    bounded_override: str | None = None
    if profile == "canary":
        if canary is None:
            raise ContractError("canary plan requires a canonical canary record")
        data = _validate_canary_record(
            canary=canary, manifest=manifest, output_root=output_root
        )
        bounded_override = "one deterministic prompt"
    elif canary is not None:
        raise ContractError("full plan must not include a canary record")

    campaign_id = _canonical_sha256(
        {
            "preflight_id": preflight["preflight_id"],
            "target": manifest["target"],
            "drafts": manifest["drafts"],
            "data": manifest["data"],
            "common": manifest["common"],
            "arms": manifest["arms"],
        }
    )

    runs: list[dict[str, Any]] = []
    for item in manifest["arms"]:
        arm = Arm(**item)
        if arm.name not in selected_names:
            continue
        capture_sizes = _capture_sizes(
            arm, manifest["request_buckets"], manifest["drafts"]
        )
        run_output = output_root / profile / arm.name
        run_name = f"q30-swe-{profile}-{arm.name}-{source_commit[:12]}"
        command = [
            "uv",
            "run",
            "--frozen",
            "--no-sync",
            manifest["entrypoint"],
            "--config",
            arm.config,
            f"policy.model_name={manifest['target']['path']}",
            f"policy.tokenizer.name={manifest['target']['path']}",
            f"data.train.data_path={data['path']}",
            f"data.validation.data_path={data['path']}",
            f"logger.log_dir={run_output / 'logs'}",
            "logger.wandb_enabled=true",
            f"logger.wandb.project={manifest['common']['wandb_project']}",
            f"logger.wandb.name={run_name}",
        ]
        runs.append(
            {
                "arm": arm.name,
                "method": arm.method,
                "num_speculative_tokens": arm.num_speculative_tokens,
                "k_semantics": "draft tokens proposed per decoding step",
                "cudagraph_capture_sizes": capture_sizes,
                "common": manifest["common"],
                "data_path": data["path"],
                "bounded_override": bounded_override,
                "config": arm.config,
                "config_sha256": arm.config_sha256,
                "environment": {
                    "WANDB_ENTITY": manifest["common"]["wandb_entity"],
                    "WANDB_PROJECT": manifest["common"]["wandb_project"],
                    "WANDB_RUN_NAME": run_name,
                },
                "output_dir": str(run_output),
                "command": command,
            }
        )

    body = {
        "schema_version": 1,
        "profile": profile,
        "workload_label": manifest["workload_label"],
        "campaign_id": campaign_id,
        "preflight_id": preflight["preflight_id"],
        "output_root": str(output_root),
        "pr_head": manifest["pr_head"],
        "source_commit": source_commit,
        "container": {
            "path": str(container),
            "sha256": container_sha256,
        },
        "target": manifest["target"],
        "drafts": manifest["drafts"],
        "data": data,
        "common": manifest["common"],
        "runs": runs,
    }
    return {**body, "plan_id": _canonical_sha256(body)}


def validate_canonical_plan(*, plan: dict[str, Any], preflight: dict[str, Any]) -> None:
    """Require the scheduler input to equal a fresh manifest-derived plan."""
    body = {key: value for key, value in plan.items() if key != "plan_id"}
    _require_equal("plan ID", plan.get("plan_id"), _canonical_sha256(body))
    try:
        profile = plan["profile"]
        canary = plan["data"] if profile == "canary" else None
        expected = render_plan(
            profile=profile,
            source_commit=plan["source_commit"],
            container=Path(plan["container"]["path"]),
            container_sha256=plan["container"]["sha256"],
            output_root=Path(plan["output_root"]),
            preflight=preflight,
            canary=canary,
        )
    except (KeyError, TypeError) as error:
        raise ContractError("canonical plan fields are incomplete") from error
    _require_equal("canonical plan", plan, expected)


def claim_submission(
    *, state_dir: Path, profile: str, arm_name: str, job_id: str
) -> dict[str, str]:
    """Create one exclusive durable submission record for a profile/arm."""
    manifest = load_manifest()
    allowed_arms = {item["name"] for item in manifest["arms"]}
    if profile not in {"canary", "full"} or arm_name not in allowed_arms:
        raise ContractError("invalid profile or arm")
    if profile == "canary" and arm_name not in manifest["canary_arms"]:
        raise ContractError(f"arm {arm_name} is not in the canary matrix")
    if not job_id.isdigit():
        raise ContractError("job ID must contain only decimal digits")

    state_dir.mkdir(parents=True, exist_ok=True)
    record_path = state_dir / f"{profile}__{arm_name}.json"
    record = {"profile": profile, "arm": arm_name, "job_id": job_id}
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(record_path, flags, 0o644)
    except FileExistsError as error:
        raise ContractError(
            f"submission already recorded for {profile}/{arm_name}"
        ) from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(record, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return record


def _validate_arm(profile: str, arm_name: str) -> dict[str, Any]:
    manifest = load_manifest()
    allowed_arms = {item["name"] for item in manifest["arms"]}
    if profile not in {"canary", "full"} or arm_name not in allowed_arms:
        raise ContractError("invalid profile or arm")
    if profile == "canary" and arm_name not in manifest["canary_arms"]:
        raise ContractError(f"arm {arm_name} is not in the canary matrix")
    return manifest


def _write_exclusive_json(path: Path, record: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise ContractError(f"record already exists: {path}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(record, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _state_path(
    state_dir: Path, campaign_id: str, profile: str, arm_name: str, kind: str
) -> Path:
    if not SHA256_PATTERN.fullmatch(campaign_id):
        raise ContractError("campaign ID must be 64 lowercase hex characters")
    return state_dir / f"{campaign_id}__{profile}__{arm_name}.{kind}.json"


def monitor_path(*, state_dir: Path, campaign_id: str, profile: str) -> Path:
    """Return the canonical grouped-monitor record path for a campaign profile."""
    if not SHA256_PATTERN.fullmatch(campaign_id):
        raise ContractError("campaign ID must be 64 lowercase hex characters")
    if profile not in {"canary", "full"}:
        raise ContractError("monitor profile must be canary or full")
    return state_dir / f"{campaign_id}__{profile}.monitor.json"


def reserve_submission(
    *, state_dir: Path, campaign_id: str, profile: str, arm_name: str
) -> dict[str, str]:
    """Exclusively reserve one arm before any scheduler side effect."""
    _validate_arm(profile, arm_name)
    reservation = {
        "status": "reserved",
        "campaign_id": campaign_id,
        "profile": profile,
        "arm": arm_name,
        "reservation_id": secrets.token_hex(16),
    }
    path = _state_path(state_dir, campaign_id, profile, arm_name, "reservation")
    try:
        _write_exclusive_json(path, reservation)
    except ContractError as error:
        raise ContractError(
            f"submission already reserved for {profile}/{arm_name}"
        ) from error
    return reservation


def record_job(
    *,
    state_dir: Path,
    campaign_id: str,
    profile: str,
    arm_name: str,
    reservation_id: str,
    job_id: str,
) -> dict[str, str]:
    """Record the scheduler ID once, bound to a prior exclusive reservation."""
    _validate_arm(profile, arm_name)
    if not job_id.isdigit():
        raise ContractError("job ID must contain only decimal digits")
    reservation_path = _state_path(
        state_dir, campaign_id, profile, arm_name, "reservation"
    )
    if not reservation_path.is_file():
        raise ContractError(f"missing reservation for {profile}/{arm_name}")
    reservation = json.loads(reservation_path.read_text(encoding="utf-8"))
    if reservation.get("reservation_id") != reservation_id:
        raise ContractError(f"reservation token mismatch for {profile}/{arm_name}")
    record = {
        "status": "submitted",
        "campaign_id": campaign_id,
        "profile": profile,
        "arm": arm_name,
        "reservation_id": reservation_id,
        "job_id": job_id,
    }
    _write_exclusive_json(
        _state_path(state_dir, campaign_id, profile, arm_name, "submission"), record
    )
    return record


def record_completion(
    *,
    state_dir: Path,
    campaign_id: str,
    profile: str,
    arm_name: str,
    job_id: str,
    exit_code: int,
) -> dict[str, str | int]:
    """Record one terminal result only when it matches the submitted job."""
    _validate_arm(profile, arm_name)
    submission_path = _state_path(
        state_dir, campaign_id, profile, arm_name, "submission"
    )
    if not submission_path.is_file():
        raise ContractError(f"missing submission for {profile}/{arm_name}")
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    if submission.get("job_id") != job_id:
        raise ContractError(f"job ID mismatch for {profile}/{arm_name}")
    record: dict[str, str | int] = {
        "status": "success" if exit_code == 0 else "failed",
        "campaign_id": campaign_id,
        "profile": profile,
        "arm": arm_name,
        "job_id": job_id,
        "exit_code": exit_code,
    }
    _write_exclusive_json(
        _state_path(state_dir, campaign_id, profile, arm_name, "completion"), record
    )
    return record


def require_successful_canary(*, state_dir: Path, campaign_id: str) -> dict[str, Any]:
    """Unlock the full matrix only after both exact canary jobs succeed."""
    manifest = load_manifest()
    job_ids: dict[str, str] = {}
    for arm_name in manifest["canary_arms"]:
        completion_path = _state_path(
            state_dir, campaign_id, "canary", arm_name, "completion"
        )
        if not completion_path.is_file():
            raise ContractError(f"missing successful canary completion for {arm_name}")
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("status") != "success" or completion.get("exit_code") != 0:
            raise ContractError(f"canary did not succeed for {arm_name}")
        job_ids[arm_name] = completion["job_id"]
    path = monitor_path(state_dir=state_dir, campaign_id=campaign_id, profile="canary")
    if not path.is_file():
        raise ContractError("missing campaign-bound canary monitor record")
    monitor = json.loads(path.read_text(encoding="utf-8"))
    _require_equal("monitor status", monitor.get("status"), "monitored")
    _require_equal("monitor campaign ID", monitor.get("campaign_id"), campaign_id)
    _require_equal("monitor profile", monitor.get("profile"), "canary")
    _require_equal("monitor job IDs", monitor.get("job_ids"), list(job_ids.values()))
    passes = monitor.get("passes")
    interval_seconds = monitor.get("interval_seconds")
    monitor_window_seconds = monitor.get("monitor_window_seconds")
    observations = monitor.get("observations")
    if (
        not isinstance(passes, int)
        or passes < 6
        or not isinstance(interval_seconds, int)
        or interval_seconds < 60
        or not isinstance(monitor_window_seconds, int)
        or monitor_window_seconds < 300
        or monitor_window_seconds != (passes - 1) * interval_seconds
        or not isinstance(observations, list)
        or len(observations) != passes
    ):
        raise ContractError("canary monitor cadence is incomplete")
    for index, observation in enumerate(observations):
        expected_observation = {
            "pass": index + 1,
            "elapsed_seconds": index * interval_seconds,
        }
        if not isinstance(observation, dict) or any(
            observation.get(key) != value for key, value in expected_observation.items()
        ):
            raise ContractError("canary monitor observations are invalid")
    return {
        "status": "full-unlocked",
        "campaign_id": campaign_id,
        "job_ids": job_ids,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    plan = subparsers.add_parser("plan")
    plan.add_argument("--profile", choices=("canary", "full"), required=True)
    plan.add_argument("--source-commit", required=True)
    plan.add_argument("--container", type=Path, required=True)
    plan.add_argument("--container-sha256", required=True)
    plan.add_argument("--output-root", type=Path, required=True)
    plan.add_argument("--preflight-record", type=Path, required=True)
    plan.add_argument("--canary-record", type=Path)

    claim = subparsers.add_parser("claim")
    claim.add_argument("--state-dir", type=Path, required=True)
    claim.add_argument("--profile", choices=("canary", "full"), required=True)
    claim.add_argument("--arm", required=True)
    claim.add_argument("--job-id", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--repo-root", type=Path, required=True)
    preflight.add_argument("--source-commit", required=True)
    preflight.add_argument("--container", type=Path, required=True)
    preflight.add_argument("--container-sha256", required=True)
    preflight.add_argument("--record", type=Path, required=True)

    canary = subparsers.add_parser("materialize-canary")
    canary.add_argument("--destination", type=Path, required=True)
    canary.add_argument("--record", type=Path, required=True)

    reserve = subparsers.add_parser("reserve")
    reserve.add_argument("--state-dir", type=Path, required=True)
    reserve.add_argument("--campaign-id", required=True)
    reserve.add_argument("--profile", choices=("canary", "full"), required=True)
    reserve.add_argument("--arm", required=True)

    record_job_parser = subparsers.add_parser("record-job")
    record_job_parser.add_argument("--state-dir", type=Path, required=True)
    record_job_parser.add_argument("--campaign-id", required=True)
    record_job_parser.add_argument(
        "--profile", choices=("canary", "full"), required=True
    )
    record_job_parser.add_argument("--arm", required=True)
    record_job_parser.add_argument("--reservation-id", required=True)
    record_job_parser.add_argument("--job-id", required=True)

    complete = subparsers.add_parser("complete")
    complete.add_argument("--state-dir", type=Path, required=True)
    complete.add_argument("--campaign-id", required=True)
    complete.add_argument("--profile", choices=("canary", "full"), required=True)
    complete.add_argument("--arm", required=True)
    complete.add_argument("--job-id", required=True)
    complete.add_argument("--exit-code", type=int, required=True)

    unlock = subparsers.add_parser("unlock-full")
    unlock.add_argument("--state-dir", type=Path, required=True)
    unlock.add_argument("--campaign-id", required=True)
    return parser


def main() -> int:
    """Run the selected benchmark planning or lifecycle action."""
    args = _build_parser().parse_args()
    try:
        if args.action == "plan":
            preflight_record = json.loads(
                args.preflight_record.read_text(encoding="utf-8")
            )
            canary_record = (
                json.loads(args.canary_record.read_text(encoding="utf-8"))
                if args.canary_record is not None
                else None
            )
            result = render_plan(
                profile=args.profile,
                source_commit=args.source_commit,
                container=args.container,
                container_sha256=args.container_sha256,
                output_root=args.output_root,
                preflight=preflight_record,
                canary=canary_record,
            )
        elif args.action == "claim":
            result = claim_submission(
                state_dir=args.state_dir,
                profile=args.profile,
                arm_name=args.arm,
                job_id=args.job_id,
            )
        elif args.action == "preflight":
            manifest = load_manifest()
            protected_paths_by_head = {
                manifest["pr_head"]: [
                    Path(manifest["recipe"]),
                    Path("examples/nemo_gym/grpo_qwen3_30ba3b_thinking_swe1.yaml"),
                    Path("examples/nemo_gym/run_qwen3_swe_rollout_only.sh"),
                    Path(manifest["entrypoint"]),
                ]
            }
            source = verify_source(
                repo_root=args.repo_root,
                source_commit=args.source_commit,
                required_ancestors=[manifest["pr_head"]],
                protected_paths_by_head=protected_paths_by_head,
                source_files_sha256={
                    Path(path): digest
                    for path, digest in manifest["source_files_sha256"].items()
                },
            )
            artifacts = verify_artifacts(
                manifest=manifest,
                container=args.container,
                container_sha256=args.container_sha256,
            )
            result = make_preflight_record(
                manifest=manifest, source=source, artifacts=artifacts
            )
            _write_exclusive_json(args.record, result)
        elif args.action == "materialize-canary":
            data = load_manifest()["data"]
            result = materialize_canary(
                source=Path(data["path"]),
                source_sha256=data["sha256"],
                source_lines=data["lines"],
                destination=args.destination,
            )
            _write_exclusive_json(args.record, result)
        elif args.action == "reserve":
            result = reserve_submission(
                state_dir=args.state_dir,
                campaign_id=args.campaign_id,
                profile=args.profile,
                arm_name=args.arm,
            )
        elif args.action == "record-job":
            result = record_job(
                state_dir=args.state_dir,
                campaign_id=args.campaign_id,
                profile=args.profile,
                arm_name=args.arm,
                reservation_id=args.reservation_id,
                job_id=args.job_id,
            )
        elif args.action == "complete":
            result = record_completion(
                state_dir=args.state_dir,
                campaign_id=args.campaign_id,
                profile=args.profile,
                arm_name=args.arm,
                job_id=args.job_id,
                exit_code=args.exit_code,
            )
        else:
            result = require_successful_canary(
                state_dir=args.state_dir, campaign_id=args.campaign_id
            )
    except ContractError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
