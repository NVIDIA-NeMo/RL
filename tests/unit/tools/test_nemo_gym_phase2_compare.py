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

import hashlib
import json
from pathlib import Path

import pytest

from tools.nemo_gym_phase2_compare import (
    build_comparison,
    load_run,
    render_comparison,
    write_comparison,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_run(
    root: Path,
    *,
    policy: str,
    repeat_id: str,
    rewards: list[int],
    workload_hash: str | None = None,
    runtime_verification_nonce: str | None = None,
) -> Path:
    directory = root / f"{policy}-{repeat_id}"
    (directory / "evaluation").mkdir(parents=True)
    matrix_invariants = {"repeat_id": repeat_id, "fixed_config": "same"}
    if runtime_verification_nonce is not None:
        matrix_invariants["experiment"] = {
            "software": {
                "runtime_verification": {
                    "driver": {
                        "artifact_sha256": runtime_verification_nonce * 64,
                        "package_inventory": {"count": 3, "sha256": "a" * 64},
                        "uv_sync_check": {
                            "command": ["uv", "sync", "--frozen", "--check"],
                            "returncode": 0,
                            "stdout": "",
                            "stderr": (
                                f"Checked 3 packages in {runtime_verification_nonce}ms"
                            ),
                        },
                    }
                }
            }
        }
    invariants_hash = _canonical_sha256(matrix_invariants)
    launch_id = f"{policy}-{repeat_id}-fresh-launch"
    manifest = {
        "run_id": f"{policy}-{repeat_id}",
        "repeat_id": repeat_id,
        "engine_launch_id": launch_id,
        "comparison_invariants_sha256": invariants_hash,
        "routing_policy": policy,
        "input_data": {
            "workload_sha256": workload_hash or f"workload-{repeat_id}",
            "workload_seed": repeat_id.removeprefix("repeat-"),
            "workload_replay_faithful": True,
            "warmup_workload_sha256": f"warmup-{repeat_id}",
            "warmup_requests": 32,
        },
        "experiment": {
            "engine": {"fresh": True, "launch_id": launch_id},
            "derived": {
                "matrix_comparison_invariants": matrix_invariants,
                "matrix_comparison_invariants_sha256": invariants_hash,
            },
        },
        "artifacts": [
            "artifact_checksums.sha256",
            "evaluation/outcomes.jsonl",
            "manifest.json",
            "summary.json",
        ],
    }
    correct = sum(rewards)
    router_hits = 80 if policy == "cache_aware" else 0
    router_misses = 20 if policy == "cache_aware" else 0
    summary = {
        "run_id": manifest["run_id"],
        "gates": {"passed": True},
        "accuracy": {
            "correct": correct,
            "evaluated": len(rewards),
            "value": correct / len(rewards),
        },
        "cache": {
            "backend_prefix_cache": {
                "hits": 70 if policy == "direct" else 90,
                "queries": 100,
                "hit_rate": 0.7 if policy == "direct" else 0.9,
            },
            "router_routing_cache": {
                "hits": router_hits,
                "misses": router_misses,
            },
        },
        "request_timing": {"p99_s": 4.0 if policy == "direct" else 3.5},
        "session_timing": {"p99_s": 5.0 if policy == "direct" else 4.5},
        "makespan_s": 20.0 if policy == "direct" else 18.0,
        "throughput": {"output_tokens_per_s": 100.0 if policy == "direct" else 110.0},
        "backend": {
            "request_distribution": {"coefficient_of_variation": 0.1},
            "generated_token_distribution": {"coefficient_of_variation": 0.2},
            "latency_seconds": {
                "backend_ttft_p99": 0.8 if policy == "direct" else 0.7,
                "backend_itl_p99": 0.04 if policy == "direct" else 0.03,
            },
        },
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with (directory / "evaluation" / "outcomes.jsonl").open(
        "w", encoding="utf-8"
    ) as output:
        for prompt_index, reward in enumerate(rewards):
            output.write(
                json.dumps(
                    {
                        "prompt_index": prompt_index,
                        "generation_index": 0,
                        "reward": reward,
                    }
                )
                + "\n"
            )
    audited_paths = [
        directory / "manifest.json",
        directory / "summary.json",
        directory / "evaluation" / "outcomes.jsonl",
    ]
    (directory / "artifact_checksums.sha256").write_text(
        "".join(
            f"{_sha256(path)}  {path.relative_to(directory).as_posix()}\n"
            for path in audited_paths
        ),
        encoding="utf-8",
    )
    return directory


def _matrix(tmp_path: Path) -> list[Path]:
    directories = []
    for repeat_id in ("repeat-1", "repeat-2"):
        directories.extend(
            [
                _write_run(
                    tmp_path,
                    policy="direct",
                    repeat_id=repeat_id,
                    rewards=[1, 0],
                ),
                _write_run(
                    tmp_path,
                    policy="cache_aware",
                    repeat_id=repeat_id,
                    rewards=[1, 1],
                ),
                _write_run(
                    tmp_path,
                    policy="consistent_hash",
                    repeat_id=repeat_id,
                    rewards=[1, 0],
                ),
            ]
        )
    return directories


def _refresh_manifest_checksum(directory: Path) -> None:
    manifest_path = directory / "manifest.json"
    checksum_path = directory / "artifact_checksums.sha256"
    checksum_rows = checksum_path.read_text(encoding="utf-8").splitlines()
    checksum_path.write_text(
        "\n".join(
            (
                f"{_sha256(manifest_path)}  manifest.json"
                if row.endswith("  manifest.json")
                else row
            )
            for row in checksum_rows
        )
        + "\n",
        encoding="utf-8",
    )


def test_build_comparison_requires_and_pairs_complete_repeated_matrix(
    tmp_path: Path,
) -> None:
    runs = [load_run(directory) for directory in _matrix(tmp_path)]

    summary, records = build_comparison(runs, min_repeats=2)

    assert summary["gates"]["passed"] is True
    assert summary["arms"]["direct"]["accuracy"]["value"] == 0.5
    assert summary["arms"]["cache_aware"]["backend_prefix_cache"]["hit_rate"] == 0.9
    cache_comparison = summary["paired_accuracy"]["cache_aware_vs_direct"]
    assert cache_comparison["common_coverage"] == 4
    assert cache_comparison["absolute_percentage_point_difference"] == 50.0
    assert cache_comparison["candidate_only_correct"] == 2
    assert cache_comparison["exact_mcnemar_p_value"] == 0.5
    assert (
        summary["paired_operational"]["cache_aware_vs_direct"]["request_p99_s"][
            "candidate_minus_direct"
        ]["mean"]
        == -0.5
    )
    assert summary["paired_operational"]["cache_aware_vs_direct"]["ttft_p99_s"][
        "candidate_minus_direct"
    ]["mean"] == pytest.approx(-0.1)
    assert summary["paired_operational"]["cache_aware_vs_direct"]["itl_p99_s"][
        "candidate_minus_direct"
    ]["mean"] == pytest.approx(-0.01)
    assert len(records) == 8
    report = render_comparison(summary)
    assert "cache_aware_vs_direct|4|50.000" in report
    assert "TTFT p99 mean" in report
    assert "ITL p99 (s)" in report


def test_build_comparison_rejects_workload_mismatch(tmp_path: Path) -> None:
    directories = _matrix(tmp_path)
    mismatched = directories[1]
    manifest_path = mismatched / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["input_data"]["workload_sha256"] = "different"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _refresh_manifest_checksum(mismatched)

    with pytest.raises(ValueError, match="inputs or outcome coverage"):
        build_comparison([load_run(path) for path in directories], min_repeats=2)


def test_build_comparison_ignores_volatile_uv_check_output(tmp_path: Path) -> None:
    directories = []
    for repeat_id in ("repeat-1", "repeat-2"):
        for nonce, policy in zip("123", ("direct", "cache_aware", "consistent_hash")):
            directories.append(
                _write_run(
                    tmp_path,
                    policy=policy,
                    repeat_id=repeat_id,
                    rewards=[1, 0],
                    runtime_verification_nonce=nonce,
                )
            )

    summary, _ = build_comparison(
        [load_run(path) for path in directories], min_repeats=2
    )

    assert summary["gates"]["comparison_invariants_matched_within_repeat"] is True
    assert summary["gates"]["passed"] is True


def test_build_comparison_rejects_reused_engine_launch_identity(
    tmp_path: Path,
) -> None:
    directories = _matrix(tmp_path)
    first_manifest = json.loads(
        (directories[0] / "manifest.json").read_text(encoding="utf-8")
    )
    reused = directories[1]
    reused_manifest_path = reused / "manifest.json"
    reused_manifest = json.loads(reused_manifest_path.read_text(encoding="utf-8"))
    launch_id = first_manifest["engine_launch_id"]
    reused_manifest["engine_launch_id"] = launch_id
    reused_manifest["experiment"]["engine"]["launch_id"] = launch_id
    reused_manifest_path.write_text(json.dumps(reused_manifest), encoding="utf-8")
    _refresh_manifest_checksum(reused)

    summary, _ = build_comparison(
        [load_run(path) for path in directories], min_repeats=2
    )

    assert summary["gates"]["fresh_engine_launch_ids_unique"] is False
    assert summary["gates"]["passed"] is False


def test_load_run_rejects_tampered_source_artifact(tmp_path: Path) -> None:
    directory = _write_run(
        tmp_path,
        policy="direct",
        repeat_id="repeat-1",
        rewards=[1, 0],
    )
    (directory / "summary.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_run(directory)


def test_write_comparison_archives_sources_and_checksums(tmp_path: Path) -> None:
    runs = [load_run(directory) for directory in _matrix(tmp_path / "runs")]
    summary, records = build_comparison(runs, min_repeats=2)
    output_dir = tmp_path / "comparison"

    write_comparison(
        output_dir,
        runs=runs,
        summary=summary,
        paired_records=records,
    )

    assert (output_dir / "comparison-summary.json").is_file()
    assert (output_dir / "comparison-report.md").is_file()
    assert (output_dir / "comparison-manifest.json").is_file()
    assert (output_dir / "paired-outcomes.jsonl").is_file()
    checksums = (output_dir / "artifact_checksums.sha256").read_text(encoding="utf-8")
    assert "  comparison-summary.json\n" in checksums
    assert "  paired-outcomes.jsonl\n" in checksums
