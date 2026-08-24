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

import json
import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = REPO_ROOT / "experiments/qwen3_30ba3b_swe_rollout_pr3733"
MANIFEST = EXPERIMENT / "benchmark_matrix.json"
CLI = EXPERIMENT / "benchmark.py"
SUBMIT_CLI = EXPERIMENT / "submit.py"
PR_HEAD = "b580dd8927b88c996470d315e74d57bf0cb4090e"
SOURCE_COMMIT = "1" * 40
CONTAINER_SHA256 = "2" * 64
CAMPAIGN_ID = "3" * 64


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location("swe_benchmark", CLI)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_submit_module():
    sys.path.insert(0, str(EXPERIMENT))
    try:
        spec = importlib.util.spec_from_file_location("swe_submit", SUBMIT_CLI)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EXPERIMENT))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _render_plan(tmp_path: Path, profile: str) -> dict[str, Any]:
    benchmark = _load_benchmark_module()
    manifest = json.loads(MANIFEST.read_text())
    container = "/lustre/containers/nemo-rl-pr3733.sqsh"
    metadata_paths = [container, manifest["data"]["path"]]
    target = manifest["target"]
    metadata_paths.extend(
        [
            f"{target['path']}/config.json",
            f"{target['path']}/model.safetensors.index.json",
            *(f"{target['path']}/{name}" for name in target["weight_sha256"]),
        ]
    )
    for draft in manifest["drafts"].values():
        metadata_paths.extend(
            [
                f"{draft['path']}/config.json",
                *(f"{draft['path']}/{name}" for name in draft["weight_sha256"]),
            ]
        )
    preflight = benchmark.make_preflight_record(
        manifest=manifest,
        source={
            "status": "verified",
            "source_commit": SOURCE_COMMIT,
            "required_ancestors": [PR_HEAD],
            "protected_paths_by_head": {
                PR_HEAD: [
                    manifest["recipe"],
                    "examples/nemo_gym/grpo_qwen3_30ba3b_thinking_swe1.yaml",
                    "examples/nemo_gym/run_qwen3_swe_rollout_only.sh",
                    manifest["entrypoint"],
                ]
            },
            "source_files_sha256": manifest["source_files_sha256"],
        },
        artifacts={
            "status": "verified",
            "container": {"path": container, "sha256": CONTAINER_SHA256},
            "target": {
                key: target[key]
                for key in (
                    "config_sha256",
                    "model_index_sha256",
                    "weight_files",
                    "weight_bytes",
                    "weight_sha256",
                )
            },
            "drafts": {
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
            },
            "data": manifest["data"],
            "file_metadata": {
                path: {"size": 1, "mtime_ns": 1, "inode": 1} for path in metadata_paths
            },
        },
    )
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(json.dumps(preflight))
    command = [
        sys.executable,
        str(CLI),
        "plan",
        "--profile",
        profile,
        "--source-commit",
        SOURCE_COMMIT,
        "--container",
        container,
        "--container-sha256",
        CONTAINER_SHA256,
        "--output-root",
        str(tmp_path / "outputs"),
        "--preflight-record",
        str(preflight_path),
    ]
    if profile == "canary":
        canary_path = tmp_path / "outputs/inputs/swebench_verified_first1.jsonl"
        canary_path.parent.mkdir(parents=True)
        canary_path.write_text('{"instance_id":"canary"}\n')
        canary = {
            "path": str(canary_path),
            "sha256": _sha256(canary_path),
            "lines": 1,
            "selection": "first JSONL record",
            "parent_path": manifest["data"]["path"],
            "parent_sha256": manifest["data"]["sha256"],
            "parent_lines": manifest["data"]["lines"],
        }
        canary_record = tmp_path / "canary.json"
        canary_record.write_text(json.dumps(canary))
        command.extend(["--canary-record", str(canary_record)])
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_manifest_pins_authoritative_swe_inputs_and_five_matched_arms() -> None:
    manifest = json.loads(MANIFEST.read_text())

    assert manifest["pr_head"] == PR_HEAD
    assert "rollout_benchmark_pr_head" not in manifest
    assert manifest["workload_label"] == "SWE trajectory-collection rollout-only"
    assert manifest["output_root_prefix"] == (
        "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/"
        "users/sna/experiments"
    )
    assert manifest["entrypoint"] == "examples/nemo_gym/run_grpo_nemo_gym.py"
    assert set(manifest["source_files_sha256"]) >= {
        manifest["entrypoint"],
        "examples/nemo_gym/run_grpo_nemo_gym.py",
        "nemo_rl/algorithms/grpo.py",
        "nemo_rl/environments/nemo_gym.py",
    }
    assert all(len(digest) == 64 for digest in manifest["source_files_sha256"].values())
    assert manifest["recipe"] == (
        "examples/configs/recipes/llm/"
        "grpo-qwen3-30ba3b-thinking-swe1-2n4g-megatron-tp2pp2-"
        "rollout-only-specdec.yaml"
    )
    target = manifest["target"]
    assert {key: value for key, value in target.items() if key != "weight_sha256"} == {
        "config_sha256": (
            "a1ee086a68d0cbfc87316da00ba4b8507bd1292978108e2496201a30a450f438"
        ),
        "path": (
            "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/"
            "sna/hf_home/hub/models--Qwen--Qwen3-30B-A3B-Thinking-2507/"
            "snapshots/144afc2f379b542fdd4e85a1fcd5e1f79112d95d"
        ),
        "snapshot": "144afc2f379b542fdd4e85a1fcd5e1f79112d95d",
        "model_index_sha256": (
            "8dde190b862c7c80ec7403c6495de00c60bbaf246ed479cee4506284989c584c"
        ),
        "weight_files": 16,
        "weight_bytes": 61066575656,
    }
    assert len(target["weight_sha256"]) == 16
    assert target["weight_sha256"]["model-00001-of-00016.safetensors"] == (
        "d6f04f15f023d0a0eff2d073b4276dd0151100bd94b1799fa27166bf99d68a1c"
    )
    assert manifest["data"] == {
        "lines": 500,
        "path": (
            "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/"
            "sna/SpecDec-RL-main-vllm020-20260606/data/"
            "swebench_verified_prompts_all.jsonl"
        ),
        "sha256": "38434589e57ac4494052cf826f2eca24eea5d75b6889cf9e37fbe9c18dc95c1a",
    }
    assert manifest["common"] == {
        "generation_batch_size": 64,
        "generation_tensor_parallel_size": 2,
        "generation_num_nodes": 1,
        "gpus_per_node": 4,
        "max_total_sequence_length": 131072,
        "num_generations_per_prompt": 1,
        "num_nodes": 2,
        "optimizer": None,
        "scheduler": None,
        "temperature": 1.0,
        "top_p": 1.0,
        "validation_batch_size": 500,
        "wandb_entity": "nvidia",
        "wandb_project": "sna-specdec",
    }
    assert manifest["request_buckets"] == [1, 2, 4, 8, 16, 32, 64, 128, 256]
    assert [arm["name"] for arm in manifest["arms"]] == [
        "baseline",
        "dflash_k5",
        "dflash_k7",
        "dspark_k5",
        "dspark_k7",
    ]
    assert manifest["canary_arms"] == ["baseline", "dflash_k5"]


def test_arm_yaml_overlays_inherit_one_authoritative_swe_recipe() -> None:
    manifest = json.loads(MANIFEST.read_text())
    expected = {
        "baseline": (None, 0),
        "dflash_k5": ("dflash", 5),
        "dflash_k7": ("dflash", 7),
        "dspark_k5": ("dspark", 5),
        "dspark_k7": ("dspark", 7),
    }

    for arm in manifest["arms"]:
        overlay = REPO_ROOT / arm["config"]
        text = overlay.read_text()
        assert (
            "defaults: ../../../examples/configs/recipes/llm/"
            "grpo-qwen3-30ba3b-thinking-swe1-2n4g-megatron-tp2pp2-"
            "rollout-only-specdec.yaml"
        ) in text
        method, k = expected[arm["name"]]
        if method is None:
            assert "speculative_config: null" in text
        else:
            assert f"method: {method}" in text
            assert f"num_speculative_tokens: {k}" in text
            assert manifest["drafts"][method]["path"] in text


def test_pr3733_trajectory_collection_batches_the_full_validation_dataset() -> None:
    runner = (REPO_ROOT / "examples/nemo_gym/run_grpo_nemo_gym.py").read_text()

    assert "config.grpo.max_val_samples = len(val_dataset)" in runner
    assert "config.grpo.val_batch_size = config.grpo.max_val_samples" in runner
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["data"]["lines"] == 500
    assert manifest["common"]["validation_batch_size"] == 500
    assert manifest["common"]["generation_tensor_parallel_size"] == 2


@pytest.mark.parametrize(
    ("arm_name", "method", "k", "capture_sizes"),
    [
        ("baseline", None, 0, [1, 2, 4, 8, 16, 32, 64, 128, 256]),
        (
            "dflash_k5",
            "dflash",
            5,
            [6, 12, 24, 48, 96, 192, 384, 768, 1536],
        ),
        (
            "dflash_k7",
            "dflash",
            7,
            [8, 16, 32, 64, 128, 256, 512, 1024, 2048],
        ),
        (
            "dspark_k5",
            "dspark",
            5,
            [
                5,
                6,
                10,
                12,
                20,
                24,
                40,
                48,
                80,
                96,
                160,
                192,
                320,
                384,
                640,
                768,
                1280,
                1536,
            ],
        ),
        (
            "dspark_k7",
            "dspark",
            7,
            [
                7,
                8,
                14,
                16,
                28,
                32,
                56,
                64,
                112,
                128,
                224,
                256,
                448,
                512,
                896,
                1024,
                1792,
                2048,
            ],
        ),
    ],
)
def test_full_plan_has_explicit_k_semantics_and_cuda_graph_coverage(
    tmp_path: Path,
    arm_name: str,
    method: str | None,
    k: int,
    capture_sizes: list[int],
) -> None:
    plan = _render_plan(tmp_path, "full")
    run = next(item for item in plan["runs"] if item["arm"] == arm_name)

    assert run["method"] == method
    assert run["num_speculative_tokens"] == k
    assert run["k_semantics"] == "draft tokens proposed per decoding step"
    assert run["cudagraph_capture_sizes"] == capture_sizes
    assert run["common"] == plan["common"]
    assert run["data_path"] == plan["data"]["path"]
    assert run["environment"]["WANDB_ENTITY"] == "nvidia"
    assert plan["workload_label"] == "SWE trajectory-collection rollout-only"
    assert "logger.wandb_enabled=true" in run["command"]
    assert run["command"][:6] == [
        "uv",
        "run",
        "--frozen",
        "--no-sync",
        "examples/nemo_gym/run_grpo_nemo_gym.py",
        "--config",
    ]
    assert run["command"][6] == run["config"]
    assert (
        f"cudagraph_capture_sizes: {capture_sizes}"
        in (REPO_ROOT / run["config"]).read_text()
    )
    assert not any("cudagraph_capture_sizes=" in item for item in run["command"])
    assert not any("speculative_config=" in item for item in run["command"])


def test_canary_is_bounded_but_keeps_native_swe_sampling_and_topology(
    tmp_path: Path,
) -> None:
    plan = _render_plan(tmp_path, "canary")

    assert [run["arm"] for run in plan["runs"]] == ["baseline", "dflash_k5"]
    assert plan["profile"] == "canary"
    assert plan["data"]["parent_lines"] == 500
    assert plan["data"]["lines"] == 1
    assert plan["common"]["temperature"] == 1.0
    assert plan["common"]["top_p"] == 1.0
    assert plan["common"]["num_nodes"] == 2
    assert plan["common"]["generation_num_nodes"] == 1
    assert plan["common"]["gpus_per_node"] == 4
    assert all(
        run["bounded_override"] == "one deterministic prompt" for run in plan["runs"]
    )


def test_claim_is_exclusive_and_records_one_submission(tmp_path: Path) -> None:
    command = [
        sys.executable,
        str(CLI),
        "claim",
        "--state-dir",
        str(tmp_path),
        "--profile",
        "canary",
        "--arm",
        "baseline",
        "--job-id",
        "12345",
    ]

    first = subprocess.run(command, capture_output=True, text=True, check=False)
    second = subprocess.run(command, capture_output=True, text=True, check=False)

    assert first.returncode == 0, first.stderr
    assert json.loads(first.stdout)["job_id"] == "12345"
    assert second.returncode == 2
    assert "already recorded" in second.stderr
    records = list(tmp_path.glob("*.json"))
    assert len(records) == 1
    assert json.loads(records[0].read_text())["arm"] == "baseline"


def test_artifact_preflight_checks_hashes_sizes_and_records(tmp_path: Path) -> None:
    benchmark = _load_benchmark_module()
    target = tmp_path / "target"
    dflash = tmp_path / "dflash"
    dspark = tmp_path / "dspark"
    for directory in (target, dflash, dspark):
        directory.mkdir()
        (directory / "config.json").write_text("{}\n")
    (target / "model.safetensors.index.json").write_text("index\n")
    (target / "a.safetensors").write_bytes(b"target")
    (dflash / "a.safetensors").write_bytes(b"dflash")
    (dspark / "a.safetensors").write_bytes(b"dspark")
    data = tmp_path / "data.jsonl"
    data.write_text('{"id": 1}\n{"id": 2}\n')
    container = tmp_path / "runtime.sqsh"
    container.write_bytes(b"container")
    manifest = {
        "target": {
            "path": str(target),
            "config_sha256": _sha256(target / "config.json"),
            "model_index_sha256": _sha256(target / "model.safetensors.index.json"),
            "weight_files": 1,
            "weight_bytes": 6,
            "weight_sha256": {"a.safetensors": _sha256(target / "a.safetensors")},
        },
        "drafts": {
            "dflash": {
                "path": str(dflash),
                "config_sha256": _sha256(dflash / "config.json"),
                "weight_files": 1,
                "weight_bytes": 6,
                "weight_sha256": {"a.safetensors": _sha256(dflash / "a.safetensors")},
            },
            "dspark": {
                "path": str(dspark),
                "config_sha256": _sha256(dspark / "config.json"),
                "weight_files": 1,
                "weight_bytes": 6,
                "weight_sha256": {"a.safetensors": _sha256(dspark / "a.safetensors")},
            },
        },
        "data": {"path": str(data), "sha256": _sha256(data), "lines": 2},
    }

    result = benchmark.verify_artifacts(
        manifest=manifest,
        container=container,
        container_sha256=_sha256(container),
    )
    assert result["status"] == "verified"
    assert result["data"]["lines"] == 2
    benchmark.validate_artifact_metadata(artifacts=result)
    (dflash / "a.safetensors").write_bytes(b"dfxxxx")
    with pytest.raises(benchmark.ContractError, match="DFlash weight SHA256"):
        benchmark.verify_artifacts(
            manifest=manifest,
            container=container,
            container_sha256=_sha256(container),
        )
    with pytest.raises(benchmark.ContractError, match="metadata drift"):
        benchmark.validate_artifact_metadata(artifacts=result)


def test_source_preflight_requires_clean_descendant_and_protects_recipe(
    tmp_path: Path,
) -> None:
    benchmark = _load_benchmark_module()
    repo = tmp_path / "repo"
    recipe = Path("examples/configs/recipes/llm/rollout.yaml")
    (repo / recipe.parent).mkdir(parents=True)
    (repo / recipe).write_text("defaults: swe.yaml\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "base",
        ],
        check=True,
    )
    pr_head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    entrypoint = Path("examples/nemo_gym/run_grpo_rollout_benchmark.py")
    (repo / entrypoint.parent).mkdir(parents=True)
    (repo / entrypoint).write_text("def main(): pass\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "rollout benchmark",
        ],
        check=True,
    )
    rollout_head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    experiment = repo / "experiments/benchmark.json"
    experiment.parent.mkdir()
    experiment.write_text("{}\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "experiment",
        ],
        check=True,
    )
    source_commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()

    result = benchmark.verify_source(
        repo_root=repo,
        source_commit=source_commit,
        required_ancestors=[pr_head, rollout_head],
        protected_paths_by_head={pr_head: [recipe], rollout_head: [entrypoint]},
        source_files_sha256={recipe: _sha256(repo / recipe)},
    )
    assert result["status"] == "verified"
    (repo / recipe).write_text("defaults: drift.yaml\n")
    with pytest.raises(benchmark.ContractError, match="not clean"):
        benchmark.verify_source(
            repo_root=repo,
            source_commit=source_commit,
            required_ancestors=[pr_head, rollout_head],
            protected_paths_by_head={pr_head: [recipe], rollout_head: [entrypoint]},
            source_files_sha256={recipe: _sha256(repo / recipe)},
        )

    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "drift",
        ],
        check=True,
    )
    drift_commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    with pytest.raises(benchmark.ContractError, match="source file SHA256"):
        benchmark.verify_source(
            repo_root=repo,
            source_commit=drift_commit,
            required_ancestors=[pr_head, rollout_head],
            protected_paths_by_head={pr_head: [], rollout_head: [entrypoint]},
            source_files_sha256={recipe: "0" * 64},
        )


def test_canary_materialization_is_first_record_and_records_parent_identity(
    tmp_path: Path,
) -> None:
    benchmark = _load_benchmark_module()
    source = tmp_path / "source.jsonl"
    source.write_text('{"instance_id":"first"}\n{"instance_id":"second"}\n')
    destination = tmp_path / "inputs/canary.jsonl"

    result = benchmark.materialize_canary(
        source=source,
        source_sha256=_sha256(source),
        source_lines=2,
        destination=destination,
    )

    assert destination.read_text() == '{"instance_id":"first"}\n'
    assert result["selection"] == "first JSONL record"
    assert result["parent_sha256"] == _sha256(source)
    assert result["sha256"] == _sha256(destination)
    with pytest.raises(benchmark.ContractError, match="already exists"):
        benchmark.materialize_canary(
            source=source,
            source_sha256=_sha256(source),
            source_lines=2,
            destination=destination,
        )


def test_reservation_precedes_job_record_and_full_requires_successful_canary(
    tmp_path: Path,
) -> None:
    benchmark = _load_benchmark_module()
    state = tmp_path / "state"

    reservations = {}
    for arm, job_id in (("baseline", "123"), ("dflash_k5", "124")):
        reserved = benchmark.reserve_submission(
            state_dir=state,
            campaign_id=CAMPAIGN_ID,
            profile="canary",
            arm_name=arm,
        )
        reservations[arm] = reserved
        with pytest.raises(benchmark.ContractError, match="already reserved"):
            benchmark.reserve_submission(
                state_dir=state, profile="canary", arm_name=arm, campaign_id=CAMPAIGN_ID
            )
        submitted = benchmark.record_job(
            state_dir=state,
            campaign_id=CAMPAIGN_ID,
            profile="canary",
            arm_name=arm,
            reservation_id=reserved["reservation_id"],
            job_id=job_id,
        )
        assert submitted["status"] == "submitted"
        completed = benchmark.record_completion(
            state_dir=state,
            campaign_id=CAMPAIGN_ID,
            profile="canary",
            arm_name=arm,
            job_id=job_id,
            exit_code=0,
        )
        assert completed["status"] == "success"

    benchmark._write_exclusive_json(
        state / f"{CAMPAIGN_ID}__canary.monitor.json",
        {
            "status": "monitored",
            "campaign_id": CAMPAIGN_ID,
            "profile": "canary",
            "job_ids": ["123", "124"],
            "passes": 6,
            "interval_seconds": 60,
            "monitor_window_seconds": 300,
            "observations": [
                {"pass": index + 1, "elapsed_seconds": index * 60, "squeue": []}
                for index in range(6)
            ],
        },
    )
    unlocked = benchmark.require_successful_canary(
        state_dir=state, campaign_id=CAMPAIGN_ID
    )
    assert unlocked["status"] == "full-unlocked"
    assert unlocked["job_ids"] == {"baseline": "123", "dflash_k5": "124"}
    with pytest.raises(benchmark.ContractError, match="missing successful canary"):
        benchmark.require_successful_canary(state_dir=state, campaign_id="4" * 64)


def test_scheduler_contract_uses_pr3733_trajectory_collection_topology(
    tmp_path: Path,
) -> None:
    submit = _load_submit_module()
    plan = _render_plan(tmp_path, "canary")
    run = plan["runs"][0]
    repo = Path("/home/sna/nemo-rl-q30-swe")

    contract = submit.build_scheduler_contract(
        plan=plan,
        run=run,
        repo_root=repo,
        account="coreai_dlalgo_nemorl",
        partition="batch",
        time_limit="04:00:00",
    )

    assert contract["sbatch_args"][:2] == ["sbatch", "--export=ALL"]
    assert "--nodes=2" in contract["sbatch_args"]
    assert "--segment=1" in contract["sbatch_args"]
    assert "--gres=gpu:4" in contract["sbatch_args"]
    assert contract["sbatch_args"][-1] == "ray.sub"
    assert contract["environment"]["CONTAINER"] == plan["container"]["path"]
    assert contract["environment"]["GPUS_PER_NODE"] == "4"
    assert contract["environment"]["MOUNTS"] == f"/lustre:/lustre,{repo}:{repo}"
    assert run["config"] in contract["environment"]["COMMAND"]
    assert "speculative_config: null" in (REPO_ROOT / run["config"]).read_text()
    assert "data.train.data_path=" in contract["environment"]["COMMAND"]
    assert "WANDB_ENTITY=nvidia" in contract["environment"]["COMMAND"]


def test_scheduler_test_only_gate_and_pre_submission_reservation(
    tmp_path: Path,
) -> None:
    submit = _load_submit_module()
    plan = _render_plan(tmp_path, "canary")
    run = plan["runs"][0]
    state_dir = Path(plan["output_root"]) / "state"
    preflight = json.loads((tmp_path / "preflight.json").read_text())
    calls: list[list[str]] = []

    def fake_runner(command, **kwargs):
        calls.append(command)
        assert Path(run["output_dir"]).is_dir()
        if "--test-only" not in command:
            assert list(state_dir.glob("*__canary__baseline.reservation.json"))
            return subprocess.CompletedProcess(command, 0, "98765\n", "")
        return subprocess.CompletedProcess(command, 0, "Job 1 to start at ...\n", "")

    with pytest.raises(
        submit.ContractError, match="matching successful sbatch --test-only"
    ):
        submit.run_scheduler_action(
            mode="submit",
            plan=plan,
            run=run,
            repo_root=Path("/home/sna/nemo-rl-q30-swe"),
            state_dir=state_dir,
            account="coreai_dlalgo_nemorl",
            partition="batch",
            time_limit="04:00:00",
            preflight_record=preflight,
            output_root_validator=lambda **_: None,
            preflight_revalidator=lambda **_: None,
            runner=fake_runner,
        )

    tested = submit.run_scheduler_action(
        mode="test-only",
        plan=plan,
        run=run,
        repo_root=Path("/home/sna/nemo-rl-q30-swe"),
        state_dir=state_dir,
        account="coreai_dlalgo_nemorl",
        partition="batch",
        time_limit="04:00:00",
        preflight_record=preflight,
        output_root_validator=lambda **_: None,
        preflight_revalidator=lambda **_: None,
        runner=fake_runner,
    )
    assert tested["status"] == "test-only-passed"
    assert "--test-only" in calls[-1]
    assert calls[-1].index("--test-only") < calls[-1].index("ray.sub")

    submitted = submit.run_scheduler_action(
        mode="submit",
        plan=plan,
        run=run,
        repo_root=Path("/home/sna/nemo-rl-q30-swe"),
        state_dir=state_dir,
        account="coreai_dlalgo_nemorl",
        partition="batch",
        time_limit="04:00:00",
        preflight_record=preflight,
        output_root_validator=lambda **_: None,
        preflight_revalidator=lambda **_: None,
        runner=fake_runner,
    )
    assert submitted["status"] == "submitted"
    assert submitted["job_id"] == "98765"
    assert "--test-only" not in calls[-1]


def test_scheduler_rejects_plan_not_bound_to_verified_preflight(tmp_path: Path) -> None:
    submit = _load_submit_module()
    plan = _render_plan(tmp_path, "canary")
    run = plan["runs"][0]
    preflight = json.loads((tmp_path / "preflight.json").read_text())
    preflight["source"]["source_commit"] = "f" * 40

    with pytest.raises(submit.ContractError, match="preflight"):
        submit.run_scheduler_action(
            mode="test-only",
            plan=plan,
            run=run,
            repo_root=Path("/home/sna/nemo-rl-q30-swe"),
            state_dir=Path(plan["output_root"]) / "state",
            account="coreai_dlalgo_nemorl",
            partition="batch",
            time_limit="04:00:00",
            preflight_record=preflight,
            output_root_validator=lambda **_: None,
            preflight_revalidator=lambda **_: None,
            runner=lambda *_args, **_kwargs: pytest.fail("scheduler was called"),
        )


def test_preflight_validator_rejects_unverified_evidence(tmp_path: Path) -> None:
    benchmark = _load_benchmark_module()
    manifest = json.loads(MANIFEST.read_text())
    container = Path("/lustre/containers/nemo-rl-pr3733.sqsh")
    preflight = {
        "status": "unverified",
        "manifest_sha256": benchmark._canonical_sha256(manifest),
        "source": {"status": "unverified", "source_commit": SOURCE_COMMIT},
        "artifacts": {
            "status": "unverified",
            "container": {"path": str(container), "sha256": CONTAINER_SHA256},
        },
    }
    preflight["preflight_id"] = benchmark._canonical_sha256(preflight)

    with pytest.raises(benchmark.ContractError, match="preflight status"):
        benchmark._validate_preflight_record(
            preflight=preflight,
            manifest=manifest,
            source_commit=SOURCE_COMMIT,
            container=container,
            container_sha256=CONTAINER_SHA256,
        )


def test_preflight_validator_requires_complete_manifest_evidence(
    tmp_path: Path,
) -> None:
    benchmark = _load_benchmark_module()
    plan = _render_plan(tmp_path, "full")
    manifest = json.loads(MANIFEST.read_text())
    preflight = json.loads((tmp_path / "preflight.json").read_text())
    preflight["artifacts"].pop("target")
    body = {key: value for key, value in preflight.items() if key != "preflight_id"}
    preflight["preflight_id"] = benchmark._canonical_sha256(body)

    with pytest.raises(benchmark.ContractError, match="artifact evidence"):
        benchmark._validate_preflight_record(
            preflight=preflight,
            manifest=manifest,
            source_commit=plan["source_commit"],
            container=Path(plan["container"]["path"]),
            container_sha256=plan["container"]["sha256"],
        )


def test_scheduler_rejects_mutated_run_after_plan_construction(tmp_path: Path) -> None:
    submit = _load_submit_module()
    plan = _render_plan(tmp_path, "canary")
    plan["runs"][0]["command"][-1] = (
        "policy.generation.vllm_kwargs.speculative_config={method:unsafe}"
    )
    run = plan["runs"][0]
    preflight = json.loads((tmp_path / "preflight.json").read_text())

    with pytest.raises(submit.ContractError, match="plan"):
        submit.run_scheduler_action(
            mode="test-only",
            plan=plan,
            run=run,
            repo_root=Path("/home/sna/nemo-rl-q30-swe"),
            state_dir=Path(plan["output_root"]) / "state",
            account="coreai_dlalgo_nemorl",
            partition="batch",
            time_limit="04:00:00",
            preflight_record=preflight,
            output_root_validator=lambda **_: None,
            preflight_revalidator=lambda **_: None,
            runner=lambda *_args, **_kwargs: pytest.fail("scheduler was called"),
        )


def test_scheduler_rejects_noncanonical_state_directory(tmp_path: Path) -> None:
    submit = _load_submit_module()
    plan = _render_plan(tmp_path, "canary")
    preflight = json.loads((tmp_path / "preflight.json").read_text())

    with pytest.raises(submit.ContractError, match="canonical state directory"):
        submit.run_scheduler_action(
            mode="test-only",
            plan=plan,
            run=plan["runs"][0],
            repo_root=Path("/home/sna/nemo-rl-q30-swe"),
            state_dir=tmp_path / "alternate-state",
            account="coreai_dlalgo_nemorl",
            partition="batch",
            time_limit="04:00:00",
            preflight_record=preflight,
            output_root_validator=lambda **_: None,
            preflight_revalidator=lambda **_: None,
            runner=lambda *_args, **_kwargs: pytest.fail("scheduler was called"),
        )


def test_planner_rejects_broad_output_root(tmp_path: Path) -> None:
    benchmark = _load_benchmark_module()
    _render_plan(tmp_path, "full")
    preflight = json.loads((tmp_path / "preflight.json").read_text())

    with pytest.raises(benchmark.ContractError, match="unsafe output root"):
        benchmark.render_plan(
            profile="full",
            source_commit=SOURCE_COMMIT,
            container=Path("/lustre/containers/nemo-rl-pr3733.sqsh"),
            container_sha256=CONTAINER_SHA256,
            output_root=Path("/"),
            preflight=preflight,
        )


def test_scheduler_rejects_output_outside_approved_lustre_prefix(
    tmp_path: Path,
) -> None:
    submit = _load_submit_module()
    plan = _render_plan(tmp_path, "full")

    with pytest.raises(submit.ContractError, match="approved prefix"):
        submit.validate_scheduler_output_root(
            plan=plan, repo_root=Path("/home/sna/nemo-rl-q30-swe")
        )


def test_canary_plan_fails_closed_if_materialized_record_drifts(tmp_path: Path) -> None:
    benchmark = _load_benchmark_module()
    plan = _render_plan(tmp_path, "canary")
    canary_path = Path(plan["data"]["path"])
    canary_path.write_text('{"instance_id":"driftx"}\n')

    with pytest.raises(benchmark.ContractError, match="canary SHA256"):
        benchmark.validate_plan_runtime_files(plan=plan)


def test_monitor_uses_one_filtered_query_per_pass_over_at_least_five_minutes(
    tmp_path: Path,
) -> None:
    submit = _load_submit_module()
    calls: list[list[str]] = []
    sleeps: list[float] = []

    def fake_runner(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(
            command, 0, "123|RUNNING|00:01|None\n124|PENDING|00:00|Resources\n", ""
        )

    result = submit.monitor_jobs(
        job_ids=["123", "124"],
        state_dir=tmp_path,
        campaign_id=CAMPAIGN_ID,
        profile="canary",
        passes=6,
        interval_seconds=60,
        runner=fake_runner,
        sleeper=sleeps.append,
    )

    assert len(calls) == 6
    assert all(call[:3] == ["squeue", "--jobs", "123,124"] for call in calls)
    assert sleeps == [60, 60, 60, 60, 60]
    assert result["monitor_window_seconds"] == 300
    assert len(result["observations"]) == 6
    assert result["campaign_id"] == CAMPAIGN_ID
    assert (tmp_path / f"{CAMPAIGN_ID}__canary.monitor.json").is_file()


def test_full_unlock_requires_campaign_bound_monitor_for_exact_canary_jobs(
    tmp_path: Path,
) -> None:
    benchmark = _load_benchmark_module()
    state = tmp_path / "state"
    for arm, job_id in (("baseline", "123"), ("dflash_k5", "124")):
        reserved = benchmark.reserve_submission(
            state_dir=state,
            campaign_id=CAMPAIGN_ID,
            profile="canary",
            arm_name=arm,
        )
        benchmark.record_job(
            state_dir=state,
            campaign_id=CAMPAIGN_ID,
            profile="canary",
            arm_name=arm,
            reservation_id=reserved["reservation_id"],
            job_id=job_id,
        )
        benchmark.record_completion(
            state_dir=state,
            campaign_id=CAMPAIGN_ID,
            profile="canary",
            arm_name=arm,
            job_id=job_id,
            exit_code=0,
        )

    with pytest.raises(benchmark.ContractError, match="monitor"):
        benchmark.require_successful_canary(state_dir=state, campaign_id=CAMPAIGN_ID)

    benchmark._write_exclusive_json(
        state / f"{CAMPAIGN_ID}__canary.monitor.json",
        {
            "status": "monitored",
            "campaign_id": CAMPAIGN_ID,
            "profile": "canary",
            "job_ids": ["123", "999"],
            "passes": 6,
            "interval_seconds": 60,
            "monitor_window_seconds": 300,
            "observations": [],
        },
    )
    with pytest.raises(benchmark.ContractError, match="monitor job IDs"):
        benchmark.require_successful_canary(state_dir=state, campaign_id=CAMPAIGN_ID)
