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

"""Compare historical recovery tests in one fixed, prepared GPU environment."""

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

GOOD = "249c394db1df268f15c7161d4cc1131c8588310e"
BAD = "6331e5bf6f40910e48ae1e4411c2125ab45882fd"
TEST = "tests/functional/grpo_dp_nixl_tq_recovery.sh"


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def classify(returncode: int, log: str) -> str:
    """Distinguish the target disconnect from other failures and empty runs."""
    if returncode in (124, 137) or returncode < 0:
        return "inconclusive"
    if returncode == 0:
        if (
            "=== Phase 2:" in log
            and "Native TQ checkpoint restored and validated" in log
            and "PASS" in log
            and "[SKIP]" not in log
        ):
            return "good"
    elif (
        "=== Phase 1:" in log
        and "=== Phase 2:" not in log
        and "NIXL_ERR_REMOTE_DISCONNECT" in log
        and "DataPlaneTransferFailed" in log
    ):
        return "bad"
    return "inconclusive"


def locate_boundary(
    commits: list[str], measure: Callable[[str], str]
) -> tuple[str, str]:
    """Locate and recheck a first-parent boundary after endpoint calibration."""
    low, high = 0, len(commits) - 1
    while high - low > 1:
        middle = (low + high) // 2
        outcome = measure(commits[middle])
        if outcome == "good":
            low = middle
        elif outcome == "bad":
            high = middle
        else:
            raise RuntimeError(
                "Inconclusive revision; no introducing commit identified"
            )
    if measure(commits[low]) != "good" or measure(commits[high]) != "bad":
        raise RuntimeError("Boundary did not repeat; no introducing commit identified")
    return commits[low], commits[high]


class Experiment:
    """Own the per-revision source trees, process deadlines and durable results."""

    def __init__(self, repo: Path, output: Path) -> None:
        self.repo = repo
        self.output = output
        self.deadline = time.monotonic() + 5400
        self.records: list[dict] = []
        self.env = dict(os.environ)
        self.env.update(
            UV_NO_SYNC="1",
            UV_FROZEN="1",
            UV_OFFLINE="1",
            UV_PROJECT_ENVIRONMENT=sys.prefix,
            NRL_FORCE_REBUILD_VENVS="false",
            NEMO_RL_VENV_DIR=os.environ.get("NEMO_RL_VENV_DIR", "/opt/ray_venvs"),
            NEMO_RL_REQUIRE_NIXL="1",
            RAY_ADDRESS="local",
            UCX_NET_DEVICES="all",
            UCX_TLS="rc,cuda_copy,cuda_ipc,self,sm",
            UCX_LOG_LEVEL="info",
        )

    def run(self, command: list[str], *, cwd: Path, log: Path, seconds: int) -> int:
        remaining = int(self.deadline - time.monotonic())
        if remaining < seconds + 60 and not command[-2:] == ["stop", "--force"]:
            raise RuntimeError(
                "Time budget exhausted; partial results are inconclusive"
            )
        with log.open("w") as stream:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=self.env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                return process.wait(timeout=seconds)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                return 124

    def stop_ray(self, label: str) -> None:
        status = self.run(
            ["uv", "run", "--no-sync", "ray", "stop", "--force"],
            cwd=self.repo,
            log=self.output / f"{label}-ray-stop.log",
            seconds=45,
        )
        if status:
            raise RuntimeError(
                "Ray cleanup failed; refusing to contaminate the next trial"
            )

    def trial(self, revision: str) -> str:
        number = len(self.records) + 1
        directory = self.output / f"trial-{number:02d}-{revision[:8]}"
        directory.mkdir()
        self.stop_ray(f"before-{number}")
        with tempfile.TemporaryDirectory(prefix="nixl-source-") as temporary:
            tree = Path(temporary)
            archive = directory / "source.tar"
            with archive.open("wb") as stream:
                subprocess.run(
                    ["git", "-C", str(self.repo), "archive", revision],
                    stdout=stream,
                    check=True,
                )
            with tarfile.open(archive) as bundle:
                bundle.extractall(tree, filter="data")
            archive.unlink()
            # Keep compiled dependencies fixed, rather than rebuilding each revision.
            for entry in (self.repo / "3rdparty").iterdir():
                target = tree / "3rdparty" / entry.name
                if entry.is_dir():
                    if target.exists():
                        shutil.rmtree(target)
                    target.symlink_to(entry.resolve(), target_is_directory=True)
            self.env["PYTHONPATH"] = str(tree)
            self.env["NEMO_RL_REPO_DIR"] = str(tree)
            actor_directory = Path(self.env["NEMO_RL_VENV_DIR"])
            for actor in (
                "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker",
                "nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker",
            ):
                if not (actor_directory / actor / "bin/python").is_file():
                    raise RuntimeError(
                        f"Prepared actor environment is missing: {actor}"
                    )
            interpreters = [Path(sys.executable)] + sorted(
                actor_directory.glob("*/bin/python")
            )
            probe = (
                "import hashlib,json,pathlib,nemo_rl; "
                "p=pathlib.Path(nemo_rl.__file__).resolve(); "
                "assert p.parent.parent==pathlib.Path.cwd(),p; "
                "print(json.dumps({'source':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))"
            )
            for index, interpreter in enumerate(interpreters):
                if self.run(
                    [str(interpreter), "-c", probe],
                    cwd=tree,
                    log=directory / f"import-{index}.log",
                    seconds=45,
                ):
                    raise RuntimeError("Historical driver/actor import probe failed")
            print(f"=== revision trial {number}: {revision} ===", flush=True)
            started = time.time()
            try:
                status = self.run(
                    ["uv", "run", "--no-sync", "bash", TEST, "nixl"],
                    cwd=tree,
                    log=directory / "stdout.log",
                    seconds=600,
                )
                log = (directory / "stdout.log").read_text(errors="replace")
                outcome = "inconclusive" if status == 124 else classify(status, log)
                # Native scripts and their assertions come from the tested revision.
                record = dict(
                    trial=number,
                    revision=revision,
                    returncode=status,
                    outcome=outcome,
                    elapsed_seconds=round(time.time() - started, 2),
                    test_sha256=hashlib.sha256((tree / TEST).read_bytes()).hexdigest(),
                )
                self.records.append(record)
                (self.output / "results.json").write_text(
                    json.dumps(self.records, indent=2)
                )
                print(json.dumps(record), flush=True)
                print(log, flush=True)
                native = tree / "tests/functional/grpo_dp_nixl_tq_recovery"
                if native.exists():
                    shutil.copytree(
                        native,
                        directory / "native",
                        ignore=shutil.ignore_patterns(
                            "checkpoints", "*.pt", "*.distcp"
                        ),
                    )
            finally:
                self.stop_ray(f"after-{number}")
        return outcome

    def measure(self, revision: str) -> str:
        outcomes = [self.trial(revision), self.trial(revision)]
        if outcomes[0] != outcomes[1] or "inconclusive" in outcomes:
            raise RuntimeError(
                f"Unstable or unrelated failure at {revision}: {outcomes}"
            )
        return outcomes[0]

    def execute(self) -> None:
        if git(self.repo, "rev-parse", "--is-shallow-repository") != "false":
            raise RuntimeError("Complete Git history is required before GPU trials")
        for revision in [GOOD, BAD]:
            git(self.repo, "cat-file", "-e", f"{revision}^{{commit}}")
        base = git(self.repo, "merge-base", GOOD, BAD)
        commits = [base] + git(
            self.repo, "rev-list", "--reverse", "--first-parent", f"{base}..{BAD}"
        ).splitlines()
        (self.output / "commits.json").write_text(json.dumps(commits, indent=2))
        (self.output / "dependency-changes.txt").write_text(
            git(
                self.repo,
                "diff",
                "--stat",
                base,
                BAD,
                "--",
                "pyproject.toml",
                "uv.lock",
                "docker",
                "3rdparty",
            )
            + "\nFixed image and prepared actor environments are reused; these dependency changes are NOT bisected.\n"
        )
        # The known passing PR head is not an ancestor of the failing main head.
        order = [GOOD, BAD, BAD, GOOD]
        outcomes = [self.trial(revision) for revision in order]
        if outcomes != ["good", "bad", "bad", "good"]:
            raise RuntimeError(
                f"Endpoints did not reproduce good/bad/bad/good: {outcomes}"
            )
        if self.measure(base) != "good":
            raise RuntimeError(
                "Common ancestor is not good; no valid main-branch bracket"
            )
        previous, candidate = locate_boundary(commits, self.measure)
        result = dict(
            previous=previous,
            candidate=candidate,
            scope="first-parent source boundary under this fixed image; not a dependency/image bisect",
        )
        (self.output / "boundary.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    experiment = Experiment(Path.cwd(), args.output.resolve())
    try:
        experiment.execute()
    except (
        RuntimeError,
        subprocess.CalledProcessError,
        OSError,
        tarfile.TarError,
    ) as error:
        (args.output / "inconclusive.txt").write_text(str(error) + "\n")
        print(f"INCONCLUSIVE: {error}", flush=True)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
