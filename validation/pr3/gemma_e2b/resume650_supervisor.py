# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the authorized Gemma continuation on sequential interactive allocations.

Only Slurm metadata/log/JSON operations execute on this login-node supervisor.
Model execution and numerical analysis remain inside each allocated container.
"""

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SLURM = HERE / "slurm"
OUT = SLURM / "resume650"
TARGET = 650
MAX_ALLOCATIONS = 4
TERMINAL = {
    "COMPLETED",
    "CANCELLED",
    "FAILED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
}
CODEX = "/lustre/fsw/portfolios/coreai/users/yuekaiz/tools/codex/bin/codex"


class HoldRequested(RuntimeError):
    """Stop allocating when the user has requested a hold."""


def command(args: list[str], **kwargs: object) -> subprocess.CompletedProcess:
    return subprocess.run(
        args, cwd=REPO, text=True, capture_output=True, timeout=60, **kwargs
    )


def checkpoint() -> int:
    path = HERE / "results/streaming-la0/checkpoints/latest_checkpoint_status.json"
    return json.loads(path.read_text())["last_checkpoint_step"] if path.exists() else 0


def job_state(job: str) -> str:
    result = command(["sacct", "-X", "-n", "-P", "-j", job, "--format=JobIDRaw,State"])
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            parts = line.split("|")
            if parts[0] == job:
                return parts[1].split()[0].rstrip("+")
    queue = command(["squeue", "-h", "-j", job, "-o", "%T"])
    if queue.returncode == 0 and queue.stdout.strip():
        return queue.stdout.strip().split()[0].rstrip("+")
    raise RuntimeError(f"Cannot determine Slurm state for {job}")


def launcher_alive(pid: int) -> bool:
    try:
        content = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return False
    return content.split(") ", 1)[1].split()[0] != "Z"


def allocation_present(job: str) -> bool:
    result = command(["squeue", "-h", "-u", "yuekaiz", "-o", "%i"])
    result.check_returncode()
    return job in result.stdout.split()


def receipt(path: Path) -> dict:
    if not path.exists():
        return {"step": 0}
    with path.open("rb") as file:
        file.seek(max(0, path.stat().st_size - 200000))
        lines = file.read().decode(errors="replace").splitlines()
    result, pending = {"step": 0}, ""
    for line in lines:
        if "step_metrics=" in line:
            pending = line
        match = re.search(
            r"train step (\d+)/(?:150|650)\s+trainer_v=(\d+)\s+lag=(\S+)", line
        )
        if match:
            metrics = {}
            for key in (
                "gen_kl_error",
                "policy_kl_error",
                "reward",
                "grad_norm",
                "policy_train_dispatches",
                "lr",
            ):
                value = re.search(r"'" + key + r"':\s*([-+0-9.eE]+)", pending)
                if value:
                    metrics[key] = value[1]
            result = {
                "step": int(match[1]),
                "version": int(match[2]),
                "lag": match[3],
                "metrics": metrics,
            }
    return result


class Supervisor:
    def __init__(self, thread: str) -> None:
        self.thread = thread
        self.path = OUT / "state.json"
        self.state = (
            json.loads(self.path.read_text())
            if self.path.exists()
            else {
                "target_steps": TARGET,
                "initial_job": (SLURM / "debug-job-id").read_text().strip(),
                "phase": "WAITING_INITIAL_150",
                "stages": [],
                "notified": [],
            }
        )
        self.last_tick = 0.0
        self.last_progress = -1
        self.progress_at = time.time()
        self.state.update(
            supervisor_pid=os.getpid(),
            thread=thread,
            max_additional_allocations=MAX_ALLOCATIONS,
            supervisor_source_sha256=hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        )

    def save(self) -> None:
        self.state["updated_at"] = time.time()
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state, indent=2) + "\n")
        temporary.replace(self.path)

    def notify(self, key: str, reason: str) -> None:
        if key in self.state["notified"]:
            return
        message = (
            "[已授权任务十分钟监控：Gemma streaming LA0续到650] " + reason + "。"
            "继续当前会话，无需新会话/子agent。用户已授权超过600步，当前目标650，覆盖旧150目标。"
            "先读validation/pr3/gemma_e2b/RESUME650.md、slurm/resume650/state.json和session/20261105_pr3_design/session_state.md最新段。"
            f"工作目录{REPO}。先查现有Slurm/driver/supervisor避免重复启动。保持Gemma4 E2B IT、batch512(32x16)、"
            "LA0/阈值16、4训练+4rollout、镜像1030和同一W&B yuekaiz/nemo-rl-automodel-sc/yuekaiz5090gemmala0s42。"
            "只使用逐次interactive debug allocation直接attach运行，不提batch训练，不并占两节点。"
            "核实checkpoint/optimizer/scheduler/数据游标和恢复后首步、lag0/真实dispatch、KL/W&B连续性。"
            "失败先定位修复再恢复，保留日志；尊重用户后续hold/cancel，HOLD标志禁止新分配。"
            "不动Qwen600/旧900/MCore/另一个RL-pr4963调查，不补baseline。历史Gemma baseline仅157/164步，"
            "650完成后按实际共同区间更新八图和报告、释放本实验节点、推送原分支。"
            "Gemma fixed-input streaming parity未验证，原Qwen1%BF16门禁仍FAILED，不能宣称验收通过。"
        )
        environment = dict(os.environ)
        environment["PATH"] = (
            "/lustre/fsw/portfolios/coreai/users/yuekaiz/tools/node/bin:/lustre/fsw/portfolios/coreai/users/yuekaiz/tools/codex/bin:/usr/local/bin:/usr/bin:/bin"
        )
        try:
            result = command(
                [
                    CODEX,
                    "queue",
                    "--thread",
                    self.thread,
                    "--message",
                    message,
                    "--resume",
                ],
                env=environment,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            # A chat notification failure must not interrupt a healthy allocation.
            with (OUT / "notifications.jsonl").open("a") as file:
                file.write(
                    json.dumps({"time": time.time(), "key": key, "error": str(error)})
                    + "\n"
                )
            return
        with (OUT / "notifications.jsonl").open("a") as file:
            file.write(
                json.dumps(
                    {
                        "time": time.time(),
                        "key": key,
                        "reason": reason,
                        "returncode": result.returncode,
                        "stderr": result.stderr,
                    }
                )
                + "\n"
            )
        if result.returncode == 0:
            self.state["notified"].append(key)
            self.save()

    def tick(self, job: str, log: Path) -> str:
        current = job_state(job)
        progress = receipt(log)
        if progress["step"] != self.last_progress:
            self.last_progress, self.progress_at = progress["step"], time.time()
        self.state.update(
            active_job=job,
            slurm_state=current,
            receipt=progress,
            durable_checkpoint=checkpoint(),
        )
        self.save()
        if time.time() - self.last_tick >= 600:
            self.last_tick = time.time()
            with (OUT / "snapshots.jsonl").open("a") as file:
                file.write(json.dumps(self.state) + "\n")
            for mark in (100, 150, 200, 300, 400, 500, 600, TARGET):
                if progress["step"] >= mark:
                    self.notify(
                        f"step{mark}",
                        f"Gemma已提交{progress['step']}步，checkpoint{checkpoint()}，allocation{job} {current}",
                    )
            if (
                self.state["phase"] == "RUNNING_DRIVER"
                and time.time() - self.progress_at > 1800
            ):
                self.notify(
                    f"stall-{job}-{progress['step']}",
                    f"Gemma allocation{job}已30分钟没有新提交步",
                )
        return current

    def held(self) -> bool:
        if (OUT / "HOLD").exists():
            self.state["phase"] = "HELD_NO_NEW_ALLOCATION"
            self.save()
            self.notify("hold", "续跑HOLD标志存在，未申请或启动新节点；请检查现有作业")
            return True
        return False

    def wait_initial(self) -> None:
        job = self.state["initial_job"]
        while True:
            if self.held():
                return
            current = self.tick(job, SLURM / "driver.log")
            exit_path = SLURM / "driver-exit-code"
            if exit_path.exists() and exit_path.read_text().strip() != "0":
                raise RuntimeError(
                    "Initial150 driver failed; inspect its logs before resuming"
                )
            if current in TERMINAL:
                assert exit_path.exists() and exit_path.read_text().strip() == "0"
                assert checkpoint() >= 150
                assert (HERE / "results/curve-comparison/comparison.png").exists()
                if not allocation_present(job):
                    return
            time.sleep(30)

    def acquire(self, index: int) -> dict:
        if self.held():
            raise HoldRequested
        name = f"yuekaiz-gemma650-debug-{index}"
        existing = command(["squeue", "-h", "-u", "yuekaiz", "-n", name, "-o", "%i"])
        existing.check_returncode()
        jobs = existing.stdout.split()
        assert len(jobs) <= 1, "Ambiguous existing allocations; refusing a duplicate"
        if jobs:
            job = jobs[0]
            info = command(["scontrol", "show", "job", job, "--oneliner"])
            info.check_returncode()
            assert "Comment=yuekaiz5090gemmala0s42-resume650" in info.stdout
            directory = re.search(r"\bWorkDir=(\S+)", info.stdout)
            assert directory and Path(directory[1]).resolve() == REPO.resolve()
        else:
            if self.held():
                raise HoldRequested
            environment = dict(os.environ)
            environment.update(
                CONTAINER="/lustre/fsw/portfolios/coreai/users/yuekaiz/containers/nemo_rl.1030.sqsh",
                BASE_LOG_DIR=str(SLURM),
                MOUNTS="/lustre:/lustre",
                COMMAND="",
                CPUS_PER_WORKER="128",
            )
            submitted = command(
                [
                    "sbatch",
                    "--parsable",
                    "--nodes=1",
                    "--account=coreai_dlalgo_nemorl",
                    "--partition=interactive",
                    "--time=4:00:00",
                    "--gres=gpu:8",
                    f"--job-name={name}",
                    "--comment=yuekaiz5090gemmala0s42-resume650",
                    f"--output={SLURM}/slurm-%j.out",
                    f"--error={SLURM}/slurm-%j.err",
                    str(REPO / "ray.sub"),
                ],
                env=environment,
            )
            submitted.check_returncode()
            job = submitted.stdout.strip().split(";")[0]
        assert job.isdigit(), job
        stage = {
            "job_id": job,
            "index": index,
            "resume_from": checkpoint(),
            "phase": "WAITING_CONTAINER",
        }
        self.state["stages"].append(stage)
        self.state["phase"] = "WAITING_CONTAINER"
        self.save()
        self.notify(
            f"allocation-{job}",
            f"已申请Gemma续跑debug allocation{job}，计划恢复checkpoint{stage['resume_from']}",
        )
        return stage

    def run_stage(self, stage: dict) -> None:
        self.state["phase"] = stage["phase"]
        self.save()
        job = stage["job_id"]
        log = SLURM / f"resume650-driver-{job}.log"
        exit_path = SLURM / f"resume650-exit-{job}"
        if stage["phase"] == "WAITING_CONTAINER":
            while (
                not (SLURM / f"{job}-logs/STARTED_RAY_HEAD").exists()
                or not (REPO / f"{job}-attach.sh").exists()
            ):
                if self.held():
                    return
                assert self.tick(job, log) not in TERMINAL, (
                    f"Allocation{job} ended before container startup"
                )
                time.sleep(15)
            if self.held():
                return
            with (SLURM / f"resume650-launcher-{job}.log").open("a") as file:
                process = subprocess.Popen(
                    ["bash", str(HERE / "launch_resume650.sh"), job],
                    cwd=REPO,
                    stdout=file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            stage.update(phase="RUNNING_DRIVER", launcher_pid=process.pid)
            self.state["phase"] = "RUNNING_DRIVER"
            self.progress_at = time.time()
            self.save()
        while not exit_path.exists():
            current = self.tick(job, log)
            assert current not in TERMINAL, f"Allocation{job} ended before driver exit"
            if not launcher_alive(stage["launcher_pid"]):
                if exit_path.exists():
                    break
                raise RuntimeError(
                    f"Launcher{stage['launcher_pid']} exited without a driver status; inspect launcher log"
                )
            progress = receipt(log)
            if progress["step"] > stage["resume_from"]:
                assert (
                    progress["version"] == progress["step"] and progress["lag"] == "0"
                )
                if "first_resumed_step" not in stage:
                    first = re.search(
                        r"train step (\d+)/650\s+trainer_v=(\d+)\s+lag=(\d+)",
                        log.read_text(errors="replace"),
                    )
                    assert (
                        first
                        and int(first[1]) == int(first[2]) == stage["resume_from"] + 1
                        and first[3] == "0"
                    )
                    stage["first_resumed_step"] = int(first[1])
                    self.save()
                self.notify(
                    f"restored-{job}",
                    f"Gemma checkpoint{stage['resume_from']}恢复后已提交{progress['step']}，version一致/lag0",
                )
            time.sleep(30)
        stage["driver_exit"] = exit_path.read_text().strip()
        self.save()
        assert stage["driver_exit"] == "0", (
            f"Gemma driver failed in allocation{job}; inspect{log}"
        )
        durable = checkpoint()
        assert stage["resume_from"] < durable <= TARGET, "No durable progress"
        assert (HERE / f"results/checkpoint{durable}-state.json").exists()
        if durable == TARGET:
            summary = json.loads(
                (HERE / "results/curve-comparison/comparison-summary.json").read_text()
            )
            assert summary["max_plot_step"] == TARGET
        # Only allocations recorded by THIS authorized supervisor are released.
        if job_state(job) not in TERMINAL:
            released = command(["scancel", job])
            released.check_returncode()
        while job_state(job) not in TERMINAL or allocation_present(job):
            time.sleep(5)
        stage.update(phase="COMPLETED", durable_checkpoint=durable)
        self.state["phase"] = "WAITING_NEXT_SEGMENT"
        self.save()
        self.notify(
            f"segment-{job}",
            f"Gemma续跑段{job}正常结束，checkpoint{durable}，本段debug节点已释放",
        )

    def run(self) -> None:
        if not self.state["stages"]:
            self.wait_initial()
        while checkpoint() < TARGET and not self.held():
            stages = self.state["stages"]
            if stages and stages[-1]["phase"] != "COMPLETED":
                stage = stages[-1]
            else:
                assert len(stages) < MAX_ALLOCATIONS, (
                    "Debug segment budget exhausted; inspect progress"
                )
                stage = self.acquire(len(stages) + 1)
            self.run_stage(stage)
        if checkpoint() == TARGET:
            self.state["phase"] = "COMPLETE_COMPUTE_PENDING_REVIEW"
            self.save()
            self.notify(
                "completed650",
                "Gemma已完成650步、最终checkpoint保存状态检查和八图，本实验debug节点已释放；请核实W&B并更新发布报告",
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread", required=True)
    args = parser.parse_args()
    assert os.environ["USER"] == "yuekaiz"
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / "supervisor.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    supervisor = Supervisor(args.thread)
    supervisor.save()
    try:
        supervisor.run()
    except HoldRequested:
        return
    except Exception as error:
        # Process boundary: persist and notify, then fail loudly. No blind retry.
        supervisor.state.update(
            phase="FAILED_NEEDS_DIAGNOSIS", error=f"{type(error).__name__}: {error}"
        )
        supervisor.save()
        supervisor.notify(f"failure-{time.time()}", supervisor.state["error"])
        raise


if __name__ == "__main__":
    main()
