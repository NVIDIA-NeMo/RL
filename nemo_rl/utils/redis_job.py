"""Optional Redis instance.

This backs nemo_rl.utils.rate_limiter, colocated on this training job's own head node as
a direct child process of this Python driver (no SLURM job step involved at all).

This runs Redis as a plain `subprocess.Popen(["singularity", "exec", ...])` launched
directly from this Python process, rather than wrapping it in its own `srun` step or
submitting a separate SLURM job. Three reasons:
  1. No new node allocation is requested -- Redis is lightweight enough to just live on
     the head node's already-idle capacity
  2. Lifecycle is automatically coupled to the training job's: this process is a direct
     child of the Python driver, and dies with it
  3. No `srun` step needed at all, so no SLURM quota/queue involved.

Exports the discovered host into os.environ.
"""

import atexit
import logging
import os
import socket
import subprocess
import time
from typing import Optional, TypedDict

logger = logging.getLogger(__name__)


class RedisJobConfig(TypedDict):
    enabled: bool
    redis_dir: str
    port: int
    startup_timeout_s: int
    poll_interval_s: float
    host_env_var: str


DEFAULT_REDIS_JOB_CONFIG: RedisJobConfig = {
    "enabled": False,
    # Dir containing redis.sif / redis.conf.
    "redis_dir": "/leonardo_scratch/fast/BOOST_EUROPA/images/redis",
    "port": 6380,  # must match <redis_dir>/redis.conf
    "startup_timeout_s": 60,
    "poll_interval_s": 0.5,
    "host_env_var": "NEMO_RL_REDIS_HOST",
}


class RedisJob:
    """Owns the background Redis process launched as a direct child of this driver.

    Call `stop()` to terminate it; also registered via `atexit`. Even without an explicit
    call, SLURM's cgroup-based job teardown kills this process along with the rest of this
    job's process tree when the job itself ends -- it's a genuine child process of the driver,
    which is itself already part of this job's process tree.
    """

    def __init__(self, process: subprocess.Popen, host: str, port: int):
        self._process: Optional[subprocess.Popen] = process
        self.host = host
        self.port = port
        atexit.register(self.stop)

    def stop(self) -> None:
        if self._process is None:
            return
        logger.info("Terminating colocated Redis process (pid %s)", self._process.pid)
        self._process.terminate()
        self._process = None


def maybe_start_redis_job(config: dict) -> Optional[RedisJob]:
    """Launch Redis as a direct child process on this training job's own head node, if
    `config["redis"]["enabled"]`.

    Returns None (no-op) if the `redis` block is absent or `enabled` is false. Must be called
    from inside a running SLURM job (relies on `SLURM_JOB_ID` purely for per-job data-dir/log
    naming) -- no node discovery or srun step is needed, since this process is already running
    on the node Redis should live on.
    """
    redis_config: RedisJobConfig = {**DEFAULT_REDIS_JOB_CONFIG, **config.get("redis", {})}
    if not redis_config["enabled"]:
        return None

    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if not slurm_job_id:
        raise RuntimeError(
            "redis.enabled=true requires SLURM_JOB_ID in the environment -- this colocates "
            "Redis on the calling process's own node, so it must run inside a SLURM job."
        )

    host = socket.gethostname()
    redis_dir = redis_config["redis_dir"]
    # Per-job data dir: unlike the old shared-job design, each training job now gets its own
    # colocated Redis instance, so a shared data dir would let two concurrent jobs'
    # redis-server processes race on the same files.
    data_dir = os.path.join(redis_dir, f"redis-data-{slurm_job_id}")
    os.makedirs(data_dir, exist_ok=True)
    log_dir = os.path.join(redis_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # redis.conf must be bound explicitly too -- singularity's default binds don't cover
    # arbitrary host paths under /leonardo_scratch, so passing the bare host path to
    # redis-server (without binding its containing dir) fails with "can't open config file
    # ... No such file or directory" even though the path exists on the host. Bind it read-only
    # to a fixed in-container path and reference THAT path in the redis-server argv instead.
    #
    # No srun wrapper: this process is already running on the target node, inside this job's
    # SLURM cgroup -- a bare child process inherits that confinement automatically. See the
    # module docstring for why an earlier srun --overlap-based design was dropped.
    cmd = [
        "singularity",
        "exec",
        "--writable-tmpfs",
        "--bind",
        f"{data_dir}:/data",
        "--bind",
        f"{os.path.join(redis_dir, 'redis.conf')}:/etc/redis.conf:ro",
        os.path.join(redis_dir, "redis.sif"),
        "redis-server",
        "/etc/redis.conf",
    ]
    logger.info("Starting colocated Redis on %s: %s", host, " ".join(cmd))

    stdout_path = os.path.join(log_dir, f"redis.{slurm_job_id}.out")
    stderr_path = os.path.join(log_dir, f"redis.{slurm_job_id}.err")
    process = subprocess.Popen(  # noqa: S603 (fixed argv, no shell)
        cmd,
        stdout=open(stdout_path, "w"),
        stderr=open(stderr_path, "w"),
    )
    _wait_for_redis_ready(
        host,
        redis_config["port"],
        process,
        redis_config["startup_timeout_s"],
        redis_config["poll_interval_s"],
    )

    os.environ[redis_config["host_env_var"]] = host
    logger.info(
        "Redis is up on %s:%d (pid %s) -- exported %s for downstream Gym servers.",
        host,
        redis_config["port"],
        process.pid,
        redis_config["host_env_var"],
    )

    return RedisJob(process=process, host=host, port=redis_config["port"])


def _wait_for_redis_ready(
    host: str,
    port: int,
    process: subprocess.Popen,
    timeout_s: float,
    poll_interval_s: float,
) -> None:
    """Poll until Redis accepts TCP connections, or the process dies trying."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Redis process died during startup (exit code: {process.returncode}). "
                "Check <redis_dir>/logs/redis.<job_id>.err for details."
            )
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError:
            time.sleep(poll_interval_s)
    raise TimeoutError(f"Redis on {host}:{port} did not become reachable within {timeout_s}s.")
