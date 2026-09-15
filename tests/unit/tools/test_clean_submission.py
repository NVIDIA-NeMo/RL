import os
import sys
from unittest.mock import patch

import pytest

from tools.super_rl.submit import main, submission_environment


def test_poisoned_environment_is_not_inherited(tmp_path):
    conf = tmp_path / "slurm.conf"
    conf.write_text("ClusterName=test\n")
    caller = {
        "PATH": "/bin",
        "HOME": "/user",
        "SLURM_CPU_BIND": "mask_cpu:0x21",
        "SLURM_JOB_ID": "old",
        "SBATCH_EXPORT": "ALL",
        "PMIX_RANK": "0",
        "SRUN_CPUS_PER_TASK": "2",
        "SLURM_CONF": "/old/config",
        "CONTAINER": "/image",
        "UNREQUESTED_SECRET": "not-a-real-key",
    }
    result = submission_environment(caller, pass_env=["CONTAINER"], slurm_conf=conf)
    assert result == {
        "PATH": "/bin",
        "HOME": "/user",
        "CONTAINER": "/image",
        "SLURM_CONF": str(conf),
    }


@pytest.mark.parametrize(
    "key",
    [
        "SLURM_JOB_ID",
        "SLURM_CONF",
        "SBATCH_EXPORT",
        "SRUN_CPUS_PER_TASK",
        "PMIX_RANK",
        "OMPI_COMM_WORLD_RANK",
    ],
)
def test_cannot_reintroduce_scheduler_state(key):
    with pytest.raises(ValueError):
        submission_environment({key: "poison"}, pass_env=[key], slurm_conf=None)


def test_missing_requested_variable_fails():
    with pytest.raises(KeyError):
        submission_environment({}, pass_env=["CONTAINER"], slurm_conf=None)


@pytest.mark.parametrize("submit", [False, True])
def test_cli_is_test_only_unless_explicit(submit):
    argv = [
        "submit.py",
        *(["--submit"] if submit else []),
        "--",
        "--account=test",
        "run.sbatch",
    ]
    with (
        patch.object(sys, "argv", argv),
        patch.dict(os.environ, {"PATH": "/bin", "SLURM_JOB_ID": "old"}, clear=True),
        patch("tools.super_rl.submit.subprocess.run") as run,
    ):
        main()
    assert run.call_args.args[0] == [
        "sbatch",
        *([] if submit else ["--test-only"]),
        "--account=test",
        "run.sbatch",
    ]
    assert run.call_args.kwargs["env"] == {"PATH": "/bin"}
