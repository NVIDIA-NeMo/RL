import os
import json
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


@pytest.mark.parametrize("submit", [False, True])
def test_bootstrap_grace_is_an_outer_sbatch_comment(submit):
    argv = [
        "submit.py",
        *(["--submit"] if submit else []),
        "--bootstrap-grace-minutes",
        "75",
        "--",
        "--account=test",
        "run.sbatch",
    ]
    with (
        patch.object(sys, "argv", argv),
        patch("tools.super_rl.submit.subprocess.run") as run,
    ):
        main()
    command = run.call_args.args[0]
    assert command[0] == "sbatch"
    assert ("--test-only" in command) is not submit
    comment = next(
        arg.removeprefix("--comment=")
        for arg in command
        if arg.startswith("--comment=")
    )
    assert json.loads(comment) == {
        "OccupiedIdleGPUsJobReaper": {
            "exemptIdleTimeMins": "75",
            "reason": "other",
            "description": "nemo-rl-run-bootstrap",
        }
    }
    assert command[-2:] == ["--account=test", "run.sbatch"]


@pytest.mark.parametrize("minutes", ["0", "-1", "not-a-number"])
def test_invalid_grace_never_calls_sbatch(minutes):
    argv = ["submit.py", "--bootstrap-grace-minutes", minutes, "--", "run.sbatch"]
    with (
        patch.object(sys, "argv", argv),
        patch("tools.super_rl.submit.subprocess.run") as run,
    ):
        with pytest.raises(SystemExit) as error:
            main()
    assert error.value.code == 2
    run.assert_not_called()


@pytest.mark.parametrize("comment", [["--comment=existing"], ["--comment", "existing"]])
def test_conflicting_first_component_comment_fails(comment):
    argv = [
        "submit.py",
        "--bootstrap-grace-minutes",
        "75",
        "--",
        *comment,
        "run.sbatch",
    ]
    with (
        patch.object(sys, "argv", argv),
        patch("tools.super_rl.submit.subprocess.run") as run,
    ):
        with pytest.raises(SystemExit):
            main()
    run.assert_not_called()


def test_bootstrap_grace_does_not_change_service_component_comment():
    batch_args = ["--nodes=64", ":", "--nodes=1", "--comment=service", "run.sbatch"]
    argv = ["submit.py", "--bootstrap-grace-minutes", "75", "--", *batch_args]
    with (
        patch.object(sys, "argv", argv),
        patch("tools.super_rl.submit.subprocess.run") as run,
    ):
        main()
    command = run.call_args.args[0]
    assert command[-len(batch_args) :] == batch_args
    assert command[2].startswith("--comment=")
