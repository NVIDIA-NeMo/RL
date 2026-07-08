import json
import logging
import re
import shlex
import shutil
import subprocess
import threading
import time

from r2egym.agenthub import CMD_TIMEOUT
from r2egym.agenthub.runtime.docker import DockerRuntime, DOCKER_PATH
from r2egym.agenthub.trajectory.swebench_utils import make_test_spec
from r2egym.commit_models.diff_classes import ParsedCommit


##############################################################################
# Local runtime
##############################################################################
class LocalRuntime(DockerRuntime):
    """
    LocalRuntime acts like DockerRuntime, except it runs all commands locally,
    assuming we're already running inside of the required environment.

    This is mainly accomplished by overriding the `run` method.
    The following methods are also overridden: __init__, copy_to_container, close.

    The methods that aren't supported in this setup raise NotImplementedError, e.g. start_container.
    The rest of the methods are inherited from DockerRuntime, e.g. _calculate_reward.
    """

    def __init__(
        self,
        ds,  # dataset entry: defaulting to this (required for all dockers moving forward)
        repo_path: str = "/testbed",  # main repo path
        alt_path: str = "/root",  # used for keeping useful scripts to be hidden from the agent
        logger=None,
        swebench_verified=False,
        swesmith=False,
    ):
        # check if ds is provided (required for all dockers moving forward)
        assert ds, "Dataset not provided"
        self.ds = ds

        self.container_name = "local"  # required by apply_patch

        self.swebench_verified = swebench_verified
        self.swesmith = swesmith
        assert not (self.swebench_verified and self.swesmith)

        if self.swebench_verified:
            # also create a test spec for swebench verified dockers (useful for grading)
            self.test_spec = make_test_spec(self.ds)

        # set runtime params
        self.repo_path = repo_path
        self.alt_path = alt_path
        self.repo_name = (
            self.ds["repo"] if self.swebench_verified or self.swesmith else self.ds["repo_name"]
        )
        if not self.swesmith:
            self.commit_json = (
                self.ds["parsed_commit"]
                if self.swebench_verified
                else self.ds["parsed_commit_content"]
            )
            self.commit = ParsedCommit(**json.loads(self.commit_json))

        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

        # Don't call self.setup_env() here because we call it separately, after the patch has been applied.
        # This is because setup_env() moves some R2E-Gym files around,
        # and we want to apply the patch to the original state of /testbed when the container starts.

    @staticmethod
    def _get_container_name(image_name: str) -> str:
        raise NotImplementedError

    def _start_kubernetes_pod(
        self, docker_image: str, command: str, pod_name: str, **docker_kwargs
    ):
        raise NotImplementedError

    def start_container(
        self, docker_image: str, command: str, ctr_name: str, **docker_kwargs
    ):
        raise NotImplementedError

    def _stop_kubernetes_pod(self):
        raise NotImplementedError

    def stop_container(self):
        raise NotImplementedError

    def _run_kubernetes(
        self,
        code: str,
        timeout: int = CMD_TIMEOUT,
        args: str = "",
        workdir: str = "",
    ) -> tuple[str, str]:
        raise NotImplementedError

    # Based on https://github.com/Kipok/SWE-bench/blob/60171e7869d6d05dd3320176235fa919c955bc25/swebench/harness/run_local_evaluation.py
    # but returns exit code instead of execution duration
    @staticmethod
    def _exec_run_with_timeout(cmd, timeout: int | None = 60):
        """
        Run a command locally with a timeout.

        Args:
            cmd (str): Command to run.
            timeout (int): Timeout in seconds.
        """
        # Local variables to store the result of executing the command
        exec_result = b""
        process = None
        exception = None
        timed_out = False

        # Wrapper function to run the command
        def run_command():
            nonlocal exec_result, process, exception
            try:
                # NeMo-RL fix: the eval harness's own python env leaked into
                # testbed subprocesses (observed: the /testbed py3.8 pytest
                # importing typing_extensions from the mounted harness's
                # py3.12 venv -> AttributeError at pytest startup -> EVERY
                # evaluation scored resolved=false; 323/323 in one batch).
                # Strip interpreter-affecting vars from the child env.
                import os as _os
                _clean_env = {
                    k: v for k, v in _os.environ.items()
                    if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")
                }
                process = subprocess.Popen(
                    cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=False,
                    env=_clean_env,
                )
                exec_result, _ = process.communicate()
            except Exception as e:
                exception = e

        # Start the command in a separate thread
        thread = threading.Thread(target=run_command)
        thread.start()
        thread.join(timeout)

        if exception:
            raise exception

        # If the thread is still alive, the command timed out
        if thread.is_alive():
            if process is not None:
                try:
                    process.terminate()
                    # Give it a moment to terminate gracefully
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            timed_out = True

        return exec_result.decode(), timed_out, process.returncode

    def run(
        self,
        code: str,
        timeout: int = CMD_TIMEOUT,
        args: str = "",
        workdir=None,
        type: str = None,
    ) -> tuple[str, str]:
        """
        General method to execute code or commands in the container, with a timeout.

        :param code: The code or command to execute.
        :param args: Arguments to pass to the code/script.
        :param workdir: The working directory inside the container (optional).
        :return: A tuple containing (output, error_message). If no error, error_message is the exit code (str).
        """
        exec_code = code
        exec_workdir = self.repo_path if workdir is None else workdir

        timeout_cmd = f"timeout {timeout} {exec_code} {args}"
        full_cmd = f"cd {exec_workdir} && PATH={DOCKER_PATH} /bin/sh -c {shlex.quote(timeout_cmd)}"
        try:
            output, timed_out, exit_code = self._exec_run_with_timeout(full_cmd, timeout + 5)

            if timed_out:
                self.logger.error(f"Internal Timeout: {timeout}s. Command: {full_cmd}")
                return f"The command took too long to execute (>{timeout}s)", "-1"

             # Remove ANSI escape codes and \r characters
            output = re.sub(r"\x1b\[[0-9;]*m|\r", "", output)

            if exit_code != 0:
                self.logger.error(f"Command failed with exit code {exit_code}. Command: {full_cmd}. Output:\n{output}")

            return output, str(exit_code)

        except Exception as e:
            return f"Error: {repr(e)}", "-1"

    def demux_run(
        self, code: str, timeout: int = CMD_TIMEOUT, args: str = "", workdir=None
    ) -> tuple[str, str]:
        raise NotImplementedError

    def _copy_to_container_kubernetes(self, src_path: str, dest_path: str):
        raise NotImplementedError

    def copy_to_container(self, src_path: str, dest_path: str):
        # We're already inside of the container, so just do a regular copy
        try:
            shutil.copy2(src_path, dest_path)
        except shutil.SameFileError:
            pass

    def reset(self):
        raise NotImplementedError

    def close(self):
        pass


    # NeMo-RL fix (companion to the env sanitize above): some testbed venvs
    # may carry .pth files referencing the harness mount path from SIF build
    # time; remove them, and strip interpreter vars at the shell level too.
    def run_tests(self, timeout: int = 300) -> tuple[str, str]:
        self.run(
            "grep -ls r2egym_setup /testbed/.venv/lib/*/site-packages/*.pth"
            " 2>/dev/null | xargs -r rm -f"
        )
        output, error_code = self.run(
            f"env -u PYTHONPATH -u PYTHONHOME bash {self.alt_path}/run_tests.sh",
            timeout=timeout,
        )
        output = re.sub(r"\x1b\[[0-9;]*m|\r", "", output)
        return output, error_code
