# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import os
import subprocess
from pathlib import Path
from typing import List, Optional

import pytest

from .types.base_config import NeMoRLTestConfig
from .types.job_dependencies import JobDependencies

# Default job dependencies: validation stage depends on training stage
DEFAULT_JOB_DEPENDENCIES = JobDependencies(
    stages={
        "training": {"depends_on": [], "needs": []},
        "validation": {"depends_on": ["training"], "needs": []},
    },
    job_groups={},
)


class BaseNeMoRLTest:
    """Base test class with common test methods for NeMo-RL tests.

    All test classes should inherit from this class and define a `config` class attribute.
    This provides the standard test_train_local() and test_train_slurm() methods.

    Use with NeMoRLTestConfig for full control over overrides, or use with
    DefaultNeMoRLTestConfig to get standard test overrides automatically applied.

    Pytest Markers:
        - @pytest.mark.runner("local"): Run only local tests
        - @pytest.mark.runner("slurm"): Run only Slurm tests
        - @pytest.mark.stage("stage_name"): Associate test with a pipeline stage
        - @pytest.mark.job_group("group_name"): Associate test with a job group

    Job Dependencies:
        - Define `job_dependencies` class attribute to configure stage and job group
          dependencies for GitLab CI pipeline generation
        - Default: validation stage depends on training stage
        - Override in subclass to customize dependencies

    Usage examples:
        # Run only local tests
        pytest -m 'runner_local' tests/test_suites/tests/

        # Run only Slurm tests
        pytest -m 'runner_slurm' tests/test_suites/tests/

        # Run all tests except Slurm
        pytest -m "not runner_slurm" tests/test_suites/tests/

        # Run a specific test by name
        pytest -k test_train_local tests/test_suites/tests/

    Example with custom config:
        class TestMyTraining(BaseNeMoRLTest):
            config = NeMoRLTestConfig(
                test_name="my-test",
                algorithm="sft",
                model_class="llm",
                test_suites=["nightly"],
                overrides={"trainer.max_steps": 100},
            )

    Example with default test overrides:
        from .base_config import DefaultNeMoRLTestConfig

        class TestMyTraining(BaseNeMoRLTest):
            config = DefaultNeMoRLTestConfig(
                test_name="my-test",
                algorithm="sft",
                model_class="llm",
                test_suites=["nightly"],
            )

    Example overriding defaults:
        from .base_config import DefaultNeMoRLTestConfig

        class TestMyTraining(BaseNeMoRLTest):
            config = DefaultNeMoRLTestConfig(
                test_name="my-test",
                algorithm="sft",
                model_class="llm",
                overrides={
                    "logger.wandb_enabled": False,  # Override default
                    "trainer.max_steps": 100,  # Add custom override
                },
            )

    Example with custom job dependencies:
        from .job_dependencies import JobDependencies

        class TestMyTraining(BaseNeMoRLTest):
            config = NeMoRLTestConfig(...)

            job_dependencies = JobDependencies(
                stages={
                    "training": {"depends_on": [], "needs": []},
                    "validation": {"depends_on": ["training"], "needs": ["train_job"]},
                },
                job_groups={
                    "job_1": {"stage": "validation", "depends_on": [], "needs": []},
                    "job_2": {"stage": "validation", "depends_on": ["job_1"], "needs": ["job_1"]},
                },
            )
    """

    config: NeMoRLTestConfig  # Must be defined by subclass
    job_dependencies: JobDependencies = (
        DEFAULT_JOB_DEPENDENCIES  # Can be overridden by subclass
    )

    def __init_subclass__(cls, **kwargs):
        """Hook to dynamically generate test methods based on config.steps_per_run.

        If config.steps_per_run is set, this removes the original test_train_local
        and test_train_slurm methods and creates numbered versions (e.g.,
        test_train_local_0, test_train_local_1) that train to incremental step counts.
        """
        super().__init_subclass__(**kwargs)

        # Check if config is defined
        if not hasattr(cls, "config"):
            return

        config = cls.config

        # If steps_per_run is not set, keep original behavior
        if config.steps_per_run is None:
            return

        # Calculate number of runs
        num_runs = config.get_num_runs()

        # Remove original test methods if they exist
        if hasattr(cls, "test_train_local"):
            delattr(cls, "test_train_local")
        if hasattr(cls, "test_train_slurm"):
            delattr(cls, "test_train_slurm")

        # Create run-specific test methods
        for run_num in range(num_runs):
            target_steps = config.get_target_steps(run_num)

            # Create local test method
            def make_local_test(run_num, target_steps):
                @pytest.mark.runner("local")
                @pytest.mark.order(run_num)
                @pytest.mark.stage("training")
                def test_method(self):
                    cmd = self.config.build_command(
                        extra_overrides={"trainer.max_steps": target_steps}
                    )
                    return_code = self.run_command(
                        cmd, self.config.run_log_path, cwd=self.config.project_root
                    )
                    assert return_code == 0, (
                        f"Training failed with return code {return_code}. "
                        f"Check {self.config.run_log_path}"
                    )

                return test_method

            # Create slurm test method
            def make_slurm_test(run_num, target_steps):
                @pytest.mark.runner("slurm")
                @pytest.mark.order(run_num)
                @pytest.mark.stage("training")
                def test_method(self, request):
                    return_code = self.run_via_slurm(
                        request.node.nodeid,
                        extra_overrides={"trainer.max_steps": target_steps},
                    )
                    assert return_code == 0, (
                        f"Training failed with return code {return_code}. "
                        f"Check {self.config.run_log_path}"
                    )

                return test_method

            # Set the methods on the class
            setattr(
                cls,
                f"test_train_local_{run_num}",
                make_local_test(run_num, target_steps),
            )
            setattr(
                cls,
                f"test_train_slurm_{run_num}",
                make_slurm_test(run_num, target_steps),
            )

    def run_command(
        self, cmd: List[str], log_file: Path, cwd: Optional[Path] = None
    ) -> int:
        """Run a command and log output to a file.

        Args:
            cmd: Command to run as a list of strings
            log_file: Path to write output to
            cwd: Working directory to run command in

        Returns:
            Return code of the command
        """
        with open(log_file, "w") as f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=cwd,
                bufsize=1,
            )

            # Stream output to both console and file
            if process.stdout:
                for line in process.stdout:
                    print(line, end="")
                    f.write(line)

            return_code = process.wait()

        return return_code

    def run_via_slurm(
        self, pytest_nodeid: str, extra_overrides: Optional[dict] = None
    ) -> int:
        """Run a test via Slurm by invoking launch_nemo_rl.py.

        This method is called when pytest is run with --slurm option.
        It submits the current test to Slurm instead of running locally.

        Args:
            pytest_nodeid: Pytest node ID (e.g., "tests/test_suites/llm/sft-test/test_file.py::TestClass::test_method")
            extra_overrides: Optional dictionary of config overrides to pass to the launch script

        Returns:
            Return code from the Slurm job
        """
        # Get the test folder from the pytest nodeid
        # Example: "tests/test_suites/llm/sft-llama3.1-8b-1n8g-fsdp2tp1-dynamicbatch/test_file.py::..."
        # Extract: "sft-llama3.1-8b-1n8g-fsdp2tp1-dynamicbatch"
        parts = Path(pytest_nodeid).parts
        if "llm" in parts or "vlm" in parts:
            # Find the test folder name (the folder containing the test file)
            test_folder_idx = -1
            for i, part in enumerate(parts):
                if part in ("llm", "vlm") and i + 1 < len(parts):
                    test_folder_idx = i + 1
                    break

            if test_folder_idx != -1:
                test_folder = parts[test_folder_idx]
            else:
                test_folder = self.config.test_name
        else:
            test_folder = self.config.test_name

        # Build launch command
        launch_script = (
            self.config.project_root / "tests" / "test_suites" / "launch_nemo_rl.py"
        )

        launch_cmd = [
            "python",
            str(launch_script),
            "--test-name",
            test_folder,
            "--num-nodes",
            str(self.config.num_nodes),
        ]

        # Add optional environment-based arguments
        env_mappings = {
            "SLURM_ACCOUNT": "--slurm-account",
            "CI_JOB_NAME": "--job-name",
            "TIME": "--job-time",
            "BUILD_IMAGE_NAME_SBATCH": "--image",
            "PARTITION": "--partition",
            "HOST": "--host",
            "RL_USER": "--user",
            "IDENTITY": "--identity",
            "NEMORUN_HOME": "--nemorun-home",
            "CI_JOB_ID": "--ci-job-id",
            "HF_HOME": "--hf-home",
            "HF_DATASETS_CACHE": "--hf-datasets-cache",
            "HF_HUB_OFFLINE": "--hf-hub-offline",
            "RL_HF_TOKEN": "--hf-token",
            "WANDB_API_KEY": "--wandb-api-key",
            "NRL_DEEPSCALER_8K_CKPT": "--nrl-deepscaler-8k-ckpt",
            "NRL_DEEPSCALER_16K_CKPT": "--nrl-deepscaler-16k-ckpt",
            "NRL_DEEPSCALER_24K_CKPT": "--nrl-deepscaler-24k-ckpt",
        }

        for env_var, arg_name in env_mappings.items():
            value = os.environ.get(env_var)
            if value:
                launch_cmd.extend([arg_name, value])

        # Add extra overrides if provided
        if extra_overrides:
            for key, value in extra_overrides.items():
                launch_cmd.extend(["--override", f"{key}={value}"])

        print(f"\n{'=' * 80}")
        print(f"Submitting test to Slurm: {test_folder}")
        print(f"Command: {' '.join(launch_cmd)}")
        print(f"{'=' * 80}\n")

        # Run the launch script
        result = subprocess.run(launch_cmd, cwd=self.config.project_root)
        return result.returncode

    @pytest.mark.runner("local")
    @pytest.mark.stage("training")
    def test_train_local(self):
        """Test that training completes successfully when run locally."""
        cmd = self.config.build_command()
        return_code = self.run_command(
            cmd, self.config.run_log_path, cwd=self.config.project_root
        )

        assert return_code == 0, (
            f"Training failed with return code {return_code}. Check {self.config.run_log_path}"
        )

    @pytest.mark.runner("slurm")
    @pytest.mark.stage("training")
    def test_train_slurm(self, request):
        """Test that training completes successfully when run via Slurm."""
        return_code = self.run_via_slurm(request.node.nodeid)

        assert return_code == 0, (
            f"Training failed with return code {return_code}. Check {self.config.run_log_path}"
        )
