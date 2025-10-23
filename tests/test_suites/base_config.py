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

import inspect
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf

from nemo_rl.utils.config import load_config, parse_hydra_overrides


@dataclass
class NeMoRLTestConfig:  # TODO(ahmadki): use native policy dicts ?
    """Test configuration with YAML base + overrides + test metadata.

    This class:
    1. References a YAML config file as the base training configuration
    2. Allows overriding specific YAML values for testing
    3. Stores test-specific metadata (test_name, algorithm, test_suites)
    4. Auto-extracts metadata from YAML for filtering/classification
    """

    #######################################################
    # Metadata we add to tests (required fields first)
    #######################################################
    test_name: str  # Test identifier
    algorithm: str  # sft, grpo, dpo
    model_class: str  # llm, vlm
    test_suites: List[str] = field(
        default_factory=lambda: ["nightly"]
    )  # Suites this test is part of (can set multiple suites)
    time_limit_minutes: int = 120  # Test timeout, used for slurms jobs

    #######################################################
    # Model config
    #######################################################
    # The model hydra/YAML config (auto-derived from test_name if None)
    yaml_config: Optional[Path] = (
        None  # The full config path, default to "examples/configs/recipes/{model_class}/{test_name}.yaml"
    )
    overrides: Dict[str, Any] = field(default_factory=dict)

    #######################################################
    # Run paths  # TODO(ahmadki): extract from yaml !!
    #######################################################
    exp_dir: Optional[Path] = (
        None  # Base experiment directory, default to test file directory
    )
    log_dir: Optional[Path] = None  # Log directory, default to {exp_dir}/logs
    ckpt_dir: Optional[Path] = None  # Checkpoint directory, default to {exp_dir}/ckpts
    run_log_path: Optional[Path] = (
        None  # Path to store run log, default to {exp_dir}/run.log
    )

    #######################################################
    # Computed fields  # TODO(ahmadki): make private ?
    #######################################################
    # These fields are extracted from meta data, config yaml + overrides
    loaded_yaml_config: DictConfig = field(init=False, repr=False, default=None)
    model_name: str = field(init=False, default="")
    backend: str = field(init=False, default="fsdp2")
    num_nodes: int = field(init=False, default=1)
    num_gpus_per_node: int = field(init=False, default=8)
    tensor_parallel: Optional[int] = field(init=False, default=None)
    pipeline_parallel: Optional[int] = field(init=False, default=None)
    sequence_parallel: bool = field(init=False, default=False)

    def __post_init__(self):
        # Get project root from git using this file folder (test_suites) as starting location
        self.project_root = Path(
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=Path(__file__).parent,
                text=True,
            ).strip()
        )

        # Get the file path of the test file that instantiated this config
        caller_frame = None
        for frame_info in inspect.stack():
            frame_file = Path(frame_info.filename)
            if frame_file != Path(__file__) and "test_suites" in str(frame_file):
                caller_frame = frame_file
                break
        self.test_file_dir = (
            caller_frame.parent if caller_frame else Path(__file__).parent
        )

        # Compute the yaml config path if not provided
        if self.yaml_config is None:
            self.yaml_config = (
                self.project_root
                / "examples"
                / "configs"
                / "recipes"
                / self.model_class
                / f"{self.test_name}.yaml"
            )
            if not self.yaml_config.exists():
                raise FileNotFoundError(
                    f"Config file not found: {self.config_path}. "
                    f"Expected to find it at the path derived from test name '{self.test_name}'"
                )
        else:
            if not self.yaml_config.exists():
                raise FileNotFoundError(f"Config file not found: {self.yaml_config}")

        # Loads YAML files with OmegaConf
        self.loaded_yaml_config = load_config(self.yaml_config)

        # Apply overrides with Hydra's parser
        if self.overrides:
            override_strings = [f"{k}={v}" for k, v in self.overrides.items()]
            self.loaded_yaml_config = parse_hydra_overrides(
                self.loaded_yaml_config, override_strings
            )
        # Extract metadata from YAML (after applying overrides)
        self._extract_metadata()

        # Validate configuration
        self.validate_config()

        # Test directories
        if self.exp_dir is None:
            self.exp_dir = self.test_file_dir

        if self.log_dir is None:
            self.log_dir = self.exp_dir / "logs"

        if self.ckpt_dir is None:
            self.ckpt_dir = self.exp_dir / "ckpts"

        if self.run_log_path is None:
            self.run_log_path = self.exp_dir / "run.log"

        # Create directories  # TODO(ahmadki): move to run funtions
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _extract_metadata(self):
        self.model_name = self._extract_model_name()
        self.backend = self._detect_backend()
        self.num_nodes = OmegaConf.select(self.loaded_yaml_config, "cluster.num_nodes")
        self.num_gpus_per_node = OmegaConf.select(
            self.loaded_yaml_config, "cluster.gpus_per_node"
        )
        self._extract_parallelism()

    def _extract_model_name(self) -> str:
        model_name = OmegaConf.select(
            self.loaded_yaml_config, "policy.model_name", default=""
        )

        # Clean up model name for filtering (remove org prefix)
        # e.g., "meta-llama/Llama-3.1-8B-Instruct" -> "llama3.1-8b-instruct"
        if "/" in model_name:
            model_name = model_name.split("/")[-1]

        model_name = model_name.lower().replace("-", "").replace("_", "")
        return model_name

    def _detect_backend(self) -> str:
        dtensor_enabled = OmegaConf.select(
            self.loaded_yaml_config, "policy.dtensor_cfg.enabled", default=False
        )
        megatron_enabled = OmegaConf.select(
            self.loaded_yaml_config, "policy.megatron_cfg.enabled", default=False
        )

        if dtensor_enabled and megatron_enabled:
            raise ValueError(
                "Both dtensor and megatron backends are enabled in the config. "
                "Please enable only one."
            )
        if dtensor_enabled:
            return "dtensor"
        if megatron_enabled:
            return "megatron"
        # Default to fsdp2
        return "fsdp2"

    def _extract_parallelism(self):
        backend = self._detect_backend()
        if backend == "dtensor":
            self.tensor_parallel = OmegaConf.select(
                self.loaded_yaml_config,
                "policy.dtensor_cfg.tensor_parallel_size",
                default=None,
            )
            self.pipeline_parallel = OmegaConf.select(
                self.loaded_yaml_config,
                "policy.dtensor_cfg.pipeline_parallel_size",
                default=None,
            )
            self.sequence_parallel = OmegaConf.select(
                self.loaded_yaml_config,
                "policy.dtensor_cfg.sequence_parallel",
                default=False,
            )
        elif backend == "megatron":
            # If not found in dtensor_cfg, try megatron_cfg
            if self.tensor_parallel is None:
                self.tensor_parallel = OmegaConf.select(
                    self.loaded_yaml_config,
                    "policy.megatron_cfg.tensor_model_parallel_size",
                    default=None,
                )
            if self.pipeline_parallel is None:
                self.pipeline_parallel = OmegaConf.select(
                    self.loaded_yaml_config,
                    "policy.megatron_cfg.pipeline_model_parallel_size",
                    default=None,
                )
            if not self.sequence_parallel:
                self.sequence_parallel = OmegaConf.select(
                    self.loaded_yaml_config,
                    "policy.megatron_cfg.sequence_parallel",
                    default=False,
                )

    def validate_config(self):
        """Validate the loaded configuration.

        This method can be extended to add more validation checks.
        Currently provides basic warnings for common issues.
        """
        # Check if critical paths exist
        if self.num_nodes < 1:
            raise ValueError(f"num_nodes must be >= 1, got {self.num_nodes}")

        if self.num_gpus_per_node < 1:
            raise ValueError(
                f"num_gpus_per_node must be >= 1, got {self.num_gpus_per_node}"
            )

        # Warn about tensor/pipeline parallel without proper GPU count
        if self.tensor_parallel and self.tensor_parallel > self.num_gpus_total:
            print(
                f"Warning: tensor_parallel_size ({self.tensor_parallel}) > "
                f"total GPUs ({self.num_gpus_total})"
            )

    @property
    def num_gpus_total(self) -> int:
        """Total number of GPUs across all nodes."""
        return self.num_nodes * self.num_gpus_per_node

    def get_run_script_path(self) -> Path:
        """Get the path to the main run script based on algorithm."""
        script_map = {
            "sft": "run_sft.py",
            "grpo": "run_grpo_math.py",
            "dpo": "run_dpo.py",
            "vlm_grpo": "run_vlm_grpo.py",
        }
        script_name = script_map.get(self.algorithm)
        if script_name is None:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        return self.project_root / "examples" / script_name

    def build_command(self) -> List[str]:
        """Build the uv run command with YAML config + overrides.

        Overrides are applied in Hydra format (key=value).
        The command structure is compatible with Hydra's override system.
        """
        cmd = [
            "uv",
            "run",
            "--no-sync",
            str(self.get_run_script_path()),
            "--config",
            str(self.config_path),
        ]

        # Apply overrides from self.overrides dict
        for key, value in self.overrides.items():
            cmd.append(f"{key}={value}")

        # Apply test-specific overrides (logs and checkpoints)
        # Using a dict makes it easier to maintain and see all overrides
        test_overrides = {
            "logger.log_dir": str(self.log_dir),
            "logger.wandb_enabled": True,
            "logger.wandb.project": "nemo-rl",
            "logger.wandb.name": self.test_name,
            "logger.monitor_gpus": True,
            "logger.tensorboard_enabled": True,
            "checkpointing.enabled": True,
            "checkpointing.checkpoint_dir": str(self.ckpt_dir),
        }

        for key, value in test_overrides.items():
            cmd.append(f"{key}={value}")

        return cmd

    def get_ci_job_id(self) -> str:
        """Get CI job ID from environment or use test name."""
        return os.environ.get("CI_JOB_ID", "local")


class BaseNeMoRLTest:  # TODO(ahmadki): to a different file, then create a default test with folder overrides
    """Base test class with common test methods for NeMo-RL tests.

    All test classes should inherit from this class and define a `config` class attribute.
    This provides the standard test_training_runs_successfully_local() and
    test_training_runs_successfully_slurm() methods.

    Example:
        class TestMyTraining(BaseNeMoRLTest):
            config = NeMoRLTestConfig(
                test_name="my-test",
                algorithm="sft",
                test_suites=["nightly"],
                time_limit_minutes=120,
            )
    """

    config: NeMoRLTestConfig  # Must be defined by subclass

    def test_train_local(self):
        """Test that training completes successfully when run locally."""
        cmd = self.config.build_command()
        return_code = run_command(
            cmd, self.config.run_log_path, cwd=self.config.project_root
        )

        assert return_code == 0, (
            f"Training failed with return code {return_code}. Check {self.config.run_log_path}"
        )

    def test_train_slurm(self, request):
        """Test that training completes successfully when run via Slurm."""
        return_code = run_via_slurm(self.config, request.node.nodeid)

        assert return_code == 0, (
            f"Training failed with return code {return_code}. Check {self.config.run_log_path}"
        )


# TODO(ahmadki): make part of BaseNeMoRLTest ?
def run_command(cmd: List[str], log_file: Path, cwd: Optional[Path] = None) -> int:
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


def run_via_slurm(config: NeMoRLTestConfig, pytest_nodeid: str) -> int:
    """Run a test via Slurm by invoking launch_nemo_rl.py.

    This function is called when pytest is run with --slurm option.
    It submits the current test to Slurm instead of running locally.

    Args:
        config: Test configuration
        pytest_nodeid: Pytest node ID (e.g., "tests/test_suites/llm/sft-test/test_file.py::TestClass::test_method")

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
            test_folder = config.test_name
    else:
        test_folder = config.test_name

    # Build launch command
    launch_script = config.project_root / "tests" / "test_suites" / "launch_nemo_rl.py"

    launch_cmd = [
        "python",
        str(launch_script),
        "--test-name",
        test_folder,
        "--num-nodes",
        str(config.num_nodes),
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

    print(f"\n{'=' * 80}")
    print(f"Submitting test to Slurm: {test_folder}")
    print(f"Command: {' '.join(launch_cmd)}")
    print(f"{'=' * 80}\n")

    # Run the launch script
    result = subprocess.run(launch_cmd, cwd=config.project_root)
    return result.returncode
