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
import math
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from omegaconf import DictConfig, OmegaConf

from nemo_rl.models.policy import DTensorConfig, MegatronConfig, PolicyConfig
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
    steps_per_run: Optional[int] = (
        None  # If set, split training into multiple runs of this many steps
    )

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
    max_steps: int = field(init=False, default=0)

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
        self.max_steps = OmegaConf.select(
            self.loaded_yaml_config, "trainer.max_steps", default=0
        )

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

        This method validates:
        - Basic cluster configuration (nodes, GPUs)
        - Policy configuration structure (using PolicyConfig types)
        - Parallelism settings match GPU counts
        - Required fields are present
        """
        # Check if critical paths exist
        if self.num_nodes < 1:
            raise ValueError(f"num_nodes must be >= 1, got {self.num_nodes}")

        if self.num_gpus_per_node < 1:
            raise ValueError(
                f"num_gpus_per_node must be >= 1, got {self.num_gpus_per_node}"
            )

        # Validate policy configuration exists
        if not self.policy_config:
            raise ValueError("Policy configuration is missing from YAML")

        # Validate required policy fields
        required_policy_fields = ["model_name", "tokenizer", "dtensor_cfg"]
        for field_name in required_policy_fields:
            if field_name not in self.policy_config:
                raise ValueError(f"Required policy field '{field_name}' is missing")

        # Validate backend configuration (exactly one should be enabled)
        dtensor_cfg = self.policy_config.get("dtensor_cfg", {})
        megatron_cfg = self.policy_config.get("megatron_cfg", {})

        dtensor_enabled = dtensor_cfg.get("enabled", False)
        megatron_enabled = megatron_cfg.get("enabled", False)

        if dtensor_enabled and megatron_enabled:
            raise ValueError(
                "Both dtensor and megatron backends are enabled. "
                "Please enable only one backend."
            )

        # Warn about tensor/pipeline parallel without proper GPU count
        if self.tensor_parallel and self.tensor_parallel > self.num_gpus_total:
            print(
                f"Warning: tensor_parallel_size ({self.tensor_parallel}) > "
                f"total GPUs ({self.num_gpus_total})"
            )

        # Validate steps_per_run if set
        if self.steps_per_run is not None:
            if self.steps_per_run <= 0:
                raise ValueError(f"steps_per_run must be > 0, got {self.steps_per_run}")

            if self.max_steps == 0:
                raise ValueError(
                    "steps_per_run is set but max_steps could not be extracted from config. "
                    "Ensure trainer.max_steps is defined in the YAML config."
                )

            if self.steps_per_run > self.max_steps:
                raise ValueError(
                    f"steps_per_run ({self.steps_per_run}) must be <= max_steps ({self.max_steps})"
                )

            # Warn if steps_per_run doesn't evenly divide max_steps
            if self.max_steps % self.steps_per_run != 0:
                print(
                    f"Warning: max_steps ({self.max_steps}) is not evenly divisible by "
                    f"steps_per_run ({self.steps_per_run}). Last run will train for "
                    f"{self.max_steps % self.steps_per_run} steps."
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
            "dapo": "run_grpo_math.py",
        }
        script_name = script_map.get(self.algorithm)
        if script_name is None:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        return self.project_root / "examples" / script_name

    def build_command(
        self, extra_overrides: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Build the uv run command with YAML config + overrides.

        Overrides are applied in Hydra format (key=value).
        The command structure is compatible with Hydra's override system.

        Args:
            extra_overrides: Optional dictionary of additional overrides to apply
                           after self.overrides. This allows callers to add
                           test-specific overrides without modifying the config.

        Returns:
            Command as a list of strings ready for subprocess execution
        """
        cmd = [
            "uv",
            "run",
            "--no-sync",
            str(self.get_run_script_path()),
            "--config",
            str(self.yaml_config),
        ]

        # Apply overrides from self.overrides dict
        for key, value in self.overrides.items():
            cmd.append(f"{key}={value}")

        # Apply extra overrides if provided
        if extra_overrides:
            for key, value in extra_overrides.items():
                cmd.append(f"{key}={value}")

        return cmd

    def get_ci_job_id(self) -> str:
        """Get CI job ID from environment or use test name."""
        return os.environ.get("CI_JOB_ID", "local")

    @property
    def policy_config(self) -> PolicyConfig:
        """Get the policy configuration from the loaded YAML as a typed PolicyConfig.

        This provides type hints and IDE support when accessing policy configuration.
        The returned config is a DictConfig but typed as PolicyConfig for convenience.

        Example:
            config = NeMoRLTestConfig(...)
            model_name = config.policy_config["model_name"]
            dtensor_enabled = config.policy_config["dtensor_cfg"]["enabled"]
        """
        return cast(PolicyConfig, OmegaConf.select(self.loaded_yaml_config, "policy"))

    def get_dtensor_config(self) -> Optional[DTensorConfig]:
        """Get DTensor configuration if dtensor is enabled.

        Returns:
            DTensorConfig dict if dtensor is enabled, None otherwise
        """
        policy = self.policy_config
        if policy and policy.get("dtensor_cfg", {}).get("enabled", False):
            return cast(DTensorConfig, policy["dtensor_cfg"])
        return None

    def get_megatron_config(self) -> Optional[MegatronConfig]:
        """Get Megatron configuration if megatron is enabled.

        Returns:
            MegatronConfig dict if megatron is enabled, None otherwise
        """
        policy = self.policy_config
        if policy and policy.get("megatron_cfg", {}).get("enabled", False):
            return cast(MegatronConfig, policy["megatron_cfg"])
        return None

    def get_num_runs(self) -> int:
        """Get the number of training runs based on steps_per_run.

        Returns:
            Number of runs (1 if steps_per_run is None, otherwise ceil(max_steps / steps_per_run))
        """
        if self.steps_per_run is None:
            return 1
        return math.ceil(self.max_steps / self.steps_per_run)

    def get_target_steps(self, run_num: int) -> int:
        """Get the target max_steps for a specific run number.

        Args:
            run_num: The run number (0-indexed)

        Returns:
            Target max_steps for this run (min of (run_num + 1) * steps_per_run and max_steps)
        """
        if self.steps_per_run is None:
            return self.max_steps
        return min((run_num + 1) * self.steps_per_run, self.max_steps)


@dataclass
class DefaultNeMoRLTestConfig(NeMoRLTestConfig):
    """Test configuration with default test overrides pre-applied.

    This config class automatically includes standard overrides for:
    - Logger configuration (wandb, tensorboard, GPU monitoring)
    - Checkpointing configuration
    - Log and checkpoint directories

    Users can still override these defaults by passing custom values in the overrides dict.

    Example:
        # Use defaults
        config = DefaultNeMoRLTestConfig(
            test_name="my-test",
            algorithm="sft",
            model_class="llm",
        )

        # Override specific defaults
        config = DefaultNeMoRLTestConfig(
            test_name="my-test",
            algorithm="sft",
            model_class="llm",
            overrides={
                "logger.wandb_enabled": False,  # Disable wandb
                "trainer.max_steps": 100,  # Add custom override
            }
        )
    """

    def __post_init__(self):
        # Apply default test overrides before parent __post_init__
        # This ensures they're part of the config but can still be overridden
        default_overrides = {
            "logger.wandb_enabled": True,
            "logger.wandb.project": "nemo-rl",
            "logger.monitor_gpus": True,
            "logger.tensorboard_enabled": True,
            "checkpointing.enabled": True,
        }

        # Merge defaults with user overrides (user overrides take precedence)
        merged_overrides = {**default_overrides, **self.overrides}
        self.overrides = merged_overrides

        # Call parent __post_init__ which will process all overrides
        super().__post_init__()

        # Add path-based overrides after directories are set up
        # These overrides reference paths that are computed in parent __post_init__
        path_overrides = {
            "logger.log_dir": str(self.log_dir),
            "logger.wandb.name": self.test_name,
            "checkpointing.checkpoint_dir": str(self.ckpt_dir),
        }

        # Merge path overrides with existing overrides (path overrides take precedence)
        self.overrides = {**self.overrides, **path_overrides}
