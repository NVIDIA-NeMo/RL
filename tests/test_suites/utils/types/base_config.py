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

from omegaconf import OmegaConf

# Import strongly-typed config classes
from nemo_rl.data import DataConfig
from nemo_rl.distributed.virtual_cluster import ClusterConfig
from nemo_rl.models.policy import DTensorConfig, MegatronConfig, PolicyConfig
from nemo_rl.utils.checkpoint import CheckpointingConfig
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import LoggerConfig

# Import type aliases from parent module
from . import Algorithm, Backend, MasterConfigUnion, ModelClass


@dataclass
class NeMoRLTestConfig:
    """Test configuration with YAML base + overrides + test metadata.

    This class:
    1. References a YAML config file as the base training configuration
    2. Allows overriding specific YAML values for testing
    3. Stores test-specific metadata (test_name, algorithm, test_suites)
    4. Auto-extracts metadata from YAML for filtering/classification
    5. Provides strongly-typed accessors for all config sections

    Strongly-typed Config Accessors:
        - master_config: MasterConfigUnion - Full config as typed MasterConfig (public attribute)
        - loaded_yaml_config: MasterConfigUnion - Deprecated alias for master_config
        - policy_config: PolicyConfig - Policy/model configuration
        - cluster_config: ClusterConfig - Cluster configuration (nodes, GPUs)
        - data_config: DataConfig - Dataset configuration
        - logger_config: LoggerConfig - Logging configuration
        - checkpointing_config: CheckpointingConfig - Checkpointing configuration
        - algorithm_config: Dict[str, Any] - Algorithm-specific config (SFT/GRPO/DPO/etc)

    Example:
        config = NeMoRLTestConfig(
            test_name="my-test",
            algorithm="sft",
            model_class="llm",
        )

        # Strongly-typed access via properties
        model_name: str = config.policy_config["model_name"]
        num_nodes: int = config.cluster_config["num_nodes"]
        dataset: str = config.data_config["dataset_name"]
        max_steps: int = config.algorithm_config["max_num_steps"]

        # Direct access to master config
        full_config: MasterConfigUnion = config.master_config
    """

    #######################################################
    # Metadata we add to tests (required fields first)
    #######################################################
    test_name: str  # Test identifier
    algorithm: Algorithm
    model_class: ModelClass
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
                    f"Config file not found: {self.yaml_config}. "
                    f"Expected to find it at the path derived from test name '{self.test_name}'"
                )
        else:
            if not self.yaml_config.exists():
                raise FileNotFoundError(f"Config file not found: {self.yaml_config}")

        # Loads YAML files with OmegaConf
        config = load_config(self.yaml_config)

        # Apply overrides with Hydra's parser
        if self.overrides:
            override_strings = [f"{k}={v}" for k, v in self.overrides.items()]
            config = parse_hydra_overrides(config, override_strings)

        # Convert to plain dict with all interpolations resolved
        self.master_config: MasterConfigUnion = OmegaConf.to_container(
            config, resolve=True
        )

        # Validate algorithm matches config structure
        self._validate_algorithm_config()

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

    #######################################################
    # Helper functions for typed config access
    #######################################################

    def _get_policy_config(self) -> PolicyConfig:
        """Get policy config with strong typing.

        Returns:
            PolicyConfig with type hints for IDE support
        """
        return cast(PolicyConfig, self.master_config.get("policy", {}))

    def _get_dtensor_config(self) -> DTensorConfig:
        """Get DTensor config with strong typing.

        Returns:
            DTensorConfig with type hints for IDE support
        """
        policy = self._get_policy_config()
        return cast(DTensorConfig, policy.get("dtensor_cfg", {}))

    def _get_megatron_config(self) -> MegatronConfig:
        """Get Megatron config with strong typing.

        Returns:
            MegatronConfig with type hints for IDE support
        """
        policy = self._get_policy_config()
        return cast(MegatronConfig, policy.get("megatron_cfg", {}))

    #######################################################
    # Validation methods
    #######################################################

    def _validate_algorithm_config(self):
        """Validate that the algorithm field matches the config structure.

        This ensures that:
        - For algorithm="sft", config has "sft" key with SFTConfig
        - For algorithm="grpo", config has "grpo" key with GRPOConfig
        - For algorithm="dpo", config has "dpo" key with DPOConfig
        - For algorithm="distillation", config has "distillation" key with DistillationConfig
        - For algorithm="rm", config has "rm" key with RMConfig

        Raises:
            ValueError: If algorithm doesn't match config structure
        """
        # Check if the algorithm-specific config section exists
        if self.algorithm not in self.master_config:
            raise ValueError(
                f"Config validation failed: algorithm='{self.algorithm}' but config "
                f"does not have '{self.algorithm}' section. "
                f"Available sections: {list(self.master_config.keys())}"
            )

        # Additional validation for required sections in MasterConfig
        required_sections = ["policy", "data", "logger", "cluster", "checkpointing"]
        missing_sections = [
            section
            for section in required_sections
            if section not in self.master_config
        ]

        if missing_sections:
            raise ValueError(
                f"Config validation failed: Missing required sections: {missing_sections}. "
                f"Available sections: {list(self.master_config.keys())}"
            )

    def _extract_model_name(self) -> str:
        """Extract and normalize model name from policy config.

        Uses strong typing from PolicyConfig for type-safe access.

        Returns:
            Normalized model name (lowercase, no dashes/underscores, no org prefix)
        """
        # Get policy config with strong typing
        policy_config_typed = self._get_policy_config()
        model_name = policy_config_typed.get("model_name", "")

        # Clean up model name for filtering (remove org prefix)
        # e.g., "meta-llama/Llama-3.1-8B-Instruct" -> "llama3.1-8b-instruct"
        if "/" in model_name:
            model_name = model_name.split("/")[-1]

        model_name = model_name.lower().replace("-", "").replace("_", "")
        return model_name

    def _detect_backend(self) -> Backend:
        """Detect the backend type (fsdp2, dtensor, or megatron).

        Uses strong typing from PolicyConfig, DTensorConfig, and MegatronConfig for type-safe access.

        Returns:
            Backend type (one of "fsdp2", "dtensor", "megatron")
        """
        # Get backend configs with strong typing
        dtensor_cfg = self._get_dtensor_config()
        megatron_cfg = self._get_megatron_config()

        dtensor_enabled = dtensor_cfg.get("enabled", False)
        megatron_enabled = megatron_cfg.get("enabled", False)

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

    def _extract_dtensor_parallelism(self) -> Dict[str, Any]:
        """Extract parallelism configuration from DTensor backend.

        Returns:
            Dictionary with keys: tensor_parallel, pipeline_parallel, sequence_parallel
        """
        dtensor_cfg = self._get_dtensor_config()
        return {
            "tensor_parallel": dtensor_cfg.get("tensor_parallel_size"),
            "pipeline_parallel": dtensor_cfg.get("pipeline_parallel_size"),
            "sequence_parallel": dtensor_cfg.get("sequence_parallel", False),
        }

    def _extract_megatron_parallelism(self) -> Dict[str, Any]:
        """Extract parallelism configuration from Megatron backend.

        Returns:
            Dictionary with keys: tensor_parallel, pipeline_parallel, sequence_parallel
        """
        megatron_cfg = self._get_megatron_config()
        return {
            "tensor_parallel": megatron_cfg.get("tensor_model_parallel_size"),
            "pipeline_parallel": megatron_cfg.get("pipeline_model_parallel_size"),
            "sequence_parallel": megatron_cfg.get("sequence_parallel", False),
        }

    def _extract_fsdp2_parallelism(self) -> Dict[str, Any]:
        """Extract parallelism configuration from FSDP2 backend.

        FSDP2 doesn't use tensor/pipeline parallelism, so all values are None/False.

        Returns:
            Dictionary with keys: tensor_parallel, pipeline_parallel, sequence_parallel
        """
        return {
            "tensor_parallel": None,
            "pipeline_parallel": None,
            "sequence_parallel": False,
        }

    def _extract_parallelism_config(self, backend: Backend) -> Dict[str, Any]:
        """Extract parallelism configuration based on backend type.

        Dispatches to backend-specific extraction functions.

        Args:
            backend: The backend type ("fsdp2", "dtensor", or "megatron")

        Returns:
            Dictionary with keys: tensor_parallel, pipeline_parallel, sequence_parallel
        """
        if backend == "dtensor":
            return self._extract_dtensor_parallelism()
        elif backend == "megatron":
            return self._extract_megatron_parallelism()
        else:  # fsdp2
            return self._extract_fsdp2_parallelism()

    #######################################################
    # Properties (computed from YAML + overrides)
    #######################################################

    @property
    def loaded_yaml_config(self) -> MasterConfigUnion:
        """Get the loaded YAML configuration as a typed MasterConfig.

        This property provides access to the loaded and parsed YAML config
        with overrides applied and all interpolations resolved.

        DEPRECATED: Use master_config directly instead.

        Returns:
            Plain dict typed as MasterConfigUnion (one of SFT, GRPO, DPO, Distillation, RM configs)
        """
        return self.master_config

    @property
    def model_name(self) -> str:
        """Extract and normalize model name from config."""
        return self._extract_model_name()

    @property
    def backend(self) -> Backend:
        """Detect the backend (fsdp2, dtensor, or megatron).

        Returns:
            Backend type (one of "fsdp2", "dtensor", "megatron")
        """
        return self._detect_backend()

    @property
    def num_nodes(self) -> int:
        """Number of nodes from cluster configuration.

        Uses strong typing from ClusterConfig for type-safe access.
        """
        return self.cluster_config.get("num_nodes", 1)

    @property
    def num_gpus_per_node(self) -> int:
        """Number of GPUs per node from cluster configuration.

        Uses strong typing from ClusterConfig for type-safe access.
        """
        return self.cluster_config.get("gpus_per_node", 8)

    @property
    def tensor_parallel(self) -> Optional[int]:
        """Tensor parallel size from backend configuration."""
        parallelism = self._extract_parallelism_config(self.backend)
        return parallelism["tensor_parallel"]

    @property
    def pipeline_parallel(self) -> Optional[int]:
        """Pipeline parallel size from backend configuration."""
        parallelism = self._extract_parallelism_config(self.backend)
        return parallelism["pipeline_parallel"]

    @property
    def sequence_parallel(self) -> bool:
        """Sequence parallel setting from backend configuration."""
        parallelism = self._extract_parallelism_config(self.backend)
        return parallelism["sequence_parallel"]

    @property
    def max_steps(self) -> int:
        """Maximum training steps from algorithm-specific configuration.

        Different algorithms have different config keys:
        - SFT: sft.max_num_steps
        - GRPO: grpo.max_num_steps
        - DPO: dpo.max_num_steps
        - Distillation: distillation.max_num_steps
        - RM: rm.max_num_steps
        """
        algo_config = self.master_config.get(self.algorithm, {})
        return algo_config.get("max_num_steps", 0)

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
        # Use helper functions for strong typing
        dtensor_cfg = self._get_dtensor_config()
        megatron_cfg = self._get_megatron_config()

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
        """Get the path to the main run script based on algorithm.

        Maps each algorithm to its corresponding run script:
        - sft -> run_sft.py
        - grpo -> run_grpo_math.py
        - dpo -> run_dpo.py
        - distillation -> run_distillation_math.py
        - rm -> run_rm.py
        """
        script_map: dict[Algorithm, str] = {
            "sft": "run_sft.py",
            "grpo": "run_grpo_math.py",
            "dpo": "run_dpo.py",
            "distillation": "run_distillation_math.py",
            "rm": "run_rm.py",
        }
        script_name = script_map[self.algorithm]  # Type-safe access, no need for get()

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
        The returned config is a plain dict typed as PolicyConfig for convenience.

        Example:
            config = NeMoRLTestConfig(...)
            model_name = config.policy_config["model_name"]
            dtensor_enabled = config.policy_config["dtensor_cfg"]["enabled"]
        """
        return cast(PolicyConfig, self.master_config.get("policy", {}))

    def get_dtensor_config(self) -> Optional[DTensorConfig]:
        """Get DTensor configuration if dtensor is enabled.

        Returns:
            DTensorConfig dict if dtensor is enabled, None otherwise
        """
        dtensor_cfg = self._get_dtensor_config()
        if dtensor_cfg.get("enabled", False):
            return dtensor_cfg
        return None

    def get_megatron_config(self) -> Optional[MegatronConfig]:
        """Get Megatron configuration if megatron is enabled.

        Returns:
            MegatronConfig dict if megatron is enabled, None otherwise
        """
        megatron_cfg = self._get_megatron_config()
        if megatron_cfg.get("enabled", False):
            return megatron_cfg
        return None

    @property
    def cluster_config(self) -> ClusterConfig:
        """Get the cluster configuration as a typed ClusterConfig.

        Returns:
            ClusterConfig dict with num_nodes and gpus_per_node

        Example:
            config = NeMoRLTestConfig(...)
            num_nodes = config.cluster_config["num_nodes"]
            gpus_per_node = config.cluster_config["gpus_per_node"]
        """
        return cast(ClusterConfig, self.master_config.get("cluster", {}))

    @property
    def data_config(self) -> DataConfig:
        """Get the data configuration as a typed DataConfig.

        Returns:
            DataConfig dict with dataset settings

        Example:
            config = NeMoRLTestConfig(...)
            dataset_name = config.data_config["dataset_name"]
            max_input_length = config.data_config["max_input_seq_length"]
        """
        return cast(DataConfig, self.master_config.get("data", {}))

    @property
    def logger_config(self) -> LoggerConfig:
        """Get the logger configuration as a typed LoggerConfig.

        Returns:
            LoggerConfig dict with logging settings

        Example:
            config = NeMoRLTestConfig(...)
            wandb_enabled = config.logger_config["wandb_enabled"]
            log_dir = config.logger_config["log_dir"]
        """
        return cast(LoggerConfig, self.master_config.get("logger", {}))

    @property
    def checkpointing_config(self) -> CheckpointingConfig:
        """Get the checkpointing configuration as a typed CheckpointingConfig.

        Returns:
            CheckpointingConfig dict with checkpointing settings

        Example:
            config = NeMoRLTestConfig(...)
            enabled = config.checkpointing_config["enabled"]
            checkpoint_dir = config.checkpointing_config["checkpoint_dir"]
        """
        return cast(CheckpointingConfig, self.master_config.get("checkpointing", {}))

    @property
    def algorithm_config(self) -> Dict[str, Any]:
        """Get the algorithm-specific configuration.

        Returns the configuration section specific to the algorithm:
        - For algorithm="sft": returns sft: SFTConfig
        - For algorithm="grpo": returns grpo: GRPOConfig
        - For algorithm="dpo": returns dpo: DPOConfig
        - For algorithm="distillation": returns distillation: DistillationConfig
        - For algorithm="rm": returns rm: RMConfig

        Returns:
            Algorithm-specific config dict

        Example:
            config = NeMoRLTestConfig(algorithm="sft", ...)
            max_steps = config.algorithm_config["max_num_steps"]
            val_period = config.algorithm_config["val_period"]
        """
        return self.master_config.get(self.algorithm, {})

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
