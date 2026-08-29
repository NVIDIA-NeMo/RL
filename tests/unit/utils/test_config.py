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
import tempfile
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

REPO_ROOT = Path(__file__).resolve().parents[3]
ULTRA_CONFIG_PATHS = [
    "examples/nemo_gym/nemotron-3-ultra/student_rlvr1.yaml",
    "examples/nemo_gym/nemotron-3-ultra/student_rlvr2.yaml",
    "examples/nemo_gym/nemotron-3-ultra/ifbench_teacher.yaml",
    "examples/nemo_gym/nemotron-3-ultra/reasoning_teacher.yaml",
    "examples/nemo_gym/nemotron-3-ultra/rlhf_teacher.yaml",
    "examples/nemo_gym/nemotron-3-ultra/swe_teacher.yaml",
    "examples/nemo_gym/nemotron-3-ultra/mopd.yaml",
]
NEMO_GYM_CONFIG_PATHS = ULTRA_CONFIG_PATHS + [
    "examples/nemo_gym/nemotron-3.5-lightning/rlvr.yaml",
]
# Select by content: every example that pins a service port range must stay
# inside the layout in virtual_cluster.py. Configs without a port key inherit
# the defaults and cannot violate the invariant themselves.
PORT_ISOLATION_CONFIG_PATHS = sorted(
    str(path.relative_to(REPO_ROOT))
    for path in (REPO_ROOT / "examples").rglob("*.yaml")
    if "port_range" in path.read_text()
)
assert PORT_ISOLATION_CONFIG_PATHS, "port-isolation config selection matched nothing"


@pytest.fixture
def temp_config_dir():
    """Create a temporary directory for test configs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def create_test_config(config_dir: Path, name: str, content: str):
    """Create a test config file."""
    config_path = config_dir / name
    config_path.write_text(content)
    return config_path


def test_single_inheritance(temp_config_dir):
    """Test basic inheritance from a single parent."""
    # Create parent config
    parent_content = """
    common:
      value: 42
    parent_only:
      value: 100
    """
    create_test_config(temp_config_dir, "parent.yaml", parent_content)

    # Create child config
    child_content = """
    defaults: parent.yaml
    common:
      value: 43
    child_only:
      value: 200
    """
    child_path = create_test_config(temp_config_dir, "child.yaml", child_content)

    # Load and verify
    config = load_config(child_path)
    assert config.common.value == 43  # Child overrides parent
    assert config.parent_only.value == 100  # Parent value preserved
    assert config.child_only.value == 200  # Child-only value exists


def test_multiple_inheritance(temp_config_dir):
    """Test inheritance from multiple parents."""
    # Create first parent
    parent1_content = """
    common:
      value: 42
    parent1_only:
      value: 100
    """
    create_test_config(temp_config_dir, "parent1.yaml", parent1_content)

    # Create second parent
    parent2_content = """
    common:
      value: 43
    parent2_only:
      value: 200
    """
    create_test_config(temp_config_dir, "parent2.yaml", parent2_content)

    # Create child config
    child_content = """
    defaults:
      - parent1.yaml
      - parent2.yaml
    common:
      value: 44
    child_only:
      value: 300
    """
    child_path = create_test_config(temp_config_dir, "child.yaml", child_content)

    # Load and verify
    config = load_config(child_path)
    assert config.common.value == 44  # Child overrides both parents
    assert config.parent1_only.value == 100  # First parent value preserved
    assert config.parent2_only.value == 200  # Second parent value preserved
    assert config.child_only.value == 300  # Child-only value exists


def test_absolute_path_inheritance(temp_config_dir):
    """Test inheritance using absolute paths."""
    # Create parent config
    parent_content = """
    common:
      value: 42
    """
    parent_path = create_test_config(temp_config_dir, "parent.yaml", parent_content)

    # Create child config with absolute path
    child_content = f"""
    defaults: {parent_path}
    common:
      value: 43
    """
    child_path = create_test_config(temp_config_dir, "child.yaml", child_content)

    # Load and verify
    config = load_config(child_path)
    assert config.common.value == 43  # Child overrides parent


def test_no_inheritance(temp_config_dir):
    """Test config without inheritance."""
    content = """
    common:
      value: 42
    """
    config_path = create_test_config(temp_config_dir, "config.yaml", content)

    # Load and verify
    config = load_config(config_path)
    assert config.common.value == 42


def test_nested_inheritance(temp_config_dir):
    """Test nested inheritance (parent inherits from grandparent)."""
    # Create grandparent config
    grandparent_content = """
    common:
      value: 42
    grandparent_only:
      value: 100
    """
    create_test_config(temp_config_dir, "grandparent.yaml", grandparent_content)

    # Create parent config
    parent_content = """
    defaults: grandparent.yaml
    common:
      value: 43
    parent_only:
      value: 200
    """
    create_test_config(temp_config_dir, "parent.yaml", parent_content)

    # Create child config
    child_content = """
    defaults: parent.yaml
    common:
      value: 44
    child_only:
      value: 300
    """
    child_path = create_test_config(temp_config_dir, "child.yaml", child_content)

    # Load and verify
    config = load_config(child_path)
    assert config.common.value == 44  # Child overrides all
    assert config.grandparent_only.value == 100  # Grandparent value preserved
    assert config.parent_only.value == 200  # Parent value preserved
    assert config.child_only.value == 300  # Child-only value exists


def test_inheritance_preserves_missing_mandatory_value(temp_config_dir):
    """Test that a mandatory parent value can be supplied after inheritance."""
    create_test_config(temp_config_dir, "parent.yaml", "required: ???")
    child_path = create_test_config(
        temp_config_dir,
        "child.yaml",
        "defaults: parent.yaml",
    )

    config = load_config(child_path)

    assert OmegaConf.is_missing(config, "required")


def test_interpolation(temp_config_dir):
    """Test that interpolation works with inherited configs."""
    # Create parent config
    parent_content = """
    base_value: 42
    derived:
      value: ${base_value}
    """
    create_test_config(temp_config_dir, "parent.yaml", parent_content)

    # Create child config
    child_content = """
    defaults: parent.yaml
    base_value: 43
    """
    child_path = create_test_config(temp_config_dir, "child.yaml", child_content)

    # Load and verify
    config = load_config(child_path)
    assert config.base_value == 43
    assert config.derived.value == 43  # Interpolation uses child's base_value


def test_add_resolver():
    """Test the arithmetic resolver used by Ultra configs."""
    register_omegaconf_resolvers()
    config = OmegaConf.create({"value": "${add:2,3}"})

    assert config.value == 5


@pytest.mark.parametrize("config_path", NEMO_GYM_CONFIG_PATHS)
def test_nemo_gym_configs_satisfy_current_grpo_contract(config_path):
    """Ensure production NeMo Gym configs satisfy the current GRPO contract."""
    from nemo_rl.algorithms.grpo import MasterConfig
    from nemo_rl.utils.checkpoint import CheckpointManager

    register_omegaconf_resolvers()
    config = load_config(REPO_ROOT / config_path)

    # These values are intentionally supplied by recipe launchers at runtime.
    config.policy.model_name = "test-model"
    for split in ("train", "validation"):
        datasets = config.data.get(split)
        if datasets is None:
            continue
        if not OmegaConf.is_list(datasets):
            datasets = [datasets]
        for dataset in datasets:
            if "data_path" in dataset:
                dataset.data_path = "/tmp/test-data.jsonl"

    if OmegaConf.is_missing(config, "sif_dir"):
        config["sif_dir"] = "/tmp/test-sifs"
    if "_teachers" in config and OmegaConf.is_missing(config["_teachers"], "general"):
        config["_teachers"]["general"] = "/tmp/test-teacher"

    resolved = OmegaConf.to_container(config, resolve=True)

    if config_path == "examples/nemo_gym/nemotron-3.5-lightning/rlvr.yaml":
        assert resolved["grpo"]["val_num_generations_per_prompt"] == 2
        assert resolved["checkpointing"]["metric_name"] is None

    # The real contract checks: the config validates against GRPO's MasterConfig
    # schema and the checkpointing block is accepted by CheckpointManager.
    master_config = MasterConfig.model_validate(resolved)
    CheckpointManager(master_config.checkpointing)


@pytest.mark.parametrize("config_path", PORT_ISOLATION_CONFIG_PATHS)
def test_example_service_port_ranges_are_isolated(config_path):
    """Resolved example configs must not bypass the below-9000 port layout."""
    from nemo_rl.distributed.virtual_cluster import (
        DEFAULT_GENERATION_PORT_RANGE_HIGH,
        DEFAULT_GENERATION_PORT_RANGE_LOW,
        DEFAULT_GENERATION_ROUTER_PORT_RANGE_HIGH,
        DEFAULT_GENERATION_ROUTER_PORT_RANGE_LOW,
        DEFAULT_GYM_PORT_RANGE_HIGH,
        DEFAULT_GYM_PORT_RANGE_LOW,
        DEFAULT_MASTER_PORT_RANGE_HIGH,
        DEFAULT_MASTER_PORT_RANGE_LOW,
    )

    config = load_config(REPO_ROOT / config_path)

    def select_int(path: str, default: int) -> int:
        value = OmegaConf.select(config, path, default=default)
        assert isinstance(value, int)
        return value

    def effective_range(
        prefix: str, default_low: int, default_high: int
    ) -> tuple[int, int]:
        return (
            select_int(f"{prefix}.port_range_low", default_low),
            select_int(f"{prefix}.port_range_high", default_high),
        )

    router_range = effective_range(
        "async_rl.generation_router",
        DEFAULT_GENERATION_ROUTER_PORT_RANGE_LOW,
        DEFAULT_GENERATION_ROUTER_PORT_RANGE_HIGH,
    )
    generation_range = effective_range(
        "policy.generation",
        DEFAULT_GENERATION_PORT_RANGE_LOW,
        DEFAULT_GENERATION_PORT_RANGE_HIGH,
    )
    gym_range = effective_range(
        "env.nemo_gym",
        DEFAULT_GYM_PORT_RANGE_LOW,
        DEFAULT_GYM_PORT_RANGE_HIGH,
    )

    # cluster uses master_port_range_low/high rather than a nested master block.
    master_range = (
        select_int(
            "cluster.master_port_range_low",
            DEFAULT_MASTER_PORT_RANGE_LOW,
        ),
        select_int(
            "cluster.master_port_range_high",
            DEFAULT_MASTER_PORT_RANGE_HIGH,
        ),
    )

    assert (
        DEFAULT_GENERATION_ROUTER_PORT_RANGE_LOW
        <= router_range[0]
        < router_range[1]
        <= DEFAULT_GENERATION_ROUTER_PORT_RANGE_HIGH
    )
    assert DEFAULT_MASTER_PORT_RANGE_LOW <= master_range[0] < master_range[1]
    assert master_range[1] <= DEFAULT_MASTER_PORT_RANGE_HIGH
    assert (
        DEFAULT_GENERATION_PORT_RANGE_LOW
        <= generation_range[0]
        < generation_range[1]
        <= DEFAULT_GENERATION_PORT_RANGE_HIGH
    )
    assert (
        DEFAULT_GYM_PORT_RANGE_LOW
        <= gym_range[0]
        < gym_range[1]
        <= DEFAULT_GYM_PORT_RANGE_HIGH
    )
    assert router_range[1] <= master_range[0]
    assert master_range[1] <= generation_range[0]
    assert generation_range[1] <= gym_range[0]
    assert gym_range[1] < 9000


def test_parse_hydra_overrides():
    """Test parsing and applying Hydra overrides."""
    from nemo_rl.utils.config import OverridesError, parse_hydra_overrides

    # Create initial config
    cfg = OmegaConf.create(
        {
            "model": {"type": "default", "hidden_size": 768},
            "training": {"batch_size": 32, "learning_rate": 1e-4},
        }
    )

    # Test basic override
    overrides = ["model.type=transformer"]
    updated_cfg = parse_hydra_overrides(cfg, overrides)
    assert updated_cfg.model.type == "transformer"
    assert updated_cfg.model.hidden_size == 768  # Unchanged

    # Test nested override
    overrides = ["model.hidden_size=1024"]
    updated_cfg = parse_hydra_overrides(cfg, overrides)
    assert updated_cfg.model.hidden_size == 1024

    # Test multiple overrides
    overrides = ["training.batch_size=64", "training.learning_rate=2e-4"]
    updated_cfg = parse_hydra_overrides(cfg, overrides)
    assert updated_cfg.training.batch_size == 64
    assert updated_cfg.training.learning_rate == 2e-4

    # Test invalid override
    overrides = ["nonexistent.key=value"]
    with pytest.raises(OverridesError):
        parse_hydra_overrides(cfg, overrides)

    # Test invalid syntax
    overrides = ["invalid.syntax"]
    with pytest.raises(OverridesError):
        parse_hydra_overrides(cfg, overrides)

    # Test empty overrides
    overrides = []
    updated_cfg = parse_hydra_overrides(cfg, overrides)
    assert updated_cfg == cfg  # Config should be unchanged

    # Test override additions and deletions
    overrides = [
        "+model.num_layers=12",
        "++model.type=transformer",
        "~training.batch_size",
    ]
    updated_cfg = parse_hydra_overrides(cfg, overrides)
    assert updated_cfg.model.num_layers == 12
    assert updated_cfg.model.type == "transformer"
    assert "batch_size" not in updated_cfg.training
