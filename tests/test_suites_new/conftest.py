"""Pytest configuration and fixtures for NeMo-RL test suites."""

import pytest


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--slurm",
        action="store_true",
        default=False,
        help="Submit tests to Slurm cluster instead of running locally",
    )

    # Custom filtering options based on TestConfig values
    parser.addoption(
        "--algorithm", help="Filter by algorithm (e.g., sft, grpo, dpo, vlm_grpo)"
    )
    parser.addoption(
        "--backend", help="Filter by backend (e.g., fsdp2, megatron, dtensor)"
    )
    parser.addoption(
        "--suite",
        help="Filter by test suite (e.g., nightly, quick, release, long, performance)",
    )
    parser.addoption("--num-gpus", type=int, help="Filter by exact GPU count")
    parser.addoption("--num-nodes", type=int, help="Filter by node count")
    parser.addoption(
        "--filter",
        help="Filter using Python expression on config (e.g., 'config.num_gpus_total >= 32 and config.backend == \"fsdp2\"')",
    )


def pytest_configure(config):
    """Register custom markers for test filtering.

    Markers are organized into groups using prefixes:
    - algo_*: Algorithm type
    - suite_*: Test suite type
    - model_size_*: Model size category
    - num_gpus_*: Number of GPUs required
    - num_nodes_*: Number of nodes required
    - engine_*: Backend/engine type
    - parallelism_*: Parallelism strategy
    - feature_*: Special features
    """
    # Algorithm markers (algo_*)
    config.addinivalue_line("markers", "algo_sft: Supervised Fine-Tuning tests")
    config.addinivalue_line(
        "markers", "algo_grpo: GRPO (Group Relative Policy Optimization) tests"
    )
    config.addinivalue_line(
        "markers", "algo_dpo: DPO (Direct Preference Optimization) tests"
    )
    config.addinivalue_line(
        "markers", "algo_vlm_grpo: Vision-Language Model GRPO tests"
    )

    # Test suite markers (suite_*)
    config.addinivalue_line("markers", "suite_nightly: Tests that run nightly")
    config.addinivalue_line("markers", "suite_release: Tests that run before release")
    config.addinivalue_line("markers", "suite_performance: Performance benchmark tests")
    config.addinivalue_line("markers", "suite_quick: Quick smoke tests")
    config.addinivalue_line("markers", "suite_long: Long-running tests")

    # Model size markers (model_size_*)
    config.addinivalue_line(
        "markers", "model_size_small: Small models (< 2B parameters)"
    )
    config.addinivalue_line(
        "markers", "model_size_medium: Medium models (2B-10B parameters)"
    )
    config.addinivalue_line(
        "markers", "model_size_large: Large models (10B-100B parameters)"
    )
    config.addinivalue_line(
        "markers", "model_size_xlarge: Extra large models (> 100B parameters)"
    )

    # GPU requirement markers (num_gpus_*)
    config.addinivalue_line("markers", "num_gpus_1: Tests requiring 1 GPU")
    config.addinivalue_line("markers", "num_gpus_2: Tests requiring 2 GPUs")
    config.addinivalue_line("markers", "num_gpus_4: Tests requiring 4 GPUs")
    config.addinivalue_line("markers", "num_gpus_8: Tests requiring 8 GPUs")
    config.addinivalue_line("markers", "num_gpus_16: Tests requiring 16 GPUs")
    config.addinivalue_line("markers", "num_gpus_32: Tests requiring 32 GPUs")
    config.addinivalue_line("markers", "num_gpus_64: Tests requiring 64 GPUs")
    config.addinivalue_line("markers", "num_gpus_128: Tests requiring 128 GPUs")
    config.addinivalue_line("markers", "num_gpus_256: Tests requiring 256 GPUs")

    # Node requirement markers (num_nodes_*)
    config.addinivalue_line("markers", "num_nodes_1: Tests requiring 1 node")
    config.addinivalue_line("markers", "num_nodes_2: Tests requiring 2 nodes")
    config.addinivalue_line("markers", "num_nodes_4: Tests requiring 4 nodes")
    config.addinivalue_line("markers", "num_nodes_8: Tests requiring 8 nodes")
    config.addinivalue_line("markers", "num_nodes_16: Tests requiring 16 nodes")
    config.addinivalue_line("markers", "num_nodes_32: Tests requiring 32 nodes")

    # Engine/Backend markers (engine_*)
    config.addinivalue_line("markers", "engine_fsdp2: Tests using FSDP2 backend")
    config.addinivalue_line(
        "markers", "engine_mcore: Tests using Megatron-Core backend"
    )
    config.addinivalue_line("markers", "engine_dtensor: Tests using DTensor backend")

    # Parallelism strategy markers (parallelism_*)
    config.addinivalue_line("markers", "parallelism_tp: Tests using tensor parallelism")
    config.addinivalue_line(
        "markers", "parallelism_pp: Tests using pipeline parallelism"
    )
    config.addinivalue_line(
        "markers", "parallelism_sp: Tests using sequence parallelism"
    )
    config.addinivalue_line("markers", "parallelism_fsdp: Tests using FSDP")

    # Feature markers (feature_*)
    config.addinivalue_line(
        "markers",
        "feature_activation_checkpointing: Tests using activation checkpointing",
    )
    config.addinivalue_line("markers", "feature_fp8: Tests using FP8 precision")
    config.addinivalue_line(
        "markers", "feature_dynamic_batch: Tests using dynamic batching"
    )
    config.addinivalue_line(
        "markers", "feature_sequence_packing: Tests using sequence packing"
    )
    config.addinivalue_line(
        "markers", "feature_non_colocated: Tests with non-colocated components"
    )


@pytest.fixture
def project_root(tmp_path):
    """Return the project root directory."""
    import subprocess
    from pathlib import Path

    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())


@pytest.fixture
def test_output_dir(tmp_path, request):
    """Create a temporary output directory for test artifacts.

    This fixture can be overridden by setting the TEST_OUTPUT_DIR environment variable
    to use a persistent directory instead of tmp_path.
    """
    import os
    from pathlib import Path

    output_dir = os.environ.get("TEST_OUTPUT_DIR")
    if output_dir:
        output_dir = Path(output_dir) / request.node.name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    return tmp_path


@pytest.fixture
def skip_if_insufficient_gpus(request):
    """Skip test if insufficient GPUs are available."""
    import torch

    # Extract GPU requirement from markers
    required_gpus = None
    for marker in request.node.iter_markers():
        if marker.name.startswith("num_gpus_"):
            required_gpus = int(marker.name.split("_")[-1])
            break

    if required_gpus is not None:
        available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if available_gpus < required_gpus:
            pytest.skip(
                f"Test requires {required_gpus} GPUs but only {available_gpus} available"
            )


@pytest.fixture(autouse=True)
def setup_git_safe_directory():
    """Mark all repos as safe in the test context.

    This is needed because wandb fetches metadata about the repo and it's a
    catch-22 to get the project root and mark it safe if you don't know the project root.
    """
    import subprocess

    subprocess.run(
        ["git", "config", "--global", "--add", "safe.directory", "*"],
        check=False,  # Don't fail if this doesn't work
    )


@pytest.fixture
def env_vars():
    """Return environment variables needed for tests."""
    import os

    return {
        "HF_HOME": os.environ.get("HF_HOME"),
        "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE"),
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1"),
        "HF_TOKEN": os.environ.get("RL_HF_TOKEN"),
        "WANDB_API_KEY": os.environ.get("WANDB_API_KEY"),
        "NRL_DEEPSCALER_8K_CKPT": os.environ.get("NRL_DEEPSCALER_8K_CKPT"),
        "NRL_DEEPSCALER_16K_CKPT": os.environ.get("NRL_DEEPSCALER_16K_CKPT"),
        "NRL_DEEPSCALER_24K_CKPT": os.environ.get("NRL_DEEPSCALER_24K_CKPT"),
    }


@pytest.fixture
def use_slurm(request):
    """Return True if tests should be submitted to Slurm."""
    return request.config.getoption("--slurm")


def pytest_collection_modifyitems(config, items):
    """Auto-generate markers from TestConfig and apply custom filters.

    This hook:
    1. Inspects each test class for a 'config' attribute (TestConfig instance)
    2. Auto-generates pytest markers based on config values
    3. Applies custom filters based on command-line options
    """
    # Get filter options
    algorithm_filter = config.getoption("--algorithm")
    backend_filter = config.getoption("--backend")
    suite_filter = config.getoption("--suite")
    num_gpus_filter = config.getoption("--num-gpus")
    num_nodes_filter = config.getoption("--num-nodes")
    filter_expr = config.getoption("--filter")

    filtered_items = []

    for item in items:
        # Check if this is a test method in a class with a config attribute
        if not (hasattr(item, "cls") and item.cls is not None):
            filtered_items.append(item)  # Keep non-class tests
            continue

        test_class = item.cls
        if not hasattr(test_class, "config"):
            filtered_items.append(item)  # Keep tests without config
            continue

        cfg = test_class.config

        # === Auto-generate markers from TestConfig ===

        # Algorithm marker
        item.add_marker(getattr(pytest.mark, f"algo_{cfg.algorithm}"))

        # Backend marker
        item.add_marker(getattr(pytest.mark, f"backend_{cfg.backend}"))

        # Resource markers
        item.add_marker(getattr(pytest.mark, f"num_nodes_{cfg.num_nodes}"))
        item.add_marker(getattr(pytest.mark, f"num_gpus_{cfg.num_gpus_total}"))

        # Test suite markers
        for suite in cfg.test_suites:
            item.add_marker(getattr(pytest.mark, f"suite_{suite}"))

        # Model size markers (derived from model_name)
        model_size = _get_model_size_from_name(cfg.model_name)
        if model_size:
            item.add_marker(getattr(pytest.mark, f"model_size_{model_size}"))

        # Parallelism markers
        if cfg.tensor_parallel:
            item.add_marker(pytest.mark.parallelism_tp)
        if cfg.pipeline_parallel:
            item.add_marker(pytest.mark.parallelism_pp)
        if cfg.sequence_parallel:
            item.add_marker(pytest.mark.parallelism_sp)
        if cfg.fsdp:
            item.add_marker(pytest.mark.parallelism_fsdp)

        # Feature markers
        if cfg.activation_checkpointing:
            item.add_marker(pytest.mark.feature_activation_checkpointing)

        # Detect features from test_name
        if "fp8" in cfg.test_name.lower():
            item.add_marker(pytest.mark.feature_fp8)
        if "dynamicbatch" in cfg.test_name.lower():
            item.add_marker(pytest.mark.feature_dynamic_batch)
        if "seqpack" in cfg.test_name.lower():
            item.add_marker(pytest.mark.feature_sequence_packing)
        if "noncolocated" in cfg.test_name.lower():
            item.add_marker(pytest.mark.feature_non_colocated)

        # === Apply custom filters ===

        # Algorithm filter
        if algorithm_filter and cfg.algorithm != algorithm_filter:
            continue

        # Backend filter
        if backend_filter and cfg.backend != backend_filter:
            continue

        # Suite filter
        if suite_filter and suite_filter not in cfg.test_suites:
            continue

        # GPU count filter
        if num_gpus_filter is not None and cfg.num_gpus_total != num_gpus_filter:
            continue

        # Node count filter
        if num_nodes_filter is not None and cfg.num_nodes != num_nodes_filter:
            continue

        # Expression-based filter
        if filter_expr:
            try:
                if not eval(filter_expr, {"config": cfg}):
                    continue
            except Exception as e:
                print(f"Warning: Failed to evaluate filter for {item.name}: {e}")
                continue

        # Test passed all filters, keep it
        filtered_items.append(item)

    # Replace items with filtered list
    items[:] = filtered_items


def _get_model_size_from_name(model_name: str) -> str:
    """Derive model size category from model name.

    Returns:
        Model size category: 'small', 'medium', 'large', or 'xlarge'
    """
    model_name_lower = model_name.lower()

    # Extract parameter count from model name
    # Examples: "llama3.1-8b", "qwen2.5-32b", "gemma3-27b", "llama3.1-70b"
    if "1b" in model_name_lower or "1.5b" in model_name_lower:
        return "small"
    elif any(x in model_name_lower for x in ["7b", "8b", "9b"]):
        return "medium"
    elif any(x in model_name_lower for x in ["13b", "27b", "30b", "32b", "70b"]):
        return "large"
    elif any(x in model_name_lower for x in ["100b", "175b", "405b"]):
        return "xlarge"

    # Default to medium if we can't determine
    return "medium"
