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
import json
import os
import random
import time
import unittest.mock
from datetime import datetime
from io import StringIO
from typing import Callable, TypedDict

import pytest
import ray
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nemo_rl.distributed.virtual_cluster import init_ray

dir_path = os.path.dirname(os.path.abspath(__file__))

TEST_ASSETS_DIR = os.path.join(dir_path, "test_assets")
UNIT_RESULTS_FILE = os.path.join(dir_path, "unit_results.json")
UNIT_RESULTS_FILE_DATED = os.path.join(
    dir_path, f"unit_results/{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
)


# Mapping between asset and absolute path (each are populated from a session level fixture)
class TEST_ASSETS:
    TINY_LLAMA_MODEL_PATH = os.path.join(
        TEST_ASSETS_DIR, "tiny_llama_with_llama3.2_tokenizer"
    )
    TINY_LLAMA_TIED_MODEL_PATH = os.path.join(
        TEST_ASSETS_DIR, "tiny_llama_tied_with_llama3.2_tokenizer"
    )
    TINY_QWEN2_MODEL_PATH = os.path.join(
        TEST_ASSETS_DIR, "tiny_qwen2_with_qwen2_tokenizer"
    )
    TINY_QWEN3_MODEL_PATH = os.path.join(
        TEST_ASSETS_DIR, "tiny_qwen3_with_qwen3_tokenizer"
    )
    TINY_GEMMA3_MODEL_PATH = os.path.join(
        TEST_ASSETS_DIR, "tiny_gemma3_with_gemma3_tokenizer"
    )


class UnitTestData(TypedDict):
    exit_status: int | str
    git_commit: str
    start_time: str
    metrics: dict
    gpu_types: list[str]
    coverage: str


###################################
# Meta Session Fixtures and Hooks #
###################################


@pytest.fixture(scope="session", autouse=True)
def _unit_test_data(request):
    """Initializes the unit test data storage at the beginning of the session."""
    session = request.session
    # Delete the unit results file at the start of a new test session
    if os.path.exists(UNIT_RESULTS_FILE):
        try:
            os.remove(UNIT_RESULTS_FILE)
            print(f"Deleted existing results file: {UNIT_RESULTS_FILE}")
        except Exception as e:
            print(f"Warning: Failed to delete results file: {e}")

    # Get the git commit hash
    try:
        import subprocess

        result = subprocess.run(
            ["git", "-C", dir_path, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        git_commit = result.stdout.strip()
    except Exception as e:
        git_commit = f"Error getting git commit: {str(e)}"

    unit_test_data = UnitTestData(
        exit_status="was not set",
        git_commit=git_commit,
        start_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        metrics={},
        gpu_types=[],
        coverage="[n/a] run with --cov=nemo_rl",
    )
    session.config._unit_test_data = unit_test_data
    return unit_test_data


@pytest.fixture(scope="session", autouse=True)
def session_data(request, init_ray_cluster, _unit_test_data):
    """Session-level fixture to store and save metrics data.

    This fixture tracks both metrics from tests and metadata about the test environment.
    The metrics are stored in the 'metrics' dictionary.

    It's set to autouse so that we track metadata and coverage even if no test selected
    explicitly track metrics.
    """
    # Pass init_ray_cluster so that we can access ray metadata

    ############################################################
    # 1. Gather all the unit test data #
    ############################################################
    unit_test_data: UnitTestData = _unit_test_data
    yield unit_test_data

    ############################################################
    # 2. Gather the ray metadata #
    ############################################################
    from nemo_rl.utils.logger import RayGpuMonitorLogger

    logger = RayGpuMonitorLogger(
        collection_interval=float("inf"),
        flush_interval=float("inf"),
        metric_prefix="test",
        step_metric="test/step",
        parent_logger=None,
    )
    unit_test_data["gpu_types"] = list(set(logger._collect_gpu_sku().values()))

    ############################################################
    # 3. Gather the coverage data #
    ############################################################
    # We directly access the coverage controller from the plugin manager
    # so we can access the coverage total before the pytest session finishes.
    cov_controller = None
    if request.config.pluginmanager.hasplugin("_cov"):
        plugin = request.config.pluginmanager.getplugin("_cov")
        if plugin.cov_controller:
            cov_controller = plugin.cov_controller

    if not cov_controller:
        # Means the user didn't run with --cov=...
        return

    # We currently don't use the cov_report since we can always access the coverage.json later, but
    # in the future if we want to report the coverage more granularly as part of the session finish,
    # we can access it here.
    cov_report = StringIO()
    cov_total = cov_controller.summary(cov_report)
    unit_test_data["coverage"] = cov_total


@pytest.fixture
def tracker(request, session_data, ray_gpu_monitor):
    """Test-level fixture that automatically captures test function info."""
    # Get fully qualified test name (module::test_function)
    module_name = request.module.__name__
    test_name = request.function.__name__
    qualified_name = f"{module_name}::{test_name}"

    # Initialize an empty dict for this test if it doesn't exist
    if qualified_name not in session_data:
        session_data["metrics"][qualified_name] = {}

    class Tracker:
        def track(self, metric_name: str, value):
            """Tracking an arbitrary metric."""
            session_data["metrics"][qualified_name][metric_name] = value

        def get_max_mem(self):
            metrics = ray_gpu_monitor._collect_metrics()
            max_mem = 0
            for m_name, m_value in metrics.items():
                if m_name.endswith(".memory"):
                    max_mem = max(max_mem, m_value)
            return max_mem

        def log_max_mem(self, metric_name: str):
            session_data["metrics"][qualified_name][metric_name] = self.get_max_mem()

    start_time = time.time()
    yield Tracker()
    end_time = time.time()
    # Prefix with `_` to indicate it's automatically collected
    session_data["metrics"][qualified_name]["_elapsed"] = end_time - start_time


def pytest_sessionfinish(session, exitstatus):
    data = session.config._unit_test_data
    data["exit_status"] = exitstatus
    print(f"\nSaving unit test data to {UNIT_RESULTS_FILE}")
    print(f"and saving to {UNIT_RESULTS_FILE_DATED}")
    with open(UNIT_RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    os.makedirs(os.path.dirname(UNIT_RESULTS_FILE_DATED), exist_ok=True)
    with open(UNIT_RESULTS_FILE_DATED, "w") as f:
        json.dump(data, f, indent=2)


################
# Ray Fixtures #
################


@pytest.fixture(scope="session", autouse=True)
def init_ray_cluster():
    """Initialize Ray for the test module and clean up afterward.

    This fixture doesn't need to be called directly.
    """
    init_ray()
    yield
    ray.shutdown()


@pytest.fixture(scope="session", autouse=True)
def ray_gpu_monitor(init_ray_cluster):
    """Initialize Ray for the test module and clean up afterward.

    This fixture doesn't need to be called directly.
    """
    from nemo_rl.utils.logger import RayGpuMonitorLogger

    gpu_monitor = RayGpuMonitorLogger(
        collection_interval=1,
        flush_interval=float("inf"),  # Disabling flushing since we will do it manually
        metric_prefix="test",
        step_metric="test/step",
        parent_logger=None,
    )
    gpu_monitor.start()
    yield gpu_monitor
    gpu_monitor.stop()


####################################
# Fixtures for Distributed Testing #
####################################


def _setup_distributed(rank, world_size, port, backend="nccl"):
    """Initialize the distributed environment for a test (internal use only)"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)  # Use the same port for all processes

    # Initialize the process group
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    # Set the device for this process
    torch.cuda.set_device(rank)


def _cleanup_distributed():
    """Clean up the distributed environment after a test (internal use only)"""
    dist.destroy_process_group()


@pytest.fixture
def distributed_test_runner():
    """Fixture that returns a function to run distributed tests.

    This fixture provides a reusable way to run a test function across multiple processes
    with PyTorch distributed communication set up.
    """

    def run_distributed_test(
        test_fn: Callable, world_size: int, backend: str = "nccl"
    ) -> None:
        """Run a test function in a distributed environment.

        Args:
            test_fn: The test function to run on each process
            world_size: Number of processes to spawn
            backend: PyTorch distributed backend to use
        """
        # Skip if CUDA is not available and using NCCL backend
        if backend == "nccl" and not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping CUDA-based test")

        # Skip if we don't have enough GPUs for NCCL backend
        if backend == "nccl" and torch.cuda.device_count() < world_size:
            pytest.skip(
                f"Not enough GPUs available. Need {world_size}, got {torch.cuda.device_count()}"
            )

        # Generate a single random port in the main process
        port = random.randint(10000, 20000)

        # Run the test on multiple processes
        mp.spawn(
            _distributed_test_wrapper,
            args=(test_fn, world_size, port, backend),
            nprocs=world_size,
            join=True,
        )

    return run_distributed_test


def _distributed_test_wrapper(
    rank: int, test_fn: Callable, world_size: int, port: int, backend: str
) -> None:
    """Wrapper function that sets up the distributed environment before running the test function.
    Internal use only - use distributed_test_runner fixture instead.

    Args:
        rank: Process rank
        test_fn: The test function to run
        world_size: Total number of processes
        port: Port to use for distributed communication
        backend: PyTorch distributed backend to use
    """
    try:
        # Setup the distributed environment
        _setup_distributed(rank, world_size, port, backend=backend)

        # Run the actual test function
        test_fn(rank, world_size)

        # Clean up
        _cleanup_distributed()
    except Exception as e:
        print(f"Error in rank {rank}: {e}")
        _cleanup_distributed()
        raise


@pytest.fixture
def mock_2gpu_distributed_env():
    """Mock distributed environment variables and initialization.

    This fixture should be used when testing the underlying ray actors that need torch distributed initialized.
    """
    # Save original environment
    old_env = os.environ.copy()

    # Set required environment variables
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "2"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Create a more sophisticated mock device mesh
    # First, create individual mesh mocks for dp and tp
    dp_mesh = unittest.mock.MagicMock()
    dp_mesh.device_type = "cuda"
    dp_mesh.size.return_value = 1  # Set dp_size

    tp_mesh = unittest.mock.MagicMock()
    tp_mesh.device_type = "cuda"
    tp_mesh.size.return_value = 2  # Set tp_size to match your test case

    # Create the 2D mesh that acts like a dictionary (this is so we can test DTensorPolicyWorker with TP > 1)
    mesh_2d = unittest.mock.MagicMock()
    mesh_2d.__getitem__.side_effect = (
        lambda key: dp_mesh if key == "dp" else tp_mesh if key == "tp" else None
    )
    mesh_2d.device_type = "cuda"

    # Mock dist.is_initialized to prevent actual initialization
    with (
        unittest.mock.patch("torch.distributed.init_process_group") as mock_init,
        unittest.mock.patch("torch.distributed.is_initialized", return_value=True),
        unittest.mock.patch("torch.distributed.get_rank", return_value=0),
        unittest.mock.patch("torch.distributed.get_world_size", return_value=2),
        unittest.mock.patch(
            "torch.distributed.device_mesh.init_device_mesh", return_value=mesh_2d
        ),
    ):
        yield mock_init

    # Restore original environment
    os.environ.clear()
    os.environ.update(old_env)


#######################
# Test Model Fixtures #
#######################


@pytest.fixture(scope="session", autouse=True)
def tiny_llama_model_path():
    """Fixture that returns a path to a tiny llama model with a dummy tokenizer."""
    import shutil

    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    model_path = TEST_ASSETS.TINY_LLAMA_MODEL_PATH
    # hidden_size//num_attention_heads = 32 (smallest value to not error due to vllm paged attention)
    # vocab_size=128256 (so we can re-use llama3.2 1b tokenizer)
    config = LlamaConfig(
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=32,
        num_attention_heads=2,
        vocab_size=128256,
        tie_word_embeddings=False,
        num_key_value_heads=None,
    )
    model = LlamaForCausalLM(config=config)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    shutil.rmtree(model_path, ignore_errors=True)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    del model, tokenizer
    yield model_path


@pytest.fixture(scope="session", autouse=True)
def tiny_llama_tied_model_path():
    """Fixture that returns a path to a tiny llama model with a dummy tokenizer."""
    import shutil

    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    model_path = TEST_ASSETS.TINY_LLAMA_TIED_MODEL_PATH
    # hidden_size//num_attention_heads = 32 (smallest value to not error due to vllm paged attention)
    # vocab_size=128256 (so we can re-use llama3.2 1b tokenizer)
    config = LlamaConfig(
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=32,
        num_attention_heads=2,
        vocab_size=128256,
        tie_word_embeddings=True,
        num_key_value_heads=None,
    )
    model = LlamaForCausalLM(config=config)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    shutil.rmtree(model_path, ignore_errors=True)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    del model, tokenizer
    yield model_path


@pytest.fixture(scope="session", autouse=True)
def tiny_qwen2_model_path():
    """Fixture that returns a path to a tiny llama model with a dummy tokenizer."""
    import shutil

    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    model_path = TEST_ASSETS.TINY_QWEN2_MODEL_PATH
    # hidden_size//num_attention_heads = 32 (smallest value to not error due to vllm paged attention)
    # vocab_size=151936 (so we can re-use qwen2 1.5b tokenizer)
    config = Qwen2Config(
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=32,
        num_attention_heads=2,
        vocab_size=151936,
        tie_word_embeddings=False,
        num_key_value_heads=None,
    )
    model = Qwen2ForCausalLM(config=config)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B")
    shutil.rmtree(model_path, ignore_errors=True)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    del model, tokenizer
    yield model_path


@pytest.fixture(scope="session", autouse=True)
def tiny_qwen3_model_path():
    """Fixture that returns a path to a tiny llama model with a dummy tokenizer."""
    import shutil

    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    model_path = TEST_ASSETS.TINY_QWEN3_MODEL_PATH
    # hidden_size//num_attention_heads = 32 (smallest value to not error due to vllm paged attention)
    # vocab_size=151936 (so we can re-use qwen2 1.5b tokenizer)
    config = Qwen3Config(
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=32,
        num_attention_heads=2,
        vocab_size=151936,
        tie_word_embeddings=False,
        num_key_value_heads=None,
    )
    model = Qwen3ForCausalLM(config=config)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    shutil.rmtree(model_path, ignore_errors=True)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    del model, tokenizer
    yield model_path


@pytest.fixture(scope="session", autouse=True)
def tiny_gemma3_model_path():
    """Fixture that returns a path to a tiny llama model with a dummy tokenizer."""
    import shutil

    from transformers import AutoTokenizer, Gemma3ForCausalLM, Gemma3TextConfig

    model_path = TEST_ASSETS.TINY_GEMMA3_MODEL_PATH
    # hidden_size//num_attention_heads = 32 (smallest value to not error due to vllm paged attention)
    # vocab_size=262144 so we can re-use gemma-3-1b tokenizer
    config = Gemma3TextConfig(
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=32,
        num_attention_heads=2,
        vocab_size=262144,
        tie_word_embeddings=True,
        num_key_value_heads=2,
    )
    model = Gemma3ForCausalLM(config=config)
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    shutil.rmtree(model_path, ignore_errors=True)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    del model, tokenizer
    yield model_path
