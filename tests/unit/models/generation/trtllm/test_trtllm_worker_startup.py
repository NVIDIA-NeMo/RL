# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""A real TRT-LLM worker comes up and generates under a Slurm launcher env.

`import tensorrt_llm` runs MPI_Init, and Open MPI reads the srun step's env as
"this process belongs to that step": with SLURM_JOBID / SLURM_NODELIST set it
aborts MPI_Init or cannot spawn the MPI workers a TP>1 engine needs. Every
actor inherits the driver's env, and TRT-LLM's own GPU workers inherit it
again, so this starts a TP=2 engine with those variables set on the driver --
it only comes up if init_ray's worker_process_setup_hook strips them first.
"""

import socket
from pathlib import Path

import pytest
import ray
import torch
from omegaconf import OmegaConf

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.trtllm.trtllm_generation import TrtllmGeneration
from nemo_rl.utils.config import load_config

pytestmark = pytest.mark.trtllm

_EXEMPLAR = (
    Path(__file__).parents[5] / "examples" / "configs" / "grpo_math_1B_trtllm.yaml"
)
# Already in CI's HF cache: the vLLM generation tests load it.
_MODEL = "Qwen/Qwen2.5-0.5B"
_TP = 2


@pytest.fixture(scope="module", autouse=True)
def _ray_cluster():
    # The directory's conftest no-ops the session Ray fixture for the
    # direct-server tests, but this one drives Ray actors. Init before the test
    # body sets the Slurm env so init_ray's worker setup hook still strips it.
    started_here = not ray.is_initialized()
    if started_here:
        init_ray()
    yield
    if started_here:
        ray.shutdown()


_MAX_NEW_TOKENS = 8


def _generation_config(tokenizer):
    """The exemplar's generation block, shrunk to one small TP=2 engine."""
    master = load_config(_EXEMPLAR)
    master.policy.model_name = _MODEL
    master.policy.max_total_sequence_length = 256
    # Only this block: the rest of the exemplar uses resolvers the training
    # entry points register.
    config = OmegaConf.to_container(master.policy.generation, resolve=True)
    config["model_name"] = _MODEL
    config["max_new_tokens"] = _MAX_NEW_TOKENS
    config["colocated"]["enabled"] = False
    config["trtllm_cfg"]["tensor_parallel_size"] = _TP
    config["trtllm_cfg"]["gpu_memory_utilization"] = 0.3
    config["trtllm_cfg"]["max_batch_size"] = 8
    config["trtllm_cfg"]["max_num_tokens"] = 256
    return configure_generation_config(config, tokenizer, is_eval=True)


@pytest.mark.skipif(
    torch.cuda.device_count() < _TP, reason=f"needs {_TP} GPUs for a TP={_TP} engine"
)
# A failed MPI spawn does not raise: it dies in a background thread and the
# engine waits for workers that never come, so bound it to fail, not hang. A
# cold node needs ~4 min (kernel downloads, first model load); warm, ~1 min.
@pytest.mark.timeout(600)
def test_trtllm_tp2_worker_starts_under_slurm_launcher_env(monkeypatch):
    # What srun hands the processes of a step. Set on the driver, so the worker
    # group forwards it to every TRT-LLM actor, as on a real Slurm job.
    for name, value in {
        "SLURM_JOBID": "1",
        "SLURM_JOB_ID": "1",
        "SLURM_NODELIST": socket.gethostname(),
        "SLURM_JOB_NODELIST": socket.gethostname(),
        "SLURM_STEP_ID": "0",
        "SLURM_STEPID": "0",
    }.items():
        monkeypatch.setenv(name, value)

    tokenizer = get_tokenizer({"name": _MODEL})
    config = _generation_config(tokenizer)
    cluster = RayVirtualCluster(
        bundle_ct_per_node_list=[_TP],
        use_gpus=True,
        max_colocated_worker_groups=1,
        num_gpus_per_node=_TP,
        name="trtllm-launcher-env-test-cluster",
    )
    generation = None
    try:
        generation = TrtllmGeneration(cluster, config)

        prompts = ["The capital of France is", "1 + 1 ="]
        encodings = tokenizer(
            prompts, padding=True, return_tensors="pt", padding_side="right"
        )
        input_lengths = encodings["attention_mask"].sum(dim=1).to(torch.int32)
        outputs = generation.generate(
            BatchedDataDict(
                {"input_ids": encodings["input_ids"], "input_lengths": input_lengths}
            ),
            greedy=True,
        )

        assert outputs["output_ids"].shape[0] == len(prompts)
        assert torch.all(outputs["generation_lengths"] > 0)
        assert torch.all(outputs["generation_lengths"] <= _MAX_NEW_TOKENS)
    finally:
        if generation is not None:
            generation.shutdown()
        cluster.shutdown()
