# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Per-process setup for the Ray workers init_ray launches.

init_ray installs :func:`ray_worker_process_setup` as the job's
``worker_process_setup_hook`` by module path, so Ray imports this module in
every worker process of the job -- every actor venv, including the workers
vLLM and TRT-LLM start themselves -- before any actor code runs. Be careful
with imports here: keep to the stdlib. Anything heavier is paid by every
worker at startup, and one that a worker's venv lacks makes that worker fail
to start.
"""

import importlib.util
import os


def ray_worker_process_setup() -> None:
    """The ``worker_process_setup_hook`` init_ray installs."""
    _drop_srun_launcher_env_for_trtllm()


def _drop_srun_launcher_env_for_trtllm() -> None:
    """Drop the srun launcher env in TRT-LLM workers only.

    ``import tensorrt_llm`` runs ``MPI_Init``, which the srun launcher env
    breaks: ``PMIX_*`` makes slurm cancel the whole Ray step when the worker
    exits, and ``SLURM_*`` makes a TP>1 engine fail to spawn its MPI workers.
    ``env_vars`` cannot unset a variable, so drop them in-process. TRT-LLM
    workers set up no telemetry, so nothing is lost.
    """
    if importlib.util.find_spec("tensorrt_llm") is None:
        return
    for key in list(os.environ):
        if key.startswith(("PMIX_", "PMI_", "MPI_", "OMPI_", "SLURM_")):
            os.environ.pop(key, None)
