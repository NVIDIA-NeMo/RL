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

import fnmatch
import logging
from copy import deepcopy
from typing import Any
import os
import time

from nemo_rl.utils.nsys import NRL_NSYS_PROFILE_STEP_RANGE, NRL_NSYS_WORKER_PATTERNS


def get_nsight_config_if_pattern_matches(worker_name: str) -> dict[str, Any]:
    """Check if worker name matches patterns in NRL_NSYS_WORKER_PATTERNS and return nsight config.

    Args:
        worker_name: Name of the worker to check against patterns

    Returns:
        Dictionary containing {"nsight": config} if pattern matches, empty dict otherwise
    """
    assert not (bool(NRL_NSYS_WORKER_PATTERNS) ^ bool(NRL_NSYS_PROFILE_STEP_RANGE)), (
        "Either both NRL_NSYS_WORKER_PATTERNS and NRL_NSYS_PROFILE_STEP_RANGE must be set, or neither. See https://github.com/NVIDIA/NeMo-RL/tree/main/docs/nsys-profiling.md for more details."
    )

    patterns_env = NRL_NSYS_WORKER_PATTERNS
    if not patterns_env:
        return {}

    # Parse CSV patterns
    patterns = [
        pattern.strip() for pattern in patterns_env.split(",") if pattern.strip()
    ]

    # Check if worker name matches any pattern
    for pattern in patterns:
        if fnmatch.fnmatch(worker_name, pattern):
            logging.info(
                f"Nsight profiling enabled for worker '{worker_name}' (matched pattern '{pattern}')"
            )
            # Replace ':' with '_' in worker name and step range for Windows compatibility
            safe_worker_name = worker_name.replace(':', '_')
            safe_step_range = NRL_NSYS_PROFILE_STEP_RANGE.replace(':', '_')
            return {
                "nsight": {
                    "t": "cuda,cudnn,cublas,nvtx",
                    "o": f"'{safe_worker_name}_{safe_step_range}_%p'",
                    "stop-on-exit": "true",
                    # Capture range is required to control the scope of the profile
                    # Profile will only start/stop when torch.cuda.profiler.start()/stop() is called
                    "capture-range": "cudaProfilerApi",
                    "capture-range-end": "stop",
                    "cuda-graph-trace": "node",
                    # "cudabacktrace": "all",
                    # "python-backtrace": "cuda",
                }
            }

    return {}

def log_worker_initialization_info(worker_name: str = "Worker") -> None:
    """Log worker initialization information including process ID and logical ranks.
    
    This function logs the process ID and available logical ranks (tensor parallel, 
    pipeline parallel, context parallel) to help map nsys profile files back to 
    specific workers and their roles in distributed training.
    
    Args:
        worker_name: Name/type of the worker being initialized
    """
    pid = os.getpid()
    
    # Get basic distributed info if available
    try:
        import torch
        if torch.distributed.is_initialized():
            global_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            
            # Try to get Megatron parallel ranks if available
            try:
                from megatron.core.parallel_state import (
                    get_tensor_model_parallel_rank,
                    get_pipeline_model_parallel_rank,
                    get_context_parallel_rank,
                    get_tensor_model_parallel_world_size,
                    get_pipeline_model_parallel_world_size,
                    get_context_parallel_world_size,
                )
                
                tp_rank = get_tensor_model_parallel_rank()
                pp_rank = get_pipeline_model_parallel_rank()
                cp_rank = get_context_parallel_rank()
                tp_size = get_tensor_model_parallel_world_size()
                pp_size = get_pipeline_model_parallel_world_size()
                cp_size = get_context_parallel_world_size()
                
                log_message = (
                    f"🚀 {worker_name} initialized: PID={pid}, GlobalRank={global_rank}/{world_size}, "
                    f"TP={tp_rank}/{tp_size}, PP={pp_rank}/{pp_size}, CP={cp_rank}/{cp_size}"
                )
            except ImportError:
                # Megatron not available, just log basic info
                log_message = f"🚀 {worker_name} initialized: PID={pid}, GlobalRank={global_rank}/{world_size}"
        else:
            # Distributed not initialized
            log_message = f"🚀 {worker_name} initialized: PID={pid} (distributed not initialized)"
            
    except ImportError:
        # PyTorch not available
        log_message = f"🚀 {worker_name} initialized: PID={pid} (torch not available)"
    
    # Write to both stdout and a dedicated log file to bypass Ray deduplication
    print(log_message)
    
    # Also write to a dedicated worker mapping file
    try:
        log_dir = os.environ.get('WORKER_LOG_DIR', '/tmp')
        
        # Include SLURM job ID in filename if available
        job_id = os.environ.get('SLURM_JOB_ID', 'unknown')
        
        # Allow custom run identifier for A/B testing (e.g., baseline vs optimized)
        run_id = os.environ.get('RUN_IDENTIFIER', '')
        if run_id:
            worker_log_file = os.path.join(log_dir, f'{job_id}_{run_id}_worker_pid_mapping.log')
        else:
            worker_log_file = os.path.join(log_dir, f'{job_id}_worker_pid_mapping.log')
        
        with open(worker_log_file, 'a') as f:
            f.write(f"{time.time():.6f}: {log_message}\n")
            f.flush()
    except Exception:
        # If file writing fails, don't crash the worker
        pass


def recursive_merge_options(
    default_options: dict[str, Any], extra_options: dict[str, Any]
) -> dict[str, Any]:
    """Recursively merge extra options into default options using OmegaConf.

    Args:
        default_options: Default options dictionary (lower precedence)
        extra_options: Extra options provided by the caller (higher precedence)

    Returns:
        Merged options dictionary with extra_options taking precedence over default_options
    """
    # Convert to OmegaConf DictConfig for robust merging
    default_conf = deepcopy(default_options)
    extra_conf = deepcopy(extra_options)

    def recursive_merge_dict(base, incoming):
        """Recursively merge incoming dict into base dict, with incoming taking precedence."""
        if isinstance(incoming, dict):
            for k, v in incoming.items():
                if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                    # Both are dicts, recurse
                    recursive_merge_dict(base[k], v)
                else:
                    # Incoming takes precedence (overwrites base) - handles all cases:
                    # - scalar replacing dict, dict replacing scalar, scalar replacing scalar
                    base[k] = deepcopy(v)

    # Handle special nsight configuration transformation (_nsight -> nsight) early
    # so that extra_options can properly override the transformed default
    # https://github.com/ray-project/ray/blob/3c4a5b65dd492503a707c0c6296820228147189c/python/ray/runtime_env/runtime_env.py#L345
    if "runtime_env" in default_conf and isinstance(default_conf["runtime_env"], dict):
        runtime_env = default_conf["runtime_env"]
        if "_nsight" in runtime_env and "nsight" not in runtime_env:
            runtime_env["nsight"] = runtime_env["_nsight"]
            del runtime_env["_nsight"]

    # Merge in place
    recursive_merge_dict(base=default_conf, incoming=extra_conf)

    return default_conf
