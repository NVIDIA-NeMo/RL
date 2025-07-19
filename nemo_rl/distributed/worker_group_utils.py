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
            return {
                "nsight": {
                    "t": "cuda,cudnn,cublas,nvtx",
                    "o": f"'{worker_name}_{NRL_NSYS_PROFILE_STEP_RANGE}_%p'",
                    "stop-on-exit": "true",
                    # Capture range is required to control the scope of the profile
                    # Profile will only start/stop when torch.cuda.profiler.start()/stop() is called
                    "capture-range": "cudaProfilerApi",
                    "capture-range-end": "stop",
                }
            }

    return {}


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
