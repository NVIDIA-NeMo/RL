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

"""Opt-in synchronized clearing at MXFP8 shared-buffer ownership boundaries.

The parameter all-gather transports BF16(master) through gradient storage.
Clear the allocation before staging masters, and clear each gathered BF16
slice only after TE has consumed it. Never clear before every microbatch:
the same allocation must retain gradients across accumulation. In particular,
a bucket's FP32 gradient view may overlap a sibling's live BF16 gather slice.
"""

import functools
import logging
import os
from collections.abc import Mapping
from typing import Any

import torch

MXFP8_RESET_SHARED_BUFFER_ENV_VAR = "NRL_MXFP8_RESET_SHARED_BUFFER"
G_PATCHED = False
logger = logging.getLogger(__name__)


def mxfp8_buffer_reset_env_vars(config_env: Mapping[str, str] | None) -> dict[str, str]:
    """Forward the exported switch to Ray, preserving explicit worker settings."""
    env_vars = dict(config_env or {})
    if MXFP8_RESET_SHARED_BUFFER_ENV_VAR in os.environ:
        env_vars.setdefault(
            MXFP8_RESET_SHARED_BUFFER_ENV_VAR,
            os.environ[MXFP8_RESET_SHARED_BUFFER_ENV_VAR],
        )
    setting = env_vars.get(MXFP8_RESET_SHARED_BUFFER_ENV_VAR, "0")
    if setting not in ("0", "1"):
        raise ValueError(f"{MXFP8_RESET_SHARED_BUFFER_ENV_VAR} must be 0 or 1")
    return env_vars


@torch.no_grad()
def reset_mxfp8_shared_buffer(tensor: torch.Tensor) -> None:
    """Synchronously zero a shared transport slice at an ownership boundary.

    Callers must first establish that no parameter payload or accumulated
    gradient in this slice is still live. Device synchronization drains CUDA
    work before the clear and completes the clear before another stream uses
    the storage. This is a diagnostic option with an intentional overlap cost.
    """
    if tensor.is_cuda:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Synchronized MXFP8 buffer reset cannot run during CUDA graph capture."
            )
        torch.cuda.synchronize(tensor.device)
    tensor.zero_()
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


def _check_collectives_finished(groups: list[Any]) -> None:
    for group in groups:
        if group.param_gather_handle is not None:
            raise RuntimeError(
                "MXFP8 buffer reset cannot clear an outstanding parameter all-gather."
            )
        if group.grad_reduce_handle is not None:
            raise RuntimeError(
                "MXFP8 buffer reset cannot clear an outstanding gradient reduction."
            )


def _shared_buffer(buffer: Any) -> torch.Tensor | None:
    shared = getattr(buffer, "shared_buffer", None)
    if shared is None or not buffer.ddp_config.reuse_grad_buf_for_mxfp8_param_ag:
        return None
    # Optional Megatron dependency is needed only in the enabled worker path.
    from megatron.core.fp8_utils import is_grouped_mxfp8tensor, is_mxfp8tensor

    if not all(is_mxfp8tensor(p) or is_grouped_mxfp8tensor(p) for p in buffer.params):
        raise RuntimeError(
            "MXFP8 buffer reset requires a bucket containing only MXFP8 parameters."
        )
    if (
        shared.data_ptr() != buffer.grad_data.data_ptr()
        or shared.numel() != buffer.grad_data.numel()
        or shared.dtype != buffer.grad_data.dtype
    ):
        raise RuntimeError("Unsupported MXFP8 shared gradient-buffer layout.")
    return shared


def install_mxfp8_buffer_reset() -> bool:
    """Install the diagnostic reset in this worker when its environment enables it.

    Unset/0 imports no Megatron code and changes no methods. Set before worker
    startup; this process-wide switch is not intended to change during a run.
    """
    global G_PATCHED
    setting = os.environ.get(MXFP8_RESET_SHARED_BUFFER_ENV_VAR, "0")
    if setting not in ("0", "1"):
        raise ValueError(f"{MXFP8_RESET_SHARED_BUFFER_ENV_VAR} must be 0 or 1")
    if setting == "0":
        return False
    if G_PATCHED:
        return True

    # These optional dependencies are imported only for enabled Megatron workers.
    from megatron.core.distributed.distributed_data_parallel import (
        DistributedDataParallel,
    )
    from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBucketGroup
    from megatron.core.fp8_utils import is_grouped_mxfp8tensor, is_mxfp8tensor
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    stage = DistributedOptimizer._copy_main_params_to_param_buffer
    post_sync = _ParamAndGradBucketGroup._post_param_sync
    zero_grad = DistributedDataParallel.zero_grad_buffer

    @functools.wraps(stage)
    def stage_with_reset(optimizer: Any) -> Any:
        if (
            optimizer.config.reuse_grad_buf_for_mxfp8_param_ag
            and not optimizer.is_stub_optimizer
        ):
            groups = [
                group
                for model_groups in optimizer.per_model_bucket_groups.values()
                for group in model_groups
            ]
            _check_collectives_finished(groups)
            shared_buffers = [
                shared
                for buffer in optimizer.buffers
                if (shared := _shared_buffer(buffer)) is not None
            ]
            for shared in shared_buffers:
                reset_mxfp8_shared_buffer(shared)
        # Write fresh BF16 masters AFTER clearing; never erase the gather source.
        return stage(optimizer)

    @functools.wraps(post_sync)
    def post_sync_with_reset(group: Any) -> Any:
        slices = []
        if group.ddp_config.reuse_grad_buf_for_mxfp8_param_ag:
            if not group.ddp_config.use_distributed_optimizer:
                raise RuntimeError(
                    "MXFP8 buffer reset requires the distributed optimizer."
                )
            for bucket in group.buckets:
                if not any(
                    is_mxfp8tensor(p) or is_grouped_mxfp8tensor(p)
                    for p in bucket.params
                ):
                    continue
                if not all(
                    is_mxfp8tensor(p) or is_grouped_mxfp8tensor(p)
                    for p in bucket.params
                ):
                    raise RuntimeError(
                        "MXFP8 buffer reset cannot clear mixed parameter buckets."
                    )
                if bucket.param_data is None or (
                    bucket.param_data.untyped_storage().data_ptr()
                    != bucket.grad_data.untyped_storage().data_ptr()
                ):
                    raise RuntimeError(
                        "Unsupported MXFP8 shared parameter-buffer layout."
                    )
                slices.append(bucket.param_data)
        result = post_sync(group)
        # The original method first consumes gathered BF16 into TE weights.
        # Clear ONLY these transport views, never the whole grad allocation:
        # a sibling all-gather can still own another portion of that allocation.
        for tensor in slices:
            reset_mxfp8_shared_buffer(tensor)
        return result

    @functools.wraps(zero_grad)
    def zero_grad_with_reset(ddp: Any) -> Any:
        shared_buffers = []
        if ddp.ddp_config.reuse_grad_buf_for_mxfp8_param_ag:
            _check_collectives_finished(
                ddp.bucket_groups + ddp.expert_parallel_bucket_groups
            )
            shared_buffers = [
                shared
                for buffer in ddp.buffers + ddp.expert_parallel_buffers
                if (shared := _shared_buffer(buffer)) is not None
            ]
        result = zero_grad(ddp)
        # This is the existing start-of-step reset, not a microbatch callback.
        for shared in shared_buffers:
            reset_mxfp8_shared_buffer(shared)
        return result

    DistributedOptimizer._copy_main_params_to_param_buffer = stage_with_reset
    _ParamAndGradBucketGroup._post_param_sync = post_sync_with_reset
    DistributedDataParallel.zero_grad_buffer = zero_grad_with_reset
    G_PATCHED = True
    logger.info(
        "%s=1: synchronized MXFP8 shared-buffer resets enabled before master staging, "
        "after parameter conversion, and at the start of gradient accumulation.",
        MXFP8_RESET_SHARED_BUFFER_ENV_VAR,
    )
    return True
