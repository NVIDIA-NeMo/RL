# Copyright 2023-2024 SGLang Team
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
# ==============================================================================
"""Standalone utility functions copied from the SGLang project.

This module contains utility functions that were originally part of the SGLang
repository (https://github.com/sgl-project/sglang). They have been copied here
to avoid requiring sglang as a runtime dependency for weight refitting functionality.

IMPORTANT: This module should NOT contain any imports from the sglang package.
All functions are standalone and self-contained.

Each function includes a permalink to its original source in the SGLang repository.
These functions were copied from sglang version 0.5.2.
"""

import io
from multiprocessing.reduction import ForkingPickler
from typing import Callable, Union

import pybase64
import torch
from torch.multiprocessing import reductions


class MultiprocessingSerializer:  # pragma: no cover
    """Serialize/deserialize Python objects using ForkingPickler for IPC.

    This class enables serialization of objects (including CUDA tensors with IPC
    handles) for transfer between processes via HTTP or other mechanisms.

    Original source (sglang v0.5.2):
    https://github.com/sgl-project/sglang/blob/v0.5.2/python/sglang/srt/utils.py#L589-L623
    """

    @staticmethod
    def serialize(obj, output_str: bool = False):
        """Serialize a Python object using ForkingPickler.

        Args:
            obj: The object to serialize.
            output_str (bool): If True, return a base64-encoded string instead of raw bytes.

        Returns:
            bytes or str: The serialized object.
        """
        buf = io.BytesIO()
        ForkingPickler(buf).dump(obj)
        buf.seek(0)
        output = buf.read()

        if output_str:
            # Convert bytes to base64-encoded string
            output = pybase64.b64encode(output).decode("utf-8")

        return output

    @staticmethod
    def deserialize(data):
        """Deserialize a previously serialized object.

        Args:
            data (bytes or str): The serialized data, optionally base64-encoded.

        Returns:
            The deserialized Python object.
        """
        if isinstance(data, str):
            # Decode base64 string to bytes
            data = pybase64.b64decode(data, validate=True)

        return ForkingPickler.loads(data)


def monkey_patch_torch_reductions():  # pragma: no cover
    """Monkey patch torch multiprocessing reductions to use GPU UUIDs.

    This patch modifies PyTorch's CUDA tensor IPC mechanism to use GPU UUIDs
    instead of device indices. This enables proper weight transfer between
    processes that may have different CUDA_VISIBLE_DEVICES configurations.

    The patch is idempotent - calling it multiple times is safe.

    This is a workaround before PyTorch https://github.com/pytorch/pytorch/pull/149248
    is merged and released.

    Original source (sglang v0.5.2):
    https://github.com/sgl-project/sglang/blob/v0.5.2/python/sglang/srt/patch_torch.py#L20-L33
    """
    if hasattr(reductions, "_reduce_tensor_original"):
        return

    reductions._reduce_tensor_original = reductions.reduce_tensor
    reductions._rebuild_cuda_tensor_original = reductions.rebuild_cuda_tensor

    reductions.reduce_tensor = _reduce_tensor_modified
    reductions.rebuild_cuda_tensor = _rebuild_cuda_tensor_modified

    reductions.init_reductions()


# The signature has not been changed for years, and we will not need this when
# the next version is released, so it looks safe to use a constant.
# Original source (sglang v0.5.2):
# https://github.com/sgl-project/sglang/blob/v0.5.2/python/sglang/srt/patch_torch.py#L36
_REDUCE_TENSOR_ARG_DEVICE_INDEX = 6


def _reduce_tensor_modified(*args, **kwargs):  # pragma: no cover
    """Modified reduce_tensor that stores GPU UUID instead of device index.

    Original source (sglang v0.5.2):
    https://github.com/sgl-project/sglang/blob/v0.5.2/python/sglang/srt/patch_torch.py#L39-L43
    """
    output_fn, output_args = reductions._reduce_tensor_original(*args, **kwargs)
    output_args = _modify_tuple(
        output_args, _REDUCE_TENSOR_ARG_DEVICE_INDEX, _device_to_uuid
    )
    return output_fn, output_args


def _rebuild_cuda_tensor_modified(*args):  # pragma: no cover
    """Modified rebuild_cuda_tensor that accepts GPU UUID or device index.

    Original source (sglang v0.5.2):
    https://github.com/sgl-project/sglang/blob/v0.5.2/python/sglang/srt/patch_torch.py#L46-L48
    """
    args = _modify_tuple(args, _REDUCE_TENSOR_ARG_DEVICE_INDEX, _device_from_maybe_uuid)
    return reductions._rebuild_cuda_tensor_original(*args)


def _device_to_uuid(device: int) -> str:  # pragma: no cover
    """Convert a device index to its UUID string.

    Original source (sglang v0.5.2):
    https://github.com/sgl-project/sglang/blob/v0.5.2/python/sglang/srt/patch_torch.py#L51-L52
    """
    return str(torch.cuda.get_device_properties(device).uuid)


def _device_from_maybe_uuid(
    device_maybe_uuid: Union[int, str],
) -> int:  # pragma: no cover
    """Convert a device UUID string or index to a device index.

    Args:
        device_maybe_uuid: Either an integer device index or a UUID string.

    Returns:
        The integer device index.

    Raises:
        Exception: If the UUID doesn't match any available device.

    Original source (sglang v0.5.2):
    https://github.com/sgl-project/sglang/blob/v0.5.2/python/sglang/srt/patch_torch.py#L55-L65
    """
    if isinstance(device_maybe_uuid, int):
        return device_maybe_uuid

    if isinstance(device_maybe_uuid, str):
        for device in range(torch.cuda.device_count()):
            if str(torch.cuda.get_device_properties(device).uuid) == device_maybe_uuid:
                return device
        raise Exception("Invalid device_uuid=" + device_maybe_uuid)

    raise Exception(f"Unknown type: {device_maybe_uuid=}")


def _modify_tuple(t, index: int, modifier: Callable):  # pragma: no cover
    """Create a new tuple with one element modified by a function.

    Original source (sglang v0.5.2):
    https://github.com/sgl-project/sglang/blob/v0.5.2/python/sglang/srt/patch_torch.py#L68-L69
    """
    return *t[:index], modifier(t[index]), *t[index + 1 :]
