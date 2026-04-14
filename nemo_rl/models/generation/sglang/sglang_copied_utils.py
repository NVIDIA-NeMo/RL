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