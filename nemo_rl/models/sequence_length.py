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

import math


def get_sequence_length_divisibility(
    *,
    context_parallel_size: int,
    tensor_parallel_size: int,
    sequence_parallel: bool,
) -> int:
    """Return the minimum global sequence-length multiple for CP and TP+SP.

    Context parallelism splits the global sequence into ``2 * CP`` balanced
    chunks. Sequence parallelism then splits the CP-local sequence across TP,
    so its equivalent global constraint is ``CP * TP``. Independent
    divisibility constraints combine with an LCM.
    """
    cp_factor = 2 * context_parallel_size if context_parallel_size > 1 else 1
    sequence_parallel_factor = (
        context_parallel_size * tensor_parallel_size
        if tensor_parallel_size > 1 and sequence_parallel
        else 1
    )
    return math.lcm(cp_factor, sequence_parallel_factor)


def validate_sequence_length_divisibility(
    make_sequence_length_divisible_by: int,
    *,
    context_parallel_size: int,
    tensor_parallel_size: int,
    sequence_parallel: bool,
) -> None:
    """Validate the sequence-padding multiple required by CP and TP+SP."""
    minimum_pad_factor = get_sequence_length_divisibility(
        context_parallel_size=context_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
        sequence_parallel=sequence_parallel,
    )

    if (
        make_sequence_length_divisible_by <= 0
        or make_sequence_length_divisible_by % minimum_pad_factor != 0
    ):
        raise ValueError(
            "make_sequence_length_divisible_by "
            f"({make_sequence_length_divisible_by}) must be a positive multiple "
            f"of the minimum pad factor ({minimum_pad_factor}).\n"
            f"Please set policy.make_sequence_length_divisible_by to a positive "
            f"multiple of {minimum_pad_factor}.\n"
            "    - CP requires `2 * context_parallel_size`.\n"
            "    - TP+SP operates on the CP-local sequence and therefore requires "
            "`context_parallel_size * tensor_parallel_size` globally.\n"
            "    - When both constraints apply, their least common multiple is used."
        )
