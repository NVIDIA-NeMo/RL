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

from unittest.mock import patch

from nemo_rl.utils.sequence_length_generator import get_sequence_length_generator


def test_sequence_length_generator_rounds_normal_sample():
    generator = get_sequence_length_generator({"mean": 8, "std": 2})

    with patch(
        "nemo_rl.utils.sequence_length_generator.np.random.normal",
        return_value=9.6,
    ):
        assert generator(123) == 10


def test_sequence_length_generator_clamps_to_minimum_one():
    generator = get_sequence_length_generator({"mean": 1, "std": 10})

    with patch(
        "nemo_rl.utils.sequence_length_generator.np.random.normal",
        return_value=-5.0,
    ):
        assert generator(0) == 1


def test_sequence_length_generator_accepts_optional_idx():
    generator = get_sequence_length_generator({"mean": 5, "std": 0})

    assert generator() == 5
