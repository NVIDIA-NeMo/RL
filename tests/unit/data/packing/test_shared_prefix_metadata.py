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

import pytest

from nemo_rl.data.packing.shared_prefix_metadata import (
    group_id_from_sample_id,
    parse_grouped_sample_id,
)


def test_group_id_from_sample_id_handles_embedded_group_markers() -> None:
    assert group_id_from_sample_id("rollout_g_retry_g15") == "rollout_g_retry"
    assert parse_grouped_sample_id("rollout_g_retry_g15") == (
        "rollout_g_retry",
        15,
    )
    with pytest.raises(ValueError, match="form"):
        group_id_from_sample_id("missing-generation-index")
    with pytest.raises(TypeError, match="strings"):
        parse_grouped_sample_id(1)  # type: ignore[arg-type]
