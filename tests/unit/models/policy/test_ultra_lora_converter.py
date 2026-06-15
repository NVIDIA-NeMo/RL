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

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from examples.converters.convert_megatron_lora_dcp_to_hf_ultra import (
    QKVLayout,
    load_qkv_layout,
    split_qkv_lora_b,
)


def test_split_qkv_lora_b_handles_grouped_query_attention_layout():
    layout = QKVLayout(num_attention_heads=4, num_key_value_heads=2, head_dim=1)
    tensor = torch.tensor(
        [
            [10.0],
            [11.0],
            [12.0],
            [13.0],
            [20.0],
            [21.0],
            [22.0],
            [23.0],
        ]
    )

    q, k, v = split_qkv_lora_b(tensor, layout=layout)

    torch.testing.assert_close(q, torch.tensor([[10.0], [11.0], [20.0], [21.0]]))
    torch.testing.assert_close(k, torch.tensor([[12.0], [22.0]]))
    torch.testing.assert_close(v, torch.tensor([[13.0], [23.0]]))


def test_load_qkv_layout_derives_head_dim_from_config(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": 8192,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
            }
        )
    )

    layout = load_qkv_layout(str(tmp_path))

    assert layout == QKVLayout(
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=256,
    )


def test_split_qkv_lora_b_rejects_unexpected_shape():
    layout = QKVLayout(num_attention_heads=4, num_key_value_heads=2, head_dim=1)

    with pytest.raises(ValueError, match="Unexpected fused QKV"):
        split_qkv_lora_b(torch.zeros(7, 1), layout=layout)
