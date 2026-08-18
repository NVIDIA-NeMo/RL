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

from types import SimpleNamespace

import pytest

from nemo_rl.weight_sync.model_express.megatron import (
    _qkv_descriptor_extras,
    model_express_role_for_mapping,
)


class AutoMapping:
    is_expert = False
    permute_dims = None

    def __init__(self, detected: str):
        self.detected = detected

    def _detect_parallelism_type(self, _module):
        return self.detected


class RowParallelMapping:
    is_expert = False


class ColumnParallelMapping:
    is_expert = False


class DirectMapping:
    is_expert = False


class QKVMapping:
    is_expert = False


class GatedMLPMapping:
    is_expert = False


@pytest.mark.parametrize(
    ("detected", "tensor_ndim", "expected"),
    [
        ("column", 2, "column"),
        ("row", 2, "row"),
        ("row", 1, "replicated"),
        ("replicated", 1, "replicated"),
    ],
)
def test_auto_mapping_uses_owning_module_classification(
    detected, tensor_ndim, expected
):
    assert (
        model_express_role_for_mapping(
            mapping=AutoMapping(detected),
            megatron_module=SimpleNamespace(),
            tensor_ndim=tensor_ndim,
        )
        == expected
    )


def test_row_parallel_bias_is_replicated():
    assert (
        model_express_role_for_mapping(
            mapping=RowParallelMapping(),
            megatron_module=SimpleNamespace(),
            tensor_ndim=1,
        )
        == "replicated"
    )


@pytest.mark.parametrize(
    ("mapping", "expected"),
    [
        (ColumnParallelMapping(), "column"),
        (DirectMapping(), "replicated"),
        (QKVMapping(), "qkv_column"),
        (GatedMLPMapping(), "gated_mlp_column"),
    ],
)
def test_explicit_mapping_roles(mapping, expected):
    assert (
        model_express_role_for_mapping(
            mapping=mapping,
            megatron_module=SimpleNamespace(),
            tensor_ndim=2,
        )
        == expected
    )


def test_qkv_bias_fails_until_alias_format_supports_it():
    with pytest.raises(NotImplementedError, match="weights only"):
        model_express_role_for_mapping(
            mapping=QKVMapping(),
            megatron_module=SimpleNamespace(),
            tensor_ndim=1,
        )


def test_unknown_mapping_fails_instead_of_assuming_replicated():
    class UnknownMapping:
        is_expert = False

    with pytest.raises(NotImplementedError, match="UnknownMapping"):
        model_express_role_for_mapping(
            mapping=UnknownMapping(),
            megatron_module=SimpleNamespace(),
            tensor_ndim=2,
        )


def test_transformed_auto_mapping_subclass_is_not_treated_as_plain_auto_mapping():
    class RMSNorm2ZeroCenteredRMSNormMapping(AutoMapping):
        pass

    with pytest.raises(NotImplementedError, match="RMSNorm2ZeroCenteredRMSNormMapping"):
        model_express_role_for_mapping(
            mapping=RMSNorm2ZeroCenteredRMSNormMapping("replicated"),
            megatron_module=SimpleNamespace(),
            tensor_ndim=1,
        )


def test_qkv_metadata_supports_fewer_kv_heads_than_tp_ranks():
    extras = _qkv_descriptor_extras(
        transformer_config=SimpleNamespace(
            num_attention_heads=64,
            num_query_groups=2,
            kv_channels=128,
        ),
        tensor_parallel_size=8,
        global_rows=8704,
    )

    assert extras == {
        "qkv_interleave": "by_head",
        "head_dim": "128",
        "num_heads": "64",
        "num_kv_heads": "2",
    }


def test_qkv_metadata_keeps_local_counts_when_both_are_divisible():
    extras = _qkv_descriptor_extras(
        transformer_config=SimpleNamespace(
            num_attention_heads=32,
            num_query_groups=8,
            kv_channels=64,
        ),
        tensor_parallel_size=8,
        global_rows=3072,
    )

    assert extras["num_heads_local"] == "4"
    assert extras["num_kv_heads_local"] == "1"


def test_qkv_metadata_rejects_rows_that_disagree_with_head_geometry():
    with pytest.raises(ValueError, match="fused QKV rows"):
        _qkv_descriptor_extras(
            transformer_config=SimpleNamespace(
                num_attention_heads=64,
                num_query_groups=2,
                kv_channels=128,
            ),
            tensor_parallel_size=8,
            global_rows=1,
        )
