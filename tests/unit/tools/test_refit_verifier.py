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

from tools.refit_verifier import (
    _prepare_dynamo_verifier_policy_config,
    _validate_dynamo_verifier_diagnostic_mode,
)


def test_dynamo_verifier_supplies_required_megatron_train_iters():
    policy_config = {"megatron_cfg": {"enabled": True}}

    _prepare_dynamo_verifier_policy_config(policy_config)

    assert policy_config["megatron_cfg"]["train_iters"] == 1


def test_dynamo_verifier_preserves_explicit_megatron_train_iters():
    policy_config = {"megatron_cfg": {"enabled": True, "train_iters": 17}}

    _prepare_dynamo_verifier_policy_config(policy_config)

    assert policy_config["megatron_cfg"]["train_iters"] == 17


def test_dynamo_verifier_allows_direct_load_control():
    _validate_dynamo_verifier_diagnostic_mode("auto", compare_before_refit=True)


def test_dynamo_verifier_rejects_comparing_dummy_weights_before_refit():
    with pytest.raises(
        ValueError,
        match="--compare-before-refit requires --initial-load-format=auto",
    ):
        _validate_dynamo_verifier_diagnostic_mode("dummy", compare_before_refit=True)
