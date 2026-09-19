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

from nemo_rl.models.generation.vllm.model_express import (
    MODEL_EXPRESS_VLLM_WORKER,
    configure_model_express_worker,
)


def test_configure_model_express_worker_uses_vllm_worker_hook():
    vllm_kwargs = {"additional_config": {"existing": True}}

    configure_model_express_worker({"refit_transport": "model_express"}, vllm_kwargs)

    assert vllm_kwargs == {
        "additional_config": {"existing": True},
        "worker_cls": MODEL_EXPRESS_VLLM_WORKER,
    }


def test_configure_model_express_worker_ignores_other_transport():
    vllm_kwargs = {}

    configure_model_express_worker({"refit_transport": "nixl"}, vllm_kwargs)

    assert vllm_kwargs == {}


def test_configure_model_express_worker_rejects_custom_worker():
    with pytest.raises(ValueError, match="worker_cls to be unset"):
        configure_model_express_worker(
            {"refit_transport": "model_express"},
            {"worker_cls": "custom.Worker"},
        )
