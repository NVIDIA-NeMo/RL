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
from contextlib import contextmanager

import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy.workers.base_policy_worker import AbstractPolicyWorker


def test_reference_logprobs_never_replay_rollout_routes():
    """R3 routes belong to the policy; the reference forward must route live."""
    calls = []

    @contextmanager
    def use_reference_model():
        calls.append("reference_model")
        yield

    def get_logprobs(**kwargs):
        calls.append(kwargs)
        return {"logprobs": torch.zeros(1, 3)}

    worker = AbstractPolicyWorker.__new__(AbstractPolicyWorker)
    worker.use_reference_model = use_reference_model
    worker.get_logprobs = get_logprobs
    data = BatchedDataDict({"input_ids": torch.zeros(1, 3, dtype=torch.long)})

    out = worker.get_reference_policy_logprobs(data=data, micro_batch_size=2)

    assert calls == [
        "reference_model",
        {"data": data, "micro_batch_size": 2, "require_router_replay": False},
    ]
    assert torch.equal(out["reference_logprobs"], torch.zeros(1, 3))
