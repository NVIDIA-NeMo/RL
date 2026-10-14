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
"""The worker extension managed Dynamo engines load for NeMo RL refit."""

import pytest

pytest.importorskip("vllm")

from nemo_rl.models.generation.vllm.dynamo_worker_extension import (  # noqa: E402
    DynamoWorkerExtension,
)
from nemo_rl.utils.staged_payload import encode_staged_payload  # noqa: E402

pytestmark = pytest.mark.vllm


def _stage(ext: DynamoWorkerExtension, payload, *, skip: int | None = None):
    staged = encode_staged_payload(payload)
    for index, data in enumerate(staged.chunks):
        if index != skip:
            ext.nrl_stage_chunk(staged.key, index, len(staged.chunks), data)
    return staged


def test_staged_reshard_plan_reaches_prepare_unchanged() -> None:
    ext = DynamoWorkerExtension()  # no __init__, as when vLLM mixes it in
    received = []
    ext.prepare_nccl_reshard_refit_info = received.append
    plan = {"per_layer_params": {"layers.0": [{"name": "w"}]}, "misc_meta": {}}

    staged = _stage(ext, plan)
    ext.nrl_prepare_nccl_reshard_refit_info_from_stage(staged.key, staged.sha256)

    assert received == [plan]
    # The buffer is consumed: the same key cannot be prepared twice.
    with pytest.raises(RuntimeError, match="No payload is staged"):
        ext.nrl_prepare_nccl_reshard_refit_info_from_stage(staged.key, staged.sha256)


def test_a_new_key_discards_a_partially_staged_payload() -> None:
    ext = DynamoWorkerExtension()
    stale = _stage(ext, {"old": True})
    fresh = _stage(ext, {"new": True})
    received = []
    ext.prepare_nccl_reshard_refit_info = received.append

    with pytest.raises(RuntimeError, match="No payload is staged"):
        ext.nrl_prepare_nccl_reshard_refit_info_from_stage(stale.key, stale.sha256)
    ext.nrl_prepare_nccl_reshard_refit_info_from_stage(fresh.key, fresh.sha256)
    assert received == [{"new": True}]


def test_a_missing_chunk_or_bad_digest_is_refused() -> None:
    ext = DynamoWorkerExtension()
    ext.prepare_nccl_reshard_refit_info = lambda info: None
    payload = {"blob": "x" * (3 << 20), "salt": list(range(1000))}

    staged = _stage(ext, payload, skip=0)
    if len(staged.chunks) > 1:
        with pytest.raises(RuntimeError, match="missing chunks"):
            ext.nrl_prepare_nccl_reshard_refit_info_from_stage(
                staged.key, staged.sha256
            )

    staged = _stage(ext, {"a": 1})
    with pytest.raises(ValueError, match="digest"):
        ext.nrl_prepare_nccl_reshard_refit_info_from_stage(staged.key, "0" * 64)


def test_a_chunk_that_does_not_fit_the_payload_is_refused() -> None:
    ext = DynamoWorkerExtension()
    ext.nrl_stage_chunk("k", 0, 2, "a")
    with pytest.raises(ValueError, match="does not fit"):
        ext.nrl_stage_chunk("k", 2, 2, "b")
    with pytest.raises(ValueError, match="does not fit"):
        ext.nrl_stage_chunk("k", 0, 3, "c")


def test_a_refit_that_reports_failure_raises() -> None:
    # Dynamo discards the RPC's return value, so False would read as success.
    ext = DynamoWorkerExtension()
    seen = []
    ext.nccl_reshard_refit = lambda refit_timeout_s: seen.append(refit_timeout_s)
    with pytest.raises(RuntimeError, match="reported failure"):
        ext.nrl_nccl_reshard_refit(30.0)
    ext.nccl_reshard_refit = lambda refit_timeout_s: True
    assert ext.nrl_nccl_reshard_refit(None) is True
    assert seen == [30.0]


def test_new_methods_do_not_shadow_the_vllm_worker() -> None:
    # vLLM refuses an extension that shadows a Worker attribute.
    from vllm.v1.worker.gpu_worker import Worker

    added = {name for name in vars(DynamoWorkerExtension) if not name.startswith("__")}
    assert "nrl_stage_chunk" in added
    assert not added & set(dir(Worker))
