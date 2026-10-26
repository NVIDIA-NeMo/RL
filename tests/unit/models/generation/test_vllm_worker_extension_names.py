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

"""The worker extension must not shadow anything on vLLM's own Worker.

``WorkerBase.init_worker`` walks ``dir(worker_extension_cls)`` and asserts
that no public attribute already exists on the worker class, so a name that
vLLM adds upstream (``synchronize_device`` in 0.29, vllm-project/vllm#52914)
turns every engine start into an ``AssertionError``. That only shows up on a
GPU job today; this mirrors vLLM's check so it fails at unit-test time.
"""

import pytest

pytestmark = pytest.mark.vllm


@pytest.mark.parametrize(
    "extension_name",
    [
        "VllmInternalWorkerExtension",
        "VllmInternalWorkerExtensionWithCheckpointEngine",
    ],
)
def test_worker_extension_does_not_shadow_vllm_worker_attributes(extension_name):
    from vllm.v1.worker.gpu_worker import Worker

    from nemo_rl.models.generation.vllm import vllm_backend

    extension = getattr(vllm_backend, extension_name)
    # Same predicate vLLM applies in WorkerBase.init_worker.
    conflicts = sorted(
        attr
        for attr in dir(extension)
        if not attr.startswith("__") and hasattr(Worker, attr)
    )
    assert not conflicts, (
        f"{extension_name} shadows vLLM Worker attribute(s) {conflicts}; vLLM "
        "refuses to load the extension when this happens. Rename them."
    )
