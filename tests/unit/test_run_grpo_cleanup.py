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

from examples import run_grpo


def test_shutdown_runtime_drains_unique_owners_before_ray(monkeypatch):
    events = []

    class Resource:
        def __init__(self, name):
            self.name = name

        def shutdown(self):
            events.append(self.name)

    generation = Resource("generation")
    teacher = Resource("teacher")
    policy = Resource("policy")
    train_cluster = Resource("train_cluster")
    inference_cluster = Resource("inference_cluster")

    monkeypatch.setattr(run_grpo.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(run_grpo.ray, "shutdown", lambda: events.append("ray_shutdown"))

    run_grpo._shutdown_runtime(
        policy,
        generation,
        (train_cluster, inference_cluster, train_cluster),
        {"teacher": teacher, "teacher_alias": teacher},
    )

    assert events == [
        "generation",
        "teacher",
        "policy",
        "train_cluster",
        "inference_cluster",
        "ray_shutdown",
    ]
