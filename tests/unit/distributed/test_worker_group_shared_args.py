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
import gc
import time

import ray
from ray.util.queue import Queue

from nemo_rl.distributed.worker_groups import _share_arguments


def test_every_argument_becomes_a_ref_to_the_original_value():
    big = b"a" * (2 << 20)
    small = {"k": "v"}
    args, kwargs = _share_arguments((big, small), {"big": big, "flag": True})
    assert all(isinstance(v, ray.ObjectRef) for v in (*args, *kwargs.values()))
    assert ray.get(list(args)) == [big, small]
    assert ray.get(kwargs["big"]) == big
    assert ray.get(kwargs["flag"]) is True


def test_an_existing_object_ref_is_passed_through_unchanged():
    ref = ray.put("value")
    args, kwargs = _share_arguments((ref,), {"r": ref})
    assert args[0] is ref
    assert kwargs["r"] is ref


def test_a_queue_argument_is_freed_once_nothing_refers_to_it():
    # lm_policy passes a RayQueue as pre_init_communication_queue. Sharing it must
    # not keep its actor alive.
    queue = Queue()
    actor_id = queue.actor._actor_id.hex()
    args, kwargs = _share_arguments((), {"pre_init_communication_queue": queue})
    del queue, args, kwargs
    gc.collect()

    deadline = time.monotonic() + 30
    state = None
    while time.monotonic() < deadline:
        state = ray._private.state.actors(actor_id)["State"]
        if state == "DEAD":
            break
        time.sleep(0.5)
    assert state == "DEAD"
