# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Print the pid of every live generation actor, one per line.

Used by the chaos and recovery functional tests to pick a victim.

WHY NOT `ps`/`pgrep`. Both tests originally matched Ray's process *title*
(``ray::VllmAsyncGenerationWorker``), which Ray sets with setproctitle. That worked on a
2-GPU workstation and found **zero** actors on a GB200 cluster (job 5861743), where the
recovery test reported "expected exactly 2 generation actors, found 0" at train step 3 with
generation demonstrably working. Titles are an implementation detail of the runtime and are
not portable; the GCS actor table is the runtime's own record and is authoritative.

WHY NOT `ray.util.state.list_actors`. That goes through the dashboard's HTTP state server,
and NeMo-RL's ``init_ray`` starts Ray with ``include_dashboard=False``, so it raises
ServerUnavailable. ``ray._private.state.actors()`` reads the GCS directly and needs no
dashboard.

Output: one pid per line, sorted. Exit 1 with a message on stderr if Ray cannot be reached.
"""

import sys

DEFAULT_MATCH = "GenerationWorker"


def main() -> int:
    match = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MATCH

    import ray

    try:
        # address="auto" attaches to the cluster the training job already started. This
        # registers a short-lived extra driver, which is why it shuts down immediately.
        ray.init(address="auto", log_to_driver=False, include_dashboard=False)
    except Exception as exc:  # noqa: BLE001 - the message is the whole point
        print(f"could not attach to a running Ray cluster: {exc}", file=sys.stderr)
        return 1

    try:
        import ray._private.state as rstate

        pids = sorted(
            rec["Pid"]
            for rec in rstate.actors().values()
            if match in rec.get("ActorClassName", "")
            and rec.get("State") == "ALIVE"
            and rec.get("Pid")
        )
    finally:
        ray.shutdown()

    for pid in pids:
        print(pid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
