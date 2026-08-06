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

"""Print the pid of every live generation actor, one per line, on stdout.

Everything else goes to stderr, so callers can do `$(... 2>/dev/null)`.

Used by the chaos and recovery functional tests to pick a victim.

WHY NOT `ps`/`pgrep`. Both tests originally matched Ray's process *title*
(``ray::VllmAsyncGenerationWorker``), set via setproctitle. That worked on a 2-GPU
workstation and found zero actors on a GB200 cluster. Titles are a runtime implementation
detail; the GCS actor table is the runtime's own record of which pid is which actor.

WHY NOT ``ray.util.state.list_actors``. It goes through the dashboard HTTP state server,
which is not reachable in every configuration. ``ray._private.state.actors()`` reads the
GCS directly.

WHY IT PRINTS SO MUCH TO STDERR. On job 5866390 this returned zero pids and said nothing,
which left no way to tell "no actors" from "wrong field name" from "wrong state string".
It now always reports what the actor table actually contained, so one run answers that.
"""

import os
import sys

DEFAULT_MATCH = "GenerationWorker"


def _err(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main() -> int:
    match = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MATCH

    import ray

    address = os.environ.get("RAY_ADDRESS", "auto")
    try:
        ray.init(address=address, log_to_driver=False, include_dashboard=False)
    except Exception as exc:  # noqa: BLE001 - the message is the whole point
        _err(f"[actors] could not attach to Ray at address={address!r}: {exc}")
        _err(
            "[actors] if the training job is still up, its cluster may use a non-default"
        )
        _err("[actors] temp dir; try RAY_ADDRESS=<ip:port> or check `ray status`.")
        return 1

    try:
        import ray._private.state as rstate

        table = rstate.actors()
        _err(f"[actors] GCS actor table: {len(table)} entries")

        rows = []
        for rec in table.values():
            rows.append(
                (
                    rec.get("ActorClassName", "<no ActorClassName>"),
                    rec.get("State", "<no State>"),
                    rec.get("Pid", 0),
                    rec.get("Name", "") or "",
                )
            )

        # Always dump the table. The whole point is to stop guessing at field values.
        for cls, state, pid, name in sorted(rows):
            _err(f"[actors]   class={cls!r} state={state!r} pid={pid} name={name!r}")

        matched = [
            (cls, state, pid)
            for cls, state, pid in ((r[0], r[1], r[2]) for r in rows)
            if match in cls
        ]
        _err(f"[actors] {len(matched)} entries match {match!r}")

        # Deliberately permissive about State: a shard that is RESTARTING or PENDING is
        # still a real process worth reporting, and over-filtering here is exactly what
        # produced a silent empty result before. Pid 0 means Ray has no process for it,
        # so those cannot be killed and are excluded, but they are reported above.
        pids = sorted({pid for _cls, _state, pid in matched if pid})
        if not pids:
            _err(f"[actors] no killable pid for any actor matching {match!r}")
    finally:
        ray.shutdown()

    for pid in pids:
        print(pid)
    return 0 if pids else 2


if __name__ == "__main__":
    raise SystemExit(main())
