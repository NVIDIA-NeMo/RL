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

"""Stop a managed Dynamo service once the process that started it is gone.

    python -I -S service_watchdog.py LIFELINE_FD PID GRACE_S

The process that starts service ``PID`` holds the only write end of a pipe whose
read end is ``LIFELINE_FD`` and never writes to it, so a read returns EOF only
once that process has exited, however it exited: SIGKILL included. The watchdog
then sends the service's process group (``PID`` leads it) SIGTERM, and SIGKILL if
the service still runs ``GRACE_S`` seconds later. On a clean shutdown the
starting process stops the service itself and then kills the watchdog.

Standard library only: it runs isolated from any virtual environment.
"""

import os
import signal
import sys
import time


def _start_time(pid: int) -> str | None:
    """``pid``'s start time while it runs; None once it has exited."""
    try:
        with open(f"/proc/{pid}/stat") as stat:
            # The command name (field 2) may hold spaces; it ends at the last ")".
            state, *fields = stat.read().rsplit(")", 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    return None if state == "Z" else fields[18]  # field 22, starttime


def _signal_group(pgid: int, signum: int) -> None:
    try:
        os.killpg(pgid, signum)
    except ProcessLookupError:
        pass


def main() -> None:
    lifeline_fd, pid, grace_s = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
    started = _start_time(pid)
    if started is None:
        return
    while os.read(lifeline_fd, 4096):
        pass
    # A different start time means the service exited and its PID was reused.
    if _start_time(pid) != started:
        return
    _signal_group(pid, signal.SIGTERM)
    deadline = time.monotonic() + grace_s
    while _start_time(pid) == started:
        if time.monotonic() >= deadline:
            _signal_group(pid, signal.SIGKILL)
            return
        time.sleep(0.1)


if __name__ == "__main__":
    main()
