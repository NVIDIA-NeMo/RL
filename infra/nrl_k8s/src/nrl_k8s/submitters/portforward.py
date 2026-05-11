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
"""Port-forward + Ray Job SDK submitter.

Dev-iteration default. Opens ``kubectl port-forward svc/<head>-svc :8265``
for the duration of each call, submits / tails / queries through Ray's
``JobSubmissionClient``, then tears the forward down.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any, Iterator

from .. import submit as _submit
from . import JobStatusStr, SubmissionHandle

_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class PortForwardSubmitter:
    """Ray Job SDK transport, reached via a short-lived ``kubectl port-forward``.

    ``working_dir=None`` is supported — the Ray job inherits whatever cwd
    the head pod's default entrypoint ran with, which is useful when
    ``infra.launch.codeSource`` is ``image`` or ``lustre`` (code already
    on disk inside the container) but the user still wants Ray's Job API
    for tracking.

    Env vars are **inlined as shell ``export``s** in the entrypoint
    rather than sent through ``runtime_env.env_vars``. The latter
    conflicts with the captured ``os.environ`` that ``ray.init(address=
    "auto")`` later hands to worker actors — Ray raises "Failed to merge
    the Job's runtime env" when it sees the same key in both. Keeping
    the transport's interface uniform with :class:`ExecSubmitter`
    (env via ``export``, not a Ray-specific channel) avoids the whole
    class.
    """

    def submit(
        self,
        cluster_name: str,
        namespace: str,
        *,
        entrypoint: str,
        run_id: str,
        env_vars: dict[str, str] | None = None,
        working_dir: Path | None = None,
    ) -> SubmissionHandle:
        from ray.job_submission import JobSubmissionClient

        wrapped_entrypoint = _prepend_env_exports(entrypoint, env_vars or {})

        with _submit.dashboard_url(cluster_name, namespace) as dash:
            runtime_env: dict[str, Any] = {}
            if working_dir is not None:
                runtime_env["working_dir"] = str(working_dir)
            client = JobSubmissionClient(dash)
            submission_id = client.submit_job(
                entrypoint=wrapped_entrypoint,
                runtime_env=runtime_env,
                submission_id=run_id,
            )
        return SubmissionHandle(
            kind="ray",
            run_id=submission_id,
            cluster_name=cluster_name,
            namespace=namespace,
            dashboard_url=None,  # port-forward doesn't survive the with-block
        )

    def follow(self, handle: SubmissionHandle) -> Iterator[str]:
        with _submit.dashboard_url(handle.cluster_name, handle.namespace) as dash:
            yield from _submit.tail_job_logs(dash, handle.run_id)

    def status(self, handle: SubmissionHandle) -> JobStatusStr:
        from ray.job_submission import JobSubmissionClient

        with _submit.dashboard_url(handle.cluster_name, handle.namespace) as dash:
            try:
                state = JobSubmissionClient(dash).get_job_status(handle.run_id)
            except Exception:  # noqa: BLE001
                return "unknown"
        return _RAY_STATUS_MAP.get(state, "unknown")

    def stop(self, handle: SubmissionHandle, *, force: bool = False) -> None:
        # Ray's API has no concept of force-kill vs graceful — stop_job asks
        # the head to SIGTERM the driver. We accept and ignore ``force``.
        del force
        from ray.job_submission import JobSubmissionClient

        with _submit.dashboard_url(handle.cluster_name, handle.namespace) as dash:
            JobSubmissionClient(dash).stop_job(handle.run_id)


def _build_ray_status_map() -> dict[Any, JobStatusStr]:
    from ray.job_submission import JobStatus

    return {
        JobStatus.PENDING: "running",
        JobStatus.RUNNING: "running",
        JobStatus.SUCCEEDED: "succeeded",
        JobStatus.FAILED: "failed",
        JobStatus.STOPPED: "stopped",
    }


class _LazyRayStatusMap:
    """Defer importing ``ray`` until first use.

    Ray is a heavy import (~2 s of startup). The CLI has commands that
    never touch a dashboard, so we don't want to pay that cost on every
    invocation just because the module imported.
    """

    _cached: dict[Any, JobStatusStr] | None = None

    def get(self, key: Any, default: JobStatusStr) -> JobStatusStr:
        if self._cached is None:
            self._cached = _build_ray_status_map()
        return self._cached.get(key, default)


_RAY_STATUS_MAP = _LazyRayStatusMap()


def _prepend_env_exports(entrypoint: str, env_vars: dict[str, str]) -> str:
    """Prepend ``export KEY=VAL`` lines to the entrypoint body.

    Ray's submit_job entrypoint runs under /bin/dash by default, so we
    avoid bash-only syntax — each export lives on its own line with
    shell-quoted values. Invalid env keys raise immediately so a typo
    doesn't produce subtle failures deep in the job.
    """
    if not env_vars:
        return entrypoint
    lines = []
    for k, v in env_vars.items():
        if not _ENV_KEY_RE.match(k):
            raise ValueError(f"invalid env var name {k!r}")
        lines.append(f"export {k}={shlex.quote(v)}")
    lines.append(entrypoint)
    return "\n".join(lines)


__all__ = ["PortForwardSubmitter"]
