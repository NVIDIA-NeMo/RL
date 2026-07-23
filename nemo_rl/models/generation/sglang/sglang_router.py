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

import logging
import multiprocessing
import time

import ray

from nemo_rl.distributed.virtual_cluster import (
    DEFAULT_SGLANG_PROMETHEUS_PORT_RANGE_HIGH,
    DEFAULT_SGLANG_PROMETHEUS_PORT_RANGE_LOW,
    DEFAULT_SGLANG_ROUTER_PORT_RANGE_HIGH,
    DEFAULT_SGLANG_ROUTER_PORT_RANGE_LOW,
    _get_free_port_local,
)
from nemo_rl.models.generation.sglang.config import SGLangRouterConfig
from nemo_rl.models.generation.sglang.utils.ip_port_utils import _wrap_ipv6
from nemo_rl.models.generation.sglang.utils.ray_utils import get_host_info
from nemo_rl.models.generation.sglang.utils.startup_deadline import (
    SGLangStartupDeadline,
)
from nemo_rl.utils.venvs import make_actor_runtime_env

logger = logging.getLogger(__name__)


def run_router(args):
    from sglang_router.launch_router import launch_router

    launch_router(args)


@ray.remote(num_cpus=1, num_gpus=0)
class RouterActor:
    """Starts and owns the sglang router subprocess.

    Runs under SGLANG_EXECUTABLE so it can import sglang_router.
    The driver (SYSTEM env) holds a handle to this actor and retrieves
    (router_ip, router_port) without ever importing sglang_router itself.
    """

    def init(self, router_cfg: SGLangRouterConfig) -> tuple[str, int]:
        from sglang_router.launch_router import RouterArgs

        router_ip = _wrap_ipv6(get_host_info()[1])
        router_port = router_cfg.get("sglang_router_port")
        if router_port is None:
            router_port = _get_free_port_local(
                DEFAULT_SGLANG_ROUTER_PORT_RANGE_LOW,
                DEFAULT_SGLANG_ROUTER_PORT_RANGE_HIGH,
            )

        router_args = RouterArgs()
        router_args.host = router_ip
        router_args.port = router_port
        if router_cfg.get("router_policy") is not None:
            router_args.router_policy = router_cfg["router_policy"]
        router_args.prometheus_port = _get_free_port_local(
            DEFAULT_SGLANG_PROMETHEUS_PORT_RANGE_LOW,
            DEFAULT_SGLANG_PROMETHEUS_PORT_RANGE_HIGH,
        )
        router_args.log_level = "warn"
        request_timeout_secs = router_cfg.get("sglang_router_request_timeout_secs")
        if request_timeout_secs is not None:
            router_args.request_timeout_secs = request_timeout_secs

        self.start(router_args)
        return router_ip, router_port

    def start(self, router_args) -> None:
        self._process = multiprocessing.Process(target=run_router, args=(router_args,))
        self._process.daemon = True
        self._process.start()

        time.sleep(3)
        assert self._process.is_alive(), "Router process died on startup"

    def stop(self, timeout: float = 1.0) -> None:
        """Terminate the router subprocess gracefully, with forced kill as fallback.

        Args:
            timeout: Seconds to wait for graceful termination before forcing kill.
        """
        if not hasattr(self, "_process") or not self._process.is_alive():
            return

        self._process.terminate()
        self._process.join(timeout=timeout)
        if self._process.is_alive():
            self._process.kill()
            self._process.join()


def _start_router(
    router_cfg: SGLangRouterConfig,
    *,
    deadline: SGLangStartupDeadline,
) -> tuple[str, int, ray.actor.ActorHandle | None]:
    """Start sgl router, returning ``(router_ip, router_port, actor_handle)``.

    When ``router_cfg.use_external_router`` is True, reuse the externally
    launched router at ``(sglang_router_ip, sglang_router_port)`` and return
    ``actor_handle=None`` (we do not own that router and must not terminate it).
    Otherwise spawn a ``RouterActor`` in sglang env to own the router process.
    """
    if router_cfg.get("use_external_router"):
        assert (
            router_cfg.get("sglang_router_ip") is not None
            and router_cfg.get("sglang_router_port") is not None
        ), (
            "sglang_router_ip and sglang_router_port must both be set "
            "when use_external_router is True"
        )
        return router_cfg["sglang_router_ip"], router_cfg["sglang_router_port"], None

    router_actor = RouterActor.options(
        runtime_env=make_actor_runtime_env(
            "nemo_rl.models.generation.sglang.sglang_worker.SGLangGenerationWorker"
        ),
    ).remote()
    try:
        router_ip, router_port = deadline.ray_get(
            router_actor.init.remote(dict(router_cfg)),
            stage="starting the rollout router",
            cancel_on_error=True,
        )
    except BaseException:
        try:
            ray.get(router_actor.stop.remote(), timeout=5)
        except Exception as cleanup_exc:
            logger.warning(
                f"Router cleanup after startup failure failed: {cleanup_exc}"
            )
        finally:
            try:
                ray.kill(router_actor, no_restart=True)
            except Exception as cleanup_exc:
                logger.warning(
                    f"Router actor kill after startup failure failed: {cleanup_exc}"
                )
        raise
    logger.info(f"Router launched at {router_ip}:{router_port}")
    return router_ip, router_port, router_actor
