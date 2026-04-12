import importlib
import logging

import ray

logger = logging.getLogger(__name__)

SGLANG_WORKER_FQN = "nemo_rl.models.generation.redesign.sglang_worker.SGLangGenerationWorker"

@ray.remote(num_cpus=1, num_gpus=0)
class RouterActor:
    """Starts and owns the sglang router subprocess.

    Runs under SGLANG_EXECUTABLE so it can import sglang_router.
    The driver (SYSTEM env) holds a handle to this actor and retrieves
    (router_ip, router_port) without ever importing sglang_router itself.
    """

    def start(self, router_cfg: dict) -> tuple[str, int]:
        import multiprocessing
        import random

        from sglang_router.launch_router import RouterArgs

        from nemo_rl.models.generation.redesign.misc import run_router
        from nemo_rl.models.generation.redesign.ray_utils import (
            _wrap_ipv6,
            find_available_port,
            get_host_info,
        )

        router_ip = _wrap_ipv6(get_host_info()[1])
        router_port = router_cfg.get("sglang_router_port")
        if router_port is None:
            router_port = find_available_port(random.randint(3000, 4000))

        router_args = RouterArgs()
        router_args.host = router_ip
        router_args.port = router_port
        if router_cfg.get("router_policy") is not None:
            router_args.router_policy = router_cfg["router_policy"]
        router_args.prometheus_port = find_available_port(
            random.randint(4000, 5000)
        )
        router_args.log_level = "warn"
        request_timeout_secs = router_cfg.get(
            "sglang_router_request_timeout_secs"
        )
        if request_timeout_secs is not None:
            router_args.request_timeout_secs = request_timeout_secs

        self._process = multiprocessing.Process(
            target=run_router, args=(router_args,)
        )
        self._process.daemon = True
        self._process.start()
        import time
        time.sleep(3)
        assert self._process.is_alive(), "Router process died on startup"
        return router_ip, router_port

    def stop(self):
        from nemo_rl.models.generation.redesign.misc import terminate_process
        if hasattr(self, "_process"):
            terminate_process(self._process)
