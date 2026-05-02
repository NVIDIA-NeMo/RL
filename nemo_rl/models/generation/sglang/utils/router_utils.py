import logging
import multiprocessing

logger = logging.getLogger(__name__)


def run_router(args):
    try:
        from sglang_router.launch_router import launch_router

        router = launch_router(args)
        if router is None:
            return 1
        return 0
    except Exception:
        # Runs inside a subprocess; surface the full traceback at ERROR level
        # so it isn't filtered by INFO config, and re-raise so the subprocess
        # exits non-zero (caller asserts on ``_process.is_alive()``).
        logger.exception("sglang router failed to launch")
        raise


def terminate_process(process: multiprocessing.Process, timeout: float = 1.0) -> None:
    """Terminate a process gracefully, with forced kill as fallback.

    Args:
        process: The process to terminate
        timeout: Seconds to wait for graceful termination before forcing kill
    """
    if not process.is_alive():
        return

    process.terminate()
    process.join(timeout=timeout)
    if process.is_alive():
        process.kill()
        process.join()
