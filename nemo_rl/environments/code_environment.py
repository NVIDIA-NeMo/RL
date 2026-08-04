# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import ast
import builtins
import multiprocessing as mp
import os
import re
import signal
import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass, replace
from io import IOBase
from multiprocessing.connection import Connection
from pprint import pformat
from types import ModuleType
from typing import Any, Dict, List, NamedTuple, NotRequired, Optional, Tuple, TypedDict

import cloudpickle
import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from nemo_rl.environments.utils import chunk_list_to_workers

try:
    import resource
except ImportError:  # pragma: no cover
    resource = None


SUBPROCESS_STARTUP_GRACE_SECONDS = 2.0


class CodeEnvConfig(TypedDict):
    num_workers: int
    # whether to terminate the execution after expression evaluation
    # if you want to execute multiple rounds of code, set this to False
    # and wrap CodeEnvironment in another environment that terminates the generation
    terminate_on_evaluation: bool
    default_timeout_seconds: NotRequired[float | None]
    default_memory_limit_bytes: NotRequired[int | None]


class CodeEnvMetadata(TypedDict):
    context: Dict[str, Any]  # Hold functions and variables defined in the code
    working_dir: str  # Working directory for file operations
    timeout_seconds: NotRequired[float | None]
    memory_limit_bytes: NotRequired[int | None]


class ExecutionLimits(NamedTuple):
    timeout_seconds: float | None
    memory_limit_bytes: int | None


@dataclass
class CodeExecutionRequest:
    code: str
    context: dict[str, Any]
    working_dir: str
    lookahead: str | None = None
    timeout_seconds: float | None = None
    memory_limit_bytes: int | None = None


@dataclass
class CodeExecutionResponse:
    formatted_result: str
    terminated: bool
    context: dict[str, Any]


def sanitize_object(obj: Any) -> Any:
    """Sanitize objects that are not safe to return through Ray."""
    if isinstance(obj, (IOBase, ModuleType)):
        return repr(obj)
    if isinstance(obj, Mapping):
        return obj.__class__(
            {sanitize_object(k): sanitize_object(v) for k, v in obj.items()}
        )
    if isinstance(obj, Sequence) and not isinstance(obj, str):
        return obj.__class__(sanitize_object(v) for v in obj)
    if hasattr(obj, "__dict__"):
        new_obj = copy(obj)
        new_obj.__dict__ = {
            sanitize_object(k): sanitize_object(v) for k, v in obj.__dict__.items()
        }
        return new_obj
    return obj


def format_result(
    result: Any, code: str | None = None, lookahead: str | None = None
) -> str:
    """Format a code execution result as an environment observation."""
    if result is None:
        return ""

    result = pformat(result)
    multiline = (code and "\n" in code) or "\n" in result
    if multiline:
        formatted_result = f"\n\n<result>\n{result}\n</result>"
    else:
        formatted_result = f"<result>{result}</result>"

    if lookahead and formatted_result.startswith(lookahead):
        # The generation may look like "</code>\n" if ">\n" is a single token.
        # We trim \n from the result if the model has already generated it.
        formatted_result = formatted_result[len(lookahead) :]

    return formatted_result


def _validate_timeout_seconds(timeout_seconds: float | None) -> float | None:
    if timeout_seconds is None:
        return None
    if isinstance(timeout_seconds, bool) or not isinstance(
        timeout_seconds, (int, float)
    ):
        raise TypeError(
            "timeout_seconds must be a positive number or None, "
            f"got {type(timeout_seconds)}"
        )

    timeout_value = float(timeout_seconds)
    if timeout_value <= 0:
        raise ValueError("timeout_seconds must be greater than 0")

    return timeout_value


def _validate_memory_limit_bytes(memory_limit_bytes: int | None) -> int | None:
    if memory_limit_bytes is None:
        return None
    if isinstance(memory_limit_bytes, bool) or not isinstance(memory_limit_bytes, int):
        raise TypeError(
            "memory_limit_bytes must be a positive integer number of bytes or None"
        )
    if memory_limit_bytes <= 0:
        raise ValueError("memory_limit_bytes must be greater than 0")

    return memory_limit_bytes


def _resolve_execution_limits(
    metadata: CodeEnvMetadata,
    *,
    default_timeout_seconds: float | None,
    default_memory_limit_bytes: int | None,
) -> ExecutionLimits:
    timeout_seconds = _validate_timeout_seconds(
        metadata.get("timeout_seconds", default_timeout_seconds)
    )
    memory_limit_bytes = _validate_memory_limit_bytes(
        metadata.get("memory_limit_bytes", default_memory_limit_bytes)
    )
    return ExecutionLimits(
        timeout_seconds=timeout_seconds,
        memory_limit_bytes=memory_limit_bytes,
    )


def _supports_memory_limit() -> bool:
    return (
        resource is not None
        and hasattr(resource, "RLIMIT_AS")
        and sys.platform.startswith("linux")
    )


def _supports_signal_timeout() -> bool:
    return hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer")


def _timeout_error(timeout_seconds: float) -> TimeoutError:
    return TimeoutError(
        "Code execution exceeded the configured timeout "
        f"of {timeout_seconds} seconds"
    )


@contextmanager
def _execution_timeout(timeout_seconds: float | None):
    if timeout_seconds is None or not _supports_signal_timeout():
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)

    def _handle_timeout(signum, frame):
        raise _timeout_error(timeout_seconds)

    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


@contextmanager
def _chdir(dir_path: str):
    """Change to a temporary directory for file operations."""
    current_dir = os.getcwd()
    os.chdir(dir_path)
    try:
        yield
    finally:
        os.chdir(current_dir)


def _safe_open(file: str, *args, **kwargs):
    """Safe version of open() that only allows access to the current directory."""
    real_file = os.path.realpath(file)
    working_dir = os.path.realpath(os.getcwd())
    if os.path.commonpath([real_file, working_dir]) != working_dir:
        raise PermissionError(
            "Access beyond the temporary working directory is blocked"
        )
    return open(file, *args, **kwargs)


def _safe_import(name: str, *args, **kwargs):
    """Safe version of import that blocks risky modules."""
    risky_modules = {
        "os",
        "shutil",  # erase filesystem
        "sys",
        "signal",  # exit the current program
        "socket",  # network communication
        "subprocess",
        "threading",
        "multiprocessing",  # spawn threads or processes
        "builtins",
        "importlib",  # bypass current blockers
    }
    if name in risky_modules:
        raise PermissionError("Importing system and network modules is blocked")
    return builtins.__import__(name, *args, **kwargs)


def _create_sandbox() -> dict[str, Any]:
    builtin_dict = {k: getattr(builtins, k) for k in dir(builtins)}
    builtin_dict["open"] = _safe_open
    builtin_dict["__import__"] = _safe_import
    return {"__builtins__": builtin_dict}


def _apply_memory_limit(memory_limit_bytes: int | None) -> None:
    if memory_limit_bytes is None:
        return
    if not _supports_memory_limit():
        raise RuntimeError(
            "memory_limit_bytes is not supported on this platform because "
            "resource.RLIMIT_AS is unavailable"
        )

    assert resource is not None  # for type checkers
    resource.setrlimit(
        resource.RLIMIT_AS,
        (memory_limit_bytes, memory_limit_bytes),
    )


def _execute_code_request(request: CodeExecutionRequest) -> CodeExecutionResponse:
    sandbox = _create_sandbox()
    result: Any = None
    terminated = False

    try:
        with _execution_timeout(request.timeout_seconds):
            tree = ast.parse(request.code)
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                exec_code = ast.unparse(tree.body[:-1])
                eval_code = ast.unparse(tree.body[-1])
            else:
                exec_code = request.code
                eval_code = None

            with _chdir(request.working_dir):
                _apply_memory_limit(request.memory_limit_bytes)
                exec(exec_code, sandbox, request.context)
                if eval_code:
                    result = eval(eval_code, sandbox, request.context)
                    terminated = True
    except Exception as err:
        result = err

    return CodeExecutionResponse(
        formatted_result=format_result(result, request.code, request.lookahead),
        terminated=terminated,
        context=request.context,
    )


def _serialize_request(
    request: CodeExecutionRequest,
) -> tuple[bytes, CodeExecutionRequest]:
    try:
        return cloudpickle.dumps(request), request
    except Exception:
        transport_request = replace(request, context=sanitize_object(request.context))
        return cloudpickle.dumps(transport_request), transport_request


def _serialize_response(response: CodeExecutionResponse) -> bytes:
    try:
        return cloudpickle.dumps(response)
    except Exception:
        sanitized_response = replace(
            response, context=sanitize_object(response.context)
        )
        return cloudpickle.dumps(sanitized_response)


def _subprocess_main(
    send_conn: Connection, request_bytes: bytes
) -> None:  # pragma: no cover
    try:
        request = cloudpickle.loads(request_bytes)
        response = _execute_code_request(request)
        send_conn.send_bytes(_serialize_response(response))
    finally:
        send_conn.close()


def _wait_for_subprocess_response(
    recv_conn: Connection,
    process: mp.Process,
    timeout_seconds: float | None,
) -> tuple[CodeExecutionResponse | None, bool]:
    wait_timeout_seconds = None
    if timeout_seconds is not None:
        wait_timeout_seconds = timeout_seconds + SUBPROCESS_STARTUP_GRACE_SECONDS
    deadline = (
        None
        if wait_timeout_seconds is None
        else time.monotonic() + wait_timeout_seconds
    )

    while True:
        poll_timeout = 0.1
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None, True
            poll_timeout = min(poll_timeout, remaining)

        if recv_conn.poll(poll_timeout):
            try:
                response = cloudpickle.loads(recv_conn.recv_bytes())
            except EOFError:
                return None, False
            return response, False

        if not process.is_alive():
            return None, False


def _execute_code_in_subprocess(
    request: CodeExecutionRequest,
) -> CodeExecutionResponse:
    ctx = mp.get_context("spawn")
    recv_conn, send_conn = ctx.Pipe(duplex=False)
    request_bytes, transport_request = _serialize_request(request)
    process = ctx.Process(
        target=_subprocess_main,
        args=(send_conn, request_bytes),
        daemon=True,
    )

    try:
        process.start()
        send_conn.close()
        response, timed_out = _wait_for_subprocess_response(
            recv_conn,
            process,
            transport_request.timeout_seconds,
        )

        if response is not None:
            process.join()
            return replace(response, context=sanitize_object(response.context))

        if timed_out and process.is_alive():
            process.kill()
            process.join()
            return CodeExecutionResponse(
                formatted_result=format_result(
                    _timeout_error(transport_request.timeout_seconds)
                ),
                terminated=False,
                context=sanitize_object(transport_request.context),
            )

        process.join()

        if (
            transport_request.memory_limit_bytes is not None
            and process.exitcode is not None
            and process.exitcode < 0
        ):
            error: Exception = MemoryError(
                "Code execution exceeded the configured memory limit "
                f"of {transport_request.memory_limit_bytes} bytes"
            )
        else:
            error = RuntimeError(
                "Code execution subprocess exited unexpectedly "
                f"with exit code {process.exitcode}"
            )
        return CodeExecutionResponse(
            formatted_result=format_result(error),
            terminated=False,
            context=sanitize_object(transport_request.context),
        )
    finally:
        recv_conn.close()
        if process.is_alive():
            process.kill()
            process.join()


@ray.remote  # pragma: no cover
class CodeExecutionWorker:
    """Helper class to process individual code execution steps."""

    def __init__(
        self,
        *,
        default_timeout_seconds: float | None,
        default_memory_limit_bytes: int | None,
    ):
        self.default_timeout_seconds = default_timeout_seconds
        self.default_memory_limit_bytes = default_memory_limit_bytes

    def execute(
        self, message_batch: list[str], metadata_batch: List[CodeEnvMetadata]
    ) -> Tuple[List[Dict[str, str]], List[bool], List[Any]]:
        """Execute code in a sandboxed environment."""
        results = []
        terminateds = []
        updated_metadata_batch: list[CodeEnvMetadata] = []

        for message, metadata in zip(message_batch, metadata_batch):
            match = re.search(r"<code>(.*)</code>(.*)", message, re.DOTALL)
            if not match:
                results.append("")
                terminateds.append(False)
                updated_metadata_batch.append(metadata)
                continue

            code, lookahead = match.groups()
            execution_limits = _resolve_execution_limits(
                metadata,
                default_timeout_seconds=self.default_timeout_seconds,
                default_memory_limit_bytes=self.default_memory_limit_bytes,
            )
            response = _execute_code_in_subprocess(
                CodeExecutionRequest(
                    code=code,
                    context=metadata["context"],
                    working_dir=metadata["working_dir"],
                    lookahead=lookahead,
                    timeout_seconds=execution_limits.timeout_seconds,
                    memory_limit_bytes=execution_limits.memory_limit_bytes,
                )
            )
            updated_metadata = dict(metadata)
            updated_metadata["context"] = response.context

            results.append(response.formatted_result)
            terminateds.append(response.terminated)
            updated_metadata_batch.append(updated_metadata)

        observations = [
            {"role": "environment", "content": result} for result in results
        ]
        updated_metadata_batch = sanitize_object(updated_metadata_batch)

        return observations, terminateds, updated_metadata_batch


@ray.remote  # pragma: no cover
class CodeEnvironment(EnvironmentInterface):
    """Code execution environment that maintains state between steps."""

    def __init__(self, cfg: CodeEnvConfig):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]
        self.terminate_on_evaluation = cfg["terminate_on_evaluation"]
        self.default_timeout_seconds = _validate_timeout_seconds(
            cfg.get("default_timeout_seconds")
        )
        self.default_memory_limit_bytes = _validate_memory_limit_bytes(
            cfg.get("default_memory_limit_bytes")
        )
        self.workers = [
            CodeExecutionWorker.options(
                runtime_env={"py_executable": PY_EXECUTABLES.SYSTEM}
            ).remote(
                default_timeout_seconds=self.default_timeout_seconds,
                default_memory_limit_bytes=self.default_memory_limit_bytes,
            )
            for _ in range(self.num_workers)
        ]

    def step(
        self,
        message_log_batch: List[LLMMessageLogType],
        metadata_batch: List[CodeEnvMetadata],
        return_extracted_answer: bool = False,
    ) -> EnvironmentReturn:
        """Process a batch of code execution steps."""
        message_batch = [ml[-1]["content"] for ml in message_log_batch]
        chunked_message_batch = chunk_list_to_workers(message_batch, self.num_workers)
        chunked_metadata_batch = chunk_list_to_workers(metadata_batch, self.num_workers)

        futures = [
            self.workers[i].execute.remote(message_chunk, metadata_chunk)
            for i, (message_chunk, metadata_chunk) in enumerate(
                zip(chunked_message_batch, chunked_metadata_batch)
            )
        ]

        results = ray.get(futures)

        observations = []
        terminateds = []
        new_metadata_batch = []

        for obs, term, meta in results:
            observations += obs
            terminateds += term
            new_metadata_batch += meta

        if self.terminate_on_evaluation:
            terminated_tensor = torch.tensor(terminateds, dtype=torch.bool)
        else:
            terminated_tensor = torch.zeros(len(terminateds), dtype=torch.bool)
        rewards_tensor = torch.zeros_like(terminated_tensor, dtype=torch.float32)

        next_stop_strings = [["</code>"]] * len(message_log_batch)

        assert return_extracted_answer == False, (
            "return_extracted_answer is not supported in CodeEnvironment. "
            "Please set it to False."
        )
        extracted_answers = None

        return EnvironmentReturn(
            observations=observations,
            metadata=new_metadata_batch,
            next_stop_strings=next_stop_strings,
            rewards=rewards_tensor,
            terminateds=terminated_tensor,
            answers=extracted_answers,
        )

    def shutdown(self):
        for worker in self.workers:
            ray.kill(worker)

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> Tuple[BatchedDataDict, dict]:
        """Compute metrics for the batch."""
        return batch, {}
