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
import os
import re
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from copy import copy
from io import IOBase
from pprint import pformat
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from nemo_rl.environments.utils import chunk_list_to_workers


class CodeEnvConfig(TypedDict):
    num_workers: int
    # whether to terminate the execution after expression evaluation
    # if you want to execute multiple rounds of code, set this to False
    # and wrap CodeEnvironment in another environment that terminates the generation
    terminate_on_evaluation: bool


class CodeEnvMetadata(TypedDict):
    context: Dict[str, Any]  # Hold functions and variables defined in the code
    working_dir: str  # Working directory for file operations


@ray.remote  # pragma: no cover
class CodeExecutionWorker:
    """Helper class to process individual code execution steps."""

    def __init__(self):
        # Create sandbox with safe builtins
        builtin_dict = {k: getattr(builtins, k) for k in dir(builtins)}
        builtin_dict["open"] = self.safe_open
        builtin_dict["__import__"] = self.safe_import
        self.sandbox = {"__builtins__": builtin_dict}

    def sanitize(self, obj: Any) -> Any:
        # TODO: better handling of unpicklable objects: custom __getstate__ and __setstate__
        # recursively remove all file objects as they are not picklable by ray
        if isinstance(obj, (IOBase, ModuleType)):
            # replace unpickable objects with a string representation
            return repr(obj)
        if isinstance(obj, Mapping):
            return obj.__class__(
                {self.sanitize(k): self.sanitize(v) for k, v in obj.items()}
            )
        if isinstance(obj, Sequence) and not isinstance(obj, str):
            return obj.__class__(self.sanitize(v) for v in obj)
        if hasattr(obj, "__dict__"):
            new_obj = copy(obj)
            new_obj.__dict__ = {
                self.sanitize(k): self.sanitize(v) for k, v in obj.__dict__.items()
            }
            return new_obj
        return obj

    def format_result(
        self, result: Any, code: Optional[str] = None, lookahead: Optional[str] = None
    ) -> str:
        if result is None:
            # no return value
            return ""
        result = pformat(result)
        multiline = (code and "\n" in code) or "\n" in result
        if multiline:
            # multi-line format
            result = f"\n\n<result>\n{result}\n</result>"
        else:
            # inline format
            result = f"<result>{result}</result>"
        if lookahead:
            if result.startswith(lookahead):
                # The generation may look like "</code>\n" if ">\n" is a single token.
                # We trim \n from the result if the model has already generated it.
                result = result[len(lookahead) :]
        return result

    def execute(
        self, message_batch: str, metadata_batch: List[CodeEnvMetadata]
    ) -> Tuple[List[Dict[str, str]], List[bool], List[Any]]:
        """Execute code in a sandboxed environment."""
        results = []
        terminateds = []

        for message, metadata in zip(message_batch, metadata_batch):
            match = re.search(r"<code>(.*)</code>(.*)", message, re.DOTALL)
            if not match:
                results.append("")
                terminateds.append(False)
                continue

            code, lookahead = match.groups()
            tree = ast.parse(code)

            if tree.body and isinstance(tree.body[-1], ast.Expr):
                # Interactive mode
                exec_code = ast.unparse(tree.body[:-1])
                eval_code = ast.unparse(tree.body[-1])
            else:
                # Silent mode
                exec_code = code
                eval_code = None

            result = None
            terminated = False
            with self.chdir(metadata["working_dir"]):
                try:
                    # isolate the code in a sandbox
                    # capture local variables in metadata["context"]
                    exec(exec_code, self.sandbox, metadata["context"])
                    if eval_code:
                        result = eval(eval_code, self.sandbox, metadata["context"])
                        terminated = True
                except Exception as err:
                    result = err

            result = self.format_result(result, code, lookahead)
            results.append(result)
            terminateds.append(terminated)

        observations = [
            {"role": "environment", "content": result} for result in results
        ]
        metadata_batch = self.sanitize(metadata_batch)

        return observations, terminateds, metadata_batch

    @contextmanager
    def chdir(self, dir: str):
        """Change to temporary directory for file operations."""
        current_dir = os.getcwd()
        os.chdir(dir)
        try:
            yield
        finally:
            os.chdir(current_dir)

    def safe_open(self, file: str, *args, **kwargs):
        """Safe version of open() that only allows access to temporary directory."""
        real_file = os.path.realpath(file)
        working_dir = os.path.realpath(os.getcwd())
        if os.path.commonpath([real_file, working_dir]) != working_dir:
            raise PermissionError(
                "Access beyond the temporary working directory is blocked"
            )
        return open(file, *args, **kwargs)

    def safe_import(self, name: str, *args, **kwargs):
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


@ray.remote  # pragma: no cover
class CodeEnvironment(EnvironmentInterface):
    """Code execution environment that maintains state between steps."""

    def __init__(self, cfg: CodeEnvConfig):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]
        self.terminate_on_evaluation = cfg["terminate_on_evaluation"]
        self.workers = [
            CodeExecutionWorker.options(
                runtime_env={"py_executable": PY_EXECUTABLES.SYSTEM}
            ).remote()
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

        # Process each chunk in parallel
        futures = [
            self.workers[i].execute.remote(message_chunk, metadata_chunk)
            for i, (message_chunk, metadata_chunk) in enumerate(
                zip(chunked_message_batch, chunked_metadata_batch)
            )
        ]

        results = ray.get(futures)

        # Unpack results
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
            "return_extracted_answer is not supported in CodeEnvironment. Please set it to False."
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
        # shutdown all workers
        for worker in self.workers:
            ray.kill(worker)

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> Tuple[BatchedDataDict, dict]:
        """Compute metrics for the batch."""
        # No specific metrics for code execution
        return batch, {}
