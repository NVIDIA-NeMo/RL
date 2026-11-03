#!/usr/bin/env python3
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
"""Verify the actual MoE module before executing a SGLang Python worker."""

from __future__ import annotations

import os
import runpy
import sys
import types
from dataclasses import dataclass
from pathlib import Path

from verify_runtime import DEFAULT_BUNDLE, install_deferred_verification


@dataclass(frozen=True)
class PythonInvocation:
    mode: str
    target: str
    arguments: tuple[str, ...]
    unbuffered: bool = False
    dont_write_bytecode: bool = False


def parse_python_invocation(arguments):
    arguments = list(arguments)
    unbuffered = dont_write_bytecode = False
    while arguments and arguments[0] in ("-u", "-B"):
        flag = arguments.pop(0)
        unbuffered |= flag == "-u"
        dont_write_bytecode |= flag == "-B"
    escaped = bool(arguments and arguments[0] == "--")
    if escaped:
        arguments.pop(0)
    if not arguments:
        raise RuntimeError(
            "Interactive Python execution is unsupported by the verified worker"
        )
    first = arguments.pop(0)
    if not escaped and first in ("-m", "-c"):
        if not arguments or (first == "-m" and not arguments[0]):
            raise RuntimeError(f"Python {first} requires a target")
        target = arguments.pop(0)
        mode = "module" if first == "-m" else "command"
    elif first == "-" or (first.startswith("-") and not escaped):
        raise RuntimeError(f"Unsupported Python worker option: {first}")
    else:
        mode, target = "script", first
    return PythonInvocation(
        mode, target, tuple(arguments), unbuffered, dont_write_bytecode
    )


def forward_python_invocation(invocation):
    if invocation.dont_write_bytecode:
        sys.dont_write_bytecode = True
    if invocation.unbuffered:
        for stream in (sys.stdout, sys.stderr):
            if callable(getattr(stream, "reconfigure", None)):
                stream.reconfigure(line_buffering=True, write_through=True)
    if invocation.mode == "script":
        sys.argv = [invocation.target, *invocation.arguments]
        sys.path[0] = str(Path(invocation.target).resolve().parent)
        return runpy.run_path(invocation.target, run_name="__main__")
    sys.path[0] = ""
    if invocation.mode == "module":
        sys.argv = [invocation.target, *invocation.arguments]
        return runpy.run_module(invocation.target, run_name="__main__", alter_sys=True)
    if invocation.mode == "command":
        sys.argv = ["-c", *invocation.arguments]
        module = types.ModuleType("__main__")
        module.__package__ = None
        module.__spec__ = None
        sys.modules["__main__"] = module
        exec(compile(invocation.target, "<string>", "exec"), module.__dict__)
        return None
    raise RuntimeError(f"Unsupported Python execution mode: {invocation.mode}")


def main(argv=None):
    invocation = parse_python_invocation(sys.argv[1:] if argv is None else argv)
    expected = os.environ.get("FLINT_FLASHINFER_MANIFEST_SHA256")
    directory = os.environ.get("FLINT_FLASHINFER_RECEIPT_DIR")
    if not expected or not directory:
        raise RuntimeError(
            "FLINT_FLASHINFER_MANIFEST_SHA256 and FLINT_FLASHINFER_RECEIPT_DIR are required"
        )
    bundle = Path(os.environ.get("FLINT_FLASHINFER_BUNDLE", str(DEFAULT_BUNDLE)))
    install_deferred_verification(bundle, expected, directory)
    return forward_python_invocation(invocation)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"Verified SGLang launch failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        sys.exit(1)
