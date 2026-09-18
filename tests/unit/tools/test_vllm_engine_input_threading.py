# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual worker lock method without importing GPU dependencies."""

import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest


def install_lock(worker):
    source = (
        Path(__file__).resolve().parents[3]
        / "nemo_rl/models/generation/vllm/vllm_worker_async.py"
    )
    tree = ast.parse(source.read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_install_engine_input_socket_lock"
    )
    namespace = {"threading": threading, "Any": Any}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    namespace[method.name](worker)


class BufferEncoder:
    def __init__(self):
        self.aux_buffers = None

    def encode(self, value):
        try:
            self.aux_buffers = [value]
            time.sleep(0.003)
            assert self.aux_buffers == [value], "concurrent encoder buffer corruption"
            if value == "fail":
                raise ValueError("synthetic encode failure")
            return self.aux_buffers
        finally:
            self.aux_buffers = None


def test_unprotected_shared_encoder_reproduces_buffer_corruption():
    encoder = BufferEncoder()
    with ThreadPoolExecutor(max_workers=8) as pool:
        with pytest.raises(AssertionError, match="buffer corruption"):
            list(pool.map(encoder.encode, range(32)))


def test_concurrent_encodes_and_sends_are_safe_and_instance_local():
    encoder = BufferEncoder()
    other = BufferEncoder()
    sent = []
    socket = SimpleNamespace(send_multipart=lambda buffers: sent.append(buffers))
    worker = SimpleNamespace(
        llm=SimpleNamespace(
            engine_core=SimpleNamespace(
                encoder=encoder,
                input_socket=SimpleNamespace(_shadow_sock=socket),
            )
        )
    )
    install_lock(worker)
    assert "encode" not in other.__dict__

    def send(value):
        buffers = encoder.encode(value)
        socket.send_multipart(buffers)
        return buffers

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(send, range(32))) == [[value] for value in range(32)]
    assert sorted(sent) == [[value] for value in range(32)]
    try:
        encoder.encode("fail")
    except ValueError as error:
        assert str(error) == "synthetic encode failure"
    else:
        raise AssertionError("encode exception was swallowed")
    assert encoder.encode(100) == [100]
