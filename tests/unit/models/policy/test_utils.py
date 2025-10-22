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

import multiprocessing
import os
import sys
import time
import traceback
import unittest.mock

import pytest
import torch
import zmq

from nemo_rl.models.policy.utils import (
    IPCProtocol,
    calculate_aligned_size,
    get_megatron_checkpoint_dir,
    rebuild_cuda_tensor_from_ipc,
    stream_weights_via_ipc_zmq_impl,
)


class TestGetMegatronCheckpointDir:
    """Test cases for the get_megatron_checkpoint_dir function."""

    def test_nrl_megatron_checkpoint_dir_takes_precedence(self):
        """Test that NRL_MEGATRON_CHECKPOINT_DIR environment variable takes highest precedence."""
        expected_dir = "/custom/nrl/checkpoint/path"

        with unittest.mock.patch.dict(
            os.environ,
            {
                "NRL_MEGATRON_CHECKPOINT_DIR": expected_dir,
                "HF_HOME": "/some/hf/home",
                "HOME": "/some/home",
            },
        ):
            result = get_megatron_checkpoint_dir()
            assert result == expected_dir

    def test_hf_home_fallback_when_nrl_not_set(self):
        """Test that HF_HOME/nemo_rl is used when NRL_MEGATRON_CHECKPOINT_DIR is not set."""
        hf_home = "/path/to/hf/home"
        expected_dir = os.path.join(hf_home, "nemo_rl")

        env_vars = {"HF_HOME": hf_home, "HOME": "/some/home"}
        # Remove NRL_MEGATRON_CHECKPOINT_DIR if it exists
        env_vars.pop("NRL_MEGATRON_CHECKPOINT_DIR", None)

        with unittest.mock.patch.dict(os.environ, env_vars, clear=True):
            result = get_megatron_checkpoint_dir()
            assert result == expected_dir

    def test_default_fallback_when_no_env_vars_set(self):
        """Test that ~/.cache/huggingface/nemo_rl is used when no environment variables are set."""
        home_dir = "/home/testuser"
        expected_dir = os.path.join(home_dir, ".cache", "huggingface", "nemo_rl")

        with unittest.mock.patch.dict(os.environ, {"HOME": home_dir}, clear=True):
            with unittest.mock.patch("os.path.expanduser") as mock_expanduser:
                mock_expanduser.return_value = home_dir
                result = get_megatron_checkpoint_dir()
                assert result == expected_dir
                mock_expanduser.assert_called_once_with("~")

    def test_nrl_checkpoint_dir_empty_string_treated_as_unset(self):
        """Test that an empty NRL_MEGATRON_CHECKPOINT_DIR is treated as unset."""
        hf_home = "/path/to/hf/home"
        expected_dir = os.path.join(hf_home, "nemo_rl")

        with unittest.mock.patch.dict(
            os.environ,
            {
                "NRL_MEGATRON_CHECKPOINT_DIR": "",
                "HF_HOME": hf_home,
                "HOME": "/some/home",
            },
        ):
            result = get_megatron_checkpoint_dir()
            assert result == expected_dir

    def test_hf_home_empty_string_treated_as_unset(self):
        """Test that an empty HF_HOME is treated as unset."""
        home_dir = "/home/testuser"
        expected_dir = os.path.join(home_dir, ".cache", "huggingface", "nemo_rl")

        with unittest.mock.patch.dict(
            os.environ, {"HF_HOME": "", "HOME": home_dir}, clear=True
        ):
            with unittest.mock.patch("os.path.expanduser") as mock_expanduser:
                mock_expanduser.return_value = home_dir
                result = get_megatron_checkpoint_dir()
                assert result == expected_dir

    def test_function_prints_selected_directory(self, capsys):
        """Test that the function prints the selected directory."""
        expected_dir = "/custom/checkpoint/dir"

        with unittest.mock.patch.dict(
            os.environ, {"NRL_MEGATRON_CHECKPOINT_DIR": expected_dir}
        ):
            result = get_megatron_checkpoint_dir()

            captured = capsys.readouterr()
            assert (
                f"Using default megatron checkpoint dir: {expected_dir}" in captured.out
            )
            assert result == expected_dir


def server_process(
    zmq_addr: str,
    known_tensors: list[tuple[str, torch.Tensor]],
    buffer_size_bytes: int,
    ready_queue: multiprocessing.Queue,
) -> None:
    """Server process that streams tensors via IPC ZMQ."""
    try:
        device = torch.device("cuda:0")
        gpu_tensors = [(name, tensor.to(device)) for name, tensor in known_tensors]

        context = zmq.Context()
        socket = context.socket(zmq.PAIR)
        socket.setsockopt(zmq.LINGER, 0)  # Close immediately on error
        socket.setsockopt(zmq.RCVTIMEO, 10000)  # 10 second timeout
        socket.bind(zmq_addr)
        ready_queue.put(("ready", None))

        stream_weights_via_ipc_zmq_impl(
            (t for t in gpu_tensors),
            buffer_size_bytes,
            socket,
            rank=0,
            worker_name="test_server",
        )
    except Exception as e:
        import sys
        import traceback

        error_details = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        ready_queue.put(("error", error_details))
        sys.exit(
            1
        )  # Exit with non-zero code so check_process_error detects the failure
    finally:
        socket.close()
        context.term()


def client_process(
    zmq_addr: str,
    known_tensors_data: list[tuple[str, tuple, torch.dtype, torch.Tensor]],
    result_queue: multiprocessing.Queue,
) -> None:
    """Client process that receives and validates tensors via IPC ZMQ."""
    try:
        device = torch.device("cuda:0")

        # Prepare expected tensors on GPU
        expected_tensors = {
            name: tensor.to(device) for name, _, _, tensor in known_tensors_data
        }
        state_dict_info = {
            name: (shape, dtype) for name, shape, dtype, _ in known_tensors_data
        }

        context = zmq.Context()
        socket = context.socket(zmq.PAIR)
        socket.setsockopt(zmq.LINGER, 0)  # Close immediately on error
        socket.setsockopt(zmq.RCVTIMEO, 10000)  # 10 second timeout
        socket.connect(zmq_addr)

        # Receive and validate loop
        while True:
            payload = socket.recv_pyobj()
            if payload == IPCProtocol.COMPLETE:
                socket.send(IPCProtocol.ACK.value.encode())
                break

            ipc_handle, list_keys, used_bytes = payload
            buffer = rebuild_cuda_tensor_from_ipc(ipc_handle, device.index)

            offset = 0
            for key in list_keys:
                shape, dtype = state_dict_info[key]
                shape = torch.Size(shape) if isinstance(shape, list) else shape
                size_in_bytes = dtype.itemsize * shape.numel()

                tensor = (
                    buffer[offset : offset + size_in_bytes]
                    .view(dtype=dtype)
                    .view(shape)
                )
                expected = expected_tensors[key]

                # Validate tensor
                assert tensor.shape == expected.shape, f"Shape mismatch for {key}"
                assert tensor.dtype == expected.dtype, f"Dtype mismatch for {key}"
                assert torch.allclose(tensor, expected, rtol=1e-7, atol=1e-7), (
                    f"Values mismatch for {key}"
                )

                offset += calculate_aligned_size(size_in_bytes)

            assert offset == used_bytes, f"Offset mismatch: {offset} != {used_bytes}"
            socket.send(b"")

        result_queue.put(("success", "All tensors validated"))
    except Exception as e:
        error_details = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        result_queue.put(("error", error_details))
        sys.exit(1)
    finally:
        socket.close()
        context.term()


def check_process_error(
    proc: multiprocessing.Process,
    queue: multiprocessing.Queue,
    process_name: str,
) -> None:
    """Check if a process failed and assert with detailed error message if available."""
    if proc.exitcode == 0:
        return

    # Get error details from queue
    error_msg = None
    while not queue.empty():
        status, msg = queue.get_nowait()
        if status == "error":
            error_msg = msg
            break

    if proc.exitcode is None:
        assert False, f"{process_name} timed out"
    else:
        details = f"\n{error_msg}" if error_msg else ""
        assert False, f"{process_name} failed (exitcode={proc.exitcode}){details}"


class TestStreamWeightsViaIPC:
    """Test suite for IPC weight streaming functionality."""

    TIMEOUT = 30  # 30 second timeout for additional overhead when running with coverage

    @pytest.mark.parametrize(
        "test_case,tensor_specs,buffer_size_bytes,test_description",
        [
            (
                "large_buffer",
                [
                    ("tensor_1", (10, 20), torch.float32),  # 0.78KB
                    ("tensor_2", (5, 15, 25), torch.float32),  # 7.32KB
                    ("tensor_3", (100,), torch.float16),  # 0.20KB
                    ("tensor_4", (50, 50), torch.bfloat16),  # 4.88KB
                    ("tensor_5", (8, 16, 32), torch.float32),  # 16.00KB
                ],  # Total: 29.18KB
                100 * 1024,  # 100 KB - large buffer for single batch (50KB per side)
                "Test with various shapes/dtypes in large buffer (single batch)",
            ),
            (
                "small_buffer",
                [
                    ("small_1", (30, 30), torch.float32),  # 3.52KB
                    ("small_2", (20, 40), torch.float16),  # 1.56KB
                    ("small_3", (128,), torch.float32),  # 0.50KB
                    ("small_4", (25, 35), torch.float32),  # 3.42KB
                ],  # Total: 9.00KB
                10 * 1024,  # 10 KB - forces multiple batches (5KB per side)
                "Test with small buffer forcing multiple batches",
            ),
        ],
    )
    def test_stream_weights_via_ipc_zmq_impl(
        self, test_case, tensor_specs, buffer_size_bytes, test_description
    ):
        """Test streaming weights via IPC ZMQ between server and client processes."""
        # Generate test tensors
        known_tensors = [
            (name, torch.randn(*shape, dtype=dtype))
            for name, shape, dtype in tensor_specs
        ]
        known_tensors_data = [
            (name, list(t.shape), t.dtype, t) for name, t in known_tensors
        ]

        # Create unique socket path and queues
        socket_path = f"/tmp/test_ipc_zmq_{test_case}_{os.getpid()}_{time.time()}"
        zmq_addr = f"ipc://{socket_path}"

        mp_context = multiprocessing.get_context("spawn")
        ready_queue = mp_context.Queue()
        result_queue = mp_context.Queue()

        # Start server and client
        server_proc = mp_context.Process(
            target=server_process,
            args=(zmq_addr, known_tensors, buffer_size_bytes, ready_queue),
        )
        server_proc.start()

        status, msg = ready_queue.get(timeout=self.TIMEOUT)
        assert status == "ready", f"Server failed: {msg}"

        client_proc = mp_context.Process(
            target=client_process,
            args=(zmq_addr, known_tensors_data, result_queue),
        )
        client_proc.start()

        # Wait and validate
        try:
            server_proc.join(timeout=self.TIMEOUT)
            client_proc.join(timeout=self.TIMEOUT)

            # Check client first since client failure often causes server to fail
            check_process_error(client_proc, result_queue, "Client")
            check_process_error(server_proc, ready_queue, "Server")

            # Verify client success message
            status, msg = result_queue.get(timeout=self.TIMEOUT)
            assert status == "success", f"Validation failed: {msg}"
        finally:
            for proc in [server_proc, client_proc]:
                if proc and proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=self.TIMEOUT)
                    if proc.is_alive():
                        proc.kill()

            if os.path.exists(socket_path):
                os.unlink(socket_path)
