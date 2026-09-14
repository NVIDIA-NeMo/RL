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
import weakref

import pytest
import torch
import zmq

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy.utils import (
    DENSE_TEACHER_IPC_FLAT_LAYOUT,
    IPCProtocol,
    aggregate_per_sample_handles,
    calculate_aligned_size,
    ensure_teacher_ipc_buffer,
    ensure_teacher_ipc_row_buffer,
    ensure_teacher_ipc_token_buffer,
    extract_batch_item_ids,
    extract_teacher_ipc_valid_lengths,
    get_dense_ipc_sequence_layout,
    get_megatron_checkpoint_dir,
    localize_teacher_ipc_valid_lengths,
    partition_teacher_ipc_row_buffer,
    partition_teacher_ipc_token_buffer,
    rebuild_cuda_tensor_from_ipc,
    resolve_model_class,
    stream_weights_via_ipc_zmq_impl,
    validate_compact_teacher_ipc_handle,
)


@pytest.mark.parametrize(
    ("model_type", "hf_class_name", "nemo_class_name"),
    [
        ("qwen2_5_vl", "hf_image_text", "nemo_image_text"),
        ("qwen2_5_omni", "hf_text_waveform", "nemo_text_waveform"),
        ("unknown_model", "hf_causal_lm", "nemo_causal_lm"),
    ],
)
@pytest.mark.parametrize("nemo_available", [False, True])
def test_resolve_model_class_selects_requested_loader(
    monkeypatch: pytest.MonkeyPatch,
    model_type: str,
    hf_class_name: str,
    nemo_class_name: str,
    nemo_available: bool,
) -> None:
    """The caller chooses plain Transformers or NeMo AutoModel classes."""
    hf_classes = {
        "hf_image_text": object(),
        "hf_text_waveform": object(),
        "hf_causal_lm": object(),
    }
    nemo_classes = {
        "nemo_image_text": object(),
        "nemo_text_waveform": object(),
        "nemo_causal_lm": object(),
    }

    monkeypatch.setattr(
        "nemo_rl.models.policy.utils.HF_AUTOMODEL_FACTORY",
        {
            "qwen2_5_vl": hf_classes["hf_image_text"],
            "qwen2_5_omni": hf_classes["hf_text_waveform"],
        },
    )
    monkeypatch.setattr(
        "nemo_rl.models.policy.utils.AUTOMODEL_FACTORY",
        {
            "qwen2_5_vl": nemo_classes["nemo_image_text"],
            "qwen2_5_omni": nemo_classes["nemo_text_waveform"],
        },
    )
    monkeypatch.setattr(
        "nemo_rl.models.policy.utils.AutoModelForCausalLM",
        hf_classes["hf_causal_lm"],
    )
    monkeypatch.setattr(
        "nemo_rl.models.policy.utils.NeMoAutoModelForCausalLM",
        nemo_classes["nemo_causal_lm"],
    )
    monkeypatch.setattr(
        "nemo_rl.models.policy.utils.NEMO_AUTOMODEL_AVAILABLE", nemo_available
    )

    assert (
        resolve_model_class(model_type, use_nemo_automodel=False)
        is hf_classes[hf_class_name]
    )
    expected_default = (
        nemo_classes[nemo_class_name] if nemo_available else hf_classes[hf_class_name]
    )
    assert resolve_model_class(model_type) is expected_default


def test_resolve_model_class_routes_gemma4_unified_to_image_text_model():
    assert "ImageTextToText" in resolve_model_class("gemma4_unified").__name__


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


class _FakeIpcSocket:
    def __init__(self):
        self.sent = []

    def send_pyobj(self, payload):
        self.sent.append(payload)

    def recv(self):
        return b""

    def getsockopt(self, _option):
        return 0


def test_stream_weights_releases_buffers_before_complete_without_full_gc(
    monkeypatch,
):
    """The final data ACK is sufficient to reclaim both acyclic IPC buffers."""

    tensor = torch.ones(4, dtype=torch.float32)
    buffer_refs = []
    events = []
    original_empty = torch.empty

    def tracking_empty(*args, **kwargs):
        buffer = original_empty(*args, **kwargs)
        buffer_refs.append(weakref.ref(buffer))
        return buffer

    def empty_cache():
        events.append("empty_cache")
        assert len(buffer_refs) == 2
        assert all(buffer_ref() is None for buffer_ref in buffer_refs)

    class ReleaseAwareSocket(_FakeIpcSocket):
        def send_pyobj(self, payload):
            if payload == IPCProtocol.COMPLETE:
                assert events == ["empty_cache"]
                assert all(buffer_ref() is None for buffer_ref in buffer_refs)
            super().send_pyobj(payload)

    monkeypatch.setattr(torch, "empty", tracking_empty)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda: unittest.mock.Mock(synchronize=lambda: None),
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    monkeypatch.setattr(
        "nemo_rl.models.policy.utils.get_handle_from_tensor",
        lambda _buffer: ("ipc-handle",),
    )
    monkeypatch.setattr(
        "nemo_rl.models.policy.utils.gc.collect",
        lambda: pytest.fail("IPC buffer cleanup must not scan the full object graph"),
    )

    socket = ReleaseAwareSocket()
    stream_weights_via_ipc_zmq_impl(
        params_generator=iter([("weight", tensor)]),
        buffer_size_bytes=4096,
        zmq_socket=socket,
        rank=0,
        worker_name="test_worker",
    )

    assert events == ["empty_cache"]
    assert socket.sent[-1] == IPCProtocol.COMPLETE


def test_stream_weights_via_ipc_zmq_uses_cuda_buffer_for_cpu_tensors(monkeypatch):
    """CPU-exported tensors should still be packed into CUDA IPC buffers."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for CUDA IPC buffer allocation")

    tensor = torch.ones(4, dtype=torch.float32)
    captured = {}

    def fake_get_handle_from_tensor(tensor):
        captured["buffer_device"] = tensor.device
        return ("ipc-handle",)

    monkeypatch.setattr(
        "nemo_rl.models.policy.utils.get_handle_from_tensor",
        fake_get_handle_from_tensor,
    )

    socket = _FakeIpcSocket()
    stream_weights_via_ipc_zmq_impl(
        params_generator=iter([("weight", tensor)]),
        buffer_size_bytes=4096,
        zmq_socket=socket,
        rank=0,
        worker_name="test_worker",
    )

    assert captured["buffer_device"].type == "cuda"
    payload = socket.sent[0]
    assert payload[0] == ("ipc-handle",)
    assert payload[1] == ["weight"]
    assert payload[2] == calculate_aligned_size(tensor.nbytes)
    assert socket.sent[-1] == IPCProtocol.COMPLETE


def test_stream_weights_via_ipc_zmq_aligns_cpu_tensor_groups(monkeypatch):
    """CPU-exported tensor groups report aligned byte offsets."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for CUDA IPC buffer allocation")

    tensors = [
        ("weight", torch.ones(4, dtype=torch.float32)),
        ("bias", torch.ones(3, dtype=torch.float16)),
    ]
    captured = {}

    def fake_get_handle_from_tensor(tensor):
        captured["buffer_device"] = tensor.device
        return ("ipc-handle",)

    monkeypatch.setattr(
        "nemo_rl.models.policy.utils.get_handle_from_tensor",
        fake_get_handle_from_tensor,
    )

    socket = _FakeIpcSocket()
    stream_weights_via_ipc_zmq_impl(
        params_generator=iter(tensors),
        buffer_size_bytes=4096,
        zmq_socket=socket,
        rank=0,
        worker_name="test_worker",
    )

    assert captured["buffer_device"].type == "cuda"
    payload = socket.sent[0]
    assert payload[1] == ["weight", "bias"]
    assert payload[2] == sum(
        calculate_aligned_size(tensor.nbytes) for _, tensor in tensors
    )
    assert socket.sent[-1] == IPCProtocol.COMPLETE


def test_stream_weights_via_ipc_zmq_preserves_cpu_and_gpu_source_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU and GPU sources must produce identical CUDA IPC staging payloads."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for CUDA IPC buffer allocation")

    source_tensors = [
        ("packed.weight", torch.tensor([[0, 1], [254, 255]], dtype=torch.uint8)),
        ("weight_scale", torch.tensor([1.0, -2.0], dtype=torch.float32)),
        ("weight_scale_2", torch.tensor([0.5], dtype=torch.float32)),
    ]

    def capture_staging_payload(
        tensors: list[tuple[str, torch.Tensor]],
    ) -> tuple[list[str], int, list[torch.Tensor]]:
        captured: dict[str, torch.Tensor] = {}

        def fake_get_handle_from_tensor(buffer: torch.Tensor) -> tuple[str]:
            captured["buffer"] = buffer.detach().cpu().clone()
            return ("ipc-handle",)

        monkeypatch.setattr(
            "nemo_rl.models.policy.utils.get_handle_from_tensor",
            fake_get_handle_from_tensor,
        )
        socket = _FakeIpcSocket()
        stream_weights_via_ipc_zmq_impl(
            params_generator=iter(tensors),
            buffer_size_bytes=4096,
            zmq_socket=socket,
            rank=0,
            worker_name="test_worker",
        )
        _, names, used_bytes = socket.sent[0]
        offset = 0
        tensor_bytes = []
        for _, tensor in source_tensors:
            tensor_bytes.append(captured["buffer"][offset : offset + tensor.nbytes])
            offset += calculate_aligned_size(tensor.nbytes)
        assert offset == used_bytes
        return names, used_bytes, tensor_bytes

    cpu_payload = capture_staging_payload(source_tensors)
    gpu_payload = capture_staging_payload(
        [(name, tensor.cuda()) for name, tensor in source_tensors]
    )

    assert cpu_payload[0] == gpu_payload[0]
    assert cpu_payload[1] == gpu_payload[1]
    assert all(
        torch.equal(cpu_bytes, gpu_bytes)
        for cpu_bytes, gpu_bytes in zip(cpu_payload[2], gpu_payload[2], strict=True)
    )


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
        self._run_stream_weights_roundtrip(test_case, known_tensors, buffer_size_bytes)

    def test_stream_weights_via_ipc_zmq_impl_non_contiguous(self):
        """Regression: tensors yielded by the params iterator may be non-contiguous.

        For example, ``Megatron-Bridge``'s ``QKVMapping.megatron_to_hf`` returns
        Q/K/V shards via advanced indexing + ``reshape`` that can produce views
        with non-canonical strides. Before the fix, ``pack_tensor`` called
        ``view(-1)`` which raises ``RuntimeError: view size is not compatible
        with input tensor's size and stride``.
        """
        # transpose(): non-contiguous, contains all elements
        t1 = torch.randn(8, 16, dtype=torch.float32).t()
        # slicing with stride: non-contiguous
        t2 = torch.randn(40, 60, dtype=torch.float32)[:, ::2]
        # permute on 3D: non-contiguous
        t3 = torch.randn(4, 8, 12, dtype=torch.bfloat16).permute(2, 0, 1)
        for t in (t1, t2, t3):
            assert not t.is_contiguous(), "test tensor must be non-contiguous"

        known_tensors = [("qkv_q_proj", t1), ("qkv_k_proj", t2), ("qkv_v_proj", t3)]
        self._run_stream_weights_roundtrip(
            "non_contiguous", known_tensors, buffer_size_bytes=100 * 1024
        )

    def _run_stream_weights_roundtrip(
        self,
        test_case: str,
        known_tensors: list[tuple[str, torch.Tensor]],
        buffer_size_bytes: int,
    ) -> None:
        """Shared driver: spawn server/client and validate the round-trip."""
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


class TestAggregatePerSampleHandles:
    @staticmethod
    def _dense_handle(
        batch_item_id,
        shard,
        *,
        tp_rank=0,
        tp_size=1,
        cp_rank=0,
        cp_size=1,
        full_seq_len=8,
        full_vocab_size=12,
        global_seq_start=None,
    ):
        local_seq_len = full_seq_len // cp_size
        local_vocab_size = full_vocab_size // tp_size
        if global_seq_start is None:
            global_seq_start = cp_rank * local_seq_len
        return {
            "batch_item_id": batch_item_id,
            "shard": shard,
            "tp_rank": tp_rank,
            "tp_size": tp_size,
            "cp_rank": cp_rank,
            "cp_size": cp_size,
            "actual_shape": (local_seq_len, local_vocab_size),
            "global_seq_start": global_seq_start,
            "full_seq_len": full_seq_len,
            "vocab_start_index": tp_rank * local_vocab_size,
            "vocab_end_index": (tp_rank + 1) * local_vocab_size,
            "full_vocab_size": full_vocab_size,
        }

    def test_orders_by_dp_rank(self):
        out = aggregate_per_sample_handles(
            [
                {"dp_rank": 1, "per_sample_handles": ["b0", "b1"]},
                {"dp_rank": 0, "per_sample_handles": ["a0", "a1"]},
            ]
        )
        assert [e["teacher_shards"] for e in out] == [["a0"], ["a1"], ["b0"], ["b1"]]

    def test_collects_replicas_per_sample(self):
        out = aggregate_per_sample_handles(
            [
                {"dp_rank": 0, "per_sample_handles": ["r0s0", "r0s1"]},
                {"dp_rank": 0, "per_sample_handles": ["r1s0", "r1s1"]},
            ]
        )
        assert [e["teacher_shards"] for e in out] == [
            ["r0s0", "r1s0"],
            ["r0s1", "r1s1"],
        ]

    def test_length_mismatch_raises(self):
        with pytest.raises(AssertionError):
            aggregate_per_sample_handles(
                [
                    {"dp_rank": 0, "per_sample_handles": ["a0", "a1"]},
                    {"dp_rank": 0, "per_sample_handles": ["b0"]},
                ]
            )

    def test_identity_aware_aggregation_uses_exact_canonical_order(self):
        out = aggregate_per_sample_handles(
            [
                {
                    "dp_rank": 1,
                    "per_sample_handles": [
                        self._dense_handle(30, "a30", tp_rank=0, tp_size=2),
                        self._dense_handle(10, "a10", tp_rank=0, tp_size=2),
                    ],
                },
                {
                    "dp_rank": 0,
                    "per_sample_handles": [
                        self._dense_handle(20, "a20", tp_rank=0, tp_size=2)
                    ],
                },
                {
                    "dp_rank": 1,
                    "per_sample_handles": [
                        self._dense_handle(10, "b10", tp_rank=1, tp_size=2),
                        self._dense_handle(30, "b30", tp_rank=1, tp_size=2),
                    ],
                },
                {
                    "dp_rank": 0,
                    "per_sample_handles": [
                        self._dense_handle(20, "b20", tp_rank=1, tp_size=2)
                    ],
                },
            ],
            canonical_batch_item_ids=(10, 20, 30),
        )

        assert [item["batch_item_id"] for item in out] == [10, 20, 30]
        assert [[h["shard"] for h in item["teacher_shards"]] for item in out] == [
            ["a10", "b10"],
            ["a20", "b20"],
            ["a30", "b30"],
        ]

    @pytest.mark.parametrize(
        "worker_results,match",
        [
            (
                [{"dp_rank": 0, "per_sample_handles": [{"shard": "missing"}]}],
                "carry batch_item_id",
            ),
            (
                [
                    {
                        "dp_rank": 0,
                        "per_sample_handles": [
                            {"batch_item_id": 1},
                            {"batch_item_id": 1},
                        ],
                    }
                ],
                "duplicate handles",
            ),
            (
                [
                    {
                        "dp_rank": 0,
                        "per_sample_handles": [{"batch_item_id": 1}],
                    },
                    {
                        "dp_rank": 0,
                        "per_sample_handles": [{"batch_item_id": 2}],
                    },
                ],
                "inconsistent batch_item_id sets",
            ),
        ],
    )
    def test_identity_aware_aggregation_rejects_invalid_records(
        self, worker_results, match
    ):
        with pytest.raises(ValueError, match=match):
            aggregate_per_sample_handles(
                worker_results,
                canonical_batch_item_ids=(1, 2),
            )

    def test_identity_aware_aggregation_rejects_missing_tp_cp_shard(self):
        worker_results = [
            {
                "dp_rank": 0,
                "per_sample_handles": [
                    self._dense_handle(
                        1,
                        f"tp{tp_rank}cp{cp_rank}",
                        tp_rank=tp_rank,
                        tp_size=2,
                        cp_rank=cp_rank,
                        cp_size=2,
                    )
                ],
            }
            for tp_rank, cp_rank in ((0, 0), (0, 1), (1, 0))
        ]
        with pytest.raises(ValueError, match="incomplete TP/CP shard coverage"):
            aggregate_per_sample_handles(worker_results, canonical_batch_item_ids=(1,))

    def test_identity_aware_aggregation_rejects_duplicate_coordinate(self):
        duplicate = self._dense_handle(1, "duplicate")
        with pytest.raises(ValueError, match="duplicate shard coordinate"):
            aggregate_per_sample_handles(
                [
                    {"dp_rank": 0, "per_sample_handles": [duplicate]},
                    {"dp_rank": 0, "per_sample_handles": [dict(duplicate)]},
                ],
                canonical_batch_item_ids=(1,),
            )

    def test_identity_aware_aggregation_rejects_sequence_coverage_gap(self):
        with pytest.raises(ValueError, match="gap or overlap"):
            aggregate_per_sample_handles(
                [
                    {
                        "dp_rank": 0,
                        "per_sample_handles": [
                            self._dense_handle(1, "cp0", cp_rank=0, cp_size=2)
                        ],
                    },
                    {
                        "dp_rank": 0,
                        "per_sample_handles": [
                            self._dense_handle(
                                1,
                                "cp1",
                                cp_rank=1,
                                cp_size=2,
                                global_seq_start=3,
                            )
                        ],
                    },
                ],
                canonical_batch_item_ids=(1,),
            )

    def test_identity_aware_aggregation_rejects_vocab_coverage_gap(self):
        tp0 = self._dense_handle(1, "tp0", tp_rank=0, tp_size=2)
        tp1 = self._dense_handle(1, "tp1", tp_rank=1, tp_size=2)
        tp1["vocab_start_index"] = 5
        tp1["vocab_end_index"] = 11
        with pytest.raises(ValueError, match="gap or overlap"):
            aggregate_per_sample_handles(
                [
                    {"dp_rank": 0, "per_sample_handles": [tp0]},
                    {"dp_rank": 0, "per_sample_handles": [tp1]},
                ],
                canonical_batch_item_ids=(1,),
            )

    def test_identity_aware_aggregation_rejects_compact_valid_length_drift(self):
        cp0 = self._dense_handle(1, "cp0", cp_rank=0, cp_size=2)
        cp1 = self._dense_handle(1, "cp1", cp_rank=1, cp_size=2)
        cp0.update(
            {
                "ipc_layout": DENSE_TEACHER_IPC_FLAT_LAYOUT,
                "payload_ipc": "cp0",
                "storage_shape": (4, 12),
                "storage_token_offset": 0,
                "storage_used_tokens": 4,
                "storage_capacity_tokens": 4,
                "stored_seq_len": 4,
                "valid_seq_len": 5,
                "dtype": torch.float32,
            }
        )
        cp1.update(
            {
                "ipc_layout": DENSE_TEACHER_IPC_FLAT_LAYOUT,
                "payload_ipc": None,
                "storage_shape": (0, 12),
                "storage_token_offset": 0,
                "storage_used_tokens": 0,
                "storage_capacity_tokens": 0,
                "stored_seq_len": 0,
                "valid_seq_len": 4,
                "dtype": torch.float32,
            }
        )

        with pytest.raises(ValueError, match="disagree on valid_seq_len"):
            aggregate_per_sample_handles(
                [
                    {"dp_rank": 0, "per_sample_handles": [cp0]},
                    {"dp_rank": 0, "per_sample_handles": [cp1]},
                ],
                canonical_batch_item_ids=(1,),
            )


class TestExtractBatchItemIds:
    def test_accepts_tensor_and_validates_cardinality(self):
        assert extract_batch_item_ids(
            {"batch_item_id": torch.tensor([7, 3])}, 2, required=True
        ) == [7, 3]
        with pytest.raises(ValueError, match="cardinality"):
            extract_batch_item_ids({"batch_item_id": [7]}, 2, required=True)

    def test_required_identity_is_not_synthesized(self):
        assert extract_batch_item_ids({}, 2, required=False) is None
        with pytest.raises(ValueError, match="requires batch_item_id"):
            extract_batch_item_ids({}, 2, required=True)


class TestDenseIpcSequenceLayout:
    def test_uses_logical_rectangle_not_packed_physical_bin(self):
        data = BatchedDataDict(
            {
                "input_ids": torch.zeros(2, 6, dtype=torch.long),
                "input_lengths": torch.tensor([3, 2]),
            }
        )
        # This is the physical scheduler geometry for a packed bin. It must not
        # leak into dense teacher handle metadata, whose sequence axis is T_t=6.
        data.micro_batch_lengths = [[12]]

        assert get_dense_ipc_sequence_layout(data, cp_rank=0, cp_size=2) == (
            6,
            3,
            0,
        )
        assert get_dense_ipc_sequence_layout(data, cp_rank=1, cp_size=2) == (
            6,
            3,
            3,
        )


class TestEnsureTeacherIpcBuffer:
    def test_alloc_reuse_and_grow(self):
        dev = torch.device("cpu")
        s, h = ensure_teacher_ipc_buffer(None, None, 2, 1, 4, 8, torch.float32, dev)
        assert s.shape == (2, 1, 4, 8) and h is not None
        s2, h2 = ensure_teacher_ipc_buffer(s, h, 2, 1, 4, 8, torch.float32, dev)
        assert s2 is s and h2 is h
        s3, _ = ensure_teacher_ipc_buffer(s, h, 3, 1, 4, 8, torch.float32, dev)
        assert s3 is not s and s3.shape == (3, 1, 4, 8)

    def test_max_cardinality_preallocation_keeps_handle_stable(self):
        dev = torch.device("cpu")
        storage, handle = ensure_teacher_ipc_buffer(
            None, None, 3, 4, 6, 8, torch.float32, dev
        )
        for variable_batch_size in (1, 4, 2):
            next_storage, next_handle = ensure_teacher_ipc_buffer(
                storage,
                handle,
                3,
                variable_batch_size,
                6,
                8,
                torch.float32,
                dev,
            )
            assert next_storage is storage
            assert next_handle is handle


class TestCompactTeacherIpcRowBuffer:
    def test_variable_cardinality_bins_use_only_logical_rows(self):
        # Qualification rank 3 is the worst old rectangular allocation:
        # 6 bins * max(14 rows) = 84 rectangles for only 32 logical rows.
        batch_sizes = [3, 5, 8, 14, 1, 1]
        seq_len = 2
        vocab_size = 3
        storage = ensure_teacher_ipc_row_buffer(
            None,
            sum(batch_sizes),
            seq_len,
            vocab_size,
            torch.float32,
            torch.device("cpu"),
        )
        partitions = partition_teacher_ipc_row_buffer(storage, batch_sizes)

        assert storage.shape == (32, seq_len, vocab_size)
        assert len(batch_sizes) * max(batch_sizes) == 84
        assert storage.numel() == 32 * seq_len * vocab_size

        expected_offset = 0
        for bin_index, (row_offset, bin_view) in enumerate(partitions):
            batch_size = batch_sizes[bin_index]
            assert row_offset == expected_offset
            assert bin_view.shape == (1, batch_size, seq_len, vocab_size)
            assert bin_view.is_contiguous()
            assert bin_view.storage_offset() == row_offset * seq_len * vocab_size
            # This is the exact indexing ABI used by dense IPC consumers.
            for sample_index_in_buf in range(batch_size):
                bin_view[0, sample_index_in_buf].fill_(bin_index + 1)
                assert torch.equal(
                    bin_view[0, sample_index_in_buf],
                    storage[row_offset + sample_index_in_buf],
                )
            expected_offset += batch_size
        assert expected_offset == storage.shape[0]

    def test_reuses_row_capacity_and_grows_before_partitioning(self):
        device = torch.device("cpu")
        storage = ensure_teacher_ipc_row_buffer(None, 8, 2, 3, torch.float32, device)
        reused = ensure_teacher_ipc_row_buffer(storage, 5, 2, 3, torch.float32, device)
        assert reused is storage

        grown = ensure_teacher_ipc_row_buffer(storage, 9, 2, 3, torch.float32, device)
        assert grown is not storage
        assert grown.shape == (9, 2, 3)

        with pytest.raises(ValueError, match="exceed storage capacity"):
            partition_teacher_ipc_row_buffer(reused, [5, 4])


class TestCompactTeacherIpcTokenBuffer:
    def test_reuses_token_capacity_and_grows_only_when_required(self):
        device = torch.device("cpu")
        storage = ensure_teacher_ipc_token_buffer(
            None,
            total_tokens=8,
            vocab_size=3,
            dtype=torch.float32,
            device=device,
        )
        reused = ensure_teacher_ipc_token_buffer(
            storage,
            total_tokens=5,
            vocab_size=3,
            dtype=torch.float32,
            device=device,
        )
        grown = ensure_teacher_ipc_token_buffer(
            storage,
            total_tokens=9,
            vocab_size=3,
            dtype=torch.float32,
            device=device,
        )

        assert reused is storage
        assert grown is not storage
        assert grown.shape == (9, 3)

    def test_qualification_like_capacity_is_sum_of_valid_cp_prefixes(self):
        full_seq_len = 3912
        valid_lengths_by_microbatch = [
            [3912, 2744, 1512],
            [2056, 1956, 1912, 264],
        ]
        flat_valid_lengths = [
            value for microbatch in valid_lengths_by_microbatch for value in microbatch
        ]
        localized_by_cp = [
            localize_teacher_ipc_valid_lengths(
                flat_valid_lengths,
                full_seq_len=full_seq_len,
                cp_rank=cp_rank,
                cp_size=2,
            )
            for cp_rank in range(2)
        ]

        assert sum(map(sum, localized_by_cp)) == sum(flat_valid_lengths)
        for local_valid_lengths in localized_by_cp:
            storage = ensure_teacher_ipc_token_buffer(
                None,
                total_tokens=sum(local_valid_lengths),
                vocab_size=3,
                dtype=torch.float32,
                device=torch.device("cpu"),
            )
            assert storage.shape == (sum(local_valid_lengths), 3)
            assert storage.shape[0] < len(flat_valid_lengths) * (full_seq_len // 2)

    def test_partitions_unequal_rows_without_allocation_or_padding(self):
        valid_lengths_by_microbatch = [[4, 2], [0, 3]]
        storage = ensure_teacher_ipc_token_buffer(
            None,
            total_tokens=9,
            vocab_size=2,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        partitions = partition_teacher_ipc_token_buffer(
            storage, valid_lengths_by_microbatch
        )

        assert [[offset for offset, _ in values] for values in partitions] == [
            [0, 4],
            [6, 6],
        ]
        assert [[tuple(view.shape) for _, view in values] for values in partitions] == [
            [(4, 2), (2, 2)],
            [(0, 2), (3, 2)],
        ]
        assert all(
            view.untyped_storage().data_ptr() == storage.untyped_storage().data_ptr()
            for values in partitions
            for _, view in values
        )

    def test_extracts_raw_lengths_only_for_packed_rows(self):
        data = {
            "input_lengths": torch.tensor([5, 2], dtype=torch.int64),
        }
        assert extract_teacher_ipc_valid_lengths(
            data, 2, full_seq_len=8, compact_padding=True
        ) == [5, 2]
        assert extract_teacher_ipc_valid_lengths(
            {}, 2, full_seq_len=8, compact_padding=False
        ) == [8, 8]

    def test_compact_handle_rejects_out_of_bounds_offset(self):
        handle = {
            "ipc_layout": DENSE_TEACHER_IPC_FLAT_LAYOUT,
            "payload_ipc": "payload",
            "storage_shape": (4, 3),
            "storage_token_offset": 3,
            "storage_used_tokens": 4,
            "storage_capacity_tokens": 4,
            "stored_seq_len": 2,
            "valid_seq_len": 2,
            "actual_shape": (4, 3),
            "global_seq_start": 0,
            "full_seq_len": 4,
            "dtype": torch.float32,
        }
        with pytest.raises(ValueError, match="outside storage bounds"):
            validate_compact_teacher_ipc_handle(handle)
