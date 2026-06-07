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
import gc
import traceback
from typing import Any

import torch
import zmq

from nemo_rl.models.policy.utils import (
    IPCProtocol,
    calculate_aligned_size,
    rebuild_cuda_tensor_from_ipc,
)
from nemo_rl.utils.nsys import wrap_with_nvtx_name
from nemo_rl.utils.packed_tensor import packed_broadcast_consumer

try:
    import vllm  # noqa: F401
except ImportError:
    raise ImportError(
        "vLLM is not installed. Please check that the py_executable in the runtime_env of VllmGenerationWorker "
        "covers the vllm dependency. You may have to update nemo_rl/distributed/ray_actor_environment_registry.py. "
        "This error can also happen if the venv creation was aborted or errored out in the middle. In that case, "
        "please run at least once with the environment variable NRL_FORCE_REBUILD_VENVS=true set to force the rebuild of the environment."
    )


def fix_gpt_oss_export_transpose(key: str, weight: torch.Tensor) -> torch.Tensor:
    """Apply GPT-OSS down_proj transpose fix to the weight.

    This is a workaround for the issue that the down_proj layout is not the same across different frameworks.
        - HF needs [in, out] layout.
        - Megatron needs [in, out] layout.
        - vLLM needs [out, in] layout.
    See https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/3271 for more details.
    """
    if key.endswith("mlp.experts.down_proj"):
        weight = weight.transpose(-2, -1).contiguous()
    return weight


class VllmInternalWorkerExtension:
    def init_collective(
        self,
        rank_prefix: int,
        ip: str,
        port: int,
        world_size: int,
        train_world_size: int,
    ) -> None:
        """Initialize the collective communication."""
        import time as _time

        from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup

        local_rank = torch.distributed.get_rank()
        # Place vLLM ranks after all training ranks so all training workers can join
        rank = train_world_size + rank_prefix + local_rank

        _t_phases: dict[str, float] = {}
        _t_overall = _time.monotonic()

        # RL-412: idempotent re-init — abort the leftover NCCL comm
        # FIRST (without sync, so we don't block on ops whose peer just
        # left), then drain + clear queued async errors via try/except'd
        # sync + empty_cache. Without the swallow, the next CUDA op
        # surfaces ``cudaErrorLaunchFailure`` and crashes the worker.
        # Mirror of base_policy_worker.init_collective.
        _t = _time.monotonic()
        old = getattr(self, "model_update_group", None)
        if old is not None:
            try:
                old.destroy()
            except Exception:  # noqa: BLE001
                pass
            try:
                torch.cuda.synchronize()
            except Exception:  # noqa: BLE001
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
        _t_phases["destroy_old"] = _time.monotonic() - _t

        _t = _time.monotonic()
        self.model_update_group = StatelessProcessGroup(  # pyrefly: ignore[implicitly-defined-attribute]  This class does not define __init__ so assignments like this should be ignored
            master_address=ip, port=port, rank=rank, world_size=world_size
        )
        _t_phases["tcp_store_ctor"] = _time.monotonic() - _t

        _t = _time.monotonic()
        self.model_update_group.init_nccl_communicator(device=self.device)
        _t_phases["nccl_init"] = _time.monotonic() - _t

        _t_total = _time.monotonic() - _t_overall
        # Only log on slow inits (>3s) to avoid noise on the steady-state
        # ~2.7s case.
        if _t_total > 3.0:
            print(
                f"[init_collective_timing] rank={rank} world={world_size} "
                f"total={_t_total:.2f}s "
                f"destroy_old={_t_phases.get('destroy_old', 0):.2f}s "
                f"tcp_store_ctor={_t_phases.get('tcp_store_ctor', 0):.2f}s "
                f"nccl_init={_t_phases.get('nccl_init', 0):.2f}s",
                flush=True,
            )

    def reset_collective(self) -> None:
        """Tear down the cross-cluster weight-sync collective on this worker.

        Order: ``group.destroy()`` FIRST, then ``cuda.synchronize()``.
        ``destroy()`` calls ``ncclCommAbort`` which immediately kills any
        in-flight NCCL op (including a broadcast hung waiting for a dead
        peer). After the abort, ``cuda.synchronize()`` drains remaining
        side-stream work and returns quickly (~ms instead of 60-100s).

        The previous order (synchronize → destroy) caused reset_collective
        to block for the full NCCL heartbeat timeout (60s+) when a peer
        died mid-broadcast, because synchronize waited for the hung op
        to complete before the abort could fire.

        Idempotent — a no-op if no collective is currently held.
        """
        group = getattr(self, "model_update_group", None)
        if group is None:
            return
        import torch

        # Abort the NCCL comm first — kills hung broadcasts immediately.
        try:
            group.destroy()
        except Exception as e:  # noqa: BLE001
            print(f"[vllm_backend.reset_collective] destroy raised {e}", flush=True)
        # Now drain side streams + clear cache (fast after abort).
        for fn in (torch.cuda.synchronize, torch.cuda.empty_cache):
            try:
                fn()
            except Exception:  # noqa: BLE001
                pass
        self.model_update_group = None  # type: ignore[assignment]

    def report_device_id(self) -> str:
        """Retrieve the UUID of the current CUDA device."""
        from nemo_rl.utils.nvml import get_device_uuid

        return get_device_uuid(self.device.index)

    def get_zmq_address(self):
        """Get the ZMQ address for the current device."""
        return f"ipc:///tmp/{self.report_device_id()}.sock"

    def maybe_init_zmq(self):
        """Initialize the ZMQ socket if it doesn't exist."""
        if not hasattr(self, "zmq_socket"):
            self.zmq_context = zmq.Context()  # pyrefly: ignore[implicitly-defined-attribute]  This class does not define __init__ so assignments like this should be ignored
            self.zmq_socket = self.zmq_context.socket(  # pyrefly: ignore[implicitly-defined-attribute]  This class does not define __init__ so assignments like this should be ignored
                zmq.REP
            )
            self.zmq_socket.setsockopt(
                zmq.SNDTIMEO, 120000
            )  # set timeout to 120 seconds
            self.zmq_socket.setsockopt(
                zmq.RCVTIMEO, 120000
            )  # set timeout to 120 seconds
            self.zmq_socket.setsockopt(zmq.LINGER, 0)
            self.zmq_socket.connect(self.get_zmq_address())

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        """Prepare state dict metadata for weight refitting and IPC streaming.

        Args:
            state_dict_info (dict): A dictionary containing the info for refit.
                e.g. {tensor_name: (shape, dtype)}
        """
        self.state_dict_info = state_dict_info  # pyrefly: ignore[implicitly-defined-attribute]  This class does not define __init__ so assignments like this should be ignored

    def warmup_nccl_library(self) -> dict[str, float]:
        """Pre-warm the NCCL library state in this worker's process.

        On freshly-spawned gen pods (TP=1/PP=1/EP=1 — the disagg recipe
        for Qwen3-30B-A3B), there is NO intra-shard NCCL group, so the
        cross-cluster ``model_update_group`` is the FIRST NCCL collective
        the engine_core process ever sees. NCCL's per-process lazy init
        runs on that first call: shared libraries get dlopened, IB
        devices get probed, NCCL workspace gets cuMemMap'd. On a fresh
        pod this can add 15-20s to the first ``init_collective`` —
        which surfaces as a step-time spike on the FIRST refit after
        the shard joins.

        We pre-pay that cost here by creating a degenerate 1-rank NCCL
        communicator and tearing it down. NCCL allows ``nranks=1``
        (operations are no-ops, but the per-process state machine
        runs through the same init paths as a real comm). The warmup
        takes ~1-2s once and amortizes across the entire lifetime of
        this worker — every subsequent ``init_collective`` reuses the
        primed library state.

        Returns a small timing dict so the caller can log how long
        the warmup actually took (useful for confirming the fix
        actually shaved time off the steady-state path).

        Idempotent. Safe to call multiple times.
        """
        import time as _time

        from nccl.core.communicator import Communicator
        from nccl.core.utils import get_unique_id

        if getattr(self, "_nccl_library_warmed", False):
            return {"skipped": True, "total_s": 0.0}

        timings: dict[str, float] = {}
        _t = _time.monotonic()
        try:
            unique_id = get_unique_id()
            timings["unique_id_s"] = _time.monotonic() - _t

            _t = _time.monotonic()
            with torch.cuda.device(self.device):
                comm = Communicator.init(
                    nranks=1,
                    rank=0,
                    unique_id=unique_id,
                )
                timings["comm_init_s"] = _time.monotonic() - _t

                _t = _time.monotonic()
                # No-op broadcast on a 1-element tensor — runs through
                # NCCL's stream/kernel submission path and forces any
                # remaining lazy init to complete.
                data = torch.ones(1, device=self.device)
                comm.broadcast(
                    sendbuf=data,
                    recvbuf=data,
                    root=0,
                    stream=int(torch.cuda.current_stream().cuda_stream),
                )
                torch.cuda.current_stream().synchronize()
                timings["warmup_op_s"] = _time.monotonic() - _t

                _t = _time.monotonic()
                for method in ("abort", "destroy", "finalize"):
                    fn = getattr(comm, method, None)
                    if callable(fn):
                        try:
                            fn()
                            break
                        except Exception:  # noqa: BLE001
                            continue
                timings["destroy_s"] = _time.monotonic() - _t

            self._nccl_library_warmed = True  # pyrefly: ignore[implicitly-defined-attribute]
            timings["total_s"] = sum(v for v in timings.values() if isinstance(v, float))
            print(
                f"[warmup_nccl_library] device={self.device} "
                f"total={timings['total_s']:.2f}s "
                f"unique_id={timings.get('unique_id_s', 0):.3f}s "
                f"comm_init={timings.get('comm_init_s', 0):.3f}s "
                f"warmup_op={timings.get('warmup_op_s', 0):.3f}s "
                f"destroy={timings.get('destroy_s', 0):.3f}s",
                flush=True,
            )
            return timings
        except Exception as e:  # noqa: BLE001
            import traceback as _tb

            print(
                f"[warmup_nccl_library] failed: {type(e).__name__}: {e}\n"
                f"{_tb.format_exc()}",
                flush=True,
            )
            return {"error": str(e), "total_s": _time.monotonic() - _t}

    def _maybe_process_fp8_kv_cache(self) -> None:
        """Process weights after loading for FP8 KV cache (static scales)."""
        use_fp8_kv_cache = False
        if hasattr(self.model_runner.vllm_config, "cache_config"):
            kv_cache_dtype = getattr(
                self.model_runner.vllm_config.cache_config, "cache_dtype", None
            )
            use_fp8_kv_cache = (
                kv_cache_dtype is not None and "fp8" in str(kv_cache_dtype).lower()
            )

        if not use_fp8_kv_cache:
            return

        # FP8 KV cache: process KV scales after weight loading
        from vllm.model_executor.model_loader.utils import (
            process_weights_after_loading,
        )

        # Get target device for processing
        target_device = next(self.model_runner.model.parameters()).device

        # Call process_weights_after_loading to handle KV scales
        process_weights_after_loading(
            self.model_runner.model,
            self.model_runner.model_config,
            target_device,
        )

    @staticmethod
    def _split_policy_and_draft_weights(
        weights: list[tuple[str, torch.Tensor]],
    ) -> tuple[list[tuple[str, torch.Tensor]], list[tuple[str, torch.Tensor]]]:
        """Split trainer-owned draft weights from policy weights.

        This path is only used for the Eagle3 online-training flow, where the
        trainer exports draft parameters under a `draft.` prefix before sending
        them to vLLM.
        This implementation is specific to the eagle model. For MTP, we can add
        similar logic to this function to split weights and send it to the drafter.
        The "draft." prefix is added here https://github.com/isomap/RL/blob/d3a5e1396d00f82fb888d9ec6800687a23bb4017/nemo_rl/models/policy/workers/megatron_policy_worker.py#L967-L997
        """
        policy_weights = []
        draft_weights = []
        for key, tensor in weights:
            if key.startswith("draft."):
                draft_weights.append((key.removeprefix("draft."), tensor))
            else:
                policy_weights.append((key, tensor))
        return policy_weights, draft_weights

    def _load_draft_weights(
        self, draft_weights: list[tuple[str, torch.Tensor]]
    ) -> None:
        if not draft_weights:
            return

        draft_owner = getattr(self.model_runner, "drafter", None)
        draft_model = getattr(draft_owner, "model", None) if draft_owner else None

        if draft_model is None:
            print(
                "[draft] Received draft weights but vLLM drafter is unavailable; skipping draft update."
            )
            return
        draft_model.load_weights(weights=draft_weights)

    def _load_weights(self, weights):
        """Load weights with GptOss transpose fix, FP8, and draft-weight support.

        Applies GPT-OSS down_proj transpose if needed, splits policy/draft
        weights, applies FP8 conversion if needed, and loads draft weights
        into the drafter model.
        """
        from nemo_rl.models.generation.vllm.quantization import fp8

        if (
            "GptOssForCausalLM"
            in self.model_runner.vllm_config.model_config.architectures
        ):
            for idx, (key, weight) in enumerate(weights):
                weight = fix_gpt_oss_export_transpose(key, weight)
                weights[idx] = (key, weight)

        policy_weights, draft_weights = self._split_policy_and_draft_weights(weights)
        if fp8.is_fp8_model(self.model_runner.vllm_config):
            fp8.load_weights(policy_weights, self.model_runner)
        else:
            self.model_runner.model.load_weights(weights=policy_weights)

        self._load_draft_weights(draft_weights)

    @wrap_with_nvtx_name("vllm_internal_worker_extension/update_weights_via_ipc_zmq")
    def update_weights_via_ipc_zmq(self) -> bool:
        """Receive and update model weights via ZMQ IPC socket.

        Returns:
            bool: True if weights were successfully updated.
        """
        buffer = None
        weights = None

        try:
            self.maybe_init_zmq()
            while True:
                # Blocking receive with timeout (this is the main operation)
                payload = self.zmq_socket.recv_pyobj()

                if payload == IPCProtocol.COMPLETE:
                    # means the update is done
                    from vllm.model_executor.model_loader.utils import (
                        process_weights_after_loading,
                    )

                    process_weights_after_loading(
                        self.model_runner.model, self.model_config, self.device
                    )
                    self.zmq_socket.send(IPCProtocol.ACK.value.encode())
                    break

                ipc_handle, list_keys, used_bytes = payload
                buffer = rebuild_cuda_tensor_from_ipc(ipc_handle, self.device.index)

                weight = None
                weights = []
                offset = 0
                for key in list_keys:
                    shape, dtype = self.state_dict_info[key]  # pyrefly
                    if isinstance(shape, list):
                        shape = torch.Size(shape)

                    # Get the weight from the buffer
                    size_in_bytes = dtype.itemsize * shape.numel()
                    weight = (
                        buffer[offset : offset + size_in_bytes]
                        .view(dtype=dtype)
                        .view(shape)
                    )
                    # apply gpt-oss transpose fix
                    if (
                        "GptOssForCausalLM"
                        in self.model_runner.vllm_config.model_config.architectures
                    ):
                        weight = fix_gpt_oss_export_transpose(key, weight)
                    weights.append((key, weight))

                    # Move offset to the next weight
                    aligned_size = calculate_aligned_size(size_in_bytes)
                    offset += aligned_size

                assert offset == used_bytes, (
                    "Offset is not equal to used bytes, usually indicate inaccurate info like keys or cached dtype in state_dict_info"
                )

                # Load weights into the model
                self._load_weights(weights)

                torch.cuda.current_stream().synchronize()

                # CRITICAL: Delete views before ACK to prevent corruption.
                # 'weights' contains views into IPC shared memory. Even though load_weights()
                # copied the data, Python may not garbage collect these view objects immediately.
                # If sender reuses the buffer before GC runs, old views would read corrupted data.
                # Explicit del ensures immediate cleanup before sending ACK.
                del weight, weights, buffer
                weight = None
                weights = None
                buffer = None
                self.zmq_socket.send(IPCProtocol.ACK.value.encode())

            # Process weights after loading for FP8 KV cache
            self._maybe_process_fp8_kv_cache()

            gc.collect()
            torch.cuda.empty_cache()
            return True
        except Exception as e:
            print(
                f"Error in VllmInternalWorkerExtension.update_weights_via_ipc_zmq: {e}.\n"
                f"{traceback.format_exc()}"
            )
            return False

    @wrap_with_nvtx_name(
        "vllm_internal_worker_extension/update_weights_from_collective"
    )
    def update_weights_from_collective(self) -> bool:
        """Update the model weights from collective communication."""
        assert self.state_dict_info is not None, (
            "state_dict_info is not prepared. "
            "Please call prepare_refit_info when initializing the worker."
        )

        load_model_weight_func = self._load_weights

        # DIAGNOSTIC: log consumer-side ordering fingerprint so we can compare
        # against the producer (megatron_policy_worker).  ``state_dict_info``
        # came from ``prepare_refit_info`` at startup; if its iteration order
        # diverges from the producer's runtime ``_iter_params_with_optional_kv_scales``
        # the packed-broadcast byte boundaries skew silently and the
        # rebuilt tensors are garbage.
        try:
            import hashlib as _hashlib
            import math as _math
            import os as _os

            _consumer_meta = [
                (
                    name,
                    tuple(shape),
                    str(dtype),
                    _math.prod(shape) * dtype.itemsize,
                )
                for name, (shape, dtype) in self.state_dict_info.items()
            ]
            _names_blob = "\n".join(m[0] for m in _consumer_meta).encode()
            _full_blob = "\n".join(
                f"{m[0]}|{m[1]}|{m[2]}|{m[3]}" for m in _consumer_meta
            ).encode()
            _names_hash = _hashlib.sha256(_names_blob).hexdigest()[:16]
            _full_hash = _hashlib.sha256(_full_blob).hexdigest()[:16]
            _total_bytes = sum(m[3] for m in _consumer_meta)
            _vrank = _os.environ.get("VLLM_DP_RANK", _os.environ.get("RANK", "?"))
            print(
                f"[refit_diag.consumer] vllm_dp_rank={_vrank} "
                f"n_params={len(_consumer_meta)} "
                f"total_bytes={_total_bytes} "
                f"names_sha={_names_hash} full_sha={_full_hash} "
                f"first3={[m[0] for m in _consumer_meta[:3]]} "
                f"last3={[m[0] for m in _consumer_meta[-3:]]}",
                flush=True,
            )
        except Exception as _e:  # noqa: BLE001
            print(
                f"[refit_diag.consumer] failed to log fingerprint: {_e}",
                flush=True,
            )

        success = True
        try:
            packed_broadcast_consumer(
                iterator=iter(self.state_dict_info.items()),
                group=self.model_update_group,
                src=0,
                post_unpack_func=load_model_weight_func,
            )

            # Process weights after loading for FP8 KV cache
            self._maybe_process_fp8_kv_cache()

        except Exception as e:
            success = False
            import traceback as _tb
            print(
                f"Error in VllmInternalWorkerExtension.update_weights_from_collective: {e}\n"
                f"{_tb.format_exc()}",
                flush=True,
            )
        finally:
            # Tear down the cross-cluster ``model_update_group`` at the END
            # of every refit — but ONLY in disaggregated mode where shards
            # can die independently and stale NCCL state would poison the
            # next refit. In single-cluster (non-disagg) mode the group is
            # stable for the entire run, so teardown is unnecessary overhead
            # (~2.5s re-init per step).
            #
            # Disagg mode is identified by DISAGG_JOB_ID env var, set by
            # run_standalone_generation_server.py's entrypoint.
            import os as _os

            is_disagg = bool(_os.environ.get("DISAGG_JOB_ID"))
            if is_disagg:
                group = getattr(self, "model_update_group", None)
                if group is not None:
                    try:
                        group.destroy()
                    except Exception as _e:  # noqa: BLE001
                        print(
                            f"[vllm_backend] post-refit destroy raised "
                            f"{type(_e).__name__}: {_e} — comm already gone",
                            flush=True,
                        )
                    self.model_update_group = None  # pyrefly: ignore
        return success

    def cleanup(self) -> None:
        """Shutdown and cleanup resources."""
        # Close ZMQ socket and context if they exist
        if hasattr(self, "zmq_socket"):
            self.zmq_socket.close()
            self.zmq_context.term()

    def start_gpu_profiling(self) -> None:
        """Start GPU profiling."""
        torch.cuda.profiler.start()

    def stop_gpu_profiling(self) -> None:
        """Stop GPU profiling."""
        torch.cuda.profiler.stop()
