"""Boundary observers for clean-A / clean-B / poisoned checkpoint replay.

Installed only by the private audit loader. GPU synchronization is deliberate:
these runs localize numerical differences, rather than measure throughput.
"""

import functools
import contextvars
import os
from pathlib import Path

import torch
import torch.distributed as dist

from precision_probe_common import (
    mxfp8_storage_snapshot,
    save_json,
    tensor_hash,
    forward_input_fingerprint,
)

ROOT = (
    Path(os.environ["NANO_AUDIT_ROOT"]) / "boundaries" / os.environ["NANO_AUDIT_PHASE"]
)
COUNTERS = {}
REFIT_INDEX = 0
ACTIVE_REQUEST_KEY = contextvars.ContextVar("precision_probe_request_key", default=None)


def emit(stage, record):
    record = {"refit_index": REFIT_INDEX, **record}
    rank = dist.get_rank() if dist.is_initialized() else -1
    folder = ROOT / f"{stage}-rank-{rank}-pid-{os.getpid()}"
    folder.mkdir(parents=True, exist_ok=True)
    index = COUNTERS.get(stage, 0)
    COUNTERS[stage] = index + 1
    save_json(folder / f"{index:05d}.json", record)
    return folder / f"{index:05d}.pt"


def rng_state():
    return {
        "cpu": tensor_hash(torch.random.get_rng_state()),
        "cuda": [tensor_hash(v) for v in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_initialized()
        else [],
    }


def model_state(model):
    records = {}
    for name, param in model.named_parameters():
        obj = param.data
        if hasattr(obj, "quantized_tensors") or hasattr(
            obj, "split_into_quantized_tensors"
        ):
            members = getattr(obj, "quantized_tensors", None)
            if members is None:
                members = obj.split_into_quantized_tensors()
            records[name] = {
                "members": [
                    mxfp8_storage_snapshot(m.shape, m.get_metadata()) for m in members
                ]
            }
        elif hasattr(obj, "get_metadata"):
            metadata = obj.get_metadata()
            records[name] = mxfp8_storage_snapshot(param.shape, metadata)
            records[name]["logical_dtype"] = str(param.dtype)
        else:
            records[name] = {
                "shape": list(param.shape),
                "dtype": str(param.dtype),
                "sha256": tensor_hash(param),
            }
    buffers = {
        name: {"shape": list(t.shape), "dtype": str(t.dtype), "sha256": tensor_hash(t)}
        for name, t in model.named_buffers()
    }
    return {"parameters": records, "buffers": buffers, "storage_comparison_version": 3}


def install(name, module):
    if name == "nemo_rl.models.policy.workers.megatron_policy_worker":
        cls = module.MegatronPolicyWorkerImpl
        original_refit = cls._nccl_reshard_refit

        @functools.wraps(original_refit)
        def refit(self, *args, **kwargs):
            global REFIT_INDEX
            REFIT_INDEX += 1
            emit(
                "trainer_before_refit",
                {
                    "model": model_state(self.model),
                    "rng": rng_state(),
                    "native": self._is_native_mxfp8_export(),
                },
            )
            result = original_refit(self, *args, **kwargs)
            emit(
                "trainer_after_refit",
                {"model": model_state(self.model), "rng": rng_state()},
            )
            return result

        cls._nccl_reshard_refit = refit

        original_misc = cls._broadcast_misc_params_packed
        original_iter = cls._iter_params_with_optional_kv_scales

        @functools.wraps(original_iter)
        def weights(self, *args, **kwargs):
            for key, tensor in original_iter(self, *args, **kwargs):
                if getattr(self, "_precision_probe_sending_misc", False):
                    emit(
                        "misc_send",
                        {
                            "name": key,
                            "shape": list(tensor.shape),
                            "dtype": str(tensor.dtype),
                            "sha256": tensor_hash(tensor),
                        },
                    )
                yield key, tensor

        @functools.wraps(original_misc)
        def misc(self, *args, **kwargs):
            self._precision_probe_sending_misc = True
            try:
                return original_misc(self, *args, **kwargs)
            finally:
                self._precision_probe_sending_misc = False

        cls._iter_params_with_optional_kv_scales = weights
        cls._broadcast_misc_params_packed = misc

    elif name == "nemo_rl.weight_sync.xferdtensor":
        original = module.xferdtensor

        @functools.wraps(original)
        def transfer(
            src_tensor,
            src_mesh,
            src_placement,
            dst_tensor,
            dst_mesh,
            dst_placement,
            process_group,
            stream=None,
        ):
            # Hash source before transfer and receive buffer after completion.
            record = {
                "global_shape": list(
                    src_tensor.shape if src_tensor is not None else dst_tensor.shape
                ),
                "source_mesh": src_mesh.mesh.tolist(),
                "destination_mesh": dst_mesh.mesh.tolist(),
            }
            if stream is not None:
                stream.synchronize()
            if src_tensor is not None:
                t = src_tensor._local_tensor
                emit(
                    "wire_send",
                    {
                        **record,
                        "local_shape": list(t.shape),
                        "dtype": str(t.dtype),
                        "sha256": tensor_hash(t),
                    },
                )
            result = original(
                src_tensor,
                src_mesh,
                src_placement,
                dst_tensor,
                dst_mesh,
                dst_placement,
                process_group,
                stream,
            )
            if stream is not None:
                stream.synchronize()
            if dst_tensor is not None:
                t = dst_tensor._local_tensor
                emit(
                    "wire_receive",
                    {
                        **record,
                        "local_shape": list(t.shape),
                        "dtype": str(t.dtype),
                        "sha256": tensor_hash(t),
                    },
                )
            return result

        module.xferdtensor = transfer

    elif name == "nemo_rl.models.generation.vllm.vllm_backend":
        cls = module.VllmInternalWorkerExtension
        original = cls._nccl_reshard_refit_impl

        @functools.wraps(original)
        def receive(self, *args, **kwargs):
            global REFIT_INDEX
            REFIT_INDEX += 1
            result = original(self, *args, **kwargs)
            model = self.model_runner.model
            torch.cuda.synchronize()
            emit(
                "sampler_after_refit",
                {
                    "model": model_state(model),
                    "rng": rng_state(),
                    "configuration": str(self.model_runner.vllm_config),
                },
            )
            if not hasattr(model, "_precision_probe_original_logits"):
                original_logits = model.compute_logits
                model._precision_probe_original_logits = original_logits
                forward_context = {}
                head_context = {}

                def capture_forward_inputs(module, args, kwargs):
                    forward_context.clear()
                    forward_context.update(
                        forward_input_fingerprint(module.forward, args, kwargs)
                    )

                model.register_forward_pre_hook(
                    capture_forward_inputs, with_kwargs=True
                )

                def capture_logits(output):
                    if (
                        isinstance(output, torch.Tensor)
                        and COUNTERS.get("sampler_logits", 0) < 256
                    ):
                        finite = bool(torch.isfinite(output).all())
                        top = output.float().topk(2, dim=-1) if finite else None
                        path = emit(
                            "sampler_logits",
                            {
                                "shape": list(output.shape),
                                "dtype": str(output.dtype),
                                "sha256": tensor_hash(output),
                                "forward_context": dict(forward_context),
                                "forward_context_complete": all(
                                    k in forward_context
                                    for k in ["input_ids", "positions"]
                                ),
                                "lm_head_input": dict(head_context),
                                "finite": finite,
                                "top2_indices": top.indices.cpu().tolist()
                                if top is not None
                                else None,
                                "top1_top2_margin": (
                                    top.values[..., 0] - top.values[..., 1]
                                )
                                .cpu()
                                .tolist()
                                if top is not None
                                else None,
                                "rng": rng_state(),
                            },
                        )
                        torch.save(output.detach().cpu(), path)

                # A V2 runner can retain an earlier bound compute_logits method.
                # Module hooks still see its LogitsProcessor invocation.
                for child in model.modules():
                    if "LogitsProcessor" in type(child).__name__:

                        def capture_head_inputs(module, args, kwargs):
                            head_context.clear()
                            head_context.update(
                                forward_input_fingerprint(
                                    module.forward,
                                    args,
                                    kwargs,
                                    fields=("hidden_states",),
                                    include_values=False,
                                )
                            )

                        child.register_forward_pre_hook(
                            capture_head_inputs, with_kwargs=True
                        )
                        child.register_forward_hook(
                            lambda module, inputs, output: capture_logits(output)
                        )

                @functools.wraps(original_logits)
                def logits(*args, **kwargs):
                    before = COUNTERS.get("sampler_logits", 0)
                    head_context.clear()
                    output = original_logits(*args, **kwargs)
                    if COUNTERS.get("sampler_logits", 0) == before:
                        capture_logits(output)
                    return output

                model.compute_logits = logits
            return result

        cls._nccl_reshard_refit_impl = receive

        original_load = cls._load_weights

        @functools.wraps(original_load)
        def load(self, weights):
            def observed():
                for name, tensor in weights:
                    if REFIT_INDEX:
                        emit(
                            "misc_receive",
                            {
                                "name": name,
                                "shape": list(tensor.shape),
                                "dtype": str(tensor.dtype),
                                "sha256": tensor_hash(tensor),
                            },
                        )
                    yield name, tensor

            return original_load(self, observed())

        cls._load_weights = load

    elif name == "nemo_rl.models.generation.vllm.vllm_worker":
        cls = module.BaseVllmGenerationWorker
        original = cls._build_sampling_params

        @functools.wraps(original)
        def sampling(self, *args, **kwargs):
            params = original(self, *args, **kwargs)
            emit(
                "sampling_parameters",
                {
                    "request_key": ACTIVE_REQUEST_KEY.get(),
                    "parameters": {
                        k: getattr(params, k, None)
                        for k in [
                            "seed",
                            "temperature",
                            "top_k",
                            "top_p",
                            "max_tokens",
                            "stop_token_ids",
                            "stop",
                            "logprobs",
                        ]
                    },
                },
            )
            return params

        cls._build_sampling_params = sampling

    elif name == "nemo_rl.models.generation.vllm.vllm_worker_async":
        cls = module.VllmAsyncGenerationWorkerImpl
        original = cls.generate_async

        @functools.wraps(original)
        async def generate(self, data, *args, **kwargs):
            request_key = (
                tensor_hash(data["input_ids"])
                + ":"
                + tensor_hash(data["input_lengths"])
            )
            emit(
                "request_inputs",
                {
                    "request_key": request_key,
                    "tensors": {
                        k: {
                            "shape": list(v.shape),
                            "dtype": str(v.dtype),
                            "sha256": tensor_hash(v),
                            "values": v.cpu().tolist()
                            if k in ["input_ids", "input_lengths"]
                            else None,
                        }
                        for k, v in data.items()
                        if isinstance(v, torch.Tensor)
                    },
                    "kwargs": {k: str(v) for k, v in kwargs.items()},
                },
            )
            token = ACTIVE_REQUEST_KEY.set(request_key)
            try:
                async for index, result in original(self, data, *args, **kwargs):
                    emit(
                        "generated_outputs",
                        {
                            "request_key": request_key,
                            "index": index,
                            "output_ids": result["output_ids"].cpu().tolist(),
                            "lengths": result["generation_lengths"].cpu().tolist(),
                            "logprobs": result["logprobs"].cpu().tolist(),
                        },
                    )
                    yield index, result
            finally:
                ACTIVE_REQUEST_KEY.reset(token)

        cls.generate_async = generate
