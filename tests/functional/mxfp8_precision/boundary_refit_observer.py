"""Read-only refit observers, enabled only by the isolated audit's import hook."""

import os
from pathlib import Path

import torch.distributed as dist

from precision_probe_common import save_json, tensor_hash
from boundary_probe_common import effective_mxfp8_fingerprint

COUNTS = {}
LAYERS = {}


def emit(stage, record):
    rank = dist.get_rank() if dist.is_initialized() else -1
    root = Path(os.environ["NANO_BOUNDARY_ROOT"]) / os.environ.get("NANO_BOUNDARY_MODE", "driver")
    folder = root / f"{stage}-rank-{rank}-pid-{os.getpid()}"
    folder.mkdir(parents=True, exist_ok=True)
    index = COUNTS.get(stage, 0)
    COUNTS[stage] = index + 1
    path = folder / f"{index:05d}"
    save_json(path.with_suffix(".json"), record)
    return path.with_suffix(".pt")


def fingerprint(tensor):
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "sha256": tensor_hash(tensor)}


def install(name, module):
    if name in {"nemo_rl.weight_sync.xferdtensor", "nemo_rl.models.generation.vllm.vllm_backend"}:
        # Reuse the tested wire, model, request-context and logits capture.
        os.environ["NANO_AUDIT_ROOT"] = os.environ["NANO_BOUNDARY_ROOT"]
        os.environ["NANO_AUDIT_PHASE"] = os.environ.get("NANO_BOUNDARY_MODE", "driver")
        import rollout_diagnostics as diagnostics

        diagnostics.emit = emit
        diagnostics.COUNTERS = COUNTS
        diagnostics.install(name, module)
        if name.endswith("vllm_backend"):
            cls = module.VllmInternalWorkerExtension
            original = cls._nccl_reshard_refit_impl

            def refit(self, *args, **kwargs):
                LAYERS.clear()
                LAYERS.update({id(m): n for n, m in self.model_runner.model.named_modules()})
                return original(self, *args, **kwargs)

            cls._nccl_reshard_refit_impl = refit
    elif name == "nemo_rl.models.generation.vllm.quantization.fp8":
        original = module.process_weights_after_loading_mxfp8_moe

        def process(self, layer):
            name = LAYERS.get(id(layer))
            if name is not None:
                values = {}
                effective = {}
                for key in ["w13_weight", "w2_weight"]:
                    scale = key + "_scale_from_checkpoint"
                    if not hasattr(layer, scale):
                        scale = key + "_scale"
                    values[key] = fingerprint(getattr(layer, key))
                    values[scale.removesuffix("_from_checkpoint")] = fingerprint(getattr(layer, scale))
                    effective[key] = effective_mxfp8_fingerprint(getattr(layer, key), getattr(layer, scale))
                emit("sampler_logical_mxfp8", {"layer": name, "weights": values, "effective_weights": effective})
            result = original(self, layer)
            if name is not None:
                emit("sampler_execution_mxfp8", {"layer": name, "weights": {
                    key: fingerprint(getattr(layer, key))
                    for key in ["w13_weight_for_apply", "w2_weight_for_apply", "w13_weight_scale", "w2_weight_scale"]
                    if hasattr(layer, key)
                }})
            return result

        module.process_weights_after_loading_mxfp8_moe = process
