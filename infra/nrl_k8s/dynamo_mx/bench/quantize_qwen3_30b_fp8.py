#!/usr/bin/env python3
"""Data-free FP8 quantization for the exact Qwen/Qwen3-30B-A3B model.

This intentionally has no fallback quantizer. If llm-compressor,
compressed-tensors, or their required runtime dependencies are unavailable, it
prints a machine-readable dependency report and exits without writing a model.

Example:
  python3 quantize_qwen3_30b_fp8.py --check-only
  python3 quantize_qwen3_30b_fp8.py \
    --output /mnt/rl-workspace/checkpoints/Qwen3-30B-A3B-FP8-block
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path
from typing import Any


MODEL_ID = "Qwen/Qwen3-30B-A3B"
MODEL_REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
EXPECTED_ARCHITECTURE = "Qwen3MoeForCausalLM"
EXPECTED_MODEL_CONFIG = {
    "model_type": "qwen3_moe",
    "hidden_size": 2048,
    "num_hidden_layers": 48,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 768,
}
REQUIRED_MODULES = {
    "torch": "torch",
    "transformers": "transformers",
    "llmcompressor": "llmcompressor",
    "compressed_tensors": "compressed-tensors",
    "safetensors": "safetensors",
}
OPTIONAL_MODULES = {"vllm": "vllm"}


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def dependency_report() -> dict[str, Any]:
    packages = {}
    missing = []
    for module, distribution in {**REQUIRED_MODULES, **OPTIONAL_MODULES}.items():
        try:
            importlib.import_module(module)
            importable = True
        except Exception as exc:  # noqa: BLE001 - report binary/import failures
            importable = False
            error = f"{type(exc).__name__}: {exc}"
        entry = {
            "distribution": distribution,
            "version": _version(distribution),
            "importable": importable,
            "required": module in REQUIRED_MODULES,
        }
        if not importable:
            entry["error"] = error
            if module in REQUIRED_MODULES:
                missing.append(module)
        packages[module] = entry
    return {
        "status": "ok" if not missing else "missing_dependencies",
        "missing_required": missing,
        "packages": packages,
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _config_dict(config: Any) -> dict[str, Any]:
    if hasattr(config, "to_dict"):
        return config.to_dict()
    raise RuntimeError(f"configuration object {type(config).__name__} has no to_dict()")


def _validate_exact_model(config: Any, requested_model: str) -> dict[str, Any]:
    raw = _config_dict(config)
    errors = []
    if requested_model != MODEL_ID:
        errors.append(f"model id must be exactly {MODEL_ID!r}, got {requested_model!r}")
    architectures = raw.get("architectures") or []
    if architectures != [EXPECTED_ARCHITECTURE]:
        errors.append(
            f"architectures must be [{EXPECTED_ARCHITECTURE!r}], got {architectures!r}"
        )
    for key, expected in EXPECTED_MODEL_CONFIG.items():
        observed = raw.get(key)
        if key == "num_experts" and observed is None:
            # Transformers 5.x normalizes the on-disk ``num_experts`` field
            # to ``num_local_experts`` in Qwen3MoeConfig.to_dict().
            observed = raw.get("num_local_experts")
        if observed != expected:
            errors.append(f"{key} must be {expected!r}, got {observed!r}")
    if errors:
        raise RuntimeError("wrong source model:\n  - " + "\n  - ".join(errors))
    return {
        "model_id": requested_model,
        "architectures": architectures,
        **{
            key: (
                raw.get("num_experts", raw.get("num_local_experts"))
                if key == "num_experts"
                else raw[key]
            )
            for key in EXPECTED_MODEL_CONFIG
        },
        "_commit_hash": getattr(config, "_commit_hash", None),
    }


def _tensor_inventory(output: Path) -> dict[str, Any]:
    import torch
    from safetensors import safe_open

    dtype_counts: dict[str, int] = {}
    scale_suffix_counts: dict[str, int] = {}
    tensor_count = 0
    float8_tensors = 0
    weight_names = set()
    scale_bases = set()
    shards = sorted(output.glob("*.safetensors"))
    if not shards:
        raise RuntimeError(f"no safetensors shards found under {output}")
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                tensor = handle.get_tensor(name)
                tensor_count += 1
                dtype = str(tensor.dtype).removeprefix("torch.")
                dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
                if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                    float8_tensors += 1
                if name.endswith(".weight"):
                    weight_names.add(name.removesuffix(".weight"))
                for suffix in (
                    ".weight_scale",
                    ".weight_scale_inv",
                    ".input_scale",
                    ".activation_scale",
                ):
                    if name.endswith(suffix):
                        scale_suffix_counts[suffix[1:]] = (
                            scale_suffix_counts.get(suffix[1:], 0) + 1
                        )
                        if suffix.startswith(".weight_scale"):
                            scale_bases.add(name.removesuffix(suffix))
                        break
    if float8_tensors == 0:
        raise RuntimeError("saved checkpoint contains no float8 tensors")
    if not scale_bases:
        raise RuntimeError("saved checkpoint contains no FP8 weight scale tensors")
    orphan_scales = sorted(scale_bases - weight_names)
    if orphan_scales:
        raise RuntimeError(
            "saved checkpoint has weight scales without weight tensors: "
            f"{orphan_scales[:10]}"
        )
    return {
        "shard_count": len(shards),
        "tensor_count": tensor_count,
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "float8_tensor_count": float8_tensors,
        "scale_suffix_counts": dict(sorted(scale_suffix_counts.items())),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_with_vllm(
    output: Path, dependencies: dict[str, Any]
) -> dict[str, Any]:
    if not dependencies["packages"]["vllm"]["importable"]:
        return {"status": "not_installed"}
    try:
        from vllm.transformers_utils.config import get_config

        config = get_config(str(output), trust_remote_code=False)
        raw = _config_dict(config)
    except Exception as exc:  # noqa: BLE001 - installed vLLM must accept output
        raise RuntimeError(
            "installed vLLM rejected the generated checkpoint configuration: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return {
        "status": "accepted",
        "model_type": raw.get("model_type"),
        "architectures": raw.get("architectures"),
        "quant_method": (raw.get("quantization_config") or {}).get("quant_method"),
    }


def quantize(args: argparse.Namespace, dependencies: dict[str, Any]) -> None:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    output = args.output.resolve()
    if output.is_file():
        raise RuntimeError(f"output path is a file: {output}")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"output directory is not empty: {output}")

    source_config = AutoConfig.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=False
    )
    source = _validate_exact_model(source_config, args.model)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        dtype="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=False
    )

    calibration_wrapper = "not_available"
    try:
        from llmcompressor.modeling import replace_modules_for_calibration

        model = replace_modules_for_calibration(model)
        calibration_wrapper = "llmcompressor.modeling.replace_modules_for_calibration"
    except ImportError:
        # Newer llm-compressor versions apply the replacement internally.
        pass

    recipe_spec = {
        "modifier": "QuantizationModifier",
        "targets": ["Linear"],
        "scheme": "FP8_BLOCK",
        "ignore": ["lm_head", "re:.*mlp.gate$"],
        "calibration": "data-free",
    }
    recipe = QuantizationModifier(
        targets=recipe_spec["targets"],
        scheme=recipe_spec["scheme"],
        ignore=recipe_spec["ignore"],
    )
    oneshot(model=model, recipe=recipe)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)

    saved_config_path = output / "config.json"
    if not saved_config_path.is_file():
        raise RuntimeError("quantizer did not emit config.json")
    saved_config = json.loads(saved_config_path.read_text())
    _validate_exact_model(
        AutoConfig.from_pretrained(output, trust_remote_code=False), args.model
    )
    quantization_config = saved_config.get("quantization_config")
    if not isinstance(quantization_config, dict):
        raise RuntimeError("saved config.json has no quantization_config object")
    if quantization_config.get("quant_method") not in {"compressed-tensors", "fp8"}:
        raise RuntimeError(
            "saved checkpoint is not marked for a vLLM FP8 loader: "
            f"{quantization_config!r}"
        )

    inventory = _tensor_inventory(output)
    vllm_validation = _validate_with_vllm(output, dependencies)
    metadata = {
        "status": "complete",
        "source": source,
        "revision_requested": args.revision,
        "recipe": recipe_spec,
        "calibration_wrapper": calibration_wrapper,
        "dependencies": dependencies,
        "output": {
            "path": str(output),
            "config_sha256": _sha256(saved_config_path),
            "quantization_config": quantization_config,
            "tensor_inventory": inventory,
            "vllm_validation": vllm_validation,
        },
    }
    _write_json(output / "fp8_quantization_metadata.json", metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID, help=argparse.SUPPRESS)
    parser.add_argument(
        "--revision",
        default=MODEL_REVISION,
        help=f"Pinned Hugging Face revision (default: {MODEL_REVISION}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Qwen3-30B-A3B-FP8-block"),
        help="New/empty output directory.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Import dependencies and print versions; do not load or write a model.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dependencies = dependency_report()
    if args.check_only or dependencies["status"] != "ok":
        stream = sys.stdout if dependencies["status"] == "ok" else sys.stderr
        print(json.dumps(dependencies, indent=2, sort_keys=True), file=stream)
        return 0 if dependencies["status"] == "ok" else 2
    try:
        quantize(args, dependencies)
    except Exception as exc:  # noqa: BLE001 - cluster entrypoint needs one clear failure
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
