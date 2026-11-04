"""100 independent BF16 A/W pairs: raw TE/vLLM MXFP8 bytes and GEMM parity.

Run ``te`` and ``vllm`` phases in their respective backend interpreters, sharing
an output directory. Only compact quantized TE oracles and BF16 outputs are
saved; the same BF16 inputs are regenerated from seeds and verified by hashes.
No scales are applied in the quantized-tensor comparisons.
"""

import argparse
import importlib.metadata
import importlib.util
import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from run_te_quantizer_parity import cpuq, digest, fi_gemm, fi_quant, te_gemm, te_quant

SHAPES = [(32, 1856, 2688), (128, 1856, 2688), (32, 2688, 1856), (128, 2688, 1856)]
SCENARIOS = ["gaussian", "uniform", "block_scaled", "tiny_weights"]


def make_inputs(
    index: int, seed_base: int
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Generate BF16 sources on CPU identically in both backend processes."""
    m, n, k = SHAPES[index % len(SHAPES)]
    scenario = SCENARIOS[(index // len(SHAPES)) % len(SCENARIOS)]
    seed = seed_base + index
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if scenario == "uniform":
        activation = torch.rand((m, k), generator=generator) * 2 - 1
        weight = (torch.rand((n, k), generator=generator) * 2 - 1) * 0.02
    else:
        activation = torch.randn((m, k), generator=generator)
        weight = torch.randn((n, k), generator=generator)
        if scenario == "block_scaled":
            ae = torch.randint(
                -6, 7, (m, k // 32), generator=generator
            ).repeat_interleave(32, -1)
            we = torch.randint(
                -6, 7, (n, k // 32), generator=generator
            ).repeat_interleave(32, -1)
            activation = torch.ldexp(activation.double(), ae)
            weight = torch.ldexp(weight.double() * 0.02, we)
        elif scenario == "tiny_weights":
            # CPU FP64 construction avoids introducing GPU FTZ into input generation.
            weight = torch.ldexp(
                weight.double(), torch.full((n, k), -122, dtype=torch.int32)
            )
        else:
            weight *= 0.02
    activation, weight = activation.bfloat16(), weight.bfloat16()
    # Include zero blocks alongside nonzero random inputs; preserve logical shapes.
    az = torch.rand((m, k // 32), generator=generator) < 0.02
    wz = torch.rand((n, k // 32), generator=generator) < 0.02
    activation.reshape(m, k // 32, 32)[az] = 0
    weight.reshape(n, k // 32, 32)[wz] = 0
    assert torch.isfinite(activation).all() and torch.isfinite(weight).all()
    metadata = {
        "trial": index,
        "seed": seed,
        "scenario": scenario,
        "shape_m_n_k": [m, n, k],
        "activation_bf16_sha256": digest(activation),
        "weight_bf16_sha256": digest(weight),
    }
    return activation, weight, metadata


def compare_quantized(
    actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]
) -> dict[str, int]:
    """Compare FP8 code bytes and valid E8M0 bytes without dequantizing."""
    assert actual["data"].shape == expected["data"].shape
    assert actual["scale"].shape == expected["scale"].shape
    return {
        "fp8_code_bytes_different": int((actual["data"] != expected["data"]).sum()),
        "e8m0_scale_bytes_different": int((actual["scale"] != expected["scale"]).sum()),
        "fp8_code_count": actual["data"].numel(),
        "scale_count": actual["scale"].numel(),
        "unscaled_fp8_values_different": int(
            (
                actual["data"].contiguous().view(torch.float8_e4m3fn).float()
                != expected["data"].contiguous().view(torch.float8_e4m3fn).float()
            ).sum()
        ),
        "signed_zero_code_differences": int(
            (
                (actual["data"] != expected["data"])
                & ((actual["data"] & 127) == 0)
                & ((expected["data"] & 127) == 0)
            ).sum()
        ),
    }


def prepare(args: argparse.Namespace) -> None:
    oracles = args.output_dir / "oracles"
    oracles.mkdir(parents=True, exist_ok=True)
    manifest = []
    with torch.no_grad():
        for index in range(args.trials):
            activation, weight, metadata = make_inputs(index, args.seed)
            aq, ar = te_quant(activation.cuda())
            wq, wr = te_quant(weight.cuda())
            output = te_gemm(wq, aq)
            assert torch.isfinite(output).all(), metadata
            torch.save(
                {
                    "metadata": metadata,
                    "activation": cpuq(ar),
                    "weight": cpuq(wr),
                    "output": output.cpu(),
                },
                oracles / f"{index:03d}.pt",
            )
            manifest.append(metadata)
            print(
                "TE",
                index + 1,
                "/",
                args.trials,
                metadata["scenario"],
                metadata["shape_m_n_k"],
                flush=True,
            )
    (args.output_dir / "te-manifest.json").write_text(
        json.dumps({"torch": torch.__version__, "trials": manifest}, indent=2)
    )


def validate(args: argparse.Namespace) -> None:
    assert os.environ.get("NRL_VLLM_MXFP8_TE_COMPAT") == "1"
    assert args.patches_file is not None
    spec = importlib.util.spec_from_file_location(
        "nrl_compat_patches", args.patches_file
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.patch_flashinfer_mxfp8_te_compat(logging.getLogger("random-parity"))
    records = []
    saved_failures = 0
    with torch.no_grad():
        for index in range(args.trials):
            activation, weight, metadata = make_inputs(index, args.seed)
            oracle = torch.load(
                args.output_dir / "oracles" / f"{index:03d}.pt",
                map_location="cuda",
                weights_only=True,
            )
            assert metadata == oracle["metadata"], (
                "BF16 inputs are not identical",
                index,
            )
            activation, weight = activation.cuda(), weight.cuda()
            aq, wq = fi_quant(activation), fi_quant(weight)
            comparisons = {
                "activation_linear": compare_quantized(aq, oracle["activation"]),
                "weight_linear": compare_quantized(wq, oracle["weight"]),
                "activation_swizzled": compare_quantized(
                    fi_quant(activation, True), oracle["activation"]
                ),
                "weight_swizzled": compare_quantized(
                    fi_quant(weight, True), oracle["weight"]
                ),
            }
            output = fi_gemm(wq, aq).contiguous()
            expected = oracle["output"].contiguous()
            # Compare actual BF16 bytes, including the sign bit of zero.
            different_bytes = int(
                (output.view(torch.uint8) != expected.view(torch.uint8)).sum()
            )
            different_values = int((output != expected).sum())
            finite = bool(torch.isfinite(output).all())
            max_abs = float((output.float() - expected.float()).abs().max())
            passed = (
                finite
                and different_bytes == 0
                and all(
                    row["fp8_code_bytes_different"]
                    == row["e8m0_scale_bytes_different"]
                    == 0
                    for row in comparisons.values()
                )
            )
            record = {
                **metadata,
                "quantization": comparisons,
                "gemm_output_bytes_different": different_bytes,
                "gemm_output_values_different": different_values,
                "gemm_max_abs": max_abs,
                "gemm_finite": finite,
                "passed": passed,
            }
            records.append(record)
            if not passed and saved_failures < 3:
                torch.save(
                    {
                        "A": activation.cpu(),
                        "W": weight.cpu(),
                        "te": {
                            "activation": cpuq(oracle["activation"]),
                            "weight": cpuq(oracle["weight"]),
                            "output": expected.cpu(),
                        },
                        "vllm": {
                            "activation": cpuq(aq),
                            "weight": cpuq(wq),
                            "output": output.cpu(),
                        },
                        "record": record,
                    },
                    args.output_dir / f"failure-{index:03d}.pt",
                )
                saved_failures += 1
            (args.output_dir / "progress.json").write_text(
                json.dumps(
                    {
                        "completed_trials": len(records),
                        "failed_trials": sum(not row["passed"] for row in records),
                        "last": record,
                    },
                    indent=2,
                )
            )
            print(
                "VLLM",
                index + 1,
                "/",
                args.trials,
                "PASS" if passed else "FAIL",
                "output bytes:",
                different_bytes,
                "max_abs:",
                max_abs,
                flush=True,
            )
    passed_count = sum(row["passed"] for row in records)
    report = {
        "completed": len(records) == args.trials,
        "env_var": "NRL_VLLM_MXFP8_TE_COMPAT=1",
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "flashinfer": importlib.metadata.version("flashinfer-python"),
        "trials": args.trials,
        "passed": passed_count,
        "failed": args.trials - passed_count,
        "all_bf16_inputs_verified_identical_by_sha256": True,
        "quantized_comparisons_apply_scales": False,
        "gemm_comparison_uses_bf16_bytes_including_zero_sign": True,
        "scenario_counts": dict(Counter(row["scenario"] for row in records)),
        "quantization_comparisons": len(records) * 4,
        "quantization_code_mismatches": sum(
            q["fp8_code_bytes_different"]
            for row in records
            for q in row["quantization"].values()
        ),
        "quantization_scale_mismatches": sum(
            q["e8m0_scale_bytes_different"]
            for row in records
            for q in row["quantization"].values()
        ),
        "unscaled_fp8_numerical_mismatches": sum(
            q["unscaled_fp8_values_different"]
            for row in records
            for q in row["quantization"].values()
        ),
        "signed_zero_code_differences": sum(
            q["signed_zero_code_differences"]
            for row in records
            for q in row["quantization"].values()
        ),
        "gemm_output_bytes_different": sum(
            row["gemm_output_bytes_different"] for row in records
        ),
        "gemm_max_abs": max(row["gemm_max_abs"] for row in records),
        "cases": records,
    }
    (args.output_dir / "result.json").write_text(json.dumps(report, indent=2))
    assert (
        report["failed"] == 0
    ), f"{report['failed']} of {args.trials} trials failed; see result.json and failure-*.pt"
    print("ALL", args.trials, "RANDOM PARITY TRIALS PASSED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["te", "vllm"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026102700)
    parser.add_argument("--patches-file", type=Path)
    args = parser.parse_args()
    if args.trials <= 0:
        parser.error("--trials must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.phase == "te":
        prepare(args)
    else:
        validate(args)
