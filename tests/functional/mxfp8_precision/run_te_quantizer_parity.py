"""Independent BF16 -> quantizer -> MXFP8 GEMM; compare codes, never dequantize."""

import argparse
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT: Path
SOURCE: Path
PATCHES_FILE: Path


def digest(x: torch.Tensor) -> str:
    return hashlib.sha256(
        x.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()


def cpuq(q: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in q.items()}


def te_quant(x: torch.Tensor) -> tuple[Any, dict[str, torch.Tensor]]:
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
    import transformer_engine_torch as tex

    matrix = x.reshape(-1, x.shape[-1]).contiguous()
    q = MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=True, columnwise=False).quantize(
        matrix
    )
    md = q.get_metadata()
    m, k = matrix.shape
    assert not md["with_gemm_swizzled_scales"]
    return q, {
        "data": md["rowwise_data"].reshape(m, k).view(torch.uint8).clone(),
        "scale": md["rowwise_scale_inv"][:m, : k // 32].clone(),
    }


def te_gemm(w: Any, a: Any) -> torch.Tensor:
    from transformer_engine.pytorch.cpp_extensions.gemm import general_grouped_gemm
    from transformer_engine.pytorch.module.grouped_linear import _2X_ACC_FPROP

    out = torch.empty((a.size(0), w.size(0)), device="cuda", dtype=torch.bfloat16)
    general_grouped_gemm(
        [w],
        [a],
        [out],
        [None],
        torch.bfloat16,
        layout="TN",
        m_splits=[a.size(0)],
        use_split_accumulator=_2X_ACC_FPROP,
    )
    return out


def prepare() -> None:
    torch.manual_seed(42)
    oracles = ROOT / "oracles"
    oracles.mkdir(exist_ok=True)
    for name in ["fc1-m32", "fc1-m128", "fc2-m32", "fc2-m128"]:
        data = torch.load(
            SOURCE / name / "direct-ab.pt", map_location="cuda", weights_only=True
        )
        a, w = data["A"], data["W"]
        qw, wr = te_quant(w)
        qa, ar = te_quant(a)
        y = te_gemm(qw, qa)
        torch.save(
            {
                "A": a.cpu(),
                "W": w.cpu(),
                "qa": cpuq(ar),
                "qw": cpuq(wr),
                "output1": y.cpu(),
            },
            oracles / (name + ".pt"),
        )
        print("TE REAL", name, flush=True)
    # All 32640 finite positive BF16 bit patterns as block maxima, with signs,
    # signed zeros, small ratios and rounding cases within each 32-value block.
    maxima = (
        torch.arange(0x7F80, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    )
    factors = torch.tensor(
        [
            1,
            -1,
            0.5,
            -0.5,
            0.25,
            -0.25,
            0.125,
            -0.125,
            0.0625,
            -0.0625,
            0.03125,
            -0.03125,
            0.015625,
            -0.015625,
            0.0078125,
            -0.0078125,
            0.00390625,
            -0.00390625,
            0.001953125,
            -0.001953125,
            0.0009765625,
            -0.0009765625,
            0.0,
            -0.0,
            0.96875,
            -0.96875,
            0.53125,
            -0.53125,
            0.28125,
            -0.28125,
            0.09375,
            -0.09375,
        ],
        dtype=torch.float64,
    )
    matrix = (maxima.double()[:, None] * factors[None, :]).bfloat16()
    for name, x in [
        ("bf16-exhaustive-small", matrix),
        ("bf16-exhaustive-large", matrix.repeat(3, 1).reshape(384, 8160)),
        ("experts-3d", torch.randn(4, 128, 128, dtype=torch.bfloat16)),
    ]:
        x = x.cuda()
        _, q = te_quant(x)
        torch.save({"x": x.cpu(), "q": cpuq(q)}, oracles / (name + ".pt"))
        print("TE PATTERN", name, flush=True)


def fi_quant(x: torch.Tensor, swizzled: bool = False) -> dict[str, torch.Tensor]:
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )

    code, scale = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=swizzled)
    m = x.numel() // x.shape[-1]
    k = x.shape[-1]
    code = code.reshape(m, k).view(torch.uint8)
    if swizzled:
        mp = (m + 127) // 128 * 128
        kp = (k // 32 + 3) // 4 * 4
        scale = (
            scale.reshape(mp // 128, kp // 4, 32, 4, 4)
            .transpose(1, 3)
            .contiguous()
            .reshape(mp, kp)[:m, : k // 32]
        )
    else:
        scale = scale.reshape(m, -1)[:, : k // 32]
    return {"data": code, "scale": scale}


def fi_gemm(w: dict[str, torch.Tensor], a: dict[str, torch.Tensor]) -> torch.Tensor:
    import flashinfer

    m, k = a["data"].shape
    n = w["data"].shape[0]
    npad = (n + 127) // 128 * 128
    kpad = (k + 255) // 256 * 256
    wd = F.pad(w["data"], (0, kpad - k, 0, npad - n))
    ad = F.pad(a["data"], (0, kpad - k))
    ws = F.pad(w["scale"], (0, (kpad - k) // 32, 0, npad - n), value=127)
    asc = torch.full(
        ((m + 127) // 128 * 128, kpad // 32), 127, dtype=torch.uint8, device="cuda"
    )
    asc[:m, : k // 32] = a["scale"]
    wd = flashinfer.shuffle_matrix_a(wd, 128).view(torch.float8_e4m3fn)
    ws = flashinfer.shuffle_matrix_sf_a(ws, 128).flatten()
    asc = flashinfer.block_scale_interleave(asc).flatten()
    return flashinfer.mm_mxfp8(
        ad.view(torch.float8_e4m3fn),
        wd.t(),
        asc,
        ws,
        out_dtype=torch.bfloat16,
        backend="trtllm",
    )[:, :n]


def compare(
    q: dict[str, torch.Tensor], ref: dict[str, torch.Tensor]
) -> dict[str, int | str]:
    return {
        "codes_different": int((q["data"] != ref["data"]).sum()),
        "scales_different": int((q["scale"] != ref["scale"]).sum()),
        "codes_sha256": digest(q["data"]),
        "scales_sha256": digest(q["scale"]),
    }


def validate(mode: str) -> None:
    records = []
    for p in sorted((ROOT / "oracles").glob("*.pt")):
        d = torch.load(p, map_location="cuda", weights_only=True)
        inputs = (
            [("A", d["A"], d["qa"]), ("W", d["W"], d["qw"])]
            if "A" in d
            else [("x", d["x"], d["q"])]
        )
        for label, x, ref in inputs:
            for swizzled in [False, True]:
                q = fi_quant(x, swizzled)
                row = {
                    "case": p.stem,
                    "input": label,
                    "swizzled": swizzled,
                    **compare(q, ref),
                }
                records.append(row)
                if mode == "on":
                    assert row["codes_different"] == row["scales_different"] == 0, row
        if "A" in d:
            qa, qw = fi_quant(d["A"]), fi_quant(d["W"])
            y = fi_gemm(qw, qa)
            equal = torch.equal(y, d["output1"])
            diff = (y.float() - d["output1"].float()).abs().max().item()
            torch.save(
                {
                    "A": d["A"].cpu(),
                    "W": d["W"].cpu(),
                    "te_quantized_A": cpuq(d["qa"]),
                    "te_quantized_W": cpuq(d["qw"]),
                    "vllm_quantized_A": cpuq(qa),
                    "vllm_quantized_W": cpuq(qw),
                    "output1": d["output1"].cpu(),
                    "output2": y.cpu(),
                },
                ROOT / f"{mode}-{p.stem}.pt",
            )
            records.append(
                {"case": p.stem, "gemm_bitwise_equal": equal, "gemm_max_abs": diff}
            )
            if mode == "on":
                assert equal, (p.stem, diff)
        print(mode, p.stem, "DONE", flush=True)
    (ROOT / (mode + ".json")).write_text(json.dumps(records, indent=2))


def install() -> None:
    spec = importlib.util.spec_from_file_location("mxfp8_te_compat", PATCHES_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.patch_flashinfer_mxfp8_te_compat(logging.getLogger("test"))
    assert module.patch_flashinfer_mxfp8_te_compat(logging.getLogger("test"))
    print("INSTALLED AND IDEMPOTENCE CHECKED", flush=True)


def finish() -> None:
    baseline = json.loads((ROOT / "baseline.json").read_text())
    for mode in ["off", "off-again"]:
        actual = json.loads((ROOT / (mode + ".json")).read_text())
        assert baseline == actual, (mode, "disabled behavior changed")
    on = json.loads((ROOT / "on.json").read_text())
    assert any(
        r.get("codes_different", 0) > 0 for r in baseline
    ), "negative control did not reproduce mismatch"
    report = {
        "completed": True,
        "env_var": "NRL_VLLM_MXFP8_TE_COMPAT=1",
        "raw_fp8_codes_and_e8m0_bytes_match_te": True,
        "disabled_mode_matches_original": True,
        "warm_cache_off_on_off_checked": True,
        "quantized_comparisons_apply_scales": False,
        "cases": on,
    }
    (ROOT / "result.json").write_text(json.dumps(report, indent=2))
    print("ALL PARITY CHECKS PASSED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=["prepare", "baseline", "install", "off", "on", "off-again", "finish"],
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--patches-file", type=Path)
    args = parser.parse_args()
    ROOT = args.output_dir
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.phase == "prepare":
        if args.input_root is None:
            parser.error("prepare requires --input-root")
        SOURCE = args.input_root
    if args.phase == "install":
        if args.patches_file is None:
            parser.error("install requires --patches-file")
        PATCHES_FILE = args.patches_file
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.phase == "prepare":
        prepare()
    elif args.phase == "install":
        install()
    elif args.phase == "finish":
        finish()
    else:
        validate(args.phase)
