"""Small GPU reproduction using the actual container's TE MXFP8 quantizer."""

import argparse
import inspect
import sys
from pathlib import Path

import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex

from precision_probe_common import (
    mxfp8_storage_snapshot,
    save_json,
    semantic_model_state,
    tensor_hash,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(
        {
            "python": sys.executable,
            "prefix": sys.prefix,
            "te_quantizer_source": inspect.getfile(MXFP8Quantizer),
        },
        flush=True,
    )
    torch.manual_seed(42)
    reports = []
    for m, k in [(64, 96), (96, 64), (128, 192)]:
        q = MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)
        q.optimize_for_gemm = False
        source = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        live = q.quantize(source)
        q.update_quantized(source * 2, live)
        fresh = q.quantize(source * 2)
        torch.cuda.synchronize()
        a, b = live.get_metadata(), fresh.get_metadata()
        assert tuple(a["columnwise_scale_inv"].shape) == tuple(
            q.get_scale_shape((m, k), True)
        )
        sa, sb = mxfp8_storage_snapshot((m, k), a), mxfp8_storage_snapshot((m, k), b)
        assert semantic_model_state(sa) == semantic_model_state(sb), (
            "In-place TE refresh differs from fresh quantization"
        )
        # Only cloned metadata is modified; the quantized weights stay untouched.
        ca = {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in a.items()
        }
        cb = {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in b.items()
        }
        for metadata, fill in [(ca, 17), (cb, 29)]:
            scale = metadata["columnwise_scale_inv"]
            scale[m // 32 :, :] = fill
            scale[: m // 32, k:] = fill
        assert semantic_model_state(
            mxfp8_storage_snapshot((m, k), ca)
        ) == semantic_model_state(mxfp8_storage_snapshot((m, k), cb))
        wrong_crop_differs = tensor_hash(
            ca["columnwise_scale_inv"][:k, : m // 32]
        ) != tensor_hash(cb["columnwise_scale_inv"][:k, : m // 32])
        if (m // 32) % 4:
            assert wrong_crop_differs, (
                "Did not reproduce the old crop's padding false failure"
            )
        cb["columnwise_scale_inv"][m // 32 - 1, k - 1] ^= 1
        assert semantic_model_state(
            mxfp8_storage_snapshot((m, k), ca)
        ) != semantic_model_state(mxfp8_storage_snapshot((m, k), cb)), (
            "Used scale corruption was ignored"
        )
        reports.append(
            {
                "shape": [m, k],
                "column_scale_allocation": list(a["columnwise_scale_inv"].shape),
                "column_scale_logical": [m // 32, k],
                "real_te_refresh_pass": True,
                "padding_ignored": True,
                "old_crop_false_failure_reproduced": wrong_crop_differs,
                "active_scale_corruption_detected": True,
            }
        )
    result = {
        "status": "PASS",
        "torch": torch.__version__,
        "te_source": inspect.getfile(MXFP8Quantizer),
        "cases": reports,
    }
    save_json(Path(args.output), result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
