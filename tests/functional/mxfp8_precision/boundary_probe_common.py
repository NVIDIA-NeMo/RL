"""Fail-closed checks shared by CPU regressions and real worker observers."""


def validate_named_spans(spans, buffer_numel):
    """Allow padding gaps, but reject overlapping, truncated or out-of-range views."""
    names = set()
    previous_end = 0
    for item in sorted(spans, key=lambda v: v["start"]):
        assert item["name"] not in names, "Duplicate parameter span"
        names.add(item["name"])
        start, end = item["start"], item["end"]
        assert 0 <= start <= end <= buffer_numel, "Parameter span outside buffer"
        assert end - start == item["numel"], "Parameter span has wrong length"
        assert start >= previous_end, "Parameter spans overlap"
        previous_end = end
    assert names, "No parameter spans observed"


def scaled_fp32_reference_input(value, scaling_factor):
    # Match production's in-place scaling in the communication dtype before
    # promoting the already-rounded values for the reference collective.
    return value.detach().clone().mul_(scaling_factor).float()


def compare_master_snapshots(actual, expected):
    """Return the first missing/different semantic master, including rank coverage."""
    if not actual or not expected:
        return {"equal": False, "reason": "missing_rank_coverage"}
    a, b = ({r["rank"]: r["masters"] for r in rows} for rows in [actual, expected])
    if len(a) != len(actual) or len(b) != len(expected) or a.keys() != b.keys():
        return {"equal": False, "reason": "rank_coverage"}
    for rank in sorted(a):
        if not a[rank] or not b[rank]:
            return {"equal": False, "reason": "missing_parameter_coverage", "rank": rank}
        for name in sorted(a[rank].keys() | b[rank].keys()):
            if a[rank].get(name) != b[rank].get(name):
                return {"equal": False, "rank": rank, "parameter": name,
                        "actual": a[rank].get(name), "expected": b[rank].get(name)}
    return {"equal": True, "ranks": len(a)}


def effective_mxfp8_fingerprint(weight, scales):
    """Compare represented values, not non-unique FP8-code/E8M0-scale pairs."""
    import torch
    from precision_probe_common import tensor_hash

    values = weight.view(torch.float8_e4m3fn) if weight.dtype == torch.uint8 else weight
    rows, cols = weight.numel() // weight.shape[-1], weight.shape[-1]
    assert cols % 32 == 0 and scales.numel() == rows * (cols // 32)
    scale_bytes = scales.view(torch.uint8).reshape(rows, cols // 32)
    decoded = torch.ldexp(values.float().reshape(rows, cols // 32, 32),
                         (scale_bytes.int() - 127).unsqueeze(-1)).reshape(weight.shape)
    assert torch.isfinite(decoded).all(), "Nonfinite effective MXFP8 weights"
    return {"shape": list(weight.shape), "sha256": tensor_hash(decoded)}
