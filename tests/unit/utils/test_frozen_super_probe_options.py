"""Reject overstated evidence before allocating native model workers."""

import pytest

from data.agentic_vbench.frozen_super_probe_options import (
    FrozenProbeOptions,
    validate_captured_manifest,
)


def _manifest() -> dict:
    return {
        "generation_weight_version": 15,
        "model_export_training_step": 15,
        "native_cumulative_token_hashes_verified": True,
        "native_extras_digests_verified": True,
    }


def test_version_only_does_not_require_or_invent_refit_audit() -> None:
    manifest = _manifest()
    options = FrozenProbeOptions(baseline_source="captured_generation_version_only")
    validate_captured_manifest(manifest, options)
    assert "refit_timing_audit" not in manifest


def test_audited_capture_still_requires_refit_evidence() -> None:
    manifest = _manifest()
    options = FrozenProbeOptions(baseline_source="captured_generation")
    with pytest.raises(KeyError, match="refit_timing_audit"):
        validate_captured_manifest(manifest, options)
    manifest["refit_timing_audit"] = {
        "all_generation_leaders_bracketed_without_refit": False
    }
    with pytest.raises(ValueError, match="refit-free"):
        validate_captured_manifest(manifest, options)
    manifest["refit_timing_audit"]["all_generation_leaders_bracketed_without_refit"] = (
        True
    )
    validate_captured_manifest(manifest, options)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_export_training_step", 13),
        ("native_cumulative_token_hashes_verified", False),
        ("native_extras_digests_verified", False),
    ],
)
def test_version_only_still_rejects_wrong_weights_or_unverified_routes(
    field: str, value: int | bool
) -> None:
    manifest = _manifest() | {field: value}
    options = FrozenProbeOptions(baseline_source="captured_generation_version_only")
    with pytest.raises(ValueError):
        validate_captured_manifest(manifest, options)


@pytest.mark.parametrize(
    "source", ["captured_generation", "captured_generation_version_only"]
)
@pytest.mark.parametrize(
    "override", [{"call_view_routes_replayed": False}, {"fixed_expert_routes": True}]
)
def test_captured_generation_cannot_substitute_policy_routes(
    source: str, override: dict
) -> None:
    with pytest.raises(ValueError, match="must replay saved routes"):
        FrozenProbeOptions.model_validate({"baseline_source": source, **override})
