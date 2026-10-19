"""Central options for the frozen Super diagnostic JSON manifests."""

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

SSMMode = Literal["native", "unfused_bf16", "unfused_fp32_scan"]


class NativeControlReference(BaseModel, extra="forbid"):
    """Bind the full-context control to an earlier actual native forward."""

    path: Path
    logprobs_path: Path
    sha256: str = Field(pattern="^[0-9a-f]{64}$")
    p99_tolerance: float = Field(default=0.001, ge=0)
    max_tolerance: float = Field(default=0.01, ge=0)


class FrozenProbeOptions(BaseModel, extra="allow"):
    """Optional diagnostic controls; training configuration is separate.

    ``generation_counterfactual`` permits mixed generation versions and fresh
    policy routing. It requires an earlier full-context native control, and
    never claims original generation-route or saved training-LP reproduction.
    ``captured_generation_version_only`` retains authenticated original routes
    and an exact version match, without claiming an unavailable refit audit.
    """

    baseline_source: Literal[
        "saved_policy",
        "captured_generation",
        "captured_generation_version_only",
        "generation_counterfactual",
    ] = "saved_policy"
    fixed_expert_routes: bool = False
    ssm_modes: list[SSMMode] = Field(
        default_factory=lambda: ["native", "unfused_bf16", "unfused_fp32_scan"]
    )
    native_control_reference: NativeControlReference | None = None
    call_view_control_path: Path | None = None
    call_view_routes_replayed: bool = True

    @model_validator(mode="after")
    def validate_comparison(self) -> Self:
        if (
            not self.ssm_modes
            or self.ssm_modes[0] != "native"
            or len(set(self.ssm_modes)) != len(self.ssm_modes)
        ):
            raise ValueError("ssm_modes must be unique and begin with native")
        if self.baseline_source == "generation_counterfactual":
            if self.native_control_reference is None:
                raise ValueError(
                    "counterfactual requires a full-context native control"
                )
            if self.call_view_routes_replayed:
                raise ValueError(
                    "fresh-routing counterfactual cannot claim generation replay"
                )
            if self.fixed_expert_routes:
                raise ValueError(
                    "fresh-routing counterfactual cannot fix native routes"
                )
        if self.baseline_source in {
            "captured_generation",
            "captured_generation_version_only",
        } and (self.fixed_expert_routes or not self.call_view_routes_replayed):
            raise ValueError("Captured-generation comparisons must replay saved routes")
        return self


def validate_captured_manifest(manifest: dict, options: FrozenProbeOptions) -> None:
    """Keep version-only evidence separate from audited refit-free generation."""
    if options.baseline_source not in {
        "captured_generation",
        "captured_generation_version_only",
    }:
        return
    if manifest["generation_weight_version"] != manifest["model_export_training_step"]:
        raise ValueError("Captured generation and checkpoint versions differ")
    if not (
        manifest["native_cumulative_token_hashes_verified"]
        and manifest["native_extras_digests_verified"]
    ):
        raise ValueError(
            "Captured generation requires verified native tokens and routes"
        )
    if (
        options.baseline_source == "captured_generation"
        and not manifest["refit_timing_audit"][
            "all_generation_leaders_bracketed_without_refit"
        ]
    ):
        raise ValueError("Audited captured generation requires a refit-free interval")
