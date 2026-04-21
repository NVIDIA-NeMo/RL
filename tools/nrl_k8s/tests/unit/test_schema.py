"""Tests for :mod:`nrl_k8s.schema`.

The schema is the contract between recipes and every downstream template, so
these tests pin the exact validation rules. Each cluster-identifying field
must be present; every ``kind=...`` sentinel with a required companion
(``queue``, ``pvcName``, ``hostPath``) must refuse a config that omits the
companion; and ``extra='forbid'`` must reject typos like ``queue_name``.
"""

from __future__ import annotations

import pytest
from nrl_k8s.schema import (
    CheckpointsKind,
    CheckpointsSpec,
    CodeSource,
    HFCacheKind,
    HFCacheSpec,
    InfraConfig,
    LaunchMode,
    LaunchSpec,
    RunMode,
    SchedulerKind,
    SchedulerSpec,
    SubmitSpec,
    SubmitterMode,
    WorkspaceKind,
    WorkspaceSpec,
)
from pydantic import ValidationError

# =============================================================================
# Top-level InfraConfig
# =============================================================================


def _min_infra() -> dict:
    """The smallest valid InfraConfig payload (required fields only)."""
    return {"namespace": "nemo-rl", "image": "nvcr.io/nvidia/nemo-rl:test"}


class TestInfraConfigRequiredFields:
    def test_minimal_config_validates(self) -> None:
        cfg = InfraConfig.model_validate(_min_infra())
        assert cfg.namespace == "nemo-rl"
        assert cfg.image == "nvcr.io/nvidia/nemo-rl:test"
        assert cfg.scheduler.kind is SchedulerKind.DEFAULT
        assert cfg.workspace.kind is WorkspaceKind.RAY_UPLOAD

    def test_missing_namespace_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            InfraConfig.model_validate({"image": "foo:bar"})

    def test_missing_image_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            InfraConfig.model_validate({"namespace": "ns"})

    def test_blank_namespace_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            InfraConfig.model_validate({"namespace": "   ", "image": "foo:bar"})


class TestInfraConfigStrictness:
    def test_unknown_top_level_key_rejected(self) -> None:
        payload = _min_infra() | {"totally_fake_key": True}
        with pytest.raises(ValidationError):
            InfraConfig.model_validate(payload)

    def test_unknown_nested_key_rejected(self) -> None:
        payload = _min_infra() | {"scheduler": {"kind": "default", "typo": "x"}}
        with pytest.raises(ValidationError):
            InfraConfig.model_validate(payload)


# =============================================================================
# Scheduler
# =============================================================================


class TestSchedulerSpec:
    def test_default_requires_no_queue(self) -> None:
        spec = SchedulerSpec()
        assert spec.kind is SchedulerKind.DEFAULT
        assert spec.queue is None

    def test_kai_without_queue_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SchedulerSpec.model_validate({"kind": "kai"})

    def test_kueue_without_queue_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SchedulerSpec.model_validate({"kind": "kueue"})

    def test_kai_with_queue_accepted(self) -> None:
        spec = SchedulerSpec.model_validate({"kind": "kai", "queue": "team-a"})
        assert spec.queue == "team-a"

    def test_invalid_kind_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SchedulerSpec.model_validate({"kind": "slurm"})


# =============================================================================
# Workspace
# =============================================================================


class TestWorkspaceSpec:
    def test_default_is_ray_upload(self) -> None:
        spec = WorkspaceSpec()
        assert spec.kind is WorkspaceKind.RAY_UPLOAD

    @pytest.mark.parametrize("kind", ["lustre", "pvc"])
    def test_pvc_kinds_require_pvc_name(self, kind: str) -> None:
        with pytest.raises(ValidationError):
            WorkspaceSpec.model_validate({"kind": kind})

    def test_lustre_with_pvc_name_accepted(self) -> None:
        spec = WorkspaceSpec.model_validate(
            {"kind": "lustre", "pvcName": "nemo-rl-lustre", "size": "1200Gi"}
        )
        assert spec.pvcName == "nemo-rl-lustre"
        assert spec.mountPath == "/mnt/nemo-rl"  # default preserved

    def test_host_path_without_host_path_rejected(self) -> None:
        with pytest.raises(ValidationError):
            WorkspaceSpec.model_validate({"kind": "hostPath"})

    def test_host_path_with_host_path_accepted(self) -> None:
        spec = WorkspaceSpec.model_validate({"kind": "hostPath", "hostPath": "/data"})
        assert spec.hostPath == "/data"


# =============================================================================
# HFCache / Checkpoints
# =============================================================================


class TestHFCacheSpec:
    def test_default_is_none(self) -> None:
        assert HFCacheSpec().kind is HFCacheKind.NONE

    @pytest.mark.parametrize("kind", ["lustre", "pvc"])
    def test_pvc_kinds_require_name(self, kind: str) -> None:
        with pytest.raises(ValidationError):
            HFCacheSpec.model_validate({"kind": kind})


class TestCheckpointsSpec:
    def test_default_is_none(self) -> None:
        assert CheckpointsSpec().kind is CheckpointsKind.NONE

    def test_lustre_without_pvc_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CheckpointsSpec.model_validate({"kind": "lustre"})


# =============================================================================
# Launch
# =============================================================================


class TestLaunchSpec:
    def test_default_is_single(self) -> None:
        spec = LaunchSpec()
        assert spec.mode is LaunchMode.SINGLE
        assert spec.peerWatcher is True

    def test_attach_requires_a_target(self) -> None:
        with pytest.raises(ValidationError):
            LaunchSpec.model_validate({"mode": "attach", "attach": {}})

    def test_attach_with_generation_only_ok(self) -> None:
        spec = LaunchSpec.model_validate(
            {"mode": "attach", "attach": {"generation": "rc-gen"}}
        )
        assert spec.attach.generation == "rc-gen"
        assert spec.attach.training is None

    def test_run_mode_defaults_to_interactive(self) -> None:
        assert LaunchSpec().runMode is RunMode.INTERACTIVE

    def test_code_source_defaults_to_upload(self) -> None:
        assert LaunchSpec().codeSource is CodeSource.UPLOAD

    def test_code_path_required_for_image(self) -> None:
        with pytest.raises(ValidationError, match="codePath is required"):
            LaunchSpec.model_validate({"codeSource": "image"})

    def test_code_path_required_for_lustre(self) -> None:
        with pytest.raises(ValidationError, match="codePath is required"):
            LaunchSpec.model_validate({"codeSource": "lustre"})

    def test_code_path_ok_with_image(self) -> None:
        spec = LaunchSpec.model_validate(
            {"codeSource": "image", "codePath": "/opt/nemo-rl"}
        )
        assert spec.codeSource is CodeSource.IMAGE
        assert spec.codePath == "/opt/nemo-rl"

    def test_code_path_not_required_for_upload(self) -> None:
        spec = LaunchSpec.model_validate({"codeSource": "upload"})
        assert spec.codePath is None


class TestSubmitterMode:
    def test_default_is_port_forward(self) -> None:
        assert SubmitSpec().submitter is SubmitterMode.PORT_FORWARD

    def test_exec_tmp_dir_default(self) -> None:
        assert SubmitSpec().execTmpDir == "/tmp"

    def test_exec_tmp_dir_override(self) -> None:
        spec = SubmitSpec.model_validate({"submitter": "exec", "execTmpDir": "/workspace/tmp"})
        assert spec.submitter is SubmitterMode.EXEC
        assert spec.execTmpDir == "/workspace/tmp"


# =============================================================================
# Placement / tolerations
# =============================================================================


class TestTolerations:
    def test_simple_toleration_accepted(self) -> None:
        cfg = InfraConfig.model_validate(
            _min_infra()
            | {
                "placement": {
                    "nodeSelector": {"gpu": "h100"},
                    "tolerations": [
                        {
                            "key": "team",
                            "operator": "Equal",
                            "value": "rl",
                            "effect": "NoSchedule",
                        }
                    ],
                }
            }
        )
        assert cfg.placement.nodeSelector == {"gpu": "h100"}
        assert len(cfg.placement.tolerations) == 1
        assert cfg.placement.tolerations[0].key == "team"

    def test_invalid_effect_rejected(self) -> None:
        with pytest.raises(ValidationError):
            InfraConfig.model_validate(
                _min_infra()
                | {
                    "placement": {
                        "tolerations": [
                            {"key": "k", "effect": "DoesntExist"},
                        ]
                    }
                }
            )
