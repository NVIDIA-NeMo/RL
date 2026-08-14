from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from nemo_rl.models.policy.workers import mamba_alignment_patches as patches


class FakeMambaMixer:
    def _ssm_prefill(
        self,
        zxBCdt,
        conv_state,
        ssm_state,
        seq_idx=None,
        cu_seqlens=None,
        batch_indices=None,
        intermediate_chunk_indices=None,
        intermediate_abs_positions=None,
        intermediate_real_count=None,
        intermediate_ssm_out=None,
        intermediate_conv_out=None,
        cu_chunk_seqlens=None,
        last_chunk_indices=None,
        seq_idx_for_varlen=None,
        cu_seqlens_list=None,
        real_token_count=None,
        conv_seq_idx=None,
        conv_seq_start=None,
    ):
        return "original-prefill"

    def _ssm_decode(
        self,
        zxBCdt,
        conv_state,
        ssm_state,
        batch_indices=None,
        intermediate_conv_state=None,
        intermediate_ssm_state=None,
    ):
        return "original-decode"


def _reshape_for_test(tensor, pattern, **axes):
    if pattern == "b s (g n) -> b s g n":
        return tensor.reshape(*tensor.shape[:-1], axes["g"], -1)
    if pattern == "b s (h p) -> b s h p":
        return tensor.reshape(*tensor.shape[:-1], -1, axes["p"])
    if pattern == "b s h p -> b s (h p)":
        return tensor.reshape(*tensor.shape[:2], -1)
    if pattern == "(h p) -> h p":
        return tensor.reshape(-1, axes["p"])
    if pattern == "l b d -> b l d":
        return tensor.transpose(0, 1)
    raise AssertionError(f"Unexpected rearrange pattern: {pattern}")


@pytest.fixture
def fake_mamba_module(monkeypatch):
    patches.restore_mamba_alignment_patch()
    module = SimpleNamespace(
        MambaMixer=FakeMambaMixer,
        rearrange=_reshape_for_test,
        mamba_chunk_scan_combined=MagicMock(),
        causal_conv1d_fn=None,
        tensor_masked_update=MagicMock(),
    )
    utils_module = SimpleNamespace(
        get_mamba_version=lambda: (_ for _ in ()).throw(
            AssertionError("original get_mamba_version imported mamba_ssm")
        ),
        _mamba_ssm_version=None,
    )

    def _import_module(name, *args, **kwargs):
        if name == "megatron.core.ssm.mamba_mixer":
            return module
        if name == "megatron.core.utils":
            return utils_module
        raise ImportError(name)

    import_module = MagicMock(side_effect=_import_module)
    monkeypatch.setattr(patches.importlib, "import_module", import_module)
    yield module, import_module
    patches.restore_mamba_alignment_patch()


def test_patch_is_idempotent_and_restorable(fake_mamba_module, capsys):
    module, import_module = fake_mamba_module
    original_prefill = module.MambaMixer._ssm_prefill
    original_decode = module.MambaMixer._ssm_decode

    patches.apply_mamba_alignment_patch()
    patches.apply_mamba_alignment_patch()

    assert module.MambaMixer._ssm_prefill is patches._nrl_patched_ssm_prefill
    assert module.MambaMixer._ssm_decode is patches._nrl_patched_ssm_decode
    assert hasattr(module.MambaMixer, "_bik_decode_buffered_scan")
    import_module.assert_any_call("megatron.core.ssm.mamba_mixer")
    import_module.assert_any_call("megatron.core.utils")
    assert capsys.readouterr().out.count("installed Mamba train/prefill/decode") == 1

    patches.restore_mamba_alignment_patch()

    assert module.MambaMixer._ssm_prefill is original_prefill
    assert module.MambaMixer._ssm_decode is original_decode
    assert not hasattr(module.MambaMixer, "_bik_decode_buffered_scan")


def test_non_bik_calls_delegate_to_original_methods(fake_mamba_module):
    patches.apply_mamba_alignment_patch()
    mixer = FakeMambaMixer()
    mixer.config = SimpleNamespace(batch_invariant_mode=False)

    assert mixer._ssm_prefill(None, None, None) == "original-prefill"
    assert mixer._ssm_decode(None, None, None) == "original-decode"


def test_bik_decode_dispatches_to_reference_helpers(fake_mamba_module):
    patches.apply_mamba_alignment_patch()
    mixer = FakeMambaMixer()
    mixer.config = SimpleNamespace(batch_invariant_mode=True)
    mixer.d_inner_local_tp = 2
    mixer.ngroups_local_tp = 1
    mixer.d_state = 1
    mixer.nheads_local_tp = 1
    mixer.rmsnorm = False
    mixer._bik_decode_conv_reference = MagicMock(return_value=torch.zeros(2, 1, 4))
    mixer._bik_decode_buffered_scan = MagicMock(return_value=torch.ones(2, 1, 2))
    batch_indices = torch.tensor([0, -1])

    output = mixer._ssm_decode(
        torch.zeros(2, 1, 7),
        torch.zeros(1),
        torch.zeros(1),
        batch_indices=batch_indices,
    )

    torch.testing.assert_close(output, torch.ones(2, 1, 2))
    mixer._bik_decode_conv_reference.assert_called_once()
    mixer._bik_decode_buffered_scan.assert_called_once()


def test_buffered_scan_zeroes_inactive_slots(fake_mamba_module):
    module, _ = fake_mamba_module
    patches.apply_mamba_alignment_patch()
    mixer = FakeMambaMixer()
    mixer.chunk_size = 4
    mixer.ngroups_local_tp = 1
    mixer.headdim = 2
    mixer.D_has_hdim = False
    mixer.rmsnorm = True
    mixer.cp = SimpleNamespace(
        get_A_log=lambda: torch.zeros(1),
        get_D=lambda: torch.ones(1),
        get_dt_bias=lambda: torch.zeros(1),
    )
    module.mamba_chunk_scan_combined.reset_mock()

    output = mixer._bik_decode_buffered_scan(
        torch.ones(2, 1, 2),
        torch.ones(2, 1, 1),
        torch.ones(2, 1, 1),
        torch.ones(2, 1, 1),
        None,
        torch.tensor([-1, -1]),
        torch.zeros(3, 1, 2, 1),
    )

    torch.testing.assert_close(output, torch.zeros(2, 1, 2))
    module.mamba_chunk_scan_combined.assert_not_called()


def test_bik_decode_rejects_speculative_rollback_buffers(fake_mamba_module):
    patches.apply_mamba_alignment_patch()
    mixer = FakeMambaMixer()
    mixer.config = SimpleNamespace(batch_invariant_mode=True)

    with pytest.raises(NotImplementedError, match="speculative-decoding"):
        mixer._ssm_decode(
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            intermediate_conv_state=torch.empty(0),
        )


class FakeMambaMixerOldApi(FakeMambaMixer):
    def _ssm_prefill(
        self,
        zxBCdt,
        conv_state,
        ssm_state,
        seq_idx=None,
        cu_seqlens=None,
        batch_indices=None,
        intermediate_chunk_indices=None,
        intermediate_abs_positions=None,
        intermediate_ssm_out=None,
        intermediate_conv_out=None,
        conv_gather_offsets=None,
        cu_chunk_seqlens=None,
        last_chunk_indices=None,
        seq_idx_for_varlen=None,
        cu_seqlens_list=None,
        real_token_count=None,
        conv_seq_idx=None,
        conv_seq_start=None,
    ):
        return "old-api-prefill"


def test_policy_uses_mamba_layers():
    assert patches.policy_uses_mamba_layers(
        {"model_name": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"}
    )
    assert not patches.policy_uses_mamba_layers({"model_name": "Qwen/Qwen3-30B-A3B"})
    assert patches.policy_uses_mamba_layers(
        {
            "model_name": "some/custom-model",
            "megatron_cfg": {"hybrid_layer_pattern": "MEME*ME"},
        }
    )


def test_patch_skips_when_not_required_and_signature_mismatch(
    fake_mamba_module, monkeypatch, capsys
):
    module, import_module = fake_mamba_module
    module.MambaMixer = FakeMambaMixerOldApi
    original_prefill = module.MambaMixer._ssm_prefill

    patches.apply_mamba_alignment_patch(required=False)

    assert module.MambaMixer._ssm_prefill is original_prefill
    assert "skipping Mamba alignment patch" in capsys.readouterr().out
    import_module.assert_any_call("megatron.core.ssm.mamba_mixer")


def test_reference_prefill_rejects_prefix_cache(fake_mamba_module):
    patches.apply_mamba_alignment_patch()
    mixer = FakeMambaMixer()

    with pytest.raises(NotImplementedError, match="prefix caching"):
        mixer._ssm_prefill_reference(
            z=torch.empty(0),
            xBC=torch.empty(0),
            dt=torch.empty(0),
            A=torch.empty(0),
            cu_seqlens=torch.empty(0),
            cu_seqlens_list=None,
            batch_indices=None,
            conv_state=None,
            ssm_state=None,
            intermediate_ssm_out=torch.empty(0),
        )


def test_get_mamba_version_uses_package_metadata(fake_mamba_module, monkeypatch):
    monkeypatch.setattr(
        "importlib.metadata.version",
        lambda name: "2.2.6.post3" if name == "mamba_ssm" else (_ for _ in ()).throw(AssertionError(name)),
    )
    patches.apply_mamba_alignment_patch()

    ver = patches._nrl_get_mamba_version()
    assert str(ver) == "2.2.6.post3"
    assert patches._MAMBA_UTILS_MODULE.get_mamba_version is patches._nrl_get_mamba_version
