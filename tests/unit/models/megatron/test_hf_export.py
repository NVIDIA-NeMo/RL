# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-only unit tests for the inline Megatron HF export helpers.

These tests must stay importable without the mcore extra: the module under
test imports Megatron only inside its subprocess entry point, and the stub
below keeps it that way.
"""

import os
import sys
from types import ModuleType

import pytest

from nemo_rl.models.megatron import hf_export


# ---------------------------------------------------------------------------
# validate_hf_export_config
# ---------------------------------------------------------------------------


def test_validate_hf_export_config_allows_absent_and_off():
    hf_export.validate_hf_export_config({})
    hf_export.validate_hf_export_config({"checkpoint": {}})
    hf_export.validate_hf_export_config({"checkpoint": {"save_consolidated": "false"}})
    hf_export.validate_hf_export_config({"checkpoint": {"save_consolidated": "final"}})


def test_validate_hf_export_config_rejects_every():
    with pytest.raises(ValueError, match="convert_megatron_to_hf.py"):
        hf_export.validate_hf_export_config(
            {"checkpoint": {"save_consolidated": "every"}}
        )


def test_validate_hf_export_config_rejects_yaml_bool():
    with pytest.raises(ValueError, match="quote the value"):
        hf_export.validate_hf_export_config({"checkpoint": {"save_consolidated": True}})


def test_validate_hf_export_config_rejects_peft_runs():
    cfg = {
        "checkpoint": {"save_consolidated": "final"},
        "peft": {"enabled": True},
    }
    with pytest.raises(ValueError, match="convert_lora_to_hf.py"):
        hf_export.validate_hf_export_config(cfg)


def test_validate_hf_export_config_allows_final_without_peft():
    cfg = {
        "checkpoint": {"save_consolidated": "final"},
        "peft": {"enabled": False},
    }
    hf_export.validate_hf_export_config(cfg)


# ---------------------------------------------------------------------------
# save_tokenizer_sidecar
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    def __init__(self, fail=False):
        self.fail = fail
        self.saved_to = None

    def save_pretrained(self, path):
        if self.fail:
            raise RuntimeError("boom")
        self.saved_to = path
        with open(os.path.join(path, "tokenizer.json"), "w") as f:
            f.write("{}")


def test_save_tokenizer_sidecar_writes_files(tmp_path):
    target = tmp_path / "policy" / "tokenizer"
    tokenizer = _FakeTokenizer()
    hf_export.save_tokenizer_sidecar(tokenizer, str(target))
    assert tokenizer.saved_to == str(target)
    assert (target / "tokenizer.json").exists()


def test_save_tokenizer_sidecar_is_warn_only(tmp_path):
    # A failing tokenizer write must not propagate: checkpoint saving wins.
    hf_export.save_tokenizer_sidecar(_FakeTokenizer(fail=True), str(tmp_path / "tok"))


# ---------------------------------------------------------------------------
# run_hf_export_subprocess
# ---------------------------------------------------------------------------


def test_run_hf_export_subprocess_builds_expected_argv():
    calls = []

    def fake_runner(argv, **kwargs):
        calls.append((argv, kwargs))

        class _R:
            returncode = 0
            stderr = ""

        return _R()

    hf_export.run_hf_export_subprocess(
        "Qwen/Qwen2.5-0.5B",
        "/ckpt/iter_0000002",
        "/ckpt_out/hf_export",
        tokenizer_dir="/ckpt/tokenizer",
        runner=fake_runner,
    )
    argv = calls[0][0]
    assert argv[:3] == [sys.executable, "-m", "nemo_rl.models.megatron.hf_export"]
    assert "--hf-model-name" in argv and "Qwen/Qwen2.5-0.5B" in argv
    assert "--megatron-ckpt-path" in argv and "/ckpt/iter_0000002" in argv
    assert "--tokenizer-dir" in argv and "/ckpt/tokenizer" in argv


def test_run_hf_export_subprocess_raises_with_stderr_tail_on_failure():
    class _R:
        returncode = 1
        stderr = "line1\nboom"

    def fake_runner(argv, **kwargs):
        return _R()

    with pytest.raises(RuntimeError, match="boom"):
        hf_export.run_hf_export_subprocess("m", "/ckpt", "/out", runner=fake_runner)


# ---------------------------------------------------------------------------
# export_final_hf_checkpoint
# ---------------------------------------------------------------------------


def _make_checkpoint_tree(tmp_path):
    weights = tmp_path / "policy" / "weights"
    (weights / "iter_0000001").mkdir(parents=True)
    (weights / "iter_0000002").mkdir(parents=True)
    return weights


def test_export_final_hf_checkpoint_converts_latest_iter_on_rank0(tmp_path):
    weights = _make_checkpoint_tree(tmp_path)
    calls, barriers = [], []

    def fake_runner(**kwargs):
        calls.append(kwargs)

    hf_export.export_final_hf_checkpoint(
        "m",
        str(weights),
        tokenizer_dir=str(tmp_path / "tok"),
        is_rank0=True,
        barrier=lambda: barriers.append(1),
        run_subprocess=fake_runner,
    )
    assert len(calls) == 1
    assert calls[0]["ckpt_path"] == str(weights / "iter_0000002")
    assert calls[0]["output_path"] == str(tmp_path / "policy" / "hf_export")
    assert barriers == [1]


def test_export_final_hf_checkpoint_non_rank0_only_barriers(tmp_path):
    weights = _make_checkpoint_tree(tmp_path)
    calls, barriers = [], []

    def fake_runner(**kwargs):
        calls.append(kwargs)

    hf_export.export_final_hf_checkpoint(
        "m",
        str(weights),
        is_rank0=False,
        barrier=lambda: barriers.append(1),
        run_subprocess=fake_runner,
    )
    assert calls == []
    assert barriers == [1]


def test_export_final_hf_checkpoint_survives_export_failure(tmp_path):
    weights = _make_checkpoint_tree(tmp_path)
    barriers = []

    def failing_runner(**kwargs):
        raise RuntimeError("conversion exploded")

    # Warn-only: the failure must not propagate so training can tear down.
    hf_export.export_final_hf_checkpoint(
        "m",
        str(weights),
        is_rank0=True,
        barrier=lambda: barriers.append(1),
        run_subprocess=failing_runner,
    )
    assert barriers == [1]


def test_export_final_hf_checkpoint_skips_without_iter_dir(tmp_path):
    weights = tmp_path / "policy" / "weights"
    weights.mkdir(parents=True)
    calls = []

    def fake_runner(**kwargs):
        calls.append(kwargs)

    hf_export.export_final_hf_checkpoint(
        "m",
        str(weights),
        is_rank0=True,
        barrier=None,
        run_subprocess=fake_runner,
    )
    assert calls == []


# ---------------------------------------------------------------------------
# main() subprocess entry point
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_community_import(monkeypatch):
    """Stub the Megatron-importing module used by main()."""
    calls = []
    stub = ModuleType("nemo_rl.models.megatron.community_import")

    def fake_export(**kwargs):
        calls.append(kwargs)
        marker = os.path.join(kwargs["output_path"], "model.safetensors")
        with open(marker, "w") as f:
            f.write("stub")

    stub.export_model_from_megatron = fake_export
    monkeypatch.setitem(sys.modules, "nemo_rl.models.megatron.community_import", stub)
    return calls


def test_main_exports_copies_tokenizer_and_renames(tmp_path, stub_community_import):
    ckpt = tmp_path / "iter_0000002"
    ckpt.mkdir()
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "tokenizer_config.json").write_text("{}")
    output = tmp_path / "hf_export"

    hf_export.main(
        [
            "--hf-model-name",
            "m",
            "--megatron-ckpt-path",
            str(ckpt),
            "--hf-ckpt-path",
            str(output),
            "--tokenizer-dir",
            str(tokenizer_dir),
        ]
    )

    assert stub_community_import[0]["input_path"] == str(ckpt)
    assert stub_community_import[0]["overwrite"] is True
    assert (output / "model.safetensors").exists()
    assert (output / "tokenizer" / "tokenizer_config.json").exists()
    # No leftover temporary directories.
    assert not [p for p in tmp_path.iterdir() if ".tmp." in p.name]


def test_main_cleans_up_tmp_on_failure(tmp_path, monkeypatch):
    stub = ModuleType("nemo_rl.models.megatron.community_import")

    def failing_export(**kwargs):
        raise RuntimeError("conversion failed")

    stub.export_model_from_megatron = failing_export
    monkeypatch.setitem(sys.modules, "nemo_rl.models.megatron.community_import", stub)

    output = tmp_path / "hf_export"
    with pytest.raises(RuntimeError, match="conversion failed"):
        hf_export.main(
            [
                "--hf-model-name",
                "m",
                "--megatron-ckpt-path",
                str(tmp_path),
                "--hf-ckpt-path",
                str(output),
            ]
        )
    assert not output.exists()
    assert not [p for p in tmp_path.iterdir() if ".tmp." in p.name]


def test_main_refuses_to_overwrite_promoted_export(tmp_path, stub_community_import):
    output = tmp_path / "hf_export"
    output.mkdir()
    (output / "keep_me").write_text("original")

    with pytest.raises(OSError):
        hf_export.main(
            [
                "--hf-model-name",
                "m",
                "--megatron-ckpt-path",
                str(tmp_path),
                "--hf-ckpt-path",
                str(output),
            ]
        )
    # The promoted export is untouched; the temporary write was cleaned up.
    assert (output / "keep_me").exists()
    assert not [p for p in tmp_path.iterdir() if ".tmp." in p.name]
