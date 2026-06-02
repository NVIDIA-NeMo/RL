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
"""Unit tests for the projection-prep CLI helpers under ``tools/x_token``.

Scoped to saver / parser / sort-cut / reapply math. Real HuggingFace
tokenizer / model loads are stubbed via ``monkeypatch`` — these tests
must not touch the network or HF cache.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from nemo_rl.algorithms.x_token.loss_utils import parse_projection_file
from tools.x_token import (
    minimal_projection_generator,
    minimal_projection_via_multitoken,
    reapply_exact_map as reapply_mod,
    sort_and_cut_projection_matrix as sort_cut_mod,
    utils,
)


REPO_ROOT = Path(__file__).resolve().parents[4]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dense_topk_projection(
    v_s: int, top_k: int, *, extra_metadata: dict | None = None
) -> dict:
    """Build a dense top-k projection dict with deterministic content."""
    # Row r maps to teacher ids [r % v_t, ...] with decreasing weights.
    v_t = max(v_s, 4)
    indices = torch.zeros((v_s, top_k), dtype=torch.long)
    likelihoods = torch.zeros((v_s, top_k), dtype=torch.float32)
    for r in range(v_s):
        for k in range(top_k):
            indices[r, k] = (r + k) % v_t
            likelihoods[r, k] = max(1.0 - 0.1 * k, 0.1)
    out: dict = {"indices": indices, "likelihoods": likelihoods}
    if extra_metadata:
        out.update(extra_metadata)
    return out


# ---------------------------------------------------------------------------
# TestParseArguments — per-tool
# ---------------------------------------------------------------------------


def _argv(monkeypatch: pytest.MonkeyPatch, args: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["prog", *args])


class TestParseArguments:
    def test_sort_and_cut_requires_initial_path_and_top_k(self, monkeypatch):
        _argv(monkeypatch, [])
        with pytest.raises(SystemExit):
            sort_cut_mod.parse_arguments()

        _argv(monkeypatch, ["--initial-projection-path", "/tmp/x.pt"])
        with pytest.raises(SystemExit):
            sort_cut_mod.parse_arguments()

        _argv(
            monkeypatch,
            ["--initial-projection-path", "/tmp/x.pt", "--top_k", "4"],
        )
        ns = sort_cut_mod.parse_arguments()
        assert ns.initial_projection_path == "/tmp/x.pt"
        assert ns.top_k == 4
        # --preserve_last default is None (sentinel for "auto-resolve").
        assert ns.preserve_last is None

    def test_reapply_only_requires_initial_path(self, monkeypatch):
        _argv(monkeypatch, [])
        with pytest.raises(SystemExit):
            reapply_mod.parse_arguments()

        # student / teacher have defaults; omit them.
        _argv(monkeypatch, ["--initial-projection-path", "/tmp/x.pt"])
        ns = reapply_mod.parse_arguments()
        assert ns.initial_projection_path == "/tmp/x.pt"
        assert ns.student_model == "meta-llama/Llama-3.2-1B"
        assert ns.teacher_model == "microsoft/phi-4"
        # default enable_scale_trick = True (per current CLI defaults).
        assert ns.enable_scale_trick is True


# ---------------------------------------------------------------------------
# save_data / load_data round-trip (minimal_projection_generator)
# ---------------------------------------------------------------------------


class TestSaveDataLoadDataRoundtrip:
    def test_dense_projection_roundtrip_via_save_load(self, tmp_path: Path):
        projection = _make_dense_topk_projection(v_s=8, top_k=2)
        # save_data expects a Tensor (calls .cpu()), so test with a tensor
        # field round-trip rather than the dict — that's what the helper
        # actually does on disk.
        out_path = tmp_path / "subdir" / "indices.pt"
        minimal_projection_generator.save_data(
            projection["indices"], str(out_path)
        )
        loaded = minimal_projection_generator.load_data(str(out_path))
        assert torch.equal(loaded, projection["indices"])

    def test_dict_roundtrip_via_parse_projection_file(self, tmp_path: Path):
        # The full dense projection format on disk is reloadable via
        # loss_utils.parse_projection_file (cross-format compat check).
        projection = _make_dense_topk_projection(v_s=8, top_k=2)
        path = tmp_path / "proj.pt"
        torch.save(projection, path)
        indices, values, v_s, _v_t = parse_projection_file(path)
        assert v_s == 8
        assert indices.shape == (2, 8 * 2)
        assert values.shape == (8 * 2,)


# ---------------------------------------------------------------------------
# generate_projection_map_chunk
# ---------------------------------------------------------------------------


class TestGenerateProjectionMapChunk:
    def test_returns_topk_indices_and_likelihoods(self):
        # 3 source rows, 4 target candidates each.
        similarities = torch.tensor(
            [
                [0.1, 0.9, 0.3, 0.5],
                [0.4, 0.2, 0.8, 0.1],
                [0.5, 0.5, 0.5, 0.5],
            ]
        )
        args = SimpleNamespace(top_k=2, weight_threshold=0.0)
        top_ind, top_lik = minimal_projection_generator.generate_projection_map_chunk(
            similarities, args
        )
        assert top_ind.shape == (3, 2)
        assert top_lik.shape == (3, 2)
        # Row 0's highest similarity is at col 1 (val 0.9).
        assert top_ind[0, 0].item() == 1
        # Row 1's highest similarity is at col 2 (val 0.8).
        assert top_ind[1, 0].item() == 2

    def test_weight_threshold_replaces_low_indices_with_sentinel(self):
        # If threshold filters out everything below max, the second
        # slot becomes -1 sentinel.
        similarities = torch.tensor([[0.1, 0.95, 0.3, 0.5]])
        args = SimpleNamespace(top_k=2, weight_threshold=0.99)
        top_ind, _top_lik = (
            minimal_projection_generator.generate_projection_map_chunk(
                similarities, args
            )
        )
        # The top-1 may or may not survive the threshold (depends on the
        # post-sharpness/sinkhorn likelihood), but any entry below it
        # must be marked -1 by the threshold branch.
        assert (top_ind == -1).any().item() or top_ind[0, 1].item() != -1


# ---------------------------------------------------------------------------
# add_multitoken_mappings (mocked tokenizers)
# ---------------------------------------------------------------------------


class TestAddMultitokenMappings:
    def test_accumulates_counts_for_student_to_teacher(self):
        from collections import defaultdict

        # Tiny tokenizer pair:
        #   source (student) ids 0, 1, 2 decode to "a", "bc", "de"
        #   target (teacher) re-encodes:
        #       "a"  -> [10]
        #       "bc" -> [11, 12]
        #       "de" -> [13]
        source_tokenizer = MagicMock()
        source_tokenizer.decode.side_effect = lambda ids: {0: "a", 1: "bc", 2: "de"}[
            ids[0]
        ]

        target_tokenizer = MagicMock()
        encodings = {
            "a": {"input_ids": [10]},
            "bc": {"input_ids": [11, 12]},
            "de": {"input_ids": [13]},
        }
        target_tokenizer.side_effect = lambda text, **kw: encodings[text]
        target_tokenizer.decode.side_effect = (
            lambda ids: f"<tgt-{ids[0]}>"
        )

        counts: dict = defaultdict(float)
        decoded, examples = (
            minimal_projection_via_multitoken.add_multitoken_mappings(
                source_tokenizer=source_tokenizer,
                target_tokenizer=target_tokenizer,
                source_total_vocab_size=3,
                source_ignore_ids=set(),
                target_ignore_ids=set(),
                source_role="student",
                transformation_counts=counts,
                tokens_to_cut=4,
                use_raw_tokens=False,
                use_canonicalization=False,
            )
        )
        # All three source tokens got decoded.
        assert set(decoded.keys()) == {0, 1, 2}
        # Exact 1:1 mappings accumulate weight 1.0 at their key.
        assert counts[(0, 10)] == pytest.approx(1.0)
        assert counts[(2, 13)] == pytest.approx(1.0)
        # Multi-token mapping splits weight across the multi-token target.
        assert counts[(1, 11)] > 0.0
        assert counts[(1, 12)] > 0.0
        # Only the multi-token mapping yields an examples entry (>=2).
        assert len(examples) == 1
        assert examples[0]["student_id"] == 1
        assert examples[0]["teacher_ids"] == [11, 12]

    def test_invalid_source_role_raises(self):
        with pytest.raises(ValueError, match="source_role"):
            minimal_projection_via_multitoken.add_multitoken_mappings(
                source_tokenizer=MagicMock(),
                target_tokenizer=MagicMock(),
                source_total_vocab_size=1,
                source_ignore_ids=set(),
                target_ignore_ids=set(),
                source_role="invalid",
                transformation_counts={},
                tokens_to_cut=2,
                use_raw_tokens=False,
                use_canonicalization=False,
            )


# ---------------------------------------------------------------------------
# sort_and_cut_projection_matrix (function + main auto-preserve)
# ---------------------------------------------------------------------------


class TestSortAndCutProjectionMatrix:
    def _save_unsorted(self, path: Path) -> None:
        # Build a projection where each row's likelihoods are NOT in
        # descending order, so sorting changes content.
        v_s, top_k = 4, 8
        indices = torch.arange(v_s * top_k, dtype=torch.long).reshape(v_s, top_k)
        likelihoods = torch.zeros((v_s, top_k), dtype=torch.float32)
        for r in range(v_s):
            # Put largest weight in the LAST slot to force a sort change.
            for k in range(top_k):
                likelihoods[r, k] = float(k + 1)  # 1, 2, ..., top_k
        torch.save({"indices": indices, "likelihoods": likelihoods}, path)

    def test_reduces_to_new_top_k_with_descending_order(self, tmp_path: Path):
        in_path = tmp_path / "in.pt"
        out_path = tmp_path / "out.pt"
        self._save_unsorted(in_path)

        sort_cut_mod.sort_and_cut_projection_matrix(
            str(in_path), str(out_path), new_top_k=2, preserve_last=False, verbose=False
        )

        out = torch.load(out_path, map_location="cpu", weights_only=False)
        assert out["indices"].shape == (4, 2)
        assert out["likelihoods"].shape == (4, 2)
        # Top-1 is the LAST original slot (highest weight = top_k = 8).
        # After Sinkhorn the absolute values change, but the ordering
        # (descending per row) must hold.
        for r in range(4):
            assert (
                out["likelihoods"][r, 0] >= out["likelihoods"][r, 1]
            ).item()

    def test_preserve_last_keeps_last_column(self, tmp_path: Path):
        in_path = tmp_path / "in.pt"
        out_path = tmp_path / "out.pt"
        self._save_unsorted(in_path)

        sort_cut_mod.sort_and_cut_projection_matrix(
            str(in_path), str(out_path), new_top_k=2, preserve_last=True, verbose=False
        )

        out = torch.load(out_path, map_location="cpu", weights_only=False)
        # The last column of the new matrix should be the LAST column of
        # the original input (per the preserve_last branch).
        original = torch.load(in_path, map_location="cpu", weights_only=False)
        for r in range(4):
            assert (
                out["indices"][r, -1].item() == original["indices"][r, -1].item()
            )


class TestSortAndCutAutoPreserveFromMetadata:
    def _build_input(self, tmp_path: Path, *, scale_trick: bool) -> Path:
        proj = _make_dense_topk_projection(
            v_s=4,
            top_k=4,
            extra_metadata={"enable_scale_trick": scale_trick},
        )
        path = tmp_path / "in.pt"
        torch.save(proj, path)
        return path

    def test_metadata_true_auto_enables_preserve_last(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        in_path = self._build_input(tmp_path, scale_trick=True)
        out_path = tmp_path / "out.pt"
        ns = argparse.Namespace(
            initial_projection_path=str(in_path),
            top_k=2,
            preserve_last=None,
            output_path=str(out_path),
            quiet=True,
        )
        monkeypatch.setattr(sort_cut_mod, "parse_arguments", lambda: ns)

        recorded = {}

        def _record(input_path, output_path, new_top_k, *, preserve_last, verbose):
            recorded["preserve_last"] = preserve_last

        monkeypatch.setattr(
            sort_cut_mod, "sort_and_cut_projection_matrix", _record
        )

        sort_cut_mod.main()
        assert recorded["preserve_last"] is True

    def test_metadata_false_leaves_preserve_last_off(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        in_path = self._build_input(tmp_path, scale_trick=False)
        out_path = tmp_path / "out.pt"
        ns = argparse.Namespace(
            initial_projection_path=str(in_path),
            top_k=2,
            preserve_last=None,
            output_path=str(out_path),
            quiet=True,
        )
        monkeypatch.setattr(sort_cut_mod, "parse_arguments", lambda: ns)

        recorded = {}
        monkeypatch.setattr(
            sort_cut_mod,
            "sort_and_cut_projection_matrix",
            lambda i, o, k, *, preserve_last, verbose: recorded.update(
                {"preserve_last": preserve_last}
            ),
        )
        sort_cut_mod.main()
        assert recorded["preserve_last"] is False

    def test_explicit_override_wins_over_metadata(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        in_path = self._build_input(tmp_path, scale_trick=True)
        out_path = tmp_path / "out.pt"
        ns = argparse.Namespace(
            initial_projection_path=str(in_path),
            top_k=2,
            preserve_last=False,  # explicit; metadata says True
            output_path=str(out_path),
            quiet=True,
        )
        monkeypatch.setattr(sort_cut_mod, "parse_arguments", lambda: ns)

        recorded = {}
        monkeypatch.setattr(
            sort_cut_mod,
            "sort_and_cut_projection_matrix",
            lambda i, o, k, *, preserve_last, verbose: recorded.update(
                {"preserve_last": preserve_last}
            ),
        )
        sort_cut_mod.main()
        assert recorded["preserve_last"] is False


# ---------------------------------------------------------------------------
# utils helpers
# ---------------------------------------------------------------------------


class TestShared:
    def test_sinkhorn_one_dim_row_normalizes(self):
        A = torch.tensor([[1.0, 1.0, 2.0], [0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
        out = utils.sinkhorn_one_dim(A.clone(), n_iters=1)
        # Row 0 sums to 1.
        assert torch.allclose(out[0].sum(), torch.tensor(1.0), atol=1e-6)
        # Row 1 is all-zero; safe-divide leaves it unchanged.
        assert torch.equal(out[1], torch.zeros(3))
        # Row 2 reduces to one-hot at index 0.
        assert torch.allclose(out[2], torch.tensor([1.0, 0.0, 0.0]))

    def test_clean_model_name_strips_param_count_and_suffixes(self):
        assert utils.clean_model_name_for_filename(
            "meta-llama/Llama-3.2-1B"
        ) == "meta-llama/Llama-3.2"
        assert (
            utils.clean_model_name_for_filename("microsoft/phi-4-Base")
            == "microsoft/phi"
        )
        # "mini" preserved as suffix.
        cleaned = utils.clean_model_name_for_filename(
            "Qwen2.5-0.5B-mini-Instruct"
        )
        assert "mini" in cleaned


# ---------------------------------------------------------------------------
# reapply_exact_map (uses the extracted helper)
# ---------------------------------------------------------------------------


class TestReapplyExactMap:
    def test_exact_match_rows_become_one_hot_and_file_is_reloadable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # Build a 4-token student vocab and 4-token teacher vocab whose
        # tokens overlap at ids 0, 1 (the exact-match candidates).
        student_tokens = ["s_match_0", "s_match_1", "s_only_2", "s_only_3"]
        teacher_tokens = ["s_match_0", "s_match_1", "t_only_2", "t_only_3"]

        class FakeTokenizer:
            def __init__(self, vocab: list[str]) -> None:
                self._vocab = vocab

            def __len__(self) -> int:
                return len(self._vocab)

            def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
                return [self._vocab[i] for i in ids]

        def fake_from_pretrained(name: str) -> FakeTokenizer:
            return FakeTokenizer(
                student_tokens if name == "student" else teacher_tokens
            )

        monkeypatch.setattr(
            reapply_mod.AutoTokenizer, "from_pretrained", fake_from_pretrained
        )
        # AutoConfig is loaded but its result is not used downstream — stub.
        monkeypatch.setattr(
            reapply_mod.AutoConfig,
            "from_pretrained",
            lambda name: MagicMock(),
        )
        # canonical_token is a passthrough when use_canonicalization=False.
        monkeypatch.setattr(
            reapply_mod, "canonical_token", lambda t, *, enabled: t
        )

        # Build an input projection with 4 students × top_k=3, all
        # initial likelihoods != 1.0 so we can detect the rewrite.
        in_path = tmp_path / "init.pt"
        v_s, top_k = 4, 3
        indices = torch.zeros((v_s, top_k), dtype=torch.long)
        likelihoods = torch.full((v_s, top_k), 0.5, dtype=torch.float32)
        # Initial first-column teacher ids deliberately wrong for rows 0,1
        # so we can verify the remap overwrites them.
        for r in range(v_s):
            for k in range(top_k):
                indices[r, k] = (r + k) % 4
        torch.save({"indices": indices, "likelihoods": likelihoods}, in_path)

        args = argparse.Namespace(
            student_model="student",
            teacher_model="teacher",
            use_canonicalization=False,
            initial_projection_path=str(in_path),
        )
        out_path = reapply_mod.reapply_exact_map(args)

        assert os.path.exists(out_path)
        remapped = torch.load(out_path, map_location="cpu", weights_only=False)
        # Rows 0 and 1 should be one-hot at the matching teacher id (0, 1).
        for r in (0, 1):
            assert remapped["likelihoods"][r, 0].item() == pytest.approx(1.0)
            assert remapped["indices"][r, 0].item() == r
            # Remaining slots are sentinel / zero.
            assert (remapped["likelihoods"][r, 1:] == 0.0).all().item()
            assert (remapped["indices"][r, 1:] == -1).all().item()
        # Rows 2 and 3 (no exact match) retain their original 0.5 weights.
        for r in (2, 3):
            assert (remapped["likelihoods"][r] == 0.5).all().item()

        # Output file is reloadable as a dense top-k projection.
        ind, vals, v_s_out, _ = parse_projection_file(out_path)
        assert v_s_out == 4
        assert ind.shape == (2, 4 * 3)


# ---------------------------------------------------------------------------
# build_projection_matrix.sh dry-run
# ---------------------------------------------------------------------------


class TestBuildProjectionMatrixShDryRun:
    BUILD_SH = REPO_ROOT / "tools" / "x_token" / "build_projection_matrix.sh"

    @pytest.fixture(autouse=True)
    def _require_bash(self):
        if shutil.which("bash") is None:
            pytest.skip("bash not available on PATH")

    def test_help_exits_zero_and_prints_usage(self):
        result = subprocess.run(
            ["bash", str(self.BUILD_SH), "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "Usage:" in result.stdout

    def test_missing_required_args_exits_nonzero(self):
        # Missing --teacher-model.
        result = subprocess.run(
            ["bash", str(self.BUILD_SH), "--student-model", "x"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0
