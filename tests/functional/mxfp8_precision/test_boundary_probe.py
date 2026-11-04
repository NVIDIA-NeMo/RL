"""Regression checks for observers that must not mask staging/resume failures."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from boundary_probe_common import (
    compare_master_snapshots,
    scaled_fp32_reference_input,
    validate_named_spans,
    effective_mxfp8_fingerprint,
)
from boundary_refit_compare import compare_refit_boundaries


def span(name, start, end, numel=None):
    return {"name": name, "start": start, "end": end,
            "numel": end - start if numel is None else numel}


def test_buffer_padding_gaps_are_valid():
    validate_named_spans([span("b", 8, 12), span("a", 0, 4)], 16)


@pytest.mark.parametrize("spans", [
    [], [span("a", -1, 3)], [span("a", 0, 17)],
    [span("a", 0, 4, 3)], [span("a", 0, 8), span("b", 4, 12)],
    [span("a", 0, 4), span("a", 8, 12)],
])
def test_invalid_staging_offsets_fail_closed(spans):
    with pytest.raises(AssertionError):
        validate_named_spans(spans, 16)


def test_fp32_reference_preserves_original_dtype_scaling_rounding():
    source = torch.tensor([1., 1.5], dtype=torch.bfloat16)
    before = source.clone()
    actual = scaled_fp32_reference_input(source, 1 / 3)
    assert torch.equal(source, before)
    assert torch.equal(actual, (source * (1 / 3)).float())
    assert not torch.equal(actual, source.float() * (1 / 3))


def test_master_comparison_cannot_pass_missing_or_duplicate_coverage():
    expected = [{"rank": 0, "masters": {"expert": "same", "norm": "same"}}]
    assert compare_master_snapshots(expected, expected)["equal"]
    for actual in [[], expected * 2, [{"rank": 1, "masters": expected[0]["masters"]}],
                   [{"rank": 0, "masters": {"expert": "same"}}],
                   [{"rank": 0, "masters": {"expert": "rounded", "norm": "same"}}]]:
        assert not compare_master_snapshots(actual, expected)["equal"]


def test_real_finalizer_boundary_stops_before_optimizer_update(tmp_path):
    # Extract only the observer's method: local CPU checks do not import Ray/MCore.
    source = ast.parse(Path(__file__).with_name("boundary_probe_worker.py").read_text())
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "BoundaryProbeImpl")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_save_gradients")
    events = []
    class Parent:
        def _save_gradients(self, folder):
            events.append("raw")
    exception = type("StopBeforeOptimizerUpdate", (Exception,), {})
    tree = ast.Module(body=[ast.ClassDef(name="Observer", bases=[ast.Name(id="Parent", ctx=ast.Load())],
        keywords=[], body=[method], decorator_list=[])], type_ignores=[])
    namespace = {"Parent": Parent, "Path": Path, "StopBeforeOptimizerUpdate": exception}
    exec(compile(ast.fix_missing_locations(tree), "observer", "exec"), namespace)
    observer = namespace["Observer"]()
    original = lambda: pytest.fail("Optimizer update must never run")
    observer.optimizer = SimpleNamespace(step=original)
    observer._save_reduced_gradients = lambda folder: events.append("reduced")
    def finish():
        events.append("production_finalizer")
        observer.optimizer.step()
        pytest.fail("Must stop immediately before the update")
    observer.finish_train_step = finish
    observer._save_gradients(tmp_path / "gradients")
    assert events == ["raw", "production_finalizer", "reduced"]
    assert observer.optimizer.step is original


def test_refit_comparison_requires_all_sampler_ranks_and_matching_context(tmp_path):
    for mode in ["a", "b"]:
        for stage in ["sampler_logical_mxfp8", "sampler_execution_mxfp8", "sampler_logits"]:
            for rank in range(4):
                folder = tmp_path / mode / f"{stage}-rank-{rank}-pid-{100 if mode == 'a' else 200}"
                folder.mkdir(parents=True)
                record = {"layer": "expert", "weights": {"weight": "same"}}
                if stage == "sampler_logical_mxfp8":
                    record["effective_weights"] = {"w13_weight": "same", "w2_weight": "same"}
                if stage == "sampler_logits":
                    record = {"forward_context_complete": True, "forward_context": {"tokens": "same"},
                              "finite": True, "sha256": "same"}
                (folder / "00000.json").write_text(json.dumps(record))
    result = compare_refit_boundaries(tmp_path, "a", "b")
    assert all(v["equal"] for v in result.values())
    path = next((tmp_path / "a").glob("sampler_logits-rank-0-*")) / "00000.json"
    record = json.loads(path.read_text());record["forward_context"] = {"tokens": "different"}
    path.write_text(json.dumps(record))
    assert compare_refit_boundaries(tmp_path, "a", "b")["sampler_logits"]["first_difference"]["reason"] == "different_token_context"
    path.unlink()
    assert not compare_refit_boundaries(tmp_path, "a", "b")["sampler_logits"]["coverage"]


def test_nonunique_mxfp8_encodings_have_equal_effective_fingerprint():
    first = torch.ones(1, 32).to(torch.float8_e4m3fn)
    second = torch.full((1, 32), 2.).to(torch.float8_e4m3fn)
    a = effective_mxfp8_fingerprint(first, torch.tensor([[127]], dtype=torch.uint8))
    b = effective_mxfp8_fingerprint(second, torch.tensor([[126]], dtype=torch.uint8))
    assert a == b
    second = torch.full((1, 32), 2.25).to(torch.float8_e4m3fn)
    assert a != effective_mxfp8_fingerprint(second, torch.tensor([[126]], dtype=torch.uint8))
