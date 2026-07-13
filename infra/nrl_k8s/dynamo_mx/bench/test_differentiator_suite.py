from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


MODULE_PATH = Path(__file__).with_name("differentiator_suite.py")
SPEC = importlib.util.spec_from_file_location("differentiator_suite", MODULE_PATH)
assert SPEC and SPEC.loader
suite = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(suite)


def artifact(**overrides):
    value = {
        "model": suite.MODEL_ID,
        "checkpoint": "0123456789abcdef",
        "checkpoint_bytes": suite.CHECKPOINT_BYTES,
        "tensor_source": "received_safetensors",
        "bytes": suite.CHECKPOINT_BYTES,
    }
    value.update(overrides)
    return value


def test_rejects_qwen3_4b_artifact() -> None:
    with pytest.raises(ValueError, match="expected model"):
        suite._artifact_metadata(
            artifact(model="Qwen/Qwen3-4B-Thinking-2507"),
            "4b.json",
        )


@pytest.mark.parametrize("tensor_source", ["synthetic", "random", "shape_only"])
def test_rejects_synthetic_tensor_sources(tensor_source: str) -> None:
    with pytest.raises(ValueError, match="synthetic tensors"):
        suite._artifact_metadata(
            artifact(tensor_source=tensor_source),
            "synthetic.json",
        )


def test_rejects_wrong_checkpoint_size() -> None:
    with pytest.raises(ValueError, match=str(suite.CHECKPOINT_BYTES)):
        suite._artifact_metadata(
            artifact(checkpoint_bytes=8_000_000_000),
            "wrong-size.json",
        )


def test_elastic_rejects_4b_receiver_results(tmp_path: Path) -> None:
    row = artifact(
        model="Qwen/Qwen3-4B-Thinking-2507",
        delay_s=0,
        pull_start_epoch=1.0,
        pull_end_epoch=2.0,
        pull_dur_s=1.0,
        gbps=1.0,
    )
    (tmp_path / "result_0.json").write_text(json.dumps(row))

    with pytest.raises(ValueError, match="expected model"):
        suite.elastic(SimpleNamespace(results=str(tmp_path)))


def test_fanout_rejects_synthetic_summary(tmp_path: Path) -> None:
    direct = artifact(
        tensor_source="synthetic",
        workers=13,
        source_count=1,
        makespan_seconds=2.0,
    )
    tree = artifact(
        workers=13,
        source_count=4,
        makespan_seconds=1.0,
    )
    direct_path = tmp_path / "direct.json"
    tree_path = tmp_path / "tree.json"
    direct_path.write_text(json.dumps(direct))
    tree_path.write_text(json.dumps(tree))

    with pytest.raises(ValueError, match="synthetic tensors"):
        suite.fanout(
            SimpleNamespace(
                direct=str(direct_path),
                tree=str(tree_path),
                workers=13,
                min_speedup=1.05,
            )
        )


def test_canonical_runner_has_no_numeric_only_fallbacks() -> None:
    runner = Path(__file__).with_name("run_differentiator_bench.sh").read_text()
    assert "--actual-bytes" not in runner
    assert 'FULL_BYTES="${FULL_BYTES:-' not in runner
    assert "EP_ARTIFACT" in runner
    assert "TP_ARTIFACT" in runner
    assert "MX_ARTIFACT" in runner
