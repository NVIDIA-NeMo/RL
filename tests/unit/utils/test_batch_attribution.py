# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Scalar batch attribution must not change sample selection or loss masks."""

import ast
import json
import math
import sys
from collections import UserDict
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch

from nemo_rl.utils.batch_attribution import (
    READY_TAG,
    TASK_TAG,
    BatchAttribution,
    GroupTrainingStats,
    attach_task_labels,
    finish_entropy_metrics,
    reconciliation_metrics,
    reduce_group_stats,
    task_metric_prefix,
)
from nemo_rl.utils.batch_attribution_torch import (
    summarize_training_groups,
    task_entropy_metrics,
)


def _production_definitions(
    relative_path: str, names: list[str], namespace: dict[str, Any]
) -> dict[str, Any]:
    """Exercise CPU seams from production source without Ray/Lens startup."""
    path = Path(__file__).resolve().parents[3] / relative_path
    tree = ast.parse(path.read_text())
    nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
    ]
    assert {n.name for n in nodes} == set(names)
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, *nodes], type_ignores=[])
            ),
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace


def test_loss_diagnostics_unchanged_when_task_attribution_is_enabled() -> None:
    namespace = {"torch": torch, "task_entropy_metrics": task_entropy_metrics}
    _production_definitions(
        "nemo_rl/algorithms/utils.py", ["masked_mean", "calculate_kl"], namespace
    )
    _production_definitions(
        "nemo_rl/algorithms/loss/clipped_pg.py",
        ["clipped_pg_diagnostic_metrics"],
        namespace,
    )
    current = torch.tensor([[-0.2, -0.3], [-0.4, -0.5], [-0.6, -0.7]])
    args = dict(
        curr_logprobs=current,
        prev_logprobs=current - 0.01,
        generation_logprobs=current + 0.01,
        ratios=torch.ones_like(current),
        ratios_clamped=torch.ones_like(current),
        mask=torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]),
        global_valid_toks=torch.tensor(3.0),
        reference_policy_kl_type="k3",
    )
    fn = namespace["clipped_pg_diagnostic_metrics"]
    baseline = fn(**args)
    diagnostic = fn(**args, task_names=["math", "code", "math"])
    for key in baseline:
        torch.testing.assert_close(baseline[key], diagnostic[key], rtol=0, atol=0)
    assert sum(
        v for k, v in diagnostic.items() if k.endswith("/entropy_contribution")
    ) == pytest.approx(baseline["approx_entropy"].item())


def test_task_labels_survive_materialization_reordering_and_packing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Load the real row-container methods without its unrelated model imports.
    namespace = {
        "torch": torch,
        "UserDict": UserDict,
        "PackedTensor": type("PackedTensor", (), {}),
    }
    path = (
        Path(__file__).resolve().parents[3] / "nemo_rl/distributed/batched_data_dict.py"
    )
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "BatchedDataDict"
    )
    cls.bases = [ast.Name(id="UserDict", ctx=ast.Load())]
    cls.body = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"__init__", "slice", "select_indices", "reorder_data"}
    ]
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[])),
            str(path),
            "exec",
        ),
        namespace,
    )
    batch_type = namespace["BatchedDataDict"]
    namespace["SlicedDataDict"] = batch_type

    plane = ModuleType("nemo_rl.data_plane")
    plane.materialize = lambda data, **kwargs: batch_type(data)
    local = ModuleType("nemo_rl.data_plane.adapters.local")
    local.materialize_local = plane.materialize
    monkeypatch.setitem(sys.modules, "nemo_rl.data_plane", plane)
    monkeypatch.setitem(sys.modules, "nemo_rl.data_plane.adapters.local", local)
    _production_definitions(
        "nemo_rl/data_plane/worker_mixin.py", ["_materialize_fetched"], namespace
    )

    class Rows(dict):
        def keys(self, **kwargs):
            return super().keys()

    rows = Rows(input_ids=torch.arange(6).reshape(3, 2))
    tags = [
        {"batch_attribution_enabled": True, TASK_TAG: name}
        for name in ["math", "code", "genrm"]
    ]
    materialize = namespace["_materialize_fetched"]

    def fetch(data: Any, **kwargs: Any) -> Any:
        result = materialize(data, **kwargs)
        attach_task_labels(result, kwargs.get("tags"))
        return result

    for local_batch in [False, True]:
        batch = fetch(
            rows,
            local_batch=local_batch,
            layout="padded",
            pad_value_dict={},
            pad_to_seqlen=0,
            tags=tags,
        )
        batch = batch.select_indices([2, 0, 1])
        batch.reorder_data([0, 2, 1])
        assert batch["batch_attribution_task_names"] == ["genrm", "code", "math"]
        assert batch.slice(1, 2)["batch_attribution_task_names"] == ["code"]

    assert "batch_attribution_task_names" not in fetch(
        rows,
        local_batch=False,
        layout="padded",
        pad_value_dict={},
        pad_to_seqlen=0,
        tags=None,
    )
    with pytest.raises(ValueError, match="mixed"):
        fetch(
            rows,
            local_batch=False,
            layout="padded",
            pad_value_dict={},
            pad_to_seqlen=0,
            tags=[tags[0], {}, tags[2]],
        )

    # The unfused sequence-packing wrapper slices metadata with each sequence.
    namespace.update(math=math, _SEQ_METRIC_MIN=set(), _SEQ_METRIC_MAX=set())
    _production_definitions(
        "nemo_rl/algorithms/loss/wrapper.py", ["SequencePackingLossWrapper"], namespace
    )
    seen = []

    def prepare(**kwargs: Any) -> tuple[dict[str, Any], Any]:
        return {}, kwargs["data"]

    def loss(**kwargs: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        seen.extend(kwargs["data"]["batch_attribution_task_names"])
        return torch.tensor(0.0), {"metric": torch.tensor(1.0)}

    wrapper = namespace["SequencePackingLossWrapper"](loss, prepare, (0, 2, 4, 6))
    wrapper(torch.zeros(1, 6, 4), batch, torch.tensor(3.0), torch.tensor(6.0))
    assert seen == ["genrm", "code", "math"]


def test_group_reduction_uses_sums_and_exact_lengths() -> None:
    groups = [
        GroupTrainingStats("g1", "math", 2, 1, 10, 1, 1, (10,)),
        GroupTrainingStats("g2", "math", 2, 2, 40, 0, 2, (5, 35)),
        GroupTrainingStats("g3", "code", 2, 0, 0, 0, 0, ()),
    ]
    m = reduce_group_stats(groups)
    prefix = task_metric_prefix("math")
    assert m[prefix + "global_valid_toks"] == 50
    assert m[prefix + "global_valid_seqs"] == 3
    assert m[prefix + "reward_mean"] == pytest.approx(1 / 3)
    assert m[prefix + "response_length_mean"] == pytest.approx(50 / 3)
    assert m[prefix + "response_length_p95"] == 35
    assert m[task_metric_prefix("code") + "reward_mean"] == 0
    assert task_metric_prefix("a/b") != task_metric_prefix("a_b")


def test_group_statistics_honor_weighted_masks_and_exclude_filtered_nan_rewards() -> (
    None
):
    weights = torch.tensor([0.5, 0.0, 1.0])
    stats = summarize_training_groups(
        group_ids=["g1", "g1", "g2"],
        task_names=["math", "math", "code"],
        rewards=torch.tensor([2.0, float("nan"), -1.0]),
        training_mask=torch.tensor([[0.5, 0.5, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        sample_mask=weights,
    )
    assert stats[0].valid_sequence_weight == 0.5
    assert stats[0].valid_tokens == 1
    assert stats[0].reward_sum == 1
    assert stats[0].response_lengths == (2,)
    assert sum(g.valid_tokens for g in stats) == 4
    m = reduce_group_stats(list(stats))
    m.update(global_valid_toks=4, global_valid_seqs=1.5, reward=0)
    assert all(value == 0 for value in reconciliation_metrics(m).values())
    m["global_valid_toks"] = 7
    assert (
        reconciliation_metrics(m)["batch_attribution/reconcile/global_valid_toks_delta"]
        == -3
    )


def test_attribution_rejects_misaligned_metadata() -> None:
    # Shape validation is strict when opted in, not an implicit row fallback.
    with pytest.raises(ValueError, match="identities"):
        summarize_training_groups(
            group_ids=["g"],
            task_names=[],
            rewards=torch.tensor([1.0]),
            training_mask=torch.ones(1, 2),
            sample_mask=torch.ones(1),
        )


def test_entropy_contributions_reconcile_across_uneven_chunks() -> None:
    current = torch.tensor([[-0.1, -0.2], [-0.3, -0.4], [-0.5, -0.6]])
    generated = current + 0.02
    mask = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    labels = ["math", "code", "math"]
    merged = {}
    for indices in ([0], [1, 2]):
        metrics = task_entropy_metrics(
            curr_logprobs=current[indices],
            generation_logprobs=generated[indices],
            mask=mask[indices],
            task_names=[labels[i] for i in indices],
            global_valid_toks=torch.tensor(3.0),
        )
        for k, v in metrics.items():
            merged[k] = merged.get(k, 0) + v.item()
    finish_entropy_metrics(merged)
    expected = (-(current - generated).exp() * current * mask).sum() / mask.sum()
    assert sum(
        v for k, v in merged.items() if k.endswith("/entropy_contribution")
    ) == pytest.approx(expected.item())
    assert merged[task_metric_prefix("math") + "approx_entropy"] == pytest.approx(
        0.1 * torch.exp(torch.tensor(-0.02)).item()
    )
    assert merged[
        task_metric_prefix("code") + "entropy_token_fraction"
    ] == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="labels"):
        task_entropy_metrics(
            curr_logprobs=current,
            generation_logprobs=generated,
            mask=mask,
            task_names=["math"],
            global_valid_toks=torch.tensor(3.0),
        )


def test_restored_provenance_and_unknown_times_are_not_fabricated(
    tmp_path: Path,
) -> None:
    tracker = BatchAttribution(str(tmp_path))
    tracker.restore_group("ready", origin="restored_complete", task_name="math")
    tracker.restore_group(
        "pending", origin="restored_unfinished", task_name="code", sealed_siblings=2
    )
    tracker.dispatch("pending", task_name="code", trainer_version=75, target_step=None)
    tags = tracker.ready("pending", task_name="code")
    assert tags[TASK_TAG] == "code" and tags[READY_TAG] > 0
    assert tracker.groups["pending"].origin == "restored_unfinished"
    tracker.selected(
        group_ids=["ready", "pending"],
        step=76,
        trainer_version=75,
        chunk=1,
        sampler="ready_first",
        inventory=[],
        min_groups=2,
        max_groups=2,
    )
    selected = [e for e in tracker.events if e["event"] == "group_selected"]
    assert selected[0]["ready_unix_s"] is None
    assert selected[0]["ready_wait_current_attempt_s"] is None
    assert selected[1]["ready_wait_current_attempt_s"] >= 0
    tracker.add_training_stats(
        (GroupTrainingStats("pending", "code", 2, 2, 30, 1, 2, (10, 20)),)
    )
    tracker.finish_step(76)
    tracker.write(tracker.drain())
    rows = [json.loads(line) for line in tracker.path.read_text().splitlines()]
    assert not tracker.events and not tracker.step_groups
    assert rows[-1]["event"] == "step_committed"
    consumed = next(e for e in rows if e["event"] == "group_consumed")
    assert consumed["origin"] == "restored_unfinished"
    assert "response_lengths" not in consumed
    assert [e["sequence"] for e in rows] == list(range(len(rows)))
