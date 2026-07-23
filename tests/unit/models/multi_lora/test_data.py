"""Strong unit-test suite for ``MultiAdapterDataLoader``.

Organized by concern:

  1. Construction & validation
  2. Iteration semantics (one-step and multi-step)
  3. Auto-restart contract
  4. ``_pack`` staticmethod   (the value-merge logic)
  5. ``_row_count`` staticmethod
  6. ``from_config`` (production entry point — monkey-patches the
     NeMo-RL data chain so the test runs without it)
  7. State isolation between instances and across iterations
  8. Determinism / repeatability

No NeMo-RL / torchdata dependency: trivial DataLoader-like objects whose
``__iter__`` yields dict batches stand in for ``StatefulDataLoader``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch

from nemo_rl.models.multi_lora.data import MultiAdapterDataLoader


# =============================================================================
# Fakes
# =============================================================================


@dataclass
class _FakeAdapter:
    name: str
    data: dict


@dataclass
class _FakeMLCfg:
    adapters: list[_FakeAdapter]
    batch_size_per_adapter: int = 2


class _FakeLoader:
    """Yields a fixed list of dict batches; repeatable across iter() calls."""

    def __init__(self, batches: list[dict[str, Any]]):
        self.batches = batches
        self.iter_count = 0  # how many times __iter__ was called

    def __iter__(self):
        self.iter_count += 1
        return iter(self.batches)


def _tensor_batch(name: str, n_rows: int, in_f: int = 4) -> dict[str, Any]:
    return {
        "input_ids": torch.arange(n_rows * in_f, dtype=torch.long).reshape(n_rows, in_f),
        "labels": torch.full((n_rows, in_f), ord(name[-1]), dtype=torch.long),
        "lengths": [in_f] * n_rows,
    }


def _message_log_batch(name: str, n_rows: int) -> dict[str, Any]:
    return {
        "message_log": [[{"role": "user", "content": f"{name}{i}"}] for i in range(n_rows)],
        "input_ids": torch.zeros(n_rows, 8, dtype=torch.long),
    }


# =============================================================================
# 1. Construction & validation
# =============================================================================


class TestConstruction:
    def test_rejects_empty_adapter_names(self):
        with pytest.raises(ValueError, match="adapter_names must be non-empty"):
            MultiAdapterDataLoader(loaders={}, adapter_names=[])

    def test_rejects_missing_loader(self):
        with pytest.raises(KeyError, match="loaders missing for adapters: \\['b'\\]"):
            MultiAdapterDataLoader(
                loaders={"a": _FakeLoader([_tensor_batch("a", 1)])},
                adapter_names=["a", "b"],
            )

    def test_rejects_multiple_missing_loaders(self):
        with pytest.raises(KeyError, match="loaders missing for adapters:"):
            MultiAdapterDataLoader(
                loaders={"a": _FakeLoader([_tensor_batch("a", 1)])},
                adapter_names=["a", "b", "c"],
            )

    def test_constructs_iter_per_adapter(self):
        loaders = {
            "a": _FakeLoader([_tensor_batch("a", 1)]),
            "b": _FakeLoader([_tensor_batch("b", 1)]),
        }
        mdl = MultiAdapterDataLoader(loaders=loaders, adapter_names=["a", "b"])
        assert set(mdl._iters.keys()) == {"a", "b"}
        # Constructing the iters does NOT consume any batches yet.
        assert loaders["a"].iter_count == 1
        assert loaders["b"].iter_count == 1

    def test_adapter_names_stored_as_list_copy(self):
        """Mutating the passed list must not affect the instance."""
        names = ["a", "b"]
        loaders = {n: _FakeLoader([_tensor_batch(n, 1)]) for n in names}
        mdl = MultiAdapterDataLoader(loaders=loaders, adapter_names=names)
        names.append("c")  # mutate caller's list
        assert mdl._adapter_names == ["a", "b"]


# =============================================================================
# 2. Iteration semantics
# =============================================================================


class TestIteration:
    def test_iter_returns_self(self):
        mdl = MultiAdapterDataLoader(
            loaders={"a": _FakeLoader([_tensor_batch("a", 1)])}, adapter_names=["a"]
        )
        assert iter(mdl) is mdl

    def test_yields_packed_batch_with_adapter_names(self):
        loaders = {
            "a": _FakeLoader([_tensor_batch("a", 2)]),
            "b": _FakeLoader([_tensor_batch("b", 3)]),
        }
        mdl = MultiAdapterDataLoader(loaders=loaders, adapter_names=["a", "b"])
        batch = next(mdl)
        assert batch["adapter_names"] == ["a", "a", "b", "b", "b"]

    def test_concatenates_tensors_along_dim_0(self):
        loaders = {
            "a": _FakeLoader([_tensor_batch("a", 2)]),
            "b": _FakeLoader([_tensor_batch("b", 3)]),
        }
        mdl = MultiAdapterDataLoader(loaders=loaders, adapter_names=["a", "b"])
        batch = next(mdl)
        assert isinstance(batch["input_ids"], torch.Tensor)
        assert batch["input_ids"].shape == (5, 4)

    def test_concatenation_preserves_row_values(self):
        loaders = {
            "a": _FakeLoader([_tensor_batch("a", 2)]),
            "b": _FakeLoader([_tensor_batch("b", 2)]),
        }
        mdl = MultiAdapterDataLoader(loaders=loaders, adapter_names=["a", "b"])
        batch = next(mdl)
        # adapter_a's labels were filled with ord('a'); adapter_b's with ord('b').
        assert (batch["labels"][:2] == ord("a")).all()
        assert (batch["labels"][2:] == ord("b")).all()

    def test_extends_lists(self):
        loaders = {
            "a": _FakeLoader([_tensor_batch("a", 2)]),
            "b": _FakeLoader([_tensor_batch("b", 3)]),
        }
        mdl = MultiAdapterDataLoader(loaders=loaders, adapter_names=["a", "b"])
        batch = next(mdl)
        assert isinstance(batch["lengths"], list)
        assert batch["lengths"] == [4, 4, 4, 4, 4]

    def test_adapter_order_dictates_row_layout(self):
        loaders = {
            "a": _FakeLoader([_tensor_batch("a", 2)]),
            "b": _FakeLoader([_tensor_batch("b", 2)]),
        }
        mdl_ab = MultiAdapterDataLoader(loaders=loaders, adapter_names=["a", "b"])
        mdl_ba = MultiAdapterDataLoader(loaders=loaders, adapter_names=["b", "a"])
        assert next(mdl_ab)["adapter_names"] == ["a", "a", "b", "b"]
        assert next(mdl_ba)["adapter_names"] == ["b", "b", "a", "a"]

    def test_message_log_dictates_row_count(self):
        loaders = {
            "a": _FakeLoader([_message_log_batch("a", 3)]),
            "b": _FakeLoader([_message_log_batch("b", 2)]),
        }
        mdl = MultiAdapterDataLoader(loaders=loaders, adapter_names=["a", "b"])
        batch = next(mdl)
        assert batch["adapter_names"] == ["a", "a", "a", "b", "b"]

    def test_unsupported_value_type_raises(self):
        loaders = {"a": _FakeLoader([{"weird": object()}])}
        mdl = MultiAdapterDataLoader(loaders=loaders, adapter_names=["a"])
        with pytest.raises(TypeError, match="unsupported"):
            next(mdl)

    def test_single_adapter_packed_batch_passes_through(self):
        loaders = {"a": _FakeLoader([_tensor_batch("a", 3)])}
        mdl = MultiAdapterDataLoader(loaders=loaders, adapter_names=["a"])
        batch = next(mdl)
        assert batch["adapter_names"] == ["a"] * 3
        assert batch["input_ids"].shape == (3, 4)

    def test_four_adapters_concatenate_in_order(self):
        loaders = {n: _FakeLoader([_tensor_batch(n, 1)]) for n in "abcd"}
        mdl = MultiAdapterDataLoader(loaders=loaders, adapter_names=list("abcd"))
        batch = next(mdl)
        assert batch["adapter_names"] == ["a", "b", "c", "d"]
        assert batch["input_ids"].shape == (4, 4)


# =============================================================================
# 3. Auto-restart contract
# =============================================================================


class TestAutoRestart:
    def test_exhausted_iterator_restarts(self):
        a_batches = [_tensor_batch("a", 1), _tensor_batch("a", 1)]
        b_batches = [_tensor_batch("b", 1), _tensor_batch("b", 1)]
        loaders = {"a": _FakeLoader(a_batches), "b": _FakeLoader(b_batches)}
        mdl = MultiAdapterDataLoader(loaders=loaders, adapter_names=["a", "b"])
        # Draw 5 packed batches from loaders that only hold 2 each.
        seen = [next(mdl) for _ in range(5)]
        assert len(seen) == 5

    def test_unbalanced_streams_restart_independently(self):
        loaders = {
            "a": _FakeLoader([_tensor_batch("a", 1)]),                # 1 batch
            "b": _FakeLoader([_tensor_batch("b", 1) for _ in range(3)]),  # 3 batches
        }
        mdl = MultiAdapterDataLoader(loaders=loaders, adapter_names=["a", "b"])
        # 6 steps means: a restarts 5 extra times, b restarts 1 extra time.
        for _ in range(6):
            next(mdl)
        assert loaders["a"].iter_count == 6
        assert loaders["b"].iter_count == 2

    def test_restart_iter_count_tracked_per_adapter(self):
        loaders = {
            "a": _FakeLoader([_tensor_batch("a", 1)]),
            "b": _FakeLoader([_tensor_batch("b", 1)]),
        }
        mdl = MultiAdapterDataLoader(loaders=loaders, adapter_names=["a", "b"])
        for _ in range(4):
            next(mdl)
        # Each step exhausts both 1-batch loaders -> 4 packed batches need 4
        # iter() calls per loader on top of the initial 1 from __init__.
        # Counting: init=1, step1 init done, step2 restart, step3 restart,
        # step4 restart -> 4 total iter() calls per loader.
        assert loaders["a"].iter_count == 4
        assert loaders["b"].iter_count == 4


# =============================================================================
# 4. _pack staticmethod
# =============================================================================


class TestPack:
    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="sub_batches must be non-empty"):
            MultiAdapterDataLoader._pack([])

    def test_preserves_order_across_adapters(self):
        sb = [
            ("a", {"input_ids": torch.tensor([[1, 1]]), "lens": [1]}),
            ("b", {"input_ids": torch.tensor([[2, 2], [2, 2]]), "lens": [2, 2]}),
        ]
        out = MultiAdapterDataLoader._pack(sb)
        assert out["adapter_names"] == ["a", "b", "b"]
        assert out["input_ids"].tolist() == [[1, 1], [2, 2], [2, 2]]
        assert out["lens"] == [1, 2, 2]

    def test_single_sub_batch_passes_through(self):
        sb = [("only", {"input_ids": torch.zeros(3, 4), "x": [0, 1, 2]})]
        out = MultiAdapterDataLoader._pack(sb)
        assert out["adapter_names"] == ["only"] * 3
        assert out["input_ids"].shape == (3, 4)
        assert out["x"] == [0, 1, 2]

    def test_unsupported_value_type_raises(self):
        sb = [("a", {"weird": object()})]
        with pytest.raises(TypeError, match="unsupported per-adapter batch value type for key 'weird'"):
            MultiAdapterDataLoader._pack(sb)

    def test_mixed_keys_across_sub_batches_raises_on_missing(self):
        """If sub-batches have inconsistent keys, the pack falls back to
        the first batch's keys; absent keys in later batches raise KeyError
        on the tensor cat / list extend."""
        sb = [
            ("a", {"input_ids": torch.zeros(1, 4)}),
            ("b", {"completely_different": torch.zeros(1, 4)}),
        ]
        with pytest.raises(KeyError):
            MultiAdapterDataLoader._pack(sb)

    def test_tensor_dtype_preserved(self):
        sb = [
            ("a", {"x": torch.tensor([[1.0]], dtype=torch.bfloat16)}),
            ("b", {"x": torch.tensor([[2.0]], dtype=torch.bfloat16)}),
        ]
        out = MultiAdapterDataLoader._pack(sb)
        assert out["x"].dtype == torch.bfloat16

    def test_tensor_concat_along_dim_0_only(self):
        """Even multi-dim tensors concat on dim 0."""
        sb = [
            ("a", {"x": torch.zeros(2, 3, 5)}),
            ("b", {"x": torch.ones(4, 3, 5)}),
        ]
        out = MultiAdapterDataLoader._pack(sb)
        assert out["x"].shape == (6, 3, 5)
        assert (out["x"][:2] == 0).all()
        assert (out["x"][2:] == 1).all()

    def test_message_log_row_count_used_for_adapter_names(self):
        sb = [
            ("a", {"message_log": [1, 2, 3], "input_ids": torch.zeros(10, 4)}),
            ("b", {"message_log": [4, 5], "input_ids": torch.zeros(99, 4)}),
        ]
        out = MultiAdapterDataLoader._pack(sb)
        # message_log lengths (3 + 2) drive adapter_names — not the tensor's
        # shape[0] (10 + 99).
        assert out["adapter_names"] == ["a", "a", "a", "b", "b"]


# =============================================================================
# 5. _row_count staticmethod
# =============================================================================


class TestRowCount:
    def test_prefers_message_log(self):
        b = {"message_log": [1, 2, 3], "input_ids": torch.zeros(10, 4)}
        assert MultiAdapterDataLoader._row_count(b) == 3

    def test_falls_back_to_tensor_leading_dim(self):
        b = {"input_ids": torch.zeros(7, 4)}
        assert MultiAdapterDataLoader._row_count(b) == 7

    def test_picks_first_tensor_found(self):
        # Two tensors -> first one wins. Python dict order is insertion
        # order, so this is well-defined.
        b = {"a": torch.zeros(5, 1), "b": torch.zeros(99, 1)}
        assert MultiAdapterDataLoader._row_count(b) == 5

    def test_raises_when_no_message_log_no_tensor(self):
        with pytest.raises(ValueError, match="neither message_log list nor any tensor"):
            MultiAdapterDataLoader._row_count({"foo": 1})

    def test_non_list_message_log_falls_through_to_tensor(self):
        b = {"message_log": "not a list", "input_ids": torch.zeros(4, 1)}
        assert MultiAdapterDataLoader._row_count(b) == 4

    def test_empty_message_log_returns_zero(self):
        b = {"message_log": [], "input_ids": torch.zeros(99, 1)}
        assert MultiAdapterDataLoader._row_count(b) == 0


# =============================================================================
# 6. from_config — monkey-patches the NeMo-RL builder
# =============================================================================


class TestFromConfig:
    def test_calls_builder_once_per_adapter_in_order(self, monkeypatch):
        calls: list[tuple[str, int, int]] = []

        def fake_build_one(adapter, base_data_cfg, tokenizer, max_seq_length, batch_size):
            calls.append((adapter.name, batch_size, max_seq_length))
            return _FakeLoader([_tensor_batch(adapter.name, 1)])

        monkeypatch.setattr(MultiAdapterDataLoader, "_build_one", staticmethod(fake_build_one))
        ml_cfg = _FakeMLCfg(
            adapters=[_FakeAdapter("a", {}), _FakeAdapter("b", {}), _FakeAdapter("c", {})],
            batch_size_per_adapter=4,
        )
        mdl = MultiAdapterDataLoader.from_config(
            ml_cfg=ml_cfg, base_data_cfg={}, tokenizer=None, max_seq_length=2048,
        )
        assert [c[0] for c in calls] == ["a", "b", "c"]
        assert all(c[1] == 4 for c in calls)
        assert all(c[2] == 2048 for c in calls)
        batch = next(mdl)
        assert batch["adapter_names"] == ["a", "b", "c"]

    def test_per_adapter_data_dict_passed_to_builder(self, monkeypatch):
        seen: list[dict] = []

        def fake_build_one(adapter, base_data_cfg, tokenizer, max_seq_length, batch_size):
            seen.append(adapter.data)
            return _FakeLoader([_tensor_batch(adapter.name, 1)])

        monkeypatch.setattr(MultiAdapterDataLoader, "_build_one", staticmethod(fake_build_one))
        ml_cfg = _FakeMLCfg(
            adapters=[
                _FakeAdapter("a", {"train": {"dataset_name": "alpha"}}),
                _FakeAdapter("b", {"train": {"dataset_name": "beta"}}),
            ]
        )
        MultiAdapterDataLoader.from_config(
            ml_cfg=ml_cfg, base_data_cfg={}, tokenizer=None, max_seq_length=1024,
        )
        assert seen[0] == {"train": {"dataset_name": "alpha"}}
        assert seen[1] == {"train": {"dataset_name": "beta"}}

    def test_returns_instance_with_adapter_order_preserved(self, monkeypatch):
        monkeypatch.setattr(
            MultiAdapterDataLoader, "_build_one",
            staticmethod(lambda adapter, **_: _FakeLoader([_tensor_batch(adapter.name, 1)])),
        )
        ml_cfg = _FakeMLCfg(adapters=[_FakeAdapter("z", {}), _FakeAdapter("a", {})])
        mdl = MultiAdapterDataLoader.from_config(
            ml_cfg=ml_cfg, base_data_cfg={}, tokenizer=None, max_seq_length=1024,
        )
        assert mdl._adapter_names == ["z", "a"]


# =============================================================================
# 7. State isolation
# =============================================================================


class TestStateIsolation:
    def test_two_instances_share_no_iterator_state(self):
        loaders = {"a": _FakeLoader([_tensor_batch("a", 1) for _ in range(3)])}
        mdl1 = MultiAdapterDataLoader(loaders=loaders, adapter_names=["a"])
        mdl2 = MultiAdapterDataLoader(loaders=loaders, adapter_names=["a"])
        next(mdl1)
        next(mdl1)
        # mdl2's iterator is fresh — it has not been advanced.
        b = next(mdl2)
        assert b["input_ids"][0, 0].item() == 0  # first batch starts at 0

    def test_one_step_does_not_corrupt_next_step(self):
        loaders = {
            "a": _FakeLoader([_tensor_batch("a", 2) for _ in range(5)]),
            "b": _FakeLoader([_tensor_batch("b", 3) for _ in range(5)]),
        }
        mdl = MultiAdapterDataLoader(loaders=loaders, adapter_names=["a", "b"])
        for _ in range(3):
            batch = next(mdl)
            assert batch["adapter_names"] == ["a", "a", "b", "b", "b"]
            assert batch["input_ids"].shape == (5, 4)

    def test_packed_batch_does_not_share_storage_with_inputs(self):
        """The merged tensor must be a new allocation — mutating it must
        not mutate the per-adapter batches."""
        a_batch = _tensor_batch("a", 1)
        b_batch = _tensor_batch("b", 1)
        original_a = a_batch["input_ids"].clone()
        loaders = {"a": _FakeLoader([a_batch]), "b": _FakeLoader([b_batch])}
        mdl = MultiAdapterDataLoader(loaders=loaders, adapter_names=["a", "b"])
        merged = next(mdl)
        merged["input_ids"][0, 0] = 999
        assert torch.equal(a_batch["input_ids"], original_a)


# =============================================================================
# 8. Determinism / repeatability
# =============================================================================


class TestDeterminism:
    def test_repeated_iter_over_same_loaders_yields_same_batches(self):
        loaders = {
            "a": _FakeLoader([_tensor_batch("a", 1) for _ in range(3)]),
            "b": _FakeLoader([_tensor_batch("b", 1) for _ in range(3)]),
        }
        mdl1 = MultiAdapterDataLoader(loaders=loaders, adapter_names=["a", "b"])
        mdl2 = MultiAdapterDataLoader(loaders=loaders, adapter_names=["a", "b"])
        batches1 = [next(mdl1) for _ in range(3)]
        batches2 = [next(mdl2) for _ in range(3)]
        for b1, b2 in zip(batches1, batches2):
            assert b1["adapter_names"] == b2["adapter_names"]
            assert torch.equal(b1["input_ids"], b2["input_ids"])

    def test_adapter_names_field_is_a_fresh_list_not_aliased(self):
        loaders = {"a": _FakeLoader([_tensor_batch("a", 1)])}
        mdl = MultiAdapterDataLoader(loaders=loaders, adapter_names=["a"])
        b1 = next(mdl)
        b2 = next(mdl)
        # Mutate b1's adapter_names; b2 must be unaffected.
        b1["adapter_names"].append("MUTATED")
        assert "MUTATED" not in b2["adapter_names"]


# =============================================================================
# 9. Worklog-derived contracts (gaps surfaced by reading 2026-05-* worklogs)
# =============================================================================


class TestBlockContiguousLayout:
    """`adapter_names` must be block-contiguous, never interleaved.
    Worklog 2026-05-05/HANDOFF §9 + 2026-05-08/03: block-contiguous layout
    is what enables BMM forward and uniform-router fast-paths.
    """

    def test_two_adapters_emit_block_layout(self):
        loaders = {
            "a": _FakeLoader([_tensor_batch("a", 3)]),
            "b": _FakeLoader([_tensor_batch("b", 2)]),
        }
        mdl = MultiAdapterDataLoader(loaders=loaders, adapter_names=["a", "b"])
        names = next(mdl)["adapter_names"]
        # Block-contiguous: all a's, then all b's. No interleaving.
        assert names == ["a", "a", "a", "b", "b"]
        # Every adapter appears as one contiguous run.
        assert names[: names.count("a")] == ["a"] * names.count("a")
        assert names[names.count("a") :] == ["b"] * names.count("b")

    def test_four_adapters_emit_block_layout(self):
        loaders = {n: _FakeLoader([_tensor_batch(n, 2)]) for n in "abcd"}
        mdl = MultiAdapterDataLoader(loaders=loaders, adapter_names=list("abcd"))
        names = next(mdl)["adapter_names"]
        assert names == ["a", "a", "b", "b", "c", "c", "d", "d"]


class TestLegacyFieldsNotEmitted:
    """`adapter_id_map` slices wrong under `BatchedDataDict.slice` (worklog
    2026-05-14/09). `adapter_ids: LongTensor` was the old positional
    convention; the slim path uses `adapter_names` only.
    """

    def test_pack_does_not_emit_adapter_id_map(self):
        sb = [
            ("a", {"input_ids": torch.zeros(1, 4)}),
            ("b", {"input_ids": torch.zeros(1, 4)}),
        ]
        out = MultiAdapterDataLoader._pack(sb)
        assert "adapter_id_map" not in out

    def test_pack_emits_canonical_adapter_ids(self):
        """`adapter_ids` is the canonical per-row GLOBAL slot-id tensor for
        the routing hook (row's index into canonical adapter order). It was
        once a legacy field; re-introduced for microbatch routing — must be
        a LongTensor[B] aligned with adapter_names."""
        sb = [
            ("a", {"input_ids": torch.zeros(1, 4)}),
            ("b", {"input_ids": torch.zeros(1, 4)}),
        ]
        out = MultiAdapterDataLoader._pack(sb)
        assert "adapter_ids" in out
        ids = out["adapter_ids"]
        assert isinstance(ids, torch.Tensor) and ids.dtype == torch.long
        assert ids.tolist() == [0, 1]
        assert out["adapter_names"] == ["a", "b"]


class TestPackKeyConsistency:
    """`_pack` must raise when later sub-batches carry extra keys not
    present in the first sub-batch. Silent drop was the latent failure
    in worklog 2026-05-07/02 M15.
    """

    def test_extra_key_in_later_sub_batch_raises(self):
        sb = [
            ("a", {"input_ids": torch.zeros(1, 4)}),
            ("b", {"input_ids": torch.zeros(1, 4), "extra_meta": [42]}),
        ]
        with pytest.raises(KeyError, match="extra_meta"):
            MultiAdapterDataLoader._pack(sb)

    def test_missing_key_in_later_sub_batch_still_raises_on_use(self):
        """Symmetric to the above: a key in first but missing in later
        must still surface as an error (not silent fill)."""
        sb = [
            ("a", {"input_ids": torch.zeros(1, 4), "lens": [1]}),
            ("b", {"input_ids": torch.zeros(1, 4)}),  # missing "lens"
        ]
        with pytest.raises(KeyError):
            MultiAdapterDataLoader._pack(sb)


class TestMergeDataCfg:
    """Per-adapter `data` dict OVERRIDES `base_data_cfg`. Worklog
    2026-04-28/01 cites config-merge inversion as a silent data-routing bug.
    """

    def test_adapter_data_wins_over_base(self):
        merged = MultiAdapterDataLoader._merge_data_cfg(
            base_data_cfg={"dataset_name": "base", "train": {"shuffle": False}},
            adapter_data={"dataset_name": "adapter"},
        )
        assert merged["dataset_name"] == "adapter"
        # Non-overlapping keys from base survive.
        assert merged["train"] == {"shuffle": False}

    def test_none_base_returns_adapter(self):
        merged = MultiAdapterDataLoader._merge_data_cfg(
            base_data_cfg=None, adapter_data={"dataset_name": "x"},
        )
        assert merged == {"dataset_name": "x"}

    def test_none_adapter_returns_base(self):
        merged = MultiAdapterDataLoader._merge_data_cfg(
            base_data_cfg={"dataset_name": "x"}, adapter_data=None,
        )
        assert merged == {"dataset_name": "x"}

    def test_both_none_returns_empty(self):
        assert MultiAdapterDataLoader._merge_data_cfg(None, None) == {}


class TestNonAlphabeticalOrder:
    """`from_config` must preserve `ml_cfg.adapters` order EXACTLY — not
    sort. Worklog 2026-05-13/01: silently re-sorting adapter order broke
    per-row routing because data and model disagreed.
    """

    def test_non_alphabetical_order_preserved_in_iteration(self, monkeypatch):
        monkeypatch.setattr(
            MultiAdapterDataLoader, "_build_one",
            staticmethod(
                lambda adapter, **_: _FakeLoader([_tensor_batch(adapter.name, 1)])
            ),
        )
        ml_cfg = _FakeMLCfg(
            adapters=[
                _FakeAdapter("zebra", {}),
                _FakeAdapter("apple", {}),
                _FakeAdapter("mango", {}),
            ]
        )
        mdl = MultiAdapterDataLoader.from_config(
            ml_cfg=ml_cfg, base_data_cfg={}, tokenizer=None, max_seq_length=1024,
        )
        batch = next(mdl)
        # Order matches ml_cfg.adapters, not alphabetical.
        assert batch["adapter_names"] == ["zebra", "apple", "mango"]
        assert mdl._adapter_names == ["zebra", "apple", "mango"]


class TestRestartContract:
    """Iterating a fully-exhausted loader and then a fresh restart must
    yield the SAME sequence (assuming the underlying loader is
    deterministic). Pins the auto-restart contract.
    """

    def test_full_pass_repeats_identical_sequence(self):
        # 2 batches per loader, 4 packed batches total = 2 full passes.
        loaders = {
            "a": _FakeLoader([_tensor_batch("a", 1), _tensor_batch("a", 1)]),
            "b": _FakeLoader([_tensor_batch("b", 1), _tensor_batch("b", 1)]),
        }
        mdl = MultiAdapterDataLoader(loaders=loaders, adapter_names=["a", "b"])
        pass1 = [next(mdl) for _ in range(2)]
        pass2 = [next(mdl) for _ in range(2)]
        for p1, p2 in zip(pass1, pass2):
            assert torch.equal(p1["input_ids"], p2["input_ids"])
            assert p1["adapter_names"] == p2["adapter_names"]


class _StatefulFakeLoader(_FakeLoader):
    """_FakeLoader + torchdata StatefulDataLoader protocol: state is the
    number of batches consumed in the current pass; iter() after a
    load_state_dict resumes mid-pass from that position."""

    def __init__(self, batches):
        super().__init__(batches)
        self._pos = 0

    def __iter__(self):
        self.iter_count += 1
        start = self._pos
        self._pos = 0  # a natural restart begins a fresh pass

        def gen():
            for i in range(start, len(self.batches)):
                self._consumed(i)
                yield self.batches[i]

        return gen()

    def _consumed(self, i):
        self._pos = i + 1

    def state_dict(self):
        return {"pos": self._pos}

    def load_state_dict(self, sd):
        self._pos = sd["pos"]


class TestCheckpointStateRoundTrip:
    """state_dict/load_state_dict must resume the packed stream exactly
    where it left off (preemption-resume contract; code5 218385 stranded
    tmp_step_1500 because MultiAdapterDataLoader had no state_dict)."""

    @staticmethod
    def _make(n=4):
        loaders = {
            "a": _StatefulFakeLoader([_tensor_batch("a", 1) for _ in range(n)]),
            "b": _StatefulFakeLoader([_tensor_batch("b", 1) for _ in range(n)]),
        }
        return MultiAdapterDataLoader(loaders=loaders, adapter_names=["a", "b"])

    def test_resume_continues_from_checkpoint(self):
        ref = self._make()
        expected = [next(ref) for _ in range(4)]

        mdl = self._make()
        got = [next(mdl) for _ in range(2)]
        sd = mdl.state_dict()

        resumed = self._make()          # fresh instance, as after preemption
        resumed.load_state_dict(sd)
        got += [next(resumed) for _ in range(2)]

        for e, g in zip(expected, got):
            assert torch.equal(e["input_ids"], g["input_ids"])
            assert torch.equal(e["adapter_ids"], g["adapter_ids"])
            assert e["adapter_names"] == g["adapter_names"]

    def test_state_dict_shape(self):
        mdl = self._make()
        next(mdl)
        sd = mdl.state_dict()
        assert set(sd["adapter_loader_states"]) == {"a", "b"}

    def test_load_state_dict_missing_adapter_raises(self):
        mdl = self._make()
        with pytest.raises(KeyError, match="missing for adapters"):
            mdl.load_state_dict({"adapter_loader_states": {"a": {"pos": 0}}})
