"""One iterator that yields packed multi-adapter SFT batches.

Public surface is one class: ``MultiAdapterDataLoader``.

- Primary constructor takes pre-built per-adapter loaders. Used by tests.
- ``from_config(...)`` builds the N NeMo-RL data chains and is the
  production entry point.

Each yielded batch is the round-robin concatenation of one micro-batch
from every adapter, with ``adapter_names`` (``list[str]`` of length B,
one name per row) attached for downstream routing. Exhausted streams
auto-restart, so iteration is effectively infinite.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Iterator

import torch

logger = logging.getLogger(__name__)


class MultiAdapterDataLoader:
    """Round-robin multi-adapter SFT dataloader.

    Args:
        loaders: dict mapping adapter name -> a torch-style DataLoader.
            Each loader yields dicts of (list | Tensor) values; lists get
            extended, tensors get concatenated along dim 0.
        adapter_names: ordered list of adapter names. Each ``__next__``
            draws one micro-batch from ``loaders[name]`` for each name in
            this order, then concatenates them. The order also dictates
            which rows in the merged batch belong to which adapter.
    """

    def __init__(self, loaders: dict[str, Any], adapter_names: list[str]):
        if not adapter_names:
            raise ValueError("adapter_names must be non-empty")
        missing = [n for n in adapter_names if n not in loaders]
        if missing:
            raise KeyError(f"loaders missing for adapters: {missing}")
        self._loaders = loaders
        self._adapter_names = list(adapter_names)
        self._iters: dict[str, Iterator] = {n: iter(loaders[n]) for n in adapter_names}

    def __len__(self) -> int:
        # Round-robin emits one merged batch per pass; truncate at the shortest loader.
        return min(len(self._loaders[n]) for n in self._adapter_names)

    @classmethod
    def from_config(
        cls,
        ml_cfg: Any,
        base_data_cfg: dict,
        tokenizer: Any,
        max_seq_length: int,
    ) -> "MultiAdapterDataLoader":
        """Build N per-adapter NeMo-RL data chains and wrap them.

        For each adapter, overlay ``adapter.data`` on ``base_data_cfg`` and
        run NeMo-RL's standard SFT data chain
        (``load_response_dataset`` -> ``AllTaskProcessedDataset`` ->
        ``StatefulDataLoader`` with ``rl_collate_fn``).
        """
        adapter_names = [a.name for a in ml_cfg.adapters]
        loaders = {
            a.name: cls._build_one(
                adapter=a,
                base_data_cfg=base_data_cfg,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                batch_size=ml_cfg.batch_size_per_adapter,
            )
            for a in ml_cfg.adapters
        }
        return cls(loaders=loaders, adapter_names=adapter_names)

    def __iter__(self) -> "MultiAdapterDataLoader":
        return self

    def __next__(self) -> dict[str, Any]:
        sub_batches = [(name, self._next_for(name)) for name in self._adapter_names]
        return self._pack(sub_batches)

    def _next_for(self, name: str) -> dict[str, Any]:
        """Draw the next micro-batch for ``name``, restarting if exhausted."""
        try:
            return next(self._iters[name])
        except StopIteration:
            self._iters[name] = iter(self._loaders[name])
            return next(self._iters[name])

    # ------------------------------------------------------------------ #
    # Checkpoint state (StatefulDataLoader protocol)
    # ------------------------------------------------------------------ #
    # NeMo-RL's SFT loop calls ``train_dataloader.state_dict()`` on every
    # checkpoint save and ``load_state_dict()`` on resume. Without these,
    # the save raises AttributeError AFTER policy weights are written but
    # BEFORE ``finalize_checkpoint`` — the checkpoint is stranded as
    # ``tmp_step_N`` (invisible to ``get_latest_checkpoint_path``) and
    # preempted runs restart from step 0 (observed: code5 multi 218385).

    def state_dict(self) -> dict[str, Any]:
        """Per-adapter StatefulDataLoader states, keyed by adapter name."""
        return {
            "adapter_loader_states": {
                n: self._loaders[n].state_dict() for n in self._adapter_names
            }
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        """Restore per-adapter loader positions; recreate live iterators.

        Iterators created in ``__init__`` predate the restore, so they are
        rebuilt here — ``iter()`` on a restored StatefulDataLoader resumes
        mid-epoch from the loaded position.
        """
        states = sd["adapter_loader_states"]
        missing = [n for n in self._adapter_names if n not in states]
        if missing:
            raise KeyError(
                f"dataloader state missing for adapters: {missing} "
                f"(have: {sorted(states)})"
            )
        for n in self._adapter_names:
            self._loaders[n].load_state_dict(states[n])
        self._iters = {n: iter(self._loaders[n]) for n in self._adapter_names}


    # ------------------------------------------------------------------ #
    # Staticmethods — pure helpers, kept on the class for cohesion.
    # ------------------------------------------------------------------ #

    @staticmethod
    def _pack(sub_batches: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
        """Concatenate one batch per adapter into a single packed batch.

        The output dict mirrors the per-adapter batches (lists are extended,
        tensors are concatenated along dim 0) plus a per-row ``adapter_names``
        field (``list[str]`` of length B) that slices correctly under
        ``BatchedDataDict.slice(start, end)``.

        Contract:
        - ``adapter_names`` is block-contiguous: all of sub_batches[0]'s rows
          first, then sub_batches[1]'s rows, etc. Never interleaved.
        - All sub-batches must share the SAME set of keys. Extra keys in
          later sub-batches raise to surface upstream-data bugs (silent drop
          was the latent failure flagged in worklog 2026-05-07/02 M15).
        - Legacy field ``adapter_id_map`` is NOT emitted (slices wrong
          under microbatching). ``adapter_ids`` IS emitted: a per-row
          ``LongTensor[B]`` of GLOBAL slot ids (row's index into the
          canonical adapter order) consumed by the routing hook /
          ``MultiLinearLoRA.lora_A[ids]``. It slices correctly under
          ``BatchedDataDict.slice`` because it is a plain dim-0 tensor.
        """
        if not sub_batches:
            raise ValueError("sub_batches must be non-empty")
        first = sub_batches[0][1]
        first_keys = set(first.keys())
        for name, b in sub_batches[1:]:
            extra = set(b.keys()) - first_keys
            if extra:
                raise KeyError(
                    f"sub_batch for adapter {name!r} has extra keys not in "
                    f"the first sub_batch: {sorted(extra)}"
                )

        merged: dict[str, Any] = {}
        for k, v0 in first.items():
            if isinstance(v0, list):
                out: list = []
                for _, b in sub_batches:
                    out.extend(b[k])
                merged[k] = out
            elif isinstance(v0, torch.Tensor):
                merged[k] = torch.cat([b[k] for _, b in sub_batches], dim=0)
            else:
                raise TypeError(
                    f"unsupported per-adapter batch value type for key {k!r}: "
                    f"{type(v0).__name__}"
                )

        adapter_names_per_row: list[str] = []
        for name, b in sub_batches:
            adapter_names_per_row.extend([name] * MultiAdapterDataLoader._row_count(b))
        merged["adapter_names"] = adapter_names_per_row
        # Canonical per-row id tensor for the routing hook. Order matches
        # the sub_batches order, which is the canonical adapter order this
        # loader was constructed with (`self._adapter_names`). Each row's id
        # is its index into that list — so it's the GLOBAL id consumed by
        # MultiLinearLoRA.lora_A[ids] / .lora_B[ids].
        global_id_by_name = {n: i for i, (n, _) in enumerate(sub_batches)}
        merged["adapter_ids"] = torch.tensor(
            [global_id_by_name[n] for n in adapter_names_per_row], dtype=torch.long
        )
        return merged

    @staticmethod
    def _row_count(batch: dict[str, Any]) -> int:
        """Row count of a single-adapter batch: ``message_log`` length if
        present (NeMo-RL SFT convention), else any tensor's leading dim."""
        msgs = batch.get("message_log")
        if isinstance(msgs, list):
            return len(msgs)
        for v in batch.values():
            if isinstance(v, torch.Tensor):
                return v.shape[0]
        raise ValueError(
            "per-adapter batch has neither message_log list nor any tensor"
        )

    @staticmethod
    def _merge_data_cfg(base_data_cfg: dict | None, adapter_data: dict | None) -> dict:
        """Merge precedence: per-adapter ``data`` OVERRIDES ``base_data_cfg``.

        Shallow merge — top-level keys from the adapter dict win. Mirror of
        ``{**base, **adapter}`` but extracted as a named method so the
        contract is testable.
        """
        return {**(base_data_cfg or {}), **(adapter_data or {})}

    @staticmethod
    def _build_one(
        adapter: Any,
        base_data_cfg: dict,
        tokenizer: Any,
        max_seq_length: int,
        batch_size: int,
    ) -> Any:
        """Run NeMo-RL's stock SFT data chain for one adapter."""
        from nemo_rl.data.collate_fn import rl_collate_fn
        from nemo_rl.data.datasets import AllTaskProcessedDataset, load_response_dataset
        from nemo_rl.data.datasets.utils import update_single_dataset_config
        from nemo_rl.data.interfaces import TaskDataSpec
        from nemo_rl.data.processors import sft_processor
        from torchdata.stateful_dataloader import StatefulDataLoader

        merged_data_cfg = MultiAdapterDataLoader._merge_data_cfg(base_data_cfg, adapter.data)

        default_task_spec = TaskDataSpec(
            task_name=f"sft_{adapter.name}",
            prompt_file=merged_data_cfg.get("default", {}).get("prompt_file"),
            system_prompt_file=merged_data_cfg.get("default", {}).get("system_prompt_file"),
        )
        task_data_processors = defaultdict(lambda: (default_task_spec, sft_processor))

        # Merge `data.default` into the per-split config before
        # `load_response_dataset`. Without this `RawDataset.set_processor`
        # silently falls back to the math processor and drops the assistant
        # turn.
        train_cfg = dict(merged_data_cfg["train"])
        if merged_data_cfg.get("default"):
            update_single_dataset_config(train_cfg, merged_data_cfg["default"])
        train_data = load_response_dataset(train_cfg)
        task_data_processors[train_data.task_name] = (
            train_data.task_spec,
            train_data.processor,
        )

        processed = AllTaskProcessedDataset(
            train_data.dataset,
            tokenizer,
            default_task_spec,
            task_data_processors,
            max_seq_length=max_seq_length,
        )

        # Honor `data.shuffle` from the merged config (defaults to True to
        # preserve historical behavior). For bit-equivalence runs the user
        # sets `data.shuffle: false` so per-adapter row order is deterministic
        # and matches a single-LoRA reference run on the same train file.
        shuffle_flag = bool(merged_data_cfg.get("shuffle", True))
        loader = StatefulDataLoader(
            processed,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            collate_fn=rl_collate_fn,
            drop_last=True,
        )
        logger.info(
            "built dataloader for adapter %s with %d examples (bspa=%d)",
            adapter.name,
            len(processed),
            batch_size,
        )
        return loader
