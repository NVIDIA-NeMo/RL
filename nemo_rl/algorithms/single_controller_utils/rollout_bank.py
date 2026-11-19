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
"""On-disk bank of collected rollout rows for offline critic pretraining.

A ``rollout_collection`` run (generation only, usually one 1-node job per data
shard) writes every finalized prompt group here: the token-capture rows PPO
would train on, their raw flags (no mask folded in), and the prompt's
``extra_env_info`` so the privileged critic's reference block can be rebuilt
without the dataset.

Layout::

    bank_dir/
      bank_meta.json            what produced the bank; a second run must match
      step_000001/
        <group_id>.pt           one prompt group: every real row, unpadded
        manifest.json           written last: the step is committed
      orphaned/                 steps a resume discarded (moved, never deleted)

A collection over many shards is a directory of such banks (one per shard job);
:func:`find_banks` expands it.

A step without ``manifest.json`` is incomplete and is never read. Resuming a
collection run from checkpoint ``step_N`` moves every step directory past ``N``
under ``orphaned/``: the resumed run re-collects those steps from the same
dataloader position, so the bank always matches its checkpoint lineage.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import torch
from tensordict import TensorDict

from nemo_rl.data_plane.schema import ROUTE_PLAN_TAG

BANK_FORMAT = "nemo_rl.rollout_bank.v1"
BANK_META_FILENAME = "bank_meta.json"
MANIFEST_FILENAME = "manifest.json"
ORPHANED_DIRNAME = "orphaned"
_STEP_PREFIX = "step_"

# Per-token columns: stored as unpadded rows and put back jagged.
TOKEN_FIELDS = (
    "input_ids",
    "token_mask",
    "generation_logprobs",
)
# Per-row columns: stored row by row and put back stacked.
ROW_FIELDS = (
    "input_lengths",
    "sample_mask",
    "mask_sample",
    "truncated",
    "total_reward",
)
BANK_FIELDS = TOKEN_FIELDS + ROW_FIELDS
# Rows cannot train the critic without these.
REQUIRED_BANK_FIELDS = (
    "input_ids",
    "token_mask",
    "input_lengths",
    "sample_mask",
    "mask_sample",
    "truncated",
    "total_reward",
)


def step_dirname(step: int) -> str:
    """``step_000042``: zero-padded so lexical order is step order."""
    return f"{_STEP_PREFIX}{step:06d}"


def _step_of(path: Path) -> Optional[int]:
    name = path.name
    if not name.startswith(_STEP_PREFIX):
        return None
    try:
        return int(name[len(_STEP_PREFIX) :])
    except ValueError:
        return None


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_write_json(path: Path, obj: Any) -> None:
    _atomic_write_bytes(
        path, json.dumps(obj, indent=2, sort_keys=True, default=str).encode()
    )


def _atomic_torch_save(path: Path, obj: Any) -> None:
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _safe_filename(group_id: str) -> str:
    """A group id as a file name; hashed when it carries path characters."""
    if group_id and all(c.isalnum() or c in "-_." for c in group_id):
        return group_id
    return hashlib.sha1(group_id.encode()).hexdigest()


def _compact(t: torch.Tensor) -> torch.Tensor:
    """int64 -> int32 when lossless (token ids, masks); the dtype is restored on load."""
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.int64 and t.numel() > 0:
        info = torch.iinfo(torch.int32)
        if int(t.min()) >= info.min and int(t.max()) <= info.max:
            return t.to(torch.int32)
    return t


def _dtype_from_name(name: str) -> torch.dtype:
    dtype = getattr(torch, name.removeprefix("torch."), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unknown dtype {name!r} in rollout bank")
    return dtype


def _clean_tag(tag: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """A row tag without the staging route plan, which means nothing offline."""
    return {k: v for k, v in (tag or {}).items() if k != ROUTE_PLAN_TAG}


def row_tensors(
    data: TensorDict, field_name: str, input_lengths: torch.Tensor
) -> list[torch.Tensor]:
    """Split a fetched column into one tensor per row.

    Per-token columns come back jagged (or padded); either way each row is cut to
    its own ``input_lengths`` so rows from different steps can be re-batched.
    """
    value = data[field_name]
    if field_name in TOKEN_FIELDS:
        if value.is_nested:
            rows = list(value.unbind())
        else:
            rows = list(value.unbind(0))
        return [row[: int(n)] for row, n in zip(rows, input_lengths.tolist())]
    if value.is_nested:
        raise ValueError(f"per-row field {field_name!r} came back jagged")
    return list(value.unbind(0))


def bank_identity(master_config: Any) -> dict[str, Any]:
    """The settings two collection runs must share to write into one bank.

    Rows from a different model, tokenizer, dataset or masking setup would be
    silently mixed into one pretraining set otherwise.
    """
    algo = master_config.ppo
    token_capture = master_config.token_capture
    policy = master_config.policy
    return {
        "format": BANK_FORMAT,
        "policy_model_name": policy["model_name"],
        "tokenizer": dict(policy["tokenizer"]),
        "max_total_sequence_length": policy["max_total_sequence_length"],
        "data": json.loads(json.dumps(dict(master_config.data), default=str)),
        "num_generations_per_prompt": algo.num_generations_per_prompt,
        "segment_rows": json.loads(
            json.dumps(
                token_capture.segment_rows.model_dump()
                if token_capture.enabled
                else None,
                default=str,
            )
        ),
    }


class RolloutBankWriter:
    """Writes a collection run's prompt groups, one committed step at a time."""

    def __init__(
        self,
        bank_dir: str | os.PathLike,
        identity: dict[str, Any],
        *,
        restored_step: int,
    ) -> None:
        """Open (or create) the bank and align it with the restored checkpoint.

        Args:
            bank_dir: Bank root.
            identity: :func:`bank_identity` of this run.
            restored_step: Steps the restored checkpoint completed (0 when fresh).

        Raises:
            ValueError: The bank was written by a run with a different identity,
                or this run starts fresh on a bank that already has steps.
        """
        self.root = Path(bank_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        meta_path = self.root / BANK_META_FILENAME
        if meta_path.exists():
            existing = json.loads(meta_path.read_text())
            mismatched = sorted(
                key
                for key in set(existing) | set(identity)
                if key != "created_at" and existing.get(key) != identity.get(key)
            )
            if mismatched:
                raise ValueError(
                    f"rollout bank {self.root} was written by a run with different "
                    f"settings ({mismatched}); point rollout_collection.bank_dir at "
                    "a new directory."
                )
        else:
            _atomic_write_json(meta_path, {**identity, "created_at": time.time()})
        self._reconcile(restored_step)
        self._pending: dict[int, list[dict[str, Any]]] = {}

    def _reconcile(self, restored_step: int) -> None:
        later = sorted(
            (step, path)
            for path in self.root.iterdir()
            if path.is_dir() and (step := _step_of(path)) is not None
            if step > restored_step or not (path / MANIFEST_FILENAME).exists()
        )
        if restored_step == 0 and any(
            (path / MANIFEST_FILENAME).exists() for _, path in later
        ):
            raise ValueError(
                f"rollout bank {self.root} already holds committed steps but this "
                "run starts from scratch; point rollout_collection.bank_dir at a new "
                "directory, or resume the run that wrote it."
            )
        if not later:
            return
        orphaned = self.root / ORPHANED_DIRNAME
        orphaned.mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%S")
        for step, path in later:
            target = orphaned / f"{path.name}.{stamp}"
            os.replace(path, target)
            print(
                f"  rollout bank: step {step} is past the restored checkpoint "
                f"(step {restored_step}) or uncommitted; moved to {target}",
                flush=True,
            )

    def write_group(
        self,
        *,
        step: int,
        group_id: str,
        prompt_idx: Optional[int],
        instance_id: Optional[str],
        extra_env_info: Mapping[str, Any],
        sample_ids: Sequence[str],
        tags: Sequence[Optional[Mapping[str, Any]]],
        sequence_lengths: Sequence[int],
        fields: Mapping[str, Sequence[torch.Tensor]],
    ) -> None:
        """Write one prompt group's real rows into ``step``'s directory.

        Args:
            step: 1-indexed training step that consumed the group.
            group_id: Prompt-group id (``{group}`` of ``{group}_g{i}[_t{j}]``).
            prompt_idx: Dataset index of the prompt, from the row tags.
            instance_id: Task identity resolved from the dataset row, if any.
            extra_env_info: The prompt's env info (reference block source).
            sample_ids: The group's row ids, pad rows excluded.
            tags: Row tags, aligned with ``sample_ids``.
            sequence_lengths: Row lengths, aligned with ``sample_ids``.
            fields: Column name -> one tensor per row (see :func:`row_tensors`).
        """
        missing = [name for name in REQUIRED_BANK_FIELDS if name not in fields]
        if missing:
            raise ValueError(f"rollout bank group {group_id} lacks {missing}")
        step_dir = self.root / step_dirname(step)
        step_dir.mkdir(exist_ok=True)
        filename = f"{_safe_filename(group_id)}.pt"
        record = {
            "format": BANK_FORMAT,
            "step": step,
            "group_id": group_id,
            "prompt_idx": prompt_idx,
            "instance_id": instance_id,
            "extra_env_info": dict(extra_env_info),
            "sample_ids": list(sample_ids),
            "tags": [_clean_tag(tag) for tag in tags],
            "sequence_lengths": [int(n) for n in sequence_lengths],
            "fields": {
                name: {
                    "dtype": str(rows[0].dtype) if rows else "torch.float32",
                    "rows": [_compact(row) for row in rows],
                }
                for name, rows in fields.items()
            },
        }
        _atomic_torch_save(step_dir / filename, record)
        rewards = torch.stack(
            [r.reshape(-1)[0].float() for r in fields["total_reward"]]
        )
        canonical = [
            int((tag or {}).get("trace_in_rollout_idx", 0)) == 0 for tag in tags
        ]
        self._pending.setdefault(step, []).append(
            {
                "file": filename,
                "group_id": group_id,
                "prompt_idx": prompt_idx,
                "instance_id": instance_id,
                "rows": len(sample_ids),
                "rollouts": int(sum(canonical)),
                "reward_mean": float(rewards[torch.tensor(canonical)].mean())
                if any(canonical)
                else float(rewards.mean()),
                "tokens": int(sum(int(n) for n in sequence_lengths)),
            }
        )

    def commit_step(self, step: int, *, trainer_version: int) -> int:
        """Write ``step``'s manifest; the step is then visible to readers.

        Returns:
            The number of groups committed.
        """
        groups = self._pending.pop(step, [])
        step_dir = self.root / step_dirname(step)
        step_dir.mkdir(exist_ok=True)
        _atomic_write_json(
            step_dir / MANIFEST_FILENAME,
            {
                "format": BANK_FORMAT,
                "step": step,
                "trainer_version": trainer_version,
                "groups": groups,
                "committed_at": time.time(),
            },
        )
        return len(groups)


# ── reader side ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BankGroupRef:
    """One committed prompt group, from its step's manifest."""

    path: Path
    step: int
    group_id: str
    prompt_idx: Optional[int]
    instance_id: Optional[str]
    rows: int
    rollouts: int
    reward_mean: float

    @property
    def task_key(self) -> str:
        """What a held-out split keys on: the task, so repeats never straddle it."""
        if self.instance_id:
            return f"instance:{self.instance_id}"
        return f"prompt:{self.prompt_idx}"


def read_bank_meta(bank_dir: str | os.PathLike) -> dict[str, Any]:
    """The bank's ``bank_meta.json``."""
    path = Path(bank_dir) / BANK_META_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: not a rollout bank")
    return json.loads(path.read_text())


def find_banks(paths: Iterable[str | os.PathLike]) -> list[Path]:
    """Expand each path to the banks it holds: itself, or the banks below it.

    A sharded collection writes one bank per shard job, so a collection root is
    searched (sorted, up to three levels down) for directories holding
    ``bank_meta.json``.
    """
    banks: list[Path] = []
    for path in map(Path, paths):
        if (path / BANK_META_FILENAME).is_file():
            banks.append(path)
            continue
        found = sorted(
            meta.parent
            for pattern in (
                "*/" + BANK_META_FILENAME,
                "*/*/" + BANK_META_FILENAME,
                "*/*/*/" + BANK_META_FILENAME,
            )
            for meta in path.glob(pattern)
        )
        if not found:
            raise FileNotFoundError(
                f"no rollout bank (bank_meta.json) in or under {path}"
            )
        banks.extend(found)
    return list(dict.fromkeys(banks))


def list_bank_groups(bank_dirs: Iterable[str | os.PathLike]) -> list[BankGroupRef]:
    """Every committed group of every bank, in (bank, step, manifest) order."""
    refs: list[BankGroupRef] = []
    for bank_dir in bank_dirs:
        root = Path(bank_dir)
        read_bank_meta(root)
        steps = sorted(
            (step, path)
            for path in root.iterdir()
            if path.is_dir() and (step := _step_of(path)) is not None
        )
        for step, path in steps:
            manifest_path = path / MANIFEST_FILENAME
            if not manifest_path.exists():
                continue
            manifest = json.loads(manifest_path.read_text())
            for group in manifest["groups"]:
                refs.append(
                    BankGroupRef(
                        path=path / group["file"],
                        step=step,
                        group_id=group["group_id"],
                        prompt_idx=group.get("prompt_idx"),
                        instance_id=group.get("instance_id"),
                        rows=int(group["rows"]),
                        rollouts=int(group["rollouts"]),
                        reward_mean=float(group["reward_mean"]),
                    )
                )
    return refs


@dataclass
class BankGroup:
    """A loaded prompt group: rows in their original dtypes."""

    group_id: str
    prompt_idx: Optional[int]
    instance_id: Optional[str]
    extra_env_info: dict[str, Any]
    sample_ids: list[str]
    tags: list[dict[str, Any]]
    sequence_lengths: list[int]
    fields: dict[str, list[torch.Tensor]]


def load_bank_group(path: str | os.PathLike) -> BankGroup:
    """Load one group file written by :class:`RolloutBankWriter`."""
    record = torch.load(path, map_location="cpu", weights_only=False)
    if record.get("format") != BANK_FORMAT:
        raise ValueError(f"{path}: not a {BANK_FORMAT} group file")
    fields = {
        name: [row.to(_dtype_from_name(entry["dtype"])) for row in entry["rows"]]
        for name, entry in record["fields"].items()
    }
    return BankGroup(
        group_id=record["group_id"],
        prompt_idx=record["prompt_idx"],
        instance_id=record["instance_id"],
        extra_env_info=dict(record["extra_env_info"]),
        sample_ids=list(record["sample_ids"]),
        tags=[dict(tag) for tag in record["tags"]],
        sequence_lengths=list(record["sequence_lengths"]),
        fields=fields,
    )


def batch_fields(
    rows_by_field: Mapping[str, Sequence[torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Re-batch per-row tensors for a data-plane put.

    Per-token columns become jagged nested tensors (the layout the token-capture
    finalizer writes); per-row columns are stacked.
    """
    batched: dict[str, torch.Tensor] = {}
    for name, rows in rows_by_field.items():
        if name in TOKEN_FIELDS:
            batched[name] = torch.nested.as_nested_tensor(
                [row.contiguous() for row in rows], layout=torch.jagged
            )
        else:
            batched[name] = torch.stack(list(rows))
    return batched
