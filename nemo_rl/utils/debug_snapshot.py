# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from __future__ import annotations

import gzip
import os
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class DebugSnapshotSaver:
    """Async saver for debug snapshots of mult_prob_error batches."""

    def __init__(self, snapshot_dir: str):
        self.snapshot_dir = snapshot_dir
        self._executor: ThreadPoolExecutor | None = None

    def _get_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="snapshot")
        return self._executor

    def _clone_field(self, value: Any, batch_size: int) -> Any:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.cpu().clone()
        if hasattr(value, "tensors"):
            return [t.cpu().clone() if t is not None else None for t in value.tensors]
        if hasattr(value, "get_element"):
            return [value.get_element(i) for i in range(batch_size)]
        return value

    def _write_to_disk(self, snapshot: dict, filepath: str, batch_size: int) -> None:
        try:
            with gzip.open(filepath, "wb", compresslevel=6) as f:
                torch.save(snapshot, f)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  💾 Saved debug snapshot to {filepath} ({size_mb:.1f} MB, batch_size={batch_size})", flush=True)
        except Exception as e:
            print(f"  ⚠️ Failed to save debug snapshot {filepath}: {e}", flush=True)

    def save(
        self,
        step: int,
        seq_mult_prob_error: torch.Tensor,
        lp_error: torch.Tensor,
        train_data: "BatchedDataDict[Any]",
        prompt_lengths: torch.Tensor,
        vllm_images: list[list[str]] | None = None,
    ) -> None:
        """Save a debug snapshot asynchronously.
        
        Saves the whole batch because MoE expert routing is batch-dependent (tokens
        compete for expert capacity across the batch). Single-sequence snapshots
        would get different routing decisions when replayed.
        """
        bad_seq_idx = seq_mult_prob_error.argmax().item()
        error_value = seq_mult_prob_error[bad_seq_idx].item()

        bad_lp_error = lp_error[bad_seq_idx]
        max_error_token_idx = bad_lp_error.argmax().item()

        input_ids = train_data["input_ids"]
        max_error_token_id = input_ids[bad_seq_idx, max_error_token_idx + 1].item()
        batch_size = input_ids.shape[0]

        snapshot = {
            "input_ids": input_ids.cpu().clone(),
            "imgs_sizes": self._clone_field(train_data.get("imgs_sizes"), batch_size),
            "token_type_ids": self._clone_field(train_data.get("token_type_ids"), batch_size),
            "image_paths": vllm_images,
            "prompt_lengths": prompt_lengths.cpu().clone(),
            "generation_logprobs": train_data["generation_logprobs"].cpu().clone(),
            "prev_logprobs": train_data["prev_logprobs"].cpu().clone(),
            "step": step,
            "batch_size": batch_size,
            "seq_mult_prob_error": seq_mult_prob_error.cpu().clone(),
            "bad_seq_idx": int(bad_seq_idx),
            "max_error_value": error_value,
            "max_error_token_idx": int(max_error_token_idx),
            "max_error_token_id": int(max_error_token_id),
        }

        os.makedirs(self.snapshot_dir, exist_ok=True)
        filepath = os.path.join(self.snapshot_dir, f"step_{step}_error_{error_value:.2e}.pt.gz")

        print(f"  📸 Queuing debug snapshot save (step={step}, error={error_value:.2e})...", flush=True)
        self._get_executor().submit(self._write_to_disk, snapshot, filepath, batch_size)
