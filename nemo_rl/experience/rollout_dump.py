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
"""Sampled per-group dump of finalized rollouts for a rollout viewer."""

import gzip
import json
import os
import socket
from typing import Any, Optional


def load_dump_config(cfg: Optional[dict[str, Any]]) -> Optional["RolloutDumper"]:
    if not cfg or not cfg.get("enabled", False):
        return None
    return RolloutDumper(cfg)


def _gen_nll(row: Any) -> Optional[float]:
    n = sum(1 for m in row.token_mask if m > 0)
    if not n:
        return None
    return -sum(lp for lp, m in zip(row.logprobs, row.token_mask) if m > 0) / n


class RolloutDumper:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self._dir = cfg["dir"]
        self._full_rows = int(cfg.get("full_rows_per_group", 4))
        self._tokenizer_path = cfg["tokenizer"]
        self._tokenizer = None
        self._file = f"{socket.gethostname()}-{os.getpid()}.jsonl.gz"

    def _segments(self, ids: list[int], mask: list[float]) -> list[list[Any]]:
        # Runs of [trained, text]; trained text is what the policy generated.
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self._tokenizer_path, trust_remote_code=True
            )
        out: list[list[Any]] = []
        start = 0
        for i in range(1, len(ids) + 1):
            if i == len(ids) or (mask[i] > 0) != (mask[start] > 0):
                out.append(
                    [bool(mask[start] > 0), self._tokenizer.decode(ids[start:i])]
                )
                start = i
        return out

    def _pick(self, raw_rewards: list[float], valid: list[bool]) -> set[int]:
        # Best and worst valid rollout first, then the rest in order.
        order = [i for i in range(len(raw_rewards)) if valid[i]]
        picks = []
        if order:
            picks += [
                max(order, key=lambda i: raw_rewards[i]),
                min(order, key=lambda i: raw_rewards[i]),
            ]
        picks += order + [i for i in range(len(raw_rewards)) if not valid[i]]
        return set(list(dict.fromkeys(picks))[: self._full_rows])

    def dump(
        self,
        *,
        version: int,
        agent_name: Optional[str],
        group_id: str,
        prompt_idx: int,
        rows: list[Any],
        raw_rewards: list[float],
        mask_sample: list[bool],
    ) -> None:
        full = self._pick(raw_rewards, [r.valid for r in rows])
        prompt = None
        rollouts = []
        for i, row in enumerate(rows):
            entry: dict[str, Any] = {
                "i": i,
                "raw_reward": raw_rewards[i],
                "reward": row.reward,
                "valid": row.valid,
                "rejection": row.rejection_reason,
                "masked": bool(mask_sample[i]),
                "tokens": len(row.token_ids),
                "gen_tokens": int(sum(row.token_mask)),
                # Sampler-side mean NLL of generated tokens, a per-env entropy proxy.
                "gen_nll": _gen_nll(row),
            }
            if i in full and row.valid:
                if prompt is None:
                    prompt = self._segments(
                        row.token_ids[: row.prompt_len],
                        row.token_mask[: row.prompt_len],
                    )
                entry["segments"] = self._segments(
                    row.token_ids[row.prompt_len :], row.token_mask[row.prompt_len :]
                )
            rollouts.append(entry)
        record = {
            "version": version,
            "agent": agent_name,
            "group_id": group_id,
            "prompt_idx": prompt_idx,
            "prompt": prompt,
            "rollouts": rollouts,
        }
        path = os.path.join(self._dir, f"v{version:04d}", self._file)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with gzip.open(path, "at") as f:
            f.write(json.dumps(record) + "\n")
