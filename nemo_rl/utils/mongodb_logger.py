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

"""MongoDB structured logging backend for NeMo-RL.

Captures per-step metrics into MongoDB for post-run analysis and
cross-backend comparison (TRT-LLM vs vLLM vs Megatron).

The logger accumulates metrics across multiple ``log_metrics()`` calls
within the same step (each with a different prefix like ``train/``,
``performance/``, ``timing/train/``) and flushes a single consolidated
document per step to the ``steps`` collection.  A companion ``runs``
collection holds run-level metadata and the full config snapshot.
"""

from __future__ import annotations

import atexit
import datetime
import uuid
from typing import Any, Mapping, NotRequired, Optional, TypedDict

import numpy as np
import torch

try:
    import pymongo
    from pymongo import MongoClient

    _PYMONGO_AVAILABLE = True
except ImportError:
    _PYMONGO_AVAILABLE = False


class MongoDBLoggerConfig(TypedDict):
    uri: str
    database: NotRequired[str]
    db_path: NotRequired[str]


class MongoDBLogger:
    """MongoDB logger backend with per-step document accumulation.

    Implements the same public surface as ``LoggerInterface`` but is *not*
    a formal subclass to keep the ``pymongo`` dependency optional.  The
    ``Logger`` dispatcher calls its methods through duck-typing.
    """

    def __init__(
        self,
        cfg: MongoDBLoggerConfig,
        run_metadata: dict[str, Any] | None = None,
    ) -> None:
        if not _PYMONGO_AVAILABLE:
            print(
                "WARNING: pymongo is not installed. MongoDB logging disabled. "
                "Install with: pip install pymongo"
            )
            self._enabled = False
            return

        uri = cfg.get("uri", "mongodb://127.0.0.1:27017")
        db_name = cfg.get("database", "nemo_rl")

        try:
            self._client: MongoClient = MongoClient(
                uri, serverSelectionTimeoutMS=5000
            )
            self._client.admin.command("ping")
        except Exception as e:
            print(f"WARNING: Cannot connect to MongoDB at {uri}: {e}. Logging disabled.")
            self._enabled = False
            return

        self._enabled = True
        self._db = self._client[db_name]
        self._runs = self._db["runs"]
        self._steps = self._db["steps"]

        self._steps.create_index([("run_id", pymongo.ASCENDING), ("step", pymongo.ASCENDING)])
        self._steps.create_index([("run_id", pymongo.ASCENDING), ("wall_time", pymongo.ASCENDING)])

        self._run_id = str(uuid.uuid4())
        self._current_step: int | None = None
        self._step_buffer: dict[str, Any] = {}
        self._pending_docs: list[dict[str, Any]] = []
        self._flush_interval = 5

        run_doc: dict[str, Any] = {
            "run_id": self._run_id,
            "started_at": datetime.datetime.now(datetime.timezone.utc),
            "finished_at": None,
            "status": "running",
        }
        if run_metadata:
            run_doc.update(run_metadata)
        self._runs.insert_one(run_doc)

        atexit.register(self.close)
        print(f"Initialized MongoDBLogger: db={db_name}, run_id={self._run_id}")

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def run_id(self) -> str:
        return self._run_id if self._enabled else ""

    def _coerce_value(self, value: Any) -> Any:
        """Convert tensors / numpy types to JSON-safe Python scalars."""
        if isinstance(value, torch.Tensor):
            if value.ndim == 0 or value.numel() == 1:
                return value.item()
            return value.tolist()
        if isinstance(value, (np.floating, np.integer, np.bool_)):
            return value.item()
        if isinstance(value, np.ndarray):
            if value.ndim == 0 or value.size == 1:
                return value.item()
            return value.tolist()
        if isinstance(value, (dict, list)):
            return None
        return value

    def _flush_step(self) -> None:
        """Flush the accumulated step buffer as a document."""
        if not self._step_buffer:
            return
        doc = {
            "run_id": self._run_id,
            "step": self._current_step,
            "wall_time": datetime.datetime.now(datetime.timezone.utc),
        }
        doc.update(self._step_buffer)
        self._pending_docs.append(doc)
        self._step_buffer = {}

        if len(self._pending_docs) >= self._flush_interval:
            self._write_pending()

    def _write_pending(self) -> None:
        if not self._pending_docs:
            return
        try:
            self._steps.insert_many(self._pending_docs, ordered=False)
        except Exception as e:
            print(f"WARNING: MongoDB bulk write failed: {e}")
        self._pending_docs = []

    def log_metrics(
        self,
        metrics: dict[str, Any],
        step: int,
        prefix: Optional[str] = "",
        step_metric: Optional[str] = None,
    ) -> None:
        if not self._enabled:
            return

        if self._current_step is not None and step != self._current_step:
            self._flush_step()

        self._current_step = step

        for name, value in metrics.items():
            coerced = self._coerce_value(value)
            if coerced is None:
                continue
            key = f"{prefix}/{name}" if prefix else name
            # MongoDB keys cannot contain dots
            key = key.replace(".", "_")
            self._step_buffer[key] = coerced

    def log_hyperparams(self, params: Mapping[str, Any]) -> None:
        if not self._enabled:
            return
        safe_params = {}
        for k, v in _flatten_dict(params).items():
            safe_key = k.replace(".", "_")
            safe_params[safe_key] = v
        try:
            self._runs.update_one(
                {"run_id": self._run_id},
                {"$set": {"config": safe_params}},
            )
        except Exception as e:
            print(f"WARNING: MongoDB hyperparams write failed: {e}")

    def log_rollouts(
        self,
        message_logs: list[list[dict[str, Any]]],
        rewards: list[float],
        step: int,
    ) -> None:
        """Store raw rollout conversations (prompt + response text) with rewards.

        Each document in the ``rollouts`` collection represents one sample:
        its conversation turns (role + content text only, no tensors) and
        the scalar reward.  Token-level data is intentionally omitted to
        keep document size manageable.
        """
        if not self._enabled:
            return

        if "rollouts" not in self.__dict__:
            self._rollouts = self._db["rollouts"]
            self._rollouts.create_index(
                [("run_id", 1), ("step", 1)],
            )

        docs = []
        for i, (msg_log, reward) in enumerate(zip(message_logs, rewards)):
            turns = []
            for msg in msg_log:
                turn = {"role": msg.get("role", "")}
                content = msg.get("content")
                if content is not None:
                    turn["content"] = str(content)
                turns.append(turn)
            doc: dict[str, Any] = {
                "run_id": self._run_id,
                "step": step,
                "sample_idx": i,
                "reward": float(reward),
                "turns": turns,
            }
            for msg in msg_log:
                if msg.get("role") == "assistant":
                    token_ids = msg.get("token_ids")
                    if token_ids is not None:
                        import torch
                        doc["token_ids"] = (
                            token_ids.tolist()
                            if isinstance(token_ids, torch.Tensor)
                            else list(token_ids)
                        )
                    origins = msg.get("spec_token_origins")
                    if origins is not None:
                        import torch
                        doc["spec_token_origins"] = (
                            origins.tolist()
                            if isinstance(origins, torch.Tensor)
                            else list(origins)
                        )
                    break
            docs.append(doc)

        if docs:
            try:
                self._rollouts.insert_many(docs, ordered=False)
            except Exception as e:
                print(f"WARNING: MongoDB rollout write failed: {e}")

    def log_histogram(self, histogram: list[Any], step: int, name: str) -> None:
        pass

    def log_plot(self, figure: Any, step: int, name: str) -> None:
        pass

    def close(self) -> None:
        """Flush remaining data and mark the run as completed."""
        if not self._enabled:
            return
        self._flush_step()
        self._write_pending()
        try:
            self._runs.update_one(
                {"run_id": self._run_id},
                {
                    "$set": {
                        "finished_at": datetime.datetime.now(datetime.timezone.utc),
                        "status": "completed",
                    }
                },
            )
        except Exception as e:
            print(f"WARNING: MongoDB run finalization failed: {e}")
        try:
            self._client.close()
        except Exception:
            pass
        self._enabled = False
        print(f"MongoDBLogger closed (run_id={self._run_id})")


def _flatten_dict(d: Mapping[str, Any], sep: str = "/") -> dict[str, Any]:
    result: dict[str, Any] = {}

    def _walk(obj: Mapping[str, Any], parent: str = "") -> None:
        for key, value in obj.items():
            full = f"{parent}{sep}{key}" if parent else key
            if isinstance(value, dict):
                _walk(value, full)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        _walk(item, f"{full}{sep}{i}")
                    else:
                        result[f"{full}{sep}{i}"] = item
            else:
                result[full] = value

    _walk(d)
    return result
