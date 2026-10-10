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

"""HF-checkpoint weight synchronization for an external stock vLLM server."""

import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional

from nemo_rl.utils.timer import Timer
from nemo_rl.weight_sync.interfaces import WeightSynchronizer


class RemoteVllmCheckpointWeightSynchronizer(WeightSynchronizer):
    """Export live Megatron weights and globally reload an external vLLM engine."""

    def __init__(self, policy: Any, generation: Any) -> None:
        self._policy = policy
        self._generation = generation
        self._checkpoint_root = Path(
            generation.remote_config.refit.checkpoint_dir
        ).expanduser()
        self._next_version = 1
        self._stale = True

    @property
    def is_stale(self) -> bool:
        return self._stale

    def init_communicator(self) -> None:
        self._checkpoint_root.mkdir(parents=True, exist_ok=True)
        existing_versions = []
        for path in self._checkpoint_root.iterdir():
            name = path.name
            if name.startswith(".version_") and name.endswith(".tmp"):
                name = name.removeprefix(".").removesuffix(".tmp")
            elif not name.startswith("version_"):
                continue
            try:
                existing_versions.append(int(name.removeprefix("version_")))
            except ValueError:
                continue
        if existing_versions:
            self._next_version = max(existing_versions) + 1

    def sync_weights(
        self,
        *,
        timer: Optional[Timer] = None,
        kv_scales: Optional[dict[str, float]] = None,
    ) -> None:
        if kv_scales is not None:
            raise NotImplementedError(
                "remote_vllm checkpoint reload does not support separate KV scales"
            )

        self._stale = True
        version = self._next_version
        final_path = self._checkpoint_root / f"version_{version:08d}"
        staging_path = self._checkpoint_root / f".version_{version:08d}.tmp"
        if final_path.exists() or staging_path.exists():
            raise FileExistsError(
                "Refusing to overwrite an existing remote-vLLM export: "
                f"{final_path if final_path.exists() else staging_path}"
            )

        export_context = (
            timer.time("prepare_for_generation/export_hf_checkpoint")
            if timer is not None
            else nullcontext()
        )
        with export_context:
            self._policy.export_hf_checkpoint(str(staging_path))
            # Publish only after every policy rank has completed Bridge export.
            os.replace(staging_path, final_path)

        reload_context = (
            timer.time("prepare_for_generation/reload_remote_vllm")
            if timer is not None
            else nullcontext()
        )
        with reload_context:
            self._generation.pause_generation_for_refit(clear_cache=True)
            # If either call fails, execution never reaches resume. A failed
            # collective RPC may have updated only some workers, so serving
            # from that state would be unsafe.
            self._generation.reload_weights(str(final_path))
            self._generation.resume_generation_after_refit()

        self._next_version += 1
        self._stale = False

    def shutdown(self) -> None:
        return None
