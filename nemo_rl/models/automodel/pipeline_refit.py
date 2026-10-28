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

"""Full-model weight stream for a pipeline-parallel Automodel policy."""

import itertools
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

import torch
from torch import nn

from nemo_rl.models.automodel.model_handle import ModelRuntime

# Maps a local model part to its (Hugging Face name, full tensor) stream.
LocalParamsFn = Callable[[nn.Module], Iterator[tuple[str, torch.Tensor]]]


@dataclass(frozen=True)
class _ManifestEntry:
    """One tensor of the full-model stream."""

    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    # Rank within the pipeline group whose stage produces the tensor.
    owner_pp_rank: int


class PipelineWeightExporter:
    """Produce the whole model's ``(name, tensor)`` stream on every PP rank.

    Refit consumers expect every training rank to stream the complete model, as
    without pipeline parallelism, but a pipeline rank holds only its stages.
    Each tensor is produced by the ranks of the stage that owns it (the same
    per-module stream the non-pipelined refit uses: full tensors, Hugging Face
    names, refit dtypes) and broadcast within the pipeline group, in one global
    order every rank agrees on. The stream does not assume its consumer, so
    colocated CUDA IPC and NCCL broadcast refits both iterate it.
    """

    def __init__(self, runtime: ModelRuntime, *, local_params: LocalParamsFn):
        """Initialize the exporter.

        Args:
            runtime: The worker's pipeline model parts.
            local_params: Yields the refit stream of one local model part.
        """
        self._runtime = runtime
        self._local_params = local_params
        self._pp_group = runtime.pipeline.pp_mesh.get_group()
        self._manifest: Optional[list[_ManifestEntry]] = None

    def _local_parts_in_stage_order(self) -> list[nn.Module]:
        return [
            part
            for _, part in sorted(
                zip(self._runtime.stage_ids, self._runtime.parts),
                key=lambda stage_and_part: stage_and_part[0],
            )
        ]

    def _build_manifest(self) -> list[_ManifestEntry]:
        """List every stage's tensors, merged in global stage order."""
        pp_rank = torch.distributed.get_rank(self._pp_group)
        local_stages = {
            stage_id: [
                (name, tuple(tensor.shape), tensor.dtype)
                for name, tensor in self._local_params(part)
            ]
            for stage_id, part in zip(self._runtime.stage_ids, self._runtime.parts)
        }
        gathered: list[Optional[tuple[int, dict]]] = [None] * (
            torch.distributed.get_world_size(self._pp_group)
        )
        torch.distributed.all_gather_object(
            gathered, (pp_rank, local_stages), group=self._pp_group
        )

        stages: dict[int, tuple[int, list]] = {}
        for owner_pp_rank, rank_stages in gathered:
            for stage_id, entries in rank_stages.items():
                if stage_id in stages:
                    raise RuntimeError(
                        f"Pipeline stage {stage_id} is owned by PP ranks "
                        f"{stages[stage_id][0]} and {owner_pp_rank}."
                    )
                stages[stage_id] = (owner_pp_rank, entries)
        num_stages = self._runtime.pipeline.info.stages[0].num_stages
        if sorted(stages) != list(range(num_stages)):
            raise RuntimeError(
                f"Pipeline weight manifest covers stages {sorted(stages)}, "
                f"expected 0..{num_stages - 1}."
            )

        manifest = []
        producing_stage: dict[str, int] = {}
        for stage_id in range(num_stages):
            owner_pp_rank, entries = stages[stage_id]
            for name, shape, dtype in entries:
                if name in producing_stage:
                    raise RuntimeError(
                        f"Weight {name!r} is produced by pipeline stages "
                        f"{producing_stage[name]} and {stage_id}."
                    )
                producing_stage[name] = stage_id
                manifest.append(_ManifestEntry(name, shape, dtype, owner_pp_rank))
        return manifest

    def manifest(self) -> list[_ManifestEntry]:
        """The full model's tensors in stream order; built once, collectively."""
        if self._manifest is None:
            self._manifest = self._build_manifest()
        return self._manifest

    def refit_info(self) -> dict[str, tuple[torch.Size, torch.dtype]]:
        """Shape and dtype of every streamed tensor, as ``prepare_refit_info``."""
        return {
            entry.name: (torch.Size(entry.shape), entry.dtype)
            for entry in self.manifest()
        }

    def iter_params(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield the full model's ``(name, tensor)`` pairs on every PP rank.

        Collective over the pipeline group (and, through the per-module stream,
        over each stage's data-parallel group): every rank must consume the
        whole stream.
        """
        manifest = self.manifest()
        pp_rank = torch.distributed.get_rank(self._pp_group)
        local_stream = itertools.chain.from_iterable(
            self._local_params(part) for part in self._local_parts_in_stage_order()
        )
        for entry in manifest:
            if entry.owner_pp_rank == pp_rank:
                name, tensor = next(local_stream)
                if name != entry.name or tuple(tensor.shape) != entry.shape:
                    raise RuntimeError(
                        f"Pipeline weight stream produced {name!r} "
                        f"{tuple(tensor.shape)} where the manifest expects "
                        f"{entry.name!r} {entry.shape}."
                    )
            else:
                tensor = torch.empty(entry.shape, dtype=entry.dtype, device="cuda")
            torch.distributed.broadcast(
                tensor,
                src=torch.distributed.get_global_rank(
                    self._pp_group, entry.owner_pp_rank
                ),
                group=self._pp_group,
            )
            yield entry.name, tensor
