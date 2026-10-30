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

from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional

import torch
from nemo_automodel.components.distributed.pipelining.autopipeline import (
    AutoPipeline,
)

# Yields this rank's pipeline stage as its (Hugging Face name, full tensor) stream.
LocalParamsFn = Callable[[], Iterator[tuple[str, torch.Tensor]]]


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
    without pipeline parallelism, but a pipeline rank holds only its stage.
    Each tensor is produced by the ranks of the stage that owns it (the same
    per-module stream the non-pipelined refit uses: full tensors, Hugging Face
    names, refit dtypes) and broadcast within the pipeline group. Under 1F1B
    PP rank ``i`` owns stage ``i``, so the stream visits the stages in PP rank
    order, an order every rank agrees on. The stream does not assume its
    consumer, so colocated CUDA IPC and NCCL broadcast refits both iterate it.
    """

    def __init__(self, pipeline: AutoPipeline, *, local_params: LocalParamsFn):
        """Initialize the exporter.

        Args:
            pipeline: The AutoPipeline that drives this rank's stage.
            local_params: Yields the refit stream of this rank's stage.
        """
        self._local_params = local_params
        self._pp_group = pipeline.pp_mesh.get_group()
        self._manifest: Optional[list[_ManifestEntry]] = None

    def _build_manifest(self) -> list[_ManifestEntry]:
        """List every stage's tensors, in PP rank order."""
        local_entries = [
            (name, tuple(tensor.shape), tensor.dtype)
            for name, tensor in self._local_params()
        ]
        # all_gather_object fills every slot with one PP rank's entries.
        gathered: list[Any] = [None] * torch.distributed.get_world_size(self._pp_group)
        torch.distributed.all_gather_object(
            gathered, local_entries, group=self._pp_group
        )

        manifest = []
        producing_rank: dict[str, int] = {}
        for owner_pp_rank, entries in enumerate(gathered):
            for name, shape, dtype in entries:
                if name in producing_rank:
                    raise RuntimeError(
                        f"Weight {name!r} is produced by the pipeline stages of "
                        f"PP ranks {producing_rank[name]} and {owner_pp_rank}."
                    )
                producing_rank[name] = owner_pp_rank
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
        local_stream = self._local_params()
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
