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
"""Stable boundary between NeMo-RL and data-plane implementations.

Wire shape adapters must support:
  * ``fields``: ``TensorDict`` with tensor leaves AND optional
    ``NonTensorStack`` / ``NonTensorData`` leaves (TQ-native non-tensor
    passthrough). TQ's storage backends handle encoding per backend
    (simple keeps Python objects; mooncake_client pickles internally).
  * ``tags``: ``list[dict[str, Any]]`` per-sample primitives (kept
    separate from ``fields`` so non-tensor metadata like
    ``input_lengths`` doesn't pollute the leaf-level schema).
  * ``keys``: per-sample string uids.
  * ``partition_id``: string-named address spaces with declared
    ``consumer_tasks`` and ``fields`` schemas.

All call sites in ``nemo_rl/algorithms``, ``nemo_rl/experience`` and
``nemo_rl/models`` go through :class:`DataPlaneClient` — never
``import transfer_queue`` directly. This is what makes the
implementation swappable.

See ``nemo_rl/data_plane/README.md`` for the full design.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, NotRequired, Sequence, TypedDict

from pydantic import BaseModel, PositiveInt
from tensordict import TensorDict


class SimpleStorageConfig(BaseModel, extra="allow"):
    """Sizing for ``backend="simple"``. Ignored by every other backend.

    ``num_storage_units`` scales with the cluster: TQ round-robins storage
    units over Ray nodes and recommends ``>= 2 x`` the node count. No static
    default is correct across cluster sizes, so this is required rather than
    defaulted — a class field cannot see ``cluster.num_nodes``, only the
    exemplar YAML can, via ``${mul:2, ${cluster.num_nodes}}``. Every recipe
    inherits that from the exemplar; set a plain int to pin it.
    """

    storage_capacity: int = 1000000  # max samples retained per partition
    num_storage_units: int


class MooncakeCpuConfig(BaseModel, extra="allow"):
    """Sizing and RDMA knobs for ``backend="mooncake_cpu"``. Ignored otherwise.

    ``global_segment_size`` / ``local_buffer_size`` are per client *process*
    (one per GPU), so a node pays ``gpus_per_node x (segment + buffer)``. Under
    RDMA that memory is pinned and resident from setup, so keep the per-node
    product in mind when raising them. Under-sizing surfaces as
    ``batch_get_tensor returned None``.

    ``reuse_registered_buffers`` keeps a pool of RDMA-registered buffers alive
    instead of registering a fresh one per transfer; set false to fall back to
    upstream's per-call registration.

    ``staging_buffer_size`` is that pool's per-slot ceiling. It is a pooling
    threshold, not a size limit: a bigger payload still transfers, just with a
    transient registration. Slots ratchet — they grow to the largest payload
    admitted and never shrink — so raise it only when a per-key payload (one
    sample of one field) genuinely exceeds it, not for headroom.

    ``use_gdr`` switches CUDA-initialized clients to TransferQueue's GDR path,
    where the staging buffer is GPU memory instead of host memory.
    ``gdr_staging_buffer_mb`` is that persistent GPU capacity per active GDR
    client. CPU-only clients (a SingleController producer, say) keep the host
    staging pool either way.

    Every RDMA rail on the host is offered to mooncake (see ``rdma_devices``).
    That is only safe with ``MC_ENABLE_DEST_DEVICE_AFFINITY=1``, which pins each
    transfer's peer rail to the local one by name; on a rail-isolated RoCE
    fabric a cross-rail pair has no route. It is set on RoCE-only hosts by
    ``nemo_rl.data_plane.adapters.transfer_queue_env.configure_engine_env``.
    """

    global_segment_size: int = 68719476736  # 64 GiB per client process
    local_buffer_size: int = 4294967296  # 4 GiB per client process
    reuse_registered_buffers: bool = True
    staging_buffer_size: int = 268435456  # 256 MiB per pool slot
    use_gdr: bool = False
    gdr_staging_buffer_mb: PositiveInt = 1024


class TransferEngineConfig(BaseModel, extra="allow"):
    """Knobs for ``backend="transfer_engine"`` (register mode). Ignored otherwise.

    Register mode has no store, so none of mooncake_cpu's sizing applies: there
    is no segment to pre-pin and no staging pool. ``put`` registers the
    producer's own allocation and publishes its address; the bytes move once,
    when a consumer reads.

    ``use_gdr`` decides where a *read* lands in the process that issues it —
    this GPU under GDR, host memory otherwise. It is per process, not per
    cluster: a CPU-only client (the SingleController driver) resolves to host
    either way and still reads a GPU-resident key correctly, because the
    producer's side of the pairing is chosen by probing the registered pointer.
    Sources are never moved: a CUDA tensor is registered where the producer
    built it, so with GDR on both ends a tensor crosses the fabric HBM-to-HBM.

    ``offload_source_to_host`` splits the two halves of that decision. With it
    on, ``put`` copies a CUDA tensor to host and registers *there*, while reads
    still land in HBM — the shape is then producer host memory straight into
    consumer HBM in one hop, with no staging buffer and no store copy on the
    way. It costs one D2H per put and buys back the HBM the producer would
    otherwise hold registered until ``clear``, which is this backend's sharpest
    cost. Leave it off to keep the payload in HBM end to end.

    ``rpc_port`` is this process's engine port; 0 lets it pick a free one,
    which is required when several clients share a node.

    The producer must outlive every key it publishes and must not mutate a
    registered buffer until ``clear``. See
    ``docs/design-docs/tq-register-mode.md`` for the read fan-out and HBM
    residency this trades against.
    """

    use_gdr: bool = True
    offload_source_to_host: bool = False
    rpc_port: int = 0


class DataPlaneConfig(TypedDict):
    """Feature-gated config; defaults to disabled.

    ``backend`` is the storage backend *inside* TransferQueue; it is owned by
    the TQ adapter, not by NeMo-RL. ``impl`` selects which adapter we go
    through.

    Backend-specific knobs live under a block named for the backend that reads
    them — ``simple:`` and ``mooncake_cpu:`` — mirroring TransferQueue's own
    ``config.yaml`` and the per-backend overlay :func:`_init_tq` builds. Only
    the block named by ``backend`` is consulted, so a config selecting
    ``simple`` never has to mention mooncake's RDMA sizing at all. An absent
    ``mooncake_cpu:`` block means "use :class:`MooncakeCpuConfig`'s
    defaults" — but ``simple:`` is **not** optional: ``num_storage_units``
    has no static default, since no single value is right across cluster
    sizes, so a ``simple`` run without the block fails validation.

    Required keys (always set in the exemplar YAML): ``enabled``, ``impl``,
    ``backend``, ``claim_meta_poll_interval_s``.

    ``storage_capacity`` / ``num_storage_units`` / ``global_segment_size`` /
    ``local_buffer_size`` / ``reuse_registered_buffers`` used to sit at this
    level. A config still using that spelling is not rejected — the flat
    key is simply never read, and :func:`backend_config` resolves the
    nested block (or its defaults) as if it were absent. See there.

    The first GDR integration also spelled ``use_gdr`` and
    ``gdr_staging_buffer_mb`` at this level. Those two are rejected rather
    than ignored, so an old GDR recipe cannot silently run with GDR disabled.
    """

    enabled: bool
    impl: Literal["transfer_queue"]
    backend: Literal["simple", "mooncake_cpu", "transfer_engine"]
    claim_meta_poll_interval_s: float
    simple: NotRequired[SimpleStorageConfig]
    mooncake_cpu: NotRequired[MooncakeCpuConfig]
    transfer_engine: NotRequired[TransferEngineConfig]
    controller_address: NotRequired[str]
    ack_timeout_ms: NotRequired[int]
    observability: NotRequired["ObservabilityConfig"]

    # The first GDR integration's deprecated flat spellings are declared only
    # so Pydantic preserves them until ``backend_config`` can raise the
    # migration error below. Without these entries, MasterConfig silently
    # drops the keys during validation.
    use_gdr: NotRequired[bool]
    gdr_staging_buffer_mb: NotRequired[int]


# PR #3501 introduced these at the top level before backend-specific nesting.
# Reject that spelling for either selected backend so rebasing an existing GDR
# recipe cannot silently disable GDR.
_FIRST_GDR_FLAT_KEYS = ("use_gdr", "gdr_staging_buffer_mb")

_BACKEND_MODELS: dict[str, type[BaseModel]] = {
    "simple": SimpleStorageConfig,
    "mooncake_cpu": MooncakeCpuConfig,
    "transfer_engine": TransferEngineConfig,
}


def backend_config(cfg: DataPlaneConfig) -> Any:
    """Return the validated sizing block for ``cfg["backend"]``.

    Reads the nested block and lets the model supply anything it omits, so no
    caller ever writes a fallback. Works whether ``cfg`` came through pydantic
    (block already coerced to a model) or as a plain dict from a test.

    Sizing is read only from the nested block. A config still using the
    pre-nesting flat sizing spelling gets this backend's defaults, not its own
    values. The first GDR integration's flat spelling is rejected instead,
    because otherwise an old GDR recipe would silently run with GDR disabled.
    """
    backend = cfg["backend"]
    stale_gdr = [k for k in _FIRST_GDR_FLAT_KEYS if k in cfg]
    if stale_gdr:
        raise ValueError(
            f"data_plane.{{{','.join(stale_gdr)}}} moved under "
            f"data_plane.mooncake_cpu (or data_plane.transfer_engine for "
            f"register mode). Update the first GDR integration's top-level "
            f"spelling before running this stack."
        )

    nested = cfg.get(backend) or {}
    if isinstance(nested, BaseModel):
        nested = nested.model_dump(exclude_unset=True)
    return _BACKEND_MODELS[backend].model_validate(nested)


class ObservabilityConfig(TypedDict):
    """Optional middleware that records per-op metrics on the client.

    Off by default. When ``enabled=True`` the factory wraps the chosen
    adapter with :class:`MetricsDataPlaneClient`. ``callback`` is
    injected programmatically (callables don't round-trip through
    YAML) — set ``cfg["observability"]["callback"] = my_fn`` before
    :func:`build_data_plane_client` to plug into wandb / file / log.
    Default callback prints one line per op for debug.
    """

    enabled: bool
    callback: NotRequired[Callable[[dict[str, Any]], None]]


@dataclass
class KVBatchMeta:
    """Per-batch metadata for data-plane KV operations.

    Carries the per-sample IDs (``sample_ids``) that address rows in the
    KV store plus per-row metadata (``fields``, ``sequence_lengths``,
    ``tags``) needed for downstream routing without fetching tensor data.
    Vocabulary is intentionally NeMo-RL-native rather than 1:1 with any
    specific backend — the adapter translates at the boundary.

    Two roles:
      * Result type returned by :meth:`DataPlaneClient.claim_meta` — callers
        extract ``.sample_ids`` / ``.partition_id`` and pass them to
        :meth:`get_samples` / :meth:`get_data`.
      * Argument type for the per-DP-rank fetch entrypoints.
        ``sequence_lengths`` lets the driver compute a balanced per-rank
        shard from metadata only (control plane), without ever
        materializing tensor data.
    """

    partition_id: str
    task_name: str | None
    sample_ids: list[str]
    fields: list[str] | None = None
    sequence_lengths: list[int] | None = None
    extra_info: dict[str, Any] = field(default_factory=dict)
    # Per-sample primitive sidecar. Aligned 1:1 with ``sample_ids`` when
    # populated. Producers stamp filter scalars (std, total_reward,
    # weight_version, …) here at ``put_samples`` time so consumers
    # can filter without fetching tensor data. Mirrors verl's pattern
    # and TQ's underlying ``KVBatchMeta.tags``.
    tags: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.tags is not None and len(self.tags) != len(self.sample_ids):
            raise ValueError(
                f"KVBatchMeta: tags ({len(self.tags)}) must align 1:1 with "
                f"sample_ids ({len(self.sample_ids)})"
            )

    @property
    def size(self) -> int:
        return len(self.sample_ids)

    def stamp_tags(self, scalars: dict[str, "Sequence[Any]"]) -> None:
        """Mirror per-row scalar columns onto :attr:`tags`.

        Each entry in ``scalars`` is a length-``size`` sequence (list,
        tensor, ndarray) whose elements are written to ``tags[i][name]``.
        Initializes ``tags`` to a list of empty dicts if currently None.
        """
        n = self.size
        if self.tags is None:
            self.tags = [{} for _ in range(n)]
        for name, values in scalars.items():
            if len(values) != n:
                raise ValueError(
                    f"stamp_tags: {name!r} has {len(values)} values, expected {n}"
                )
            for i, v in enumerate(values):
                self.tags[i][name] = v  # type: ignore[bad-specialization]

    # ── Pure-metadata transforms (no I/O) ──────────────────────────────
    # Used by dynamic_sampling on the meta path: filter zero-std rows
    # (subset), accumulate survivors across iterations (concat), trim
    # an over-full cache to the training batch size (slice). Each
    # returns a fresh KVBatchMeta — caller is responsible for clear_samples-
    # ing any uids dropped from the working set.

    def _replace(
        self,
        *,
        sample_ids: list[str],
        sequence_lengths: list[int] | None,
        tags: list[dict[str, Any]] | None = None,
    ) -> "KVBatchMeta":
        """Return a copy with new sample_ids/sequence_lengths/tags, same metadata otherwise."""
        return KVBatchMeta(
            partition_id=self.partition_id,
            task_name=self.task_name,
            sample_ids=list(sample_ids),
            fields=self.fields,
            sequence_lengths=list(sequence_lengths)
            if sequence_lengths is not None
            else None,
            extra_info=dict(self.extra_info or {}),
            tags=list(tags) if tags is not None else None,
        )

    def subset(self, indices: "Sequence[int]") -> "KVBatchMeta":
        """Return a new meta with only the rows at ``indices`` (any order)."""
        return self._replace(
            sample_ids=[self.sample_ids[i] for i in indices],
            sequence_lengths=(
                [self.sequence_lengths[i] for i in indices]
                if self.sequence_lengths is not None
                else None
            ),
            tags=([self.tags[i] for i in indices] if self.tags is not None else None),
        )

    def slice(self, start: int, stop: int) -> "KVBatchMeta":
        """Return a new meta with rows in the contiguous range ``[start, stop)``."""
        return self._replace(
            sample_ids=self.sample_ids[start:stop],
            sequence_lengths=(
                self.sequence_lengths[start:stop]
                if self.sequence_lengths is not None
                else None
            ),
            tags=self.tags[start:stop] if self.tags is not None else None,
        )

    def concat(self, *others: "KVBatchMeta") -> "KVBatchMeta":
        """Append ``others`` to ``self``. All metas must share ``partition_id``."""
        if any(o.partition_id != self.partition_id for o in others):
            raise ValueError("KVBatchMeta.concat: partition_ids must match")
        all_m = (self, *others)
        sample_ids = [k for m in all_m for k in m.sample_ids]
        all_have_lens = all(m.sequence_lengths is not None for m in all_m)
        seq_lens = (
            [s for m in all_m for s in (m.sequence_lengths or [])]
            if all_have_lens
            else None
        )
        all_have_tags = all(m.tags is not None for m in all_m)
        tags = [t for m in all_m for t in (m.tags or [])] if all_have_tags else None
        return self._replace(
            sample_ids=sample_ids, sequence_lengths=seq_lens, tags=tags
        )

    def drop(self, indices: "Sequence[int]") -> "KVBatchMeta | None":
        """Complement of :meth:`subset`. Returns ``None`` when all rows are dropped."""
        dropped = set(indices)
        keep = [i for i in range(self.size) if i not in dropped]
        if not keep:
            return None
        return self.subset(keep)

    def with_fields(self, field_names: "Sequence[str]") -> "KVBatchMeta":
        """Return a copy with ``field_names`` merged into ``fields`` (deduped, order-preserving)."""
        merged = list(dict.fromkeys([*(self.fields or []), *field_names]))
        return KVBatchMeta(
            partition_id=self.partition_id,
            task_name=self.task_name,
            sample_ids=list(self.sample_ids),
            fields=merged,
            sequence_lengths=(
                list(self.sequence_lengths)
                if self.sequence_lengths is not None
                else None
            ),
            extra_info=dict(self.extra_info or {}),
            tags=[dict(tag) for tag in self.tags] if self.tags is not None else None,
        )


class DataPlaneClient(ABC):
    """Stable, swappable data-plane boundary.

    The methods are split into three groups by intent. Argument order
    mirrors the underlying ``transfer_queue`` API 1:1 so a future adapter
    (e.g. ``nv-dataplane``) is a thin pass-through too.

    A. *Task-mediated* — used by stages that wait for upstream production
       via the per-task consumer counter:
       :meth:`register_partition`, :meth:`claim_meta`, :meth:`get_data`,
       :meth:`check_consumption_status`.
    B. *Direct-by-key* — used by stages that already know the exact uids
       (e.g. driver-side fan-out to DP ranks):
       :meth:`put_samples`, :meth:`get_samples`, :meth:`clear_samples`.
    C. *Lifecycle* — :meth:`close`.

    Stage-completion signal: there is intentionally no ``mark_consumed``.
    The authoritative signal in TransferQueue is *field production* —
    when a stage calls :meth:`put_samples` for a new field, the controller
    flips ``production_status[sample, field] = 1``. Downstream consumers
    waiting on that field only see those samples once produced.
    """

    @property
    def put_device(self) -> str:
        """Device a producer should hand tensors to :meth:`put_samples` on.

        ``"cpu"`` for every backend that copies the payload into a store, which
        is why producers move results to host before writing back. Register
        mode overrides it: it registers the tensor where it already is, so a
        write-back that copies to host first would be pure loss.
        """
        return "cpu"

    # ── (A) task-mediated ───────────────────────────────────────────────

    @abstractmethod
    def register_partition(
        self,
        partition_id: str,
        fields: list[str],
        num_samples: int,
        consumer_tasks: list[str],
        grpo_group_size: int | None = None,
        enums: dict[str, list[str]] | None = None,
    ) -> None:
        """Declare the partition schema and consumer tasks.

        Args:
            partition_id: Partition name.
            fields: Superset of fields any producer may write here.
            num_samples: Expected total samples; sizes controller arrays.
            consumer_tasks: Named tasks; each gets its own consumption cursor.
            grpo_group_size: Group size for GRPO balanced sampling.
            enums: Per-field fixed-vocab string codec, shipped once at register.
        """

    @abstractmethod
    def claim_meta(
        self,
        partition_id: str,
        task_name: str,
        required_fields: list[str],
        batch_size: int,
        dp_rank: int | None = None,
        blocking: bool = True,
        timeout_s: float = 60.0,
    ) -> KVBatchMeta:
        """Discover and **claim** up to ``batch_size`` ready samples.

        Advances ``task_name``'s per-sample consumption cursor (TQ's
        ``mode='fetch'``); claimed uids won't be returned again. Samples
        stay readable via :meth:`get_samples` until :meth:`clear_samples`.

        Args:
            partition_id: Partition to claim from.
            task_name: Consumer task whose cursor is advanced.
            required_fields: Fields that must be produced for a sample to be claimable.
            batch_size: Max samples to claim.
            dp_rank: Reserved; driver-side balancing via :func:`shard_meta_for_dp` is used today.
            blocking: Block until the batch can be claimed.
            timeout_s: Max blocking time before raising.

        Returns:
            ``KVBatchMeta`` for the claimed batch; pass to :meth:`get_data`.
        """

    @abstractmethod
    def get_data(
        self,
        meta: KVBatchMeta,
        select_fields: list[str] | None = None,
    ) -> TensorDict:
        """Resolve a meta to tensor data.

        Field-set resolution: (1) explicit ``select_fields``; (2)
        ``meta.fields`` if non-None; (3) *fail loudly* — never silently
        fetch all fields.

        Args:
            meta: From :meth:`claim_meta` or hand-built with explicit keys.
            select_fields: Subset of fields to fetch.

        Returns:
            ``TensorDict`` keyed by field name, batched along ``meta.sample_ids``.
        """

    @abstractmethod
    def check_consumption_status(
        self, partition_id: str, task_names: list[str]
    ) -> bool:
        """True iff every task has consumed all samples in the partition.

        Authoritative across workers — uses TQ's controller-side counter,
        not the per-process client cache.

        Args:
            partition_id: Partition to check.
            task_names: Tasks whose consumption cursors are inspected.

        Returns:
            ``True`` iff every task in ``task_names`` has consumed all samples.
        """

    # ── (B) direct-by-key (TQ-aligned signatures) ──────────────────────

    @abstractmethod
    def put_samples(
        self,
        sample_ids: list[str],
        partition_id: str,
        fields: TensorDict | None = None,
        tags: list[dict[str, Any]] | None = None,
    ) -> KVBatchMeta:
        """Write fields for ``sample_ids`` — the producer entrypoint.

        Writing a field flips the controller's ``production_status`` bit
        for ``(sample, field)``; that flip is the "stage finished" signal
        downstream consumers wait on. Tensor and ``NonTensorStack`` leaves
        both pass through to TQ; non-tensor encoding is per-backend.

        Args:
            sample_ids: Per-sample uids being written.
            partition_id: Partition these samples belong to.
            fields: Tensor / ``NonTensorStack`` leaves to write.
            tags: Optional per-sample primitive metadata.

        Returns:
            ``KVBatchMeta`` covering ``sample_ids`` — usable for direct :meth:`get_samples`.
        """

    @abstractmethod
    def get_samples(
        self,
        sample_ids: list[str],
        partition_id: str,
        select_fields: list[str],
    ) -> TensorDict:
        """Direct fetch by uids.

        Used by per-DP-rank slice fetches. Does NOT advance any per-task
        consumption cursor — that only happens via :meth:`claim_meta`.

        ``select_fields`` is required (no implicit "fetch every field"
        fallback): bulk schemas are wide and silent over-fetch is the
        most expensive shape the wire can take. Callers must name what
        they read.

        Args:
            sample_ids: Uids to fetch.
            partition_id: Partition the samples live in.
            select_fields: Subset of fields to fetch.

        Returns:
            ``TensorDict`` keyed by field name, batched along ``sample_ids``.
        """

    @abstractmethod
    def clear_samples(
        self,
        sample_ids: list[str] | None,
        partition_id: str,
    ) -> None:
        """Drop key-value pairs.

        Explicit form (``sample_ids=[...]``) drops exactly those uids and
        is the form callers should use whenever they have the meta in
        hand — both sync GRPO callers (driver passes ``meta.sample_ids``)
        and future async-RL data-loader actors that don't share a
        process-local registry with the producer.

        Convenience form (``sample_ids=None``) drops "everything this
        process knows produced in this partition". Adapters implement
        this via a local registry populated by :meth:`put_samples`, with
        a fallback query to the underlying store. Useful for step-end
        teardown when the caller is the producer (driver in sync GRPO).
        Workers / loader actors that didn't produce the samples should
        pass explicit IDs — the ``None`` form may silently no-op for
        them, and adapters are expected to warn when that happens.

        Args:
            sample_ids: Uids to drop; ``None`` clears every uid this
                process produced in the partition.
            partition_id: Partition the samples live in.
        """

    # ── (C) lifecycle ──────────────────────────────────────────────────

    @abstractmethod
    def close(self) -> None:
        """Release controller / storage handles. Idempotent."""
