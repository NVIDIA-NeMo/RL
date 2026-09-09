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

from __future__ import annotations

import os
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, ContextManager, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.utils import unwrap_model
from torch import Tensor, nn
from torch.multiprocessing.reductions import reduce_tensor


def get_eagle3_aux_hidden_state_layers(num_layers: int) -> tuple[int, ...]:
    """Pick the default auxiliary policy layers whose activations feed Eagle training."""
    candidate_indices = (
        1,
        max(0, num_layers // 2 - 1),
        max(1, num_layers - 4),
    )
    valid_indices = sorted(set(candidate_indices))
    return tuple(valid_indices)


@dataclass
class CapturedStates:
    """Container for hidden states captured from the policy model."""

    hidden_states: Optional[Tensor] = None
    inputs_embeds: Optional[Tensor] = None


def _handle_done(handle: torch.cuda.Event) -> bool:
    """Non-blocking done check for a completion event (IPC and net paths)."""
    return handle.query()


class TapChannel:
    """Tap transport for PP > 1 draft co-training (design §4.6, G v2).

    Under PP > 1 the draft model lives on the last pipeline stage, but the
    policy layers it taps (and the input embeddings) live on earlier stages.
    The last stage pre-allocates one ring of landing slots per source stage;
    the transport into a slot is chosen per source at setup:

    - same host as the draft stage: the ring is exported over CUDA IPC and
      the source writes one-sided on a side stream, then stamps a header the
      consumer polls. No recv, no pairing, no message ordering — it cannot
      interleave with (or deadlock against) Megatron's own pipeline P2P, the
      failure mode of putting send/recv inside forward callbacks.
    - different host (multi-node PP makes every stage pair cross-host under
      Megatron's pp-outermost rank order): NCCL send/recv on a DEDICATED side
      communicator per PP column (GPUDirect RDMA on the wire). Two-sided, but
      it cannot deadlock the schedule either: it never shares a communicator
      with Megatron's P2P, both ends post ops in the same monotone microbatch
      order (NCCL matches per-pair FIFO), and an unmatched send merely parks
      on the channel's own stream without blocking the host.

    Rendezvous is by position, not by id: with a non-interleaved schedule
    every stage forwards microbatches in the same order, so the writer's n-th
    put and the consumer's n-th get are the same microbatch. A header stamp
    carries the writer's sequence number, so an ordering violation surfaces
    as a loud stamp mismatch instead of silent training on the wrong
    microbatch's taps. Interleaved (virtual PP) schedules break this
    invariant and are rejected at build time.

    Reserved memory is exact and static: per source stage,
    ``ring_slots × slot_rows × width × dtype`` where width covers the stage's
    tap layers (plus the input embeddings on the first stage) and slot_rows
    is the per-rank microbatch token budget.
    """

    def __init__(
        self,
        *,
        aux_layer_ids: list[int],
        local_layer_ids: list[int],
        has_embedding: bool,
        hidden_size: int,
        slot_rows: int,
        dtype: torch.dtype,
        pp_group: dist.ProcessGroup,
        net_group: Optional[dist.ProcessGroup] = None,
        net_sources_override: Optional[list[int]] = None,
        needs_mask_row: bool = False,
        mask_token_id: Optional[int] = None,
        ring_slots: Optional[int] = None,
        put_timeout_s: float = 120.0,
        # Strictly below torch's 600s NCCL watchdog: if a rank is starved of
        # taps, OUR error (with microbatch + source stage) must fire before
        # the watchdog aborts the job and masks it.
        get_timeout_s: float = 240.0,
    ):
        self.aux_layer_ids = tuple(int(i) for i in aux_layer_ids)
        self.hidden_size = int(hidden_size)
        self.slot_rows = int(slot_rows)
        self.dtype = dtype
        self.pp_group = pp_group
        self.needs_mask_row = needs_mask_row
        self.mask_token_id = mask_token_id
        self.mask_row: Optional[Tensor] = None
        self.put_timeout_s = put_timeout_s
        self.get_timeout_s = get_timeout_s

        self.group_ranks = dist.get_process_group_ranks(pp_group)
        self.pp_size = len(self.group_ranks)
        self.pp_rank = dist.get_rank(pp_group)
        assert self.pp_rank == parallel_state.get_pipeline_model_parallel_rank()
        self.is_last_stage = self.pp_rank == self.pp_size - 1

        width = (self.hidden_size if has_embedding else 0) + len(
            local_layer_ids
        ) * self.hidden_size
        meta = [None] * self.pp_size
        dist.all_gather_object(
            meta,
            {
                "host": os.uname().nodename,
                "width": width,
                "has_embedding": has_embedding,
            },
            group=pp_group,
        )
        last_host = meta[-1]["host"]
        # Sources = every earlier stage with something to send. The first stage
        # always sends (it owns the input embeddings the draft consumes).
        self.sources = [r for r in range(self.pp_size - 1) if meta[r]["width"] > 0]
        if not self.sources or self.sources[0] != 0 or not meta[0]["has_embedding"]:
            raise RuntimeError(
                "[draft] Tap channel expects the first pipeline stage to own the "
                f"policy embedding; got per-stage meta {meta}."
            )
        # Transport per source: same host as the draft stage -> one-sided CUDA
        # IPC; different host -> NCCL send/recv on the dedicated side
        # communicator. net_sources_override is a test knob.
        if net_sources_override is not None:
            self.net_sources = [
                r for r in self.sources if r in set(net_sources_override)
            ]
        else:
            self.net_sources = [r for r in self.sources if meta[r]["host"] != last_host]
        self.ipc_sources = [r for r in self.sources if r not in self.net_sources]
        self.net_group = net_group
        if self.net_sources and net_group is None:
            raise ValueError(
                "[draft] Cross-host tap sources need the dedicated NCCL side "
                "communicator; build the channel via build_tap_channel."
            )
        if self.net_sources:
            # NCCL initializes lazily on first use, and BOTH steps are
            # peer-synchronous: the communicator (all group members) and each
            # point-to-point connection (its src/dst pair). Left lazy, a
            # source's first in-schedule isend host-blocks until the
            # consumer's first irecv — which waits on activations from that
            # very forward. Warm both up here, outside the schedule.
            dist.barrier(group=net_group)
            probe = torch.zeros(1, dtype=torch.int32, device="cuda")
            if self.pp_rank in self.net_sources:
                self._net_ops(dist.isend, probe, self.pp_size - 1).synchronize()
            elif self.is_last_stage:
                for src in self.net_sources:
                    self._net_ops(dist.irecv, probe, src).synchronize()
        self._widths = {r: meta[r]["width"] for r in self.sources}
        # In-flight bound: 1F1B keeps stage s at most (pp_size - s) microbatches
        # ahead of the last stage; forward-only passes can run slightly further
        # ahead (bounded by the activation P2P rendezvous), so double it.
        self._slots = {
            r: int(ring_slots) if ring_slots is not None else 2 * (self.pp_size - r)
            for r in self.sources
        }

        # The consumer allocates the landing rings for every source. IPC
        # sources additionally get an exported header ring (int32 [ring, 4] =
        # (stamp, rows, batch, 0); stamp 0 marks a free slot); net sources
        # ship the header as a message and keep their ring consumer-side only.
        handles: list[Any] = [None]
        if self.is_last_stage:
            self._data = {
                r: torch.zeros(
                    self._slots[r],
                    self.slot_rows,
                    self._widths[r],
                    dtype=dtype,
                    device="cuda",
                )
                for r in self.sources
            }
            self._headers = {
                r: torch.zeros(self._slots[r], 4, dtype=torch.int32, device="cuda")
                for r in self.ipc_sources
            }
            # Persistent landing buffers for net headers. The hot loop must
            # never allocate: per-microbatch torch.empty headers were observed
            # (in full trainer runs) landing on allocator blocks double-booked
            # with live int64 tensors — the received header kept mutating
            # AFTER delivery. Ctor-time buffers never re-enter the pool.
            self._net_headers = {
                r: torch.zeros(4, dtype=torch.int32, device="cuda")
                for r in self.net_sources
            }
            self._consume_event: Optional[torch.cuda.Event] = None
            handles = [
                {
                    r: (reduce_tensor(self._data[r]), reduce_tensor(self._headers[r]))
                    for r in self.ipc_sources
                }
            ]
            reserved = sum(t.numel() * t.element_size() for t in self._data.values())
            print(
                f"[draft] Tap channel reserved {reserved / (1 << 20):.0f} MiB on the "
                f"draft stage: rows/slot={self.slot_rows}, "
                f"per-source (width, slots)="
                f"{[(self._widths[r], self._slots[r]) for r in self.sources]}.",
                flush=True,
            )
        dist.broadcast_object_list(handles, src=self.group_ranks[-1], group=pp_group)
        self.is_source = self.pp_rank in self.sources
        self.is_net_source = self.pp_rank in self.net_sources
        if self.is_source:
            if not self.is_net_source:
                (data_fn, data_args), (hdr_fn, hdr_args) = handles[0][self.pp_rank]
                self._remote_data = data_fn(*data_args)
                self._remote_header = hdr_fn(*hdr_args)
            # Persistent header staging ring (see _net_headers): staged rows
            # are reused only after their send/copy retired via the ring bound.
            self._header_ring = torch.zeros(
                self._slots[self.pp_rank], 4, dtype=torch.int32, device="cuda"
            )
            self._io_stream = torch.cuda.Stream()
            self._pending: list[tuple[Tensor, torch.cuda.Event]] = []
        if self.is_last_stage:
            self._poll_stream = torch.cuda.Stream()

        self._put_seq = 0
        self._get_seq = 0
        self._wait_seconds = 0.0

    def _net_ops(self, op: Any, tensor: Tensor, peer: int) -> torch.cuda.Event:
        """One batched p2p op on the net communicator; returns a completion event.

        Batched (not bare send/recv): unbatched p2p takes torch's lazy
        per-rank-pair subcommunicator path (the ProcessGroupNCCL size-2
        warning) — the same special path Megatron's PP=2 activation p2p is
        forced onto for the SAME rank pair; batched ops stay on net_group's
        own communicator, like Megatron's `_batched_p2p_ops` and the DFlash
        ring.

        Completion is signalled by a CUDA event recorded behind the NCCL end
        event on the CALLER'S ACTIVE stream, never by Work.is_completed():
        the latter was observed returning True while the recv'd buffer was
        still mutating (read-before-delivery), and on the send side it would
        release the pending chunk ref while NCCL is still reading it.
        """
        work = dist.batch_isend_irecv(
            [dist.P2POp(op, tensor, self.group_ranks[peer], self.net_group)]
        )[0]
        work.wait()  # the active stream now depends on the NCCL end event
        done = torch.cuda.Event()
        done.record()
        return done

    def begin_pass(self, model: Any) -> None:
        """Per-pass refresh outside the schedule: prune writer refs, fetch the mask row.

        The DFlash/DSpark mask embedding is the policy's LIVE
        ``embed_tokens[mask_token_id]`` row, which only the first stage owns —
        broadcast it over the PP group so the draft stage matches the PP = 1
        semantics each pass.
        """
        if self.is_source and self._pending and _handle_done(self._pending[-1][-1]):
            self._pending.clear()  # io stream / NCCL pair are FIFO: last done => all done
        if not self.needs_mask_row:
            return
        from nemo_rl.models.megatron.draft.utils import get_policy_embedding_row

        chunks = model if isinstance(model, list) else [model]
        if self.pp_rank == 0:
            row = get_policy_embedding_row(
                unwrap_model(chunks[0]), int(self.mask_token_id)
            )
            row = row.detach().to(self.dtype).contiguous()
        else:
            row = torch.empty(self.hidden_size, dtype=self.dtype, device="cuda")
        dist.broadcast(row, src=self.group_ranks[0], group=self.pp_group)
        self.mask_row = row

    def put(self, chunk: Tensor) -> None:
        """Ship this stage's (S, B, width) tap chunk for the next microbatch."""
        assert self.is_source
        seq_len, batch, width = chunk.shape
        rows = seq_len * batch
        if width != self._widths[self.pp_rank] or rows > self.slot_rows:
            raise RuntimeError(
                f"[draft] Tap chunk (rows={rows}, width={width}) does not fit its "
                f"slot (rows={self.slot_rows}, width={self._widths[self.pp_rank]}); "
                "the microbatch token budget changed after channel setup."
            )
        slots = self._slots[self.pp_rank]
        slot = self._put_seq % slots
        deadline = time.perf_counter() + self.put_timeout_s
        if self.is_net_source:
            # Unmatched sends park harmlessly on the io stream; bound the
            # in-flight chunks we keep alive to the same ring depth as IPC.
            while len(self._pending) >= slots:
                if _handle_done(self._pending[0][-1]):
                    self._pending.pop(0)
                    continue
                if time.perf_counter() > deadline:
                    raise RuntimeError(
                        f"[draft] Tap send for put #{self._put_seq - slots} still "
                        f"unconsumed after {self.put_timeout_s}s at put "
                        f"#{self._put_seq} — the draft stage is more than {slots} "
                        "microbatches behind, which no supported schedule produces."
                    )
        else:
            while int(self._remote_header[slot, 0].item()) != 0:
                if time.perf_counter() > deadline:
                    raise RuntimeError(
                        f"[draft] Tap slot {slot} still unconsumed after "
                        f"{self.put_timeout_s}s at put #{self._put_seq} — the draft "
                        f"stage is more than {slots} microbatches behind, which no "
                        "supported schedule produces."
                    )
        staged = self._header_ring[slot]
        staged[0], staged[1], staged[2] = self._put_seq + 1, seq_len, batch
        ready = torch.cuda.Event()
        ready.record()
        with torch.cuda.stream(self._io_stream):
            self._io_stream.wait_event(ready)
            if self.is_net_source:
                self._net_ops(dist.isend, staged, self.pp_size - 1)
                done = self._net_ops(
                    dist.isend, chunk.reshape(rows, width), self.pp_size - 1
                )
            else:
                self._remote_data[slot, :rows].copy_(
                    chunk.reshape(rows, width), non_blocking=True
                )
                # Same stream: the data lands before the stamp becomes visible.
                self._remote_header[slot].copy_(staged, non_blocking=True)
                done = torch.cuda.Event()
                done.record(self._io_stream)
        self._pending.append((chunk, done))
        while self._pending and _handle_done(self._pending[0][-1]):
            self._pending.pop(0)
        self._put_seq += 1

    def get(self, local_hidden_chunks: list[Tensor]) -> tuple[Tensor, Tensor]:
        """Assemble (inputs_embeds, hidden_states) for the next microbatch.

        Polls every source's header, concatenates the remote chunks (ascending
        stage order == ascending tap-layer order) with the last stage's own
        chunks, then frees the slots. Returned tensors own their storage.
        """
        assert self.is_last_stage
        stamp = self._get_seq + 1
        views = {}
        rows = batch = None
        wait_t0 = time.perf_counter()
        for src in self.sources:
            slot = self._get_seq % self._slots[src]
            deadline = time.perf_counter() + self.get_timeout_s
            if src in self.net_sources:
                head = self._recv_net(src, slot, stamp, deadline)
            else:
                while True:
                    with torch.cuda.stream(self._poll_stream):
                        head = [int(v) for v in self._headers[src][slot].tolist()]
                    if head[0] == stamp:
                        break
                    if head[0] > stamp or time.perf_counter() > deadline:
                        raise RuntimeError(
                            f"[draft] Tap rendezvous failed for microbatch "
                            f"#{self._get_seq} from stage {src}: expected stamp "
                            f"{stamp}, header={head} (timeout {self.get_timeout_s}s). "
                            "Source and draft stages disagree on the microbatch order."
                        )
            if rows is None:
                rows, batch = head[1], head[2]
            elif (rows, batch) != (head[1], head[2]):
                raise RuntimeError(
                    f"[draft] Tap shape mismatch across stages at microbatch "
                    f"#{self._get_seq}: {(rows, batch)} vs {tuple(head[1:3])}."
                )
            views[src] = self._data[src][slot, : rows * batch].view(
                rows, batch, self._widths[src]
            )
        self._wait_seconds += time.perf_counter() - wait_t0
        inputs_embeds = views[0][..., : self.hidden_size].clone()
        pieces = [views[0][..., self.hidden_size :]]
        pieces += [views[src] for src in self.sources[1:]]
        pieces += list(local_hidden_chunks)
        pieces = [p for p in pieces if p.shape[-1] > 0]
        hidden_states = torch.cat(pieces, dim=-1)
        for src in self.ipc_sources:  # stream-ordered after the reads above
            self._headers[src][self._get_seq % self._slots[src]].zero_()
        if self.net_sources:
            # Ring reuse fence: the next recv into these slots must not start
            # before the cats above finished reading them.
            self._consume_event = torch.cuda.Event()
            self._consume_event.record()
        self._get_seq += 1
        return inputs_embeds, hidden_states

    def pop_rendezvous_wait_s(self) -> float:
        """Consumer-side rendezvous wait accumulated by get() since the last pop.

        Covers stamp polling plus (net sources) header/payload recv — the time
        the draft stage's forward is blocked on taps, excluding the assembly
        compute that follows.
        """
        seconds, self._wait_seconds = self._wait_seconds, 0.0
        return seconds

    def _recv_net(self, src: int, slot: int, stamp: int, deadline: float) -> list[int]:
        """Receive one microbatch's header + payload from a cross-host source."""
        header = self._net_headers[src]
        with torch.cuda.stream(self._poll_stream):
            if self._consume_event is not None:
                self._poll_stream.wait_event(self._consume_event)
            done = self._net_ops(dist.irecv, header, src)
        self._wait_net_event(done, src, stamp, deadline)
        with torch.cuda.stream(self._poll_stream):
            head = [int(v) for v in header.tolist()]
        if head[0] != stamp:
            time.sleep(0.2)  # forensic: a changed re-read = read-before-delivery
            with torch.cuda.stream(self._poll_stream):
                recheck = [int(v) for v in header.tolist()]
            raise RuntimeError(
                f"[draft] Tap rendezvous failed for microbatch #{self._get_seq} "
                f"from stage {src}: expected stamp {stamp}, header={head}, "
                f"recheck after 200ms={recheck}. "
                "Source and draft stages disagree on the microbatch order."
            )
        with torch.cuda.stream(self._poll_stream):
            done = self._net_ops(
                dist.irecv, self._data[src][slot, : head[1] * head[2]], src
            )
        self._wait_net_event(done, src, stamp, deadline)
        return head

    def _wait_net_event(
        self, event: torch.cuda.Event, src: int, stamp: int, deadline: float
    ) -> None:
        while not event.query():
            if time.perf_counter() > deadline:
                raise RuntimeError(
                    f"[draft] Tap rendezvous failed for microbatch #{self._get_seq} "
                    f"from stage {src}: NCCL recv (stamp {stamp}) did not complete "
                    f"within {self.get_timeout_s}s."
                )


def tap_slot_rows(policy_cfg: dict[str, Any]) -> int:
    """Per-rank landing-slot row budget from the config's microbatch token budget."""
    for key in ("sequence_packing", "dynamic_batching"):
        sub = policy_cfg.get(key) or {}
        if sub.get("enabled"):
            tokens = int(sub["train_mb_tokens"])
            break
    else:
        tokens = int(policy_cfg["max_total_sequence_length"]) * int(
            policy_cfg["train_micro_batch_size"]
        )
    cp = parallel_state.get_context_parallel_world_size()
    # + slack for the pad-to-multiple (sequence_length_round × CP zigzag) the
    # packers apply on top of the budget.
    return tokens // cp + 1024


def build_tap_channel(
    model: Any, draft_config: dict[str, Any], policy_cfg: dict[str, Any]
) -> TapChannel:
    """Scan the local model chunk and build the PP tap channel (all PP ranks)."""
    from nemo_rl.models.megatron.draft.utils import resolve_draft_aux_layer_ids

    chunks = model if isinstance(model, list) else [model]
    vpp = parallel_state.get_virtual_pipeline_model_parallel_world_size()
    if len(chunks) > 1 or (vpp or 1) > 1:
        raise NotImplementedError(
            "[draft] PP draft co-training does not support interleaved (virtual) "
            "pipeline schedules: the tap channel matches microbatches by "
            "per-stage forward order, which interleaving breaks."
        )
    chunk = unwrap_model(chunks[0])
    hidden_size = int(chunk.config.hidden_size)
    aux_layer_ids = resolve_draft_aux_layer_ids(
        draft_config, int(chunk.config.num_layers)
    )
    local_layer_ids = sorted(
        int(layer.layer_number) - 1
        for layer in chunk.decoder.layers
        if int(layer.layer_number) - 1 in aux_layer_ids
    )

    needs_mask_row = (draft_config.get("speculator_type") or "eagle3") in (
        "dflash",
        "dspark",
    )
    mask_token_id = draft_config.get("mask_token_id")
    if needs_mask_row and mask_token_id is None:
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(draft_config["model_name"]).to_dict()
        mask_token_id = (hf_config.get("dflash_config") or {}).get(
            "mask_token_id"
        ) or hf_config.get("mask_token_id")
    if needs_mask_row and mask_token_id is None:
        raise ValueError("policy.draft.mask_token_id is required for dflash/dspark.")

    # One dedicated NCCL side communicator per PP column, used by cross-host
    # sources. dist.new_group is collective over WORLD, so every rank creates
    # every column's group (cheap: NCCL initializes a communicator lazily on
    # first use) and keeps its own.
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    pp_ranks = dist.get_process_group_ranks(pp_group)
    stride = pp_ranks[1] - pp_ranks[0]
    if pp_ranks != [pp_ranks[0] + s * stride for s in range(len(pp_ranks))] or (
        dist.get_world_size() != stride * len(pp_ranks)
    ):
        raise RuntimeError(
            f"[draft] Unexpected PP rank layout {pp_ranks}; the tap side "
            "communicators assume Megatron's uniform-stride pp-outermost order."
        )
    net_group = None
    for column in range(stride):
        group = dist.new_group([column + s * stride for s in range(len(pp_ranks))])
        if column == pp_ranks[0]:
            net_group = group

    return TapChannel(
        aux_layer_ids=aux_layer_ids,
        local_layer_ids=local_layer_ids,
        has_embedding=hasattr(chunk, "embedding"),
        hidden_size=hidden_size,
        slot_rows=tap_slot_rows(policy_cfg),
        dtype=chunk.config.params_dtype,
        pp_group=pp_group,
        net_group=net_group,
        needs_mask_row=needs_mask_row,
        mask_token_id=None if mask_token_id is None else int(mask_token_id),
    )


class HiddenStateCapture:
    """Capture policy embeddings and auxiliary hidden states for draft training."""

    def __init__(
        self,
        model: nn.Module,
        aux_layer_indices: Optional[Tuple[int, ...]] = None,
        tap_channel: Optional[TapChannel] = None,
    ):
        self.model = unwrap_model(model)
        self.num_layers = self.model.config.num_layers
        # Trunk SP shards layer/embedding outputs into contiguous dim-0 chunks
        # over TP (mcore hard-requires SP for MoE training with TP > 1). Taps
        # are detached, so a plain all_gather restores the SP-off row layout;
        # the draft module itself always runs with SP disabled.
        self._sp_world = (
            parallel_state.get_tensor_model_parallel_world_size()
            if getattr(self.model.config, "sequence_parallel", False)
            else 1
        )

        self.aux_layer_indices = (
            aux_layer_indices
            if aux_layer_indices is not None
            else get_eagle3_aux_hidden_state_layers(self.num_layers)
        )
        self._tap_channel = tap_channel

        self.pp_size = parallel_state.get_pipeline_model_parallel_world_size()
        self.pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        self.is_first_stage = parallel_state.is_pipeline_first_stage()
        self.is_last_stage = parallel_state.is_pipeline_last_stage()

        self._global_to_local: Dict[int, int] = {}
        self._local_aux_indices: List[int] = []
        self._compute_local_layer_mapping()

        self._captured: Dict[str, Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

    def _compute_local_layer_mapping(self) -> None:
        for local_idx, layer in enumerate(self.model.decoder.layers):
            global_idx = int(layer.layer_number) - 1
            if global_idx in self.aux_layer_indices:
                self._global_to_local[global_idx] = local_idx
                self._local_aux_indices.append(local_idx)

    def _full_rows(self, t: Tensor) -> Tensor:
        """Return an owned copy of a captured tensor with full sequence rows.

        Under trunk SP the collective is safe inside forward hooks (including
        the recompute replay): TP peers fire the same hooks at the same layer
        boundaries, so the all_gather order matches across the group.
        """
        if self._sp_world == 1:
            return t.clone()
        out = t.new_empty((t.shape[0] * self._sp_world, *t.shape[1:]))
        dist.all_gather_into_tensor(
            out,
            t.contiguous(),
            group=parallel_state.get_tensor_model_parallel_group(),
        )
        return out

    def _make_layer_output_hook(self, global_idx: int):
        def hook(_module, _args, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            if hidden_states is None:
                return
            self._captured[f"layer_{global_idx}"] = self._full_rows(
                hidden_states.detach()
            )

        return hook

    def _make_embedding_hook(self):
        def hook(_module, _args, output):
            self._captured["embeds"] = self._full_rows(output.detach())

        return hook

    def register_hooks(self) -> None:
        self.clear_hooks()
        self._captured.clear()

        if self.is_first_stage and hasattr(self.model, "embedding"):
            self._hooks.append(
                self.model.embedding.register_forward_hook(self._make_embedding_hook())
            )

        for local_idx in self._local_aux_indices:
            layer = self.model.decoder.layers[local_idx]
            global_idx = int(layer.layer_number) - 1
            self._hooks.append(
                layer.register_forward_hook(self._make_layer_output_hook(global_idx))
            )

    def clear_hooks(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    @contextmanager
    def capture_context(self):
        try:
            self.register_hooks()
            yield self
        finally:
            self.clear_hooks()

    def _assemble_local_states(self) -> CapturedStates:
        embeds = self._captured.get("embeds")

        hidden_chunks = []
        for global_idx in sorted(self.aux_layer_indices):
            tensor = self._captured.get(f"layer_{global_idx}")
            if tensor is not None:
                hidden_chunks.append(tensor)

        if not hidden_chunks:
            return CapturedStates(hidden_states=None, inputs_embeds=embeds)

        return CapturedStates(
            hidden_states=torch.cat(hidden_chunks, dim=-1),
            inputs_embeds=embeds,
        )

    def get_captured_states(self) -> CapturedStates:
        if self.pp_size == 1:
            return self._assemble_local_states()
        if self._tap_channel is None:
            raise RuntimeError(
                "Draft hidden-state capture with pipeline_model_parallel_size > 1 "
                "requires the tap channel built at worker setup."
            )
        local = self._assemble_local_states()
        if not self.is_last_stage:
            # Embeds-first matches the channel's slot layout; the consumer
            # splits the first stage's chunk back apart in TapChannel.get.
            parts = [
                t for t in (local.inputs_embeds, local.hidden_states) if t is not None
            ]
            if parts:
                self._tap_channel.put(
                    torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
                )
            return CapturedStates()
        local_hidden = [local.hidden_states] if local.hidden_states is not None else []
        inputs_embeds, hidden_states = self._tap_channel.get(local_hidden)
        return CapturedStates(hidden_states=hidden_states, inputs_embeds=inputs_embeds)


def get_capture_context(
    model: nn.Module,
    enabled: bool = False,
    aux_layer_indices: Optional[Tuple[int, ...]] = None,
    tap_channel: Optional[TapChannel] = None,
) -> Tuple[ContextManager, Optional[HiddenStateCapture]]:
    """Return a no-op context unless draft training needs hidden-state capture for this step."""
    if not enabled:
        return nullcontext(), None
    if tap_channel is not None:
        # Single source of truth under PP > 1: source stages have no draft
        # model to read the ids from, and a mismatch would silently capture
        # (and train on) the wrong layers.
        aux_layer_indices = tap_channel.aux_layer_ids
    capture = HiddenStateCapture(
        model=model,
        aux_layer_indices=aux_layer_indices,
        tap_channel=tap_channel,
    )
    return capture.capture_context(), capture
