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

"""PP > 1 tap-channel tests: one-sided transport, ring reuse, capture wiring."""

import time
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

pytestmark = pytest.mark.mcore

from megatron.core import parallel_state  # noqa: E402

from nemo_rl.models.megatron.draft.hidden_capture import (  # noqa: E402
    HiddenStateCapture,
    TapChannel,
)
from nemo_rl.models.megatron.draft.utils import (  # noqa: E402
    resolve_draft_aux_layer_ids,
)

requires_2_gpus = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs >= 2 GPUs"
)
requires_4_gpus = pytest.mark.skipif(
    torch.cuda.device_count() < 4, reason="needs >= 4 GPUs"
)

HIDDEN = 16
# (seq_len, batch) per microbatch; 6 microbatches wrap the default pp=2 ring
# (2 * pp_size = 4 slots) and exercise varying shapes within one slot budget.
MB_SHAPES = [(8, 2), (16, 1), (5, 3), (12, 2), (16, 2), (3, 1)]


def _init_pp(rank: int, world: int, init_file: str) -> None:
    # File rendezvous: fixed TCP ports fall inside the cluster's ephemeral
    # range (9000-65000) and collide with lingering sockets of earlier tests.
    dist.init_process_group(
        "nccl", rank=rank, world_size=world, init_method=f"file://{init_file}"
    )
    torch.cuda.set_device(rank)
    parallel_state.initialize_model_parallel(pipeline_model_parallel_size=world)


def _teardown() -> None:
    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


def _pattern(tag: float, seq_len: int, batch: int, width: int) -> torch.Tensor:
    base = torch.arange(seq_len * batch * width, device="cuda", dtype=torch.float32)
    return (base.view(seq_len, batch, width) * 1e-3 + tag).to(torch.bfloat16)


def _make_channel(rank: int, local_layer_ids, net=None, **kwargs) -> TapChannel:
    """net: list of source ranks forced onto the NCCL side channel (None = all IPC)."""
    net_group = (
        dist.new_group(list(range(dist.get_world_size()))) if net is not None else None
    )
    return TapChannel(
        aux_layer_ids=kwargs.pop("aux_layer_ids", [1, 3]),
        local_layer_ids=local_layer_ids,
        has_embedding=rank == 0,
        hidden_size=HIDDEN,
        slot_rows=kwargs.pop("slot_rows", 64),
        dtype=torch.bfloat16,
        pp_group=parallel_state.get_pipeline_model_parallel_group(),
        net_group=net_group,
        net_sources_override=net,
        get_timeout_s=30.0,
        **kwargs,
    )


def _roundtrip_worker(rank: int, world: int, init_file: str, net) -> None:
    _init_pp(rank, world, init_file)
    channel = _make_channel(rank, local_layer_ids=[1] if rank == 0 else [3], net=net)
    for i, (seq_len, batch) in enumerate(MB_SHAPES):
        if rank == 0:
            channel.put(_pattern(i, seq_len, batch, 2 * HIDDEN))
        else:
            local = _pattern(i + 1000.0, seq_len, batch, HIDDEN)
            embeds, hidden = channel.get([local])
            ref = _pattern(i, seq_len, batch, 2 * HIDDEN)
            torch.testing.assert_close(embeds, ref[..., :HIDDEN])
            torch.testing.assert_close(
                hidden, torch.cat([ref[..., HIDDEN:], local], dim=-1)
            )
    _teardown()


@requires_2_gpus
@pytest.mark.parametrize("net", [None, [0]], ids=["ipc", "net"])
def test_channel_roundtrip_pp2(net, tmp_path):
    mp.spawn(
        _roundtrip_worker, args=(2, str(tmp_path / "pg_init"), net), nprocs=2, join=True
    )


def _pp4_worker(rank: int, world: int, init_file: str, net) -> None:
    _init_pp(rank, world, init_file)
    # Stage 2 owns no tap layers -> inactive source (EAGLE-3-at-PP=4 shape).
    local_ids = {0: [1], 1: [4, 5], 2: [], 3: [7]}[rank]
    channel = _make_channel(
        rank, local_layer_ids=local_ids, aux_layer_ids=[1, 4, 5, 7], net=net
    )
    assert channel.sources == [0, 1]
    for i, (seq_len, batch) in enumerate(MB_SHAPES[:3]):
        if rank == 0:
            channel.put(_pattern(i, seq_len, batch, 2 * HIDDEN))
        elif rank == 1:
            channel.put(_pattern(i + 10.0, seq_len, batch, 2 * HIDDEN))
        elif rank == 3:
            local = _pattern(i + 1000.0, seq_len, batch, HIDDEN)
            embeds, hidden = channel.get([local])
            ref0 = _pattern(i, seq_len, batch, 2 * HIDDEN)
            ref1 = _pattern(i + 10.0, seq_len, batch, 2 * HIDDEN)
            torch.testing.assert_close(embeds, ref0[..., :HIDDEN])
            torch.testing.assert_close(
                hidden, torch.cat([ref0[..., HIDDEN:], ref1, local], dim=-1)
            )
    _teardown()


@requires_4_gpus
@pytest.mark.parametrize("net", [None, [0], [0, 1]], ids=["ipc", "mixed", "net"])
def test_channel_pp4_uneven_and_inactive_source(net, tmp_path):
    mp.spawn(_pp4_worker, args=(4, str(tmp_path / "pg_init"), net), nprocs=4, join=True)


def _overrun_worker(rank: int, world: int, init_file: str, net) -> None:
    _init_pp(rank, world, init_file)
    # The net path only backs up once the payload exceeds NCCL's per-connection
    # FIFO buffering (tiny sends complete without a matched recv), so its
    # variant uses a multi-MB chunk; IPC blocks on the header regardless.
    rows = (1 << 20) if net is not None else 4
    channel = _make_channel(
        rank,
        local_layer_ids=[1] if rank == 0 else [3],
        net=net,
        slot_rows=rows,
        ring_slots=1,
        put_timeout_s=0.5,
    )
    if rank == 0:
        channel.put(_pattern(0.0, rows, 1, 2 * HIDDEN))
        try:
            channel.put(_pattern(1.0, rows, 1, 2 * HIDDEN))  # slot never freed
            raise AssertionError("second put into a full ring should time out")
        except RuntimeError as e:
            assert "still unconsumed" in str(e), str(e)
    else:
        time.sleep(2.0)  # let rank 0 hit its put timeout first
        channel.get([_pattern(1000.0, rows, 1, HIDDEN)])  # then drain put #1
    _teardown()


@requires_2_gpus
@pytest.mark.parametrize("net", [None, [0]], ids=["ipc", "net"])
def test_channel_slot_overrun_raises(net, tmp_path):
    mp.spawn(
        _overrun_worker, args=(2, str(tmp_path / "pg_init"), net), nprocs=2, join=True
    )


class _AddLayer(nn.Module):
    """Stand-in decoder layer: output = input + layer_number (1-based, global)."""

    def __init__(self, layer_number: int):
        super().__init__()
        self.layer_number = layer_number

    def forward(self, x):
        return x + float(self.layer_number)


class _FakeTrunk(nn.Module):
    def __init__(self, layer_numbers, with_embedding: bool):
        super().__init__()
        self.config = SimpleNamespace(num_layers=4)
        self.decoder = SimpleNamespace(
            layers=nn.ModuleList([_AddLayer(n) for n in layer_numbers])
        )
        if with_embedding:
            self.embedding = nn.Identity()


# layer_number (1-based) per stage; PP=4 mirrors the dflash-8b shape scaled
# down: uneven remote taps (1/2/0 layers), an inactive middle source stage,
# and one tap local to the draft stage. Embeds always come from stage 0.
_CAPTURE_LAYERS = {
    2: {0: (1, 2), 1: (3, 4)},
    4: {0: (1, 2), 1: (3, 4), 2: (5, 6), 3: (7, 8)},
}
_CAPTURE_AUX = {2: (1, 3), 4: (1, 2, 3, 7)}


def _stage_chain(layer_numbers, x):
    """Replay a stage's _AddLayer chain bf16-exactly: {global_layer_idx: output}."""
    outputs, h = {}, x
    for n in layer_numbers:
        h = h + float(n)
        outputs[n - 1] = h
    return outputs


def _capture_worker(rank: int, world: int, init_file: str, net) -> None:
    _init_pp(rank, world, init_file)
    layers = _CAPTURE_LAYERS[world]
    aux = _CAPTURE_AUX[world]
    local_ids = [n - 1 for n in layers[rank] if n - 1 in aux]
    channel = _make_channel(
        rank, local_layer_ids=local_ids, aux_layer_ids=list(aux), net=net
    )
    trunk = _FakeTrunk(layers[rank], with_embedding=rank == 0)
    capture = HiddenStateCapture(
        model=trunk, aux_layer_indices=aux, tap_channel=channel
    )
    for i, (seq_len, batch) in enumerate(MB_SHAPES[:3]):
        x = _pattern(i + 50.0 * rank, seq_len, batch, HIDDEN)  # this stage's input
        with capture.capture_context():
            h = trunk.embedding(x) if rank == 0 else x
            for layer in trunk.decoder.layers:
                h = layer(h)
        states = capture.get_captured_states()
        if rank < world - 1:
            assert states.hidden_states is None and states.inputs_embeds is None
        else:
            x0 = _pattern(i, seq_len, batch, HIDDEN)
            torch.testing.assert_close(states.inputs_embeds, x0)
            owner = {n - 1: r for r, nums in layers.items() for n in nums}
            chains = {
                r: _stage_chain(
                    layers[r], _pattern(i + 50.0 * r, seq_len, batch, HIDDEN)
                )
                for r in {owner[g] for g in aux}
            }
            expected = torch.cat([chains[owner[g]][g] for g in sorted(aux)], dim=-1)
            torch.testing.assert_close(states.hidden_states, expected)
    _teardown()


@requires_2_gpus
@pytest.mark.parametrize("net", [None, [0]], ids=["ipc", "net"])
def test_capture_pp2_matches_local_semantics(net, tmp_path):
    mp.spawn(
        _capture_worker, args=(2, str(tmp_path / "pg_init"), net), nprocs=2, join=True
    )


@requires_4_gpus
@pytest.mark.parametrize("net", [None, [0, 1]], ids=["ipc", "net"])
def test_capture_pp4_uneven_and_inactive_source(net, tmp_path):
    mp.spawn(
        _capture_worker, args=(4, str(tmp_path / "pg_init"), net), nprocs=4, join=True
    )


def _mask_row_worker(rank: int, world: int, init_file: str) -> None:
    _init_pp(rank, world, init_file)
    channel = _make_channel(
        rank,
        local_layer_ids=[1] if rank == 0 else [3],
        needs_mask_row=True,
        mask_token_id=5,
    )
    torch.manual_seed(7)  # same weights on both ranks for the reference
    table = nn.Embedding(11, HIDDEN).cuda().to(torch.bfloat16)
    chunk = SimpleNamespace(embedding=SimpleNamespace(word_embeddings=table))
    channel.begin_pass(chunk if rank == 0 else None)
    torch.testing.assert_close(channel.mask_row, table.weight[5].detach())
    _teardown()


@requires_2_gpus
def test_begin_pass_broadcasts_mask_row(tmp_path):
    mp.spawn(_mask_row_worker, args=(2, str(tmp_path / "pg_init")), nprocs=2, join=True)


def test_resolve_draft_aux_layer_ids_cpu():
    """The tap channel and the builders must agree on the aux ids on every rank."""
    explicit = {"speculator_type": "dflash", "aux_layer_indices": [1, 9, 17]}
    assert resolve_draft_aux_layer_ids(explicit, 36) == (1, 9, 17)
    defaults = {"speculator_type": "eagle3"}
    assert resolve_draft_aux_layer_ids(defaults, 36) == (1, 17, 32)
