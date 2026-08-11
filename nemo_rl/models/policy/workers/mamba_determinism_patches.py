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

"""Runtime Mamba inference patches for zero train/generation mismatch.

This ports the final ``mamba_mixer.py`` behavior from YigongQin/Megatron-LM
commits 41fce34, 6e08a0d, d3f7bea, and 0f78b26. The patch is intentionally
kept in NeMo-RL so zero-KL runs do not require a custom Megatron-LM fork.
"""

import importlib
import inspect
from types import ModuleType
from typing import Any, Callable, Mapping, Optional

import torch
import torch.nn.functional as F

_MAMBA_MIXER_MODULE: Optional[ModuleType] = None
_ORIGINAL_SSM_PREFILL: Optional[Callable[..., torch.Tensor]] = None
_ORIGINAL_SSM_DECODE: Optional[Callable[..., torch.Tensor]] = None

_EXPECTED_PREFILL_PARAMETERS = (
    "self",
    "zxBCdt",
    "conv_state",
    "ssm_state",
    "seq_idx",
    "cu_seqlens",
    "batch_indices",
    "intermediate_chunk_indices",
    "intermediate_abs_positions",
    "intermediate_ssm_out",
    "intermediate_conv_out",
    "conv_gather_offsets",
    "cu_chunk_seqlens",
    "last_chunk_indices",
    "seq_idx_for_varlen",
    "cu_seqlens_list",
    "real_token_count",
    "conv_seq_idx",
    "conv_seq_start",
)
_EXPECTED_DECODE_PARAMETERS = (
    "self",
    "zxBCdt",
    "conv_state",
    "ssm_state",
    "batch_indices",
    "intermediate_conv_state",
    "intermediate_ssm_state",
)
_HELPER_METHODS = {
    "_ssm_prefill_reference": "_nrl_ssm_prefill_reference",
    "_bik_ensure_chunk_buffers": "_nrl_bik_ensure_chunk_buffers",
    "_bik_seed_decode_buffers": "_nrl_bik_seed_decode_buffers",
    "_bik_decode_buffered_scan": "_nrl_bik_decode_buffered_scan",
    "_bik_decode_conv_reference": "_nrl_bik_decode_conv_reference",
}


def _mamba_module() -> ModuleType:
    if _MAMBA_MIXER_MODULE is None:
        raise RuntimeError("Mamba determinism patch has not been installed")
    return _MAMBA_MIXER_MODULE


def _nrl_bik_ensure_chunk_buffers(
    self: Any,
    max_batch: int,
    nh: int,
    p: int,
    ng: int,
    n: int,
    device: torch.device,
    x_dtype: torch.dtype,
    dt_dtype: torch.dtype,
    B_dtype: torch.dtype,
    C_dtype: torch.dtype,
    z_dtype: torch.dtype,
) -> None:
    """Lazily allocate per-slot buffers for training-kernel decode scans."""
    if hasattr(self, "_bik_chunk_x_buf"):
        return

    buf_len = self.chunk_size
    self._bik_chunk_x_buf = torch.zeros(
        max_batch, buf_len, nh, p, device=device, dtype=x_dtype
    )
    self._bik_chunk_dt_buf = torch.zeros(
        max_batch, buf_len, nh, device=device, dtype=dt_dtype
    )
    self._bik_chunk_B_buf = torch.zeros(
        max_batch, buf_len, ng, n, device=device, dtype=B_dtype
    )
    self._bik_chunk_C_buf = torch.zeros(
        max_batch, buf_len, ng, n, device=device, dtype=C_dtype
    )
    self._bik_chunk_z_buf = torch.zeros(
        max_batch, buf_len, nh, p, device=device, dtype=z_dtype
    )
    self._bik_chunk_count = torch.zeros(max_batch, device=device, dtype=torch.int32)
    self._bik_state_is_zero = torch.zeros(max_batch, device=device, dtype=torch.bool)


def _nrl_bik_seed_decode_buffers(
    self: Any,
    x: torch.Tensor,
    dt: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    z: torch.Tensor,
    cu_seqlens: torch.Tensor,
    batch_indices: Optional[torch.Tensor],
    ssm_state: torch.Tensor,
) -> None:
    """Seed decode buffers with each prefill request's partial chunk tail."""
    max_batch = ssm_state.shape[0]
    nh, p = x.shape[-2:]
    ng, n = B.shape[-2:]
    self._bik_ensure_chunk_buffers(
        max_batch,
        nh,
        p,
        ng,
        n,
        x.device,
        x.dtype,
        dt.dtype,
        B.dtype,
        C.dtype,
        z.dtype,
    )
    z_flat = z.squeeze(0) if z.dim() == 4 else z
    num_prefill = cu_seqlens.numel() - 1
    for request_idx in range(num_prefill):
        slot = (
            int(batch_indices[request_idx].item())
            if batch_indices is not None
            else request_idx
        )
        if slot < 0:
            continue
        start = int(cu_seqlens[request_idx].item())
        end = int(cu_seqlens[request_idx + 1].item())
        prefill_len = end - start
        if prefill_len < self.chunk_size:
            tail = prefill_len
            self._bik_state_is_zero[slot] = True
        else:
            tail = prefill_len % self.chunk_size
            self._bik_state_is_zero[slot] = False
        self._bik_chunk_count[slot] = tail
        if tail > 0:
            tail_start = end - tail
            self._bik_chunk_x_buf[slot, :tail] = x[tail_start:end]
            self._bik_chunk_dt_buf[slot, :tail] = dt[tail_start:end]
            self._bik_chunk_B_buf[slot, :tail] = B[tail_start:end]
            self._bik_chunk_C_buf[slot, :tail] = C[tail_start:end]
            self._bik_chunk_z_buf[slot, :tail] = z_flat[tail_start:end]


def _nrl_ssm_prefill_reference(
    self: Any,
    *,
    z: torch.Tensor,
    xBC: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_list: Optional[list[int]],
    batch_indices: Optional[torch.Tensor],
    conv_state: Optional[torch.Tensor],
    ssm_state: Optional[torch.Tensor],
    intermediate_ssm_out: Optional[torch.Tensor],
) -> torch.Tensor:
    """Run dynamic prefill through the same conv and scan kernels as training."""
    if intermediate_ssm_out is not None:
        raise NotImplementedError(
            "batch_invariant_mode reference prefill does not support Mamba "
            "prefix caching (set enable_prefix_caching=false)."
        )

    module = _mamba_module()
    chunk_scan = module.mamba_chunk_scan_combined
    if chunk_scan is None:
        raise RuntimeError(
            "mamba_ssm is required for the batch-invariant reference prefill"
        )
    if self.cp.cp_size != 1 and not self.rmsnorm:
        raise NotImplementedError(
            "Context parallel is unsupported for reference prefill when rmsnorm=False"
        )

    batch, total_sequence_length, _ = xBC.shape
    if batch != 1:
        raise ValueError(
            "dynamic-batching prefill expects a flattened [1, tokens, hidden] layout"
        )
    cumulative_lengths = (
        cu_seqlens_list
        if cu_seqlens_list is not None
        else [int(value) for value in cu_seqlens.tolist()]
    )
    num_sequences = len(cumulative_lengths) - 1
    nheads = self.cp.nheads_local_tpcp
    ngroups = self.cp.ngroups_local_tpcp
    rearrange = module.rearrange

    D = (
        rearrange(self.cp.get_D().float(), "(h p) -> h p", p=self.headdim)
        if self.D_has_hdim
        else self.cp.get_D()
    )
    dt_bias = self.cp.get_dt_bias().float()
    conv_weight = rearrange(self.cp.get_conv1d_weight(), "d 1 w -> d w")
    conv_bias = self.cp.get_conv1d_bias()

    y_flat = torch.zeros(
        total_sequence_length,
        nheads,
        self.headdim,
        device=xBC.device,
        dtype=xBC.dtype,
    )
    x_scan = torch.zeros_like(y_flat)
    z_scan = torch.zeros_like(y_flat)
    B_scan = torch.zeros(
        total_sequence_length,
        ngroups,
        self.d_state,
        device=xBC.device,
        dtype=xBC.dtype,
    )
    C_scan = torch.zeros_like(B_scan)
    dt_flat = dt.squeeze(0).contiguous()

    for request_idx in range(num_sequences):
        start = int(cumulative_lengths[request_idx])
        end = int(cumulative_lengths[request_idx + 1])
        if end <= start:
            continue
        slot = (
            int(batch_indices[request_idx].item())
            if batch_indices is not None
            else request_idx
        )
        if slot < 0:
            continue
        sequence_length = end - start

        xBC_request = rearrange(xBC[:, start:end], "b l d -> b d l").contiguous()
        if conv_state is not None:
            conv_state[slot].copy_(
                F.pad(
                    xBC_request,
                    (self.d_conv - xBC_request.shape[-1], 0),
                ).squeeze(0)
            )
        if module.causal_conv1d_fn is None:
            xBC_convolved = self.act(self.cp.conv1d(xBC_request)[..., :sequence_length])
        else:
            if self.activation not in ("silu", "swish"):
                raise ValueError(
                    f"Unsupported Mamba activation for reference conv: {self.activation}"
                )
            xBC_convolved = module.causal_conv1d_fn(
                x=xBC_request,
                weight=conv_weight,
                bias=conv_bias,
                activation=self.activation,
            )
        xBC_convolved = rearrange(xBC_convolved, "b d l -> b l d").contiguous()

        x_request, B_request, C_request = torch.split(
            xBC_convolved,
            [
                self.cp.d_inner_local_tpcp,
                ngroups * self.d_state,
                ngroups * self.d_state,
            ],
            dim=-1,
        )
        x_request = rearrange(
            x_request, "b l (h p) -> b l h p", p=self.headdim
        ).contiguous()
        B_request = rearrange(
            B_request, "b l (g n) -> b l g n", n=self.d_state
        ).contiguous()
        C_request = rearrange(
            C_request, "b l (g n) -> b l g n", n=self.d_state
        ).contiguous()
        z_request = rearrange(
            z[:, start:end], "b l (h p) -> b l h p", p=self.headdim
        ).contiguous()
        dt_request = dt[:, start:end].contiguous()

        y_request = chunk_scan(
            x_request,
            dt_request,
            A,
            B_request,
            C_request,
            self.chunk_size,
            D=D,
            z=z_request if not self.rmsnorm else None,
            dt_bias=dt_bias,
            dt_softplus=True,
            return_final_states=ssm_state is not None,
            initial_states=None,
        )
        if ssm_state is not None:
            y_request, last_state = y_request
            tail = (
                sequence_length % self.chunk_size
                if sequence_length >= self.chunk_size
                else sequence_length
            )
            if sequence_length >= self.chunk_size and tail > 0:
                boundary = sequence_length - tail
                _, boundary_state = chunk_scan(
                    x_request[:, :boundary],
                    dt_request[:, :boundary],
                    A,
                    B_request[:, :boundary],
                    C_request[:, :boundary],
                    self.chunk_size,
                    D=D,
                    z=(z_request[:, :boundary] if not self.rmsnorm else None),
                    dt_bias=dt_bias,
                    dt_softplus=True,
                    return_final_states=True,
                    initial_states=None,
                )
                ssm_state[slot].copy_(boundary_state.squeeze(0))
            else:
                ssm_state[slot].copy_(last_state.squeeze(0))

        y_flat[start:end] = y_request.squeeze(0)
        x_scan[start:end] = x_request.squeeze(0)
        B_scan[start:end] = B_request.squeeze(0)
        C_scan[start:end] = C_request.squeeze(0)
        z_scan[start:end] = z_request.squeeze(0)

    if ssm_state is not None:
        self._bik_seed_decode_buffers(
            x_scan,
            dt_flat,
            B_scan,
            C_scan,
            z_scan,
            cu_seqlens,
            batch_indices,
            ssm_state,
        )

    y = rearrange(y_flat.unsqueeze(0), "b l h p -> l b (h p)").contiguous()
    y = self.cp.post_conv_ssm(y)
    if self.rmsnorm:
        z_transposed = rearrange(z, "b l d -> l b d").contiguous()
        z_transposed = self.cp.post_conv_ssm(z_transposed)
        y = self.norm(y, z_transposed)
    return y


def _nrl_bik_decode_buffered_scan(
    self: Any,
    x: torch.Tensor,
    dt: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    z: Optional[torch.Tensor],
    batch_indices: Optional[torch.Tensor],
    ssm_state: torch.Tensor,
) -> torch.Tensor:
    """Run batched buffered chunk scans matching full-sequence training scans."""
    module = _mamba_module()
    rearrange = module.rearrange
    B = rearrange(B, "b s (g n) -> b s g n", g=self.ngroups_local_tp)
    C = rearrange(C, "b s (g n) -> b s g n", g=self.ngroups_local_tp)
    x = rearrange(x, "b s (h p) -> b s h p", p=self.headdim)
    z = (
        rearrange(z, "b s (h p) -> b s h p", p=self.headdim)
        if z is not None and not self.rmsnorm
        else None
    )

    A = -torch.exp(self.cp.get_A_log().float())
    D = (
        rearrange(self.cp.get_D().float(), "(h p) -> h p", p=self.headdim)
        if self.D_has_hdim
        else self.cp.get_D()
    )
    dt_bias = self.cp.get_dt_bias().float()
    batch_size, sequence_length, nheads, head_dim = x.shape
    ngroups, state_dim = B.shape[-2:]
    if sequence_length != 1:
        raise NotImplementedError(
            "BIK Mamba decode supports one token per request and no speculative decoding"
        )

    self._bik_ensure_chunk_buffers(
        ssm_state.shape[0],
        nheads,
        head_dim,
        ngroups,
        state_dim,
        x.device,
        x.dtype,
        dt.dtype,
        B.dtype,
        C.dtype,
        z.dtype if z is not None else x.dtype,
    )
    slots = (
        batch_indices.to(torch.long)
        if batch_indices is not None
        else torch.arange(batch_size, device=x.device, dtype=torch.long)
    )
    safe_slots = slots.clamp(min=0)
    slots_list = slots.tolist()
    counts_list = self._bik_chunk_count.index_select(0, safe_slots).tolist()
    zero_state_list = self._bik_state_is_zero.index_select(0, safe_slots).tolist()

    active_rows = [index for index, slot in enumerate(slots_list) if slot >= 0]
    y = torch.zeros(
        batch_size,
        1,
        nheads,
        head_dim,
        device=x.device,
        dtype=x.dtype,
    )
    if not active_rows:
        return rearrange(y, "b s h p -> b s (h p)")

    slot_indices = torch.tensor(
        [slots_list[index] for index in active_rows],
        device=x.device,
        dtype=torch.long,
    )
    count_indices = torch.tensor(
        [counts_list[index] for index in active_rows],
        device=x.device,
        dtype=torch.long,
    )
    row_indices = torch.tensor(active_rows, device=x.device, dtype=torch.long)
    self._bik_chunk_x_buf[slot_indices, count_indices] = x[row_indices, 0]
    self._bik_chunk_dt_buf[slot_indices, count_indices] = dt[row_indices, 0]
    self._bik_chunk_B_buf[slot_indices, count_indices] = B[row_indices, 0]
    self._bik_chunk_C_buf[slot_indices, count_indices] = C[row_indices, 0]
    if z is not None:
        self._bik_chunk_z_buf[slot_indices, count_indices] = z[row_indices, 0]

    groups: dict[tuple[int, bool], list[int]] = {}
    for row in active_rows:
        key = (int(counts_list[row]), bool(zero_state_list[row]))
        groups.setdefault(key, []).append(row)

    chunk_scan = module.mamba_chunk_scan_combined
    if chunk_scan is None:
        raise RuntimeError("mamba_ssm is required for BIK Mamba decode")
    for (count, initial_state_is_zero), group_rows in groups.items():
        group_slots = torch.tensor(
            [slots_list[row] for row in group_rows],
            device=x.device,
            dtype=torch.long,
        )
        length = count + 1
        x_buffer = self._bik_chunk_x_buf.index_select(0, group_slots)[
            :, :length
        ].contiguous()
        dt_buffer = self._bik_chunk_dt_buf.index_select(0, group_slots)[
            :, :length
        ].contiguous()
        B_buffer = self._bik_chunk_B_buf.index_select(0, group_slots)[
            :, :length
        ].contiguous()
        C_buffer = self._bik_chunk_C_buf.index_select(0, group_slots)[
            :, :length
        ].contiguous()
        z_buffer = (
            self._bik_chunk_z_buf.index_select(0, group_slots)[:, :length].contiguous()
            if z is not None
            else None
        )
        initial_state = (
            None
            if initial_state_is_zero
            else ssm_state.index_select(0, group_slots).contiguous()
        )
        y_run, new_state = chunk_scan(
            x_buffer,
            dt_buffer,
            A,
            B_buffer,
            C_buffer,
            self.chunk_size,
            D=D,
            z=z_buffer,
            dt_bias=dt_bias,
            dt_softplus=True,
            initial_states=initial_state,
            return_final_states=True,
        )
        group_row_indices = torch.tensor(group_rows, device=x.device, dtype=torch.long)
        y[group_row_indices] = y_run[:, -1:].to(y.dtype)
        if length == self.chunk_size:
            ssm_state.index_copy_(0, group_slots, new_state.to(ssm_state.dtype))
            self._bik_chunk_count.index_fill_(0, group_slots, 0)
            self._bik_state_is_zero.index_fill_(0, group_slots, False)
        else:
            self._bik_chunk_count.index_fill_(0, group_slots, length)

    return rearrange(y, "b s h p -> b s (h p)")


def _nrl_bik_decode_conv_reference(
    self: Any,
    xBC: torch.Tensor,
    conv_state: torch.Tensor,
    batch_indices: Optional[torch.Tensor],
) -> torch.Tensor:
    """Run decode convolution through the full-sequence training kernel."""
    module = _mamba_module()
    batch_size, sequence_length, _ = xBC.shape
    if sequence_length != 1:
        raise NotImplementedError(
            "BIK Mamba decode conv supports one token per request"
        )

    dtype = xBC.dtype
    slots = (
        batch_indices.to(torch.long)
        if batch_indices is not None
        else torch.arange(batch_size, device=xBC.device, dtype=torch.long)
    )
    safe_slots = slots.clamp(min=0)
    active = (slots >= 0).view(-1, 1)
    windows = conv_state.index_select(0, safe_slots)
    windows = torch.cat(
        [
            windows[:, :, 1:],
            xBC[:, 0].to(conv_state.dtype).unsqueeze(-1),
        ],
        dim=-1,
    )
    module.tensor_masked_update(conv_state, slots, windows)

    window_input = windows.to(dtype)
    if module.causal_conv1d_fn is None:
        output = self.act(self.cp.conv1d(window_input)[..., : window_input.shape[-1]])[
            ..., -1
        ]
    else:
        if self.activation not in ("silu", "swish"):
            raise ValueError(
                f"Unsupported Mamba activation for reference conv: {self.activation}"
            )
        output = module.causal_conv1d_fn(
            x=window_input,
            weight=module.rearrange(self.cp.get_conv1d_weight(), "d 1 w -> d w"),
            bias=self.cp.get_conv1d_bias(),
            activation=self.activation,
        )[..., -1]
    output = torch.where(
        active, output.to(dtype), torch.zeros_like(output, dtype=dtype)
    )
    return output.unsqueeze(1)


def _nrl_patched_ssm_prefill(
    self: Any,
    zxBCdt: torch.Tensor,
    conv_state: Optional[torch.Tensor],
    ssm_state: Optional[torch.Tensor],
    seq_idx: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    batch_indices: Optional[torch.Tensor] = None,
    intermediate_chunk_indices: Optional[torch.Tensor] = None,
    intermediate_abs_positions: Optional[torch.Tensor] = None,
    intermediate_ssm_out: Optional[torch.Tensor] = None,
    intermediate_conv_out: Optional[torch.Tensor] = None,
    conv_gather_offsets: Optional[torch.Tensor] = None,
    cu_chunk_seqlens: Optional[torch.Tensor] = None,
    last_chunk_indices: Optional[torch.Tensor] = None,
    seq_idx_for_varlen: Optional[torch.Tensor] = None,
    cu_seqlens_list: Optional[list[int]] = None,
    real_token_count: Optional[int] = None,
    conv_seq_idx: Optional[torch.Tensor] = None,
    conv_seq_start: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Intercept only dynamic BIK prefill; delegate every other path."""
    if not (seq_idx is not None and self.config.batch_invariant_mode):
        if _ORIGINAL_SSM_PREFILL is None:
            raise RuntimeError("Original Mamba prefill method is unavailable")
        return _ORIGINAL_SSM_PREFILL(
            self,
            zxBCdt,
            conv_state,
            ssm_state,
            seq_idx=seq_idx,
            cu_seqlens=cu_seqlens,
            batch_indices=batch_indices,
            intermediate_chunk_indices=intermediate_chunk_indices,
            intermediate_abs_positions=intermediate_abs_positions,
            intermediate_ssm_out=intermediate_ssm_out,
            intermediate_conv_out=intermediate_conv_out,
            conv_gather_offsets=conv_gather_offsets,
            cu_chunk_seqlens=cu_chunk_seqlens,
            last_chunk_indices=last_chunk_indices,
            seq_idx_for_varlen=seq_idx_for_varlen,
            cu_seqlens_list=cu_seqlens_list,
            real_token_count=real_token_count,
            conv_seq_idx=conv_seq_idx,
            conv_seq_start=conv_seq_start,
        )
    if cu_seqlens is None:
        raise ValueError("BIK dynamic prefill requires cu_seqlens")

    module = _mamba_module()
    zxBCdt = module.rearrange(zxBCdt, "l b d -> b l d").contiguous()
    A = -torch.exp(self.cp.get_A_log().float())
    z, xBC, dt = torch.split(
        zxBCdt,
        [
            self.cp.d_inner_local_tpcp,
            self.cp.d_inner_local_tpcp + 2 * self.cp.ngroups_local_tpcp * self.d_state,
            self.cp.nheads_local_tpcp,
        ],
        dim=-1,
    )
    return self._ssm_prefill_reference(
        z=z,
        xBC=xBC,
        dt=dt,
        A=A,
        cu_seqlens=cu_seqlens,
        cu_seqlens_list=cu_seqlens_list,
        batch_indices=batch_indices,
        conv_state=conv_state,
        ssm_state=ssm_state,
        intermediate_ssm_out=intermediate_ssm_out,
    )


def _nrl_patched_ssm_decode(
    self: Any,
    zxBCdt: torch.Tensor,
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
    batch_indices: Optional[torch.Tensor] = None,
    intermediate_conv_state: Optional[torch.Tensor] = None,
    intermediate_ssm_state: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Intercept only BIK decode; delegate the standard MCore path."""
    if not self.config.batch_invariant_mode:
        if _ORIGINAL_SSM_DECODE is None:
            raise RuntimeError("Original Mamba decode method is unavailable")
        return _ORIGINAL_SSM_DECODE(
            self,
            zxBCdt,
            conv_state,
            ssm_state,
            batch_indices=batch_indices,
            intermediate_conv_state=intermediate_conv_state,
            intermediate_ssm_state=intermediate_ssm_state,
        )
    if intermediate_conv_state is not None or intermediate_ssm_state is not None:
        raise NotImplementedError(
            "BIK Mamba decode does not support speculative-decoding rollback buffers"
        )

    z, xBC, dt = torch.split(
        zxBCdt,
        [
            self.d_inner_local_tp,
            self.d_inner_local_tp + 2 * self.ngroups_local_tp * self.d_state,
            self.nheads_local_tp,
        ],
        dim=-1,
    )
    xBC = self._bik_decode_conv_reference(xBC, conv_state, batch_indices)
    x, B, C = torch.split(
        xBC,
        [
            self.d_inner_local_tp,
            self.ngroups_local_tp * self.d_state,
            self.ngroups_local_tp * self.d_state,
        ],
        dim=-1,
    )
    y = self._bik_decode_buffered_scan(x, dt, B, C, z, batch_indices, ssm_state)
    if self.rmsnorm:
        y = self.norm(y, z)
    return y


_MAMBA_MODEL_NAME_MARKERS = (
    "nemotron-nano",
    "nemotron_nano",
    "nemotron-3-nano",
    "nemotron_3_nano",
    "nemotron-3-super",
    "nemotron_3_super",
    "falconh1",
    "falcon-h1",
    "falcon_h1",
)


def policy_uses_mamba_layers(config: Mapping[str, Any]) -> bool:
    """Return True when the policy model includes MambaMixer layers."""
    model_name = str(config.get("model_name", "")).lower()
    if any(marker in model_name for marker in _MAMBA_MODEL_NAME_MARKERS):
        return True
    if "mamba" in model_name and "qwen" not in model_name:
        return True

    megatron_cfg = config.get("megatron_cfg") or {}
    if megatron_cfg.get("is_hybrid_model"):
        return True

    hybrid_layer_pattern = megatron_cfg.get("hybrid_layer_pattern")
    if hybrid_layer_pattern is not None and "M" in str(hybrid_layer_pattern).upper():
        return True

    return False


def _validate_method_signature(
    method: Callable[..., torch.Tensor],
    expected_parameters: tuple[str, ...],
    method_name: str,
) -> None:
    actual_parameters = tuple(inspect.signature(method).parameters)
    if actual_parameters != expected_parameters:
        raise RuntimeError(
            "Cannot install Mamba determinism patch: "
            f"MambaMixer.{method_name} signature changed from "
            f"{expected_parameters} to {actual_parameters}. Rebase the patch "
            "onto the installed Megatron-LM version."
        )


def apply_mamba_determinism_patch(*, required: bool = True) -> None:
    """Install the zero-KL Mamba prefill and decode paths once per process.

    When ``required`` is False (non-Mamba models), signature mismatches are
    logged and the patch is skipped instead of failing worker initialization.
    """
    global _MAMBA_MIXER_MODULE, _ORIGINAL_SSM_DECODE, _ORIGINAL_SSM_PREFILL
    if _ORIGINAL_SSM_PREFILL is not None:
        return

    module = importlib.import_module("megatron.core.ssm.mamba_mixer")
    mixer_class = module.MambaMixer
    original_prefill = mixer_class._ssm_prefill
    original_decode = mixer_class._ssm_decode
    try:
        _validate_method_signature(
            original_prefill, _EXPECTED_PREFILL_PARAMETERS, "_ssm_prefill"
        )
        _validate_method_signature(
            original_decode, _EXPECTED_DECODE_PARAMETERS, "_ssm_decode"
        )
    except RuntimeError as exc:
        if not required:
            print(
                "[zero_train_gen_mismatch] skipping Mamba determinism patch "
                f"(model has no Mamba layers): {exc}",
                flush=True,
            )
            return
        raise
    for method_name in _HELPER_METHODS:
        if hasattr(mixer_class, method_name):
            raise RuntimeError(
                "Cannot install NeMo-RL Mamba determinism patch because "
                f"MambaMixer.{method_name} already exists. The installed "
                "Megatron-LM may already contain an equivalent upstream patch."
            )

    _MAMBA_MIXER_MODULE = module
    _ORIGINAL_SSM_PREFILL = original_prefill
    _ORIGINAL_SSM_DECODE = original_decode
    for method_name, function_name in _HELPER_METHODS.items():
        setattr(mixer_class, method_name, globals()[function_name])
    mixer_class._ssm_prefill = _nrl_patched_ssm_prefill
    mixer_class._ssm_decode = _nrl_patched_ssm_decode
    print(
        "[zero_train_gen_mismatch] installed batch-invariant Mamba "
        "reference prefill/decode paths",
        flush=True,
    )


def restore_mamba_determinism_patch() -> None:
    """Restore original MambaMixer methods for isolated tests."""
    global _MAMBA_MIXER_MODULE, _ORIGINAL_SSM_DECODE, _ORIGINAL_SSM_PREFILL
    if _ORIGINAL_SSM_PREFILL is None or _MAMBA_MIXER_MODULE is None:
        return

    mixer_class = _MAMBA_MIXER_MODULE.MambaMixer
    mixer_class._ssm_prefill = _ORIGINAL_SSM_PREFILL
    mixer_class._ssm_decode = _ORIGINAL_SSM_DECODE
    for method_name, function_name in _HELPER_METHODS.items():
        if getattr(mixer_class, method_name, None) is globals()[function_name]:
            delattr(mixer_class, method_name)
    _ORIGINAL_SSM_PREFILL = None
    _ORIGINAL_SSM_DECODE = None
    _MAMBA_MIXER_MODULE = None
