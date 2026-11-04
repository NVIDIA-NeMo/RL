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
"""Per-worker draft co-training runtimes: loss computation and capture lifecycle."""

import contextlib
import warnings
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor

from nemo_rl.models.automodel.draft.common import DSparkForwardOutput
from nemo_rl.models.automodel.draft.draft_qwen3 import Qwen3DSparkModel
from nemo_rl.models.automodel.draft.eagle3_llama import Eagle3DraftModel
from nemo_rl.models.automodel.draft.hidden_capture import DSparkHiddenCapture
from nemo_rl.models.automodel.draft.loss import compute_dspark_loss
from nemo_rl.models.policy import Eagle3DraftConfig


def anchor_sampling_seed(
    dp_rank: int, global_batch_index: int, microbatch_index: int
) -> int:
    """Deterministic seed for the draft's anchor sampling.

    A pure function of the DP rank and the training progress counters: every
    TP/CP peer of the same DP slice derives the identical seed (the draft is
    replicated across them, so they must sample identical anchors or the
    replicas drift apart), while DP ranks, global batches, and microbatches
    all draw differently. Independent of each rank's ambient RNG state.
    """
    seed = 0x5EED_D5BA
    for value in (dp_rank, global_batch_index, microbatch_index):
        # Splitmix64-style mixing keeps nearby counter values decorrelated.
        seed = (seed ^ (value + 0x9E3779B97F4A7C15)) * 0xBF58476D1CE4E5B9
        seed &= (1 << 63) - 1
    return seed


class _DraftRuntimeBase:
    """Shared per-worker draft co-training state and distributed plumbing.

    Subclasses supply the loss (``compute_loss``) and the layer-id source for
    the hidden capture (``_capture_layer_ids``); everything here is common to
    every drafter family: the option/group state, the microbatch-slot
    normalization, and the teacher-logits stash.
    """

    # Algo family name used in error messages.
    _algo_label = "Draft"

    def __init__(
        self,
        draft_model: nn.Module,
        options: Any,
        loss_weight: float,
        dp_group: Optional[dist.ProcessGroup],
        tp_group: Optional[dist.ProcessGroup] = None,
        cp_group: Optional[dist.ProcessGroup] = None,
    ):
        self.draft_model = draft_model
        self.options = options
        self.loss_weight = float(loss_weight)
        self.dp_group = dp_group
        self.tp_group = tp_group
        # Under context parallelism the captured hiddens, teacher logits, and
        # input_ids are sequence-sharded (load-balanced); compute_loss gathers
        # them to the full sequence and every CP peer runs the identical
        # full-sequence draft forward (grads stay consistent through the
        # draft's dp_cp FSDP mesh, like the TP-replicated case).
        self.cp_group = cp_group
        self.capture: Optional[DSparkHiddenCapture] = None
        self._teacher_logits: Optional[torch.Tensor] = None
        self._num_microbatch_slots: int = 1
        self._dp_rank: int = dist.get_rank(dp_group) if dp_group is not None else 0
        self._global_batch_index: int = 0
        self._microbatch_index: int = 0

    @property
    def _cp_size(self) -> int:
        return dist.get_world_size(self.cp_group) if self.cp_group is not None else 1

    def begin_global_batch(self, num_microbatch_slots: int) -> None:
        """Record this global batch's microbatch-slot count (identical on all ranks).

        The training loop sums per-microbatch losses across gradient
        accumulation, while each draft microbatch loss is a mean over that
        microbatch's (DP-all-reduced) supervised tokens. Dividing every
        microbatch term by the slot count turns the sum into an average of
        per-slot global means, keeping the draft gradient scale independent of
        gbs/mbs — the same effective scale as the policy loss, which normalizes
        each microbatch by the whole-global-batch token denominator instead.
        """
        self._num_microbatch_slots = max(int(num_microbatch_slots), 1)
        self._global_batch_index += 1
        self._microbatch_index = 0

    def _capture_layer_ids(self) -> list[int]:
        raise NotImplementedError

    def attach_capture(self, policy_model: nn.Module) -> None:
        if self.capture is None:
            self.capture = DSparkHiddenCapture(policy_model, self._capture_layer_ids())

    def stash_teacher_logits(self, logits: torch.Tensor, need_clone: bool) -> None:
        """Record the policy's raw logits before any in-place temperature scaling.

        With temperature 1.0 no scaling happens and a detached view is exact and
        free; otherwise the in-place div would corrupt the view, so clone first.

        Under TP the logits arrive as a vocab-sharded DTensor; it is stashed
        as-is and gathered to the full vocab lazily in ``compute_loss`` so the
        full-size tensor does not coexist with the policy-loss peak.
        """
        if not isinstance(logits, DTensor) and self.tp_group is not None:
            if dist.get_world_size(self.tp_group) > 1:
                raise RuntimeError(
                    f"{self._algo_label} co-training with tensor_parallel_size "
                    "> 1 expects the policy logits as a vocab-sharded DTensor, "
                    f"but got a plain {type(logits).__name__}; a bare local "
                    "shard cannot be gathered safely."
                )
        raw = logits.detach()
        self._teacher_logits = raw.clone() if need_clone else raw

    def _require_capture_and_teacher(self) -> torch.Tensor:
        """Common compute_loss preamble: guards plus the stashed teacher."""
        if self.capture is None or not self.capture.active:
            raise RuntimeError(
                f"{self._algo_label} loss requested but hidden capture is not "
                "active; the worker must activate capture around the training "
                "forward."
            )
        if self._teacher_logits is None:
            raise RuntimeError(
                f"{self._algo_label} loss requested but no teacher logits were "
                "stashed for this microbatch."
            )
        return self._teacher_logits


class DSparkRuntime(_DraftRuntimeBase):
    """Per-worker DSpark co-training state: draft model, capture, and loss."""

    _algo_label = "DSpark"

    def __init__(
        self,
        draft_model: Qwen3DSparkModel,
        dspark_options: dict[str, Any],
        loss_weight: float,
        dp_group: Optional[dist.ProcessGroup],
        tp_group: Optional[dist.ProcessGroup] = None,
        cp_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__(
            draft_model=draft_model,
            options=dspark_options,
            loss_weight=loss_weight,
            dp_group=dp_group,
            tp_group=tp_group,
            cp_group=cp_group,
        )

    def _capture_layer_ids(self) -> list[int]:
        return list(self.draft_model.config.target_layer_ids)

    def compute_loss(self, data_dict: Any) -> tuple[torch.Tensor, dict[str, float]]:
        """Run the draft forward on the captured hiddens and compute the DSpark loss."""
        teacher_logits = self._require_capture_and_teacher()
        if isinstance(teacher_logits, DTensor):
            # Vocab-sharded under TP; the draft's teacher gather needs the full
            # vocab dimension.
            teacher_logits = teacher_logits.full_tensor()
        if getattr(self.draft_model, "draft_vocab_size", None) is not None:
            # Reduced-vocab drafts compare distributions over the draft vocab;
            # mapping the teacher through d2t BEFORE the CP allgather shrinks
            # the gathered tensor by vocab_ratio (e.g. 248320 -> 32000). Note
            # d2t stores offsets (target_id = draft_idx + d2t[draft_idx]).
            teacher_logits = teacher_logits.index_select(
                -1, self.draft_model._get_d2t_target_ids().to(teacher_logits.device)
            )
        target_hidden_states = self.capture.collect()

        input_ids = data_dict["input_ids"]
        if isinstance(input_ids, DTensor):
            # Under CP the loss prep wraps the load-balanced LOCAL shard as a
            # DTensor with a (nominal) contiguous Shard(1) placement, so
            # full_tensor() would return a mis-ordered sequence. Take the
            # local shard; the CP branch below restores the true order via
            # allgather_cp_sharded_tensor.
            input_ids = input_ids.to_local()
        loss_mask = data_dict["token_mask"].float()
        if "sample_mask" in data_dict:
            loss_mask = loss_mask * data_dict["sample_mask"].float().unsqueeze(-1)

        if self._cp_size > 1:
            from nemo_rl.distributed.model_utils import allgather_cp_sharded_tensor

            # Captured tensors are sequence-local (load-balanced CP shards);
            # restore the full contiguous sequence on every CP peer. loss_mask
            # comes from the un-sharded data dict and is already full-length.
            teacher_logits = allgather_cp_sharded_tensor(
                teacher_logits, self.cp_group, seq_dim=1
            )
            target_hidden_states = allgather_cp_sharded_tensor(
                target_hidden_states, self.cp_group, seq_dim=1
            )
            if input_ids.size(1) != loss_mask.size(1):
                input_ids = allgather_cp_sharded_tensor(
                    input_ids, self.cp_group, seq_dim=1
                )
            if input_ids.size(1) != loss_mask.size(1) or target_hidden_states.size(
                1
            ) != loss_mask.size(1):
                raise RuntimeError(
                    "DSpark CP gather produced inconsistent sequence lengths: "
                    f"input_ids={input_ids.size(1)}, "
                    f"hiddens={target_hidden_states.size(1)}, "
                    f"loss_mask={loss_mask.size(1)}."
                )

        self._microbatch_index += 1
        anchor_generator = torch.Generator(device=input_ids.device)
        anchor_generator.manual_seed(
            anchor_sampling_seed(
                self._dp_rank, self._global_batch_index, self._microbatch_index
            )
        )
        outputs: DSparkForwardOutput = self.draft_model(
            input_ids=input_ids,
            target_hidden_states=target_hidden_states,
            loss_mask=loss_mask,
            teacher_logits=teacher_logits,
            anchor_generator=anchor_generator,
        )
        loss, terms = compute_dspark_loss(
            outputs=outputs,
            loss_decay_gamma=float(self.options["loss_decay_gamma"]),
            ce_loss_alpha=float(self.options["ce_loss_alpha"]),
            l1_loss_alpha=float(self.options["l1_loss_alpha"]),
            confidence_head_alpha=float(self.options["confidence_loss_alpha"]),
            process_group=self.dp_group,
            return_terms=True,
        )
        # See begin_global_batch: average the DP-global-denominator terms
        # across microbatch slots BEFORE logging, so the summed backward and
        # the logged train/draft_loss both match the policy loss's
        # global-mean gradient scale (grpo.py sums per-microbatch metrics
        # across microbatches and DP ranks by default).
        loss = loss / self._num_microbatch_slots
        metrics = self._terms_to_metrics(terms, self._num_microbatch_slots)

        # Free per-microbatch stash; the next forward re-stashes.
        self._teacher_logits = None
        self.capture.clear()
        return loss, metrics

    @staticmethod
    def _terms_to_metrics(
        terms: dict[str, torch.Tensor], microbatch_slots: int
    ) -> dict[str, float]:
        """Flatten the loss terms into shape-stable scalar metrics.

        ``loss``/``ce_loss``/``l1_loss``/``confidence_loss`` use DP-global
        denominators and are divided by ``microbatch_slots`` to match the
        backward value (see ``compute_loss``). Ratios (tau, accept_rate) are
        emitted as raw ``*_num``/``*_den`` pairs, NOT pre-divided: the
        training-loop metric aggregation (``grpo.py``'s per-key reduction)
        sums every metric not on its small mean-reduction allowlist across
        all microbatches and DP ranks, so a pre-divided per-microbatch ratio
        would be summed into a meaningless value (see
        ``nemo_rl.algorithms.utils.finalize_draft_ratio_metrics``, which the
        training loop calls after that reduction to turn the summed num/den
        pairs back into the correct token-weighted global ratio).
        """
        metrics: dict[str, float] = {
            "draft_loss": float(terms["loss"].item()) / microbatch_slots,
            "draft_ce_loss": float(terms["ce_loss"].item()) / microbatch_slots,
            "draft_tv_loss": float(terms["l1_loss"].item()) / microbatch_slots,
            "draft_conf_loss": float(terms["confidence_loss"].item())
            / microbatch_slots,
            "draft_tau_num": float(terms["tau_num"].item()),
            "draft_tau_den": float(terms["tau_den"].item()),
        }
        # One host transfer per vector instead of one sync per position.
        pos_nums = terms["accept_rate_per_pos_num"].tolist()
        pos_dens = terms["accept_rate_per_pos_den"].tolist()
        for k, (num_k, den_k) in enumerate(zip(pos_nums, pos_dens)):
            metrics[f"draft_accept_rate_num@{k + 1}"] = num_k
            metrics[f"draft_accept_rate_den@{k + 1}"] = den_k
        return metrics


@contextlib.contextmanager
def draft_capture_ctx(runtime: Any):
    """Keep the hidden-capture hooks active for the duration of a train call.

    Works for any draft runtime exposing ``capture`` (dspark/dflash/eagle3).
    """
    assert runtime.capture is not None, "attach_capture must run before training."
    runtime.capture.activate()
    try:
        yield
    finally:
        runtime.capture.deactivate()


def next_token_position_mask(token_mask: torch.Tensor) -> torch.Tensor:
    """Shift a per-TOKEN mask onto the POSITIONS whose logits predict it.

    Rollout ``token_mask[t]`` marks token t as supervised (a response token),
    but logits at position t predict token t+1 (the policy loss paths apply
    ``token_mask[:, 1:]`` for the same reason). The eagle3 TTT forward gates
    loss at logit/teacher positions, so its mask must be
    ``mask[t] = token_mask[t + 1]`` with a zero tail: the final position has
    no next-token label, and the last-prompt-token position (whose label is
    the FIRST response token — exactly where drafting starts) is supervised.
    """
    shifted = torch.zeros_like(token_mask)
    shifted[:, :-1] = token_mask[:, 1:]
    return shifted


def _shift_left_with_zero(tensor: torch.Tensor) -> torch.Tensor:
    """Shift a sequence tensor left by one position along dim 1, zero-tailed.

    ``shifted[:, t] = tensor[:, t + 1]``; the last position is zeroed.
    Matches Automodel's ``Eagle3Target._shift_left_with_zero`` convention
    (``speculative/eagle/target.py``).
    """
    shifted = torch.zeros_like(tensor)
    shifted[:, :-1] = tensor[:, 1:]
    return shifted


class Eagle3Runtime(_DraftRuntimeBase):
    """Per-worker EAGLE3 co-training state: draft model, capture, TTT loss.

    Exposes the same runtime protocol as DSparkRuntime so the worker and
    loss wrapper treat all draft algos uniformly.
    """

    _algo_label = "EAGLE3"

    def __init__(
        self,
        draft_model: Eagle3DraftModel,
        eagle3_options: Eagle3DraftConfig,
        loss_weight: float,
        dp_group: Optional[dist.ProcessGroup],
        tp_group: Optional[dist.ProcessGroup] = None,
        cp_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__(
            draft_model=draft_model,
            options=eagle3_options,
            loss_weight=loss_weight,
            dp_group=dp_group,
            tp_group=tp_group,
            cp_group=cp_group,
        )

    def _capture_layer_ids(self) -> list[int]:
        return list(self.draft_model.target_layer_ids)

    def compute_loss(self, data_dict: Any) -> tuple[torch.Tensor, dict[str, float]]:
        """Run the TTT draft forward on captured hiddens and compute the loss."""
        teacher_logits = self._require_capture_and_teacher()
        if isinstance(teacher_logits, DTensor):
            teacher_logits = teacher_logits.full_tensor()
        # Map the teacher into draft-vocab order before any CP gather so the
        # gathered tensor is draft_vocab wide, not target_vocab wide.
        teacher_logits = teacher_logits.index_select(
            -1, self.draft_model.get_d2t_target_ids().to(teacher_logits.device)
        )

        fused_hidden = self.capture.collect()

        input_ids = data_dict["input_ids"]
        if isinstance(input_ids, DTensor):
            # CP loss prep wraps the load-balanced local shard with a nominal
            # contiguous placement; take the local and restore order below.
            input_ids = input_ids.to_local()
        if "input_lengths" not in data_dict:
            raise RuntimeError(
                "EAGLE3 co-training requires input_lengths in the microbatch to "
                "mark padding for the packed-row document mask."
            )
        input_lengths = data_dict["input_lengths"].to(fused_hidden.device)
        # See next_token_position_mask: the TTT forward gates loss at logit
        # positions (position t supervises token t + 1), not at token indices.
        loss_mask = next_token_position_mask(data_dict["token_mask"].float())
        if "sample_mask" in data_dict:
            loss_mask = loss_mask * data_dict["sample_mask"].float().unsqueeze(-1)

        if self._cp_size > 1:
            from nemo_rl.distributed.model_utils import allgather_cp_sharded_tensor

            teacher_logits = allgather_cp_sharded_tensor(
                teacher_logits, self.cp_group, seq_dim=1
            )
            fused_hidden = allgather_cp_sharded_tensor(
                fused_hidden, self.cp_group, seq_dim=1
            )
            if input_ids.size(1) != loss_mask.size(1):
                input_ids = allgather_cp_sharded_tensor(
                    input_ids, self.cp_group, seq_dim=1
                )

        # vLLM's eagle proposer feeds h[t] + token[t + 1] and predicts
        # token[t + 2] (see llm_base_proposer.py's "Shift the input ids by
        # one token"); shift input_ids and teacher_logits left by one to
        # match that contract -- hidden states stay unshifted since they're
        # the target's own per-position captures. loss_mask needs the same
        # shift on top of next_token_position_mask's existing one.
        input_ids = _shift_left_with_zero(input_ids)
        teacher_logits = _shift_left_with_zero(teacher_logits)
        loss_mask = _shift_left_with_zero(loss_mask)

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Flatten the padded batch into one packed row: document ids separate
        # the sequences in the attention mask (padding marked -1) and
        # per-sequence position ids restart at 1, matching the speculators
        # packed-row layout.
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        valid = positions < input_lengths.unsqueeze(1)
        document_ids = torch.where(
            valid,
            torch.arange(batch_size, device=device).unsqueeze(1),
            torch.full_like(positions.expand(batch_size, -1), -1),
        )
        position_ids = (1 + positions).expand(batch_size, -1)

        total = batch_size * seq_len
        # Every sample occupies a fixed seq_len-wide slot in the packed row
        # (padding inside the slot, at its tail); for the flash_attention_2
        # path each slot is declared as one FlashAttention-varlen "document"
        # of that fixed width -- see Eagle3DraftModel.forward's seq_lens
        # docstring for why that's safe despite the internal padding. Built
        # unconditionally; Eagle3DraftModel ignores it on the eager path.
        seq_lens = torch.full((1, batch_size), seq_len, dtype=torch.long, device=device)

        # Called as self.draft_model(...) (nn.Module.__call__), not a bare
        # method: FSDP2's fully_shard unshard/reshard hooks fire on
        # __call__, and this is the only entry point that runs the WHOLE
        # TTT unroll under a single top-level call (see Eagle3DraftModel).
        terms = self.draft_model(
            fused_hidden_states=fused_hidden.reshape(1, total, -1),
            input_ids=input_ids.reshape(1, total),
            document_ids=document_ids.reshape(1, total),
            loss_mask=(loss_mask > 0.5).reshape(1, total),
            teacher_logits=teacher_logits.reshape(1, total, -1),
            position_ids=position_ids.reshape(1, total),
            ttt_steps=int(self.options.ttt_steps),
            seq_lens=seq_lens,
        )

        decay = float(self.options.ttt_step_loss_decay)
        # One all-reduce over the stacked per-step denominators instead of one
        # tiny collective per TTT step (elementwise sum-reduce is identical).
        dens_global = torch.stack([den.detach() for den in terms.loss_dens])
        if self.dp_group is not None and dist.is_initialized():
            dist.all_reduce(dens_global, group=self.dp_group)
        loss = fused_hidden.new_zeros((), dtype=torch.float32)
        for step, num in enumerate(terms.loss_nums):
            loss = loss + (decay**step) * num / dens_global[step]

        # Metrics: one host transfer for all per-step scalars instead of ~5
        # device syncs per step (see DSparkRuntime._terms_to_metrics).
        stats = (
            torch.stack(
                [
                    torch.stack(terms.loss_nums).detach(),
                    torch.stack(terms.loss_dens).detach(),
                    torch.stack(terms.full_acc_nums),
                    torch.stack(terms.full_acc_dens),
                    torch.stack(terms.cond_acc_nums),
                    torch.stack(terms.cond_acc_dens),
                ]
            )
            .float()
            .tolist()
        )
        loss_nums, loss_dens, full_nums, full_dens, cond_nums, cond_dens = stats
        # Raw num/den pairs, NOT pre-divided: see
        # DSparkRuntime._terms_to_metrics /
        # nemo_rl.algorithms.utils.finalize_draft_ratio_metrics for why
        # (grpo.py sums per-microbatch metrics across microbatches and DP
        # ranks by default; a pre-divided ratio would be summed into a
        # meaningless value).
        metrics: dict[str, float] = {}
        for step in range(len(loss_nums)):
            metrics[f"draft_ttt_loss_num@{step}"] = loss_nums[step]
            metrics[f"draft_ttt_loss_den@{step}"] = loss_dens[step]
            metrics[f"draft_full_acc_num@{step}"] = full_nums[step]
            metrics[f"draft_full_acc_den@{step}"] = full_dens[step]
            metrics[f"draft_cond_acc_num@{step}"] = cond_nums[step]
            metrics[f"draft_cond_acc_den@{step}"] = cond_dens[step]
        # See DSparkRuntime.begin_global_batch: average across microbatch
        # slots so the summed backward -- and the logged train/draft_loss --
        # keep a global-mean gradient scale.
        loss = loss / self._num_microbatch_slots
        metrics["draft_loss"] = float(loss.item())

        self._teacher_logits = None
        self.capture.clear()
        return loss, metrics


def draft_refit_export_name(name: str) -> str:
    """Strip the eagle3 draft's model. prefix so refit keys match what the vLLM drafter expects."""
    return name.removeprefix("model.")


def build_draft_runtime(
    draft_model: nn.Module,
    draft_config: Any,
    policy_model: nn.Module,
    policy_config: dict[str, Any],
    dp_group: Optional[dist.ProcessGroup],
    tp_group: Optional[dist.ProcessGroup] = None,
    cp_group: Optional[dist.ProcessGroup] = None,
) -> "_DraftRuntimeBase":
    """Build the per-worker draft runtime, attach hidden capture to the policy.

    Also validates ``generation.vllm_kwargs.speculative_config.num_speculative_tokens``
    against the drafter's actual proposal count (eagle3's ttt_steps, or the
    block drafter's block_size minus the unsupervised anchor slot for
    dflash) -- this needs the built drafter's own config (block_size /
    sample_from_anchor), so config validation alone cannot do it.
    """
    loss_weight = float(draft_config.loss_weight)
    common_groups = dict(dp_group=dp_group, tp_group=tp_group, cp_group=cp_group)
    if draft_config.speculator_type == "eagle3":
        runtime: "_DraftRuntimeBase" = Eagle3Runtime(
            draft_model=draft_model,
            eagle3_options=draft_config,
            loss_weight=loss_weight,
            **common_groups,
        )
    else:
        runtime = DSparkRuntime(
            draft_model=draft_model,
            dspark_options=draft_config.model_dump(),
            loss_weight=loss_weight,
            **common_groups,
        )
    runtime.attach_capture(policy_model)

    spec_cfg = (
        policy_config.get("generation", {})
        .get("vllm_kwargs", {})
        .get("speculative_config", {})
    ) or {}
    num_spec_tokens = spec_cfg.get("num_speculative_tokens")
    if draft_config.speculator_type == "eagle3":
        expected_spec = int(runtime.options.ttt_steps)
        expected_desc = f"the eagle3 ttt_steps={expected_spec}"
    else:
        # dspark blocks predict at every slot; dflash's anchor slot is an
        # unsupervised bonus token, so it proposes one fewer.
        block_size = int(draft_model.config.block_size)
        sample_from_anchor = bool(
            getattr(draft_model.config, "sample_from_anchor", True)
        )
        expected_spec = block_size if sample_from_anchor else block_size - 1
        expected_desc = (
            f"the draft's proposal count {expected_spec} "
            f"(block_size={block_size}, sample_from_anchor={sample_from_anchor})"
        )
    if num_spec_tokens is not None and int(num_spec_tokens) != expected_spec:
        warnings.warn(
            f"speculative_config.num_speculative_tokens={num_spec_tokens} "
            f"does not match {expected_desc}; the drafter proposes "
            f"{expected_spec} tokens per step."
        )
    return runtime
