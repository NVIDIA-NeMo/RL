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

"""TrainingEngine subclass that runs on NeMo RL's microbatch processing and loss."""

from typing import Any

import megatron.bridge.engine.recipes as engine_recipes
import torch
from megatron.bridge.engine import (
    ModelOutput,
    PreparedMicrobatch,
    TrainingEngine,
    TrainingRecipe,
)
from megatron.bridge.training.utils.packed_seq_utils import (
    get_thd_cp_partition_indices,
)

from nemo_rl.algorithms.loss import ClippedPGLossFn
from nemo_rl.algorithms.loss.interfaces import LossFunction, LossType
from nemo_rl.algorithms.loss.wrapper import SequencePackingFusionLossWrapper
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import allgather_cp_sharded_tensor
from nemo_rl.models.megatron.data import ProcessedInputs, process_microbatch
from nemo_rl.models.policy.training_engine_adapter.config import TrainingEngineConfig
from nemo_rl.utils.sequence_lengths import CpuIntTuple, to_cpu_int_tuple

# Metrics the loss returns that are not sums over tokens, so they cannot be added across
# microbatches and ranks: its own total, a per-rank sample count, a mean with a separate
# denominator, and extrema.
# TODO: reduce the ratio extrema with min and max instead of dropping them.
_NOT_ADDITIVE = frozenset(
    {
        "loss",
        "num_valid_samples",
        "positive_nll_loss",
        "probs_ratio_min",
        "probs_ratio_max",
        "probs_ratio_clamped_min",
        "probs_ratio_clamped_max",
    }
)


def prepare_packed_logprobs(
    logits: torch.Tensor,
    data: BatchedDataDict[Any],
    loss_fn: LossFunction | None,
    cu_seqlens_q: CpuIntTuple,
    cu_seqlens_q_padded: CpuIntTuple,
    vocab_parallel_rank: int | None = None,
    vocab_parallel_group: torch.distributed.ProcessGroup | None = None,
    context_parallel_group: torch.distributed.ProcessGroup | None = None,
) -> tuple[dict[str, Any], BatchedDataDict[Any]]:
    """Whole-sequence [B, S-1] logprobs from the logprobs the engine's model returns.

    The prepare step of NeMo RL's packed loss wrapper, with the signature of
    ``prepare_packed_loss_input``. That one turns packed logits into logprobs; the
    engine's model already returns one logprob per packed position this rank holds
    (passed as ``logits``, [1, T // CP]), so only the second half is left: gather each
    sequence across context-parallel ranks, keeping gradients, and put it on its row.
    """
    cp_group = context_parallel_group
    cp_size = 1 if cp_group is None else torch.distributed.get_world_size(cp_group)
    packed = logits.squeeze(0).to(torch.float32)
    num_rows, width = data["input_ids"].shape
    logprobs = packed.new_zeros((num_rows, width - 1))
    for row, length in enumerate(data["input_lengths"].tolist()):
        start, end = cu_seqlens_q_padded[row], cu_seqlens_q_padded[row + 1]
        sequence = packed[start // cp_size : end // cp_size]
        if cp_size > 1:
            sequence = allgather_cp_sharded_tensor(sequence, cp_group, seq_dim=0)
        # Position t predicts token t + 1; the last token and the padding predict nothing.
        logprobs[row, : length - 1] = sequence[: length - 1]
    return {"next_token_logprobs": logprobs}, data


def check_supported_loss(loss_fn: LossFunction) -> None:
    """Reject loss options the TrainingEngine policy has not been verified with."""
    if not isinstance(loss_fn, ClippedPGLossFn):
        raise NotImplementedError(
            "The TrainingEngine policy supports ClippedPGLossFn only, got "
            f"{type(loss_fn).__name__}."
        )
    unsupported = {
        "token_level_loss=false": loss_fn.loss_type != LossType.TOKEN_LEVEL,
        "sequence_level_importance_ratios": loss_fn.sequence_level_importance_ratios,
        "seq_logprob_error_in_loss": loss_fn.seq_logprob_error_in_loss,
        "truncated_importance_sampling_type=seq-mask-tis": (
            loss_fn.truncated_importance_sampling_type == "seq-mask-tis"
        ),
        "score_centering": loss_fn.score_centering,
        "opd_full": loss_fn.opd_full is not None,
        "positive_example_nll_weight": loss_fn.positive_example_nll_weight > 0,
    }
    enabled = [name for name, on in unsupported.items() if on]
    if enabled:
        # TODO: the loss sees whole sequences, but these divide by something other than
        # the token count the engine normalizes with (sequences, filter survivors, a
        # second term's own tokens), or need the model's logits.
        raise NotImplementedError(
            f"The TrainingEngine policy does not support loss_fn options: {', '.join(enabled)}."
        )


def build_recipe(engine_config: TrainingEngineConfig) -> TrainingRecipe:
    """The recipe ``policy.training_engine`` names, with its overrides applied."""
    recipe = getattr(engine_recipes, engine_config["recipe"])(
        seq_length=engine_config["seq_length"], **engine_config["recipe_kwargs"]
    )
    engine_recipes.apply_recipe_overrides(recipe, engine_config["overrides"])
    return recipe


def _unpacked(
    microbatch: PreparedMicrobatch,
) -> tuple[BatchedDataDict[Any], CpuIntTuple]:
    """What the loss and the score gathering need to get back from a packed microbatch.

    Returns:
        The microbatch as NeMo RL holds it (right-padded [B, S] fields, on the GPU), and
        where each sample starts in the packed row before the context-parallel split,
        with the end of the last one.
    """
    (rows,) = microbatch.sample_inputs
    return rows, to_cpu_int_tuple(microbatch.packed_seq_params.cu_seqlens_q_padded)


class NeMoRLTrainingEngine(TrainingEngine):
    """TrainingEngine fed NeMo RL batches, with NeMo RL's packing and loss.

    By default the engine receives the global batch as Datums on every rank, plans the
    microbatches, deals them to data-parallel replicas and packs them. NeMo RL plans on
    the driver and sends each replica only its own samples, so the data hooks are
    overridden: the engine takes a NeMo RL batch with its plan and packs each planned
    microbatch with NeMo RL's own microbatch processing. Only the model inputs are
    packed; the loss reads NeMo RL's fields where they are.
    """

    # Each sample is padded to this multiple when packed. It is the policy's
    # make_sequence_length_divisible_by, which the driver's plan rounds with too.
    sample_pad_multiple: int

    @classmethod
    def from_recipe(
        cls, recipe: TrainingRecipe, *, sample_pad_multiple: int, **kwargs: Any
    ) -> "NeMoRLTrainingEngine":
        """Build the engine as ``TrainingEngine.from_recipe`` does and set its sample padding."""
        engine = super().from_recipe(recipe, **kwargs)
        engine.sample_pad_multiple = sample_pad_multiple
        return engine

    def preprocess(
        self, data: BatchedDataDict[Any], *, training: bool
    ) -> list[PreparedMicrobatch]:
        """Pack each planned microbatch and keep this rank's context-parallel slice.

        Args:
            data: This replica's samples with the driver's packing plan, the same on
                every rank of the replica.
            training: Select the positions NeMo RL's masks select (True) or every
                position that has a target (False).
        """
        model = self.recipe.model
        microbatches = []
        for rows in data.make_microbatch_iterator_for_packable_sequences():
            if rows.size == 0:
                raise ValueError("NeMo RL planned a microbatch with no samples.")
            rows = rows.to("cuda")
            packed = process_microbatch(
                rows,
                seq_length_key="input_lengths",
                pad_individual_seqs_to_multiple_of=self.sample_pad_multiple,
                pad_full_seq_to=(
                    model.seq_length if self.recipe.pad_to_max_length else None
                ),
                pack_sequences=True,
                create_packed_seq_padding_mask=True,
                # Makes it build the packed positions, which the engine always passes
                # to the model.
                mtp_enabled=True,
            )
            microbatches.append(self._prepared(rows, packed, training=training))
        return microbatches

    def _prepared(
        self, rows: BatchedDataDict[Any], packed: ProcessedInputs, *, training: bool
    ) -> PreparedMicrobatch:
        """Build the engine's microbatch from one microbatch NeMo RL has packed.

        NeMo RL's packing gives the model inputs. The engine scores targets inside the
        model and counts loss tokens itself, so this adds ``target_ids`` and
        ``loss_mask``, packed and context-parallel split exactly like the input tokens.

        Shapes (B samples in the microbatch, S their padded width, T tokens once packed):

        - ``rows``, not packed: ``rows["input_ids"]`` and ``rows["token_mask"]`` are
          [B, S], one right-padded sample per row; ``rows["input_lengths"]`` and
          ``rows["sample_mask"]`` are [B].
        - ``packed.input_ids`` is [1, T]: the B samples one after another in a single
          row, each padded to the engine's pad multiple. Sample ``i`` starts at
          ``packed.cu_seqlens_padded[i]`` ([B + 1]).
        - ``packed.input_ids_cp_sharded``, ``packed.position_ids`` and
          ``packed.padding_mask`` are [1, T // CP]: the part of that row this
          context-parallel rank holds.
        - Every field of the returned microbatch is [1, T // CP].

        Args:
            rows: One planned microbatch as NeMo RL holds it.
            packed: What NeMo RL's ``process_microbatch`` made of ``rows``.
            training: Take the loss mask from NeMo RL's masks (True) or select every
                position that has a target (False).
        """
        device = packed.input_ids.device
        total_tokens = packed.input_ids.shape[1]
        # The packed-row positions this rank holds, in the order the model sees them.
        cp_group = self.process_groups.cp
        if cp_group is None or cp_group.size() == 1:
            cp_indices = torch.arange(total_tokens, device=device)
        else:
            cp_indices = get_thd_cp_partition_indices(
                packed.cu_seqlens_padded,
                total_tokens=total_tokens,
                cp_group=cp_group,
                device=device,
            )

        width = rows["input_ids"].shape[1]
        # [T]: for each packed position, where in the flattened [B, S] fields of ``rows``
        # (index row * S + column) the token it must predict is: the next token of the
        # same sample. -1 where there is none (padding and each sample's last token).
        target_index = torch.full((total_tokens,), -1, device=device)
        offsets = packed.cu_seqlens_padded.tolist()
        for row, length in enumerate(rows["input_lengths"].tolist()):
            target_index[offsets[row] : offsets[row] + length - 1] = (
                row * width + torch.arange(1, length, device=device)
            )
        # [T // CP]: the same for this rank's positions.
        target_index = target_index[cp_indices]
        has_target = target_index >= 0
        source = target_index.clamp(min=0)

        tokens = rows["input_ids"].flatten().to(torch.int64)
        input_ids = packed.input_ids_cp_sharded.to(torch.int64)
        # The input at a position must be the token just before its target. This fails
        # if NeMo RL's packing or split ever differs from the index map above.
        if not torch.equal(input_ids[0][has_target], tokens[source[has_target] - 1]):
            raise ValueError(
                "NeMo RL's packed row does not have the layout the TrainingEngine "
                "policy maps targets with."
            )
        # Read the [B, S] fields of ``rows`` through the index map: this both packs them
        # and keeps this rank's positions.
        if training:
            selected = rows["token_mask"] * rows["sample_mask"].unsqueeze(-1)
            loss_mask = selected.flatten()[source].to(torch.float32) * has_target
        else:
            loss_mask = has_target.to(torch.float32)
        # mcore-contract: All predefined tensor fields in PreparedMicrobatch are CP sharded
        return PreparedMicrobatch(
            input_ids=input_ids,
            target_ids=(tokens[source] * has_target).unsqueeze(0),
            position_ids=packed.position_ids,
            loss_mask=loss_mask.unsqueeze(0),
            padding_mask=packed.padding_mask,
            packed_seq_params=packed.packed_seq_params,
            # Not packed and not split: the loss and the score gathering read NeMo RL's
            # own microbatch. One entry for the whole microbatch, not one per sample.
            sample_inputs=[rows],
        )

    def postprocess_forward_only(
        self,
        microbatches: list[PreparedMicrobatch],
        logprobs: list[torch.Tensor],
        *,
        gather_output: bool,
    ) -> torch.Tensor:
        """This replica's scores in NeMo RL's layout, in the order the samples came in.

        Gathers each sequence across context-parallel ranks with the prepare step the
        loss uses. Nothing is collected across data-parallel replicas: NeMo RL does that
        on the driver.

        Returns:
            [B, S] float32 on the GPU: position i holds the logprob of token i; the first
            token's and the padding's are 0.
        """
        if not gather_output:
            # TODO: return only the positions this context-parallel rank holds.
            raise NotImplementedError(
                "The TrainingEngine policy always gathers scores across context-parallel ranks."
            )
        scores = []
        for microbatch, packed in zip(microbatches, logprobs):
            rows, cu_seqlens_padded = _unpacked(microbatch)
            loss_input, _ = prepare_packed_logprobs(
                logits=packed.unsqueeze(0),
                data=rows,
                loss_fn=None,
                cu_seqlens_q=cu_seqlens_padded,
                cu_seqlens_q_padded=cu_seqlens_padded,
                context_parallel_group=self.process_groups.cp,
            )
            scores.append(loss_input["next_token_logprobs"])
        scores = torch.cat(scores)
        return torch.cat([torch.zeros_like(scores[:, :1]), scores], dim=1)

    def clipped_pg_loss(
        self,
        batch: PreparedMicrobatch,
        output: ModelOutput,
        *,
        loss_fn: ClippedPGLossFn,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Token-level clipped policy-gradient loss, summed over the microbatch.

        Runs as in NeMo RL's Megatron policy: its packed loss wrapper first gathers the
        logprobs into whole sequences, then calls the loss on NeMo RL's own microbatch.
        Every context-parallel rank therefore computes the same loss, and the gather
        adds their gradients up, so the loss and metrics are divided by the number of
        those ranks. Global counts of 1 turn the loss's means into sums; the engine
        divides the accumulated gradient by the global token count once per update.
        """
        rows, cu_seqlens_padded = _unpacked(batch)
        cp_group = self.process_groups.cp
        whole_sequence_loss = SequencePackingFusionLossWrapper(
            loss_fn=loss_fn,
            prepare_fn=prepare_packed_logprobs,
            cu_seqlens_q=cu_seqlens_padded,
            cu_seqlens_q_padded=cu_seqlens_padded,
            context_parallel_group=cp_group,
        )
        device = output.logprobs.device
        one = torch.ones((), device=device)
        loss, metrics = whole_sequence_loss(output.logprobs, rows, one, one)
        cp_size = 1 if cp_group is None else cp_group.size()
        sums = {
            name: torch.as_tensor(value, dtype=torch.float64, device=device) / cp_size
            for name, value in metrics.items()
            if name not in _NOT_ADDITIVE
        }
        return loss / cp_size, sums
