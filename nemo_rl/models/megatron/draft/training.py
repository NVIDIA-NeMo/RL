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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Protocol, cast

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import MegatronModule
from megatron.core.utils import unwrap_model
import torch
from torch import Tensor
from torch.distributed._functional_collectives import all_gather_tensor_autograd

from nemo_rl.algorithms.loss.draft import (
    DraftLossStats,
    dflash_projected_vocab_parallel_soft_ce,
)
from nemo_rl.algorithms.loss.loss_functions import DraftCrossEntropyLossFn
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.megatron.draft.block_plan import (
    DFlashBatchPlan,
    DSparkBatchPlan,
    build_dflash_batch_plan,
    build_dspark_batch_plan,
)

from nemo_rl.models.megatron.draft.dflash import DFlashBody, DFlashBodyConfig
from nemo_rl.models.megatron.draft.utils import (
    DRAFT_GRAD_NORM_GROUP,
    build_draft_model,
    export_dflash_weights_to_hf,
    export_dspark_heads_to_hf,
    export_eagle_weights_to_hf,
    get_policy_lm_head_weight,
    load_hf_weights_to_dflash,
    load_hf_weights_to_dspark,
    register_draft_grad_norm_group,
)
from nemo_rl.models.policy.draft_config import (
    DFlashDraftConfig,
    DSparkDraftConfig,
    DraftConfig,
    Eagle3DraftConfig,
)

if TYPE_CHECKING:
    from megatron.bridge.models.model_provider import ModelProviderMixin
    from nemo_rl.models.megatron.draft.hidden_capture import CapturedStates
    from nemo_rl.models.megatron.draft.sequence_layout import DraftSequenceLayout


@dataclass(frozen=True, slots=True)
class DFlashForwardOutput:
    """DFlash hidden output paired with the immutable plan used to produce it."""

    hidden: Tensor
    plan: DFlashBatchPlan
    output_weight: Tensor
    sequence_layout: DraftSequenceLayout | None = None
    selected_teacher_logits: Tensor | None = None


@dataclass(frozen=True, slots=True)
class DSparkForwardOutput:
    """DSpark hidden states and immutable token-alignment metadata."""

    hidden: Tensor
    plan: DSparkBatchPlan
    output_weight: Tensor
    previous_token_ids: Tensor
    hard_labels: Tensor
    adapter: Any
    sequence_layout: DraftSequenceLayout | None = None
    selected_teacher_logits: Tensor | None = None


def _embed_draft_mask_tokens(
    word_embeddings: torch.nn.Module,
    mask_ids: Tensor,
    *,
    tensor_parallel_group: torch.distributed.ProcessGroup | None,
) -> Tensor:
    """Embed selected mask blocks while undoing target sequence parallelism."""
    if not bool(getattr(word_embeddings, "reduce_scatter_embeddings", False)):
        return word_embeddings(mask_ids)
    if tensor_parallel_group is None or not torch.distributed.is_initialized():
        raise RuntimeError(
            "DFlash target-SP mask embeddings require an initialized TP group"
        )

    tp_size = torch.distributed.get_world_size(tensor_parallel_group)
    gamma = mask_ids.shape[1]
    padded_gamma = ((gamma + tp_size - 1) // tp_size) * tp_size
    if padded_gamma != gamma:
        padding = mask_ids[:, :1].expand(-1, padded_gamma - gamma)
        mask_ids = torch.cat((mask_ids, padding), dim=1)

    local_embeddings = word_embeddings(mask_ids)
    gathered_embeddings = all_gather_tensor_autograd(
        local_embeddings,
        gather_dim=0,
        group=tensor_parallel_group,
    )
    if gathered_embeddings.shape[0] != padded_gamma:
        raise RuntimeError(
            "target-SP mask embedding gather did not reconstruct the padded block"
        )
    return gathered_embeddings[:gamma].transpose(0, 1).contiguous()


class DraftTrainingProvider(Protocol):
    """Method-neutral model, objective, plan, and checkpoint training seam."""

    config: DraftConfig
    supports_context_parallel: bool
    supports_sequence_packing: bool
    supports_target_sequence_parallel: bool
    requires_full_cp_local_capture: bool

    def build_model(
        self,
        *,
        model_provider: ModelProviderMixin,
        pg_collection: ProcessGroupCollection,
        policy_model_chunk: MegatronModule,
    ) -> MegatronModule | None:
        """Build the draft model for the policy chunk that owns it."""

    def capture_layer_ids(self) -> tuple[int, ...] | None:
        """Return explicit target layers required by this training method."""

    def prepare_batch(
        self,
        data: BatchedDataDict[Any],
        *,
        optimizer_step: int,
        sequence_layout: DraftSequenceLayout | None = None,
    ) -> DFlashBatchPlan | None:
        """Build a method-specific immutable plan when one is required."""

    def normalization_counts(
        self,
        data: BatchedDataDict[Any],
        *,
        optimizer_step: int,
        sequence_layout: DraftSequenceLayout | None = None,
    ) -> Tensor | None:
        """Return local full-batch objective counts for synchronous training."""

    def forward(
        self,
        *,
        policy_model: MegatronModule,
        draft_model: MegatronModule,
        captured_states: CapturedStates,
        input_ids_cp_local: Tensor,
        attention_mask: Tensor | None,
        data: BatchedDataDict[Any],
        optimizer_step: int,
        sequence_layout: DraftSequenceLayout | None,
        context_parallel_group: torch.distributed.ProcessGroup | None,
        tensor_parallel_group: torch.distributed.ProcessGroup | None,
    ) -> None:
        """Run method-specific draft forward and attach its transient output."""

    def loss_stats(
        self,
        *,
        target_logits: Tensor,
        data: BatchedDataDict[Any],
        prepare_fn: Callable[..., Any],
        vocab_parallel_rank: int | None,
        vocab_parallel_group: torch.distributed.ProcessGroup | None,
        context_parallel_group: torch.distributed.ProcessGroup | None,
    ) -> DraftLossStats:
        """Return raw method-specific objective bins."""

    def export_weights(self, model: MegatronModule) -> list[tuple[str, Tensor]]:
        """Return logical draft-body weights for later runtime adapters."""


DraftSpeculator = DraftTrainingProvider


@dataclass(frozen=True)
class Eagle3Speculator:
    """Current EAGLE-3 draft speculator."""

    supports_context_parallel: ClassVar[bool] = (
        Eagle3DraftConfig.supports_context_parallel
    )
    supports_sequence_packing: ClassVar[bool] = (
        Eagle3DraftConfig.supports_sequence_packing
    )
    supports_target_sequence_parallel: ClassVar[bool] = (
        Eagle3DraftConfig.supports_target_sequence_parallel
    )
    requires_full_cp_local_capture: ClassVar[bool] = False
    config: Eagle3DraftConfig

    def build_model(
        self,
        *,
        model_provider: ModelProviderMixin,
        pg_collection: ProcessGroupCollection,
        policy_model_chunk: MegatronModule,
    ) -> MegatronModule | None:
        """Build the EAGLE-3 draft model with the existing implementation."""
        return build_draft_model(
            model_provider=model_provider,
            draft_config=self.config,
            pg_collection=pg_collection,
            policy_model_chunk=policy_model_chunk,
        )

    def capture_layer_ids(self) -> tuple[int, ...] | None:
        return (
            tuple(self.config.aux_layer_indices)
            if self.config.aux_layer_indices is not None
            else None
        )

    def prepare_batch(
        self,
        data: BatchedDataDict[Any],
        *,
        optimizer_step: int,
        sequence_layout: DraftSequenceLayout | None = None,
    ) -> None:
        del data, optimizer_step, sequence_layout
        return None

    def normalization_counts(
        self,
        data: BatchedDataDict[Any],
        *,
        optimizer_step: int,
        sequence_layout: DraftSequenceLayout | None = None,
    ) -> None:
        del data, optimizer_step, sequence_layout
        return None

    def forward(
        self,
        *,
        policy_model: MegatronModule,
        draft_model: MegatronModule,
        captured_states: CapturedStates,
        input_ids_cp_local: Tensor,
        attention_mask: Tensor | None,
        data: BatchedDataDict[Any],
        optimizer_step: int,
        sequence_layout: DraftSequenceLayout | None,
        context_parallel_group: torch.distributed.ProcessGroup | None,
        tensor_parallel_group: torch.distributed.ProcessGroup | None,
    ) -> None:
        del policy_model
        del input_ids_cp_local
        del optimizer_step
        del sequence_layout
        del tensor_parallel_group
        from megatron.core.transformer.multi_token_prediction import roll_tensor

        if captured_states.inputs_embeds is None:
            raise RuntimeError("EAGLE draft training did not capture target embeddings")
        shifted_input_embeds = roll_tensor(
            captured_states.inputs_embeds,
            shifts=-1,
            dims=0,
            cp_group=context_parallel_group,
        )[0]
        data["student_logits"] = draft_model(
            hidden_states=captured_states.hidden_states,
            input_embeds=shifted_input_embeds,
            attention_mask=attention_mask,
        )

    def loss_stats(
        self,
        *,
        target_logits: Tensor,
        data: BatchedDataDict[Any],
        prepare_fn: Callable[..., Any],
        vocab_parallel_rank: int | None,
        vocab_parallel_group: torch.distributed.ProcessGroup | None,
        context_parallel_group: torch.distributed.ProcessGroup | None,
    ) -> DraftLossStats:
        loss_fn = DraftCrossEntropyLossFn(
            vocab_parallel_group=vocab_parallel_group,
        )
        loss_input, prepared_data = prepare_fn(
            logits=target_logits,
            data=data,
            loss_fn=loss_fn,
            vocab_parallel_rank=vocab_parallel_rank,
            vocab_parallel_group=vocab_parallel_group,
            context_parallel_group=context_parallel_group,
        )
        return loss_fn.loss_stats(data=prepared_data, **loss_input)

    def export_weights(self, model: MegatronModule) -> list[tuple[str, Tensor]]:
        return export_eagle_weights_to_hf(model)


@dataclass(frozen=True)
class DFlashSpeculator:
    """DFlash body provider sharing embeddings and output head with the target."""

    supports_context_parallel: ClassVar[bool] = (
        DFlashDraftConfig.supports_context_parallel
    )
    supports_sequence_packing: ClassVar[bool] = (
        DFlashDraftConfig.supports_sequence_packing
    )
    supports_target_sequence_parallel: ClassVar[bool] = (
        DFlashDraftConfig.supports_target_sequence_parallel
    )
    requires_full_cp_local_capture: ClassVar[bool] = True
    config: DFlashDraftConfig

    def build_model(
        self,
        *,
        model_provider: ModelProviderMixin,
        pg_collection: ProcessGroupCollection,
        policy_model_chunk: MegatronModule,
    ) -> MegatronModule | None:
        """Build or load the exact-schema public DFlash body."""
        del policy_model_chunk
        if not self.config.enabled:
            return None
        invalid_taps = [
            layer_id
            for layer_id in self.config.target_hidden_state_layer_ids
            if layer_id >= int(model_provider.num_layers)
        ]
        if invalid_taps:
            raise ValueError(
                "DFlash target hidden-state layer IDs exceed the target model: "
                + ", ".join(str(layer_id) for layer_id in invalid_taps)
            )

        from megatron.core.model_parallel_config import ModelParallelConfig

        parallel_config = ModelParallelConfig(
            tensor_model_parallel_size=model_provider.tensor_model_parallel_size,
            use_cpu_initialization=model_provider.use_cpu_initialization,
            fp16=model_provider.fp16,
            bf16=model_provider.bf16,
            params_dtype=model_provider.params_dtype,
            sequence_parallel=False,
        )
        body = DFlashBody(
            DFlashBodyConfig(
                hidden_size=model_provider.hidden_size,
                intermediate_size=model_provider.ffn_hidden_size,
                num_attention_heads=model_provider.num_attention_heads,
                num_key_value_heads=model_provider.num_query_groups,
                head_dim=model_provider.kv_channels,
                num_hidden_layers=self.config.num_layers,
                num_target_taps=len(self.config.target_hidden_state_layer_ids),
                rope_theta=model_provider.rotary_base,
                rms_norm_eps=model_provider.layernorm_epsilon,
                initializer_range=model_provider.init_method_std,
            ),
            tp_group=getattr(pg_collection, "tp", None),
            parallel_config=parallel_config,
        )
        if self.config.model_name is not None:
            missing_keys, unexpected_keys = load_hf_weights_to_dflash(
                body,
                self.config.model_name,
                model_revision=self.config.model_revision,
            )
            if missing_keys or unexpected_keys:
                raise RuntimeError(
                    "[draft] DFlash checkpoint schema mismatch: "
                    f"missing={missing_keys}, unexpected={unexpected_keys}."
                )
        register_draft_grad_norm_group()
        for parameter in body.parameters():
            parameter.grad_norm_group = DRAFT_GRAD_NORM_GROUP
        return cast(MegatronModule, body)

    def capture_layer_ids(self) -> tuple[int, ...]:
        return tuple(self.config.target_hidden_state_layer_ids)

    def prepare_batch(
        self,
        data: BatchedDataDict[Any],
        *,
        optimizer_step: int,
        sequence_layout: DraftSequenceLayout | None = None,
    ) -> DFlashBatchPlan:
        required = ("input_ids", "input_lengths", "draft_sample_ids")
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(
                "DFlash training requires stable plan inputs: " + ", ".join(missing)
            )
        input_ids = data["input_ids"]
        sequence_positions = torch.arange(
            input_ids.shape[1],
            device=input_ids.device,
        )
        token_valid_mask = sequence_positions.unsqueeze(0) < data["input_lengths"].to(
            device=input_ids.device
        ).unsqueeze(1)
        plan = build_dflash_batch_plan(
            token_valid_mask,
            data["draft_sample_ids"].to(device=input_ids.device, dtype=torch.int64),
            anchors_per_sample=self.config.anchors_per_sample,
            gamma=self.config.gamma,
            optimizer_step=optimizer_step,
            seed=self.config.seed,
            sequence_layout=sequence_layout,
        )
        total_windows = plan.excluded_window_count + plan.eligible_window_count
        exclusion_fraction = plan.excluded_window_count.to(torch.float32) / (
            total_windows.clamp_min(1).to(torch.float32)
        )
        exceeds_limit = (total_windows > 0) & (
            exclusion_fraction > self.config.max_cp_boundary_exclusion_fraction
        )
        if bool(exceeds_limit.item()):
            raise ValueError(
                "DFlash CP boundary exclusion exceeds "
                "max_cp_boundary_exclusion_fraction"
            )
        return plan

    def forward(
        self,
        *,
        policy_model: MegatronModule,
        draft_model: MegatronModule,
        captured_states: CapturedStates,
        input_ids_cp_local: Tensor,
        attention_mask: Tensor | None,
        data: BatchedDataDict[Any],
        optimizer_step: int,
        sequence_layout: DraftSequenceLayout | None,
        context_parallel_group: torch.distributed.ProcessGroup | None,
        tensor_parallel_group: torch.distributed.ProcessGroup | None,
    ) -> None:
        del attention_mask
        if captured_states.hidden_states is None:
            raise RuntimeError("DFlash training did not capture target hidden states")
        if captured_states.inputs_embeds is None:
            raise RuntimeError("DFlash training did not capture target embeddings")
        if sequence_layout is not None and captured_states.output_hidden is None:
            raise RuntimeError(
                "DFlash packed target output hidden states were not captured"
            )
        if captured_states.sequence_layout is not sequence_layout:
            captured_layout = captured_states.sequence_layout
            layouts_match = (
                captured_layout is not None
                and sequence_layout is not None
                and captured_layout.cp_rank == sequence_layout.cp_rank
                and captured_layout.tp_rank == sequence_layout.tp_rank
                and torch.equal(captured_layout.descriptor, sequence_layout.descriptor)
            )
            if not layouts_match:
                raise RuntimeError(
                    "DFlash target captures and microbatch use different sequence layouts"
                )
        if sequence_layout is not None:
            expected_input_shape = (
                1,
                sequence_layout.cp_global_positions.numel(),
            )
            if input_ids_cp_local.shape != expected_input_shape:
                raise ValueError(
                    "DFlash packed input_ids_cp_local must have shape "
                    f"{expected_input_shape}, got {tuple(input_ids_cp_local.shape)}"
                )
            if (
                sequence_layout.tp_size > 1
                and not captured_states.sequence_is_reconstructed
            ):
                raise RuntimeError(
                    "DFlash target sequence-parallel captures were not reconstructed"
                )
        plan = self.prepare_batch(
            data,
            optimizer_step=optimizer_step,
            sequence_layout=sequence_layout,
        )
        hidden_size = int(cast(DFlashBody, draft_model).config.hidden_size)
        target_taps = (
            captured_states.hidden_states.detach()
            .unflatten(-1, (len(self.capture_layer_ids()), hidden_size))
            .permute(1, 0, 2, 3)
            .contiguous()
        )
        input_embeddings = captured_states.inputs_embeds.detach().transpose(0, 1)
        if sequence_layout is None:
            anchor_embeddings = input_embeddings[
                plan.sample_rows,
                plan.anchor_positions,
            ].unsqueeze(1)
        else:
            anchor_embeddings = input_embeddings[
                0,
                plan.local_anchor_positions,
            ].unsqueeze(1)
        policy = unwrap_model(policy_model)
        word_embeddings = policy.embedding.word_embeddings
        mask_ids = torch.full(
            (plan.sample_rows.numel(), self.config.gamma),
            self.config.mask_token_id,
            dtype=torch.int64,
            device=input_embeddings.device,
        )
        with torch.no_grad():
            mask_embeddings = _embed_draft_mask_tokens(
                word_embeddings,
                mask_ids,
                tensor_parallel_group=tensor_parallel_group,
            )
        block_embeddings = torch.cat((anchor_embeddings, mask_embeddings), dim=1)
        output_weight = get_policy_lm_head_weight(policy_model).detach()
        selected_teacher_logits = None
        if sequence_layout is not None:
            assert captured_states.output_hidden is not None
            output_hidden = captured_states.output_hidden.detach().transpose(0, 1)
            selected_teacher_hidden = output_hidden[
                0,
                plan.local_query_positions[:, :-1],
            ]
            selected_teacher_logits = selected_teacher_hidden @ output_weight.T
        data["dflash_output"] = DFlashForwardOutput(
            hidden=draft_model(
                target_taps=target_taps,
                block_embeddings=block_embeddings,
                plan=plan,
                sequence_layout=sequence_layout,
                context_parallel_group=context_parallel_group,
            ),
            plan=plan,
            output_weight=output_weight,
            sequence_layout=sequence_layout,
            selected_teacher_logits=selected_teacher_logits,
        )

    @staticmethod
    def _loss_mask(
        plan: DFlashBatchPlan,
        data: BatchedDataDict[Any],
        sequence_layout: DraftSequenceLayout | None = None,
    ) -> Tensor:
        loss_mask = plan.loss_mask
        if "token_mask" in data:
            response_mask = data["token_mask"].to(dtype=torch.bool)
            response_positions = (
                plan.label_positions
                if sequence_layout is None
                else plan.packed_rope_positions
            )
            loss_mask = (
                loss_mask
                & response_mask[
                    plan.sample_rows[:, None],
                    response_positions,
                ]
            )
        if "sample_mask" in data:
            loss_mask = (
                loss_mask
                & data["sample_mask"].to(dtype=torch.bool)[plan.sample_rows, None]
            )
        return loss_mask

    def normalization_counts(
        self,
        data: BatchedDataDict[Any],
        *,
        optimizer_step: int,
        sequence_layout: DraftSequenceLayout | None = None,
    ) -> Tensor:
        plan = self.prepare_batch(
            data,
            optimizer_step=optimizer_step,
            sequence_layout=sequence_layout,
        )
        return self._loss_mask(plan, data, sequence_layout)[:, 1:].sum(
            dim=0,
            dtype=torch.float32,
        )

    def loss_stats(
        self,
        *,
        target_logits: Tensor,
        data: BatchedDataDict[Any],
        prepare_fn: Callable[..., Any],
        vocab_parallel_rank: int | None,
        vocab_parallel_group: torch.distributed.ProcessGroup | None,
        context_parallel_group: torch.distributed.ProcessGroup | None,
    ) -> DraftLossStats:
        del prepare_fn, vocab_parallel_rank, context_parallel_group
        output = data.get("dflash_output")
        if not isinstance(output, DFlashForwardOutput):
            raise RuntimeError("DFlash forward output is unavailable")
        plan = output.plan
        loss_mask = self._loss_mask(plan, data, output.sequence_layout)
        if output.sequence_layout is None:
            teacher_logits = target_logits
            teacher_sample_rows = plan.sample_rows
            teacher_label_positions = plan.label_positions
        else:
            if output.selected_teacher_logits is None:
                raise RuntimeError("DFlash selected teacher logits are unavailable")
            teacher_logits = output.selected_teacher_logits
            teacher_sample_rows = torch.arange(
                plan.sample_rows.numel(),
                dtype=torch.int64,
                device=plan.sample_rows.device,
            )
            teacher_label_positions = torch.arange(
                plan.block_size,
                dtype=torch.int64,
                device=plan.sample_rows.device,
            ).expand_as(plan.local_label_positions)
        return dflash_projected_vocab_parallel_soft_ce(
            draft_hidden=output.hidden,
            output_weight=output.output_weight,
            teacher_logits=teacher_logits,
            sample_rows=teacher_sample_rows,
            label_positions=teacher_label_positions,
            loss_mask=loss_mask,
            position_decay=self.config.position_decay,
            token_chunk_size=self.config.vocab_tile_size,
            tp_group=vocab_parallel_group,
        )

    def export_weights(self, model: MegatronModule) -> list[tuple[str, Tensor]]:
        return export_dflash_weights_to_hf(model)


@dataclass(frozen=True)
class DSparkSpeculator:
    """DSpark provider reusing the DFlash body with Markov/confidence heads."""

    supports_context_parallel: ClassVar[bool] = (
        DSparkDraftConfig.supports_context_parallel
    )
    supports_sequence_packing: ClassVar[bool] = (
        DSparkDraftConfig.supports_sequence_packing
    )
    supports_target_sequence_parallel: ClassVar[bool] = (
        DSparkDraftConfig.supports_target_sequence_parallel
    )
    requires_full_cp_local_capture: ClassVar[bool] = True
    config: DSparkDraftConfig

    def build_model(
        self,
        *,
        model_provider: ModelProviderMixin,
        pg_collection: ProcessGroupCollection,
        policy_model_chunk: MegatronModule,
    ) -> MegatronModule | None:
        if not self.config.enabled:
            return None
        target_vocab_size = int(model_provider.vocab_size)
        draft_vocab_size = self.config.resolve_draft_vocab_size(
            target_vocab_size=target_vocab_size
        )
        body_config = DFlashDraftConfig(
            enabled=True,
            model_name=None,
            loss_weight=self.config.loss_weight,
            gamma=self.config.block_size - 1,
            anchors_per_sample=self.config.anchors_per_sample,
            mask_token_id=self.config.mask_token_id,
            target_hidden_state_layer_ids=self.config.target_hidden_state_layer_ids,
            num_layers=self.config.num_layers,
            seed=self.config.seed,
            vocab_tile_size=self.config.vocab_tile_size,
            max_cp_boundary_exclusion_fraction=(
                self.config.max_cp_boundary_exclusion_fraction
            ),
            optimizer=self.config.optimizer,
        )
        body = DFlashSpeculator(body_config).build_model(
            model_provider=model_provider,
            pg_collection=pg_collection,
            policy_model_chunk=policy_model_chunk,
        )
        if body is None:
            raise RuntimeError("enabled DSpark provider did not build its body")

        from nemo_rl.models.megatron.draft.dspark_provider import (
            build_dspark_provider,
        )

        tp_group = getattr(pg_collection, "tp", None)
        tp_size = int(model_provider.tensor_model_parallel_size)
        tp_rank = 0
        if tp_group is not None and torch.distributed.is_initialized():
            tp_size = torch.distributed.get_world_size(tp_group)
            tp_rank = torch.distributed.get_rank(tp_group)
        if draft_vocab_size % tp_size:
            raise ValueError("DSpark draft vocabulary must be divisible by TP size")
        local_vocab_size = draft_vocab_size // tp_size
        body_weight = next(body.parameters())
        adapter = build_dspark_provider(
            body=body,
            target_vocab_size=target_vocab_size,
            draft_vocab_size=draft_vocab_size,
            hidden_size=int(model_provider.hidden_size),
            markov_rank=self.config.markov_rank,
            confidence_enabled=self.config.confidence_enabled,
            confidence_with_markov=self.config.confidence_with_markov,
            draft_vocab_start_index=tp_rank * local_vocab_size,
            draft_vocab_end_index=(tp_rank + 1) * local_vocab_size,
            tensor_parallel_group=tp_group,
            device=body_weight.device,
            dtype=body_weight.dtype,
        )
        if self.config.model_name is not None:
            missing_keys, unexpected_keys = load_hf_weights_to_dspark(
                adapter,
                self.config.model_name,
                model_revision=self.config.model_revision,
            )
            if missing_keys or unexpected_keys:
                raise RuntimeError(
                    "[draft] DSpark checkpoint schema mismatch: "
                    f"missing={missing_keys}, unexpected={unexpected_keys}."
                )
        register_draft_grad_norm_group()
        for parameter in adapter.parameters():
            parameter.grad_norm_group = DRAFT_GRAD_NORM_GROUP
        return cast(MegatronModule, adapter)

    def capture_layer_ids(self) -> tuple[int, ...]:
        return tuple(self.config.target_hidden_state_layer_ids)

    def prepare_batch(
        self,
        data: BatchedDataDict[Any],
        *,
        optimizer_step: int,
        sequence_layout: DraftSequenceLayout | None = None,
    ) -> DSparkBatchPlan:
        required = ("input_ids", "input_lengths", "draft_sample_ids")
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(
                "DSpark training requires stable plan inputs: " + ", ".join(missing)
            )
        input_ids = data["input_ids"]
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        token_valid_mask = positions.unsqueeze(0) < data["input_lengths"].to(
            device=input_ids.device
        ).unsqueeze(1)
        plan = build_dspark_batch_plan(
            token_valid_mask,
            data["draft_sample_ids"].to(device=input_ids.device, dtype=torch.int64),
            anchors_per_sample=self.config.anchors_per_sample,
            block_size=self.config.block_size,
            optimizer_step=optimizer_step,
            seed=self.config.seed,
            sequence_layout=sequence_layout,
        )
        total_windows = plan.excluded_window_count + plan.eligible_window_count
        exclusion_fraction = plan.excluded_window_count.to(torch.float32) / (
            total_windows.clamp_min(1).to(torch.float32)
        )
        exceeds_limit = (total_windows > 0) & (
            exclusion_fraction > self.config.max_cp_boundary_exclusion_fraction
        )
        if bool(exceeds_limit.item()):
            raise ValueError(
                "DSpark CP boundary exclusion exceeds "
                "max_cp_boundary_exclusion_fraction"
            )
        return plan

    @staticmethod
    def _loss_mask(
        plan: DSparkBatchPlan,
        data: BatchedDataDict[Any],
        sequence_layout: DraftSequenceLayout | None = None,
    ) -> Tensor:
        loss_mask = plan.loss_mask
        if "token_mask" in data:
            response_mask = data["token_mask"].to(dtype=torch.bool)
            response_positions = (
                plan.label_positions
                if sequence_layout is None
                else plan.packed_label_rope_positions
            )
            loss_mask = (
                loss_mask
                & response_mask[
                    plan.sample_rows[:, None],
                    response_positions,
                ]
            )
        if "sample_mask" in data:
            loss_mask = (
                loss_mask
                & data["sample_mask"].to(dtype=torch.bool)[plan.sample_rows, None]
            )
        return loss_mask

    def normalization_counts(
        self,
        data: BatchedDataDict[Any],
        *,
        optimizer_step: int,
        sequence_layout: DraftSequenceLayout | None = None,
    ) -> Tensor:
        plan = self.prepare_batch(
            data,
            optimizer_step=optimizer_step,
            sequence_layout=sequence_layout,
        )
        return self._loss_mask(plan, data, sequence_layout).sum(
            dim=0,
            dtype=torch.float32,
        )

    def forward(
        self,
        *,
        policy_model: MegatronModule,
        draft_model: MegatronModule,
        captured_states: CapturedStates,
        input_ids_cp_local: Tensor,
        attention_mask: Tensor | None,
        data: BatchedDataDict[Any],
        optimizer_step: int,
        sequence_layout: DraftSequenceLayout | None,
        context_parallel_group: torch.distributed.ProcessGroup | None,
        tensor_parallel_group: torch.distributed.ProcessGroup | None,
    ) -> None:
        del attention_mask
        if captured_states.hidden_states is None:
            raise RuntimeError("DSpark training did not capture target hidden states")
        if captured_states.inputs_embeds is None:
            raise RuntimeError("DSpark training did not capture target embeddings")
        if sequence_layout is not None and captured_states.output_hidden is None:
            raise RuntimeError(
                "DSpark packed target output hidden states were not captured"
            )
        if captured_states.sequence_layout is not sequence_layout:
            captured_layout = captured_states.sequence_layout
            layouts_match = (
                captured_layout is not None
                and sequence_layout is not None
                and captured_layout.cp_rank == sequence_layout.cp_rank
                and captured_layout.tp_rank == sequence_layout.tp_rank
                and torch.equal(captured_layout.descriptor, sequence_layout.descriptor)
            )
            if not layouts_match:
                raise RuntimeError(
                    "DSpark target captures and microbatch use different sequence layouts"
                )
        if sequence_layout is not None:
            expected_input_shape = (
                1,
                sequence_layout.cp_global_positions.numel(),
            )
            if input_ids_cp_local.shape != expected_input_shape:
                raise ValueError(
                    "DSpark packed input_ids_cp_local must have shape "
                    f"{expected_input_shape}, got {tuple(input_ids_cp_local.shape)}"
                )
            if (
                sequence_layout.tp_size > 1
                and not captured_states.sequence_is_reconstructed
            ):
                raise RuntimeError(
                    "DSpark target sequence-parallel captures were not reconstructed"
                )
        plan = self.prepare_batch(
            data,
            optimizer_step=optimizer_step,
            sequence_layout=sequence_layout,
        )
        adapter = cast(Any, unwrap_model(draft_model))
        hidden_size = int(adapter.body.config.hidden_size)
        target_taps = (
            captured_states.hidden_states.detach()
            .unflatten(-1, (len(self.capture_layer_ids()), hidden_size))
            .permute(1, 0, 2, 3)
            .contiguous()
        )
        input_embeddings = captured_states.inputs_embeds.detach().transpose(0, 1)
        if sequence_layout is None:
            anchor_embeddings = input_embeddings[
                plan.sample_rows,
                plan.anchor_positions,
            ].unsqueeze(1)
            input_rows = plan.sample_rows
        else:
            anchor_embeddings = input_embeddings[
                0,
                plan.local_anchor_positions,
            ].unsqueeze(1)
            input_rows = torch.zeros_like(plan.sample_rows)
        policy = unwrap_model(policy_model)
        mask_ids = torch.full(
            (plan.sample_rows.numel(), self.config.block_size - 1),
            self.config.mask_token_id,
            dtype=torch.int64,
            device=input_embeddings.device,
        )
        with torch.no_grad():
            mask_embeddings = _embed_draft_mask_tokens(
                policy.embedding.word_embeddings,
                mask_ids,
                tensor_parallel_group=tensor_parallel_group,
            )
        block_embeddings = torch.cat((anchor_embeddings, mask_embeddings), dim=1)
        output_weight = get_policy_lm_head_weight(policy_model).detach()
        selected_teacher_logits = None
        if sequence_layout is not None:
            assert captured_states.output_hidden is not None
            output_hidden = captured_states.output_hidden.detach().transpose(0, 1)
            selected_teacher_hidden = output_hidden[
                0,
                plan.local_query_positions,
            ]
            selected_teacher_logits = selected_teacher_hidden @ output_weight.T
        data["dspark_output"] = DSparkForwardOutput(
            hidden=adapter.body(
                target_taps=target_taps,
                block_embeddings=block_embeddings,
                plan=plan,
                sequence_layout=sequence_layout,
                context_parallel_group=context_parallel_group,
            ),
            plan=plan,
            output_weight=output_weight,
            previous_token_ids=input_ids_cp_local[
                input_rows[:, None],
                plan.local_query_positions,
            ],
            hard_labels=input_ids_cp_local[
                input_rows[:, None],
                plan.local_label_positions,
            ],
            adapter=adapter,
            sequence_layout=sequence_layout,
            selected_teacher_logits=selected_teacher_logits,
        )

    def loss_stats(
        self,
        *,
        target_logits: Tensor,
        data: BatchedDataDict[Any],
        prepare_fn: Callable[..., Any],
        vocab_parallel_rank: int | None,
        vocab_parallel_group: torch.distributed.ProcessGroup | None,
        context_parallel_group: torch.distributed.ProcessGroup | None,
    ) -> DraftLossStats:
        del prepare_fn, vocab_parallel_rank, context_parallel_group
        output = data.get("dspark_output")
        if not isinstance(output, DSparkForwardOutput):
            raise RuntimeError("DSpark forward output is unavailable")
        plan = output.plan
        if output.sequence_layout is None:
            selected_target_logits = target_logits[
                plan.sample_rows[:, None],
                plan.query_positions,
            ]
        else:
            if output.selected_teacher_logits is None:
                raise RuntimeError("DSpark selected teacher logits are unavailable")
            selected_target_logits = output.selected_teacher_logits
        slot_bins = torch.arange(
            self.config.block_size,
            dtype=torch.int64,
            device=target_logits.device,
        ).expand_as(plan.query_positions)
        slot_weights = torch.exp(
            -torch.arange(
                self.config.block_size,
                dtype=torch.float32,
                device=target_logits.device,
            )
            / self.config.loss_decay_gamma
        )
        stats = output.adapter.objective_stats(
            draft_hidden=output.hidden,
            target_output_weight=output.output_weight,
            target_logits=selected_target_logits,
            previous_token_ids=output.previous_token_ids,
            hard_labels=output.hard_labels,
            valid_mask=self._loss_mask(plan, data, output.sequence_layout),
            slot_bins=slot_bins,
            slot_weights=slot_weights,
            loss_weights=(
                self.config.ce_loss_weight,
                self.config.tv_loss_weight,
                self.config.confidence_loss_weight,
            ),
            token_chunk_size=self.config.vocab_tile_size,
            tp_group=vocab_parallel_group,
        )
        return DraftLossStats(
            numerators=stats.combined.numerators,
            counts=stats.combined.counts,
            weights=stats.combined.weights,
        )

    def export_weights(self, model: MegatronModule) -> list[tuple[str, Tensor]]:
        adapter = cast(Any, unwrap_model(model))
        return [
            *export_dflash_weights_to_hf(adapter.body),
            *export_dspark_heads_to_hf(adapter),
        ]


_SPECULATOR_FACTORIES: dict[
    str,
    type[Eagle3Speculator] | type[DFlashSpeculator] | type[DSparkSpeculator],
] = {
    "eagle3": Eagle3Speculator,
    "dflash": DFlashSpeculator,
    "dspark": DSparkSpeculator,
}


def resolve_draft_speculator(
    config: DraftConfig | None,
) -> DraftTrainingProvider | None:
    """Resolve an enabled draft configuration to its speculator."""
    if config is None or not config.enabled:
        return None
    return cast(
        DraftTrainingProvider,
        _SPECULATOR_FACTORIES[config.speculator_type](config),
    )
