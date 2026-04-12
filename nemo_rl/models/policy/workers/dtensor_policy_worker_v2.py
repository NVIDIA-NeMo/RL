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

import gc
import itertools
import os
import warnings
from collections import defaultdict
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Any, Generator, Optional, cast

import ray
import torch
from accelerate import init_empty_weights
from hydra.utils import get_class
from nemo_automodel import (
    NeMoAutoModelForSequenceClassification,
)
from nemo_automodel._transformers.registry import ModelRegistry
from nemo_automodel.components._peft.lora import (
    PeftConfig,
    apply_lora_to_linear_modules,
)
from nemo_automodel.components.config.loader import _resolve_target
from nemo_automodel.components.distributed.config import FSDP2Config
from nemo_automodel.components.distributed.cp_utils import (
    create_context_parallel_ctx,
    get_train_context,
)
from nemo_automodel.components.distributed.device_mesh import create_device_mesh
from nemo_automodel.components.distributed.fsdp2 import (
    FSDP2Manager,
)
from nemo_automodel.components.distributed.tensor_utils import (
    get_cpu_state_dict,
    to_local_if_dtensor,
)
from nemo_automodel.components.moe.parallelizer import (
    parallelize_model as moe_parallelize_model,
)
from nemo_automodel.components.training.utils import scale_grads_and_clip_grad_norm
from torch import nn
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
)
from torch.distributed.tensor import DTensor, Shard
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedModel,
)
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM

from nemo_rl.algorithms.loss.interfaces import LossFunction, LossType
from nemo_rl.algorithms.loss.loss_functions import SequencePackingLossWrapper
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import (
    _compute_distributed_log_softmax,
    allgather_cp_sharded_tensor,
    distributed_vocab_topk,
    get_logprobs_from_vocab_parallel_logits,
)
from nemo_rl.models.policy.utils import get_handle_from_tensor, rebuild_cuda_tensor_from_ipc
from nemo_rl.models.huggingface.common import (
    get_flash_attention_kwargs,
    pack_sequences,
)
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import (
    ColocatablePolicyInterface,
    LogprobOutputSpec,
    ScoreOutputSpec,
)
from nemo_rl.models.policy.utils import (
    configure_dynamo_cache,
    get_runtime_env_for_policy_worker,
    resolve_model_class,
)
from nemo_rl.models.policy.workers.base_policy_worker import AbstractPolicyWorker
from nemo_rl.models.policy.workers.patches import (
    apply_transformer_engine_patch,
)
from nemo_rl.models.automodel.setup import (
    setup_distributed,
    setup_model_and_optimizer,
    setup_reference_model_state,
    validate_and_prepare_config,
)
from nemo_rl.models.automodel.checkpoint import AutomodelCheckpointManager
from nemo_rl.utils.checkpoint import CheckpointingConfig
from nemo_rl.utils.nsys import wrap_with_nvtx_name
from nemo_rl.utils.packed_tensor import packed_broadcast_producer

STRING_TO_DTYPE = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("dtensor_policy_worker_v2")
)  # pragma: no cover
class DTensorPolicyWorkerV2(AbstractPolicyWorker, ColocatablePolicyInterface):
    def __repr__(self) -> str:
        """Customizes the actor's prefix in the Ray logs.

        This makes it easier to identify which worker is producing specific log messages.
        """
        if torch.distributed.is_initialized():
            return f"{self.__class__.__qualname__}[rank={torch.distributed.get_rank()}]"
        else:
            return f"{self.__class__.__qualname__}"

    def __init__(
        self,
        config: PolicyConfig,
        tokenizer: AutoTokenizer,
        processor: Optional[AutoProcessor] = None,
        weights_path: Optional[str] = None,
        optimizer_path: Optional[str] = None,
        init_optimizer: bool = True,
        init_reference_model: bool = True,
        **kwargs: Any,
    ):
        """Initialize the DTensorPolicyWorkerV2."""
        # Apply TE patch until TE is upgraded to 2.10.0
        apply_transformer_engine_patch()

        self.cfg = config

        # Keep legacy call compatibility (tokenizer/processor passed from caller),
        # but default to main-branch local reconstruction path when needed.
        if tokenizer is None:
            from nemo_rl.models.automodel.setup import get_tokenizer

            use_processor = config["tokenizer"].get("use_processor", False)
            result = get_tokenizer(config["tokenizer"], get_processor=use_processor)
            if use_processor:
                self.processor = result
                self.tokenizer = result.tokenizer
            else:
                self.tokenizer = result
                self.processor = None
        else:
            self.tokenizer = tokenizer
            self.processor = processor
        self.is_vlm = self.processor is not None
        print(f"Initializing DTensorPolicyWorkerV2 with is_vlm={self.is_vlm}")

        self.checkpoint_manager: Optional[AutomodelCheckpointManager] = None
        self.lora_enabled = (
            config["dtensor_cfg"].get("lora_cfg", {}).get("enabled", False)
        )

        runtime_config = validate_and_prepare_config(
            config=config,
            processor=self.processor,
            rank=0,
        )
        distributed_context = setup_distributed(
            config=config,
            runtime_config=runtime_config,
        )
        self.rank = torch.distributed.get_rank()
        self.device_mesh = distributed_context.device_mesh
        self.dp_cp_mesh = self.device_mesh["dp_cp"]
        self.dp_mesh = self.device_mesh["dp"]
        self.tp_mesh = self.device_mesh["tp"]
        self.cp_mesh = self.device_mesh["cp"]
        self.moe_mesh = distributed_context.moe_mesh
        self.dp_size = distributed_context.dp_size
        self.tp_size = distributed_context.tp_size
        self.cp_size = distributed_context.cp_size

        self._init_checkpoint_manager(
            config_updates={
                "model_repo_id": config["model_name"],
                "dequantize_base_checkpoint": config.get(
                    "dequantize_base_checkpoint", False
                ),
                "is_peft": self.lora_enabled,
                "is_async": True,
            },
        )

        model_and_optimizer_state = setup_model_and_optimizer(
            config=config,
            tokenizer=self.tokenizer,
            runtime_config=runtime_config,
            distributed_context=distributed_context,
            checkpoint_manager=self.checkpoint_manager,
            is_vlm=self.is_vlm,
            init_optimizer=init_optimizer,
            weights_path=weights_path,
            optimizer_path=optimizer_path,
        )
        (
            self.model,
            self.optimizer,
            self.scheduler,
            self.is_hf_model,
            self.is_moe_model,
            self._is_reward_model,
            self.model_class,
            self.model_config,
            self.peft_config,
            self.autocast_enabled,
        ) = model_and_optimizer_state

        self._fix_phi_rope_meta_buffers()

        self.reference_model_state_dict = None
        if init_reference_model:
            self.reference_model_state_dict = setup_reference_model_state(self.model)

        (
            self.model_class,
            self.model_config,
            self.hf_config_overrides,
            self.allow_flash_attn_args,
            self.attn_impl,
            self.dtype,
            self.enable_seq_packing,
            self.max_grad_norm,
            self.cpu_offload,
            self.offload_optimizer_for_logprob,
            self.is_generation_colocated,
            self.sampling_params,
            _runtime_is_reward_model,
        ) = runtime_config

    def _fix_phi_rope_meta_buffers(self) -> None:
        """Repair Phi RoPE original_inv_freq when loaded from meta init.

        Some Phi remote-code revisions keep `original_inv_freq` as a meta tensor
        after model materialization, which later crashes in `.to(device)` calls.
        """
        model_name = str(self.cfg.get("model_name", "")).lower()
        architectures = getattr(getattr(self, "model_config", None), "architectures", [])
        arch_blob = " ".join(
            a.lower() for a in architectures if isinstance(a, str)
        )
        # Scope to Phi-4/Phi3-style remote-code paths only.
        if "phi-4" not in model_name and "phi4" not in model_name and "phi3" not in arch_blob:
            return

        fixed_count = 0
        for module in self.model.modules():
            if not hasattr(module, "inv_freq"):
                continue
            inv_freq = getattr(module, "inv_freq")
            original_inv_freq = getattr(module, "original_inv_freq", None)

            inv_needs_repair = torch.is_tensor(inv_freq) and (
                getattr(inv_freq, "is_meta", False) or (not torch.isfinite(inv_freq).all())
            )
            original_needs_repair = torch.is_tensor(original_inv_freq) and (
                getattr(original_inv_freq, "is_meta", False)
                or (not torch.isfinite(original_inv_freq).all())
            )

            if not inv_needs_repair and not original_needs_repair:
                continue

            if hasattr(module, "rope_init_fn") and hasattr(module, "config"):
                try:
                    repaired_inv_freq, _ = module.rope_init_fn(
                        module.config, torch.device("cuda")
                    )
                    if hasattr(module, "_buffers") and "inv_freq" in module._buffers:
                        module._buffers["inv_freq"] = repaired_inv_freq
                    else:
                        module.inv_freq = repaired_inv_freq
                    if hasattr(module, "original_inv_freq"):
                        module.original_inv_freq = repaired_inv_freq.detach().clone()
                    fixed_count += 1
                except Exception:
                    # Keep original behavior if re-init fails for any model variant.
                    pass

        if fixed_count > 0 and self.rank == 0:
            print(
                f"[Phi4 compatibility] repaired {fixed_count} RoPE meta buffer(s) after model setup."
            )

    def _apply_temperature_scaling(self, logits: torch.Tensor, skip: bool = False) -> torch.Tensor:
        if skip:
            return logits
        if "generation" in self.cfg and self.cfg["generation"] is not None:
            temp = self.cfg["generation"]["temperature"]
            if temp > 0:
                logits.div_(temp)
        return logits

    def check_model_allow_flash_attn_args(self, model_config) -> bool:
        # Some models doesn't support flash_attn_kwargs
        # Check nemotron nas.
        if (
            model_config.architectures[0] == "DeciLMForCausalLM"
            and model_config.model_type == "nemotron-nas"
        ):
            return False

        return True

    def init_cross_tokenizer_loss_fn(self, loss_config, token_aligner_config):
        """Build CrossTokenizerDistillationLossFn locally from config + shared filesystem."""
        from nemo_rl.algorithms.x_token import TokenAligner
        from nemo_rl.algorithms.loss.loss_functions import CrossTokenizerDistillationLossFn

        aligner = TokenAligner(
            teacher_tokenizer_name=token_aligner_config["teacher_model"],
            student_tokenizer_name=token_aligner_config["student_model"],
            max_comb_len=token_aligner_config.get("max_comb_len", 4),
            projection_matrix_multiplier=token_aligner_config.get("projection_matrix_multiplier", 1.0),
        )
        aligner._load_logits_projection_map(
            file_path=token_aligner_config["projection_matrix_path"],
            use_sparse_format=token_aligner_config.get("use_sparse_format", True),
            learnable=token_aligner_config.get("learnable", False),
            device="cpu",
        )
        if token_aligner_config.get("project_teacher_to_student", False):
            aligner.create_reverse_projection_matrix(device="cpu")
        self._cached_loss_fn = CrossTokenizerDistillationLossFn(loss_config, aligner)

    def update_cross_tokenizer_data(self, teacher_input_ids, aligned_pairs) -> None:
        """Update per-step cross-tokenizer data on the cached loss function."""
        if hasattr(self, '_cached_loss_fn') and self._cached_loss_fn is not None:
            self._cached_loss_fn.set_cross_tokenizer_data(
                teacher_input_ids=teacher_input_ids,
                aligned_pairs=aligned_pairs,
            )

    @wrap_with_nvtx_name("dtensor_policy_worker_v2/train")
    def train(
        self,
        data: BatchedDataDict[Any],
        loss_fn: LossFunction,
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
        is_teacher: bool = False,
        teacher_logits: Optional[Any] = None,
        topk_logits: Optional[int] = None,
    ) -> dict[str, Any]:
        """Train the policy on a batch of data with a given loss function."""
        if loss_fn is None:
            loss_fn = getattr(self, '_cached_loss_fn', None)
        if gbs is None:
            gbs = self.cfg["train_global_batch_size"]
        if mbs is None:
            mbs = self.cfg["train_micro_batch_size"]
        local_gbs = gbs // self.dp_size
        total_dataset_size = torch.tensor(data.size, device="cuda")
        torch.distributed.all_reduce(
            total_dataset_size,
            op=torch.distributed.ReduceOp.SUM,
            group=self.dp_mesh.get_group(),
        )
        num_global_batches = int(total_dataset_size.item()) // gbs

        # dim 1 is always assumed to be the sequence dim, sanity check this here
        sequence_dim = 1
        seq_dim_size = data.get("input_ids").shape[sequence_dim]
        for k, v in data.items():
            if torch.is_tensor(v) and len(v.shape) > 1:
                assert v.shape[sequence_dim] == seq_dim_size, (
                    f"Dim 1 must be the sequence dim, expected dim 1={seq_dim_size} but got shape {v.shape}"
                )

        if is_teacher:
            ctx: AbstractContextManager[Any] = torch.no_grad()
            self.model.eval()
        elif eval_mode:
            ctx = torch.no_grad()
            self.model.eval()
        else:
            ctx = nullcontext()
            self.model.train()

        with ctx:
            # Get data from batch and move to device
            data.to("cuda")

            losses = []
            all_mb_metrics = []
            for gb_idx in range(num_global_batches):
                global_batch = data.get_batch(batch_idx=gb_idx, batch_size=local_gbs)

                assert "sample_mask" in global_batch, (
                    "sample_mask must be present in the data!"
                )
                ## get the normalization factor for the loss
                local_valid_seqs = torch.sum(global_batch["sample_mask"])

                if "token_mask" not in global_batch:
                    local_valid_toks = (
                        local_valid_seqs * global_batch["input_ids"].shape[1]
                    )
                else:
                    local_valid_toks = torch.sum(
                        global_batch["token_mask"][:, 1:]
                        * global_batch["sample_mask"].unsqueeze(-1)
                    )

                to_reduce = torch.tensor([local_valid_seqs, local_valid_toks]).cuda()
                torch.distributed.all_reduce(to_reduce, group=self.dp_mesh.get_group())
                global_valid_seqs, global_valid_toks = to_reduce[0], to_reduce[1]

                if (
                    hasattr(loss_fn, "loss_type")
                    and loss_fn.loss_type == LossType.TOKEN_LEVEL
                ):
                    assert "token_mask" in global_batch, (
                        "token_mask must be present in the data when using token-level loss"
                    )

                if not is_teacher:
                    self.optimizer.zero_grad()
                mb_losses = []
                batch = data.get_batch(batch_idx=gb_idx, batch_size=local_gbs)
                # Calculate number of microbatches to process
                # make_microbatch_iterator assumes that the batch size is a multiple of the microbatch size
                # so its safe to not check for the case where the last data slice is smaller than mbs
                dummy_iterator = iter([])
                if self.cfg.get("dynamic_batching", {}).get("enabled", False):
                    mb_iterator = batch.make_microbatch_iterator_with_dynamic_shapes()
                    iterator_len = batch.get_microbatch_iterator_dynamic_shapes_len()
                elif self.enable_seq_packing:
                    mb_iterator = (
                        batch.make_microbatch_iterator_for_packable_sequences()
                    )
                    iterator_len, max_seqlen = (
                        batch.get_microbatch_iterator_for_packable_sequences_len()
                    )
                    max_batch_ct = torch.tensor([iterator_len], device="cuda")
                    torch.distributed.all_reduce(
                        max_batch_ct, op=torch.distributed.ReduceOp.MAX
                    )

                    # Sequence packing can end up with unevenly distributed batch counts across DP ranks.
                    # We add dummy batches to the end of the iterator to make the batch counts equal.
                    dummy_batch_ct = int(max_batch_ct.item() - iterator_len)
                    dummy_iterator = (
                        batch.make_microbatch_iterator_for_packable_sequences()
                    )
                    dummy_iterator = itertools.islice(
                        itertools.cycle(dummy_iterator), dummy_batch_ct
                    )
                else:
                    mb_iterator = batch.make_microbatch_iterator(mbs)
                    iterator_len = batch.size // mbs

                empty_cache_steps = self.cfg.get("dtensor_cfg", {}).get(
                    "clear_cache_every_n_steps"
                )
                if empty_cache_steps:
                    warnings.warn(
                        f"Emptying cache every {empty_cache_steps} microbatches, doing so unnnecessarily would incur a large performance overhead."
                    )

                _teacher_mb_handles = []
                _teacher_is_topk = False
                if not is_teacher and teacher_logits is not None:
                    rank = torch.distributed.get_rank()
                    worker_result = teacher_logits[rank]
                    _teacher_mb_handles = worker_result['microbatch_handles']
                    _teacher_is_topk = worker_result.get('is_topk', False)

                for mb_idx, mb in enumerate(
                    itertools.chain(mb_iterator, dummy_iterator)
                ):
                    # Conditioanlly empty cache when sensitive to fragmentation
                    if empty_cache_steps and mb_idx % empty_cache_steps == 0:
                        torch.cuda.empty_cache()

                    with torch.autocast(device_type="cuda", dtype=self.dtype):
                        if self.enable_seq_packing:
                            input_ids = mb.get("input_ids").cuda()
                            input_ids, position_ids, _ = pack_sequences(
                                input_ids=input_ids,
                                input_lengths=mb["input_lengths"],
                                packed_sequence_size=[
                                    len(mb["input_lengths"])
                                ],  # flash attention 2 expects flattened input
                                padding_value=self.tokenizer.eos_token_id,
                                return_attention_mask=False,
                                min_seq_len=self.cfg["sequence_packing"][
                                    "train_mb_tokens"
                                ],  # TODO: this is a WAR for sequence packing, we should fix this. Without this, backward will fail when TP is enabled.
                            )
                            seq_len = input_ids.shape[1]
                            attention_mask = None
                            flash_attn_kwargs = get_flash_attention_kwargs(
                                input_lengths=mb["input_lengths"],
                            )

                        else:
                            input_ids = mb.get("input_ids").cuda()
                            batch_size, seq_len = input_ids.shape

                            attention_mask = torch.ones(
                                (batch_size, seq_len),
                                dtype=torch.bool,
                                device=input_ids.device,
                            )
                            position_ids = torch.arange(
                                seq_len, device=input_ids.device
                            ).repeat(batch_size, 1)
                            flash_attn_kwargs = {}

                        # add vlm kwargs to model call
                        vlm_kwargs = mb.get_multimodal_dict(
                            as_tensors=True, device=input_ids.device
                        )
                        if len(vlm_kwargs) > 0:
                            position_ids = None
                            assert not self.cfg["dtensor_cfg"]["sequence_parallel"], (
                                "Sequence parallel is not supported with multimodal since there's an issue when you do not pass position_ids. See https://github.com/NVIDIA-NeMo/Automodel/issues/652"
                            )

                    context_parallel_ctx = None
                    if self.cp_size > 1:
                        assert len(vlm_kwargs) == 0, (
                            f"multimodal kwargs={vlm_kwargs} are not supported for context parallel"
                        )
                        seq_index = torch.arange(
                            seq_len, device=input_ids.device
                        ).repeat(1, 1)
                        cp_buffers = (
                            [input_ids, position_ids, seq_index]
                            if self.cp_size > 1
                            else []
                        )

                        # Create context parallel context
                        context_parallel_ctx = create_context_parallel_ctx(
                            cp_mesh=self.cp_mesh,
                            cp_buffers=cp_buffers,
                            cp_seq_dims=[sequence_dim] * len(cp_buffers),
                            cp_no_restore_buffers=set(cp_buffers),
                        )

                    with get_train_context(False, False, context_parallel_ctx)():
                        with torch.autocast(device_type="cuda", dtype=self.dtype):
                            model_args = dict(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                use_cache=False,
                                flash_attn_kwargs=flash_attn_kwargs,
                                **vlm_kwargs,
                            )

                            if self._is_reward_model:
                                # `flash_attn_kwarg` is not supported for `LlamaForSequenceClassification`.
                                # Note that it should be empty anyway since sequence packing
                                # is not supported for reward models.
                                assert not flash_attn_kwargs
                                del model_args["flash_attn_kwargs"]
                            # remove flash_attn_kwargs if there are multimodal kwargs
                            if len(vlm_kwargs) > 0:
                                del model_args["flash_attn_kwargs"]

                            if (
                                not self.allow_flash_attn_args
                                and "flash_attn_kwargs" in model_args
                            ):
                                del model_args["flash_attn_kwargs"]

                            outputs = self.model(**model_args)

                        # Get logprobs
                        if isinstance(outputs, (torch.Tensor, DTensor)):
                            # custom models (e.g., those coming from AutoModel) can output logits directly
                            logits = outputs
                        elif not hasattr(outputs, "logits"):
                            logits = self.model.lm_head(outputs.last_hidden_state)
                        else:
                            logits = outputs.logits
                        del outputs

                        # Temperature scaling is only for inference/generation, not training
                        logits = self._apply_temperature_scaling(logits, skip=True)

                        if self.cp_size > 1:
                            seq_index_dtensor = (
                                DTensor.from_local(
                                    seq_index,
                                    device_mesh=self.cp_mesh,
                                    placements=[Shard(1)],
                                )
                                .full_tensor()
                                .squeeze(0)
                            )

                            mb["seq_index"] = seq_index_dtensor

                            for tensor_name in mb:
                                current_tensor = mb[tensor_name]
                                for buffer in cp_buffers:
                                    if current_tensor is buffer:
                                        assert type(current_tensor) == torch.Tensor, (
                                            f"tensor {tensor_name} is not a tensor"
                                        )
                                        mb[tensor_name] = DTensor.from_local(
                                            current_tensor,
                                            device_mesh=self.cp_mesh,
                                            placements=[Shard(sequence_dim)],
                                        )
                                        break

                            if isinstance(logits, DTensor):
                                # Must be tp sharded
                                assert (
                                    logits.device_mesh.ndim == 1
                                    and logits.device_mesh.mesh_dim_names[0] == "tp"
                                ), "logits must be tp sharded"

                                # CP is implicitly sharded on the seq dim, so we need to redistribute to the tp dim
                                logits = DTensor.from_local(
                                    logits.to_local(),
                                    device_mesh=self.device_mesh[("cp", "tp")],
                                    placements=[Shard(sequence_dim), Shard(-1)],
                                )
                            else:
                                logits = DTensor.from_local(
                                    logits,
                                    device_mesh=self.device_mesh[("cp", "tp")],
                                    placements=[Shard(sequence_dim), Shard(-1)],
                                )

                        if is_teacher:
                            with torch.no_grad():
                                if isinstance(logits, DTensor):
                                    mb_logits_local = logits.to_local()
                                else:
                                    mb_logits_local = logits
                                del logits

                                tp_group = self.tp_mesh.get_group()
                                tp_rank = torch.distributed.get_rank(tp_group)
                                V_local = int(mb_logits_local.shape[-1])
                                vocab_start_index = tp_rank * V_local
                                vocab_end_index = (tp_rank + 1) * V_local

                                mb_logits_local = mb_logits_local.to(torch.float32)
                                mb_log_prob = _compute_distributed_log_softmax(mb_logits_local, group=tp_group)
                                del mb_logits_local

                                if isinstance(mb_log_prob, DTensor):
                                    mb_log_prob = mb_log_prob.to_local()

                                if self.cfg.get('is_mdlm', False):
                                    shared_seq_len = int(mb_log_prob.shape[1] / 2)
                                    mb_log_prob = mb_log_prob[:, shared_seq_len:, :]

                                if topk_logits is not None:
                                    mb_topk_vals, mb_topk_idx = distributed_vocab_topk(
                                        mb_log_prob,
                                        k=topk_logits,
                                        tp_group=tp_group,
                                        vocab_start_index=vocab_start_index,
                                        vocab_end_index=vocab_end_index,
                                    )
                                    del mb_log_prob

                                    B_mb, S_mb, K_mb = mb_topk_vals.shape
                                    buf_idx = len(_teacher_mb_handles)
                                    self._ensure_teacher_mb_topk_buffer(
                                        buf_idx, B_mb, K_mb,
                                        mb_topk_vals.dtype, mb_topk_idx.dtype, mb_topk_vals.device,
                                    )
                                    self._teacher_mb_vals_buffers[buf_idx][:B_mb, :S_mb, :K_mb].copy_(mb_topk_vals)
                                    self._teacher_mb_idx_buffers[buf_idx][:B_mb, :S_mb, :K_mb].copy_(mb_topk_idx)
                                    del mb_topk_vals, mb_topk_idx

                                    rank = torch.distributed.get_rank()
                                    _teacher_mb_handles.append({
                                        rank: self._teacher_mb_vals_ipcs[buf_idx],
                                        'actual_shape': (B_mb, S_mb, K_mb),
                                        'topk_indices_ipc': self._teacher_mb_idx_ipcs[buf_idx],
                                    })
                                else:
                                    B_mb, S_mb, V_mb = mb_log_prob.shape
                                    buf_idx = len(_teacher_mb_handles)
                                    self._ensure_teacher_mb_logits_buffer(
                                        buf_idx, B_mb, V_mb,
                                        mb_log_prob.dtype, mb_log_prob.device,
                                    )
                                    self._teacher_mb_logits_buffers[buf_idx][:B_mb, :S_mb, :V_mb].copy_(mb_log_prob)
                                    del mb_log_prob

                                    rank = torch.distributed.get_rank()
                                    _teacher_mb_handles.append({
                                        rank: self._teacher_mb_logits_ipcs[buf_idx],
                                        'actual_shape': (B_mb, S_mb, V_mb),
                                    })
                        else:
                            if self.enable_seq_packing:
                                loss_fn_ = SequencePackingLossWrapper(
                                    loss_fn=loss_fn,
                                    cu_seqlens_q=flash_attn_kwargs.cu_seqlens_q,
                                    cu_seqlens_q_padded=flash_attn_kwargs.cu_seqlens_q,
                                )
                            else:
                                loss_fn_ = loss_fn

                            # ── NaN debug: inspect logits on first microbatch of first 2 steps ──
                            if mb_idx == 0 and gb_idx == 0 and self.rank == 0 and len(losses) < 2:
                                _local_logits = logits.to_local() if isinstance(logits, DTensor) else logits
                                _lf = _local_logits.float()
                                print(
                                    f"  [NaN debug rank-0] logits shape={_local_logits.shape}, "
                                    f"dtype={_local_logits.dtype}, "
                                    f"min={_lf.min().item():.4f}, max={_lf.max().item():.4f}, "
                                    f"has_nan={torch.isnan(_lf).any().item()}, "
                                    f"has_inf={torch.isinf(_lf).any().item()}, "
                                    f"global_valid_toks={global_valid_toks.item():.0f}, "
                                    f"global_valid_seqs={global_valid_seqs.item():.0f}",
                                    flush=True,
                                )
                                del _local_logits, _lf

                            if _teacher_mb_handles and mb_idx < len(_teacher_mb_handles) and not self.enable_seq_packing:
                                rank = torch.distributed.get_rank()
                                current_device_id = torch.cuda.current_device()
                                handle = _teacher_mb_handles[mb_idx]
                                aB, aS, aK = handle['actual_shape']

                                teacher_logits_tensor = rebuild_cuda_tensor_from_ipc(
                                    handle[rank], current_device_id
                                ).detach()
                                teacher_logits_tensor = teacher_logits_tensor[:aB, :aS, :aK].clone()

                                teacher_topk_indices_tensor = None
                                if _teacher_is_topk and 'topk_indices_ipc' in handle:
                                    teacher_topk_indices_tensor = rebuild_cuda_tensor_from_ipc(
                                        handle['topk_indices_ipc'], current_device_id
                                    ).detach()
                                    teacher_topk_indices_tensor = teacher_topk_indices_tensor[:aB, :aS, :aK].clone()

                                loss, loss_metrics = loss_fn_(
                                    logits,
                                    mb,
                                    global_valid_seqs,
                                    global_valid_toks,
                                    teacher_logits=teacher_logits_tensor,
                                    teacher_topk_indices_ipc=teacher_topk_indices_tensor,
                                )
                                del teacher_logits_tensor
                                if teacher_topk_indices_tensor is not None:
                                    del teacher_topk_indices_tensor
                            else:
                                loss, loss_metrics = loss_fn_(
                                    logits,
                                    mb,
                                    global_valid_seqs,
                                    global_valid_toks,
                                )
                            del logits

                            # skip the update for dummy batches
                            if mb_idx < iterator_len:
                                ## scale by the number of global batches so we get the correct
                                ## value when summing metrics across all microbatches
                                for k in loss_metrics.keys():
                                    loss_metrics[k] /= num_global_batches
                                num_valid_samples = loss_metrics["num_valid_samples"]
                                loss_metrics["lr"] = self.optimizer.param_groups[0]["lr"]
                                loss_metrics["global_valid_seqs"] = global_valid_seqs.item()
                                loss_metrics["global_valid_toks"] = global_valid_toks.item()
                            else:
                                loss *= 0

                            # Backward pass
                            if not eval_mode:
                                loss *= self.dp_size * self.cp_size
                                loss.backward()

                    if not is_teacher:
                        if num_valid_samples > 0:
                            mb_losses.append(loss.item())
                            all_mb_metrics.append(loss_metrics)

                if is_teacher:
                    # Ensure writes to IPC-exported buffers are complete before
                    # returning handles to student workers.
                    torch.cuda.current_stream().synchronize()
                    self.teacher_logits = {
                        'microbatch_handles': _teacher_mb_handles,
                        'is_topk': topk_logits is not None,
                    }
                    return self.teacher_logits

                grad_norm: Optional[float | torch.Tensor] = None
                if not is_teacher and not eval_mode:
                    grad_norm = scale_grads_and_clip_grad_norm(
                        self.max_grad_norm,
                        [self.model],
                        norm_type=2.0,
                        pp_enabled=False,
                        device_mesh=self.device_mesh,
                        moe_mesh=self.moe_mesh,
                        ep_axis_name="ep"
                        if self.moe_mesh is not None
                        and "ep" in self.moe_mesh.mesh_dim_names
                        else None,
                        pp_axis_name=None,
                        foreach=True,
                        num_label_tokens=1,
                        dp_group_size=self.dp_size * self.cp_size,
                    )
                    grad_norm = torch.tensor(
                        grad_norm, device="cpu", dtype=torch.float32
                    )

                    # Update parameters
                    self.optimizer.step()

                if not is_teacher:
                    losses.append(torch.tensor(mb_losses).sum().item())

            # release gradient memory before rollouts
            self.optimizer.zero_grad()
            if not eval_mode:
                self.scheduler.step()
            torch.cuda.empty_cache()

            # Compute global loss across all ranks
            with torch.no_grad():
                global_loss = torch.tensor(losses, device="cuda")
                torch.distributed.all_reduce(
                    global_loss, group=self.dp_mesh.get_group()
                )
            # Aggregate metrics across all microbatches
            mb_metrics = defaultdict(list)
            for m in all_mb_metrics:
                for k, v in m.items():
                    mb_metrics[k].append(v)

            metrics = {
                "global_loss": global_loss.cpu(),
                "grad_norm": grad_norm,
                "rank": torch.distributed.get_rank(),
                "gpu_name": torch.cuda.get_device_name(),
                "model_dtype": self.dtype,
                "all_mb_metrics": dict(mb_metrics),
            }

            return metrics

    # TODO @Rayen Tian: Related Issue: Refactor shared logic between score() and get_logprobs() (https://github.com/NVIDIA-NeMo/RL/issues/1094)
    @wrap_with_nvtx_name("dtensor_policy_worker_v2/get_logprobs")
    def get_logprobs(
        self, data: BatchedDataDict[Any], micro_batch_size: Optional[int] = None
    ) -> BatchedDataDict[LogprobOutputSpec]:
        """Get the logprobs of the model for a batch of data.

        Uses the configured logprob_batch_size to do microbatching.

        Input data is assumed to be right-padded. The method internally converts to
        left-padded format for computation, and returns outputs in right-padded format.

        Returns:
          a BatchedDataDict with key "logprobs" and shape [batch_size, sequence_length].
          We use the convention that the logprob of the first token is 0 so that the sequence length is maintained.
          The logprob of input token i is specified at position i in the output logprobs tensor.
        """
        logprob_batch_size = (
            micro_batch_size
            if micro_batch_size is not None
            else self.cfg["logprob_batch_size"]
        )
        logprob_chunk_size = self.cfg.get("logprob_chunk_size", None)

        # dim 1 is always assumed to be the sequence dim, sanity check this here
        sequence_dim = 1
        seq_dim_size = data.get("input_ids").shape[sequence_dim]
        for k, v in data.items():
            if torch.is_tensor(v) and len(v.shape) > 1:
                assert v.shape[sequence_dim] == seq_dim_size, (
                    f"Dim 1 must be the sequence dim, expected dim 1={seq_dim_size} but got shape {v.shape}"
                )

        all_log_probs = []
        self.model.eval()

        with torch.no_grad():
            data.to("cuda")
            dummy_iterator = iter([])
            if self.cfg.get("dynamic_batching", {}).get("enabled", False):
                mb_iterator = data.make_microbatch_iterator_with_dynamic_shapes()
                iterator_len = data.get_microbatch_iterator_dynamic_shapes_len()
            elif self.enable_seq_packing:
                mb_iterator = data.make_microbatch_iterator_for_packable_sequences()
                iterator_len, max_seqlen = (
                    data.get_microbatch_iterator_for_packable_sequences_len()
                )
                max_batch_ct = torch.tensor([iterator_len], device="cuda")
                torch.distributed.all_reduce(
                    max_batch_ct, op=torch.distributed.ReduceOp.MAX
                )

                # Sequence packing can end up with unevenly distributed batch counts across DP ranks.
                # We add dummy batches to the end of the iterator to make the batch counts equal.
                dummy_batch_ct = int(max_batch_ct.item() - iterator_len)
                dummy_iterator = data.make_microbatch_iterator_for_packable_sequences()
                dummy_iterator = itertools.islice(
                    itertools.cycle(dummy_iterator), dummy_batch_ct
                )
            else:
                mb_iterator = data.make_microbatch_iterator(logprob_batch_size)
                iterator_len = data.size // logprob_batch_size

            step = 0
            for batch_idx, lp_batch in enumerate(
                itertools.chain(mb_iterator, dummy_iterator)
            ):
                step += 1
                input_ids = lp_batch.get("input_ids").cuda()
                input_lengths = lp_batch.get("input_lengths")
                vlm_kwargs = lp_batch.get_multimodal_dict(
                    as_tensors=True, device=input_ids.device
                )

                batch_size, seq_len = input_ids.shape
                if self.enable_seq_packing:
                    assert len(vlm_kwargs) == 0, (
                        "multimodal kwargs are not supported for sequence packing"
                    )
                    input_ids, position_ids, _ = pack_sequences(
                        input_ids=input_ids,
                        input_lengths=input_lengths,
                        packed_sequence_size=[
                            batch_size
                        ],  # flash attention 2 expects flattened input
                        padding_value=self.tokenizer.eos_token_id,
                        return_attention_mask=False,
                    )
                    seq_len = input_ids.shape[1]
                    attention_mask = None
                    flash_attn_kwargs = get_flash_attention_kwargs(
                        input_lengths=input_lengths,
                    )
                else:
                    # Create post_attention_mask for right-padded data for masking token after forwarding.
                    post_attention_mask = torch.zeros(
                        (batch_size, seq_len), dtype=torch.bool, device=input_ids.device
                    )
                    for i, length in enumerate(input_lengths):
                        # For right-padded sequence, set 1s at the beginning of the sequence
                        post_attention_mask[i, :length] = 1

                    # explicitly create position ids for the input, otherwise the sharding
                    # for DTensor will be incorrect
                    position_ids = torch.arange(
                        seq_len, device=input_ids.device
                    ).repeat(batch_size, 1)
                    flash_attn_kwargs = {}

                    # DTensor requires the casual attention kernel to hit,
                    # yet our attention mask above is not always all 1s
                    # this is fine because we mask with the actual attention mask
                    # later, but for input it has to be all 1s
                    attention_mask = torch.ones(
                        (batch_size, seq_len),
                        dtype=torch.bool,
                        device=input_ids.device,
                    )

                # if there are multimodal kwargs, we don't need to add position_ids (computed internally)
                if len(vlm_kwargs) > 0:
                    position_ids = None

                context_parallel_ctx = None
                if self.cp_size > 1:
                    assert len(vlm_kwargs) == 0, (
                        "multimodal kwargs are not supported for context parallel"
                    )
                    seq_index = torch.arange(seq_len, device=input_ids.device).repeat(
                        1, 1
                    )
                    cp_buffers = [input_ids, position_ids, seq_index]

                    # Create context parallel context
                    context_parallel_ctx = create_context_parallel_ctx(
                        cp_mesh=self.cp_mesh,
                        cp_buffers=cp_buffers,
                        cp_seq_dims=[sequence_dim] * len(cp_buffers),
                        cp_no_restore_buffers=set(cp_buffers),
                    )

                with get_train_context(False, False, context_parallel_ctx)():
                    with torch.autocast(device_type="cuda", dtype=self.dtype):
                        model_args = dict(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            use_cache=False,
                            flash_attn_kwargs=flash_attn_kwargs,
                            **vlm_kwargs,
                        )
                        if len(vlm_kwargs) > 0:
                            del model_args["flash_attn_kwargs"]

                        if (
                            not self.allow_flash_attn_args
                            and "flash_attn_kwargs" in model_args
                        ):
                            del model_args["flash_attn_kwargs"]

                        outputs = self.model(**model_args)

                    logits = outputs.logits if hasattr(outputs, "logits") else outputs

                    # Apply temperature scaling
                    logits = self._apply_temperature_scaling(logits)

                    if self.cp_size > 1:
                        seq_index_tensor = (
                            DTensor.from_local(
                                seq_index,
                                device_mesh=self.cp_mesh,
                                placements=[Shard(1)],
                            )
                            .full_tensor()
                            .squeeze(0)
                        )

                        input_ids_dtensor = DTensor.from_local(
                            input_ids,
                            device_mesh=self.cp_mesh,
                            placements=[Shard(sequence_dim)],
                        )

                        if isinstance(logits, DTensor):
                            # Must be tp sharded
                            assert (
                                logits.device_mesh.ndim == 1
                                and logits.device_mesh.mesh_dim_names[0] == "tp"
                            ), "logits must be tp sharded"

                            # CP is implicitly sharded on the seq dim, so we need to redistribute to the tp dim
                            logits = DTensor.from_local(
                                logits.to_local(),
                                device_mesh=self.device_mesh[("cp", "tp")],
                                placements=[Shard(sequence_dim), Shard(-1)],
                            )
                        else:
                            logits = DTensor.from_local(
                                logits,
                                device_mesh=self.device_mesh[("cp", "tp")],
                                placements=[Shard(sequence_dim), Shard(-1)],
                            )

                        token_logprobs = get_logprobs_from_vocab_parallel_logits(
                            logits,
                            input_ids_dtensor,
                            seq_index_tensor,
                            chunk_size=logprob_chunk_size,
                        )

                        assert token_logprobs.shape[1] == seq_len - 1
                    else:
                        if isinstance(logits, DTensor):
                            token_logprobs = get_logprobs_from_vocab_parallel_logits(
                                logits,
                                input_ids,
                                chunk_size=logprob_chunk_size,
                            )
                        else:
                            if logprob_chunk_size is not None:
                                logits_seq_len = int(logits.shape[1])
                                num_chunks = (
                                    logits_seq_len + logprob_chunk_size - 1
                                ) // logprob_chunk_size
                                chunked_log_probs = []
                                for chunk_idx in range(num_chunks):
                                    chunk_start = chunk_idx * logprob_chunk_size
                                    chunk_end = min(
                                        logits_seq_len,
                                        (chunk_idx + 1) * logprob_chunk_size,
                                    )
                                    chunk_logits = logits[
                                        :, chunk_start:chunk_end, :
                                    ].to(torch.float32)
                                    log_probs = torch.nn.functional.log_softmax(
                                        chunk_logits, dim=-1
                                    )
                                    chunked_log_probs.append(log_probs)
                                log_probs = torch.cat(chunked_log_probs, dim=1)
                                del chunked_log_probs
                            else:
                                logits = logits.to(torch.float32)
                                log_probs = torch.nn.functional.log_softmax(
                                    logits, dim=-1
                                )
                            # Extract logprobs for each token in the sequence by gathering the logprob
                            # corresponding to the next token at each position
                            # Input shapes:
                            #   log_probs: [batch_size, sequence_length, vocab_size] - logits for each position
                            #   token_ids: [batch_size, sequence_length] - actual tokens
                            # Output shape: [batch_size, sequence_length] - logprob of each token given previous
                            # We get logprob of token[t+1] from logits[t], prepending 0 to maintain sequence length
                            next_tokens = input_ids[:, 1:]
                            log_probs = log_probs[:, :-1]
                            token_logprobs = log_probs.gather(
                                dim=-1, index=next_tokens.unsqueeze(-1)
                            ).squeeze(-1)
                            del log_probs

                del outputs, logits

                token_logprobs = torch.cat(
                    [torch.zeros_like(token_logprobs[:, :1]), token_logprobs], dim=1
                )

                # skip keeping the logprobs for the dummy batches
                if batch_idx >= iterator_len:
                    continue

                if not self.enable_seq_packing:
                    # Apply mask to zero out padding tokens logprobs
                    token_logprobs = token_logprobs * post_attention_mask
                else:
                    # For packed sequences, unpack logprobs
                    unpacked_logprobs = torch.zeros(
                        (batch_size, seq_dim_size),
                        dtype=token_logprobs.dtype,
                        device=token_logprobs.device,
                    )
                    cu_seqlens = flash_attn_kwargs.cu_seqlens_q
                    for i in range(batch_size):
                        start = cu_seqlens[i].item() + 1
                        end = cu_seqlens[i + 1].item()
                        seq_len_actual = input_lengths[i].item()
                        unpacked_logprobs[i, 1:seq_len_actual] = token_logprobs[
                            0, start:end
                        ]
                    token_logprobs = unpacked_logprobs

                all_log_probs.append(token_logprobs)

        # Concatenate all batches
        return_data = BatchedDataDict[LogprobOutputSpec]()

        all_log_probs_padded = []
        for lp in all_log_probs:
            padding_needed = seq_dim_size - lp.shape[1]
            if padding_needed > 0:
                lp = torch.nn.functional.pad(
                    lp, (0, padding_needed), mode="constant", value=0.0
                )
            all_log_probs_padded.append(lp)
        return_data["logprobs"] = torch.cat(all_log_probs_padded, dim=0).cpu()

        return return_data

    # TODO @Rayen Tian: Related Issue: Refactor shared logic between score() and get_logprobs() (https://github.com/NVIDIA-NeMo/RL/issues/1094)
    @wrap_with_nvtx_name("dtensor_policy_worker_v2/score")
    def score(self, data: BatchedDataDict) -> BatchedDataDict[ScoreOutputSpec]:
        global_batch_size = min(self.cfg["batch_size"], data.size)

        sequence_dim = 1
        seq_dim_size = data.get("input_ids").shape[sequence_dim]
        for k, v in data.items():
            if torch.is_tensor(v) and len(v.shape) > 1:
                assert v.shape[sequence_dim] == seq_dim_size, (
                    f"Dim 1 must be the sequence dim, expected dim 1={seq_dim_size} but got shape {v.shape}"
                )
        self.model.eval()
        print("Begin to batch datas")
        with torch.no_grad():
            data.to("cuda")
            dummy_iterator = iter([])
            if self.cfg.get("dynamic_batching", {}).get("enabled", False):
                mb_iterator = data.make_microbatch_iterator_with_dynamic_shapes()
                iterator_len = data.get_microbatch_iterator_dynamic_shapes_len()
            elif self.enable_seq_packing:
                mb_iterator = data.make_microbatch_iterator_for_packable_sequences()
                iterator_len, max_seqlen = (
                    data.get_microbatch_iterator_for_packable_sequences_len()
                )
                max_batch_ct = torch.tensor([iterator_len], device="cuda")
                torch.distributed.all_reduce(
                    max_batch_ct, op=torch.distributed.ReduceOp.MAX
                )
                dummy_batch_ct = int(max_batch_ct.item() - iterator_len)
                dummy_iterator = data.make_microbatch_iterator_for_packable_sequences()
                dummy_iterator = itertools.islice(
                    itertools.cycle(dummy_iterator), dummy_batch_ct
                )
            else:
                mb_iterator = data.make_microbatch_iterator(global_batch_size)
                iterator_len = data.size // global_batch_size
            step = 0
            all_rm_scores = []
            for batch_idx, generate_batch in enumerate(
                itertools.chain(mb_iterator, dummy_iterator)
            ):
                step += 1
                input_ids = generate_batch.get("input_ids").cuda()
                input_lengths = generate_batch.get("input_lengths")
                batch_size, seq_len = input_ids.shape
                if self.enable_seq_packing:
                    input_ids, position_ids, _ = pack_sequences(
                        input_ids=input_ids,
                        input_lengths=input_lengths,
                        packed_sequence_size=[
                            batch_size
                        ],  # flash attention 2 expects flattened input
                        padding_value=self.tokenizer.eos_token_id,
                        return_attention_mask=False,
                    )
                    seq_len = input_ids.shape[1]
                    attention_mask = None
                    flash_attn_kwargs = get_flash_attention_kwargs(
                        input_lengths=input_lengths,
                    )
                else:
                    # Create attention mask for right-padded data
                    post_attention_mask = torch.zeros(
                        (batch_size, seq_len), dtype=torch.bool, device=input_ids.device
                    )
                    for i, length in enumerate(input_lengths):
                        # For right-padded sequence, set 1s at the beginning of the sequence
                        post_attention_mask[i, :length] = 1
                    position_ids = torch.arange(
                        seq_len, device=input_ids.device
                    ).repeat(batch_size, 1)

                    attention_mask = torch.ones(
                        (batch_size, seq_len),
                        dtype=torch.bool,
                        device=input_ids.device,
                    )
                context_parallel_ctx = None
                if self.cp_size > 1:
                    seq_index = torch.arange(seq_len, device=input_ids.device).repeat(
                        1, 1
                    )
                    cp_buffers = [input_ids, position_ids, seq_index]

                    # Create context parallel context
                    context_parallel_ctx = create_context_parallel_ctx(
                        cp_mesh=self.cp_mesh,
                        cp_buffers=cp_buffers,
                        cp_seq_dims=[sequence_dim] * len(cp_buffers),
                        cp_no_restore_buffers=set(cp_buffers),
                    )
                with get_train_context(False, False, context_parallel_ctx)():
                    with torch.autocast(device_type="cuda", dtype=self.dtype):
                        model_args = dict(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            use_cache=False,
                        )
                        outputs = self.model(**model_args)

                    if not hasattr(outputs, "logits"):
                        logits = self.model.lm_head(outputs.last_hidden_state)
                    else:
                        logits = outputs.logits
                    # Apply temperature scaling
                    logits = self._apply_temperature_scaling(logits)
                if isinstance(logits, DTensor):
                    logits = logits.to(torch.float32)
                else:
                    logits = outputs.logits.to(torch.float32)

                rm_scores = to_local_if_dtensor(logits)
                rm_scores = rm_scores.squeeze(-1)
                all_rm_scores.append(rm_scores)

        all_rm_scores = torch.cat(all_rm_scores, dim=0)
        all_rm_scores = all_rm_scores.squeeze(-1).cpu()
        return_data = BatchedDataDict[ScoreOutputSpec](
            {
                "scores": all_rm_scores,
            }
        )
        return return_data

    @wrap_with_nvtx_name("dtensor_policy_worker_v2/get_topk_logits")
    def get_topk_logits(
        self,
        data: BatchedDataDict[Any],
        k: int,
        micro_batch_size: Optional[int] = None,
    ) -> BatchedDataDict[Any]:
        """Return per-position top-k logits and corresponding global indices.

        Notes:
        - Return shapes are [B, S, k].
        - Computes top-k over the full sequence (no trimming of the last position).
        - If alignment with next-token targets is required, the caller should handle it.
        - If logits are TP-sharded DTensor, performs distributed global top-k across TP.
        - Supports context parallelism with proper CP gather.
        - Otherwise, computes local top-k on full-vocab tensor.
        """
        topk_batch_size = (
            micro_batch_size
            if micro_batch_size is not None
            else self.cfg["logprob_batch_size"]
        )

        sequence_dim = 1
        seq_dim_size = data.get("input_ids").shape[sequence_dim]

        out_topk_vals = []
        out_topk_idx = []
        self.model.eval()

        with torch.no_grad():
            data.to("cuda")
            dummy_iterator = iter([])
            if self.cfg.get("dynamic_batching", {}).get("enabled", False):
                # dynamic batching support (no CP/packed)
                mb_iterator = data.make_microbatch_iterator_with_dynamic_shapes()
                iterator_len = data.get_microbatch_iterator_dynamic_shapes_len()
            elif self.enable_seq_packing:
                mb_iterator = data.make_microbatch_iterator_for_packable_sequences()
                iterator_len, max_seqlen = (
                    data.get_microbatch_iterator_for_packable_sequences_len()
                )
                max_batch_ct = torch.tensor([iterator_len], device="cuda")
                torch.distributed.all_reduce(
                    max_batch_ct, op=torch.distributed.ReduceOp.MAX
                )

                # Sequence packing can end up with unevenly distributed batch counts across DP ranks.
                # We add dummy batches to the end of the iterator to make the batch counts equal.
                dummy_batch_ct = int(max_batch_ct.item() - iterator_len)
                dummy_iterator = data.make_microbatch_iterator_for_packable_sequences()
                dummy_iterator = itertools.islice(
                    itertools.cycle(dummy_iterator), dummy_batch_ct
                )
            else:
                mb_iterator = data.make_microbatch_iterator(topk_batch_size)
                iterator_len = data.size // topk_batch_size

            for batch_idx, lp_batch in enumerate(
                itertools.chain(mb_iterator, dummy_iterator)
            ):
                input_ids = lp_batch.get("input_ids").cuda()
                input_lengths = lp_batch.get("input_lengths")

                batch_size, seq_len = input_ids.shape
                # Store original shapes for unpacking later
                original_batch_size = batch_size
                original_seq_len = seq_len

                if self.enable_seq_packing:
                    input_ids, position_ids, _ = pack_sequences(
                        input_ids=input_ids,
                        input_lengths=input_lengths,
                        packed_sequence_size=[
                            batch_size
                        ],  # flash attention 2 expects flattened input
                        padding_value=self.tokenizer.eos_token_id,
                        return_attention_mask=False,
                    )
                    seq_len = input_ids.shape[1]
                    attention_mask = None
                    flash_attn_kwargs = get_flash_attention_kwargs(
                        input_lengths=input_lengths,
                    )
                else:
                    # Build attention mask (right-padded inputs)
                    attention_mask = torch.zeros(
                        (batch_size, seq_len), dtype=torch.long, device=input_ids.device
                    )
                    for i, length in enumerate(input_lengths):
                        attention_mask[i, :length] = 1

                    position_ids = torch.arange(
                        seq_len, device=input_ids.device
                    ).repeat(batch_size, 1)

                    flash_attn_kwargs = {}

                with torch.autocast(device_type="cuda", dtype=self.dtype):
                    attention_mask_input_all_ones = torch.ones(
                        (batch_size, seq_len), dtype=torch.long, device=input_ids.device
                    )

                context_parallel_ctx = None
                if self.cp_size > 1:
                    seq_index = torch.arange(seq_len, device=input_ids.device).repeat(
                        1, 1
                    )
                    cp_buffers = [input_ids, position_ids, seq_index]

                    # Create context parallel context
                    context_parallel_ctx = create_context_parallel_ctx(
                        cp_mesh=self.cp_mesh,
                        cp_buffers=cp_buffers,
                        cp_seq_dims=[sequence_dim] * len(cp_buffers),
                        cp_no_restore_buffers=set(cp_buffers),
                    )

                with get_train_context(False, False, context_parallel_ctx)():
                    with torch.autocast(device_type="cuda", dtype=self.dtype):
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask_input_all_ones,
                            position_ids=position_ids,
                            use_cache=False,
                            flash_attn_kwargs=flash_attn_kwargs,
                        )

                    if not hasattr(outputs, "logits"):
                        logits = self.model.lm_head(outputs.last_hidden_state)
                    else:
                        logits = outputs.logits
                    del outputs

                    # Apply temperature scaling
                    logits = self._apply_temperature_scaling(logits)

                    if self.cp_size > 1:
                        if isinstance(logits, DTensor):
                            # Must be tp sharded
                            assert (
                                logits.device_mesh.ndim == 1
                                and logits.device_mesh.mesh_dim_names[0] == "tp"
                            ), "logits must be tp sharded"

                            # CP is implicitly sharded on the seq dim, so we need to redistribute to the tp dim
                            logits = DTensor.from_local(
                                logits.to_local(),
                                device_mesh=self.device_mesh[("cp", "tp")],
                                placements=[Shard(sequence_dim), Shard(-1)],
                            )
                        else:
                            logits = DTensor.from_local(
                                logits,
                                device_mesh=self.device_mesh[("cp", "tp")],
                                placements=[Shard(sequence_dim), Shard(-1)],
                            )

                        # deal with TP first
                        local_logits = logits.to_local()  # [B, S_cp, V_tp]

                        tp_group = self.tp_mesh.get_group()
                        tp_rank = torch.distributed.get_rank(tp_group)
                        V_local = int(local_logits.shape[-1])
                        vocab_start_index = tp_rank * V_local
                        vocab_end_index = (tp_rank + 1) * V_local

                        vals, idx = distributed_vocab_topk(
                            local_logits,
                            k=k,
                            tp_group=tp_group,
                            vocab_start_index=vocab_start_index,
                            vocab_end_index=vocab_end_index,
                        )
                        # [B, S_cp, k]

                        cp_group = self.cp_mesh.get_group()

                        vals = allgather_cp_sharded_tensor(
                            vals, cp_group, seq_dim=sequence_dim
                        )
                        idx = allgather_cp_sharded_tensor(
                            idx, cp_group, seq_dim=sequence_dim
                        )
                        # [B, S, k]
                    else:
                        # Compute top-k over full sequence length (do not drop last position)
                        if isinstance(logits, DTensor):
                            local_logits = logits.to_local()  # [B, S, V_local]
                            tp_group = self.tp_mesh.get_group()
                            tp_rank = torch.distributed.get_rank(tp_group)
                            V_local = int(local_logits.shape[-1])
                            vocab_start_index = tp_rank * V_local
                            vocab_end_index = (tp_rank + 1) * V_local

                            local_logits = local_logits.to(torch.float32)
                            local_log_probs = _compute_distributed_log_softmax(local_logits, group=tp_group)
                            del logits, local_logits

                            if isinstance(local_log_probs, DTensor):
                                local_log_probs = local_log_probs.to_local()

                            if self.cfg.get('is_mdlm', False):
                                shared_sequence_length = int(local_log_probs.shape[1] // 2)
                                local_log_probs = local_log_probs[:, shared_sequence_length:, :]

                            vals, idx = distributed_vocab_topk(
                                local_log_probs,
                                k=k,
                                tp_group=tp_group,
                                vocab_start_index=vocab_start_index,
                                vocab_end_index=vocab_end_index,
                            )
                            del local_log_probs
                        else:
                            full_logits = logits.to(torch.float32)
                            vals, idx = torch.topk(full_logits, k=k, dim=-1)

                # Handle sequence packing unpacking
                if self.enable_seq_packing:
                    # Unpack top-k results from packed format back to original batch format
                    # vals: [1, packed_seq_len, k] -> [original_batch_size, original_seq_len, k]
                    # idx: [1, packed_seq_len, k] -> [original_batch_size, original_seq_len, k]

                    # Create tensors to store unpacked results
                    unpacked_vals = torch.zeros(
                        (original_batch_size, original_seq_len, k),
                        dtype=vals.dtype,
                        device=vals.device,
                    )
                    unpacked_idx = torch.zeros(
                        (original_batch_size, original_seq_len, k),
                        dtype=idx.dtype,
                        device=idx.device,
                    )

                    # Get cumulative sequence lengths for unpacking
                    cu_seqlens = flash_attn_kwargs.cu_seqlens_q

                    for i in range(original_batch_size):
                        start = cu_seqlens[i].item()
                        end = cu_seqlens[i + 1].item()
                        seq_len_actual = input_lengths[i].item()

                        # Extract the corresponding portion from packed results
                        # Note: vals and idx are [1, packed_seq_len, k] due to packing
                        unpacked_vals[i, :seq_len_actual, :] = vals[0, start:end, :]
                        unpacked_idx[i, :seq_len_actual, :] = idx[0, start:end, :]

                    # Replace with unpacked results
                    vals = unpacked_vals
                    idx = unpacked_idx

                    # Update batch_size and seq_len for consistency
                    batch_size = original_batch_size
                    seq_len = original_seq_len

                # Shapes remain [B, S, k].
                B_mb, S_mb, K_mb = vals.shape
                target_dtype = vals.dtype
                target_device = vals.device

                # Pre-allocate two IPC buffers (values + indices) exactly once.
                if not hasattr(self, '_teacher_topk_vals_buffer') or self._teacher_topk_vals_buffer is None:
                    max_S = self.cfg.get("max_total_sequence_length", S_mb)
                    vals_buf_shape = (B_mb, max_S, K_mb)
                    self._teacher_topk_vals_buffer = torch.empty(
                        vals_buf_shape, dtype=target_dtype, device=target_device
                    )
                    self._teacher_topk_vals_ipc = {
                        torch.distributed.get_rank(): get_handle_from_tensor(self._teacher_topk_vals_buffer)
                    }
                    idx_buf_shape = (B_mb, max_S, K_mb)
                    self._teacher_topk_idx_buffer = torch.empty(
                        idx_buf_shape, dtype=idx.dtype, device=target_device
                    )
                    self._teacher_topk_idx_ipc = {
                        torch.distributed.get_rank(): get_handle_from_tensor(self._teacher_topk_idx_buffer)
                    }
                    print(f" rank {torch.distributed.get_rank()} Allocated topk IPC buffers: "
                          f"vals={vals_buf_shape} ({self._teacher_topk_vals_buffer.numel() * self._teacher_topk_vals_buffer.element_size() / 1e9:.4f} GB), "
                          f"idx={idx_buf_shape} ({self._teacher_topk_idx_buffer.numel() * self._teacher_topk_idx_buffer.element_size() / 1e9:.4f} GB) "
                          f"(actual data: [{B_mb}, {S_mb}, {K_mb}])")

                # Copy actual data into the top-left slice of the buffers
                self._teacher_topk_vals_buffer[:B_mb, :S_mb, :K_mb].copy_(vals)
                self._teacher_topk_idx_buffer[:B_mb, :S_mb, :K_mb].copy_(idx)
                del vals, idx

                out_topk_vals.append(self._teacher_topk_vals_buffer[:B_mb, :S_mb, :K_mb].cpu())
                out_topk_idx.append(self._teacher_topk_idx_buffer[:B_mb, :S_mb, :K_mb].cpu())

        ret = BatchedDataDict[Any]()
        all_topk_vals_padded = []
        all_topk_idx_padded = []
        target_seq_len = seq_dim_size
        for vals, idx in zip(out_topk_vals, out_topk_idx):
            pad_needed = target_seq_len - vals.shape[1]
            if pad_needed > 0:
                vals = torch.nn.functional.pad(
                    vals, (0, 0, 0, pad_needed, 0, 0), mode="constant", value=0.0
                )
                idx = torch.nn.functional.pad(
                    idx, (0, 0, 0, pad_needed, 0, 0), mode="constant", value=0
                )
            all_topk_vals_padded.append(vals)
            all_topk_idx_padded.append(idx)

        ret["topk_logits"] = (
            torch.cat(all_topk_vals_padded, dim=0)
            if len(all_topk_vals_padded) > 1
            else all_topk_vals_padded[0]
        ).cpu()
        ret["topk_indices"] = (
            torch.cat(all_topk_idx_padded, dim=0)
            if len(all_topk_idx_padded) > 1
            else all_topk_idx_padded[0]
        ).cpu()
        return ret

    def _ensure_teacher_mb_topk_buffer(self, buf_idx, B, K, vals_dtype, idx_dtype, device):
        """Lazily grow the per-microbatch IPC buffer pool for top-k teacher logits."""
        if not hasattr(self, '_teacher_mb_vals_buffers'):
            self._teacher_mb_vals_buffers = []
            self._teacher_mb_vals_ipcs = []
            self._teacher_mb_idx_buffers = []
            self._teacher_mb_idx_ipcs = []
        max_S = self.cfg.get("max_total_sequence_length", 1)
        while len(self._teacher_mb_vals_buffers) <= buf_idx:
            vals_buf = torch.empty((B, max_S, K), dtype=vals_dtype, device=device)
            idx_buf = torch.empty((B, max_S, K), dtype=idx_dtype, device=device)
            self._teacher_mb_vals_buffers.append(vals_buf)
            self._teacher_mb_vals_ipcs.append(get_handle_from_tensor(vals_buf))
            self._teacher_mb_idx_buffers.append(idx_buf)
            self._teacher_mb_idx_ipcs.append(get_handle_from_tensor(idx_buf))

    def _ensure_teacher_mb_logits_buffer(self, buf_idx, B, V, dtype, device):
        """Lazily grow the per-microbatch IPC buffer pool for full-vocab teacher logits."""
        if not hasattr(self, '_teacher_mb_logits_buffers'):
            self._teacher_mb_logits_buffers = []
            self._teacher_mb_logits_ipcs = []
        max_S = self.cfg.get("max_total_sequence_length", 1)
        while len(self._teacher_mb_logits_buffers) <= buf_idx:
            buf = torch.empty((B, max_S, V), dtype=dtype, device=device)
            self._teacher_mb_logits_buffers.append(buf)
            self._teacher_mb_logits_ipcs.append(get_handle_from_tensor(buf))

    @contextmanager
    def use_reference_model(self) -> Generator[None, None, None]:
        """Context manager that temporarily swaps the reference model and active model.

        On entry: Moves model to CPU, moves reference_model to CUDA. Swaps the references
        On exit: Restores original references and re-flips cuda/cpu
        """
        with torch.no_grad():
            try:
                # Save train model state_dict
                curr_state_dict = get_cpu_state_dict(
                    self.model.state_dict().items(), pin_memory=True
                )

                # Swap reference model state_dict to self.model
                for k, v in self.model.state_dict().items():
                    val = to_local_if_dtensor(v)
                    val.copy_(self.reference_model_state_dict[k])

                # - self.model is the original reference_model, now on CUDA
                # - curr_state_dict is the train model, now on CPU
                yield

            finally:
                # Restore train model state_dict
                for k, v in self.model.state_dict().items():
                    val = to_local_if_dtensor(v)
                    val.copy_(curr_state_dict[k])

    def _add_noise_to_weights(self) -> None:
        """Add small Gaussian noise to the weights of the model. Note that this is used for testing purposes only."""
        noise_std = 0.01  # Standard deviation for the noise
        for p in self.model.parameters():
            if p.requires_grad:
                noise = torch.randn_like(p.data) * noise_std
                p.data.add_(noise)  # Add noise in-place
        torch.cuda.synchronize()

    def return_state_dict(self):
        return self.model.state_dict()

    def return_model_config(self) -> dict[str, Any]:
        """Return the model configuration as a dictionary.

        Returns:
            dict: Model configuration dictionary
        """
        return self.model.config

    @torch.no_grad()
    def prepare_refit_info(self) -> Optional[dict[str, Any]]:
        """Prepare state dict metadata for weight refitting and IPC streaming."""
        state_dict_info = {}
        for name, tensor in self.model.state_dict().items():
            # all tensor will be casted to self.dtype in stream_weights_via_ipc_zmq/broadcast_weights_for_collective
            state_dict_info[name] = (tensor.shape, self.dtype)

        return state_dict_info

    @torch.no_grad()
    def calibrate_qkv_fp8_scales(
        self,
        data: BatchedDataDict[Any],
        micro_batch_size: Optional[int] = None,
        percentile: float = 99.9,
        margin: float = 1.05,
        include_q: bool = False,
    ) -> dict[str, Any]:
        """Placeholder for FP8 Q/K/V scale calibration, not implemented for DTensorPolicyWorkerV2."""
        raise NotImplementedError(
            "calibrate_qkv_fp8_scales is not implemented for DTensorPolicyWorkerV2"
        )

    @torch.no_grad()
    @wrap_with_nvtx_name("dtensor_policy_worker_v2/stream_weights_via_ipc_zmq")
    def stream_weights_via_ipc_zmq(
        self,
        buffer_size_bytes: int = 0,
        kv_scales: Optional[dict[str, float]] = None,
    ) -> None:
        """Stream model weights to peer process via ZMQ IPC socket."""
        if kv_scales is not None:
            raise NotImplementedError(
                "FP8 kvcache is not currently supported for DTensor path, we will support it in the future."
            )

        self.maybe_init_zmq()
        # Manually move model to cuda for cpu offload case
        if self.cpu_offload:
            self.model = self.move_to_cuda(self.model)

        from nemo_rl.models.policy.utils import stream_weights_via_ipc_zmq_impl

        def dtensor_params_generator():
            """Generator that yields (name, tensor) pairs, converting DTensors to local tensors."""
            for name, tensor in self.model.state_dict().items():
                if isinstance(tensor, DTensor):
                    # Convert DTensor to full tensor for streaming
                    full_tensor = tensor.full_tensor()
                    # Convert to target dtype
                    yield (
                        name,
                        full_tensor.to(self.dtype, non_blocking=True).contiguous(),
                    )
                else:
                    # Convert to target dtype
                    yield name, tensor.to(self.dtype, non_blocking=True).contiguous()

        # Use the shared implementation
        stream_weights_via_ipc_zmq_impl(
            params_generator=dtensor_params_generator(),
            buffer_size_bytes=buffer_size_bytes,
            zmq_socket=self.zmq_socket,
            rank=self.rank,
            worker_name=str(self),
        )

    @torch.no_grad()
    def broadcast_weights_for_collective(
        self, kv_scales: Optional[dict[str, float]] = None
    ) -> None:
        """Broadcast the weights for collective communication."""
        if kv_scales is not None:
            raise NotImplementedError(
                "FP8 kvcache is not currently supported for DTensor path, we will support it in the future."
            )

        # Manually move model to cuda for cpu offload case
        if self.cpu_offload:
            print(
                "[WARNING]: Unless you are lacking of memory, it is not recommended to enable cpu_offload when "
                "using non-colocated generation since it will have an extra onload and offload at refit stage."
            )
            self.model = self.move_to_cuda(self.model)

        def _dtensor_post_iter_func(tensor, dtype):
            if isinstance(tensor, DTensor):
                tensor = tensor.full_tensor()
            tensor = tensor.to(dtype, non_blocking=True)
            return tensor

        # param_iterator will return (name, tensor), we only need tensor
        dtensor_post_iter_func = lambda x: _dtensor_post_iter_func(x[1], self.dtype)

        packed_broadcast_producer(
            iterator=iter(self.model.state_dict().items()),
            group=self.model_update_group,
            src=0,
            post_iter_func=dtensor_post_iter_func,
        )

        # Manually move model to cpu for cpu offload case
        # cpu offload needs model on CPU before model forward
        if self.cpu_offload:
            self.model = self.move_to_cpu(self.model)

    @wrap_with_nvtx_name("dtensor_policy_worker_v2/prepare_for_lp_inference")
    def prepare_for_lp_inference(self) -> None:
        # onload model to cuda
        if not self.cpu_offload:
            self.move_to_cuda(self.model)
        else:
            self.model = self.move_buffer_to_device(self.model, "cuda")

        self.model.eval()

        # offload optimizer to cpu
        torch.randn(1).cuda()  # wake up torch allocator
        if self.optimizer is not None and self.offload_optimizer_for_logprob:
            self.move_optimizer_to_device("cpu")

        gc.collect()
        torch.cuda.empty_cache()

    @wrap_with_nvtx_name("dtensor_policy_worker_v2/prepare_for_training")
    def prepare_for_training(self, *args, **kwargs) -> None:
        # onload models and optimizer state to cuda
        if not self.cpu_offload:
            self.move_to_cuda(self.model)
        else:
            # when cpu offload is enabled, the buffers do not get moved
            # to cuda automatically, so we need to do that manually
            self.model = self.move_buffer_to_device(self.model, "cuda")

        self.model.train()
        # Move optimizer state to CUDA if it exists
        # colocated generation will always offload optimizer to cuda before refit
        if (
            self.optimizer is not None
            and not self.cpu_offload
            and (self.offload_optimizer_for_logprob or self.is_generation_colocated)
        ):
            self.move_optimizer_to_device("cuda")

        torch.cuda.empty_cache()

    @torch.no_grad()
    @wrap_with_nvtx_name("dtensor_policy_worker_v2/offload_before_refit")
    def offload_before_refit(self) -> None:
        """Offload the optimizer to the CPU."""
        torch.randn(1).cuda()  # wake up torch allocator
        if self.optimizer is not None:
            self.move_optimizer_to_device("cpu")

        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    @wrap_with_nvtx_name("dtensor_policy_worker_v2/offload_after_refit")
    def offload_after_refit(self) -> None:
        """Offload as much as possible on the CPU."""
        self.model = self.move_to_cpu(self.model)
        self.model.eval()
        torch.randn(1).cuda()  # wake up torch allocator
        self.offload_before_refit()  # rerun the old offload function

        # Print memory stats after offloading
        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory after optimizer offload: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )

    def move_optimizer_to_device(self, device: str | torch.device) -> None:
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, (DTensor, torch.Tensor)):
                    state[k] = v.to(device)

    def move_to_device(self, model: nn.Module, device: str | torch.device) -> nn.Module:
        model = self.move_buffer_to_device(model, device)
        return model.to(device)

    def move_buffer_to_device(
        self, model: nn.Module, device: str | torch.device
    ) -> nn.Module:
        # FSDP modules do not move buffers to the device automatically
        for v in model.buffers():
            torch.utils.swap_tensors(v, v.to(device))

        return model

    def move_to_cuda(self, model: torch.nn.Module) -> torch.nn.Module:
        model = self.move_to_device(model, "cuda")
        gc.collect()
        torch.cuda.empty_cache()
        return model

    def move_to_cpu(self, model: torch.nn.Module) -> torch.nn.Module:
        model = self.move_to_device(model, "cpu")
        gc.collect()
        torch.cuda.empty_cache()
        return model

    def save_checkpoint(
        self,
        weights_path: str,
        optimizer_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        checkpointing_cfg: Optional[CheckpointingConfig] = None,
    ) -> None:
        """Save a checkpoint of the model.

        the optimizer states are saved only if `optimizer` and `optimizer_path` are provided.
        """
        self.checkpoint_manager.save_checkpoint(
            model=self.model,
            weights_path=weights_path,
            optimizer=self.optimizer,
            optimizer_path=optimizer_path,
            scheduler=self.scheduler,
            tokenizer=self.tokenizer if tokenizer_path is None else None,
            tokenizer_path=tokenizer_path,
            checkpointing_cfg=checkpointing_cfg,
            lora_enabled=self.lora_enabled,
            peft_config=self.peft_config,
        )

    def load_checkpoint(
        self,
        weights_path: str,
        optimizer_path: Optional[str] = None,
    ) -> None:
        """Load a checkpoint into the model using Automodel Checkpointer."""
        self.checkpoint_manager.load_checkpoint(
            model=self.model,
            weights_path=weights_path,
            optimizer=self.optimizer,
            optimizer_path=optimizer_path,
            scheduler=self.scheduler,
        )

    def _init_checkpoint_manager(
        self,
        config_updates: Optional[dict[str, Any]] = None,
        checkpoint_root: Optional[str] = None,
    ) -> None:
        """Initialize the AutomodelCheckpointManager for this worker.

        This creates the checkpoint manager bound to this worker's device meshes
        and initializes its underlying checkpointer.

        Args:
            config_updates: Dict of CheckpointingConfig fields to set during initialization.
            checkpoint_root: Optional root directory for checkpoints.
        """
        if self.checkpoint_manager is None:
            self.checkpoint_manager = AutomodelCheckpointManager(
                dp_mesh=self.dp_mesh,
                tp_mesh=self.tp_mesh,
                moe_mesh=self.moe_mesh,
            )
            self.checkpoint_manager.init_checkpointer(
                config_updates=config_updates,
                checkpoint_root=checkpoint_root,
            )
