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
from typing import Any, Generator, Optional

import ray
import torch
import zmq
from nemo_automodel.components.checkpoint._backports.filesystem import (
    SerializationFormat,
)
from nemo_automodel.components.checkpoint.checkpointing import (
    Checkpointer,
)
from nemo_automodel.components.checkpoint.checkpointing import (
    CheckpointingConfig as AutomodelCheckpointingConfig,
)
from nemo_automodel.components.distributed.cp_utils import (
    create_context_parallel_ctx,
    get_train_context,
)
from nemo_automodel.components.distributed.tensor_utils import (
    get_cpu_state_dict,
    to_local_if_dtensor,
)
from torch import nn
from torch.distributed.tensor import DTensor
from transformers import (
    AutoProcessor,
    AutoTokenizer,
)

from nemo_rl.algorithms.interfaces import LossFunction
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.automodel.data import (
    get_microbatch_iterator,
    process_global_batch,
    process_microbatch,
)
from nemo_rl.models.automodel.setup import (
    setup_distributed,
    setup_model_and_optimizer,
    validate_and_set_config,
)
from nemo_rl.models.automodel.train import (
    cleanup_after_training,
    forward_backward,
    forward_with_processor,
    get_logprobs,
    get_topk_logits,
    optimizer_step,
    setup_train_loop,
)
from nemo_rl.models.automodel.types import LossInputs
from nemo_rl.models.huggingface.common import (
    get_flash_attention_kwargs,
    pack_sequences,
)
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import (
    LogprobOutputSpec,
    ReferenceLogprobOutputSpec,
    ScoreOutputSpec,
)
from nemo_rl.models.policy.utils import (
    get_gpu_info,
    get_runtime_env_for_policy_worker,
)
from nemo_rl.utils.checkpoint import CheckpointingConfig
from nemo_rl.utils.nsys import wrap_with_nvtx_name
from nemo_rl.utils.packed_tensor import packed_broadcast_producer


def _maybe_adapt_tensor_to_hf(
    model_part: nn.Module, fqn: str, tensor: torch.Tensor, quantization: bool = False
) -> list[tuple[str, torch.Tensor]]:
    adapter = getattr(model_part, "state_dict_adapter", None)
    if adapter:
        return adapter.convert_single_tensor_to_hf(
            fqn,
            tensor,
            exclude_key_regex=r".*_extra_state.*",
            quantization=quantization,
        )
    return [(fqn, tensor)]


@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("dtensor_policy_worker_v2")
)  # pragma: no cover
class DTensorPolicyWorkerV2:
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
        # Store tokenizer and processor
        self.tokenizer = tokenizer
        self.processor = processor

        # Store configuration
        self.cfg = config

        # Initialize checkpointer references
        self.checkpointer = None
        self.checkpoint_config = None

        # Validate configuration and set derived values
        # This needs a rank, but we don't have it yet, so we'll pass 0 for now
        # and update after distributed init
        validated_state = validate_and_set_config(
            config=config,
            processor=processor,
            rank=0,  # Temporary, will be updated after distributed init
        )

        # Extract runtime_config from validated_state
        self.runtime_config = validated_state.runtime_config

        print(
            f"Initializing DTensorPolicyWorkerV2 with is_vlm={self.runtime_config.is_vlm}"
        )

        # Set up distributed environment
        self.distributed_state = setup_distributed(
            config=config,
            runtime_config=self.runtime_config,
        )

        # Set up model and optimizer
        self.model_and_optimizer = setup_model_and_optimizer(
            config=config,
            tokenizer=tokenizer,
            runtime_config=self.runtime_config,
            distributed_state=self.distributed_state,
            worker_instance=self,
            init_optimizer=init_optimizer,
            init_reference_model=init_reference_model,
        )

        # Direct accessors for frequently used fields
        self.model = self.model_and_optimizer.model
        self.optimizer = self.model_and_optimizer.optimizer

        # Load checkpoint if provided
        if weights_path:
            self.load_checkpoint(weights_path, optimizer_path)
        else:
            print(
                "No weights path provided. Loaded base HF weights via Checkpointer (default policy init)"
            )

    def _apply_temperature_scaling(self, logits: torch.Tensor) -> torch.Tensor:
        if "generation" in self.cfg and self.cfg["generation"] is not None:
            logits.div_(self.cfg["generation"]["temperature"])
        return logits

    def init_collective(
        self, ip: str, port: int, world_size: int, *, train_world_size: int
    ) -> None:
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup

        pg = StatelessProcessGroup.create(
            host=ip, port=port, rank=self.distributed_state.rank, world_size=world_size
        )
        device = torch.cuda.current_device()
        self.model_update_group = PyNcclCommunicator(pg, device=device)

    def is_alive(self) -> bool:
        return True

    def reset_peak_memory_stats(self) -> None:
        torch.cuda.reset_peak_memory_stats()

    def get_gpu_info(self) -> dict[str, Any]:
        """Return information about the GPU being used by this worker."""
        return get_gpu_info(self.model)

    @wrap_with_nvtx_name("dtensor_policy_worker_v2/train")
    def train(
        self,
        data: BatchedDataDict[Any],
        loss_fn: LossFunction,
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
    ) -> dict[str, Any]:
        """Train the policy on a batch of data with a given loss function."""
        if gbs is None:
            gbs = self.cfg["train_global_batch_size"]
        if mbs is None:
            mbs = self.cfg["train_micro_batch_size"]

        # Setup training loop parameters
        setup_params = setup_train_loop(
            data, gbs, self.distributed_state.dp_size, self.distributed_state.dp_mesh
        )
        local_gbs = setup_params["local_gbs"]
        num_global_batches = setup_params["num_global_batches"]

        if eval_mode:
            ctx: AbstractContextManager[Any] = torch.no_grad()
            self.model.eval()
        else:
            ctx = nullcontext()
            # Ensure model is in training mode
            self.model.train()

        with ctx:
            # Get data from batch and move to device
            data.to("cuda")

            losses = []
            all_mb_metrics = []
            for gb_idx in range(num_global_batches):
                # Process global batch and get normalization factors
                gb_result = process_global_batch(
                    data, gb_idx, local_gbs, loss_fn, self.distributed_state.dp_mesh
                )
                batch = gb_result["batch"]
                global_valid_seqs = gb_result["global_valid_seqs"]
                global_valid_toks = gb_result["global_valid_toks"]

                self.optimizer.zero_grad()
                mb_losses = []

                # Get microbatch iterator
                mb_iterator, iterator_len, dummy_iterator = get_microbatch_iterator(
                    batch,
                    self.cfg,
                    self.runtime_config.enable_seq_packing,
                    mbs,
                    self.distributed_state.dp_mesh,
                )

                empty_cache_steps = self.cfg.get("dtensor_cfg", {}).get(
                    "clear_cache_every_n_steps"
                )
                if empty_cache_steps:
                    warnings.warn(
                        f"Emptying cache every {empty_cache_steps} microbatches, doing so unnnecessarily would incur a large performance overhead."
                    )

                for mb_idx, mb in enumerate(
                    itertools.chain(mb_iterator, dummy_iterator)
                ):
                    # Conditioanlly empty cache when sensitive to fragmentation
                    if empty_cache_steps and mb_idx % empty_cache_steps == 0:
                        torch.cuda.empty_cache()

                    with torch.autocast(
                        device_type="cuda", dtype=self.runtime_config.dtype
                    ):
                        # Process microbatch inputs
                        processed_inputs = process_microbatch(
                            mb,
                            self.tokenizer,
                            self.runtime_config.enable_seq_packing,
                            self.cfg,
                            self.distributed_state.cp_size,
                        )

                    # Create loss inputs
                    loss_inputs = LossInputs(
                        microbatch=mb,
                        loss_fn=loss_fn,
                        global_valid_seqs=global_valid_seqs,
                        global_valid_toks=global_valid_toks,
                        apply_temperature_fn=self._apply_temperature_scaling,
                    )

                    # Forward and backward pass
                    loss, loss_metrics = forward_backward(
                        model=self.model,
                        processed_inputs=processed_inputs,
                        loss_inputs=loss_inputs,
                        runtime_config=self.runtime_config,
                        distributed_state=self.distributed_state,
                        eval_mode=eval_mode,
                    )

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

                    if num_valid_samples > 0:
                        mb_losses.append(loss.item())
                        all_mb_metrics.append(loss_metrics)

                grad_norm: Optional[float | torch.Tensor] = None
                if not eval_mode:
                    grad_norm = optimizer_step(
                        optimizer=self.optimizer,
                        model=self.model,
                        runtime_config=self.runtime_config,
                        distributed_state=self.distributed_state,
                    )

                losses.append(torch.tensor(mb_losses).sum().item())

            # Cleanup after training batches
            cleanup_after_training(
                self.optimizer, self.model_and_optimizer.scheduler, eval_mode
            )

            # Compute global loss across all ranks
            with torch.no_grad():
                global_loss = torch.tensor(losses, device="cuda")
                torch.distributed.all_reduce(
                    global_loss, group=self.distributed_state.dp_mesh.get_group()
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
                "model_dtype": self.runtime_config.dtype,
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

            # Get microbatch iterator
            mb_iterator, iterator_len, dummy_iterator = get_microbatch_iterator(
                data,
                self.cfg,
                self.runtime_config.enable_seq_packing,
                logprob_batch_size,
                self.distributed_state.dp_mesh,
            )

            step = 0
            for batch_idx, lp_batch in enumerate(
                itertools.chain(mb_iterator, dummy_iterator)
            ):
                step += 1
                input_lengths = lp_batch.get("input_lengths")
                original_batch_size, original_seq_len = lp_batch.get("input_ids").shape

                # Process microbatch inputs
                with torch.autocast(
                    device_type="cuda", dtype=self.runtime_config.dtype
                ):
                    processed_inputs = process_microbatch(
                        lp_batch,
                        self.tokenizer,
                        self.runtime_config.enable_seq_packing,
                        self.cfg,
                        self.distributed_state.cp_size,
                    )

                input_ids = processed_inputs["input_ids"]

                # Create post_attention_mask for non-packed sequences (for masking later)
                post_attention_mask = None
                if not self.runtime_config.enable_seq_packing:
                    post_attention_mask = torch.zeros(
                        (original_batch_size, original_seq_len),
                        dtype=torch.bool,
                        device=input_ids.device,
                    )
                    for i, length in enumerate(input_lengths):
                        post_attention_mask[i, :length] = 1

                # Model forward pass and logprobs processing
                token_logprobs = forward_with_processor(
                    model=self.model,
                    processor_fn=get_logprobs,
                    processed_inputs=processed_inputs,
                    runtime_config=self.runtime_config,
                    distributed_state=self.distributed_state,
                    processor_kwargs={
                        "input_ids": input_ids,
                        "apply_temperature_fn": self._apply_temperature_scaling,
                        "logprob_chunk_size": logprob_chunk_size,
                    },
                )

                # skip keeping the logprobs for the dummy batches
                if batch_idx >= iterator_len:
                    continue

                if not self.runtime_config.enable_seq_packing:
                    # Apply mask to zero out padding tokens logprobs
                    token_logprobs = token_logprobs * post_attention_mask
                else:
                    # For packed sequences, unpack logprobs
                    unpacked_logprobs = torch.zeros(
                        (original_batch_size, seq_dim_size),
                        dtype=token_logprobs.dtype,
                        device=token_logprobs.device,
                    )
                    cu_seqlens = processed_inputs["flash_attn_kwargs"].cu_seqlens_q
                    for i in range(original_batch_size):
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
            if self.cfg["dynamic_batching"]["enabled"]:
                mb_iterator = data.make_microbatch_iterator_with_dynamic_shapes()
                iterator_len = data.get_microbatch_iterator_dynamic_shapes_len()
            elif self.runtime_config.enable_seq_packing:
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
                if self.runtime_config.enable_seq_packing:
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
                if self.distributed_state.cp_size > 1:
                    seq_index = torch.arange(seq_len, device=input_ids.device).repeat(
                        1, 1
                    )
                    cp_buffers = [input_ids, position_ids, seq_index]

                    # Create context parallel context
                    context_parallel_ctx = create_context_parallel_ctx(
                        cp_mesh=self.distributed_state.cp_mesh,
                        cp_buffers=cp_buffers,
                        cp_seq_dims=[sequence_dim] * len(cp_buffers),
                        cp_no_restore_buffers=set(cp_buffers),
                    )
                with get_train_context(False, False, context_parallel_ctx)():
                    with torch.autocast(
                        device_type="cuda", dtype=self.runtime_config.dtype
                    ):
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

            # Get microbatch iterator
            mb_iterator, iterator_len, dummy_iterator = get_microbatch_iterator(
                data,
                self.cfg,
                self.runtime_config.enable_seq_packing,
                topk_batch_size,
                self.distributed_state.dp_mesh,
            )

            for batch_idx, lp_batch in enumerate(
                itertools.chain(mb_iterator, dummy_iterator)
            ):
                input_lengths = lp_batch.get("input_lengths")
                original_batch_size, original_seq_len = lp_batch.get("input_ids").shape

                # Process microbatch inputs
                with torch.autocast(
                    device_type="cuda", dtype=self.runtime_config.dtype
                ):
                    processed_inputs = process_microbatch(
                        lp_batch,
                        self.tokenizer,
                        self.runtime_config.enable_seq_packing,
                        self.cfg,
                        self.distributed_state.cp_size,
                    )

                # Model forward pass and top-k processing
                vals, idx = forward_with_processor(
                    model=self.model,
                    processor_fn=get_topk_logits,
                    processed_inputs=processed_inputs,
                    runtime_config=self.runtime_config,
                    distributed_state=self.distributed_state,
                    processor_kwargs={
                        "k": k,
                        "apply_temperature_fn": self._apply_temperature_scaling,
                    },
                )

                # Skip keeping the results for the dummy batches
                if batch_idx >= iterator_len:
                    continue

                # Handle sequence packing unpacking
                if self.runtime_config.enable_seq_packing:
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
                    cu_seqlens = processed_inputs["flash_attn_kwargs"].cu_seqlens_q

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

                # Keep only real sequence tokens (no trimming here; padded positions can be masked downstream)
                # Shapes remain [B, S, k].
                out_topk_vals.append(vals.cpu())
                out_topk_idx.append(idx.cpu())

        ret = BatchedDataDict[Any]()
        # Pad each micro-batch result on sequence dim to common length (S), similar to get_logprobs
        all_topk_vals_padded = []
        all_topk_idx_padded = []
        target_seq_len = seq_dim_size
        for vals, idx in zip(out_topk_vals, out_topk_idx):
            pad_needed = target_seq_len - vals.shape[1]
            if pad_needed > 0:
                # pad along sequence dimension (second dim): (last_dim_pad_left, last_dim_pad_right, seq_pad_left, seq_pad_right, batch_pad_left, batch_pad_right)
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
                    val.copy_(self.model_and_optimizer.reference_model_state_dict[k])

                # - self.model is the original reference_model, now on CUDA
                # - curr_state_dict is the train model, now on CPU
                yield

            finally:
                # Restore train model state_dict
                for k, v in self.model.state_dict().items():
                    val = to_local_if_dtensor(v)
                    val.copy_(curr_state_dict[k])

    @wrap_with_nvtx_name("dtensor_policy_worker_v2/get_reference_policy_logprobs")
    def get_reference_policy_logprobs(
        self, data: BatchedDataDict[Any], micro_batch_size: Optional[int] = None
    ) -> BatchedDataDict[ReferenceLogprobOutputSpec]:
        """Get the logprobs from the reference policy for a batch of data.

        Returns:
          a BatchedDataDict with key "reference_logprobs" and shape [batch_size, sequence_length].
          We use the convention that the logprob of the first token is 0 so that the sequence length is maintained.
          The logprob of input token i is specified at position i in the output logprobs tensor.
        """
        with self.use_reference_model():
            reference_logprobs = self.get_logprobs(data, micro_batch_size)

        return_data = BatchedDataDict[ReferenceLogprobOutputSpec]()
        return_data["reference_logprobs"] = reference_logprobs["logprobs"].cpu()
        return return_data

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

    def report_device_id(self) -> str:
        """Report the UUID of the current CUDA device using NVML.

        Returns:
            str: UUID of the device in the format "GPU-xxxxx"
        """
        from nemo_rl.utils.nvml import get_device_uuid

        # Get current device index from torch
        device_idx = torch.cuda.current_device()
        # Get device UUID using NVML
        return get_device_uuid(device_idx)

    def get_zmq_address(self):
        """Get the ZMQ address for the current device."""
        return f"ipc:///tmp/{self.report_device_id()}.sock"

    def maybe_init_zmq(self):
        """Initialize the ZMQ socket if it doesn't exist."""
        if not hasattr(self, "zmq_socket"):
            self.zmq_context = zmq.Context()
            self.zmq_socket = self.zmq_context.socket(zmq.REQ)
            self.zmq_socket.setsockopt(
                zmq.SNDTIMEO, 120000
            )  # set timeout to 120 seconds
            self.zmq_socket.setsockopt(
                zmq.RCVTIMEO, 120000
            )  # set timeout to 120 seconds
            self.zmq_socket.setsockopt(zmq.LINGER, 0)
            self.zmq_socket.bind(self.get_zmq_address())

    @torch.no_grad()
    def prepare_refit_info(self) -> Optional[dict[str, Any]]:
        """Prepare state dict metadata for weight refitting and IPC streaming."""
        state_dict_info = {}
        for name, tensor in self.model.state_dict().items():
            full_tensor = (
                tensor.full_tensor() if isinstance(tensor, DTensor) else tensor
            )
            # all tensor will be casted to self.runtime_config.dtype in stream_weights_via_ipc_zmq/broadcast_weights_for_collective
            adapted_fqn_tensors = _maybe_adapt_tensor_to_hf(
                self.model, name, full_tensor
            )
            for adapted_fqn, adapted_tensor in adapted_fqn_tensors:
                state_dict_info[adapted_fqn] = (
                    adapted_tensor.shape,
                    self.runtime_config.dtype,
                )

        return state_dict_info

    def get_free_memory_bytes(self) -> int:
        """Get the available free memory."""
        from nemo_rl.utils.nvml import get_free_memory_bytes

        device_idx = torch.cuda.current_device()
        return get_free_memory_bytes(device_idx)

    @torch.no_grad()
    @wrap_with_nvtx_name("dtensor_policy_worker_v2/stream_weights_via_ipc_zmq")
    def stream_weights_via_ipc_zmq(self, buffer_size_bytes: int = 0) -> None:
        """Stream model weights to peer process via ZMQ IPC socket."""
        self.maybe_init_zmq()
        # Manually move model to cuda for cpu offload case
        if self.runtime_config.cpu_offload:
            self.model = self.move_to_cuda(self.model)

        from nemo_rl.models.policy.utils import stream_weights_via_ipc_zmq_impl

        def dtensor_params_generator():
            """Generator that yields (name, tensor) pairs, converting DTensors to local tensors."""
            for name, tensor in self.model.state_dict().items():
                full_tensor = (
                    tensor.full_tensor() if isinstance(tensor, DTensor) else tensor
                )
                adapted_fqn_tensors = _maybe_adapt_tensor_to_hf(
                    self.model, name, full_tensor
                )
                for adapted_fqn, adapted_tensor in adapted_fqn_tensors:
                    # Convert to target dtype
                    yield (
                        adapted_fqn,
                        adapted_tensor.to(
                            self.runtime_config.dtype, non_blocking=True
                        ).contiguous(),
                    )

        # Use the shared implementation
        stream_weights_via_ipc_zmq_impl(
            params_generator=dtensor_params_generator(),
            buffer_size_bytes=buffer_size_bytes,
            zmq_socket=self.zmq_socket,
            rank=self.distributed_state.rank,
            worker_name=str(self),
        )

    @torch.no_grad()
    def broadcast_weights_for_collective(self) -> None:
        """Broadcast the weights for collective communication."""
        # Manually move model to cuda for cpu offload case
        if self.runtime_config.cpu_offload:
            print(
                "[WARNING]: Unless you are lacking of memory, it is not recommended to enable cpu_offload when "
                "using non-colocated generation since it will have an extra onload and offload at refit stage."
            )
            self.model = self.move_to_cuda(self.model)

        def dtensor_params_generator():
            """Generator that yields (name, tensor) pairs, converting DTensors to local tensors and adapting to HF format."""
            for name, tensor in self.model.state_dict().items():
                full_tensor = (
                    tensor.full_tensor() if isinstance(tensor, DTensor) else tensor
                )
                adapted_fqn_tensors = _maybe_adapt_tensor_to_hf(
                    self.model, name, full_tensor
                )
                for adapted_fqn, adapted_tensor in adapted_fqn_tensors:
                    # Convert to target dtype
                    yield (
                        adapted_fqn,
                        adapted_tensor.to(
                            self.runtime_config.dtype, non_blocking=True
                        ).contiguous(),
                    )

        # param_iterator will return (name, tensor), we only need tensor
        dtensor_post_iter_func = lambda x: x[1]

        packed_broadcast_producer(
            iterator=dtensor_params_generator(),
            group=self.model_update_group,
            src=0,
            post_iter_func=dtensor_post_iter_func,
        )

        # Manually move model to cpu for cpu offload case
        # cpu offload needs model on CPU before model forward
        if self.runtime_config.cpu_offload:
            self.model = self.move_to_cpu(self.model)

    @wrap_with_nvtx_name("dtensor_policy_worker_v2/prepare_for_lp_inference")
    def prepare_for_lp_inference(self) -> None:
        # onload model to cuda
        if not self.runtime_config.cpu_offload:
            self.move_to_cuda(self.model)
        else:
            self.model = self.move_buffer_to_device(self.model, "cuda")

        self.model.eval()

        # offload optimizer to cpu
        torch.randn(1).cuda()  # wake up torch allocator
        if (
            self.optimizer is not None
            and self.runtime_config.offload_optimizer_for_logprob
        ):
            self.move_optimizer_to_device("cpu")

        gc.collect()
        torch.cuda.empty_cache()

    @wrap_with_nvtx_name("dtensor_policy_worker_v2/prepare_for_training")
    def prepare_for_training(self, *args, **kwargs) -> None:
        # onload models and optimizer state to cuda
        if not self.runtime_config.cpu_offload:
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
            and not self.runtime_config.cpu_offload
            and (
                self.runtime_config.offload_optimizer_for_logprob
                or self.runtime_config.is_generation_colocated
            )
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
            v.data = v.data.to(device)

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
        if checkpointing_cfg is None:
            raise ValueError(
                "checkpointing_cfg must be provided when saving checkpoint"
            )

        # Extract only the checkpointing configuration keys that exist
        checkpoint_kwargs = {
            key: value
            for key, value in checkpointing_cfg.items()
            if key
            in {
                "model_save_format",
                "save_consolidated",
                "is_peft",
                "peft_config",
                "model_cache_dir",
                "model_repo_id",
                "is_async",
                "dequantize_base_checkpoint",
            }
        }

        checkpoint_root = _infer_checkpoint_root(weights_path)

        # Ensure a persistent Checkpointer exists and is configured
        self._ensure_checkpointer(
            config_updates=checkpoint_kwargs, checkpoint_root=checkpoint_root
        )

        self.checkpointer.save_model(
            model=self.model,
            weights_path=weights_path,
            peft_config=checkpoint_kwargs.get("peft_config"),
            tokenizer=self.tokenizer if tokenizer_path is None else None,
        )

        if optimizer_path and self.optimizer is not None:
            self.checkpointer.save_optimizer(
                optimizer=self.optimizer,
                model=self.model,
                weights_path=optimizer_path,
                scheduler=self.model_and_optimizer.scheduler,
            )

        # TODO: needed?
        if tokenizer_path and self.tokenizer is not None:
            print(f"Saving tokenizer (or processor) to {tokenizer_path}")
            self.tokenizer.save_pretrained(tokenizer_path)

    def load_checkpoint(
        self,
        weights_path: str,
        optimizer_path: Optional[str] = None,
    ) -> None:
        """Load a checkpoint into the model using Automodel Checkpointer."""
        print(f"Loading weights from {weights_path}")

        model_save_format, is_peft = detect_checkpoint_format(weights_path)

        weights_dir = os.path.dirname(weights_path)
        checkpoint_root = (
            os.path.dirname(weights_dir)
            if weights_dir.endswith("weights")
            else weights_dir
        )

        # Ensure a persistent Checkpointer exists and is configured
        self._ensure_checkpointer(
            config_updates={
                "model_save_format": model_save_format,
                "is_peft": is_peft,
            },
            checkpoint_root=checkpoint_root,
        )

        model_dir = (
            weights_path
            if weights_path.endswith("/model")
            else os.path.join(weights_path, "model")
        )

        self.checkpointer.load_model(
            model=self.model,
            model_path=model_dir,
        )

        if optimizer_path and self.optimizer is not None:
            self.checkpointer.load_optimizer(
                optimizer=self.optimizer,
                model=self.model,
                weights_path=optimizer_path,
                scheduler=self.model_and_optimizer.scheduler,
            )

    def _ensure_checkpointer(
        self, config_updates=None, checkpoint_root: Optional[str] = None
    ) -> None:
        """Create or update a persistent Automodel Checkpointer bound to this worker ranks.

        Args:
            config_updates: Dict of CheckpointingConfig fields to update.
            checkpoint_root: Optional root directory for checkpoints.
        """
        if config_updates is None:
            config_updates = {}

        # Compute dp/tp ranks
        dp_rank = torch.distributed.get_rank(self.distributed_state.dp_mesh.get_group())
        tp_rank = torch.distributed.get_rank(self.distributed_state.tp_mesh.get_group())
        pp_rank = 0

        if self.checkpointer is None:
            # Initialize a base config with sensible defaults
            base_cfg = AutomodelCheckpointingConfig(
                enabled=True,
                checkpoint_dir=checkpoint_root or "",
                model_save_format=config_updates.get(
                    "model_save_format", "safetensors"
                ),
                model_cache_dir=config_updates.get("model_cache_dir", ""),
                model_repo_id=config_updates.get("model_repo_id", ""),
                save_consolidated=config_updates.get("save_consolidated", False),
                is_peft=config_updates.get("is_peft", False),
                model_state_dict_keys=getattr(self, "model_state_dict_keys", None),
                is_async=config_updates.get("is_async", False),
                dequantize_base_checkpoint=config_updates.get(
                    "dequantize_base_checkpoint", False
                ),
            )
            self.checkpoint_config = base_cfg
            self.checkpointer = Checkpointer(
                config=base_cfg,
                dp_rank=dp_rank,
                tp_rank=tp_rank,
                pp_rank=pp_rank,
                moe_mesh=self.distributed_state.moe_mesh,
            )
        else:
            # Update mutable config fields on the existing instance
            cfg = self.checkpointer.config
            if checkpoint_root is not None:
                cfg.checkpoint_dir = checkpoint_root
            for k, v in config_updates.items():
                if k == "model_save_format":
                    # Ensure enum type
                    v = SerializationFormat[v.upper()] if isinstance(v, str) else v
                setattr(cfg, k, v)
            # Ensure model_state_dict_keys is current
            if getattr(self, "model_state_dict_keys", None) is not None:
                cfg.model_state_dict_keys = (
                    self.model_and_optimizer.model_state_dict_keys
                )

    def shutdown(self) -> None:
        """Shutdown the policy."""
        # Clean up extension resources like ZMQ sockets
        if hasattr(self, "zmq_socket"):
            self.zmq_socket.close()
            self.zmq_context.term()
        # Close checkpointer resources
        if hasattr(self, "checkpointer") and self.checkpointer is not None:
            self.checkpointer.close()

    def start_gpu_profiling(self) -> None:
        """Start GPU profiling."""
        torch.cuda.profiler.start()

    def stop_gpu_profiling(self) -> None:
        """Stop GPU profiling."""
        torch.cuda.profiler.stop()

    def report_node_ip_and_gpu_id(self) -> list[tuple[str, int]]:
        """Report the node IP and GPU ID of the current worker."""
        ip = ray._private.services.get_node_ip_address()
        gpu_id = ray.get_gpu_ids()[0]
        return (ip, gpu_id)


def detect_checkpoint_format(weights_path: str) -> tuple[str, bool]:
    """Detect model save format and PEFT status from checkpoint directory.

    Args:
        weights_path: Path to the checkpoint directory (e.g., weights/model)

    Returns:
        tuple: (model_save_format, is_peft) where:
               model_save_format is "torch_save" for DCP or "safetensors" for safetensors
               is_peft is True if PEFT/adapter patterns are detected
    """
    is_peft = False
    model_save_format = "safetensors"
    try:
        # Iterate through all subdirectories and files recursively
        all_files = []
        for root, dirs, files in os.walk(weights_path):
            all_files.extend(files)

        if any(f.endswith(".distcp") for f in all_files):
            model_save_format = "torch_save"
        elif any(f.endswith(".safetensors") for f in all_files):
            model_save_format = "safetensors"
        elif any(f.endswith((".bin", ".pt", ".pth")) for f in all_files):
            model_save_format = "torch_save"

        if not is_peft:
            is_peft = any("adapter" in f.lower() for f in all_files)

    except (OSError, PermissionError):
        pass

    return model_save_format, is_peft


def _infer_checkpoint_root(weights_path: str) -> str:
    """Infer checkpoint root directory from weights path.

    When weights_path ends with "…/weights/model", we need the parent of
    the weights directory (the checkpoint root), not the weights directory itself.

    Args:
        weights_path: Path to model weights (e.g., "/path/to/policy/weights/model")

    Returns:
        str: Checkpoint root directory (e.g., "/path/to/policy")
    """
    weights_dir = os.path.dirname(weights_path)
    if weights_dir.endswith("weights"):
        return os.path.dirname(weights_dir)
    return weights_dir
