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
import os
import warnings
from collections import defaultdict
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Any, Generator, Optional, cast

import ray
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel,
    MixedPrecision,
)
from torch.distributed.fsdp.api import ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase
from transformers.integrations.accelerate import find_tied_parameters

from nemo_rl.algorithms.interfaces import LossFunction, LossType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import (
    LogprobOutputSpec,
    ReferenceLogprobOutputSpec,
)
from nemo_rl.models.policy.utils import (
    get_gpu_info,
    import_class_from_path,
    sliding_window_overwrite,
)
from nemo_rl.utils.native_checkpoint import (
    load_checkpoint,
    save_checkpoint,
)


@ray.remote
class FSDP1PolicyWorker:
    def __repr__(self) -> str:
        """Customizes the actor's prefix in the Ray logs.

        This makes it easier to identify which worker is producing specific log messages.
        """
        if torch.distributed.is_initialized():
            return f"{self.__class__.__name__}[rank={torch.distributed.get_rank()}]"
        else:
            return f"{self.__class__.__name__}"

    def __init__(
        self,
        config: PolicyConfig,
        tokenizer: PreTrainedTokenizerBase,
        weights_path: Optional[str] = None,
        optimizer_path: Optional[str] = None,
        init_optimizer: bool = True,
        init_reference_model: bool = True,
        **kwargs: Any,
    ):
        self.cfg = config
        # torch distributed init. Envars for rank, world_size, and master_addr and master_port are set from the ray remote call
        torch.distributed.init_process_group(backend="nccl")
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        model_name = self.cfg["model_name"]
        if self.cfg["precision"] == "float32":
            self.dtype = torch.float32
        elif self.cfg["precision"] == "bfloat16":
            self.dtype = torch.bfloat16
        else:
            raise ValueError(f"Unknown precision: {self.cfg['precision']}")

        print(f"[Rank {rank}] Loading model {model_name} on CPU...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cpu",  # load weights onto CPU initially
            # Always load the model in float32 to keep master weights in float32.
            # Keeping the master weights in lower precision has shown to cause issues with convergence.
            # https://github.com/NVIDIA-NeMo/RL/issues/279 will fix the issue of CPU OOM for larger models.
            torch_dtype=torch.float32,
            trust_remote_code=True,
            **sliding_window_overwrite(
                model_name
            ),  # due to https://github.com/huggingface/transformers/issues/38002
        )
        # caching since this property is not always preserved after FSDP
        self.num_tied_weights = len(find_tied_parameters(self.model))

        if init_reference_model:
            self.reference_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="cpu",  # load weights onto CPU initially
                torch_dtype=torch.float32,  # use full precision in sft until https://github.com/NVIDIA-NeMo/RL/issues/13 is fixed
                trust_remote_code=True,
                **sliding_window_overwrite(
                    model_name
                ),  # due to https://github.com/huggingface/transformers/issues/38002
            )
        else:
            self.reference_model = None

        self.tokenizer = tokenizer

        # ------------------------------------------------
        # 3) Move to GPU + Composable FSDP
        #    (Initialize device mesh, shard submodules, then shard entire model)
        # ------------------------------------------------

        def do_fsdp(model: torch.nn.Module) -> torch.nn.Module:
            if world_size == 1:
                print(
                    "[INFO] Using a single GPU - skipping FSDP wrapper to avoid GPU memory offloading issues"
                )
                return model

            # Create a device mesh with 'world_size' GPUs in a 1D arrangement.
            mesh = init_device_mesh("cuda", (world_size,))
            mp_policy = MixedPrecision(
                param_dtype=self.dtype,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
            )

            cpu_offload = (
                CPUOffload(offload_params=True)
                if self.cfg["fsdp_offload_enabled"]
                else None
            )

            return FullyShardedDataParallel(
                model,
                device_mesh=mesh,
                auto_wrap_policy=size_based_auto_wrap_policy,
                mixed_precision=mp_policy,
                cpu_offload=cpu_offload,
            )

        self.model.to("cuda")
        if self.cfg["activation_checkpointing_enabled"]:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.model = do_fsdp(self.model)
        self.model = self.manual_offload_to_cpu(self.model)
        if self.reference_model is not None:
            self.reference_model.to("cuda")
            self.reference_model = do_fsdp(self.reference_model)
            self.reference_model = self.manual_offload_to_cpu(self.reference_model)
        self.model = self.manual_load_to_gpu(self.model)

        # used for streaming update inference engine weights
        self._held_sharded_state_dict_reference: Optional[dict[str, Any]] = None
        self._held_streamed_param_reference: Optional[dict[str, Any]] = None

        # register_fsdp_forward_method(self.model, "generate")
        if init_optimizer:
            optimizer_cls = import_class_from_path(self.cfg["optimizer"]["name"])
            self.optimizer = optimizer_cls(
                self.model.parameters(), **self.cfg["optimizer"]["kwargs"]
            )
        else:
            self.optimizer = None

        if "scheduler" in self.cfg and self.optimizer is not None:
            if isinstance(self.cfg["scheduler"], dict):
                scheduler_cls = import_class_from_path(
                    cast(str, self.cfg["scheduler"]["name"])
                )
                self.scheduler = scheduler_cls(
                    self.optimizer, **self.cfg["scheduler"]["kwargs"]
                )
            else:
                schedulers = []
                for scheduler_cfg in self.cfg["scheduler"]:
                    if "name" in scheduler_cfg:
                        schedulers.append(
                            import_class_from_path(scheduler_cfg["name"])(
                                self.optimizer, **scheduler_cfg["kwargs"]
                            )
                        )
                    else:
                        assert "milestones" in scheduler_cfg, (
                            "unknown scheduler config: ",
                            scheduler_cfg,
                        )
                        milestones: list[int] = scheduler_cfg["milestones"]

                self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer, schedulers, milestones
                )

        elif self.optimizer is not None:
            ## default to a passthrough LR schedule
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda epoch: 1
            )

        # restore
        if weights_path:
            self.load_checkpoint(
                weights_path,
                optimizer_path,
            )
        else:
            print(
                "No weights path provided. Starting from scratch (default policy init)"
            )

    def is_alive(self) -> bool:
        return True

    def reset_peak_memory_stats(self) -> None:
        torch.cuda.reset_peak_memory_stats()

    def get_gpu_info(self) -> dict[str, Any]:
        """Return information about the GPU being used by this worker."""
        return get_gpu_info(self.model)

    def train(
        self,
        data: BatchedDataDict[Any],
        loss_fn: LossFunction,
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
    ) -> dict[str, Any]:
        """Train the policy on a batch of data with a given loss function."""
        # Check if the model has tied weights
        skip_tie_check = os.environ.get("NRL_SKIP_TIED_WEIGHT_CHECK")
        if self.num_tied_weights != 0 and not skip_tie_check:
            raise ValueError(
                f"Using FSP1 with a model ({self.cfg['model_name']}) that has tied weights (num_tied_weights={self.num_tied_weights}) is not supported (https://github.com/NVIDIA-NeMo/RL/issues/227). Please use dtensor policy with tensor parallel == 1 instead."
            )

        if gbs is None:
            gbs = self.cfg["train_global_batch_size"]
        if mbs is None:
            mbs = self.cfg["train_micro_batch_size"]
        local_gbs = gbs // torch.distributed.get_world_size()
        dataset_size = data["input_ids"].shape[0]
        num_global_batches = dataset_size // local_gbs

        if eval_mode:
            ctx: AbstractContextManager = torch.no_grad()
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
            for gb_start in range(0, dataset_size, local_gbs):
                global_batch: BatchedDataDict[Any] = data.slice(
                    gb_start, gb_start + local_gbs
                )

                assert "sample_mask" in global_batch, (
                    "sample_mask must be present in the data!"
                )
                ## get the normalization factor for the loss
                local_valid_seqs = torch.sum(global_batch["sample_mask"])

                if not "token_mask" in global_batch:
                    local_valid_toks = (
                        local_valid_seqs * global_batch["input_ids"].shape[1]
                    )
                else:
                    local_valid_toks = torch.sum(
                        global_batch["token_mask"][:, 1:]
                        * global_batch["sample_mask"].unsqueeze(-1)
                    )

                to_reduce = torch.tensor([local_valid_seqs, local_valid_toks]).cuda()
                torch.distributed.all_reduce(to_reduce)
                global_valid_seqs, global_valid_toks = to_reduce[0], to_reduce[1]

                if (
                    hasattr(loss_fn, "loss_type")
                    and loss_fn.loss_type == LossType.TOKEN_LEVEL
                ):
                    assert "token_mask" in global_batch, (
                        "token_mask must be present in the data when using token-level loss"
                    )

                self.optimizer.zero_grad()
                mb_losses = []

                # Calculate number of microbatches to process
                # make_microbatch_iterator assumes that the batch size is a multiple of the microbatch size
                # so its safe to not check for the case where the last data slice is smaller than mbs
                num_microbatches = min(local_gbs, dataset_size - gb_start) // mbs

                for mb in global_batch.make_microbatch_iterator(mbs):
                    input_ids = mb["input_ids"]

                    input_lengths = mb["input_lengths"]
                    batch_size, seq_len = input_ids.shape
                    attention_mask = torch.ones(
                        (batch_size, seq_len), dtype=torch.long, device=input_ids.device
                    )
                    for i, length in enumerate(input_lengths):
                        # For right-padded sequence, set 1s at the beginning of the sequence
                        attention_mask[i, :length] = 1

                    with torch.autocast(device_type="cuda", dtype=self.dtype):
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=False,
                        )
                        # Get logprobs
                        if not hasattr(outputs, "logits"):
                            logits = self.model.lm_head(outputs.last_hidden_state)
                        else:
                            logits = outputs.logits

                    # Divide logits by temperature
                    if "generation" in self.cfg and self.cfg["generation"] is not None:
                        logits.div_(self.cfg["generation"]["temperature"])

                    loss, loss_metrics = loss_fn(
                        logits, mb, global_valid_seqs, global_valid_toks
                    )
                    ## scale by the number of global batches so we get the correct
                    ## value when summing metrics across all microbatches
                    for k in loss_metrics.keys():
                        loss_metrics[k] /= num_global_batches
                    num_valid_samples = loss_metrics["num_valid_samples"]
                    loss_metrics["lr"] = self.optimizer.param_groups[0]["lr"]
                    loss_metrics["global_valid_seqs"] = global_valid_seqs.item()
                    loss_metrics["global_valid_toks"] = global_valid_toks.item()

                    # Backward pass
                    if not eval_mode:
                        ## NOTE: invalid samples should be multiplied
                        ## by zero in the loss function to prevent them
                        ## from affecting the gradient calculation

                        # when FSDP reduces the gradients over the DP dim, they're automatically averaged
                        # but we want to sum them so we cancel out the average here
                        loss *= torch.distributed.get_world_size()
                        loss.backward()
                    if num_valid_samples > 0:
                        mb_losses.append(loss.item())
                        all_mb_metrics.append(loss_metrics)

                # Clip gradients
                if not eval_mode:
                    if self.cfg["max_grad_norm"] is None:
                        max_grad_norm = 9999999999.0
                    else:
                        max_grad_norm = self.cfg["max_grad_norm"]

                    if isinstance(self.model, FullyShardedDataParallel):
                        # when using FSDP1, use FSDP's clip_grad_norm_
                        # to ensure grad norm is being computed over all parameters
                        # see https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.clip_grad_norm_
                        grad_norm = self.model.clip_grad_norm_(max_norm=max_grad_norm)
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=max_grad_norm
                        )
                    grad_norm = grad_norm.cpu()

                    # Update parameters
                    self.optimizer.step()
                else:
                    grad_norm = None
                losses.append(torch.tensor(mb_losses).sum().item())

            # increment scheduler after all batches in rollout are processed
            if not eval_mode:
                self.scheduler.step()

            # Compute global loss across all ranks
            with torch.no_grad():
                global_loss = torch.tensor(losses, device="cuda")
                torch.distributed.all_reduce(global_loss)

            # Aggregate metrics across all microbatches
            mb_metrics = defaultdict(list)
            for m in all_mb_metrics:
                for k, v in m.items():
                    mb_metrics[k].append(v)

            metrics = {
                "global_loss": global_loss.cpu(),
                "grad_norm": grad_norm,
                "rank": torch.distributed.get_rank(),
                "all_mb_metrics": dict(mb_metrics),
            }

            return metrics

    def get_logprobs(
        self, data: BatchedDataDict[Any], micro_batch_size: Optional[int] = None
    ) -> BatchedDataDict[LogprobOutputSpec]:
        """Get the logprobs of the model for a batch of data.

        If no micro-batch size is provided, uses the configured logprob_batch_size to do microbatching.

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
        all_log_probs = []
        self.model.eval()

        # Process in batches
        with torch.no_grad():
            data.to("cuda")
            for lp_batch in data.make_microbatch_iterator(logprob_batch_size):
                input_ids = lp_batch["input_ids"]
                batch_size, seq_len = input_ids.shape

                # Create attention mask
                input_lengths = lp_batch["input_lengths"]

                # Create attention mask for right-padded data
                attention_mask = torch.zeros(
                    (batch_size, seq_len), dtype=torch.long, device=input_ids.device
                )
                for i, length in enumerate(input_lengths):
                    # For right-padded sequence, set 1s at the beginning of the sequence
                    attention_mask[i, :length] = 1

                # Process with the model directly using right-padded inputs
                with torch.autocast(device_type="cuda", dtype=self.dtype):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
                log_probs = torch.nn.functional.log_softmax(
                    outputs.logits.to(torch.float32), dim=-1
                )

                # Extract logprobs for each token in the sequence by gathering the logprob
                # corresponding to the next token at each position
                # Input shapes:
                #   log_probs: [batch_size, sequence_length, vocab_size] - logits for each position
                #   token_ids: [batch_size, sequence_length] - actual tokens
                # Output shape: [batch_size, sequence_length] - logprob of each token given previous
                # We get logprob of token[t+1] from logits[t], prepending 0 to maintain sequence length
                token_ids = input_ids
                next_tokens = token_ids[:, 1:]  # Skip first token
                log_probs = log_probs[:, :-1]  # Remove last position's logits
                token_logprobs = log_probs.gather(
                    dim=-1, index=next_tokens.unsqueeze(-1)
                ).squeeze(-1)

                # Prepend 0 logprob for first token to maintain same sequence length as input
                token_logprobs = torch.cat(
                    [torch.zeros_like(token_logprobs[:, :1]), token_logprobs], dim=1
                )

                # Apply mask to zero out padding tokens logprobs
                token_logprobs = token_logprobs * attention_mask
                all_log_probs.append(token_logprobs)

        # Concatenate all batches
        return_data = BatchedDataDict[LogprobOutputSpec]()
        return_data["logprobs"] = torch.cat(all_log_probs, dim=0).cpu()

        return return_data

    @contextmanager
    def use_reference_model(self) -> Generator[None, None, None]:
        """Context manager that temporarily swaps the reference model and active model.

        On entry: Moves model to CPU, moves reference_model to CUDA. Swaps the references
        On exit: Restores original references and re-flips cuda/cpu

        """
        try:
            # Save original references
            original_model = self.model
            original_reference_model = self.reference_model

            self.model = self.manual_offload_to_cpu(self.model)
            self.reference_model = self.manual_load_to_gpu(self.reference_model)

            # Swap the references
            self.model, self.reference_model = self.reference_model, self.model
            gc.collect()
            torch.cuda.empty_cache()

            # - self.model is the original reference_model, now on CUDA
            # - self.reference_model is the original model, now on CPU
            yield

        finally:
            # Restore original references and device placement
            self.reference_model = self.manual_offload_to_cpu(original_reference_model)
            self.model = self.manual_load_to_gpu(original_model)
            gc.collect()
            torch.cuda.empty_cache()

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

    def generate(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Generate a batch of data using huggingface framework generation.

        Args:
            data: BatchedDataDict containing input_ids and input_lengths tensors

        Returns:
            BatchedDataDict conforming to GenerationOutputSpec:
                - output_ids: input + generated token IDs
                - logprobs: Log probabilities for each token
                - generation_lengths: Lengths of each response
        """
        # Verify input is right padded
        assert isinstance(data, BatchedDataDict), (
            f"data must be a BatchedDataDict, got type: {type(data)}"
        )
        assert "input_ids" in data and "input_lengths" in data, (
            f"input_ids and input_lengths must be present in the BatchedDataDict, got keys: {data.keys()}"
        )
        is_right_padded, error_msg = verify_right_padding(
            data, pad_value=self.tokenizer.pad_token_id
        )
        if not is_right_padded:
            warnings.warn(
                f"Input to vLLM worker is not properly right-padded: {error_msg}"
            )

        self.model.eval()

        # Right padded tokens are converted to left padded tokens for HF generate (https://huggingface.co/docs/transformers/main/en/llm_tutorial?padding=right+pad#padding-side)
        with torch.distributed.fsdp.FullyShardedDataParallel.summon_full_params(
            self.model, recurse=False
        ):
            # Get generation config from self.cfg
            generation_batch_size = self.cfg["generation_batch_size"]
            assert self.cfg["generation"] is not None, (
                "Generation config is not set while trying to generate"
            )
            gen_cfg = self.cfg["generation"]

            micro_batches = []

            # Process in batches
            max_length = 0
            for gen_batch in data.make_microbatch_iterator(generation_batch_size):
                # Create attention mask from input_lengths if needed for the model
                input_ids = gen_batch["input_ids"].cuda()
                input_lengths = gen_batch["input_lengths"].cuda()
                batch_size, seq_len = input_ids.shape

                # Convert right padding to left padding
                left_padded_input_ids = torch.full_like(
                    input_ids, gen_cfg["pad_token_id"]
                )
                left_padded_attention_mask = torch.zeros(
                    (batch_size, seq_len), dtype=torch.long, device=input_ids.device
                )

                for i, length in enumerate(input_lengths):
                    # Move tokens to the end of the sequence (left padding)
                    left_padded_input_ids[i, seq_len - length :] = input_ids[i, :length]
                    # Set attention mask for the actual tokens (at the end for left padding)
                    left_padded_attention_mask[i, seq_len - length :] = 1

                # this function requires all generations have the same stop strings, so we collect all here
                batch_stop_strings: list[list[str]] = gen_batch.get("stop_strings", [])
                stop_strings = set()
                for sample_stop_strings in batch_stop_strings:
                    if sample_stop_strings:
                        stop_strings.update(sample_stop_strings)

                # Add default stop strings from config
                if gen_cfg.get("stop_strings", None):
                    stop_strings.update(gen_cfg["stop_strings"])

                stop_strings: list[str] | None = (
                    list(stop_strings) if len(stop_strings) > 0 else None
                )

                if isinstance(
                    self.model, torch.distributed.fsdp.FullyShardedDataParallel
                ):
                    generation_module = self.model.module
                else:
                    generation_module = self.model
                outputs = generation_module.generate(  # type: ignore # we know it's a nn.Module
                    input_ids=left_padded_input_ids,
                    attention_mask=left_padded_attention_mask,
                    max_new_tokens=gen_cfg["max_new_tokens"],
                    do_sample=not greedy,
                    temperature=gen_cfg["temperature"],
                    top_p=gen_cfg["top_p"],
                    top_k=gen_cfg["top_k"],
                    pad_token_id=gen_cfg["pad_token_id"],
                    eos_token_id=gen_cfg["stop_token_ids"],
                    stop_strings=stop_strings,
                    tokenizer=self.tokenizer,  # needs for stop_strings
                    return_dict_in_generate=True,
                    output_scores=True,
                    synced_gpus=True,
                )
                # Get the generated sequences
                max_length = max(max_length, outputs.sequences.size(1))

                # Convert scores to log probabilities and extract the logprob of the chosen token
                scores = torch.stack(
                    outputs.scores, dim=1
                )  # [batch_size, seq_len, vocab_size]
                logprobs = torch.nn.functional.log_softmax(scores, dim=-1)

                # Get the logprobs of the actually generated tokens
                # outputs.sequences[:, -scores.size(1):] gives us just the newly generated tokens
                generated_tokens = outputs.sequences[:, -scores.size(1) :]
                token_logprobs = logprobs.gather(
                    dim=-1, index=generated_tokens.unsqueeze(-1)
                ).squeeze(-1)

                # Prepend zeros for input tokens based on original input lengths, not the padded length
                mb = {}
                mb["orig_input_lengths"] = input_lengths.clone()
                mb["generation_logprobs"] = token_logprobs
                mb["left_padded_output_ids"] = outputs.sequences

                micro_batches.append(mb)

            # Get lengths, pad, and concatenate all batches
            return_data: BatchedDataDict[GenerationOutputSpec] = (
                BatchedDataDict.from_batches(
                    micro_batches,
                    pad_value_dict={
                        "left_padded_output_ids": self.cfg["generation"]["pad_token_id"]
                    },
                )
            )

            # Calculate the lengths of generations for each sequence by finding stop tokens
            generation_lengths = []
            unpadded_sequence_lengths = []
            input_length = data["input_ids"].size(1)

            # Convert left-padded outputs back to right-padded format
            batch_size = len(return_data["left_padded_output_ids"])
            max_seq_len = max(
                [seq.size(0) for seq in return_data["left_padded_output_ids"]]
            )
            right_padded_output_ids = torch.full(
                (batch_size, max_seq_len),
                self.cfg["generation"]["pad_token_id"],
                dtype=return_data["left_padded_output_ids"][0].dtype,
                device=return_data["left_padded_output_ids"][0].device,
            )

            for idx, seq in enumerate(return_data["left_padded_output_ids"]):
                # Get only the generated part (excluding input)
                original_length = return_data["orig_input_lengths"][idx].item()
                seq_len = seq.size(0)

                # The generated content starts after the left-padded input
                generated_part = seq[-(seq_len - input_length) :]

                eos_positions = (generated_part == self.tokenizer.eos_token_id).nonzero(
                    as_tuple=True
                )[0]
                # TODO @sahilj: handle different stopping criteria
                # Calculate generation length
                if len(eos_positions) > 0:
                    gen_length = (
                        eos_positions[0].item() + 1
                    )  # +1 to include the EOS token
                else:
                    gen_length = len(generated_part)

                generation_lengths.append(gen_length)

                valid_length = original_length + gen_length
                unpadded_sequence_lengths.append(valid_length)

                # Extract the original input tokens from the left-padded sequence
                # For left-padded sequences, tokens are at the end of the input section
                valid_input_part = (
                    seq[input_length - original_length : input_length]
                    if original_length > 0
                    else torch.tensor([], device=seq.device, dtype=seq.dtype)
                )

                # Combine with generated part
                valid_generated_part = generated_part[:gen_length]
                valid_tokens = torch.cat([valid_input_part, valid_generated_part])

                # Place at the beginning of the right-padded sequence
                right_padded_output_ids[idx, :valid_length] = valid_tokens

            # Store the right-padded outputs
            return_data["output_ids"] = right_padded_output_ids

            # Align generation_logprobs with right-padded output format
            batch_size = len(return_data["generation_logprobs"])
            right_padded_logprobs = torch.zeros(
                (batch_size, max_seq_len),
                dtype=return_data["generation_logprobs"][0].dtype,
                device=return_data["generation_logprobs"][0].device,
            )

            for idx, logprob_seq in enumerate(return_data["generation_logprobs"]):
                original_length = return_data["orig_input_lengths"][idx].item()
                gen_length = generation_lengths[idx]

                # For right-padded format, we need:
                # 1. Zeros for the original input tokens (at the beginning)
                # 2. Actual logprobs for generated tokens (after the zeros)
                # 3. Zeros padding at the end (if needed)

                right_padded_seq = torch.zeros(
                    max_seq_len, dtype=logprob_seq.dtype, device=logprob_seq.device
                )
                right_padded_seq[original_length : original_length + gen_length] = (
                    logprob_seq[:gen_length]
                )
                right_padded_logprobs[idx] = right_padded_seq
                valid_length = original_length + gen_length

            # Remove the temporary data we added
            if "generation_logprobs" in return_data:
                del return_data["generation_logprobs"]
            if "orig_input_lengths" in return_data:
                del return_data["orig_input_lengths"]
            if "left_padded_output_ids" in return_data:
                del return_data["left_padded_output_ids"]

            # Ensure consistent data types and device placement
            return_data["output_ids"] = right_padded_output_ids
            return_data["logprobs"] = right_padded_logprobs
            return_data["generation_lengths"] = torch.tensor(
                generation_lengths, dtype=torch.long
            )
            return_data["unpadded_sequence_lengths"] = torch.tensor(
                unpadded_sequence_lengths, dtype=torch.long
            )

            # Move everything to CPU before returning
            return_data.to("cpu")

            return return_data

    def _add_noise_to_weights(self) -> None:
        """Add small Gaussian noise to the weights of the model. Note that this is used for testing purposes only."""
        # TODO @sahilj: do this without a summon (maybe FSDP2)
        noise_std = 0.01  # Standard deviation for the noise
        with torch.distributed.fsdp.FullyShardedDataParallel.summon_full_params(
            self.model, recurse=True
        ):
            for p in self.model.parameters():
                if p.requires_grad:
                    noise = torch.randn_like(p.data) * noise_std
                    p.data.add_(noise)  # Add noise in-place
        torch.cuda.synchronize()

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

    @torch.no_grad()
    def prepare_refit_info(self) -> Optional[dict[str, Any]]:
        if not isinstance(self.model, FullyShardedDataParallel):
            state_dict = self.model.state_dict()
        else:
            with FullyShardedDataParallel.state_dict_type(
                self.model,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            ):
                state_dict = self.model.state_dict()

        # Collect info for streaming multiple tensors
        self.refit_param_info = []
        for name, tensor in state_dict.items():
            # dtensor's numel will return complete tensor instead of only local tensor
            size_in_bytes = tensor.element_size() * tensor.numel()
            self.refit_param_info.append((name, size_in_bytes))

    @torch.no_grad()
    def prepare_weights_for_ipc(self) -> tuple[list[tuple[str, int]], float]:
        """Prepare the weights for IPC.

        This function:
        - Prepares the state_dict of the model.
        - Collects the info for streaming multiple tensors.

        Returns:
            list: The list of parameters sizes.
            float: The total available memory in bytes.
        """
        from nemo_rl.utils.nvml import get_free_memory_bytes

        # If the model is not FSDP, then we need to manually move it to the GPU
        # For an FSDP model, model.state_dict() will move the params to the GPU
        if not isinstance(self.model, FullyShardedDataParallel):
            self.model = self.manual_load_to_gpu(self.model)
            self._held_sharded_state_dict_reference = self.model.state_dict()
        else:
            # Get sharded state dict instead of full state dict for FSDP1
            with FullyShardedDataParallel.state_dict_type(
                self.model,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            ):
                self._held_sharded_state_dict_reference = self.model.state_dict()

        # Collect current available memory for refit
        ## Get current device index from torch
        device_idx = torch.cuda.current_device()
        ## Get device free memory using NVML
        total_available_bytes = get_free_memory_bytes(device_idx)
        ## Use 80% of the free memory for safety
        total_available_bytes *= 0.8

        return self.refit_param_info, total_available_bytes

    @torch.no_grad()
    def get_weights_ipc_handles(self, keys: list[str]) -> dict[str, Any]:
        from torch.distributed.tensor import DTensor
        from torch.multiprocessing.reductions import reduce_tensor

        assert self._held_sharded_state_dict_reference is not None, (
            "prepare_weights_for_ipc must be called before get_weights_ipc_handles"
        )

        # Clean up the held tensors to reduce peak memory
        if self._held_streamed_param_reference is not None:
            del self._held_streamed_param_reference
            self._held_streamed_param_reference = None

        converted_params = {}
        for key in keys:
            # Get full_tensor for dtensor (GPU > 1)
            tensor = self._held_sharded_state_dict_reference[key]
            if isinstance(tensor, DTensor):
                full_tensor = tensor.full_tensor()
            else:
                full_tensor = tensor
            # Convert parameters to the configured dtype
            converted_params[key] = full_tensor.to(self.dtype, non_blocking=True)

        # Temporary record the full tensor for cleanup
        # It is needed for cleanup the last full_tensor in the refit process
        self._held_streamed_param_reference = converted_params

        # Get device UUID for IPC
        device_uuid = self.report_device_id()
        # Create handles for the tensors
        all_handles = []
        for key, p in converted_params.items():
            handle = reduce_tensor(p.detach())
            all_handles.append((key, handle))

        return {device_uuid: all_handles}

    def prepare_for_lp_inference(self) -> None:
        self.model = self.manual_load_to_gpu(self.model)
        self.model.eval()
        self.offload_before_refit()

    def prepare_for_training(self, *args: Any, **kwargs: Any) -> None:
        # onload models and optimizer state to cuda
        self.model = self.manual_load_to_gpu(self.model)
        self.model.train()

        if not self.cfg["fsdp_offload_enabled"]:
            # Move optimizer state to CUDA if it exists
            if hasattr(self, "optimizer") and self.optimizer is not None:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v) and not v.is_cuda:
                            state[k] = v.to("cuda")

        torch.cuda.empty_cache()

    @torch.no_grad()
    def offload_before_refit(self) -> None:
        """Offload the optimizer and buffers to the CPU."""
        torch.randn(1).cuda()  # wake up torch allocator
        if not self.cfg["fsdp_offload_enabled"]:
            if hasattr(self, "optimizer") and self.optimizer is not None:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to("cpu")

        gc.collect()
        torch.cuda.empty_cache()

        # Print memory stats after offloading
        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory after optimizer offload: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )

    @torch.no_grad()
    def offload_after_refit(self) -> None:
        # Offload as much as possible on the CPU
        self.model = self.manual_offload_to_cpu(self.model)
        self.model.eval()
        torch.randn(1).cuda()  # wake up torch allocator
        self.offload_before_refit()  # rerun the old offload function

        # Clean up the held tensors
        if self._held_sharded_state_dict_reference is not None:
            del self._held_sharded_state_dict_reference
            self._held_sharded_state_dict_reference = None
        if self._held_streamed_param_reference is not None:
            del self._held_streamed_param_reference
            self._held_streamed_param_reference = None

        gc.collect()
        torch.cuda.empty_cache()

        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory after refit complete: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )

    def manual_offload_to_cpu(self, model: torch.nn.Module) -> torch.nn.Module:
        if self.cfg["fsdp_offload_enabled"]:
            return model

        for param in model.parameters():
            param.data = param.data.to("cpu", non_blocking=True)
            if hasattr(param, "_local_shard"):
                param._local_shard = param.data
            if param.grad is not None:
                param.grad = param.grad.to("cpu", non_blocking=True)
        for buffer in model.buffers():
            buffer.data = buffer.data.to("cpu", non_blocking=True)

        if hasattr(model, "_fsdp_wrapped_module"):
            wrapped_module = model._fsdp_wrapped_module
            assert isinstance(wrapped_module, torch.nn.Module), (
                f"wrapped_module is not a torch.nn.Module: instead, {type(wrapped_module)}"
            )
            self.manual_offload_to_cpu(wrapped_module)

        return model

    def manual_load_to_gpu(self, model: torch.nn.Module) -> torch.nn.Module:
        if self.cfg["fsdp_offload_enabled"]:
            return model

        for param in model.parameters():
            param.data = param.data.to("cuda", non_blocking=True)
            if hasattr(param, "_local_shard"):
                param._local_shard = param.data
            if param.grad is not None:
                param.grad = param.grad.to("cuda", non_blocking=True)
        for buffer in model.buffers():
            buffer.data = buffer.data.to("cuda", non_blocking=True)

        if hasattr(model, "_fsdp_wrapped_module"):
            wrapped_module = model._fsdp_wrapped_module
            assert isinstance(wrapped_module, torch.nn.Module), (
                f"wrapped_module is not a torch.nn.Module: instead, {type(wrapped_module)}"
            )
            self.manual_load_to_gpu(wrapped_module)

        return model

    def save_checkpoint(
        self,
        weights_path: str,
        optimizer_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
    ) -> None:
        """Save a checkpoint of the model.

        The checkpoint is saved in the following format:

        weights_path/
            __0_1.distcp
            __1_0.distcp
            ...
        optimizer_path/
            __0_0.distcp
            __1_0.distcp
            ...

        the optimizer states are saved only if `optimizer` and `optimizer_path` are provided.
        """
        save_checkpoint(
            model=self.model,
            weights_path=weights_path,
            optimizer=self.optimizer if optimizer_path else None,
            scheduler=self.scheduler if optimizer_path else None,
            optimizer_path=optimizer_path,
            tokenizer=self.tokenizer if tokenizer_path else None,
            tokenizer_path=tokenizer_path,
        )

    def load_checkpoint(
        self, weights_path: str, optimizer_path: Optional[str] = None
    ) -> None:
        """Load a checkpoint into the model."""
        load_checkpoint(
            model=self.model,
            weights_path=weights_path,
            optimizer=self.optimizer if optimizer_path else None,
            scheduler=self.scheduler if optimizer_path else None,
            optimizer_path=optimizer_path,
        )

    def shutdown(self) -> None:
        """Shutdown the policy."""
        pass

    def start_gpu_profiling(self) -> None:
        """Start GPU profiling."""
        torch.cuda.profiler.start()

    def stop_gpu_profiling(self) -> None:
        """Stop GPU profiling."""
        torch.cuda.profiler.stop()
