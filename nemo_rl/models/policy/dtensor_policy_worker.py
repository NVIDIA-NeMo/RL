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

import contextlib
import gc
import os
from collections import defaultdict
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Any, Generator, Iterable, List, Optional, Set, Union, cast

import ray
import torch
from torch import nn
from torch.distributed.fsdp import (
    FSDPModule,
)
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor.experimental import context_parallel
from torch.distributed.tensor.experimental._attention import (
    set_rotate_method,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.integrations.accelerate import find_tied_parameters
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM

from nemo_rl.algorithms.interfaces import LossFunction, LossType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.worker_group_utils import get_nsight_config_if_pattern_matches
from nemo_rl.models.dtensor.parallelize import (
    _parallelize_model,
    clip_grad_by_total_norm_,
    get_grad_norm,
    get_logprobs_from_vocab_parallel_logits,
    to_local_if_dtensor,
)
from nemo_rl.models.huggingface.common import ModelFlag
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


@contextmanager
def unshard_fsdp2_model(model: nn.Module) -> Generator[None, None, None]:
    """Explicitly unshard and then reshard the FSDP2 modules. Useful for logprob inference."""
    try:
        for module in model.modules():
            if isinstance(module, FSDPModule):
                module.unshard()
        yield
    finally:
        for module in model.modules():
            if isinstance(module, FSDPModule):
                module.reshard()


@torch.no_grad()
def get_cpu_state_dict(
    state_generator: Iterable[tuple[str, Union[torch.Tensor, DTensor]]],
    pin_memory: bool = False,
) -> dict[str, torch.Tensor]:
    """Copy the state dict generator to CPU memory.

    Args:
        state_generator (Iterable[tuple[str, Union[torch.Tensor, DTensor]]]):
            An iterable that yields (key, tensor) pairs from a model state.
        pin_memory (bool, optional):
            Whether to allocate the CPU tensors in pinned memory for faster GPU transfer.
            Defaults to False.

    Returns:
        dict[str, torch.Tensor]: A dictionary mapping parameter names to CPU tensors.
    """
    new_state_dict = {}
    for k, v in state_generator:
        val = to_local_if_dtensor(v)

        if len(val.shape) == 0:
            new_state_dict[k] = val.cpu()
        else:
            cpu_tensor = torch.empty(
                *val.shape, device="cpu", pin_memory=pin_memory, dtype=val.dtype
            )
            cpu_tensor.copy_(val, non_blocking=True)
            new_state_dict[k] = cpu_tensor

    torch.cuda.synchronize()
    return new_state_dict


@ray.remote(
    runtime_env={
        # TODO: This option causes a crash on Ampere. It's okay to enable on Hopper.
        # "env_vars": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        **get_nsight_config_if_pattern_matches("dtensor_policy_worker"),
    }
)
class DTensorPolicyWorker:
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
        weights_path: Optional[str] = None,
        optimizer_path: Optional[str] = None,
        init_optimizer: bool = True,
        init_reference_model: bool = True,
        **kwargs: Any,
    ):
        self.cfg = config
        # torch distributed init. Envars for rank, world_size, and master_addr and master_port are set from the ray remote call
        torch.distributed.init_process_group(backend="nccl")
        self.rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        model_name = self.cfg["model_name"]

        self.cpu_offload = self.cfg["dtensor_cfg"]["cpu_offload"]
        self.max_grad_norm = self.cfg["max_grad_norm"]

        if self.cfg["precision"] == "float32":
            self.dtype = torch.float32
        elif self.cfg["precision"] == "bfloat16":
            self.dtype = torch.bfloat16
        elif self.cfg["precision"] == "float16":
            self.dtype = torch.float16
        else:
            raise ValueError(f"Unknown precision: {self.cfg['precision']}")

        print(f"[Rank {self.rank}] Loading model {model_name} on CPU...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cpu",  # load weights onto CPU initially
            # Always load the model in float32 to keep master weights in float32.
            # Keeping the master weights in lower precision has shown to cause issues with convergence.
            # https://github.com/NVIDIA/NeMo-RL/issues/279 will fix the issue of CPU OOM for larger models.
            torch_dtype=torch.float32,
            trust_remote_code=True,
            **sliding_window_overwrite(
                model_name
            ),  # due to https://github.com/huggingface/transformers/issues/38002
        )
        # caching since this property is not always preserved after FSDP
        self.num_tied_weights = len(find_tied_parameters(self.model))
        self.skip_tie_check = os.environ.get(
            "NRL_SKIP_TIED_WEIGHT_CHECK"
        ) or ModelFlag.SKIP_DTENSOR_TIED_WEIGHTS_CHECK.matches(model_name)

        self.tokenizer = tokenizer
        # ------------------------------------------------
        # 3) Move to GPU + Composable FSDP
        #    (Initialize device mesh, shard submodules, then shard entire model)
        # ------------------------------------------------

        tp_size = self.cfg["dtensor_cfg"]["tensor_parallel_size"]
        cp_size = self.cfg["dtensor_cfg"]["context_parallel_size"]
        dp_size = world_size // tp_size // cp_size
        assert world_size == dp_size * tp_size * cp_size, (
            f"World size({world_size}) must equal to dp_size({dp_size}) * tp_size({tp_size}) * cp_size({cp_size}) to use DTensor"
        )

        if cp_size > 1:
            assert not isinstance(self.model, Gemma3ForCausalLM), (
                "Context parallel is not supported for Gemma3ForCausalLM. Torch context parallel has many limitations. Please refer to https://github.com/NVIDIA/NeMo-RL/blob/main/docs/model-quirks.md#context-parallel-with-fsdp2 for more details."
            )

        device_mesh = torch.distributed.device_mesh.init_device_mesh(
            "cuda", (dp_size, cp_size, tp_size), mesh_dim_names=("dp", "cp", "tp")
        )

        self.dp_cp_mesh = device_mesh[("dp", "cp")]._flatten(mesh_dim_name="dp_cp")

        self.dp_mesh, self.tp_mesh, self.cp_mesh = (
            device_mesh["dp"],
            device_mesh["tp"],
            device_mesh["cp"],
        )
        self.dp_size = dp_size
        self.tp_size = tp_size
        self.cp_size = cp_size
        self.device_mesh = device_mesh

        self.model = _parallelize_model(
            self.model,
            self.dp_cp_mesh,
            self.tp_mesh,
            param_dtype=self.dtype,
            sequence_parallel=self.cfg["dtensor_cfg"]["sequence_parallel"],
            cpu_offload=self.cpu_offload,
            activation_checkpointing=self.cfg["dtensor_cfg"][
                "activation_checkpointing"
            ],
            custom_parallel_plan=self.cfg["dtensor_cfg"]["custom_parallel_plan"],
        )

        if self.cpu_offload:
            self.model = self.move_buffer_to_device(self.model, "cpu")

        # used for streaming update inference engine weights
        self._held_sharded_state_dict_reference: Optional[dict[str, torch.Tensor]] = (
            None
        )
        self._held_streamed_param_reference: Optional[dict[str, torch.Tensor]] = None

        if init_reference_model:
            self.reference_model_state_dict = get_cpu_state_dict(
                self.model.state_dict().items(), pin_memory=True
            )

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
            self.load_checkpoint(weights_path, optimizer_path)
        else:
            print(
                "No weights path provided. Starting from scratch (default policy init)"
            )

    # Refer to nemo impl. Below is original comment.
    # based on https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/utils.py#L113
    @staticmethod
    def create_context_parallel_ctx(
        cp_mesh: torch.distributed.device_mesh.DeviceMesh,
        cp_buffers: List[torch.Tensor],
        cp_seq_dims: List[int],
        cp_no_restore_buffers: Set[torch.Tensor],
        cp_rotate_method: Optional[str] = None,
    ):
        """Create a context parallel context.

        Args:
            cp_mesh (DeviceMesh): The device mesh for context parallel.
            cp_buffers (List[torch.Tensor]): The buffers for context parallel.
            cp_seq_dims (List[int]): The sequence dimensions for context parallel.
            cp_no_restore_buffers (Set[torch.Tensor]): The no restore buffers for context parallel.
            cp_rotate_method (str): The rotation method for context parallel, such as "allgather" or "addtoall".
        """
        if cp_rotate_method is not None:
            set_rotate_method(cp_rotate_method)

        return context_parallel(
            cp_mesh,
            buffers=cp_buffers,
            buffer_seq_dims=cp_seq_dims,
            no_restore_buffers=cp_no_restore_buffers,
        )

    # Refer to nemo impl. Below is original comment.
    # based on https://github.com/pytorch/torchtitan/blob/cddd7dc809f36fe0ed51cdaaea0671c084d75442/torchtitan/distributed/utils.py#L178
    @staticmethod
    @contextlib.contextmanager
    def train_context(cp_context: Optional[Generator[None, None, None]] = None):
        with contextlib.ExitStack() as stack:
            if cp_context is not None:
                from torch.nn.attention import SDPBackend, sdpa_kernel
                # TODO (xilunwu): support cuDNN backend

                stack.enter_context(
                    sdpa_kernel(
                        [
                            SDPBackend.FLASH_ATTENTION,
                            SDPBackend.EFFICIENT_ATTENTION,
                        ]
                    )
                )

                stack.enter_context(cp_context)

            yield

    def init_collective(self, ip: str, port: int, world_size: int) -> None:
        """Initialize the collective communication."""
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup

        # keep the same behavior as vllm
        # see https://github.com/vllm-project/vllm/blob/v0.8.5/vllm/env_override.py#L25
        if not os.path.exists("/dev/nvidia-caps-imex-channels"):
            os.environ["NCCL_CUMEM_ENABLE"] = "0"

        if self.rank == 0:
            pg = StatelessProcessGroup.create(
                host=ip, port=port, rank=0, world_size=world_size
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
        if (
            self.num_tied_weights != 0
            and self.cfg["dtensor_cfg"]["tensor_parallel_size"] > 1
            and not self.skip_tie_check
        ):
            raise ValueError(
                f"Using dtensor policy with tp size {self.cfg['dtensor_cfg']['tensor_parallel_size']} for model ({self.cfg['model_name']}) that has tied weights (num_tied_weights={self.num_tied_weights}) is not supported (https://github.com/NVIDIA/NeMo-RL/issues/227). Please use dtensor policy with tensor parallel == 1 instead."
            )
        if gbs is None:
            gbs = self.cfg["train_global_batch_size"]
        if mbs is None:
            mbs = self.cfg["train_micro_batch_size"]
        local_gbs = gbs // self.dp_size
        dataset_size = data["input_ids"].shape[0]
        num_global_batches = dataset_size // local_gbs

        # dim 1 is always assumed to be the sequence dim, sanity check this here
        sequence_dim = 1
        seq_dim_size = data.get("input_ids").shape[sequence_dim]
        for k, v in data.items():
            if torch.is_tensor(v) and len(v.shape) > 1:
                assert v.shape[sequence_dim] == seq_dim_size, (
                    f"Dim 1 must be the sequence dim, expected dim 1={seq_dim_size} but got shape {v.shape}"
                )

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
            for gb_idx, gb_start in enumerate(range(0, dataset_size, local_gbs)):
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
                torch.distributed.all_reduce(to_reduce, group=self.dp_mesh.get_group())
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
                batch = data.get_batch(batch_idx=gb_idx, batch_size=local_gbs)
                # Calculate number of microbatches to process
                # make_microbatch_iterator assumes that the batch size is a multiple of the microbatch size
                # so its safe to not check for the case where the last data slice is smaller than mbs
                if self.cfg["dynamic_batching"]["enabled"]:
                    mb_iterator = batch.make_microbatch_iterator_with_dynamic_shapes()
                else:
                    mb_iterator = batch.make_microbatch_iterator(mbs)

                for mb in mb_iterator:
                    input_ids = mb.get("input_ids").cuda()
                    input_lengths = mb.get("input_lengths")
                    batch_size, seq_len = input_ids.shape

                    attention_mask = torch.zeros(
                        (batch_size, seq_len), dtype=torch.long, device=input_ids.device
                    )
                    for i, length in enumerate(input_lengths):
                        # For right-padded sequence, set 1s at the beginning of the sequence
                        attention_mask[i, :length] = 1

                    with torch.autocast(device_type="cuda", dtype=self.dtype):
                        attention_mask_input_all_ones = torch.ones(
                            (batch_size, seq_len),
                            dtype=torch.long,
                            device=input_ids.device,
                        )
                        position_ids = torch.arange(
                            seq_len, device=input_ids.device
                        ).repeat(batch_size, 1)

                    context_parallel_ctx = None
                    if self.cp_size > 1:
                        seq_index = torch.arange(
                            seq_len, device=input_ids.device
                        ).repeat(1, 1)
                        cp_buffers = (
                            [input_ids, position_ids, seq_index]
                            if self.cp_size > 1
                            else []
                        )

                        # Create context parallel context
                        context_parallel_ctx = self.create_context_parallel_ctx(
                            cp_mesh=self.cp_mesh,
                            cp_buffers=cp_buffers,
                            cp_seq_dims=[sequence_dim] * len(cp_buffers),
                            cp_no_restore_buffers=set(cp_buffers),
                        )

                    with DTensorPolicyWorker.train_context(context_parallel_ctx):
                        with torch.autocast(device_type="cuda", dtype=self.dtype):
                            outputs = self.model(
                                input_ids=input_ids,
                                attention_mask=attention_mask_input_all_ones,
                                position_ids=position_ids,
                                use_cache=False,
                            )

                        # Get logprobs
                        if not hasattr(outputs, "logits"):
                            logits = self.model.lm_head(outputs.last_hidden_state)
                        else:
                            logits = outputs.logits

                        # Divide logits by temperature
                        if (
                            "generation" in self.cfg
                            and self.cfg["generation"] is not None
                        ):
                            logits.div_(self.cfg["generation"]["temperature"])

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
                            _, sorted_indices = torch.sort(seq_index_dtensor)

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
                                        ).full_tensor()[:, sorted_indices]
                                        break

                            if isinstance(logits, DTensor):
                                logits = logits.full_tensor()

                            logits_dtensor = DTensor.from_local(
                                logits,
                                device_mesh=self.cp_mesh,
                                placements=[Shard(sequence_dim)],
                            )
                            logits = logits_dtensor.full_tensor()[:, sorted_indices]

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
                            loss *= self.dp_size * self.cp_size
                            loss.backward()

                    if num_valid_samples > 0:
                        mb_losses.append(loss.item())
                        all_mb_metrics.append(loss_metrics)

                grad_norm: Optional[float | torch.Tensor] = None
                if not eval_mode:
                    with torch.no_grad():
                        grad_norm = get_grad_norm(
                            self.model.parameters(),
                            dp_cp_group=self.dp_cp_mesh.get_group(),
                            tp_group=self.tp_mesh.get_group(),
                            dtype=torch.float32,
                        )
                        if self.max_grad_norm is not None:
                            clip_grad_by_total_norm_(
                                self.model.parameters(),
                                max_grad_norm=self.max_grad_norm,
                                total_norm=grad_norm,
                                dtype=torch.float32,
                            )
                        grad_norm = torch.tensor([grad_norm])

                    # Update parameters
                    self.optimizer.step()

                losses.append(torch.tensor(mb_losses).sum().item())

            # increment scheduler after all batches in rollout are processed
            if not eval_mode:
                self.scheduler.step()
            # dynamic batch and sequence dims causes alot of fragmentation, so clear
            # the memory allocator before moving on
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
                "all_mb_metrics": dict(mb_metrics),
            }

            return metrics

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

        with unshard_fsdp2_model(self.model), torch.no_grad():
            data.to("cuda")
            if self.cfg["dynamic_batching"]["enabled"]:
                mb_iterator = data.make_microbatch_iterator_with_dynamic_shapes()
            else:
                mb_iterator = data.make_microbatch_iterator(logprob_batch_size)

            for lp_batch in mb_iterator:
                input_ids = lp_batch.get("input_ids").cuda()
                input_lengths = lp_batch.get("input_lengths")

                batch_size, seq_len = input_ids.shape
                # Create attention mask for right-padded data
                attention_mask = torch.zeros(
                    (batch_size, seq_len), dtype=torch.long, device=input_ids.device
                )
                for i, length in enumerate(input_lengths):
                    # For right-padded sequence, set 1s at the beginning of the sequence
                    attention_mask[i, :length] = 1

                # explicitly create position ids for the input, otherwise the sharding
                # for DTensor will be incorrect
                position_ids = torch.arange(seq_len, device=input_ids.device).repeat(
                    batch_size, 1
                )

                with torch.autocast(device_type="cuda", dtype=self.dtype):
                    # DTensor requires the casual attention kernel to hit,
                    # yet our attention mask above is not always all 1s
                    # this is fine because we mask with the actual attention mask
                    # later, but for input it has to be all 1s
                    attention_mask_input_all_ones = torch.ones(
                        (batch_size, seq_len), dtype=torch.long, device=input_ids.device
                    )

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask_input_all_ones,
                        position_ids=position_ids,
                        use_cache=False,
                    )

                if isinstance(outputs.logits, DTensor):
                    token_logprobs = get_logprobs_from_vocab_parallel_logits(
                        outputs.logits.to(torch.float32), input_ids
                    )
                else:
                    # Extract logprobs for each token in the sequence by gathering the logprob
                    # corresponding to the next token at each position
                    # Input shapes:
                    #   log_probs: [batch_size, sequence_length, vocab_size] - logits for each position
                    #   token_ids: [batch_size, sequence_length] - actual tokens
                    # Output shape: [batch_size, sequence_length] - logprob of each token given previous
                    # We get logprob of token[t+1] from logits[t], prepending 0 to maintain sequence length

                    log_probs = torch.nn.functional.log_softmax(
                        outputs.logits.to(torch.float32), dim=-1
                    )
                    next_tokens = input_ids[:, 1:]
                    log_probs = log_probs[:, :-1]
                    token_logprobs = log_probs.gather(
                        dim=-1, index=next_tokens.unsqueeze(-1)
                    ).squeeze(-1)

                token_logprobs = torch.cat(
                    [torch.zeros_like(token_logprobs[:, :1]), token_logprobs], dim=1
                )

                # Apply mask to zero out padding tokens logprobs
                token_logprobs = token_logprobs * attention_mask
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

        # Get state_dict
        self.model = self.move_to_cuda(self.model)
        self._held_sharded_state_dict_reference: dict[str, torch.Tensor] = (
            self.model.state_dict()
        )

        # Collect info for streaming multiple tensors
        state_dict_info = []
        for name, tensor in self._held_sharded_state_dict_reference.items():
            # dtensor's numel will return complete tensor instead of only local tensor
            size_in_bytes = tensor.element_size() * tensor.numel()
            state_dict_info.append((name, size_in_bytes))

        # Collect current available memory for refit
        ## Get current device index from torch
        device_idx = torch.cuda.current_device()
        ## Get device free memory using NVML
        total_available_bytes = get_free_memory_bytes(device_idx)
        ## Use 80% of the free memory for safety
        total_available_bytes *= 0.8

        return state_dict_info, total_available_bytes

    @torch.no_grad()
    def get_weights_ipc_handles(self, keys: Iterable[str]) -> dict[str, Any]:
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

    @torch.no_grad()
    def prepare_info_for_collective(self) -> dict[str, Any]:
        """Prepare the info for collective communication.

        Returns:
            dict: A dictionary containing the info for collective communication.
        """
        # Get state_dict
        self.model = self.move_to_cuda(self.model)
        state_dict = self.model.state_dict()

        # Collect info for collective communication
        state_dict_info = {}
        for name, tensor in state_dict.items():
            state_dict_info[name] = (tensor.shape, self.dtype)

        return state_dict_info

    @torch.no_grad()
    def broadcast_weights_for_collective(self) -> None:
        """Broadcast the weights for collective communication."""
        for _, tensor in self.model.state_dict().items():
            if isinstance(tensor, DTensor):
                tensor = tensor.full_tensor()
            if self.rank == 0:
                tensor = tensor.to(self.dtype, non_blocking=True)
                self.model_update_group.broadcast(tensor.data, src=0)

    def prepare_for_lp_inference(self) -> None:
        if not self.cpu_offload:
            self.move_to_cuda(self.model)
        else:
            self.model = self.move_buffer_to_device(self.model, "cuda")

        self.model.eval()
        self.offload_before_refit()

    def prepare_for_training(self, *args, **kwargs) -> None:
        # onload models and optimizer state to cuda
        if not self.cpu_offload:
            self.move_to_cuda(self.model)
        else:
            # when cpu offload is enabled, the buffers do not get moved
            # to cuda automatically, so we need to do that manually
            self.model = self.move_buffer_to_device(self.model, "cuda")

        # have to move buffers to cuda manually for cpu offload case
        self.move_buffer_to_device(self.model, "cuda")

        self.model.train()
        # Move optimizer state to CUDA if it exists
        if (
            hasattr(self, "optimizer")
            and self.optimizer is not None
            and not self.cpu_offload
        ):
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, (DTensor, torch.Tensor)):
                        state[k] = v.to("cuda")

        torch.cuda.empty_cache()

    @torch.no_grad()
    def offload_before_refit(self) -> None:
        """Offload the optimizer to the CPU."""
        torch.randn(1).cuda()  # wake up torch allocator
        if hasattr(self, "optimizer") and self.optimizer is not None:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, (DTensor, torch.Tensor)):
                        state[k] = v.to("cpu")

        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def offload_after_refit(self) -> None:
        # Offload as much as possible on the CPU
        self.model = self.move_to_cpu(self.model)
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

        # Print memory stats after offloading
        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory after optimizer offload: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )

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
    ) -> None:
        """Save a checkpoint of the model.

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

    def start_gpu_profiling(self) -> None:
        """Start GPU profiling."""
        torch.cuda.profiler.start()

    def stop_gpu_profiling(self) -> None:
        """Stop GPU profiling."""
        torch.cuda.profiler.stop()
