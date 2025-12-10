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
import math
import os
import warnings
from collections import defaultdict
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Any, Generator, Optional, cast

import nemo_automodel.components._peft.lora as _lora_mod
import ray
import torch
from accelerate import init_empty_weights
from nemo_automodel import (
    NeMoAutoModelForSequenceClassification,
)
from nemo_automodel.components._peft.lora import (
    PeftConfig,
    apply_lora_to_linear_modules,
)
from nemo_automodel.components.distributed.cp_utils import (
    create_context_parallel_ctx,
    get_train_context,
)
from nemo_automodel.components.distributed.grad_utils import (
    clip_grad_by_total_norm_,
    get_grad_norm,
)
from nemo_automodel.components.distributed.parallelizer import (
    fsdp2_strategy_parallelize,
)
from nemo_automodel.components.distributed.tensor_utils import (
    get_cpu_state_dict,
    to_local_if_dtensor,
)
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    set_model_state_dict,
)
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    OffloadPolicy,
)
from torch.distributed.tensor import DTensor, Shard
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
)
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM

from nemo_rl.algorithms.interfaces import LossFunction, LossType
from nemo_rl.algorithms.loss_functions import SequencePackingLossWrapper
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import (
    allgather_cp_sharded_tensor,
    distributed_vocab_topk,
    get_logprobs_from_vocab_parallel_logits,
)
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
    import_class_from_path,
    resolve_model_class,
)
from nemo_rl.models.policy.workers.base_policy_worker import AbstractPolicyWorker
from nemo_rl.utils.automodel_checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from nemo_rl.utils.checkpoint import CheckpointingConfig
from nemo_rl.utils.nsys import wrap_with_nvtx_name
from nemo_rl.utils.packed_tensor import packed_broadcast_producer


# TODO: @ruit remove this once the bump Automodel to 2d20e33a19d5e53a271b1403b507475e68ad14dc (https://github.com/NVIDIA-NeMo/RL/issues/1586)
def _patched_init_lora_weights(self, init_method: str):
    if init_method == "xavier":
        nn.init.xavier_normal_(self.lora_A.weight.data)
    else:
        nn.init.kaiming_uniform_(self.lora_A.weight.data, a=math.sqrt(5))
    self.lora_B.weight.data.zero_()


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
        self.tokenizer = tokenizer
        self.processor = processor
        self.is_vlm = processor is not None

        print(f"Initializing DTensorPolicyWorkerV2 with is_vlm={self.is_vlm}")

        self.is_generation_colocated = None
        if "generation" in config and config["generation"] is not None:
            self.is_generation_colocated = config["generation"]["colocated"]["enabled"]

        # Explicitly set NCCL_CUMEM_ENABLE to 1 to avoid the P2P initialization error for PyNCCLCommunicator.
        # See https://github.com/NVIDIA-NeMo/RL/issues/564 for more details.
        if not self.is_generation_colocated:
            os.environ["NCCL_CUMEM_ENABLE"] = "1"

        # Disable dynamo autotune_local_cache to avoid crash when there's already a cache
        # with different order of node_bundles
        configure_dynamo_cache()

        self.cfg = config
        # torch distributed init. Envars for rank, world_size, and master_addr and master_port are set from the ray remote call
        torch.distributed.init_process_group(backend="nccl")
        self.rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        model_name = self.cfg["model_name"]

        self.cpu_offload = self.cfg["dtensor_cfg"]["cpu_offload"]
        self.offload_optimizer_for_logprob = self.cfg["offload_optimizer_for_logprob"]
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
        self.enable_seq_packing = self.cfg["sequence_packing"]["enabled"]
        if self.enable_seq_packing:
            assert not self.is_vlm, (
                "Sequence packing is not supported for VLM models. Please set policy.sequence_packing.enabled = False to train VLM models."
            )
            print(
                f"[Rank {self.rank}] Sequence packing is enabled for model {model_name}"
            )
            print(f"[Rank {self.rank}] Using FlashAttention2 for sequence packing")

        hf_config_overrides = self.cfg.get("hf_config_overrides", {}) or {}

        model_config = AutoConfig.from_pretrained(
            model_name,
            # Always load the model in float32 to keep master weights in float32.
            # Keeping the master weights in lower precision has shown to cause issues with convergence.
            torch_dtype=torch.float32,
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
            if self.enable_seq_packing
            else None,
            **hf_config_overrides,
        )

        self.allow_flash_attn_args = self.check_model_allow_flash_attn_args(
            model_config
        )

        self._is_reward_model = (
            "reward_model_cfg" in self.cfg and self.cfg["reward_model_cfg"]["enabled"]
        )
        if self._is_reward_model:
            # Ensure sequence packing is disabled.
            if self.enable_seq_packing:
                raise NotImplementedError(
                    "Sequence packing is not supported for reward models"
                )
            # Load model as a Reward Model.
            rm_type = self.cfg["reward_model_cfg"]["reward_model_type"]
            if rm_type == "bradley_terry":
                model_class = NeMoAutoModelForSequenceClassification
                if model_config.num_labels != 1:
                    # For Bradley-Terry reward models, the linear head has a single output.
                    # In the transformers library, the default setting for model_config.num_labels is 2
                    # (https://github.com/huggingface/transformers/blob/v4.52.4/src/transformers/configuration_utils.py#L259).
                    # Since num_labels is used as the out_features for the linear head
                    # (https://github.com/huggingface/transformers/blob/v4.52.4/src/transformers/models/llama/modeling_llama.py#L738)
                    # if num_labels is not 1, we set it to 1. This change may trigger a warning that some weights are not initialized
                    # from the model checkpoint and are instead initialized using model_config.initializer_range
                    # (https://github.com/huggingface/transformers/blob/v4.52.4/src/transformers/models/llama/configuration_llama.py#L62).
                    print(
                        "model_config.num_labels is not 1. Setting it to 1 since this value is used as the out_features "
                        "for the linear head of Bradley-Terry reward models."
                    )
                    model_config.num_labels = 1
            else:
                raise ValueError(f"Unknown reward model type: {rm_type}")
        else:
            # DO NOT assume AutoModelForCausalLM, multimodal models can inherit from AutoModelForImageTextToText, AutoModelForTextToWaveform, etc.
            model_class = resolve_model_class(model_config.model_type)

        full_state_dict = None
        model_state_dict_keys = None

        # lora config
        lora_cfg = self.cfg["dtensor_cfg"].get("lora_cfg", None)
        self.peft_config = None
        self.lora_enabled = lora_cfg is not None and lora_cfg["enabled"]
        # patch the init_lora_weights method to use the xavier initialization
        _lora_mod.LinearLoRA.init_lora_weights = _patched_init_lora_weights
        if self.lora_enabled:
            if self.cfg["dtensor_cfg"]["tensor_parallel_size"] > 1:
                assert not lora_cfg["use_triton"], (
                    "Triton is not supported when tensor_parallel_size > 1"
                )
            # Always use float32 since FSDP requires all parameters to be in the same dtype.
            # autocast should cast the weights to the correct dtype during the forward pass.
            cfg_dict_with_dtype = {**lora_cfg, "lora_dtype": "torch.float32"}
            self.peft_config = PeftConfig.from_dict(cfg_dict_with_dtype)

        if self.rank == 0:
            print(f"[Rank {self.rank}] Loading model {model_name} on CPU...")
            model = model_class.from_pretrained(
                model_name,
                device_map="cpu",  # load weights onto CPU initially
                trust_remote_code=True,
                config=model_config,
                use_liger_kernel=False,
                torch_dtype=str(model_config.torch_dtype),
            )

            if self.lora_enabled:
                apply_lora_to_linear_modules(model, self.peft_config)

            full_state_dict = model.state_dict()
            # Store the original model state dict keys before any parallelization
            model_state_dict_keys = list(full_state_dict.keys())
            del model

        print(f"[Rank {self.rank}] Initializing empty model for FSDP...")
        # All ranks initialize model on meta device, so FSDP can shard it.
        # The actual weights will be broadcast from rank 0.

        with init_empty_weights():
            # NeMoAutoModelForCausalLM uses flash_attention_2 by default
            # so we need to set it to None if sequence packing is disabled
            # https://github.com/NVIDIA-NeMo/Automodel/blob/7e748be260651349307862426c0c168cebdeeec3/nemo_automodel/components/_transformers/auto_model.py#L180
            self.model = model_class.from_config(
                model_config,
                attn_implementation="flash_attention_2"
                if self.enable_seq_packing
                else None,
                use_liger_kernel=False,
                trust_remote_code=True,
                torch_dtype=str(model_config.torch_dtype),
            )
            if self.lora_enabled:
                apply_lora_to_linear_modules(self.model, self.peft_config)

        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = tokenizer.pad_token_id

        tp_size = self.cfg["dtensor_cfg"]["tensor_parallel_size"]
        cp_size = self.cfg["dtensor_cfg"]["context_parallel_size"]
        if cp_size > 1 and self.enable_seq_packing:
            raise ValueError(
                "Context parallel is not supported for sequence packing. Refer to https://github.com/NVIDIA/NeMo-RL/blob/main/docs/model-quirks.md#context-parallel-with-fsdp2 for more details."
            )
        dp_size = world_size // tp_size // cp_size
        sequence_parallel_enabled = self.cfg["dtensor_cfg"]["sequence_parallel"]
        assert world_size == dp_size * tp_size * cp_size, (
            f"World size({world_size}) must equal to dp_size({dp_size}) * tp_size({tp_size}) * cp_size({cp_size}) to use DTensor"
        )

        if sequence_parallel_enabled and tp_size == 1:
            print(
                "[WARNING]: sequence_parallel=True, but tp_size=1 which has no effect. Enable tp_size > 1 to use sequence parallelism."
            )
        elif sequence_parallel_enabled and tp_size > 1:
            raise RuntimeError(
                "Sequence parallel + tp_size >1 is currently broken in torch==2.8.0. See https://github.com/NVIDIA-NeMo/Automodel/issues/652 for more details."
            )

        if cp_size > 1:
            assert not isinstance(self.model, Gemma3ForCausalLM), (
                "Context parallel is not supported for Gemma3ForCausalLM. Torch context parallel has many limitations. "
                "Please refer to https://github.com/NVIDIA/NeMo-RL/blob/main/docs/model-quirks.md#context-parallel-with-fsdp2 for more details."
            )

            assert not (tp_size > 1 and sequence_parallel_enabled), (
                "It's a known issue that context parallel can't be used together with sequence parallel in DTensor worker. "
                "Please either set cp_size = 1 or disable sequence parallel. "
                "See https://github.com/NVIDIA-NeMo/RL/issues/659 for more details."
            )

            assert not self.is_vlm, (
                "Context parallel is yet not supported for VLM models. Please set cp_size = 1 to train VLM models."
            )

        # For FSDP2 compatibility, we need to support HSDP structure
        # For now, we use dp_replicate_size = 1 (no hybrid sharding)
        dp_replicate_size = 1
        dp_shard_size = dp_size

        # torch==2.8 uses LOCAL_RANK to set the device here (https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/torch/distributed/device_mesh.py#L500),
        # but CUDA_VISIBLE_DEVICES is set to only 1 gpu, so we need to temporarily set LOCAL_RANK to 0.
        # TODO: consider changing the default LOCAL_RANK set in worker_groups.py
        prev_local_rank = os.environ["LOCAL_RANK"]
        os.environ["LOCAL_RANK"] = "0"

        # Create device mesh with HSDP structure for FSDP2 compatibility
        device_mesh = torch.distributed.device_mesh.init_device_mesh(
            "cuda",
            (dp_replicate_size, dp_shard_size, cp_size, tp_size),
            mesh_dim_names=("dp_replicate", "dp_shard", "cp", "tp"),
        )
        os.environ["LOCAL_RANK"] = prev_local_rank

        # Create flattened submeshes for different use cases
        # Flatten dp_replicate + dp_shard for the "dp" dimension (backward compatibility)
        device_mesh[("dp_replicate", "dp_shard")]._flatten(mesh_dim_name="dp")

        # Flatten dp_shard + cp for FSDP2 sharding
        device_mesh[("dp_shard", "cp")]._flatten(mesh_dim_name="dp_shard_cp")

        # Flatten dp_replicate + dp_shard + cp for gradient operations
        device_mesh[("dp_replicate", "dp_shard", "cp")]._flatten(mesh_dim_name="dp_cp")

        # Store mesh references for backward compatibility
        self.dp_cp_mesh = device_mesh["dp_cp"]
        self.dp_mesh = device_mesh["dp"]
        self.tp_mesh = device_mesh["tp"]
        self.cp_mesh = device_mesh["cp"]

        self.dp_size = dp_size
        self.tp_size = tp_size
        self.cp_size = cp_size
        self.device_mesh = device_mesh

        # ------------------------------------------------
        # 3) Move to GPU + Composable FSDP
        #    (Initialize device mesh, shard submodules, then shard entire model)
        # ------------------------------------------------
        self.model = fsdp2_strategy_parallelize(
            self.model,
            device_mesh=self.device_mesh,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=self.dtype,
                reduce_dtype=torch.float32,
                output_dtype=torch.float32,
            ),
            offload_policy=CPUOffloadPolicy(pin_memory=False)
            if self.cpu_offload
            else OffloadPolicy(),
            sequence_parallel=sequence_parallel_enabled,
            activation_checkpointing=self.cfg["dtensor_cfg"][
                "activation_checkpointing"
            ],
            tp_shard_plan=self.cfg["dtensor_cfg"]["custom_parallel_plan"],
            dp_replicate_mesh_name="dp_replicate",
            dp_shard_cp_mesh_name="dp_shard_cp",
            tp_mesh_name="tp",
        )

        print(f"[Rank {self.rank}] Loading state dict from rank 0...")
        # This will broadcast the state dict from rank 0 to all other ranks
        # and load it into the FSDP model.
        set_model_state_dict(
            self.model,
            model_state_dict=full_state_dict,
            options=StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=True,
            ),
        )

        # Broadcast model state dict keys to all ranks and store as instance variable
        keys_to_broadcast = [model_state_dict_keys]
        torch.distributed.broadcast_object_list(keys_to_broadcast, src=0)
        self.model_state_dict_keys = keys_to_broadcast[0]

        # Handle tied word embeddings after loading the state dict
        # We need to actually tie the parameters at the model level
        is_tied_lm_head = hasattr(self.model, "lm_head") and getattr(
            getattr(self.model, "config", {}), "tie_word_embeddings", False
        )
        if is_tied_lm_head:
            embed_tokens_weight = None
            for name, param in self.model.named_parameters():
                if "embed_tokens" in name and name.endswith(".weight"):
                    embed_tokens_weight = param
                    break

            if embed_tokens_weight is not None:
                self.model.lm_head.weight = embed_tokens_weight

        # Manually broadcast buffers
        for _, buf in self.model.named_buffers():
            torch.distributed.broadcast(to_local_if_dtensor(buf), src=0)

        if self.cpu_offload:
            self.model = self.move_to_device(self.model, "cpu")

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
            host=ip, port=port, rank=self.rank, world_size=world_size
        )
        device = torch.cuda.current_device()
        self.model_update_group = PyNcclCommunicator(pg, device=device)

    def check_model_allow_flash_attn_args(self, model_config) -> bool:
        # Some models doesn't support flash_attn_kwargs
        # Check nemotron nas.
        if (
            model_config.architectures[0] == "DeciLMForCausalLM"
            and model_config.model_type == "nemotron-nas"
        ):
            return False

        return True

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
                global_batch = data.get_batch(batch_idx=gb_idx, batch_size=local_gbs)

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
                dummy_iterator = iter([])
                if self.cfg["dynamic_batching"]["enabled"]:
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
                        if not hasattr(outputs, "logits"):
                            logits = self.model.lm_head(outputs.last_hidden_state)
                        else:
                            logits = outputs.logits
                        del outputs

                        # Apply temperature scaling
                        logits = self._apply_temperature_scaling(logits)

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

                        if self.enable_seq_packing:
                            loss_fn_ = SequencePackingLossWrapper(
                                loss_fn=loss_fn,
                                cu_seqlens_q=flash_attn_kwargs.cu_seqlens_q,
                                cu_seqlens_q_padded=flash_attn_kwargs.cu_seqlens_q,
                            )
                        else:
                            loss_fn_ = loss_fn

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
                            )
                        grad_norm = torch.tensor([grad_norm])

                    # Update parameters
                    self.optimizer.step()

                losses.append(torch.tensor(mb_losses).sum().item())

            # release gradient memory before rollouts
            self.optimizer.zero_grad()
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
            if self.cfg["dynamic_batching"]["enabled"]:
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

                    logits = outputs.logits

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
            if self.cfg["dynamic_batching"]["enabled"]:
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
            if self.cfg["dynamic_batching"]["enabled"]:
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

                            vals, idx = distributed_vocab_topk(
                                local_logits,
                                k=k,
                                tp_group=tp_group,
                                vocab_start_index=vocab_start_index,
                                vocab_end_index=vocab_end_index,
                            )
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
            }
        }
        if self.lora_enabled:
            checkpoint_kwargs["is_peft"] = True
            checkpoint_kwargs["peft_config"] = self.peft_config

        save_checkpoint(
            model=self.model,
            weights_path=weights_path,
            optimizer=self.optimizer if optimizer_path else None,
            scheduler=self.scheduler if optimizer_path else None,
            optimizer_path=optimizer_path,
            tokenizer=self.tokenizer if tokenizer_path else None,
            tokenizer_path=tokenizer_path,
            model_state_dict_keys=self.model_state_dict_keys,
            **checkpoint_kwargs,
        )

    def load_checkpoint(
        self,
        weights_path: str,
        optimizer_path: Optional[str] = None,
    ) -> None:
        """Load a checkpoint into the model."""
        load_checkpoint(
            model=self.model,
            weights_path=weights_path,
            optimizer=self.optimizer if optimizer_path else None,
            scheduler=self.scheduler if optimizer_path else None,
            optimizer_path=optimizer_path,
        )
