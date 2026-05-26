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
import warnings
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Generator, Optional

import ray
import torch
from nemo_automodel.components.distributed.cp_utils import (
    create_context_parallel_ctx,
)
from nemo_automodel.components.distributed.cp_utils import (
    get_train_context as get_train_context_automodel,
)
from nemo_automodel.components.training.utils import scale_grads_and_clip_grad_norm
from torch import nn
from transformers import (
    AutoTokenizer,
)

from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.automodel.data import (
    check_sequence_dim,
    get_microbatch_iterator,
    process_global_batch,
)
from nemo_rl.models.automodel.setup import (
    setup_distributed,
    setup_model_and_optimizer,
    validate_and_prepare_config,
)
from nemo_rl.models.automodel.train import (
    # NOTE(C5 Path A): LossPostProcessor / ScorePostProcessor /
    # automodel_forward_backward / forward_with_post_processing_fn were
    # used by the previous integrated-score-head implementation of train()
    # and get_values(). After Path A we drive the backbone+ValueHead loop
    # directly inside this worker, so only the metric aggregator remains.
    aggregate_training_statistics,
)
from nemo_rl.models.policy.workers.base_policy_worker import AbstractPolicyWorker
from nemo_rl.models.policy.workers.patches import apply_transformer_engine_patch
from nemo_rl.models.value.config import ValueConfig
from nemo_rl.models.value.interfaces import ValueOutputSpec
# NOTE(ppo-dtensor port): bg51717/ppo originally imported AutomodelCheckpointManager
# from nemo_rl.utils.automodel_checkpoint, but in this branch the module lives at
# nemo_rl.models.automodel.checkpoint. Re-pointed to match.
from nemo_rl.models.automodel.checkpoint import AutomodelCheckpointManager
from nemo_rl.utils.checkpoint import CheckpointingConfig
from nemo_rl.utils.nsys import wrap_with_nvtx_name


@contextlib.contextmanager
def get_train_context(
    cp_size: int,
    cp_mesh: Any,
    cp_buffers: list,
    sequence_dim: int,
    dtype: torch.dtype,
    autocast_enabled: bool = True,
) -> Generator[None, None, None]:
    """Create combined context manager for training with context parallel and autocast."""
    with contextlib.ExitStack() as stack:
        context_parallel_ctx = None
        if cp_size > 1:
            # Create context parallel context
            context_parallel_ctx = create_context_parallel_ctx(
                cp_mesh=cp_mesh,
                cp_buffers=cp_buffers,
                cp_seq_dims=[sequence_dim] * len(cp_buffers),
                cp_no_restore_buffers=set(cp_buffers),
            )

        stack.enter_context(
            get_train_context_automodel(False, False, context_parallel_ctx)()
        )
        if autocast_enabled:
            stack.enter_context(torch.autocast(device_type="cuda", dtype=dtype))
        yield


# ---------------------------------------------------------------------------
# C5 (PORT_NOTES): Path A — standalone fp32 ValueHead mirroring megatron.
#
# The HF AutoModelForTokenClassification path used by the previous version of
# this worker put `score = nn.Linear(hidden_size, 1)` INSIDE the FSDP-wrapped
# model, with bf16 params + bf16 AdamW states. At value LR 1e-5 this hit the
# bf16 representable-update floor (lr*m/sqrt(v) ≈ 1e-5 << bf16 ulp at
# magnitude ~1), so the value head essentially didn't learn, leaving the
# per-step value_loss 5-12x larger than the megatron A/B twin (see DTENSOR
# _PORT_NOTES.md C5 side-by-side table).
#
# This standalone fp32 ValueHead mirrors megatron_value_worker.py:84 verbatim
# (same fp32 autocast forward + nn.Linear(hidden_size, 1, bias=True) shape),
# so the two backends now share the value-head numerics. The head lives
# outside FSDP (each DP rank holds a full fp32 copy); per-step DP allreduce
# on its gradients keeps the copies in sync (see _allreduce_value_head_grads
# below). A separate torch.optim.AdamW with fp32 params + fp32 Adam moments
# is paired with the head (see DTensorValueWorkerV2._add_value_head_to_optimizer).
# ---------------------------------------------------------------------------


class ValueHead(nn.Module):
    """Standalone fp32 linear value head — port of megatron_value_worker.ValueHead.

    Maps hidden states `[B, S, hidden_size]` to scalar values `[B, S, 1]`.
    Forward runs entirely in fp32 (autocast + .float() on input) for parity
    with the megatron value head; weights/biases are held in fp32 so the
    paired torch.optim.AdamW keeps fp32 Adam moments.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1, bias=True)
        self.linear.to(dtype=torch.float32)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        with torch.autocast(
            device_type=hidden_states.device.type, dtype=torch.float32
        ):
            return self.linear(hidden_states.float())


def _resolve_hidden_size(model_config: Any) -> int:
    """Find hidden_size on an HF config object.

    Most LMs put it directly on the top-level config (Qwen2 / Llama / etc.);
    VLM configs expose it via `text_config`. Falls back to model.config (when
    callers pass the model itself) before erroring loudly so the caller can
    add a model-family-specific path explicitly rather than silently
    constructing the head with the wrong dim.
    """
    cfg = model_config
    if hasattr(cfg, "config"):
        # `cfg` is the model itself rather than its config
        cfg = cfg.config
    if hasattr(cfg, "hidden_size"):
        return int(cfg.hidden_size)
    if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
        return int(cfg.text_config.hidden_size)
    raise ValueError(
        f"Cannot determine hidden_size from model config of type {type(cfg)}; "
        "add an explicit path in _resolve_hidden_size() for this model family."
    )


def _unwrap_to_backbone(model: nn.Module) -> nn.Module:
    """Return the inner causal-LM / backbone module of an HF classification model.

    Path A bypasses the HF `score` head by calling the backbone directly so
    we can route the produced `last_hidden_state` through our standalone
    fp32 `ValueHead` (mirrors megatron's `_ValueOutputLayerBypass` trick).

    For Qwen2ForTokenClassification / Qwen2ForCausalLM the backbone is
    `model.model` (Qwen2Model). For PEFT-wrapped models the structure is
    `model.base_model.model.model` — we only handle the non-PEFT case for now
    since the value worker disables LoRA by default; raise a clear error if a
    caller turns LoRA on so the failure mode is loud rather than silent.
    """
    # PEFT / LoRA: not supported by Path A today (LoRA wraps the backbone
    # with adapter layers that change the forward signature). The value
    # recipe uses `lora_cfg.enabled=false`, so this is just a guard.
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        raise NotImplementedError(
            "Path A standalone ValueHead does not yet support PEFT/LoRA-wrapped "
            "value models. Disable lora_cfg.enabled on the value config or extend "
            "_unwrap_to_backbone() to handle the PEFT wrapper structure."
        )
    if hasattr(model, "model"):
        return model.model
    raise ValueError(
        f"Cannot locate backbone (.model attr) on {type(model).__name__}; "
        "extend _unwrap_to_backbone() for this model family."
    )


def get_runtime_env_for_value_worker(worker_type: str) -> dict:
    """Get runtime environment for value worker."""
    from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker

    # Reuse policy worker runtime env
    return get_runtime_env_for_policy_worker("dtensor_policy_worker_v2")


@ray.remote(
    runtime_env=get_runtime_env_for_value_worker("dtensor_value_worker_v2")
)  # pragma: no cover
class DTensorValueWorkerV2(AbstractPolicyWorker):
    def __repr__(self) -> str:
        """Customizes the actor's prefix in the Ray logs."""
        if torch.distributed.is_initialized():
            return f"{self.__class__.__qualname__}[rank={torch.distributed.get_rank()}]"
        else:
            return f"{self.__class__.__qualname__}"

    def __init__(
        self,
        config: ValueConfig,
        tokenizer: AutoTokenizer,
        weights_path: Optional[str] = None,
        optimizer_path: Optional[str] = None,
        init_optimizer: bool = True,
        **kwargs: Any,
    ):
        """Initialize the DTensorValueWorkerV2.

        Note: Value models don't need a reference model since they don't compute KL divergence.
        """
        # Apply patches
        apply_transformer_engine_patch()
        # NOTE(ppo-dtensor port): apply_torch_aten_alias_tensor_patch isn't exported
        # from policy.workers.patches in this branch; the corresponding policy
        # worker (dtensor_policy_worker_v2.py) also omits this call, so drop it.

        # Store configuration and tokenizer
        self.cfg = config
        self.tokenizer = tokenizer
        self.lora_enabled = (
            config["dtensor_cfg"].get("lora_cfg", {}).get("enabled", False)
        )

        # Ensure reward model config is set for value models
        if (
            "reward_model_cfg" not in config
            or not config["reward_model_cfg"]["enabled"]
        ):
            # Value models use the reward model architecture but predict values instead
            config["reward_model_cfg"] = {
                "enabled": True,
                "reward_model_type": "regression",  # Value is a regression task
            }

        print("Initializing DTensorValueWorkerV2")

        # Initialize checkpoint manager
        self.checkpoint_manager: Optional[AutomodelCheckpointManager] = None

        if "hf_config_overrides" not in config:
            config["hf_config_overrides"] = {}
        config["hf_config_overrides"]["num_labels"] = 1

        # Validate configuration and prepare runtime settings
        runtime_config = validate_and_prepare_config(
            config=config,
            processor=None,  # Value models don't use vision processors
            rank=0,  # Temporary, will be updated after distributed init
        )

        # Set up distributed environment
        distributed_manager = setup_distributed(
            config=config,
            runtime_config=runtime_config,
        )

        # Set instance attributes from distributed manager
        self.rank = torch.distributed.get_rank()
        self.device_mesh = distributed_manager.device_mesh
        self.dp_cp_mesh = self.device_mesh["dp_cp"]
        self.dp_mesh = self.device_mesh["dp"]
        self.tp_mesh = self.device_mesh["tp"]
        self.cp_mesh = self.device_mesh["cp"]
        self.moe_mesh = distributed_manager.moe_mesh
        self.dp_size = distributed_manager.dp_size
        self.tp_size = distributed_manager.tp_size
        self.cp_size = distributed_manager.cp_size

        # Initialize checkpoint manager
        self._init_checkpoint_manager(
            config_updates={
                "model_repo_id": config["model_name"],
                "dequantize_base_checkpoint": config.get(
                    "dequantize_base_checkpoint", False
                ),
                "is_peft": self.lora_enabled,
                "skip_task_head_prefixes_for_base_model": ["score."],
            },
        )

        # Set up model and optimizer
        # NOTE: bg51717/ppo branch named the local var `distributed_manager`
        # but the formal parameter on setup_model_and_optimizer
        # (RL/nemo_rl/models/automodel/setup.py:533) is `distributed_context`
        # (returns DistributedContext). Pre-fix this raised:
        #   TypeError: setup_model_and_optimizer() got an unexpected keyword
        #   argument 'distributed_manager'. Did you mean 'distributed_context'?
        # Sibling Nemo-RL-ppo (PR #2027) uses `distributed_context` for both
        # the local var and the kwarg. We only fix the kwarg here (keeping
        # the local var name minimally diffed from bg51717/ppo) so future
        # rebases don't fight us. Drop this comment block if bg51717 renames
        # the var or the formal parameter.
        model_and_optimizer_state = setup_model_and_optimizer(
            config=config,
            tokenizer=tokenizer,
            runtime_config=runtime_config,
            distributed_context=distributed_manager,
            checkpoint_manager=self.checkpoint_manager,
            is_vlm=False,  # Value models don't use vision
            init_optimizer=init_optimizer,
            weights_path=weights_path,
            optimizer_path=optimizer_path,
            # optimizer_module_filter=["score."],
        )

        # Set instance attributes from model and optimizer state
        # NOTE: bg51717/ppo branch unpacked an extra `self.model_state_dict_keys`
        # slot (between model and optimizer) but the ModelAndOptimizerState
        # NamedTuple (RL/nemo_rl/models/automodel/config.py:76-92) only has 10
        # fields and does not include model_state_dict_keys. Without this fix
        # `ValueError: not enough values to unpack (expected 11, got 10)` would
        # fire here. Nothing else in this file reads self.model_state_dict_keys
        # (see grep at fix time), so dropping the slot is safe. Sibling
        # Nemo-RL-ppo (PR #2027) matches this 10-slot unpacking.
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

        # Set instance attributes from runtime config
        # NOTE: bg51717/ppo branch skipped the `sampling_params` slot in the
        # unpacking, but RuntimeConfig (RL/nemo_rl/models/automodel/config.py:
        # 41-73) has 13 fields including sampling_params right before
        # is_reward_model. Without this fix `ValueError: too many values to
        # unpack (expected 12)` would fire here. bg51717 also stripped all
        # references to self.sampling_params from train()/inference paths in
        # this file (see the original kwargs `sampling_params=self.sampling_
        # params` that bg51717 removed from automodel_forward_backward and
        # forward_with_post_processing_fn calls), so we discard the value
        # rather than assigning self.sampling_params. Sibling Nemo-RL-ppo
        # (PR #2027) keeps self.sampling_params and the kwarg pass-throughs.
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
            _runtime_sampling_params,  # bg51717 strips usage, see NOTE above
            _runtime_is_reward_model,
        ) = runtime_config

        # ------------------------------------------------------------------
        # C5 (PORT_NOTES Path A): install standalone fp32 ValueHead + paired
        # AdamW. The HF model loaded above still has a built-in `score` Linear
        # (because we set num_labels=1); we freeze it so it doesn't waste FSDP
        # gradient bandwidth or interfere with the optimizer's param coverage.
        # The standalone head + dedicated AdamW restore parity with
        # megatron_value_worker.py's two-optimizer / fp32-head architecture
        # (see megatron_value_worker.py:84-109 ValueHead and :504-523
        # _add_value_head_to_optimizer). This is the central fix for the
        # value_loss 5-12x gap diagnosed in DTENSOR_PORT_NOTES.md C5.
        # ------------------------------------------------------------------
        if self.cp_size > 1:
            raise NotImplementedError(
                "DTensorValueWorkerV2 Path A (standalone ValueHead) does not yet "
                "support context_parallel_size > 1. Add CP-aware allgather on "
                "value head outputs (mirror megatron_value_worker.py:887-890) "
                "before enabling CP on the value side."
            )

        hidden_size = _resolve_hidden_size(self.model)
        self.value_head = ValueHead(hidden_size).cuda()

        # Freeze the HF score head so it doesn't enter optimizer updates or
        # FSDP grad reductions. We don't delete it because automodel's
        # checkpoint manager may still serialize it (and because we keep
        # automodel_forward_backward callable for other backends to use).
        if hasattr(self.model, "score"):
            for p in self.model.score.parameters():
                p.requires_grad_(False)

        # Standalone AdamW for value head — params are fp32 so Adam moments
        # are also fp32 (avoids the bf16 ulp-rounding update floor that the
        # main FSDP optimizer hits at lr=1e-5).
        self.value_head_optimizer: Optional[torch.optim.AdamW] = None
        if init_optimizer:
            self._add_value_head_to_optimizer(config)

        # Restore value head weights from training checkpoint if present
        # (mirror megatron_value_worker.py:397-405). HF base-model weights
        # are loaded by setup_model_and_optimizer above; we never load from
        # the HF score head because (a) Qwen2-1.5B-Instruct doesn't ship one
        # and (b) our recipes have load_value_head_from_model=false.
        self._maybe_load_value_head_weights(weights_path)

    def _add_value_head_to_optimizer(self, config: dict) -> None:
        """Create a standalone fp32 AdamW for the value head.

        Mirrors megatron_value_worker.py:504-523 in intent. Reads the LR /
        weight_decay / betas / eps from `value.optimizer.kwargs` so the value
        head shares the same nominal optimizer hyperparams as the rest of
        the value side (currently `value.optimizer.kwargs.lr=1.0e-5`, same as
        the megatron twin's `value.megatron_cfg.optimizer.lr=1.0e-5`).
        """
        opt_cfg = config.get("optimizer") or {}
        opt_kwargs = opt_cfg.get("kwargs") or {}
        lr = opt_kwargs.get("lr", 1.0e-5)
        weight_decay = opt_kwargs.get("weight_decay", 0.0)
        betas = tuple(opt_kwargs.get("betas", (0.9, 0.999)))
        eps = opt_kwargs.get("eps", 1.0e-8)

        self.value_head_optimizer = torch.optim.AdamW(
            self.value_head.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )

    def _maybe_load_value_head_weights(self, weights_path: Optional[str]) -> None:
        """Load `value_head.pt` from a training checkpoint dir if it exists.

        Path mirrors megatron_value_worker.py:397-405. The file is written by
        save_checkpoint below on rank 0; absent means this is a fresh run
        and the value head keeps its default nn.Linear init.
        """
        if weights_path is None:
            return
        value_head_path = os.path.join(weights_path, "value_head.pt")
        if not os.path.exists(value_head_path):
            return
        state = torch.load(value_head_path, map_location="cuda", weights_only=True)
        self.value_head.load_state_dict(state)
        if self.rank == 0:
            print(f"[DTensorValueWorkerV2] Loaded value head from {value_head_path}")

    def _backbone_forward(
        self,
        processed_inputs: Any,
    ) -> torch.Tensor:
        """Run the FSDP-wrapped top-level forward and capture last_hidden_state.

        Mirrors megatron_value_worker.py's `_ValueOutputLayerBypass` trick in
        intent: capture hidden states before the task head and route them
        through our standalone ValueHead instead.

        Why we MUST go through the wrapped top-level forward (not call the
        backbone directly): when nemo_automodel wraps the model with FSDP2,
        the wrapper's forward is responsible for casting plain torch.Tensor
        inputs (input_ids etc.) to the DTensor layout that the sharded
        embedding weight expects. Calling `self.model.model(...)` directly
        bypasses that cast and fails inside `F.embedding` with:

            RuntimeError: aten.embedding.default got mixed torch.Tensor and
            DTensor, need to convert all torch.Tensor to DTensor before
            calling distributed operators!

        (Job 12110047 hit exactly this in `get_values` after Path A landed;
        see DTENSOR_PORT_NOTES.md C5b.)

        So we register a forward hook on the inner Qwen2Model that captures
        `last_hidden_state` from the BaseModelOutputWithPast it returns. The
        outer Qwen2ForTokenClassification still runs `self.score(hidden)`
        downstream of the captured tensor — but score is `requires_grad_
        (False)` (frozen in __init__), so its bf16 Linear forward is just a
        ~negligible wasted FLOP and its output never enters the backward
        pass we drive ourselves.
        """
        backbone = _unwrap_to_backbone(self.model)
        captured: dict[str, torch.Tensor] = {}

        def _capture_hook(_module, _args, output):
            # Qwen2Model returns BaseModelOutputWithPast; some HF versions
            # expose last_hidden_state directly, others (older) only via
            # output[0]. Cover both. The captured tensor stays in-graph for
            # our subsequent value_head + loss.backward().
            if hasattr(output, "last_hidden_state"):
                captured["hidden_states"] = output.last_hidden_state
            else:
                captured["hidden_states"] = output[0]

        handle = backbone.register_forward_hook(_capture_hook)
        try:
            # Call the FSDP-wrapped top-level forward — this is the one that
            # properly converts input_ids to a DTensor matching the sharded
            # embed_tokens.weight. We discard outputs.logits (it's the
            # frozen score head's output); only outputs of our captured
            # backbone hidden_states feed the value_head + loss path.
            _outputs = self.model(
                input_ids=processed_inputs.input_ids,
                attention_mask=processed_inputs.attention_mask,
                position_ids=processed_inputs.position_ids,
                use_cache=False,
            )
            # Help the GC drop the frozen score head's autograd graph early
            # (it depends on hidden_states but we never backprop through it).
            del _outputs
        finally:
            handle.remove()

        if "hidden_states" not in captured:
            raise RuntimeError(
                "Backbone forward hook did not capture last_hidden_state. "
                "The model returned an output type without `.last_hidden_state` "
                "or `[0]` access — extend _capture_hook in _backbone_forward()."
            )
        return captured["hidden_states"]  # [B, S, H]

    @staticmethod
    def _right_shift_values(values: torch.Tensor) -> torch.Tensor:
        """Shift values right by 1 along the sequence dim.

        After the shift `values[t] = V(state BEFORE token t)`, with
        `values[0] = 0`. This is the alignment convention megatron's
        value worker uses (see megatron_value_worker.py:210-213 train
        path and :892-895 inference path) so that training targets
        (returns, old_values) computed off inference values are aligned
        with the loss-time values. Path A adopts the same convention so
        the GAE math in the PPO loop is consistent across backends.
        """
        return torch.cat([torch.zeros_like(values[:, :1]), values[:, :-1]], dim=1)

    def _allreduce_value_head_grads(self) -> None:
        """Average value head gradients across the DP group.

        The standalone ValueHead lives outside FSDP, so each DP rank holds a
        full fp32 copy that accumulates a per-rank gradient from its local
        microbatches. Without this allreduce the per-rank copies would drift
        out of sync after the optimizer step. Mirrors
        megatron_value_worker.py:678-685.
        """
        dp_group = self.dp_mesh.get_group()
        for p in self.value_head.parameters():
            if p.grad is not None:
                torch.distributed.all_reduce(
                    p.grad,
                    op=torch.distributed.ReduceOp.AVG,
                    group=dp_group,
                )

    @wrap_with_nvtx_name("dtensor_value_worker_v2/train")
    def train(
        self,
        data: BatchedDataDict[Any],
        loss_fn: LossFunction,
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
    ) -> dict[str, Any]:
        """Train the value function on a batch of data with a given loss function.

        Path A (C5): bypasses `automodel_forward_backward` so we can route the
        backbone's `last_hidden_state` through our standalone fp32 `ValueHead`
        (instead of HF's bf16 `score` head), apply the megatron-style right-
        shift by 1 token (V[t] = V(state before token t)), and step the
        backbone + value_head with their dedicated optimizers. See
        megatron_value_worker.py:533-792 for the reference implementation
        we mirror.
        """
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

        # Validate sequence dimension
        sequence_dim, _ = check_sequence_dim(data)

        if eval_mode:
            ctx: AbstractContextManager[Any] = torch.no_grad()
            self.model.eval()
            self.value_head.eval()
        else:
            ctx = nullcontext()
            self.model.train()
            self.value_head.train()

        # Create train context factory (autocast + optional CP context).
        # Same factory the previous automodel-driven train() used; we still
        # need it so the backbone bf16 forward is wrapped in the right
        # autocast context.
        def train_context_fn(processed_inputs):
            return get_train_context(
                cp_size=self.cp_size,
                cp_mesh=self.cp_mesh,
                cp_buffers=processed_inputs.cp_buffers,
                sequence_dim=sequence_dim,
                dtype=self.dtype,
                autocast_enabled=self.autocast_enabled,
            )

        empty_cache_steps = self.cfg.get("dtensor_cfg", {}).get(
            "clear_cache_every_n_steps"
        )
        if empty_cache_steps:
            warnings.warn(
                f"Emptying cache every {empty_cache_steps} microbatches; doing so unnecessarily would incur a large performance overhead.",
            )

        def on_microbatch_start(mb_idx):
            if empty_cache_steps and mb_idx % empty_cache_steps == 0:
                torch.cuda.empty_cache()

        grad_norm: Optional[float | torch.Tensor] = None

        with ctx:
            data = data.to("cuda")

            losses = []
            all_mb_metrics = []
            for gb_idx in range(num_global_batches):
                gb_result = process_global_batch(
                    data,
                    loss_fn,
                    self.dp_mesh.get_group(),
                    batch_idx=gb_idx,
                    batch_size=local_gbs,
                )
                batch = gb_result["batch"]
                global_valid_seqs = gb_result["global_valid_seqs"]
                global_valid_toks = gb_result["global_valid_toks"]

                # Zero both optimizers' grads (mirror megatron_value_worker.py:622-624)
                self.optimizer.zero_grad()
                if self.value_head_optimizer is not None:
                    self.value_head_optimizer.zero_grad()

                processed_iterator, iterator_len = get_microbatch_iterator(
                    batch,
                    self.cfg,
                    mbs,
                    self.dp_mesh,
                    tokenizer=self.tokenizer,
                    cp_size=self.cp_size,
                )

                # ---- C5 Path A custom microbatch loop (replaces
                # ----    automodel_forward_backward). Per microbatch:
                # ----      1. backbone forward → last_hidden_state (bf16)
                # ----      2. standalone fp32 ValueHead → values (fp32)
                # ----      3. right-shift values by 1 for V(state before t)
                # ----      4. loss_fn(values, ...) → loss
                # ----      5. (loss * dp_size * cp_size).backward() to cancel
                # ----         FSDP's implicit average and recover sum-of-mbs
                # ----         semantics, identical to automodel_forward_backward
                # ----         line 482-483.
                mb_losses = []
                for mb_idx, processed_mb in enumerate(processed_iterator):
                    on_microbatch_start(mb_idx)
                    processed_inputs = processed_mb.processed_inputs

                    with train_context_fn(processed_inputs):
                        hidden_states = self._backbone_forward(processed_inputs)
                        # ValueHead does fp32 autocast internally, output is fp32
                        values = self.value_head(hidden_states).squeeze(-1)  # [B, S]
                        values = self._right_shift_values(values)

                        # MseValueLossFn (input_type=LossInputType.LOGIT) accepts
                        # the values tensor as the `logits` kwarg; data carries
                        # returns / values / token_mask / sample_mask.
                        loss, loss_metrics = loss_fn(
                            logits=values,
                            data=processed_mb.data_dict,
                            global_valid_seqs=global_valid_seqs,
                            global_valid_toks=global_valid_toks,
                        )

                        is_dummy = mb_idx >= iterator_len
                        if not is_dummy:
                            for k in list(loss_metrics.keys()):
                                if "_min" in k or "_max" in k:
                                    continue
                                loss_metrics[k] = loss_metrics[k] / num_global_batches
                        else:
                            loss = loss * 0

                        if not eval_mode:
                            # Match automodel_forward_backward line 482-483: FSDP
                            # averages grads over DP dim but loss_fn already
                            # normalizes by global tokens, so we multiply back
                            # by dp_size * cp_size to get a sum-of-mbs grad.
                            scaled_loss = loss * self.dp_size * self.cp_size
                            scaled_loss.backward()

                    if not is_dummy:
                        num_valid_samples = loss_metrics.get("num_valid_samples", 0)
                        loss_metrics["lr"] = self.optimizer.param_groups[0]["lr"]
                        loss_metrics["global_valid_seqs"] = global_valid_seqs.item()
                        loss_metrics["global_valid_toks"] = global_valid_toks.item()
                        if num_valid_samples > 0:
                            mb_losses.append(loss.item())
                            all_mb_metrics.append(loss_metrics)

                if not eval_mode:
                    # 1. Sync the standalone value head's gradients across DP
                    #    (mirrors megatron_value_worker.py:678-685) — the
                    #    head isn't in FSDP so each rank holds an
                    #    independent fp32 copy that needs allreduce.
                    self._allreduce_value_head_grads()
                    # 2. Clip value head grads (mirrors megatron_value_worker
                    #    .py:686-691).
                    if self.max_grad_norm is not None and self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.value_head.parameters(), self.max_grad_norm
                        )
                    # 3. Backbone grad norm + clip via the FSDP-aware utility
                    #    (same call the previous train() used; this still
                    #    handles the DTensor-sharded backbone params).
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
                    # 4. Step both optimizers (mirrors megatron_value_worker
                    #    .py:654-692).
                    self.optimizer.step()
                    if self.value_head_optimizer is not None:
                        self.value_head_optimizer.step()

                losses.append(torch.tensor(mb_losses).sum().item())

            # Release gradient memory on both optimizers
            self.optimizer.zero_grad()
            if self.value_head_optimizer is not None:
                self.value_head_optimizer.zero_grad()
            # Increment scheduler (only the backbone scheduler; value head
            # uses a flat LR same as megatron, which doesn't wrap the value
            # head optimizer in a scheduler either).
            if not eval_mode:
                self.scheduler.step()
            torch.cuda.empty_cache()

            metrics = aggregate_training_statistics(
                losses=losses,
                all_mb_metrics=all_mb_metrics,
                grad_norm=grad_norm,
                dp_group=self.dp_mesh.get_group(),
                dtype=self.dtype,
            )

            return metrics

    @wrap_with_nvtx_name("dtensor_value_worker_v2/get_values")
    def get_values(
        self, data: BatchedDataDict[Any], micro_batch_size: Optional[int] = None
    ) -> BatchedDataDict[ValueOutputSpec]:
        """Get per-token value predictions for a batch of data.

        Path A (C5): mirrors the train-time forward path — backbone produces
        last_hidden_state, our standalone fp32 ValueHead produces values,
        and we right-shift by 1 so values[t] = V(state before token t). This
        matches the megatron value worker's inference convention (see
        megatron_value_worker.py:865-895) so the GAE / returns computed
        downstream in the PPO loop see the same value semantics regardless
        of backend.
        """
        value_batch_size = (
            micro_batch_size
            if micro_batch_size is not None
            else self.cfg.get("logprob_batch_size", self.cfg["train_micro_batch_size"])
        )

        # Validate sequence dimension
        sequence_dim, seq_dim_size = check_sequence_dim(data)

        all_values = []
        self.model.eval()
        self.value_head.eval()

        with torch.no_grad():
            data.to("cuda")
            processed_iterator, iterator_len = get_microbatch_iterator(
                data,
                self.cfg,
                value_batch_size,
                self.dp_mesh,
                tokenizer=self.tokenizer,
                cp_size=self.cp_size,
            )

            for batch_idx, processed_mb in enumerate(processed_iterator):
                processed_inputs = processed_mb.processed_inputs

                with get_train_context(
                    cp_size=self.cp_size,
                    cp_mesh=self.cp_mesh,
                    cp_buffers=processed_inputs.cp_buffers,
                    sequence_dim=sequence_dim,
                    dtype=self.dtype,
                    autocast_enabled=self.autocast_enabled,
                ):
                    hidden_states = self._backbone_forward(processed_inputs)
                    values = self.value_head(hidden_states).squeeze(-1)  # [B, S]
                    values = self._right_shift_values(values)

                # Skip dummy batches
                if batch_idx >= iterator_len:
                    continue

                all_values.append(values)

        return_data = BatchedDataDict[ValueOutputSpec]()

        all_values_padded = []
        for val in all_values:
            padding_needed = seq_dim_size - val.shape[1]
            if padding_needed > 0:
                val = torch.nn.functional.pad(
                    val, (0, padding_needed), mode="constant", value=0.0
                )
            all_values_padded.append(val)
        return_data["values"] = torch.cat(all_values_padded, dim=0).cpu()

        return return_data

    @wrap_with_nvtx_name("dtensor_value_worker_v2/prepare_for_training")
    def prepare_for_training(self, *args, **kwargs) -> None:
        """Prepare for training by loading model and optimizer to GPU."""
        if not self.cpu_offload:
            self.move_to_cuda(self.model)
        else:
            self.model = self.move_buffer_to_device(self.model, "cuda")

        self.model.train()
        # C5 Path A: also ensure the standalone fp32 ValueHead is on CUDA
        # and in train mode (mirror megatron_value_worker.py:942).
        self.value_head.cuda().train()

        if self.optimizer is not None and not self.cpu_offload:
            self.move_optimizer_to_device("cuda")

        torch.cuda.empty_cache()

    def finish_training(self, *args, **kwargs) -> None:
        pass

    def prepare_for_inference(self, *args, **kwargs) -> None:
        """Prepare for inference by setting model to eval mode."""
        self.model.eval()
        # C5 Path A: value head also needs eval (mirror
        # megatron_value_worker.py:958).
        self.value_head.eval()
        torch.cuda.empty_cache()

    def finish_inference(self, *args, **kwargs) -> None:
        pass

    def move_optimizer_to_device(self, device: str | torch.device) -> None:
        """Move optimizer state to specified device.

        Handles both the FSDP-wrapped backbone optimizer and the standalone
        ValueHead optimizer (mirror megatron_value_worker.py:1015-1037).
        """
        from torch.distributed.tensor import DTensor

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, (DTensor, torch.Tensor)):
                    state[k] = v.to(device)

        # C5 Path A: also move the standalone value head optimizer
        if self.value_head_optimizer is not None:
            for state in self.value_head_optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

    def move_to_device(self, model: nn.Module, device: str | torch.device) -> nn.Module:
        """Move model to specified device."""
        model = self.move_buffer_to_device(model, device)
        return model.to(device)

    def move_buffer_to_device(
        self, model: nn.Module, device: str | torch.device
    ) -> nn.Module:
        """Move model buffers to specified device."""
        for v in model.buffers():
            torch.utils.swap_tensors(v, v.to(device))
        return model

    def move_to_cuda(self, model: torch.nn.Module) -> torch.nn.Module:
        """Move model to CUDA."""
        model = self.move_to_device(model, "cuda")
        gc.collect()
        torch.cuda.empty_cache()
        return model

    def move_to_cpu(self, model: torch.nn.Module) -> torch.nn.Module:
        """Move model to CPU."""
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
        """Save a checkpoint of the value model.

        Saves the backbone via the standard AutomodelCheckpointManager and,
        as a side car, the standalone fp32 ValueHead's state_dict to
        `<weights_path>/value_head.pt` (mirror megatron_value_worker.py:
        1096-1108). Only rank 0 writes the value head file to avoid
        concurrent-write conflicts; all ranks see the same head state
        after the per-step DP allreduce so the rank-0 copy is canonical.
        """
        self.checkpoint_manager.save_checkpoint(
            model=self.model,
            weights_path=weights_path,
            optimizer=self.optimizer,
            optimizer_path=optimizer_path,
            scheduler=self.scheduler,
            tokenizer=self.tokenizer if tokenizer_path else None,
            tokenizer_path=tokenizer_path,
            checkpointing_cfg=checkpointing_cfg,
            lora_enabled=self.lora_enabled,
            peft_config=self.peft_config,
        )

        # C5 Path A: save standalone value head + (optionally) its optimizer
        if self.rank == 0:
            os.makedirs(weights_path, exist_ok=True)
            value_head_path = os.path.join(weights_path, "value_head.pt")
            torch.save(self.value_head.state_dict(), value_head_path)
            if optimizer_path is not None and self.value_head_optimizer is not None:
                os.makedirs(optimizer_path, exist_ok=True)
                vh_opt_path = os.path.join(optimizer_path, "value_head_optimizer.pt")
                torch.save(self.value_head_optimizer.state_dict(), vh_opt_path)
        # Barrier so non-rank-0 processes don't race ahead and try to read
        # a half-written value_head.pt on subsequent loads.
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def load_checkpoint(
        self,
        weights_path: str,
        optimizer_path: Optional[str] = None,
    ) -> None:
        """Load a checkpoint into the value model.

        Loads the backbone via the standard AutomodelCheckpointManager, then
        (if present) loads the standalone fp32 ValueHead state from
        `<weights_path>/value_head.pt` and the head optimizer state from
        `<optimizer_path>/value_head_optimizer.pt`. Both side cars are
        optional — fresh-init runs won't have them and we just keep the
        default nn.Linear init / fresh AdamW state.
        """
        self.checkpoint_manager.load_checkpoint(
            model=self.model,
            weights_path=weights_path,
            optimizer=self.optimizer,
            optimizer_path=optimizer_path,
            scheduler=self.scheduler,
        )

        # C5 Path A: try to restore value head + its optimizer from side cars
        self._maybe_load_value_head_weights(weights_path)
        if optimizer_path is not None and self.value_head_optimizer is not None:
            vh_opt_path = os.path.join(optimizer_path, "value_head_optimizer.pt")
            if os.path.exists(vh_opt_path):
                state = torch.load(vh_opt_path, map_location="cuda", weights_only=True)
                self.value_head_optimizer.load_state_dict(state)
                if self.rank == 0:
                    print(
                        f"[DTensorValueWorkerV2] Loaded value head optimizer "
                        f"state from {vh_opt_path}"
                    )

    def _init_checkpoint_manager(
        self,
        config_updates: Optional[dict[str, Any]] = None,
        checkpoint_root: Optional[str] = None,
    ) -> None:
        """Initialize the AutomodelCheckpointManager for this worker."""
        if self.checkpoint_manager is None:
            # NOTE: bg51717/ppo + ppo-dtensor branches originally passed
            # `model_state_dict_keys=getattr(self, "model_state_dict_keys",
            # None)` here, but (a) `self.model_state_dict_keys` is only set
            # later by setup_model_and_optimizer (line 214), so at this point
            # the getattr always returned None, and (b) the pinned
            # AutomodelCheckpointManager.__init__ in this worktree (and in the
            # sibling Nemo-RL-ppo PR #2027 worktree) does not accept that
            # kwarg, causing `TypeError: ... unexpected keyword argument
            # 'model_state_dict_keys'`. We drop the dead kwarg to align with
            # the sibling worktree; restore it once the manager actually
            # consumes a real model_state_dict_keys value.
            self.checkpoint_manager = AutomodelCheckpointManager(
                dp_mesh=self.dp_mesh,
                tp_mesh=self.tp_mesh,
                moe_mesh=self.moe_mesh,
            )
            self.checkpoint_manager.init_checkpointer(
                config_updates=config_updates,
                checkpoint_root=checkpoint_root,
            )
