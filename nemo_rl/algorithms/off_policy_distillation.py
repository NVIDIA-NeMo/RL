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
# See the License for the specific language governing permissions and limitations.
# limitations under the License.

"""Off-policy distillation algorithm.

This module implements off-policy distillation where:
- A fixed dataset of prompt-response pairs is used (no student generation).
- Teacher provides logits for the fixed responses.
- Student aligns with teacher using KL divergence loss.

Key differences from on-policy distillation (``distillation.py``):
- No student generation step; uses pre-existing responses from the dataset.
- No environment needed for reward computation.
- Simpler training loop without rollout generation.
"""

import importlib.util
import multiprocessing
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, NotRequired, Optional, TypedDict, TypeVar, Union, cast

import numpy as np
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoConfig, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.algorithms.loss import (
    CrossTokenizerDistillationLossFn,
    DistillationLossConfig,
    DistillationLossDataDict,
    DistillationLossFn,
)
from nemo_rl.algorithms.utils import maybe_pad_last_batch, set_seed
from nemo_rl.data import DataConfig
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import (
    ClusterConfig,
    RayVirtualCluster,
)
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import ColocatablePolicyInterface
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.checkpoint import CheckpointingConfig, CheckpointManager
from nemo_rl.utils.logger import Logger, LoggerConfig
from nemo_rl.utils.nsys import maybe_gpu_profile_step
from nemo_rl.utils.timer import TimeoutChecker, Timer

# ===============================================================================
# Configuration
# ===============================================================================
TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizerBase)
AnyDistillationLossFn = Union[DistillationLossFn, CrossTokenizerDistillationLossFn]


class TokenAlignerConfig(TypedDict, total=False):
    """Configuration for cross-tokenizer distillation via TokenAligner.

    When enabled, teacher and student may use different tokenizers/vocabularies.
    A precomputed projection matrix maps between the two vocabulary spaces.
    """

    enabled: bool  # Master switch for cross-tokenizer mode
    projection_matrix_path: str  # Path to .pt projection matrix file
    use_sparse_format: bool  # True = sparse COO format, False = dense indices/values
    loss_type: str  # 'KL', 'cross_entropy', or 'chunked_ce'
    exact_token_match_only: bool  # Only use 1:1 aligned token positions for loss
    temperature: float  # Softmax temperature for KL computation
    vocab_topk: int  # Reduce teacher vocab to top-k for speed (0 = all)
    reverse_kl: bool  # If True, use reverse KL direction
    projection_matrix_multiplier: float  # Scaling factor for projection matrix
    max_comb_len: int  # Max combination length for token alignment DP
    learnable: bool  # If True, projection matrix is trainable
    project_teacher_to_student: (
        bool  # If True, project teacher->student instead of student->teacher
    )
    use_char_offset: bool  # If True, try char-offset alignment before DP fallback
    force_dp_only: bool  # If True, disable char-offset path and run DP for all samples
    use_cuda_dp: (
        bool  # If True, patch TokenAligner chunked DP base case with CUDA kernel
    )
    dp_chunk_size: int  # Chunk size used by DP chunked solver
    use_align_fast: (
        bool  # If True, use align_fast for DP path; default False for parity
    )


class OffPolicyDistillationConfig(TypedDict):
    """Configuration for off-policy distillation training.

    Simplified compared to on-policy:
    - No num_generations_per_prompt (we use fixed responses)
    - No max_rollout_turns (no generation)
    """

    num_prompts_per_step: int  # Batch size
    max_num_steps: int  # Maximum number of steps to train for
    max_num_epochs: int  # Maximum number of epochs to train for
    topk_logits_k: int  # Top-k logits for sparse KL loss
    seed: int
    # Validation settings
    val_period: NotRequired[int]  # Run validation every N steps (0 = disabled)
    val_batches: NotRequired[int]  # Number of validation batches (0 = all)
    val_global_batch_size: NotRequired[int]  # Validation batch size
    val_micro_batch_size: NotRequired[int]  # Validation micro batch size
    val_at_start: NotRequired[bool]  # Run validation before training starts
    # CPU processes for parallel cross-tokenizer decode/encode/align (None = auto, 1 = sequential)
    cross_tokenizer_num_workers: NotRequired[Optional[int]]


class OffPolicyDistillationSaveState(TypedDict):
    """State to save for checkpointing."""

    total_steps: int  # Track total number of steps across all epochs
    current_epoch: int  # Track current epoch
    current_step: int  # Track step within current epoch
    consumed_samples: int
    total_valid_tokens: int  # Track total number of non-padding tokens during training


def _default_distillation_save_state() -> OffPolicyDistillationSaveState:
    return {
        "current_epoch": 0,
        "current_step": 0,
        "total_steps": 0,
        "consumed_samples": 0,
        "total_valid_tokens": 0,
    }


class OffPolicyMasterConfig(TypedDict):
    """Main configuration structure for off-policy distillation.

    Key difference from on-policy MasterConfig:
    - No 'env' config (no environment needed)
    """

    policy: PolicyConfig  # Student model configuration
    teacher: PolicyConfig  # Teacher model configuration
    loss_fn: DistillationLossConfig  # Loss function configuration
    data: DataConfig  # Data configuration
    distillation: OffPolicyDistillationConfig  # Distillation configuration
    logger: LoggerConfig  # Logger configuration
    cluster: ClusterConfig  # Cluster configuration
    checkpointing: CheckpointingConfig  # Checkpointing configuration
    token_aligner: NotRequired[TokenAlignerConfig]  # Cross-tokenizer config (optional)


class _PrefetchedBatchPack(TypedDict):
    batch: BatchedDataDict[DatumSpec]
    flat_messages: BatchedDataDict[DatumSpec]
    input_lengths: torch.Tensor
    train_data: BatchedDataDict[DistillationLossDataDict]
    ct_future: Any


# ===============================================================================
# Cross-Tokenizer Parallel Processing
# ===============================================================================

# Module-level global set by _init_align_worker for each pool process.
_ct_token_aligner = None
_ct_dp_chunk_size = 128
_ct_use_align_fast = False


def _init_align_worker(token_aligner, dp_chunk_size: int, use_align_fast: bool):
    """Initializer for ProcessPoolExecutor workers.

    Stores the TokenAligner once per process to avoid re-pickling every call.
    """
    global _ct_token_aligner, _ct_dp_chunk_size, _ct_use_align_fast
    _ct_token_aligner = token_aligner
    _ct_dp_chunk_size = int(dp_chunk_size)
    _ct_use_align_fast = bool(use_align_fast)
    if getattr(_ct_token_aligner, "_use_cuda_dp", False):
        cuda_dp_path_str = getattr(_ct_token_aligner, "_cuda_dp_module_path", "")
        if cuda_dp_path_str:
            spec = importlib.util.spec_from_file_location(
                "x_token_cuda_dp_worker", cuda_dp_path_str
            )
            if spec is not None and spec.loader is not None:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                mod.monkeypatch_tokenaligner_cuda_basecase()


def _align_chunk(args):
    """Align a chunk of (student, teacher) token-ID pairs.

    Called by ProcessPoolExecutor. Uses align_fast only when enabled;
    otherwise falls back to regular align.

    Args:
        args: (student_ids_chunk, teacher_ids_chunk) — both list[list[int]].

    Returns:
        aligned_pairs from ``TokenAligner.align_fast`` or ``TokenAligner.align``.
    """
    student_ids_chunk, teacher_ids_chunk = args
    student_t = torch.tensor(student_ids_chunk)
    teacher_t = torch.tensor(teacher_ids_chunk)
    if _ct_use_align_fast and _ct_token_aligner._student_canon_map is not None:
        return _ct_token_aligner.align_fast(
            student_t, teacher_t, chunk_size=_ct_dp_chunk_size
        )
    return _ct_token_aligner.align(student_t, teacher_t, chunk_size=_ct_dp_chunk_size)


def _align_by_char_offsets(
    s_content: list[tuple[int, int, int]],
    t_content: list[tuple[int, int, int]],
) -> list[tuple]:
    """Align tokens via character offsets in O(n+m).

    Both sequences are tokenizations of the same text, so their character
    spans partition the same string.  A two-pointer walk groups tokens
    whose character boundaries converge.

    Args:
        s_content: [(char_start, char_end, token_position), ...] for student,
                   sorted by char_start, excluding special/pad tokens.
        t_content: same for teacher.

    Returns:
        aligned_pairs in the standard 7-tuple format:
        (s1_tokens, s2_tokens, s_pos_start, s_pos_end, t_pos_start, t_pos_end, is_correct)
    """
    pairs: list[tuple] = []
    si, ti = 0, 0
    n_s, n_t = len(s_content), len(t_content)

    while si < n_s and ti < n_t:
        s_group_start = si
        t_group_start = ti
        s_char_end = s_content[si][1]
        t_char_end = t_content[ti][1]

        while s_char_end != t_char_end:
            if s_char_end < t_char_end:
                si += 1
                if si >= n_s:
                    break
                s_char_end = s_content[si][1]
            else:
                ti += 1
                if ti >= n_t:
                    break
                t_char_end = t_content[ti][1]

        if s_char_end != t_char_end:
            break

        pairs.append(
            (
                [],
                [],
                s_content[s_group_start][2],
                s_content[si][2] + 1,
                t_content[t_group_start][2],
                t_content[ti][2] + 1,
                True,
            )
        )
        si += 1
        ti += 1

    for i in range(si, n_s):
        pos = s_content[i][2]
        pairs.append(([], [], pos, pos + 1, -1, -1, False))
    for i in range(ti, n_t):
        pos = t_content[i][2]
        pairs.append(([], [], -1, -1, pos, pos + 1, False))

    return pairs


def _process_cross_tokenizer_batch(
    train_input_ids: torch.Tensor,
    batch_loss_multiplier: torch.Tensor,
    extra_env: Any,
    tokenizer: PreTrainedTokenizerBase,
    teacher_tokenizer: PreTrainedTokenizerBase,
    token_aligner: Any,
    use_char_offset: bool,
    use_align_fast: bool,
    dp_chunk_size: int,
    ct_pool: Optional[ProcessPoolExecutor],
    max_teacher_len_rt: int,
) -> tuple[torch.Tensor, list[Any], BatchedDataDict]:
    """Prepare teacher inputs + aligned pairs for one training batch."""
    import time as _time

    student_ids = train_input_ids
    batch_size_ct = student_ids.shape[0]

    _t0 = _time.time()
    has_raw_text = (
        extra_env
        and len(extra_env) == batch_size_ct
        and all(
            info is not None and isinstance(info, dict) and "raw_text" in info
            for info in extra_env
        )
    )
    if has_raw_text:
        texts = [info["raw_text"] for info in extra_env]
    else:
        # Fallback only when raw text is unavailable for the batch.
        texts = tokenizer.batch_decode(student_ids.tolist(), skip_special_tokens=True)
    _t1 = _time.time()

    teacher_encoded = teacher_tokenizer(
        texts,
        max_length=max_teacher_len_rt,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    teacher_input_ids = teacher_encoded["input_ids"]
    teacher_attention_mask = teacher_encoded["attention_mask"]
    teacher_offsets = teacher_encoded["offset_mapping"]

    student_re = tokenizer(
        texts,
        max_length=student_ids.shape[1],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    # Align against student/teacher IDs tokenized from the same raw text.
    # This keeps alignment semantics symmetric across tokenizers.
    student_align_ids = student_re["input_ids"]
    student_offsets = student_re["offset_mapping"]
    _t2 = _time.time()

    # --- Vectorized pre-check: which samples can try char-offset? ---
    s_off_np = student_offsets.numpy()
    t_off_np = teacher_offsets.numpy()

    s_nonzero = (s_off_np[:, :, 0] != 0) | (s_off_np[:, :, 1] != 0)
    t_nonzero = (t_off_np[:, :, 0] != 0) | (t_off_np[:, :, 1] != 0)

    s_has = s_nonzero.any(axis=1)
    t_has = t_nonzero.any(axis=1)

    s_last = s_nonzero.shape[1] - 1 - np.flip(s_nonzero, axis=1).argmax(axis=1)
    t_last = t_nonzero.shape[1] - 1 - np.flip(t_nonzero, axis=1).argmax(axis=1)
    s_last_end = s_off_np[np.arange(batch_size_ct), s_last, 1]
    t_last_end = t_off_np[np.arange(batch_size_ct), t_last, 1]

    if not use_char_offset:
        can_try_offset = np.zeros(batch_size_ct, dtype=bool)
    else:
        can_try_offset = s_has & t_has & (s_last_end == t_last_end)

    # --- Vectorized offset filtering (avoid per-sample .tolist()) ---
    # Pre-extract content indices per sample using numpy
    s_content_per_sample = []
    t_content_per_sample = []
    for idx in range(batch_size_ct):
        if can_try_offset[idx]:
            s_mask = s_nonzero[idx]
            t_mask = t_nonzero[idx]
            s_positions = np.where(s_mask)[0]
            t_positions = np.where(t_mask)[0]
            s_content_per_sample.append(
                [
                    (int(s_off_np[idx, p, 0]), int(s_off_np[idx, p, 1]), int(p))
                    for p in s_positions
                ]
            )
            t_content_per_sample.append(
                [
                    (int(t_off_np[idx, p, 0]), int(t_off_np[idx, p, 1]), int(p))
                    for p in t_positions
                ]
            )
        else:
            s_content_per_sample.append(None)
            t_content_per_sample.append(None)

    # --- Char-offset alignment (sequential, fast O(n+m) per sample) ---
    aligned_pairs: list[Any] = [None] * batch_size_ct
    dp_samples_s = []
    dp_samples_t = []
    dp_slot_indices = []

    for idx in range(batch_size_ct):
        if not can_try_offset[idx]:
            dp_samples_s.append(student_align_ids[idx : idx + 1].tolist())
            dp_samples_t.append(teacher_input_ids[idx : idx + 1].tolist())
            dp_slot_indices.append(idx)
            continue

        pairs = _align_by_char_offsets(
            s_content_per_sample[idx], t_content_per_sample[idx]
        )
        n_correct = sum(1 for p in pairs if p[6])
        if n_correct == 0 or n_correct / len(pairs) < 0.5:
            dp_samples_s.append(student_align_ids[idx : idx + 1].tolist())
            dp_samples_t.append(teacher_input_ids[idx : idx + 1].tolist())
            dp_slot_indices.append(idx)
        else:
            aligned_pairs[idx] = pairs

    dp_fallback = len(dp_slot_indices)
    n_offsets = batch_size_ct - dp_fallback

    # --- DP alignment for fallbacks (parallelized) ---
    if dp_fallback > 0:
        if ct_pool is not None and dp_fallback > 1:
            chunks = list(zip(dp_samples_s, dp_samples_t))
            dp_results = list(ct_pool.map(_align_chunk, chunks))
            for i, slot in enumerate(dp_slot_indices):
                aligned_pairs[slot] = dp_results[i][0]
        else:
            for i, slot in enumerate(dp_slot_indices):
                s_t = torch.tensor(dp_samples_s[i])
                t_t = torch.tensor(dp_samples_t[i])
                if use_align_fast and token_aligner._student_canon_map is not None:
                    dp_result = token_aligner.align_fast(
                        s_t, t_t, chunk_size=dp_chunk_size
                    )
                else:
                    dp_result = token_aligner.align(s_t, t_t, chunk_size=dp_chunk_size)
                aligned_pairs[slot] = dp_result[0]

    _t3 = _time.time()
    print(
        f"  [CT timing] decode={_t1 - _t0:.2f}s, "
        f"encode={_t2 - _t1:.2f}s, "
        f"align={_t3 - _t2:.2f}s "
        f"(offsets: {n_offsets}, "
        f"dp_fallback: {dp_fallback})",
        flush=True,
    )

    teacher_input_lengths_ct = teacher_attention_mask.sum(dim=1)

    teacher_token_mask = torch.zeros_like(teacher_input_ids, dtype=torch.float32)
    for i in range(batch_size_ct):
        teacher_token_mask[i, : teacher_input_lengths_ct[i]] = 1.0

    teacher_data = BatchedDataDict(
        {
            "input_ids": teacher_input_ids,
            "input_lengths": teacher_input_lengths_ct,
            "token_mask": teacher_token_mask,
            "sample_mask": batch_loss_multiplier,
        }
    )
    teacher_data.to("cpu")

    return teacher_input_ids, aligned_pairs, teacher_data


# ===============================================================================
# Setup & Initialization
# ===============================================================================
def check_vocab_equality(
    tokenizer: TokenizerType, student_model_name: str, teacher_model_name: str
) -> None:
    """Check if the vocab of the tokenizer (student) and the teacher tokenizer are equal."""
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)

    skip_hint = "Set NRL_SKIP_DISTILLATION_TOKENIZER_CHECK=true to skip this check."

    # 1) Exact token->id mapping equality
    vocab_a = tokenizer.get_vocab()
    vocab_b = teacher_tokenizer.get_vocab()
    assert vocab_a == vocab_b, (
        f"Token->ID mapping differs between student and teacher. {skip_hint}"
    )

    # 2) Size consistency (sanity checks)
    assert len(tokenizer) == len(teacher_tokenizer), (
        f"Effective vocab sizes differ between student and teacher. {skip_hint}"
    )

    # 3) Check model.config.vocab_size to guarantee the last dimension of the logits is the same
    student_config = AutoConfig.from_pretrained(student_model_name)
    teacher_config = AutoConfig.from_pretrained(teacher_model_name)
    assert student_config.vocab_size == teacher_config.vocab_size, (
        f"Model config vocab sizes differ between student and teacher. {skip_hint}"
    )


def _setup_cross_tokenizer(
    master_config: "OffPolicyMasterConfig",
) -> tuple[Any, Optional[PreTrainedTokenizerBase]]:
    """Initialize TokenAligner + teacher tokenizer for cross-tokenizer distillation.

    Returns ``(None, None)`` when cross-tokenizer mode is disabled, otherwise
    returns the configured ``TokenAligner`` and teacher tokenizer.
    """
    token_aligner_cfg = master_config.get("token_aligner", {})
    if not token_aligner_cfg.get("enabled", False):
        return None, None

    from nemo_rl.algorithms.x_token import TokenAligner

    policy_config = master_config["policy"]
    teacher_config = master_config["teacher"]

    print("\n▶ Setting up cross-tokenizer distillation (TokenAligner)...", flush=True)
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_config["model_name"])
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    token_aligner = TokenAligner(
        teacher_tokenizer_name=teacher_config["model_name"],
        student_tokenizer_name=policy_config["model_name"],
        max_comb_len=token_aligner_cfg.get("max_comb_len", 4),
        projection_matrix_multiplier=token_aligner_cfg.get(
            "projection_matrix_multiplier", 1.0
        ),
    )
    token_aligner._load_logits_projection_map(
        file_path=token_aligner_cfg["projection_matrix_path"],
        use_sparse_format=token_aligner_cfg.get("use_sparse_format", True),
        learnable=token_aligner_cfg.get("learnable", False),
        device="cpu",
    )
    if token_aligner_cfg.get("project_teacher_to_student", False):
        token_aligner.create_reverse_projection_matrix(device="cpu")

    print(
        f"  ✓ TokenAligner initialized "
        f"({policy_config['model_name']} → {teacher_config['model_name']})",
        flush=True,
    )

    token_aligner.precompute_canonical_maps()
    if token_aligner_cfg.get("use_cuda_dp", False):
        cuda_dp_path = (
            Path(__file__).resolve().parents[2] / "x_token" / "cuda_tokenalign_dp.py"
        )
        if not cuda_dp_path.exists():
            raise FileNotFoundError(
                f"Requested token_aligner.use_cuda_dp=true but file not found: {cuda_dp_path}"
            )
        spec = importlib.util.spec_from_file_location(
            "x_token_cuda_dp", str(cuda_dp_path)
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load CUDA DP module from: {cuda_dp_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.monkeypatch_tokenaligner_cuda_basecase()
        token_aligner._use_cuda_dp = True
        token_aligner._cuda_dp_module_path = str(cuda_dp_path)
        print("  ✓ CUDA DP monkeypatch enabled for TokenAligner", flush=True)
    if token_aligner_cfg.get("force_dp_only", False):
        print("  ✓ force_dp_only enabled (char-offset disabled)", flush=True)

    return token_aligner, teacher_tokenizer


def _ensure_topk_logprobs_for_non_ipc(
    teacher_topk_logits: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    """Normalize teacher top-k values to log-probs for non-IPC distillation.

    Depending on worker/backend path, `get_topk_logits` may return either:
    - top-k log-probabilities, or
    - raw top-k logits.
    Distillation loss expects log-probs in this non-IPC data-dict path.
    """
    teacher_topk_logits = teacher_topk_logits.to(torch.float32)
    topk_mass = teacher_topk_logits.exp().sum(dim=-1)
    looks_like_logprobs = bool(
        (teacher_topk_logits.max() <= 1e-6).item()
        and (topk_mass.max() <= 1.0001).item()
    )
    if looks_like_logprobs:
        return teacher_topk_logits, False
    return torch.nn.functional.log_softmax(teacher_topk_logits, dim=-1), True


def setup(
    master_config: OffPolicyMasterConfig,
    tokenizer: TokenizerType,
    train_dataset: AllTaskProcessedDataset,
    val_dataset: Optional[AllTaskProcessedDataset] = None,
) -> tuple[
    ColocatablePolicyInterface,  # student_policy
    ColocatablePolicyInterface,  # teacher_policy
    StatefulDataLoader,  # train_dataloader
    Optional[StatefulDataLoader],  # val_dataloader
    AnyDistillationLossFn,
    Logger,
    CheckpointManager,
    OffPolicyDistillationSaveState,
    OffPolicyMasterConfig,
    Any,  # token_aligner (None unless cross-tokenizer mode)
    Optional[
        PreTrainedTokenizerBase
    ],  # teacher_tokenizer (None unless cross-tokenizer)
]:
    """Set up off-policy distillation training components.

    Differs from on-policy :func:`nemo_rl.algorithms.distillation.setup`:
      - No ``student_generation`` interface (responses come from the dataset).
      - Single training cluster (no inference cluster needed).
      - Optionally initializes a ``TokenAligner`` for cross-tokenizer mode.

    Returns:
        ``(student_policy, teacher_policy, train_dataloader, val_dataloader,
        loss_fn, logger, checkpointer, save_state, master_config,
        token_aligner, teacher_tokenizer)``.
    """
    # Extract configuration
    policy_config = master_config["policy"]
    teacher_config = master_config["teacher"]
    loss_config = master_config["loss_fn"]
    distillation_config = master_config["distillation"]
    data_config = master_config["data"]
    logger_config = master_config["logger"]
    cluster_config = master_config["cluster"]

    # Disallow SP + packing for dtensor path
    for cfg, who in ((policy_config, "student"), (teacher_config, "teacher")):
        dtensor_enabled = cfg["dtensor_cfg"]["enabled"]
        sequence_packing_enabled = (
            "sequence_packing" in cfg and cfg["sequence_packing"]["enabled"]
        )
        sequence_parallel_enabled = (
            "sequence_parallel" in cfg["dtensor_cfg"]
            and cfg["dtensor_cfg"]["sequence_parallel"]
        )

        if dtensor_enabled and sequence_packing_enabled and sequence_parallel_enabled:
            raise AssertionError(
                f"Distillation does not support DTensor sequence parallel + sequence packing ({who} policy). "
                "Please refer to https://github.com/NVIDIA-NeMo/RL/issues/1178 for more details."
            )

    # Set random seed
    set_seed(distillation_config["seed"])

    # ==========================
    #         Logger
    # ==========================
    logger = Logger(logger_config)
    logger.log_hyperparams(master_config)

    # ==========================
    #      Checkpointing
    # ==========================
    checkpointer = CheckpointManager(master_config["checkpointing"])
    last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
    distillation_save_state: Optional[OffPolicyDistillationSaveState] = cast(
        Optional[OffPolicyDistillationSaveState],
        checkpointer.load_training_info(last_checkpoint_path),
    )
    if distillation_save_state is None:
        distillation_save_state = _default_distillation_save_state()

    # ==========================
    #           Data
    # ==========================
    dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=distillation_config["num_prompts_per_step"],
        shuffle=data_config.get("shuffle", True),
        collate_fn=rl_collate_fn,
        drop_last=True,
    )

    if last_checkpoint_path:
        dataloader_state_dict = torch.load(
            os.path.join(last_checkpoint_path, "train_dataloader.pt")
        )
        dataloader.load_state_dict(dataloader_state_dict)

    print(
        f"  ✓ Training dataloader loaded with {len(train_dataset)} samples", flush=True
    )

    # Load validation dataloader if provided
    val_dataloader: Optional[StatefulDataLoader] = None
    val_period = distillation_config.get("val_period", 0)
    val_at_start = distillation_config.get("val_at_start", False)
    if val_period > 0 or val_at_start:
        assert val_dataset is not None, (
            "Validation dataset is required if validation is enabled "
            "(val_period > 0 or val_at_start = True)"
        )
        val_dataloader = StatefulDataLoader(
            val_dataset,
            batch_size=distillation_config.get(
                "val_global_batch_size", distillation_config["num_prompts_per_step"]
            ),
            shuffle=False,
            collate_fn=rl_collate_fn,
            drop_last=False,
        )
        print(
            f"  ✓ Validation dataloader loaded with {len(val_dataset)} samples",
            flush=True,
        )

    # ==========================
    #          Cluster
    # ==========================
    # For off-policy distillation, we only need a training cluster
    # No inference cluster needed since we don't generate responses
    print("\n▶ Setting up compute cluster...", flush=True)

    cluster = RayVirtualCluster(
        name="off_policy_distillation_cluster",
        bundle_ct_per_node_list=[cluster_config["gpus_per_node"]]
        * cluster_config["num_nodes"],
        use_gpus=True,
        num_gpus_per_node=cluster_config["gpus_per_node"],
        max_colocated_worker_groups=3,
    )
    print(
        f"  ✓ Ray cluster initialized with {cluster_config['num_nodes']} nodes",
        flush=True,
    )

    # ==========================
    #      Cross-Tokenizer Setup
    # ==========================
    token_aligner, teacher_tokenizer = _setup_cross_tokenizer(master_config)
    cross_tokenizer_enabled = token_aligner is not None

    # ==========================
    #      Teacher Policy
    # ==========================
    print("\n▶ Setting up teacher policy...", flush=True)

    if not cross_tokenizer_enabled:
        if not bool(os.getenv("NRL_SKIP_DISTILLATION_TOKENIZER_CHECK", False)):
            check_vocab_equality(
                tokenizer, policy_config["model_name"], teacher_config["model_name"]
            )

    if "megatron_cfg" in teacher_config and teacher_config["megatron_cfg"]["enabled"]:
        ## NOTE: this is equal to the total number of scheduler steps
        total_train_iters = min(
            distillation_config["max_num_steps"],
            distillation_config["max_num_epochs"] * len(dataloader),
        )
        teacher_config["megatron_cfg"]["train_iters"] = total_train_iters

    teacher_policy = Policy(
        name_prefix="teacher",
        cluster=cluster,
        config=teacher_config,
        tokenizer=teacher_tokenizer if cross_tokenizer_enabled else tokenizer,
        weights_path=None,
        optimizer_path=None,
        init_optimizer=False,
        init_reference_model=False,
    )
    teacher_policy.offload_after_refit()

    # ==========================
    #      Student Policy
    # ==========================
    # Note: No student_generation interface for off-policy distillation
    print("\n▶ Setting up student policy...", flush=True)

    # Checkpoint paths
    weights_path = None
    optimizer_path = None
    if last_checkpoint_path:
        weights_path = Path(last_checkpoint_path) / "policy" / "weights"
        optimizer_path = Path(last_checkpoint_path) / "policy" / "optimizer"

    if "megatron_cfg" in policy_config and policy_config["megatron_cfg"]["enabled"]:
        ## NOTE: this is equal to the total number of scheduler steps
        total_train_iters = min(
            distillation_config["max_num_steps"],
            distillation_config["max_num_epochs"] * len(dataloader),
        )
        policy_config["megatron_cfg"]["train_iters"] = total_train_iters

    student_policy = Policy(
        name_prefix="student",
        cluster=cluster,
        config=policy_config,
        tokenizer=tokenizer,
        weights_path=weights_path,
        optimizer_path=optimizer_path,
        init_optimizer=True,
        init_reference_model=False,
    )

    if cross_tokenizer_enabled:
        loss_fn = CrossTokenizerDistillationLossFn(loss_config, token_aligner)
    else:
        loss_fn = DistillationLossFn(loss_config)

    print("\n" + "=" * 60)
    print(" " * 12 + "OFF-POLICY DISTILLATION SETUP COMPLETE")
    print("=" * 60 + "\n", flush=True)

    return (
        student_policy,
        teacher_policy,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        distillation_save_state,
        master_config,
        token_aligner,
        teacher_tokenizer,
    )


# ===============================================================================
# Training & Validation
# ===============================================================================


def off_policy_distillation_train(
    student_policy: ColocatablePolicyInterface,
    teacher_policy: ColocatablePolicyInterface,
    dataloader: StatefulDataLoader,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer: TokenizerType,
    loss_fn: AnyDistillationLossFn,
    logger: Logger,
    checkpointer: CheckpointManager,
    distillation_save_state: OffPolicyDistillationSaveState,
    master_config: OffPolicyMasterConfig,
    eval_hook: Optional[Callable] = None,
    eval_hook_period: int = 0,
    eval_hook_at_start: bool = False,
    token_aligner=None,
    teacher_tokenizer=None,
) -> None:
    """Run off-policy distillation training algorithm.

    Key differences from on-policy distillation train():
    - No student_generation parameter (we don't generate responses)
    - No task_to_env / val_task_to_env (no environment scoring)
    - No rollout generation step - uses fixed responses from dataset directly

    Training loop:
    1. Load batch with prompt-response pairs (responses already in dataset)
    2. Add loss masks (train on assistant tokens only)
    3. Get teacher top-k logits for the fixed responses
    4. Train student with KL divergence loss

    Args:
        eval_hook: Optional callback ``(step, student_policy, teacher_policy, logger) -> dict``
            called every *eval_hook_period* steps.  Return value (if dict) is
            logged under ``prefix="eval_hook"`` and used for checkpoint metric lookup.
        eval_hook_period: How often (in steps) to call *eval_hook*. 0 = disabled.
        eval_hook_at_start: If True, call eval_hook before the first training step.
    """
    timer = Timer()
    timeout = TimeoutChecker(
        timeout=master_config["checkpointing"].get("checkpoint_must_save_by", None),
        fit_last_save_time=True,
    )
    timeout.start_iterations()

    # common config/state items
    current_epoch = distillation_save_state["current_epoch"]  # current epoch
    current_step = distillation_save_state[
        "current_step"
    ]  # current step within current epoch
    total_steps = distillation_save_state[
        "total_steps"
    ]  # total number of steps across all epochs
    consumed_samples = distillation_save_state["consumed_samples"]
    total_valid_tokens = distillation_save_state["total_valid_tokens"]
    max_epochs = master_config["distillation"][
        "max_num_epochs"
    ]  # max number of epochs to train for
    max_steps = master_config["distillation"][
        "max_num_steps"
    ]  # max number of steps to train for

    # Validation configuration
    val_period = master_config["distillation"].get("val_period", 0)
    val_at_start = master_config["distillation"].get("val_at_start", False)

    # Run validation at the start if configured
    if val_at_start and total_steps == 0:
        print("\n🔍 Running initial validation...", flush=True)
        val_metrics, validation_timings = validate(
            student_policy,
            teacher_policy,
            val_dataloader,
            tokenizer,
            loss_fn,
            step=0,
            master_config=master_config,
        )
        logger.log_metrics(val_metrics, total_steps, prefix="validation")
        logger.log_metrics(validation_timings, total_steps, prefix="timing/validation")

    # Run eval hook at start if configured
    eval_hook_metrics = None
    if eval_hook and eval_hook_at_start and total_steps == 0:
        print("\n🔍 Running initial eval hook...", flush=True)
        eval_hook_metrics = eval_hook(
            step=0,
            student_policy=student_policy,
            teacher_policy=teacher_policy,
            logger=logger,
        )
        if isinstance(eval_hook_metrics, dict):
            logger.log_metrics(eval_hook_metrics, 0, prefix="eval_hook")

    # Run off-policy distillation training
    batch: BatchedDataDict[DatumSpec]

    # Create a process pool for cross-tokenizer processing (if enabled).
    cross_tokenizer_enabled = (
        token_aligner is not None and teacher_tokenizer is not None
    )
    token_aligner_cfg = master_config.get("token_aligner", {})
    dp_chunk_size = int(token_aligner_cfg.get("dp_chunk_size", 128))
    # Default to DP-only for parity/stability; char-offset is opt-in.
    use_char_offset = bool(token_aligner_cfg.get("use_char_offset", False))
    if bool(token_aligner_cfg.get("force_dp_only", False)):
        # Backward-compatible override for older configs.
        use_char_offset = False
    use_align_fast = bool(token_aligner_cfg.get("use_align_fast", False))
    ct_num_workers = master_config["distillation"].get(
        "cross_tokenizer_num_workers", None
    )
    if ct_num_workers is None:
        ct_num_workers = os.cpu_count() or 1
    ct_pool: Optional[ProcessPoolExecutor] = None
    if cross_tokenizer_enabled and ct_num_workers > 1:
        mp_ctx = multiprocessing.get_context("forkserver")
        ct_pool = ProcessPoolExecutor(
            max_workers=ct_num_workers,
            mp_context=mp_ctx,
            initializer=_init_align_worker,
            initargs=(token_aligner, dp_chunk_size, use_align_fast),
        )
        print(
            f"  ✓ Cross-tokenizer process pool created with {ct_num_workers} workers "
            f"(dp_chunk_size={dp_chunk_size})",
            flush=True,
        )
    if cross_tokenizer_enabled:
        print(f"  ✓ TokenAligner mode: use_char_offset={use_char_offset}", flush=True)
        print(f"  ✓ TokenAligner DP mode: use_align_fast={use_align_fast}", flush=True)

    ct_prefetch_pool: Optional[ThreadPoolExecutor] = None
    if cross_tokenizer_enabled:
        ct_prefetch_pool = ThreadPoolExecutor(max_workers=1)

    def _shutdown_alignment_pools() -> None:
        if ct_prefetch_pool is not None:
            ct_prefetch_pool.shutdown(wait=False, cancel_futures=True)
        if ct_pool is not None:
            ct_pool.shutdown(wait=False)

    def _prepare_train_batch_data(batch_obj: BatchedDataDict[DatumSpec]):
        # Add loss mask for each message (train on assistant tokens only)
        # Skip if token_loss_mask already exists from data processor
        for message_log in batch_obj["message_log"]:
            for message in message_log:
                if "token_loss_mask" not in message:
                    if message["role"] == "assistant":
                        message["token_loss_mask"] = torch.ones_like(
                            message["token_ids"]
                        )
                    else:
                        message["token_loss_mask"] = torch.zeros_like(
                            message["token_ids"]
                        )

        # Convert message_log to flat format for training
        flat_messages_obj, input_lengths_obj = batched_message_log_to_flat_message(
            batch_obj["message_log"],
            pad_value_dict={"token_ids": tokenizer.pad_token_id},
            make_sequence_length_divisible_by=master_config["policy"].get(
                "make_sequence_length_divisible_by", 1
            ),
        )

        train_data_obj = BatchedDataDict[DistillationLossDataDict](
            {
                "input_ids": flat_messages_obj["token_ids"],
                "input_lengths": input_lengths_obj,
                "token_mask": flat_messages_obj["token_loss_mask"],
                "sample_mask": batch_obj["loss_multiplier"],
            }
        )
        train_data_obj.update(flat_messages_obj.get_multimodal_dict(as_tensors=False))
        train_data_obj.to("cpu")
        return flat_messages_obj, input_lengths_obj, train_data_obj

    def _get_max_teacher_len() -> int:
        return int(
            master_config["teacher"].get(
                "max_total_sequence_length",
                master_config["policy"]["max_total_sequence_length"],
            )
        )

    def _resolve_cross_tokenizer_batch_data(
        train_data_obj: BatchedDataDict[DistillationLossDataDict],
        batch_obj: BatchedDataDict[DatumSpec],
        ct_future_obj: Any,
    ) -> tuple[torch.Tensor, list[Any], BatchedDataDict]:
        if ct_future_obj is not None:
            return ct_future_obj.result()
        return _process_cross_tokenizer_batch(
            train_input_ids=train_data_obj["input_ids"],
            batch_loss_multiplier=batch_obj["loss_multiplier"],
            extra_env=batch_obj.get("extra_env_info"),
            tokenizer=tokenizer,
            teacher_tokenizer=teacher_tokenizer,
            token_aligner=token_aligner,
            use_char_offset=use_char_offset,
            use_align_fast=use_align_fast,
            dp_chunk_size=dp_chunk_size,
            ct_pool=ct_pool,
            max_teacher_len_rt=_get_max_teacher_len(),
        )

    def _maybe_prefetch_next_batch(
        dataloader_iter_obj: Any,
    ) -> Optional[_PrefetchedBatchPack]:
        if not cross_tokenizer_enabled or ct_prefetch_pool is None:
            return None

        try:
            next_batch_obj = next(dataloader_iter_obj)
        except StopIteration:
            return None

        next_flat_messages, next_input_lengths, next_train_data = (
            _prepare_train_batch_data(next_batch_obj)
        )
        next_ct_future = ct_prefetch_pool.submit(
            _process_cross_tokenizer_batch,
            next_train_data["input_ids"],
            next_batch_obj["loss_multiplier"],
            next_batch_obj.get("extra_env_info"),
            tokenizer,
            teacher_tokenizer,
            token_aligner,
            use_char_offset,
            use_align_fast,
            dp_chunk_size,
            ct_pool,
            _get_max_teacher_len(),
        )
        return {
            "batch": next_batch_obj,
            "flat_messages": next_flat_messages,
            "input_lengths": next_input_lengths,
            "train_data": next_train_data,
            "ct_future": next_ct_future,
        }

    while total_steps < max_steps and current_epoch < max_epochs:
        print(
            f"\n{'=' * 25} Epoch {current_epoch + 1}/{max_epochs} {'=' * 25}",
            flush=True,
        )

        dataloader_iter = iter(dataloader)
        prefetched_batch_pack: Optional[_PrefetchedBatchPack] = None
        while total_steps < max_steps:
            if prefetched_batch_pack is not None:
                batch = prefetched_batch_pack["batch"]
                flat_messages = prefetched_batch_pack["flat_messages"]
                input_lengths = prefetched_batch_pack["input_lengths"]
                train_data = prefetched_batch_pack["train_data"]
                ct_future = prefetched_batch_pack["ct_future"]
                prefetched_batch_pack = None
                loaded_from_prefetch = True
            else:
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    break
                loaded_from_prefetch = False
                ct_future = None

            print(
                f"\n{'=' * 25} Step {current_step + 1}/{min(len(dataloader), max_steps)} {'=' * 25}",
                flush=True,
            )
            maybe_gpu_profile_step(student_policy, total_steps + 1)
            val_metrics, validation_timings = None, None

            with timer.time("total_step_time"):
                # ==== Data Processing ====
                # Off-policy: Use responses from dataset directly (no generation)
                if not loaded_from_prefetch:
                    print(
                        "▶ Processing batch data (off-policy - using fixed responses)...",
                        flush=True,
                    )
                    with timer.time("data_processing"):
                        flat_messages, input_lengths, train_data = (
                            _prepare_train_batch_data(batch)
                        )
                else:
                    print("▶ Using prefetched batch data...", flush=True)
                    # Keep timing key stable in logs when using prefetched data.
                    with timer.time("data_processing"):
                        pass

                # ==== Cross-Tokenizer Data Processing ====
                teacher_data = None

                if cross_tokenizer_enabled:
                    with timer.time("cross_tokenizer_processing"):
                        teacher_input_ids, aligned_pairs, teacher_data = (
                            _resolve_cross_tokenizer_batch_data(
                                train_data_obj=train_data,
                                batch_obj=batch,
                                ct_future_obj=ct_future,
                            )
                        )

                        loss_fn.set_cross_tokenizer_data(
                            teacher_input_ids=teacher_input_ids,
                            aligned_pairs=aligned_pairs,
                        )

                # Prepare one-step-ahead cross-tokenizer preprocessing in the background.
                if cross_tokenizer_enabled and prefetched_batch_pack is None:
                    prefetched_batch_pack = _maybe_prefetch_next_batch(dataloader_iter)

                # ==== Teacher Logprob Inference ====
                use_ipc = bool(master_config["distillation"].get("use_ipc", True))
                topk_k = master_config["distillation"]["topk_logits_k"]

                print("▶ Preparing for teacher logprob inference...", flush=True)
                with timer.time("teacher_logprob_inference_prep"):
                    student_policy.offload_after_refit()
                    teacher_policy.prepare_for_lp_inference()

                teacher_fwd_data = (
                    teacher_data if cross_tokenizer_enabled else train_data
                )
                teacher_topk_k = None if cross_tokenizer_enabled else topk_k

                if use_ipc:
                    print("▶ Computing teacher logprobs (IPC)...", flush=True)
                    with timer.time("teacher_logprob_inference"):
                        teacher_logits = teacher_policy.train(
                            teacher_fwd_data,
                            None,
                            eval_mode=True,
                            is_teacher=True,
                            topk_logits=teacher_topk_k,
                            gbs=master_config["policy"]["train_global_batch_size"],
                            mbs=master_config["policy"]["train_micro_batch_size"],
                        )
                else:
                    if cross_tokenizer_enabled:
                        raise NotImplementedError(
                            "Cross-tokenizer distillation requires use_ipc=True. "
                            "Set distillation.use_ipc: true in the config."
                        )
                    print(
                        "▶ Computing teacher logprobs (non-IPC, data dict)...",
                        flush=True,
                    )
                    with timer.time("teacher_logprob_inference"):
                        teacher_topk = teacher_policy.get_topk_logits(
                            train_data, k=topk_k
                        )
                        teacher_topk_logprobs, converted_to_logprobs = (
                            _ensure_topk_logprobs_for_non_ipc(
                                teacher_topk["topk_logits"]
                            )
                        )
                        train_data["teacher_topk_logits"] = teacher_topk_logprobs
                        train_data["teacher_topk_indices"] = teacher_topk[
                            "topk_indices"
                        ]
                        if (
                            converted_to_logprobs
                            and total_steps == 0
                            and current_step == 0
                        ):
                            print(
                                "⚠️ teacher.get_topk_logits returned raw logits in non-IPC mode; "
                                "normalizing with log_softmax before distillation loss.",
                                flush=True,
                            )
                        del teacher_topk

                # ==== Student Training ====
                print("▶ Preparing for training...", flush=True)
                with timer.time("training_prep"):
                    teacher_policy.offload_after_refit()
                    student_policy.prepare_for_training()

                if cross_tokenizer_enabled:
                    if not getattr(student_policy, "_loss_fn_initialized", False):
                        student_policy._loss_fn_initialized = True
                        token_aligner_cfg = master_config.get("token_aligner", {})
                        student_policy.init_cross_tokenizer_loss_fn(
                            loss_config=master_config["loss_fn"],
                            token_aligner_config={
                                "teacher_model": master_config["teacher"]["model_name"],
                                "student_model": master_config["policy"]["model_name"],
                                "projection_matrix_path": token_aligner_cfg[
                                    "projection_matrix_path"
                                ],
                                "use_sparse_format": token_aligner_cfg.get(
                                    "use_sparse_format", True
                                ),
                                "learnable": token_aligner_cfg.get("learnable", False),
                                "max_comb_len": token_aligner_cfg.get(
                                    "max_comb_len", 4
                                ),
                                "projection_matrix_multiplier": token_aligner_cfg.get(
                                    "projection_matrix_multiplier", 1.0
                                ),
                                "project_teacher_to_student": token_aligner_cfg.get(
                                    "project_teacher_to_student", False
                                ),
                            },
                        )
                    student_policy.update_cross_tokenizer_data(
                        teacher_input_ids=teacher_input_ids,
                        aligned_pairs=aligned_pairs,
                    )

                student_loss_fn = None if cross_tokenizer_enabled else loss_fn
                print("▶ Training policy...", flush=True)
                with timer.time("policy_training"):
                    train_kwargs: dict[str, Any] = {}
                    if use_ipc:
                        train_kwargs["teacher_logits"] = teacher_logits
                        train_kwargs["use_teacher_ipc_loss_postprocessor"] = True

                    train_results = student_policy.train(
                        train_data,
                        student_loss_fn,
                        **train_kwargs,
                    )

                    if use_ipc:
                        del teacher_logits

                is_last_step = (total_steps + 1 >= max_steps) or (
                    (current_epoch + 1 == max_epochs)
                    and (current_step + 1 == len(dataloader))
                )

                # ==== Validation ====
                if val_period > 0 and (total_steps + 1) % val_period == 0:
                    val_metrics, validation_timings = validate(
                        student_policy,
                        teacher_policy,
                        val_dataloader,
                        tokenizer,
                        loss_fn,
                        step=total_steps + 1,
                        master_config=master_config,
                    )
                    logger.log_metrics(
                        validation_timings, total_steps + 1, prefix="timing/validation"
                    )
                    logger.log_metrics(
                        val_metrics, total_steps + 1, prefix="validation"
                    )

                # ==== Eval Hook (e.g., generation-based MATH/MMLU eval) ====
                if (
                    eval_hook
                    and eval_hook_period > 0
                    and (total_steps + 1) % eval_hook_period == 0
                ):
                    print(
                        f"\n🔍 Running eval hook at step {total_steps + 1}...",
                        flush=True,
                    )
                    with timer.time("eval_hook"):
                        eval_hook_metrics = eval_hook(
                            step=total_steps + 1,
                            student_policy=student_policy,
                            teacher_policy=teacher_policy,
                            logger=logger,
                        )
                    if isinstance(eval_hook_metrics, dict):
                        logger.log_metrics(
                            eval_hook_metrics, total_steps + 1, prefix="eval_hook"
                        )
                    student_policy.prepare_for_training()

                # ==== Metrics ====
                metrics = {
                    "loss": train_results["loss"].numpy(),
                    "grad_norm": train_results["grad_norm"].numpy(),
                    "mean_seq_length": batch["length"].numpy().mean(),
                    "total_num_tokens": input_lengths.numpy().sum(),
                }
                metrics.update(train_results["all_mb_metrics"])
                for k, v in metrics.items():
                    if k in {
                        "lr",
                        "wd",
                        "global_valid_seqs",
                        "global_valid_toks",
                        "mean_seq_length",
                    }:
                        metrics[k] = np.mean(v).item()
                    else:
                        metrics[k] = np.sum(v).item()
                total_valid_tokens += metrics["global_valid_toks"]

                ## Checkpointing
                consumed_samples += master_config["distillation"][
                    "num_prompts_per_step"
                ]
                timeout.mark_iteration()

                should_save_by_step = (
                    is_last_step
                    or (total_steps + 1) % master_config["checkpointing"]["save_period"]
                    == 0
                )
                # Check if timeout-based checkpointing is enabled in config.
                should_save_by_timeout = timeout.check_save()

                if master_config["checkpointing"]["enabled"] and (
                    should_save_by_step or should_save_by_timeout
                ):
                    student_policy.prepare_for_training()

                    distillation_save_state["current_epoch"] = current_epoch
                    distillation_save_state["current_step"] = current_step + 1
                    distillation_save_state["total_steps"] = total_steps + 1
                    distillation_save_state["total_valid_tokens"] = total_valid_tokens
                    distillation_save_state["consumed_samples"] = consumed_samples

                    full_metric_name = master_config["checkpointing"]["metric_name"]
                    if full_metric_name is not None:
                        assert full_metric_name.startswith(
                            "train:"
                        ) or full_metric_name.startswith("val:"), (
                            f"metric_name={full_metric_name} must start with 'val:' or 'train:',\n"
                            f'followed by the corresponding name in the "val" or "train" metrics dictionary. '
                            f"Example: 'train:loss' or 'val:val_loss'"
                        )
                        prefix, metric_name = full_metric_name.split(":", 1)
                        metrics_source = metrics if prefix == "train" else val_metrics
                        if not metrics_source:
                            warnings.warn(
                                f"You asked to save checkpoints based on {metric_name} but no {prefix} metrics were collected. "
                                "This checkpoint will not be saved as top-k.",
                                stacklevel=2,
                            )
                            if full_metric_name in distillation_save_state:
                                del distillation_save_state[full_metric_name]
                        elif metric_name not in metrics_source:
                            raise ValueError(
                                f"Metric {metric_name} not found in {prefix} metrics"
                            )
                        else:
                            distillation_save_state[full_metric_name] = metrics_source[
                                metric_name
                            ]

                    with timer.time("checkpointing"):
                        print(
                            f"Saving checkpoint for step {total_steps + 1}...",
                            flush=True,
                        )
                        checkpoint_path = checkpointer.init_tmp_checkpoint(
                            total_steps + 1, distillation_save_state, master_config
                        )
                        student_policy.save_checkpoint(
                            weights_path=os.path.join(
                                checkpoint_path, "policy", "weights"
                            ),
                            optimizer_path=os.path.join(
                                checkpoint_path, "policy", "optimizer"
                            )
                            if checkpointer.save_optimizer
                            else None,
                            tokenizer_path=os.path.join(
                                checkpoint_path, "policy", "tokenizer"
                            ),
                            checkpointing_cfg=master_config["checkpointing"],
                        )
                        torch.save(
                            dataloader.state_dict(),
                            os.path.join(checkpoint_path, "train_dataloader.pt"),
                        )
                        checkpointer.finalize_checkpoint(checkpoint_path)

            # Logging
            # Log training data
            log_data = {"content": flat_messages["content"]}
            log_data["input_lengths"] = input_lengths.tolist()
            logger.log_batched_dict_as_jsonl(
                log_data, f"train_data_step{total_steps + 1}.jsonl"
            )

            timing_metrics: dict[str, float] = timer.get_timing_metrics(
                reduction_op="sum"
            )  # type: ignore

            print("\n📊 Training Results:")

            print(f"  • Loss: {metrics['loss']:.4f}")
            print(f"  • Grad Norm: {metrics['grad_norm']:.4f}")
            print(f"  • Mean Sequence Length: {metrics['mean_seq_length']:.1f}")

            if "total_flops" in train_results:
                total_time = timing_metrics.get("total_step_time", 0)
                total_tflops = (
                    train_results["total_flops"]
                    / timing_metrics["policy_training"]
                    / 1e12
                )
                num_ranks = train_results["num_ranks"]
                print(
                    f"  • Training FLOPS: {total_tflops:.2f} TFLOPS ({total_tflops / num_ranks:.2f} TFLOPS per rank)",
                    flush=True,
                )
                if "theoretical_tflops" in train_results:
                    theoretical_tflops = train_results["theoretical_tflops"]
                    print(
                        f"  • Training Model Floating Point Utilization: {100 * total_tflops / theoretical_tflops:.2f}%",
                        flush=True,
                    )
                    metrics["train_fp_utilization"] = total_tflops / theoretical_tflops

            print("\n⏱️  Timing:", flush=True)
            # Display total time first, separately
            total_time = timing_metrics.get("total_step_time", 0)

            total_num_gpus = (
                master_config["cluster"]["num_nodes"]
                * master_config["cluster"]["gpus_per_node"]
            )
            metrics.update(
                {
                    "tokens_per_sec_per_gpu": metrics["total_num_tokens"]
                    / total_time
                    / total_num_gpus
                }
            )

            print(f"  • Total step time: {total_time:.2f}s", flush=True)

            # Display all other timing metrics
            for k, v in sorted(
                timing_metrics.items(), key=lambda item: item[1], reverse=True
            ):
                if k != "total_step_time":
                    percent = (v / total_time * 100) if total_time > 0 else 0
                    print(f"  • {k}: {v:.2f}s ({percent:.1f}%)", flush=True)

            timing_metrics["valid_tokens_per_sec_per_gpu"] = (
                metrics["global_valid_toks"] / total_time / total_num_gpus
            )
            logger.log_metrics(metrics, total_steps + 1, prefix="train")
            logger.log_metrics(timing_metrics, total_steps + 1, prefix="timing/train")

            timer.reset()
            current_step += 1
            total_steps += 1
            if should_save_by_timeout:
                print("Timeout has been reached, stopping training early", flush=True)
                _shutdown_alignment_pools()
                return
            if total_steps >= max_steps:
                print(
                    "Max number of steps has been reached, stopping training early",
                    flush=True,
                )
                _shutdown_alignment_pools()
                return

        # End of epoch
        current_epoch += 1
        current_step = 0  # Reset step counter for new epoch

    _shutdown_alignment_pools()


def validate(
    student_policy: ColocatablePolicyInterface,
    teacher_policy: ColocatablePolicyInterface,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer: TokenizerType,
    loss_fn: AnyDistillationLossFn,
    step: int,
    master_config: OffPolicyMasterConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run validation for off-policy distillation.

    Computes teacher top-k logits and student distillation loss on the
    validation set in eval mode (no gradient updates). Supports both the
    IPC teacher-logit path and the data-dict (non-IPC) path.
    """
    if val_dataloader is None:
        print("  ⚠️ No validation dataloader provided, skipping validation", flush=True)
        return {}, {}

    timer = Timer()

    with timer.time("total_validation_time"):
        print(f"▶ Starting validation at step {step}...", flush=True)

        val_metrics: dict[str, Any] = {"val_loss": 0.0}
        sum_num_valid_tokens = 0

        val_batches = master_config["distillation"].get("val_batches", 0)
        val_batch_size = master_config["distillation"].get(
            "val_global_batch_size",
            master_config["distillation"]["num_prompts_per_step"],
        )
        val_mbs = master_config["distillation"].get(
            "val_micro_batch_size", val_batch_size
        )

        for batch_idx, val_batch in enumerate(val_dataloader):
            # Add loss masks for assistant tokens.
            for message_log in val_batch["message_log"]:
                for message in message_log:
                    if "token_loss_mask" not in message:
                        if message["role"] == "assistant":
                            message["token_loss_mask"] = torch.ones_like(
                                message["token_ids"]
                            )
                        else:
                            message["token_loss_mask"] = torch.zeros_like(
                                message["token_ids"]
                            )

            # Flatten messages.
            flat_messages, input_lengths = batched_message_log_to_flat_message(
                val_batch["message_log"],
                pad_value_dict={"token_ids": tokenizer.pad_token_id},
                make_sequence_length_divisible_by=master_config["policy"].get(
                    "make_sequence_length_divisible_by", 1
                ),
            )

            val_data = BatchedDataDict[DistillationLossDataDict](
                {
                    "input_ids": flat_messages["token_ids"],
                    "input_lengths": input_lengths,
                    "token_mask": flat_messages["token_loss_mask"],
                    "sample_mask": val_batch["loss_multiplier"],
                }
            )
            val_data.update(flat_messages.get_multimodal_dict(as_tensors=False))
            val_data.to("cpu")

            # Pad partial batch if needed (drop_last=False for val).
            # Must pad BEFORE teacher logits to avoid size mismatch: teacher.get_topk_logits
            # internally pads for its own DP sharding and returns padded-size outputs,
            # so all inputs must be uniformly padded first.
            if val_data.size < val_batch_size:
                dp_size = student_policy.sharding_annotations.get_axis_size(
                    "data_parallel"
                )
                val_data = maybe_pad_last_batch(val_data, dp_size, val_mbs)

            # Teacher top-k logits.
            use_ipc = master_config["distillation"].get("use_ipc", True)
            topk_k = master_config["distillation"]["topk_logits_k"]

            teacher_policy.prepare_for_lp_inference()
            if use_ipc:
                teacher_logits = teacher_policy.train(
                    val_data,
                    None,
                    eval_mode=True,
                    is_teacher=True,
                    topk_logits=topk_k,
                    gbs=val_data.size,
                    mbs=val_mbs,
                )
            else:
                teacher_topk = teacher_policy.get_topk_logits(val_data, k=topk_k)
                teacher_topk_logprobs, _ = _ensure_topk_logprobs_for_non_ipc(
                    teacher_topk["topk_logits"]
                )
                val_data["teacher_topk_logits"] = teacher_topk_logprobs
                val_data["teacher_topk_indices"] = teacher_topk["topk_indices"]
                del teacher_topk
            teacher_policy.offload_after_refit()

            # Student validation loss (eval mode, no gradient updates).
            student_policy.prepare_for_training()
            if use_ipc:
                val_results = student_policy.train(
                    val_data,
                    loss_fn,
                    eval_mode=True,
                    gbs=val_data.size,
                    mbs=val_mbs,
                    teacher_logits=teacher_logits,
                    use_teacher_ipc_loss_postprocessor=isinstance(
                        loss_fn, CrossTokenizerDistillationLossFn
                    ),
                )
                del teacher_logits
            else:
                val_results = student_policy.train(
                    val_data,
                    loss_fn,
                    eval_mode=True,
                    gbs=val_data.size,
                    mbs=val_mbs,
                )

            if len(val_results["all_mb_metrics"]) == 0:
                warnings.warn(
                    "No validation metrics were collected for this batch."
                    " This is likely because there were no valid samples."
                )
            else:
                num_valid_tokens = (
                    val_data["sample_mask"].unsqueeze(-1) * val_data["token_mask"]
                ).sum()
                val_metrics["val_loss"] += float(val_results["loss"]) * num_valid_tokens
                sum_num_valid_tokens += num_valid_tokens

            if val_batches > 0 and batch_idx >= val_batches - 1:
                break

        if sum_num_valid_tokens > 0:
            val_metrics["val_loss"] /= sum_num_valid_tokens
        else:
            warnings.warn(
                "No validation metrics were collected."
                " This is likely because there were no valid samples in the validation set."
            )

        student_policy.prepare_for_training()

    timing_metrics = timer.get_timing_metrics(reduction_op="sum")
    validation_time = timing_metrics.get("total_validation_time", 0)

    if sum_num_valid_tokens > 0:
        print("\n📊 Validation Results:")
        print(f"    • Validation loss: {val_metrics['val_loss']:.4f}")
        print("\n  ⏱️  Validation Timing:")
        print(f"    • Total validation time: {validation_time:.2f}s")

    timer.reset()

    return val_metrics, timing_metrics
