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

import logging
import os
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer

try:
    from numba import njit

    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

##### define the format of projection matrix
##### go for dense as it is easier to train, and gradient is only computed for top_k
##### we will not have "A_to_B" and "B_to_A" to simplify, no bidirectional projection

#### skip backprop if accuracy of alignment is <0.9

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


if _NUMBA_AVAILABLE:

    @njit(cache=True)
    def _dp_core_numba(
        ids1,
        ids2,
        joined1,
        joined2,
        n1,
        n2,
        exact_match_score,
        gap_penalty,
        comb_mul,
        max_comb_len,
    ):
        """Numba-accelerated DP core for token alignment.

        Uses the same algorithm as align_tokens_with_combinations_numpy but
        with integer ID comparisons instead of Python string operations.

        Trace codes: 0=start, 1=diag, 2=up, 3=left,
                     10+k = comb_s1_over_s2_k, 20+k = comb_s2_over_s1_k
        """
        INVALID = np.int64(-1)
        dp = np.zeros((n1 + 1, n2 + 1), dtype=np.float32)
        trace = np.zeros((n1 + 1, n2 + 1), dtype=np.int32)

        for i in range(1, n1 + 1):
            dp[i, 0] = dp[i - 1, 0] + gap_penalty
            trace[i, 0] = 2
        for j in range(1, n2 + 1):
            dp[0, j] = dp[0, j - 1] + gap_penalty
            trace[0, j] = 3

        for i in range(1, n1 + 1):
            id_i = ids1[i - 1]
            for j in range(1, n2 + 1):
                id_j = ids2[j - 1]

                if id_i == id_j:
                    best = dp[i - 1, j - 1] + exact_match_score
                else:
                    best = dp[i - 1, j - 1] - exact_match_score
                best_m = np.int32(1)

                s = dp[i - 1, j] + gap_penalty
                if s > best:
                    best = s
                    best_m = np.int32(2)

                s = dp[i, j - 1] + gap_penalty
                if s > best:
                    best = s
                    best_m = np.int32(3)

                k_max_s2 = min(j, max_comb_len)
                for k in range(2, k_max_s2 + 1):
                    jid = joined2[j, k]
                    if jid != INVALID and id_i == jid:
                        s = dp[i - 1, j - k] + comb_mul * np.float32(k)
                        if s > best:
                            best = s
                            best_m = np.int32(10 + k)

                k_max_s1 = min(i, max_comb_len)
                for k in range(2, k_max_s1 + 1):
                    jid = joined1[i, k]
                    if jid != INVALID and id_j == jid:
                        s = dp[i - k, j - 1] + comb_mul * np.float32(k)
                        if s > best:
                            best = s
                            best_m = np.int32(20 + k)

                dp[i, j] = best
                trace[i, j] = best_m

        return dp, trace
else:
    _dp_core_numba = None


class TokenAligner(nn.Module):
    def __init__(
        self,
        max_comb_len=4,
        teacher_tokenizer_name=None,
        student_tokenizer_name=None,
        init_hf_tokenizers=True,
        track_rules=False,
        projection_matrix_multiplier=1.0,
        enable_scale_trick=None,
    ):
        super().__init__()
        self.teacher_tokenizer_name = teacher_tokenizer_name
        self.student_tokenizer_name = student_tokenizer_name
        self.track_rules = track_rules  # Control whether to track alignment rules
        self.projection_matrix_multiplier = (
            projection_matrix_multiplier  # Multiplier for projection matrix scaling
        )
        self.enable_scale_trick = (
            enable_scale_trick  # Override for SCALE_TRICK (if None, use default False)
        )

        if init_hf_tokenizers:
            self.teacher_tokenizer = AutoTokenizer.from_pretrained(
                teacher_tokenizer_name
            )
            self.student_tokenizer = AutoTokenizer.from_pretrained(
                student_tokenizer_name
            )
            if self.teacher_tokenizer.pad_token is None:
                self.teacher_tokenizer.pad_token = self.teacher_tokenizer.eos_token
            if self.student_tokenizer.pad_token is None:
                self.student_tokenizer.pad_token = self.student_tokenizer.eos_token
        else:
            self.teacher_tokenizer = None
            self.student_tokenizer = None

        self.forward_rules = set()  # (seq1_tuple, seq2_tuple)
        self.reverse_rules = set()  # (seq2_tuple, seq1_tuple)
        self.max_combination_len = max_comb_len
        self.sparse_transformation_matrix = None
        # Cached CSR for dense top-k projection (built from indices/values) to avoid scatter path
        self._dense_proj_csr = None
        self._dense_proj_csr_device = None

        # Precomputed canonical ID maps (built by precompute_canonical_maps)
        self._student_canon_map = None
        self._teacher_canon_map = None
        self._canon_id_to_str = None

    def precompute_canonical_maps(self):
        """Build token_id → canonical_string lookup tables for both tokenizers.

        Call once at startup. After this, align_fast() can skip
        convert_ids_to_tokens and _canonicalize_sequence entirely.
        """
        import time as _time

        _t0 = _time.time()

        canon_str_to_id: dict[str, int] = {}
        next_id = [0]

        def _get_canon_id(s: str) -> int:
            cid = canon_str_to_id.get(s)
            if cid is None:
                cid = next_id[0]
                canon_str_to_id[s] = cid
                next_id[0] += 1
            return cid

        student_vocab_size = len(self.student_tokenizer)
        teacher_vocab_size = len(self.teacher_tokenizer)

        student_map = np.zeros(student_vocab_size, dtype=np.int64)
        for tid in range(student_vocab_size):
            tok = self.student_tokenizer.convert_ids_to_tokens(tid)
            canon = self._canonical_token(tok)
            student_map[tid] = _get_canon_id(canon)

        teacher_map = np.zeros(teacher_vocab_size, dtype=np.int64)
        for tid in range(teacher_vocab_size):
            tok = self.teacher_tokenizer.convert_ids_to_tokens(tid)
            canon = self._canonical_token(tok)
            teacher_map[tid] = _get_canon_id(canon)

        self._student_canon_map = student_map
        self._teacher_canon_map = teacher_map
        self._canon_id_to_str = {v: k for k, v in canon_str_to_id.items()}

        _t1 = _time.time()
        print(
            f"  [TokenAligner] Precomputed canonical maps in {_t1 - _t0:.2f}s "
            f"(student_vocab={student_vocab_size}, teacher_vocab={teacher_vocab_size}, "
            f"unique_canonical={len(canon_str_to_id)})",
            flush=True,
        )

    def align_fast(
        self,
        student_ids,
        teacher_ids,
        exact_match_score=3,
        combination_score_multiplier=1.5,
        gap_penalty=-1.5,
        chunk_size=128,
        post_process=True,
        anchor_lengths=[
            3,
        ],
        ignore_leading_char_diff=False,
    ):
        """Fast alignment using precomputed canonical ID maps.

        Skips convert_ids_to_tokens and _canonicalize_sequence by looking up
        canonical strings directly from token IDs via precomputed numpy arrays.
        Falls back to regular align() if precomputed maps are not available.
        """
        if self._student_canon_map is None:
            return self.align(
                student_ids,
                teacher_ids,
                exact_match_score=exact_match_score,
                combination_score_multiplier=combination_score_multiplier,
                gap_penalty=gap_penalty,
                chunk_size=chunk_size,
                post_process=post_process,
                anchor_lengths=anchor_lengths,
                ignore_leading_char_diff=ignore_leading_char_diff,
            )

        if isinstance(student_ids, torch.Tensor):
            student_ids = student_ids.cpu().numpy()
        if isinstance(teacher_ids, torch.Tensor):
            teacher_ids = teacher_ids.cpu().numpy()

        if student_ids.ndim == 1:
            student_ids = student_ids[np.newaxis, :]
            teacher_ids = teacher_ids[np.newaxis, :]

        import time as _time

        _t_lookup_total = 0.0
        _t_anchors_dp_total = 0.0
        _t_postprocess_total = 0.0
        _t_mask_total = 0.0

        all_aligned_pairs = []
        for i in range(student_ids.shape[0]):
            s_ids = student_ids[i]
            t_ids = teacher_ids[i]

            _tl0 = _time.time()
            s_canon_strs = [
                self._canon_id_to_str[self._student_canon_map[tid]] for tid in s_ids
            ]
            t_canon_strs = [
                self._canon_id_to_str[self._teacher_canon_map[tid]] for tid in t_ids
            ]
            _tl1 = _time.time()
            _t_lookup_total += _tl1 - _tl0

            align_kwargs = {
                "exact_match_score": exact_match_score,
                "combination_score_multiplier": combination_score_multiplier,
                "gap_penalty": gap_penalty,
                "max_combination_len": self.max_combination_len,
                "ignore_leading_char_diff": False,
                "chunk_size": chunk_size,
                "anchor_lengths": anchor_lengths,
            }

            aligned_pairs, _ = self._align_with_anchors(
                s_canon_strs, t_canon_strs, **align_kwargs
            )
            _tl2 = _time.time()
            _t_anchors_dp_total += _tl2 - _tl1

            if post_process:
                aligned_pairs = self.post_process_alignment_optimized(
                    aligned_pairs,
                    ignore_leading_char_diff=ignore_leading_char_diff,
                    exact_match_score=exact_match_score,
                    combination_score_multiplier=combination_score_multiplier,
                    gap_penalty=gap_penalty,
                    max_combination_len=self.max_combination_len,
                )
            _tl3 = _time.time()
            _t_postprocess_total += _tl3 - _tl2

            mask = self.get_alignment_mask(
                aligned_pairs,
                use_canonicalization=True,
                ignore_leading_char_diff=ignore_leading_char_diff,
            )
            aligned_pairs = [
                (s1_tokens, s2_tokens, s1_start, s1_end, s2_start, s2_end, mask_value)
                for (
                    s1_tokens,
                    s2_tokens,
                    s1_start,
                    s1_end,
                    s2_start,
                    s2_end,
                ), mask_value in zip(aligned_pairs, mask)
            ]
            _tl4 = _time.time()
            _t_mask_total += _tl4 - _tl3

            all_aligned_pairs.append(aligned_pairs)

        n = student_ids.shape[0]
        _t_total = (
            _t_lookup_total + _t_anchors_dp_total + _t_postprocess_total + _t_mask_total
        )
        if _t_total > 0.5 or n > 1:
            print(
                f"    [align_fast timing] lookup={_t_lookup_total:.3f}s, "
                f"anchors+DP={_t_anchors_dp_total:.3f}s, "
                f"postprocess={_t_postprocess_total:.3f}s, "
                f"mask={_t_mask_total:.3f}s, "
                f"total={_t_total:.3f}s (n={n})",
                flush=True,
            )

        return all_aligned_pairs

    def _convert_student_tokens_to_teacher_tokens(
        self, student_tokens: torch.Tensor
    ) -> torch.Tensor:
        device = student_tokens.device
        dtype = student_tokens.dtype
        if student_tokens.device != "cpu":
            student_tokens = student_tokens.cpu()

        # Decode each sequence in the batch, not each individual token
        text = [
            self.student_tokenizer.decode(sequence.tolist(), skip_special_tokens=True)
            for sequence in student_tokens
        ]
        teacher_tokens = [
            self.teacher_tokenizer.encode(
                text_single,
                max_length=student_tokens.shape[1],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).squeeze(0)
            for text_single in text
        ]

        teacher_tokens = torch.stack(teacher_tokens).to(device).to(dtype)
        return teacher_tokens

    def _load_logits_projection_map(
        self,
        folder_location: str = "cross_tokenizer_data",
        file_path: str = None,
        top_k: int = 100,
        device: str = "cuda",
        use_sparse_format: bool = False,
        learnable: bool = False,
    ):
        """Load projection map for cross-tokenizer likelihood projection.

        Always creates student→teacher mapping.

        Args:
            folder_location: Directory containing the projection files
            file_path: Specific file path (overrides folder_location)
            top_k: Number of top entries per row (only used for old format)
            device: Device to load tensors on
            use_sparse_format: If True, load sparse transformation matrix format (from multi-token mapping)
                             If False, load old dense indices/values format
            learnable: If True, make the transformation matrix learnable
        """
        self.learnable = learnable
        if use_sparse_format:
            # Load sparse transformation matrix format
            if file_path is None:
                file_path = f"{folder_location}/transformation_counts_via_multitoken.pt"

            if not os.path.exists(file_path):
                raise FileNotFoundError(
                    f"Sparse transformation matrix file not found: {file_path}. Please generate it first."
                )

            # Load transformation counts dictionary
            transformation_counts = torch.load(
                file_path, map_location="cpu", weights_only=False
            )

            # Get tokenizer vocab sizes
            teacher_vocab_size = (
                len(self.teacher_tokenizer) if self.teacher_tokenizer else 151669
            )  # fallback
            student_vocab_size = (
                len(self.student_tokenizer) if self.student_tokenizer else 128256
            )  # fallback
            if 1:
                # get vocab sizes from autoconfig
                if (
                    "gemma" not in self.teacher_tokenizer_name.lower()
                    and "qwen3.5" not in self.teacher_tokenizer_name.lower()
                ):
                    teacher_vocab_size = AutoConfig.from_pretrained(
                        self.teacher_tokenizer_name
                    ).vocab_size
                else:
                    teacher_vocab_size = AutoConfig.from_pretrained(
                        self.teacher_tokenizer_name
                    ).text_config.vocab_size
                if (
                    "gemma" not in self.student_tokenizer_name.lower()
                    and "qwen3.5" not in self.student_tokenizer_name.lower()
                ):
                    student_vocab_size = AutoConfig.from_pretrained(
                        self.student_tokenizer_name
                    ).vocab_size
                else:
                    student_vocab_size = AutoConfig.from_pretrained(
                        self.student_tokenizer_name
                    ).text_config.vocab_size
                # teacher_vocab_size = AutoConfig.from_pretrained(self.teacher_tokenizer_name).vocab_size
                # student_vocab_size = AutoConfig.from_pretrained(self.student_tokenizer_name).vocab_size

            # Debug vocab sizes
            print(
                f"Teacher vocab size: {teacher_vocab_size}, Student vocab size: {student_vocab_size}"
            )

            # Convert dictionary to sparse tensor
            if transformation_counts:
                indices = list(transformation_counts.keys())
                values = list(transformation_counts.values())

                student_indices = [idx[0] for idx in indices]
                teacher_indices = [idx[1] for idx in indices]

                # Always create student→teacher mapping: rows = student vocab, cols = teacher vocab
                indices_tensor = torch.LongTensor([student_indices, teacher_indices])
                values_tensor = (
                    torch.FloatTensor(values) / self.projection_matrix_multiplier
                )
                matrix_shape = (student_vocab_size, teacher_vocab_size)

                print(
                    f"Creating sparse matrix: student→teacher ({student_vocab_size} x {teacher_vocab_size})"
                )

                sparse_transformation_matrix = torch.sparse_coo_tensor(
                    indices_tensor,
                    values_tensor,
                    (
                        student_vocab_size,
                        teacher_vocab_size,
                    ),  # student_vocab × teacher_vocab
                    device=device,
                    dtype=torch.float32,
                )

                # Optionally make the sparse matrix learnable (values only)
                if learnable:
                    self.sparse_transformation_matrix = nn.Parameter(
                        sparse_transformation_matrix.coalesce(), requires_grad=True
                    )
                else:
                    # Register as buffer for non-learnable parameters (ensures proper device handling)
                    self.register_buffer(
                        "sparse_transformation_matrix",
                        sparse_transformation_matrix.coalesce(),
                        persistent=True,
                    )

                # Store a flag for downstream code
                self.is_sparse_learnable = learnable
                print(
                    f"Loaded sparse transformation matrix with {len(transformation_counts)} entries"
                )
            else:
                # Empty transformation matrix (student→teacher)
                matrix_shape = (student_vocab_size, teacher_vocab_size)

                empty_sparse = torch.sparse_coo_tensor(
                    torch.zeros(2, 0, dtype=torch.long),
                    torch.zeros(0, dtype=torch.float32),
                    matrix_shape,
                    device=device,
                )

                if learnable:
                    self.sparse_transformation_matrix = nn.Parameter(
                        empty_sparse, requires_grad=True
                    )
                else:
                    # Register as buffer for non-learnable parameters
                    self.register_buffer(
                        "sparse_transformation_matrix", empty_sparse, persistent=True
                    )

                self.is_sparse_learnable = learnable
                print("Warning: Empty transformation matrix loaded")
        else:
            # Load old dense indices/values format
            if file_path is None:
                file_path = f"{folder_location}/projection_map_Llama-3.1_to_Qwen3_bidirectional_top_10.pt"

            if not os.path.exists(file_path):
                raise FileNotFoundError(
                    f"Projection map file not found: {file_path}. Please generate it first."
                )

            projection_data = torch.load(
                file_path, map_location="cpu", weights_only=False
            )
            # Always use B_to_A direction for student->teacher projection
            # projection_data = projection_data["B_to_A"]
            # projection_data = projection_data["A_to_B"]

            indices = projection_data["indices"]
            likelihoods = (
                projection_data["likelihoods"] / self.projection_matrix_multiplier
            )

            # Register indices as buffer (always non-learnable)
            self.register_buffer(
                "likelihood_projection_indices", indices.to(device), persistent=True
            )
            if learnable:
                if 1:
                    likelihoods = (likelihoods + 1e-10).log()

                # Use instance variable if set, otherwise use default (False)
                # scale_trick_enabled = self.enable_scale_trick if self.enable_scale_trick is not None else False

                # if scale_trick_enabled:
                #     #trick with last column being multiplier - set to -4.0
                #     likelihoods[:,-1] = likelihoods[:,-1]*0.0 - 4.0
                # lets introduce some noise to encourage training. will remove later.
                if 0:
                    likelihoods = likelihoods + torch.randn_like(likelihoods) * 1e-1
                    likelihoods = likelihoods / 2.0

                self.likelihood_projection_matrix = nn.Parameter(
                    likelihoods.to(device), requires_grad=True
                )
                # print(self.likelihood_projection_matrix[0])
                # print(self.likelihood_projection_matrix[:,-1])
                # exit()
                # add small gaussian noise to the projection matrix
                # use log form
            else:
                # Register as buffer for non-learnable parameters
                self.register_buffer(
                    "likelihood_projection_matrix",
                    likelihoods.to(device),
                    persistent=True,
                )

            print(f"Loaded dense projection map with shape {indices.shape}")
            # Invalidate cached CSR; will rebuild on first use
            self._dense_proj_csr = None
            self._dense_proj_csr_device = None

    def create_reverse_projection_matrix(self, device="cuda"):
        """Create a reverse (transposed) projection matrix for teacher→student projection.

        For sparse format: Transposes the sparse_transformation_matrix from [student_vocab, teacher_vocab]
                          to [teacher_vocab, student_vocab]
        For dense format: Builds a reverse index mapping from teacher tokens to student tokens

        This enables projecting teacher logits into student vocabulary space.
        """
        if (
            hasattr(self, "sparse_transformation_matrix")
            and self.sparse_transformation_matrix is not None
        ):
            # Transpose sparse matrix
            print("Creating reverse projection matrix (sparse format): teacher→student")
            sparse_matrix = self.sparse_transformation_matrix.coalesce()
            indices = sparse_matrix.indices()
            values = sparse_matrix.values()

            # Swap student and teacher indices (transpose)
            transposed_indices = torch.stack(
                [indices[1], indices[0]], dim=0
            )  # Swap rows: [teacher, student]
            teacher_vocab_size, student_vocab_size = (
                sparse_matrix.shape[1],
                sparse_matrix.shape[0],
            )

            reverse_sparse = torch.sparse_coo_tensor(
                transposed_indices,
                values,
                (teacher_vocab_size, student_vocab_size),
                device=device,
                dtype=torch.float32,
            ).coalesce()

            # Store as buffer or parameter based on learnability
            if self.is_sparse_learnable:
                self.reverse_sparse_transformation_matrix = nn.Parameter(
                    reverse_sparse, requires_grad=True
                )
            else:
                self.register_buffer(
                    "reverse_sparse_transformation_matrix",
                    reverse_sparse,
                    persistent=True,
                )

            print(
                f"Created reverse sparse matrix: teacher→student ({teacher_vocab_size} x {student_vocab_size})"
            )
            print(f"Reverse matrix has {len(values)} non-zero entries")

        elif (
            hasattr(self, "likelihood_projection_indices")
            and self.likelihood_projection_indices is not None
        ):
            # Build reverse index for dense format
            print("Creating reverse projection matrix (dense format): teacher→student")

            # Current: likelihood_projection_indices is [student_vocab, topk]
            # We need to build: [teacher_vocab, variable_k] where variable_k depends on how many students map to each teacher token

            student_vocab_size = self.likelihood_projection_indices.shape[0]
            topk = self.likelihood_projection_indices.shape[1]

            # Infer teacher vocab size from the max index
            teacher_vocab_size = self.likelihood_projection_indices.max().item() + 1

            # Build reverse mapping: for each teacher token, collect all (student_token, value) pairs
            from collections import defaultdict

            teacher_to_students = defaultdict(list)

            for student_idx in range(student_vocab_size):
                for k in range(topk):
                    teacher_idx = self.likelihood_projection_indices[
                        student_idx, k
                    ].item()
                    if hasattr(self, "likelihood_projection_matrix"):
                        value = self.likelihood_projection_matrix[student_idx, k].item()
                    else:
                        value = 1.0  # Default value if no matrix

                    # Check for valid entries: teacher_idx must be valid, and value must be finite (not -inf)
                    # If matrix is in log-space, valid log-probs are finite negative values
                    # Threshold at -20 to filter out padding values like -22.3197
                    if (
                        teacher_idx >= 0 and value > -20.0
                    ):  # Skip invalid or padding entries
                        teacher_to_students[teacher_idx].append((student_idx, value))

            # Find max number of students mapping to any teacher token
            raw_max_students = (
                max([len(v) for v in teacher_to_students.values()])
                if teacher_to_students
                else 1
            )
            print(
                f"Max students mapping to any teacher token (before filtering): {raw_max_students}"
            )

            # Limit to top-K students per teacher token to avoid explosion
            # Keep only the top-K highest probability mappings per teacher
            max_students_per_teacher = min(
                topk, raw_max_students
            )  # Use same topk as forward direction
            print(
                f"Limiting to top-{max_students_per_teacher} students per teacher token"
            )

            # Sort each teacher's student list by value (descending) and keep only top-K
            for teacher_idx in teacher_to_students:
                student_list = teacher_to_students[teacher_idx]
                # Sort by value (descending - higher log-prob = less negative)
                student_list_sorted = sorted(
                    student_list, key=lambda x: x[1], reverse=True
                )
                teacher_to_students[teacher_idx] = student_list_sorted[
                    :max_students_per_teacher
                ]

            # Create dense reverse index [teacher_vocab, max_students_per_teacher]
            # Use 0 instead of -1 for padding (valid index), with very negative values to nullify contribution
            reverse_indices = torch.zeros(
                (teacher_vocab_size, max_students_per_teacher),
                dtype=torch.long,
                device=device,
            )
            # Initialize with very negative values (padding sentinel, similar to forward direction)
            reverse_values = torch.full(
                (teacher_vocab_size, max_students_per_teacher),
                -22.3197,
                dtype=torch.float32,
                device=device,
            )

            for teacher_idx, student_list in teacher_to_students.items():
                for k, (student_idx, value) in enumerate(student_list):
                    reverse_indices[teacher_idx, k] = student_idx
                    reverse_values[teacher_idx, k] = value

            print(
                f"Created reverse dense projection: teacher→student ({teacher_vocab_size} x {max_students_per_teacher})"
            )

            # Store as buffer or parameter
            self.register_buffer(
                "reverse_likelihood_projection_indices",
                reverse_indices,
                persistent=True,
            )
            if self.learnable:
                self.reverse_likelihood_projection_matrix = nn.Parameter(
                    reverse_values, requires_grad=True
                )
            else:
                self.register_buffer(
                    "reverse_likelihood_projection_matrix",
                    reverse_values,
                    persistent=True,
                )

            print(
                f"Created reverse dense projection: teacher→student ({teacher_vocab_size} x {max_students_per_teacher})"
            )
        else:
            raise ValueError(
                "No projection matrix loaded. Cannot create reverse projection."
            )

    def update_transformation_matrix_from_checkpoint(
        self, transformation_data, device="cuda"
    ):
        """Update the transformation matrix from loaded checkpoint data.

        Args:
            transformation_data: Dictionary containing 'indices' and 'likelihoods' from checkpoint
            device: Device to load the matrix on

        Returns:
            bool: True if update was successful, False if skipped due to validation errors
        """
        if transformation_data is None:
            print("No transformation matrix data to load")
            return False

        try:
            indices = transformation_data["indices"].to(device)
            likelihoods = (
                transformation_data["likelihoods"].to(device)
                / self.projection_matrix_multiplier
            )

            # Debug: print shapes and check compatibility
            max_index = indices.max().item() if indices.numel() > 0 else -1
            min_index = indices.min().item() if indices.numel() > 0 else 0
            print(
                f"Checkpoint data - indices shape: {indices.shape}, likelihoods shape: {likelihoods.shape}"
            )
            print(f"Checkpoint data - indices range: [{min_index}, {max_index}]")

            if (
                hasattr(self, "likelihood_projection_indices")
                and self.likelihood_projection_indices is not None
            ):
                current_max_index = (
                    self.likelihood_projection_indices.max().item()
                    if self.likelihood_projection_indices.numel() > 0
                    else -1
                )
                current_min_index = (
                    self.likelihood_projection_indices.min().item()
                    if self.likelihood_projection_indices.numel() > 0
                    else 0
                )
                print(
                    f"Current - indices shape: {self.likelihood_projection_indices.shape}"
                )
                print(
                    f"Current - indices range: [{current_min_index}, {current_max_index}]"
                )
            if (
                hasattr(self, "likelihood_projection_matrix")
                and self.likelihood_projection_matrix is not None
            ):
                current_matrix_shape = (
                    self.likelihood_projection_matrix.shape
                    if hasattr(self.likelihood_projection_matrix, "shape")
                    else self.likelihood_projection_matrix.data.shape
                )
                print(f"Current - matrix shape: {current_matrix_shape}")
                print(f"Current - matrix vocab size (dim 0): {current_matrix_shape[0]}")

            # Check for dimension compatibility before updating
            if (
                hasattr(self, "likelihood_projection_indices")
                and self.likelihood_projection_indices is not None
            ):
                if indices.shape != self.likelihood_projection_indices.shape:
                    print(
                        f"WARNING: Indices shape mismatch! Checkpoint: {indices.shape} vs Current: {self.likelihood_projection_indices.shape}"
                    )
                    print("Skipping transformation matrix update due to shape mismatch")
                    return False

            if (
                hasattr(self, "likelihood_projection_matrix")
                and self.likelihood_projection_matrix is not None
            ):
                current_matrix_shape = (
                    self.likelihood_projection_matrix.shape
                    if hasattr(self.likelihood_projection_matrix, "shape")
                    else self.likelihood_projection_matrix.data.shape
                )
                if likelihoods.shape != current_matrix_shape:
                    print(
                        f"WARNING: Matrix shape mismatch! Checkpoint: {likelihoods.shape} vs Current: {current_matrix_shape}"
                    )
                    print("Skipping transformation matrix update due to shape mismatch")
                    return False

            # Additional validation: check if indices contain valid teacher vocabulary indices
            # Since we project student→teacher, indices represent teacher vocabulary positions
            # Get teacher vocab size from tokenizer or current matrix
            max_teacher_vocab = None

            # Try to get teacher vocab size from tokenizer
            if (
                hasattr(self, "teacher_tokenizer")
                and self.teacher_tokenizer is not None
            ):
                max_teacher_vocab = len(self.teacher_tokenizer.get_vocab())
            elif (
                hasattr(self, "teacher_tokenizer_name")
                and self.teacher_tokenizer_name is not None
            ):
                try:
                    from transformers import AutoTokenizer

                    temp_tokenizer = AutoTokenizer.from_pretrained(
                        self.teacher_tokenizer_name, trust_remote_code=True
                    )
                    max_teacher_vocab = len(temp_tokenizer.get_vocab())
                except Exception as e:
                    print(
                        f"Warning: Could not load teacher tokenizer to check vocab size: {e}"
                    )

            # Fallback: infer from current target vocab size being used
            if (
                max_teacher_vocab is None
                and hasattr(self, "likelihood_projection_matrix")
                and self.likelihood_projection_matrix is not None
            ):
                current_matrix_shape = (
                    self.likelihood_projection_matrix.shape
                    if hasattr(self.likelihood_projection_matrix, "shape")
                    else self.likelihood_projection_matrix.data.shape
                )
                print(
                    f"Warning: Using matrix shape to infer teacher vocab size: {current_matrix_shape}"
                )
                # This is likely wrong, but we'll use it as a fallback
                max_teacher_vocab = (
                    current_matrix_shape[1]
                    if len(current_matrix_shape) > 1
                    else current_matrix_shape[0]
                )

            if max_teacher_vocab is not None:
                max_index = indices.max().item() if indices.numel() > 0 else -1
                min_index = indices.min().item() if indices.numel() > 0 else 0
                if max_index >= max_teacher_vocab or min_index < 0:
                    print(
                        f"ERROR: Index out of bounds! Indices range [{min_index}, {max_index}] but teacher vocab size is {max_teacher_vocab}"
                    )
                    print(
                        "This indicates the transformation matrix was saved with a different teacher tokenizer"
                    )
                    print(
                        f"Current teacher: {getattr(self, 'teacher_tokenizer_name', 'unknown')}"
                    )
                    print(
                        "Skipping transformation matrix update to prevent CUDA index errors"
                    )
                    return False

            if 1:
                # we store transformation matrix after softmax, so need to redo here
                # had a bug before when the very first matrix was loaded correctly, but after restarting the checkpoint it was not, here is the fix
                likelihoods = (likelihoods + 1e-10).log()
            # Check if we're using sparse format or dense format
            if (
                hasattr(self, "sparse_transformation_matrix")
                and self.sparse_transformation_matrix is not None
            ):
                print(
                    "Warning: Cannot update sparse transformation matrix from dense checkpoint data"
                )
                print("Sparse matrix updates not yet implemented")
                return False

            # Update dense format matrices
            if hasattr(self, "likelihood_projection_indices") and hasattr(
                self, "likelihood_projection_matrix"
            ):
                self.likelihood_projection_indices = indices

                # Handle both learnable and non-learnable cases
                if hasattr(self.likelihood_projection_matrix, "data"):
                    # It's a Parameter - update the data
                    self.likelihood_projection_matrix.data = likelihoods
                    print("Updated learnable transformation matrix from checkpoint")
                else:
                    # It's a regular tensor
                    self.likelihood_projection_matrix = likelihoods
                    print("Updated fixed transformation matrix from checkpoint")

                # Invalidate cached CSR; will rebuild on first use
                self._dense_proj_csr = None
                self._dense_proj_csr_device = None
                return True
            else:
                print(
                    "Warning: No existing transformation matrix structure found to update"
                )
                return False

        except Exception as e:
            print(f"Error updating transformation matrix from checkpoint: {e}")
            print("Continuing with original transformation matrix")
            return False

    def get_transformation_matrix_for_checkpoint(self):
        """Get the transformation matrix data for saving to checkpoint.

        Returns:
            Dictionary containing 'indices' and 'likelihoods' for checkpoint saving,
            or None if no transformation matrix is available.
        """
        # Check if we have dense format transformation matrix
        if hasattr(self, "likelihood_projection_indices") and hasattr(
            self, "likelihood_projection_matrix"
        ):
            if (
                self.likelihood_projection_indices is not None
                and self.likelihood_projection_matrix is not None
            ):
                print("TokenAligner.get_transformation_matrix_for_checkpoint:")
                print(
                    f"  Teacher: {getattr(self, 'teacher_tokenizer_name', 'unknown')}"
                )
                print(
                    f"  Student: {getattr(self, 'student_tokenizer_name', 'unknown')}"
                )
                print(f"  Indices shape: {self.likelihood_projection_indices.shape}")

                # Get the matrix data (handle both Parameter and Tensor cases)
                if hasattr(self.likelihood_projection_matrix, "data"):
                    # It's a Parameter - get the data
                    matrix_data = self.likelihood_projection_matrix.data
                    print(f"  Matrix type: Parameter, shape: {matrix_data.shape}")
                else:
                    # It's a regular Tensor
                    matrix_data = self.likelihood_projection_matrix
                    print(f"  Matrix type: Tensor, shape: {matrix_data.shape}")

                print(f"  Matrix dtype: {matrix_data.dtype}")
                print(
                    f"  Projection matrix multiplier: {self.projection_matrix_multiplier}"
                )

                # Apply the projection matrix multiplier for saving (reverse the division done during loading)
                likelihoods_for_save = matrix_data * self.projection_matrix_multiplier

                # Apply softmax to get probabilities for saving (reverse the log operation done during loading)
                likelihoods_for_save = torch.softmax(likelihoods_for_save, dim=-1)

                print(
                    f"  Final likelihoods shape: {likelihoods_for_save.shape}, dtype: {likelihoods_for_save.dtype}"
                )

                return {
                    "indices": self.likelihood_projection_indices.clone(),
                    "likelihoods": likelihoods_for_save.clone(),
                }

        # Check if we have sparse format transformation matrix
        if (
            hasattr(self, "sparse_transformation_matrix")
            and self.sparse_transformation_matrix is not None
        ):
            print(
                "Warning: Saving sparse transformation matrix to checkpoint not yet implemented"
            )
            print("Returning None - sparse matrix will not be saved to checkpoint")
            return None

        print("No transformation matrix available for checkpoint saving")
        return None

    # @staticmethod
    # def project_token_likelihoods(input_likelihoods, projection_map_indices, projection_map_values, target_vocab_size, device, use_sparse_format=False, sparse_matrix=None, use_vectorized=True, projection_matrix_multiplier=1.0, gpu_optimized_scatter=True):
    #     """
    #     Projects token likelihoods from a source to a target vocabulary using either dense or sparse projection.

    #     Args:
    #         input_likelihoods: Input likelihood tensor (batch_size, seq_len, source_vocab_size)
    #         projection_map_indices: Indices for dense format (source_vocab_size, top_k)
    #         projection_map_values: Values for dense format (source_vocab_size, top_k)
    #         target_vocab_size: Size of target vocabulary
    #         device: Device to run computation on
    #         use_sparse_format: If True, use sparse matrix projection
    #         sparse_matrix: Sparse transformation matrix (teacher_vocab_size, student_vocab_size)
    #         use_vectorized: If True (and use_sparse_format=False), use vectorized dense approach;
    #                        If False, use sparse CSR matrix approach (only for dense format)
    #         gpu_optimized_scatter: If True, uses a more GPU-friendly scatter operation for dense projection.
    #     """
    #     if use_sparse_format:
    #         if sparse_matrix is None:
    #             raise ValueError("sparse_matrix must be provided when use_sparse_format=True")
    #         return TokenAligner.project_token_likelihoods_sparse(input_likelihoods, sparse_matrix*projection_matrix_multiplier, device)
    #     else:
    #         return TokenAligner.project_token_likelihoods_dense(input_likelihoods, projection_map_indices, projection_map_values*projection_matrix_multiplier, target_vocab_size, device, use_vectorized, gpu_optimized_scatter=gpu_optimized_scatter, enable_scale_trick=None)

    def project_token_likelihoods_ultra_fast(
        self, input_likelihoods, sparse_matrix=None, target_vocab_reduced_indices=None
    ):
        """Ultra-fast projection optimized for sparse matrices and reduced vocabularies.

        Args:
            input_likelihoods: Input probabilities (B, S, V_student)
            sparse_matrix: Sparse transformation matrix
            target_vocab_reduced_indices: If provided, only project to these teacher vocab positions
        """
        if sparse_matrix is None:
            sparse_matrix = self.sparse_transformation_matrix

        if sparse_matrix is None:
            raise ValueError("No sparse matrix available for ultra-fast projection")

        # Cache CSR conversion for repeated use
        if not hasattr(self, "_sparse_csr_cache") or self._sparse_csr_cache.get(
            "matrix_id"
        ) != id(sparse_matrix):
            sparse_csr = sparse_matrix.to_sparse_csr()
            self._sparse_csr_cache = {
                "matrix_id": id(sparse_matrix),
                "csr_matrix": sparse_csr,
            }
        else:
            sparse_csr = self._sparse_csr_cache["csr_matrix"]

        # Ultra-fast sparse matmul with shape optimization
        bsz, seqlen, vs = input_likelihoods.shape
        x2d = input_likelihoods.reshape(bsz * seqlen, vs)

        # Use optimized sparse matmul (often faster than dense)
        out2d = torch.sparse.mm(sparse_csr.t(), x2d.t()).t()

        result = out2d.reshape(bsz, seqlen, -1)

        # If target vocab is reduced, slice early
        if target_vocab_reduced_indices is not None:
            result = result[:, :, target_vocab_reduced_indices]

        return result

    def project_token_likelihoods_instance(
        self,
        input_likelihoods,
        projection_map_indices,
        projection_map_values,
        target_vocab_size,
        device,
        use_sparse_format=False,
        sparse_matrix=None,
        use_vectorized=True,
        gpu_optimized_scatter=True,
        global_top_indices=None,
    ):
        """Instance method wrapper for project_token_likelihoods that can access instance variables.

        Args:
            global_top_indices: Optional tensor of shape (K,) containing indices of tokens to project to.
                               If provided, only projects to these K tokens instead of full target_vocab_size.
                               Results in (batch, seq, K) output instead of (batch, seq, target_vocab_size).
        """
        if use_sparse_format:
            if sparse_matrix is None:
                raise ValueError(
                    "sparse_matrix must be provided when use_sparse_format=True"
                )

            if global_top_indices is not None:
                # For sparse format with global_top_indices, project to full vocab then slice
                full_projection = TokenAligner.project_token_likelihoods_sparse(
                    input_likelihoods,
                    sparse_matrix * self.projection_matrix_multiplier,
                    device,
                )
                return full_projection[:, :, global_top_indices]
            else:
                return TokenAligner.project_token_likelihoods_sparse(
                    input_likelihoods,
                    sparse_matrix * self.projection_matrix_multiplier,
                    device,
                )
        else:
            # If projection map is learnable, fall back to dense scatter path to preserve gradients
            if getattr(projection_map_values, "requires_grad", False):
                scale_trick_enabled = (
                    self.enable_scale_trick
                    if self.enable_scale_trick is not None
                    else False
                )
                return TokenAligner.project_token_likelihoods_dense(
                    input_likelihoods,
                    projection_map_indices,
                    projection_map_values * self.projection_matrix_multiplier,
                    target_vocab_size,
                    device,
                    use_vectorized=True,
                    gpu_optimized_scatter=gpu_optimized_scatter,
                    enable_scale_trick=scale_trick_enabled,
                    global_top_indices=global_top_indices,
                )

            # Otherwise, use stateless CSR matmul (no caching) for memory efficiency
            vs = projection_map_indices.shape[0]
            top_k = projection_map_indices.shape[1]
            # Ensure device/dtype for indices/values
            idx = projection_map_indices.to(device)
            val = (projection_map_values * self.projection_matrix_multiplier).to(device)
            if val.dtype != input_likelihoods.dtype:
                val = val.to(input_likelihoods.dtype)
            # Build CSR once per call outside autograd to keep checkpoint recomputation identical
            with torch.no_grad():
                crow_indices = torch.arange(
                    0, (vs + 1) * top_k, top_k, device=device, dtype=torch.long
                )
                col_indices = idx.reshape(-1)
                values = val.reshape(-1)
                proj_csr = torch.sparse_csr_tensor(
                    crow_indices,
                    col_indices,
                    values,
                    size=(vs, target_vocab_size),
                    device=device,
                )
            # Matmul: [B, S, Vs] -> [B*S, Vs] @ [Vs, Vt] -> [B*S, Vt] -> [B, S, Vt]
            bsz, seqlen, vs_in = input_likelihoods.shape
            if vs_in != vs:
                # In case logits have extra vocab tail, slice to match
                x = input_likelihoods[:, :, :vs]
            else:
                x = input_likelihoods
            x2d = x.reshape(bsz * seqlen, vs)
            out2d = torch.matmul(x2d.to(torch.float32), proj_csr.to(torch.float32))
            out = out2d.reshape(bsz, seqlen, target_vocab_size).to(
                input_likelihoods.dtype
            )
            return out

    @staticmethod
    def project_token_likelihoods_dense(
        input_likelihoods,
        projection_map_indices,
        projection_map_values,
        target_vocab_size,
        device,
        use_vectorized=True,
        gpu_optimized_scatter=True,
        enable_scale_trick=None,
        global_top_indices=None,
    ):
        """Projects token likelihoods from a source to a target vocabulary using dense indices/values format.

        Args:
            global_top_indices: Optional tensor of shape (K,) containing indices of target tokens to project to.
                               If provided, only projects to these K tokens instead of full target_vocab_size.
                               Results in (batch, seq, K) output instead of (batch, seq, target_vocab_size).
                               MAJOR SPEEDUP: Reduces both memory and compute significantly.
        """
        batch_size, seq_len, source_vocab_size = input_likelihoods.shape
        if abs(source_vocab_size - projection_map_indices.shape[0]) > 1000:
            raise ValueError(
                f"Source vocab size of input ({source_vocab_size}) mismatches projection map size ({projection_map_indices.shape[0]})"
            )

        top_k = projection_map_indices.shape[1]
        input_likelihoods = input_likelihoods.to(device)
        if projection_map_indices.device != device:
            projection_map_indices = projection_map_indices.to(device)
        if projection_map_values.device != device:
            projection_map_values = projection_map_values.to(device)
        # do for dtype
        if projection_map_values.dtype != input_likelihoods.dtype:
            projection_map_values = projection_map_values.to(input_likelihoods.dtype)

        # else:
        #     projection_map_values = projection_map_values.to(device)

        if use_vectorized:
            # Solution 1: Efficient dense implementation using vectorized operations for small top_k
            source_vocab_size_fixed = projection_map_indices.shape[0]
            input_likelihoods_fixed = input_likelihoods[:, :, :source_vocab_size_fixed]

            # OPTIMIZATION: Use reduced vocabulary if global_top_indices provided
            if global_top_indices is not None:
                k_indices = len(global_top_indices)
                global_top_indices = global_top_indices.to(device)

                # Create mapping from full target indices to reduced indices [0, 1, 2, ..., k-1]
                full_to_reduced_map = torch.full(
                    (target_vocab_size,), -1, device=device, dtype=torch.long
                )
                full_to_reduced_map[global_top_indices] = torch.arange(
                    k_indices, device=device
                )

                # Initialize smaller output tensor - MAJOR MEMORY SAVINGS
                projected_likelihoods = torch.zeros(
                    batch_size,
                    seq_len,
                    k_indices,
                    device=device,
                    dtype=input_likelihoods.dtype,
                )
                effective_vocab_size = k_indices

                # Filter projection matrices to only include mappings to global_top_indices
                # This will be used in the scatter operations below
                use_reduced_projection = True
            else:
                # Initialize full output tensor
                projected_likelihoods = torch.zeros(
                    batch_size,
                    seq_len,
                    target_vocab_size,
                    device=device,
                    dtype=input_likelihoods.dtype,
                )
                effective_vocab_size = target_vocab_size
                use_reduced_projection = False

            # Optimized chunked processing with multiple speedup techniques
            # Use larger chunks for better amortization of fixed costs
            max_memory_mb = 200  # Increased for better performance
            # max_memory_mb = 500  # Increased for better performance
            elements_per_chunk = max_memory_mb * 1024 * 1024 // 4  # 4 bytes per float32
            chunk_size = max(
                512,
                min(
                    source_vocab_size_fixed,
                    elements_per_chunk // (batch_size * seq_len),
                ),
            )

            use_masking = False
            # Process vocabulary in optimized chunks
            for chunk_start in range(0, source_vocab_size_fixed, chunk_size):
                chunk_end = min(chunk_start + chunk_size, source_vocab_size_fixed)
                chunk_len = chunk_end - chunk_start

                input_chunk = input_likelihoods_fixed[
                    :, :, chunk_start:chunk_end
                ]  # (B, S, chunk_len)
                indices_chunk = projection_map_indices[
                    chunk_start:chunk_end, :
                ]  # (chunk_len, top_k)
                values_chunk = projection_map_values[
                    chunk_start:chunk_end, :
                ]  # (chunk_len, top_k)

                # Extract input chunk once per chunk (not per k) - major speedup
                # Determine effective top_k (exclude last column if scale trick is enabled)
                scale_trick_enabled = (
                    enable_scale_trick if enable_scale_trick is not None else False
                )
                effective_top_k = top_k - 1 if scale_trick_enabled else top_k
                # effective_top_k = 1

                if gpu_optimized_scatter:
                    if use_masking:
                        # Process one k at a time to reduce peak memory usage
                        for k in range(effective_top_k):
                            values_k = values_chunk[:, k]
                            valid_mask_k = values_k > 1e-4
                            if not valid_mask_k.any():
                                continue

                            source_indices_k = torch.nonzero(
                                valid_mask_k, as_tuple=True
                            )[0]

                            input_subset_k = input_chunk[:, :, source_indices_k]
                            values_subset_k = values_k[source_indices_k]

                            indices_k = indices_chunk[:, k]
                            target_indices_subset_k = indices_k[source_indices_k]

                            weighted_inputs_k = input_subset_k * values_subset_k.view(
                                1, 1, -1
                            )
                            expanded_target_indices_k = target_indices_subset_k.view(
                                1, 1, -1
                            ).expand(batch_size, seq_len, -1)

                            projected_likelihoods.scatter_add_(
                                2, expanded_target_indices_k, weighted_inputs_k
                            )
                    else:
                        # Compact, un-masked implementation
                        # Process only effective columns without creating intermediate tensors
                        input_expanded = input_chunk.unsqueeze(
                            -1
                        )  # (B, S, chunk_len, 1)

                        for k in range(effective_top_k):
                            values_k = values_chunk[
                                :, k : k + 1
                            ]  # (chunk_len, 1) - view, no copy
                            indices_k = indices_chunk[:, k]  # (chunk_len,)

                            if use_reduced_projection:
                                # OPTIMIZATION: Only project to indices in global_top_indices
                                # Map full indices to reduced indices and filter out invalid ones
                                reduced_indices_k = full_to_reduced_map[
                                    indices_k
                                ]  # (chunk_len,)
                                valid_mask = (
                                    reduced_indices_k != -1
                                )  # Only keep indices in global_top_indices

                                if not valid_mask.any():
                                    continue  # Skip if no valid indices in this chunk

                                # Filter to only valid entries - MAJOR COMPUTE SAVINGS
                                valid_indices = torch.nonzero(
                                    valid_mask, as_tuple=True
                                )[0]
                                reduced_indices_filtered = reduced_indices_k[
                                    valid_indices
                                ]
                                values_filtered = values_k.squeeze(-1)[
                                    valid_indices
                                ]  # (valid_count,)
                                input_filtered = input_chunk[
                                    :, :, valid_indices
                                ]  # (B, S, valid_count)

                                weighted_k = input_filtered * values_filtered.unsqueeze(
                                    0
                                ).unsqueeze(0)
                                indices_expanded = (
                                    reduced_indices_filtered.unsqueeze(0)
                                    .unsqueeze(0)
                                    .expand(batch_size, seq_len, -1)
                                )
                                projected_likelihoods.scatter_add_(
                                    2, indices_expanded, weighted_k
                                )
                            else:
                                # Standard full projection
                                weighted_k = input_expanded * values_k.unsqueeze(
                                    0
                                ).unsqueeze(0)  # (B, S, chunk_len, 1)
                                weighted_k = weighted_k.squeeze(-1)  # (B, S, chunk_len)

                                indices_expanded = (
                                    indices_k.unsqueeze(0)
                                    .unsqueeze(0)
                                    .expand(batch_size, seq_len, -1)
                                )
                                projected_likelihoods.scatter_add_(
                                    2, indices_expanded, weighted_k
                                )
                else:
                    # Original implementation with a loop over top_k
                    if True:  # For small top_k, process all k together
                        # Broadcast input: (B, S, chunk_len, 1) * (1, 1, chunk_len, top_k) -> (B, S, chunk_len, top_k)
                        weighted_inputs = input_chunk.unsqueeze(
                            -1
                        ) * values_chunk.unsqueeze(0).unsqueeze(0)

                        # Process all k simultaneously using advanced indexing
                        for k in range(effective_top_k):
                            target_indices_k = indices_chunk[:, k]  # (chunk_len,)
                            weighted_k = weighted_inputs[
                                :, :, :, k
                            ]  # (B, S, chunk_len)

                            if use_reduced_projection:
                                # OPTIMIZATION: Only project to indices in global_top_indices
                                reduced_indices_k = full_to_reduced_map[
                                    target_indices_k
                                ]  # (chunk_len,)
                                valid_mask = reduced_indices_k != -1

                                if not valid_mask.any():
                                    continue  # Skip if no valid indices

                                # Filter to only valid entries - MAJOR COMPUTE SAVINGS
                                valid_indices = torch.nonzero(
                                    valid_mask, as_tuple=True
                                )[0]
                                reduced_indices_filtered = reduced_indices_k[
                                    valid_indices
                                ]
                                weighted_filtered = weighted_k[
                                    :, :, valid_indices
                                ]  # (B, S, valid_count)

                                target_expanded = reduced_indices_filtered.view(
                                    1, 1, -1
                                ).expand(batch_size, seq_len, len(valid_indices))
                                projected_likelihoods.scatter_add_(
                                    2, target_expanded, weighted_filtered
                                )
                            else:
                                # Use optimized scatter with pre-expanded indices (avoid .expand() in loop)
                                target_expanded = target_indices_k.view(
                                    1, 1, -1
                                ).expand(batch_size, seq_len, chunk_len)
                                projected_likelihoods.scatter_add_(
                                    2, target_expanded, weighted_k
                                )

                # else:  # For larger top_k, use optimized sequential processing
                #     for k in range(top_k):
                #         target_indices_k = indices_chunk[:, k]  # (chunk_len,)
                #         target_values_k = values_chunk[:, k]    # (chunk_len,)

                #         # Skip projections marked with -1
                #         valid_mask = target_values_k > -0.00001
                #         if not valid_mask.any():
                #             continue

                #         # Only process valid projections
                #         valid_target_indices = target_indices_k[valid_mask]
                #         valid_target_values = target_values_k[valid_mask]
                #         valid_input = input_chunk[valid_mask]

                #         weighted_input = valid_input * valid_target_values.view(-1, 1, 1)

                #         # Direct scatter (simpler and often faster than index caching)
                #         target_expanded = valid_target_indices.view(1, 1, -1).expand(batch_size, seq_len, valid_target_indices.size(0))
                #         projected_likelihoods.scatter_add_(2, target_expanded, weighted_input)

            return projected_likelihoods
        else:
            # Solution 2: Sparse matrix approach (original implementation)
            source_vocab_size_fixed = projection_map_indices.shape[0]

            # Create sparse CSR matrix
            crow_indices = torch.arange(
                0,
                (source_vocab_size_fixed + 1) * top_k,
                top_k,
                device=device,
                dtype=torch.long,
            )
            col_indices = projection_map_indices.flatten()
            values = projection_map_values.flatten()

            sparse_projection_matrix = torch.sparse_csr_tensor(
                crow_indices,
                col_indices,
                values,
                size=(source_vocab_size_fixed, target_vocab_size),
                device=device,
            )

            # Apply sparse matrix multiplication
            input_likelihoods_fixed = input_likelihoods[:, :, :source_vocab_size_fixed]
            reshaped_input = input_likelihoods_fixed.reshape(
                batch_size * seq_len, source_vocab_size
            )

            projected_likelihoods_reshaped = torch.matmul(
                reshaped_input.to(torch.float32),
                sparse_projection_matrix.to(torch.float32),
            )

            return projected_likelihoods_reshaped.reshape(
                batch_size, seq_len, target_vocab_size
            ).to(input_likelihoods.dtype)

    @staticmethod
    def project_token_likelihoods_sparse(input_likelihoods, sparse_matrix, device):
        """Projects token likelihoods using a sparse transformation matrix."""
        batch_size, seq_len, source_vocab_size = input_likelihoods.shape

        # Get dimensions from sparse matrix
        matrix_input_size, matrix_output_size = sparse_matrix.shape

        if abs(source_vocab_size - matrix_input_size) > 1000:
            raise ValueError(
                f"Source vocab size of input ({source_vocab_size}) mismatches sparse matrix input size ({matrix_input_size})"
            )

        # Move to correct device and dtype
        # input_likelihoods = input_likelihoods.to(device)
        # sparse_matrix = sparse_matrix.to(device)

        # Adjust input size to match matrix dimensions
        # next 2 lines required when we used vocab length from tokenizer, now we use the size of logits
        # source_vocab_size_fixed = min(source_vocab_size, matrix_input_size)
        # input_likelihoods_fixed = input_likelihoods[:, :, :source_vocab_size_fixed]
        input_likelihoods_fixed = input_likelihoods

        # Reshape for matrix multiplication
        reshaped_input = input_likelihoods_fixed.reshape(
            batch_size * seq_len, source_vocab_size
        )

        # Project using sparse matrix multiplication
        projected_likelihoods_reshaped = torch.matmul(
            reshaped_input.to(torch.float32), sparse_matrix.to(torch.float32)
        )

        # Reshape back to original format
        return projected_likelihoods_reshaped.reshape(
            batch_size, seq_len, matrix_output_size
        ).to(input_likelihoods.dtype)

    def align(
        self,
        student_seq: Union[List[str], List[List[str]], List[int], List[List[int]]],
        teacher_seq: Union[List[str], List[List[str]], List[int], List[List[int]]],
        exact_match_score=3,
        combination_score_multiplier=1.5,
        gap_penalty=-1.5,
        ignore_leading_char_diff=False,
        chunk_size=128,
        post_process=True,
        convert_ids_to_tokens=True,
        anchor_lengths=[
            3,
        ],
        track_rules=None,
        _debug_timing=False,
    ):
        """Align two sequences (or batches) and update the internal rule set.

        Identifies translation rules between the two token sequences and
        updates ``self.forward_rules`` / ``self.reverse_rules`` accordingly.
        """
        import time as _time

        should_track_rules = (
            track_rules if track_rules is not None else self.track_rules
        )

        seq1 = student_seq
        seq2 = teacher_seq

        original_seq1_ids = None
        original_seq2_ids = None

        _t_convert = 0.0
        if isinstance(seq1, torch.Tensor):
            original_seq1_ids = seq1.cpu().tolist()
            original_seq2_ids = seq2.cpu().tolist()

            seq1 = seq1.cpu().tolist()
            seq2 = seq2.cpu().tolist()
            if convert_ids_to_tokens:
                _tc0 = _time.time()
                seq1 = [
                    self.student_tokenizer.convert_ids_to_tokens(seq1_single)
                    for seq1_single in seq1
                ]
                seq2 = [
                    self.teacher_tokenizer.convert_ids_to_tokens(seq2_single)
                    for seq2_single in seq2
                ]
                _t_convert = _time.time() - _tc0

        is_batched = (
            isinstance(seq1, list) and len(seq1) > 0 and isinstance(seq1[0], list)
        )

        _t_canon_total = 0.0
        _t_anchors_dp_total = 0.0
        _t_postprocess_total = 0.0
        _t_mask_total = 0.0

        if is_batched:
            if not (
                isinstance(seq2, list)
                and len(seq2) == len(seq1)
                and (len(seq2) == 0 or isinstance(seq2[0], list))
            ):
                raise ValueError(
                    "For batched input, seq1 and seq2 must be lists of lists with the same length."
                )

            all_aligned_pairs = []
            for i, (s1, s2) in enumerate(zip(seq1, seq2)):
                s1_ids = original_seq1_ids[i] if original_seq1_ids else None
                s2_ids = original_seq2_ids[i] if original_seq2_ids else None
                aligned_pairs, timings = self._align_single(
                    s1,
                    s2,
                    exact_match_score,
                    combination_score_multiplier,
                    gap_penalty,
                    ignore_leading_char_diff,
                    chunk_size,
                    post_process,
                    anchor_lengths,
                    s1_ids,
                    s2_ids,
                    should_track_rules,
                    _return_timings=True,
                )
                all_aligned_pairs.append(aligned_pairs)
                _t_canon_total += timings.get("canon", 0)
                _t_anchors_dp_total += timings.get("anchors_dp", 0)
                _t_postprocess_total += timings.get("postprocess", 0)
                _t_mask_total += timings.get("mask", 0)
        else:
            s1_ids = original_seq1_ids[0] if original_seq1_ids else None
            s2_ids = original_seq2_ids[0] if original_seq2_ids else None
            aligned_pairs, timings = self._align_single(
                seq1,
                seq2,
                exact_match_score,
                combination_score_multiplier,
                gap_penalty,
                ignore_leading_char_diff,
                chunk_size,
                post_process,
                anchor_lengths,
                s1_ids,
                s2_ids,
                should_track_rules,
                _return_timings=True,
            )
            all_aligned_pairs = [aligned_pairs]
            _t_canon_total += timings.get("canon", 0)
            _t_anchors_dp_total += timings.get("anchors_dp", 0)
            _t_postprocess_total += timings.get("postprocess", 0)
            _t_mask_total += timings.get("mask", 0)

        if _debug_timing:
            n = len(all_aligned_pairs)
            print(
                f"    [align timing] convert_ids={_t_convert:.3f}s, "
                f"canonicalize={_t_canon_total:.3f}s, "
                f"anchors+DP={_t_anchors_dp_total:.3f}s, "
                f"postprocess={_t_postprocess_total:.3f}s, "
                f"mask={_t_mask_total:.3f}s "
                f"(n={n})",
                flush=True,
            )

        return all_aligned_pairs

    def _align_single(
        self,
        seq1,
        seq2,
        exact_match_score=3,
        combination_score_multiplier=1.5,
        gap_penalty=-1.5,
        ignore_leading_char_diff=True,
        chunk_size=0,
        post_process=True,
        anchor_lengths=None,
        seq1_token_ids=None,
        seq2_token_ids=None,
        track_rules=None,
        _return_timings=False,
    ):
        """Align two sequences and update the internal rule set.

        Identifies translation rules between the two token sequences and
        updates ``self.forward_rules`` / ``self.reverse_rules`` accordingly.
        """
        import time as _time

        _tc0 = _time.time()
        seq1_canon = TokenAligner._canonicalize_sequence(seq1)
        seq2_canon = TokenAligner._canonicalize_sequence(seq2)
        _tc1 = _time.time()

        align_kwargs = {
            "exact_match_score": exact_match_score,
            "combination_score_multiplier": combination_score_multiplier,
            "gap_penalty": gap_penalty,
            "max_combination_len": self.max_combination_len,
            "ignore_leading_char_diff": False,
            "chunk_size": chunk_size,
            "anchor_lengths": anchor_lengths,
        }

        aligned_pairs, _ = self._align_with_anchors(
            seq1_canon, seq2_canon, **align_kwargs
        )
        _tc2 = _time.time()

        if post_process:
            aligned_pairs = self.post_process_alignment_optimized(
                aligned_pairs,
                ignore_leading_char_diff=ignore_leading_char_diff,
                exact_match_score=exact_match_score,
                combination_score_multiplier=combination_score_multiplier,
                gap_penalty=gap_penalty,
                max_combination_len=self.max_combination_len,
            )
        _tc3 = _time.time()

        mask = self.get_alignment_mask(
            aligned_pairs,
            use_canonicalization=True,
            ignore_leading_char_diff=ignore_leading_char_diff,
        )
        aligned_pairs = [
            (s1_tokens, s2_tokens, s1_start, s1_end, s2_start, s2_end, mask_value)
            for (
                s1_tokens,
                s2_tokens,
                s1_start,
                s1_end,
                s2_start,
                s2_end,
            ), mask_value in zip(aligned_pairs, mask)
        ]
        _tc4 = _time.time()

        if track_rules:
            self._update_rules(
                aligned_pairs, seq1_token_ids, seq2_token_ids, seq1, seq2
            )

        timings = {
            "canon": _tc1 - _tc0,
            "anchors_dp": _tc2 - _tc1,
            "postprocess": _tc3 - _tc2,
            "mask": _tc4 - _tc3,
        }

        if _return_timings:
            return aligned_pairs, timings
        return aligned_pairs

    def compute_accuracy(
        self, aligned_pairs, ignore_student_ids=None, ignore_teacher_ids=None
    ):
        """Compute alignment accuracy from aligned pairs with support for batched input.

        Args:
            aligned_pairs: Either a single list of aligned pairs or a list of lists (batched)
            ignore_student_ids: Set of student token IDs to ignore when computing accuracy
            ignore_teacher_ids: Set of teacher token IDs to ignore when computing accuracy

        Returns:
            For single input: Single accuracy value (float)
            For batched input: List of accuracy values (List[float])
        """
        if ignore_student_ids is None:
            ignore_student_ids = set()
        if ignore_teacher_ids is None:
            ignore_teacher_ids = set()

        def is_not_ignored(token_or_tokens, ignore_ids):
            """Check if token(s) should not be ignored in accuracy computation."""
            if isinstance(token_or_tokens, (list, tuple)):
                return all(tok not in ignore_ids for tok in token_or_tokens)
            else:
                return token_or_tokens not in ignore_ids

        def compute_single_accuracy(single_aligned_pairs):
            """Compute accuracy for a single sequence's aligned pairs."""
            if not single_aligned_pairs:
                return 0.0

            mask_values = [
                pair[6]  # is_correct mask value
                for pair in single_aligned_pairs
                if is_not_ignored(pair[0], ignore_student_ids)
                and is_not_ignored(pair[1], ignore_teacher_ids)
            ]

            if not mask_values:
                return 0.0

            return sum(mask_values) / float(len(mask_values))

        # Check if input is batched (list of lists of aligned pairs)
        # First check if it's a list and has elements
        if isinstance(aligned_pairs, list) and len(aligned_pairs) > 0:
            # Check if the first element is itself a list of tuples (indicating batched input)
            if (
                isinstance(aligned_pairs[0], list)
                and len(aligned_pairs[0]) > 0
                and (
                    isinstance(aligned_pairs[0][0], tuple)
                    or isinstance(aligned_pairs[0][0], list)
                )
                and len(aligned_pairs[0][0]) == 7
            ):
                # Batched input: compute accuracy for each batch item
                return [
                    compute_single_accuracy(batch_pairs)
                    for batch_pairs in aligned_pairs
                ]
            elif isinstance(aligned_pairs[0], tuple) and len(aligned_pairs[0]) == 7:
                # Single sequence input
                return compute_single_accuracy(aligned_pairs)

        # Empty or invalid input
        return 0.0

    def _align_with_anchors(
        self,
        seq1,
        seq2,
        anchor_lengths=[
            3,
        ],
        **kwargs,
    ):
        """Optimized alignment using unique 1-to-1 matches as anchors."""
        # CRITICAL FIX: If anchor_lengths is empty, disable anchor optimization completely
        if not anchor_lengths:
            return self._perform_dp_alignment(seq1, seq2, **kwargs)

        if anchor_lengths is None:
            anchor_lengths = [3, 2]  # Default: check 3-token, then 2-token sequences

        # Debug output
        debug = kwargs.get("debug", False)

        # 1. Find high-confidence anchor points using unique token matches.
        s1_counts = {}
        for i, t in enumerate(seq1):
            if t not in s1_counts:
                s1_counts[t] = []
            s1_counts[t].append(i)

        s2_counts = {}
        for i, t in enumerate(seq2):
            if t not in s2_counts:
                s2_counts[t] = []
            s2_counts[t].append(i)

        # Find potential anchors using consecutive token sequences
        potential_anchors = []

        # FIXED: Don't break early - collect anchors from all lengths and then choose the best
        all_potential_anchors = []

        # Check for anchors of different lengths
        for anchor_len in anchor_lengths:
            anchors_for_this_len = []

            if anchor_len == 1:
                # Handle single token anchors
                common_tokens = s1_counts.keys() & s2_counts.keys()
                for token in common_tokens:
                    if len(s1_counts[token]) == 1 and len(s2_counts[token]) == 1:
                        i = s1_counts[token][0]
                        j = s2_counts[token][0]
                        anchors_for_this_len.append((i, j, anchor_len))
            else:
                # Handle multi-token anchors
                s1_ngram_counts = {}
                for i in range(len(seq1) - anchor_len + 1):
                    ngram = tuple(seq1[i : i + anchor_len])
                    if ngram not in s1_ngram_counts:
                        s1_ngram_counts[ngram] = []
                    s1_ngram_counts[ngram].append(i)

                s2_ngram_counts = {}
                for i in range(len(seq2) - anchor_len + 1):
                    ngram = tuple(seq2[i : i + anchor_len])
                    if ngram not in s2_ngram_counts:
                        s2_ngram_counts[ngram] = []
                    s2_ngram_counts[ngram].append(i)

                # Find n-grams that appear exactly once in both sequences
                common_ngrams = s1_ngram_counts.keys() & s2_ngram_counts.keys()
                for ngram in common_ngrams:
                    if (
                        len(s1_ngram_counts[ngram]) == 1
                        and len(s2_ngram_counts[ngram]) == 1
                    ):
                        i = s1_ngram_counts[ngram][0]
                        j = s2_ngram_counts[ngram][0]
                        # ADDED: Verify the anchor is actually correct
                        if (
                            i + anchor_len <= len(seq1)
                            and j + anchor_len <= len(seq2)
                            and seq1[i : i + anchor_len] == seq2[j : j + anchor_len]
                        ):
                            anchors_for_this_len.append((i, j, anchor_len))

            all_potential_anchors.extend(anchors_for_this_len)

        # IMPROVED: Choose the best set of anchors
        # Prefer longer anchors, but if shorter anchors give better coverage, use them

        # Sort by position and filter for monotonic ordering
        all_potential_anchors.sort()

        # IMPROVED: Better anchor selection - use greedy approach to maximize coverage
        selected_anchors = []
        used_positions_seq1 = set()
        used_positions_seq2 = set()

        # Sort by anchor length (descending) then by position
        all_potential_anchors.sort(key=lambda x: (-x[2], x[0], x[1]))

        for i, j, anchor_len in all_potential_anchors:
            # Check if this anchor conflicts with already selected ones
            seq1_range = set(range(i, i + anchor_len))
            seq2_range = set(range(j, j + anchor_len))

            if not (seq1_range & used_positions_seq1) and not (
                seq2_range & used_positions_seq2
            ):
                # This anchor doesn't conflict - we can use it
                selected_anchors.append((i, j, anchor_len))
                used_positions_seq1.update(seq1_range)
                used_positions_seq2.update(seq2_range)

        # Re-sort selected anchors by position for processing
        selected_anchors.sort()

        # IMPROVED: Additional validation of selected anchors
        validated_anchors = []
        last_j = -1
        for i, j, anchor_len in selected_anchors:
            # Ensure monotonic ordering and no overlaps
            if j > last_j:
                # Double-check the anchor is valid
                if (
                    i + anchor_len <= len(seq1)
                    and j + anchor_len <= len(seq2)
                    and seq1[i : i + anchor_len] == seq2[j : j + anchor_len]
                ):
                    validated_anchors.append((i, j, anchor_len))
                    last_j = j + anchor_len - 1

        anchors = validated_anchors

        if not anchors:
            # If no anchors are found, fall back to the standard alignment.
            return self._perform_dp_alignment(seq1, seq2, **kwargs)

        # 2. Align segments between anchors.
        full_alignment = []
        last_i, last_j = 0, 0

        for anchor_idx, (i, j, anchor_len) in enumerate(anchors):
            # Align segment before the current anchor.
            seg1, seg2 = seq1[last_i:i], seq2[last_j:j]

            if seg1 or seg2:
                aligned_segment, _ = self._perform_dp_alignment(seg1, seg2, **kwargs)

                # Adjust indices to be relative to the full sequence and split exact matches.
                for (
                    s1_toks,
                    s2_toks,
                    s1_start,
                    s1_end,
                    s2_start,
                    s2_end,
                ) in aligned_segment:
                    new_s1_start = s1_start + last_i if s1_start != -1 else -1
                    new_s1_end = s1_end + last_i if s1_end != -1 else -1
                    new_s2_start = s2_start + last_j if s2_start != -1 else -1
                    new_s2_end = s2_end + last_j if s2_end != -1 else -1

                    # Split if both sides have the same tokens
                    if (
                        len(s1_toks) > 1
                        and len(s2_toks) > 1
                        and len(s1_toks) == len(s2_toks)
                        and s1_toks == s2_toks
                    ):
                        # Split into individual 1-to-1 matches
                        for k in range(len(s1_toks)):
                            full_alignment.append(
                                (
                                    [s1_toks[k]],
                                    [s2_toks[k]],
                                    new_s1_start + k,
                                    new_s1_start + k + 1,
                                    new_s2_start + k,
                                    new_s2_start + k + 1,
                                )
                            )
                    else:
                        full_alignment.append(
                            (
                                s1_toks,
                                s2_toks,
                                new_s1_start,
                                new_s1_end,
                                new_s2_start,
                                new_s2_end,
                            )
                        )

            # Add the anchor itself (consecutive tokens), also split if needed.
            anchor_seq1 = seq1[i : i + anchor_len]
            anchor_seq2 = seq2[j : j + anchor_len]

            # Split anchor into individual matches since they should be identical
            for k in range(anchor_len):
                full_alignment.append(
                    (
                        [anchor_seq1[k]],
                        [anchor_seq2[k]],
                        i + k,
                        i + k + 1,
                        j + k,
                        j + k + 1,
                    )
                )

            last_i, last_j = i + anchor_len, j + anchor_len

        # 3. Align the final segment after the last anchor.
        seg1, seg2 = seq1[last_i:], seq2[last_j:]

        if seg1 or seg2:
            aligned_segment, _ = self._perform_dp_alignment(seg1, seg2, **kwargs)

            for s1_toks, s2_toks, s1_start, s1_end, s2_start, s2_end in aligned_segment:
                new_s1_start = s1_start + last_i if s1_start != -1 else -1
                new_s1_end = s1_end + last_i if s1_end != -1 else -1
                new_s2_start = s2_start + last_j if s2_start != -1 else -1
                new_s2_end = s2_end + last_j if s2_end != -1 else -1

                # Split if both sides have the same tokens
                if (
                    len(s1_toks) > 1
                    and len(s2_toks) > 1
                    and len(s1_toks) == len(s2_toks)
                    and s1_toks == s2_toks
                ):
                    # Split into individual 1-to-1 matches
                    for k in range(len(s1_toks)):
                        full_alignment.append(
                            (
                                [s1_toks[k]],
                                [s2_toks[k]],
                                new_s1_start + k,
                                new_s1_start + k + 1,
                                new_s2_start + k,
                                new_s2_start + k + 1,
                            )
                        )
                else:
                    full_alignment.append(
                        (
                            s1_toks,
                            s2_toks,
                            new_s1_start,
                            new_s1_end,
                            new_s2_start,
                            new_s2_end,
                        )
                    )

        return full_alignment, 0  # Return 0 for score as it's not well-defined here

    def _perform_dp_alignment(self, seq1, seq2, **kwargs):
        """Helper function to run the core DP-based alignment."""
        chunk_size = kwargs.get("chunk_size", 0)
        kwargs.pop("chunk_size", None)
        kwargs.pop("anchor_lengths", None)

        if chunk_size > 0:
            return self.align_tokens_combinations_chunked(
                seq1, seq2, chunk_size=chunk_size, **kwargs
            )
        else:
            return self.align_tokens_with_combinations_numpy_jit(seq1, seq2, **kwargs)

    @staticmethod
    def _align_chunked_fast(
        seq1,
        seq2,
        exact_match_score=3,
        combination_score_multiplier=1.5,
        gap_penalty=-1.5,
        max_combination_len=4,
        ignore_leading_char_diff=False,
        chunk_size=256,
    ):
        """Chunked processing using the fast DP as the base case."""
        n1, n2 = len(seq1), len(seq2)

        if n1 <= chunk_size and n2 <= chunk_size:
            return TokenAligner.align_tokens_with_combinations_numpy_fast(
                seq1,
                seq2,
                exact_match_score,
                combination_score_multiplier,
                gap_penalty,
                max_combination_len,
                ignore_leading_char_diff,
            )

        mid1, mid2 = n1 // 2, n2 // 2

        left_aligned, left_score = TokenAligner._align_chunked_fast(
            seq1[:mid1],
            seq2[:mid2],
            exact_match_score,
            combination_score_multiplier,
            gap_penalty,
            max_combination_len,
            ignore_leading_char_diff,
            chunk_size,
        )
        right_aligned, right_score = TokenAligner._align_chunked_fast(
            seq1[mid1:],
            seq2[mid2:],
            exact_match_score,
            combination_score_multiplier,
            gap_penalty,
            max_combination_len,
            ignore_leading_char_diff,
            chunk_size,
        )

        adjusted_right = []
        for s1_tokens, s2_tokens, s1_start, s1_end, s2_start, s2_end in right_aligned:
            adjusted_right.append(
                (
                    s1_tokens,
                    s2_tokens,
                    s1_start + mid1 if s1_start >= 0 else -1,
                    s1_end + mid1 if s1_end >= 0 else -1,
                    s2_start + mid2 if s2_start >= 0 else -1,
                    s2_end + mid2 if s2_end >= 0 else -1,
                )
            )

        return left_aligned + adjusted_right, left_score + right_score

    @staticmethod
    def _canonical_token(token: str) -> str:
        """Return a canonical representation of a tokenizer token."""
        if not token:
            return token

        # 1. Normalize space prefixes first
        if token.startswith(" "):
            token = "Ġ" + token[1:]
        elif token.startswith("_"):
            token = "Ġ" + token[1:]
        elif token.startswith("▁"):  # SentencePiece-style space prefix
            token = "Ġ" + token[1:]

        # 1.5. Normalize newline and whitespace representations
        if token == "Ċ":  # GPT-style newline (used by Llama)
            token = "\n"
        elif token == "\\n":  # Escaped newline representation
            token = "\n"
        elif token == "ĉ":  # Alternative newline representation
            token = "\n"
        elif token == "Ġ\n":  # Space + newline combination
            token = "\n"
        elif "Ċ" in token:  # Handle Ċ embedded in other tokens
            token = token.replace("Ċ", "\n")
        elif "\\n" in token:  # Handle escaped newlines in compound tokens
            token = token.replace("\\n", "\n")

        # 1.6. Handle space-separated punctuation normalization
        if token == "Ġ,":  # Space + comma
            token = ","
        elif token == "Ġ.":  # Space + period
            token = "."
        elif token == "Ġ;":  # Space + semicolon
            token = ";"
        elif token == "Ġ:":  # Space + colon
            token = ":"

        # 2. Handle SentencePiece byte fallback tokens like <0x20>
        if token.startswith("<0x") and token.endswith(">") and len(token) == 6:
            try:
                byte_val = int(token[3:5], 16)
                if 0 <= byte_val <= 255:
                    return chr(byte_val)
            except ValueError:
                pass

        # 3. Normalize common Unicode encoding issues
        unicode_fixes = {
            # Spanish
            "Ã±": "ñ",
            "Ã¡": "á",
            "Ã©": "é",
            "Ã­": "í",
            "Ã³": "ó",
            "Ãº": "ú",
            "Ã": "À",
            "Ã¢": "â",
            # French / Spanish shared
            "Ã§": "ç",
            "Ã¨": "è",
            "Ã«": "ë",
            "Ã®": "î",
            "Ã´": "ô",
            "Ã¹": "ù",
            "Ã»": "û",
            "Ã¿": "ÿ",
            # Chinese (common encoding artifacts)
            "ä¸Ń": "中",
            "æĸĩ": "文",
            "æĹ¥æľ¬": "日本",
            "èªŀ": "語",
            # Russian
            "ÐłÑĥÑģ": "Рус",
            "ÑģÐºÐ¸Ð¹": "ский",
            # Arabic
            "Ø§ÙĦØ¹Ø±Ø¨ÙĬØ©": "العربية",
            # Hindi
            "à¤¹": "ह",
            "à¤¿à¤Ĥ": "हिं",
            "à¤¦à¥Ģ": "दी",
            # Mathematical symbols (common artifacts)
            "âĪĳ": "∑",
            "âĪı": "∏",
            "âĪĤ": "∂",
            "âĪĩ": "∇",
            "âĪŀ": "∞",
            "âĪļ": "√",
            "âĪ«": "∫",
            "âīĪ": "≈",
            "âīł": "≠",
            "âī¤": "≤",
            "âī¥": "≥",
        }

        # Apply Unicode fixes
        for broken, fixed in unicode_fixes.items():
            if broken in token:
                token = token.replace(broken, fixed)

        # 4. Normalize special tokens
        special_token_map = {
            "<|begin_of_text|>": "<bos>",  # Llama-style BOS token
            "<bos>": "<bos>",  # Standard BOS token
            "<pad>": "",  # Padding tokens → empty (will be handled by alignment)
            "": " ",  # End tokens
        }

        if token in special_token_map:
            return special_token_map[token]

        return token

    @staticmethod
    def _canonicalize_sequence(seq: List[str]) -> List[str]:
        """Canonicalize every token in a sequence (list of str)."""
        # First, handle multi-token encoding artifacts (before individual canonicalization)
        merged_artifacts = TokenAligner._merge_encoding_artifacts(seq)

        # Then, canonicalize individual tokens
        canon_tokens = [TokenAligner._canonical_token(tok) for tok in merged_artifacts]

        # Finally, merge consecutive byte tokens into proper Unicode characters
        return TokenAligner._merge_consecutive_bytes(canon_tokens)

    @staticmethod
    def _merge_encoding_artifacts(tokens: List[str]) -> List[str]:
        """Merge consecutive tokens that represent multi-token encoding artifacts."""
        if not tokens:
            return tokens

        # Common multi-token encoding artifacts that should be merged
        multi_token_fixes = [
            # Mathematical symbols split across tokens
            (["ĠâĪ", "ĳ"], ["Ġ∑"]),  # Sum symbol
            (["âĪ", "ĳ"], ["∑"]),  # Sum symbol (no space)
            (["ĠâĪ", "ı"], ["Ġ∏"]),  # Product symbol
            (["âĪ", "ı"], ["∏"]),  # Product symbol (no space)
            (["ĠâĪ", "Ĥ"], ["Ġ∂"]),  # Partial derivative
            (["âĪ", "Ĥ"], ["∂"]),  # Partial derivative (no space)
            (["ĠâĪ", "ĩ"], ["Ġ∇"]),  # Nabla/gradient
            (["âĪ", "ĩ"], ["∇"]),  # Nabla/gradient (no space)
            (["ĠâĪ", "ŀ"], ["Ġ∞"]),  # Infinity
            (["âĪ", "ŀ"], ["∞"]),  # Infinity (no space)
            (["ĠâĪ", "ļ"], ["Ġ√"]),  # Square root
            (["âĪ", "ļ"], ["√"]),  # Square root (no space)
            (["ĠâĪ", "«"], ["Ġ∫"]),  # Integral
            (["âĪ", "«"], ["∫"]),  # Integral (no space)
            (["Ġâī", "ł"], ["Ġ≠"]),  # Not equal
            (["âī", "ł"], ["≠"]),  # Not equal (no space)
            # Other common multi-token artifacts
            (["Ġä¸", "Ń"], ["Ġ中"]),  # Chinese character
            (["ä¸", "Ń"], ["中"]),  # Chinese character (no space)
            (["æĸ", "ĩ"], ["文"]),  # Chinese character
            (["Ġæĸ", "ĩ"], ["Ġ文"]),  # Chinese character (with space)
        ]

        result = []
        i = 0

        while i < len(tokens):
            # Check if current position matches any multi-token pattern
            matched = False

            for pattern, replacement in multi_token_fixes:
                pattern_len = len(pattern)
                if i + pattern_len <= len(tokens):
                    # Check if the tokens match the pattern
                    if tokens[i : i + pattern_len] == pattern:
                        # Replace with the fixed version
                        result.extend(replacement)
                        i += pattern_len
                        matched = True
                        break

            if not matched:
                # No pattern matched, keep the original token
                result.append(tokens[i])
                i += 1

        return result

    @staticmethod
    def _merge_consecutive_bytes(tokens: List[str]) -> List[str]:
        """Merge consecutive tokens that represent UTF-8 byte sequences."""
        if not tokens:
            return tokens

        result = []
        byte_buffer = []

        for token in tokens:
            # Check if this token represents byte(s)
            clean_token = token.lstrip("Ġ")

            # Check if all characters in the token are visual bytes
            all_chars_are_bytes = True
            if len(clean_token) == 0:
                all_chars_are_bytes = False
            else:
                for char in clean_token:
                    if TokenAligner._get_byte_value(char) is None:
                        all_chars_are_bytes = False
                        break

            if all_chars_are_bytes:
                byte_buffer.append(token)
            else:
                # Not a byte token, flush buffer first
                if byte_buffer:
                    merged = TokenAligner._try_merge_byte_buffer(byte_buffer)
                    result.extend(merged)
                    byte_buffer = []
                result.append(token)

        # Flush any remaining bytes
        if byte_buffer:
            merged = TokenAligner._try_merge_byte_buffer(byte_buffer)
            result.extend(merged)

        return result

    @staticmethod
    def _try_merge_byte_buffer(byte_tokens: List[str]) -> List[str]:
        """Try to merge a buffer of potential byte tokens into a Unicode character."""
        if not byte_tokens:
            return []

        # If only one token, just return it unless it's a multi-character byte token
        if len(byte_tokens) == 1:
            token = byte_tokens[0]
            clean_token = token.lstrip("Ġ")
            if len(clean_token) <= 1:
                return byte_tokens
            # Continue processing multi-character token

        # Extract space prefix from first token
        first_token = byte_tokens[0]
        space_prefix = "Ġ" if first_token.startswith("Ġ") else ""

        # Extract raw bytes from all characters in all tokens
        raw_bytes = []
        for token in byte_tokens:
            clean_token = token.lstrip("Ġ")
            for char in clean_token:
                byte_value = TokenAligner._get_byte_value(char)
                if byte_value is not None:
                    raw_bytes.append(byte_value)
                else:
                    # If any character is not a byte, return original tokens
                    return byte_tokens

        # Only try to merge if we have 2-4 bytes (typical for emoji/multi-byte chars)
        if len(raw_bytes) < 2 or len(raw_bytes) > 4:
            return byte_tokens

        # Try to decode as UTF-8
        try:
            decoded_text = bytes(raw_bytes).decode("utf-8")
            # Only merge if the result is a single Unicode character (like an emoji)
            if len(decoded_text) == 1 and ord(decoded_text) > 127:
                return [space_prefix + decoded_text]
            else:
                # If it's not a single special character, keep original tokens
                return byte_tokens
        except UnicodeDecodeError:
            # If decoding fails, return original tokens
            return byte_tokens

    # Common visual byte representations used by some tokenizers (especially for emojis)
    VISUAL_BYTE_MAP = {
        # Common emoji byte range (240-255)
        "ð": 240,
        "Ɩ": 241,
        "Ɨ": 242,
        "Ƙ": 243,
        "ƙ": 244,
        "ƚ": 245,
        "ƛ": 246,
        "Ɯ": 247,
        "Ɲ": 248,
        "ƞ": 249,
        "Ɵ": 250,
        "Ơ": 251,
        "ơ": 252,
        "Ƣ": 253,
        "ƣ": 254,
        "Ƥ": 255,
        # Other common byte representations (0-255 only)
        "Ł": 156,
        "ł": 157,
        "Ń": 158,
        "ń": 159,
        "ĺ": 149,
        "Ļ": 150,
        "ļ": 151,
        "Ľ": 152,
        "ľ": 153,
        "Ŀ": 154,
        "ŀ": 155,
        "Ĭ": 135,
        "ĭ": 136,
        "Į": 137,
        "į": 138,
        "İ": 139,
        "ı": 140,
        "Ĳ": 141,
        "ĳ": 142,
        "Ĵ": 143,
        "ĵ": 144,
        "Ķ": 145,
        "ķ": 146,
        "ĸ": 147,
        "Ĺ": 148,
        "ĥ": 128,
        "Ħ": 129,
        "ħ": 130,
        "Ĩ": 131,
        "ĩ": 132,
        "Ī": 133,
        "ī": 134,
        "Ģ": 162,
        "ģ": 163,
        "Ĝ": 28,
        "ĝ": 29,
        "Ğ": 30,
        "ğ": 31,
    }

    @staticmethod
    def _get_byte_value(token_char: str) -> int:
        """Get the byte value for a character, handling both direct bytes and visual representations."""
        if len(token_char) != 1:
            return None

        char_ord = ord(token_char)

        # Direct byte (0-255)
        if char_ord < 256:
            return char_ord

        # Visual byte representation
        if token_char in TokenAligner.VISUAL_BYTE_MAP:
            return TokenAligner.VISUAL_BYTE_MAP[token_char]

        return None

    @staticmethod
    def _strings_equal_flexible(s1, s2, ignore_leading_char_diff):
        if not ignore_leading_char_diff:
            return s1 == s2

        # Use our comprehensive canonicalization for robust comparison
        s1_canonical = TokenAligner._canonical_token(s1)
        s2_canonical = TokenAligner._canonical_token(s2)

        return s1_canonical == s2_canonical

    @staticmethod
    def align_tokens_with_combinations_numpy_fast(
        seq1,
        seq2,
        exact_match_score=3,
        combination_score_multiplier=1.5,
        gap_penalty=-1.5,
        max_combination_len=4,
        ignore_leading_char_diff=False,
        band_width=None,
    ):
        """DP alignment using integer token IDs, int32 trace, and optional band constraint.

        Produces the same result as ``align_tokens_with_combinations_numpy`` but
        replaces per-cell Python string comparisons with integer comparisons and
        uses a compact int32 trace array instead of a Python object array.

        When *band_width* is set (recommended for cross-tokenizer alignment where
        both sequences encode the same text), the DP is restricted to a diagonal
        band of width ``2 * band_width + 1``, reducing complexity from
        O(n1 * n2) to O(n1 * band_width).

        Note: ``ignore_leading_char_diff`` must be False.  The caller
        (``_align_single``) canonicalizes sequences before calling the DP,
        so flexible comparison is never needed here.
        """
        if ignore_leading_char_diff:
            return TokenAligner.align_tokens_with_combinations_numpy(
                seq1,
                seq2,
                exact_match_score,
                combination_score_multiplier,
                gap_penalty,
                max_combination_len,
                ignore_leading_char_diff,
            )
        n1, n2 = len(seq1), len(seq2)
        if n1 == 0 and n2 == 0:
            return [], 0.0
        if n1 == 0:
            return [
                ([], [seq2[j]], -1, -1, j, j + 1) for j in range(n2)
            ], n2 * gap_penalty
        if n2 == 0:
            return [
                ([seq1[i]], [], i, i + 1, -1, -1) for i in range(n1)
            ], n1 * gap_penalty

        token_to_id: dict[str, int] = {}
        _next = [0]

        def _id(s: str) -> int:
            tid = token_to_id.get(s)
            if tid is None:
                tid = _next[0]
                token_to_id[s] = tid
                _next[0] += 1
            return tid

        ids1 = [_id(t) for t in seq1]
        ids2 = [_id(t) for t in seq2]

        joined_ids1: dict[tuple[int, int], int] = {}
        for i in range(n1 + 1):
            for k in range(2, min(i, max_combination_len) + 1):
                joined_ids1[(i - k, i)] = _id("".join(seq1[i - k : i]))

        joined_ids2: dict[tuple[int, int], int] = {}
        for j in range(n2 + 1):
            for k in range(2, min(j, max_combination_len) + 1):
                joined_ids2[(j - k, j)] = _id("".join(seq2[j - k : j]))

        use_band = band_width is not None
        NEG_INF = np.float32(-1e9)
        dp = np.full((n1 + 1, n2 + 1), NEG_INF, dtype=np.float32)
        # Trace codes: 0=start, 1=diag, 2=up, 3=left,
        #   10+k = comb_s1_over_s2_k, 20+k = comb_s2_over_s1_k
        trace = np.zeros((n1 + 1, n2 + 1), dtype=np.int32)

        dp[0, 0] = 0.0
        for i in range(1, n1 + 1):
            dp[i, 0] = dp[i - 1, 0] + gap_penalty
            trace[i, 0] = 2
        for j in range(1, n2 + 1):
            dp[0, j] = dp[0, j - 1] + gap_penalty
            trace[0, j] = 3

        scale = n2 / max(n1, 1)
        exact = np.float32(exact_match_score)
        neg_exact = np.float32(-exact_match_score)
        gap = np.float32(gap_penalty)
        comb_mul = np.float32(combination_score_multiplier)

        for i in range(1, n1 + 1):
            if use_band:
                ej = int(i * scale)
                j_lo = max(1, ej - band_width)
                j_hi = min(n2, ej + band_width)
            else:
                j_lo = 1
                j_hi = n2

            id_i = ids1[i - 1]

            for j in range(j_lo, j_hi + 1):
                id_j = ids2[j - 1]

                best = dp[i - 1, j - 1] + (exact if id_i == id_j else neg_exact)
                best_m = 1

                s = dp[i - 1, j] + gap
                if s > best:
                    best = s
                    best_m = 2

                s = dp[i, j - 1] + gap
                if s > best:
                    best = s
                    best_m = 3

                for k in range(2, min(j + 1, max_combination_len + 1)):
                    key = (j - k, j)
                    if key in joined_ids2 and id_i == joined_ids2[key]:
                        s = dp[i - 1, j - k] + comb_mul * k
                        if s > best:
                            best = s
                            best_m = 10 + k

                for k in range(2, min(i + 1, max_combination_len + 1)):
                    key = (i - k, i)
                    if key in joined_ids1 and id_j == joined_ids1[key]:
                        s = dp[i - k, j - 1] + comb_mul * k
                        if s > best:
                            best = s
                            best_m = 20 + k

                dp[i, j] = best
                trace[i, j] = best_m

        aligned: list = []
        i, j = n1, n2
        while i > 0 or j > 0:
            m = int(trace[i, j])
            if m == 1:
                aligned.append(([seq1[i - 1]], [seq2[j - 1]], i - 1, i, j - 1, j))
                i -= 1
                j -= 1
            elif m == 2:
                aligned.append(([seq1[i - 1]], [], i - 1, i, -1, -1))
                i -= 1
            elif m == 3:
                aligned.append(([], [seq2[j - 1]], -1, -1, j - 1, j))
                j -= 1
            elif 10 <= m < 20:
                k = m - 10
                aligned.append(([seq1[i - 1]], seq2[j - k : j], i - 1, i, j - k, j))
                i -= 1
                j -= k
            elif 20 <= m < 30:
                k = m - 20
                aligned.append((seq1[i - k : i], [seq2[j - 1]], i - k, i, j - 1, j))
                i -= k
                j -= 1
            else:
                break

        aligned.reverse()
        return aligned, float(dp[n1, n2])

    @staticmethod
    def align_tokens_with_combinations_numpy(
        seq1,
        seq2,
        exact_match_score=3,
        combination_score_multiplier=1.5,
        gap_penalty=-1.5,
        max_combination_len=4,
        ignore_leading_char_diff=False,
    ):
        n1, n2 = len(seq1), len(seq2)
        dp = np.zeros((n1 + 1, n2 + 1), dtype=np.float32)
        trace = np.full((n1 + 1, n2 + 1), "", dtype=object)

        # Initialize DP edges with gap penalties
        for i in range(1, n1 + 1):
            dp[i, 0] = dp[i - 1, 0] + gap_penalty
            trace[i, 0] = "up"
        for j in range(1, n2 + 1):
            dp[0, j] = dp[0, j - 1] + gap_penalty
            trace[0, j] = "left"

        # Precompute joined substrings for all valid k-length spans
        joined_seq1 = {
            (i - k, i): "".join(seq1[i - k : i])
            for i in range(n1 + 1)
            for k in range(1, min(i, max_combination_len) + 1)
        }
        joined_seq2 = {
            (j - k, j): "".join(seq2[j - k : j])
            for j in range(n2 + 1)
            for k in range(1, min(j, max_combination_len) + 1)
        }

        # Fill DP table
        for i in range(1, n1 + 1):
            for j in range(1, n2 + 1):
                s1_val, s2_val = seq1[i - 1], seq2[j - 1]
                match_score = (
                    exact_match_score
                    if TokenAligner._strings_equal_flexible(
                        s1_val, s2_val, ignore_leading_char_diff
                    )
                    else -exact_match_score
                )
                score_diag = dp[i - 1, j - 1] + match_score
                score_up = dp[i - 1, j] + gap_penalty
                score_left = dp[i, j - 1] + gap_penalty

                max_score = score_diag
                best_move = "diag"
                if score_up > max_score:
                    max_score = score_up
                    best_move = "up"
                if score_left > max_score:
                    max_score = score_left
                    best_move = "left"

                # Check for seq1[i-1] == join(seq2[j-k:j])
                for k in range(2, min(j + 1, max_combination_len + 1)):
                    if (
                        j - k,
                        j,
                    ) in joined_seq2 and TokenAligner._strings_equal_flexible(
                        s1_val, joined_seq2[(j - k, j)], ignore_leading_char_diff
                    ):
                        comb_score = dp[i - 1, j - k] + combination_score_multiplier * k
                        if comb_score > max_score:
                            max_score = comb_score
                            best_move = f"comb_s1_over_s2_{k}"

                # Check for seq2[j-1] vs seq1[i-k:i]
                for k in range(2, min(i + 1, max_combination_len + 1)):
                    if (
                        i - k,
                        i,
                    ) in joined_seq1 and TokenAligner._strings_equal_flexible(
                        s2_val, joined_seq1[(i - k, i)], ignore_leading_char_diff
                    ):
                        comb_score = dp[i - k, j - 1] + combination_score_multiplier * k
                        if comb_score > max_score:
                            max_score = comb_score
                            best_move = f"comb_s2_over_s1_{k}"

                dp[i, j] = max_score
                trace[i, j] = best_move

        # Backtrack to extract alignment
        aligned = []
        i, j = n1, n2
        while i > 0 or j > 0:
            move = trace[i, j]
            if move == "diag":
                aligned.append(([seq1[i - 1]], [seq2[j - 1]], i - 1, i, j - 1, j))
                i -= 1
                j -= 1
            elif move == "up":
                aligned.append(([seq1[i - 1]], [], i - 1, i, -1, -1))
                i -= 1
            elif move == "left":
                aligned.append(([], [seq2[j - 1]], -1, -1, j - 1, j))
                j -= 1
            elif move.startswith("comb_s1_over_s2_"):
                k = int(move.rsplit("_", 1)[-1])
                aligned.append(([seq1[i - 1]], seq2[j - k : j], i - 1, i, j - k, j))
                i -= 1
                j -= k
            elif move.startswith("comb_s2_over_s1_"):
                k = int(move.rsplit("_", 1)[-1])
                aligned.append((seq1[i - k : i], [seq2[j - 1]], i - k, i, j - 1, j))
                i -= k
                j -= 1
            else:
                break

        aligned.reverse()
        return aligned, dp[n1, n2]

    @staticmethod
    def align_tokens_with_combinations_numpy_jit(
        seq1,
        seq2,
        exact_match_score=3,
        combination_score_multiplier=1.5,
        gap_penalty=-1.5,
        max_combination_len=4,
        ignore_leading_char_diff=False,
    ):
        """Numba-accelerated version of align_tokens_with_combinations_numpy.

        Pre-converts string tokens to integer IDs, runs the DP in a Numba
        @njit kernel, then backtracks using the original string tokens.
        Falls back to the pure-Python original when Numba is unavailable or
        when ignore_leading_char_diff is True (requires Python string logic).
        """
        if not _NUMBA_AVAILABLE or ignore_leading_char_diff:
            return TokenAligner.align_tokens_with_combinations_numpy(
                seq1,
                seq2,
                exact_match_score,
                combination_score_multiplier,
                gap_penalty,
                max_combination_len,
                ignore_leading_char_diff,
            )

        n1, n2 = len(seq1), len(seq2)
        if n1 == 0 and n2 == 0:
            return [], 0.0
        if n1 == 0:
            return [
                ([], [seq2[j]], -1, -1, j, j + 1) for j in range(n2)
            ], n2 * gap_penalty
        if n2 == 0:
            return [
                ([seq1[i]], [], i, i + 1, -1, -1) for i in range(n1)
            ], n1 * gap_penalty

        token_to_id: dict[str, int] = {}
        _next_id = [0]

        def _get_id(s: str) -> int:
            tid = token_to_id.get(s)
            if tid is None:
                tid = _next_id[0]
                token_to_id[s] = tid
                _next_id[0] += 1
            return tid

        ids1 = np.array([_get_id(t) for t in seq1], dtype=np.int64)
        ids2 = np.array([_get_id(t) for t in seq2], dtype=np.int64)

        INVALID = np.int64(-1)
        joined1 = np.full((n1 + 1, max_combination_len + 1), INVALID, dtype=np.int64)
        for i in range(n1 + 1):
            for k in range(2, min(i, max_combination_len) + 1):
                joined1[i, k] = _get_id("".join(seq1[i - k : i]))

        joined2 = np.full((n2 + 1, max_combination_len + 1), INVALID, dtype=np.int64)
        for j in range(n2 + 1):
            for k in range(2, min(j, max_combination_len) + 1):
                joined2[j, k] = _get_id("".join(seq2[j - k : j]))

        dp, trace = _dp_core_numba(
            ids1,
            ids2,
            joined1,
            joined2,
            n1,
            n2,
            np.float32(exact_match_score),
            np.float32(gap_penalty),
            np.float32(combination_score_multiplier),
            max_combination_len,
        )

        aligned = []
        i, j = n1, n2
        while i > 0 or j > 0:
            m = trace[i, j]
            if m == 1:
                aligned.append(([seq1[i - 1]], [seq2[j - 1]], i - 1, i, j - 1, j))
                i -= 1
                j -= 1
            elif m == 2:
                aligned.append(([seq1[i - 1]], [], i - 1, i, -1, -1))
                i -= 1
            elif m == 3:
                aligned.append(([], [seq2[j - 1]], -1, -1, j - 1, j))
                j -= 1
            elif 10 <= m < 20:
                k = m - 10
                aligned.append(([seq1[i - 1]], seq2[j - k : j], i - 1, i, j - k, j))
                i -= 1
                j -= k
            elif 20 <= m < 30:
                k = m - 20
                aligned.append((seq1[i - k : i], [seq2[j - 1]], i - k, i, j - 1, j))
                i -= k
                j -= 1
            else:
                break

        aligned.reverse()
        return aligned, float(dp[n1, n2])

    @staticmethod
    def align_tokens_combinations_chunked(
        seq1: List[str],
        seq2: List[str],
        exact_match_score: float = 3.0,
        combination_score_multiplier: float = 1.5,
        gap_penalty: float = -1.5,
        max_combination_len: int = 4,
        ignore_leading_char_diff: bool = False,
        chunk_size: int = 256,
    ):
        """Chunked processing for very large sequences."""
        n1, n2 = len(seq1), len(seq2)

        # If sequences are small enough, use regular algorithm
        if n1 <= chunk_size and n2 <= chunk_size:
            return TokenAligner.align_tokens_with_combinations_numpy_jit(
                seq1,
                seq2,
                exact_match_score,
                combination_score_multiplier,
                gap_penalty,
                max_combination_len,
                ignore_leading_char_diff,
            )

        # For very large sequences, use divide-and-conquer approach
        if n1 > chunk_size or n2 > chunk_size:
            # Find approximate midpoint alignment using simplified algorithm
            mid1, mid2 = n1 // 2, n2 // 2

            # Recursively align left and right parts
            left_aligned, left_score = TokenAligner.align_tokens_combinations_chunked(
                seq1[:mid1],
                seq2[:mid2],
                exact_match_score,
                combination_score_multiplier,
                gap_penalty,
                max_combination_len,
                ignore_leading_char_diff,
                chunk_size=chunk_size,
            )

            right_aligned, right_score = TokenAligner.align_tokens_combinations_chunked(
                seq1[mid1:],
                seq2[mid2:],
                exact_match_score,
                combination_score_multiplier,
                gap_penalty,
                max_combination_len,
                ignore_leading_char_diff,
                chunk_size=chunk_size,
            )

            # Adjust indices for right part
            adjusted_right = []
            for (
                s1_tokens,
                s2_tokens,
                s1_start,
                s1_end,
                s2_start,
                s2_end,
            ) in right_aligned:
                new_s1_start = s1_start + mid1 if s1_start >= 0 else -1
                new_s1_end = s1_end + mid1 if s1_end >= 0 else -1
                new_s2_start = s2_start + mid2 if s2_start >= 0 else -1
                new_s2_end = s2_end + mid2 if s2_end >= 0 else -1
                adjusted_right.append(
                    (
                        s1_tokens,
                        s2_tokens,
                        new_s1_start,
                        new_s1_end,
                        new_s2_start,
                        new_s2_end,
                    )
                )

            # Combine results
            combined_aligned = left_aligned + adjusted_right
            combined_score = left_score + right_score

            return combined_aligned, combined_score

        # Fallback to regular algorithm
        return TokenAligner.align_tokens_with_combinations_numpy_jit(
            seq1,
            seq2,
            exact_match_score,
            combination_score_multiplier,
            gap_penalty,
            max_combination_len,
            ignore_leading_char_diff,
        )

    # @staticmethod
    # def post_process_alignment_optimized(
    #     aligned_pairs: List,
    #     ignore_leading_char_diff: bool = False,
    #     exact_match_score: float = 3.0,
    #     combination_score_multiplier: float = 1.5,
    #     gap_penalty: float = -1.5,
    #     max_combination_len: int = 4
    # ) -> List:
    #     """
    #     Optimized version of post_process_alignment with better performance.
    #     """
    #     if not aligned_pairs:
    #         return []

    #     # Precompute joined strings for all pairs to avoid repeated concatenation
    #     # Use canonicalization for robust comparison
    #     pair_strings = []
    #     for i, (s1_tokens, s2_tokens, *rest) in enumerate(aligned_pairs):
    #         # Canonicalize individual tokens before joining for better matching
    #         s1_canonical_tokens = [TokenAligner._canonical_token(tok) for tok in s1_tokens] if s1_tokens else []
    #         s2_canonical_tokens = [TokenAligner._canonical_token(tok) for tok in s2_tokens] if s2_tokens else []
    #         s1_str = "".join(s1_canonical_tokens)
    #         s2_str = "".join(s2_canonical_tokens)
    #         is_match = TokenAligner._strings_equal_flexible(s1_str, s2_str, ignore_leading_char_diff)
    #         pair_strings.append((s1_str, s2_str, is_match))

    #     processed_pairs = []
    #     alignment_cache = {}  # Cache for repeated alignment patterns
    #     i = 0

    #     while i < len(aligned_pairs):
    #         s1_tokens, s2_tokens, *_ = aligned_pairs[i]

    #         # Handle coarse alignments that can be split (optimized)
    #         if len(s1_tokens) > 1 and len(s1_tokens) == len(s2_tokens) and s1_tokens == s2_tokens:
    #             s1_start, s1_end, s2_start, s2_end = aligned_pairs[i][2:6]
    #             # Vectorized creation of split pairs
    #             for k in range(len(s1_tokens)):
    #                 processed_pairs.append(
    #                     ([s1_tokens[k]], [s2_tokens[k]],
    #                      s1_start + k, s1_start + k + 1,
    #                      s2_start + k, s2_start + k + 1)
    #                 )
    #             i += 1
    #             continue

    #         # Find bad regions more efficiently using precomputed strings
    #         start_bad_region = -1
    #         for j in range(i, len(aligned_pairs)):
    #             if not pair_strings[j][2]:  # is_match is False
    #                 start_bad_region = j
    #                 break

    #         if start_bad_region == -1:
    #             # No more bad regions - add remaining pairs and exit
    #             processed_pairs.extend(aligned_pairs[i:])
    #             break

    #         # Add good pairs before bad region
    #         processed_pairs.extend(aligned_pairs[i:start_bad_region])

    #         # Optimized chunk processing with early termination
    #         found_fix = False
    #         max_chunk_size = min(10, len(aligned_pairs) - start_bad_region)  # Limit search space

    #         for chunk_size in range(2, max_chunk_size + 1):
    #             chunk = aligned_pairs[start_bad_region : start_bad_region + chunk_size]

    #             # Efficient token extraction using list comprehension
    #             chunk_s1_tokens = []
    #             chunk_s2_tokens = []
    #             s1_indices = []
    #             s2_indices = []

    #             for s1_toks, s2_toks, s1_start, s1_end, s2_start, s2_end in chunk:
    #                 chunk_s1_tokens.extend(s1_toks)
    #                 chunk_s2_tokens.extend(s2_toks)
    #                 if s1_toks:
    #                     s1_indices.extend([s1_start, s1_end])
    #                 if s2_toks:
    #                     s2_indices.extend([s2_start, s2_end])

    #             # Quick string comparison using canonicalization
    #             chunk_s1_canonical = [TokenAligner._canonical_token(tok) for tok in chunk_s1_tokens]
    #             chunk_s2_canonical = [TokenAligner._canonical_token(tok) for tok in chunk_s2_tokens]
    #             chunk_s1_str = "".join(chunk_s1_canonical)
    #             chunk_s2_str = "".join(chunk_s2_canonical)

    #             if not TokenAligner._strings_equal_flexible(chunk_s1_str, chunk_s2_str, ignore_leading_char_diff):
    #                 continue

    #             # Create cache key for alignment
    #             cache_key = (tuple(chunk_s1_tokens), tuple(chunk_s2_tokens))

    #             if cache_key in alignment_cache:
    #                 sub_aligned_pairs, realign_is_perfect = alignment_cache[cache_key]
    #             else:
    #                 # Perform alignment
    #                 sub_aligned_pairs, _ = TokenAligner.align_tokens_with_combinations_numpy(
    #                     chunk_s1_tokens,
    #                     chunk_s2_tokens,
    #                     exact_match_score=exact_match_score,
    #                     combination_score_multiplier=combination_score_multiplier,
    #                     gap_penalty=gap_penalty,
    #                     max_combination_len=max_combination_len,
    #                     ignore_leading_char_diff=ignore_leading_char_diff
    #                 )

    #                 # Check if re-alignment was successful using canonicalization
    #                 realign_is_perfect = all(
    #                     TokenAligner._strings_equal_flexible(
    #                         "".join([TokenAligner._canonical_token(tok) for tok in p[0]]),
    #                         "".join([TokenAligner._canonical_token(tok) for tok in p[1]]),
    #                         ignore_leading_char_diff
    #                     )
    #                     for p in sub_aligned_pairs
    #                 )

    #                 # Cache the result
    #                 alignment_cache[cache_key] = (sub_aligned_pairs, realign_is_perfect)

    #             # Vectorized index calculations
    #             s1_chunk_start = min(s1_indices[::2]) if s1_indices else -1
    #             s2_chunk_start = min(s2_indices[::2]) if s2_indices else -1

    #             if realign_is_perfect:
    #                 # Add granular aligned pairs
    #                 for s1_toks, s2_toks, s1_start, s1_end, s2_start, s2_end, *_ in sub_aligned_pairs:
    #                     new_s1_start = s1_chunk_start + s1_start if s1_start != -1 else -1
    #                     new_s1_end = s1_chunk_start + s1_end if s1_end != -1 else -1
    #                     new_s2_start = s2_chunk_start + s2_start if s2_start != -1 else -1
    #                     new_s2_end = s2_chunk_start + s2_end if s2_end != -1 else -1
    #                     processed_pairs.append((s1_toks, s2_toks, new_s1_start, new_s1_end, new_s2_start, new_s2_end))
    #             else:
    #                 # Create merged pair
    #                 s1_chunk_end = max(s1_indices[1::2]) if s1_indices else -1
    #                 s2_chunk_end = max(s2_indices[1::2]) if s2_indices else -1
    #                 merged_pair = (chunk_s1_tokens, chunk_s2_tokens, s1_chunk_start, s1_chunk_end, s2_chunk_start, s2_chunk_end)
    #                 processed_pairs.append(merged_pair)

    #             i = start_bad_region + chunk_size
    #             found_fix = True
    #             break

    #         if not found_fix:
    #             processed_pairs.append(aligned_pairs[start_bad_region])
    #             i = start_bad_region + 1

    #     return processed_pairs

    @staticmethod
    def _combine_consecutive_misaligned_tokens(
        aligned_pairs: List, pair_strings: List, end_mismatch_threshold: float = 0.2
    ) -> List:
        """Combine consecutive misaligned tokens into single chunks to improve alignment.

        This addresses cases where multiple tokens are individually misaligned but
        collectively represent the same content. Avoids combining tokens near the
        end of sequences that might be misaligned due to length differences.

        Args:
            aligned_pairs: List of alignment pairs
            pair_strings: Precomputed string representations and match status
            end_mismatch_threshold: Fraction of sequence from end to avoid chunking

        Returns:
            Modified aligned_pairs with consecutive misaligned tokens combined
        """
        if not aligned_pairs or len(aligned_pairs) < 2:
            return aligned_pairs

        # Calculate the boundary for avoiding end mismatches
        sequence_length = len(aligned_pairs)
        end_boundary = int(sequence_length * (1 - end_mismatch_threshold))

        processed_pairs = []
        i = 0

        while i < len(aligned_pairs):
            # Check if current pair is misaligned and not near the end
            if (
                i < end_boundary
                and not pair_strings[i][2]  # Current pair is misaligned
                and i + 1 < len(aligned_pairs)
            ):  # Not the last pair
                # Find consecutive misaligned pairs
                consecutive_misaligned = [i]
                j = i + 1

                # Look ahead for more consecutive misaligned pairs (up to end boundary)
                while (
                    j < end_boundary
                    and j < len(aligned_pairs)
                    and not pair_strings[j][2]
                ):  # Next pair is also misaligned
                    consecutive_misaligned.append(j)
                    j += 1

                # Only combine if we have multiple consecutive misaligned pairs
                if len(consecutive_misaligned) >= 2:
                    # Combine all consecutive misaligned pairs into one chunk
                    combined_s1_tokens = []
                    combined_s2_tokens = []
                    s1_indices = []
                    s2_indices = []

                    for idx in consecutive_misaligned:
                        (
                            s1_tokens,
                            s2_tokens,
                            s1_start,
                            s1_end,
                            s2_start,
                            s2_end,
                            *rest,
                        ) = aligned_pairs[idx]
                        combined_s1_tokens.extend(s1_tokens)
                        combined_s2_tokens.extend(s2_tokens)

                        if s1_tokens and s1_start != -1:
                            s1_indices.extend([s1_start, s1_end])
                        if s2_tokens and s2_start != -1:
                            s2_indices.extend([s2_start, s2_end])

                    # Calculate combined indices
                    combined_s1_start = min(s1_indices[::2]) if s1_indices else -1
                    combined_s1_end = max(s1_indices[1::2]) if s1_indices else -1
                    combined_s2_start = min(s2_indices[::2]) if s2_indices else -1
                    combined_s2_end = max(s2_indices[1::2]) if s2_indices else -1

                    # Create combined pair
                    combined_pair = (
                        combined_s1_tokens,
                        combined_s2_tokens,
                        combined_s1_start,
                        combined_s1_end,
                        combined_s2_start,
                        combined_s2_end,
                    )

                    processed_pairs.append(combined_pair)
                    i = j  # Skip to after the combined region
                else:
                    # Only one misaligned pair, keep as is
                    processed_pairs.append(aligned_pairs[i])
                    i += 1
            else:
                # Current pair is aligned or near the end, keep as is
                processed_pairs.append(aligned_pairs[i])
                i += 1

        return processed_pairs

    @staticmethod
    def post_process_alignment_optimized(
        aligned_pairs: List,
        ignore_leading_char_diff: bool = False,
        exact_match_score: float = 3.0,
        combination_score_multiplier: float = 1.5,
        gap_penalty: float = -1.5,
        max_combination_len: int = 4,
        combine_misaligned_chunks: bool = True,
        end_mismatch_threshold: float = 0.2,
    ) -> List:
        """Optimized version of post_process_alignment with better performance.

        Key optimizations:
        1. Precompute string concatenations to avoid repeated joins
        2. Early termination when no bad regions are found
        3. Cache alignment results for repeated chunk patterns
        4. Vectorized index calculations
        5. Reduced nested loop complexity
        6. Combine multiple consecutive misaligned tokens into single chunks

        Args:
            combine_misaligned_chunks: If True, combine consecutive misaligned tokens into chunks
            end_mismatch_threshold: Fraction of sequence length from end to avoid chunking (0.2 = last 20%)
        """
        if not aligned_pairs:
            return []

        # Precompute joined strings for all pairs to avoid repeated concatenation
        # Use canonicalization for robust comparison
        pair_strings = []
        for i, (s1_tokens, s2_tokens, *rest) in enumerate(aligned_pairs):
            # Canonicalize individual tokens before joining for better matching
            s1_canonical_tokens = (
                [TokenAligner._canonical_token(tok) for tok in s1_tokens]
                if s1_tokens
                else []
            )
            s2_canonical_tokens = (
                [TokenAligner._canonical_token(tok) for tok in s2_tokens]
                if s2_tokens
                else []
            )
            s1_str = "".join(s1_canonical_tokens)
            s2_str = "".join(s2_canonical_tokens)
            is_match = TokenAligner._strings_equal_flexible(
                s1_str, s2_str, ignore_leading_char_diff
            )
            pair_strings.append((s1_str, s2_str, is_match))

        # Step 1: Handle consecutive misaligned chunks if enabled
        if combine_misaligned_chunks:
            aligned_pairs = TokenAligner._combine_consecutive_misaligned_tokens(
                aligned_pairs, pair_strings, end_mismatch_threshold
            )

            # Recompute pair_strings after combining misaligned chunks
            pair_strings = []
            for i, (s1_tokens, s2_tokens, *rest) in enumerate(aligned_pairs):
                s1_canonical_tokens = (
                    [TokenAligner._canonical_token(tok) for tok in s1_tokens]
                    if s1_tokens
                    else []
                )
                s2_canonical_tokens = (
                    [TokenAligner._canonical_token(tok) for tok in s2_tokens]
                    if s2_tokens
                    else []
                )
                s1_str = "".join(s1_canonical_tokens)
                s2_str = "".join(s2_canonical_tokens)
                is_match = TokenAligner._strings_equal_flexible(
                    s1_str, s2_str, ignore_leading_char_diff
                )
                pair_strings.append((s1_str, s2_str, is_match))

        processed_pairs = []
        alignment_cache = {}  # Cache for repeated alignment patterns
        i = 0

        while i < len(aligned_pairs):
            s1_tokens, s2_tokens, *_ = aligned_pairs[i]

            # Handle coarse alignments that can be split (optimized)
            if (
                len(s1_tokens) > 1
                and len(s1_tokens) == len(s2_tokens)
                and s1_tokens == s2_tokens
            ):
                s1_start, s1_end, s2_start, s2_end = aligned_pairs[i][2:6]
                # Vectorized creation of split pairs
                for k in range(len(s1_tokens)):
                    processed_pairs.append(
                        (
                            [s1_tokens[k]],
                            [s2_tokens[k]],
                            s1_start + k,
                            s1_start + k + 1,
                            s2_start + k,
                            s2_start + k + 1,
                        )
                    )
                i += 1
                continue

            # Find bad regions more efficiently using precomputed strings
            start_bad_region = -1
            for j in range(i, len(aligned_pairs)):
                if not pair_strings[j][2]:  # is_match is False
                    start_bad_region = j
                    break

            if start_bad_region == -1:
                # No more bad regions - add remaining pairs and exit
                processed_pairs.extend(aligned_pairs[i:])
                break

            # Add good pairs before bad region
            processed_pairs.extend(aligned_pairs[i:start_bad_region])

            # Optimized chunk processing with early termination
            found_fix = False
            max_chunk_size = min(
                10, len(aligned_pairs) - start_bad_region
            )  # Limit search space

            for chunk_size in range(2, max_chunk_size + 1):
                chunk = aligned_pairs[start_bad_region : start_bad_region + chunk_size]

                # Efficient token extraction using list comprehension
                chunk_s1_tokens = []
                chunk_s2_tokens = []
                s1_indices = []
                s2_indices = []

                for (
                    s1_toks,
                    s2_toks,
                    s1_start,
                    s1_end,
                    s2_start,
                    s2_end,
                    *rest,
                ) in chunk:
                    chunk_s1_tokens.extend(s1_toks)
                    chunk_s2_tokens.extend(s2_toks)
                    if s1_toks:
                        s1_indices.extend([s1_start, s1_end])
                    if s2_toks:
                        s2_indices.extend([s2_start, s2_end])

                # Quick string comparison using canonicalization
                chunk_s1_canonical = [
                    TokenAligner._canonical_token(tok) for tok in chunk_s1_tokens
                ]
                chunk_s2_canonical = [
                    TokenAligner._canonical_token(tok) for tok in chunk_s2_tokens
                ]
                chunk_s1_str = "".join(chunk_s1_canonical)
                chunk_s2_str = "".join(chunk_s2_canonical)

                if not TokenAligner._strings_equal_flexible(
                    chunk_s1_str, chunk_s2_str, ignore_leading_char_diff
                ):
                    continue

                # Create cache key for alignment
                cache_key = (tuple(chunk_s1_tokens), tuple(chunk_s2_tokens))

                if cache_key in alignment_cache:
                    sub_aligned_pairs, realign_is_perfect = alignment_cache[cache_key]
                else:
                    # Perform alignment
                    sub_aligned_pairs, _ = (
                        TokenAligner.align_tokens_with_combinations_numpy(
                            chunk_s1_tokens,
                            chunk_s2_tokens,
                            exact_match_score=exact_match_score,
                            combination_score_multiplier=combination_score_multiplier,
                            gap_penalty=gap_penalty,
                            max_combination_len=max_combination_len,
                            ignore_leading_char_diff=ignore_leading_char_diff,
                        )
                    )

                    # Check if re-alignment was successful using canonicalization
                    realign_is_perfect = all(
                        TokenAligner._strings_equal_flexible(
                            "".join(
                                [TokenAligner._canonical_token(tok) for tok in p[0]]
                            ),
                            "".join(
                                [TokenAligner._canonical_token(tok) for tok in p[1]]
                            ),
                            ignore_leading_char_diff,
                        )
                        for p in sub_aligned_pairs
                    )

                    # Cache the result
                    alignment_cache[cache_key] = (sub_aligned_pairs, realign_is_perfect)

                # Vectorized index calculations
                s1_chunk_start = min(s1_indices[::2]) if s1_indices else -1
                s2_chunk_start = min(s2_indices[::2]) if s2_indices else -1

                if realign_is_perfect:
                    # Add granular aligned pairs
                    for (
                        s1_toks,
                        s2_toks,
                        s1_start,
                        s1_end,
                        s2_start,
                        s2_end,
                        *_,
                    ) in sub_aligned_pairs:
                        new_s1_start = (
                            s1_chunk_start + s1_start if s1_start != -1 else -1
                        )
                        new_s1_end = s1_chunk_start + s1_end if s1_end != -1 else -1
                        new_s2_start = (
                            s2_chunk_start + s2_start if s2_start != -1 else -1
                        )
                        new_s2_end = s2_chunk_start + s2_end if s2_end != -1 else -1
                        processed_pairs.append(
                            (
                                s1_toks,
                                s2_toks,
                                new_s1_start,
                                new_s1_end,
                                new_s2_start,
                                new_s2_end,
                            )
                        )
                else:
                    # Create merged pair
                    s1_chunk_end = max(s1_indices[1::2]) if s1_indices else -1
                    s2_chunk_end = max(s2_indices[1::2]) if s2_indices else -1
                    merged_pair = (
                        chunk_s1_tokens,
                        chunk_s2_tokens,
                        s1_chunk_start,
                        s1_chunk_end,
                        s2_chunk_start,
                        s2_chunk_end,
                    )
                    processed_pairs.append(merged_pair)

                i = start_bad_region + chunk_size
                found_fix = True
                break

            if not found_fix:
                processed_pairs.append(aligned_pairs[start_bad_region])
                i = start_bad_region + 1

        return processed_pairs

    @staticmethod
    def get_alignment_mask(
        aligned_pairs: List,
        use_canonicalization: bool = True,
        ignore_leading_char_diff: bool = False,
    ) -> List[bool]:
        """Get a boolean mask indicating which alignments are correct."""
        if not aligned_pairs:
            return []

        # Handle batch case - take first batch
        if (
            isinstance(aligned_pairs, list)
            and aligned_pairs
            and isinstance(aligned_pairs[0], list)
        ):
            pairs_to_verify = aligned_pairs[0]
        else:
            pairs_to_verify = aligned_pairs

        mask = []
        for (
            s1_tokens,
            s2_tokens,
            s1_start,
            s1_end,
            s2_start,
            s2_end,
            *rest,
        ) in pairs_to_verify:
            # Concatenate tokens into strings
            s1_str = "".join(s1_tokens) if s1_tokens else ""
            s2_str = "".join(s2_tokens) if s2_tokens else ""

            # Apply canonicalization if requested
            if use_canonicalization:
                s1_canonical = (
                    "".join([TokenAligner._canonical_token(tok) for tok in s1_tokens])
                    if s1_tokens
                    else ""
                )
                s2_canonical = (
                    "".join([TokenAligner._canonical_token(tok) for tok in s2_tokens])
                    if s2_tokens
                    else ""
                )
                is_correct = TokenAligner._strings_equal_flexible(
                    s1_canonical, s2_canonical, ignore_leading_char_diff
                )
            else:
                if ignore_leading_char_diff:
                    is_correct = TokenAligner._strings_equal_flexible(
                        s1_str, s2_str, ignore_leading_char_diff
                    )
                else:
                    is_correct = s1_str == s2_str

            mask.append(is_correct)

        return mask

    def _update_rules(
        self,
        aligned_pairs,
        student_token_ids=None,
        teacher_token_ids=None,
        student_sequence=None,
        teacher_sequence=None,
    ):
        """Update rule tracking with aligned pairs."""
        # Track how many times each rule is triggered
        if not hasattr(self, "forward_rule_counts"):
            self.forward_rule_counts = {}
        if not hasattr(self, "reverse_rule_counts"):
            self.reverse_rule_counts = {}
        if not hasattr(self, "forward_rules_with_ids"):
            self.forward_rules_with_ids = {}  # Maps (token_strings) -> (token_ids, count)
        if not hasattr(self, "reverse_rules_with_ids"):
            self.reverse_rules_with_ids = {}  # Maps (token_strings) -> (token_ids, count)
        if not hasattr(self, "rule_conflicts"):
            self.rule_conflicts = {}  # Track conflicting rules: source -> set of conflicting targets
        if not hasattr(self, "rule_conflict_counts"):
            self.rule_conflict_counts = {}  # Track counts: (source, target) -> count
        if not hasattr(self, "conflict_contexts"):
            self.conflict_contexts = {}  # Track full context when conflicts occur: conflict_id -> context_data

        for (
            s1_elems,
            s2_elems,
            s1_start,
            s1_end,
            s2_start,
            s2_end,
            *rest,
        ) in aligned_pairs:
            # Extract mask value if available, default to True for backward compatibility
            is_correct = rest[0] if rest else True

            # Only add rules if the alignment is correct (mask is positive)
            if not is_correct:
                continue

            s1_tuple = tuple(s1_elems)
            s2_tuple = tuple(s2_elems)

            if s1_tuple and s2_tuple:
                # Extract token IDs if available
                s1_ids = None
                s2_ids = None
                if student_token_ids is not None and s1_start != -1 and s1_end != -1:
                    s1_ids = tuple(student_token_ids[s1_start:s1_end])
                if teacher_token_ids is not None and s2_start != -1 and s2_end != -1:
                    s2_ids = tuple(teacher_token_ids[s2_start:s2_end])

                # Check for conflicts in existing rules
                existing_targets = [
                    rule[1] for rule in self.forward_rules if rule[0] == s1_tuple
                ]
                if existing_targets and s2_tuple not in existing_targets:
                    # Initialize conflict tracking for this source if not exists
                    if s1_tuple not in self.rule_conflicts:
                        self.rule_conflicts[s1_tuple] = set()

                    # Add all targets (existing + new) to the conflict set
                    all_targets = set(existing_targets + [s2_tuple])
                    old_conflict_size = len(self.rule_conflicts[s1_tuple])
                    self.rule_conflicts[s1_tuple].update(all_targets)

                    # Store the full context of this conflict
                    if len(self.rule_conflicts[s1_tuple]) > old_conflict_size:
                        conflict_id = (
                            f"{hash((s1_tuple, s2_tuple, len(self.conflict_contexts)))}"
                        )
                        context_data = {
                            "conflict_source": s1_tuple,
                            "new_target": s2_tuple,
                            "existing_targets": existing_targets,
                            "student_sequence": student_sequence,
                            "teacher_sequence": teacher_sequence,
                            "student_token_ids": student_token_ids,
                            "teacher_token_ids": teacher_token_ids,
                            "full_alignment": aligned_pairs,
                            "conflict_position": (s1_start, s1_end, s2_start, s2_end),
                            "student_ids_at_conflict": s1_ids,
                            "teacher_ids_at_conflict": s2_ids,
                            "timestamp": __import__("time").time(),
                        }
                        self.conflict_contexts[conflict_id] = context_data

                # Store string-based rules (backward compatibility)
                self.forward_rules.add((s1_tuple, s2_tuple))
                self.reverse_rules.add((s2_tuple, s1_tuple))

                # Track conflict counts for this specific source-target pair
                conflict_key = (s1_tuple, s2_tuple)

                # After adding to rules, check if this is now part of a conflict
                # and update conflict counts accordingly
                if (
                    s1_tuple in self.rule_conflicts
                    and s2_tuple in self.rule_conflicts[s1_tuple]
                ):
                    # This rule is part of a conflict, count it
                    self.rule_conflict_counts[conflict_key] = (
                        self.rule_conflict_counts.get(conflict_key, 0) + 1
                    )

                    # Also retroactively count any other conflicting rules for this source
                    # that we may have missed when they weren't conflicts yet
                    for target in self.rule_conflicts[s1_tuple]:
                        target_key = (s1_tuple, target)
                        if (
                            target_key != conflict_key
                            and target_key not in self.rule_conflict_counts
                        ):
                            # Count how many times this rule has been used
                            rule_count = self.forward_rule_counts.get(target_key, 0)
                            if rule_count > 0:
                                self.rule_conflict_counts[target_key] = rule_count

                # Store rules with token IDs
                forward_key = (s1_tuple, s2_tuple)
                reverse_key = (s2_tuple, s1_tuple)

                if forward_key not in self.forward_rules_with_ids:
                    self.forward_rules_with_ids[forward_key] = {
                        "student_ids": s1_ids,
                        "teacher_ids": s2_ids,
                        "count": 0,
                    }
                if reverse_key not in self.reverse_rules_with_ids:
                    self.reverse_rules_with_ids[reverse_key] = {
                        "teacher_ids": s1_ids,  # Note: reversed
                        "student_ids": s2_ids,  # Note: reversed
                        "count": 0,
                    }

                # Count how many times the rule was triggered
                self.forward_rule_counts[forward_key] = (
                    self.forward_rule_counts.get(forward_key, 0) + 1
                )
                self.reverse_rule_counts[reverse_key] = (
                    self.reverse_rule_counts.get(reverse_key, 0) + 1
                )

                # Update counts in the ID-based rules
                self.forward_rules_with_ids[forward_key]["count"] += 1
                self.reverse_rules_with_ids[reverse_key]["count"] += 1

    def translate(self, sequence, direction="forward"):
        """Translate a sequence using current rules.

        ``direction`` is ``'forward'`` (seq1 → seq2) or ``'reverse'`` (seq2 → seq1).
        """
        rules = self.forward_rules if direction == "forward" else self.reverse_rules
        rule_map = {src: tgt for src, tgt in rules}
        sorted_rules = sorted(rule_map.items(), key=lambda x: -len(x[0]))

        output = []
        i = 0
        while i < len(sequence):
            matched = False
            for src, tgt in sorted_rules:
                src_len = len(src)
                if tuple(sequence[i : i + src_len]) == src:
                    output.extend(tgt)
                    i += src_len
                    matched = True
                    break
            if not matched:
                output.append(sequence[i])
                i += 1
        return output

    def translate_via_alignment_spans(
        self, source_sequence, aligned_pairs, source_is_seq1=True
    ):
        """Translate a sequence using explicit alignment spans without reconstructing rules."""
        translated = []
        source_idx = 0

        for (
            s1_tokens,
            s2_tokens,
            s1_start,
            s1_end,
            s2_start,
            s2_end,
            *_,
        ) in aligned_pairs:
            current_source = s1_tokens if source_is_seq1 else s2_tokens
            target = s2_tokens if source_is_seq1 else s1_tokens

            if not current_source:
                # Insertion in target, not aligned in source → emit target
                translated.extend(target)
            else:
                match_span = source_sequence[
                    source_idx : source_idx + len(current_source)
                ]
                if match_span == current_source:
                    translated.extend(target)
                    source_idx += len(current_source)
                else:
                    source_idx += len(current_source)  # Skip to avoid infinite loop

        return translated

    def get_rules(self, direction="forward"):
        return self.forward_rules if direction == "forward" else self.reverse_rules

    def reset_rules(self):
        self.forward_rules.clear()
        self.reverse_rules.clear()

    def enable_rule_tracking(self):
        """Enable rule tracking for future alignments."""
        self.track_rules = True

    def disable_rule_tracking(self):
        """Disable rule tracking for future alignments."""
        self.track_rules = False

    def is_rule_tracking_enabled(self):
        """Check if rule tracking is currently enabled."""
        return self.track_rules

    def clear_all_rules(self):
        """Clear all collected rules and reset rule tracking data."""
        self.forward_rules.clear()
        self.reverse_rules.clear()

        # Clear rule counts if they exist
        if hasattr(self, "forward_rule_counts"):
            self.forward_rule_counts.clear()
        if hasattr(self, "reverse_rule_counts"):
            self.reverse_rule_counts.clear()

        # Clear conflict tracking if it exists
        if hasattr(self, "rule_conflicts"):
            self.rule_conflicts.clear()
        if hasattr(self, "rule_conflict_counts"):
            self.rule_conflict_counts.clear()

        # Clear rules with IDs if they exist
        if hasattr(self, "forward_rules_with_ids"):
            self.forward_rules_with_ids.clear()
        if hasattr(self, "reverse_rules_with_ids"):
            self.reverse_rules_with_ids.clear()

        # Clear conflict contexts if they exist
        if hasattr(self, "conflict_contexts"):
            self.conflict_contexts.clear()

    def compute_loss(
        self,
        aligned_pairs,
        student_logits,
        teacher_logits,
        input_ids_student,
        input_ids_teacher,
        loss_type="chunked_ce",
        exact_token_match_only=False,
        temperature=1.0,
        loss_on_non_zero_only=False,
        debug_verbose=False,
        kd_topk: int = 0,
        vocab_topk: int = 8192,
        reverse_kl: bool = False,
        project_teacher_logits_to_student: bool = False,
        log_softmax: str = "together",
        token_weights=None,
        gold_loss: bool = False,
        xtoken_loss: bool = False,
    ) -> float:
        """Compute the loss between two sequences of tokens.

        Args:
            aligned_pairs: Aligned token pairs with alignment mask (7th element indicates correctness)
            student_logits: Student model logits
            teacher_logits: Teacher model logits
            input_ids_student: Student input token IDs
            input_ids_teacher: Teacher input token IDs
            loss_type: 'chunked_ce' -> compute loss on chunks of tokens, from tokenkit
                      'KL' -> compute KL divergence between teacher and student logits
                      'cross_entropy' -> compute cross-entropy loss
            exact_token_match_only: If True, only use 1-1 token mappings that are correct according to the mask
                                   If False, use all alignments that are correct according to the mask
            temperature: Temperature for softening probability distributions (used in KL loss)
            loss_on_non_zero_only: If True, computes KL divergence only on non-zero vocabulary subset
                                 (only used in KL loss type)
            project_teacher_logits_to_student: If True, project teacher logits to student space (reverse projection)

        Returns:
            Computed loss value
        """
        # make sure aligned_pairs are present
        if not aligned_pairs:
            raise ValueError(
                "No aligned pairs found. Please align the sequences first."
            )

        topk_accuracy = 0.0

        # create list of tokenids with correct alignments using the alignment mask
        # for exact_token_match_only, add constraint that it should be 1-1 mapping
        if (
            isinstance(aligned_pairs, list)
            and aligned_pairs
            and isinstance(aligned_pairs[0], list)
        ):
            if exact_token_match_only:
                # Use mask + 1-1 mapping constraint
                tokenids_with_exact_match = [
                    [
                        el
                        for el in batch_pairs
                        if len(el) > 6 and el[6] and len(el[0]) == 1 and len(el[1]) == 1
                    ]
                    for batch_pairs in aligned_pairs
                ]
            else:
                # Use only the alignment mask (with fallback to old behavior if mask not available)
                tokenids_with_exact_match = [
                    [el for el in batch_pairs if len(el) > 6 and el[6]]
                    if batch_pairs and len(batch_pairs[0]) > 6
                    else [
                        el for el in batch_pairs if el[0] == el[1]
                    ]  # fallback to old behavior
                    for batch_pairs in aligned_pairs
                ]
        else:
            if exact_token_match_only:
                # Use mask + 1-1 mapping constraint
                tokenids_with_exact_match = [
                    [
                        (s1_elems, s2_elems, s1_start, s1_end, s2_start, s2_end, *rest)
                        for s1_elems, s2_elems, s1_start, s1_end, s2_start, s2_end, *rest in aligned_pairs
                        if len(rest) > 0
                        and rest[0]
                        and len(s1_elems) == 1
                        and len(s2_elems) == 1
                    ]
                ]
            else:
                # Use only the alignment mask (with fallback to old behavior if mask not available)
                if aligned_pairs and len(aligned_pairs[0]) > 6:
                    tokenids_with_exact_match = [
                        [
                            (
                                s1_elems,
                                s2_elems,
                                s1_start,
                                s1_end,
                                s2_start,
                                s2_end,
                                *rest,
                            )
                            for s1_elems, s2_elems, s1_start, s1_end, s2_start, s2_end, *rest in aligned_pairs
                            if len(rest) > 0 and rest[0]
                        ]
                    ]
                else:
                    # Fallback to old behavior for backward compatibility
                    tokenids_with_exact_match = [
                        [
                            (s1_elems, s2_elems, s1_start, s1_end, s2_start, s2_end)
                            for s1_elems, s2_elems, s1_start, s1_end, s2_start, s2_end, *rest in aligned_pairs
                            if s1_elems == s2_elems
                        ]
                    ]

        # compute the loss
        if loss_type == "chunked_ce":
            # from tokenkit
            loss = self.compute_ce_loss(
                aligned_pairs,
                student_logits,
                teacher_logits,
                input_ids_student,
                input_ids_teacher,
                tokenids_with_exact_match,
                exact_token_match_only,
            )
        elif loss_type == "cross_entropy":
            # considering only correct alignments based on mask
            # go over batch size dimension
            if exact_token_match_only:
                losses = []
                for batch_idx in range(student_logits.shape[0]):
                    for alignment_pair in tokenids_with_exact_match[batch_idx]:
                        # Extract components from alignment pair
                        _, _, start1, end1, _, _ = alignment_pair[:6]
                        if (start1 == -1 and end1 == -1) or (
                            start1 >= input_ids_student.shape[1]
                        ):
                            continue  # remove out of bounds indices
                        logits = student_logits[batch_idx, start1:end1, :]
                        targets = input_ids_student[
                            batch_idx, start1 + 1 : end1 + 1
                        ]  # dont forget shift
                        losses.append(
                            torch.nn.functional.cross_entropy(
                                logits.view(-1, student_logits.size(-1)),
                                targets.view(-1),
                            )
                        )
                if losses:
                    loss = torch.stack(losses).mean()
                else:
                    loss = torch.tensor(
                        0.0, device=student_logits.device, requires_grad=True
                    )
            else:
                loss = torch.nn.functional.cross_entropy(
                    student_logits[:, :-1].reshape(-1, student_logits.size(-1)),
                    input_ids_student[:, 1:].reshape(-1),
                )

        elif loss_type == "KL":
            # Use ultra-fast version for maximum speed (vocab_topk < 8192)
            if (
                vocab_topk <= -1
                and hasattr(self, "sparse_transformation_matrix")
                and self.sparse_transformation_matrix is not None
            ):
                loss, topk_accuracy = self.compute_KL_loss_ultra_fast(
                    aligned_pairs,
                    student_logits,
                    teacher_logits,
                    input_ids_student,
                    input_ids_teacher,
                    tokenids_with_exact_match,
                    exact_token_match_only,
                    temperature=temperature,
                    vocab_topk=vocab_topk,
                    use_mixed_precision=True,
                    reverse_kl=reverse_kl,
                )
            else:
                loss, topk_accuracy = self.compute_KL_loss_optimized(
                    aligned_pairs,
                    student_logits,
                    teacher_logits,
                    input_ids_student,
                    input_ids_teacher,
                    tokenids_with_exact_match,
                    exact_token_match_only,
                    temperature=temperature,
                    loss_on_non_zero_only=loss_on_non_zero_only,
                    debug_verbose=debug_verbose,
                    kd_topk=kd_topk,
                    vocab_topk=vocab_topk,
                    reverse_kl=reverse_kl,
                    project_teacher_logits_to_student=project_teacher_logits_to_student,
                    log_softmax=log_softmax,
                    token_weights=token_weights,
                    gold_loss=gold_loss,
                    xtoken_loss=xtoken_loss,
                )
        else:
            raise ValueError(f"Loss type {loss_type} not supported")

        return loss, topk_accuracy

    def compute_feature_mse_loss(
        self,
        aligned_pairs,
        student_features,
        teacher_features,
        exact_token_match_only=True,
    ):
        """Compute MSE loss between student and teacher features for exactly matching tokens.

        Args:
            aligned_pairs: List of alignment information for each batch
            student_features: Tensor of shape (batch_size, seq_len, hidden_dim) - student hidden states
            teacher_features: Tensor of shape (batch_size, seq_len, hidden_dim) - teacher hidden states
            exact_token_match_only: If True, only compute loss for tokens that exactly match

        Returns:
            MSE loss tensor
        """
        if not aligned_pairs:
            raise ValueError(
                "No aligned pairs found. Please align the sequences first."
            )

        # Create list of tokenids with exact token text match between teacher and student
        if (
            isinstance(aligned_pairs, list)
            and aligned_pairs
            and isinstance(aligned_pairs[0], list)
        ):
            tokenids_with_exact_match = [
                [el for el in batch_pairs if el[0] == el[1]]
                for batch_pairs in aligned_pairs
            ]
        else:
            tokenids_with_exact_match = [
                s1_elems
                for s1_elems, s2_elems, *_ in aligned_pairs
                if s1_elems == s2_elems
            ]

        if exact_token_match_only:
            # Collect features for exactly matching tokens
            student_features_matched = []
            teacher_features_matched = []

            for batch_idx in range(student_features.shape[0]):
                for _, _, start1, end1, start2, end2, *_ in tokenids_with_exact_match[
                    batch_idx
                ]:
                    # Skip invalid indices
                    if (start1 == -1 and end1 == -1) or (
                        start1 >= student_features.shape[1]
                    ):
                        continue
                    if (start2 == -1 and end2 == -1) or (
                        start2 >= teacher_features.shape[1]
                    ):
                        continue

                    # Extract features for the matching token spans
                    student_span_features = student_features[
                        batch_idx, start1:end1, :
                    ]  # Shape: (span_len, hidden_dim)
                    teacher_span_features = teacher_features[
                        batch_idx, start2:end2, :
                    ]  # Shape: (span_len, hidden_dim)

                    # Handle different span lengths by taking mean pooling or truncating
                    if student_span_features.shape[0] != teacher_span_features.shape[0]:
                        # Use mean pooling to handle different span lengths
                        student_span_mean = student_span_features.mean(
                            dim=0, keepdim=True
                        )  # Shape: (1, hidden_dim)
                        teacher_span_mean = teacher_span_features.mean(
                            dim=0, keepdim=True
                        )  # Shape: (1, hidden_dim)
                        student_features_matched.append(student_span_mean)
                        teacher_features_matched.append(teacher_span_mean)
                    else:
                        # Same span length, add all tokens
                        student_features_matched.append(student_span_features)
                        teacher_features_matched.append(teacher_span_features)

            # If no matching tokens found, return zero loss with gradient
            if not student_features_matched:
                return torch.tensor(
                    0.0,
                    device=student_features.device,
                    dtype=student_features.dtype,
                    requires_grad=True,
                )

            # Concatenate all matched features
            student_features_matched = torch.cat(
                student_features_matched, dim=0
            )  # Shape: (total_matched_tokens, hidden_dim)
            teacher_features_matched = torch.cat(
                teacher_features_matched, dim=0
            )  # Shape: (total_matched_tokens, hidden_dim)

            # Debug: Check for NaN before MSE computation
            student_has_nan = torch.isnan(student_features_matched).any()
            teacher_has_nan = torch.isnan(teacher_features_matched).any()

            if student_has_nan or teacher_has_nan:
                print(
                    f"DEBUG: NaN detected before MSE - student: {student_has_nan.item()}, teacher: {teacher_has_nan.item()}"
                )
                print(
                    f"Student matched shape: {student_features_matched.shape}, Teacher matched shape: {teacher_features_matched.shape}"
                )
                if student_has_nan:
                    print(
                        f"Student features stats: min={student_features_matched.min().item()}, max={student_features_matched.max().item()}"
                    )
                if teacher_has_nan:
                    print(
                        f"Teacher features stats: min={teacher_features_matched.min().item()}, max={teacher_features_matched.max().item()}"
                    )
                # Return zero loss to avoid NaN propagation
                return torch.tensor(
                    0.0,
                    device=student_features.device,
                    dtype=student_features.dtype,
                    requires_grad=True,
                )

            # Check for extreme values that might cause NaN
            if (
                torch.isinf(student_features_matched).any()
                or torch.isinf(teacher_features_matched).any()
            ):
                print("DEBUG: Infinite values detected in matched features")
                return torch.tensor(
                    0.0,
                    device=student_features.device,
                    dtype=student_features.dtype,
                    requires_grad=True,
                )

            # Compute MSE loss
            mse_loss = torch.nn.functional.mse_loss(
                student_features_matched, teacher_features_matched
            )

            # Debug: Check if MSE computation resulted in NaN
            if torch.isnan(mse_loss):
                print(
                    f"DEBUG: MSE loss is NaN! student_matched stats: min={student_features_matched.min().item():.6f}, max={student_features_matched.max().item():.6f}"
                )
                print(
                    f"teacher_matched stats: min={teacher_features_matched.min().item():.6f}, max={teacher_features_matched.max().item():.6f}"
                )
                print(
                    f"Difference stats: min={(student_features_matched - teacher_features_matched).min().item():.6f}, max={(student_features_matched - teacher_features_matched).max().item():.6f}"
                )
                return torch.tensor(
                    0.0,
                    device=student_features.device,
                    dtype=student_features.dtype,
                    requires_grad=True,
                )

        else:
            # Compute MSE loss over all positions (not recommended for cross-tokenizer alignment)
            # This assumes student and teacher sequences have the same length
            min_seq_len = min(student_features.shape[1], teacher_features.shape[1])
            student_features_truncated = student_features[:, :min_seq_len, :]
            teacher_features_truncated = teacher_features[:, :min_seq_len, :]
            mse_loss = torch.nn.functional.mse_loss(
                student_features_truncated, teacher_features_truncated
            )

        return mse_loss

    def transform_logits(self, input_logits):
        """Project student logits to teacher-vocabulary space via binary sparse ``P``.

        ``P`` has shape ``[student_vocab, teacher_vocab]``. The projection keeps
        logit semantics: any teacher token with *no* mapping is set to -inf so
        its probability is 0 after softmax.
        """
        P = self.sparse_transformation_matrix  # binary CSR matrix already on GPU
        if P is None:
            return None

        with torch.no_grad():
            # 1. Sparse matmul in bf16/fp16  (no big fp32 tensors)
            projected = TokenAligner.project_token_likelihoods_sparse(
                input_logits.softmax(dim=-1), P, input_logits.device
            )

            # 2. Columns with no mapping → –inf   (probability 0)
            # with torch.no_grad():
            #     # `column_has_data` is 1 for columns that receive at least one copy-over
            #     column_has_data = (P.sum(dim=0) > 0).to(projected.dtype)   # shape (teacher_vocab,)
            #     minus_inf = -torch.finfo(projected.dtype).max
            # projected = projected * column_has_data + minus_inf * (1.0 - column_has_data)
            projected = torch.log(projected + 1e-8)

        return projected

    def transform_learned_matrix_instance(
        self, x: torch.Tensor, dim: int = -1
    ) -> torch.Tensor:
        """Instance method version that uses instance variables."""
        scale_trick_enabled = (
            self.enable_scale_trick if self.enable_scale_trick is not None else False
        )
        return TokenAligner.transform_learned_matrix(
            x, dim, enable_scale_trick=scale_trick_enabled
        )

    @staticmethod
    def transform_learned_matrix(
        x: torch.Tensor, dim: int = -1, enable_scale_trick=None
    ) -> torch.Tensor:
        """Compute Quite Attention over tensor x along specified dimension.

        Args:
            x: Input tensor.
            dim: Dimension to apply attention over (default: -1).

        Returns:
            Tensor of same shape with quite attention applied.
        """
        if 0:
            exp_x = torch.exp(x)
            denom = 1 + torch.sum(exp_x, dim=dim, keepdim=True)
            return exp_x / denom
            # write as a single lambda function
            # return lambda x: torch.exp(x) / (1 + torch.sum(torch.exp(x), dim=dim, keepdim=True))
        else:
            scale_trick_enabled = (
                enable_scale_trick if enable_scale_trick is not None else False
            )
            if scale_trick_enabled:
                # trick with last column being multiplier of 0..1, or try with c instead of 1 in qa.
                scores = torch.nn.functional.softmax(x, dim=dim)
                # Create a mask to zero out the last column while preserving gradients
                # mask = torch.ones_like(scores)
                # mask[:, -1] = 0.0
                # scores = scores * mask
                # Alternative approach using sigmoid (commented out):
                # scores = scores * torch.sigmoid(x[:, -1].unsqueeze(-1))
                return scores
            else:
                # normal softmax
                return torch.nn.functional.softmax(x, dim=dim)
            return torch.nn.functional.softmax(x, dim=dim)

    def compute_KL_loss(
        self,
        aligned_pairs,
        student_logits,
        teacher_logits,
        input_ids_student,
        input_ids_teacher,
        tokenids_with_exact_match=None,
        exact_token_match_only=False,
        temperature=0.1,
        loss_on_non_zero_only=False,
        debug_verbose=False,
        kd_topk: int = 0,
    ):
        """Compute KL divergence loss between student and teacher logits.

        Always uses student->teacher projection: ``KL(student_projected || teacher)``.

        Args:
            aligned_pairs: List of alignment information.
            student_logits: Logits from the student model.
            teacher_logits: Logits from the teacher model.
            input_ids_student: Input token IDs for the student.
            input_ids_teacher: Input token IDs for the teacher.
            tokenids_with_exact_match: Pre-filtered list of alignment pairs.
            exact_token_match_only: If True, computes loss only on 1-to-1 matching tokens.
            temperature: Temperature for softening probability distributions.
            loss_on_non_zero_only: If True, computes KL divergence only on non-zero vocabulary subset.

        Returns:
            Computed KL divergence loss tensor.
        """
        # Always use student->teacher projection: KL(student_projected || teacher)
        # Project student logits to teacher's vocabulary space
        # student_probs = torch.nn.functional.softmax(student_logits / temperature, dim=-1)
        if tokenids_with_exact_match is None:
            tokenids_with_exact_match = [
                [
                    el
                    for el in batch_pairs
                    if len(el) > 6 and el[6] and len(el[0]) == 1 and len(el[1]) == 1
                ]
                for batch_pairs in aligned_pairs
            ]

        # july 26th, lets project logits not probabilities as before

        # student_logits = student_logits / temperature
        student_probs = torch.nn.functional.softmax(
            student_logits / temperature, dim=-1
        )

        # Detect which format is loaded and use appropriate projection
        if (
            hasattr(self, "sparse_transformation_matrix")
            and self.sparse_transformation_matrix is not None
        ):
            # Use sparse matrix projection (student→teacher)
            student_logits_projected = self.project_token_likelihoods_instance(
                student_probs,
                None,
                None,
                None,  # Not used for sparse format
                teacher_logits.device,
                use_sparse_format=True,
                sparse_matrix=self.sparse_transformation_matrix,
            )
        elif (
            hasattr(self, "likelihood_projection_indices")
            and self.likelihood_projection_indices is not None
        ):
            # Use dense projection (student→teacher)
            student_logits_projected = self.project_token_likelihoods_instance(
                student_probs,
                self.likelihood_projection_indices,
                self.transform_learned_matrix_instance(
                    self.likelihood_projection_matrix
                )
                if self.learnable
                else self.likelihood_projection_matrix,
                teacher_logits.shape[-1],
                teacher_logits.device,
                use_sparse_format=False,
                # global_top_indices=self.global_top_indices
            )
        else:
            raise ValueError(
                "No projection matrix loaded. Please call _load_logits_projection_map() first."
            )

        # Get teacher log-probabilities (target distribution)
        # Ensure teacher logits match the projected vocabulary size
        # projected_vocab_size = student_probs_projected.shape[-1]
        # teacher_vocab_size = teacher_logits.shape[-1]

        # Debug stats for teacher_logits (original)
        if debug_verbose:
            print(f"DEBUG: teacher_logits (original) - shape: {teacher_logits.shape}")
            print(
                f"DEBUG: teacher_logits (original) - min: {teacher_logits.min().item():.6f}, max: {teacher_logits.max().item():.6f}, mean: {teacher_logits.mean().item():.6f}"
            )
            print(
                f"DEBUG: teacher_logits (original) - has NaN: {torch.isnan(teacher_logits).any().item()}, has inf: {torch.isinf(teacher_logits).any().item()}"
            )

            # Debug stats for projected student probabilities
            # Note: 'projected_probs' is defined below, but at this point we can infer the shape from logits
            print(
                f"DEBUG: projected student distribution (after projection) - expected vocab: {teacher_logits.shape[-1]}"
            )

        # if projected_vocab_size != teacher_vocab_size:
        #     # Truncate or pad teacher logits to match projected vocabulary size
        #     if projected_vocab_size < teacher_vocab_size:
        #         # Truncate teacher logits to match projected size
        #         teacher_logits_matched = teacher_logits[:, :, :projected_vocab_size]
        #         print(f"Warning: Truncating teacher logits from {teacher_vocab_size} to {projected_vocab_size} to match projected vocabulary")
        #     else:
        #         # Pad teacher logits with very negative values (near zero probability)
        #         padding_size = projected_vocab_size - teacher_vocab_size
        #         padding = torch.full((*teacher_logits.shape[:-1], padding_size), -1e8,
        #                            device=teacher_logits.device, dtype=teacher_logits.dtype)
        #         teacher_logits_matched = torch.cat([teacher_logits, padding], dim=-1)
        #         print(f"Warning: Padding teacher logits from {teacher_vocab_size} to {projected_vocab_size} to match projected vocabulary")
        # else:
        #     teacher_logits_matched = teacher_logits

        teacher_logits_matched = teacher_logits

        teacher_log_probs = torch.nn.functional.log_softmax(
            teacher_logits_matched / temperature, dim=-1
        )

        # Debug stats for teacher_logits_matched and teacher_log_probs
        if debug_verbose:
            print(
                f"DEBUG: teacher_logits_matched - shape: {teacher_logits_matched.shape}"
            )
            print(
                f"DEBUG: teacher_logits_matched - min: {teacher_logits_matched.min().item():.6f}, max: {teacher_logits_matched.max().item():.6f}, mean: {teacher_logits_matched.mean().item():.6f}"
            )
            print(f"DEBUG: teacher_log_probs - shape: {teacher_log_probs.shape}")
            print(
                f"DEBUG: teacher_log_probs - min: {teacher_log_probs.min().item():.6f}, max: {teacher_log_probs.max().item():.6f}, mean: {teacher_log_probs.mean().item():.6f}"
            )
            print(
                f"DEBUG: teacher_log_probs - has NaN: {torch.isnan(teacher_log_probs).any().item()}, has inf: {torch.isinf(teacher_log_probs).any().item()}"
            )

        # Use student_probs_projected as P and teacher_log_probs as Q
        # KL(P || Q) = KL(student_projected || teacher)
        # projected_probs = torch.nn.functional.softmax(student_logits_projected, dim=-1)
        projected_probs = student_logits_projected
        target_log_probs = teacher_log_probs

        # Optional teacher top-k with renormalization (argument-driven)
        # Only enable top-k in the exact-match path; the chunk path expects full vocab shapes
        if kd_topk and exact_token_match_only and not loss_on_non_zero_only:
            k = min(int(kd_topk), teacher_logits_matched.shape[-1])
            if k > 0 and k < teacher_logits_matched.shape[-1]:
                # Teacher: top-k logits and renormalized log-probs over k
                topk = torch.topk(teacher_logits_matched, k=k, dim=-1)
                topk_indices = topk.indices
                topk_logits = topk.values / temperature
                target_log_probs = torch.nn.functional.log_softmax(topk_logits, dim=-1)

                # Student: gather projected probs at teacher top-k indices and renormalize over k
                gathered = torch.gather(projected_probs, dim=-1, index=topk_indices)
                denom = gathered.sum(dim=-1, keepdim=True).clamp_min(1e-10)
                projected_probs = gathered / denom

        # Debug stats for projected_probs and target_log_probs
        if debug_verbose:
            print(
                f"DEBUG: projected_probs - min: {projected_probs.min().item():.6f}, max: {projected_probs.max().item():.6f}, mean: {projected_probs.mean().item():.6f}"
            )
            print(
                f"DEBUG: projected_probs - has NaN: {torch.isnan(projected_probs).any().item()}, has inf: {torch.isinf(projected_probs).any().item()}"
            )
            print(
                f"DEBUG: target_log_probs - min: {target_log_probs.min().item():.6f}, max: {target_log_probs.max().item():.6f}, mean: {target_log_probs.mean().item():.6f}"
            )
            print(
                f"DEBUG: target_log_probs - has NaN: {torch.isnan(target_log_probs).any().item()}, has inf: {torch.isinf(target_log_probs).any().item()}"
            )

        if loss_on_non_zero_only:
            if (
                not hasattr(self, "sparse_transformation_matrix")
                or self.sparse_transformation_matrix is None
            ):
                raise ValueError(
                    "loss_on_non_zero_only=True requires a sparse transformation matrix to be loaded."
                )

            # Cache the mask for efficiency
            # For student→teacher projection, non-zero indices are in the teacher vocabulary (columns)
            if not hasattr(self, "_non_zero_teacher_vocab_mask"):
                with torch.no_grad():
                    # Get the unique column indices from the sparse matrix, which correspond to the teacher vocabulary
                    non_zero_indices = (
                        self.sparse_transformation_matrix.coalesce()
                        .indices()[1]
                        .unique()
                    )
                    # Create a mask for the full vocabulary
                    mask = torch.zeros(
                        teacher_logits_matched.shape[-1],
                        dtype=torch.bool,
                        device=teacher_logits_matched.device,
                    )
                    mask[non_zero_indices] = True
                    self._non_zero_teacher_vocab_mask = mask
            vocab_mask = self._non_zero_teacher_vocab_mask

            # Apply mask to both projected probabilities and target logits
            # Zero out probabilities for tokens not in the transformation matrix
            projected_probs = projected_probs * vocab_mask.unsqueeze(0).unsqueeze(0)
            # renormalize projected probabilities
            projected_probs = projected_probs / projected_probs.sum(
                -1, keepdim=True
            )  # didnt check it before hand

            # Zero out target logits for tokens not in the transformation matrix (before softmax)
            masked_teacher_logits = teacher_logits_matched * vocab_mask.unsqueeze(
                0
            ).unsqueeze(0)
            # Set masked positions to very negative values so they don't contribute to softmax
            masked_teacher_logits = masked_teacher_logits + (~vocab_mask).unsqueeze(
                0
            ).unsqueeze(0) * (-1e9)
            target_log_probs = torch.nn.functional.log_softmax(
                masked_teacher_logits / temperature, dim=-1
            )

        if exact_token_match_only:
            # Create boolean masks to select only the distributions for exactly matched tokens
            projected_mask = torch.zeros(
                projected_probs.shape[:2],
                dtype=torch.bool,
                device=student_logits.device,
            )
            target_mask = torch.zeros(
                target_log_probs.shape[:2],
                dtype=torch.bool,
                device=student_logits.device,
            )

            for example_idx in range(student_logits.shape[0]):
                for alignment_pair in aligned_pairs[example_idx]:
                    # Extract components from alignment pair
                    s1text, s2text, start1, end1, start2, end2 = alignment_pair[:6]
                    if (start1 == -1 and end1 == -1) or (
                        start1 >= input_ids_student.shape[1]
                    ):
                        continue
                    if (start2 == -1 and end2 == -1) or (
                        start2 >= input_ids_teacher.shape[1]
                    ):
                        continue
                    if start1 == 0 or start2 == 0:
                        continue
                    if (end1 - start1 != end2 - start2) and (end1 - start1 != 1):
                        continue
                    # print(f"s1text: {s1text}, s2text: {s2text}, start1: {start1}, end1: {end1}, start2: {start2}, end2: {end2}")
                    # For student→teacher projection: start1 is student (projected), start2 is teacher (target)
                    projected_mask[example_idx, start1 - 1] = True  # student positions
                    target_mask[example_idx, start2 - 1] = True  # teacher positions

            # Apply masks to get distributions for aligned tokens
            # Select only positions where mask is True
            projected_probs_masked = projected_probs[
                projected_mask
            ]  # Shape: (num_true_positions, vocab_size)
            target_log_probs_masked = target_log_probs[
                target_mask
            ]  # Shape: (num_true_positions, vocab_size)

            projected_log_probs_masked = torch.log(projected_probs_masked + 1e-10)

            # If no tokens are aligned, loss is 0, but we need a tensor with grad_fn
            if projected_probs_masked.numel() == 0:
                return torch.tensor(
                    0.0, device=student_logits.device, requires_grad=True
                )

            # Compute KL divergence on the masked distributions: KL(projected || target)
            # Check if shapes match and handle gracefully
            # if projected_probs_masked.shape != target_log_probs_masked.shape:
            #     print(f"Warning: Shape mismatch in KL loss computation - projected: {projected_probs_masked.shape}, target: {target_log_probs_masked.shape}")
            #     print("This should not happen after vocabulary size matching. Returning zero loss.")
            #     loss_kl = torch.tensor(0.0, device=student_logits.device, requires_grad=True)
            # else:
            #     loss_kl = torch.nn.functional.kl_div(target_log_probs_masked, projected_probs_masked, reduction="batchmean", log_target=False)
            loss_kl = torch.nn.functional.kl_div(
                projected_log_probs_masked,
                target_log_probs_masked,
                reduction="batchmean",
                log_target=True,
            )

            if 1:
                # for debugging
                # Compute top-5 accuracy for exact token matching
                with torch.no_grad():
                    if projected_probs_masked.numel() > 0:
                        # Use masked versions for exact token matching
                        student_top1_masked = torch.topk(
                            projected_probs_masked,
                            k=min(1, projected_probs_masked.shape[-1]),
                            dim=-1,
                        ).indices
                        teacher_probs_masked = torch.exp(target_log_probs_masked)
                        teacher_top1_masked = torch.topk(
                            teacher_probs_masked,
                            k=min(1, teacher_probs_masked.shape[-1]),
                            dim=-1,
                        ).indices

                        # Calculate overlap between top-5 predictions
                        matches = 0
                        total = 0
                        for i in range(student_top1_masked.shape[0]):
                            student_set = set(student_top1_masked[i].cpu().numpy())
                            teacher_set = set(teacher_top1_masked[i].cpu().numpy())
                            if len(student_set.intersection(teacher_set)) > 0:
                                matches += 1
                            total += 1

                        top1_accuracy = matches / total if total > 0 else 0.0

        else:
            # Chunk-based alignment with proper averaging to handle many-to-many token mappings
            max_length_projected = projected_probs.shape[1]
            max_length_target = target_log_probs.shape[1]
            max_n_chunks = min(max_length_projected, max_length_target)
            n_examples = student_logits.shape[0]

            projected_tokens_to_chunks = torch.zeros(
                (n_examples, max_length_projected, max_n_chunks), dtype=torch.bool
            ).to(student_logits.device)
            target_tokens_to_chunks = torch.zeros(
                (n_examples, max_length_target, max_n_chunks), dtype=torch.bool
            ).to(student_logits.device)

            # Build alignment masks
            for example_idx in range(n_examples):
                chunk_idx = 0
                for alignment_pair in aligned_pairs[example_idx]:
                    # Extract components from alignment pair
                    s1text, s2text, start1, end1, start2, end2 = alignment_pair[:6]
                    if start1 != -1 and start2 != -1 and chunk_idx < max_n_chunks:
                        # For student→teacher projection: start1 is student (projected), start2 is teacher (target)
                        projected_tokens_to_chunks[
                            example_idx, start1:end1, chunk_idx
                        ] = 1  # student positions
                        target_tokens_to_chunks[example_idx, start2:end2, chunk_idx] = (
                            1  # teacher positions
                        )
                        chunk_idx += 1

            # Compute chunk-averaged distributions
            # For student (projected probabilities): average probabilities within each chunk
            projected_chunk_probs = torch.bmm(
                projected_tokens_to_chunks.transpose(1, 2).to(
                    projected_probs.dtype
                ),  # (batch, max_n_chunks, max_length_projected)
                projected_probs,  # (batch, max_length_projected, vocab_size)
            )  # Result: (batch, max_n_chunks, vocab_size)

            # Normalize by number of tokens in each chunk to get proper averages
            chunk_sizes_projected = projected_tokens_to_chunks.sum(
                dim=1, keepdim=True
            ).float()  # (batch, 1, max_n_chunks)
            chunk_sizes_projected = chunk_sizes_projected.transpose(
                1, 2
            )  # (batch, max_n_chunks, 1)
            projected_chunk_probs = projected_chunk_probs / (
                chunk_sizes_projected + 1e-10
            )  # Avoid division by zero

            # Renormalize to ensure probabilities sum to 1 (handles numerical precision errors)
            projected_chunk_probs = projected_chunk_probs / (
                projected_chunk_probs.sum(dim=-1, keepdim=True) + 1e-10
            )

            # Convert projected chunk probabilities to log probabilities (will recompute after optional top-k)
            projected_chunk_log_probs = torch.log(projected_chunk_probs + 1e-10)

            # Alternative: Geometric mean instead of arithmetic mean (uncomment to try)
            # This computes (P1 * P2 * ... * Pn)^(1/n) for each chunk
            # projected_chunk_probs_geom = torch.ones_like(projected_chunk_probs)
            # for example_idx in range(n_examples):
            #     for chunk_idx in range(max_n_chunks):
            #         mask = projected_tokens_to_chunks[example_idx, :, chunk_idx]
            #         if mask.any():
            #             chunk_tokens = projected_probs[example_idx][mask]  # (num_tokens_in_chunk, vocab_size)
            #             chunk_product = torch.prod(chunk_tokens + 1e-10, dim=0)  # Product across tokens
            #             chunk_geom_mean = torch.pow(chunk_product, 1.0 / mask.sum().float())
            #             projected_chunk_probs_geom[example_idx, chunk_idx] = chunk_geom_mean

            # For teacher: convert logits to probabilities first, then average probabilities (consistent with student)
            teacher_probs = torch.softmax(
                teacher_logits_matched / temperature, dim=-1
            )  # Convert to probabilities first
            target_chunk_probs = torch.bmm(
                target_tokens_to_chunks.transpose(1, 2).to(
                    teacher_probs.dtype
                ),  # (batch, max_n_chunks, max_length_target)
                teacher_probs,  # (batch, max_length_target, vocab_size)
            )  # Result: (batch, max_n_chunks, vocab_size)

            # Normalize by number of tokens in each chunk to get proper averages
            chunk_sizes_target = target_tokens_to_chunks.sum(
                dim=1, keepdim=True
            ).float()  # (batch, 1, max_n_chunks)
            chunk_sizes_target = chunk_sizes_target.transpose(
                1, 2
            )  # (batch, max_n_chunks, 1)
            target_chunk_probs = target_chunk_probs / (
                chunk_sizes_target + 1e-10
            )  # Avoid division by zero

            # Renormalize to ensure probabilities sum to 1 (handles numerical precision errors)
            target_chunk_probs = target_chunk_probs / (
                target_chunk_probs.sum(dim=-1, keepdim=True) + 1e-10
            )

            # Optional top-k over chunks (argument-driven)
            if kd_topk and not loss_on_non_zero_only:
                k = min(int(kd_topk), target_chunk_probs.shape[-1])
                if k > 0 and k < target_chunk_probs.shape[-1]:
                    topk = torch.topk(target_chunk_probs, k=k, dim=-1)
                    indices_k = topk.indices
                    target_probs_k = topk.values
                    projected_probs_k = torch.gather(
                        projected_chunk_probs, dim=-1, index=indices_k
                    )
                    # Renormalize over k
                    t_denom = target_probs_k.sum(dim=-1, keepdim=True).clamp_min(1e-10)
                    s_denom = projected_probs_k.sum(dim=-1, keepdim=True).clamp_min(
                        1e-10
                    )
                    target_chunk_probs = target_probs_k / t_denom
                    projected_chunk_probs = projected_probs_k / s_denom
                    # Recompute projected log-probs after slicing to keep shapes aligned
                    projected_chunk_log_probs = torch.log(projected_chunk_probs + 1e-10)

            # Convert target chunk probabilities to log probabilities
            target_chunk_log_probs = torch.log(target_chunk_probs + 1e-10)

            # Create mask for valid chunks (chunks that have tokens from both sides)
            chunk_mask = (chunk_sizes_projected.squeeze(-1) > 0) & (
                chunk_sizes_target.squeeze(-1) > 0
            )

            # Compute KL divergence: KL(projected_chunk_probs || target_chunk_log_probs)
            loss_kl = torch.nn.functional.kl_div(
                projected_chunk_log_probs,
                target_chunk_log_probs,
                reduction="none",
                log_target=True,
            )

            # Apply chunk mask and compute weighted average
            if chunk_mask.sum() > 0:
                loss_kl_weighted = (
                    loss_kl * chunk_mask.unsqueeze(-1)
                ).sum() / chunk_mask.sum()
                loss_kl = loss_kl_weighted
            else:
                loss_kl = torch.tensor(
                    0.0, device=student_logits.device, requires_grad=True
                )

            if 1:
                # Compute top-1 accuracy for chunk-based alignment
                with torch.no_grad():
                    if chunk_mask.sum() > 0:
                        # Get top-1 predictions from chunk-averaged distributions
                        student_top1_indices = torch.argmax(
                            projected_chunk_probs, dim=-1
                        )  # (batch, max_n_chunks)
                        teacher_top1_indices = torch.argmax(
                            target_chunk_probs, dim=-1
                        )  # (batch, max_n_chunks)

                        # Count matches only for valid chunks
                        matches = (
                            (
                                (student_top1_indices == teacher_top1_indices)
                                & chunk_mask
                            )
                            .sum()
                            .item()
                        )
                        total = chunk_mask.sum().item()

                        top1_accuracy = matches / total if total > 0 else 0.0
                    else:
                        top1_accuracy = 0.0

        # Scale loss by temperature squared
        return loss_kl * (temperature**2), top1_accuracy

    def compute_projected_logits_KL_loss(
        self,
        aligned_pairs,
        projected_student_logits,
        teacher_logits,
        input_ids_student,
        input_ids_teacher,
        tokenids_with_exact_match=None,
        exact_token_match_only=True,
        temperature=1.0,
        rewrite_with_sparse_projection=False,
        kd_topk: int = 0,
    ):
        """Compute KL divergence loss for pre-projected student logits.

        Student logits are assumed to already be projected into the teacher's
        vocabulary space; this function does NOT use the internal
        ``transformation_matrix`` by default. It relies on alignment pairs to
        match tokens between student and teacher sequences (which may have
        different lengths).

        Args:
            aligned_pairs: List of alignment information for each batch.
            projected_student_logits: Student logits projected to the teacher's vocabulary space.
                                      Shape: (batch_size, student_seq_len, teacher_vocab_size).
            teacher_logits: Original teacher model logits.
                            Shape: (batch_size, teacher_seq_len, teacher_vocab_size).
            input_ids_student: Student input token IDs.
            input_ids_teacher: Teacher input token IDs.
            tokenids_with_exact_match: Pre-filtered list of alignment pairs. If None, it will be computed.
            exact_token_match_only: If True, computes loss only on 1-to-1 token mappings that are textually identical.
                                     If False, uses chunk-based alignment similar to compute_KL_loss.
            temperature: Temperature for softening probability distributions.
            rewrite_with_sparse_projection: If True, rewrites elements in projected_student_logits using the
                                           sparse transformation matrix for vocabulary positions that have
                                           non-zero entries in the matrix. Requires sparse_transformation_matrix
                                           to be loaded.

        Returns:
            tuple: (Computed KL divergence loss tensor, top-1 accuracy float)
        """
        if tokenids_with_exact_match is None:
            if (
                isinstance(aligned_pairs, list)
                and aligned_pairs
                and isinstance(aligned_pairs[0], list)
            ):
                if exact_token_match_only:
                    # Use mask + 1-1 mapping constraint
                    tokenids_with_exact_match = [
                        [
                            el
                            for el in batch_pairs
                            if len(el) > 6
                            and el[6]
                            and len(el[0]) == 1
                            and len(el[1]) == 1
                        ]
                        for batch_pairs in aligned_pairs
                    ]
                else:
                    # Use only the alignment mask (with fallback to old behavior if mask not available)
                    tokenids_with_exact_match = [
                        [el for el in batch_pairs if len(el) > 6 and el[6]]
                        if batch_pairs and len(batch_pairs[0]) > 6
                        else [
                            el for el in batch_pairs if el[0] == el[1]
                        ]  # fallback to old behavior
                        for batch_pairs in aligned_pairs
                    ]
            else:
                raise ValueError(
                    "aligned_pairs must be a list of lists (batched input)."
                )

        # Optionally rewrite projected logits using sparse transformation matrix
        if rewrite_with_sparse_projection:
            if (
                not hasattr(self, "sparse_transformation_matrix")
                or self.sparse_transformation_matrix is None
            ):
                raise ValueError(
                    "rewrite_with_sparse_projection=True requires a sparse transformation matrix to be loaded."
                )

            # Since the sparse matrix contains only 0s and 1s, we can work directly with logits
            # and avoid expensive softmax/log conversions

            # Get the sparse matrix indices for direct mapping
            sparse_indices = (
                self.sparse_transformation_matrix.coalesce().indices()
            )  # Shape: [2, num_nonzero]
            student_vocab_indices = sparse_indices[
                0
            ]  # Student vocabulary indices (rows)
            teacher_vocab_indices = sparse_indices[
                1
            ]  # Teacher vocabulary indices (columns)

            # Clone to avoid in-place modification
            projected_student_logits = projected_student_logits.clone()

            # For binary sparse matrices (0s and 1s), directly copy teacher logits to corresponding positions
            # This overwrites the projected logits with teacher logits for vocabulary positions that have
            # non-zero entries in the transformation matrix
            projected_student_logits[:, :, teacher_vocab_indices] = teacher_logits[
                :, :, teacher_vocab_indices
            ]

        # Get probabilities from logits (after potential rewriting)
        projected_student_probs = torch.nn.functional.softmax(
            projected_student_logits / temperature, dim=-1
        )
        teacher_probs = torch.nn.functional.softmax(
            teacher_logits / temperature, dim=-1
        )
        projected_student_log_probs = torch.nn.functional.log_softmax(
            projected_student_logits / temperature, dim=-1
        )

        if exact_token_match_only:
            # Create boolean masks to select distributions for exactly matched tokens
            student_mask = torch.zeros(
                projected_student_logits.shape[:2],
                dtype=torch.bool,
                device=projected_student_logits.device,
            )
            teacher_mask = torch.zeros(
                teacher_logits.shape[:2], dtype=torch.bool, device=teacher_logits.device
            )

            for batch_idx in range(projected_student_logits.shape[0]):
                if batch_idx >= len(tokenids_with_exact_match):
                    continue

                for _, _, start1, end1, start2, end2, *_ in tokenids_with_exact_match[
                    batch_idx
                ]:
                    # Skip invalid indices or non 1-to-1 matches
                    if (
                        start1 == -1
                        or start2 == -1
                        or (end1 - start1 != 1)
                        or (end2 - start2 != 1)
                    ):
                        continue

                    if start1 == 0 or start2 == 0:
                        continue

                    # Ensure indices are within bounds (accounting for shift)
                    if (
                        start1 - 1 >= projected_student_logits.shape[1]
                        or start2 - 1 >= teacher_logits.shape[1]
                    ):
                        continue
                    if start1 - 1 < 0 or start2 - 1 < 0:
                        continue
                    # we will shift here - use logits at position t-1 to predict token at position t
                    student_mask[batch_idx, start1 - 1] = True
                    teacher_mask[batch_idx, start2 - 1] = True

            # Apply masks to get distributions for aligned tokens
            student_log_probs_masked = projected_student_log_probs[student_mask]
            teacher_probs_masked = teacher_probs[teacher_mask]

            # If no tokens are aligned, loss is 0
            if student_log_probs_masked.numel() == 0:
                return torch.tensor(
                    0.0, device=projected_student_logits.device, requires_grad=True
                ), 0.0

            # Ensure the number of matched tokens is consistent
            if student_log_probs_masked.shape[0] != teacher_probs_masked.shape[0]:
                # This case can indicate a bug in alignment or masking logic.
                # It's safer to return 0 loss than to proceed with mismatched tensors.
                return torch.tensor(
                    0.0, device=projected_student_logits.device, requires_grad=True
                ), 0.0

            # Compute KL divergence on the masked distributions: KL(teacher || student)
            loss_kl = torch.nn.functional.kl_div(
                student_log_probs_masked,
                teacher_probs_masked,
                reduction="batchmean",
                log_target=False,
            )

            # Compute top-1 accuracy for exact token matching (projected logits)
            with torch.no_grad():
                if student_log_probs_masked.numel() > 0:
                    # Convert log probabilities to probabilities for masked student predictions
                    student_probs_masked = torch.exp(student_log_probs_masked)

                    # Get top-1 predictions for both
                    student_top1_masked = torch.topk(
                        student_probs_masked,
                        k=min(1, student_probs_masked.shape[-1]),
                        dim=-1,
                    ).indices
                    teacher_top1_masked = torch.topk(
                        teacher_probs_masked,
                        k=min(1, teacher_probs_masked.shape[-1]),
                        dim=-1,
                    ).indices

                    # Calculate overlap between top-1 predictions
                    matches = 0
                    total = 0
                    for i in range(student_top1_masked.shape[0]):
                        student_set = set(student_top1_masked[i].cpu().numpy())
                        teacher_set = set(teacher_top1_masked[i].cpu().numpy())
                        if len(student_set.intersection(teacher_set)) > 0:
                            matches += 1
                        total += 1

                    top1_accuracy = matches / total if total > 0 else 0.0
                    # print(f"Top-1 accuracy (projected exact match): {top1_accuracy:.4f} ({matches}/{total})")
                else:
                    top1_accuracy = 0.0
                    # print("Top-1 accuracy (projected exact match): 0.0000 (0/0)")

        else:
            # Chunk-based alignment similar to compute_KL_loss
            # print("chunk-based alignment")
            max_length_teacher = teacher_logits.shape[1]
            max_length_student = projected_student_logits.shape[1]
            max_n_chunks = min(max_length_teacher, max_length_student)
            n_examples = projected_student_logits.shape[0]  # batch size

            teacher_tokens_to_chunks = torch.zeros(
                (n_examples, max_length_teacher, max_n_chunks), dtype=torch.bool
            ).to(projected_student_logits.device)
            student_tokens_to_chunks = torch.zeros(
                (n_examples, max_length_student, max_n_chunks), dtype=torch.bool
            ).to(projected_student_logits.device)

            # Use alignment mask to filter correct alignments
            for example_idx in range(n_examples):
                chunk_idx = 0
                # for alignment_pair in tokenids_with_exact_match[example_idx]:
                for alignment_pair in aligned_pairs[example_idx]:
                    # Extract components from alignment pair
                    s1text, s2text, start1, end1, start2, end2 = alignment_pair[:6]
                    # if start1 == 0 or start2 == 0:
                    #     continue
                    if start1 != -1 and start2 != -1:
                        student_tokens_to_chunks[
                            example_idx, start1:end1, chunk_idx
                        ] = 1
                        teacher_tokens_to_chunks[
                            example_idx, start2:end2, chunk_idx
                        ] = 1
                        chunk_idx += 1

            chunk_mask = (teacher_tokens_to_chunks.sum(-2) > 0) & (
                student_tokens_to_chunks.sum(-2) > 0
            )

            if 0:
                teacher_chunk_probs = torch.bmm(
                    teacher_tokens_to_chunks.transpose(1, 2).to(teacher_probs.dtype),
                    teacher_probs,
                )
                student_chunk_probs = torch.bmm(
                    student_tokens_to_chunks.transpose(1, 2).to(
                        projected_student_log_probs.dtype
                    ),
                    projected_student_log_probs.exp(),
                )
                # or equivalently, student_tokens_to_chunks[:, 1:].sum(-2) > 0

                # Compute KL divergence over the entire sequences
                loss_kl = torch.nn.functional.kl_div(
                    torch.log(student_chunk_probs + 1e-10),
                    torch.log(teacher_chunk_probs + 1e-10),
                    reduction="none",
                    log_target=True,
                )
            else:
                # redo in logits space
                teacher_logits_chunk = torch.bmm(
                    teacher_tokens_to_chunks.transpose(1, 2).to(teacher_logits.dtype),
                    teacher_logits,
                )
                student_logits_chunk = torch.bmm(
                    student_tokens_to_chunks.transpose(1, 2).to(
                        projected_student_logits.dtype
                    ),
                    projected_student_logits,
                )
                # do log_softmax
                student_log_probs_chunk = torch.nn.functional.log_softmax(
                    student_logits_chunk, dim=-1
                )
                teacher_log_probs_chunk = torch.nn.functional.log_softmax(
                    teacher_logits_chunk, dim=-1
                )
                # Convert to probs for optional top-k slicing
                teacher_chunk_probs = torch.exp(teacher_log_probs_chunk)
                student_chunk_probs = torch.exp(student_log_probs_chunk)
                # Optional top-k over chunks (argument-driven). Applies to chunk mode as well.
                if kd_topk:
                    k = min(int(kd_topk), teacher_chunk_probs.shape[-1])
                    if k > 0 and k < teacher_chunk_probs.shape[-1]:
                        topk = torch.topk(teacher_chunk_probs, k=k, dim=-1)
                        idx = topk.indices
                        teacher_probs_k = topk.values
                        student_probs_k = torch.gather(
                            student_chunk_probs, dim=-1, index=idx
                        )
                        # Renormalize over k
                        t_denom = teacher_probs_k.sum(dim=-1, keepdim=True).clamp_min(
                            1e-10
                        )
                        s_denom = student_probs_k.sum(dim=-1, keepdim=True).clamp_min(
                            1e-10
                        )
                        teacher_probs_k = teacher_probs_k / t_denom
                        student_probs_k = student_probs_k / s_denom
                        # Replace log-probs and probs with k-sliced versions
                        teacher_chunk_probs = teacher_probs_k
                        student_chunk_probs = student_probs_k
                        teacher_log_probs_chunk = torch.log(teacher_probs_k + 1e-10)
                        student_log_probs_chunk = torch.log(student_probs_k + 1e-10)
                # Compute KL on (possibly) reduced distributions
                loss_kl = torch.nn.functional.kl_div(
                    student_log_probs_chunk,
                    teacher_log_probs_chunk,
                    reduction="none",
                    log_target=True,
                )

            # apply chunk mask
            loss_kl_weighted = (
                loss_kl * chunk_mask[:, :, None]
            ).sum() / chunk_mask.sum()
            loss_kl = loss_kl_weighted

            # Compute top-1 accuracy for chunk-based alignment (projected logits)
            with torch.no_grad():
                # Get top-1 predictions from projected student probabilities and teacher probabilities
                student_top1_indices = torch.topk(
                    student_chunk_probs, k=1, dim=-1
                ).indices
                teacher_top1_indices = torch.topk(
                    teacher_chunk_probs, k=1, dim=-1
                ).indices

                batch_size, seq_len = projected_student_probs.shape[:2]
                matches = 0
                total = 0

                for b in range(batch_size):
                    for t in range(seq_len):
                        student_set = set(student_top1_indices[b, t].cpu().numpy())
                        teacher_set = set(teacher_top1_indices[b, t].cpu().numpy())
                        if len(student_set.intersection(teacher_set)) > 0:
                            matches += 1
                        total += 1

                top1_accuracy = matches / total if total > 0 else 0.0
                # print(f"Top-1 accuracy (projected chunk-based): {top1_accuracy:.4f} ({matches}/{total})")

        # Scale loss by temperature squared
        return loss_kl * (temperature**2), top1_accuracy

    def compute_ce_loss(
        self,
        aligned_pairs,
        student_logits,
        teacher_logits,
        input_ids_student,
        input_ids_teacher,
        tokenids_with_exact_match=None,
        exact_token_match_only=False,
    ):
        # need to understand this function
        max_length_teacher = teacher_logits.shape[1]
        max_length_student = student_logits.shape[1]
        max_n_chunks = min(max_length_teacher, max_length_student)
        n_examples = student_logits.shape[0]  # batch size

        teacher_tokens_to_chunks = torch.zeros(
            (n_examples, max_length_teacher, max_n_chunks), dtype=torch.bool
        ).to(student_logits.device)
        student_tokens_to_chunks = torch.zeros(
            (n_examples, max_length_student, max_n_chunks), dtype=torch.bool
        ).to(student_logits.device)

        # Use alignment mask to filter correct alignments
        for example_idx in range(n_examples):
            chunk_idx = 0
            for alignment_pair in tokenids_with_exact_match[example_idx]:
                # Extract components from alignment pair
                s1text, s2text, start1, end1, start2, end2 = alignment_pair[:6]
                if start1 != -1 and start2 != -1:
                    teacher_tokens_to_chunks[example_idx, start2:end2, chunk_idx] = 1
                    student_tokens_to_chunks[example_idx, start1:end1, chunk_idx] = 1
                    chunk_idx += 1

        teacher_logprobs = torch.log_softmax(teacher_logits, -1)
        student_logprobs = torch.log_softmax(student_logits, -1)

        # shift is happening here
        teacher_main_path_logprobs = torch.take_along_dim(
            teacher_logprobs[:, :-1], input_ids_teacher[:, 1:, None], dim=-1
        ).squeeze(-1)
        student_main_path_logprobs = torch.take_along_dim(
            student_logprobs[:, :-1], input_ids_student[:, 1:, None], dim=-1
        ).squeeze(-1)

        def log1mexp(x):
            """Computes log(1 - exp(x)) in a numerically stable way for x < 0."""
            # For x < log(0.5), use log1p(-exp(x)) directly
            # For x >= log(0.5), use log(-expm1(x)) to avoid precision issues
            log_half = -torch.log(torch.tensor(2, device=x.device))
            return torch.where(
                x < log_half, torch.log1p(-torch.exp(x)), torch.log(-torch.expm1(x))
            )

        def distance_fn(log_y_true, log_y_pred, temp=100, epsilon=1e-6):
            log_y_true = (log_y_true.to(torch.float32) / temp) - epsilon
            log_y_pred = (log_y_pred.to(torch.float32) / temp) - epsilon

            return -(
                torch.exp(log_y_true) * log_y_pred
                + (-torch.expm1(log_y_true) * log1mexp(log_y_pred))
            )

        teacher_chunk_logprobs = torch.matmul(
            teacher_main_path_logprobs[:, None, :],
            teacher_tokens_to_chunks[:, 1:].to(teacher_main_path_logprobs.dtype),
        )
        student_chunk_logprobs = torch.matmul(
            student_main_path_logprobs[:, None, :],
            student_tokens_to_chunks[:, 1:].to(student_main_path_logprobs.dtype),
        )
        # or equivalently, student_tokens_to_chunks[:, 1:].sum(-2) > 0
        chunk_mask = (teacher_tokens_to_chunks[:, 1:].sum(-2) > 0) & (
            student_tokens_to_chunks[:, 1:].sum(-2) > 0
        )
        # is it the place to put only exact matches?
        elementwise_loss = distance_fn(teacher_chunk_logprobs, student_chunk_logprobs)

        loss = (elementwise_loss * chunk_mask).mean() / chunk_mask.to(
            torch.float32
        ).mean()
        return loss

    def compute_KL_loss_with_checkpointing(
        self,
        aligned_pairs,
        student_logits,
        teacher_logits,
        input_ids_student,
        input_ids_teacher,
        tokenids_with_exact_match=None,
        exact_token_match_only=False,
        temperature=0.1,
        loss_on_non_zero_only=False,
        debug_verbose=False,
        kd_topk: int = 0,
    ):
        """Memory-efficient KL loss using gradient checkpointing only.

        This is a drop-in replacement for compute_KL_loss that uses gradient checkpointing
        to reduce memory usage during the backward pass. The forward computation is identical.

        Args:
            Same as compute_KL_loss

        Returns:
            Same as compute_KL_loss: (loss_tensor, accuracy_float)
        """
        # If exact-token mode, fall back to original compute with checkpointing
        if exact_token_match_only:
            return torch.utils.checkpoint.checkpoint(
                self.compute_KL_loss,
                aligned_pairs,
                student_logits,
                teacher_logits,
                input_ids_student,
                input_ids_teacher,
                tokenids_with_exact_match,
                exact_token_match_only,
                temperature,
                loss_on_non_zero_only,
                debug_verbose,
                kd_topk,
                use_reentrant=False,
            )

        # Sequence microbatching for chunk-based KL to reduce peak memory
        device = student_logits.device
        batch_size = student_logits.shape[0]
        student_seq_len = student_logits.shape[1]
        teacher_seq_len = teacher_logits.shape[1]
        teacher_vocab_size = teacher_logits.shape[-1]

        # Build alignment masks (same as in compute_KL_loss chunk path)
        max_n_chunks = min(student_seq_len, teacher_seq_len)
        projected_tokens_to_chunks = torch.zeros(
            (batch_size, student_seq_len, max_n_chunks), dtype=torch.bool, device=device
        )
        target_tokens_to_chunks = torch.zeros(
            (batch_size, teacher_seq_len, max_n_chunks), dtype=torch.bool, device=device
        )

        for example_idx in range(batch_size):
            chunk_idx = 0
            for alignment_pair in aligned_pairs[example_idx]:
                s1text, s2text, start1, end1, start2, end2 = alignment_pair[:6]
                if start1 != -1 and start2 != -1 and chunk_idx < max_n_chunks:
                    projected_tokens_to_chunks[example_idx, start1:end1, chunk_idx] = 1
                    target_tokens_to_chunks[example_idx, start2:end2, chunk_idx] = 1
                    chunk_idx += 1

        # Accumulators for chunk sums
        projected_chunk_sums = torch.zeros(
            (batch_size, max_n_chunks, teacher_vocab_size),
            dtype=student_logits.dtype,
            device=device,
        )
        target_chunk_sums = torch.zeros(
            (batch_size, max_n_chunks, teacher_vocab_size),
            dtype=teacher_logits.dtype,
            device=device,
        )

        # Windowed student projection and accumulation
        window = 128
        # Determine projection mode
        use_sparse = hasattr(self, "sparse_transformation_matrix") and (
            self.sparse_transformation_matrix is not None
        )
        has_dense = hasattr(self, "likelihood_projection_indices") and (
            self.likelihood_projection_indices is not None
        )

        for s in range(0, student_seq_len, window):
            e = min(s + window, student_seq_len)
            # Student slice probs
            student_probs_slice = torch.softmax(
                student_logits[:, s:e, :] / temperature, dim=-1
            )
            # Project slice to teacher vocab
            if use_sparse:
                projected_slice = self.project_token_likelihoods_instance(
                    student_probs_slice,
                    None,
                    None,
                    None,
                    device,
                    use_sparse_format=True,
                    sparse_matrix=self.sparse_transformation_matrix,
                )
            elif has_dense:
                projected_slice = self.project_token_likelihoods_instance(
                    student_probs_slice,
                    self.likelihood_projection_indices,
                    self.transform_learned_matrix_instance(
                        self.likelihood_projection_matrix
                    )
                    if getattr(self, "learnable", False)
                    else self.likelihood_projection_matrix,
                    teacher_vocab_size,
                    device,
                    use_sparse_format=False,
                )
            else:
                raise ValueError(
                    "No projection matrix loaded. Please call _load_logits_projection_map() first."
                )

            mask_slice = projected_tokens_to_chunks[
                :, s:e, :
            ]  # (B, window_len, max_n_chunks)
            # (B, max_n_chunks, window_len) @ (B, window_len, Vt) -> (B, max_n_chunks, Vt)
            partial = torch.bmm(
                mask_slice.transpose(1, 2).to(projected_slice.dtype), projected_slice
            )
            projected_chunk_sums += partial

        # Windowed teacher accumulation
        for s in range(0, teacher_seq_len, window):
            e = min(s + window, teacher_seq_len)
            teacher_probs_slice = torch.softmax(
                teacher_logits[:, s:e, :] / temperature, dim=-1
            )
            mask_slice = target_tokens_to_chunks[:, s:e, :]
            partial = torch.bmm(
                mask_slice.transpose(1, 2).to(teacher_probs_slice.dtype),
                teacher_probs_slice,
            )
            target_chunk_sums += partial

        # Normalize by chunk sizes (mean over tokens inside chunk)
        chunk_sizes_projected = (
            projected_tokens_to_chunks.sum(dim=1, keepdim=True).float().transpose(1, 2)
        )  # (B, max_n_chunks, 1)
        chunk_sizes_target = (
            target_tokens_to_chunks.sum(dim=1, keepdim=True).float().transpose(1, 2)
        )  # (B, max_n_chunks, 1)

        projected_chunk_probs = projected_chunk_sums / (chunk_sizes_projected + 1e-10)
        target_chunk_probs = target_chunk_sums / (chunk_sizes_target + 1e-10)

        # Renormalize to ensure probabilities sum to 1
        projected_chunk_probs = projected_chunk_probs / (
            projected_chunk_probs.sum(dim=-1, keepdim=True) + 1e-10
        )
        target_chunk_probs = target_chunk_probs / (
            target_chunk_probs.sum(dim=-1, keepdim=True) + 1e-10
        )

        # Optional top-k slicing over chunks
        if kd_topk and not loss_on_non_zero_only:
            k = min(int(kd_topk), target_chunk_probs.shape[-1])
            if k > 0 and k < target_chunk_probs.shape[-1]:
                topk = torch.topk(target_chunk_probs, k=k, dim=-1)
                indices_k = topk.indices
                target_probs_k = topk.values
                projected_probs_k = torch.gather(
                    projected_chunk_probs, dim=-1, index=indices_k
                )
                # Renormalize over k
                t_denom = target_probs_k.sum(dim=-1, keepdim=True).clamp_min(1e-10)
                s_denom = projected_probs_k.sum(dim=-1, keepdim=True).clamp_min(1e-10)
                target_chunk_probs = target_probs_k / t_denom
                projected_chunk_probs = projected_probs_k / s_denom

        # Convert to log-probs
        projected_chunk_log_probs = torch.log(projected_chunk_probs + 1e-10)
        target_chunk_log_probs = torch.log(target_chunk_probs + 1e-10)

        # Valid chunk mask
        chunk_mask = (chunk_sizes_projected.squeeze(-1) > 0) & (
            chunk_sizes_target.squeeze(-1) > 0
        )

        # KL divergence per chunk
        loss_kl = torch.nn.functional.kl_div(
            projected_chunk_log_probs,
            target_chunk_log_probs,
            reduction="none",
            log_target=True,
        )
        if chunk_mask.sum() > 0:
            loss_kl = (loss_kl * chunk_mask.unsqueeze(-1)).sum() / chunk_mask.sum()
        else:
            loss_kl = torch.tensor(0.0, device=device, requires_grad=True)

        # Top-1 accuracy over chunks
        with torch.no_grad():
            if chunk_mask.sum() > 0:
                student_top1 = torch.argmax(projected_chunk_probs, dim=-1)
                teacher_top1 = torch.argmax(target_chunk_probs, dim=-1)
                matches = ((student_top1 == teacher_top1) & chunk_mask).sum().item()
                total = chunk_mask.sum().item()
                top1_accuracy = matches / total if total > 0 else 0.0
            else:
                top1_accuracy = 0.0

        return loss_kl * (temperature**2), top1_accuracy

    def compute_KL_loss_optimized(
        self,
        aligned_pairs,
        student_logits,
        teacher_logits,
        input_ids_student,
        input_ids_teacher,
        tokenids_with_exact_match=None,
        exact_token_match_only=False,
        temperature=1.0,
        loss_on_non_zero_only=False,
        debug_verbose=False,
        kd_topk: int = 0,
        vocab_topk: int = 8192,
        reverse_kl: bool = False,
        project_teacher_logits_to_student: bool = False,
        log_softmax: str = "together",
        token_weights=None,
        gold_loss: bool = False,
        xtoken_loss: bool = False,
    ):
        """Heavily optimized KL loss computation for large vocabularies.

        Key optimizations:
        - Pre-filter vocabulary to top-K teacher tokens globally
        - Fused softmax + log operations
        - Reduced intermediate tensor allocations
        - Early exit for empty alignments

        Args:
            vocab_topk: Reduce effective vocabulary size to this many tokens based on teacher logits
            project_teacher_logits_to_student: If True, project teacher logits to student space (instead of student to teacher)
            gold_loss: If True, use gold loss computation (no vocab transformation for chunks, direct logit averaging)
            Other args same as compute_KL_loss
        """
        if not aligned_pairs or not any(aligned_pairs):
            return torch.tensor(
                0.0, device=student_logits.device, requires_grad=True
            ), 0.0

        if 0:
            # print aligned_pairs
            # go over each entry and print the alignment pairs
            for aligned_pair in aligned_pairs:
                for alignment_pair in aligned_pair:
                    print(alignment_pair)
            exit()

        device = student_logits.device
        batch_size, student_seq_len, student_vocab_size = student_logits.shape
        teacher_seq_len, teacher_vocab_size = (
            teacher_logits.shape[1],
            teacher_logits.shape[2],
        )

        # Gold loss path: split into exact-mapped (common) and non-exact (uncommon) vocab
        if gold_loss:
            # Step 1: Create exact token map from projection matrix
            # Only include student tokens that have exactly one strong mapping to a teacher token
            if (
                not hasattr(self, "likelihood_projection_indices")
                or self.likelihood_projection_indices is None
            ):
                raise ValueError(
                    "gold_loss requires likelihood_projection_indices to be loaded"
                )

            projection_indices = (
                self.likelihood_projection_indices
            )  # (student_vocab, top_k)
            projection_matrix = (
                self.transform_learned_matrix_instance(
                    self.likelihood_projection_matrix
                )
                if getattr(self, "learnable", False)
                else self.likelihood_projection_matrix
            )

            # Find student tokens with exactly one strong mapping
            # Sort projection weights for each student token to find strongest mappings
            sorted_values, sorted_indices_in_topk = torch.sort(
                projection_matrix, dim=-1, descending=True
            )

            # A student token has exact mapping if:
            # - First value is high (>0.9) indicating strong mapping
            # - Second value is low (<0.1) indicating no other strong mappings

            if xtoken_loss:
                # remove multitoken projections
                # consider ones with top1 proj > 0.6 prob in the transformation matrix as exact mappings; with GOLD, anything that has <1.0 prob is considered non exact mapping for ULD loss
                # avoid collisions, it's makes KL loss shoot up
                has_exact_map = sorted_values[:, 0] >= 0.6
            else:
                has_exact_map = (sorted_values[:, 0] == 1.0) & (
                    projection_indices[:, 1] == -1
                )  # & (sorted_values[:, 1] < 0.1)

            # import pdb
            # pdb.set_trace()

            # Get the actual teacher token indices for exact mappings
            # projection_indices[student_idx, k] gives the teacher token for the k-th strongest mapping
            student_indices_with_exact_map = torch.where(has_exact_map)[0]
            teacher_indices_for_exact_map = projection_indices[
                student_indices_with_exact_map,
                sorted_indices_in_topk[student_indices_with_exact_map, 0],
            ]

            # Create mapping dictionaries for quick lookup
            student_to_teacher_exact_map = {}
            teacher_to_student_exact_map = {}
            teacher_collision_count = 0
            teacher_collisions = []  # Track which teacher tokens have multiple student mappings

            # for s_idx, t_idx in zip(student_indices_with_exact_map.tolist(), teacher_indices_for_exact_map.tolist()):
            #     # Only keep if teacher index is valid
            #     if 0 <= t_idx < teacher_vocab_size:
            #         if t_idx not in teacher_to_student_exact_map:# or xtoken_loss:
            #             # New mapping
            #             student_to_teacher_exact_map[s_idx] = t_idx
            #             teacher_to_student_exact_map[t_idx] = s_idx
            #         else:
            #             # Collision: teacher token already mapped to different student token
            #             teacher_collision_count += 1
            #             existing_s_idx = teacher_to_student_exact_map[t_idx]
            # teacher_collisions.append((t_idx, existing_s_idx, s_idx))

            for s_idx, t_idx in zip(
                student_indices_with_exact_map.tolist(),
                teacher_indices_for_exact_map.tolist(),
            ):
                # Only keep if teacher index is valid
                if 0 <= t_idx < teacher_vocab_size:
                    if t_idx not in teacher_to_student_exact_map or xtoken_loss:
                        # New mapping

                        if t_idx in teacher_to_student_exact_map:
                            prev_student_token = teacher_to_student_exact_map[t_idx]
                            prev_prob = sorted_values[prev_student_token, 0]

                            if prev_prob >= sorted_values[s_idx, 0]:
                                # print(f"Skipping: prev_prob={prev_prob} > new_prob={sorted_values[s_idx, 0]}")
                                continue
                            else:
                                del student_to_teacher_exact_map[prev_student_token]
                                # print(f"replacing student token {prev_student_token} {prev_prob} with {s_idx} {sorted_values[s_idx, 0]}")

                        student_to_teacher_exact_map[s_idx] = t_idx
                        teacher_to_student_exact_map[t_idx] = s_idx
                    else:
                        # Collision: teacher token already mapped to different student token
                        teacher_collision_count += 1
                        existing_s_idx = teacher_to_student_exact_map[t_idx]
                        teacher_collisions.append((t_idx, existing_s_idx, s_idx))

            # # Print collision diagnostics
            # if teacher_collision_count > 0:
            #     print(f"⚠️  Teacher token collision warning: {teacher_collision_count} student tokens tried to map to already-mapped teacher tokens")
            #     if len(teacher_collisions) <= 10:
            #         # Print all collisions if there are few
            #         for t_idx, existing_s, new_s in teacher_collisions:
            #             print(f"   Teacher token {t_idx} already mapped to student {existing_s}, skipping student {new_s}")
            #     else:
            #         # Print first 5 and last 5 if there are many
            #         print(f"   Showing first 5 and last 5 collisions:")
            #         for t_idx, existing_s, new_s in teacher_collisions[:5]:
            #             print(f"   Teacher token {t_idx} already mapped to student {existing_s}, skipping student {new_s}")
            #         print(f"   ... ({len(teacher_collisions) - 10} more collisions) ...")
            #         for t_idx, existing_s, new_s in teacher_collisions[-5:]:
            #             print(f"   Teacher token {t_idx} already mapped to student {existing_s}, skipping student {new_s}")

            # Step 2: Split indices into common (exact match) and uncommon (no exact match)
            common_student_indices = sorted(student_to_teacher_exact_map.keys())
            common_teacher_indices = [
                student_to_teacher_exact_map[s] for s in common_student_indices
            ]

            all_student_indices = set(range(student_vocab_size))
            all_teacher_indices = set(range(teacher_vocab_size))
            uncommon_student_indices = sorted(
                all_student_indices - set(common_student_indices)
            )
            uncommon_teacher_indices = sorted(
                all_teacher_indices - set(common_teacher_indices)
            )

            # print(f"Gold loss: {len(common_student_indices)} exact token mappings, "
            #   f"{len(uncommon_student_indices)} uncommon student tokens, "
            #   f"{len(uncommon_teacher_indices)} uncommon teacher tokens")

            # Step 3: Compute loss using chunk-based masking
            # Build chunk masks for all alignments (not just exact 1-to-1)
            max_n_chunks = min(student_seq_len, teacher_seq_len)

            student_chunk_mask = torch.zeros(
                (batch_size, student_seq_len, max_n_chunks),
                dtype=torch.bool,
                device=device,
            )
            teacher_chunk_mask = torch.zeros(
                (batch_size, teacher_seq_len, max_n_chunks),
                dtype=torch.bool,
                device=device,
            )

            # Fill chunk masks from alignment pairs
            for batch_idx in range(batch_size):
                for chunk_idx, alignment_pair in enumerate(
                    aligned_pairs[batch_idx][:max_n_chunks]
                ):
                    s1text, s2text, start1, end1, start2, end2 = alignment_pair[:6]
                    if start1 != -1 and start2 != -1:
                        student_chunk_mask[batch_idx, start1:end1, chunk_idx] = True
                        teacher_chunk_mask[batch_idx, start2:end2, chunk_idx] = True

            # Compute log_softmax on original logits BEFORE averaging
            student_log_probs = torch.log_softmax(student_logits / temperature, dim=-1)
            teacher_log_probs = torch.log_softmax(teacher_logits / temperature, dim=-1)

            # Average log probabilities within chunks for FULL vocabularies
            student_chunk_log_probs_full = torch.bmm(
                student_chunk_mask.transpose(1, 2).to(student_log_probs.dtype),
                student_log_probs,
            )  # (batch, max_n_chunks, student_vocab_size)

            teacher_chunk_log_probs_full = torch.bmm(
                teacher_chunk_mask.transpose(1, 2).to(teacher_log_probs.dtype),
                teacher_log_probs,
            )  # (batch, max_n_chunks, teacher_vocab_size)

            # Normalize by chunk sizes
            student_chunk_sizes = (
                student_chunk_mask.sum(dim=1, keepdim=True).float().transpose(1, 2)
            )  # (batch, max_n_chunks, 1)
            teacher_chunk_sizes = (
                teacher_chunk_mask.sum(dim=1, keepdim=True).float().transpose(1, 2)
            )  # (batch, max_n_chunks, 1)

            student_chunk_log_probs_full = student_chunk_log_probs_full / (
                student_chunk_sizes + 1e-10
            )
            teacher_chunk_log_probs_full = teacher_chunk_log_probs_full / (
                teacher_chunk_sizes + 1e-10
            )

            # Valid chunk mask
            chunk_mask = (student_chunk_sizes.squeeze(-1) > 0) & (
                teacher_chunk_sizes.squeeze(-1) > 0
            )

            if not chunk_mask.any():
                return torch.tensor(0.0, device=device, requires_grad=True), 0.0

            # Now split chunk-averaged log probs into common and uncommon vocab
            # Extract common and uncommon from chunk-averaged log probs
            if len(common_student_indices) > 0:
                common_student_indices_tensor = torch.tensor(
                    common_student_indices, device=device
                )
                common_teacher_indices_tensor = torch.tensor(
                    common_teacher_indices, device=device
                )

                student_chunk_common_log_probs = student_chunk_log_probs_full[
                    :, :, common_student_indices_tensor
                ]  # (B, chunks, num_common)
                teacher_chunk_common_log_probs = teacher_chunk_log_probs_full[
                    :, :, common_teacher_indices_tensor
                ]  # (B, chunks, num_common)
            else:
                student_chunk_common_log_probs = torch.empty(
                    batch_size, max_n_chunks, 0, device=device
                )
                teacher_chunk_common_log_probs = torch.empty(
                    batch_size, max_n_chunks, 0, device=device
                )

            if len(uncommon_student_indices) > 0:
                uncommon_student_indices_tensor = torch.tensor(
                    uncommon_student_indices, device=device
                )
                student_chunk_uncommon_log_probs = student_chunk_log_probs_full[
                    :, :, uncommon_student_indices_tensor
                ]  # (B, chunks, num_uncommon_s)
            else:
                student_chunk_uncommon_log_probs = torch.empty(
                    batch_size, max_n_chunks, 0, device=device
                )

            if len(uncommon_teacher_indices) > 0:
                uncommon_teacher_indices_tensor = torch.tensor(
                    uncommon_teacher_indices, device=device
                )
                teacher_chunk_uncommon_log_probs = teacher_chunk_log_probs_full[
                    :, :, uncommon_teacher_indices_tensor
                ]  # (B, chunks, num_uncommon_t)
            else:
                teacher_chunk_uncommon_log_probs = torch.empty(
                    batch_size, max_n_chunks, 0, device=device
                )

            # Part 1: KL loss on common (aligned) vocab - using pre-computed log probs
            loss_kl_common = torch.tensor(0.0, device=device, requires_grad=True)
            if student_chunk_common_log_probs.shape[-1] > 0:
                # Compute KL divergence per chunk using pre-computed log probs
                if not reverse_kl:
                    loss_kl_per_elem = torch.nn.functional.kl_div(
                        student_chunk_common_log_probs,
                        teacher_chunk_common_log_probs,
                        reduction="none",
                        log_target=True,
                    )
                else:
                    loss_kl_per_elem = torch.nn.functional.kl_div(
                        teacher_chunk_common_log_probs,
                        student_chunk_common_log_probs,
                        reduction="none",
                        log_target=True,
                    )

                # Sum across vocab dimension
                # print(f"student {student_chunk_common_log_probs} teahcer {teacher_chunk_common_log_probs}")
                # import pdb
                # pdb.set_trace()
                loss_kl_per_chunk = loss_kl_per_elem.sum(
                    dim=-1
                )  # (batch, max_n_chunks)

                # Mask invalid chunks
                loss_kl_per_chunk = loss_kl_per_chunk * chunk_mask

                if chunk_mask.sum() > 0:
                    if token_weights is not None:
                        # Map chunk losses to teacher token positions
                        loss_kl_per_teacher_token = torch.bmm(
                            teacher_chunk_mask.to(loss_kl_per_chunk.dtype),
                            loss_kl_per_chunk.unsqueeze(-1),
                        ).squeeze(-1)

                        weighted_loss_per_token = (
                            loss_kl_per_teacher_token * token_weights
                        )

                        valid_teacher_sizes = (
                            teacher_chunk_sizes.squeeze(-1) * chunk_mask
                        )
                        total_teacher_token_participations = valid_teacher_sizes.sum()
                        if total_teacher_token_participations > 0:
                            loss_kl_common = (
                                weighted_loss_per_token.sum()
                                / total_teacher_token_participations
                            )
                    else:
                        loss_kl_common = loss_kl_per_chunk.sum() / chunk_mask.sum()
                        # pdb.set_trace()

            # Part 2: L1 loss on uncommon (unaligned) vocab - sort chunk-averaged probabilities
            loss_l1_uncommon = torch.tensor(0.0, device=device, requires_grad=True)
            # import pdb
            # pdb.set_trace()
            if (
                student_chunk_uncommon_log_probs.shape[-1] > 0
                or teacher_chunk_uncommon_log_probs.shape[-1] > 0
            ):
                # import pdb
                # pdb.set_trace()
                # Get valid chunks only
                student_uncommon_valid = student_chunk_uncommon_log_probs[
                    chunk_mask
                ]  # (num_valid_chunks, num_uncommon_s)
                teacher_uncommon_valid = teacher_chunk_uncommon_log_probs[
                    chunk_mask
                ]  # (num_valid_chunks, num_uncommon_t)

                if student_uncommon_valid.shape[0] > 0:
                    # Convert log probabilities to probabilities using exp - use in-place operations where possible
                    with torch.no_grad():
                        # Use topk instead of full sort to reduce memory - only need sorted values, not indices
                        # Limit the vocab size for uncommon distributions to prevent OOM
                        max_uncommon_vocab = min(
                            student_uncommon_valid.shape[-1],
                            teacher_uncommon_valid.shape[-1],
                            8192,  # Cap at reasonable size to prevent OOM
                        )

                    if max_uncommon_vocab > 0:
                        student_uncommon_probs = torch.exp(student_uncommon_valid)
                        teacher_uncommon_probs = torch.exp(teacher_uncommon_valid)

                        # Use topk for memory efficiency - we only need the top probabilities
                        # topk is much more memory efficient than full sort
                        if student_uncommon_probs.shape[-1] > max_uncommon_vocab:
                            student_uncommon_sorted, _ = torch.topk(
                                student_uncommon_probs,
                                k=max_uncommon_vocab,
                                dim=-1,
                                largest=True,
                            )
                        else:
                            student_uncommon_sorted = torch.sort(
                                student_uncommon_probs, dim=-1, descending=True
                            )[0]

                        if teacher_uncommon_probs.shape[-1] > max_uncommon_vocab:
                            teacher_uncommon_sorted, _ = torch.topk(
                                teacher_uncommon_probs,
                                k=max_uncommon_vocab,
                                dim=-1,
                                largest=True,
                            )
                        else:
                            teacher_uncommon_sorted = torch.sort(
                                teacher_uncommon_probs, dim=-1, descending=True
                            )[0]

                        # Free intermediate tensors immediately
                        del student_uncommon_probs, teacher_uncommon_probs

                        # Take minimum length for comparison
                        min_uncommon_len = min(
                            student_uncommon_sorted.shape[-1],
                            teacher_uncommon_sorted.shape[-1],
                        )
                        if min_uncommon_len > 0:
                            student_uncommon_sorted = student_uncommon_sorted[
                                :, :min_uncommon_len
                            ]
                            teacher_uncommon_sorted = teacher_uncommon_sorted[
                                :, :min_uncommon_len
                            ]

                            # Compute L1 loss on sorted uncommon probabilities
                            # print(f"ULD student {student_uncommon_sorted} teacher {teacher_uncommon_sorted}")
                            loss_l1_per_chunk = torch.nn.functional.l1_loss(
                                student_uncommon_sorted,
                                teacher_uncommon_sorted,
                                reduction="none",
                            ).sum(dim=-1)  # Sum over vocab dimension

                            # Free sorted tensors immediately after computing loss
                            del student_uncommon_sorted, teacher_uncommon_sorted

                            # Apply token weights if provided
                            if token_weights is not None:
                                # Expand chunk mask to get chunk indices
                                chunk_indices = torch.nonzero(
                                    chunk_mask, as_tuple=False
                                )  # (num_valid_chunks, 2) - [batch_idx, chunk_idx]

                                # Map chunks back to teacher tokens for weighting
                                weighted_l1_per_chunk = torch.zeros_like(
                                    loss_l1_per_chunk
                                )
                                for valid_idx, (batch_idx, chunk_idx) in enumerate(
                                    chunk_indices
                                ):
                                    # Get teacher tokens participating in this chunk
                                    teacher_tokens_in_chunk = teacher_chunk_mask[
                                        batch_idx, :, chunk_idx
                                    ]
                                    if teacher_tokens_in_chunk.any():
                                        # Average token weights for tokens in this chunk
                                        chunk_weight = token_weights[
                                            batch_idx, teacher_tokens_in_chunk
                                        ].mean()
                                        weighted_l1_per_chunk[valid_idx] = (
                                            loss_l1_per_chunk[valid_idx] * chunk_weight
                                        )

                                loss_l1_uncommon = weighted_l1_per_chunk.mean()
                                del weighted_l1_per_chunk, chunk_indices
                            else:
                                loss_l1_uncommon = loss_l1_per_chunk.mean()
                                # pdb.set_trace()

                            del loss_l1_per_chunk

            # Combine losses
            loss_total = loss_kl_common + loss_l1_uncommon
            # print(f"loss_kl_common: {loss_kl_common}, loss_l1_uncommon: {loss_l1_uncommon}")

            # Free large tensors before accuracy computation to ensure memory is available
            # These are no longer needed for the loss computation
            del student_chunk_log_probs_full, teacher_chunk_log_probs_full
            del student_chunk_mask, teacher_chunk_mask
            if len(uncommon_student_indices) > 0:
                del student_chunk_uncommon_log_probs
            if len(uncommon_teacher_indices) > 0:
                del teacher_chunk_uncommon_log_probs

            # Accuracy computation on common vocab - using pre-computed log probs
            # MEMORY OPTIMIZED: argmax works directly on log probs without needing exp()
            with torch.no_grad():
                if student_chunk_common_log_probs.shape[-1] > 0 and chunk_mask.any():
                    # Get predictions for valid chunks BEFORE exp to save memory
                    # argmax is invariant to monotonic transformations, so argmax(log_probs) == argmax(probs)
                    student_chunk_log_probs_valid = student_chunk_common_log_probs[
                        chunk_mask
                    ]
                    teacher_chunk_log_probs_valid = teacher_chunk_common_log_probs[
                        chunk_mask
                    ]

                    # Compute argmax directly on log probabilities (saves massive memory)
                    student_top1 = student_chunk_log_probs_valid.argmax(dim=-1)
                    teacher_top1 = teacher_chunk_log_probs_valid.argmax(dim=-1)
                    matches = (student_top1 == teacher_top1).sum().item()
                    top1_accuracy = matches / chunk_mask.sum().item()

                    # Clean up accuracy computation tensors
                    del student_chunk_log_probs_valid, teacher_chunk_log_probs_valid
                    del student_top1, teacher_top1
                else:
                    top1_accuracy = 0.0

            # Clean up remaining tensors
            del chunk_mask
            if len(common_student_indices) > 0:
                del student_chunk_common_log_probs, teacher_chunk_common_log_probs

            return loss_total * (temperature**2), top1_accuracy

        if project_teacher_logits_to_student:
            # REVERSE PROJECTION: Teacher → Student
            # print(f"Using REVERSE projection: Teacher → Student vocabulary space")

            # Step 1: Project teacher_logits (via probs) to student space
            if log_softmax == "separate":
                teacher_probs = torch.softmax(teacher_logits / temperature, dim=-1)
            else:
                teacher_probs = teacher_logits  # torch.softmax(teacher_logits / temperature, dim=-1)

            if (
                hasattr(self, "reverse_sparse_transformation_matrix")
                and self.reverse_sparse_transformation_matrix is not None
            ):
                # print(f"Using REVERSE sparse matrix projection for teacher_probs")
                projected_teacher_probs_full = self.project_token_likelihoods_instance(
                    teacher_probs,
                    None,
                    None,
                    None,
                    device,
                    use_sparse_format=True,
                    sparse_matrix=self.reverse_sparse_transformation_matrix,
                )
                # projected_teacher_probs_full is now in student space, full vocab (B, T, student_vocab_size)

            elif (
                hasattr(self, "reverse_likelihood_projection_indices")
                and self.reverse_likelihood_projection_indices is not None
            ):
                # print(f"Using REVERSE dense matrix projection for teacher_probs")
                reverse_matrix = self.reverse_likelihood_projection_matrix
                if getattr(self, "learnable", False):
                    reverse_matrix = self.transform_learned_matrix_instance(
                        reverse_matrix
                    )
                # print(f"reverse_matrix: {reverse_matrix}")

                # print(f"reverse_likelihood_projection_indices: {self.reverse_likelihood_projection_indices}")
                projected_teacher_probs_full = self.project_token_likelihoods_instance(
                    teacher_probs,
                    self.reverse_likelihood_projection_indices,
                    reverse_matrix,
                    student_vocab_size,
                    device,
                    use_sparse_format=False,
                )
                # projected_teacher_probs_full is now in student space, full vocab (B, T, student_vocab_size)
            else:
                raise ValueError(
                    "Reverse projection matrices not found. Please call create_reverse_projection_matrix() first."
                )

            # Step 2: Compute global_top_indices based on projected teacher probs (in student vocab space)
            # Use max probability per vocab position to find important student vocab tokens
            with torch.no_grad():
                if vocab_topk == 0 or vocab_topk >= student_vocab_size:
                    # Use all vocabulary tokens (no reduction)
                    global_top_indices = torch.arange(student_vocab_size, device=device)
                else:
                    # Get globally most important STUDENT tokens based on projected teacher probs
                    projected_teacher_flat = projected_teacher_probs_full.view(
                        -1, student_vocab_size
                    )
                    global_teacher_importance = projected_teacher_flat.max(dim=0)[
                        0
                    ]  # Max prob per vocab token
                    _, global_top_indices = torch.topk(
                        global_teacher_importance,
                        k=min(vocab_topk, student_vocab_size),
                        dim=-1,
                    )
                    global_top_indices = global_top_indices.sort()[
                        0
                    ]  # Keep sorted for efficiency

            # Step 3: Apply softmax on student_logits
            student_probs = torch.softmax(student_logits / temperature, dim=-1)

            # Step 4: Slice both distributions with global_top_indices
            if log_softmax == "together":
                projected_teacher_probs_reduced = torch.log_softmax(
                    projected_teacher_probs_full / temperature, dim=-1
                )[:, :, global_top_indices]  # (B, T, vocab_topk)
            else:
                projected_teacher_probs_reduced = projected_teacher_probs_full[
                    :, :, global_top_indices
                ]  # (B, T, vocab_topk)

            student_probs_reduced = student_probs[
                :, :, global_top_indices
            ]  # (B, S, vocab_topk)

            # Step 5: Apply log to get target_log_probs (projected probs are already in probability space)
            # For consistency with the rest of the code, set projected_probs to student and target to teacher
            if log_softmax == "separate":
                target_log_probs = torch.log(projected_teacher_probs_reduced + 1e-10)
            else:
                target_log_probs = projected_teacher_probs_reduced  # torch.log(projected_teacher_probs_reduced + 1e-10)  # Log of projected teacher probs
            projected_probs = student_probs_reduced  # Student probs (sliced)

        else:
            # FORWARD PROJECTION: Student → Teacher (original behavior)
            # Step 1: Global vocabulary filtering (major speedup)
            with torch.no_grad():
                if vocab_topk == 0 or vocab_topk >= teacher_vocab_size:
                    # Use all vocabulary tokens (no reduction)
                    global_top_indices = torch.arange(teacher_vocab_size, device=device)
                else:
                    # Get globally most important teacher tokens across all positions
                    teacher_flat = teacher_logits.view(-1, teacher_vocab_size)
                    global_teacher_importance = teacher_flat.max(dim=0)[
                        0
                    ]  # Max logit per vocab token
                    _, global_top_indices = torch.topk(
                        global_teacher_importance,
                        k=min(vocab_topk, teacher_vocab_size),
                        dim=-1,
                    )
                    global_top_indices = global_top_indices.sort()[
                        0
                    ]  # Keep sorted for efficiency

            # Step 2: Project student to reduced teacher vocabulary
            student_probs = torch.softmax(student_logits / temperature, dim=-1)

            if (
                hasattr(self, "sparse_transformation_matrix")
                and self.sparse_transformation_matrix is not None
            ):
                # print(f"Using sparse matrix projection for student_probs")
                projected_probs_full = self.project_token_likelihoods_instance(
                    student_probs,
                    None,
                    None,
                    None,
                    device,
                    use_sparse_format=True,
                    sparse_matrix=self.sparse_transformation_matrix,
                )
                projected_probs = projected_probs_full[
                    :, :, global_top_indices
                ]  # (B, S, vocab_topk)

            else:
                projected_probs_full = self.project_token_likelihoods_instance(
                    student_probs,
                    self.likelihood_projection_indices,
                    self.transform_learned_matrix_instance(
                        self.likelihood_projection_matrix
                    )
                    if getattr(self, "learnable", False)
                    else self.likelihood_projection_matrix,
                    teacher_vocab_size,
                    device,
                    use_sparse_format=False,
                )
                projected_probs = projected_probs_full[
                    :, :, global_top_indices
                ]  # (B, S, vocab_topk)

            # Step 3: Slice to reduced vocabulary
            teacher_logits_reduced = teacher_logits[
                :, :, global_top_indices
            ]  # (B, T, vocab_topk)

            # Step 4: Efficient target log-probabilities (fused softmax+log)
            # print(f"teacher top 50 max probs after topk: {torch.sort(torch.softmax(teacher_logits_reduced, dim=-1), descending=True)[0][:, :50]}")
            target_log_probs = torch.log_softmax(
                teacher_logits_reduced / temperature, dim=-1
            )

        if exact_token_match_only:
            # Optimized exact matching with reduced vocab
            student_mask = torch.zeros(
                batch_size, student_seq_len, dtype=torch.bool, device=device
            )
            teacher_mask = torch.zeros(
                batch_size, teacher_seq_len, dtype=torch.bool, device=device
            )

            for batch_idx in range(batch_size):
                for alignment_pair in aligned_pairs[batch_idx]:
                    s1text, s2text, start1, end1, start2, end2 = alignment_pair[:6]
                    if (
                        start1 > 0
                        and start2 > 0
                        and end1 - start1 == 1
                        and end2 - start2 == 1
                        and start1 - 1 < student_seq_len
                        and start2 - 1 < teacher_seq_len
                    ):
                        student_mask[batch_idx, start1 - 1] = True
                        teacher_mask[batch_idx, start2 - 1] = True
            # print(f"student_mask: {student_mask}")
            # print(f"teacher_mask: {teacher_mask}")
            # print(target_log_probs)

            if not student_mask.any():
                return torch.tensor(0.0, device=device, requires_grad=True), 0.0

            projected_probs_masked = projected_probs[student_mask]
            target_log_probs_masked = target_log_probs[teacher_mask]

            # print(f"projected_probs_masked.shape: {projected_probs_masked.shape}")
            # print(f"target_log_probs_masked.shape: {target_log_probs_masked.shape}")
            # exit()
            # Fused log + KL computation
            projected_log_probs_masked = torch.log(projected_probs_masked + 1e-10)
            if not reverse_kl:
                loss_kl_per_token = torch.nn.functional.kl_div(
                    projected_log_probs_masked,
                    target_log_probs_masked,
                    reduction="none",
                    log_target=True,
                )
            else:
                # print("Computing reverse KL1")
                loss_kl_per_token = torch.nn.functional.kl_div(
                    target_log_probs_masked,
                    projected_log_probs_masked,
                    reduction="none",
                    log_target=True,
                )

            # Sum across vocab dimension: (num_matched_tokens, vocab_topk) -> (num_matched_tokens,)
            loss_kl_per_token = loss_kl_per_token.sum(dim=-1)

            # Apply token weights if provided
            if token_weights is not None:
                # token_weights are based on teacher tokens, so use teacher_mask
                # token_weights shape: (batch_size, teacher_seq_len)
                token_weights_masked = token_weights[
                    teacher_mask
                ]  # (num_matched_tokens,)
                weighted_loss_per_token = loss_kl_per_token * token_weights_masked
                # Normalize by number of tokens to ensure comparable loss magnitudes
                # while weights still control relative contribution per token
                num_matched_tokens = teacher_mask.sum()
                if num_matched_tokens > 0:
                    loss_kl = weighted_loss_per_token.sum() / num_matched_tokens
                else:
                    loss_kl = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                # Regular batchmean reduction
                loss_kl = loss_kl_per_token.mean()

            # Fast accuracy computation
            with torch.no_grad():
                matches = (
                    (
                        projected_probs_masked.argmax(dim=-1)
                        == torch.exp(target_log_probs_masked).argmax(dim=-1)
                    )
                    .sum()
                    .item()
                )
                top1_accuracy = matches / projected_probs_masked.shape[0]

        else:
            # Chunk-based with reduced vocabulary - similar to original but on smaller vocab
            max_n_chunks = min(student_seq_len, teacher_seq_len)

            # Pre-allocate masks (more memory efficient)
            proj_mask = torch.zeros(
                (batch_size, student_seq_len, max_n_chunks),
                dtype=torch.bool,
                device=device,
            )
            tgt_mask = torch.zeros(
                (batch_size, teacher_seq_len, max_n_chunks),
                dtype=torch.bool,
                device=device,
            )

            # Fill masks efficiently
            for batch_idx in range(batch_size):
                for chunk_idx, alignment_pair in enumerate(
                    aligned_pairs[batch_idx][:max_n_chunks]
                ):
                    s1text, s2text, start1, end1, start2, end2 = alignment_pair[:6]
                    if start1 != -1 and start2 != -1:
                        proj_mask[batch_idx, start1:end1, chunk_idx] = True
                        tgt_mask[batch_idx, start2:end2, chunk_idx] = True

            # Efficient chunk averaging using bmm
            proj_chunks = torch.bmm(
                proj_mask.transpose(1, 2).to(projected_probs.dtype), projected_probs
            )
            tgt_log_chunks = torch.bmm(
                tgt_mask.transpose(1, 2).to(target_log_probs.dtype), target_log_probs
            )

            # Normalize by chunk sizes
            proj_sizes = proj_mask.sum(dim=1, keepdim=True).transpose(1, 2)
            tgt_sizes = tgt_mask.sum(dim=1, keepdim=True).transpose(1, 2)

            proj_chunks = proj_chunks / (proj_sizes + 1e-10)
            tgt_log_chunks = tgt_log_chunks / (tgt_sizes + 1e-10)

            # Renormalize and compute loss
            proj_chunks = proj_chunks / (proj_chunks.sum(dim=-1, keepdim=True) + 1e-10)
            proj_log_chunks = torch.log(proj_chunks + 1e-10)

            chunk_mask = (proj_sizes.squeeze(-1) > 0) & (tgt_sizes.squeeze(-1) > 0)

            if not reverse_kl:
                loss_kl_per_elem = torch.nn.functional.kl_div(
                    proj_log_chunks, tgt_log_chunks, reduction="none", log_target=True
                )
            else:
                # print("Computing reverse KL2")
                loss_kl_per_elem = torch.nn.functional.kl_div(
                    tgt_log_chunks, proj_log_chunks, reduction="none", log_target=True
                )

            # Sum across vocab dimension: (batch_size, max_n_chunks, vocab_topk) -> (batch_size, max_n_chunks)
            loss_kl_per_chunk = loss_kl_per_elem.sum(dim=-1)

            # Mask invalid chunks
            loss_kl_per_chunk = loss_kl_per_chunk * chunk_mask

            if chunk_mask.sum() > 0:
                if token_weights is not None:
                    # Map chunk losses back to TEACHER token positions using the teacher chunk mask
                    # token_weights are based on teacher tokens, shape: (batch_size, teacher_seq_len)
                    # tgt_mask shape: (batch_size, teacher_seq_len, max_n_chunks)
                    # loss_kl_per_chunk shape: (batch_size, max_n_chunks)

                    # For each teacher token, accumulate loss from all chunks it participates in
                    # (batch_size, teacher_seq_len, max_n_chunks) @ (batch_size, max_n_chunks, 1) -> (batch_size, teacher_seq_len, 1)
                    loss_kl_per_teacher_token = torch.bmm(
                        tgt_mask.to(
                            loss_kl_per_chunk.dtype
                        ),  # (batch_size, teacher_seq_len, max_n_chunks)
                        loss_kl_per_chunk.unsqueeze(
                            -1
                        ),  # (batch_size, max_n_chunks, 1)
                    ).squeeze(-1)  # -> (batch_size, teacher_seq_len)

                    # Weight the loss per teacher token
                    weighted_loss_per_token = loss_kl_per_teacher_token * token_weights

                    # Sum over teacher tokens and normalize
                    # Normalize by total teacher token participations in VALID chunks only
                    # comment from sharath: how many teacher tokens present in this chunk as you go from chunk to teacher space with the BMM
                    # tgt_sizes shape: (batch_size, max_n_chunks, 1), chunk_mask shape: (batch_size, max_n_chunks)
                    valid_tgt_sizes = (
                        tgt_sizes.squeeze(-1) * chunk_mask
                    )  # (batch_size, max_n_chunks)
                    total_teacher_token_participations = valid_tgt_sizes.sum()
                    if total_teacher_token_participations > 0:
                        # No need to mask: tokens only in invalid chunks already have loss=0
                        loss_kl = (
                            weighted_loss_per_token.sum()
                            / total_teacher_token_participations
                        )
                    else:
                        loss_kl = torch.tensor(0.0, device=device, requires_grad=True)
                else:
                    # Regular reduction by number of valid chunks
                    loss_kl = loss_kl_per_chunk.sum() / chunk_mask.sum()
            else:
                loss_kl = torch.tensor(0.0, device=device, requires_grad=True)
            # Accuracy computation
            with torch.no_grad():
                if chunk_mask.sum() > 0:
                    proj_top1 = proj_chunks.argmax(dim=-1)
                    tgt_top1 = torch.exp(tgt_log_chunks).argmax(dim=-1)
                    matches = ((proj_top1 == tgt_top1) & chunk_mask).sum().item()
                    top1_accuracy = matches / chunk_mask.sum().item()
                else:
                    top1_accuracy = 0.0

        return loss_kl * (temperature**2), top1_accuracy

    def compute_KL_loss_ultra_fast(
        self,
        aligned_pairs,
        student_logits,
        teacher_logits,
        input_ids_student,
        input_ids_teacher,
        tokenids_with_exact_match=None,
        exact_token_match_only=False,
        temperature=1.0,
        vocab_topk: int = 4096,
        use_mixed_precision=True,
        reverse_kl: bool = False,
    ):
        """Ultra-fast KL loss with maximum optimizations for production use.

        Key optimizations:
        - Aggressive vocabulary pruning (4k default vs 128k)
        - In-place operations where possible
        - Pre-allocated tensor reuse
        - Fused softmax-log operations
        - Mixed precision (fp16/bf16) for intermediate computations
        - Minimal tensor copying
        """
        if not aligned_pairs or not any(aligned_pairs):
            return torch.tensor(
                0.0, device=student_logits.device, requires_grad=True
            ), 0.0

        device = student_logits.device
        batch_size, student_seq_len = student_logits.shape[:2]
        teacher_seq_len, teacher_vocab_size = (
            teacher_logits.shape[1],
            teacher_logits.shape[2],
        )

        # Cache key for vocab filtering to avoid recomputation
        cache_key = (teacher_logits.shape, temperature, vocab_topk)

        if (
            not hasattr(self, "_vocab_cache")
            or self._vocab_cache.get("key") != cache_key
        ):
            with torch.no_grad():
                # More aggressive vocabulary filtering - use mean instead of max for better coverage
                teacher_flat = teacher_logits.view(-1, teacher_vocab_size)
                # Combine max and mean for better token selection
                global_importance = 0.7 * teacher_flat.max(dim=0)[
                    0
                ] + 0.3 * teacher_flat.mean(dim=0)
                _, top_indices = torch.topk(
                    global_importance, k=min(vocab_topk, teacher_vocab_size)
                )
                top_indices = top_indices.sort()[0]

                # Cache the indices and create index mapping for fast lookup
                self._vocab_cache = {
                    "key": cache_key,
                    "indices": top_indices,
                    "inv_mapping": torch.full(
                        (teacher_vocab_size,), -1, dtype=torch.long, device=device
                    ),
                }
                self._vocab_cache["inv_mapping"][top_indices] = torch.arange(
                    len(top_indices), device=device
                )

        top_indices = self._vocab_cache["indices"]

        # Use mixed precision for intermediate computations if available
        compute_dtype = (
            torch.float16
            if use_mixed_precision and torch.cuda.is_available()
            else student_logits.dtype
        )

        # Step 1: Slice vocabularies early to reduce all subsequent operations
        teacher_logits_reduced = teacher_logits[:, :, top_indices]  # (B, T, vocab_topk)

        # Step 2: Project student efficiently with ultra-fast projection
        student_probs = torch.softmax(student_logits / temperature, dim=-1)

        if (
            hasattr(self, "sparse_transformation_matrix")
            and self.sparse_transformation_matrix is not None
        ):
            # Use ultra-fast projection with direct vocabulary slicing
            projected_probs = self.project_token_likelihoods_ultra_fast(
                student_probs,
                sparse_matrix=self.sparse_transformation_matrix,
                target_vocab_reduced_indices=top_indices,
            ).to(compute_dtype)
        else:
            # Fallback to regular projection + slicing
            projected_probs_full = self.project_token_likelihoods_instance(
                student_probs,
                self.likelihood_projection_indices,
                self.transform_learned_matrix_instance(
                    self.likelihood_projection_matrix
                )
                if getattr(self, "learnable", False)
                else self.likelihood_projection_matrix,
                teacher_vocab_size,
                device,
                use_sparse_format=False,
            )
            projected_probs = projected_probs_full[:, :, top_indices].to(compute_dtype)

        # Step 3: Fused log-softmax on reduced teacher logits
        teacher_log_probs = torch.log_softmax(
            teacher_logits_reduced / temperature, dim=-1
        ).to(compute_dtype)

        if exact_token_match_only:
            # Ultra-fast exact matching using vectorized operations
            valid_positions = []

            for batch_idx in range(batch_size):
                batch_positions = []
                for alignment_pair in aligned_pairs[batch_idx]:
                    s1text, s2text, start1, end1, start2, end2 = alignment_pair[:6]
                    if (
                        start1 > 0
                        and start2 > 0
                        and end1 - start1 == 1
                        and end2 - start2 == 1
                        and start1 - 1 < student_seq_len
                        and start2 - 1 < teacher_seq_len
                    ):
                        batch_positions.append((start1 - 1, start2 - 1))
                valid_positions.append(batch_positions)

            # If no valid positions, return zero
            total_positions = sum(len(positions) for positions in valid_positions)
            if total_positions == 0:
                return torch.tensor(0.0, device=device, requires_grad=True), 0.0

            # Vectorized gathering of valid positions
            proj_list = []
            tgt_list = []
            for batch_idx, positions in enumerate(valid_positions):
                for s_pos, t_pos in positions:
                    proj_list.append(projected_probs[batch_idx, s_pos])
                    tgt_list.append(teacher_log_probs[batch_idx, t_pos])

            projected_probs_masked = torch.stack(proj_list, dim=0)
            target_log_probs_masked = torch.stack(tgt_list, dim=0)

            # In-place log operation for projected probs
            projected_log_probs_masked = torch.log(projected_probs_masked + 1e-10)

            # Fused KL computation
            if not reverse_kl:
                loss_kl = torch.nn.functional.kl_div(
                    projected_log_probs_masked,
                    target_log_probs_masked,
                    reduction="batchmean",
                    log_target=True,
                ).to(student_logits.dtype)
            else:
                # print("reverse KL 3")
                loss_kl = torch.nn.functional.kl_div(
                    target_log_probs_masked,
                    projected_log_probs_masked,
                    reduction="batchmean",
                    log_target=True,
                ).to(student_logits.dtype)

            # Fast accuracy - use argmax directly on reduced vocab
            with torch.no_grad():
                proj_argmax = projected_probs_masked.argmax(dim=-1)
                tgt_argmax = torch.exp(target_log_probs_masked).argmax(dim=-1)
                top1_accuracy = (proj_argmax == tgt_argmax).float().mean().item()

        else:
            # Optimized chunk-based processing
            max_n_chunks = min(
                student_seq_len, teacher_seq_len, 512
            )  # Limit chunks for speed

            # Pre-allocate reusable tensors
            if (
                not hasattr(self, "_chunk_cache")
                or self._chunk_cache["batch_size"] != batch_size
            ):
                self._chunk_cache = {
                    "batch_size": batch_size,
                    "proj_mask": torch.zeros(
                        (batch_size, student_seq_len, max_n_chunks),
                        dtype=torch.bool,
                        device=device,
                    ),
                    "tgt_mask": torch.zeros(
                        (batch_size, teacher_seq_len, max_n_chunks),
                        dtype=torch.bool,
                        device=device,
                    ),
                }

            proj_mask = self._chunk_cache["proj_mask"]
            tgt_mask = self._chunk_cache["tgt_mask"]

            # Clear masks (in-place)
            proj_mask.zero_()
            tgt_mask.zero_()

            # Fill masks efficiently (limit number of chunks processed)
            for batch_idx in range(batch_size):
                for chunk_idx, alignment_pair in enumerate(
                    aligned_pairs[batch_idx][:max_n_chunks]
                ):
                    s1text, s2text, start1, end1, start2, end2 = alignment_pair[:6]
                    if start1 != -1 and start2 != -1:
                        proj_mask[batch_idx, start1:end1, chunk_idx] = True
                        tgt_mask[batch_idx, start2:end2, chunk_idx] = True

            # Efficient chunk computation with reduced precision
            proj_chunks = torch.bmm(
                proj_mask[:, :, :max_n_chunks].transpose(1, 2).to(compute_dtype),
                projected_probs,
            )
            tgt_log_chunks = torch.bmm(
                tgt_mask[:, :, :max_n_chunks].transpose(1, 2).to(compute_dtype),
                teacher_log_probs,
            )

            # Fast normalization
            proj_sizes = (
                proj_mask[:, :, :max_n_chunks]
                .sum(dim=1, keepdim=True)
                .to(compute_dtype)
                .transpose(1, 2)
            )
            tgt_sizes = (
                tgt_mask[:, :, :max_n_chunks]
                .sum(dim=1, keepdim=True)
                .to(compute_dtype)
                .transpose(1, 2)
            )

            proj_chunks.div_(proj_sizes + 1e-10)
            tgt_log_chunks.div_(tgt_sizes + 1e-10)

            # In-place renormalization and log
            proj_chunks.div_(proj_chunks.sum(dim=-1, keepdim=True) + 1e-10)
            proj_log_chunks = torch.log(proj_chunks + 1e-10)

            chunk_mask = (proj_sizes.squeeze(-1) > 0) & (tgt_sizes.squeeze(-1) > 0)

            # Compute loss
            if not reverse_kl:
                loss_kl = torch.nn.functional.kl_div(
                    proj_log_chunks, tgt_log_chunks, reduction="none", log_target=True
                )
            else:
                # print("reverse KL4")
                loss_kl = torch.nn.functional.kl_div(
                    tgt_log_chunks, proj_log_chunks, reduction="none", log_target=True
                )

            if chunk_mask.sum() > 0:
                loss_kl = (loss_kl * chunk_mask.unsqueeze(-1)).sum() / chunk_mask.sum()
            else:
                loss_kl = torch.tensor(0.0, device=device, requires_grad=True)

            loss_kl = loss_kl.to(student_logits.dtype)

            # Fast accuracy
            with torch.no_grad():
                if chunk_mask.sum() > 0:
                    proj_argmax = proj_chunks.argmax(dim=-1)
                    tgt_argmax = torch.exp(tgt_log_chunks).argmax(dim=-1)
                    matches = ((proj_argmax == tgt_argmax) & chunk_mask).sum().item()
                    top1_accuracy = matches / chunk_mask.sum().item()
                else:
                    top1_accuracy = 0.0

        return loss_kl * (temperature**2), top1_accuracy
