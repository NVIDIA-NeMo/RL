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
import os
from dataclasses import dataclass
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

os.environ["TOKENIZERS_PARALLELISM"] = "false"


if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _dp_core_numba(ids1, ids2, joined1, joined2, n1, n2,
                       exact_match_score, gap_penalty, comb_mul, max_comb_len):
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


@dataclass(frozen=True)
class VocabPartition:
    """Projection-matrix-derived vocab partition for the gold-loss path.

    Built once per (xtoken_loss, teacher_vocab_size) by
    `TokenAligner.build_vocab_partition`. All tensors are long, 1-D, and
    live on the aligner's projection-matrix device.
    """

    common_student_indices: torch.Tensor
    common_teacher_indices: torch.Tensor
    uncommon_student_indices: torch.Tensor
    uncommon_teacher_indices: torch.Tensor


class TokenAligner(nn.Module):
    def __init__(self, max_comb_len=4, teacher_tokenizer_name=None, student_tokenizer_name=None, init_hf_tokenizers=True, projection_matrix_multiplier=1.0, enable_scale_trick=None):
        super().__init__()
        self.teacher_tokenizer_name = teacher_tokenizer_name
        self.student_tokenizer_name = student_tokenizer_name
        self.projection_matrix_multiplier = projection_matrix_multiplier
        self.enable_scale_trick = enable_scale_trick

        if init_hf_tokenizers:
            self.teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_tokenizer_name)
            self.student_tokenizer = AutoTokenizer.from_pretrained(student_tokenizer_name)
            if self.teacher_tokenizer.pad_token is None:
                self.teacher_tokenizer.pad_token = self.teacher_tokenizer.eos_token
            if self.student_tokenizer.pad_token is None:
                self.student_tokenizer.pad_token = self.student_tokenizer.eos_token
        else:
            self.teacher_tokenizer = None
            self.student_tokenizer = None

        self.max_combination_len = max_comb_len
        self.sparse_transformation_matrix  = None
        # Cached CSR for dense top-k projection (built from indices/values) to avoid scatter path
        self._dense_proj_csr = None
        self._dense_proj_csr_device = None

        # Precomputed canonical ID maps (built by precompute_canonical_maps)
        self._student_canon_map = None
        self._teacher_canon_map = None
        self._canon_id_to_str = None

        # Cached gold-loss vocab partitions keyed by (xtoken_loss, teacher_vocab_size).
        # Populated lazily by build_vocab_partition(); bypassed when learnable=True.
        self._vocab_partition_cache: dict[tuple[bool, int], VocabPartition] = {}

    @torch.no_grad()
    def build_vocab_partition(
        self, xtoken_loss: bool, teacher_vocab_size: int
    ) -> VocabPartition:
        """Derive (and cache) the gold-loss vocab partition.

        Computes exactly the state that CrossTokenizerDistillationLossFn._compute_gold_loss
        previously rebuilt every microbatch: the set of student/teacher tokens that have an
        exact 1:1 projection (common) and the complement (uncommon). Cached by
        (xtoken_loss, teacher_vocab_size) so repeat calls cost nothing. Caching is skipped
        when the projection matrix is learnable, since the partition then depends on
        gradient-updated values.
        """
        if (
            not hasattr(self, "likelihood_projection_indices")
            or self.likelihood_projection_indices is None
        ):
            raise ValueError(
                "build_vocab_partition requires likelihood_projection_indices to be loaded"
            )

        learnable = bool(getattr(self, "learnable", False))
        key = (bool(xtoken_loss), int(teacher_vocab_size))
        if not learnable:
            cached = self._vocab_partition_cache.get(key)
            if cached is not None:
                return cached

        projection_indices = self.likelihood_projection_indices
        projection_matrix = (
            self.transform_learned_matrix_instance(self.likelihood_projection_matrix)
            if learnable
            else self.likelihood_projection_matrix
        )
        device = projection_matrix.device
        student_vocab_size = int(projection_matrix.shape[0])

        sorted_values, sorted_indices_in_topk = torch.sort(
            projection_matrix, dim=-1, descending=True
        )

        if xtoken_loss:
            has_exact_map = sorted_values[:, 0] >= 0.6
        else:
            has_exact_map = (sorted_values[:, 0] == 1.0) & (projection_indices[:, 1] == -1)

        student_indices_with_exact_map = torch.where(has_exact_map)[0]
        teacher_indices_for_exact_map = projection_indices[
            student_indices_with_exact_map,
            sorted_indices_in_topk[student_indices_with_exact_map, 0],
        ]

        student_to_teacher_exact_map: dict[int, int] = {}
        teacher_to_student_exact_map: dict[int, int] = {}
        for s_idx, t_idx in zip(
            student_indices_with_exact_map.tolist(),
            teacher_indices_for_exact_map.tolist(),
        ):
            if 0 <= t_idx < teacher_vocab_size:
                if t_idx not in teacher_to_student_exact_map or xtoken_loss:
                    if t_idx in teacher_to_student_exact_map:
                        prev_student_token = teacher_to_student_exact_map[t_idx]
                        if sorted_values[prev_student_token, 0] >= sorted_values[s_idx, 0]:
                            continue
                        del student_to_teacher_exact_map[prev_student_token]
                    student_to_teacher_exact_map[s_idx] = t_idx
                    teacher_to_student_exact_map[t_idx] = s_idx

        common_student = sorted(student_to_teacher_exact_map.keys())
        common_teacher = [student_to_teacher_exact_map[s] for s in common_student]
        uncommon_student = sorted(
            set(range(student_vocab_size)) - set(common_student)
        )
        uncommon_teacher = sorted(
            set(range(teacher_vocab_size)) - set(common_teacher)
        )

        partition = VocabPartition(
            common_student_indices=torch.tensor(
                common_student, dtype=torch.long, device=device
            ),
            common_teacher_indices=torch.tensor(
                common_teacher, dtype=torch.long, device=device
            ),
            uncommon_student_indices=torch.tensor(
                uncommon_student, dtype=torch.long, device=device
            ),
            uncommon_teacher_indices=torch.tensor(
                uncommon_teacher, dtype=torch.long, device=device
            ),
        )

        if not learnable:
            self._vocab_partition_cache[key] = partition
        return partition

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
        print(f"  [TokenAligner] Precomputed canonical maps in {_t1-_t0:.2f}s "
              f"(student_vocab={student_vocab_size}, teacher_vocab={teacher_vocab_size}, "
              f"unique_canonical={len(canon_str_to_id)})", flush=True)

    def align_fast(self, student_ids, teacher_ids,
                   exact_match_score=3,
                   combination_score_multiplier=1.5,
                   gap_penalty=-1.5,
                   chunk_size=128,
                   post_process=True,
                   anchor_lengths=[3,],
                   ignore_leading_char_diff=False):
        """Fast alignment using precomputed canonical ID maps.

        Skips convert_ids_to_tokens and _canonicalize_sequence by looking up
        canonical strings directly from token IDs via precomputed numpy arrays.
        Falls back to regular align() if precomputed maps are not available.
        """
        if self._student_canon_map is None:
            return self.align(student_ids, teacher_ids,
                              exact_match_score=exact_match_score,
                              combination_score_multiplier=combination_score_multiplier,
                              gap_penalty=gap_penalty,
                              chunk_size=chunk_size,
                              post_process=post_process,
                              anchor_lengths=anchor_lengths,
                              ignore_leading_char_diff=ignore_leading_char_diff)

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
            s_canon_strs = [self._canon_id_to_str[self._student_canon_map[tid]] for tid in s_ids]
            t_canon_strs = [self._canon_id_to_str[self._teacher_canon_map[tid]] for tid in t_ids]
            _tl1 = _time.time()
            _t_lookup_total += _tl1 - _tl0

            align_kwargs = {
                'exact_match_score': exact_match_score,
                'combination_score_multiplier': combination_score_multiplier,
                'gap_penalty': gap_penalty,
                'max_combination_len': self.max_combination_len,
                'ignore_leading_char_diff': False,
                'chunk_size': chunk_size,
                'anchor_lengths': anchor_lengths,
            }

            aligned_pairs, _ = self._align_with_anchors(s_canon_strs, t_canon_strs, **align_kwargs)
            _tl2 = _time.time()
            _t_anchors_dp_total += _tl2 - _tl1

            if post_process:
                aligned_pairs = self.post_process_alignment_optimized(
                    aligned_pairs,
                    ignore_leading_char_diff=ignore_leading_char_diff,
                    exact_match_score=exact_match_score,
                    combination_score_multiplier=combination_score_multiplier,
                    gap_penalty=gap_penalty,
                    max_combination_len=self.max_combination_len
                )
            _tl3 = _time.time()
            _t_postprocess_total += _tl3 - _tl2

            mask = self.get_alignment_mask(aligned_pairs, use_canonicalization=True,
                                           ignore_leading_char_diff=ignore_leading_char_diff)
            aligned_pairs = [
                (s1_tokens, s2_tokens, s1_start, s1_end, s2_start, s2_end, mask_value)
                for (s1_tokens, s2_tokens, s1_start, s1_end, s2_start, s2_end), mask_value
                in zip(aligned_pairs, mask)
            ]
            _tl4 = _time.time()
            _t_mask_total += _tl4 - _tl3

            all_aligned_pairs.append(aligned_pairs)

        n = student_ids.shape[0]
        _t_total = _t_lookup_total + _t_anchors_dp_total + _t_postprocess_total + _t_mask_total
        if _t_total > 0.5 or n > 1:
            print(f"    [align_fast timing] lookup={_t_lookup_total:.3f}s, "
                  f"anchors+DP={_t_anchors_dp_total:.3f}s, "
                  f"postprocess={_t_postprocess_total:.3f}s, "
                  f"mask={_t_mask_total:.3f}s, "
                  f"total={_t_total:.3f}s (n={n})", flush=True)

        return all_aligned_pairs

    def _load_logits_projection_map(
            self,
            folder_location: str = "cross_tokenizer_data",
            file_path: str = None,
            top_k: int = 100,
            device: str = "cuda",
            use_sparse_format: bool = False,
            learnable: bool = False,
        ):
        """
        Load projection map for cross-tokenizer likelihood projection.
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
                raise FileNotFoundError(f"Sparse transformation matrix file not found: {file_path}. Please generate it first.")
            
            # Load transformation counts dictionary
            transformation_counts = torch.load(file_path, map_location='cpu', weights_only=False)
            
            # Get tokenizer vocab sizes
            teacher_vocab_size = len(self.teacher_tokenizer) if self.teacher_tokenizer else 151669  # fallback
            student_vocab_size = len(self.student_tokenizer) if self.student_tokenizer else 128256  # fallback
            if 1:
                # get vocab sizes from autoconfig
                if "gemma" not in self.teacher_tokenizer_name.lower() and "qwen3.5" not in self.teacher_tokenizer_name.lower():
                    teacher_vocab_size = AutoConfig.from_pretrained(self.teacher_tokenizer_name).vocab_size
                else:
                    teacher_vocab_size = AutoConfig.from_pretrained(self.teacher_tokenizer_name).text_config.vocab_size
                if "gemma" not in self.student_tokenizer_name.lower() and "qwen3.5" not in self.student_tokenizer_name.lower():
                    student_vocab_size = AutoConfig.from_pretrained(self.student_tokenizer_name).vocab_size
                else:
                    student_vocab_size = AutoConfig.from_pretrained(self.student_tokenizer_name).text_config.vocab_size
                # teacher_vocab_size = AutoConfig.from_pretrained(self.teacher_tokenizer_name).vocab_size
                # student_vocab_size = AutoConfig.from_pretrained(self.student_tokenizer_name).vocab_size

            
            # Debug vocab sizes
            print(f"Teacher vocab size: {teacher_vocab_size}, Student vocab size: {student_vocab_size}")
            
            # Convert dictionary to sparse tensor
            if transformation_counts:
                
                
                indices = list(transformation_counts.keys())
                values = list(transformation_counts.values())
                
                student_indices = [idx[0] for idx in indices]
                teacher_indices = [idx[1] for idx in indices]
                
                # Always create student→teacher mapping: rows = student vocab, cols = teacher vocab
                indices_tensor = torch.LongTensor([student_indices, teacher_indices])
                values_tensor = torch.FloatTensor(values)/self.projection_matrix_multiplier
                matrix_shape = (student_vocab_size, teacher_vocab_size)
                
                print(f"Creating sparse matrix: student→teacher ({student_vocab_size} x {teacher_vocab_size})")
                                
                sparse_transformation_matrix = torch.sparse_coo_tensor(
                    indices_tensor, 
                    values_tensor, 
                    (student_vocab_size, teacher_vocab_size),  # student_vocab × teacher_vocab
                    device=device,
                    dtype=torch.float32
                )
                
                # Optionally make the sparse matrix learnable (values only)
                if learnable:
                    self.sparse_transformation_matrix = nn.Parameter(
                        sparse_transformation_matrix.coalesce(), requires_grad=True
                    )
                else:
                    # Register as buffer for non-learnable parameters (ensures proper device handling)
                    self.register_buffer('sparse_transformation_matrix', 
                                       sparse_transformation_matrix.coalesce(), 
                                       persistent=True)
                
                # Store a flag for downstream code
                self.is_sparse_learnable = learnable
                print(f"Loaded sparse transformation matrix with {len(transformation_counts)} entries")
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
                    self.sparse_transformation_matrix = nn.Parameter(empty_sparse, requires_grad=True)
                else:
                    # Register as buffer for non-learnable parameters
                    self.register_buffer('sparse_transformation_matrix', empty_sparse, persistent=True)

                self.is_sparse_learnable = learnable
                print("Warning: Empty transformation matrix loaded")
        else:
            # Load old dense indices/values format
            if file_path is None:
                file_path = f"{folder_location}/projection_map_Llama-3.1_to_Qwen3_bidirectional_top_10.pt"
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Projection map file not found: {file_path}. Please generate it first.")
                
            projection_data = torch.load(file_path, map_location='cpu', weights_only=False)
            # Always use B_to_A direction for student->teacher projection
            # projection_data = projection_data["B_to_A"]
            # projection_data = projection_data["A_to_B"]
            
            indices     = projection_data["indices"]
            likelihoods = projection_data["likelihoods"]/self.projection_matrix_multiplier
            
            # Register indices as buffer (always non-learnable)
            self.register_buffer('likelihood_projection_indices', indices.to(device), persistent=True)
            if learnable:
                if 1:
                    likelihoods = (likelihoods+1e-10).log()
                    
                # Use instance variable if set, otherwise use default (False)
                # scale_trick_enabled = self.enable_scale_trick if self.enable_scale_trick is not None else False
                
                # if scale_trick_enabled:
                #     #trick with last column being multiplier - set to -4.0
                #     likelihoods[:,-1] = likelihoods[:,-1]*0.0 - 4.0
                #lets introduce some noise to encourage training. will remove later.
                if 0:
                    likelihoods = likelihoods + torch.randn_like(likelihoods) * 1e-1
                    likelihoods = likelihoods/2.0
                
                self.likelihood_projection_matrix = nn.Parameter(likelihoods.to(device), requires_grad=True)
                # print(self.likelihood_projection_matrix[0])
                # print(self.likelihood_projection_matrix[:,-1])
                # exit()
                #add small gaussian noise to the projection matrix
                #use log form
            else:
                # Register as buffer for non-learnable parameters
                self.register_buffer('likelihood_projection_matrix', likelihoods.to(device), persistent=True)
            
            
            print(f"Loaded dense projection map with shape {indices.shape}")
            # Invalidate cached CSR; will rebuild on first use
            self._dense_proj_csr = None
            self._dense_proj_csr_device = None
    
    def create_reverse_projection_matrix(self, device="cuda"):
        """
        Create a reverse (transposed) projection matrix for teacher→student projection.
        
        For sparse format: Transposes the sparse_transformation_matrix from [student_vocab, teacher_vocab] 
                          to [teacher_vocab, student_vocab]
        For dense format: Builds a reverse index mapping from teacher tokens to student tokens
        
        This enables projecting teacher logits into student vocabulary space.
        """
        if hasattr(self, 'sparse_transformation_matrix') and self.sparse_transformation_matrix is not None:
            # Transpose sparse matrix
            print("Creating reverse projection matrix (sparse format): teacher→student")
            sparse_matrix = self.sparse_transformation_matrix.coalesce()
            indices = sparse_matrix.indices()
            values = sparse_matrix.values()
            
            # Swap student and teacher indices (transpose)
            transposed_indices = torch.stack([indices[1], indices[0]], dim=0)  # Swap rows: [teacher, student]
            teacher_vocab_size, student_vocab_size = sparse_matrix.shape[1], sparse_matrix.shape[0]
            
            reverse_sparse = torch.sparse_coo_tensor(
                transposed_indices,
                values,
                (teacher_vocab_size, student_vocab_size),
                device=device,
                dtype=torch.float32
            ).coalesce()
            
            # Store as buffer or parameter based on learnability
            if self.is_sparse_learnable:
                self.reverse_sparse_transformation_matrix = nn.Parameter(reverse_sparse, requires_grad=True)
            else:
                self.register_buffer('reverse_sparse_transformation_matrix', reverse_sparse, persistent=True)
            
            print(f"Created reverse sparse matrix: teacher→student ({teacher_vocab_size} x {student_vocab_size})")
            print(f"Reverse matrix has {len(values)} non-zero entries")
            
        elif hasattr(self, 'likelihood_projection_indices') and self.likelihood_projection_indices is not None:
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
                    teacher_idx = self.likelihood_projection_indices[student_idx, k].item()
                    if hasattr(self, 'likelihood_projection_matrix'):
                        value = self.likelihood_projection_matrix[student_idx, k].item()
                    else:
                        value = 1.0  # Default value if no matrix
                    
                    # Check for valid entries: teacher_idx must be valid, and value must be finite (not -inf)
                    # If matrix is in log-space, valid log-probs are finite negative values
                    # Threshold at -20 to filter out padding values like -22.3197
                    if teacher_idx >= 0 and value > -20.0:  # Skip invalid or padding entries
                        teacher_to_students[teacher_idx].append((student_idx, value))
            
            # Find max number of students mapping to any teacher token
            raw_max_students = max([len(v) for v in teacher_to_students.values()]) if teacher_to_students else 1
            print(f"Max students mapping to any teacher token (before filtering): {raw_max_students}")
            
            # Limit to top-K students per teacher token to avoid explosion
            # Keep only the top-K highest probability mappings per teacher
            max_students_per_teacher = min(topk, raw_max_students)  # Use same topk as forward direction
            print(f"Limiting to top-{max_students_per_teacher} students per teacher token")
            
            # Sort each teacher's student list by value (descending) and keep only top-K
            for teacher_idx in teacher_to_students:
                student_list = teacher_to_students[teacher_idx]
                # Sort by value (descending - higher log-prob = less negative)
                student_list_sorted = sorted(student_list, key=lambda x: x[1], reverse=True)
                teacher_to_students[teacher_idx] = student_list_sorted[:max_students_per_teacher]
            
            # Create dense reverse index [teacher_vocab, max_students_per_teacher]
            # Use 0 instead of -1 for padding (valid index), with very negative values to nullify contribution
            reverse_indices = torch.zeros((teacher_vocab_size, max_students_per_teacher), 
                                        dtype=torch.long, device=device)
            # Initialize with very negative values (padding sentinel, similar to forward direction)
            reverse_values = torch.full((teacher_vocab_size, max_students_per_teacher), -22.3197,
                                        dtype=torch.float32, device=device)
            
            for teacher_idx, student_list in teacher_to_students.items():
                for k, (student_idx, value) in enumerate(student_list):
                    reverse_indices[teacher_idx, k] = student_idx
                    reverse_values[teacher_idx, k] = value
            
            print(f"Created reverse dense projection: teacher→student ({teacher_vocab_size} x {max_students_per_teacher})")
            
            # Store as buffer or parameter
            self.register_buffer('reverse_likelihood_projection_indices', reverse_indices, persistent=True)
            if self.learnable:
                self.reverse_likelihood_projection_matrix = nn.Parameter(reverse_values, requires_grad=True)
            else:
                self.register_buffer('reverse_likelihood_projection_matrix', reverse_values, persistent=True)
            
            print(f"Created reverse dense projection: teacher→student ({teacher_vocab_size} x {max_students_per_teacher})")
        else:
            raise ValueError("No projection matrix loaded. Cannot create reverse projection.")
    
    def project_token_likelihoods_instance(self, input_likelihoods, projection_map_indices, projection_map_values, target_vocab_size, device, use_sparse_format=False, sparse_matrix=None, use_vectorized=True, gpu_optimized_scatter=True, global_top_indices=None):
        """
        Instance method wrapper for project_token_likelihoods that can access instance variables.
        
        Args:
            global_top_indices: Optional tensor of shape (K,) containing indices of tokens to project to.
                               If provided, only projects to these K tokens instead of full target_vocab_size.
                               Results in (batch, seq, K) output instead of (batch, seq, target_vocab_size).
        """
        if use_sparse_format:
            if sparse_matrix is None:
                raise ValueError("sparse_matrix must be provided when use_sparse_format=True")
            
            if global_top_indices is not None:
                # For sparse format with global_top_indices, project to full vocab then slice
                full_projection = TokenAligner.project_token_likelihoods_sparse(input_likelihoods, sparse_matrix*self.projection_matrix_multiplier, device)
                return full_projection[:, :, global_top_indices]
            else:
                return TokenAligner.project_token_likelihoods_sparse(input_likelihoods, sparse_matrix*self.projection_matrix_multiplier, device)
        else:
            # If projection map is learnable, fall back to dense scatter path to preserve gradients
            if getattr(projection_map_values, "requires_grad", False):
                scale_trick_enabled = self.enable_scale_trick if self.enable_scale_trick is not None else False
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
                crow_indices = torch.arange(0, (vs + 1) * top_k, top_k, device=device, dtype=torch.long)
                col_indices = idx.reshape(-1)
                values = val.reshape(-1)
                # _exact_map_remapped matrices use -1 as a padding sentinel for missing
                # entries.  A -1 column index is illegal in a CSR tensor and causes a
                # CUDA illegal-memory-access during the subsequent matmul.  Clamp those
                # entries to column 0 and zero their value so they contribute nothing.
                pad_mask = col_indices < 0
                if pad_mask.any():
                    col_indices = col_indices.clone()
                    col_indices[pad_mask] = 0
                    values = values.clone()
                    values[pad_mask] = 0.0
                proj_csr = torch.sparse_csr_tensor(
                    crow_indices, col_indices, values, size=(vs, target_vocab_size), device=device
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
            out = out2d.reshape(bsz, seqlen, target_vocab_size).to(input_likelihoods.dtype)
            return out
    
    @staticmethod
    def project_token_likelihoods_dense(input_likelihoods, projection_map_indices, projection_map_values, target_vocab_size, device, use_vectorized=True, gpu_optimized_scatter=True, enable_scale_trick=None, global_top_indices=None):
        """
        Projects token likelihoods from a source to a target vocabulary using dense indices/values format.
        
        Args:
            global_top_indices: Optional tensor of shape (K,) containing indices of target tokens to project to.
                               If provided, only projects to these K tokens instead of full target_vocab_size.
                               Results in (batch, seq, K) output instead of (batch, seq, target_vocab_size).
                               MAJOR SPEEDUP: Reduces both memory and compute significantly.
        """
        batch_size, seq_len, source_vocab_size = input_likelihoods.shape
        if abs(source_vocab_size - projection_map_indices.shape[0]) > 1000:
            raise ValueError(f"Source vocab size of input ({source_vocab_size}) mismatches projection map size ({projection_map_indices.shape[0]})")

        top_k = projection_map_indices.shape[1]
        input_likelihoods = input_likelihoods.to(device)
        if projection_map_indices.device != device:
            projection_map_indices = projection_map_indices.to(device)
        if projection_map_values.device != device:
            projection_map_values = projection_map_values.to(device)
        #do for dtype
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
                full_to_reduced_map = torch.full((target_vocab_size,), -1, device=device, dtype=torch.long)
                full_to_reduced_map[global_top_indices] = torch.arange(k_indices, device=device)
                
                # Initialize smaller output tensor - MAJOR MEMORY SAVINGS
                projected_likelihoods = torch.zeros(batch_size, seq_len, k_indices, 
                                                  device=device, dtype=input_likelihoods.dtype)
                effective_vocab_size = k_indices
                
                # Filter projection matrices to only include mappings to global_top_indices
                # This will be used in the scatter operations below
                use_reduced_projection = True
            else:
                # Initialize full output tensor
                projected_likelihoods = torch.zeros(batch_size, seq_len, target_vocab_size, 
                                                  device=device, dtype=input_likelihoods.dtype)
                effective_vocab_size = target_vocab_size
                use_reduced_projection = False
            
            # Optimized chunked processing with multiple speedup techniques
            # Use larger chunks for better amortization of fixed costs
            max_memory_mb = 200  # Increased for better performance
            # max_memory_mb = 500  # Increased for better performance
            elements_per_chunk = max_memory_mb * 1024 * 1024 // 4  # 4 bytes per float32  
            chunk_size = max(512, min(source_vocab_size_fixed, elements_per_chunk // (batch_size * seq_len)))
            
            
            use_masking = False
            # Process vocabulary in optimized chunks
            for chunk_start in range(0, source_vocab_size_fixed, chunk_size):
                chunk_end = min(chunk_start + chunk_size, source_vocab_size_fixed)
                chunk_len = chunk_end - chunk_start
                
                
                input_chunk = input_likelihoods_fixed[:, :, chunk_start:chunk_end]  # (B, S, chunk_len)
                indices_chunk = projection_map_indices[chunk_start:chunk_end, :]    # (chunk_len, top_k)
                values_chunk = projection_map_values[chunk_start:chunk_end, :]      # (chunk_len, top_k)
                
                # Extract input chunk once per chunk (not per k) - major speedup
                # Determine effective top_k (exclude last column if scale trick is enabled)
                scale_trick_enabled = enable_scale_trick if enable_scale_trick is not None else False
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
                            
                            source_indices_k = torch.nonzero(valid_mask_k, as_tuple=True)[0]
                            
                            input_subset_k = input_chunk[:, :, source_indices_k]
                            values_subset_k = values_k[source_indices_k]
                            
                            indices_k = indices_chunk[:, k]
                            target_indices_subset_k = indices_k[source_indices_k]
                            
                            weighted_inputs_k = input_subset_k * values_subset_k.view(1, 1, -1)
                            expanded_target_indices_k = target_indices_subset_k.view(1, 1, -1).expand(batch_size, seq_len, -1)
                            
                            projected_likelihoods.scatter_add_(2, expanded_target_indices_k, weighted_inputs_k)
                    else:
                        # Compact, un-masked implementation
                        # Process only effective columns without creating intermediate tensors
                        input_expanded = input_chunk.unsqueeze(-1)  # (B, S, chunk_len, 1)
                        
                        for k in range(effective_top_k):
                            values_k = values_chunk[:, k:k+1]  # (chunk_len, 1) - view, no copy
                            indices_k = indices_chunk[:, k]     # (chunk_len,)
                            
                            if use_reduced_projection:
                                # OPTIMIZATION: Only project to indices in global_top_indices
                                # Map full indices to reduced indices and filter out invalid ones
                                reduced_indices_k = full_to_reduced_map[indices_k]  # (chunk_len,)
                                valid_mask = reduced_indices_k != -1  # Only keep indices in global_top_indices
                                
                                if not valid_mask.any():
                                    continue  # Skip if no valid indices in this chunk
                                    
                                # Filter to only valid entries - MAJOR COMPUTE SAVINGS
                                valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]
                                reduced_indices_filtered = reduced_indices_k[valid_indices]
                                values_filtered = values_k.squeeze(-1)[valid_indices]  # (valid_count,)
                                input_filtered = input_chunk[:, :, valid_indices]  # (B, S, valid_count)
                                
                                weighted_k = input_filtered * values_filtered.unsqueeze(0).unsqueeze(0)
                                indices_expanded = reduced_indices_filtered.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
                                projected_likelihoods.scatter_add_(2, indices_expanded, weighted_k)
                            else:
                                # Standard full projection
                                weighted_k = input_expanded * values_k.unsqueeze(0).unsqueeze(0)  # (B, S, chunk_len, 1)
                                weighted_k = weighted_k.squeeze(-1)  # (B, S, chunk_len)
                                
                                indices_expanded = indices_k.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
                                projected_likelihoods.scatter_add_(2, indices_expanded, weighted_k)
                else:
                    # Original implementation with a loop over top_k
                    if True:  # For small top_k, process all k together
                        # Broadcast input: (B, S, chunk_len, 1) * (1, 1, chunk_len, top_k) -> (B, S, chunk_len, top_k)
                        weighted_inputs = input_chunk.unsqueeze(-1) * values_chunk.unsqueeze(0).unsqueeze(0)
                        
                        # Process all k simultaneously using advanced indexing
                        for k in range(effective_top_k):
                            target_indices_k = indices_chunk[:, k]  # (chunk_len,)
                            weighted_k = weighted_inputs[:, :, :, k]  # (B, S, chunk_len)
                            
                            if use_reduced_projection:
                                # OPTIMIZATION: Only project to indices in global_top_indices
                                reduced_indices_k = full_to_reduced_map[target_indices_k]  # (chunk_len,)
                                valid_mask = reduced_indices_k != -1
                                
                                if not valid_mask.any():
                                    continue  # Skip if no valid indices
                                
                                # Filter to only valid entries - MAJOR COMPUTE SAVINGS
                                valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]
                                reduced_indices_filtered = reduced_indices_k[valid_indices]
                                weighted_filtered = weighted_k[:, :, valid_indices]  # (B, S, valid_count)
                                
                                target_expanded = reduced_indices_filtered.view(1, 1, -1).expand(batch_size, seq_len, len(valid_indices))
                                projected_likelihoods.scatter_add_(2, target_expanded, weighted_filtered)
                            else:
                                # Use optimized scatter with pre-expanded indices (avoid .expand() in loop)
                                target_expanded = target_indices_k.view(1, 1, -1).expand(batch_size, seq_len, chunk_len)
                                projected_likelihoods.scatter_add_(2, target_expanded, weighted_k)
                        
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
            crow_indices = torch.arange(0, (source_vocab_size_fixed + 1) * top_k, top_k, device=device, dtype=torch.long)
            col_indices = projection_map_indices.flatten()
            values = projection_map_values.flatten()
            
            sparse_projection_matrix = torch.sparse_csr_tensor(
                crow_indices, col_indices, values, size=(source_vocab_size_fixed, target_vocab_size), device=device
            )
            
            # Apply sparse matrix multiplication
            input_likelihoods_fixed = input_likelihoods[:, :, :source_vocab_size_fixed]
            reshaped_input = input_likelihoods_fixed.reshape(batch_size * seq_len, source_vocab_size)
            
            projected_likelihoods_reshaped = torch.matmul(reshaped_input.to(torch.float32), sparse_projection_matrix.to(torch.float32))
            
            return projected_likelihoods_reshaped.reshape(batch_size, seq_len, target_vocab_size).to(input_likelihoods.dtype)
    
    @staticmethod
    def project_token_likelihoods_sparse(input_likelihoods, sparse_matrix, device):
        """Projects token likelihoods using a sparse transformation matrix."""
        batch_size, seq_len, source_vocab_size = input_likelihoods.shape
        
        # Get dimensions from sparse matrix
        matrix_input_size, matrix_output_size = sparse_matrix.shape
        
        if abs(source_vocab_size - matrix_input_size) > 1000:
            raise ValueError(f"Source vocab size of input ({source_vocab_size}) mismatches sparse matrix input size ({matrix_input_size})")
        
        # Move to correct device and dtype
        # input_likelihoods = input_likelihoods.to(device)
        # sparse_matrix = sparse_matrix.to(device)
        
        # Adjust input size to match matrix dimensions
        # next 2 lines required when we used vocab length from tokenizer, now we use the size of logits
        # source_vocab_size_fixed = min(source_vocab_size, matrix_input_size)
        # input_likelihoods_fixed = input_likelihoods[:, :, :source_vocab_size_fixed]
        input_likelihoods_fixed = input_likelihoods
        
        # Reshape for matrix multiplication
        reshaped_input = input_likelihoods_fixed.reshape(batch_size * seq_len, source_vocab_size)
        
        # Project using sparse matrix multiplication
        projected_likelihoods_reshaped = torch.matmul(reshaped_input.to(torch.float32), sparse_matrix.to(torch.float32))
        
        # Reshape back to original format
        return projected_likelihoods_reshaped.reshape(batch_size, seq_len, matrix_output_size).to(input_likelihoods.dtype)

    def align(self, student_seq: Union[List[str], List[List[str]], List[int], List[List[int]]],
              teacher_seq: Union[List[str], List[List[str]], List[int], List[List[int]]],
              exact_match_score=3,
              combination_score_multiplier=1.5,
              gap_penalty=-1.5,
              ignore_leading_char_diff=False,
              chunk_size=128,
              post_process=True,
              convert_ids_to_tokens=True,
              anchor_lengths=[3,],
              _debug_timing=False):
        """Align two sequences of tokens (or batches of sequences)."""
        import time as _time

        seq1 = student_seq
        seq2 = teacher_seq

        _t_convert = 0.0
        if isinstance(seq1, torch.Tensor):
            seq1 = seq1.cpu().tolist()
            seq2 = seq2.cpu().tolist()
            if convert_ids_to_tokens:
                _tc0 = _time.time()
                seq1 = [self.student_tokenizer.convert_ids_to_tokens(seq1_single) for seq1_single in seq1]
                seq2 = [self.teacher_tokenizer.convert_ids_to_tokens(seq2_single) for seq2_single in seq2]
                _t_convert = _time.time() - _tc0

        is_batched = isinstance(seq1, list) and len(seq1) > 0 and isinstance(seq1[0], list)

        _t_canon_total = 0.0
        _t_anchors_dp_total = 0.0
        _t_postprocess_total = 0.0
        _t_mask_total = 0.0

        if is_batched:
            if not (isinstance(seq2, list) and len(seq2) == len(seq1) and (len(seq2) == 0 or isinstance(seq2[0], list))):
                 raise ValueError("For batched input, seq1 and seq2 must be lists of lists with the same length.")

            all_aligned_pairs = []
            for s1, s2 in zip(seq1, seq2):
                aligned_pairs, timings = self._align_single(s1, s2, exact_match_score, combination_score_multiplier, gap_penalty, ignore_leading_char_diff, chunk_size, post_process, anchor_lengths, _return_timings=True)
                all_aligned_pairs.append(aligned_pairs)
                _t_canon_total += timings.get("canon", 0)
                _t_anchors_dp_total += timings.get("anchors_dp", 0)
                _t_postprocess_total += timings.get("postprocess", 0)
                _t_mask_total += timings.get("mask", 0)
        else:
            aligned_pairs, timings = self._align_single(seq1, seq2, exact_match_score, combination_score_multiplier, gap_penalty, ignore_leading_char_diff, chunk_size, post_process, anchor_lengths, _return_timings=True)
            all_aligned_pairs = [aligned_pairs]
            _t_canon_total += timings.get("canon", 0)
            _t_anchors_dp_total += timings.get("anchors_dp", 0)
            _t_postprocess_total += timings.get("postprocess", 0)
            _t_mask_total += timings.get("mask", 0)

        if _debug_timing:
            n = len(all_aligned_pairs)
            print(f"    [align timing] convert_ids={_t_convert:.3f}s, "
                  f"canonicalize={_t_canon_total:.3f}s, "
                  f"anchors+DP={_t_anchors_dp_total:.3f}s, "
                  f"postprocess={_t_postprocess_total:.3f}s, "
                  f"mask={_t_mask_total:.3f}s "
                  f"(n={n})", flush=True)

        return all_aligned_pairs 

    def _align_single(self, seq1, seq2,
            exact_match_score=3,
            combination_score_multiplier=1.5,
            gap_penalty=-1.5,
            ignore_leading_char_diff=True,
            chunk_size=0,
            post_process=True,
            anchor_lengths=None,
            _return_timings=False):
        """Align two sequences of tokens."""
        import time as _time

        _tc0 = _time.time()
        seq1_canon = TokenAligner._canonicalize_sequence(seq1)
        seq2_canon = TokenAligner._canonicalize_sequence(seq2)
        _tc1 = _time.time()

        align_kwargs = {
            'exact_match_score': exact_match_score,
            'combination_score_multiplier': combination_score_multiplier,
            'gap_penalty': gap_penalty,
            'max_combination_len': self.max_combination_len,
            'ignore_leading_char_diff': False,
            'chunk_size': chunk_size,
            'anchor_lengths': anchor_lengths,
        }

        aligned_pairs, _ = self._align_with_anchors(seq1_canon, seq2_canon, **align_kwargs)
        _tc2 = _time.time()

        if post_process:
            aligned_pairs = self.post_process_alignment_optimized(
                aligned_pairs,
                ignore_leading_char_diff=ignore_leading_char_diff,
                exact_match_score=exact_match_score,
                combination_score_multiplier=combination_score_multiplier,
                gap_penalty=gap_penalty,
                max_combination_len=self.max_combination_len
            )
        _tc3 = _time.time()

        mask = self.get_alignment_mask(aligned_pairs, use_canonicalization=True, ignore_leading_char_diff=ignore_leading_char_diff)
        aligned_pairs = [
            (s1_tokens, s2_tokens, s1_start, s1_end, s2_start, s2_end, mask_value)
            for (s1_tokens, s2_tokens, s1_start, s1_end, s2_start, s2_end), mask_value in zip(aligned_pairs, mask)
        ]
        _tc4 = _time.time()

        timings = {
            "canon": _tc1 - _tc0,
            "anchors_dp": _tc2 - _tc1,
            "postprocess": _tc3 - _tc2,
            "mask": _tc4 - _tc3,
        }

        if _return_timings:
            return aligned_pairs, timings
        return aligned_pairs
    
    
    def _align_with_anchors(self, seq1, seq2, anchor_lengths=[3,], **kwargs):
        """
        Optimized alignment using unique 1-to-1 matches as anchors.
        """
        # CRITICAL FIX: If anchor_lengths is empty, disable anchor optimization completely
        if not anchor_lengths:
            return self._perform_dp_alignment(seq1, seq2, **kwargs)
            
        if anchor_lengths is None:
            anchor_lengths = [3, 2]  # Default: check 3-token, then 2-token sequences
        
        # Debug output
        debug = kwargs.get('debug', False)
        
        # 1. Find high-confidence anchor points using unique token matches.
        s1_counts = {}
        for i, t in enumerate(seq1):
            if t not in s1_counts: s1_counts[t] = []
            s1_counts[t].append(i)
        
        s2_counts = {}
        for i, t in enumerate(seq2):
            if t not in s2_counts: s2_counts[t] = []
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
                    ngram = tuple(seq1[i:i + anchor_len])
                    if ngram not in s1_ngram_counts:
                        s1_ngram_counts[ngram] = []
                    s1_ngram_counts[ngram].append(i)
                
                s2_ngram_counts = {}
                for i in range(len(seq2) - anchor_len + 1):
                    ngram = tuple(seq2[i:i + anchor_len])
                    if ngram not in s2_ngram_counts:
                        s2_ngram_counts[ngram] = []
                    s2_ngram_counts[ngram].append(i)
                
                # Find n-grams that appear exactly once in both sequences
                common_ngrams = s1_ngram_counts.keys() & s2_ngram_counts.keys()
                for ngram in common_ngrams:
                    if len(s1_ngram_counts[ngram]) == 1 and len(s2_ngram_counts[ngram]) == 1:
                        i = s1_ngram_counts[ngram][0]
                        j = s2_ngram_counts[ngram][0]
                        # ADDED: Verify the anchor is actually correct
                        if (i + anchor_len <= len(seq1) and j + anchor_len <= len(seq2) and
                            seq1[i:i + anchor_len] == seq2[j:j + anchor_len]):
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
            
            if not (seq1_range & used_positions_seq1) and not (seq2_range & used_positions_seq2):
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
                if (i + anchor_len <= len(seq1) and j + anchor_len <= len(seq2) and
                    seq1[i:i + anchor_len] == seq2[j:j + anchor_len]):
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
                for s1_toks, s2_toks, s1_start, s1_end, s2_start, s2_end in aligned_segment:
                    new_s1_start = s1_start + last_i if s1_start != -1 else -1
                    new_s1_end = s1_end + last_i if s1_end != -1 else -1
                    new_s2_start = s2_start + last_j if s2_start != -1 else -1
                    new_s2_end = s2_end + last_j if s2_end != -1 else -1
                    
                    # Split if both sides have the same tokens
                    if (len(s1_toks) > 1 and len(s2_toks) > 1 and 
                        len(s1_toks) == len(s2_toks) and s1_toks == s2_toks):
                        # Split into individual 1-to-1 matches
                        for k in range(len(s1_toks)):
                            full_alignment.append((
                                [s1_toks[k]], [s2_toks[k]], 
                                new_s1_start + k, new_s1_start + k + 1,
                                new_s2_start + k, new_s2_start + k + 1
                            ))
                    else:
                        full_alignment.append((s1_toks, s2_toks, new_s1_start, new_s1_end, new_s2_start, new_s2_end))
            
            # Add the anchor itself (consecutive tokens), also split if needed.
            anchor_seq1 = seq1[i:i + anchor_len]
            anchor_seq2 = seq2[j:j + anchor_len]
            
            # Split anchor into individual matches since they should be identical
            for k in range(anchor_len):
                full_alignment.append((
                    [anchor_seq1[k]], [anchor_seq2[k]], 
                    i + k, i + k + 1, 
                    j + k, j + k + 1
                ))
            
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
                if (len(s1_toks) > 1 and len(s2_toks) > 1 and 
                    len(s1_toks) == len(s2_toks) and s1_toks == s2_toks):
                    # Split into individual 1-to-1 matches
                    for k in range(len(s1_toks)):
                        full_alignment.append((
                            [s1_toks[k]], [s2_toks[k]], 
                            new_s1_start + k, new_s1_start + k + 1,
                            new_s2_start + k, new_s2_start + k + 1
                        ))
                else:
                    full_alignment.append((s1_toks, s2_toks, new_s1_start, new_s1_end, new_s2_start, new_s2_end))

        return full_alignment, 0 # Return 0 for score as it's not well-defined here

    def _perform_dp_alignment(self, seq1, seq2, **kwargs):
        """
        Helper function to run the core DP-based alignment.
        """
        chunk_size = kwargs.get('chunk_size', 0)
        kwargs.pop('chunk_size', None)
        kwargs.pop('anchor_lengths', None)

        if chunk_size > 0:
            return self.align_tokens_combinations_chunked(seq1, seq2, chunk_size=chunk_size, **kwargs)
        else:
            return self.align_tokens_with_combinations_numpy_jit(seq1, seq2, **kwargs)

    @staticmethod
    def _canonical_token(token: str) -> str:
        """Return a canonical representation of a tokenizer token."""
        if not token:
            return token
        
        # 1. Normalize space prefixes first
        if token.startswith(' '):
            token = 'Ġ' + token[1:]
        elif token.startswith('_'):
            token = 'Ġ' + token[1:]
        elif token.startswith('▁'):  # SentencePiece-style space prefix
            token = 'Ġ' + token[1:]
        
        # 1.5. Normalize newline and whitespace representations
        if token == 'Ċ':  # GPT-style newline (used by Llama)
            token = '\n'
        elif token == '\\n':  # Escaped newline representation
            token = '\n'
        elif token == 'ĉ':  # Alternative newline representation
            token = '\n'
        elif token == 'Ġ\n':  # Space + newline combination
            token = '\n'
        elif 'Ċ' in token:  # Handle Ċ embedded in other tokens
            token = token.replace('Ċ', '\n')
        elif '\\n' in token:  # Handle escaped newlines in compound tokens
            token = token.replace('\\n', '\n')
        
        # 1.6. Handle space-separated punctuation normalization
        if token == 'Ġ,':  # Space + comma
            token = ','
        elif token == 'Ġ.':  # Space + period
            token = '.'
        elif token == 'Ġ;':  # Space + semicolon
            token = ';'
        elif token == 'Ġ:':  # Space + colon
            token = ':'
        
        # 2. Handle SentencePiece byte fallback tokens like <0x20>
        if token.startswith('<0x') and token.endswith('>') and len(token) == 6:
            try:
                byte_val = int(token[3:5], 16)
                if 0 <= byte_val <= 255:
                    return chr(byte_val)
            except ValueError:
                pass
        
        # 3. Normalize common Unicode encoding issues
        unicode_fixes = {
            # Spanish
            'Ã±': 'ñ', 'Ã¡': 'á', 'Ã©': 'é', 'Ã­': 'í', 'Ã³': 'ó', 'Ãº': 'ú',
            'Ã': 'À', 'Ã¢': 'â', 'Ã§': 'ç',
            # French  
            'Ã§': 'ç', 'Ã¨': 'è', 'Ã©': 'é', 'Ã«': 'ë', 'Ã®': 'î', 'Ã´': 'ô',
            'Ã¹': 'ù', 'Ã»': 'û', 'Ã¿': 'ÿ',
            # Chinese (common encoding artifacts)
            'ä¸Ń': '中', 'æĸĩ': '文', 'æĹ¥æľ¬': '日本', 'èªŀ': '語',
            # Russian
            'ÐłÑĥÑģ': 'Рус', 'ÑģÐºÐ¸Ð¹': 'ский',
            # Arabic
            'Ø§ÙĦØ¹Ø±Ø¨ÙĬØ©': 'العربية',
            # Hindi
            'à¤¹': 'ह', 'à¤¿à¤Ĥ': 'हिं', 'à¤¦à¥Ģ': 'दी',
            # Mathematical symbols (common artifacts)
            'âĪĳ': '∑', 'âĪı': '∏', 'âĪĤ': '∂', 'âĪĩ': '∇', 
            'âĪŀ': '∞', 'âĪļ': '√', 'âĪ«': '∫', 'âīĪ': '≈',
            'âīł': '≠', 'âī¤': '≤', 'âī¥': '≥',
        }
        
        # Apply Unicode fixes
        for broken, fixed in unicode_fixes.items():
            if broken in token:
                token = token.replace(broken, fixed)
        
        # 4. Normalize special tokens
        special_token_map = {
            '<|begin_of_text|>': '<bos>',  # Llama-style BOS token
            '<bos>': '<bos>',               # Standard BOS token
            '<pad>': '',                    # Padding tokens → empty (will be handled by alignment)
            '': ' ',                        # End tokens
            '': ' ',                        # End tokens
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
            (['ĠâĪ', 'ĳ'], ['Ġ∑']),          # Sum symbol
            (['âĪ', 'ĳ'], ['∑']),            # Sum symbol (no space)
            (['ĠâĪ', 'ı'], ['Ġ∏']),          # Product symbol  
            (['âĪ', 'ı'], ['∏']),            # Product symbol (no space)
            (['ĠâĪ', 'Ĥ'], ['Ġ∂']),          # Partial derivative
            (['âĪ', 'Ĥ'], ['∂']),            # Partial derivative (no space)
            (['ĠâĪ', 'ĩ'], ['Ġ∇']),          # Nabla/gradient
            (['âĪ', 'ĩ'], ['∇']),            # Nabla/gradient (no space)
            (['ĠâĪ', 'ŀ'], ['Ġ∞']),          # Infinity
            (['âĪ', 'ŀ'], ['∞']),            # Infinity (no space)
            (['ĠâĪ', 'ļ'], ['Ġ√']),          # Square root
            (['âĪ', 'ļ'], ['√']),            # Square root (no space)
            (['ĠâĪ', '«'], ['Ġ∫']),          # Integral
            (['âĪ', '«'], ['∫']),            # Integral (no space)
            (['Ġâī', 'ł'], ['Ġ≠']),          # Not equal
            (['âī', 'ł'], ['≠']),            # Not equal (no space)
            # Other common multi-token artifacts
            (['Ġä¸', 'Ń'], ['Ġ中']),         # Chinese character
            (['ä¸', 'Ń'], ['中']),           # Chinese character (no space)
            (['æĸ', 'ĩ'], ['文']),           # Chinese character
            (['Ġæĸ', 'ĩ'], ['Ġ文']),        # Chinese character (with space)
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
                    if tokens[i:i+pattern_len] == pattern:
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
            clean_token = token.lstrip('Ġ')
            
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
            clean_token = token.lstrip('Ġ')
            if len(clean_token) <= 1:
                return byte_tokens
            # Continue processing multi-character token
        
        # Extract space prefix from first token
        first_token = byte_tokens[0]
        space_prefix = 'Ġ' if first_token.startswith('Ġ') else ''
        
        # Extract raw bytes from all characters in all tokens
        raw_bytes = []
        for token in byte_tokens:
            clean_token = token.lstrip('Ġ')
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
            decoded_text = bytes(raw_bytes).decode('utf-8')
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
        'ð': 240, 'Ɩ': 241, 'Ɨ': 242, 'Ƙ': 243, 'ƙ': 244, 'ƚ': 245, 'ƛ': 246, 'Ɯ': 247, 
        'Ɲ': 248, 'ƞ': 249, 'Ɵ': 250, 'Ơ': 251, 'ơ': 252, 'Ƣ': 253, 'ƣ': 254, 'Ƥ': 255,
        # Other common byte representations (0-255 only)
        'Ł': 156, 'ł': 157, 'Ń': 158, 'ń': 159, 'ĺ': 149, 'Ļ': 150, 'ļ': 151, 'Ľ': 152, 
        'ľ': 153, 'Ŀ': 154, 'ŀ': 155, 'Ĭ': 135, 'ĭ': 136, 'Į': 137, 'į': 138, 'İ': 139,
        'ı': 140, 'Ĳ': 141, 'ĳ': 142, 'Ĵ': 143, 'ĵ': 144, 'Ķ': 145, 'ķ': 146, 'ĸ': 147, 
        'Ĺ': 148, 'ĥ': 128, 'Ħ': 129, 'ħ': 130, 'Ĩ': 131, 'ĩ': 132, 'Ī': 133, 'ī': 134,
        'Ģ': 162, 'ģ': 163, 'Ĝ': 28, 'ĝ': 29, 'Ğ': 30, 'ğ': 31,
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

    def align_tokens_with_combinations_numpy(seq1, seq2,
                                            exact_match_score=3,
                                            combination_score_multiplier=1.5,
                                            gap_penalty=-1.5,
                                            max_combination_len=4,
                                            ignore_leading_char_diff=False):
        n1, n2 = len(seq1), len(seq2)
        dp = np.zeros((n1 + 1, n2 + 1), dtype=np.float32)
        trace = np.full((n1 + 1, n2 + 1), '', dtype=object)

        # Initialize DP edges with gap penalties
        for i in range(1, n1 + 1):
            dp[i, 0] = dp[i - 1, 0] + gap_penalty
            trace[i, 0] = 'up'
        for j in range(1, n2 + 1):
            dp[0, j] = dp[0, j - 1] + gap_penalty
            trace[0, j] = 'left'

        # Precompute joined substrings for all valid k-length spans
        joined_seq1 = {(i - k, i): ''.join(seq1[i - k:i])
                    for i in range(n1 + 1)
                    for k in range(1, min(i, max_combination_len) + 1)}
        joined_seq2 = {(j - k, j): ''.join(seq2[j - k:j])
                    for j in range(n2 + 1)
                    for k in range(1, min(j, max_combination_len) + 1)}

        # Fill DP table
        for i in range(1, n1 + 1):
            for j in range(1, n2 + 1):
                s1_val, s2_val = seq1[i - 1], seq2[j - 1]
                match_score = exact_match_score if TokenAligner._strings_equal_flexible(s1_val, s2_val, ignore_leading_char_diff) else -exact_match_score
                score_diag = dp[i - 1, j - 1] + match_score
                score_up = dp[i - 1, j] + gap_penalty
                score_left = dp[i, j - 1] + gap_penalty

                max_score = score_diag
                best_move = 'diag'
                if score_up > max_score:
                    max_score = score_up
                    best_move = 'up'
                if score_left > max_score:
                    max_score = score_left
                    best_move = 'left'

                # Check for seq1[i-1] == join(seq2[j-k:j])
                for k in range(2, min(j + 1, max_combination_len + 1)):
                    if (j - k, j) in joined_seq2 and TokenAligner._strings_equal_flexible(s1_val, joined_seq2[(j - k, j)], ignore_leading_char_diff):
                        comb_score = dp[i - 1, j - k] + combination_score_multiplier * k
                        if comb_score > max_score:
                            max_score = comb_score
                            best_move = f'comb_s1_over_s2_{k}'

                # Check for seq2[j-1] vs seq1[i-k:i]
                for k in range(2, min(i + 1, max_combination_len + 1)):
                    if (i - k, i) in joined_seq1 and TokenAligner._strings_equal_flexible(s2_val, joined_seq1[(i - k, i)], ignore_leading_char_diff):
                        comb_score = dp[i - k, j - 1] + combination_score_multiplier * k
                        if comb_score > max_score:
                            max_score = comb_score
                            best_move = f'comb_s2_over_s1_{k}'

                dp[i, j] = max_score
                trace[i, j] = best_move

        # Backtrack to extract alignment
        aligned = []
        i, j = n1, n2
        while i > 0 or j > 0:
            move = trace[i, j]
            if move == 'diag':
                aligned.append(([seq1[i - 1]], [seq2[j - 1]], i - 1, i, j - 1, j))
                i -= 1
                j -= 1
            elif move == 'up':
                aligned.append(([seq1[i - 1]], [], i - 1, i, -1, -1))
                i -= 1
            elif move == 'left':
                aligned.append(([], [seq2[j - 1]], -1, -1, j - 1, j))
                j -= 1
            elif move.startswith('comb_s1_over_s2_'):
                k = int(move.rsplit('_', 1)[-1])
                aligned.append(([seq1[i - 1]], seq2[j - k:j], i - 1, i, j - k, j))
                i -= 1
                j -= k
            elif move.startswith('comb_s2_over_s1_'):
                k = int(move.rsplit('_', 1)[-1])
                aligned.append((seq1[i - k:i], [seq2[j - 1]], i - k, i, j - 1, j))
                i -= k
                j -= 1
            else:
                break

        aligned.reverse()
        return aligned, dp[n1, n2]

    @staticmethod
    def align_tokens_with_combinations_numpy_jit(
        seq1, seq2,
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
                seq1, seq2, exact_match_score, combination_score_multiplier,
                gap_penalty, max_combination_len, ignore_leading_char_diff,
            )

        n1, n2 = len(seq1), len(seq2)
        if n1 == 0 and n2 == 0:
            return [], 0.0
        if n1 == 0:
            return [([], [seq2[j]], -1, -1, j, j + 1) for j in range(n2)], n2 * gap_penalty
        if n2 == 0:
            return [([seq1[i]], [], i, i + 1, -1, -1) for i in range(n1)], n1 * gap_penalty

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
                joined1[i, k] = _get_id(''.join(seq1[i - k:i]))

        joined2 = np.full((n2 + 1, max_combination_len + 1), INVALID, dtype=np.int64)
        for j in range(n2 + 1):
            for k in range(2, min(j, max_combination_len) + 1):
                joined2[j, k] = _get_id(''.join(seq2[j - k:j]))

        dp, trace = _dp_core_numba(
            ids1, ids2, joined1, joined2, n1, n2,
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
                aligned.append(([seq1[i - 1]], seq2[j - k:j], i - 1, i, j - k, j))
                i -= 1
                j -= k
            elif 20 <= m < 30:
                k = m - 20
                aligned.append((seq1[i - k:i], [seq2[j - 1]], i - k, i, j - 1, j))
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
        """
        Chunked processing for very large sequences.
        """
        n1, n2 = len(seq1), len(seq2)
        
        # If sequences are small enough, use regular algorithm
        if n1 <= chunk_size and n2 <= chunk_size:
            return TokenAligner.align_tokens_with_combinations_numpy_jit(
                seq1, seq2, exact_match_score, combination_score_multiplier,
                gap_penalty, max_combination_len, ignore_leading_char_diff
            )
        
        # For very large sequences, use divide-and-conquer approach
        if n1 > chunk_size or n2 > chunk_size:
            # Find approximate midpoint alignment using simplified algorithm
            mid1, mid2 = n1 // 2, n2 // 2
            
            # Recursively align left and right parts
            left_aligned, left_score = TokenAligner.align_tokens_combinations_chunked(
                seq1[:mid1], seq2[:mid2], exact_match_score, combination_score_multiplier,
                gap_penalty, max_combination_len, ignore_leading_char_diff, 
                chunk_size=chunk_size
            )
            
            right_aligned, right_score = TokenAligner.align_tokens_combinations_chunked(
                seq1[mid1:], seq2[mid2:], exact_match_score, combination_score_multiplier,
                gap_penalty, max_combination_len, ignore_leading_char_diff, 
                chunk_size=chunk_size
            )
            
            # Adjust indices for right part
            adjusted_right = []
            for s1_tokens, s2_tokens, s1_start, s1_end, s2_start, s2_end in right_aligned:
                new_s1_start = s1_start + mid1 if s1_start >= 0 else -1
                new_s1_end = s1_end + mid1 if s1_end >= 0 else -1
                new_s2_start = s2_start + mid2 if s2_start >= 0 else -1
                new_s2_end = s2_end + mid2 if s2_end >= 0 else -1
                adjusted_right.append((s1_tokens, s2_tokens, new_s1_start, new_s1_end, new_s2_start, new_s2_end))
            
            # Combine results
            combined_aligned = left_aligned + adjusted_right
            combined_score = left_score + right_score
            
            return combined_aligned, combined_score
        
        # Fallback to regular algorithm
        return TokenAligner.align_tokens_with_combinations_numpy_jit(
            seq1, seq2, exact_match_score, combination_score_multiplier,
            gap_penalty, max_combination_len, ignore_leading_char_diff
        )

    @staticmethod
    def _combine_consecutive_misaligned_tokens(
        aligned_pairs: List,
        pair_strings: List,
        end_mismatch_threshold: float = 0.2
    ) -> List:
        """
        Combine consecutive misaligned tokens into single chunks to improve alignment.
        
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
            if (i < end_boundary and 
                not pair_strings[i][2] and  # Current pair is misaligned
                i + 1 < len(aligned_pairs)):  # Not the last pair
                
                # Find consecutive misaligned pairs
                consecutive_misaligned = [i]
                j = i + 1
                
                # Look ahead for more consecutive misaligned pairs (up to end boundary)
                while (j < end_boundary and 
                       j < len(aligned_pairs) and 
                       not pair_strings[j][2]):  # Next pair is also misaligned
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
                        s1_tokens, s2_tokens, s1_start, s1_end, s2_start, s2_end, *rest = aligned_pairs[idx]
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
                        combined_s2_end
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
        end_mismatch_threshold: float = 0.2
    ) -> List:
        """
        Optimized version of post_process_alignment with better performance.
        
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
            s1_canonical_tokens = [TokenAligner._canonical_token(tok) for tok in s1_tokens] if s1_tokens else []
            s2_canonical_tokens = [TokenAligner._canonical_token(tok) for tok in s2_tokens] if s2_tokens else []
            s1_str = "".join(s1_canonical_tokens)
            s2_str = "".join(s2_canonical_tokens)
            is_match = TokenAligner._strings_equal_flexible(s1_str, s2_str, ignore_leading_char_diff)
            pair_strings.append((s1_str, s2_str, is_match))

        # Step 1: Handle consecutive misaligned chunks if enabled
        if combine_misaligned_chunks:
            aligned_pairs = TokenAligner._combine_consecutive_misaligned_tokens(
                aligned_pairs, pair_strings, end_mismatch_threshold
            )
            
            # Recompute pair_strings after combining misaligned chunks
            pair_strings = []
            for i, (s1_tokens, s2_tokens, *rest) in enumerate(aligned_pairs):
                s1_canonical_tokens = [TokenAligner._canonical_token(tok) for tok in s1_tokens] if s1_tokens else []
                s2_canonical_tokens = [TokenAligner._canonical_token(tok) for tok in s2_tokens] if s2_tokens else []
                s1_str = "".join(s1_canonical_tokens)
                s2_str = "".join(s2_canonical_tokens)
                is_match = TokenAligner._strings_equal_flexible(s1_str, s2_str, ignore_leading_char_diff)
                pair_strings.append((s1_str, s2_str, is_match))

        processed_pairs = []
        alignment_cache = {}  # Cache for repeated alignment patterns
        i = 0
        
        while i < len(aligned_pairs):
            s1_tokens, s2_tokens, *_ = aligned_pairs[i]
            
            # Handle coarse alignments that can be split (optimized)
            if len(s1_tokens) > 1 and len(s1_tokens) == len(s2_tokens) and s1_tokens == s2_tokens:
                s1_start, s1_end, s2_start, s2_end = aligned_pairs[i][2:6]
                # Vectorized creation of split pairs
                for k in range(len(s1_tokens)):
                    processed_pairs.append(
                        ([s1_tokens[k]], [s2_tokens[k]], 
                         s1_start + k, s1_start + k + 1, 
                         s2_start + k, s2_start + k + 1)
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
            max_chunk_size = min(10, len(aligned_pairs) - start_bad_region)  # Limit search space
            
            for chunk_size in range(2, max_chunk_size + 1):
                chunk = aligned_pairs[start_bad_region : start_bad_region + chunk_size]
                
                # Efficient token extraction using list comprehension
                chunk_s1_tokens = []
                chunk_s2_tokens = []
                s1_indices = []
                s2_indices = []
                
                for s1_toks, s2_toks, s1_start, s1_end, s2_start, s2_end, *rest in chunk:
                    chunk_s1_tokens.extend(s1_toks)
                    chunk_s2_tokens.extend(s2_toks)
                    if s1_toks:
                        s1_indices.extend([s1_start, s1_end])
                    if s2_toks:
                        s2_indices.extend([s2_start, s2_end])

                # Quick string comparison using canonicalization
                chunk_s1_canonical = [TokenAligner._canonical_token(tok) for tok in chunk_s1_tokens]
                chunk_s2_canonical = [TokenAligner._canonical_token(tok) for tok in chunk_s2_tokens]
                chunk_s1_str = "".join(chunk_s1_canonical)
                chunk_s2_str = "".join(chunk_s2_canonical)
                
                if not TokenAligner._strings_equal_flexible(chunk_s1_str, chunk_s2_str, ignore_leading_char_diff):
                    continue
                
                # Create cache key for alignment
                cache_key = (tuple(chunk_s1_tokens), tuple(chunk_s2_tokens))
                
                if cache_key in alignment_cache:
                    sub_aligned_pairs, realign_is_perfect = alignment_cache[cache_key]
                else:
                    # Perform alignment
                    sub_aligned_pairs, _ = TokenAligner.align_tokens_with_combinations_numpy(
                        chunk_s1_tokens,
                        chunk_s2_tokens,
                        exact_match_score=exact_match_score,
                        combination_score_multiplier=combination_score_multiplier,
                        gap_penalty=gap_penalty,
                        max_combination_len=max_combination_len,
                        ignore_leading_char_diff=ignore_leading_char_diff
                    )
                    
                    # Check if re-alignment was successful using canonicalization
                    realign_is_perfect = all(
                        TokenAligner._strings_equal_flexible(
                            "".join([TokenAligner._canonical_token(tok) for tok in p[0]]),
                            "".join([TokenAligner._canonical_token(tok) for tok in p[1]]),
                            ignore_leading_char_diff
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
                    for s1_toks, s2_toks, s1_start, s1_end, s2_start, s2_end, *_ in sub_aligned_pairs:
                        new_s1_start = s1_chunk_start + s1_start if s1_start != -1 else -1
                        new_s1_end = s1_chunk_start + s1_end if s1_end != -1 else -1
                        new_s2_start = s2_chunk_start + s2_start if s2_start != -1 else -1
                        new_s2_end = s2_chunk_start + s2_end if s2_end != -1 else -1
                        processed_pairs.append((s1_toks, s2_toks, new_s1_start, new_s1_end, new_s2_start, new_s2_end))
                else:
                    # Create merged pair
                    s1_chunk_end = max(s1_indices[1::2]) if s1_indices else -1
                    s2_chunk_end = max(s2_indices[1::2]) if s2_indices else -1
                    merged_pair = (chunk_s1_tokens, chunk_s2_tokens, s1_chunk_start, s1_chunk_end, s2_chunk_start, s2_chunk_end)
                    processed_pairs.append(merged_pair)
                
                i = start_bad_region + chunk_size
                found_fix = True
                break
            
            if not found_fix:
                processed_pairs.append(aligned_pairs[start_bad_region])
                i = start_bad_region + 1

        return processed_pairs

    @staticmethod
    def get_alignment_mask(aligned_pairs: List, use_canonicalization: bool = True, 
                          ignore_leading_char_diff: bool = False) -> List[bool]:
        """
        Get a boolean mask indicating which alignments are correct.
        """
        if not aligned_pairs:
            return []
        
        # Handle batch case - take first batch
        if isinstance(aligned_pairs, list) and aligned_pairs and isinstance(aligned_pairs[0], list):
            pairs_to_verify = aligned_pairs[0]
        else:
            pairs_to_verify = aligned_pairs
        
        mask = []
        for s1_tokens, s2_tokens, s1_start, s1_end, s2_start, s2_end, *rest in pairs_to_verify:
            # Concatenate tokens into strings
            s1_str = "".join(s1_tokens) if s1_tokens else ""
            s2_str = "".join(s2_tokens) if s2_tokens else ""
            
            # Apply canonicalization if requested
            if use_canonicalization:
                s1_canonical = "".join([TokenAligner._canonical_token(tok) for tok in s1_tokens]) if s1_tokens else ""
                s2_canonical = "".join([TokenAligner._canonical_token(tok) for tok in s2_tokens]) if s2_tokens else ""
                is_correct = TokenAligner._strings_equal_flexible(s1_canonical, s2_canonical, ignore_leading_char_diff)
            else:
                if ignore_leading_char_diff:
                    is_correct = TokenAligner._strings_equal_flexible(s1_str, s2_str, ignore_leading_char_diff)
                else:
                    is_correct = s1_str == s2_str
            
            mask.append(is_correct)
        
        return mask

    
    def transform_learned_matrix_instance(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Instance method version that uses instance variables.
        """
        scale_trick_enabled = self.enable_scale_trick if self.enable_scale_trick is not None else False
        return TokenAligner.transform_learned_matrix(x, dim, enable_scale_trick=scale_trick_enabled)
    
    @staticmethod
    def transform_learned_matrix(x: torch.Tensor, dim: int = -1, enable_scale_trick=None) -> torch.Tensor:
        """
        Compute Quite Attention over tensor x along specified dimension.

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
            scale_trick_enabled = enable_scale_trick if enable_scale_trick is not None else False
            if scale_trick_enabled:
                #trick with last column being multiplier of 0..1, or try with c instead of 1 in qa.
                scores = torch.nn.functional.softmax(x, dim=dim)
                # Create a mask to zero out the last column while preserving gradients
                # mask = torch.ones_like(scores)
                # mask[:, -1] = 0.0
                # scores = scores * mask
                # Alternative approach using sigmoid (commented out):
                # scores = scores * torch.sigmoid(x[:, -1].unsqueeze(-1))
                return scores
            else:
                #normal softmax
                return torch.nn.functional.softmax(x, dim=dim)
            return torch.nn.functional.softmax(x, dim=dim)
