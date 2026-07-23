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
import argparse
import difflib
import os
import re
from collections import defaultdict

import torch
import tqdm
from transformers import AutoConfig, AutoTokenizer

from nemo_rl.algorithms.x_token.token_aligner import canonical_token
from tools.x_token.utils import (
    clean_model_name_for_filename,
    sinkhorn_one_dim,
)

# remove all special tokens that start with <| and end with |>


# compare 3 ways to estimate likelihood matrix:
# 1. using embeddings from another model, like was done in minimal_projection_generator.py
# 2. using text analysis like in tokenalign_likelihood_estimate.py
# 3. use one token to multiple and assign those as transformation matrix

# this file implements 3rd way


def create_weight_distribution(num_tokens):
    """Create weight distribution for multi-token mappings using exponential decay."""
    weights = []
    base = 0.9
    for i in range(num_tokens):
        if i == 0:
            weights.append(base)
        else:
            weights.append(base * (0.1**i))

    # Normalize to sum to 1
    total = sum(weights)
    weights = [w / total for w in weights]
    return weights


def print_projection_map_examples(
    transformation_counts,
    source_tokenizer,
    target_tokenizer,
    *,
    direction: str = "",
    num_examples: int = 50,
    use_raw_tokens: bool = False,
    use_canonicalization: bool = False,
) -> None:
    """Print a sample of projection mappings with decoded tokens and weights.

    Args:
        transformation_counts: Dict mapping (source_id, target_id) -> weight.
        source_tokenizer: HuggingFace tokenizer for source-side decoding.
        target_tokenizer: HuggingFace tokenizer for target-side decoding.
        direction: Human-readable direction label, e.g. "student->teacher".
        num_examples: Number of source tokens to print (highest-id first).
        use_raw_tokens: If True, use ``convert_ids_to_tokens`` instead of ``decode``.
        use_canonicalization: If True, apply :func:`canonical_token` after decoding.
    """
    print(
        f"\n--- Projection map examples {direction} (showing {num_examples} examples) ---"
    )

    # Group transformation_counts by source token (student token)
    source_to_targets = defaultdict(list)
    for (source_id, target_id), weight in transformation_counts.items():
        source_to_targets[source_id].append((target_id, weight))

    # Take the highest-id `num_examples` source tokens for inspection.
    sorted_sources = sorted(source_to_targets.keys())[-num_examples:]

    for source_id in sorted_sources:
        try:
            if use_raw_tokens:
                source_token = source_tokenizer.convert_ids_to_tokens([source_id])[0]
            else:
                source_token = source_tokenizer.decode([source_id])
            source_token = canonical_token(source_token, enabled=use_canonicalization)
            source_token_str = repr(source_token)
        except Exception:
            source_token_str = f"<ID:{source_id}>"

        targets_weights = sorted(
            source_to_targets[source_id], key=lambda x: x[1], reverse=True
        )

        target_parts = []
        for target_id, weight in targets_weights:
            try:
                if use_raw_tokens:
                    target_token = target_tokenizer.convert_ids_to_tokens([target_id])[
                        0
                    ]
                else:
                    target_token = target_tokenizer.decode([target_id])
                target_token = canonical_token(
                    target_token, enabled=use_canonicalization
                )
                target_token_str = repr(target_token)
            except Exception:
                target_token_str = f"<ID:{target_id}>"
            target_parts.append(f"{target_token_str}({weight:.4f})")

        print(f"{source_token_str} -> {' '.join(target_parts)}")


def add_multitoken_mappings(
    *,
    source_tokenizer,
    target_tokenizer,
    source_total_vocab_size: int,
    source_ignore_ids,
    target_ignore_ids,
    source_role: str,
    transformation_counts,
    tokens_to_cut: int,
    use_raw_tokens: bool,
    use_canonicalization: bool,
) -> tuple[dict, list]:
    """Re-tokenize every source-vocab token with the target tokenizer and accumulate weighted mappings.

    The two passes of the multi-token projection (student->teacher and
    teacher->student) share this same logic — only the source/target
    tokenizer pair and the role labels change.

    Args:
        source_tokenizer: Tokenizer to decode source-vocab tokens.
        target_tokenizer: Tokenizer used to re-encode each decoded source token.
        source_total_vocab_size: Total source-vocab size to iterate.
        source_ignore_ids: Source token ids to skip entirely.
        target_ignore_ids: Target token ids that, if present in any encoding, drop the whole mapping.
        source_role: Either ``"student"`` or ``"teacher"``. Determines which
            position of the ``transformation_counts`` key the source id fills
            (the key is always ``(student_id, teacher_id)``).
        transformation_counts: Mutable mapping ``(student_id, teacher_id) -> float``;
            weights are accumulated in place.
        tokens_to_cut: Cap the re-encoding at this many target tokens.
        use_raw_tokens: If True, decode via ``convert_ids_to_tokens`` instead of ``decode``.
        use_canonicalization: If True, apply :func:`canonical_token` after decoding.

    Returns:
        ``(decoded_source_tokens, examples)`` where ``decoded_source_tokens`` is a
        ``{source_id: decoded_str}`` dict and ``examples`` is a list of multi-token
        examples (only those with ``>= 2`` target tokens), keyed by
        ``f"{source_role}_token"`` / ``f"{target_role}_tokens"`` etc.

    Raises:
        ValueError: If ``source_role`` is not "student" or "teacher".
    """
    if source_role not in ("student", "teacher"):
        raise ValueError(
            f"source_role must be 'student' or 'teacher', got {source_role!r}"
        )
    target_role = "teacher" if source_role == "student" else "student"

    decoded_source: dict[int, str] = {}
    print(f"Decoding {source_role} tokens...")
    for token_id in tqdm.tqdm(
        range(source_total_vocab_size), desc=f"Decoding {source_role} tokens"
    ):
        if token_id in source_ignore_ids:
            continue
        try:
            if use_raw_tokens:
                decoded = source_tokenizer.convert_ids_to_tokens([token_id])[0]
            else:
                decoded = source_tokenizer.decode([token_id])
            if decoded.startswith("<|") and decoded.endswith("|>"):
                print(f"Skipping special token: {decoded}")
                continue
            decoded = canonical_token(decoded, enabled=use_canonicalization)
            decoded_source[token_id] = decoded
        except Exception:
            continue
    print(f"Successfully decoded {len(decoded_source)} {source_role} tokens")

    examples: list[dict] = []
    print(f"Finding {source_role}->{target_role} multi-token mappings...")
    for source_token_id, source_token_str in tqdm.tqdm(
        decoded_source.items(), desc=f"Processing {source_role} tokens"
    ):
        target_encoding = target_tokenizer(
            source_token_str, add_special_tokens=False, return_attention_mask=False
        )
        target_token_ids = target_encoding["input_ids"]
        if any(tid in target_ignore_ids for tid in target_token_ids):
            continue

        target_token_ids = target_token_ids[:tokens_to_cut]
        weights = create_weight_distribution(len(target_token_ids))

        for target_token_id, weight in zip(target_token_ids, weights):
            if source_role == "student":
                key = (source_token_id, target_token_id)
            else:
                key = (target_token_id, source_token_id)
            transformation_counts[key] += weight

        if len(target_token_ids) >= 2:
            decoded_targets = [
                target_tokenizer.decode([tid]) for tid in target_token_ids
            ]
            examples.append(
                {
                    f"{source_role}_token": source_token_str,
                    f"{source_role}_id": source_token_id,
                    f"{target_role}_tokens": decoded_targets,
                    f"{target_role}_ids": target_token_ids,
                    "weights": weights,
                }
            )

    return decoded_source, examples


def print_projection_statistics(
    *,
    transformation_counts,
    student_to_teacher_examples: list,
    teacher_to_student_examples: list,
    enable_reverse_pass: bool,
    max_examples_shown: int = 10,
) -> None:
    """Print a summary of the multi-token mappings collected so far."""
    print("\n=== SUMMARY ===")
    print(
        f"Found {len(student_to_teacher_examples)} student tokens that map "
        f"to multiple teacher tokens"
    )

    if student_to_teacher_examples:
        print("\nExamples of student->teacher multi-token mappings:")
        for example in student_to_teacher_examples[:max_examples_shown]:
            print(
                f"  Student '{example['student_token']}' -> Teacher "
                f"{example['teacher_tokens']} (weights: {example['weights']})"
            )
        if len(student_to_teacher_examples) > max_examples_shown:
            print(
                f"  ... and {len(student_to_teacher_examples) - max_examples_shown} more."
            )

    if enable_reverse_pass:
        print(
            f"\nReverse pass enabled — added "
            f"{len(teacher_to_student_examples)} teacher->student bidirectional examples"
        )

    print(f"\nTotal transformation entries: {len(transformation_counts)}")


def find_similar_special_tokens(
    tokenizer_a, tokenizer_b, similarity_threshold=0.4, top_k_matches=3
):
    """Find similar special tokens between two tokenizers using string similarity."""

    def is_special_token(token_str):
        """Check if a token looks like a special token."""
        return (
            (token_str.startswith("<|") and token_str.endswith("|>"))
            or (token_str.startswith("<") and token_str.endswith(">"))
            or token_str in ["<eos>", "<bos>", "<pad>", "<unk>", "<s>", "</s>"]
        )

    def extract_special_tokens(tokenizer):
        """Extract all special tokens from a tokenizer with their IDs."""
        special_tokens = {}
        vocab = tokenizer.get_vocab()
        for token_str, token_id in vocab.items():
            if is_special_token(token_str):
                special_tokens[token_id] = token_str
        return special_tokens

    def calculate_similarity(token_a, token_b):
        """Calculate similarity between two token strings."""
        # Use difflib for sequence similarity
        seq_similarity = difflib.SequenceMatcher(None, token_a, token_b).ratio()

        # Extract key words from special tokens for semantic matching
        def extract_keywords(token):
            # Remove special token markers and split by common separators
            cleaned = re.sub(r"[<>|_]", " ", token.lower())
            words = [w for w in cleaned.split() if len(w) > 2]  # Filter short words
            return set(words)

        keywords_a = extract_keywords(token_a)
        keywords_b = extract_keywords(token_b)

        # Jaccard similarity for keywords
        if keywords_a or keywords_b:
            keyword_similarity = len(keywords_a.intersection(keywords_b)) / len(
                keywords_a.union(keywords_b)
            )
        else:
            keyword_similarity = 0.0

        # Combined similarity (weighted average)
        return 0.6 * seq_similarity + 0.4 * keyword_similarity

    print("Extracting special tokens...")
    special_tokens_a = extract_special_tokens(tokenizer_a)  # student
    special_tokens_b = extract_special_tokens(tokenizer_b)  # teacher

    print(f"Found {len(special_tokens_a)} special tokens in student tokenizer")
    print(f"Found {len(special_tokens_b)} special tokens in teacher tokenizer")

    # Find matches
    special_token_mappings = []

    print("Finding similar special tokens...")
    for token_id_a, token_str_a in special_tokens_a.items():
        similarities = []
        for token_id_b, token_str_b in special_tokens_b.items():
            similarity = calculate_similarity(token_str_a, token_str_b)
            if similarity >= similarity_threshold:
                similarities.append((token_id_b, token_str_b, similarity))

        # Sort by similarity and take top-k
        similarities.sort(key=lambda x: x[2], reverse=True)
        for token_id_b, token_str_b, similarity in similarities[:top_k_matches]:
            special_token_mappings.append(
                {
                    "student_id": token_id_a,
                    "student_token": token_str_a,
                    "teacher_id": token_id_b,
                    "teacher_token": token_str_b,
                    "similarity": similarity,
                }
            )

    return special_token_mappings


def parse_arguments():
    """Parse command line arguments for the multi-token projection script."""
    parser = argparse.ArgumentParser(
        description="Generate multi-token projection mappings between tokenizers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model selection arguments
    parser.add_argument(
        "--student-model",
        type=str,
        required=True,
        help="Student model name or path",
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        required=True,
        help="Teacher model name or path",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=0,
        help=(
            "Print this many projection-map examples after building the "
            "transformation matrix (0 disables; useful for spot-checking)."
        ),
    )

    # Boolean flags
    parser.add_argument(
        "--enable-scale-trick",
        action="store_true",
        default=True,
        help="Enable scale trick (set last column likelihood to 0.2)",
    )
    parser.add_argument(
        "--disable-scale-trick",
        action="store_false",
        dest="enable_scale_trick",
        help="Disable scale trick",
    )
    parser.add_argument(
        "--enable-reverse-pass",
        action="store_true",
        default=True,
        help="Enable second pass: student tokens -> teacher tokens",
    )
    parser.add_argument(
        "--disable-reverse-pass",
        action="store_false",
        dest="enable_reverse_pass",
        help="Disable reverse pass",
    )
    parser.add_argument(
        "--enable-exact-match",
        action="store_true",
        default=False,
        help="Enable exact match enforcement for identical tokens",
    )
    parser.add_argument(
        "--use-raw-tokens",
        action="store_true",
        default=False,
        help="Use convert_ids_to_tokens instead of decode, should be False",
    )
    parser.add_argument(
        "--enable-special-token-mapping",
        action="store_true",
        default=True,
        help="Enable mapping of similar special tokens",
    )
    parser.add_argument(
        "--disable-special-token-mapping",
        action="store_false",
        dest="enable_special_token_mapping",
        help="Disable special token mapping",
    )
    parser.add_argument(
        "--use-canonicalization",
        action="store_true",
        default=False,
        help="Apply token canonicalization before processing to normalize different tokenizer representations (e.g., Ġ vs ▁ prefixes, Ċ vs \\n)",
    )

    # Numeric parameters
    parser.add_argument(
        "--tokens-to-cut",
        type=int,
        default=4,
        help="Maximum number of tokens to consider for multi-token mappings",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=32,
        help="Number of top projections to keep for each token",
    )
    parser.add_argument(
        "--special-token-similarity-threshold",
        type=float,
        default=0.3,
        help="Minimum similarity threshold for special token matching",
    )
    parser.add_argument(
        "--special-token-top-k",
        type=int,
        default=None,
        help="Top K matches for each special token (defaults to --top-k value)",
    )

    # File paths
    parser.add_argument(
        "--initial-projection-path",
        type=str,
        default=None,
        help="Path to initial projection map to load and extend",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="cross_tokenizer_data",
        help="Output directory for saving projection maps",
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default=None,
        help=(
            "Optional output filename stem (without extension). When unset, "
            "the stem is auto-derived from cleaned student and teacher model "
            "names; recipe-driven runs (e.g. tokenalign/commands.txt) pass "
            "an explicit stem like 'llama_qwen_best' to lock the filename."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # Configuration from arguments
    ENABLE_SCALE_TRICK = args.enable_scale_trick
    ENABLE_REVERSE_PASS = args.enable_reverse_pass
    ENABLE_EXACT_MATCH = args.enable_exact_match

    TOKENS_TO_CUT = args.tokens_to_cut
    TOP_K = args.top_k
    USE_RAW_TOKENS = args.use_raw_tokens
    INITIAL_PROJECTION_PATH = args.initial_projection_path
    ENABLE_SPECIAL_TOKEN_MAPPING = args.enable_special_token_mapping
    SPECIAL_TOKEN_SIMILARITY_THRESHOLD = args.special_token_similarity_threshold
    SPECIAL_TOKEN_TOP_K = (
        args.special_token_top_k if args.special_token_top_k is not None else TOP_K
    )
    USE_CANONICALIZATION = args.use_canonicalization

    # Model names from arguments
    teacher_model_name = args.teacher_model
    student_model_name = args.student_model

    # Print configuration
    print("=== Configuration ===")
    print(f"Student model: {student_model_name}")
    print(f"Teacher model: {teacher_model_name}")
    print(f"Enable scale trick: {ENABLE_SCALE_TRICK}")
    print(f"Enable reverse pass: {ENABLE_REVERSE_PASS}")
    print(f"Enable exact match: {ENABLE_EXACT_MATCH}")
    print(f"Use raw tokens: {USE_RAW_TOKENS}")
    print(f"Use canonicalization: {USE_CANONICALIZATION}")
    print(f"Tokens to cut: {TOKENS_TO_CUT}")
    print(f"Top K: {TOP_K}")
    print(f"Enable special token mapping: {ENABLE_SPECIAL_TOKEN_MAPPING}")
    if ENABLE_SPECIAL_TOKEN_MAPPING:
        print(
            f"Special token similarity threshold: {SPECIAL_TOKEN_SIMILARITY_THRESHOLD}"
        )
        print(f"Special token top K: {SPECIAL_TOKEN_TOP_K}")
    print(f"Initial projection path: {INITIAL_PROJECTION_PATH}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 25)

    tokenizer_student = AutoTokenizer.from_pretrained(student_model_name)
    tokenizer_teacher = AutoTokenizer.from_pretrained(teacher_model_name)

    tokenizer_student_total_vocab_size = len(tokenizer_student)
    tokenizer_teacher_total_vocab_size = len(tokenizer_teacher)
    model_A_config = AutoConfig.from_pretrained(student_model_name)
    model_B_config = AutoConfig.from_pretrained(teacher_model_name)
    # Gemma and Qwen3.5 nest `vocab_size` under `config.text_config`; the
    # rest of the supported families expose it directly on the top-level
    # config. Mirrors the PT reference.
    student_name_lower = student_model_name.lower()
    if "gemma" in student_name_lower or "qwen3.5" in student_name_lower:
        source_vocab_size = model_A_config.text_config.vocab_size
    else:
        source_vocab_size = model_A_config.vocab_size

    teacher_name_lower = teacher_model_name.lower()
    if "gemma" in teacher_name_lower or "qwen3.5" in teacher_name_lower:
        target_vocab_size = model_B_config.text_config.vocab_size
    else:
        target_vocab_size = model_B_config.vocab_size

    tokenizer_student_total_vocab_size = source_vocab_size
    tokenizer_teacher_total_vocab_size = target_vocab_size
    # print(f"Source top k tokens: {model_A_top_k_tokens}")
    # print(f"Target top k tokens: {model_B_top_k_tokens}")

    print(f"Student tokenizer total vocab size: {tokenizer_student_total_vocab_size}")
    print(f"Teacher tokenizer total vocab size: {tokenizer_teacher_total_vocab_size}")

    # Print token processing mode
    if USE_RAW_TOKENS:
        print("Using raw token representation (convert_ids_to_tokens)")
    else:
        print("Using decoded token representation (decode)")

    transformation_counts = defaultdict(float)
    import os

    if INITIAL_PROJECTION_PATH and os.path.exists(INITIAL_PROJECTION_PATH):
        print(f"Loading initial projection from: {INITIAL_PROJECTION_PATH}")
        initial_projection_map = torch.load(INITIAL_PROJECTION_PATH, map_location="cpu")

        if (
            isinstance(initial_projection_map, dict)
            and "indices" in initial_projection_map
            and "likelihoods" in initial_projection_map
        ):
            print(
                "Loading from sparse top-k format and converting to transformation_counts."
            )
            indices = initial_projection_map["indices"]
            likelihoods = initial_projection_map["likelihoods"]

            # Accept the new `student_model_id`/`teacher_model_id` keys and fall back to the
            # legacy `model_A_id`/`model_B_id` keys produced by older generator runs.
            loaded_student_model = initial_projection_map.get(
                "student_model_id", initial_projection_map.get("model_A_id")
            )
            loaded_teacher_model = initial_projection_map.get(
                "teacher_model_id", initial_projection_map.get("model_B_id")
            )

            if loaded_student_model and loaded_student_model != student_model_name:
                print(
                    f"Warning: Student model mismatch. Loaded: {loaded_student_model}, Current: {student_model_name}"
                )
            if loaded_teacher_model and loaded_teacher_model != teacher_model_name:
                print(
                    f"Warning: Teacher model mismatch. Loaded: {loaded_teacher_model}, Current: {teacher_model_name}"
                )

            num_student_tokens = indices.shape[0]
            top_k = indices.shape[1]

            for student_id in tqdm.tqdm(
                range(num_student_tokens),
                desc="Converting initial projection to counts",
            ):
                for k in range(top_k):
                    teacher_id = indices[student_id, k].item()
                    if teacher_id != -1:
                        likelihood = likelihoods[student_id, k].item()
                        if likelihood > 0:
                            transformation_counts[(student_id, teacher_id)] = likelihood

        elif torch.is_tensor(initial_projection_map):
            if initial_projection_map.is_sparse:
                print(
                    "Loading from sparse tensor and converting to transformation_counts."
                )
                sparse_matrix = initial_projection_map.coalesce()
                map_indices = sparse_matrix.indices()
                map_values = sparse_matrix.values()
                for i in tqdm.tqdm(
                    range(map_indices.shape[1]),
                    desc="Converting sparse tensor to counts",
                ):
                    student_id = map_indices[0, i].item()
                    teacher_id = map_indices[1, i].item()
                    weight = map_values[i].item()
                    if weight > 0:
                        transformation_counts[(student_id, teacher_id)] = weight
            else:
                print(
                    "Loading from dense matrix and converting to transformation_counts."
                )
                dense_matrix = initial_projection_map
                non_zero_indices = torch.nonzero(dense_matrix, as_tuple=False)
                for idx in tqdm.tqdm(
                    range(non_zero_indices.shape[0]),
                    desc="Converting dense projection to counts",
                ):
                    student_id = non_zero_indices[idx, 0].item()
                    teacher_id = non_zero_indices[idx, 1].item()
                    weight = dense_matrix[student_id, teacher_id].item()
                    if weight > 0:
                        transformation_counts[(student_id, teacher_id)] = weight
        else:
            print(
                f"Warning: Unrecognized format for initial projection map at {INITIAL_PROJECTION_PATH}. Skipping."
            )

        print(
            f"Initialized transformation_counts with {len(transformation_counts)} entries."
        )

    ignore_tokens = ["<|endoftext|>", "<eos>"]
    ignore_student_ids = {
        tokenizer_student.convert_tokens_to_ids(token)
        for token in ignore_tokens
        if token in tokenizer_student.get_vocab()
    }
    ignore_teacher_ids = {
        tokenizer_teacher.convert_tokens_to_ids(token)
        for token in ignore_tokens
        if token in tokenizer_teacher.get_vocab()
    }

    # First pass: student tokens -> teacher tokens.
    print("\n=== FIRST PASS: Student tokens -> Teacher tokens ===")
    _, student_to_teacher_examples = add_multitoken_mappings(
        source_tokenizer=tokenizer_student,
        target_tokenizer=tokenizer_teacher,
        source_total_vocab_size=tokenizer_student_total_vocab_size,
        source_ignore_ids=ignore_student_ids,
        target_ignore_ids=ignore_teacher_ids,
        source_role="student",
        transformation_counts=transformation_counts,
        tokens_to_cut=TOKENS_TO_CUT,
        use_raw_tokens=USE_RAW_TOKENS,
        use_canonicalization=USE_CANONICALIZATION,
    )

    # Second pass: teacher tokens -> student tokens (opposite direction).
    teacher_to_student_examples: list[dict] = []
    if ENABLE_REVERSE_PASS:
        print("\n=== SECOND PASS: Teacher tokens -> Student tokens ===")
        _, teacher_to_student_examples = add_multitoken_mappings(
            source_tokenizer=tokenizer_teacher,
            target_tokenizer=tokenizer_student,
            source_total_vocab_size=tokenizer_teacher_total_vocab_size,
            source_ignore_ids=ignore_teacher_ids,
            target_ignore_ids=ignore_student_ids,
            source_role="teacher",
            transformation_counts=transformation_counts,
            tokens_to_cut=TOKENS_TO_CUT,
            use_raw_tokens=USE_RAW_TOKENS,
            use_canonicalization=USE_CANONICALIZATION,
        )

    print("\n=== ADDING SPECIAL TOKEN MAPPINGS ===")

    # Find and add special token mappings (if enabled)
    special_token_mappings = []
    if ENABLE_SPECIAL_TOKEN_MAPPING:
        special_token_mappings = find_similar_special_tokens(
            tokenizer_student,
            tokenizer_teacher,
            similarity_threshold=SPECIAL_TOKEN_SIMILARITY_THRESHOLD,
            top_k_matches=SPECIAL_TOKEN_TOP_K,
        )
    else:
        print("Special token mapping disabled")

    if special_token_mappings:
        print(f"\nFound {len(special_token_mappings)} special token mappings:")
        initial_transformation_count = len(transformation_counts)

        # Add ALL mappings to transformation matrix
        for mapping in special_token_mappings:
            student_id = mapping["student_id"]
            teacher_id = mapping["teacher_id"]
            similarity = mapping["similarity"]

            # Add mapping with weight based on similarity
            weight = similarity * 0.8  # Scale similarity to reasonable weight
            transformation_counts[(student_id, teacher_id)] += weight

        # Group mappings by student token and show top 2 matches per student token
        from collections import defaultdict

        student_mappings = defaultdict(list)
        for mapping in special_token_mappings:
            student_mappings[mapping["student_id"]].append(mapping)

        # Sort each student's mappings by similarity and show top 2
        print("Top 2 matches per student special token:")
        shown_count = 0
        for student_id, mappings in student_mappings.items():
            # Sort by similarity (highest first)
            sorted_mappings = sorted(
                mappings, key=lambda x: x["similarity"], reverse=True
            )

            # Show top 2 for this student token
            student_token = sorted_mappings[0][
                "student_token"
            ]  # Get student token name
            print(f"  {student_token}:")

            for mapping in sorted_mappings[:2]:
                similarity = mapping["similarity"]
                weight = similarity * 0.8
                print(
                    f"    -> '{mapping['teacher_token']}' (similarity: {similarity:.3f}, weight: {weight:.3f})"
                )
                shown_count += 1

            if len(sorted_mappings) > 2:
                print(f"    ... and {len(sorted_mappings) - 2} more matches")

        total_hidden = len(special_token_mappings) - shown_count
        if total_hidden > 0:
            print(f"Total mappings not shown: {total_hidden}")

        added_count = len(transformation_counts) - initial_transformation_count
        print(f"Added {added_count} new special token transformation entries")
    else:
        print("No similar special tokens found")

    print_projection_statistics(
        transformation_counts=transformation_counts,
        student_to_teacher_examples=student_to_teacher_examples,
        teacher_to_student_examples=teacher_to_student_examples,
        enable_reverse_pass=ENABLE_REVERSE_PASS,
    )

    if ENABLE_EXACT_MATCH:
        print("Checking for exact token matches and setting exact mappings...")
        # check exact match between student and teacher tokens and set those as perfect 1-to-1 mappings
        # Convert all tokens to strings at once for vectorized comparison
        tokens_student = [
            canonical_token(
                tokenizer_student.convert_ids_to_tokens([i])[0],
                enabled=USE_CANONICALIZATION,
            )
            for i in range(tokenizer_student_total_vocab_size)
        ]
        tokens_teacher = [
            canonical_token(
                tokenizer_teacher.convert_ids_to_tokens([j])[0],
                enabled=USE_CANONICALIZATION,
            )
            for j in range(tokenizer_teacher_total_vocab_size)
        ]

        map_teacher_token_to_idx = {token: j for j, token in enumerate(tokens_teacher)}

        # Find indices in student and teacher where the tokens are identical
        match_indices_student = []
        match_indices_teacher = []
        for i, token_student in enumerate(tokens_student):
            if token_student in map_teacher_token_to_idx:
                j = map_teacher_token_to_idx[token_student]
                match_indices_student.append(i)
                match_indices_teacher.append(j)

        if match_indices_student:
            print(
                f"Found {len(match_indices_student)} exact matches. Setting perfect 1-to-1 mappings."
            )

            # For tokens that match exactly, we want their mapping to be 1.0
            # and they should not be mapped to any other token.
            # First, remove all existing mappings for these student tokens
            match_indices_student_set = set(match_indices_student)
            keys_to_remove = []
            for key in transformation_counts.keys():
                student_id, teacher_id = key
                if student_id in match_indices_student_set:
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del transformation_counts[key]
            # Then, set the perfect 1-to-1 mappings for exact matches
            for student_id, teacher_id in zip(
                match_indices_student, match_indices_teacher
            ):
                transformation_counts[(student_id, teacher_id)] = 1.0

    # Create transformation matrix (student -> teacher projection)
    indices = list(transformation_counts.keys())
    values = list(transformation_counts.values())

    teacher_indices = [idx[1] for idx in indices]
    student_indices = [idx[0] for idx in indices]

    # Create sparse tensor with student tokens as rows, teacher tokens as columns
    # This creates a student -> teacher projection matrix
    indices_tensor = torch.LongTensor([student_indices, teacher_indices])
    values_tensor = torch.FloatTensor(values)

    transformation_matrix_sparse = torch.sparse_coo_tensor(
        indices_tensor,
        values_tensor,
        (tokenizer_student_total_vocab_size, tokenizer_teacher_total_vocab_size),
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16,
    )

    # indices, values = torch.topk(transformation_matrix_sparse, k=1000, dim=1)

    print(
        f"Created sparse student->teacher projection matrix with shape: {transformation_matrix_sparse.shape}"
    )
    print(f"Non-zero elements: {transformation_matrix_sparse._nnz()}")

    # Convert sparse matrix to same format as minimal_projection_generator.py
    os.makedirs(args.output_dir, exist_ok=True)

    # Convert defaultdict to regular dict for saving
    transformation_counts_dict = dict(transformation_counts)

    # Show some examples of the projection mappings
    if args.num_examples > 0:
        print_projection_map_examples(
            transformation_counts_dict,
            tokenizer_student,
            tokenizer_teacher,
            direction="student->teacher",
            num_examples=args.num_examples,
            use_raw_tokens=USE_RAW_TOKENS,
            use_canonicalization=USE_CANONICALIZATION,
        )

    print(f"\nConverting sparse matrix to top-{TOP_K} dense format...")

    # Convert sparse matrix to dense and get top-k values per row
    print("Converting to dense matrix on CPU to avoid memory issues...")
    dense_matrix = (
        transformation_matrix_sparse.cpu().to_dense()
    )  # Move to CPU to handle memory
    print(f"Dense matrix shape: {dense_matrix.shape}")

    # Get top-k values and indices for each row (each source token)
    print(f"Extracting top-{TOP_K} values per token...")

    # Apply sinkhorn normalization on CPU
    print("Applying Sinkhorn normalization on CPU...")
    dense_matrix = sinkhorn_one_dim(dense_matrix, n_iters=1)

    # Extract top-k on CPU
    top_k_likelihoods, top_k_indices = torch.topk(
        dense_matrix, k=min(TOP_K, dense_matrix.shape[1]), dim=1
    )
    # exit()
    # Handle case where vocabulary has fewer tokens than TOP_K
    actual_k = top_k_indices.shape[1]
    if actual_k < TOP_K:
        print(
            f"Warning: Target vocabulary size ({dense_matrix.shape[1]}) is smaller than TOP_K ({TOP_K}). Using k={actual_k}"
        )
        # Pad with -1 indices and 0.0 likelihoods to maintain consistent shape
        pad_size = TOP_K - actual_k
        top_k_indices = torch.cat(
            [
                top_k_indices,
                torch.full(
                    (top_k_indices.shape[0], pad_size), -1, dtype=top_k_indices.dtype
                ),
            ],
            dim=1,
        )
        top_k_likelihoods = torch.cat(
            [
                top_k_likelihoods,
                torch.zeros(
                    (top_k_likelihoods.shape[0], pad_size),
                    dtype=top_k_likelihoods.dtype,
                ),
            ],
            dim=1,
        )

    # Apply SCALE_TRICK: set last column to -4 if enabled
    if ENABLE_SCALE_TRICK:
        print("ENABLE_SCALE_TRICK is True: Setting last column of likelihoods to -4.0")
        top_k_likelihoods[:, -1] = 0.2
        if ENABLE_EXACT_MATCH:
            for indices in match_indices_student:
                top_k_likelihoods[indices, -1] = 0.0
            print(
                f"Set last column of likelihoods to 0.0 for {len(match_indices_student)} exact matches as exact match is enabled"
            )
        # Apply sinkhorn normalization on CPU
        print("Applying Sinkhorn normalization on CPU...")
        top_k_likelihoods = sinkhorn_one_dim(top_k_likelihoods, n_iters=1)

    # set indices to -1 where likelihood is 0

    # Build the output filename. If the caller provided an explicit
    # `--output-filename` stem, honor it (recipe-driven runs lock the name);
    # otherwise auto-derive from cleaned student/teacher names so ad-hoc
    # runs produce a self-describing default that matches the format used by
    # minimal_projection_generator.py.
    if args.output_filename is not None:
        output_filename = args.output_filename
    else:
        student_clean_name = clean_model_name_for_filename(
            student_model_name.split("/")[-1]
        )
        teacher_clean_name = clean_model_name_for_filename(
            teacher_model_name.split("/")[-1]
        )
        output_filename = (
            f"projection_map_{student_clean_name}_to_{teacher_clean_name}"
            f"_multitoken_top_{TOP_K}_double"
        )
    if ENABLE_SPECIAL_TOKEN_MAPPING:
        output_filename += "_special"
    if not output_filename.endswith(".pt"):
        output_filename += ".pt"
    output_path = os.path.join(args.output_dir, output_filename)

    # Save in same format as minimal_projection_generator.py.
    # `enable_scale_trick` is persisted so downstream tools (e.g.
    # `sort_and_cut_projection_matrix.py`) can decide whether the last
    # column carries a tunable scale slot without the user re-specifying.
    # Metadata keys use the student/teacher framing; legacy `model_A_id`/
    # `model_B_id` produced by older PT-reference artifacts are accepted on
    # the load side (see the fallback near line 574).
    torch.save(
        {
            "indices": top_k_indices,
            "likelihoods": top_k_likelihoods,
            "student_model_id": student_model_name,
            "teacher_model_id": teacher_model_name,
            "enable_scale_trick": ENABLE_SCALE_TRICK,
        },
        output_path,
    )

    print(f"Saved projection map to: {output_path}")
    print(
        f"Format: indices shape {top_k_indices.shape}, likelihoods shape {top_k_likelihoods.shape}"
    )
    print("Compatible with minimal_projection_generator.py format")
    print(
        f"Token processing mode: {'Raw tokens (convert_ids_to_tokens)' if USE_RAW_TOKENS else 'Decoded tokens (decode)'}"
    )
    if ENABLE_REVERSE_PASS:
        print(
            "File includes bidirectional mappings (teacher->student and student->teacher)"
        )
    if ENABLE_SPECIAL_TOKEN_MAPPING:
        print(
            f"File includes special token mappings (similarity_threshold={SPECIAL_TOKEN_SIMILARITY_THRESHOLD}, top_k={SPECIAL_TOKEN_TOP_K})"
        )
    # exit()

    # Test projection function compatibility (same as minimal_projection_generator.py)
    print("\n--- Testing projection function compatibility ---")

    def project_token_likelihoods(
        input_likelihoods,
        projection_map_indices,
        projection_map_values,
        target_vocab_size,
        device,
    ):
        """Projects token likelihoods from a source to a target vocabulary using a sparse map."""
        batch_size, seq_len, source_vocab_size = input_likelihoods.shape
        if source_vocab_size != projection_map_indices.shape[0]:
            raise ValueError(
                f"Source vocab size of input ({source_vocab_size}) mismatches projection map size ({projection_map_indices.shape[0]})"
            )

        top_k = projection_map_indices.shape[1]
        input_likelihoods = input_likelihoods.to(device)
        projection_map_indices = projection_map_indices.to(device)
        projection_map_values = projection_map_values.to(device)

        crow_indices = torch.arange(
            0, (source_vocab_size + 1) * top_k, top_k, device=device, dtype=torch.long
        )
        col_indices = projection_map_indices.flatten()
        values = projection_map_values.flatten()

        sparse_projection_matrix = torch.sparse_csr_tensor(
            crow_indices,
            col_indices,
            values,
            size=(source_vocab_size, target_vocab_size),
            device=device,
        )

        reshaped_input = input_likelihoods.reshape(
            batch_size * seq_len, source_vocab_size
        )
        projected_likelihoods_reshaped = torch.matmul(
            reshaped_input, sparse_projection_matrix
        )
        return projected_likelihoods_reshaped.reshape(
            batch_size, seq_len, target_vocab_size
        )

    # Create a dummy likelihood tensor: [BATCH, SEQ, source_vocab_size]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_tensor = torch.randn(
        1, 4096, tokenizer_student_total_vocab_size, device=device, dtype=torch.float32
    )

    # Transform this tensor using the projection map
    projected_tensor = project_token_likelihoods(
        dummy_tensor,
        top_k_indices.to(device),
        top_k_likelihoods.float().to(device),
        tokenizer_teacher_total_vocab_size,
        device,
    )
    print(f"Input tensor shape: {dummy_tensor.shape}")
    print(f"Projected tensor shape: {projected_tensor.shape}")
    print("Projection test successful - format is fully compatible!")
