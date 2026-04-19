import argparse

import torch
from tokenalign import TokenAligner
from transformers import AutoConfig, AutoTokenizer


def apply_canonicalization_if_enabled(token_str, use_canonicalization):
    """Apply canonicalization to token string if enabled."""
    if use_canonicalization:
        return TokenAligner._canonical_token(token_str)
    return token_str


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
        default="meta-llama/Llama-3.2-1B",
        help="Student model name or path",
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        default="microsoft/phi-4",
        help="Teacher model name or path",
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

    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    # Model names from arguments
    teacher_model_name = args.teacher_model
    student_model_name = args.student_model
    USE_CANONICALIZATION = args.use_canonicalization

    tokenizer_student = AutoTokenizer.from_pretrained(student_model_name)
    tokenizer_teacher = AutoTokenizer.from_pretrained(teacher_model_name)

    tokenizer_student_total_vocab_size = len(tokenizer_student)
    tokenizer_teacher_total_vocab_size = len(tokenizer_teacher)
    model_A_config = AutoConfig.from_pretrained(student_model_name)
    model_B_config = AutoConfig.from_pretrained(teacher_model_name)

    tokens_student = [
        apply_canonicalization_if_enabled(
            tokenizer_student.convert_ids_to_tokens([i])[0], USE_CANONICALIZATION
        )
        for i in range(tokenizer_student_total_vocab_size)
    ]
    tokens_teacher = [
        apply_canonicalization_if_enabled(
            tokenizer_teacher.convert_ids_to_tokens([j])[0], USE_CANONICALIZATION
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

    # load intial projection map
    initial_projection_path = args.initial_projection_path
    if initial_projection_path is not None:
        initial_projection_map = torch.load(initial_projection_path)
    else:
        initial_projection_map = None

    # go through token in projection map. For each token present in match_indices_student, set it's likelihoods and incices to 1.0 and the exact match teacher token
    non_exact_map_tokens = list(range(len(initial_projection_map["likelihoods"])))
    all_student_token_ids = list(range(len(initial_projection_map["likelihoods"])))

    show_remapping = 5
    if show_remapping > 0:
        print(f"Showing remapping for the last {show_remapping} exact matches.")
    else:
        print("Not showing remapping.")

    for i, exact_token_student in enumerate(match_indices_student):
        exact_token_teacher = match_indices_teacher[i]

        index_ = all_student_token_ids.index(exact_token_student)
        likelihoods = initial_projection_map["likelihoods"][index_]
        indices = initial_projection_map["indices"][index_]

        if len(match_indices_student) - i <= show_remapping:
            print(f"prior to remapping: likelihoods {likelihoods} indices {indices}")

        topk = indices.shape[0]

        remapped_indices = torch.ones_like(indices) * -1
        remapped_likelihoods = torch.zeros_like(likelihoods)

        remapped_likelihoods[0] = 1.0
        remapped_indices[0] = exact_token_teacher

        # if exact_token_student == 5159:
        #     import pdb
        #     pdb.set_trace()

        initial_projection_map["likelihoods"][index_] = remapped_likelihoods
        initial_projection_map["indices"][index_] = remapped_indices

        if len(match_indices_student) - i <= show_remapping:
            print(
                f"after remapping {tokens_student[exact_token_student]}:{exact_token_student} -> {tokens_teacher[exact_token_teacher]}:{exact_token_teacher}: likelihoods {initial_projection_map['likelihoods'][index_]} indices {initial_projection_map['indices'][index_]}"
            )
        non_exact_map_tokens.remove(index_)

    # import pdb
    # pdb.set_trace()
    # print(f"non exact map tokens: {non_exact_map_tokens}")
    # pdb.set_trace()
    save_path = args.initial_projection_path.split(".")[0] + "_exact_map_remapped.pt"
    torch.save(initial_projection_map, save_path)
    print(f"Saved remapped projection map to: {save_path}")
    print(
        f"remapped {len(match_indices_student)} tokens. Retained remaining {len(non_exact_map_tokens)} tokens as is."
    )
