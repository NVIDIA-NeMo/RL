import argparse
import os
import re

import torch
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

##### verify KL and top5 with this matrix


###### use config vocab size, not tokenizer

EXACT_MATCH_ONLY = False

# --- Configuration and Setup ---
parser = argparse.ArgumentParser(
    description="Generate a sparse projection map between two tokenizers."
)
parser.add_argument(
    "--model_a_index",
    type=int,
    default=1,
    help="Index of the source model (Model A / Student).",
)
parser.add_argument(
    "--model_b_index",
    type=int,
    default=0,
    help="Index of the target model (Model B / Teacher).",
)
parser.add_argument(
    "--model_a_name",
    type=str,
    default=None,
    help="HuggingFace model name for source model (Model A / Student). If provided, overrides model_a_index.",
)
parser.add_argument(
    "--model_b_name",
    type=str,
    default=None,
    help="HuggingFace model name for target model (Model B / Teacher). If provided, overrides model_b_index.",
)
parser.add_argument(
    "--keep_top_tokens",
    type=int,
    default=-1,
    help="Number of top tokens to keep for each vocabulary. -1 means all.",
)
parser.add_argument(
    "--data_dir",
    type=str,
    default="cross_tokenizer_data/",
    help="Directory for importance scores and cached data.",
)
parser.add_argument(
    "--top_k",
    type=int,
    default=10,
    help="Number of top projections to keep for each token.",
)
parser.add_argument(
    "--weight_threshold",
    type=float,
    default=0.0,
    help="Minimum weight threshold to keep a projection. Values below this will be filtered out.",
)
parser.add_argument(
    "--force_recompute",
    action="store_true",
    help="Force recomputation of embeddings even if cached files exist.",
)
parser.add_argument(
    "--skip_exact_enforcement",
    action="store_true",
    help="Skip enforcing exact matches between tokens.",
)
parser.add_argument(
    "--use_canonicalization",
    action="store_true",
    help="Apply token canonicalization before generating embeddings to normalize different tokenizer representations (e.g., Ġ vs ▁ prefixes, Ċ vs \\n).",
)
args = parser.parse_args()

args.skip_exact_enforcement = True

MODEL_LIST = [
    "nvidia/Mistral-NeMo-Minitron-8B-Base",
    "Qwen/Qwen3-8B-Base",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.1-8B",
    "google/gemma-3-4b-it",
    "google/gemma-2b",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "openai/gpt-oss-20b",
    "microsoft/phi-4",
    "google/gemma-3-12b-pt",
]
EMBEDDING_MODEL_CHOICES = [
    {
        "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "type": "sbert",
    },
    {"name": "sentence-transformers/all-mpnet-base-v2", "type": "sbert"},
    {"name": "sentence-transformers/all-MiniLM-L6-v2", "type": "sbert"},
    {"name": "Qwen/Qwen3-Embedding-4B", "type": "llm_first_layer"},
    {"name": "Qwen/Qwen3-Embedding-0.6B", "type": "llm_first_layer"},
]

MAX_SEQ_LENGTH_EMBEDDING = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sinkhorn(A, n_iters=10):
    for _ in range(n_iters):
        if _ % 2 == 0:
            # A = A / (A.sum(dim=0, keepdim=True) + 1e-6)
            col_sums = A.sum(dim=0, keepdim=True)
            safe_col_sums = torch.where(
                col_sums == 0, torch.ones_like(col_sums), col_sums
            )
            A = A / safe_col_sums
        else:
            # 0, 2, 4, 6
            # A = A / (A.sum(dim=1, keepdim=True) + 1e-6)
            row_sums = A.sum(dim=1, keepdim=True)
            safe_row_sums = torch.where(
                row_sums == 0, torch.ones_like(row_sums), row_sums
            )
            A = A / safe_row_sums

    return A


def sinkhorn_one_dim(A, n_iters=1):
    for _ in range(n_iters):
        # A = A / (A.sum(dim=1, keepdim=True) + 1e-6)
        row_sums = A.sum(dim=1, keepdim=True)
        safe_row_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
        A = A / safe_row_sums

    return A


# --- Helper Functions ---


def clean_model_name_for_filename(name: str) -> str:
    """Removes parameter counts and common suffixes from model names for cleaner filenames."""
    # Removes patterns like -8B, -1.5B, -4b, -125m etc.
    cleaned_name = re.sub(r"-?[0-9\.]+[bBmB]", "", name, flags=re.IGNORECASE)
    # Remove common suffixes
    cleaned_name = (
        cleaned_name.replace("-Base", "").replace("-it", "").replace("-Instruct", "")
    )
    # Clean up any leading/trailing hyphens that might result
    cleaned_name = cleaned_name.strip("-_")
    if "mini" in name:
        cleaned_name += "_mini"
    return cleaned_name


def load_tokenizer(model_id_or_path):
    """Loads a HuggingFace tokenizer, setting a pad token if necessary."""
    try:
        tok = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok
    except Exception as e:
        print(f"Error loading tokenizer for model '{model_id_or_path}': {e}")
        print(f"Available models in MODEL_LIST (indices 0-{len(MODEL_LIST) - 1}):")
        for i, model in enumerate(MODEL_LIST):
            print(f"  {i}: {model}")
        raise


def validate_model_selection(args):
    """Validates that the model selection arguments are valid."""
    # Check if both name and index are provided for the same model
    if args.model_a_name is not None and args.model_a_index != 1:  # 1 is the default
        print(
            "Warning: Both --model_a_name and --model_a_index provided. Using --model_a_name."
        )

    if args.model_b_name is not None and args.model_b_index != 0:  # 0 is the default
        print(
            "Warning: Both --model_b_name and --model_b_index provided. Using --model_b_name."
        )

    # Validate indices if names are not provided
    if args.model_a_name is None:
        if args.model_a_index < 0 or args.model_a_index >= len(MODEL_LIST):
            raise ValueError(
                f"model_a_index {args.model_a_index} is out of range. Available models: 0-{len(MODEL_LIST) - 1}"
            )

    if args.model_b_name is None:
        if args.model_b_index < 0 or args.model_b_index >= len(MODEL_LIST):
            raise ValueError(
                f"model_b_index {args.model_b_index} is out of range. Available models: 0-{len(MODEL_LIST) - 1}"
            )

    # Check if the same model is selected for both A and B
    model_a_id = (
        args.model_a_name
        if args.model_a_name is not None
        else MODEL_LIST[args.model_a_index]
    )
    model_b_id = (
        args.model_b_name
        if args.model_b_name is not None
        else MODEL_LIST[args.model_b_index]
    )

    if model_a_id == model_b_id:
        raise ValueError(f"Cannot use the same model for both A and B: {model_a_id}")


def save_data(data, filename):
    """Saves data to a torch file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(data.cpu(), filename)
    print(f"Data saved to {filename}")


def load_data(filename):
    """Loads data from a torch file."""
    return torch.load(filename)


def get_llm_first_layer_embeddings(
    decoded_tokens_list,
    llm_embedding_tokenizer,
    llm_embedding_model,
    max_seq_length_embedding,
    device,
    batch_size=32,
):
    """Generates embeddings using the first layer of a given LLM."""
    all_embeddings = []
    llm_embedding_model.eval()
    embedding_dim = llm_embedding_model.config.hidden_size

    for i in tqdm(
        range(0, len(decoded_tokens_list), batch_size), desc="Encoding tokens with LLM"
    ):
        batch_tokens = decoded_tokens_list[i : i + batch_size]
        inputs = llm_embedding_tokenizer(
            batch_tokens,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length_embedding,
            add_special_tokens=False,
        ).to(device)

        with torch.no_grad():
            outputs = llm_embedding_model(**inputs, output_hidden_states=True)
            first_layer_output = outputs.hidden_states[0]

            for k in range(first_layer_output.shape[0]):
                valid_token_mask = inputs["attention_mask"][k] == 1
                if valid_token_mask.sum() > 0:
                    pooled_embedding = first_layer_output[k, valid_token_mask].mean(
                        dim=0
                    )
                    all_embeddings.append(pooled_embedding)
                else:
                    all_embeddings.append(torch.zeros(embedding_dim, device=device))

    return torch.stack(all_embeddings).to(device)


def compute_chunked_projection_map(
    embeddings_query, embeddings_corpus, args, device, chunk_size=1000
):
    """Computes projection map in chunks to save memory."""
    num_queries = embeddings_query.shape[0]
    target_vocab_size = embeddings_corpus.shape[0]

    # Pre-allocate result tensors
    all_top_k_indices = torch.zeros((num_queries, args.top_k), dtype=torch.long)
    all_top_k_likelihoods = torch.zeros((num_queries, args.top_k), dtype=torch.float32)

    # Normalize corpus embeddings once
    embeddings_corpus_norm = torch.nn.functional.normalize(
        embeddings_corpus.to(device).float(), p=2, dim=1
    )

    for chunk_start in tqdm(
        range(0, num_queries, chunk_size), desc="Processing chunks"
    ):
        chunk_end = min(chunk_start + chunk_size, num_queries)
        chunk_query = embeddings_query[chunk_start:chunk_end].to(device).float()

        with torch.no_grad():
            # Compute similarities for this chunk
            chunk_query_norm = torch.nn.functional.normalize(chunk_query, p=2, dim=1)
            similarities = torch.matmul(chunk_query_norm, embeddings_corpus_norm.t())

            # Generate projection map for this chunk
            chunk_top_k_indices, chunk_top_k_likelihoods = (
                generate_projection_map_chunk(similarities, args)
            )

            # Store results
            all_top_k_indices[chunk_start:chunk_end] = chunk_top_k_indices.cpu()
            all_top_k_likelihoods[chunk_start:chunk_end] = chunk_top_k_likelihoods.cpu()

            # Clear GPU memory
            del (
                similarities,
                chunk_query_norm,
                chunk_top_k_indices,
                chunk_top_k_likelihoods,
            )
            torch.cuda.empty_cache()

    return all_top_k_indices, all_top_k_likelihoods


def generate_projection_map_chunk(similarities, args):
    """Calculates the sparse likelihood map from a similarity matrix chunk."""
    similarities = similarities.abs()
    similarities[similarities > 0.999999999] = 1.0
    max_similarities = torch.max(similarities, dim=1, keepdim=True)[0]
    sharpness = 10.0 * max_similarities
    likelihood = similarities**sharpness

    # Normalize rows
    likelihood = sinkhorn_one_dim(likelihood)

    # Extract final top-k values from the normalized sparse likelihood matrix
    top_k_likelihood, top_k_indices = likelihood.topk(args.top_k, dim=1)

    # Apply weight threshold filtering if specified
    if args.weight_threshold > 0.0:
        threshold_mask = top_k_likelihood >= args.weight_threshold
        top_k_indices = top_k_indices.where(
            threshold_mask, torch.full_like(top_k_indices, -1)
        )

    return top_k_indices, top_k_likelihood


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

    reshaped_input = input_likelihoods.reshape(batch_size * seq_len, source_vocab_size)
    projected_likelihoods_reshaped = torch.matmul(
        reshaped_input, sparse_projection_matrix
    )
    return projected_likelihoods_reshaped.reshape(
        batch_size, seq_len, target_vocab_size
    )


def debug_projection_map(
    top_k_indices,
    top_k_likelihood,
    source_tokenizer,
    target_tokenizer,
    direction="",
    N=2000,
):
    """Debug function to show first N rows with decoded tokens and weights."""
    N = min(N, top_k_indices.shape[0])  # Show first N rows or less
    print(f"\n--- Debugging projection map {direction} (first {N} rows) ---")

    for row_idx in range(N):
        # for row_idx in range(-N,-1):
        # Decode source token
        try:
            token_id = row_idx if row_idx >= 0 else top_k_indices.shape[0] + row_idx
            source_token = source_tokenizer.decode([token_id])
            # source_token = source_tokenizer.convert_ids_to_tokens([token_id])[0]
            source_token_str = repr(source_token)  # Use repr to show special chars
        except:
            source_token_str = f"<ID:{row_idx}>"

        # Build the target tokens with weights string
        row_indices = top_k_indices[row_idx].cpu().numpy()
        row_weights = top_k_likelihood[row_idx].float().cpu().numpy()

        weight_total = 0
        target_parts = []

        if row_weights.max() != row_weights[-1]:
            continue

        for target_idx, weight in zip(row_indices, row_weights):
            try:
                target_token = target_tokenizer.decode([target_idx])
                target_token_str = repr(target_token)
            except:
                target_token_str = f"<ID:{target_idx}>"

            target_parts.append(f"{target_token_str}({weight:.4f})")
            weight_total += weight

        target_string = " ".join(target_parts)
        # print(f"Weight total: {weight_total:.4f}")
        print(f"{source_token_str} -> {target_string}")


def generate_projection_map(similarities, args):
    """Calculates the sparse likelihood map from a similarity matrix."""
    similarities = similarities.abs()
    similarities[similarities > 0.999999999] = 1.0
    max_similarities = torch.max(similarities, dim=1, keepdim=True)[0]
    sharpness = 10.0 * max_similarities
    likelihood = similarities**sharpness

    # Create a sparse representation by keeping only top-k values
    # top_k_likelihood_pre_norm, _ = likelihood.topk(args.top_k, dim=1)
    # likelihood = likelihood.where(likelihood >= top_k_likelihood_pre_norm[:, -1:], torch.zeros_like(likelihood))

    # Normalize the row to sum to 1, handling rows that are all zero
    # row_sums = likelihood.sum(dim=1, keepdim=True)
    # safe_row_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
    # likelihood = likelihood / safe_row_sums
    # pdb.set_trace()
    # likelihood = sinkhorn_one_dim(likelihood)

    # Get the final top-k values and their indices from the sparse, normalized likelihood matrix
    top_k_likelihood, top_k_indices = likelihood.topk(args.top_k, dim=1)

    # Store top-k values before zeroing (to avoid losing them)
    row_indices = torch.arange(likelihood.shape[0]).unsqueeze(1).expand(-1, args.top_k)
    top_k_values = likelihood[row_indices, top_k_indices].clone()

    # Zero out entire likelihood matrix in-place, then restore only top-k elements
    likelihood.zero_()
    likelihood[row_indices, top_k_indices] = top_k_values

    # likelihood = sinkhorn(likelihood, n_iters=1)
    # likelihood = sinkhorn(likelihood, n_iters=1) works the best

    likelihood = sinkhorn_one_dim(likelihood)

    # Extract final top-k values from the normalized sparse likelihood matrix
    top_k_likelihood, top_k_indices = likelihood.topk(args.top_k, dim=1)

    # Apply weight threshold filtering if specified
    if args.weight_threshold > 0.0:
        print(f"Applying weight threshold filter: {args.weight_threshold}")
        # Create mask for values above threshold
        # pdb.set_trace()
        threshold_mask = top_k_likelihood >= args.weight_threshold

        # set indices to -1 where threshold is not met
        top_k_indices = top_k_indices.where(
            threshold_mask, torch.full_like(top_k_indices, -1)
        )

        # # Count how many values per row are above threshold
        # valid_counts = threshold_mask.sum(dim=1)
        # total_filtered = (valid_counts == 0).sum().item()
        # total_kept = threshold_mask.sum().item()
        # total_possible = top_k_likelihood.numel()

        # print(f"Kept {total_kept}/{total_possible} ({100*total_kept/total_possible:.1f}%) projections above threshold")

        # if total_filtered > 0:
        #     print(f"Warning: {total_filtered} tokens have no projections above threshold {args.weight_threshold}")

        # # Zero out values below threshold
        # filtered_likelihood = top_k_likelihood * threshold_mask.to(top_k_likelihood.dtype)
        # filtered_indices = top_k_indices.clone()

        # # For rows with no values above threshold, keep the top value to avoid empty rows
        # empty_rows = valid_counts == 0
        # if empty_rows.any():
        #     print(f"Keeping top projection for {empty_rows.sum().item()} tokens with no values above threshold")
        #     filtered_likelihood[empty_rows, 0] = top_k_likelihood[empty_rows, 0]

        # top_k_likelihood = filtered_likelihood
        # top_k_indices = filtered_indices

    # pdb.set_trace()

    return top_k_indices, top_k_likelihood


# --- Main Execution ---
if __name__ == "__main__":
    # Validate model selection arguments
    validate_model_selection(args)

    # 1. Load Tokenizers and deterministically assign A and B
    # Use model names if provided, otherwise use indices
    if args.model_a_name is not None:
        model_1 = {"id": args.model_a_name}
        print(f"Using provided model A name: {args.model_a_name}")
    else:
        model_1 = {"id": MODEL_LIST[args.model_a_index]}
        print(f"Using model A from index {args.model_a_index}: {model_1['id']}")

    model_1["name"] = model_1["id"].split("/")[-1]
    print(f"Loading first tokenizer: {model_1['name']}")
    model_1["tokenizer"] = load_tokenizer(model_1["id"])

    if args.model_b_name is not None:
        model_2 = {"id": args.model_b_name}
        print(f"Using provided model B name: {args.model_b_name}")
    else:
        model_2 = {"id": MODEL_LIST[args.model_b_index]}
        print(f"Using model B from index {args.model_b_index}: {model_2['id']}")

    model_2["name"] = model_2["id"].split("/")[-1]
    print(f"Loading second tokenizer: {model_2['name']}")
    model_2["tokenizer"] = load_tokenizer(model_2["id"])

    # Deterministically assign model_A and model_B based on alphabetical order of names
    if model_1["name"] > model_2["name"]:
        model_A, model_B = model_2, model_1
    else:
        model_A, model_B = model_1, model_2

    print(f"\nAssigned Source (A): {model_A['name']}")
    print(f"Assigned Target (B): {model_B['name']}")

    source_vocab_size = model_A["tokenizer"].vocab_size
    target_vocab_size = model_B["tokenizer"].vocab_size
    # get the top k tokens from the source and target vocab from model config file
    model_A_config = AutoConfig.from_pretrained(
        model_A["id"], trust_remote_code=True if "nvidia" in model_A["id"] else False
    )
    model_B_config = AutoConfig.from_pretrained(
        model_B["id"], trust_remote_code=True if "nvidia" in model_B["id"] else False
    )
    # pdb.set_trace()
    source_vocab_size = model_A_config.vocab_size
    if "gemma" not in model_B["id"]:
        target_vocab_size = model_B_config.vocab_size
    else:
        target_vocab_size = model_B_config.text_config.vocab_size
    # print(f"Source top k tokens: {model_A_top_k_tokens}")
    # print(f"Target top k tokens: {model_B_top_k_tokens}")

    print(f"Source vocab size (full): {source_vocab_size}")
    print(f"Target vocab size (full): {target_vocab_size}")
    # exit()

    if 0:
        # just debugging learned projection map
        # learned_projection_map = torch.load("models/runs/s4_l1q4b_lr0_kl1_ce0_k1_emb_top10_transformation_matrices/learned_projection_map_latest.pt")
        # learned_projection_map = torch.load("cross_tokenizer_data/projection_map_Llama-3.2_to_Qwen3_multitoken_top_64_double.pt")
        learned_projection_map = torch.load(
            "cross_tokenizer_data/projection_matrix_learned_llama_qwen_top5.pt"
        )
        top_k_indices_A_to_B = learned_projection_map["indices"]
        top_k_likelihood_A_to_B = learned_projection_map["likelihoods"]
        debug_projection_map(
            top_k_indices_A_to_B,
            top_k_likelihood_A_to_B,
            model_A["tokenizer"],
            model_B["tokenizer"],
            "A -> B",
            N=150000,
        )
        exit()

    # 2. Select and Load Embedding Model
    embedding_model_index = 3  # Default to a good LLM embedder
    selected_model_info = EMBEDDING_MODEL_CHOICES[embedding_model_index]
    embedding_model_name = selected_model_info["name"]
    embedding_model_type = selected_model_info["type"]
    print(f"\nUsing embedding model: {embedding_model_name} ({embedding_model_type})")

    # 3. Generate or Load Embeddings
    canonicalization_suffix = "_canonical" if args.use_canonicalization else "_raw"
    embeddings_path_A = os.path.join(
        args.data_dir,
        f"embeddings_{model_A['name']}_{embedding_model_name.replace('/', '_')}_full{canonicalization_suffix}.pt",
    )
    embeddings_path_B = os.path.join(
        args.data_dir,
        f"embeddings_{model_B['name']}_{embedding_model_name.replace('/', '_')}_full{canonicalization_suffix}.pt",
    )

    if (
        not args.force_recompute
        and os.path.exists(embeddings_path_A)
        and os.path.exists(embeddings_path_B)
    ):
        print("Loading cached embeddings...")
        model_A["embeddings"] = load_data(embeddings_path_A).to(DEVICE)
        model_B["embeddings"] = load_data(embeddings_path_B).to(DEVICE)
    else:
        print("Generating new embeddings...")

        # Generate raw decoded tokens
        raw_tokens_A = [
            model_A["tokenizer"].decode([idx])
            for idx in range(model_A["tokenizer"].vocab_size)
        ]
        raw_tokens_B = [
            model_B["tokenizer"].decode([idx])
            for idx in range(model_B["tokenizer"].vocab_size)
        ]

        # Apply canonicalization if requested
        if args.use_canonicalization:
            # Import canonicalization function
            import sys

            sys.path.append(".")
            from tokenalign import TokenAligner

            print("Applying token canonicalization before embedding generation...")
            decoded_tokens_A = [
                TokenAligner._canonical_token(token) for token in raw_tokens_A
            ]
            decoded_tokens_B = [
                TokenAligner._canonical_token(token) for token in raw_tokens_B
            ]

            # Show some examples of canonicalization
            print("Canonicalization examples:")
            for i in range(min(10, len(raw_tokens_A))):
                if raw_tokens_A[i] != decoded_tokens_A[i]:
                    print(f"  Model A: '{raw_tokens_A[i]}' -> '{decoded_tokens_A[i]}'")
            for i in range(min(10, len(raw_tokens_B))):
                if raw_tokens_B[i] != decoded_tokens_B[i]:
                    print(f"  Model B: '{raw_tokens_B[i]}' -> '{decoded_tokens_B[i]}'")

            print(
                f"Applied canonicalization to {len(decoded_tokens_A)} tokens for model A and {len(decoded_tokens_B)} tokens for model B"
            )
        else:
            print("Using raw decoded tokens without canonicalization")
            decoded_tokens_A = raw_tokens_A
            decoded_tokens_B = raw_tokens_B

        if embedding_model_type == "sbert":
            from sentence_transformers import SentenceTransformer

            sbert_model = SentenceTransformer(embedding_model_name, device=DEVICE)
            model_A["embeddings"] = sbert_model.encode(
                decoded_tokens_A, convert_to_tensor=True, show_progress_bar=True
            )
            model_B["embeddings"] = sbert_model.encode(
                decoded_tokens_B, convert_to_tensor=True, show_progress_bar=True
            )
        elif embedding_model_type == "llm_first_layer":
            llm_tokenizer = AutoTokenizer.from_pretrained(
                embedding_model_name, trust_remote_code=True
            )
            if llm_tokenizer.pad_token is None:
                llm_tokenizer.pad_token = llm_tokenizer.eos_token
            llm_model = AutoModel.from_pretrained(
                embedding_model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
            ).to(DEVICE)
            model_A["embeddings"] = get_llm_first_layer_embeddings(
                decoded_tokens_A,
                llm_tokenizer,
                llm_model,
                MAX_SEQ_LENGTH_EMBEDDING,
                DEVICE,
            )
            model_B["embeddings"] = get_llm_first_layer_embeddings(
                decoded_tokens_B,
                llm_tokenizer,
                llm_model,
                MAX_SEQ_LENGTH_EMBEDDING,
                DEVICE,
            )

        save_data(model_A["embeddings"], embeddings_path_A)
        save_data(model_B["embeddings"], embeddings_path_B)

    # 4. Compute Similarity and Generate Projection Maps (chunked to save memory)
    print("\nComputing projection map in chunks to save memory...")
    chunk_size = 500  # Process 500 tokens at a time to avoid OOM
    top_k_indices_A_to_B, top_k_likelihood_A_to_B = compute_chunked_projection_map(
        model_A["embeddings"],
        model_B["embeddings"],
        args,
        DEVICE,
        chunk_size=chunk_size,
    )

    # Note: Exact match enforcement is skipped in chunked mode for simplicity
    # The chunked approach processes similarities in small batches to avoid OOM
    if 0:
        debug_projection_map(
            top_k_indices_A_to_B,
            top_k_likelihood_A_to_B,
            model_A["tokenizer"],
            model_B["tokenizer"],
            "A -> B",
        )

    # print("Generating B -> A projection map...")
    # top_k_indices_B_to_A, top_k_likelihood_B_to_A = generate_projection_map(similarities.T, args)
    # debug_projection_map(top_k_indices_B_to_A, top_k_likelihood_B_to_A, model_B['tokenizer'], model_A['tokenizer'], "B -> A")

    # 5. Save the Combined Projection Map
    print("\nSaving combined projection map...")
    model_a_clean_name = clean_model_name_for_filename(model_A["name"])
    model_b_clean_name = clean_model_name_for_filename(model_B["name"])
    # output_filename = f"temp_projection_map_{model_a_clean_name}_to_{model_b_clean_name}_bidirectional_top_{args.top_k}.pt"
    output_filename = f"temp_projection_map_{model_a_clean_name}_to_{model_b_clean_name}_top_{args.top_k}"
    # if args.skip_exact_enforcement:
    #     output_filename += "_no_exact"
    output_filename += ".pt"
    if args.weight_threshold > 0.0:
        output_filename = output_filename.replace(
            ".pt", f"_thresh_{args.weight_threshold:.3f}.pt"
        )
    output_path = os.path.join(args.data_dir, output_filename)

    torch.save(
        {
            "indices": top_k_indices_A_to_B.cpu(),
            "likelihoods": top_k_likelihood_A_to_B.cpu(),
            "model_A_id": model_A["id"],
            "model_B_id": model_B["id"],
        },
        output_path,
    )

    # torch.save({
    #     "A_to_B": {
    #         "indices": top_k_indices_A_to_B.cpu(),
    #         "likelihoods": top_k_likelihood_A_to_B.cpu()
    #     },
    #     "B_to_A": {
    #         "indices": top_k_indices_B_to_A.cpu(),
    #         "likelihoods": top_k_likelihood_B_to_A.cpu()
    #     },
    #     "model_A_id": model_A['id'],
    #     "model_B_id": model_B['id'],
    # }, output_path)
    print(f"Saved combined projection map to: {output_path}")

    # 6. Example Usage of the Projection Function
    print("\n--- Testing projection function (A -> B) ---")
    # Create a dummy likelihood tensor: [BATCH, SEQ, vocab_size_A]
    source_vocab_size_A = model_A["embeddings"].shape[0]
    target_vocab_size_B = model_B["embeddings"].shape[0]
    dummy_tensor = torch.randn(
        1, 4096, source_vocab_size_A, device=DEVICE, dtype=torch.bfloat16
    )

    # Transform this tensor using the projection map (convert to float32 for compatibility)
    projected_tensor = project_token_likelihoods(
        dummy_tensor.float(),
        top_k_indices_A_to_B,
        top_k_likelihood_A_to_B,
        target_vocab_size_B,
        DEVICE,
    )
    print(f"Input tensor shape: {dummy_tensor.shape}")
    print(f"Projected tensor shape: {projected_tensor.shape}")
    print("Projection test successful.")
