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
import os

import torch
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

from tools.x_token._shared import (
    clean_model_name_for_filename,
    project_token_likelihoods,
    sinkhorn_one_dim,
)


EXACT_MATCH_ONLY = False


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments for projection-matrix generation."""
    parser = argparse.ArgumentParser(
        description="Generate a sparse projection map between a student and a teacher tokenizer.",
    )
    parser.add_argument(
        "--student-model",
        type=str,
        required=True,
        help="HuggingFace model name for the student tokenizer (source vocabulary).",
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        required=True,
        help="HuggingFace model name for the teacher tokenizer (target vocabulary).",
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
        "--use_canonicalization",
        action="store_true",
        help=(
            "Apply token canonicalization before generating embeddings to normalize "
            "different tokenizer representations (e.g. Ġ vs ▁ prefixes, Ċ vs \\n)."
        ),
    )
    return parser.parse_args()


EMBEDDING_MODEL_CHOICES = [
    {"name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "type": "sbert"},
    {"name": "sentence-transformers/all-mpnet-base-v2", "type": "sbert"},
    {"name": "sentence-transformers/all-MiniLM-L6-v2", "type": "sbert"},
    {"name": "Qwen/Qwen3-Embedding-4B", "type": "llm_first_layer"},
    {"name": "Qwen/Qwen3-Embedding-0.6B", "type": "llm_first_layer"},
]

MAX_SEQ_LENGTH_EMBEDDING = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Helper Functions ---

def load_tokenizer(model_id_or_path):
    """Loads a HuggingFace tokenizer, setting a pad token if necessary."""
    tok = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def save_data(data, filename):
    """Saves data to a torch file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(data.cpu(), filename)
    print(f"Data saved to {filename}")

def load_data(filename):
    """Loads data from a torch file."""
    return torch.load(filename)

def get_llm_first_layer_embeddings(decoded_tokens_list, llm_embedding_tokenizer, llm_embedding_model, max_seq_length_embedding, device, batch_size=32):
    """Generates embeddings using the first layer of a given LLM."""
    all_embeddings = []
    llm_embedding_model.eval()
    embedding_dim = llm_embedding_model.config.hidden_size

    for i in tqdm(range(0, len(decoded_tokens_list), batch_size), desc="Encoding tokens with LLM"):
        batch_tokens = decoded_tokens_list[i:i + batch_size]
        inputs = llm_embedding_tokenizer(
            batch_tokens, return_tensors="pt", padding=True, truncation=True,
            max_length=max_seq_length_embedding, add_special_tokens=False,
        ).to(device)

        with torch.no_grad():
            outputs = llm_embedding_model(**inputs, output_hidden_states=True)
            first_layer_output = outputs.hidden_states[0]

            for k in range(first_layer_output.shape[0]):
                valid_token_mask = inputs['attention_mask'][k] == 1
                if valid_token_mask.sum() > 0:
                    pooled_embedding = first_layer_output[k, valid_token_mask].mean(dim=0)
                    all_embeddings.append(pooled_embedding)
                else:
                    all_embeddings.append(torch.zeros(embedding_dim, device=device))

    return torch.stack(all_embeddings).to(device)


def compute_chunked_projection_map(embeddings_query, embeddings_corpus, args, device, chunk_size=1000):
    """Computes projection map in chunks to save memory."""
    num_queries = embeddings_query.shape[0]
    target_vocab_size = embeddings_corpus.shape[0]

    # Pre-allocate result tensors
    all_top_k_indices = torch.zeros((num_queries, args.top_k), dtype=torch.long)
    all_top_k_likelihoods = torch.zeros((num_queries, args.top_k), dtype=torch.float32)

    # Normalize corpus embeddings once
    embeddings_corpus_norm = torch.nn.functional.normalize(embeddings_corpus.to(device).float(), p=2, dim=1)

    for chunk_start in tqdm(range(0, num_queries, chunk_size), desc="Processing chunks"):
        chunk_end = min(chunk_start + chunk_size, num_queries)
        chunk_query = embeddings_query[chunk_start:chunk_end].to(device).float()

        with torch.no_grad():
            # Compute similarities for this chunk
            chunk_query_norm = torch.nn.functional.normalize(chunk_query, p=2, dim=1)
            similarities = torch.matmul(chunk_query_norm, embeddings_corpus_norm.t())

            # Generate projection map for this chunk
            chunk_top_k_indices, chunk_top_k_likelihoods = generate_projection_map_chunk(similarities, args)

            # Store results
            all_top_k_indices[chunk_start:chunk_end] = chunk_top_k_indices.cpu()
            all_top_k_likelihoods[chunk_start:chunk_end] = chunk_top_k_likelihoods.cpu()

            # Clear GPU memory
            del similarities, chunk_query_norm, chunk_top_k_indices, chunk_top_k_likelihoods
            torch.cuda.empty_cache()

    return all_top_k_indices, all_top_k_likelihoods

def generate_projection_map_chunk(similarities, args):
    """Calculates the sparse likelihood map from a similarity matrix chunk."""
    similarities = similarities.abs()
    similarities[similarities > 0.999999999] = 1.0
    max_similarities = torch.max(similarities, dim=1, keepdim=True)[0]
    sharpness = 10.0 * max_similarities
    likelihood = similarities ** sharpness

    # Normalize rows
    likelihood = sinkhorn_one_dim(likelihood)

    # Extract final top-k values from the normalized sparse likelihood matrix
    top_k_likelihood, top_k_indices = likelihood.topk(args.top_k, dim=1)

    # Apply weight threshold filtering if specified
    if args.weight_threshold > 0.0:
        threshold_mask = top_k_likelihood >= args.weight_threshold
        top_k_indices = top_k_indices.where(threshold_mask, torch.full_like(top_k_indices, -1))

    return top_k_indices, top_k_likelihood

# --- Main Execution ---
if __name__ == "__main__":
    args = parse_arguments()

    if args.student_model == args.teacher_model:
        raise ValueError(
            f"Cannot use the same model for both student and teacher: {args.student_model}"
        )

    # 1. Load student and teacher tokenizers directly from --student-model / --teacher-model.
    #    No alphabetical swap — the projection direction follows the CLI args.
    student = {"id": args.student_model}
    student["name"] = student["id"].split("/")[-1]
    print(f"Loading student tokenizer: {student['name']}")
    student["tokenizer"] = load_tokenizer(student["id"])

    teacher = {"id": args.teacher_model}
    teacher["name"] = teacher["id"].split("/")[-1]
    print(f"Loading teacher tokenizer: {teacher['name']}")
    teacher["tokenizer"] = load_tokenizer(teacher["id"])

    print(f"\nSource (student): {student['name']}")
    print(f"Target (teacher): {teacher['name']}")

    student_config = AutoConfig.from_pretrained(
        student["id"], trust_remote_code="nvidia" in student["id"]
    )
    teacher_config = AutoConfig.from_pretrained(
        teacher["id"], trust_remote_code="nvidia" in teacher["id"]
    )
    source_vocab_size = student_config.vocab_size
    if "gemma" in teacher["id"]:
        target_vocab_size = teacher_config.text_config.vocab_size
    else:
        target_vocab_size = teacher_config.vocab_size

    print(f"Source vocab size (full): {source_vocab_size}")
    print(f"Target vocab size (full): {target_vocab_size}")

    # 2. Select and Load Embedding Model
    embedding_model_index = 3 # Default to a good LLM embedder
    selected_model_info = EMBEDDING_MODEL_CHOICES[embedding_model_index]
    embedding_model_name = selected_model_info["name"]
    embedding_model_type = selected_model_info["type"]
    print(f"\nUsing embedding model: {embedding_model_name} ({embedding_model_type})")

    # 3. Generate or Load Embeddings
    canonicalization_suffix = "_canonical" if args.use_canonicalization else "_raw"
    embeddings_path_student = os.path.join(
        args.data_dir,
        f"embeddings_{student['name']}_{embedding_model_name.replace('/', '_')}_full{canonicalization_suffix}.pt",
    )
    embeddings_path_teacher = os.path.join(
        args.data_dir,
        f"embeddings_{teacher['name']}_{embedding_model_name.replace('/', '_')}_full{canonicalization_suffix}.pt",
    )

    if (
        not args.force_recompute
        and os.path.exists(embeddings_path_student)
        and os.path.exists(embeddings_path_teacher)
    ):
        print("Loading cached embeddings...")
        student["embeddings"] = load_data(embeddings_path_student).to(DEVICE)
        teacher["embeddings"] = load_data(embeddings_path_teacher).to(DEVICE)
    else:
        print("Generating new embeddings...")

        # Generate raw decoded tokens
        raw_tokens_student = [
            student["tokenizer"].decode([idx])
            for idx in range(student["tokenizer"].vocab_size)
        ]
        raw_tokens_teacher = [
            teacher["tokenizer"].decode([idx])
            for idx in range(teacher["tokenizer"].vocab_size)
        ]

        # Apply canonicalization if requested
        if args.use_canonicalization:
            from nemo_rl.algorithms.x_token.token_aligner import canonical_token

            print("Applying token canonicalization before embedding generation...")
            decoded_tokens_student = [canonical_token(token) for token in raw_tokens_student]
            decoded_tokens_teacher = [canonical_token(token) for token in raw_tokens_teacher]

            # Show some examples of canonicalization
            print("Canonicalization examples:")
            for i in range(min(10, len(raw_tokens_student))):
                if raw_tokens_student[i] != decoded_tokens_student[i]:
                    print(f"  student: '{raw_tokens_student[i]}' -> '{decoded_tokens_student[i]}'")
            for i in range(min(10, len(raw_tokens_teacher))):
                if raw_tokens_teacher[i] != decoded_tokens_teacher[i]:
                    print(f"  teacher: '{raw_tokens_teacher[i]}' -> '{decoded_tokens_teacher[i]}'")

            print(
                f"Applied canonicalization to {len(decoded_tokens_student)} student tokens "
                f"and {len(decoded_tokens_teacher)} teacher tokens"
            )
        else:
            print("Using raw decoded tokens without canonicalization")
            decoded_tokens_student = raw_tokens_student
            decoded_tokens_teacher = raw_tokens_teacher

        if embedding_model_type == "sbert":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "The sbert embedding path requires `sentence-transformers` to be installed. "
                    "Install it with `uv pip install sentence-transformers`, "
                    "or pick an embedding model with type `llm_first_layer` instead."
                ) from e
            sbert_model = SentenceTransformer(embedding_model_name, device=DEVICE)
            student["embeddings"] = sbert_model.encode(decoded_tokens_student, convert_to_tensor=True, show_progress_bar=True)
            teacher["embeddings"] = sbert_model.encode(decoded_tokens_teacher, convert_to_tensor=True, show_progress_bar=True)
        elif embedding_model_type == "llm_first_layer":
            llm_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name, trust_remote_code=True)
            if llm_tokenizer.pad_token is None:
                llm_tokenizer.pad_token = llm_tokenizer.eos_token
            llm_model = AutoModel.from_pretrained(
                embedding_model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
            ).to(DEVICE)
            student["embeddings"] = get_llm_first_layer_embeddings(decoded_tokens_student, llm_tokenizer, llm_model, MAX_SEQ_LENGTH_EMBEDDING, DEVICE)
            teacher["embeddings"] = get_llm_first_layer_embeddings(decoded_tokens_teacher, llm_tokenizer, llm_model, MAX_SEQ_LENGTH_EMBEDDING, DEVICE)

        save_data(student["embeddings"], embeddings_path_student)
        save_data(teacher["embeddings"], embeddings_path_teacher)

    # 4. Compute Similarity and Generate Projection Maps (chunked to save memory)
    print("\nComputing projection map in chunks to save memory...")
    chunk_size = 500  # Process 500 tokens at a time to avoid OOM
    top_k_indices_student_to_teacher, top_k_likelihood_student_to_teacher = compute_chunked_projection_map(
        student["embeddings"], teacher["embeddings"], args, DEVICE, chunk_size=chunk_size
    )

    # Note: Exact match enforcement is skipped in chunked mode for simplicity.
    # The chunked approach processes similarities in small batches to avoid OOM.

    # 5. Save the Combined Projection Map
    print("\nSaving combined projection map...")
    student_clean_name = clean_model_name_for_filename(student["name"])
    teacher_clean_name = clean_model_name_for_filename(teacher["name"])
    output_filename = f"temp_projection_map_{student_clean_name}_to_{teacher_clean_name}_top_{args.top_k}.pt"
    if args.weight_threshold > 0.0:
        output_filename = output_filename.replace(".pt", f"_thresh_{args.weight_threshold:.3f}.pt")
    output_path = os.path.join(args.data_dir, output_filename)

    # Metadata keys use the student/teacher framing; legacy `model_A_id`/
    # `model_B_id` from older PT-reference artifacts are accepted on the
    # load side in minimal_projection_via_multitoken.py.
    torch.save(
        {
            "indices": top_k_indices_student_to_teacher.cpu(),
            "likelihoods": top_k_likelihood_student_to_teacher.cpu(),
            "student_model_id": student["id"],
            "teacher_model_id": teacher["id"],
        },
        output_path,
    )

    print(f"Saved combined projection map to: {output_path}")

    # 6. Example Usage of the Projection Function
    print("\n--- Testing projection function (student -> teacher) ---")
    source_vocab_size_student = student["embeddings"].shape[0]
    target_vocab_size_teacher = teacher["embeddings"].shape[0]
    dummy_tensor = torch.randn(
        1, 4096, source_vocab_size_student, device=DEVICE, dtype=torch.bfloat16
    )

    # Transform this tensor using the projection map (convert to float32 for compatibility)
    projected_tensor = project_token_likelihoods(
        dummy_tensor.float(),
        top_k_indices_student_to_teacher,
        top_k_likelihood_student_to_teacher,
        target_vocab_size_teacher,
        DEVICE,
    )
    print(f"Input tensor shape: {dummy_tensor.shape}")
    print(f"Projected tensor shape: {projected_tensor.shape}")
    print("Projection test successful.")
