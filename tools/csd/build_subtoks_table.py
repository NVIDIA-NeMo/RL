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
"""Build one direction of the CSD sub-token tables on CPU.

Invoke once for student-to-teacher and once with the tokenizers reversed.
Each source vocabulary row contains a complete destination token chain;
unmapped rows have length zero and padding is -1. No model weights are loaded.
Both tokenizers must use a BPE model with a ByteLevel decoder.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from tokenizers.decoders import ByteLevel
from tokenizers.models import BPE
from transformers import AutoTokenizer, PreTrainedTokenizerFast


def special_roles(tokenizer: PreTrainedTokenizerFast) -> dict[str, int | None]:
    """Return semantic roles independently of their token spellings."""
    return {
        "bos": tokenizer.bos_token_id,
        "eos": tokenizer.eos_token_id,
        "pad": tokenizer.pad_token_id,
        "unk": tokenizer.unk_token_id,
        "sep": tokenizer.sep_token_id,
        "cls": tokenizer.cls_token_id,
        "mask": tokenizer.mask_token_id,
    }


def hard_special_ids(tokenizer: PreTrainedTokenizerFast) -> set[int]:
    """Include hard added specials without excluding ordinary added tokens."""
    return set(tokenizer.all_special_ids) | {
        token_id
        for token_id, token in tokenizer.added_tokens_decoder.items()
        if token.special
    }


def tokenizer_fingerprint(tokenizer: PreTrainedTokenizerFast) -> str:
    """Fingerprint vocabulary, tokenizer behavior, and semantic special roles."""
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    # Encode calls update these transient batch options; neither changes the
    # vocabulary events represented by a table.
    backend.pop("padding", None)
    backend.pop("truncation", None)
    payload = {
        "backend": backend,
        "roles": special_roles(tokenizer),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def byte_bpe_model(tokenizer: PreTrainedTokenizerFast) -> BPE | None:
    """Return the byte-preserving BPE model, when this tokenizer has one."""
    backend = tokenizer.backend_tokenizer
    if isinstance(backend.model, BPE) and isinstance(backend.decoder, ByteLevel):
        return backend.model
    return None


def require_supported_tokenizer(
    tokenizer: PreTrainedTokenizerFast, *, label: str
) -> None:
    """Reject decoders whose single-token text can change with context.

    Metaspace and WordPiece decoders can add or remove whitespace or word
    markers depending on token position. Isolated decoding cannot establish
    the complete events needed by a static table for those tokenizers.
    """
    if byte_bpe_model(tokenizer) is None:
        raise ValueError(
            f"{label}: CSD table preparation and validation require a fast "
            "tokenizer with a BPE model and ByteLevel decoder. Other decoder "
            "families, including Metaspace and WordPiece, are unsupported "
            "because isolated token decoding can change events with context."
        )


def role_mapping(
    source: PreTrainedTokenizerFast, destination: PreTrainedTokenizerFast
) -> dict[int, int]:
    """Map unambiguous shared special roles; never resolve aliases by order."""
    destination_roles = special_roles(destination)
    candidates: dict[int, set[int]] = {}
    for role, source_id in special_roles(source).items():
        destination_id = destination_roles[role]
        if source_id is not None and destination_id is not None:
            candidates.setdefault(source_id, set()).add(destination_id)
    return {
        source_id: next(iter(targets))
        for source_id, targets in candidates.items()
        if len(targets) == 1
    }


def special_match_is_valid(
    source: PreTrainedTokenizerFast,
    destination: PreTrainedTokenizerFast,
    *,
    source_id: int,
    destination_id: int,
) -> bool:
    """Permit identical hard-special spellings only when roles do not conflict."""
    source_roles = {k for k, v in special_roles(source).items() if v == source_id}
    destination_roles = {
        k for k, v in special_roles(destination).items() if v == destination_id
    }
    return not (
        source_roles
        and destination_roles
        and source_roles.isdisjoint(destination_roles)
    ) and source.convert_ids_to_tokens(source_id) == destination.convert_ids_to_tokens(
        destination_id
    )


def ordinary_event_matches(
    source: PreTrainedTokenizerFast,
    destination: PreTrainedTokenizerFast,
    *,
    source_id: int,
    destination_ids: list[int],
) -> bool:
    """Check complete events without replacing incomplete UTF-8 fragments.

    Byte-level BPE pairs compare their reversible byte surfaces. Ordinary added
    tokens compare decoded text and conservatively reject replacement characters.
    """
    if not destination_ids:
        return False
    if (
        byte_bpe_model(source) is not None
        and byte_bpe_model(destination) is not None
        and source_id not in source.added_tokens_decoder
        and not any(i in destination.added_tokens_decoder for i in destination_ids)
    ):
        return source.convert_ids_to_tokens(source_id) == "".join(
            destination.convert_ids_to_tokens(destination_ids)
        )
    source_text = source.decode([source_id], clean_up_tokenization_spaces=False)
    destination_text = destination.decode(
        destination_ids, clean_up_tokenization_spaces=False
    )
    return (
        bool(source_text)
        and "\ufffd" not in source_text
        and source_text == destination_text
    )


def build_subtoks_table(
    source: PreTrainedTokenizerFast,
    destination: PreTrainedTokenizerFast,
    *,
    max_subtokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build complete destination chains for every source vocabulary ID.

    Args:
        source: Source tokenizer, including its added tokens.
        destination: Tokenizer whose IDs populate each chain.
        max_subtokens: Positive maximum chain length; longer events are unmapped.

    Returns:
        Integer tensors ``subtoks[V_source, max_subtokens]`` and
        ``lengths[V_source]``, ready for the CSD loss table loader.
    """
    if max_subtokens < 1:
        raise ValueError("max_subtokens must be positive")
    require_supported_tokenizer(source, label="source tokenizer")
    require_supported_tokenizer(destination, label="destination tokenizer")
    subtoks = torch.full((len(source), max_subtokens), -1, dtype=torch.long)
    lengths = torch.zeros(len(source), dtype=torch.long)
    source_specials = hard_special_ids(source)
    destination_specials = hard_special_ids(destination)
    roles = role_mapping(source, destination)
    destination_vocab = destination.get_vocab()
    destination_bpe = byte_bpe_model(destination)
    byte_pair = byte_bpe_model(source) is not None and destination_bpe is not None

    for source_id in range(len(source)):
        surface = source.convert_ids_to_tokens(source_id)
        exact_id = destination_vocab.get(surface)
        destination_ids: list[int] = []
        if source_id in source_specials:
            if source_id in roles:
                destination_ids = [roles[source_id]]
            elif (
                exact_id is not None
                and exact_id in destination_specials
                and special_match_is_valid(
                    source, destination, source_id=source_id, destination_id=exact_id
                )
            ):
                destination_ids = [exact_id]
        else:
            if (
                exact_id is not None
                and exact_id not in destination_specials
                and ordinary_event_matches(
                    source, destination, source_id=source_id, destination_ids=[exact_id]
                )
            ):
                destination_ids = [exact_id]
            else:
                text = source.decode([source_id], clean_up_tokenization_spaces=False)
                candidate = destination.encode(text, add_special_tokens=False)
                if (
                    byte_pair
                    and source_id not in source.added_tokens_decoder
                    and not ordinary_event_matches(
                        source,
                        destination,
                        source_id=source_id,
                        destination_ids=candidate,
                    )
                ):
                    # Token-level Unicode decoding can replace incomplete byte
                    # fragments. BPE on the reversible surface preserves them.
                    assert destination_bpe is not None
                    candidate = [
                        piece.id for piece in destination_bpe.tokenize(surface)
                    ]
                if not any(
                    i in destination_specials for i in candidate
                ) and ordinary_event_matches(
                    source, destination, source_id=source_id, destination_ids=candidate
                ):
                    destination_ids = candidate
        if 0 < len(destination_ids) <= max_subtokens:
            lengths[source_id] = len(destination_ids)
            subtoks[source_id, : len(destination_ids)] = torch.tensor(destination_ids)
    return subtoks, lengths


def load_tokenizer(name: str, *, revision: str | None) -> PreTrainedTokenizerFast:
    """Load only tokenizer assets and reject unsupported slow implementations."""
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, use_fast=True)
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError("CSD table preparation requires a fast tokenizer")
    require_supported_tokenizer(tokenizer, label=name)
    return tokenizer


def main() -> None:
    """Build and save a directional table with tokenizer identity metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tokenizer", required=True)
    parser.add_argument("--destination-tokenizer", required=True)
    parser.add_argument("--source-revision", default=None)
    parser.add_argument("--destination-revision", default=None)
    parser.add_argument("--max-subtokens", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        parser.error(f"{args.output} already exists; use --overwrite to replace it")
    source = load_tokenizer(args.source_tokenizer, revision=args.source_revision)
    destination = load_tokenizer(
        args.destination_tokenizer, revision=args.destination_revision
    )
    subtoks, lengths = build_subtoks_table(
        source, destination, max_subtokens=args.max_subtokens
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "subtoks": subtoks,
            "lengths": lengths,
            "src": args.source_tokenizer,
            "dst": args.destination_tokenizer,
            "N_max": args.max_subtokens,
            "V_src": len(source),
            "V_dst": len(destination),
            "source_fingerprint": tokenizer_fingerprint(source),
            "destination_fingerprint": tokenizer_fingerprint(destination),
        },
        args.output,
    )
    print(
        f"Saved {args.output}: {int((lengths > 0).sum())}/{len(source)} mapped rows, "
        f"{int((lengths == 0).sum())} unmapped rows"
    )


if __name__ == "__main__":
    main()
