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
"""Validate both CSD tables against the exact student and teacher tokenizers."""

import argparse
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerFast

from tools.csd.build_subtoks_table import (
    hard_special_ids,
    load_tokenizer,
    ordinary_event_matches,
    require_supported_tokenizer,
    role_mapping,
    special_match_is_valid,
    tokenizer_fingerprint,
)


def validate_table(
    path: Path,
    *,
    source: PreTrainedTokenizerFast,
    destination: PreTrainedTokenizerFast,
) -> None:
    """Check tensor structure, tokenizer identity, roles, and every mapped event.

    Args:
        path: Table saved by ``tools.csd.build_subtoks_table``.
        source: Tokenizer whose vocabulary indexes rows.
        destination: Tokenizer whose IDs populate each chain.

    Raises:
        ValueError: The table is inconsistent or maps a different token event.
    """
    require_supported_tokenizer(source, label="source tokenizer")
    require_supported_tokenizer(destination, label="destination tokenizer")
    table = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(table, dict):
        raise ValueError(f"{path}: expected a table dictionary")
    for key, expected in (
        ("V_src", len(source)),
        ("V_dst", len(destination)),
        ("source_fingerprint", tokenizer_fingerprint(source)),
        ("destination_fingerprint", tokenizer_fingerprint(destination)),
    ):
        if table.get(key) != expected:
            raise ValueError(
                f"{path}: {key} does not match the supplied tokenizer pair"
            )
    subtoks, lengths = table.get("subtoks"), table.get("lengths")
    if not isinstance(subtoks, torch.Tensor) or not isinstance(lengths, torch.Tensor):
        raise ValueError(f"{path}: missing subtoks or lengths tensors")
    if subtoks.dtype != torch.long or lengths.dtype != torch.long:
        raise ValueError(f"{path}: subtoks and lengths must be int64")
    if (
        subtoks.ndim != 2
        or subtoks.shape[0] != len(source)
        or subtoks.shape[1] < 1
        or lengths.shape != (len(source),)
        or table.get("N_max") != subtoks.shape[1]
    ):
        raise ValueError(f"{path}: invalid tensor shapes or N_max metadata")
    if bool(((lengths < 0) | (lengths > subtoks.shape[1])).any()):
        raise ValueError(f"{path}: invalid chain lengths")
    valid = torch.arange(subtoks.shape[1])[None, :] < lengths[:, None]
    if bool(((subtoks[valid] < 0) | (subtoks[valid] >= len(destination))).any()):
        raise ValueError(f"{path}: destination token ID out of range")
    if bool((subtoks[~valid] != -1).any()):
        raise ValueError(f"{path}: padding must be -1")

    roles = role_mapping(source, destination)
    source_specials = hard_special_ids(source)
    destination_specials = hard_special_ids(destination)
    for source_id, destination_id in roles.items():
        if (
            lengths[source_id].item() != 1
            or subtoks[source_id, 0].item() != destination_id
        ):
            raise ValueError(
                f"{path}: source row {source_id} has an incorrect semantic role"
            )
    for source_id in (lengths > 0).nonzero().flatten().tolist():
        destination_ids = subtoks[source_id, : lengths[source_id]].tolist()
        if source_id in source_specials:
            valid_event = len(destination_ids) == 1 and (
                roles.get(source_id) == destination_ids[0]
                or (
                    destination_ids[0] in destination_specials
                    and special_match_is_valid(
                        source,
                        destination,
                        source_id=source_id,
                        destination_id=destination_ids[0],
                    )
                )
            )
        else:
            valid_event = not any(
                i in destination_specials for i in destination_ids
            ) and ordinary_event_matches(
                source,
                destination,
                source_id=source_id,
                destination_ids=destination_ids,
            )
        if not valid_event:
            raise ValueError(
                f"{path}: source row {source_id} does not preserve its token event"
            )
    print(f"Validated {path}: {int((lengths > 0).sum())} mapped rows")


def main() -> None:
    """Validate directional tables before starting a GPU training job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-tokenizer", required=True)
    parser.add_argument("--teacher-tokenizer", required=True)
    parser.add_argument("--student-revision", default=None)
    parser.add_argument("--teacher-revision", default=None)
    parser.add_argument("--student-to-teacher", type=Path, required=True)
    parser.add_argument("--teacher-to-student", type=Path, required=True)
    args = parser.parse_args()
    student = load_tokenizer(args.student_tokenizer, revision=args.student_revision)
    teacher = load_tokenizer(args.teacher_tokenizer, revision=args.teacher_revision)
    validate_table(args.student_to_teacher, source=student, destination=teacher)
    validate_table(args.teacher_to_student, source=teacher, destination=student)


if __name__ == "__main__":
    main()
