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

"""Stage checksum-authorized CCC problems with only the Python standard library."""

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


def reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number: {value}")


def finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite JSON number: {value}")
    return result


def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(data: str | bytes) -> Any:
    return json.loads(
        data,
        parse_constant=reject_constant,
        parse_float=finite_float,
        object_pairs_hook=unique_keys,
    )


def json_bytes(value: Any, *, sort_keys: bool = False) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=sort_keys,
        allow_nan=False,
    ).encode("utf-8")


def stage(manifest_path: Path, output: Path) -> dict[str, Any]:
    """Validate all source bytes and problem hashes before publishing output.

    Paths in the manifest are relative to the manifest directory. Existing
    output is never replaced. Problem hashes are authoritative: a historical
    serializer mismatch is an error, not permission to recompute the authority.
    """
    manifest_bytes = manifest_path.read_bytes()
    manifest = read_json(manifest_bytes)
    expected = {}
    for record in manifest["problems"]:
        key = (str(record["competition_id"]), str(record["problem_id"]))
        if key in expected:
            raise ValueError(f"Duplicate problem in manifest: {key}")
        expected[key] = record["sha256"]
    if not expected or not manifest["sources"]:
        raise ValueError("Manifest requires nonempty problems and sources")
    if output.exists():
        raise FileExistsError(output)
    seen = set()
    digest = hashlib.sha256()
    # Same-directory staging allows atomic, no-clobber publication via link().
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(fd, "wb") as destination:
            for source_record in manifest["sources"]:
                source = manifest_path.parent / source_record["path"]
                source_digest = hashlib.sha256()
                with source.open("rb") as stream:
                    for line in stream:
                        source_digest.update(line)
                        if not line.strip():
                            continue
                        entry = read_json(line)
                        competition = str(
                            entry.get("competition_id")
                            or entry.get("competition")
                            or ""
                        )
                        field = (
                            "metadata"
                            if entry.get("metadata") is not None
                            else "problems"
                        )
                        selected = {}
                        for problem, value in entry[field].items():
                            key = (competition, problem)
                            if key not in expected:
                                continue
                            actual = hashlib.sha256(
                                json_bytes(value, sort_keys=True)
                            ).hexdigest()
                            if actual != expected[key]:
                                raise ValueError(f"Problem checksum mismatch: {key}")
                            if key not in seen:
                                selected[problem] = value
                                seen.add(key)
                        if selected:
                            payload = json_bytes(entry | {field: selected}) + b"\n"
                            destination.write(payload)
                            digest.update(payload)
                if source_digest.hexdigest() != source_record["sha256"]:
                    raise ValueError(f"Source checksum mismatch: {source}")
            if seen != expected.keys():
                raise ValueError(f"Missing problems: {sorted(expected.keys() - seen)}")
            destination.flush()
            os.fsync(destination.fileno())
        os.link(temporary, output)
    finally:
        os.unlink(temporary)
    return {
        "path": str(output),
        "sha256": digest.hexdigest(),
        "verified_problems": len(seen),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stage(args.manifest, args.output), indent=2))


if __name__ == "__main__":
    main()
