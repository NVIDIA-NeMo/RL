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

"""Split a JSON payload into bounded string chunks and reassemble it.

Managed Dynamo reaches each vLLM engine through an HTTP admin route whose request
body is capped at 2 MiB, which refit metadata for a large model can exceed. The
driver encodes the payload once, sends the chunks one request at a time, and the
NeMo RL worker extension reassembles them inside the engine.

Stdlib only: the driver and the vLLM worker processes both import this.
"""

import base64
import hashlib
import json
import uuid
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

# Characters of base64 text per chunk. With the JSON envelope around it, one chunk
# stays well under Dynamo's 2 MiB request body limit.
STAGED_CHUNK_CHARS = 1 << 20


@dataclass(frozen=True)
class StagedPayload:
    """One encoded payload: a fresh key, the digest of its JSON, and its chunks."""

    key: str
    sha256: str
    chunks: tuple[str, ...]


def encode_staged_payload(payload: Any) -> StagedPayload:
    """Encode ``payload`` as strict JSON, compress it, and split it into chunks.

    Strict means no NaN/Infinity and no fallback encoder, so anything that is not
    plain JSON (tensors, dtypes, tuples as keys) fails here rather than arriving
    altered. Dict order is preserved, which callers rely on for ordered metadata.
    """
    raw = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
    text = base64.b64encode(zlib.compress(raw)).decode("ascii")
    chunks = tuple(
        text[start : start + STAGED_CHUNK_CHARS]
        for start in range(0, len(text), STAGED_CHUNK_CHARS)
    )
    return StagedPayload(
        key=uuid.uuid4().hex,
        sha256=hashlib.sha256(raw).hexdigest(),
        chunks=chunks,
    )


def decode_staged_payload(chunks: Sequence[str], sha256: str) -> Any:
    """Reassemble chunks in order and return the payload, verifying its digest."""
    raw = zlib.decompress(base64.b64decode("".join(chunks)))
    digest = hashlib.sha256(raw).hexdigest()
    if digest != sha256:
        raise ValueError(
            f"Staged payload digest {digest} does not match the expected {sha256}."
        )
    return json.loads(raw)
