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

"""Router-replay routes of a cut call: staged per chunk, joined on restore."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from nemo_rl.experience.route_assembly import (
    ROUTE_MISSING_SENTINEL,
    verify_route_fragment_integrity,
)


def _join_route_pieces(pieces: Sequence[Any]) -> np.ndarray:
    """Join route pieces into one integer ``[rows, layers, topk]`` array.

    vLLM publishes every piece of a request as a numpy array of one dtype, so
    one C-level join in that dtype checks them all; anything else is checked
    piece by piece.

    Raises:
        ValueError: a piece is not integer ``[rows, layers, topk]`` with the
            others' dimensions.
    """
    first = pieces[0]
    try:
        if not isinstance(first, np.ndarray):
            raise TypeError("not a numpy route piece")
        joined = np.concatenate(pieces, axis=0, dtype=first.dtype, casting="no")
    except (TypeError, ValueError):
        arrays = [np.asarray(piece) for piece in pieces]
        shape = tuple(arrays[0].shape[1:])
        if any(array.ndim != 3 or tuple(array.shape[1:]) != shape for array in arrays):
            raise ValueError(
                "vLLM route pieces must be integer [rows, layers, topk]"
            ) from None
        joined = np.concatenate(arrays, axis=0)
    if joined.ndim != 3 or min(joined.shape[1:]) < 1 or joined.dtype.kind not in "iu":
        raise ValueError("vLLM route pieces must be integer [rows, layers, topk]")
    return joined


def _final_route_placeholder(
    layers: int, topk: int, dtype: torch.dtype
) -> torch.Tensor:
    """The route row staged for a prefix's last token, which no step has routed.

    A cut builds it when staging and a restore rebuilds it when joining
    chunks; both must use this helper, or every resumed call fails the
    digest check. It holds ``ROUTE_MISSING_SENTINEL``, as the final row of an
    uncut call's routes does.
    """
    return torch.full((1, layers, topk), ROUTE_MISSING_SENTINEL, dtype=dtype)


def _prefix_routes_sha256(
    delta_routes: torch.Tensor | None, *, generated_prefix_tokens: int
) -> str:
    """Hash the routes a resumed call keeps for its restored generated prefix.

    ``delta_routes`` holds one row per prefix delta token. The kept rows are
    the generated prefix tokens' except the last, whose route the resumed
    call's own prefill computes. Restore and completion both log this, so the
    staged row can be checked against the routes restored from TQ.

    Returns:
        A SHA-256 hex digest over the kept rows as int32, or ``"none"`` when
        the call carries no routes.
    """
    if delta_routes is None:
        return "none"
    end = int(delta_routes.shape[0])
    kept = delta_routes[end - generated_prefix_tokens : end - 1]
    return hashlib.sha256(
        kept.to(torch.int32).contiguous().numpy().tobytes()
    ).hexdigest()


class _EngineRouteCache:
    """The engine's route pieces for one request, joined as they arrive.

    The joined routes live in a buffer that doubles when full, so a cut copies
    only the pieces vLLM appended since the previous one and the total copying
    is linear in the request's length. It follows the engine, not the cuts, so
    a rolled-back cut does not rewind it. Only the call's lifecycle owner
    touches it.
    """

    __slots__ = ("_dtype", "_buffer", "_rows", "_piece_count", "_last_piece")

    def __init__(self, dtype: torch.dtype) -> None:
        self._dtype = torch.empty((), dtype=dtype).numpy().dtype
        self._buffer: np.ndarray | None = None
        self._rows = 0
        self._piece_count = 0
        self._last_piece: Any = None

    def absorb(self, pieces: Sequence[Any]) -> torch.Tensor:
        """Copy the engine's new route pieces in; return the joined routes.

        ``pieces`` is the request's whole route list as vLLM holds it. vLLM
        only appends to it and never replaces a piece, so the pieces already
        joined are skipped. Returns the request's full-sequence routes as a
        view of the buffer; rows already returned are never written again.

        Raises:
            ValueError: the list shrank, a joined piece was replaced, or a new
                piece is not integer ``[rows, layers, topk]`` with this
                request's model dimensions.
        """
        covered = self._piece_count
        if len(pieces) < covered:
            raise ValueError(
                f"vLLM's route list shrank: {len(pieces)} pieces, {covered} already joined"
            )
        if covered and pieces[covered - 1] is not self._last_piece:
            raise ValueError("vLLM replaced a route piece that was already joined")
        if len(pieces) > covered:
            added = _join_route_pieces(pieces[covered:])
            shape = tuple(added.shape[1:])
            buffer = self._buffer
            if buffer is not None and tuple(buffer.shape[1:]) != shape:
                raise ValueError("vLLM route pieces changed model dimensions")
            dtype = self._dtype
            # vLLM publishes unsigned expert IDs (uint8, or uint16 past 256
            # experts); the signed checkpoint dtype holds the -1 placeholder.
            # Numpy would wrap an ID that does not fit, so check rather than
            # stage a wrong expert.
            if added.size and not np.can_cast(added.dtype, dtype, casting="safe"):
                limits = np.iinfo(dtype)
                if int(added.min()) < limits.min or int(added.max()) > limits.max:
                    raise ValueError(
                        f"vLLM routed expert IDs do not fit the checkpoint's {dtype}"
                    )
            rows = self._rows + len(added)
            if buffer is None or rows > len(buffer):
                capacity = rows if buffer is None else max(rows, 2 * len(buffer))
                grown = np.empty((capacity, *shape), dtype=dtype)
                if buffer is not None:
                    grown[: self._rows] = buffer[: self._rows]
                buffer = grown
            buffer[self._rows : rows] = added
            # Commit only after every check and copy has succeeded.
            self._buffer = buffer
            self._rows = rows
            self._piece_count = len(pieces)
            self._last_piece = pieces[-1]
        if self._buffer is None:
            raise ValueError("vLLM has not routed any token of the request yet")
        return torch.from_numpy(self._buffer[: self._rows])


def _cut_chunk_routes(
    engine_routes: Any | None,
    *,
    dtype: torch.dtype,
    prev_len: int,
    prompt_len: int,
    restored: torch.Tensor | None,
    staged: int,
    frozen: int,
) -> torch.Tensor | None:
    """The routes of the chunk a cut stages: one row per chunk token.

    ``engine_routes`` are the request's full-sequence routes, read with the
    frozen tokens; ``staged`` and ``frozen`` count the generated tokens already
    staged and frozen by this cut. vLLM only appends routes, so earlier rows
    never change and every chunk lines up with what earlier cuts staged.

    A call's routes are delta-aligned: one row per prompt token past
    ``prev_len`` and per generated token, the last being a placeholder for the
    final token, which no step has routed yet. The first chunk is the whole
    prefix so far, placeholder included; a later chunk's first row replaces
    the previous chunk's placeholder.

    Returns:
        The chunk's routes, or ``None`` when the cut stages no new tokens.

    Raises:
        ValueError: the routes are missing, malformed, or do not cover every
            frozen token.
    """
    if frozen == staged:
        return None
    if engine_routes is None:
        raise ValueError(
            "the request left vLLM before its routes were read for the cut"
        )
    raw = torch.as_tensor(engine_routes, dtype=dtype).cpu()
    if raw.dim() != 3 or raw.shape[1] < 1 or raw.shape[2] < 1:
        raise ValueError("routed experts must have shape [tokens, layers, topk]")
    layers, topk = int(raw.shape[1]), int(raw.shape[2])
    # Delta rows before the generated tokens: the prompt past prev_len, or the
    # restored prefix's rows.
    lead = prompt_len - prev_len
    # Rows parent lineage or the restored cut already own.
    skip = prev_len
    if restored is not None:
        if tuple(restored.shape[1:]) != (layers, topk):
            raise ValueError("restored routed experts changed model dimensions")
        lead = int(restored.shape[0])
        skip += lead - 1
    # The last frozen token is not routed until a later step prefills it.
    needed = lead + frozen - 1
    if restored is not None:
        needed -= lead - 1
    routes = raw[skip : skip + needed]
    if routes.shape[0] < needed:
        # vLLM can omit route rows (prefix caching with chunked prefill).
        # Padding would shift later rows onto the wrong tokens, and a later
        # cut would see different rows than this one staged.
        raise ValueError(
            "vLLM returned fewer routed experts than the frozen tokens need: "
            f"routes={routes.shape[0]} needed={needed}"
        )
    final = _final_route_placeholder(layers, topk, dtype)
    parts = [routes, final] if restored is None else [restored[:-1], routes, final]
    cumulative = torch.cat(parts, dim=0).contiguous()
    if staged == 0 and restored is None:
        return cumulative
    return cumulative[lead + staged - 1 : -1]


def _join_chunk_routes(chunks: Sequence[Any], *, required: bool) -> torch.Tensor | None:
    """Verify each cut chunk's routes against its digest and join them.

    The first chunk stores a normal full, delta-aligned route tensor whose
    last row is the unused final-position placeholder. Every later chunk
    stores one route per chunk token: its first row replaces that
    placeholder, and a new placeholder is appended at the new boundary.

    Raises:
        RuntimeError: routes are missing when ``required``, only some chunks
            carry them, or a chunk's routes fail their checks.
    """
    fragments = [chunk.fragment for chunk in chunks]
    if all(fragment is None for fragment in fragments):
        if required:
            raise RuntimeError("generation-cut checkpoint has no router-replay payload")
        return None
    if any(fragment is None for fragment in fragments):
        raise RuntimeError("generation-cut route fragments are incomplete")

    routes: torch.Tensor | None = None
    for chunk, fragment in zip(chunks, fragments, strict=True):
        snapshot = chunk.snapshot
        if chunk.routed_len != snapshot.delta_len:
            raise RuntimeError(
                "generation-cut route length does not match token delta: "
                f"routes={chunk.routed_len} tokens={snapshot.delta_len}"
            )
        if not verify_route_fragment_integrity(
            fragment,
            extras_digest_version=snapshot.extras_digest_version,
            expected_extras_digest=snapshot.extras_digest,
        ):
            raise RuntimeError("generation-cut route fragment failed integrity check")
        fragment_routes = fragment.routes.cpu().contiguous()
        if fragment_routes.dim() != 3:
            raise RuntimeError("generation-cut routes must have rank 3")
        if routes is None:
            routes = fragment_routes
            continue
        if tuple(fragment_routes.shape[1:]) != tuple(routes.shape[1:]):
            raise RuntimeError("generation-cut route model dimensions changed")
        final = _final_route_placeholder(
            int(fragment_routes.shape[1]),
            int(fragment_routes.shape[2]),
            fragment_routes.dtype,
        )
        routes = torch.cat((routes[:-1], fragment_routes, final), dim=0)

    assert routes is not None
    expected_len = sum(chunk.snapshot.delta_len for chunk in chunks)
    if routes.shape[0] != expected_len:
        raise RuntimeError(
            "rebuilt generation-cut routes do not match cumulative token delta"
        )
    return routes
