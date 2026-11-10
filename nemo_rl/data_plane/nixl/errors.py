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
"""Exceptions raised across the data plane."""

from __future__ import annotations


class UnitFull(Exception):
    """A storage unit cannot satisfy an allocation."""


class DataPlaneLostKeys(Exception):
    """Some keys are unreadable because the unit holding them is gone.

    ``keys`` lists the affected keys so the caller can drop and regenerate.
    """

    def __init__(self, keys: list[str], reason: str = "") -> None:
        self.keys = list(keys)
        self.reason = reason
        super().__init__(
            f"{len(self.keys)} key(s) lost{': ' + reason if reason else ''}"
        )

    # Inside TransferQueue the storage-level key is ``f"{global_index}@{field}"``.
    # Callers that hold TQ metadata can map indexes back to their sample ids
    # with ``client.kv_retrieve_keys(global_indexes, partition_id)``.
    def global_indexes(self) -> list[int]:
        out: list[int] = []
        for k in self.keys:
            head, _, _ = k.partition("@")
            if head.isdigit():
                g = int(head)
                if g not in out:
                    out.append(g)
        return out

    def fields(self) -> list[str]:
        out: list[str] = []
        for k in self.keys:
            _, sep, f = k.partition("@")
            if sep and f not in out:
                out.append(f)
        return out


# None of the retryable classes below subclasses DataPlaneLostKeys, so an
# ``except DataPlaneLostKeys`` handler never treats a timeout as data loss.
class DataPlaneTransferFailed(Exception):
    """A transfer or a unit reply failed, but the unit still holds the data.

    Retry the operation. ``in_flight`` is True when a NIXL transfer may still
    be running; its local buffer is retired and the retry gets a fresh one.
    """

    def __init__(self, keys: list[str], reason: str = "", *, in_flight: bool = False):
        self.keys = list(keys)
        self.reason = reason
        self.in_flight = in_flight
        super().__init__(
            f"transfer of {len(self.keys)} key(s) failed"
            f"{': ' + reason if reason else ''}"
        )


class DataPlaneTimeout(Exception):
    """An operation ran out of its deadline (``op_timeout_s``). Retryable."""


class DataPlaneBusy(Exception):
    """A unit is temporarily unable to serve; retried inside the operation."""

    def __init__(self, msg: str = "", reason: str | None = None) -> None:
        super().__init__(msg)
        self.reason = reason


class DataPlaneFatal(RuntimeError):
    """This process can no longer move data safely. Not retryable.

    Raised once the endpoint is POISONED (unfinished transfers retain more
    than ``graveyard_max_bytes``) or the client pool has no buffer left.
    """


class TransferError(Exception):
    """A NIXL transfer finished in the ERR state or timed out.

    ``in_flight`` is True when the transfer was posted and never seen DONE.
    Releasing its handle does not stop it (UCX cancels asynchronously, POSIX
    just drops the handle), so its destination may still be written later and
    must not be reused.
    """

    def __init__(self, msg: str = "", *, in_flight: bool = False) -> None:
        super().__init__(msg)
        self.in_flight = in_flight


class LostBlobs(Exception):
    """Some blobs are unreadable for good; the client maps them back to keys.

    Raised only when the bytes are confirmed gone: the unit replied that it
    no longer has the blob, or the unit itself is dead.
    """

    def __init__(self, blob_ids: list[str], reason: str) -> None:
        self.blob_ids = list(blob_ids)
        self.reason = reason
        super().__init__(f"{len(self.blob_ids)} blob(s) lost: {reason}")


class StaleRead(Exception):
    """A read outlived its pin; the caller must re-resolve."""


class TransportPolicyError(RuntimeError):
    """UCX is configured so that NIXL could move bytes over TCP sockets.

    the NIXL data plane only runs over RDMA (plus same-host shared memory). NIXL's
    UCX backend honours ``UCX_*`` environment variables over anything passed in
    ``backend_init_params``, so the environment is where this is decided.
    """
