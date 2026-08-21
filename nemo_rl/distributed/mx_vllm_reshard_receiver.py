# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Compatibility adapter around ModelExpress' vLLM reshard receiver.

MX exposes ``VllmReshardReceiver.update_weights(step)`` but two things this path
needs are missing from its public surface: a way to require a publisher step
before the receiver's first lazy discovery, and a receiver shutdown method.

So this adapter owns the extra metadata client used for the version preflight,
and confines the one private dependency it cannot avoid
(``receiver._manager.shutdown()``) to a boundary that is checked at
construction rather than assumed at teardown. If MX grows public equivalents,
this adapter is what shrinks.
"""

from __future__ import annotations

import json
import time
from inspect import signature
from typing import Any, Optional


class MxVllmReshardReceiver:
    """Version-gated lifecycle around ModelExpress' reshard receiver API."""

    def __init__(
        self,
        *,
        model: Any,
        vllm_config: Any,
        model_config: Any,
        model_name: str,
        server_url: str,
        agent_name: str,
        local_rank: int,
        global_rank: int,
        num_trainer_sources: int,
        device: Any,
        listen_port: int,
        timeout: float,
    ) -> None:
        from modelexpress.client import MxClient
        from modelexpress.engines.vllm.refit.receiver import VllmReshardReceiver
        from modelexpress.refit.reshard.rendezvous import MxReshardRendezvous

        self._num_trainer_sources = num_trainer_sources
        self._timeout = timeout
        self._global_rank = global_rank
        # Resolved on first discovery and cached: the installed MX cannot change
        # under a live receiver.
        self._tensorless_discovery: Optional[bool] = None
        # Held for parameter-equality verification, which needs the live params
        # both before and after an install. MX's receiver keeps its own reference.
        self._model = model
        self._client = MxClient(server_url=server_url)
        self._rendezvous = MxReshardRendezvous(
            self._client,
            role="inference",
            rank=global_rank,
            model_name=model_name,
        )
        try:
            self._receiver = VllmReshardReceiver(
                model=model,
                vllm_config=vllm_config,
                model_config=model_config,
                model_name=model_name,
                mx_server=server_url,
                agent_name=agent_name,
                local_rank=local_rank,
                global_rank=global_rank,
                num_trainer_sources=num_trainer_sources,
                device=device,
                listen_port=listen_port,
                timeout=timeout,
            )
        except Exception:
            self._client.close()
            raise
        manager = getattr(self._receiver, "_manager", None)
        if manager is None or not callable(getattr(manager, "shutdown", None)):
            self._client.close()
            raise RuntimeError(
                "ModelExpress compatibility error: "
                "VllmReshardReceiver._manager.shutdown() is unavailable. This "
                "adapter needs it to release the receiver's NIXL registrations; "
                "check the installed modelexpress version."
            )
        self._closed = False

    def update_weights(self, version: int) -> dict[str, Any]:
        """Require every visible trainer stamp before MX can install weights."""
        # Timed separately because MX_REFIT_STAGE records cover only the install
        # that follows, leaving the majority of a MoE refit unattributed. The
        # quorum check costs one list_sources plus a get_metadata round-trip per
        # trainer rank, and each response carries that rank's whole shard table,
        # so it scales with sources rather than with bytes moved.
        discover_t0 = time.perf_counter()
        payloads = self._discover_for_quorum()
        discover_s = time.perf_counter() - discover_t0
        observed = [payload.publisher_step for payload in payloads]
        if len(observed) != self._num_trainer_sources or any(
            step != version for step in observed
        ):
            raise RuntimeError(
                f"ModelExpress publisher version mismatch: requested {version}, "
                f"observed {observed}"
            )
        from nemo_rl.distributed import mx_refit_verify

        verifying = mx_refit_verify.enabled()
        before = mx_refit_verify.fingerprint_model(self._model) if verifying else {}

        install_t0 = time.perf_counter()
        result = self._receiver.update_weights(version, timeout=self._timeout)
        install_s = time.perf_counter() - install_t0

        if verifying and before:
            mx_refit_verify.report(
                version,
                self._global_rank,
                before,
                mx_refit_verify.fingerprint_model(self._model),
            )
        self._report_phases(version, discover_s, install_s, payloads)
        return result

    def _discover_for_quorum(self) -> list:
        """Discover trainer ranks for the version check only.

        This check needs one integer per rank. It does not read shard geometry --
        MX's own receiver discovers that once in ``_prepare`` and keeps it -- so
        asking for the shard tables here rebuilds, every step, a table that is
        identical every step. On Qwen3-30B-A3B that is 78,760 entries across 16
        ranks, and skipping the rebuild removes ~0.8 s of a ~5.2 s check.

        Omits the flag against an MX that predates it, so this does not have to
        land in lockstep with the client change.
        """
        kwargs: dict[str, Any] = {"timeout": self._timeout}
        if self._supports_tensorless_discovery():
            kwargs["with_tensors"] = False
        return self._rendezvous.discover_trainers(self._num_trainer_sources, **kwargs)

    def _supports_tensorless_discovery(self) -> bool:
        """Whether the installed MX accepts ``with_tensors``.

        Asked of the signature rather than by calling and catching TypeError: a
        genuine TypeError raised *inside* discovery would otherwise be swallowed
        and retried as the expensive full fetch, turning a real bug into an
        unexplained per-step slowdown.
        """
        if self._tensorless_discovery is None:
            try:
                parameters = signature(self._rendezvous.discover_trainers).parameters
            except (TypeError, ValueError):
                # Unintrospectable callable (C extension, some mocks). Assume the
                # older contract; the full fetch is slower but always correct.
                self._tensorless_discovery = False
            else:
                self._tensorless_discovery = "with_tensors" in parameters
        return self._tensorless_discovery

    def _report_phases(
        self,
        version: int,
        discover_s: float,
        install_s: float,
        payloads: list,
    ) -> None:
        """Emit the phase split, and never let doing so break a refit.

        Telemetry must not be able to fail the operation it measures. The first
        version of this read ``payload.tensors`` directly and raised
        AttributeError on any payload shape that lacked it, which would abort a
        refit that had already succeeded.
        """

        def entries(payload) -> int:
            # Prefer the recorded count: the quorum path deliberately does not
            # build the shard tables, so len(payload.tensors) would read 0 and
            # hide the figure that showed this cost tracks source count.
            counter = getattr(payload, "entry_count", None)
            if callable(counter):
                return int(counter())
            return len(getattr(payload, "tensors", ()) or ())

        try:
            tensors_seen = sum(entries(p) for p in payloads)
        except TypeError:
            tensors_seen = -1
        try:
            print(
                "MX_RECV_PHASE "
                + json.dumps(
                    {
                        "schema": "mx-recv-phase-v1",
                        "step": version,
                        "rank": self._global_rank,
                        "discover_s": round(discover_s, 6),
                        "mx_update_s": round(install_s, 6),
                        "tensors_seen": tensors_seen,
                        "trainer_sources": len(payloads),
                    }
                ),
                flush=True,
            )
        except Exception:  # noqa: BLE001 - reporting is never worth a failed refit
            pass

    def shutdown(self) -> None:
        """Release the receiver-owned NIXL manager and both gRPC clients."""
        if self._closed:
            return
        self._closed = True
        manager = self._receiver._manager
        try:
            manager.shutdown()
        finally:
            receiver_client = getattr(self._receiver, "_mx_client", None)
            if receiver_client is not None and callable(
                getattr(receiver_client, "close", None)
            ):
                receiver_client.close()
            self._client.close()


__all__ = ["MxVllmReshardReceiver"]
