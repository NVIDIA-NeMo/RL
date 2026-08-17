# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Pinned compatibility adapter for ModelExpress #635 reshard receivers.

ModelExpress #635 exposes ``VllmReshardReceiver.update_weights(step)`` but no
public API to require a publisher step before the first lazy discovery, and no
receiver shutdown method. This adapter owns the extra metadata client used for
the version preflight and confines the one pinned private cleanup dependency
(``receiver._manager.shutdown()``) to a checked boundary.
"""

from __future__ import annotations

import json
import time
from typing import Any


class MxVllmReshardReceiver:
    """Version-gated lifecycle around the exact ModelExpress #635 receiver API."""

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
                "ModelExpress #635 compatibility error: "
                "VllmReshardReceiver._manager.shutdown() is unavailable"
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
        payloads = self._rendezvous.discover_trainers(
            self._num_trainer_sources,
            timeout=self._timeout,
        )
        discover_s = time.perf_counter() - discover_t0
        observed = [payload.publisher_step for payload in payloads]
        if len(observed) != self._num_trainer_sources or any(
            step != version for step in observed
        ):
            raise RuntimeError(
                f"ModelExpress publisher version mismatch: requested {version}, "
                f"observed {observed}"
            )
        install_t0 = time.perf_counter()
        result = self._receiver.update_weights(version, timeout=self._timeout)
        install_s = time.perf_counter() - install_t0
        self._report_phases(version, discover_s, install_s, payloads)
        return result

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
        try:
            tensors_seen = sum(len(getattr(p, "tensors", ()) or ()) for p in payloads)
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
        """Release #635's receiver-owned NIXL manager and both gRPC clients."""
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
