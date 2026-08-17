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
        payloads = self._rendezvous.discover_trainers(
            self._num_trainer_sources,
            timeout=self._timeout,
        )
        observed = [payload.publisher_step for payload in payloads]
        if len(observed) != self._num_trainer_sources or any(
            step != version for step in observed
        ):
            raise RuntimeError(
                f"ModelExpress publisher version mismatch: requested {version}, "
                f"observed {observed}"
            )
        return self._receiver.update_weights(version, timeout=self._timeout)

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
