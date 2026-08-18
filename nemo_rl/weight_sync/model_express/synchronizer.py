# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""ModelExpress control-plane lifecycle for versioned RL refit."""

from __future__ import annotations

import uuid
from contextlib import nullcontext
from typing import Any, Optional

from nemo_rl.utils.timer import Timer
from nemo_rl.weight_sync.interfaces import WeightSynchronizer


class ModelExpressWeightSynchronizer(WeightSynchronizer):
    """Coordinate MX clients embedded in policy and generation actors."""

    def __init__(
        self,
        *,
        policy: Any,
        generation: Any,
        control_client: Any | None = None,
        payload_format: Any | None = None,
        server_url: str | None = None,
    ) -> None:
        if control_client is None or payload_format is None:
            try:
                from modelexpress_rl import (
                    ModelExpressControlClient,
                    WeightPayloadFormat,
                )
            except ImportError as error:
                raise RuntimeError(
                    "refit_transport='model_express' requires the 'modelexpress' "
                    "distribution from a revision exporting the modelexpress_rl "
                    "control, trainer, and generator clients"
                ) from error
            if control_client is None:
                control_client = ModelExpressControlClient.connect(
                    server_url=server_url
                )
            if payload_format is None:
                payload_format = WeightPayloadFormat.FULL_TENSOR

        self._policy = policy
        self._generation = generation
        self._control = control_client
        self._payload_format = payload_format
        self._server_url = server_url
        self._model_name = generation.cfg["model_name"]
        self._run_id = uuid.uuid4().hex[:8]
        self._next_version_number = 1
        self._next_attempt_number = 1
        self._source_slots: list[str] | None = None
        self._stale = True

    @property
    def is_stale(self) -> bool:
        return self._stale

    def mark_stale(self) -> None:
        self._stale = True

    def init_communicator(self) -> None:
        slots = self._policy.initialize_model_express(server_url=self._server_url)
        if not slots or any(not slot for slot in slots):
            raise RuntimeError(
                "ModelExpress trainer initialization returned no source slots"
            )
        if len(set(slots)) != len(slots):
            raise RuntimeError("ModelExpress trainer source slots must be unique")
        self._generation.initialize_model_express(server_url=self._server_url)
        self._source_slots = sorted(slots)

    def _create_weight_version(self) -> Any:
        assert self._source_slots is not None
        version_number = self._next_version_number
        version = self._control.create_weight_version(
            model_name=self._model_name,
            version_number=version_number,
            idempotency_key=(
                f"{self._run_id}:{version_number}:{self._next_attempt_number}"
            ),
            payload_format=self._payload_format,
            expected_source_slots=self._source_slots,
        )
        self._next_attempt_number += 1
        return version

    def sync_weights(
        self,
        *,
        timer: Optional[Timer] = None,
        kv_scales: Optional[dict[str, float]] = None,
    ) -> None:
        if kv_scales:
            raise NotImplementedError(
                "ModelExpress refit does not yet publish generated KV-cache scales"
            )
        if self._source_slots is None:
            raise RuntimeError(
                "init_communicator() must be called before sync_weights()"
            )

        timer_context = (
            timer.time("prepare_for_generation/transfer_and_update_weights")
            if timer is not None
            else nullcontext()
        )
        with timer_context:
            version = self._create_weight_version()
            try:
                self._policy.publish_model_express_version(version.ref)
                ready = self._control.get_weight_version(version.version_id)
                if ready.state.value != "READY":
                    raise RuntimeError(
                        f"ModelExpress weight version {version.version_id!r} "
                        "did not become READY after all trainer RPCs completed"
                    )
                self._generation.update_weights_from_model_express(version.ref)
            finally:
                self._control.delete_weight_version(version.version_id)
                self._policy.release_model_express_version(version.ref)

        self._next_version_number += 1
        self._stale = False

    def shutdown(self) -> None:
        self._control.close()


__all__ = ["ModelExpressWeightSynchronizer"]
