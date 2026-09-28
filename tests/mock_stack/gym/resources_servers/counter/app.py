# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re

from fastapi import Request
from nemo_gym.rollout_correlation import current_logical_rollout_id
from resources_servers.example_session_state_mgmt.app import (
    StatefulCounterResourcesServer,
    StatefulCounterVerifyRequest,
)


class Counter(StatefulCounterResourcesServer):
    async def verify(self, request: Request, body: StatefulCounterVerifyRequest):
        result = await super().verify(request, body)
        if result.reward != 1:
            raise RuntimeError(
                "The resource counter does not match the expected number of turns"
            )
        sibling = re.search(r"_g(\d+)$", current_logical_rollout_id())
        if sibling is None:
            raise RuntimeError("Missing stable sibling identity")
        return result.model_copy(update={"reward": float(int(sibling[1]) + 1)})


if __name__ == "__main__":
    Counter.run_webserver()
