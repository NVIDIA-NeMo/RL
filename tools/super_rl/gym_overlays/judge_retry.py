# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded retries around one judge decision, never around a policy rollout."""

import asyncio
from functools import wraps
import logging


def retry_judge_errors(error_type: type[Exception]):
    """Decorate a judge-only method; valid negative decisions return immediately.

    Passing Gym's exception type explicitly keeps this helper dependency-free.
    The decorated method must rebuild only the same judge request from immutable
    arguments. Do not decorate agent.run(), verify(), or tool execution.
    """

    def decorate(method):
        @wraps(method)
        async def wrapper(self, *args, **kwargs):
            attempts = self.config.judge_max_attempts
            if type(attempts) is not int or not 1 <= attempts <= 8:
                raise ValueError("judge_max_attempts must be an integer in [1, 8]")
            for attempt in range(1, attempts + 1):
                try:
                    return await method(self, *args, **kwargs)
                except error_type as error:
                    logging.getLogger("nemo_gym.judge_retry").warning(
                        "Judge attempt %d/%d failed: resource=%s method=%s error=%s",
                        attempt,
                        attempts,
                        self.config.name,
                        method.__name__,
                        str(error)[:512],
                    )
                    if attempt == attempts:
                        raise
                    await asyncio.sleep(min(2 ** (attempt - 1), 8))

        return wrapper

    return decorate
