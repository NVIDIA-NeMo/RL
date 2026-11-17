"""Distributed, Redis-backed rate limiter for outbound tool calls (e.g. web-search APIs).

Works across any number of workers/nodes since the count lives in Redis, not a local
process state -- one shared Redis instance can back independent rate limits for multiple
tools at once, keyed by `key`.
"""

import asyncio
import logging
import random
import time

from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class RateLimiter:
    """Distributed rate limiter.

    Attributes:
        redis_host (str): Redis host.
        redis_port (int): Redis port.
        redis_pswd (str): Redis password.
        rate_limit_window_sec (int): Rate limit time window in seconds.
        rate_limit_count (int): Maximum number of requests in the time
            window.
        key (str): Rate limit key.
        raise_after (int): Maximum number of attempts before raising
            an error.
        max_wait_sec (int): Maximum time waited between retries.
    """

    def __init__(
        self,
        redis_host: str,
        redis_port: int,
        redis_pswd: str,
        rate_limit_window_sec: int,
        rate_limit_count: int,
        key: str = "rate_limit",
        raise_after: int = 60,
        max_wait_sec: int = 10,
    ):
        """Initialize the rate limiter.

        Args:
            redis_host (str): Redis host (e.g., lrdn0910).
            redis_port (int): Redis port.
            redis_pswd (str): Redis password.
            rate_limit_window_sec (int): Rate limit time window in
                seconds (e.g. 1 for a "N requests per second" API limit).
            rate_limit_count (int): Maximum number of requests in
                the time window.
            key (str, optional): Unique key of this rate limiter.
                Use a distinct key per tool to run independent limits off one
                shared Redis instance. Defaults to "rate_limit".
            raise_after (int, optional): Maximum number of attempts before
                raising an error. Defaults to 60.
            max_wait_sec (int, optional): Upper bound of the jittered retry
                wait -- retries wait a random amount of time in
                [rate_limit_window_sec, max_wait_sec], so this redistributes
                contending requests instead of having them all retry in lockstep.
                Must be >= rate_limit_window_sec. Defaults to 10.
        """
        if max_wait_sec < rate_limit_window_sec:
            raise ValueError(
                f"max_wait_sec ({max_wait_sec}) must be >= rate_limit_window_sec "
                f"({rate_limit_window_sec}): max_wait_sec is the upper bound of the jittered "
                "retry wait and rate_limit_window_sec its lower bound."
            )
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_pswd = redis_pswd
        self.rate_limit_window_sec = rate_limit_window_sec
        self.rate_limit_count = rate_limit_count
        self.key = key
        self.raise_after = raise_after
        self.max_wait_sec = max_wait_sec

    async def acquire_lock(self) -> None:
        """Try to acquire the lock."""
        async with Redis(
            host=self.redis_host,
            port=self.redis_port,
            password=self.redis_pswd,
            decode_responses=True,
        ) as client:
            for i in range(self.raise_after):
                logger.debug("Acquiring lock - turn %s", i)
                now = time.time()
                count = await self._get_concurrent_request_count(now, client)
                logger.info("Number of concurrent requests %s", count)
                if count <= self.rate_limit_count:
                    return

                logger.warning(
                    "Number of concurrent requests '%s' is greater"
                    " than the rate limit '%s'",
                    count,
                    self.rate_limit_count,
                )
                await self._remove_value(now, client)
                await self._wait_jitted()
            else:
                msg = f"Failed to acquire lock after {self.raise_after} turns"
                raise RuntimeError(msg)

    async def _get_concurrent_request_count(
        self, now: float, client: Redis
    ) -> int:
        """Get the number of concurrent requests.

        The pipeline has four steps:
        1. Add one to the count
        2. Remove all timestamps older than the window
        3. Get the count of the remaining timestamps
        4. Set the expiration for autocleaning.

        Args:
            now (float): Lock value.
            client (Redis): Client.

        Returns:
            int: Number of concurrent requests.
        """
        async with client.pipeline(transaction=True) as pipe:
            pipe.zadd(self.key, {str(now): now})
            pipe.zremrangebyscore(
                self.key, 0, now - self.rate_limit_window_sec
            )
            pipe.zcard(self.key)
            pipe.expire(self.key, self.rate_limit_window_sec)
            results = await pipe.execute(raise_on_error=True)

            return results[2]

    async def _remove_value(self, now: float, client: Redis) -> None:
        """Remove one to the number of requests.

        Lock has not been acquired, so the count must be
        reduced by one.

        Args:
            now (float): Lock value.
            client (Redis): Client.
        """
        async with client.pipeline(transaction=True) as pipe:
            pipe.zrem(self.key, str(now))
            await pipe.execute(raise_on_error=True)

    async def _wait_jitted(self) -> None:
        """Wait a jittered amount of time before retry"""
        jitted_wait = random.randint(self.rate_limit_window_sec, self.max_wait_sec)
        await asyncio.sleep(jitted_wait)
