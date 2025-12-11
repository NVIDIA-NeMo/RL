# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
import os
import uuid
from typing import AsyncGenerator, Optional

import zmq
from vllm.config import (
    CacheConfig,
    ModelConfig,
    ObservabilityConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.distributed.kv_events import KVEventsConfig
from vllm.inputs.data import TokensPrompt
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.metrics.loggers import StatLoggerBase
from vllm.v1.metrics.stats import IterationStats, SchedulerStats

logger = logging.getLogger(__name__)


class MetricsPublisher(StatLoggerBase):
    """Stat logger publisher. Wrapper for the WorkerMetricsPublisher to match the StatLoggerBase interface."""

    def __init__(self, port: int) -> None:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{port}")
        self.port = port
        
        logger.info(f"[MetricsPublisher] ZMQ PUB socket bound to tcp://*:{port}")

    def record(
        self,
        scheduler_stats: SchedulerStats,
        iteration_stats: Optional[IterationStats],
        engine_idx: int = 0,
    ):
        # Send metrics over ZMQ
        metrics_data = {
            "num_waiting_reqs": scheduler_stats.num_waiting_reqs,
            "kv_cache_usage": scheduler_stats.kv_cache_usage,
        }

        self.socket.send_json(metrics_data)

    def log_engine_initialized(self) -> None:
        print(f"[MetricsPublisher] log_engine_initialized() called on port {self.port}")


class LoggerFactory:
    """Factory for creating stat logger publishers. Required by vLLM."""

    def __init__(self, port: int) -> None:
        self.port = port

    def __call__(self, vllm_config: VllmConfig, dp_rank: int) -> StatLoggerBase:
        return MetricsPublisher(port=self.port)

