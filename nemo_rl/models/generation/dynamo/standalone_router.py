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

import argparse
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
import uvicorn
import zmq
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from dynamo._core import RadixTree, ZmqKvEventListener

logger = logging.getLogger(__name__)


class RouterRequest(BaseModel):
    local_hashes: List[int]
    num_tokens: int


class RouterResponse(BaseModel):
    worker_id: int


class LoadMetrics(BaseModel):
    kv_cache_usage: float
    num_waiting_reqs: int


def setup_zmq_subscriber(context: zmq.Context, endpoint: str) -> zmq.Socket[bytes]:
    socket = context.socket(zmq.SUB)
    socket.connect(endpoint)
    socket.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all messages
    socket.setsockopt(zmq.CONFLATE, 1)  # Only keep latest message
    socket.setsockopt(zmq.RCVTIMEO, 1)  # 1ms timeout (very short)
    return socket


class KvRouter:
    def __init__(
        self,
        block_size: int = 64,
        num_workers: int = 4,
        base_kv_events_port: int = 5557,
        base_metrics_port: int = 5657,
        worker_ips: Optional[List[str]] = None,
        # Router config (matching Rust KvRouterConfig defaults)
        overlap_score_weight: float = 1.0,
        router_temperature: float = 0.0,  # 0 = greedy, >0 = softmax sampling
    ):
        self.num_workers = num_workers
        self.block_size = block_size
        self.overlap_score_weight = overlap_score_weight
        self.router_temperature = router_temperature

        self.radix_tree = RadixTree()

        self.kv_usages = [0.0] * num_workers
        self.waitings = [0] * num_workers
        # Track active decode blocks per worker (for cost function)
        self.decode_blocks = [0] * num_workers
        
        self.round_robin_counter = 0

        # Use worker IPs if provided, otherwise fall back to localhost
        if worker_ips is None:
            worker_ips = ["localhost"] * num_workers
        
        self.context = zmq.Context()
        self.load_listeners = []
        self.kv_listeners = []
        
        for worker_id in range(num_workers):
            worker_ip = worker_ips[worker_id] if worker_id < len(worker_ips) else "localhost"
            metrics_endpoint = f"tcp://{worker_ip}:{base_metrics_port + worker_id}"
            kv_endpoint = f"tcp://{worker_ip}:{base_kv_events_port + worker_id}"
            
            print(f"[Router] Worker {worker_id}: metrics={metrics_endpoint}, kv={kv_endpoint}")
            
            self.load_listeners.append(
                setup_zmq_subscriber(self.context, metrics_endpoint)
            )
            self.kv_listeners.append(
                ZmqKvEventListener(kv_endpoint, "", block_size)
            )

        self.background_tasks: list[asyncio.Task] = []
        self.load_tasks: list[asyncio.Task] = []
        self.indexer_tasks: list[asyncio.Task] = []
        
        # Configure logging for this module if not already configured
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        logger.info("Router initialized")

    async def start_background_tasks(self):
        """Start background tasks for load and indexer updates"""
        logger.info("Starting router background tasks...")
        self.background_tasks.append(asyncio.create_task(self.periodic_update_load()))
        self.background_tasks.append(
            asyncio.create_task(self.periodic_update_indexer())
        )

    async def periodic_update_load(self):
        async def update_load(worker_id: int):
            try:
                while True:
                    try:
                        metrics_dict = self.load_listeners[worker_id].recv_json(
                            zmq.NOBLOCK
                        )
                        metrics = LoadMetrics.model_validate(metrics_dict)
                        self.kv_usages[worker_id] = metrics.kv_cache_usage
                        self.waitings[worker_id] = metrics.num_waiting_reqs
                    except zmq.Again:
                        pass
                    except Exception as e:
                        logger.warning(
                            f"Error receiving metrics for worker {worker_id}: {e}"
                        )

                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                logger.debug(
                    "Load update task cancelled for worker %s", worker_id
                )
                raise

        worker_tasks: list[asyncio.Task] = []
        for worker_id in range(self.num_workers):
            worker_tasks.append(asyncio.create_task(update_load(worker_id)))
        self.load_tasks = worker_tasks

        try:
            if worker_tasks:
                await asyncio.gather(*worker_tasks)
        except asyncio.CancelledError:
            raise
        finally:
            for task in worker_tasks:
                task.cancel()
            if worker_tasks:
                await asyncio.gather(*worker_tasks, return_exceptions=True)
            self.load_tasks = []

    async def periodic_update_indexer(self):
        async def update_tree(worker_id: int):
            try:
                while True:
                    try:
                        kv_events: list[str] = await self.kv_listeners[
                            worker_id
                        ].get_events()
                        for event in kv_events:
                            event = json.loads(event)
                            self.radix_tree.apply_event(
                                worker_id, json.dumps(event).encode("utf-8")
                            )
                    except zmq.Again:
                        pass
                    except Exception as e:
                        logger.warning(
                            f"Error receiving KV events for worker {worker_id}: {e}"
                        )

                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                logger.debug(
                    "Indexer update task cancelled for worker %s", worker_id
                )
                raise

        worker_tasks: list[asyncio.Task] = []
        for worker_id in range(self.num_workers):
            worker_tasks.append(asyncio.create_task(update_tree(worker_id)))
        self.indexer_tasks = worker_tasks

        try:
            if worker_tasks:
                await asyncio.gather(*worker_tasks)
        except asyncio.CancelledError:
            raise
        finally:
            for task in worker_tasks:
                task.cancel()
            if worker_tasks:
                await asyncio.gather(*worker_tasks, return_exceptions=True)
            self.indexer_tasks = []

    async def get_best_worker(self, local_hashes: list[int], num_tokens: int) -> int:
        """Select the best worker using the Dynamo KV router cost function.
        
        Cost function (lower is better):
            logit = overlap_weight * prefill_blocks + decode_blocks
            
        Where:
            - prefill_blocks = tokens that need prefilling / block_size
            - overlap = cached blocks that can be reused (reduces prefill)
            - decode_blocks = active decode blocks on the worker
            
        Selection:
            - temperature=0: Greedy (select min logit, random tie-break)
            - temperature>0: Softmax sampling (lower logit = higher probability)
        """
        try:
            if num_tokens <= 0:
                raise ValueError("num_tokens must be positive")

            # local_hashes can be empty
            raw_scores = self.radix_tree.find_matches(local_hashes).scores
            
            # Debug: log raw scores and input info periodically
            if hasattr(self, '_get_best_worker_count'):
                self._get_best_worker_count += 1
            else:
                self._get_best_worker_count = 1
            
            if self._get_best_worker_count <= 3 or self._get_best_worker_count % 100 == 0:
                print(
                    f"[Router] get_best_worker #{self._get_best_worker_count}: "
                    f"num_hashes={len(local_hashes)}, num_tokens={num_tokens}, "
                    f"raw_scores={dict(raw_scores)}"
                )

            # raw_scores keys are tuples (worker_id, rank) - aggregate by worker_id
            # Sum all scores for each worker_id regardless of the second element
            # overlap = number of cached blocks for this worker
            overlap_blocks = {}
            for key, score in raw_scores.items():
                if isinstance(key, tuple):
                    wid = key[0]
                else:
                    wid = key
                overlap_blocks[wid] = overlap_blocks.get(wid, 0) + score

            # Calculate logits for each worker (lower is better)
            worker_logits = {}
            for worker_id in range(self.num_workers):
                # Number of cached blocks for this worker
                overlap = overlap_blocks.get(worker_id, 0)
                
                # prefill_tokens = total tokens - cached tokens
                # This is the number of tokens that need to be prefilled
                cached_tokens = overlap * self.block_size
                prefill_tokens = max(0, num_tokens - cached_tokens)
                prefill_blocks = prefill_tokens / self.block_size
                
                # decode_blocks = active decode blocks on this worker
                # Use waiting requests as a proxy for decode load
                decode_blocks = self.decode_blocks[worker_id] + self.waitings[worker_id]
                
                # Cost function: overlap_weight * prefill_blocks + decode_blocks
                # Lower is better (minimize prefill cost + decode load)
                logit = self.overlap_score_weight * prefill_blocks + decode_blocks
                worker_logits[worker_id] = logit
                
                logger.info(
                    f"worker_id: {worker_id}, logit = {self.overlap_score_weight:.1f} * {prefill_blocks:.3f} + {decode_blocks} "
                    f"= {logit:.3f} (overlap={overlap} blocks, cached={cached_tokens} tokens)"
                )

            # Select worker based on temperature
            if self.router_temperature <= 0:
                # Greedy selection: pick minimum logit with random tie-break
                min_logit = min(worker_logits.values())
                min_workers = [wid for wid, logit in worker_logits.items() if logit == min_logit]
                best_worker_id = int(np.random.choice(min_workers))
            else:
                # Softmax sampling (lower logit = higher probability)
                best_worker_id = self._softmax_sample(worker_logits, self.router_temperature)

            # Predictive update for handling concurrent request bursts
            self.waitings[best_worker_id] += 1

            return best_worker_id

        except Exception as e:
            logger.error(f"Error in get_best_worker: {e}")
            raise
    
    def _softmax_sample(self, logits: dict[int, float], temperature: float) -> int:
        """Softmax sampling over logits (lower logit = higher probability).
        
        Matches the Rust softmax_sample implementation in scheduler.rs.
        """
        if len(logits) == 1:
            return list(logits.keys())[0]
        
        keys = list(logits.keys())
        values = list(logits.values())
        
        min_val = min(values)
        max_val = max(values)
        
        if min_val == max_val:
            # All values same, uniform probability
            return int(np.random.choice(keys))
        
        # Normalize and negate (lower is better, so negate for softmax)
        normalized = [-(v / (max_val - min_val)) for v in values]
        
        # Apply temperature and softmax
        scaled = [v / temperature for v in normalized]
        max_scaled = max(scaled)
        exp_values = [np.exp(v - max_scaled) for v in scaled]  # Subtract max for numerical stability
        sum_exp = sum(exp_values)
        probabilities = [v / sum_exp for v in exp_values]
        
        # Sample from distribution
        return int(np.random.choice(keys, p=probabilities))

    async def get_worker_round_robin(self, local_hashes: list[int], num_tokens: int) -> int:
        # Select worker in round robin fashion
        worker_id = self.round_robin_counter
        self.round_robin_counter = (self.round_robin_counter + 1) % self.num_workers
        
        logger.info(f"Round robin selected worker_id: {worker_id}")
        
        # Predictive update for handling concurrent request bursts
        self.waitings[worker_id] += 1
        
        return worker_id

    async def shutdown(self):
        """Shutdown ZMQ listeners, context, and background tasks"""
        logger.info("Shutting down KvRouter...")

        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()

        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        self.background_tasks = []

        # Close load listeners (ZMQ sockets)
        for listener in self.load_listeners:
            try:
                listener.close()
            except Exception as e:
                logger.error(f"Error closing load listener: {e}")

        # Terminate ZMQ context
        try:
            self.context.term()
            logger.info("ZMQ context terminated successfully")
        except Exception as e:
            logger.error(f"Error terminating ZMQ context: {e}")

        logger.info("KvRouter shutdown completed")


class RouterAPI:
    def __init__(
        self,
        block_size: int = 64,
        num_workers: int = 4,
        base_kv_events_port: int = 5557,
        base_metrics_port: int = 5657,
        port: int = 7000,
        # Router config (matching Rust KvRouterConfig defaults)
        overlap_score_weight: float = 1.0,
        router_temperature: float = 0.0,
    ):
        self.port = port
        self.block_size = block_size
        self.num_workers = num_workers
        self.base_kv_events_port = base_kv_events_port
        self.base_metrics_port = base_metrics_port
        self.overlap_score_weight = overlap_score_weight
        self.router_temperature = router_temperature
        self.router = None
        self.server: Optional[uvicorn.Server] = None
        self.app = FastAPI(
            title="KV Router API", version="0.0.1", lifespan=self.lifespan
        )
        self.setup_routes()

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        # Startup
        self.router = KvRouter(
            block_size=self.block_size,
            num_workers=self.num_workers,
            base_kv_events_port=self.base_kv_events_port,
            base_metrics_port=self.base_metrics_port,
            overlap_score_weight=self.overlap_score_weight,
            router_temperature=self.router_temperature,
        )
        await self.router.start_background_tasks()
        logger.info("Router API started successfully")

        yield

        # Shutdown
        if self.router:
            await self.router.shutdown()

    def setup_routes(self):
        @self.app.post("/find_best_worker", response_model=RouterResponse)
        async def find_best_worker(request: RouterRequest):
            if self.router is None:
                raise HTTPException(status_code=503, detail="Router not initialized")

            try:
                worker_id = await self.router.get_best_worker(
                    request.local_hashes, request.num_tokens
                )
                return RouterResponse(worker_id=worker_id)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"Error finding best worker: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")

        @self.app.post("/find_worker_round_robin", response_model=RouterResponse)
        async def find_worker_round_robin(request: RouterRequest):
            if self.router is None:
                raise HTTPException(status_code=503, detail="Router not initialized")

            try:
                worker_id = await self.router.get_worker_round_robin(
                    request.local_hashes, request.num_tokens
                )

                return RouterResponse(worker_id=worker_id)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"Error finding worker round robin: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")

    async def start(self):
        """Start the router API server"""
        logger.info(f"Starting Router API server on port {self.port}")
        config = uvicorn.Config(
            self.app, host="0.0.0.0", port=self.port, log_level="info"
        )
        self.server = uvicorn.Server(config)
        try:
            await self.server.serve()
        finally:
            # ensure reference cleared if serve exits unexpectedly
            self.server = None

    async def stop(self):
        """Request the server to shutdown gracefully."""
        if self.server is None:
            logger.warning("RouterAPI.stop() called but server is not running")
            return

        logger.info("Stopping Router API server...")
        self.server.should_exit = True
        try:
            await self.server.shutdown()
        finally:
            self.server = None


def main():
    parser = argparse.ArgumentParser(description="KV Router API Server")

    parser.add_argument(
        "--block-size", type=int, default=64, help="Block size for caching"
    )
    parser.add_argument(
        "--num-workers", type=int, default=2, help="Number of worker processes"
    )
    parser.add_argument(
        "--base-kv-events-port", type=int, default=5557, help="Base port for KV events"
    )
    parser.add_argument(
        "--base-metrics-port", type=int, default=5657, help="Base port for metrics"
    )
    parser.add_argument(
        "--port", type=int, default=7000, help="Port to serve the Router API on"
    )
    parser.add_argument(
        "--overlap-score-weight", type=float, default=1.0,
        help="Weight for overlap score in cost function (default: 1.0)"
    )
    parser.add_argument(
        "--router-temperature", type=float, default=0.0,
        help="Temperature for softmax sampling (0.0 = greedy, default: 0.0)"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    api = RouterAPI(
        block_size=args.block_size,
        num_workers=args.num_workers,
        base_kv_events_port=args.base_kv_events_port,
        base_metrics_port=args.base_metrics_port,
        port=args.port,
        overlap_score_weight=args.overlap_score_weight,
        router_temperature=args.router_temperature,
    )

    async def run_with_shutdown():
        try:
            await api.start()
        except KeyboardInterrupt:
            logger.info(
                "Received KeyboardInterrupt, shutting down Router API server..."
            )
        except Exception as e:
            logger.exception(f"Unhandled exception: {e}")

    try:
        asyncio.run(run_with_shutdown())
    except KeyboardInterrupt:
        logger.info("Force shutdown via KeyboardInterrupt.")


if __name__ == "__main__":
    main()
