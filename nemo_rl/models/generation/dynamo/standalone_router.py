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

# Configure logger to write to file
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create file handler
file_handler = logging.FileHandler("standalone_router.log", mode="a")
file_handler.setLevel(logging.DEBUG)

# Create formatter
formatter = logging.Formatter(
    "%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(file_handler)


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
        worker_addresses: Optional[list[str]] = None,
        worker_ports: Optional[list[int]] = None,  # Actual KV event ports
    ):
        self.num_workers = num_workers
        self.block_size = block_size

        self.radix_tree = RadixTree()

        self.kv_usages = [0.0] * num_workers
        self.waitings = [0] * num_workers
        
        self.round_robin_counter = 0
        
        # Use localhost for all workers if addresses not provided (single-node setup)
        if worker_addresses is None:
            self.worker_addresses = ["localhost"] * num_workers
            logger.info("No worker addresses provided, using localhost for all workers (single-node mode)")
        else:
            if len(worker_addresses) != num_workers:
                raise ValueError(
                    f"Number of worker addresses ({len(worker_addresses)}) must match "
                    f"number of workers ({num_workers})"
                )
            self.worker_addresses = worker_addresses
            logger.info(f"Using distributed mode with worker addresses: {worker_addresses}")
        
        # Use provided ports or compute them from base_port + worker_id
        if worker_ports is not None:
            if len(worker_ports) != num_workers:
                raise ValueError(
                    f"Number of worker ports ({len(worker_ports)}) must match "
                    f"number of workers ({num_workers})"
                )
            self.worker_kv_ports = worker_ports
            self.worker_metrics_ports = [port + (base_metrics_port - base_kv_events_port) for port in worker_ports]
            logger.info(f"Using explicit worker ports for KV events: {self.worker_kv_ports}")
        else:
            # Default: compute ports from base_port + worker_id (for single-node or simple setups)
            self.worker_kv_ports = [base_kv_events_port + i for i in range(num_workers)]
            self.worker_metrics_ports = [base_metrics_port + i for i in range(num_workers)]

        self.context = zmq.Context()
        
        # Create listeners with detailed logging
        self.load_listeners = []
        for worker_id in range(num_workers):
            endpoint = f"tcp://{self.worker_addresses[worker_id]}:{self.worker_metrics_ports[worker_id]}"
            logger.info(f"Router connecting to worker {worker_id} metrics at: {endpoint}")
            self.load_listeners.append(setup_zmq_subscriber(self.context, endpoint))
        
        self.kv_listeners = []
        for worker_id in range(num_workers):
            endpoint = f"tcp://{self.worker_addresses[worker_id]}:{self.worker_kv_ports[worker_id]}"
            logger.info(f"Router connecting to worker {worker_id} KV events at: {endpoint}")
            self.kv_listeners.append(ZmqKvEventListener(endpoint, "", block_size))

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

    async def start_background_tasks(self, enable_kv_indexer: bool = True):
        """Start background tasks for load and indexer updates
        
        Args:
            enable_kv_indexer: If False, skip KV event tracking (useful for round-robin routing)
        """
        logger.info("Starting router background tasks...")
        self.background_tasks.append(asyncio.create_task(self.periodic_update_load()))
        
        if enable_kv_indexer:
            self.background_tasks.append(
                asyncio.create_task(self.periodic_update_indexer())
            )
            logger.info("KV indexer enabled (for KV-aware routing)")
        else:
            logger.info("KV indexer disabled (round-robin mode - no KV tracking needed)")

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
            logger.info(f"Starting KV event listener for worker {worker_id}")
            events_received_count = 0
            events_applied_count = 0
            events_failed_count = 0
            
            try:
                logger.info(f"[periodic_update_indexer] Starting indexer update task for worker {worker_id}")
                while True:
                    try:
                        logger.debug(f"Polling for KV events from worker {worker_id}...")
                        kv_events: list[str] = await self.kv_listeners[
                            worker_id
                        ].get_events()
                        
                        if kv_events:
                            logger.info(
                                f"Received {len(kv_events)} KV event(s) from worker {worker_id}"
                            )
                            events_received_count += len(kv_events)
                        
                        for idx, event in enumerate(kv_events):
                            try:
                                event_dict = json.loads(event)
                                
                                # Dynamo ZmqKvEventListener format: {"event_id": ..., "data": {...}, "dp_rank": ...}
                                # RadixTree expects this EXACT format - do NOT transform it!
                                
                                # Determine event type for logging only
                                event_type = "unknown"
                                if "data" in event_dict:
                                    data = event_dict["data"]
                                    if isinstance(data, dict):
                                        if "stored" in data:
                                            event_type = "block_stored"
                                        elif "removed" in data:
                                            event_type = "block_removed"
                                    elif data == "cleared":
                                        event_type = "all_blocks_cleared"
                                
                                # Debug: Log event info for first few events
                                if idx == 0 and events_received_count <= 10:
                                    block_hashes = "N/A"
                                    if event_type == "block_stored" and isinstance(event_dict.get("data"), dict):
                                        stored = event_dict["data"].get("stored", {})
                                        blocks = stored.get("blocks", [])
                                        block_hashes = [b.get("block_hash") for b in blocks]
                                    
                                    logger.info(
                                        f"[KV Event] Worker {worker_id}: type={event_type}, "
                                        f"block_hashes={block_hashes}"
                                    )
                                
                                logger.debug(
                                    f"Worker {worker_id} event {idx+1}/{len(kv_events)}: "
                                    f"type={event_type}"
                                )
                                
                                # Pass the ORIGINAL event format to RadixTree
                                self.radix_tree.apply_event(
                                    worker_id, event.encode("utf-8") if isinstance(event, str) else json.dumps(event_dict).encode("utf-8")
                                )
                                events_applied_count += 1
                                logger.debug(
                                    f"Successfully applied {event_type} event from worker {worker_id}"
                                )
                                
                            except Exception as apply_error:
                                events_failed_count += 1
                                # Events can arrive out of order or reference parent blocks
                                # that don't exist yet. Log at debug level and continue.
                                error_msg = str(apply_error)
                                if "Failed to find parent block" in error_msg:
                                    logger.debug(
                                        f"Skipping out-of-order event for worker {worker_id}: {error_msg}"
                                    )
                                else:
                                    logger.warning(
                                        f"Error applying KV event for worker {worker_id}: {error_msg}"
                                    )
                    except zmq.Again:
                        logger.debug(f"No KV events available for worker {worker_id} (timeout)")
                        pass
                    except Exception as e:
                        # Downgrade "Failed to find parent block" to DEBUG level
                        # This is expected in round-robin routing or with diverse prompts
                        error_msg = str(e)
                        if "Failed to find parent block" in error_msg:
                            logger.debug(
                                f"KV event for worker {worker_id}: {e} (expected in round-robin mode)"
                            )
                        else:
                            logger.warning(
                                f"Error receiving KV events for worker {worker_id}: {e}"
                            )

                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                logger.info(
                    f"Indexer update task cancelled for worker {worker_id} "
                    f"(received: {events_received_count}, applied: {events_applied_count}, "
                    f"failed: {events_failed_count})"
                )
                raise

        logger.info(f"Starting periodic_update_indexer for {self.num_workers} workers")
        worker_tasks: list[asyncio.Task] = []
        for worker_id in range(self.num_workers):
            worker_tasks.append(asyncio.create_task(update_tree(worker_id)))
        self.indexer_tasks = worker_tasks

        try:
            if worker_tasks:
                await asyncio.gather(*worker_tasks)
        except asyncio.CancelledError:
            logger.info("periodic_update_indexer cancelled")
            raise
        finally:
            logger.info("Cleaning up indexer tasks...")
            for task in worker_tasks:
                task.cancel()
            if worker_tasks:
                await asyncio.gather(*worker_tasks, return_exceptions=True)
            self.indexer_tasks = []
            logger.info("Indexer tasks cleanup completed")

    async def get_best_worker(self, local_hashes: list[int], num_tokens: int) -> int:
        try:
            if num_tokens <= 0:
                raise ValueError("num_tokens must be positive")

            # local_hashes can be empty
            logger.debug(
                f"[get_best_worker] num_tokens={num_tokens}, "
                f"num_hashes={len(local_hashes)}, "
                f"first_3_hashes={local_hashes[:3] if local_hashes else []}"
            )
            raw_scores = self.radix_tree.find_matches(local_hashes).scores
            logger.debug(
                f"[get_best_worker] raw_scores from RadixTree: {raw_scores}"
            )

            # RadixTree returns scores with tuple keys (worker_id, lora_id)
            # Convert to simple worker_id -> score mapping (sum across all lora_ids)
            worker_scores = {}
            for (worker_id, lora_id), score in raw_scores.items():
                worker_scores[worker_id] = worker_scores.get(worker_id, 0) + score
            
            logger.debug(
                f"[get_best_worker] worker_scores (aggregated): {worker_scores}"
            )

            overlap_scores = {
                worker_id: worker_scores.get(worker_id, 0) * self.block_size / num_tokens
                for worker_id in range(self.num_workers)
            }

            kv_usages = self.kv_usages[:]
            waitings = self.waitings[:]

            max_waiting = max(waitings) if waitings else 0
            waitings_normalized = [
                waiting / max_waiting if max_waiting else 0.0 for waiting in waitings
            ]

            logits = []
            for worker_id in range(self.num_workers):
                overlap = overlap_scores[worker_id]
                usage = kv_usages[worker_id]
                waiting = waitings_normalized[worker_id]
                logit = 2 * overlap - usage - waiting
                logits.append(logit)
                logger.info(
                    f"worker_id: {worker_id}, logit = 2 * {overlap:.3f} - {usage:.3f} - {waiting:.3f} = {logit:.3f}"
                )

            logits_array = np.array(logits)
            best_worker_id = int(
                np.random.choice(np.flatnonzero(logits_array == logits_array.max()))
            )

            # this is a predictive update which will be reset as new metrics are polled
            # but it is helpful for handling short bursts of highly concurrent requests
            # we omit updating the gpu_usage_perc as done in the rusty router for simplicity
            # as this requires obtaining num_gpu_blocks from the engines and can be intrusive
            # no need for async lock here, as the state is intended to be continuously overwritten
            self.waitings[best_worker_id] += 1

            return best_worker_id

        except Exception as e:
            logger.error(f"Error in get_best_worker: {e}")
            raise

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
        
        # Close KV event listeners
        for listener in self.kv_listeners:
            try:
                listener.close()
            except Exception as e:
                logger.error(f"Error closing KV listener: {e}")

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
    ):
        self.port = port
        self.block_size = block_size
        self.num_workers = num_workers
        self.base_kv_events_port = base_kv_events_port
        self.base_metrics_port = base_metrics_port
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

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    api = RouterAPI(
        block_size=args.block_size,
        num_workers=args.num_workers,
        base_kv_events_port=args.base_kv_events_port,
        base_metrics_port=args.base_metrics_port,
        port=args.port,
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
