#!/usr/bin/env python3
"""
Load balancer for distributing requests across multiple LLaDA/Nemotron server workers.

This load balancer sits in front of multiple worker servers (each running on a different GPU)
and distributes incoming requests using a round-robin strategy.

Usage:
    python llada_load_balancer.py --worker-ports 8001 8002 8003 8004 --port 8000
"""

import argparse
import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import List, Optional
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PendingRequest:
    """Represents a request waiting to be distributed to a worker."""
    request_id: str
    method: str
    path: str
    body: bytes
    headers: dict
    query_params: dict
    future: asyncio.Future
    timestamp: float


class BatchAccumulator:
    """
    Accumulates requests into batches before distributing them to workers.
    
    This ensures consistent batch sizes across GPUs by forming batches at the
    load balancer level before distribution, rather than per-worker batching.
    """
    
    def __init__(self, batch_size: int, max_wait_time: float, num_workers: int):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.num_workers = num_workers
        self.pending_requests: deque = deque()
        self.lock = asyncio.Lock()
        self.processing = False
        self._distribution_task = None
        self._worker_pool = None
        self._next_worker_idx = 0  # Track which worker to send the next batch to
    
    async def add_request(
        self, 
        method: str, 
        path: str, 
        body: bytes, 
        headers: dict, 
        query_params: dict
    ) -> Response:
        """Add a request to the batch accumulator and wait for its result."""
        request_id = str(uuid.uuid4())
        future = asyncio.Future()
        pending_request = PendingRequest(
            request_id=request_id,
            method=method,
            path=path,
            body=body,
            headers=headers,
            query_params=query_params,
            future=future,
            timestamp=time.time()
        )
        
        async with self.lock:
            self.pending_requests.append(pending_request)
            logger.debug(f"[BatchAccumulator] Added request {request_id} to batch queue. Queue size: {len(self.pending_requests)}")
        
        # Wait for the result
        try:
            return await future
        except Exception as e:
            logger.error(f"[BatchAccumulator] Request {request_id} failed: {e}")
            raise
    
    async def _batch_distribution_loop(self):
        """Continuously check if batches are ready and distribute them."""
        while True:
            try:
                await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
                
                if not self.pending_requests or self.processing:
                    continue
                
                # Check if we should distribute a batch
                should_distribute = False
                async with self.lock:
                    if len(self.pending_requests) >= self.batch_size:
                        should_distribute = True
                    elif self.pending_requests:
                        oldest_request_time = self.pending_requests[0].timestamp
                        if time.time() - oldest_request_time >= self.max_wait_time:
                            should_distribute = True
                
                if should_distribute:
                    if hasattr(self, '_worker_pool') and self._worker_pool is not None:
                        await self._distribute_batch(self._worker_pool)
                    else:
                        # Worker pool not set yet, skip distribution
                        logger.debug("[BatchAccumulator] Worker pool not ready, skipping batch distribution")
                    
            except Exception as e:
                logger.error(f"[BatchAccumulator] Error in batch distribution loop: {e}")
    
    async def _distribute_batch(self, worker_pool):
        """Distribute a batch of requests to workers round-robin."""
        if self.processing:
            return
        
        self.processing = True
        batch_requests = []
        
        try:
            # Extract requests from the queue
            async with self.lock:
                while self.pending_requests and len(batch_requests) < self.batch_size:
                    batch_requests.append(self.pending_requests.popleft())
            
            if not batch_requests:
                return
            
            # Send entire batch to a single worker for optimal batching
            # Round-robin across workers: this batch goes to one worker, next batch goes to next worker
            worker_idx = self._next_worker_idx
            self._next_worker_idx = (self._next_worker_idx + 1) % self.num_workers
            
            logger.info(f"[BatchAccumulator] Sending batch of {len(batch_requests)} requests to worker {worker_idx}")
            
            # Forward all requests to the selected worker concurrently
            # This ensures they arrive together and the worker's batch processor can handle them as a batch
            async def forward_single_request(pending_req: PendingRequest):
                try:
                    response = await worker_pool.forward_request_to_worker(
                        worker_idx=worker_idx,
                        method=pending_req.method,
                        path=pending_req.path,
                        body=pending_req.body,
                        headers=pending_req.headers,
                        query_params=pending_req.query_params
                    )
                    if not pending_req.future.done():
                        pending_req.future.set_result(response)
                except Exception as e:
                    logger.error(f"[BatchAccumulator] Failed to forward request {pending_req.request_id} to worker {worker_idx}: {e}")
                    if not pending_req.future.done():
                        pending_req.future.set_exception(e)
            
            # Forward all requests concurrently to the same worker
            # Workers have their own batch processors, so sending them together increases
            # the chance they'll be processed in the same batch
            await asyncio.gather(*[
                forward_single_request(req) for req in batch_requests
            ])
            
        except Exception as e:
            logger.error(f"[BatchAccumulator] Batch distribution failed: {e}")
            # Set exception for all pending requests
            for pending_req in batch_requests:
                if not pending_req.future.done():
                    pending_req.future.set_exception(e)
        finally:
            self.processing = False
    
    def get_pending_count(self) -> int:
        """Get the number of pending requests."""
        return len(self.pending_requests)
    
    def set_worker_pool(self, worker_pool):
        """Set the worker pool for batch distribution."""
        self._worker_pool = worker_pool
    
    def start(self):
        """Start the batch distribution loop. Must be called when event loop is running."""
        if self._distribution_task is None:
            self._distribution_task = asyncio.create_task(self._batch_distribution_loop())
    
    async def stop(self):
        """Stop the batch distribution loop."""
        if self._distribution_task is not None:
            self._distribution_task.cancel()
            try:
                await self._distribution_task
            except asyncio.CancelledError:
                pass
            self._distribution_task = None


class WorkerPool:
    """Manages a pool of worker servers and distributes requests among them."""
    
    def __init__(self, worker_urls: List[str], health_check_interval: float = 30.0, request_timeout: float = 600.0):
        self.worker_urls = worker_urls
        self.current_index = 0
        self.healthy_workers = set(range(len(worker_urls)))
        self.health_check_interval = health_check_interval
        self.request_timeout = request_timeout
        self.lock = asyncio.Lock()
        
        # Configure httpx client with proper timeouts and connection limits
        # This is crucial for long-running evaluations with many concurrent requests
        timeout_config = httpx.Timeout(
            connect=10.0,              # 10s to establish connection
            read=request_timeout,       # Long read timeout for diffusion generation
            write=30.0,                # 30s to send request
            pool=None                  # No timeout waiting for connection from pool
        )
        limits = httpx.Limits(
            max_connections=1000,      # Support many concurrent requests
            max_keepalive_connections=100,  # Keep more connections alive
            keepalive_expiry=300.0     # Keep connections alive for 5 minutes
        )
        self.client = httpx.AsyncClient(timeout=timeout_config, limits=limits)
        
        # Stats
        self.total_requests = 0
        self.worker_request_counts = [0] * len(worker_urls)
        self.worker_error_counts = [0] * len(worker_urls)
        
        logger.info(f"Initialized worker pool with {len(worker_urls)} workers:")
        for i, url in enumerate(worker_urls):
            logger.info(f"  Worker {i}: {url}")
        logger.info(f"HTTP client configuration:")
        logger.info(f"  Request timeout: {request_timeout}s (read)")
        logger.info(f"  Connection limits: {limits.max_connections} max, {limits.max_keepalive_connections} keepalive")
        logger.info(f"  Keepalive expiry: {limits.keepalive_expiry}s")
    
    async def start_health_checks(self):
        """Start periodic health checks for all workers."""
        while True:
            await asyncio.sleep(self.health_check_interval)
            await self._check_all_workers()
    
    async def _check_all_workers(self):
        """Check health of all workers."""
        for i, url in enumerate(self.worker_urls):
            try:
                response = await self.client.get(f"{url}/health", timeout=5.0)
                if response.status_code == 200:
                    async with self.lock:
                        self.healthy_workers.add(i)
                else:
                    logger.warning(f"Worker {i} ({url}) unhealthy: status {response.status_code}")
                    async with self.lock:
                        self.healthy_workers.discard(i)
            except Exception as e:
                logger.warning(f"Worker {i} ({url}) health check failed: {e}")
                async with self.lock:
                    self.healthy_workers.discard(i)
    
    async def get_next_worker(self) -> Optional[tuple[int, str]]:
        """Get the next healthy worker using round-robin."""
        async with self.lock:
            if not self.healthy_workers:
                return None
            
            # Round-robin through healthy workers
            attempts = 0
            while attempts < len(self.worker_urls):
                worker_idx = self.current_index % len(self.worker_urls)
                self.current_index += 1
                
                if worker_idx in self.healthy_workers:
                    return worker_idx, self.worker_urls[worker_idx]
                
                attempts += 1
            
            return None
    
    async def forward_request(self, method: str, path: str, request: Request) -> Response:
        """Forward a request to the next available worker."""
        worker_info = await self.get_next_worker()
        
        if worker_info is None:
            logger.error("No healthy workers available")
            raise HTTPException(status_code=503, detail="No healthy workers available")
        
        worker_idx, worker_url = worker_info
        body = await request.body()
        headers = dict(request.headers)
        headers.pop('host', None)
        headers.pop('content-length', None)
        
        return await self.forward_request_to_worker(
            worker_idx=worker_idx,
            method=method,
            path=path,
            body=body,
            headers=headers,
            query_params=dict(request.query_params)
        )
    
    async def forward_request_to_worker(
        self,
        worker_idx: int,
        method: str,
        path: str,
        body: bytes,
        headers: dict,
        query_params: dict
    ) -> Response:
        """Forward a request to a specific worker."""
        if worker_idx not in self.healthy_workers:
            # Try to find another healthy worker
            worker_info = await self.get_next_worker()
            if worker_info is None:
                raise HTTPException(status_code=503, detail="No healthy workers available")
            worker_idx, worker_url = worker_info
        else:
            worker_url = self.worker_urls[worker_idx]
        
        target_url = f"{worker_url}{path}"
        
        try:
            # Forward the request
            logger.debug(f"Forwarding {method} {path} to worker {worker_idx} ({worker_url})")
            
            response = await self.client.request(
                method=method,
                url=target_url,
                content=body,
                headers=headers,
                params=query_params
            )
            
            # Update stats
            async with self.lock:
                self.total_requests += 1
                self.worker_request_counts[worker_idx] += 1
            
            # Return the response
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.headers.get('content-type')
            )
            
        except Exception as e:
            logger.error(f"Error forwarding request to worker {worker_idx}: {e}")
            
            # Update error stats
            async with self.lock:
                self.worker_error_counts[worker_idx] += 1
                # Mark worker as unhealthy if it's failing
                self.healthy_workers.discard(worker_idx)
            
            raise HTTPException(status_code=502, detail=f"Worker error: {e}")
    
    async def get_stats(self) -> dict:
        """Get load balancer statistics."""
        async with self.lock:
            return {
                "total_workers": len(self.worker_urls),
                "healthy_workers": len(self.healthy_workers),
                "total_requests": self.total_requests,
                "workers": [
                    {
                        "index": i,
                        "url": url,
                        "healthy": i in self.healthy_workers,
                        "requests_served": self.worker_request_counts[i],
                        "errors": self.worker_error_counts[i]
                    }
                    for i, url in enumerate(self.worker_urls)
                ]
            }
    
    async def cleanup(self):
        """Cleanup resources."""
        await self.client.aclose()


# Global worker pool and batch accumulator
worker_pool: Optional[WorkerPool] = None
batch_accumulator: Optional[BatchAccumulator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager."""
    global worker_pool, batch_accumulator
    
    logger.info("Starting load balancer...")
    
    # Wait for workers to start up and load models
    # Increased from 5s to 20s to allow for:
    # - Model weight loading from HuggingFace cache
    # - CUDA context initialization
    # - Uvicorn server startup
    logger.info("Waiting 20 seconds for workers to initialize...")
    await asyncio.sleep(20)
    
    # Initial health check
    logger.info("Performing initial health check...")
    await worker_pool._check_all_workers()
    
    healthy_count = len(worker_pool.healthy_workers)
    total_count = len(worker_pool.worker_urls)
    
    if healthy_count == 0:
        logger.error("No healthy workers found! Server may not work properly.")
    else:
        logger.info(f"Found {healthy_count}/{total_count} healthy workers")
    
    # Initialize batch accumulator if batching is enabled
    if batch_accumulator is not None:
        batch_accumulator.set_worker_pool(worker_pool)
        batch_accumulator.start()  # Start the batch distribution loop now that event loop is running
        logger.info(f"Batch accumulator initialized: batch_size={batch_accumulator.batch_size}, max_wait_time={batch_accumulator.max_wait_time}s")
    
    # Start periodic health checks
    health_check_task = asyncio.create_task(worker_pool.start_health_checks())
    
    yield
    
    # Cleanup
    health_check_task.cancel()
    if batch_accumulator is not None:
        await batch_accumulator.stop()
    await worker_pool.cleanup()
    logger.info("Load balancer shut down")


# Create FastAPI app
app = FastAPI(
    title="LLaDA/Nemotron Load Balancer",
    description="Load balancer for distributing requests across multiple LLaDA/Nemotron workers",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint for the load balancer."""
    stats = await worker_pool.get_stats()
    
    return {
        "status": "healthy" if stats["healthy_workers"] > 0 else "unhealthy",
        "load_balancer": "active",
        "workers": stats
    }


@app.get("/stats")
async def get_stats():
    """Get detailed load balancer statistics."""
    stats = await worker_pool.get_stats()
    
    # Add batch accumulator stats if enabled
    if batch_accumulator is not None:
        stats["batch_accumulator"] = {
            "enabled": True,
            "batch_size": batch_accumulator.batch_size,
            "max_wait_time": batch_accumulator.max_wait_time,
            "pending_requests": batch_accumulator.get_pending_count()
        }
    else:
        stats["batch_accumulator"] = {"enabled": False}
    
    return stats


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy_request(path: str, request: Request):
    """Proxy all other requests to workers."""
    global batch_accumulator
    
    # If batch accumulator is enabled, use it for batching before distribution
    if batch_accumulator is not None and request.method == "POST" and path.startswith("v1/chat/completions"):
        # For chat completion requests, use batch accumulator
        body = await request.body()
        headers = dict(request.headers)
        headers.pop('host', None)
        headers.pop('content-length', None)
        
        return await batch_accumulator.add_request(
            method=request.method,
            path=f"/{path}",
            body=body,
            headers=headers,
            query_params=dict(request.query_params)
        )
    else:
        # For other requests, forward directly (no batching)
        return await worker_pool.forward_request(request.method, f"/{path}", request)


def main():
    global worker_pool, batch_accumulator
    
    parser = argparse.ArgumentParser(
        description="Load Balancer for LLaDA/Nemotron API Servers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load balance across 4 workers on ports 8001-8004
  python llada_load_balancer.py --worker-ports 8001 8002 8003 8004 --port 8000
  
  # Load balance with pre-load-balancer batching (consistent batch sizes across GPUs)
  python llada_load_balancer.py --worker-ports 8001 8002 8003 8004 --port 8000 --batch-size 8 --max-wait-time 0.1
  
  # Load balance with custom host
  python llada_load_balancer.py --worker-ports 8001 8002 --port 8000 --worker-host localhost
        """
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind load balancer to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind load balancer to")
    parser.add_argument("--worker-host", default="localhost", help="Host where workers are running")
    parser.add_argument("--worker-ports", nargs="+", type=int, required=True,
                       help="Ports of worker servers (e.g., 8001 8002 8003 8004)")
    parser.add_argument("--health-check-interval", type=float, default=30.0,
                       help="Interval between health checks in seconds")
    parser.add_argument("--request-timeout", type=float, default=600.0,
                       help="Request timeout in seconds (default: 600, increase for long evaluations)")
    parser.add_argument("--timeout-keep-alive", type=int, default=300,
                       help="HTTP keep-alive timeout in seconds (default: 300, increase for long evaluations)")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Enable pre-load-balancer batching with this batch size (ensures consistent batch sizes across GPUs)")
    parser.add_argument("--max-wait-time", type=float, default=0.1,
                       help="Maximum time to wait for batch to fill in seconds (default: 0.1, only used with --batch-size)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("🔍 VERBOSE MODE ENABLED")
    
    # Create worker URLs
    worker_urls = [f"http://{args.worker_host}:{port}" for port in args.worker_ports]
    
    # Initialize worker pool
    worker_pool = WorkerPool(
        worker_urls, 
        health_check_interval=args.health_check_interval,
        request_timeout=args.request_timeout
    )
    
    # Initialize batch accumulator if batch size is specified
    if args.batch_size is not None:
        batch_accumulator = BatchAccumulator(
            batch_size=args.batch_size,
            max_wait_time=args.max_wait_time,
            num_workers=len(worker_urls)
        )
        logger.info(f"Pre-load-balancer batching ENABLED: batch_size={args.batch_size}, max_wait_time={args.max_wait_time}s")
        logger.info(f"This ensures consistent batch sizes across {len(worker_urls)} GPUs")
    else:
        batch_accumulator = None
        logger.info("Pre-load-balancer batching DISABLED (using direct round-robin distribution)")
    
    logger.info(f"Starting load balancer on {args.host}:{args.port}")
    logger.info(f"Distributing across {len(worker_urls)} workers")
    logger.info(f"Timeout settings: request={args.request_timeout}s, keep-alive={args.timeout_keep_alive}s")
    
    # Start the server with proper timeout configuration
    uvicorn.run(
        app, 
        host=args.host, 
        port=args.port,
        timeout_keep_alive=args.timeout_keep_alive,
        timeout_graceful_shutdown=30,
    )


if __name__ == "__main__":
    main()

