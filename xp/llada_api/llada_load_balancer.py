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


class WorkerPool:
    """Manages a pool of worker servers and distributes requests among them."""
    
    def __init__(self, worker_urls: List[str], health_check_interval: float = 30.0, request_timeout: float = 600.0):
        self.worker_urls = worker_urls
        self.current_index = 0
        self.healthy_workers = set(range(len(worker_urls)))
        self.health_check_interval = health_check_interval
        self.request_timeout = request_timeout
        self.lock = asyncio.Lock()
        self.client = httpx.AsyncClient(timeout=request_timeout)
        
        # Stats
        self.total_requests = 0
        self.worker_request_counts = [0] * len(worker_urls)
        self.worker_error_counts = [0] * len(worker_urls)
        
        logger.info(f"Initialized worker pool with {len(worker_urls)} workers:")
        for i, url in enumerate(worker_urls):
            logger.info(f"  Worker {i}: {url}")
        logger.info(f"Request timeout: {request_timeout}s")
    
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
        target_url = f"{worker_url}{path}"
        
        # Get request body
        body = await request.body()
        
        # Forward headers (exclude host-related headers)
        headers = dict(request.headers)
        headers.pop('host', None)
        headers.pop('content-length', None)
        
        try:
            # Forward the request
            logger.debug(f"Forwarding {method} {path} to worker {worker_idx} ({worker_url})")
            
            response = await self.client.request(
                method=method,
                url=target_url,
                content=body,
                headers=headers,
                params=dict(request.query_params)
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


# Global worker pool
worker_pool: Optional[WorkerPool] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager."""
    global worker_pool
    
    logger.info("Starting load balancer...")
    
    # Wait a bit for workers to start up
    logger.info("Waiting 5 seconds for workers to initialize...")
    await asyncio.sleep(5)
    
    # Initial health check
    logger.info("Performing initial health check...")
    await worker_pool._check_all_workers()
    
    healthy_count = len(worker_pool.healthy_workers)
    total_count = len(worker_pool.worker_urls)
    
    if healthy_count == 0:
        logger.error("No healthy workers found! Server may not work properly.")
    else:
        logger.info(f"Found {healthy_count}/{total_count} healthy workers")
    
    # Start periodic health checks
    health_check_task = asyncio.create_task(worker_pool.start_health_checks())
    
    yield
    
    # Cleanup
    health_check_task.cancel()
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
    return await worker_pool.get_stats()


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy_request(path: str, request: Request):
    """Proxy all other requests to workers."""
    return await worker_pool.forward_request(request.method, f"/{path}", request)


def main():
    global worker_pool
    
    parser = argparse.ArgumentParser(
        description="Load Balancer for LLaDA/Nemotron API Servers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load balance across 4 workers on ports 8001-8004
  python llada_load_balancer.py --worker-ports 8001 8002 8003 8004 --port 8000
  
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

