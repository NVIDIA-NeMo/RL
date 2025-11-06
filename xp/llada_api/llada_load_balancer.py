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
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
from dataclasses import dataclass
from collections import deque
import threading

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_worker_error(worker_idx: int, worker_url: str, error_type: str, details: dict, context: str = ""):
    """
    Centralized worker error logging - only logs informative errors, not timeouts.
    
    Args:
        worker_idx: Index of the failing worker
        worker_url: URL of the failing worker  
        error_type: Type of error (HTTP_ERROR, EXCEPTION, TIMEOUT, etc.)
        details: Dictionary with error details  
        context: Additional context (batch info, request info, etc.)
    """
    # Skip logging for common health check timeouts - they're not informative
    if error_type == "HEALTH_CHECK_EXCEPTION" and "ReadTimeout" in str(details.get("error_type", "")):
        return
    if error_type == "HEALTH_CHECK_EXCEPTION" and "timeout" in str(details.get("error", "")).lower():
        return
    
    # Log concise, informative worker errors only
    if error_type == "HTTP_ERROR":
        status_code = details.get("status_code", "unknown")
        response_body = details.get("response_body", "")
        # Only show first few lines of response body
        body_preview = response_body.split('\n')[0][:200] if response_body else ""
        logger.error(f"🚨 Worker #{worker_idx} ({worker_url}) HTTP {status_code}: {body_preview}")
        
        # Show full response body only if it contains useful error info (not just HTML)
        if response_body and len(response_body) < 1000 and not response_body.startswith('<!DOCTYPE'):
            logger.error(f"   Response: {response_body}")
            
    elif error_type == "HEALTH_CHECK_FAILED":
        status_code = details.get("status_code", "unknown") 
        response_body = details.get("response_body", "")
        body_preview = response_body.split('\n')[0][:200] if response_body else ""
        logger.error(f"🚨 Worker #{worker_idx} ({worker_url}) health check failed (HTTP {status_code}): {body_preview}")
        
    elif error_type == "EXCEPTION":
        error = details.get("error", "unknown error")
        error_type_name = details.get("error_type", "")
        logger.error(f"🚨 Worker #{worker_idx} ({worker_url}) {error_type_name}: {error}")
        
    else:
        # Generic error logging for other types
        error = details.get("error", str(details))
        logger.error(f"🚨 Worker #{worker_idx} ({worker_url}) {error_type}: {error}")


@dataclass
class BatchRequest:
    """Represents a single request waiting to be batched."""
    request_id: str
    request_body: bytes
    headers: Dict[str, str]
    query_params: Dict[str, str]
    future: asyncio.Future
    timestamp: float


class CentralizedBatchProcessor:
    """Handles centralized batching at the load balancer level with parallel multi-worker dispatch."""
    
    def __init__(self, max_batch_size: int = 8, max_wait_time: float = 0.02, worker_pool=None):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time  # Reduced default timeout for better responsiveness
        self.worker_pool = worker_pool
        self.pending_requests: deque = deque()
        self.lock = asyncio.Lock()
        self.worker_round_robin_index = 0  # For distributing batches across workers
        
        # Statistics
        self.total_batches_processed = 0
        self.total_requests_batched = 0
        self.batch_size_histogram = {}  # Track batch size distribution
        
        logger.info(f"Parallel multi-worker batching enabled: batch_size={max_batch_size}, wait_time={max_wait_time}s")
        
        # Task will be started later when event loop is running
        self._batch_task = None
    
    async def start_batch_processing(self):
        """Start the batch processing loop (call this when event loop is running)."""
        if self._batch_task is None:
            self._batch_task = asyncio.create_task(self._batch_processing_loop())
            logger.info("Started centralized batch processing loop")
    
    async def stop_batch_processing(self):
        """Stop the batch processing loop."""
        if self._batch_task:
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass
            self._batch_task = None
            logger.info("Stopped centralized batch processing loop")
    
    async def add_request(self, request_body: bytes, headers: Dict[str, str], query_params: Dict[str, str]) -> Dict[str, Any]:
        """Add a chat completion request to the batch and wait for its result."""
        request_id = str(uuid.uuid4())
        future = asyncio.Future()
        batch_request = BatchRequest(
            request_id=request_id,
            request_body=request_body,
            headers=headers,
            query_params=query_params,
            future=future,
            timestamp=time.time()
        )
        
        async with self.lock:
            self.pending_requests.append(batch_request)
            logger.debug(f"Added request {request_id} to centralized batch queue. Queue size: {len(self.pending_requests)}")
        
        # Wait for the result
        try:
            return await future
        except Exception as e:
            logger.error(f"Request {request_id} failed: {e}")
            raise
    
    async def _batch_processing_loop(self):
        """Continuously process batches of requests with parallel multi-worker dispatch."""
        while True:
            try:
                await asyncio.sleep(0.005)  # Reduced delay for better responsiveness
                
                if not self.pending_requests:
                    continue
                
                # Check if we should process a batch - allow multiple batches in parallel
                should_process = False
                async with self.lock:
                    if len(self.pending_requests) >= self.max_batch_size:
                        should_process = True
                        logger.debug(f"Triggering batch: reached max_batch_size ({self.max_batch_size})")
                    elif self.pending_requests:
                        oldest_request_time = self.pending_requests[0].timestamp
                        if time.time() - oldest_request_time >= self.max_wait_time:
                            should_process = True
                            logger.debug(f"Triggering batch: exceeded max_wait_time ({self.max_wait_time}s)")

                if should_process:
                    # Process batch in parallel - don't block other batch formation
                    asyncio.create_task(self._process_batch())
                    
            except Exception as e:
                logger.error(f"Error in centralized batch processing loop: {e}")
    
    async def _process_batch(self):
        """Process a batch of requests by dispatching to workers in parallel across multiple workers."""
        batch_requests = []
        
        try:
            # Extract requests from the queue (keep lock time minimal)  
            async with self.lock:
                while self.pending_requests and len(batch_requests) < self.max_batch_size:
                    batch_requests.append(self.pending_requests.popleft())
            
            if not batch_requests:
                return
            
            batch_size = len(batch_requests)
            batch_start_time = time.time()
            
            # Get next worker using round-robin distribution
            worker_info = await self._get_next_worker_round_robin()
            if worker_info is None:
                # No workers available - return errors
                error = HTTPException(status_code=503, detail="No healthy workers available")
                for batch_request in batch_requests:
                    batch_request.future.set_exception(error)
                return
            
            worker_idx, worker_url = worker_info
            
            logger.info(f"🚀 Dispatching batch of {batch_size} requests → Worker {worker_idx} (parallel processing)")
            
            # Safety check: Warn if batch size is unexpectedly small
            if batch_size < self.max_batch_size:
                logger.warning(f"⚠️ Sending partial batch: {batch_size}/{self.max_batch_size} requests")
            
            # Send all requests in parallel to the selected worker
            # Don't wait for completion - fire and forget to enable parallel batching
            tasks = []
            for batch_request in batch_requests:
                task = asyncio.create_task(
                    self._send_single_request_with_future(worker_idx, worker_url, batch_request)
                )
                tasks.append(task)
            
            # Update statistics immediately (don't wait for completion)
            self.total_batches_processed += 1
            self.total_requests_batched += batch_size
            self.batch_size_histogram[batch_size] = self.batch_size_histogram.get(batch_size, 0) + 1
            
            # Start all requests in parallel but don't block - this enables the next batch to form immediately
            asyncio.create_task(self._wait_for_batch_completion(tasks, batch_start_time, batch_size, worker_idx))
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"❌ ERROR: Batch dispatch failed:")
            logger.error(f"   Batch size: {len(batch_requests)}")
            logger.error(f"   Error: {e}")
            logger.error(f"   Full traceback:")
            for line in error_details.split('\n'):
                if line.strip():
                    logger.error(f"     {line}")
            
            # Set exception for all pending requests that haven't been handled
            for batch_request in batch_requests:
                if not batch_request.future.done():
                    batch_request.future.set_exception(HTTPException(status_code=500, detail=f"Batch processing failed: {e}"))
    
    async def _get_next_worker_round_robin(self) -> Optional[tuple[int, str]]:
        """Get next worker using round-robin distribution across healthy workers."""
        if not self.worker_pool or not self.worker_pool.healthy_workers:
            return None
        
        healthy_workers = list(self.worker_pool.healthy_workers)
        if not healthy_workers:
            return None
        
        # Round-robin through healthy workers
        worker_idx = healthy_workers[self.worker_round_robin_index % len(healthy_workers)]
        self.worker_round_robin_index += 1
        
        return worker_idx, self.worker_pool.worker_urls[worker_idx]
    
    async def _wait_for_batch_completion(self, tasks: List[asyncio.Task], batch_start_time: float, batch_size: int, worker_idx: int):
        """Wait for batch completion and log results without blocking batch formation."""
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
            batch_time = time.time() - batch_start_time
            throughput = batch_size / batch_time if batch_time > 0 else 0
            logger.info(f"✅ Batch of {batch_size} completed in {batch_time:.3f}s → Worker {worker_idx} ({throughput:.1f} req/s)")
        except Exception as e:
            logger.error(f"❌ Batch completion monitoring failed: {e}")
    
    async def _send_single_request_with_future(self, worker_idx: int, worker_url: str, batch_request: BatchRequest):
        """Send a single request to a worker and handle the future result."""
        try:
            # Track that we're sending a request to this worker
            self.worker_pool.worker_last_activity[worker_idx] = time.time()
            
            result = await self._send_single_request(worker_idx, worker_url, batch_request)
            batch_request.future.set_result(result)
        except Exception as e:
            batch_request.future.set_exception(e)
    
    async def _send_single_request(self, worker_idx: int, worker_url: str, batch_request: BatchRequest) -> Dict[str, Any]:
        """Send a single request to a worker."""
        target_url = f"{worker_url}/v1/chat/completions"
        
        try:
            response = await self.worker_pool.client.request(
                method="POST",
                url=target_url,
                content=batch_request.request_body,
                headers=batch_request.headers,
                params=batch_request.query_params
            )
            
            # Check for HTTP errors (4xx, 5xx status codes)
            if response.status_code >= 400:
                error_body = ""
                try:
                    error_body = response.text
                except:
                    error_body = f"<Could not decode response body>"
                
                # Use centralized error logging
                log_worker_error(
                    worker_idx=worker_idx,
                    worker_url=worker_url,
                    error_type="HTTP_ERROR",
                    details={
                        "status_code": response.status_code,
                        "url": target_url,
                        "request_id": batch_request.request_id,
                        "response_headers": dict(response.headers),
                        "response_body": error_body
                    },
                    context=f"Batch request processing"
                )
                
                # Update error stats
                async with self.worker_pool.lock:
                    self.worker_pool.worker_error_counts[worker_idx] += 1
                    # Only mark worker unhealthy after multiple consecutive server errors
                    # Single 500s during processing are expected and shouldn't remove workers
                    if response.status_code >= 500:
                        # Track consecutive errors instead of immediate removal
                        if not hasattr(self.worker_pool, 'consecutive_errors'):
                            self.worker_pool.consecutive_errors = [0] * len(self.worker_pool.worker_urls)
                        
                        self.worker_pool.consecutive_errors[worker_idx] += 1
                        
                        # Only mark unhealthy after 3 consecutive errors
                        if self.worker_pool.consecutive_errors[worker_idx] >= 3:
                            self.worker_pool.healthy_workers.discard(worker_idx)
                            logger.warning(f"⚠️ Worker {worker_idx} marked UNHEALTHY after {self.worker_pool.consecutive_errors[worker_idx]} consecutive errors")
                        else:
                            logger.debug(f"Worker {worker_idx} error count: {self.worker_pool.consecutive_errors[worker_idx]}/3")
                    else:
                        # Reset consecutive error count on successful response
                        if hasattr(self.worker_pool, 'consecutive_errors'):
                            self.worker_pool.consecutive_errors[worker_idx] = 0
                
                raise HTTPException(
                    status_code=response.status_code, 
                    detail=f"Worker {worker_idx} returned {response.status_code}: {error_body[:200]}{'...' if len(error_body) > 200 else ''}"
                )
            
            # Update worker stats for successful requests
            async with self.worker_pool.lock:
                self.worker_pool.worker_request_counts[worker_idx] += 1
                # Reset consecutive error count on successful response
                if hasattr(self.worker_pool, 'consecutive_errors'):
                    self.worker_pool.consecutive_errors[worker_idx] = 0
                # Track worker activity to avoid false unhealthy marking
                self.worker_pool.worker_last_activity[worker_idx] = time.time()
            
            return response.json()
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            
            # Use centralized error logging
            log_worker_error(
                worker_idx=worker_idx,
                worker_url=worker_url,
                error_type="EXCEPTION",
                details={
                    "request_id": batch_request.request_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": error_details
                },
                context=f"Batch request processing failed"
            )
            
            async with self.worker_pool.lock:
                self.worker_pool.worker_error_counts[worker_idx] += 1
                self.worker_pool.healthy_workers.discard(worker_idx)
            raise HTTPException(status_code=502, detail=f"Worker {worker_idx} error: {e}")
    
    
    def get_stats(self) -> Dict[str, Any]:
        """Get centralized batching statistics."""
        return {
            "enabled": True,
            "max_batch_size": self.max_batch_size,
            "max_wait_time": self.max_wait_time,
            "pending_requests": len(self.pending_requests),
            "total_batches_processed": self.total_batches_processed,
            "total_requests_batched": self.total_requests_batched,
            "avg_batch_size": self.total_requests_batched / max(1, self.total_batches_processed),
            "batch_size_histogram": self.batch_size_histogram
        }


class WorkerPool:
    """Manages a pool of worker servers and distributes requests among them."""
    
    def __init__(self, worker_urls: List[str], health_check_interval: float = 60.0, request_timeout: float = 600.0, 
                 enable_centralized_batching: bool = True, batch_size: int = 8, batch_wait_time: float = 0.02):
        self.worker_urls = worker_urls
        self.current_index = 0
        self.healthy_workers = set(range(len(worker_urls)))
        self.health_check_interval = health_check_interval  # Increased from 30s to 60s
        self.request_timeout = request_timeout
        self.lock = asyncio.Lock()
        
        # Track when workers are busy to avoid marking them unhealthy
        self.worker_last_activity = [time.time()] * len(worker_urls)
        self.worker_busy_count = [0] * len(worker_urls)
        
        # Centralized batching
        self.enable_centralized_batching = enable_centralized_batching
        self.centralized_batch_processor = None
        if enable_centralized_batching:
            self.centralized_batch_processor = CentralizedBatchProcessor(
                max_batch_size=batch_size,
                max_wait_time=batch_wait_time,
                worker_pool=self
            )
        
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
        last_status_report = 0
        status_report_interval = 300  # Report status every 5 minutes
        
        while True:
            await asyncio.sleep(self.health_check_interval)
            await self._check_all_workers()
            
            # Periodic status report
            current_time = time.time()
            if current_time - last_status_report >= status_report_interval:
                await self._log_worker_status_summary()
                last_status_report = current_time
    
    async def _log_worker_status_summary(self):
        """Log a summary of worker health status."""
        async with self.lock:
            healthy_count = len(self.healthy_workers)
            total_count = len(self.worker_urls)
            
            logger.info(f"📊 Worker Status: {healthy_count}/{total_count} healthy")
            
            # Only log details for unhealthy workers or if there are errors
            has_issues = False
            for i, url in enumerate(self.worker_urls):
                if i not in self.healthy_workers:
                    errors = self.worker_error_counts[i]
                    logger.info(f"  Worker {i} ({url}): UNHEALTHY (errors: {errors})")
                    has_issues = True
                elif self.worker_error_counts[i] > 0:
                    requests = self.worker_request_counts[i]
                    errors = self.worker_error_counts[i]
                    logger.info(f"  Worker {i} ({url}): HEALTHY but has errors (reqs: {requests}, errors: {errors})")
                    has_issues = True
            
            if healthy_count == 0:
                logger.error("🚨 WARNING: No healthy workers! All requests will fail.")
            elif not has_issues:
                total_requests = sum(self.worker_request_counts)
                logger.info(f"  All workers healthy, {total_requests} total requests processed")
    
    async def _check_all_workers(self):
        """Check health of all workers with consideration for busy workers."""
        for i, url in enumerate(self.worker_urls):
            was_healthy = i in self.healthy_workers
            current_time = time.time()
            last_activity = self.worker_last_activity[i]
            time_since_activity = current_time - last_activity
            
            try:
                # Use longer timeout for health checks (15s instead of 5s) to account for busy workers
                health_timeout = 15.0 if time_since_activity < 120 else 5.0  # Longer timeout if recently active
                response = await self.client.get(f"{url}/health", timeout=health_timeout)
                
                if response.status_code == 200:
                    async with self.lock:
                        self.healthy_workers.add(i)
                    # Log when a worker becomes healthy again
                    if not was_healthy:
                        logger.info(f"✅ Worker {i} ({url}) is now HEALTHY")
                else:
                    # Get response body for more details
                    error_body = ""
                    try:
                        error_body = response.text
                    except:
                        error_body = "<Could not decode response body>"
                    
                    # Be more lenient with health check failures if worker was recently active
                    if time_since_activity < 60:  # If active within last 60 seconds
                        logger.debug(f"Worker {i} health check failed but was recently active ({time_since_activity:.1f}s ago) - keeping healthy")
                        async with self.lock:
                            self.healthy_workers.add(i)  # Keep it healthy
                    else:
                        # Only log informative health check failures (not just status codes)
                        if response.status_code not in [503, 500] or "model" in error_body.lower() or "error" in error_body.lower():
                            log_worker_error(
                                worker_idx=i,
                                worker_url=url,
                                error_type="HEALTH_CHECK_FAILED",
                                details={
                                    "status_code": response.status_code,
                                    "response_body": error_body,
                                    "time_since_activity": time_since_activity
                                }
                            )
                        
                        async with self.lock:
                            self.healthy_workers.discard(i)
                            
            except Exception as e:
                # Be more lenient with timeouts if worker was recently active
                is_timeout = "timeout" in str(e).lower() or "ReadTimeout" in type(e).__name__
                
                if is_timeout and time_since_activity < 90:  # If active within last 90 seconds
                    logger.debug(f"Worker {i} health check timeout but was recently active ({time_since_activity:.1f}s ago) - keeping healthy")
                    async with self.lock:
                        self.healthy_workers.add(i)  # Keep it healthy
                else:
                    # Only log non-timeout exceptions (timeouts are expected during high load)
                    if not is_timeout:
                        log_worker_error(
                            worker_idx=i,
                            worker_url=url,
                            error_type="HEALTH_CHECK_EXCEPTION",
                            details={
                                "error": str(e),
                                "error_type": type(e).__name__,
                                "time_since_activity": time_since_activity
                            }
                        )
                    
                    async with self.lock:
                        self.healthy_workers.discard(i)
                        
                        # If all workers become unhealthy, try to recover by assuming they're just busy
                        if len(self.healthy_workers) == 0:
                            logger.warning("⚠️ All workers marked unhealthy - likely overloaded. Attempting recovery...")
                            for j in range(len(self.worker_urls)):
                                if current_time - self.worker_last_activity[j] < 300:  # Active within 5 minutes
                                    self.healthy_workers.add(j)
                                    logger.info(f"🔄 Restored Worker {j} to healthy (was active {current_time - self.worker_last_activity[j]:.1f}s ago)")
    
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
    
    async def get_least_loaded_worker(self) -> Optional[tuple[int, str]]:
        """Get the least loaded healthy worker for batch processing."""
        async with self.lock:
            if not self.healthy_workers:
                return None
            
            # Find the worker with the least requests processed
            min_requests = float('inf')
            best_worker_idx = None
            
            for worker_idx in self.healthy_workers:
                request_count = self.worker_request_counts[worker_idx]
                if request_count < min_requests:
                    min_requests = request_count
                    best_worker_idx = worker_idx
            
            if best_worker_idx is not None:
                return best_worker_idx, self.worker_urls[best_worker_idx]
            
            return None
    
    async def forward_request(self, method: str, path: str, request: Request) -> Response:
        """Forward a request to the next available worker, using centralized batching for chat completions."""
        
        # Use centralized batching for chat completion requests
        if (self.enable_centralized_batching and 
            self.centralized_batch_processor and 
            method.upper() == "POST" and 
            path in ["/v1/chat/completions", "v1/chat/completions"]):
            
            logger.debug(f"Using centralized batching for {method} {path}")
            
            # Get request body and headers
            body = await request.body()
            headers = dict(request.headers)
            headers.pop('host', None)
            headers.pop('content-length', None)
            query_params = dict(request.query_params)
            
            try:
                # Use centralized batching
                result = await self.centralized_batch_processor.add_request(
                    request_body=body,
                    headers=headers,
                    query_params=query_params
                )
                
                # Return the result as a JSON response
                return JSONResponse(content=result)
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Centralized batching failed: {e}")
                # Fallback to regular forwarding
                pass
        
        # Regular forwarding for non-chat completion requests or if batching is disabled/failed
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
            
            # Check for HTTP errors (4xx, 5xx status codes)
            if response.status_code >= 400:
                error_body = ""
                try:
                    error_body = response.text
                except:
                    error_body = f"<Could not decode response body>"
                
                # Use centralized error logging
                log_worker_error(
                    worker_idx=worker_idx,
                    worker_url=worker_url,
                    error_type="HTTP_ERROR",
                    details={
                        "status_code": response.status_code,
                        "url": target_url,
                        "method": f"{method} {path}",
                        "response_headers": dict(response.headers),
                        "response_body": error_body
                    },
                    context=f"Direct request forwarding"
                )
                
                # Update error stats
                async with self.lock:
                    self.worker_error_counts[worker_idx] += 1
                    # Only mark worker unhealthy after multiple consecutive server errors
                    if response.status_code >= 500:
                        if not hasattr(self, 'consecutive_errors'):
                            self.consecutive_errors = [0] * len(self.worker_urls)
                        
                        self.consecutive_errors[worker_idx] += 1
                        
                        # Only mark unhealthy after 3 consecutive errors
                        if self.consecutive_errors[worker_idx] >= 3:
                            self.healthy_workers.discard(worker_idx)
                            logger.warning(f"⚠️ Worker {worker_idx} marked UNHEALTHY after {self.consecutive_errors[worker_idx]} consecutive errors (HTTP {response.status_code})")
                        else:
                            logger.debug(f"Worker {worker_idx} error count: {self.consecutive_errors[worker_idx]}/3 (HTTP {response.status_code})")
                    else:
                        # Reset consecutive error count on successful response
                        if hasattr(self, 'consecutive_errors'):
                            self.consecutive_errors[worker_idx] = 0
            else:
                # Update stats only for successful requests
                async with self.lock:
                    self.total_requests += 1
                    self.worker_request_counts[worker_idx] += 1
                    # Reset consecutive error count on successful response
                    if hasattr(self, 'consecutive_errors'):
                        self.consecutive_errors[worker_idx] = 0
                    # Track worker activity
                    self.worker_last_activity[worker_idx] = time.time()
            
            # Return the response (including errors, so client can see them)
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.headers.get('content-type')
            )
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            
            # Use centralized error logging
            log_worker_error(
                worker_idx=worker_idx,
                worker_url=worker_url,
                error_type="EXCEPTION",
                details={
                    "method": f"{method} {path}",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": error_details
                },
                context=f"Direct request forwarding failed"
            )
            
            # Update error stats
            async with self.lock:
                self.worker_error_counts[worker_idx] += 1
                # Only mark unhealthy after consecutive errors (not single failures)
                if not hasattr(self, 'consecutive_errors'):
                    self.consecutive_errors = [0] * len(self.worker_urls)
                
                self.consecutive_errors[worker_idx] += 1
                
                # Mark unhealthy after 3 consecutive errors
                if self.consecutive_errors[worker_idx] >= 3:
                    self.healthy_workers.discard(worker_idx)
                    logger.warning(f"⚠️ Worker {worker_idx} marked UNHEALTHY after {self.consecutive_errors[worker_idx]} consecutive errors (last error: {e})")
                else:
                    logger.debug(f"Worker {worker_idx} error count: {self.consecutive_errors[worker_idx]}/3 (error: {e})")
            
            raise HTTPException(status_code=502, detail=f"Worker {worker_idx} error: {e}")
    
    async def get_stats(self) -> dict:
        """Get load balancer statistics."""
        async with self.lock:
            stats = {
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
            
            # Add centralized batching statistics if enabled
            if self.centralized_batch_processor:
                stats["centralized_batching"] = self.centralized_batch_processor.get_stats()
            else:
                stats["centralized_batching"] = {"enabled": False}
            
            # Add consecutive error counts for debugging
            if hasattr(self, 'consecutive_errors'):
                stats["worker_consecutive_errors"] = self.consecutive_errors
            else:
                stats["worker_consecutive_errors"] = [0] * len(self.worker_urls)
            
            return stats
    
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
        logger.error("🚨 NO HEALTHY WORKERS FOUND!")
        logger.error("Common issues: workers not started yet, model loading failed, or network issues")
        logger.error("Check worker error messages above (if any) or check worker processes directly")
    else:
        logger.info(f"✅ Found {healthy_count}/{total_count} healthy workers")
        
        # Only log details if some workers are unhealthy
        if healthy_count < total_count:
            for i, url in enumerate(worker_pool.worker_urls):
                if i not in worker_pool.healthy_workers:
                    logger.info(f"  Worker {i}: {url} - ❌ UNHEALTHY")
    
    # Start periodic health checks
    health_check_task = asyncio.create_task(worker_pool.start_health_checks())
    
    # Start centralized batch processing if enabled
    if worker_pool.centralized_batch_processor:
        await worker_pool.centralized_batch_processor.start_batch_processing()
    
    yield
    
    # Cleanup
    health_check_task.cancel()
    
    # Stop centralized batch processing
    if worker_pool.centralized_batch_processor:
        await worker_pool.centralized_batch_processor.stop_batch_processing()
    
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

@app.get("/worker-status") 
async def get_worker_status():
    """Get simple worker health status for monitoring."""
    stats = await worker_pool.get_stats()
    current_time = time.time()
    
    worker_status = []
    for worker in stats["workers"]:
        worker_idx = worker["index"]
        last_activity = worker_pool.worker_last_activity[worker_idx]
        time_since_activity = current_time - last_activity
        
        status = {
            "worker_id": worker_idx,
            "url": worker["url"],
            "healthy": worker["healthy"],
            "requests_served": worker["requests_served"],
            "errors": worker["errors"],
            "consecutive_errors": stats["worker_consecutive_errors"][worker_idx],
            "last_activity_seconds_ago": round(time_since_activity, 1),
            "recently_active": time_since_activity < 90,
            "status": "busy" if time_since_activity < 30 else "idle" if worker["healthy"] else "unhealthy"
        }
        worker_status.append(status)
    
    return {
        "timestamp": current_time,
        "healthy_workers": stats["healthy_workers"],
        "total_workers": stats["total_workers"],
        "workers": worker_status,
        "system_status": "overloaded" if all(w["recently_active"] for w in worker_status) else "normal"
    }


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
    parser.add_argument("--enable-centralized-batching", action="store_true", default=True,
                       help="Enable centralized batching at load balancer level (default: True)")
    parser.add_argument("--disable-centralized-batching", action="store_true", 
                       help="Disable centralized batching (use round-robin only)")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Centralized batch size (default: 8)")
    parser.add_argument("--batch-wait-time", type=float, default=0.02,
                       help="Maximum wait time for centralized batching in seconds (default: 0.02)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("🔍 VERBOSE MODE ENABLED")
    else:
        # Ensure we see ERROR messages even in non-verbose mode
        logging.getLogger().setLevel(logging.INFO)
    
    # Ensure ERROR level messages are visible (avoid duplicate handlers)
    if not any(isinstance(handler, logging.StreamHandler) for handler in logging.getLogger().handlers):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.ERROR)
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logging.getLogger().addHandler(console_handler)
    
    # Create worker URLs
    worker_urls = [f"http://{args.worker_host}:{port}" for port in args.worker_ports]
    
    # Determine centralized batching setting
    if args.disable_centralized_batching:
        enable_batching = False
    else:
        enable_batching = args.enable_centralized_batching  # Default is True now
    
    # Initialize worker pool
    worker_pool = WorkerPool(
        worker_urls, 
        health_check_interval=args.health_check_interval,
        request_timeout=args.request_timeout,
        enable_centralized_batching=enable_batching,
        batch_size=args.batch_size,
        batch_wait_time=args.batch_wait_time
    )
    
    logger.info(f"Starting load balancer on {args.host}:{args.port}")
    logger.info(f"Distributing across {len(worker_urls)} workers")
    logger.info(f"Centralized batching: {'ENABLED' if enable_batching else 'DISABLED'}")
    if enable_batching:
        logger.info(f"  Batch size: {args.batch_size}")
        logger.info(f"  Batch wait time: {args.batch_wait_time}s")
        logger.info(f"  ✅ Centralized batching will improve GPU utilization by grouping requests")
    else:
        logger.info(f"  Using round-robin distribution - workers will batch naturally based on concurrent load")
    logger.info(f"Timeout settings: request={args.request_timeout}s, keep-alive={args.timeout_keep_alive}s")
    logger.info(f"🔍 Enhanced error logging: ENABLED (detailed errors will appear in stdout)")
    
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

