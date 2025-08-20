#!/usr/bin/env python3
"""
Modifications to enable batch processing in the existing llada_openai_server.py

This file shows how to modify the existing server to support batch processing
with minimal code changes.
"""

import asyncio
import time
import logging
from collections import deque
from dataclasses import dataclass
from typing import List, Union
import torch
import uuid

logger = logging.getLogger(__name__)

@dataclass
class BatchRequest:
    """Represents a single request in a batch."""
    request_id: str
    request: 'ChatCompletionRequest'
    future: asyncio.Future
    timestamp: float

def generate_fast_dllm_batch(
    model, prompts_batch, steps=128, gen_length=128, block_length=32,
    temperature=0.0, remasking='low_confidence', mask_id=126336,
    use_cache=True, use_dual_cache=True, threshold=None, factor=None
):
    """
    Batch version of the Fast-dLLM generation function.
    
    Args:
        model: LLaDA model
        prompts_batch: Batch of input tensors of shape (batch_size, L)
        steps: Sampling steps
        gen_length: Generated answer length
        block_length: Block length for generation
        temperature: Categorical distribution sampling temperature
        remasking: Remasking strategy ('low_confidence' or 'random')
        mask_id: The token id of [MASK]
        use_cache: Enable KV caching
        use_dual_cache: Enable dual cache (both prefix and suffix)
        threshold: Confidence threshold for parallel decoding
        factor: Factor for dynamic parallel decoding
    """
    if not FAST_DLLM_AVAILABLE:
        raise RuntimeError("Fast-dLLM is not available. Please check the installation.")
    
    try:
        # Choose the appropriate generation function based on cache settings
        if use_cache:
            if use_dual_cache:
                logger.info(f"Using Fast-dLLM dual cache batch generation for batch size {prompts_batch.shape[0]}")
                output, nfe = generate_with_dual_cache(
                    model=model,
                    prompt=prompts_batch,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=block_length,
                    temperature=temperature,
                    remasking=remasking,
                    mask_id=mask_id,
                    threshold=threshold,
                    factor=factor
                )
            else:
                logger.info(f"Using Fast-dLLM prefix cache batch generation for batch size {prompts_batch.shape[0]}")
                output, nfe = generate_with_prefix_cache(
                    model=model,
                    prompt=prompts_batch,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=block_length,
                    temperature=temperature,
                    remasking=remasking,
                    mask_id=mask_id,
                    threshold=threshold,
                    factor=factor
                )
        else:
            logger.info(f"Using Fast-dLLM basic batch generation for batch size {prompts_batch.shape[0]}")
            output, nfe = generate(
                model=model,
                prompt=prompts_batch,
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                temperature=temperature,
                remasking=remasking,
                mask_id=mask_id,
                threshold=threshold,
                factor=factor
            )
        
        logger.info(f"Batch generation completed with {nfe} forward passes")
        return output
        
    except Exception as e:
        logger.error(f"Fast-dLLM batch generation failed: {e}")
        raise

class SimpleBatchProcessor:
    """Simple batch processor that can be integrated into the existing server."""
    
    def __init__(self, max_batch_size: int = 4, max_wait_time: float = 0.05):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.pending_requests: deque = deque()
        self.lock = asyncio.Lock()
        self.processing = False
    
    async def process_request(self, request):
        """Process a single request, potentially batching it with others."""
        request_id = str(uuid.uuid4())
        future = asyncio.Future()
        batch_request = BatchRequest(
            request_id=request_id,
            request=request,
            future=future,
            timestamp=time.time()
        )
        
        async with self.lock:
            self.pending_requests.append(batch_request)
        
        # Check if we should process immediately
        if len(self.pending_requests) >= self.max_batch_size:
            asyncio.create_task(self._try_process_batch())
        else:
            # Schedule processing after wait time
            asyncio.create_task(self._schedule_batch_processing())
        
        return await future
    
    async def _schedule_batch_processing(self):
        """Schedule batch processing after wait time."""
        await asyncio.sleep(self.max_wait_time)
        await self._try_process_batch()
    
    async def _try_process_batch(self):
        """Try to process a batch if not already processing."""
        if self.processing or not self.pending_requests:
            return
        
        async with self.lock:
            if self.processing or not self.pending_requests:
                return
            self.processing = True
            
            # Extract batch
            batch_requests = []
            while self.pending_requests and len(batch_requests) < self.max_batch_size:
                batch_requests.append(self.pending_requests.popleft())
        
        try:
            logger.info(f"Processing batch of {len(batch_requests)} requests")
            results = await self._process_batch(batch_requests)
            
            # Return results
            for batch_req, result in zip(batch_requests, results):
                if isinstance(result, Exception):
                    batch_req.future.set_exception(result)
                else:
                    batch_req.future.set_result(result)
                    
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            for batch_req in batch_requests:
                if not batch_req.future.done():
                    batch_req.future.set_exception(e)
        finally:
            self.processing = False
    
    async def _process_batch(self, batch_requests: List[BatchRequest]):
        """Process batch of requests - implement the actual batch logic here."""
        # This would contain the batch processing logic
        # Similar to what's in the full batch server above
        return [await self._process_single_request(req.request) for req in batch_requests]
    
    async def _process_single_request(self, request):
        """Fallback to process single request."""
        # Import the original generate_chat_completion function and use it
        from llada_openai_server import generate_chat_completion
        return await generate_chat_completion(request)

# Example integration into existing server:
"""
To integrate this into your existing llada_openai_server.py:

1. Add this at the top of the file:
from batch_modifications import SimpleBatchProcessor

2. Initialize the batch processor globally:
batch_processor = SimpleBatchProcessor(max_batch_size=4, max_wait_time=0.05)

3. Modify the create_chat_completion endpoint:
@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        if request.stream:
            # Handle streaming normally (no batching for streaming)
            result = await generate_chat_completion(request)
            return StreamingResponse(result, ...)
        else:
            # Use batch processor for non-streaming requests
            result = await batch_processor.process_request(request)
            return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
"""
