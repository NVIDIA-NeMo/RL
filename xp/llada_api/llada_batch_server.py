#!/usr/bin/env python3
"""
Batch-enabled OpenAI-compatible API server for LLaDA models with Fast-dLLM acceleration.

This server provides batching capabilities by accumulating incoming requests
and processing them together using Fast-dLLM's native batch processing.

Usage:
    python llada_batch_server.py --model-path /path/to/checkpoint --batch-size 8
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional, Union
from dataclasses import dataclass
from collections import deque
import threading

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoConfig, AutoModel, AutoTokenizer

# Add Fast-dLLM to Python path
FAST_DLLM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../3rdparty/Fast-dLLM/llada')
if os.path.exists(FAST_DLLM_PATH):
    sys.path.insert(0, FAST_DLLM_PATH)
    try:
        from generate import generate, generate_with_prefix_cache, generate_with_dual_cache
        from model.modeling_llada import LLaDAModelLM
        FAST_DLLM_AVAILABLE = True
    except ImportError as e:
        logging.warning(f"Failed to import Fast-dLLM: {e}")
        FAST_DLLM_AVAILABLE = False
        generate = None
        generate_with_prefix_cache = None
        generate_with_dual_cache = None
        LLaDAModelLM = None
else:
    logging.warning(f"Fast-dLLM path not found: {FAST_DLLM_PATH}")
    FAST_DLLM_AVAILABLE = False

# Import original server components
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llada_openai_server import (
    ChatMessage, ChatCompletionRequest, ChatCompletionChoice, ChatCompletionResponse,
    MASK_ID
)

# Import generation registry
from llada_generate import (
    get_algorithm, 
    list_available_algorithms,
    list_algorithms,
    GenerationAlgorithm
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables
model = None
tokenizer = None
device = None
model_config = None


def load_model_from_hf(model_path: str):
    """Load LLaDA model from HuggingFace format (local path or HuggingFace model name)."""
    global model, tokenizer, device, model_config
    
    # Determine if this is a local path or HuggingFace model name
    is_local_path = os.path.exists(model_path)
    model_type = "local path" if is_local_path else "HuggingFace model name"
    
    logger.info(f"Loading LLaDA model from {model_type}: {model_path}")
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        # Load tokenizer (works with both local paths and HF model names)
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Load model using Fast-dLLM optimized model if available
        logger.info("Loading model...")
        if FAST_DLLM_AVAILABLE and LLaDAModelLM is not None:
            logger.info("Using Fast-dLLM optimized model class")
            model = LLaDAModelLM.from_pretrained(
                model_path, 
                trust_remote_code=True, 
                torch_dtype=torch.bfloat16
            )
        else:
            logger.warning("Fast-dLLM not available, falling back to standard AutoModel")
            model = AutoModel.from_pretrained(
                model_path, 
                trust_remote_code=True, 
                torch_dtype=torch.bfloat16
            )
        
        model = model.to(device).eval()
        
        # Load config (works with both local paths and HF model names)
        logger.info("Loading model config...")
        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        logger.info(f"Model loaded successfully from {model_type}!")
        logger.info(f"Fast-dLLM optimizations: {'enabled' if FAST_DLLM_AVAILABLE else 'disabled'}")
        return True
    except Exception as e:
        logger.error(f"Failed to load model from {model_type} '{model_path}': {e}")
        return False


def load_model_from_dcp(dcp_path: str, base_model: str, temp_dir: str = "/tmp/llada_hf_converted"):
    """Load LLaDA model from DCP checkpoint by converting to HF format first.""" 
    global model, tokenizer, device, model_config
    
    try:
        from nemo_rl.utils.native_checkpoint import convert_dcp_to_hf, convert_structured_dcp_to_hf, load_checkpoint
        NEMO_RL_AVAILABLE = True
    except ImportError:
        NEMO_RL_AVAILABLE = False
    
    if not NEMO_RL_AVAILABLE:
        logger.error("NeMo-RL is not available. DCP checkpoint loading requires nemo_rl.utils.native_checkpoint.")
        logger.error("For local execution, please:")
        logger.error("1. Set PYTHONPATH to include NeMo-RL: export PYTHONPATH=/path/to/NeMo-RL:$PYTHONPATH")
        logger.error("2. Install NeMo-RL dependencies: uv sync --locked --no-install-project") 
        logger.error("3. Or use a HuggingFace model instead: --model-path GSAI-ML/LLaDA-8B-Instruct")
        return False
    
    logger.info(f"Converting DCP checkpoint to HuggingFace format...")
    logger.info(f"DCP path: {dcp_path}")
    logger.info(f"Base model: {base_model}")
    logger.info(f"Temp HF path: {temp_dir}")
    
    try:
        # Check if this is a structured checkpoint (has weights/ and tokenizer/ subdirectories)
        weights_dir = os.path.join(dcp_path, "weights")
        tokenizer_dir = os.path.join(dcp_path, "tokenizer")
        
        if os.path.exists(weights_dir) and os.path.exists(tokenizer_dir):
            logger.info(f"Detected structured DCP checkpoint with weights/ and tokenizer/ directories")
            logger.info(f"  Weights: {weights_dir}")
            logger.info(f"  Tokenizer: {tokenizer_dir}")
            
            # Use the new structured conversion function
            hf_path = convert_structured_dcp_to_hf(
                dcp_root_path=dcp_path,
                hf_ckpt_path=temp_dir,
                model_name_or_path=base_model,
                overwrite=True
            )
        else:
            logger.info(f"Using legacy DCP checkpoint format (direct weights path)")
            # Convert DCP to HF format using the old method
            hf_path = convert_dcp_to_hf(
                dcp_ckpt_path=dcp_path,
                hf_ckpt_path=temp_dir,
                model_name_or_path=base_model,
                tokenizer_name_or_path=base_model,
                overwrite=True
            )
        
        logger.info(f"Conversion completed. Loading from: {hf_path}")
        
        # Now load from HF format
        return load_model_from_hf(hf_path)
        
    except Exception as e:
        logger.error(f"Failed to convert or load DCP checkpoint: {e}")
        return False


@dataclass
class BatchRequest:
    """Represents a single request in a batch."""
    request_id: str
    request: ChatCompletionRequest
    future: asyncio.Future
    timestamp: float

class BatchProcessor:
    """Handles batching of requests and processing them together."""
    
    def __init__(self, max_batch_size: int = 8, max_wait_time: float = 0.1):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.pending_requests: deque = deque()
        self.processing = False
        self.lock = asyncio.Lock()
        
        # Start the batch processing loop
        asyncio.create_task(self._batch_processing_loop())
    
    async def add_request(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Add a request to the batch and wait for its result."""
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
            logger.debug(f"Added request {request_id} to batch queue. Queue size: {len(self.pending_requests)}")
        
        # Wait for the result
        try:
            return await future
        except Exception as e:
            logger.error(f"Request {request_id} failed: {e}")
            raise
    
    async def _batch_processing_loop(self):
        """Continuously process batches of requests."""
        while True:
            try:
                await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
                
                if not self.pending_requests or self.processing:
                    continue
                
                # Check if we should process a batch
                should_process = False
                async with self.lock:
                    if len(self.pending_requests) >= self.max_batch_size:
                        should_process = True
                    elif self.pending_requests:
                        oldest_request_time = self.pending_requests[0].timestamp
                        if time.time() - oldest_request_time >= self.max_wait_time:
                            should_process = True
                
                if should_process:
                    await self._process_batch()
                    
            except Exception as e:
                logger.error(f"Error in batch processing loop: {e}")
    
    async def _process_batch(self):
        """Process a batch of requests."""
        if self.processing:
            return
        
        self.processing = True
        batch_requests = []
        
        try:
            # Extract requests from the queue
            async with self.lock:
                while self.pending_requests and len(batch_requests) < self.max_batch_size:
                    batch_requests.append(self.pending_requests.popleft())
            
            if not batch_requests:
                return
            
            logger.info(f"Processing batch of {len(batch_requests)} requests")
            batch_start_time = time.time()
            
            # Process the batch
            results = await self._process_batch_requests(batch_requests)
            
            batch_time = time.time() - batch_start_time
            logger.info(f"Batch of {len(batch_requests)} completed in {batch_time:.3f}s ({len(batch_requests)/batch_time:.1f} req/s)")
            
            # Return results to waiting futures
            for batch_request, result in zip(batch_requests, results):
                if isinstance(result, Exception):
                    batch_request.future.set_exception(result)
                else:
                    batch_request.future.set_result(result)
                    
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Set exception for all pending requests
            for batch_request in batch_requests:
                if not batch_request.future.done():
                    batch_request.future.set_exception(e)
        finally:
            self.processing = False
    
    async def _process_batch_requests(self, batch_requests: List[BatchRequest]) -> List[Union[ChatCompletionResponse, Exception]]:
        """Process a batch of requests using Fast-dLLM batch capabilities."""
        try:
            # Group requests by generation algorithm for efficient batching
            algorithm_groups = {}
            request_to_group = {}
            
            for i, batch_req in enumerate(batch_requests):
                request = batch_req.request
                
                # Determine generation algorithm for this request
                algorithm_name = request.generation_algorithm or "dual_cache"
                
                # Group by algorithm
                if algorithm_name not in algorithm_groups:
                    algorithm_groups[algorithm_name] = []
                algorithm_groups[algorithm_name].append((i, batch_req))
                request_to_group[i] = algorithm_name
            
            logger.info(f"Processing batch with {len(algorithm_groups)} algorithm groups: {list(algorithm_groups.keys())}")
            
            # Process each algorithm group separately and collect results
            all_results = [None] * len(batch_requests)  # Maintain original order
            
            for algorithm_name, group_requests in algorithm_groups.items():
                group_indices = [idx for idx, _ in group_requests]
                group_batch_requests = [req for _, req in group_requests]
                
                logger.info(f"Processing {len(group_batch_requests)} requests with {algorithm_name} algorithm")
                
                try:
                    group_results = await self._process_algorithm_group(algorithm_name, group_batch_requests)
                    
                    # Place results back in original order
                    for result_idx, original_idx in enumerate(group_indices):
                        all_results[original_idx] = group_results[result_idx]
                        
                except Exception as e:
                    logger.error(f"Failed to process {algorithm_name} group: {e}")
                    # Set error for all requests in this group
                    error = HTTPException(status_code=500, detail=f"Algorithm {algorithm_name} failed: {e}")
                    for _, original_idx in enumerate(group_indices):
                        all_results[original_idx] = error
            
            return all_results
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return [HTTPException(status_code=500, detail=f"Batch processing failed: {e}") for _ in batch_requests]
    
    async def _process_algorithm_group(self, algorithm_name: str, batch_requests: List[BatchRequest]) -> List[Union[ChatCompletionResponse, Exception]]:
        """Process a group of requests that all use the same algorithm."""
        try:
            # Get the generation algorithm
            algorithm = get_algorithm(algorithm_name)
            if algorithm is None:
                raise RuntimeError(f"Unknown generation algorithm: {algorithm_name}")
            
            if not algorithm.is_available():
                raise RuntimeError(f"Generation algorithm '{algorithm_name}' is not available")
            
            # Prepare batch data
            batch_prompts = []
            batch_configs = []
            
            for batch_req in batch_requests:
                request = batch_req.request
                
                # Format messages into prompt
                try:
                    formatted_prompt = tokenizer.apply_chat_template(
                        [{"role": msg.role, "content": msg.content} for msg in request.messages],
                        add_generation_prompt=True,
                        tokenize=False
                    )
                    batch_prompts.append(formatted_prompt)
                    
                    # Store configuration for each request
                    batch_configs.append({
                        'steps': request.steps,
                        'gen_length': request.max_tokens or 128,
                        'block_length': request.block_length,
                        'temperature': request.temperature,
                        'remasking': request.remasking,
                        'threshold': request.threshold,
                        'factor': request.factor
                    })
                except Exception as e:
                    logger.error(f"Failed to format prompt for request {batch_req.request_id}: {e}")
                    return [HTTPException(status_code=400, detail=f"Failed to format chat template: {e}") for _ in batch_requests]
            
            # Tokenize batch prompts
            try:
                # Find max length and pad all prompts
                tokenized_prompts = [tokenizer(prompt, return_tensors="pt")['input_ids'] for prompt in batch_prompts]
                max_prompt_length = max(p.shape[1] for p in tokenized_prompts)
                
                # Pad prompts to same length
                padded_prompts = []
                for prompt_ids in tokenized_prompts:
                    if prompt_ids.shape[1] < max_prompt_length:
                        padding = torch.full(
                            (1, max_prompt_length - prompt_ids.shape[1]), 
                            tokenizer.pad_token_id, 
                            dtype=torch.long, 
                            device=device
                        )
                        # Move prompt_ids to same device as padding before concatenation
                        prompt_ids = prompt_ids.to(device)
                        prompt_ids = torch.cat([padding, prompt_ids], dim=1)
                    else:
                        # Move to device even if no padding needed
                        prompt_ids = prompt_ids.to(device)
                    padded_prompts.append(prompt_ids)
                
                # Stack into batch tensor
                batch_input_ids = torch.cat(padded_prompts, dim=0).to(device)
                logger.debug(f"Batch input shape: {batch_input_ids.shape}")
                
            except Exception as e:
                logger.error(f"Failed to tokenize batch: {e}")
                return [HTTPException(status_code=500, detail=f"Failed to tokenize input: {e}") for _ in batch_requests]
            
            # Use the first request's config for the entire batch (assumes similar configs)
            # TODO: Handle heterogeneous batch configurations
            config = batch_configs[0]
            gen_length = config['gen_length']
            block_length = min(config['block_length'], gen_length)
            
            # Adjust gen_length to be divisible by block_length
            if gen_length % block_length != 0:
                gen_length = ((gen_length // block_length) + 1) * block_length
                logger.debug(f"Adjusted gen_length to {gen_length} to be divisible by block_length {block_length}")
            
            # Pretty print sample messages from the first request in the batch
            sample_request = batch_requests[0]
            sample_request_id = sample_request.request_id
            logger.info("=" * 80)
            logger.info(f"📝 BATCH SAMPLE INPUT ({len(batch_requests)} requests total) [Request ID: {sample_request_id}]:")
            logger.info("=" * 80)
            for i, msg in enumerate(sample_request.request.messages):
                logger.info(f"[{i+1}] {msg.role.upper()}:")
                # Handle multi-line content better
                content_lines = msg.content.strip().split('\n')
                if len(content_lines) == 1:
                    logger.info(f"    {content_lines[0]}")
                else:
                    for line in content_lines:
                        logger.info(f"    {line}")
                if i < len(sample_request.request.messages) - 1:  # Add separator between messages
                    logger.info("    " + "-" * 60)
            
            # Generate with selected algorithm (batch processing)
            try:
                logger.info(f"Using {algorithm.name} generation for {batch_input_ids.shape[0]} requests: {algorithm.description}")
                
                # Log detailed tensor information before generation
                logger.info(f"🔍 GENERATION DEBUG INFO:")
                logger.info(f"  batch_input_ids.shape: {batch_input_ids.shape}")
                logger.info(f"  batch_input_ids.device: {batch_input_ids.device}")
                logger.info(f"  batch_input_ids.dtype: {batch_input_ids.dtype}")
                logger.info(f"  config: {config}")
                logger.info(f"  gen_length: {gen_length}")
                logger.info(f"  block_length: {block_length}")
                logger.info(f"  algorithm: {algorithm.name}")
                
                # Log individual prompt shapes for debugging
                for i, prompt_ids in enumerate(padded_prompts):
                    logger.info(f"  prompt[{i}] shape: {prompt_ids.shape}")
                
                output, nfe = algorithm.generate(
                    model=model,
                    prompt=batch_input_ids,
                    steps=config['steps'],
                    gen_length=gen_length,
                    block_length=block_length,
                    temperature=config['temperature'],
                    remasking=config['remasking'],
                    threshold=config['threshold'],
                    factor=config['factor']
                )
                
                logger.info(f"✅ Batch generation completed with {nfe} forward passes")
                logger.info(f"  output.shape: {output.shape}")
                logger.info(f"  output.device: {output.device}")
                logger.info(f"  output.dtype: {output.dtype}")
                
            except Exception as e:
                import traceback
                logger.error(f"❌ Fast-dLLM batch generation failed with detailed context:")
                logger.error(f"  Error: {e}")
                logger.error(f"  Error type: {type(e).__name__}")
                logger.error(f"  batch_input_ids.shape: {batch_input_ids.shape}")
                logger.error(f"  gen_length: {gen_length}")
                logger.error(f"  block_length: {block_length}")
                logger.error(f"  algorithm: {algorithm.name}")
                logger.error(f"  config: {config}")
                logger.error(f"  Full traceback:")
                for line in traceback.format_exc().split('\n'):
                    logger.error(f"    {line}")
                return [HTTPException(status_code=500, detail=f"Generation failed: {e}") for _ in batch_requests]
            
            # Decode and format responses
            results = []
            sample_response_text = None  # Store sample response for pretty printing
            sample_output_request_id = None  # Store the request ID of the sample output
            
            logger.info(f"🔍 RESPONSE DECODING DEBUG INFO:")
            logger.info(f"  output.shape: {output.shape}")
            logger.info(f"  len(batch_requests): {len(batch_requests)}")
            logger.info(f"  len(padded_prompts): {len(padded_prompts)}")
            
            for i, (batch_req, prompt_ids) in enumerate(zip(batch_requests, padded_prompts)):
                try:
                    logger.debug(f"  Processing response {i}/{len(batch_requests)}")
                    logger.debug(f"    batch_req.request_id: {batch_req.request_id}")
                    logger.debug(f"    prompt_ids.shape: {prompt_ids.shape}")
                    logger.debug(f"    output[{i}] shape would be: {output[i:i+1].shape}")
                    logger.debug(f"    slice [{i}:{i+1}, {prompt_ids.shape[1]}:] from output.shape {output.shape}")
                    
                    # Extract generated tokens for this request
                    generated_tokens = output[i:i+1, prompt_ids.shape[1]:]
                    logger.debug(f"    generated_tokens.shape: {generated_tokens.shape}")
                    
                    generated_text = tokenizer.batch_decode(
                        generated_tokens, 
                        skip_special_tokens=True
                    )[0].strip()
                    
                    logger.debug(f"    generated_text length: {len(generated_text)}")
                    
                    # Store response from the same request we printed input for
                    if batch_req.request_id == sample_request_id:
                        sample_response_text = generated_text
                        sample_output_request_id = batch_req.request_id
                    
                    # Create response
                    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
                    created = int(time.time())
                    
                    # Calculate token usage
                    input_tokens = prompt_ids.shape[1]
                    output_tokens = generated_tokens.shape[1]
                    total_tokens = input_tokens + output_tokens
                    
                    response = ChatCompletionResponse(
                        id=completion_id,
                        created=created,
                        model=batch_req.request.model,
                        choices=[
                            ChatCompletionChoice(
                                index=0,
                                message=ChatMessage(role="assistant", content=generated_text),
                                finish_reason="stop"
                            )
                        ],
                        usage={
                            "prompt_tokens": input_tokens,
                            "completion_tokens": output_tokens,
                            "total_tokens": total_tokens
                        }
                    )
                    results.append(response)
                    
                except Exception as e:
                    import traceback
                    logger.error(f"❌ Failed to decode response for request {batch_req.request_id}:")
                    logger.error(f"  Error: {e}")
                    logger.error(f"  Error type: {type(e).__name__}")
                    logger.error(f"  Request index: {i}")
                    logger.error(f"  prompt_ids.shape: {prompt_ids.shape}")
                    logger.error(f"  output.shape: {output.shape}")
                    logger.error(f"  Attempted slice: output[{i}:{i+1}, {prompt_ids.shape[1]}:]")
                    logger.error(f"  Full traceback:")
                    for line in traceback.format_exc().split('\n'):
                        logger.error(f"    {line}")
                    results.append(HTTPException(status_code=500, detail=f"Failed to decode response: {e}"))
            
            # Pretty print sample response from the same request we printed input for
            if sample_response_text is not None and sample_output_request_id == sample_request_id:
                logger.info("=" * 80)
                logger.info(f"🤖 BATCH SAMPLE OUTPUT [Request ID: {sample_output_request_id}]:")
                logger.info("=" * 80)
                # Handle multi-line responses better
                response_lines = sample_response_text.strip().split('\n')
                if len(response_lines) == 1:
                    logger.info(f"    {response_lines[0]}")
                else:
                    for line in response_lines:
                        logger.info(f"    {line}")
                logger.info("=" * 80)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return [HTTPException(status_code=500, detail=f"Batch processing failed: {e}") for _ in batch_requests]

# Global batch processor
batch_processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for model loading."""
    global batch_processor
    logger.info("Starting LLaDA Batch OpenAI API Server...")
    
    if model is None:
        logger.error("Model not loaded! Server will not work properly.")
    
    # Initialize batch processor
    batch_processor = BatchProcessor(
        max_batch_size=app.state.batch_size,
        max_wait_time=app.state.max_wait_time
    )
    
    yield
    
    logger.info("Shutting down LLaDA Batch OpenAI API Server...")

# Create FastAPI app
app = FastAPI(
    title="LLaDA Batch OpenAI API",
    description="Batch-enabled OpenAI-compatible API for LLaDA diffusion language models",
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

@app.get("/v1/models")
async def list_models():
    """List available models."""
    from llada_openai_server import ModelList, ModelInfo
    return ModelList(
        object="list",
        data=[
            ModelInfo(
                id="llada-8b-instruct",
                created=int(time.time()),
                owned_by="llada"
            )
        ]
    )

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Create chat completion using batched LLaDA model processing."""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming is not supported in batch mode")
    
    try:
        # Add request to batch processor
        result = await batch_processor.add_request(request)
        return result
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    available_algorithms = list_available_algorithms()
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device,
        "batch_processor_active": batch_processor is not None,
        "pending_requests": len(batch_processor.pending_requests) if batch_processor else 0,
        "available_generation_algorithms": available_algorithms,
        "total_generation_algorithms": len(list_algorithms())
    }

@app.get("/batch/stats")
async def batch_stats():
    """Get batch processing statistics."""
    if not batch_processor:
        raise HTTPException(status_code=503, detail="Batch processor not initialized")
    
    return {
        "max_batch_size": batch_processor.max_batch_size,
        "max_wait_time": batch_processor.max_wait_time,
        "pending_requests": len(batch_processor.pending_requests),
        "currently_processing": batch_processor.processing
    }

@app.get("/generation/algorithms")
async def list_generation_algorithms():
    """List all available generation algorithms."""
    from llada_generate import registry
    
    algorithms = []
    for name in registry.list_algorithms():
        algorithm_info = registry.get_algorithm_info(name)
        if algorithm_info:
            algorithms.append(algorithm_info)
    
    return {
        "algorithms": algorithms,
        "note": "Generation algorithm is now specified per request using the 'generation_algorithm' parameter"
    }

def main():
    global model, tokenizer, device, model_config
    
    parser = argparse.ArgumentParser(description="LLaDA Batch OpenAI API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--model-path", help="Path to HuggingFace model")
    parser.add_argument("--dcp-path", help="Path to DCP checkpoint")
    parser.add_argument("--base-model", default="GSAI-ML/LLaDA-8B-Instruct", 
                       help="Base model name for DCP conversion")
    parser.add_argument("--temp-dir", default="/tmp/llada_hf_converted",
                       help="Temporary directory for DCP conversion")
    parser.add_argument("--batch-size", type=int, default=8, 
                       help="Maximum batch size for processing requests")
    parser.add_argument("--max-wait-time", type=float, default=0.1,
                       help="Maximum time to wait for batch to fill (seconds)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose debug logging (very verbose, use for troubleshooting)")
    
    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("🔍 VERBOSE MODE ENABLED - Logging will be very verbose!")
    else:
        logging.getLogger().setLevel(logging.INFO)
    
    # Load model
    if args.model_path:
        success = load_model_from_hf(args.model_path)
    elif args.dcp_path:
        success = load_model_from_dcp(args.dcp_path, args.base_model, args.temp_dir)
    else:
        logger.error("Either --model-path or --dcp-path must be provided")
        return
    
    if not success:
        logger.error("Failed to load model. Exiting.")
        return
    
    # Log available generation algorithms
    available_algorithms = list_available_algorithms()
    logger.info(f"Available generation algorithms: {available_algorithms}")
    if not available_algorithms:
        logger.warning("No generation algorithms are available. Check Fast-dLLM installation.")
    
    # Store batch configuration in app state
    app.state.batch_size = args.batch_size
    app.state.max_wait_time = args.max_wait_time
    
    logger.info(f"Starting batch server on {args.host}:{args.port}")
    logger.info(f"Batch size: {args.batch_size}, Max wait time: {args.max_wait_time}s")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
