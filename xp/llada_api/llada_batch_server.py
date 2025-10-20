#!/usr/bin/env python3
"""
Batch-enabled OpenAI-compatible API server for LLaDA and Nemotron models.

This server provides batching capabilities by accumulating incoming requests
and processing them together. Supports both LLaDA models with Fast-dLLM 
acceleration and Nemotron models with built-in diffusion generation.

Usage:
    # For LLaDA models:
    python llada_batch_server.py --model-path /path/to/llada/checkpoint --batch-size 8
    python llada_batch_server.py --dcp-path /path/to/llada_dcp --base-model GSAI-ML/LLaDA-8B-Instruct --batch-size 8
    
    # For Nemotron models:
    python llada_batch_server.py --model-path nvidia/Nemotron-Diffusion-Research-4B-v0 --batch-size 8
    python llada_batch_server.py --dcp-path /path/to/nemotron_dcp --base-model nvidia/Nemotron-Diffusion-Research-4B-v0 --batch-size 8
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

# Import server components
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

# Global variables - now store algorithm instances instead of raw models
algorithm_instances = {}  # Dict[algorithm_name, GenerationAlgorithm]
default_algorithm = None  # The default algorithm instance to use
model_type = None  # 'llada' or 'nemotron'


def load_model_with_algorithm(model_path: str = None, dcp_path: str = None, base_model: str = None, temp_dir: str = "/tmp/model_hf_converted", algorithm_name: str = "dual_cache"):
    """
    Load model using the generation algorithm registry.
    This replaces the old load_model_from_hf and load_model_from_dcp functions.
    
    Args:
        model_path: Path to HuggingFace model (optional)
        dcp_path: Path to DCP checkpoint (optional)
        base_model: Base model name for DCP conversion
        temp_dir: Temporary directory for DCP conversion
        algorithm_name: Name of the algorithm to use for loading
        
    Returns:
        True if successful, False otherwise
    """
    global algorithm_instances, default_algorithm, model_type
    
    # Get the algorithm from the registry
    algorithm = get_algorithm(algorithm_name)
    if algorithm is None:
        logger.error(f"Unknown generation algorithm: {algorithm_name}")
        return False
    
    if not algorithm.is_available():
        logger.error(f"Generation algorithm '{algorithm_name}' is not available")
        return False
    
    logger.info(f"Loading model with {algorithm.name} algorithm: {algorithm.description}")
    
    # Load the model using the algorithm
    try:
        if model_path:
            success = algorithm.load_model_from_hf(model_path)
        elif dcp_path:
            success = algorithm.load_model_from_dcp(dcp_path, base_model, temp_dir)
        else:
            logger.error("Either model_path or dcp_path must be provided")
            return False
        
        if not success:
            return False
        
        # Store the algorithm instance
        algorithm_instances[algorithm_name] = algorithm
        default_algorithm = algorithm
        model_type = algorithm.model_type
        
        logger.info(f"Model loaded successfully with {algorithm_name} algorithm!")
        logger.info(f"Model type: {model_type}")
        
        # Also load models for other compatible algorithms (shared model/tokenizer)
        # This allows switching between algorithms without reloading
        _load_other_algorithms(algorithm)
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model with algorithm '{algorithm_name}': {e}")
        return False


def _load_other_algorithms(source_algorithm: GenerationAlgorithm):
    """
    Load other compatible algorithms using the same model and tokenizer.
    This allows algorithm switching without reloading the model.
    """
    global algorithm_instances
    
    # Get all available algorithms
    all_algorithms = list_algorithms()
    
    for algo_name in all_algorithms:
        if algo_name in algorithm_instances:
            continue  # Already loaded
        
        algo = get_algorithm(algo_name)
        if algo is None or not algo.is_available():
            continue
        
        # Share the model, tokenizer, and config from the source algorithm
        algo.model = source_algorithm.model
        algo.tokenizer = source_algorithm.tokenizer
        algo.device = source_algorithm.device
        algo.model_config = source_algorithm.model_config
        algo.model_type = source_algorithm.model_type
        
        algorithm_instances[algo_name] = algo
        logger.info(f"Loaded {algo_name} algorithm (shared model with {source_algorithm.name})")


def get_algorithm_instance(algorithm_name: str) -> Optional[GenerationAlgorithm]:
    """Get a loaded algorithm instance by name."""
    return algorithm_instances.get(algorithm_name, default_algorithm)




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
                # For Nemotron models, default to "nemotron" algorithm
                if model_type == "nemotron":
                    algorithm_name = request.generation_algorithm or "nemotron"
                else:
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
            # Get the loaded algorithm instance (works for both LLaDA and Nemotron)
            algorithm = get_algorithm_instance(algorithm_name)
            if algorithm is None:
                raise RuntimeError(f"Algorithm '{algorithm_name}' not loaded")
            
            if algorithm.model is None or algorithm.tokenizer is None:
                raise RuntimeError(f"Algorithm '{algorithm_name}' does not have a loaded model")
            
            # Prepare batch data
            batch_prompts = []
            batch_messages = []
            batch_configs = []
            
            for batch_req in batch_requests:
                request = batch_req.request
                
                # Format messages into prompt
                try:
                    if hasattr(app.state, 'no_chat_template') and app.state.no_chat_template:
                        # Skip chat template - use raw content from last message
                        if request.messages:
                            formatted_prompt = request.messages[-1].content
                        else:
                            formatted_prompt = ""
                        logger.debug(f"Using raw text (no chat template): {formatted_prompt[:100]}...")
                        batch_prompts.append(formatted_prompt)
                        batch_messages.append(None)  # No messages for raw text mode
                    else:
                        # Store messages for chat template application by the algorithm
                        messages_list = [{"role": msg.role, "content": msg.content} for msg in request.messages]
                        batch_messages.append(messages_list)
                        batch_prompts.append(None)  # Will be formatted by algorithm
                        logger.debug(f"Prepared messages for chat template")
                    
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
            
            # Tokenize batch prompts using the algorithm's tokenization method
            try:
                # Determine whether to use chat template
                use_chat_template = not (hasattr(app.state, 'no_chat_template') and app.state.no_chat_template)
                
                if use_chat_template:
                    # Use algorithm's tokenization with chat template
                    batch_input_ids = algorithm.tokenize_prompts(
                        prompts=[],  # Empty since we're using messages
                        apply_chat_template=True,
                        messages=batch_messages
                    )
                else:
                    # Use algorithm's tokenization without chat template
                    batch_input_ids = algorithm.tokenize_prompts(
                        prompts=batch_prompts,
                        apply_chat_template=False,
                        messages=None
                    )
                
                logger.debug(f"Batch input shape: {batch_input_ids.shape}")
                
                # Store individual prompt lengths for later decoding
                prompt_lengths = [batch_input_ids.shape[1]] * batch_input_ids.shape[0]
                
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
            
            # Generate using the selected algorithm from the registry
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
                
                output, nfe = algorithm.generate(
                    model=algorithm.model,
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
                logger.error(f"❌ Batch generation failed with detailed context:")
                logger.error(f"  Error: {e}")
                logger.error(f"  Error type: {type(e).__name__}")
                logger.error(f"  Algorithm: {algorithm.name}")
                logger.error(f"  Model type: {algorithm.model_type}")
                logger.error(f"  batch_input_ids.shape: {batch_input_ids.shape}")
                logger.error(f"  gen_length: {gen_length}")
                logger.error(f"  block_length: {block_length}")
                logger.error(f"  config: {config}")
                logger.error(f"  Full traceback:")
                for line in traceback.format_exc().split('\n'):
                    logger.error(f"    {line}")
                return [HTTPException(status_code=500, detail=f"Generation failed: {e}") for _ in batch_requests]
            
            # Decode and format responses using the algorithm's decoding method
            results = []
            sample_response_text = None  # Store sample response for pretty printing
            sample_output_request_id = None  # Store the request ID of the sample output
            
            logger.info(f"🔍 RESPONSE DECODING DEBUG INFO:")
            logger.info(f"  output.shape: {output.shape}")
            logger.info(f"  len(batch_requests): {len(batch_requests)}")
            logger.info(f"  prompt_lengths: {prompt_lengths}")
            
            # Use algorithm's decode_outputs method
            try:
                generated_texts = algorithm.decode_outputs(output, prompt_lengths)
                
                for i, (batch_req, generated_text) in enumerate(zip(batch_requests, generated_texts)):
                    logger.debug(f"  Processing response {i}/{len(batch_requests)}")
                    logger.debug(f"    batch_req.request_id: {batch_req.request_id}")
                    logger.debug(f"    generated_text length: {len(generated_text)}")
                    
                    # Store response from the same request we printed input for
                    if batch_req.request_id == sample_request_id:
                        sample_response_text = generated_text
                        sample_output_request_id = batch_req.request_id
                    
                    # Create response
                    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
                    created = int(time.time())
                    
                    # Calculate token usage
                    input_tokens = prompt_lengths[i]
                    output_tokens = output.shape[1] - input_tokens
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
                logger.error(f"❌ Failed to decode responses:")
                logger.error(f"  Error: {e}")
                logger.error(f"  Error type: {type(e).__name__}")
                logger.error(f"  output.shape: {output.shape}")
                logger.error(f"  prompt_lengths: {prompt_lengths}")
                logger.error(f"  Full traceback:")
                for line in traceback.format_exc().split('\n'):
                    logger.error(f"    {line}")
                return [HTTPException(status_code=500, detail=f"Failed to decode responses: {e}") for _ in batch_requests]
            
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
    
    if default_algorithm is None or default_algorithm.model is None:
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
    title="LLaDA/Nemotron Batch OpenAI API",
    description="Batch-enabled OpenAI-compatible API for LLaDA and Nemotron diffusion language models",
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
    
    if default_algorithm is None or default_algorithm.model is None:
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
    loaded_algorithms = list(algorithm_instances.keys())
    return {
        "status": "healthy",
        "model_loaded": default_algorithm is not None and default_algorithm.model is not None,
        "model_type": model_type,
        "device": default_algorithm.device if default_algorithm else None,
        "batch_processor_active": batch_processor is not None,
        "pending_requests": len(batch_processor.pending_requests) if batch_processor else 0,
        "available_generation_algorithms": available_algorithms,
        "loaded_generation_algorithms": loaded_algorithms,
        "total_generation_algorithms": len(list_algorithms()),
        "default_algorithm": "nemotron" if model_type == "nemotron" else "dual_cache",
        "chat_template_enabled": not (hasattr(app.state, 'no_chat_template') and app.state.no_chat_template)
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
    global model_type
    
    parser = argparse.ArgumentParser(description="LLaDA/Nemotron Batch OpenAI API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--model-path", help="Path to HuggingFace model")
    parser.add_argument("--dcp-path", help="Path to DCP checkpoint (supports both LLaDA and Nemotron models)")
    parser.add_argument("--base-model", default="GSAI-ML/LLaDA-8B-Instruct", 
                       help="Base model name for DCP conversion (e.g., GSAI-ML/LLaDA-8B-Instruct, nvidia/Nemotron-Diffusion-Research-4B-v0)")
    parser.add_argument("--temp-dir", default="/tmp/model_hf_converted",
                       help="Temporary directory for DCP conversion")
    parser.add_argument("--algorithm", default="dual_cache",
                       help="Generation algorithm to use for loading the model (basic, prefix_cache, dual_cache, nemotron)")
    parser.add_argument("--batch-size", type=int, default=8, 
                       help="Maximum batch size for processing requests")
    parser.add_argument("--max-wait-time", type=float, default=0.1,
                       help="Maximum time to wait for batch to fill (seconds)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose debug logging (very verbose, use for troubleshooting)")
    parser.add_argument("--no-chat-template", action="store_true",
                       help="Disable chat template application (feed raw text to tokenizer)")
    
    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("🔍 VERBOSE MODE ENABLED - Logging will be very verbose!")
    else:
        logging.getLogger().setLevel(logging.INFO)
    
    # Determine algorithm to use based on model type if not explicitly specified
    if args.algorithm == "dual_cache" and args.base_model and "nemotron" in args.base_model.lower():
        args.algorithm = "nemotron"
        logger.info(f"Auto-detected Nemotron model, switching to 'nemotron' algorithm")
    elif args.algorithm == "dual_cache" and args.model_path and "nemotron" in args.model_path.lower():
        args.algorithm = "nemotron"
        logger.info(f"Auto-detected Nemotron model, switching to 'nemotron' algorithm")
    
    # Load model using the new algorithm-based loading
    success = load_model_with_algorithm(
        model_path=args.model_path,
        dcp_path=args.dcp_path,
        base_model=args.base_model,
        temp_dir=args.temp_dir,
        algorithm_name=args.algorithm
    )
    
    if not success:
        logger.error("Failed to load model. Exiting.")
        return
    
    # Log available and loaded generation algorithms
    available_algorithms = list_available_algorithms()
    loaded_algorithms = list(algorithm_instances.keys())
    logger.info(f"Available generation algorithms: {available_algorithms}")
    logger.info(f"Loaded generation algorithms: {loaded_algorithms}")
    if not available_algorithms:
        logger.warning("No generation algorithms are available. Check Fast-dLLM installation.")
    
    # Store batch configuration in app state
    app.state.batch_size = args.batch_size
    app.state.max_wait_time = args.max_wait_time
    app.state.no_chat_template = args.no_chat_template
    
    logger.info(f"Starting batch server on {args.host}:{args.port}")
    logger.info(f"Batch size: {args.batch_size}, Max wait time: {args.max_wait_time}s")
    logger.info(f"Default algorithm: {default_algorithm.name if default_algorithm else 'None'}")
    if args.no_chat_template:
        logger.info("Chat template disabled - using raw text input")
    else:
        logger.info("Chat template enabled (default)")  
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
