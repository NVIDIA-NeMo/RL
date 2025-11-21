#!/usr/bin/env python3
"""
⚠️ DEPRECATED: Please use llada_batch_server.py instead ⚠️

This server is deprecated and no longer maintained. It lacks the following features:
- Engine-based architecture (fast-dllm, dinfer, nemotron)
- dInfer support (10x+ faster than Fast-dLLM)
- Systematic model/engine validation
- Per-request algorithm switching within same engine
- Latest optimizations and bug fixes

RECOMMENDED ALTERNATIVE:
    Use llada_batch_server.py which supports:
    - All features from this server
    - Batch processing for 3-5x additional speedup
    - Full engine architecture with dInfer support
    - Better error handling and validation

MIGRATION:
    Old: python llada_openai_server.py --model-path /path/to/model
    New: python llada_batch_server.py --model-path /path/to/model

    The batch server is backward compatible with all OpenAI API clients.
    Streaming is not supported, but most use cases don't need it.

---

OpenAI-compatible API server for LLaDA and Nemotron models (LEGACY VERSION).

This server provides OpenAI API compatibility for diffusion language models:
- LLaDA models with Fast-dLLM optimizations including KV cache and parallel decoding
- Nemotron models with built-in diffusion generation

Usage (DEPRECATED - use llada_batch_server.py):
    # For LLaDA models:
    python llada_openai_server.py --model-path /path/to/llada/checkpoint
    
    # For Nemotron models:
    python llada_openai_server.py --model-path nvidia/Nemotron-Diffusion-Research-4B-v0

For DCP checkpoints (both LLaDA and Nemotron):
    python llada_openai_server.py --dcp-path /path/to/dcp --base-model GSAI-ML/LLaDA-8B-Instruct
    python llada_openai_server.py --dcp-path /path/to/nemotron_dcp --base-model nvidia/Nemotron-Diffusion-Research-4B-v0
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

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# Import generation registry
from llada_generate import (
    get_algorithm, 
    list_available_algorithms,
    list_algorithms,
    GenerationAlgorithm
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables - now store algorithm instances instead of raw models
algorithm_instances = {}  # Dict[algorithm_name, GenerationAlgorithm]
default_algorithm = None  # The default algorithm instance to use
model_type = None  # 'llada' or 'nemotron'

# LLaDA specific constants
MASK_ID = 126336  # The token id of [MASK] in LLaDA tokenizer


# OpenAI API Models
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")


class ChatCompletionRequest(BaseModel):
    model_config = {"extra": "allow"}  # Allow extra fields for LLaDA-specific parameters
    
    model: str = Field(default="llada-8b-instruct")
    messages: List[ChatMessage] = Field(..., description="List of messages")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)  # Fast-dLLM default
    max_tokens: Optional[int] = Field(default=128, gt=0)  # Fast-dLLM default
    stream: bool = Field(default=False)
    # Standard OpenAI parameters
    top_p: float = Field(default=0.95, ge=0.0, le=1.0, description="Top-p (nucleus) sampling")
    top_k: int = Field(default=-1, description="Top-k sampling (-1 to disable)")
    # Fast-dLLM specific parameters
    steps: int = Field(default=128, ge=1, le=65536, description="Diffusion steps")
    block_length: int = Field(default=32, ge=1, description="Block length for generation")
    cfg_scale: float = Field(default=0.0, ge=0.0, description="Classifier-free guidance scale")
    remasking: str = Field(default="low_confidence", description="Remasking strategy")
    # Generation algorithm selection
    generation_algorithm: Optional[str] = Field(default="dual_cache", description="Generation algorithm to use (basic, prefix_cache, dual_cache)")
    # Fast-dLLM parameters
    threshold: Optional[float] = Field(default=None, description="Confidence threshold for parallel decoding")
    factor: Optional[float] = Field(default=None, description="Factor for dynamic parallel decoding")


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, int]


class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "llada"


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# Helper functions for model loading (similar to batch server)
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






async def generate_chat_completion(request: ChatCompletionRequest) -> Union[ChatCompletionResponse, AsyncGenerator]:
    """Generate chat completion using algorithm-based model."""
    
    # Log generation parameters for verification
    logger.info(f"Generation request received:")
    logger.info(f"  Model: {request.model}")
    logger.info(f"  Temperature: {request.temperature}")
    logger.info(f"  Max tokens: {request.max_tokens}")
    logger.info(f"  Top-p: {request.top_p} (NOTE: Not used in LLaDA diffusion generation)")
    logger.info(f"  Top-k: {request.top_k} (NOTE: Not used in LLaDA diffusion generation)")
    logger.info(f"  LLaDA steps: {request.steps}")
    logger.info(f"  Block length: {request.block_length}")
    logger.info(f"  CFG scale: {request.cfg_scale}")
    logger.info(f"  Remasking: {request.remasking}")
    logger.info(f"  Generation algorithm: {request.generation_algorithm}")
    logger.info(f"  Fast-dLLM threshold: {request.threshold}")
    logger.info(f"  Fast-dLLM factor: {request.factor}")
    
    # Log any extra parameters received via NeMo-Skills extra_body
    request_values = request.model_dump()
    extra_fields = {k: v for k, v in request_values.items() if k not in request.model_fields}
    if extra_fields:
        logger.info(f"  Extra parameters received: {extra_fields}")
    
    # Format messages into prompt
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty")
    
    # Log user messages with better formatting
    logger.info("=" * 80)
    logger.info("📝 USER MESSAGES:")
    logger.info("=" * 80)
    for i, msg in enumerate(request.messages):
        logger.info(f"[{i+1}] {msg.role.upper()}:")
        # Handle multi-line content better
        content_lines = msg.content.strip().split('\n')
        if len(content_lines) == 1:
            logger.info(f"    {content_lines[0]}")
        else:
            for line in content_lines:
                logger.info(f"    {line}")
        if i < len(request.messages) - 1:  # Add separator between messages
            logger.info("    " + "-" * 60)
    
    # Select algorithm based on model type and request
    if model_type == "nemotron":
        algorithm_name = request.generation_algorithm or "nemotron"
        logger.info(f"Using Nemotron generation with algorithm: {algorithm_name}")
    else:
        algorithm_name = request.generation_algorithm or "dual_cache"
        logger.info(f"Using LLaDA generation with algorithm: {algorithm_name}")
    
    # Get the algorithm instance
    algorithm = get_algorithm_instance(algorithm_name)
    if algorithm is None:
        raise HTTPException(status_code=500, detail=f"Algorithm '{algorithm_name}' not loaded")
    
    if algorithm.model is None or algorithm.tokenizer is None:
        raise HTTPException(status_code=500, detail=f"Algorithm '{algorithm_name}' does not have a loaded model")
    
    # Prepare prompt using algorithm's method
    try:
        use_chat_template = not (hasattr(app.state, 'no_chat_template') and app.state.no_chat_template)
        
        if use_chat_template:
            # Use algorithm's tokenization with chat template
            messages_list = [{"role": msg.role, "content": msg.content} for msg in request.messages]
            input_ids = algorithm.tokenize_prompts(
                prompts=[],
                apply_chat_template=True,
                messages=[messages_list]
            )
        else:
            # Use raw text from last message
            if request.messages:
                formatted_prompt = request.messages[-1].content
            else:
                formatted_prompt = ""
            logger.debug(f"Using raw text (no chat template): {formatted_prompt[:100]}...")
            input_ids = algorithm.tokenize_prompts(
                prompts=[formatted_prompt],
                apply_chat_template=False,
                messages=None
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to tokenize input: {e}")
    
    # Generate using the algorithm
    try:
        gen_length = request.max_tokens or 128
        
        # Ensure block_length compatibility for LLaDA algorithms
        block_length = min(request.block_length, gen_length)
        if gen_length % block_length != 0:
            # Adjust gen_length to be divisible by block_length
            gen_length = ((gen_length // block_length) + 1) * block_length
            logger.info(f"Adjusted gen_length to {gen_length} to be divisible by block_length {block_length}")
        
        output, nfe = algorithm.generate(
            model=algorithm.model,
            prompt=input_ids,
            steps=request.steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=request.temperature,
            remasking=request.remasking,
            threshold=request.threshold,
            factor=request.factor,
            **extra_fields
        )
        
        logger.info(f"Generation completed with {nfe} forward passes")
        
        # Decode generated text using algorithm's method
        prompt_length = input_ids.shape[1]
        generated_texts = algorithm.decode_outputs(output, [prompt_length])
        generated_text = generated_texts[0]
        
        # Log model response with better formatting
        logger.info("=" * 80)
        logger.info("🤖 MODEL RESPONSE:")
        logger.info("=" * 80)
        # Handle multi-line responses better
        response_lines = generated_text.strip().split('\n')
        if len(response_lines) == 1:
            logger.info(f"    {response_lines[0]}")
        else:
            for line in response_lines:
                logger.info(f"    {line}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
    
    # Create response
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    
    # Calculate token usage
    input_tokens = input_ids.shape[1]
    output_tokens = output.shape[1] - input_tokens
    total_tokens = input_tokens + output_tokens
    
    if request.stream:
        return generate_stream_response(completion_id, created, request.model, generated_text, total_tokens)
    else:
        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=request.model,
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


async def generate_stream_response(completion_id: str, created: int, model_name: str, text: str, total_tokens: int):
    """Generate streaming response for chat completion."""
    
    # Split text into words for streaming effect
    words = text.split()
    
    for i, word in enumerate(words):
        chunk = ChatCompletionStreamResponse(
            id=completion_id,
            created=created,
            model=model_name,
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta={"content": word + (" " if i < len(words) - 1 else "")},
                    finish_reason=None
                )
            ]
        )
        
        yield f"data: {chunk.model_dump_json()}\n\n"
        await asyncio.sleep(0.05)  # Small delay for streaming effect
    
    # Send final chunk
    final_chunk = ChatCompletionStreamResponse(
        id=completion_id,
        created=created,
        model=model_name,
        choices=[
            ChatCompletionStreamChoice(
                index=0,
                delta={},
                finish_reason="stop"
            )
        ]
    )
    
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for model loading."""
    logger.info("Starting LLaDA OpenAI API Server...")
    
    # Model loading is handled in main() before server starts
    if default_algorithm is None or default_algorithm.model is None:
        logger.error("Model not loaded! Server will not work properly.")
    
    yield
    
    logger.info("Shutting down LLaDA OpenAI API Server...")


# Create FastAPI app
app = FastAPI(
    title="LLaDA/Nemotron OpenAI API",
    description="OpenAI-compatible API for LLaDA and Nemotron diffusion language models",
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
    """Create chat completion using algorithm-based model."""
    
    if default_algorithm is None or default_algorithm.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = await generate_chat_completion(request)
        
        if request.stream:
            return StreamingResponse(
                result,
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        else:
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
        "available_generation_algorithms": available_algorithms,
        "loaded_generation_algorithms": loaded_algorithms,
        "default_algorithm": "nemotron" if model_type == "nemotron" else "dual_cache",
        "chat_template_enabled": not (hasattr(app.state, 'no_chat_template') and app.state.no_chat_template),
        "note": "Use 'generation_algorithm' parameter in requests to specify algorithm"
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
        "note": "Generation algorithm is specified per request using the 'generation_algorithm' parameter"
    }


def main():
    global model_type
    
    # Print deprecation warning
    logger.warning("=" * 80)
    logger.warning("⚠️  DEPRECATION WARNING ⚠️")
    logger.warning("=" * 80)
    logger.warning("This server (llada_openai_server.py) is DEPRECATED.")
    logger.warning("")
    logger.warning("Please use llada_batch_server.py instead, which includes:")
    logger.warning("  • Engine architecture (fast-dllm, dinfer, nemotron)")
    logger.warning("  • dInfer support (10x+ faster than Fast-dLLM)")
    logger.warning("  • Batch processing (3-5x additional speedup)")
    logger.warning("  • Per-request algorithm switching")
    logger.warning("  • Latest bug fixes and optimizations")
    logger.warning("")
    logger.warning("Migration:")
    logger.warning("  Old: python llada_openai_server.py --model-path MODEL")
    logger.warning("  New: python llada_batch_server.py --model-path MODEL")
    logger.warning("=" * 80)
    logger.warning("")
    
    parser = argparse.ArgumentParser(
        description="⚠️ DEPRECATED - LLaDA/Nemotron OpenAI API Server (use llada_batch_server.py instead)"
    )
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
    parser.add_argument("--no-chat-template", action="store_true",
                       help="Disable chat template application (feed raw text to tokenizer)")
    
    args = parser.parse_args()
    
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
    
    # Store configuration in app state
    app.state.no_chat_template = args.no_chat_template
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info(f"Default algorithm: {default_algorithm.name if default_algorithm else 'None'}")
    if args.no_chat_template:
        logger.info("Chat template disabled - using raw text input")
    else:
        logger.info("Chat template enabled (default)")  
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
