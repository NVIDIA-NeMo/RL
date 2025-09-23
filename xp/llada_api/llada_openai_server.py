#!/usr/bin/env python3
"""
OpenAI-compatible API server for LLaDA and Nemotron models.

This server provides OpenAI API compatibility for diffusion language models:
- LLaDA models with Fast-dLLM optimizations including KV cache and parallel decoding
- Nemotron models with built-in diffusion generation

Usage:
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
from transformers import AutoConfig, AutoModel, AutoTokenizer

# Add Fast-dLLM to Python path
FAST_DLLM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../3rdparty/Fast-dLLM/llada')
if os.path.exists(FAST_DLLM_PATH):
    sys.path.insert(0, FAST_DLLM_PATH)
    # Import Fast-dLLM generation functions
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
    generate = None
    generate_with_prefix_cache = None
    generate_with_dual_cache = None
    LLaDAModelLM = None

# Import NeMo-RL utilities for DCP handling (optional for local mode)
try:
    from nemo_rl.utils.native_checkpoint import convert_dcp_to_hf, convert_structured_dcp_to_hf, load_checkpoint
    NEMO_RL_AVAILABLE = True
except ImportError:
    NEMO_RL_AVAILABLE = False
    convert_dcp_to_hf = None
    load_checkpoint = None

# Import generation registry
from llada_generate import (
    get_algorithm, 
    list_available_algorithms,
    GenerationAlgorithm
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variables
model = None
tokenizer = None
device = None
model_config = None
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
    steps: int = Field(default=128, ge=1, le=512, description="Diffusion steps")
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


# Registry-based generation function
@torch.no_grad()
def generate_with_algorithm(
    model, prompt, algorithm_name="dual_cache",
    steps=128, gen_length=128, block_length=32, temperature=0.0, 
    remasking='low_confidence', threshold=None, factor=None
):
    """
    Generate using the specified algorithm from the registry.
    
    Args:
        model: LLaDA model
        prompt: Input tensor of shape (1, L)
        algorithm_name: Name of the generation algorithm to use
        steps: Sampling steps
        gen_length: Generated answer length
        block_length: Block length for generation
        temperature: Categorical distribution sampling temperature
        remasking: Remasking strategy ('low_confidence' or 'random')
        threshold: Confidence threshold for parallel decoding
        factor: Factor for dynamic parallel decoding
    """
    algorithm = get_algorithm(algorithm_name)
    if algorithm is None:
        raise RuntimeError(f"Unknown generation algorithm: {algorithm_name}")
    
    if not algorithm.is_available():
        raise RuntimeError(f"Generation algorithm '{algorithm.name}' is not available")
    
    logger.info(f"Using {algorithm.name} generation: {algorithm.description}")
    
    try:
        output, nfe = algorithm.generate(
            model=model,
            prompt=prompt,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            remasking=remasking,
            threshold=threshold,
            factor=factor
        )
        
        logger.info(f"Generation completed with {nfe} forward passes")
        return output
        
    except Exception as e:
        logger.error(f"Generation failed with {algorithm.name}: {e}")
        raise




def load_model_from_hf(model_path: str):
    """Load model from HuggingFace format (supports both LLaDA and Nemotron models)."""
    global model, tokenizer, device, model_config, model_type
    
    # Determine if this is a local path or HuggingFace model name
    is_local_path = os.path.exists(model_path)
    path_type = "local path" if is_local_path else "HuggingFace model name"
    
    # Detect model type based on model path (if not already set by DCP loading)
    if model_type is None:
        if "nemotron" in model_path.lower() or "Nemotron" in model_path:
            model_type = "nemotron"
            logger.info(f"Detected Nemotron model from {path_type}: {model_path}")
        else:
            model_type = "llada"
            logger.info(f"Loading LLaDA model from {path_type}: {model_path}")
    else:
        logger.info(f"Using pre-detected model type '{model_type}' for {path_type}: {model_path}")
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        # Load tokenizer (works with both local paths and HF model names)
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Load model based on detected type
        logger.info("Loading model...")
        if model_type == "nemotron":
            logger.info("Loading Nemotron model with standard AutoModel")
            model = AutoModel.from_pretrained(
                model_path, 
                trust_remote_code=True, 
                torch_dtype=torch.bfloat16
            )
            model = model.to(device).eval()
        else:
            # LLaDA model loading
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
        
        logger.info(f"Model loaded successfully from {path_type}!")
        logger.info(f"Model type: {model_type}")
        if model_type == "llada":
            logger.info(f"Fast-dLLM optimizations: {'enabled' if FAST_DLLM_AVAILABLE else 'disabled'}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to load model from {path_type} '{model_path}': {e}")
        return False


def load_model_from_dcp(dcp_path: str, base_model: str, temp_dir: str = "/tmp/model_hf_converted"):
    """Load model from DCP checkpoint by converting to HF format first (supports both LLaDA and Nemotron models)."""
    global model, tokenizer, device, model_config, model_type
    
    if not NEMO_RL_AVAILABLE:
        logger.error("NeMo-RL is not available. DCP checkpoint loading requires nemo_rl.utils.native_checkpoint.")
        logger.error("For local execution, please:")
        logger.error("1. Set PYTHONPATH to include NeMo-RL: export PYTHONPATH=/path/to/NeMo-RL:$PYTHONPATH")
        logger.error("2. Install NeMo-RL dependencies: uv sync --locked --no-install-project") 
        logger.error("3. Or use a HuggingFace model instead:")
        logger.error("   - LLaDA: --model-path GSAI-ML/LLaDA-8B-Instruct")
        logger.error("   - Nemotron: --model-path nvidia/Nemotron-Diffusion-Research-4B-v0")
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
        
        # Detect model type from base_model parameter (not converted path)
        # This is important because the converted path doesn't contain model type info
        global model_type
        if "nemotron" in base_model.lower() or "Nemotron" in base_model:
            model_type = "nemotron"
            logger.info(f"Detected Nemotron model from base model: {base_model}")
        else:
            model_type = "llada"
            logger.info(f"Detected LLaDA model from base model: {base_model}")
        
        # Now load from HF format
        return load_model_from_hf(hf_path)
        
    except Exception as e:
        logger.error(f"Failed to convert or load DCP checkpoint: {e}")
        return False


async def generate_chat_completion(request: ChatCompletionRequest) -> Union[ChatCompletionResponse, AsyncGenerator]:
    """Generate chat completion using LLaDA model."""
    
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
    extra_fields = {k: v for k, v in request.__dict__.items() if k not in request.model_fields}
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
    
    # Apply chat template or use raw text
    try:
        if hasattr(app.state, 'no_chat_template') and app.state.no_chat_template:
            # Skip chat template - use raw content from last message
            if request.messages:
                formatted_prompt = request.messages[-1].content
            else:
                formatted_prompt = ""
            logger.debug(f"Using raw text (no chat template): {formatted_prompt[:100]}...")
        else:
            # Apply chat template (default behavior)
            formatted_prompt = tokenizer.apply_chat_template(
                [{"role": msg.role, "content": msg.content} for msg in request.messages],
                add_generation_prompt=True,
                tokenize=False
            )
            logger.debug(f"Applied chat template: {formatted_prompt[:100]}...")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to format chat template: {e}")
    
    # Tokenize
    try:
        input_ids = tokenizer(formatted_prompt, return_tensors="pt")['input_ids'].to(device)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to tokenize input: {e}")
    
    # Generate using the appropriate algorithm from registry
    try:
        gen_length = request.max_tokens or 128
        
        # Select algorithm based on model type and request
        if model_type == "nemotron":
            algorithm_name = request.generation_algorithm or "nemotron"
            logger.info(f"Using Nemotron generation with algorithm: {algorithm_name}")
        else:
            algorithm_name = request.generation_algorithm or "dual_cache"
            logger.info(f"Using LLaDA generation with algorithm: {algorithm_name}")
            
            # Ensure block_length compatibility for LLaDA algorithms
            block_length = min(request.block_length, gen_length)
            if gen_length % block_length != 0:
                # Adjust gen_length to be divisible by block_length
                gen_length = ((gen_length // block_length) + 1) * block_length
                logger.info(f"Adjusted gen_length to {gen_length} to be divisible by block_length {block_length}")
        
        output = generate_with_algorithm(
            model=model,
            prompt=input_ids,
            algorithm_name=algorithm_name,
            steps=request.steps,
            gen_length=gen_length,
            block_length=request.block_length,
            temperature=request.temperature,
            remasking=request.remasking,
            threshold=request.threshold,
            factor=request.factor
        )
        
        # Decode generated text (excluding the input prompt)
        generated_text = tokenizer.batch_decode(
            output[:, input_ids.shape[1]:], 
            skip_special_tokens=True
        )[0].strip()
        
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
    if model is None:
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
    """Create chat completion using LLaDA model."""
    
    if model is None:
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
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": model_type,
        "device": device,
        "available_generation_algorithms": available_algorithms,
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
    global model, tokenizer, device, model_config, model_type
    
    parser = argparse.ArgumentParser(description="LLaDA/Nemotron OpenAI API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--model-path", help="Path to HuggingFace model")
    parser.add_argument("--dcp-path", help="Path to DCP checkpoint (supports both LLaDA and Nemotron models)")
    parser.add_argument("--base-model", default="GSAI-ML/LLaDA-8B-Instruct", 
                       help="Base model name for DCP conversion (e.g., GSAI-ML/LLaDA-8B-Instruct, nvidia/Nemotron-Diffusion-Research-4B-v0)")
    parser.add_argument("--temp-dir", default="/tmp/model_hf_converted",
                       help="Temporary directory for DCP conversion")
    parser.add_argument("--no-chat-template", action="store_true",
                       help="Disable chat template application (feed raw text to tokenizer)")
    
    args = parser.parse_args()
    
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
    
    # Store configuration in app state
    app.state.no_chat_template = args.no_chat_template
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    if args.no_chat_template:
        logger.info("Chat template disabled - using raw text input")
    else:
        logger.info("Chat template enabled (default)")  
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
