#!/usr/bin/env python3
"""
OpenAI-compatible API server for LLaDA models with DCP checkpoint support.

This server provides OpenAI API compatibility for LLaDA diffusion language models,
supporting both direct HuggingFace model loading and DCP checkpoint conversion.

Usage:
    python llada_openai_server.py --model-path /path/to/checkpoint

For DCP checkpoints:
    python llada_openai_server.py --dcp-path /path/to/dcp --base-model GSAI-ML/LLaDA-8B-Instruct
"""

import argparse
import asyncio
import json
import logging
import os
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

# Import NeMo-RL utilities for DCP handling (optional for local mode)
try:
    from nemo_rl.utils.native_checkpoint import convert_dcp_to_hf, load_checkpoint
    NEMO_RL_AVAILABLE = True
except ImportError:
    NEMO_RL_AVAILABLE = False
    convert_dcp_to_hf = None
    load_checkpoint = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variables
model = None
tokenizer = None
device = None
model_config = None

# LLaDA specific constants
MASK_ID = 126336  # The token id of [MASK] in LLaDA tokenizer


# OpenAI API Models
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="llada-8b-instruct")
    messages: List[ChatMessage] = Field(..., description="List of messages")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=128, gt=0)
    stream: bool = Field(default=False)
    # LLaDA specific parameters
    steps: int = Field(default=64, ge=1, le=512, description="Diffusion steps")
    block_length: int = Field(default=64, ge=1, description="Block length for generation")
    cfg_scale: float = Field(default=0.0, ge=0.0, description="Classifier-free guidance scale")
    remasking: str = Field(default="low_confidence", description="Remasking strategy")


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


# LLaDA Generation Functions
def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves 
    perplexity score but reduces generation quality. Thus, we use float64.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule, the expected number of 
    tokens transitioned at each step should be consistent.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
    ) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@torch.no_grad()
def generate_llada(
    model, prompt, steps=64, gen_length=128, block_length=64, 
    temperature=0.0, cfg_scale=0.0, remasking='low_confidence', mask_id=MASK_ID
):
    """
    LLaDA diffusion generation function.
    
    Args:
        model: Mask predictor model
        prompt: Input tensor of shape (1, L)
        steps: Sampling steps
        gen_length: Generated answer length
        block_length: Block length for semi-autoregressive generation
        temperature: Categorical distribution sampling temperature
        cfg_scale: Classifier-free guidance scale
        remasking: Remasking strategy ('low_confidence' or 'random')
        mask_id: The token id of [MASK]
    """
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    # Ensure block_length divides gen_length evenly
    if gen_length % block_length != 0:
        gen_length = ((gen_length // block_length) + 1) * block_length
        x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
        x[:, :prompt.shape[1]] = prompt.clone()

    num_blocks = gen_length // block_length
    if steps % num_blocks != 0:
        steps = ((steps // num_blocks) + 1) * num_blocks
    
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        
        for i in range(steps_per_block):
            mask_index = (x == mask_id)
            
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == 'low_confidence':
                import torch.nn.functional as F
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(f"Remasking strategy '{remasking}' not implemented")

            x0_p[:, block_end:] = float('-inf')

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, float('-inf'))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                if num_transfer_tokens[j, i] > 0:
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


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
        
        # Load model (works with both local paths and HF model names)
        logger.info("Loading model...")
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
        return True
    except Exception as e:
        logger.error(f"Failed to load model from {model_type} '{model_path}': {e}")
        return False


def load_model_from_dcp(dcp_path: str, base_model: str, temp_dir: str = "/tmp/llada_hf_converted"):
    """Load LLaDA model from DCP checkpoint by converting to HF format first."""
    global model, tokenizer, device, model_config
    
    if not NEMO_RL_AVAILABLE:
        logger.error("NeMo-RL is not available. DCP checkpoint loading requires nemo_rl.utils.native_checkpoint.")
        logger.error("For local execution, please:")
        logger.error("1. Set PYTHONPATH to include NeMo-RL: export PYTHONPATH=/path/to/NeMo-RL:$PYTHONPATH")
        logger.error("2. Install NeMo-RL dependencies: uv sync --locked --extra vllm --no-install-project") 
        logger.error("3. Or use a HuggingFace model instead: --model-path GSAI-ML/LLaDA-8B-Instruct")
        return False
    
    logger.info(f"Converting DCP checkpoint to HuggingFace format...")
    logger.info(f"DCP path: {dcp_path}")
    logger.info(f"Base model: {base_model}")
    logger.info(f"Temp HF path: {temp_dir}")
    
    try:
        # Convert DCP to HF format
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


async def generate_chat_completion(request: ChatCompletionRequest) -> Union[ChatCompletionResponse, AsyncGenerator]:
    """Generate chat completion using LLaDA model."""
    
    # Format messages into prompt
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty")
    
    # Apply chat template
    try:
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": msg.role, "content": msg.content} for msg in request.messages],
            add_generation_prompt=True,
            tokenize=False
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to format chat template: {e}")
    
    # Tokenize
    try:
        input_ids = tokenizer(formatted_prompt, return_tensors="pt")['input_ids'].to(device)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to tokenize input: {e}")
    
    # Generate with LLaDA
    try:
        gen_length = request.max_tokens or 128
        output = generate_llada(
            model=model,
            prompt=input_ids,
            steps=request.steps,
            gen_length=gen_length,
            block_length=min(request.block_length, gen_length),
            temperature=request.temperature,
            cfg_scale=request.cfg_scale,
            remasking=request.remasking,
        )
        
        # Decode generated text (excluding the input prompt)
        generated_text = tokenizer.batch_decode(
            output[:, input_ids.shape[1]:], 
            skip_special_tokens=True
        )[0].strip()
        
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
    title="LLaDA OpenAI API",
    description="OpenAI-compatible API for LLaDA diffusion language models",
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
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device
    }


def main():
    parser = argparse.ArgumentParser(description="LLaDA OpenAI API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--model-path", help="Path to HuggingFace model")
    parser.add_argument("--dcp-path", help="Path to DCP checkpoint")
    parser.add_argument("--base-model", default="GSAI-ML/LLaDA-8B-Instruct", 
                       help="Base model name for DCP conversion")
    parser.add_argument("--temp-dir", default="/tmp/llada_hf_converted",
                       help="Temporary directory for DCP conversion")
    
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
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
