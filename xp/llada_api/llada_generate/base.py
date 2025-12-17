#!/usr/bin/env python3
"""
Base interface for LLaDA generation algorithms.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import os
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


class GenerationAlgorithm(ABC):
    """Base class for all LLaDA generation algorithms."""
    
    def __init__(self, name: str, description: str, engine: str = "unknown"):
        """
        Initialize generation algorithm.
        
        Args:
            name: Algorithm name (e.g., 'basic', 'dinfer_blockwise')
            description: Human-readable description
            engine: Inference engine (e.g., 'fast-dllm', 'dinfer', 'nemotron', 'hf')
        """
        self.name = name
        self.description = description
        self.engine = engine  # Inference engine: 'fast-dllm', 'dinfer', 'nemotron', 'hf'
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.device: Optional[str] = None
        self.model_config: Optional[Any] = None
        self.model_type: Optional[str] = None  # 'llada' or 'nemotron'
        self.use_chat_template: bool = True  # Whether to use chat template (default: True)
    
    @abstractmethod
    def generate(
        self,
        model: PreTrainedModel,
        prompt: torch.Tensor,
        steps: int,
        gen_length: int,
        block_length: int,
        temperature: float = 1.0,
        remasking: bool = True,
        threshold: float = 0.95,
        factor: float = 1.0,
        **kwargs
    ) -> Tuple[torch.Tensor, int]:
        """
        Generate text using the specific algorithm.
        
        Args:
            model: The LLaDA model
            prompt: Input prompt tensor of shape (batch_size, seq_len)
            steps: Number of diffusion steps
            gen_length: Length of text to generate
            block_length: Block length for generation
            temperature: Sampling temperature
            remasking: Whether to use remasking
            threshold: Threshold parameter for algorithm
            factor: Factor parameter for algorithm
            **kwargs: Additional algorithm-specific parameters
        
        Returns:
            Tuple of (generated_tokens, num_forward_passes)
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this generation algorithm is available (dependencies loaded)."""
        pass
    
    @abstractmethod
    def load_model_class(self, model_path: str, **kwargs) -> PreTrainedModel:
        """
        Load the model class specific to this algorithm.
        Each algorithm can override this to use its own optimized model class.
        
        Args:
            model_path: Path to the model
            **kwargs: Additional arguments for model loading
            
        Returns:
            Loaded model instance
        """
        pass
    
    def load_model_from_hf(self, model_path: str, model_type: Optional[str] = None) -> bool:
        """
        Load model from HuggingFace format.
        
        Args:
            model_path: Path to HuggingFace model (local or remote)
            model_type: Optional model type override ('llada' or 'nemotron')
            
        Returns:
            True if successful, False otherwise
        """
        # Determine if this is a local path or HuggingFace model name
        is_local_path = os.path.exists(model_path)
        path_type = "local path" if is_local_path else "HuggingFace model name"
        
        # Detect model type based on model path (if not already set)
        if model_type is None:
            if "nemotron" in model_path.lower() or "Nemotron" in model_path:
                self.model_type = "nemotron"
                logger.info(f"Detected Nemotron model from {path_type}: {model_path}")
            else:
                self.model_type = "llada"
                logger.info(f"Loading LLaDA model from {path_type}: {model_path}")
        else:
            self.model_type = model_type
            logger.info(f"Using specified model type '{model_type}' for {path_type}: {model_path}")
        
        try:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Using device: {self.device}")
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            # Load model using algorithm-specific loader
            logger.info("Loading model...")
            self.model = self.load_model_class(model_path, torch_dtype=torch.bfloat16)
            self.model = self.model.to(self.device).eval()
            
            # Load config
            logger.info("Loading model config...")
            self.model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            
            logger.info(f"Model loaded successfully from {path_type}!")
            logger.info(f"Model type: {self.model_type}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to load model from {path_type} '{model_path}': {e}")
            return False
    
    def load_model_from_dcp(self, dcp_path: str, base_model: str, temp_dir: str = "/tmp/model_hf_converted") -> bool:
        """
        Load model from DCP checkpoint by converting to HF format first.
        
        Args:
            dcp_path: Path to DCP checkpoint
            base_model: Base model name for conversion
            temp_dir: Temporary directory for conversion
            
        Returns:
            True if successful, False otherwise
        """
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
            logger.error("3. Or use a HuggingFace model instead:")
            logger.error("   - LLaDA: --model-path GSAI-ML/LLaDA-8B-Instruct")
            logger.error("   - Nemotron: --model-path nvidia/Nemotron-Diffusion-Research-4B-v0")
            return False
        
        logger.info(f"Converting DCP checkpoint to HuggingFace format...")
        logger.info(f"DCP path: {dcp_path}")
        logger.info(f"Base model: {base_model}")
        logger.info(f"Temp HF path: {temp_dir}")
        
        # Detect model type from base_model parameter
        if "nemotron" in base_model.lower() or "Nemotron" in base_model:
            model_type = "nemotron"
            logger.info(f"Detected Nemotron model from base model: {base_model}")
        else:
            model_type = "llada"
            logger.info(f"Detected LLaDA model from base model: {base_model}")
        
        try:
            # Check if this is a structured checkpoint
            weights_dir = os.path.join(dcp_path, "weights")
            tokenizer_dir = os.path.join(dcp_path, "tokenizer")
            
            if os.path.exists(weights_dir) and os.path.exists(tokenizer_dir):
                logger.info("Detected structured DCP checkpoint with weights/ and tokenizer/ directories")

                try:
                    logger.info("Attempting to load checkpoint without HF conversion...")
                    res = self.load_model_from_hf(base_model, model_type=model_type)
                    if res:
                        load_checkpoint(
                            model=self.model,
                            weights_path=weights_dir,
                            optimizer=None,
                            scheduler=None,
                            optimizer_path=None,
                        )
                        logger.info("Checkpoint successfully loaded WITHOUT HF conversion!")
                        return True
                    else:
                        raise RuntimeError("Failed to load HF checkpoint")
                except:
                    logger.info("Direct checkpoint loading fail. Falling back to explicit HF conversion")
                    hf_path = convert_structured_dcp_to_hf(
                        dcp_root_path=dcp_path,
                        hf_ckpt_path=temp_dir,
                        model_name_or_path=base_model,
                        overwrite=True
                    )
            else:
                logger.info("Using legacy DCP checkpoint format (direct weights path)")
                hf_path = convert_dcp_to_hf(
                    dcp_ckpt_path=dcp_path,
                    hf_ckpt_path=temp_dir,
                    model_name_or_path=base_model,
                    tokenizer_name_or_path=base_model,
                    overwrite=True
                )
            
            logger.info(f"Conversion completed. Loading from: {hf_path}")
            
            # Load from converted HF format
            return self.load_model_from_hf(hf_path, model_type=model_type)
            
        except Exception as e:
            logger.error(f"Failed to convert or load DCP checkpoint: {e}")
            return False
    
    def tokenize_prompts(
        self, 
        prompts: Union[str, List[str]], 
        apply_chat_template: bool = True,
        messages: Optional[List[List[Dict[str, str]]]] = None
    ) -> torch.Tensor:
        """
        Tokenize prompts with optional chat template application.
        
        Args:
            prompts: Single prompt string or list of prompts
            apply_chat_template: Whether to apply chat template
            messages: Optional list of message lists (for chat template)
            
        Returns:
            Tokenized and padded batch tensor
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load_model_from_hf or load_model_from_dcp first.")
        
        # Convert single prompt to list
        if isinstance(prompts, str):
            prompts = [prompts]
        
        # Apply chat template if requested and messages provided
        if apply_chat_template and messages is not None:
            formatted_prompts = []
            for msg_list in messages:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    msg_list,
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=True
                )
                formatted_prompts.append(formatted_prompt)
            prompts = formatted_prompts
        
        # Tokenize all prompts
        tokenized_prompts = [self.tokenizer(prompt, return_tensors="pt")['input_ids'] for prompt in prompts]
        max_prompt_length = max(p.shape[1] for p in tokenized_prompts)
        
        # Pad prompts to same length
        padded_prompts = []
        for prompt_ids in tokenized_prompts:
            if prompt_ids.shape[1] < max_prompt_length:
                padding = torch.full(
                    (1, max_prompt_length - prompt_ids.shape[1]),
                    self.tokenizer.pad_token_id,
                    dtype=torch.long,
                    device=self.device
                )
                prompt_ids = prompt_ids.to(self.device)
                prompt_ids = torch.cat([padding, prompt_ids], dim=1)
            else:
                prompt_ids = prompt_ids.to(self.device)
            padded_prompts.append(prompt_ids)
        
        # Stack into batch tensor
        batch_input_ids = torch.cat(padded_prompts, dim=0).to(self.device)
        
        return batch_input_ids
    
    def decode_outputs(
        self,
        output_ids: torch.Tensor,
        prompt_lengths: Optional[List[int]] = None,
        skip_special_tokens: bool = True
    ) -> List[str]:
        """
        Decode output token IDs to text.
        
        Args:
            output_ids: Output tensor of shape (batch_size, seq_len)
            prompt_lengths: Optional list of prompt lengths to exclude from output
            skip_special_tokens: Whether to skip special tokens in decoding
            
        Returns:
            List of decoded text strings
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load_model_from_hf or load_model_from_dcp first.")
        
        batch_size = output_ids.shape[0]
        decoded_texts = []
        
        for i in range(batch_size):
            if prompt_lengths is not None and i < len(prompt_lengths):
                # Extract only generated tokens
                generated_tokens = output_ids[i:i+1, prompt_lengths[i]:]
            else:
                generated_tokens = output_ids[i:i+1]
            
            decoded_text = self.tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=skip_special_tokens
            )[0].strip()
            decoded_texts.append(decoded_text)
        
        return decoded_texts
    
    def generate_batch(
        self,
        prompts: Union[str, List[str]],
        apply_chat_template: bool = True,
        messages: Optional[List[List[Dict[str, str]]]] = None,
        steps: int = 128,
        gen_length: int = 128,
        block_length: int = 32,
        temperature: float = 1.0,
        remasking: bool = True,
        threshold: float = 0.95,
        factor: float = 1.0,
        **kwargs
    ) -> Tuple[List[str], int]:
        """
        High-level batch generation with tokenization and decoding.
        
        Args:
            prompts: Single prompt string or list of prompts
            apply_chat_template: Whether to apply chat template
            messages: Optional list of message lists (for chat template)
            steps: Number of diffusion steps
            gen_length: Length of text to generate
            block_length: Block length for generation
            temperature: Sampling temperature
            remasking: Whether to use remasking
            threshold: Threshold parameter
            factor: Factor parameter
            **kwargs: Additional algorithm-specific parameters
            
        Returns:
            Tuple of (generated_texts, num_forward_passes)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model_from_hf or load_model_from_dcp first.")
        
        # Tokenize prompts
        batch_input_ids = self.tokenize_prompts(prompts, apply_chat_template, messages)
        
        # Store prompt lengths for decoding
        prompt_lengths = [batch_input_ids.shape[1]] * batch_input_ids.shape[0]
        
        # Generate
        output_ids, nfe = self.generate(
            model=self.model,
            prompt=batch_input_ids,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            remasking=remasking,
            threshold=threshold,
            factor=factor,
            **kwargs
        )
        
        # Decode outputs
        generated_texts = self.decode_outputs(output_ids, prompt_lengths)
        
        return generated_texts, nfe
    
    def get_required_args(self) -> Dict[str, Any]:
        """Get the required arguments and their default values for this algorithm."""
        return {
            'steps': 16,
            'gen_length': 128,
            'block_length': 32,
            'temperature': 1.0,
            'remasking': True,
            'threshold': 0.95,
            'factor': 1.0
        }
    
    def validate_args(self, **kwargs) -> Dict[str, Any]:
        """Validate and set default values for generation arguments."""
        required_args = self.get_required_args()
        validated_args = {}
        
        for key, default_value in required_args.items():
            validated_args[key] = kwargs.get(key, default_value)
        
        # Ensure gen_length is divisible by block_length
        gen_length = validated_args['gen_length']
        block_length = validated_args['block_length']
        
        if gen_length % block_length != 0:
            validated_args['gen_length'] = ((gen_length // block_length) + 1) * block_length
        
        return validated_args
    
    def prepare_prompts(self, messages_list: List[List[Dict[str, str]]]) -> List[Optional[str]]:
        """
        Prepare prompts from messages based on use_chat_template setting.
        
        Args:
            messages_list: List of message lists, where each message list contains
                          dicts with 'role' and 'content' keys
        
        Returns:
            List of prepared prompts (or None if using chat template directly)
        """
        if self.use_chat_template:
            # Return None - will apply chat template during tokenization
            return [None] * len(messages_list)
        else:
            # Extract raw text from last message in each conversation
            prompts = []
            for messages in messages_list:
                if messages:
                    # Use the content from the last message
                    prompts.append(messages[-1]['content'])
                else:
                    prompts.append("")
            return prompts
    
    def tokenize_batch(
        self, 
        messages_list: List[List[Dict[str, str]]]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Tokenize a batch of messages using the appropriate method.
        
        This is a unified interface that handles both chat template and raw text modes.
        Subclasses can override for engine-specific behavior (e.g., dInfer left-padding).
        
        Args:
            messages_list: List of message lists to tokenize
        
        Returns:
            Either input_ids tensor (for Fast-dLLM/Nemotron) or 
            tuple of (input_ids, attention_mask) for dInfer
        """
        # Default implementation: use tokenize_prompts for standard algorithms
        if self.use_chat_template:
            input_ids = self.tokenize_prompts(
                prompts=[],
                apply_chat_template=True,
                messages=messages_list
            )
        else:
            # Prepare raw prompts
            prompts = self.prepare_prompts(messages_list)
            input_ids = self.tokenize_prompts(
                prompts=prompts,
                apply_chat_template=False,
                messages=None
            )
        
        return input_ids
