#!/usr/bin/env python3
"""
Base class for dInfer generation algorithms.

This module provides the common base class for all dInfer-based generation algorithms.
dInfer uses a different architecture than Fast-dLLM, wrapping the model with a
BlockWiseDiffusionLLM that handles the diffusion process.
"""

import logging
from abc import abstractmethod
from typing import Optional, Tuple
import torch
from transformers import PreTrainedModel

from ..base import GenerationAlgorithm
from ._imports import LLaDAModelLM, DINFER_AVAILABLE, MASK_ID, EOS_ID

logger = logging.getLogger(__name__)


class DInferGeneration(GenerationAlgorithm):
    """
    Base class for dInfer generation algorithms.
    
    Engine: 'dinfer'
    
    dInfer uses a different architecture:
    - Model is wrapped in a BlockWiseDiffusionLLM (or variant)
    - Decoder handles parallel decoding strategies
    - Iterator factory manages diffusion iterations
    - Cache factory handles KV-cache management
    
    Subclasses should implement:
    - create_diffusion_llm(): Create the specific dInfer diffusion LLM wrapper
    - is_available(): Check if their specific components are available
    """
    
    def __init__(self, name: str, description: str):
        super().__init__(name, description, engine="dinfer")
        self.diffusion_llm = None  # Will be created after model loading
    
    def load_model_class(self, model_path: str, **kwargs) -> PreTrainedModel:
        """
        Load model class optimized for dInfer.
        
        Uses dInfer's LLaDAModelLM for optimal performance.
        
        Args:
            model_path: Path to the model (local or HuggingFace)
            **kwargs: Additional arguments for model loading
            
        Returns:
            Loaded model instance
        """
        if DINFER_AVAILABLE and LLaDAModelLM is not None:
            logger.info("Loading model with dInfer optimized LLaDAModelLM class")
            return LLaDAModelLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                **kwargs
            )
        else:
            raise RuntimeError(
                "dInfer is not available. Please install dInfer or use Fast-dLLM algorithms instead."
            )
    
    def load_model_from_hf(self, model_path: str, model_type: Optional[str] = None) -> bool:
        """
        Load model from HuggingFace format and create diffusion LLM wrapper.
        
        Extends parent method to also create the dInfer diffusion LLM wrapper.
        """
        # Call parent to load model, tokenizer, config
        success = super().load_model_from_hf(model_path, model_type)
        
        if success:
            # Ensure tokenizer has pad token for batching
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                logger.info("Set pad_token to eos_token for batch padding")
            
            # Create the diffusion LLM wrapper
            try:
                self.diffusion_llm = self.create_diffusion_llm()
                logger.info(f"Created dInfer diffusion LLM wrapper: {type(self.diffusion_llm).__name__}")
            except Exception as e:
                logger.error(f"Failed to create diffusion LLM wrapper: {e}")
                return False
        
        return success
    
    @abstractmethod
    def create_diffusion_llm(self):
        """
        Create the dInfer diffusion LLM wrapper.
        
        This should instantiate the appropriate dInfer class (e.g., BlockWiseDiffusionLLM)
        with the loaded model and configured decoder.
        
        Returns:
            dInfer diffusion LLM instance
        """
        pass
    
    def tokenize_prompts_dinfer(
        self, 
        prompts: list[str], 
        apply_chat_template: bool = True,
        messages: Optional[list[list[dict[str, str]]]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize prompts with left-padding for dInfer batch generation.
        
        dInfer requires left-padding for efficient batch processing.
        This is a dInfer-specific method that returns attention_mask.
        
        Args:
            prompts: List of prompt strings
            apply_chat_template: Whether to apply chat template
            messages: Optional list of message lists (for chat template)
            
        Returns:
            Tuple of (input_ids, attention_mask)
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
                    tokenize=False
                )
                formatted_prompts.append(formatted_prompt)
            prompts = formatted_prompts
        
        # Set padding side to left for decoder-only models
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = 'left'
        
        # Tokenize with left-padding
        tokenized = self.tokenizer(
            prompts,
            padding='longest',
            padding_side='left',
            return_tensors='pt',
            truncation=True,
            max_length=2048
        )
        
        # Restore original padding side
        self.tokenizer.padding_side = original_padding_side
        
        input_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        return input_ids, attention_mask
    
    def decode_outputs_dinfer(
        self,
        output_ids: torch.Tensor,
        input_ids: torch.Tensor,
        skip_special_tokens: bool = True
    ) -> list[str]:
        """
        Decode output token IDs to text.
        
        For dInfer, we need to exclude padding and prompt tokens.
        
        Args:
            output_ids: Output tensor of shape (batch_size, seq_len)
            input_ids: Input tensor (to calculate prompt lengths)
            skip_special_tokens: Whether to skip special tokens in decoding
            
        Returns:
            List of decoded text strings
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded.")
        
        batch_size = output_ids.shape[0]
        decoded_texts = []
        
        for i in range(batch_size):
            # Calculate actual prompt length (excluding padding)
            prompt_length = (input_ids[i] != self.tokenizer.pad_token_id).sum().item()
            
            # Extract only generated tokens (after the prompt)
            generated_tokens = output_ids[i:i+1, prompt_length:]
            
            decoded_text = self.tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=skip_special_tokens
            )[0].strip()
            decoded_texts.append(decoded_text)
        
        return decoded_texts
    
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
        Generate text using dInfer diffusion LLM.
        
        Note: dInfer doesn't use 'steps' in the same way as Fast-dLLM.
        The diffusion iterations are managed by the iterator factory.
        
        Args:
            model: The model (not used directly, diffusion_llm wraps it)
            prompt: Input prompt tensor of shape (batch_size, seq_len)
            steps: Number of diffusion steps (not used by dInfer in the same way)
            gen_length: Length of text to generate
            block_length: Block length for generation
            temperature: Sampling temperature (handled by decoder)
            remasking: Not used by dInfer
            threshold: Threshold for parallel decoding (handled by decoder)
            factor: Not used by dInfer
            **kwargs: Additional algorithm-specific parameters
        
        Returns:
            Tuple of (generated_tokens, num_forward_passes)
        """
        if self.diffusion_llm is None:
            raise RuntimeError("Diffusion LLM not created. Call load_model_from_hf first.")
        
        validated_args = self.validate_args(
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            remasking=remasking,
            threshold=threshold,
            factor=factor,
            **kwargs
        )
        
        logger.debug(f"Using dInfer generation with args: {validated_args}")
        
        # Generate using dInfer
        with torch.no_grad():
            output_ids = self.diffusion_llm.generate(
                prompt=prompt,
                gen_length=validated_args['gen_length'],
                block_length=validated_args['block_length']
            )
        
        # dInfer doesn't return NFE directly, estimate it
        # For now, we'll use a placeholder (can be improved later)
        nfe = -1  # Placeholder - dInfer doesn't expose this
        
        return output_ids, nfe
    
    def get_required_args(self):
        """Get the required arguments with dInfer-specific defaults."""
        return {
            'steps': 128,  # Not used by dInfer but kept for compatibility
            'gen_length': 256,
            'block_length': 64,
            'temperature': 1.0,
            'remasking': True,  # Not used by dInfer
            'threshold': 0.9,
            'factor': 1.0,  # Not used by dInfer
        }
    
    def tokenize_batch(self, messages_list):
        """
        dInfer-specific batch tokenization with left-padding.
        
        Returns:
            Tuple of (input_ids, attention_mask) - dInfer needs attention mask
        """
        if self.use_chat_template:
            input_ids, attention_mask = self.tokenize_prompts_dinfer(
                prompts=[],
                apply_chat_template=True,
                messages=messages_list
            )
        else:
            # Prepare raw prompts
            prompts = self.prepare_prompts(messages_list)
            input_ids, attention_mask = self.tokenize_prompts_dinfer(
                prompts=prompts,
                apply_chat_template=False,
                messages=None
            )
        
        return input_ids, attention_mask


__all__ = [
    'DINFER_AVAILABLE',
    'LLaDAModelLM',
    'BlockWiseDiffusionLLM',
    'SlidingWindowDiffusionLLM',
    'BlockWiseDiffusionLLMWithSP',
    'ThresholdParallelDecoder',
    'CreditThresholdParallelDecoder',
    'HierarchyDecoder',
    'BlockIteratorFactory',
    'KVCacheFactory',
]
