import sys
import os

# Disable torch compilation to avoid backend compiler errors
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'

import torch
# Disable torch.compile globally
torch._dynamo.config.disable = True

import numpy as np
import torch.nn.functional as F
import time
import logging
from transformers import AutoTokenizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add dInfer path
DINFER_PATH = os.path.abspath('3rdparty/dInfer/python')
if os.path.exists(DINFER_PATH) and DINFER_PATH not in sys.path:
    sys.path.insert(0, DINFER_PATH)
    logger.info(f"Added dInfer path: {DINFER_PATH}")

from dinfer.model import LLaDAModelLM
from dinfer.decoding.parallel_strategy import (
    ParallelDecoder,
    get_num_transfer_tokens,
    get_transfer_index,
)
from dinfer import (
    BlockWiseDiffusionLLM,
    BlockIteratorFactory,
    KVCacheFactory,
)
from dinfer.decoding.utils import TokenArray

# Constants
MASK_ID = 126336
EOS_ID = 126081

class _FixedParallelDecoder(ParallelDecoder):
    """ This decoder decodes tokens in a fixed number of steps.
    """
    def __init__(self, temperature, steps, remasking='low_confidence', mask_id=MASK_ID, eos_id=EOS_ID):
        super().__init__(temperature, remasking, mask_id)
        self.steps = steps
        self.iter = 0
        self.eos_id = eos_id

    def block_init(self, block_x, block_id):
        # TODO(zhengda) we need to handle steps correctly here when the distributed version changes the gen length.
        block_mask_index = block_x == self.mask_id
        self.num_transfer_tokens = get_num_transfer_tokens(block_mask_index, self.steps)
        self.iter = 0

    def decode(self, logits, block_start, block_end, x, iter_threshold = None):
        """ Decode the logits in a block.
        """
        mask_index = (x[block_start:block_end] == self.mask_id)
        assert mask_index.shape[1] == logits.shape[1]

        curr_x = x[block_start:block_end]
        x0, transfer_index = get_transfer_index(logits, self.temperature, self.remasking, mask_index, curr_x, self.num_transfer_tokens[:, self.iter], None)
        self.iter += 1
        x[block_start:block_end][transfer_index] = x0[transfer_index]

class BlockWiseSoftTokenLLM:
    def __init__(self, model, decoder, iterator_factory, early_stop=True, cache_factory=None, maximum_unroll=4, expected_tpf=8, soft_token_ratio=0.2, treat_soft_tokens_as_candidates=False):
        self.model = model
        self.cache_factory = cache_factory
        self.decoder = decoder
        self.iterator_factory = iterator_factory
        self.num_forwards = 0
        self.cache_updates = 0
        self.early_stop = early_stop
        self.maximum_unroll = maximum_unroll
        self.expected_tpf = expected_tpf
        self.soft_token_ratio = soft_token_ratio
        self.treat_soft_tokens_as_candidates = treat_soft_tokens_as_candidates
        self.input_embeddings = self.model.get_input_embeddings()

    def validate_schedule(self, block_length):
        """ Validates that the decoding schedule can be satisfied with the given soft token ratio.
        """
        if not hasattr(self.decoder, 'steps') or self.treat_soft_tokens_as_candidates:
            return

        steps = self.decoder.steps
        current_masks = block_length
        
        # Calculate the schedule for a full block
        base = current_masks // steps
        remainder = current_masks % steps
        
        schedule = []
        for i in range(steps):
            count = base + (1 if i < remainder else 0)
            schedule.append(count)
            
        # Simulate decoding
        for step_idx, num_to_decode in enumerate(schedule):
            num_soft = int(current_masks * self.soft_token_ratio)
            available = current_masks - num_soft
            
            if available < num_to_decode:
                raise ValueError(
                    f"Decoding Schedule Violation: Step {step_idx} requires decoding {num_to_decode} tokens, "
                    f"but only {available} masks are available ({current_masks} total - {num_soft} soft tokens). "
                    f"Reduce soft_token_ratio or enable treat_soft_tokens_as_candidates."
                )
            current_masks -= num_to_decode
        
    @torch.no_grad()
    def generate(self, prompt, gen_length=128, block_length=128):
        ''' Generate tokens with diffusion iterations block by block using Soft Token Sampling.
        '''
        self.validate_schedule(block_length)

        x = TokenArray(prompt, gen_length, self.decoder.mask_id, self.decoder.eos_id, self.model.device)
        it = self.iterator_factory.create(x, block_length)

        iter_no = 0
        kv_cache = self.cache_factory.create() if self.cache_factory is not None else None
        
        for block_id, (block_loc, block) in enumerate(it):
            self.decoder.block_init(block, block_id)

            while (block == self.decoder.mask_id).sum() > 0:
                
                # Pre-check: Ensure we can satisfy the soft token ratio without violating the decoding schedule
                # if we choose to exclude soft tokens from candidacy.
                current_masks = (x[block_loc.start:block_loc.end] == self.decoder.mask_id).sum().item()
                num_soft = int(current_masks * self.soft_token_ratio)
                
                # The decoder wants to transfer (decode) N tokens this step
                num_to_decode = self.decoder.num_transfer_tokens[0, self.decoder.iter].item()
                
                if not self.treat_soft_tokens_as_candidates:
                    # If soft tokens CANNOT be decoded, we must have enough pure masks left to satisfy decoder demand
                    available_for_decoding = current_masks - num_soft
                    if available_for_decoding < num_to_decode:
                        raise ValueError(
                            f"Decoding Schedule Violation: Step {self.decoder.iter} requires decoding {num_to_decode} tokens, "
                            f"but only {available_for_decoding} masks are available ({current_masks} total - {num_soft} soft tokens). "
                            f"Reduce soft_token_ratio or enable treat_soft_tokens_as_candidates."
                        )
                
                # Helper to run model with correct context and embeddings
                def run_model(use_input_embeds=None):
                    # Determine input context based on cache type
                    if kv_cache is None:
                        # Full context (no cache)
                        if use_input_embeds is not None:
                            logits = self.model(inputs_embeds=use_input_embeds).logits
                        else:
                            logits = self.model(x.data).logits
                        return logits[:, block_loc.start:block_loc.end]
                        
                    elif kv_cache.cache_type == 'prefix':
                        # Prefix Cache: past_key_values contains context up to block_start
                        past_key_values, replace_position = kv_cache.get_key_values(block_loc.start, block_loc.end)
                        
                        if use_input_embeds is not None:
                            # Input embeddings should correspond to x[block_loc.start:]
                            logits = self.model(inputs_embeds=use_input_embeds, past_key_values=past_key_values, use_cache=True,
                                              replace_position=replace_position).logits
                        else:
                            logits = self.model(x[block_loc.start:], past_key_values=past_key_values, use_cache=True,
                                              replace_position=replace_position).logits
                        
                        curr_len = block_loc.end - block_loc.start
                        return logits[:, :curr_len]

                    else:
                        # Dual/Sliding Cache: typically uses block context
                        past_key_values, replace_position = kv_cache.get_key_values(block_loc.start, block_loc.end)
                        
                        if use_input_embeds is not None:
                             logits = self.model(inputs_embeds=use_input_embeds, past_key_values=past_key_values, use_cache=True,
                                              replace_position=replace_position).logits
                        else:
                             logits = self.model(block, past_key_values=past_key_values, use_cache=True,
                                              replace_position=replace_position).logits
                        return logits

                # 1. Handle KV Cache Update (Initial step for block or periodically)
                if kv_cache is not None and kv_cache.require_update(iter_no, block_loc.start, block_loc.end):
                    output = self.model(x.data, use_cache=True)
                    self.num_forwards += 1
                    
                    # Update cache
                    kv_cache.update(output.past_key_values)
                    self.cache_updates += 1
                    
                    # Decode using these initial logits
                    self.decoder.decode(output.logits[:, block_loc.start:block_loc.end], block_loc.start, block_loc.end, x)
                    iter_no += 1
                    continue

                # 2. Pass 1: Standard Logits (with current masks)
                logits1 = run_model(use_input_embeds=None)
                self.num_forwards += 1
                
                decoding_logits = logits1
                soft_indices = None
                
                # 3. Soft Token Logic
                # Identify masks in the current block
                curr_block_ids = x[block_loc.start:block_loc.end]
                mask_mask = (curr_block_ids == self.decoder.mask_id)
                mask_indices = torch.nonzero(mask_mask).flatten() # Indices relative to block start
                
                if mask_indices.numel() > 0 and self.soft_token_ratio > 0:
                    if num_soft > 0:
                        perm = torch.randperm(mask_indices.numel(), device=self.model.device)
                        soft_indices = mask_indices[perm[:num_soft]] # Indices relative to block start
                        
                        # Extract logits for these positions
                        # logits1 shape: [1, block_len, vocab]
                        selected_logits = logits1[0, soft_indices]
                        probs = torch.softmax(selected_logits, dim=-1)
                        
                        # Compute Soft Embeddings: Weighted average of token embeddings
                        # [num_soft, vocab] @ [vocab, d_model] -> [num_soft, d_model]
                        soft_embeds = torch.matmul(probs, self.input_embeddings.weight)
                        
                        # Prepare Input Embeddings
                        target_ids = None
                        global_offset = 0
                        
                        if kv_cache is None:
                            target_ids = x.data
                            global_offset = block_loc.start # Offset in target_ids
                        elif kv_cache.cache_type == 'prefix':
                            target_ids = x[block_loc.start:]
                            global_offset = 0 # relative to start of target_ids
                        else:
                            target_ids = curr_block_ids
                            global_offset = 0
                        
                        # Get base embeddings for the input context
                        inputs_embeds = self.input_embeddings(target_ids).clone() # [1, len, d_model]
                        
                        # Replace masks with soft embeddings
                        inputs_embeds[0, global_offset + soft_indices] = soft_embeds
                        
                        # Pass 2: Get logits with Soft Tokens
                        logits2 = run_model(use_input_embeds=inputs_embeds)
                        self.num_forwards += 1
                        decoding_logits = logits2

                # 4. Decode using the latest logits
                # IMPORTANT: If treat_soft_tokens_as_candidates is False, we must mask out the logits
                # for soft tokens so they are not selected by the decoder (which picks top-k confidence usually)
                # However, FixedParallelDecoder usually relies on 'transfer_index' which is determined by confidence.
                # We can force the logits of soft tokens to be very low confidence (high entropy) or -inf
                # but FixedParallelDecoder logic for selection is 'get_transfer_index'.
                
                if not self.treat_soft_tokens_as_candidates and soft_indices is not None and soft_indices.numel() > 0:
                    # We want to prevent these indices from being selected.
                    # The decoder selects based on confidence. We can artificially lower the confidence 
                    # of soft tokens to -inf (or uniform distribution) so they are last in line.
                    # But we can't modify logits in-place if they are used for next step, so clone or careful modify.
                    
                    # Actually, simpler: mask them out in the 'mask_index' passed to get_transfer_index inside decode?
                    # But we can't change the decoder code easily from here without subclassing.
                    # Best approach: Modify the logits passed to decode() so that soft token positions look like garbage.
                    
                    # Set logits for soft tokens to a uniform distribution (max entropy -> min confidence)
                    # or set them to -inf if we want to be sure.
                    # But FixedParallelDecoder calculates confidence. 
                    
                    # Let's just zero out their logits to make them uniform -> low confidence
                    # decoding_logits is [1, len, vocab]
                    # We have soft_indices relative to block start.
                    
                    # Create a mask for soft tokens
                    # Set their logits to 0 (uniform probability after softmax, low max-prob)
                    decoding_logits_modified = decoding_logits.clone()
                    decoding_logits_modified[0, soft_indices] = -1000.0 # Effectively zero probability for all tokens
                    
                    self.decoder.decode(decoding_logits_modified, block_loc.start, block_loc.end, x)
                else:
                    self.decoder.decode(decoding_logits, block_loc.start, block_loc.end, x)
                    
                iter_no += 1

            # Early stop at EOS
            if self.early_stop and torch.any(x[block_loc.start:block_loc.end] == self.decoder.eos_id):
                x[block_loc.end:] = self.decoder.eos_id
                break

        logger.info(f'SoftTokenLLM - Total diffusion iterations: {self.num_forwards}')
        return x.get_generated_tokens()

def main():
    MODEL_PATH = "GSAI-ML/LLaDA-8B-Instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Loading model from {MODEL_PATH} on {device}...")
    model = LLaDAModelLM.from_pretrained(
        MODEL_PATH, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16, 
        init_device=str(device)
    ).eval().to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    # Configuration
    generation_config = {
        "gen_length": 128,
        "steps": 32,
        "block_length": 128,
        "cache_type": "dual",
        "soft_token_ratio": 0.2,
        "treat_soft_tokens_as_candidates": False
    }
    
    # Setup components
    iterator_factory = BlockIteratorFactory(True)
    cache_factory = KVCacheFactory(generation_config["cache_type"])
    
    prompt = "The future of AI involves"
    logger.info(f"Prompt: {prompt}")
    
    # ---------------------------------------------------------
    # 1. Baseline: BlockWiseDiffusionLLM with FixedParallelDecoder
    # ---------------------------------------------------------
    logger.info("-" * 50)
    logger.info("Running Baseline (Standard FixedParallelDecoder)...")
    
    decoder_baseline = _FixedParallelDecoder(
        temperature=0, 
        steps=generation_config["steps"],
        mask_id=MASK_ID,
        eos_id=EOS_ID
    )
    
    baseline_llm = BlockWiseDiffusionLLM(
        model=model,
        decoder=decoder_baseline,
        iterator_factory=iterator_factory,
        cache_factory=cache_factory,
        early_stop=True
    )
    
    start_time = time.time()
    out_baseline = baseline_llm.generate(
        tokenizer(prompt, return_tensors="pt")['input_ids'].to(device),
        gen_length=generation_config["gen_length"],
        block_length=generation_config["block_length"]
    )
    time_baseline = time.time() - start_time
    
    text_baseline = tokenizer.decode(out_baseline[0], skip_special_tokens=True)
    logger.info(f"Baseline Output: {text_baseline}")
    logger.info(f"Baseline Time: {time_baseline:.2f}s, Forwards: {baseline_llm.num_forwards}")
    
    # ---------------------------------------------------------
    # 2. New Method: BlockWiseSoftTokenLLM with FixedParallelDecoder
    # ---------------------------------------------------------
    logger.info("-" * 50)
    logger.info(f"Running SoftToken Method (Ratio={generation_config['soft_token_ratio']})...")
    
    decoder_soft = _FixedParallelDecoder(
        temperature=0, 
        steps=generation_config["steps"],
        mask_id=MASK_ID,
        eos_id=EOS_ID
    )
    
    soft_llm = BlockWiseSoftTokenLLM(
        model=model,
        decoder=decoder_soft,
        iterator_factory=iterator_factory,
        cache_factory=cache_factory,
        early_stop=True,
        soft_token_ratio=generation_config["soft_token_ratio"],
        treat_soft_tokens_as_candidates=generation_config["treat_soft_tokens_as_candidates"]
    )
    
    start_time = time.time()
    out_soft = soft_llm.generate(
        tokenizer(prompt, return_tensors="pt")['input_ids'].to(device),
        gen_length=generation_config["gen_length"],
        block_length=generation_config["block_length"]
    )
    time_soft = time.time() - start_time
    
    text_soft = tokenizer.decode(out_soft[0], skip_special_tokens=True)
    logger.info(f"SoftToken Output: {text_soft}")
    logger.info(f"SoftToken Time: {time_soft:.2f}s, Forwards: {soft_llm.num_forwards}")

if __name__ == "__main__":
    main()

