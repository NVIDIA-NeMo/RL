import copy
from typing import Callable, Optional, Tuple, Union
import random

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

from transformers.processing_utils import Unpack

from transformers.cache_utils import Cache, DynamicCache

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.generation import GenerationMixin

import math

# Import from multi-batch model (for batch_size > 1)
# Using relative imports within models/MB/ package
from .modeling_qwen3_MB import (
    Qwen3InfModel,
    Qwen3InfPreTrainedModel,
    Qwen3InfAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)
# Import config from root model directory (model_path is added to sys.path in chat.py)
from configuration_nvrdiff import NVRDiffConfig

# @torch.compile(dynamic=True, mode="reduce-overhead")
@torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs", dynamic=False)
def fused_flex_attention(q, k, v, block_mask=None):
    return flex_attention(q, k, v, block_mask=block_mask)

# with reference to https://github.com/pytorch-labs/attention-gym/blob/main/examples/flex_attn.ipynb
class Qwen3InfFlexAttention(Qwen3InfAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
 
        self.max_seq_length = self.config.seq_length
        self.prefix_len_orig = int(self.config.seq_length * self.config.prefix_ratio)
        self.block_size_orig = self.config.block_size

        # Remove mask cache: rely on precomputed masks and on-demand builds
        self._mask_cache_key = None
        self._mask_cache_val = None

        if self.config.dlm_paradigm == 'bidirectional':
            self.bidirectional_mask = self.compute_block_mask(mode='bidirectional')
        elif self.config.dlm_paradigm == 'prefix_bidirectional':
            self.prefix_bidirectional_mask = self.compute_block_mask(mode='prefix_bidirectional', prefix_len=self.prefix_len_orig)
        elif self.config.dlm_paradigm == 'efficient_block_diff':
            self.efficient_block_diff_mask = self.compute_block_mask(mode='efficient_block_diff', block_size=self.block_size_orig)
        elif self.config.dlm_paradigm == 'block_diff':
            self.block_diff_mask = self.compute_block_mask(mode='block_diff', block_size=self.block_size_orig)
        else:
            raise ValueError(f"Unknown attention mode: {self.config.dlm_paradigm}")

        self.prefix_len = self.prefix_len_orig
        self.block_size = self.block_size_orig
        self.mode = 'bidirectional'

        import torch._dynamo.config as dcfg
        dcfg.cache_size_limit = 512


    def set_attention_mode(self, mode, prefix_len=None, block_size=None):
        self.mode = mode
        self.prefix_len = prefix_len
        self.block_size = block_size


    def compute_block_mask(self, mode, prefix_len=None, q_len=None, block_size=None):

        def bidirectional_mask(b, h, q, kv): 
            return (q >= kv) | (q < kv)
        
        def prefix_bidirectional_mask(prefix_len, b, h, q, kv):
            return (kv <= prefix_len) | (q >= prefix_len)

        def efficient_block_diff_mask(block_size, b, h, q, kv):
            return (q // block_size) >= (kv // block_size)

        def block_diff_mask(block_size, b, h, q_idx, kv_idx, n):
            """
            Constructs the specialized block diffusion attention mask for training
            composed of three masks:
            - **Block Diagonal Mask (M_BD)**: Self-attention within noised blocks
            - **Offset Block Causal Mask (M_OBC)**: Cross-attention for conditional context
            - **Block Causal Mask (M_BC)**: Attention to update x0

            Args:
                b, h: Batch and head indices (ignored for mask logic).
                q_idx, kv_idx: Query and Key indices.
                seq_len: Total sequence length.
                block_size: Defines the block structure.

            Returns:
                A boolean attention mask.
            """

            # Indicate whether token belongs to xt or x0
            x0_flag_q = (q_idx >= n)
            x0_flag_kv = (kv_idx >= n)

            # Compute block indices
            block_q = torch.where(x0_flag_q == 1,
                                    (q_idx - n) // block_size,
                                    q_idx // block_size)
            block_kv = torch.where(x0_flag_kv == 1,
                                    (kv_idx - n) // block_size,
                                    kv_idx // block_size)

            # **1. Block Diagonal Mask (M_BD) **
            block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)

            # **2. Offset Block-Causal Mask (M_OBC) **
            offset_block_causal = (
                (block_q > block_kv)
                & (x0_flag_kv == 1)
                & (x0_flag_q == 0)
            )

            # **3. Block-Causal Mask (M_BC) **
            block_causal = (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1)

            # **4. Combine Masks **
            return block_diagonal | offset_block_causal | block_causal

        if mode == 'bidirectional':
            attn_mask = bidirectional_mask
        elif mode == 'prefix_bidirectional':
            assert prefix_len is not None
            attn_mask = lambda b, h, q, kv: prefix_bidirectional_mask(prefix_len, b, h, q, kv)
        elif mode == 'efficient_block_diff':
            assert block_size is not None
            attn_mask = lambda b, h, q, kv: efficient_block_diff_mask(block_size, b, h, q, kv)
        elif mode == 'block_diff':
            assert block_size is not None
            attn_mask = lambda b, h, q, kv: block_diff_mask(block_size, b, h, q, kv, self.max_seq_length)
        else:
            raise ValueError(f"Unknown attention mode: {mode}")

        if q_len is not None:
            Q_LEN = q_len
        else:
            if mode == 'block_diff':
                Q_LEN = self.max_seq_length * 2
            else:
                Q_LEN = self.max_seq_length

        # No caching here to avoid Python overhead

        block_mask = create_block_mask(
            attn_mask, B=None, H=None, Q_LEN=Q_LEN, KV_LEN=Q_LEN
        )

        return block_mask


    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        is_training: bool = True,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings

        if self.mode == 'block_diff' and is_training:
            # Split query and key states in half along sequence length dimension
            q1, q2 = query_states.chunk(2, dim=2)
            k1, k2 = key_states.chunk(2, dim=2)
            
            # Apply RoPE independently to each half
            q1, k1 = apply_rotary_pos_emb(q1, k1, cos, sin)
            q2, k2 = apply_rotary_pos_emb(q2, k2, cos, sin)
            
            # Recombine the halves
            query_states = torch.cat([q1, q2], dim=2)
            key_states = torch.cat([k1, k2], dim=2)
        else:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if self.num_key_value_groups != 1:
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
        

        if self.mode == 'bidirectional':
            if q_len != self.bidirectional_mask.shape[-2]:
                block_mask = self.compute_block_mask(mode='bidirectional', prefix_len=self.prefix_len, q_len=q_len)
                # Persist resized mask for reuse
                self.bidirectional_mask = block_mask
            else:
                block_mask = self.bidirectional_mask

        elif self.mode == 'prefix_bidirectional':
            if self.prefix_len != self.prefix_len_orig or q_len != self.prefix_bidirectional_mask.shape[-2]:
                block_mask = self.compute_block_mask(mode='prefix_bidirectional', prefix_len=self.prefix_len, q_len=q_len)
                # Persist resized/re-parameterized mask
                self.prefix_bidirectional_mask = block_mask
            else:
                block_mask = self.prefix_bidirectional_mask
        elif self.mode == 'efficient_block_diff':
            if self.block_size != self.block_size_orig or q_len != self.efficient_block_diff_mask.shape[-2]:
                block_mask = self.compute_block_mask(mode='efficient_block_diff', block_size=self.block_size, q_len=q_len)
                # Persist resized/re-parameterized mask
                self.efficient_block_diff_mask = block_mask
            else:
                block_mask = self.efficient_block_diff_mask
        elif self.mode == 'block_diff':
            if self.block_size != self.block_size_orig or q_len != self.block_diff_mask.shape[-2]:
                block_mask = self.compute_block_mask(mode='block_diff', block_size=self.block_size, q_len=q_len)
                # Persist resized/re-parameterized mask
                self.block_diff_mask = block_mask
            else:
                block_mask = self.block_diff_mask
        else:
            raise ValueError(f"Unknown attention mode: {self.mode}")

        attn_output = fused_flex_attention(query_states, key_states, value_states, block_mask=block_mask)
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()

        attn_output = self.o_proj(attn_output)

        return attn_output, None


@torch.compile(fullgraph=True, mode="reduce-overhead")
def gumbel_topk(log_w: torch.Tensor, k: int) -> torch.Tensor:
    """Return a Bool mask of length len(log_w) with exactly k True."""
    g = -torch.log(-torch.log(torch.rand_like(log_w) + 1e-9) + 1e-9)
    topk = torch.topk(log_w + g, k).indices
    mask = torch.zeros_like(log_w, dtype=torch.bool)
    mask[topk] = True
    return mask
            

class DiffEncoderModel(Qwen3InfPreTrainedModel, GenerationMixin):
    """
    A single model with:
      - a bidirectional encoder + diffusion‐LM head over A
      - a causal decoder + LM head over B, conditioned on F_A
    """

    def __init__(self, config: NVRDiffConfig):
        super().__init__(config)

        self.mask_token_id = config.mask_token_id

        # Ensure attention backend is propagated to internal modules
        try:
            if getattr(config, "attn_implementation", None) is not None:
                config._attn_implementation = config.attn_implementation
        except Exception:
            pass

        diffusion_config = copy.deepcopy(config)
        try:
            if getattr(diffusion_config, "attn_implementation", None) is not None:
                diffusion_config._attn_implementation = diffusion_config.attn_implementation
        except Exception:
            pass
        diffusion_config.diffusion_lm = True

        if config.dlm_paradigm in ['prefix_bidirectional', 'efficient_block_diff', 'block_diff']:
            diffusion_config.attn_class = Qwen3InfFlexAttention
        elif config.dlm_paradigm in ['bidirectional', 'autoregressive']:
            diffusion_config.attn_class = Qwen3InfAttention

            if config.dlm_paradigm == 'autoregressive':
                diffusion_config.diffusion_lm = False
        else:
            raise ValueError(f"Unsupported DLM paradigm: {config.dlm_paradigm}")
        
        self.encoder = Qwen3InfModel(diffusion_config)
        self.diffusion_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.vocab_size = config.vocab_size

        self.post_init()

    


    def forward_process(self, input_ids, eps=1e-3, block_size=None, loss_mask=None):
        b, l = input_ids.shape
        device = input_ids.device

        t = torch.rand(b, device=device)

        p_mask = (1 - eps) * t + eps  # shape: (b,)
        p_mask = p_mask[:, None].expand(-1, l)  # shape: (b, l)

        masked_indices = torch.rand((b, l), device=device) < p_mask

        if loss_mask is not None:
            masked_indices[loss_mask == 0] = 0

        noisy_batch = torch.where(masked_indices, self.mask_token_id, input_ids)        

        return noisy_batch, masked_indices, p_mask


    def forward_process_exp(
        self,
        input_ids: torch.Tensor,
        eps: float = 1e-3,
        block_size: int | None = None,
        half_life_ratio: float = 0.25, # λ = ln 2 / (half_life_ratio·L)
        loss_mask: Optional[torch.Tensor] = None,
    ):
        """
        Two-stage corruption with optional per-block sampling.

        • Stage 1:  m ~ U(eps, 1)   →   k = round(m · len)  (exact budget).
        • Stage 2:  sample exactly k positions with weights
                    w_i(m) = exp[ λ · (1−m) · i ]   (late-heavy when m→0,
                                                     uniform when m→1).

          If `block_size` is given, the procedure is run *independently*
          inside each contiguous block of that length (last block may be shorter).
          When block_size is provided, m is sampled per-block and p_mask is per-block.

        Args
        ----
        input_ids : (B, L)  LongTensor
        eps       : minimum corruption ratio
        block_size: if not None, operate block-wise with per-block m sampling
        half_life_ratio : controls steepness when m→0
        """
        B, L = input_ids.shape
        device = input_ids.device
        dtype  = torch.float32

        masked_indices = torch.zeros((B, L), dtype=torch.bool, device=device)
        p_mask = torch.zeros((B, L), dtype=dtype, device=device)

        # ---------- Stage 1 & 2: whole-sentence or block-wise -------------------
        for b in range(B):
            if block_size is None:
                # ---------- Per-batch sampling (original behavior) ----------
                m = eps + (1.0 - eps) * torch.rand(1, device=device).item()   # scalar
                k_tot = int(round(m * L))
                k_tot = max(1, min(k_tot, L))  # clamp to [1, L]
                
                # Fill p_mask for this batch
                p_mask[b, :] = m
                
                slope = 1.0 - m          # ∈ [0,1]; 0 ⇒ uniform, 1 ⇒ late-heavy
                
                # ------- single pool over the whole sentence -------------
                lam_base = math.log(2.0) / (half_life_ratio * L) # base decay rate (λ when slope=1)

                pos   = torch.arange(L, device=device, dtype=dtype)
                log_w = (lam_base * slope * pos).clone()

                masked_indices[b] = gumbel_topk(log_w, k_tot)

            else:
                # ---------- Per-block sampling ----------
                num_blocks = math.ceil(L / block_size)
                lam_base = math.log(2.0) / (half_life_ratio * block_size) # base decay rate (λ when slope=1)

                for blk in range(num_blocks):
                    start = blk * block_size
                    end   = min((blk + 1) * block_size, L)
                    blk_len = end - start

                    # Sample m per block
                    m_blk = eps + (1.0 - eps) * torch.rand(1, device=device).item()
                    
                    # Fill p_mask for this block
                    p_mask[b, start:end] = m_blk
                    
                    # per-block budget
                    k_blk = int(round(m_blk * blk_len))
                    k_blk = max(0, min(k_blk, blk_len))
                    if k_blk == 0:
                        continue

                    slope = 1.0 - m_blk          # ∈ [0,1]; 0 ⇒ uniform, 1 ⇒ late-heavy

                    pos   = torch.arange(blk_len, device=device, dtype=dtype)
                    log_w = lam_base * slope * pos

                    blk_mask = gumbel_topk(log_w, k_blk)
                    masked_indices[b, start:end] = blk_mask

        if loss_mask is not None:
            masked_indices[loss_mask == 0] = 0            

        noisy_batch = torch.where(masked_indices, self.mask_token_id, input_ids)
        return noisy_batch, masked_indices, p_mask
    

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor]   = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor]       = None,
        split_len: Optional[int]                 = None,
        past_key_values: Optional[Cache]         = None,
        block_size: Optional[int]                = None,
        block_diff_ppl: bool                     = False,
        eps: float                               = 1e-3,
        is_teacher: bool                        = False,
        masked_indices: Optional[torch.Tensor]   = None,
        p_mask: Optional[torch.Tensor]           = None,
        loss_mask: Optional[torch.Tensor] = None,
        nvte: bool = False,  # Enable NVIDIA Transformer Engine attention
        **kwargs,
    ) -> CausalLMOutputWithPast:
        # Store nvte in kwargs so it gets passed to encoder
        kwargs['nvte'] = nvte

        batch_size, seq_len = input_ids.shape

        if self.config.dlm_paradigm == 'bidirectional':
            if labels is not None and torch.rand(1) < self.config.random_length_prob:
                random_length = torch.randint(2, input_ids.shape[1] + 1, (1,))
                input_ids = input_ids[:, :random_length]
                labels = labels[:, :random_length]
                
                if attention_mask is not None:
                    attention_mask = attention_mask[:, :random_length]
                if position_ids is not None:
                    position_ids = position_ids[:, :random_length]

        elif self.config.dlm_paradigm == 'prefix_bidirectional':
            if labels is not None and split_len is None:
                if torch.rand(1) < self.config.random_length_prob:
                    split_len = torch.randint(1, seq_len//64, (1,)).item() * 64  ## [64, seq_len] divisible by 64
                else:
                    split_len = int(seq_len * self.config.prefix_ratio)

        elif self.config.dlm_paradigm == 'efficient_block_diff':
            if labels is not None and block_size is None:
                if torch.rand(1) < self.config.random_length_prob:
                    block_size = torch.randint(1, 8, (1,)).item() * 4  ## [4, 32] divisible by 4
                else:
                    block_size = self.config.block_size

        elif self.config.dlm_paradigm == 'block_diff':
            if labels is not None and block_size is None:
                if torch.rand(1) < self.config.random_length_prob:
                    block_size = torch.randint(1, 8, (1,)).item() * 4  ## [4, 32] divisible by 4
                else:
                    block_size = self.config.block_size

        if labels is not None and self.config.dlm_paradigm != 'autoregressive':
            if masked_indices is not None:
                assert p_mask is not None

                if loss_mask is not None:
                    masked_indices[loss_mask == 0] = 0

                noisy_inputs = torch.where(masked_indices, self.mask_token_id, input_ids)

            else:
                if self.config.tok_mask_half_life_ratio is not None:
                    noisy_inputs, masked_indices, p_mask = self.forward_process_exp(input_ids, eps=eps, block_size=block_size, half_life_ratio=self.config.tok_mask_half_life_ratio, loss_mask=loss_mask)
                else:
                    noisy_inputs, masked_indices, p_mask = self.forward_process(input_ids, eps=eps, block_size=block_size, loss_mask=loss_mask)

        else:
            noisy_inputs = input_ids
            masked_indices = None
            p_mask = None

        if self.config.dlm_paradigm in ['prefix_bidirectional', 'efficient_block_diff', 'block_diff']:
            for layer in self.encoder.layers:
                if hasattr(layer.self_attn, 'set_attention_mode'):
                    layer.self_attn.set_attention_mode(self.config.dlm_paradigm, prefix_len=split_len, block_size=block_size)

        input_ids_len = noisy_inputs.shape[1]
        if labels is not None and self.config.dlm_paradigm == 'block_diff':
            if position_ids is None:
                position_ids = torch.arange(input_ids_len, device=noisy_inputs.device).unsqueeze(0)
            noisy_inputs = torch.cat([noisy_inputs, input_ids], dim=1)

        if block_diff_ppl:
            if position_ids is None:
                position_ids = torch.arange(input_ids_len // 2, device=noisy_inputs.device).unsqueeze(0)

        enc_out  = self.encoder(
            past_key_values=past_key_values,
            input_ids=noisy_inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            is_training=(labels is not None) or (block_diff_ppl),
            **kwargs,
        )

        logits = self.diffusion_head(enc_out.last_hidden_state)  # (batch, len_B, vocab)

        if labels is not None and self.config.dlm_paradigm == 'block_diff':
            logits = logits[:, :input_ids_len]

        loss = None
        if labels is not None:  
            if self.config.dlm_paradigm == 'autoregressive':
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                if loss_mask is None:
                    loss_fct = CrossEntropyLoss()
                    shift_logits = shift_logits.reshape(-1, shift_logits.size(-1))
                    shift_labels = shift_labels.reshape(-1)
                    loss = loss_fct(shift_logits, shift_labels)

                else:
                    loss_mask = loss_mask[..., 1:].contiguous()

                    loss_fct = CrossEntropyLoss(reduction='none')
                    shift_logits = shift_logits.reshape(-1, shift_logits.size(-1))
                    shift_labels = shift_labels.reshape(-1)
                    shift_labels = shift_labels.to(shift_logits.device)
                    
                    token_losses = loss_fct(shift_logits, shift_labels)
                                    
                    loss = token_losses[loss_mask].sum() / loss_mask.sum()

            else:
                # Handle DREAM vs LLADA style losses
                if hasattr(self.config, 'dlm_type') and self.config.dlm_type == 'dream':
                    logits = logits[..., :-1, :].contiguous()
                    labels = labels[..., 1:].contiguous()
                    masked_indices = masked_indices[:, 1:]
                    p_mask = p_mask[:, 1:]

                # Original boolean-mask indexing path (fast for dense masks)
                token_loss = torch.nn.functional.cross_entropy(
                    logits[masked_indices],
                    labels[masked_indices],
                    reduction='none'
                ) / p_mask[masked_indices]
                loss = token_loss.sum() / masked_indices.sum()

        return CausalLMOutputWithPast(
            loss=loss if not is_teacher else logits,
            logits=logits,
            past_key_values=enc_out.past_key_values,
            hidden_states=None,
            attentions=enc_out.attentions, 
        )