# --------------------------------------------------------
# Adapted from https://huggingface.co/OpenGVLab/InternVL2-Llama3-76B under MIT License
#     LICENSE is in incl_licenses directory.
# --------------------------------------------------------


import warnings
from typing import Optional, Tuple, Union

import torch
import transformers
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModel, AutoModelForCausalLM, GenerationConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration import NemotronH_Nano_VL_V2_Config
from .modeling_nemotron_h import HybridMambaAttentionDynamicCache, NemotronHCausalLMOutput

logger = logging.get_logger(__name__)


"""
The following code is adapted from the
https://huggingface.co/OpenGVLab/InternVL2-Llama3-76B/blob/main/modeling_internvl_chat.py repository

The chat function is adapted to handle NVLM 1-D tile-tagging design for dynamic high-resolution images.
"""


class SquaredReLU(nn.Module):
    def forward(self, x):
        return torch.pow(torch.nn.functional.relu(x), 2)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight.to(torch.float32) * hidden_states).to(input_dtype)


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


class NemotronH_Nano_VL_V2(PreTrainedModel):
    config_class = NemotronH_Nano_VL_V2_Config
    main_input_name = 'pixel_values'
    _supports_flash_attn_2 = True
    _no_split_modules = ['NemotronHBlock']

    def __init__(self, config: NemotronH_Nano_VL_V2_Config):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.36.2', 'ge')
        image_size = config.force_image_size
        patch_size = config.patch_size
        self.patch_size = patch_size
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        self.image_tag_type = config.image_tag_type

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')

        self.language_model = AutoModelForCausalLM.from_config(config.text_config, trust_remote_code=True)
        self.vision_model = AutoModel.from_config(config.vision_config, trust_remote_code=True)
        self.vision_model.model._initialize_weights = self.vision_model.model._init_weights  # WAR for transformers issue 38358 
        self.vision_model.radio_model.make_preprocessor_external()
        self.vision_model = self.vision_model.to(self.language_model.config.torch_dtype)

        self.drop_vision_class_token = True

        # Construct the vision projection.
        # Default
        vit_hidden_size = config.vit_hidden_size
        vision_projection_hidden_size = config.projector_hidden_size
        llm_hidden_size = config.text_config.hidden_size

        self.mlp1 = nn.Sequential(
            RMSNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, eps=1e-5),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, vision_projection_hidden_size, bias=False),
            SquaredReLU(),
            nn.Linear(vision_projection_hidden_size, llm_hidden_size, bias=False)
        )
        self.mlp1 = self.mlp1.to(self.language_model.config.torch_dtype)

        self.img_context_token_id = self.config.image_context_token_id

        for p in self.vision_model.parameters():
            p.requires_grad = False
        self.vision_model.eval()
        for p in self.mlp1.parameters():
            p.requires_grad = False
        self.mlp1.eval()

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            cache_params: Optional[HybridMambaAttentionDynamicCache] = None,
            labels: Optional[torch.LongTensor] = None,
            inputs_embeds = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, NemotronHCausalLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        if image_flags is not None:
            image_flags = image_flags.squeeze(-1)

            B, N, C = inputs_embeds.shape
            inputs_embeds = inputs_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)

            vit_batch_size = pixel_values.shape[0]
            vit_embeds = self.extract_feature(pixel_values)

            del pixel_values

            # if torch.distributed.get_rank() == 0:
            #     print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

            vit_embeds = vit_embeds[image_flags == 1].reshape(-1, C)
            num_img_tokens = int(selected.sum().item())
            # FIXME(jseppanen): defensive merging of vit embeddings to input embeddings is needed
            # because sometimes the number of image placeholder tokens (num_img_tokens) doesn't
            # match with the number of image embeddings (vit_embeds.shape[0]) due to an upstream
            # data processing bug.
            if num_img_tokens != vit_embeds.shape[0]:
                warnings.warn(f"The number of image placeholder tokens ({num_img_tokens}) doesn't match the number of vit embeddings ({vit_embeds.shape[0]})")
            selected_ids = torch.nonzero(selected, as_tuple=False).squeeze(-1)
            k = min(num_img_tokens, vit_embeds.shape[0])
            inputs_embeds.index_copy_(0, selected_ids[:k], vit_embeds[:k])
            inputs_embeds.index_fill_(0, selected_ids[k:], 0.0)

            del vit_embeds

            inputs_embeds = inputs_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_params=cache_params,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return NemotronHCausalLMOutput(
            loss=loss,
            logits=logits,
            cache_params=outputs.cache_params,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        vit_embeds = self.vision_model(pixel_values).features
        vit_embeds = vit_embeds.to(dtype=torch.bfloat16)
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds).to(torch.bfloat16)
        return vit_embeds

    def train(self, mode: bool = True):
        super().train(mode)
        self.vision_model.eval()
        self.mlp1.eval()
        return self

    def chat(
        self,
        tokenizer, 
        image_preprocessor,
        messages,
        images,
        generation_config,
        img_start_token='<img>',
        img_end_token='</img>',
        img_context_token='<image>',
    ):
        eos_token_id = tokenizer.eos_token_id

        query = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if len(images) > 0:
            image_inputs = image_preprocessor(images)
            parts = query.split(img_context_token)
            assert len(parts) - 1 == len(image_inputs['num_patches']), f"Number of {img_context_token} tokens ({len(parts) - 1}) doesn't match num_patches_list length ({len(image_inputs['num_patches'])})"
            
            processed_query = parts[0]
            for num_patches, part in zip(image_inputs['num_patches'], parts[1:]):
                feature_size = num_patches * self.num_image_token
                image_repl = img_start_token + img_context_token * feature_size + img_end_token
                processed_query += image_repl + part
            pixel_values = image_inputs['pixel_values'].cuda()
        else:
            pixel_values = None
            processed_query = query

        model_inputs = tokenizer(processed_query, return_tensors='pt', add_special_tokens=False)
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config,
        )
        response = tokenizer.batch_decode(generation_output)[0]
        response = response.split(tokenizer.eos_token)[0].strip()
        return response

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features.cuda()
                vit_embeds = self.mlp1(vit_embeds)
            else:
                pixel_values = pixel_values.to(dtype=self.vision_model.config.torch_dtype)
                vit_embeds = self.extract_feature(pixel_values)
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = inputs_embeds.shape
            inputs_embeds = inputs_embeds.reshape(B * N, C)
            input_ids_copy = input_ids.reshape(B * N)
            selected = (input_ids_copy == self.img_context_token_id)
            assert selected.sum() != 0
            inputs_embeds[selected] = vit_embeds.reshape(-1, C).to(inputs_embeds.device, inputs_embeds.dtype)

            inputs_embeds = inputs_embeds.reshape(B, N, C)

        else:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        outputs = self.language_model.generate(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs
