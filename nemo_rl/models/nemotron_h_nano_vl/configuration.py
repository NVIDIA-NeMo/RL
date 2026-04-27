# --------------------------------------------------------
# Adapted from https://huggingface.co/OpenGVLab/InternVL2-Llama3-76B under MIT License
#     LICENSE is in incl_licenses directory.
# --------------------------------------------------------

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from .configuration_nemotron_h import NemotronHConfig

logger = logging.get_logger(__name__)

class NemotronH_Nano_VL_V2_Config(PretrainedConfig):
    model_type = 'NemotronH_Nano_VL_V2'
    is_composition = True

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        force_image_size=None,
        downsample_ratio=0.5,
        template=None,
        ps_version='v1',
        image_tag_type="internvl",
        projector_hidden_size=4096,
        vit_hidden_size=1280,
        attn_implementation="flash_attention_2",
        **kwargs
    ):
        super().__init__(**kwargs)
        
        if vision_config is not None:
            assert "auto_map" in vision_config and "AutoConfig" in vision_config["auto_map"]
            vision_auto_config = get_class_from_dynamic_module(*vision_config["auto_map"]["AutoConfig"].split("--")[::-1])
            self.vision_config = vision_auto_config(**vision_config)
            # self.vision_config = RADIOConfig(**vision_config)
        else:
            self.vision_config = PretrainedConfig()

        # Handle both cases: when loading from JSON (text_config is dict) and when called internally by transformers (text_config is None)
        if text_config is not None:
            self.text_config = NemotronHConfig(**text_config)
        else:
            self.text_config = NemotronHConfig()

        # Assign configuration values
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
        self.template = template  # TODO move out of here and into the tokenizer
        self.ps_version = ps_version  # Pixel shuffle version
        self.image_tag_type = image_tag_type # TODO: into the tokenizer too?
        self.projector_hidden_size = projector_hidden_size
        self.vit_hidden_size = vit_hidden_size

        self.layers_block_type = self.text_config.layers_block_type
        # self.vocab_size = getattr(self.text_config, "vocab_size", 0)
        # self.hidden_size = getattr(self.text_config, "hidden_size", 0)

        self._attn_implementation = attn_implementation
        self.vision_config.use_flash_attn = self._attn_implementation is not None and "flash_attention" in self._attn_implementation
        self.text_config._attn_implementation = self._attn_implementation
