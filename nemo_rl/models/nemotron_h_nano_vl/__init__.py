from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoProcessor, AutoImageProcessor

from .configuration import NemotronH_Nano_VL_V2_Config
from .modeling import NemotronH_Nano_VL_V2
from .processing import NemotronNanoVLV2Processor
from .image_processing import NemotronNanoVLV2ImageProcessor
from .configuration_nemotron_h import NemotronHConfig
from .modeling_nemotron_h import NemotronHForCausalLM


def register():
    AutoConfig.register("NemotronH_Nano_VL_V2", NemotronH_Nano_VL_V2_Config)
    AutoModel.register(NemotronH_Nano_VL_V2_Config, NemotronH_Nano_VL_V2)
    AutoModelForCausalLM.register(NemotronH_Nano_VL_V2_Config, NemotronH_Nano_VL_V2)
    AutoProcessor.register(NemotronH_Nano_VL_V2_Config, NemotronNanoVLV2Processor)
    AutoImageProcessor.register(NemotronH_Nano_VL_V2_Config, fast_image_processor_class=NemotronNanoVLV2ImageProcessor)

    AutoConfig.register("nemotron_h", NemotronHConfig)
    AutoModelForCausalLM.register(NemotronHConfig, NemotronHForCausalLM)
