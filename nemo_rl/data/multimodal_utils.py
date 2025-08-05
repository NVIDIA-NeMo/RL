from transformers import PreTrainedTokenizerBase
from typing import Union, Optional
import torch
import numpy as np
from collections import defaultdict
import re

class PackedGenericDataItem:

    """Wrapper around a torch tensor that contains multimodal data"""
    def __init__(self, tensor: torch.Tensor, dim_to_pack: int = 0):
        self.tensor = tensor
        self.dim_to_pack = dim_to_pack
    
    def __call__(self):
        return self.tensor

    @classmethod
    def concat_packed_items(cls, packed_data: list["PackedGenericDataItem"], return_as_item: bool = False) -> Union["PackedGenericDataBatch", "PackedGenericDataItem"]:
        """Concatenate a list of PackedGenericDataItem objects into a single PackedGenericDataBatch.
        """
        dim = [item.dim_to_pack for item in packed_data]
        assert len(set(dim)) == 1, "All packed multimodal data must have the same dim_to_pack"
        dim = dim[0]
        tensors = [item.tensor for item in packed_data]
        # packed batch
        batch = cls.BatchClass(tensors, dim)
        if return_as_item:
            return batch.as_packed_data_item()
        return batch

class PackedGenericDataBatch:

    ItemClass = PackedGenericDataItem
    """Wrapper around a torch tensor that contains multimodal data"""
    def __init__(self, tensors: Union[list[torch.Tensor], list[PackedGenericDataItem]], dim_to_pack: int):
        # this is a concatenated databatch of packed multimodal data
        if isinstance(tensors[0], torch.Tensor):
            self.tensors = tensors
        elif isinstance(tensors[0], PackedGenericDataItem):
            self.tensors = [item.tensor for item in tensors]
        else:
            raise ValueError("tensor must be a torch.Tensor or a list of PackedGenericDataItem objects")
        self.dim_to_pack = dim_to_pack
    
    def as_tensor(self, as_tensors: bool = True, device: Optional[torch.device] = None) -> torch.Tensor:
        # return (self.tensor.to(device) if device is not None else self.tensor) if as_tensors else self
        if not as_tensors:
            if device is not None:
                self.tensors = [item.to(device) for item in self.tensors]
            return self
        # as tensors
        if device is not None:
            self.tensors = [item.to(device) for item in self.tensors]
        return torch.cat(self.tensors, dim=self.dim_to_pack)
    
    def __len__(self) -> int:
        # this is the number of items in the batch
        return len(self.tensors)
    
    def to(self, device: str | torch.device) -> "PackedGenericDataBatch":
        self.tensors = [item.to(device) for item in self.tensors]
        return self

    def slice(self, indices: Union[list[int], torch.Tensor]) -> "PackedGenericDataBatch":
        idx = indices.tolist() if isinstance(indices, torch.Tensor) else indices
        tensors = [self.tensors[i] for i in idx]
        return PackedGenericDataBatch(tensors, self.dim_to_pack)
    
    def as_packed_data_item(self) -> PackedGenericDataItem:
        # you can collapse the batch into a single item by calling this function
        # called inside `message_log_to_flat_messages` to convert the multimodal tensors from turns into a conversation
        ItemClass = self.ItemClass
        return ItemClass(torch.cat(self.tensors, dim=self.dim_to_pack), self.dim_to_pack)
    
    def repeat_interleave(self, num_repeats: int) -> "PackedGenericDataBatch":
        """Repeats the batch num_repeats times."""
        raise NotImplementedError("Why are you interleaving batches? Interleaving is supposed to happen at the message log level. \
                                  If you are adding new functionality, implement this function.")
    
    @classmethod
    def concat_packed_batches(cls, packed_batches: list["PackedGenericDataBatch"]) -> "PackedGenericDataBatch":
        """Concatenate a list of PackedGenericDataBatch objects into a single PackedGenericDataBatch.

        Each batch must have the same dim_to_pack.
        """
        dim_to_packs = [batch.dim_to_pack for batch in packed_batches]
        assert len(set(dim_to_packs)) == 1, "All packed multimodal data must have the same dim_to_pack"
        # concatenate the tensors
        # tensors = [batch.tensor for batch in packed_batches]
        tensors = []
        for batch in packed_batches:
            tensors.extend(batch.tensors)
        dim_to_pack = dim_to_packs[0]
        return cls(tensors, dim_to_pack)

# Set up circular references after both classes are defined
PackedGenericDataItem.BatchClass = PackedGenericDataBatch
PackedGenericDataBatch.ItemClass = PackedGenericDataItem

# define a class for packed multimodal data item
class PackedMultimodalDataItem(PackedGenericDataItem):
    pass

# define a class for packed multimodal data batch
class PackedMultimodalDataBatch(PackedGenericDataBatch):
    pass

# Set up circular references after both classes are defined
PackedMultimodalDataItem.BatchClass = PackedMultimodalDataBatch
PackedMultimodalDataBatch.ItemClass = PackedMultimodalDataItem

def get_multimodal_keys_from_processor(processor) -> list[str]:
    '''
    Get keys of the multimodal data that can be used as model inputs.

    This will be used in the data_processor function to determine which keys to use as model inputs.
    '''
    if isinstance(processor, PreTrainedTokenizerBase):
        return []
    
    all_keys = set()
    if hasattr(processor, "image_processor"):
        all_keys.update(processor.image_processor.model_input_names)
    if hasattr(processor, "video_processor"):
        all_keys.update(processor.video_processor.model_input_names)
    if hasattr(processor, "feature_extractor"):
        all_keys.update(processor.feature_extractor.model_input_names)
    # all_keys.update(processor.model_input_names)
    all_keys.difference_update(set(processor.tokenizer.model_input_names))
    return list(all_keys)


def reroute_processor_model_name_patch(model_name: str) -> str:
    '''
    for certain models, the processor is configured incorrectly, so we use another processor that is safer

    First, we try to match the model name to an exact match in the registry.
    If not found, we try to match the model name to a regex in the registry.
    '''
    PROCESSOR_REROUTE_REGEX_REGISTRY = {}
    PROCESSOR_REROUTE_EXACT_REGISTRY = {}
    regex_registry = PROCESSOR_REROUTE_REGEX_REGISTRY

    # Qwen2-VL models have a processor that produces empty strings in `apply_chat_template` function
    regex_registry.update({'qwen/qwen2-vl-*': 'Qwen/Qwen2.5-VL-3B-Instruct'})

    if model_name.lower() in PROCESSOR_REROUTE_EXACT_REGISTRY:
        print(f"Rerouting processor for {model_name} to {PROCESSOR_REROUTE_EXACT_REGISTRY[model_name.lower()]}")
        return PROCESSOR_REROUTE_EXACT_REGISTRY[model_name.lower()]

    for regex, replacement in regex_registry.items():
        if re.match(regex, model_name.lower()):
            print(f"Rerouting processor for {model_name} to {replacement}")
            return replacement
    return model_name
        
def get_dim_to_pack_along(processor, key: str) -> int:
    '''
    Special considerations for packing certain keys from certain processors

    In most cases, the packed items are along dim 0
    '''
    if processor.__class__.__name__ == "SmolVLMProcessor":
        return 1
    # return zero by default
    return 0
