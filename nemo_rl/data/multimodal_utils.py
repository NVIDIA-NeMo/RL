from transformers import PreTrainedTokenizerBase
from typing import Union, Optional
import torch


class PackedTensor:
    """Wrapper around a list of torch tensors and a dimension along which to pack the tensors.

    This class is used to wrap a list of tensors along with a `dim_to_pack` parameter.
    It can be used for data that can be packed along different dimensions (such as multimodal data).
    `dim_to_pack` is used to specify the dimension along which to pack the tensors.
    The list of tensors can be returned as a single packed tensor by calling `as_tensor` which will concatenate the tensors along the `dim_to_pack` dimension.
    """

    def __init__(self, tensors: Union[torch.Tensor, list[torch.Tensor]], dim_to_pack: int) -> None:
        assert tensors is not None, "Input tensors to PackedTensor cannot be None"

        if isinstance(tensors, torch.Tensor):
            self.tensors: list[torch.Tensor] = [tensors]
        elif isinstance(tensors, list):
            assert len(tensors) > 0, "Input tensors to PackedTensor must be a non-empty list"
            self.tensors: list[torch.Tensor] = tensors
        else:
            raise ValueError(f"Unsupported type for input tensors to PackedTensor: {type(tensors)}")
        self.dim_to_pack = dim_to_pack 

    def as_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        if device is not None:
            self.tensors = [item.to(device) for item in self.tensors]
        return torch.cat(self.tensors, dim=self.dim_to_pack).to(device)
    
    def __len__(self) -> int:
        return len(self.tensors)
    
    def to(self, device: str | torch.device) -> "PackedTensor":
        self.tensors = [item.to(device) for item in self.tensors]
        return self

    def slice(self, indices: Union[list[int], torch.Tensor]) -> "PackedTensor":
        idx = indices.tolist() if isinstance(indices, torch.Tensor) else indices
        tensors = [self.tensors[i] for i in idx]
        return PackedTensor(tensors, self.dim_to_pack)
    
    @classmethod
    def concat(cls, packed_batches: list["PackedTensor"]) -> "PackedTensor":
        """Concatenate a list of PackedTensor objects into a single PackedTensor.

        Each batch must have the same dim_to_pack.
        """
        dim_to_packs = [batch.dim_to_pack for batch in packed_batches]
        assert len(set(dim_to_packs)) == 1, "All PackedTensors must have the same dim_to_pack"
        # concatenate the tensors
        tensors = []
        for batch in packed_batches:
            tensors.extend(batch.tensors)
        dim_to_pack = dim_to_packs[0]
        return cls(tensors, dim_to_pack)

    @classmethod
    def concat_as_tensor(cls, packed_datas: list["PackedTensor"], device: Optional[torch.device] = None) -> torch.Tensor:
        return cls.concat(packed_datas).as_tensor(device)


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

        
def get_dim_to_pack_along(processor, key: str) -> int:
    '''
    Special considerations for packing certain keys from certain processors

    In most cases, the packed items are along dim 0
    '''
    if processor.__class__.__name__ == "SmolVLMProcessor":
        return 1
    # return zero by default
    return 0
