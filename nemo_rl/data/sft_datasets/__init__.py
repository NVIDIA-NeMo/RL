from functools import partial

from nemo_rl.data.sft_datasets.local_sft_dataset import LocalSFTDataset
from nemo_rl.data.sft_datasets.oasst import OasstDataset
from nemo_rl.data.sft_datasets.squad import SquadDataset
from nemo_rl.data.sft_datasets.openmathinstruct2 import OpenMathInstruct2Dataset
from nemo_rl.data.sft_datasets.oai_format_dataset import OpenAIFormatDataset
from nemo_rl.data.sft_datasets.clevr import CLEVRCoGenTDataset


def load_sft_dataset(data_config, seed: int):
    """Loads SFT dataset."""
    dataset_name = data_config["dataset_name"]

    if dataset_name == "open_assistant":
        base_dataset = OasstDataset(
            output_dir="/tmp/open_assistant",
            seed=seed,
        )
    elif dataset_name == "squad":
        base_dataset = SquadDataset()
    elif dataset_name == "openmathinstruct2":
        base_dataset = OpenMathInstruct2Dataset(
            split=data_config["split"],
            output_key=data_config["output_key"],
            prompt_file=data_config["prompt_file"],
            seed=seed,
        )
    elif dataset_name == "clevr_cogent":
        from nemo_rl.data.hf_datasets.clevr import format_clevr_cogent_dataset

        base_dataset = CLEVRCoGenTDataset(
            split=data_config["split"],
            prompt_file=data_config["prompt_file"],
        )
        datum_preprocessor = partial(format_clevr_cogent_dataset, return_pil=True)
    elif dataset_name == "openai_format":
        base_dataset = OpenAIFormatDataset(
            data_config["train_data_path"],
            data_config["val_data_path"],
            data_config["chat_key"],
            data_config["system_key"],
            data_config["system_prompt"],
        )
    # fall back to local dataset
    else:
        base_dataset = LocalSFTDataset(
            data_config["train_data_path"],
            data_config["val_data_path"],
            data_config["input_key"],
            data_config["output_key"],
        )

    return base_dataset

__all__ = [
    "OasstDataset",
    "SquadDataset",
    "OpenMathInstruct2Dataset",
    "CLEVRCoGenTDataset",
    "OpenAIFormatDataset",
    "LocalSFTDataset",
]
