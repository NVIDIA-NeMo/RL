from hydra.utils import get_class
from torchdata.stateful_dataloader import StatefulDataLoader


class MultipleDataloaderWrapper:
    """Wrapper for multiple dataloaders.

    This wrapper is used to sample data from multiple dataloaders using a custom dataloader function.
    """

    def __init__(
        self,
        num_prompts_per_step: int,
        data_config: dict,
        dataloaders: dict[str, StatefulDataLoader],
    ):
        self.num_prompts_per_step = num_prompts_per_step
        self.data_config = data_config
        self.dataloaders = dataloaders

        # init data iterators
        self.data_iterators = {
            task_name: iter(dataloader) for task_name, dataloader in dataloaders.items()
        }

        self.custom_dataloader_func = get_class(data_config["custom_dataloader"])
        self.records = {}

    def __iter__(self):
        result, self.data_iterators = self.custom_dataloader_func(
            self.data_iterators, self.dataloaders, **self.records
        )
        assert len(result) == self.num_prompts_per_step, (
            f"Expected {self.num_prompts_per_step} prompts, but got {len(result)}"
        )

        # reset records
        self.records = {}

        return result

    def set_records(self, records: dict):
        """Set the records for the custom dataloader.

        Records are used to pass additional information to the custom dataloader to decide how to sample the data from the dataloaders.
        """
        self.records.update(records)
