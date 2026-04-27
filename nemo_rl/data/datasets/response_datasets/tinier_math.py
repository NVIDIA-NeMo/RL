import json
import random
from typing import Optional

from datasets import Dataset, Features, Image as ImageFeature, Sequence, Value
from PIL import Image

from nemo_rl.data.datasets.raw_dataset import RawDataset
from nemo_rl.data.interfaces import TaskDataSpec


class TinierMathDataset(RawDataset):
    def __init__(
        self,
        train_data_path: Optional[str] = None,
        prompt_file: Optional[str] = None,
    ):
        self.task_name = "tinier_math"
        if not train_data_path:
            raise ValueError("TinierMathDataset requires a JSONL path")
        self.formatted_ds = {
            "train": self._load_jsonl(train_data_path),
            "validation": None,
        }
        self.task_spec = TaskDataSpec(task_name="tinier_math", prompt_file=prompt_file)

    def _load_jsonl(self, path: str) -> Dataset:
        """Load a JSONL with image path and conversations into a Dataset."""
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                image = obj.get("image")
                if not image:
                    # educhat_math has also text-only samples, but nemo-rl VLM code path
                    # doesn't support mixed text and image samples
                    size = (random.randint(32, 512), random.randint(32, 512))
                    color = (
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255),
                    )
                    image = Image.new("RGB", size, color=color)
                conv = obj.get("conversations", [])
                q = conv[0]["value"] if conv and conv[0].get("from") == "human" else ""
                a = (
                    conv[1]["value"]
                    if len(conv) > 1 and conv[1].get("from") == "gpt"
                    else ""
                )
                rows.append(
                    {
                        "images": [image],
                        "question": q,
                        "answer": a,
                        "task_name": self.task_name,
                    }
                )
        features = Features(
            {
                "images": Sequence(ImageFeature()),
                "question": Value("string"),
                "answer": Value("string"),
                "task_name": Value("string"),
            }
        )
        return Dataset.from_list(rows, features=features)
