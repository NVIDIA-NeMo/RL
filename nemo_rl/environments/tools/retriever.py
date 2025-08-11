# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import re
from collections import Counter
from typing import Any, Dict, List, TypedDict

import ray
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn


class RAGEnvConfig(TypedDict):
    dataset_name: str  # Name of the dataset to load
    dataset_split: str  # Split of the dataset to use
    text_column: str  # Column name containing the text to retrieve
    num_results: int  # Number of documents to retrieve
    k1: float  # BM25 parameter
    b: float  # BM25 parameter
    device: str  # Device to compute BM25


class BM25Retriever:
    """Sparse BM25 retriever.

    Args:
        documents: list of documents to retrieve from
        num_result: retrieve top-k documents
        k1: parameter of BM25. Values in [1.2, 2.0] are recommended.
        b: parameter of BM25. 0.75 is recommended.
        device: device to compute BM25
    """

    def __init__(
        self,
        documents: List[str] = None,
        num_result: int = 10,
        k1: float = 1.5,
        b: float = 0.75,
        device: str = "cpu",
    ):
        if documents is None:
            dataset = load_dataset("wikimedia/wikipedia", "20231101.en")
            self.documents = [sample["text"] for sample in dataset["train"]]
        else:
            self.documents = documents
        self.tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased", use_fast=True
        )
        self.num_result = num_result
        self.k1 = k1
        self.b = b
        self.device = device
        self.corpus_size = len(self.documents)
        self.vocab_size = self.tokenizer.vocab_size

        self.build_index()

    def build_index(self):
        doc_ids = []
        token_ids = []
        tfs = []
        lengths = []

        for i, document in enumerate(
            tqdm(self.documents, "Build index for BM25Retriever")
        ):
            input_ids = self.tokenizer.encode(document, add_special_tokens=False)
            token2cnt = Counter(input_ids)
            token_ids += token2cnt.keys()
            tfs += token2cnt.values()
            doc_ids += [i] * len(token2cnt)
            lengths.append(len(input_ids))

        avg_dl = sum(lengths) / self.corpus_size
        for i, doc_id in enumerate(doc_ids):
            tfs[i] = (
                tfs[i]
                * (self.k1 + 1)
                / (tfs[i] + self.k1 * (1 - self.b + self.b * lengths[doc_id] / avg_dl))
            )

        indices = torch.tensor([doc_ids, token_ids], device=self.device)
        values = torch.tensor(tfs, device=self.device)
        self.doc_tfs = torch.sparse_coo_tensor(
            indices, values, (self.corpus_size, self.vocab_size)
        )

        idfs = [0] * self.vocab_size
        token2df = Counter(token_ids)
        for token_id, df in token2df.items():
            idfs[token_id] = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1)
        self.idfs = idfs

    def __call__(self, query: str) -> List[str]:
        input_ids = self.tokenizer.encode(query, add_special_tokens=False)
        token2cnt = Counter(input_ids)
        token_ids = []
        query_idfs = []
        for token_id, query_tf in token2cnt.items():
            token_ids.append(token_id)
            query_idfs.append(query_tf * self.idfs[token_id])

        indices = torch.tensor([token_ids, [0] * len(token_ids)], device=self.device)
        values = torch.tensor(query_idfs, device=self.device)
        query_idfs = torch.sparse_coo_tensor(indices, values, (self.vocab_size, 1))

        scores = torch.sparse.mm(self.doc_tfs, query_idfs)
        scores = scores.to_dense().squeeze(-1)
        results = []
        for i in scores.topk(k=self.num_result).indices.tolist():
            results.append(self.documents[i])

        return results


@ray.remote  # pragma: no cover
class RAGEnvironment(EnvironmentInterface):
    """RAG environment that uses BM25 for document retrieval."""

    def __init__(self, cfg: RAGEnvConfig):
        self.cfg = cfg

        # Load dataset
        dataset = load_dataset(cfg["dataset_name"], split=cfg["dataset_split"])
        documents = [sample[cfg["text_column"]] for sample in dataset]

        # Initialize BM25 retriever
        self.retriever = BM25Retriever(
            documents=documents,
            num_result=cfg["num_results"],
            k1=cfg["k1"],
            b=cfg["b"],
            device=cfg["device"],
        )

    def format_result(self, retrieved_docs: List[str]) -> str:
        result = "<result>\n"
        for i, doc in enumerate(retrieved_docs):
            result += f"<{i + 1}>\n{doc}\n</{i + 1}>\n"
        result += "</result>\n"
        return result

    def step(
        self,
        message_log_batch: List[LLMMessageLogType],
        metadata_batch: List[Dict[str, Any]],
        return_extracted_answer: bool = False,
    ) -> EnvironmentReturn:
        """Process a batch of retrieval steps."""
        # Extract queries from the last message in each log
        messages = [ml[-1]["content"] for ml in message_log_batch]

        # Retrieve documents for each query
        results = []
        for message in messages:
            match = re.search(r"<retrieve>(.*)</retrieve>", message, re.DOTALL)
            if not match:
                results.append(
                    {"role": "environment", "content": "No retrieval query found!"}
                )
                continue
            query = match.group(1)
            retrieved_docs = self.retriever(query)
            result = self.format_result(retrieved_docs)
            results.append({"role": "environment", "content": result})

        batch_size = len(message_log_batch)
        rewards_tensor = torch.zeros(batch_size, dtype=torch.float32)
        terminated_tensor = torch.ones(batch_size, dtype=torch.bool)
        next_stop_strings = [["</retrieve>"]] * batch_size

        assert return_extracted_answer == False, (
            "return_extracted_answer is not supported in RAGEnvironment. Please set it to False."
        )
        extracted_answers = None

        return EnvironmentReturn(
            observations=results,
            metadata=metadata_batch,
            next_stop_strings=next_stop_strings,
            rewards=rewards_tensor,
            terminateds=terminated_tensor,
            answers=extracted_answers,
        )

    def shutdown(self):
        """Clean up resources."""
        pass

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict]:
        """Compute metrics for the batch."""
        # No specific metrics for RAG
        return batch, {}
