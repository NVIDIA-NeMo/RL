import json

import ray

from vllm import LLM, SamplingParams


@ray.remote
class VLLMWorker:
    def __init__(self, llm_kwargs: dict, sampling_params: dict):
        self.sampling_params_dict = sampling_params

        self.llm = LLM(**llm_kwargs)

    def generate(self, prompt: dict[str, list[int]], sampling_params: dict):
        input_text = self.llm.get_tokenizer().decode(prompt["prompt_token_ids"])
        outputs = self.llm.generate(prompt, SamplingParams(**(self.sampling_params_dict | sampling_params)))

        return {
            "input_text": input_text,
            "output_text": outputs[0].outputs[0].text,
        }


if __name__ == "__main__":
    with open("vllm_args.json") as f:
        vllm_args = json.load(f)
    llm_kwargs = vllm_args["vllm_llm_kwargs"]
    sampling_params = vllm_args["sampling_params"]

    NUM_GPUS = 8
    TENSOR_PARALLEL_SIZE = 1

    workers = []
    for _ in range(NUM_GPUS // TENSOR_PARALLEL_SIZE):
        worker = VLLMWorker.options(num_gpus=TENSOR_PARALLEL_SIZE).remote(llm_kwargs, sampling_params)
        workers.append(worker)

    with open("prompt_token_ids.json") as f:
        all_prompt_token_ids = json.load(f)

    futures = []
    for i, (prompt, worker) in enumerate(zip(all_prompt_token_ids, workers)):
        futures.append(worker.generate.remote({"prompt_token_ids": prompt}, {"seed": i}))

    responses = ray.get(futures)

    with open("temp.json", "w") as f:
        json.dump(responses, f, indent=4)
