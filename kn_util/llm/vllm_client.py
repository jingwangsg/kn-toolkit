import requests
from transformers import AutoTokenizer

from ..utils.multiproc import map_async_with_thread


class ResponseUnwrapMixin:
    def unwrap_generated_text(self, responses):
        return [response["choices"][0]["text"] for response in responses]

    def unwrap_total_tokens(self, responses):
        return [response["usage"]["total_tokens"] for response in responses]


class LLMClient(ResponseUnwrapMixin):
    def __init__(
        self,
        model_path,
        model,
        max_tokens=8192,
        temperature=0,
        port=8091,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
        )
        self.endpoint = f"http://localhost:{port}/"
        self.params = {
            "model": model,
            "temperature": temperature,
        }
        self.max_tokens = max_tokens
        self.model = model


    def get_token_length(self, prompts):
        return self.tokenizer(prompts, return_length=True)["length"]

    def generate(self, prompt):
        tokenized_prompt = self.tokenizer(prompt, return_length=True)
        length = tokenized_prompt["length"][0]

        response = requests.post(
            self.endpoint + "v1/completions",
            json={
                "prompt": prompt,
                "max_tokens": self.max_tokens - length,
                **self.params,
            },
        )

        return response.json()

    def generate_batch(
        self,
        prompts,
        num_thread=32,
    ):
        tokenized_prompt = self.tokenizer(prompts, return_length=True)
        lengths = tokenized_prompt["length"]

        params = [
            {
                "prompt": prompt,
                "max_tokens": self.max_tokens - length,
                **self.params,
            }
            for prompt, length in zip(prompts, lengths)
        ]

        responses = map_async_with_thread(
            iterable=params,
            func=lambda param: requests.post(self.endpoint + "v1/completions", json=param),
            verbose=True,
            num_thread=num_thread,
        )

        return [response.json() for response in responses]
