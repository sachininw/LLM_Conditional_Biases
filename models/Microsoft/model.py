from core.base import LLM
from openai import OpenAI
import yaml
import os


class PhiThreePointFive(LLM):
    """Phi-4 via DeepInfra (OpenAI-compatible API)."""

    def __init__(
        self, randomly_flip_options: bool = False, shuffle_answer_options: bool = False
    ):
        super().__init__(
            randomly_flip_options=randomly_flip_options,
            shuffle_answer_options=shuffle_answer_options,
        )
        if "DEEPINFRA_API" not in os.environ:
            raise ValueError(
                "Cannot access DeepInfra API due to missing API key. "
                "Please store your API key in environment variable 'DEEPINFRA_API'."
            )
        self._CLIENT = OpenAI(
            base_url="https://api.deepinfra.com/v1/openai",
            api_key=os.environ["DEEPINFRA_API"],
        )
        self.NAME = "microsoft/phi-4"
        self.RESPONSE_FORMAT = None
        with open("./models/Microsoft/prompts.yml") as f:
            self._PROMPTS = yaml.safe_load(f)

    def prompt(self, prompt: str, temperature: float = 0.0, seed: int = 42) -> str:
        response = self._CLIENT.chat.completions.create(
            model=self.NAME,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    def prompt_multiturn(self, messages: list[dict], temperature: float = 0.0, seed: int = 42) -> str:
        response = self._CLIENT.chat.completions.create(
            model=self.NAME,
            temperature=temperature,
            messages=messages,
        )
        return response.choices[0].message.content
