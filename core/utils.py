from core.base import TestGenerator, LLM, RatioScaleMetric
import importlib


SUPPORTED_MODELS = [
    "GPT-4o",
    "Llama-3.1-8B",
    "Llama-3.1-70B",
    "Gemini-1.5-Pro",
    "Gemini-2.5-Flash",
    "Claude-3.5-Haiku",
    "Gemma-2-9B-IT",
    "Qwen-2.5-72B-Instruct",
    "DeepSeek-V3",
    "Phi-3.5",
]


def get_generator(bias: str) -> TestGenerator:
    try:
        module = importlib.import_module(f'tests.{bias}.test')
        return getattr(module, f'{bias}TestGenerator')()
    except (ModuleNotFoundError, AttributeError) as e:
        raise ImportError(f"Could not find the generator for bias '{bias}': {e}")


def get_metric(bias: str) -> RatioScaleMetric:
    try:
        module = importlib.import_module(f'tests.{bias}.test')
        return getattr(module, f'{bias}Metric')
    except (ModuleNotFoundError, AttributeError) as e:
        raise ImportError(f"Could not find the metric for bias '{bias}': {e}")


def get_model(model_name: str, randomly_flip_options: bool = False, shuffle_answer_options: bool = False) -> LLM:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Model '{model_name}' is not supported. Choose one of: {SUPPORTED_MODELS}")

    if model_name == "GPT-4o":
        from models.OpenAI.model import GptFourO
        return GptFourO(randomly_flip_options, shuffle_answer_options)
    elif model_name == "Llama-3.1-8B":
        from models.Meta.model import LlamaThreePointOneEightB
        return LlamaThreePointOneEightB(randomly_flip_options, shuffle_answer_options)
    elif model_name == "Llama-3.1-70B":
        from models.Meta.model import LlamaThreePointOneSeventyB
        return LlamaThreePointOneSeventyB(randomly_flip_options, shuffle_answer_options)
    elif model_name == "Gemini-1.5-Pro":
        from models.Google.model import GeminiOneFivePro
        return GeminiOneFivePro(randomly_flip_options, shuffle_answer_options)
    elif model_name == "Gemini-2.5-Flash":
        from models.Google.model import GeminiTwoFiveFlash
        return GeminiTwoFiveFlash(randomly_flip_options, shuffle_answer_options)
    elif model_name == "Claude-3.5-Haiku":
        from models.Anthropic.model import ClaudeThreeHaiku
        return ClaudeThreeHaiku(randomly_flip_options, shuffle_answer_options)
    elif model_name == "Gemma-2-9B-IT":
        from models.Google.model import GemmaTwoNineB
        return GemmaTwoNineB(randomly_flip_options, shuffle_answer_options)
    elif model_name == "Qwen-2.5-72B-Instruct":
        from models.Alibaba.model import QwenTwoPointFiveSeventyTwoB
        return QwenTwoPointFiveSeventyTwoB(randomly_flip_options, shuffle_answer_options)
    elif model_name == "DeepSeek-V3":
        from models.DeepSeek.model import DeepSeekV3
        return DeepSeekV3(randomly_flip_options, shuffle_answer_options)
    elif model_name == "Phi-3.5":
        from models.Microsoft.model import PhiThreePointFive
        return PhiThreePointFive(randomly_flip_options, shuffle_answer_options)

    raise ValueError(f"Model '{model_name}' is not supported.")


def get_supported_models() -> list[str]:
    return SUPPORTED_MODELS
