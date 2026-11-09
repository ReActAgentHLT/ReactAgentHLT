import re
import typing
from dataclasses import dataclass, field

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    TokenizersBackend,
)


def parse_thinking(content: str) -> list[str]:
    match = re.search(r"<think>(.*?)</think>(.*)", content, flags=re.DOTALL)
    if match:
        return [match.group(1).strip(), match.group(2).strip()]
    return ["", content.strip()]


class SupportsGeneration(typing.Protocol):
    def generate(self, **kwargs: typing.Any) -> torch.Tensor: ...


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    thinking_tokens: int = 0
    answer_tokens: int = 0


@dataclass
class Response:
    answer: str
    thinking: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)


class LanguageModel:
    """Core language model interface used by agents"""

    def __init__(self, model_id: str) -> None:
        self.tokenizer = typing.cast(
            TokenizersBackend, AutoTokenizer.from_pretrained(model_id)
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype="auto"
        )
        self.model.eval()

    @torch.inference_mode()
    def generate(
        self,
        messages: list[dict[str, str]],
        generation_specs: dict[str, typing.Any],
    ) -> Response:
        inputs = typing.cast(
            BatchEncoding,
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ),
        ).to(self.model.device)

        # account token usage
        usage = TokenUsage()
        usage.prompt_tokens = inputs["input_ids"].shape[-1]

        # generate answer
        generator = typing.cast(SupportsGeneration, self.model)
        outputs = generator.generate(**inputs, **generation_specs)[0]

        # decode answer and parse thinking
        generated = str(
            self.tokenizer.decode(
                outputs[usage.prompt_tokens :], skip_special_tokens=True
            )
        )

        thinking, answer = parse_thinking(generated)

        tokenized_answer = self.tokenizer(answer, return_dict=True, return_tensors="pt")
        tokenized_thinking = self.tokenizer(
            thinking, return_dict=True, return_tensors="pt"
        )

        usage.answer_tokens = tokenized_answer["input_ids"][0].shape[0]
        usage.thinking_tokens = tokenized_thinking["input_ids"][0].shape[0]

        return Response(answer, thinking, usage)
