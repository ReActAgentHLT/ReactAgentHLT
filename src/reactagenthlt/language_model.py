import typing

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    TokenizersBackend,
)


class SupportsGeneration(typing.Protocol):
    def generate(self, **kwargs: typing.Any) -> torch.Tensor: ...


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
    ) -> str:
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

        generator = typing.cast(SupportsGeneration, self.model)
        outputs = generator.generate(**inputs, **generation_specs)
        generated = outputs[0, inputs["input_ids"].shape[-1] :]

        return str(self.tokenizer.decode(generated, skip_special_tokens=True))
