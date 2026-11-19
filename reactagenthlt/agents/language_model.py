import json
from dataclasses import asdict, dataclass
from typing import cast

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    TextStreamer,
    TokenizersBackend,
)


@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int


class LanguageModel:
    def __init__(self, configpath: str) -> None:
        with open(configpath, "r") as configfile:
            config = json.load(configfile)

        self.config = config
        self.tokenizer = cast(
            TokenizersBackend, AutoTokenizer.from_pretrained(config["model"])
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config["model"], device_map="auto"
        )
        self.model.eval()

    @torch.inference_mode()
    def generate(
        self,
        messages: list[dict[str, str]],
        do_sample: bool = False,
        stream: bool = False,
    ) -> tuple[str, TokenUsage]:

        # tokenize the input
        inputs = cast(
            BatchEncoding,
            self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ),
        ).to(self.model.device)

        # count input tokens
        input_tokens = inputs["input_ids"].shape[-1]

        # used for live demo in order to generate tokens interactively
        streamer = (
            TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            if stream
            else None
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config["max_new_tokens"],
            do_sample=do_sample,
            streamer=streamer,
            **self.config["sampling"] if do_sample else dict(),  # generation parameters
        )

        generated = outputs[0][input_tokens:]  # get only output tokens
        output_tokens = len(generated)  # count output tokens

        output = str(self.tokenizer.decode(generated, skip_special_tokens=True))

        return (
            output.strip(),
            TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
        )
