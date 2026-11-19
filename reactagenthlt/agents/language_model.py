import json
import re
from dataclasses import asdict, dataclass
from typing import cast

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    StoppingCriteria,
    StoppingCriteriaList,
    TextStreamer,
    TokenizersBackend,
)


@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int


class RegexStop(StoppingCriteria):
    def __init__(self, tokenizer, pattern, prompt_length):
        self.tokenizer = tokenizer
        self.regex = re.compile(pattern)
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores, **kwargs):
        generated = input_ids[0, self.prompt_length :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return bool(self.regex.search(text))


class LanguageModel:
    def __init__(
        self,
        configpath: str,
        do_sample: bool = False,
        stream: bool = False,
        stop_criterias: list[str] = [],
    ) -> None:
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

        self.do_sample = do_sample
        self.stream = stream
        self.stop_criterias = stop_criterias

    @torch.inference_mode()
    def generate(self, messages: list[dict[str, str]]) -> tuple[str, TokenUsage]:

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
            if self.stream
            else None
        )

        # generate list of generation stopping criteria if some regex is encountered
        criteria = StoppingCriteriaList(
            [
                RegexStop(self.tokenizer, regex, inputs["input_ids"].shape[1])
                for regex in self.stop_criterias
            ]
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config["max_new_tokens"],
            do_sample=self.do_sample,
            stopping_criteria=criteria,
            streamer=streamer,
            **self.config["sampling"]
            if self.do_sample
            else dict(),  # generation parameters
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
