import json
import re
from dataclasses import dataclass
from typing import cast

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    DynamicCache,
    StoppingCriteria,
    StoppingCriteriaList,
    TextStreamer,
    TokenizersBackend,
)


@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0  # input tokens whose KV entries were reused


class RegexStop(StoppingCriteria):
    """
    Stops the model generation when the pattern is generated. Note that the
    pattern will be present in the generated string as well.
    """

    def __init__(self, tokenizer, pattern, prompt_length):
        self.tokenizer = tokenizer
        self.regex = re.compile(pattern)
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores, **kwargs):
        generated = input_ids[0, self.prompt_length :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return bool(self.regex.search(text))


class LanguageModel:
    """
    Wrapper around HuggingFace transfomers API for causal text generation.

    With `"kv_cache": true` in the config (the default), the KV cache of the
    previous call is kept. The conversation is still re-tokenized in full at
    every call; when the previous call's input tokens are exactly the start
    of the new input, their cached keys and values are reused and only the
    rest (the previous reply, re-tokenized, and the new turn) is computed.
    Any other input, such as a new question, starts from an empty cache.
    """

    def __init__(self, configpath: str) -> None:
        """Load the model and tokenizer described by a JSON config file.

        Args:
            configpath: Path to the model configuration file.
        """
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

        self.use_kv_cache = bool(config.get("kv_cache", True))
        self._cache: DynamicCache | None = None  # left by the previous call
        self._cached_input_ids: torch.Tensor | None = None  # that call's input

    def reset_cache(self) -> None:
        """Forget the KV cache, so the next call computes its whole input."""
        self._cache = None
        self._cached_input_ids = None

    def _new_cache(self) -> DynamicCache:
        cache = DynamicCache(config=self.model.config)
        # Sliding-window layers (Gemma) normally drop tokens that leave the
        # window as soon as they are computed, after which the cache cannot
        # be cut back. Recording keeps them until the next `crop`.
        cache.activate_past_recording()
        return cache

    def _prepare_cache(self, input_ids: torch.Tensor) -> DynamicCache:
        """
        Return the KV cache to generate with: the previous one cut back to the
        previous input when that input is an exact prefix of this one, or a
        new empty cache otherwise.

        The previous reply's entries are always dropped, because the reply
        is re-tokenized as part of the new input and computed again.
        """
        cache, previous = self._cache, self._cached_input_ids
        self.reset_cache()  # a failed call must not leave a stale cache
        if cache is None or previous is None:
            return self._new_cache()

        length = previous.shape[-1]
        if input_ids.shape[-1] <= length or not torch.equal(
            input_ids[0, :length], previous[0]
        ):
            return self._new_cache()

        # `crop` takes the (negative) number of trailing tokens to remove.
        cache.crop(-(cache.get_seq_length() - length))
        return cache

    @torch.inference_mode()
    def generate(
        self,
        messages: list[dict[str, str]],
        do_sample: bool = False,
        stream: bool = False,
        stop_criteria: tuple[str, ...] = (),
        *,
        strip_output: bool = True,
    ) -> tuple[str, TokenUsage]:
        """
        Args:
            messages: Chat messages containing role and content fields.
            do_sample: Use the sampling parameters from the model config.
            stream: Print generated tokens to stdout as they are produced.
            stop_criteria: Regex patterns that stop generation when matched.
                The matched text remains in the generated output.
            strip_output: Trim surrounding whitespace from the reply.

        Returns:
            Generated text and token usage for this model call.
        """

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

        # generate list of generation stopping criteria if some regex is encountered
        criteria = StoppingCriteriaList(
            [
                RegexStop(self.tokenizer, regex, inputs["input_ids"].shape[1])
                for regex in stop_criteria
            ]
        )

        # reuse the previous call's KV cache when its input is a prefix of this one
        cache = self._prepare_cache(inputs["input_ids"]) if self.use_kv_cache else None
        cached_tokens = cache.get_seq_length() if cache is not None else 0

        outputs = self.model.generate(
            **inputs,
            past_key_values=cache,
            use_cache=True,
            return_dict_in_generate=True,
            max_new_tokens=self.config["max_new_tokens"],
            do_sample=do_sample,
            stopping_criteria=criteria,
            streamer=streamer,
            **self.config["sampling"] if do_sample else dict(),  # generation parameters
        )

        if self.use_kv_cache:
            self._cache = outputs.past_key_values
            self._cached_input_ids = inputs["input_ids"]

        generated = outputs.sequences[0][input_tokens:]  # get only output tokens
        output_tokens = len(generated)  # count output tokens

        output = str(self.tokenizer.decode(generated, skip_special_tokens=True))

        return (
            output.strip() if strip_output else output,
            TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_tokens=cached_tokens,
            ),
        )
