"""Model backends: GGUF via llama.cpp, Hugging Face folders via transformers."""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Callable, Iterator

# The caller can send a plain prompt or a chat-style list of messages.
ChatInput = str | list[dict[str, str]]


def _as_messages(prompt: ChatInput) -> list[dict[str, str]]:
    """Treat a plain string as one user message."""
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return prompt


def _plain_text(prompt: ChatInput) -> str:
    """Fallback for models that have no chat template."""
    messages = _as_messages(prompt)
    if len(messages) == 1:
        return messages[0]["content"]
    return "\n\n".join(message["content"] for message in messages)


def detect_backend(model_path: str | Path) -> str:
    """Return 'gguf' for a .gguf file, 'hf' for a local transformers folder."""
    path = Path(model_path)
    if path.is_file() and path.suffix.lower() == ".gguf":
        return "gguf"
    if path.is_dir() and (path / "config.json").exists():
        return "hf"
    raise ValueError(
        f"Unsupported model path: {path}. Use a .gguf file or a Hugging Face "
        "model directory with config.json."
    )


def _device_info() -> dict:
    """Small model/GPU report."""
    import torch

    if not torch.cuda.is_available():
        return {"device": "cpu", "gpu": None, "vram_gb": None}
    free, total = torch.cuda.mem_get_info()
    return {
        "device": "cuda",
        "gpu": torch.cuda.get_device_name(0),
        "vram_gb": round(total / 1024**3, 2),
        "vram_free_gb": round(free / 1024**3, 2),
    }


class LlamaCppBackend:
    """GGUF model through llama-cpp-python, all layers on the GPU."""

    name = "llama.cpp"

    def __init__(
        self,
        model_path: str | Path,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
        seed: int = 0,
    ):
        import torch

        # On Windows, llama.dll needs torch's CUDA 12 DLLs before it is imported.
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(Path(torch.__file__).parent / "lib"))

        from llama_cpp import Llama

        self.path = Path(model_path)
        self.n_ctx = n_ctx
        self.llm = Llama(
            model_path=str(self.path),
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            seed=seed,
            verbose=False,
        )
        self.chat_template = (self.llm.metadata or {}).get("tokenizer.chat_template")
        self.last_completion_tokens = 0

    def render(self, prompt: ChatInput) -> str:
        """Apply the model's chat template, or use plain text if it has none."""
        if not self.chat_template:
            return _plain_text(prompt)

        from jinja2.sandbox import ImmutableSandboxedEnvironment

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        template = env.from_string(self.chat_template)
        return template.render(
            messages=_as_messages(prompt),
            add_generation_prompt=True,
        )

    def count_tokens(self, text: str) -> int:
        return len(self.llm.tokenize(text.encode("utf-8"), add_bos=False, special=True))

    def stream(
        self,
        rendered_prompt: str,
        max_new_tokens: int,
        stop_strings: tuple[str, ...],
        should_stop: Callable[[str], bool],
    ) -> Iterator[str]:
        self.last_completion_tokens = 0
        text = ""
        stream = self.llm.create_completion(
            rendered_prompt,
            max_tokens=max_new_tokens,
            temperature=0.0,
            stop=[s for s in stop_strings if s],
            stream=True,
        )
        for chunk in stream:
            piece = chunk["choices"][0].get("text", "")
            if not piece:
                continue
            text += piece
            yield piece

            usage = chunk.get("usage") or {}
            if usage.get("completion_tokens"):
                self.last_completion_tokens = usage["completion_tokens"]
            if should_stop(text):
                break

        if not self.last_completion_tokens:
            self.last_completion_tokens = self.count_tokens(text)

    def info(self) -> dict:
        return {
            "backend": self.name,
            "model": str(self.path),
            "chat_template": bool(self.chat_template),
            "n_ctx": self.n_ctx,
            **_device_info(),
        }

    def unload(self) -> None:
        import torch

        self.llm = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class HFBackend:
    """Unquantized Hugging Face model folder through transformers."""

    name = "transformers"

    def __init__(self, model_path: str | Path, device_map: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.path = Path(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.path))
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.has_template = bool(getattr(self.tokenizer, "chat_template", None))

        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
        self.dtype = str(dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.path),
            torch_dtype=dtype,
            device_map=device_map,
        )
        self.model.eval()
        self.device = self.model.device
        self.last_completion_tokens = 0
        self._error: Exception | None = None

    def render(self, prompt: ChatInput) -> str:
        """Apply the tokenizer's chat template, or use plain text if it has none."""
        messages = _as_messages(prompt)
        if not self.has_template:
            return _plain_text(messages)
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def stream(
        self,
        rendered_prompt: str,
        max_new_tokens: int,
        stop_strings: tuple[str, ...],
        should_stop: Callable[[str], bool],
    ) -> Iterator[str]:
        import threading

        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        class StopOnText(StoppingCriteria):
            """Stop as soon as a stop string or the caller's rule matches."""

            def __init__(self, tokenizer, prompt_len, stop_strings, should_stop):
                super().__init__()
                self.tokenizer = tokenizer
                self.prompt_len = prompt_len
                self.stop_strings = stop_strings
                self.should_stop = should_stop
                self.n_generated = 0

            def __call__(self, input_ids, scores, **kwargs):
                generated = input_ids[0, self.prompt_len:]
                self.n_generated = int(generated.shape[0])
                text = self.tokenizer.decode(generated, skip_special_tokens=True)
                return any(s in text for s in self.stop_strings) or self.should_stop(text)

        inputs = self.tokenizer(
            rendered_prompt,
            return_tensors="pt",
            add_special_tokens=not self.has_template,
        )
        prompt_len = int(inputs["input_ids"].shape[1])
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        criteria = StopOnText(
            self.tokenizer, prompt_len, stop_strings, should_stop
        )
        kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "streamer": streamer,
            "stopping_criteria": StoppingCriteriaList([criteria]),
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        self._error = None
        self.last_completion_tokens = 0

        def generate():
            try:
                with torch.inference_mode():
                    self.model.generate(**kwargs)
            except Exception as exc:
                self._error = exc
                try:
                    streamer.end()
                except Exception:
                    pass

        thread = threading.Thread(target=generate, daemon=True)
        thread.start()
        text = ""
        try:
            for piece in streamer:
                if not piece:
                    continue
                text += piece
                yield piece
                if should_stop(text):
                    break
        finally:
            thread.join(timeout=2.0)
            self.last_completion_tokens = criteria.n_generated or self.count_tokens(text)
            if self._error:
                raise self._error

    def info(self) -> dict:
        import torch

        info = {
            "backend": self.name,
            "model": str(self.path),
            "chat_template": self.has_template,
            "dtype": self.dtype,
            **_device_info(),
            "device": str(self.device),
        }
        device_map = getattr(self.model, "hf_device_map", None)
        if device_map:
            info["device_map"] = {str(k): str(v) for k, v in device_map.items()}
            info["cpu_offload"] = any(
                str(v) in ("cpu", "disk") for v in device_map.values()
            )
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
        return info

    def unload(self) -> None:
        import torch

        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
