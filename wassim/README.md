# wassim — model inference

Small engine for loading a model and streaming text. It does **not** know about
ReAct, actions, tools, tasks, or datasets. The agent supplies prompts, history,
stop rules, and tools.

## Files

- `backends.py` — GGUF via `llama.cpp`, Hugging Face folder via `transformers`.
- `engine.py` — `ModelEngine.generate(...)`.
- `test_engine.py` — manual test for both paths and stop rules.

## Install

```powershell
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install llama-cpp-python transformers accelerate jinja2 safetensors sentencepiece
```

For CUDA, install the prebuilt CUDA wheel of `llama-cpp-python==0.3.4`.

## Model path

- A `.gguf` file → `llama.cpp`, `n_gpu_layers=-1`, `n_ctx=8192`.
- A folder with `config.json` → `transformers`, bf16/fp16, `device_map="auto"`.
- Gemma is gated: accept the license and run `huggingface-cli login`.

## Use

```python
from wassim.engine import ModelEngine

engine = ModelEngine("qwen2.5-3b-instruct-q4_k_m.gguf")

result = engine.generate(
    [{"role": "user", "content": "What is 2 + 2? Reply only Finish[number]."}],
    max_new_tokens=64,
    stop_strings=("Observation:",),
    should_stop=lambda text: "Finish[" in text and "]" in text,
    on_token=lambda piece: print(piece, end="", flush=True),
)

print(result.text)
print(result.completion_tokens)
engine.unload()
```

## `generate(...)` parameters

| Parameter | Meaning |
|---|---|
| `prompt` | A string, or a list of chat messages (`role` + `content`). |
| `max_new_tokens` | Maximum generated tokens. Default `256`. |
| `stop_strings` | Literal strings that stop generation. |
| `should_stop` | Optional `should_stop(text) -> bool` rule from the agent. |
| `on_token` | Optional callback for each streamed piece. |

## Return

```python
Generation(text: str, completion_tokens: int)
```

Only those two fields. Prompt tokens, TTFT, inference time, stop reason, tools,
and evaluation are the caller's responsibility.

## Test

Run from the project root:

```powershell
python -m wassim.test_engine --model .\qwen2.5-3b-instruct-q4_k_m.gguf
```

For a Hugging Face folder:

```powershell
python -m wassim.test_engine --model "C:\models\gemma-2-2b-it"
```

The test checks backend detection, loading, streaming, chat messages, stop
strings, completion tokens, and unload.
