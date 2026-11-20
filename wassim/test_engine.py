r"""Manual test for the wassim model engine.

Run from the project root:

    python -m wassim.test_engine --model .\qwen2.5-3b-instruct-q4_k_m.gguf

For a Hugging Face model folder:

    python -m wassim.test_engine --model "C:\models\gemma-2-2b-it"
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

from wassim.backends import detect_backend
from wassim.engine import ModelEngine


def timed_generate(engine, prompt, **kwargs):
    """The engine returns no timing, so the caller measures it."""
    start = time.perf_counter()
    first_token = None

    def on_token(piece: str) -> None:
        nonlocal first_token
        if first_token is None:
            first_token = time.perf_counter()
        print(piece, end="", flush=True)

    result = engine.generate(prompt, on_token=on_token, **kwargs)
    end = time.perf_counter()

    ttft = (first_token - start) if first_token else 0.0
    inference = end - start
    speed = result.completion_tokens / inference if inference > 0 else 0.0
    return result, ttft, inference, speed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="qwen2.5-3b-instruct-q4_k_m.gguf",
        help=".gguf file or Hugging Face model folder",
    )
    parser.add_argument(
        "--prompt",
        default="What is 2 + 2? Reply only Finish[number].",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("1. Backend detection")
    print("=" * 60)
    if not Path(args.model).exists():
        print(f"Model not found: {args.model}")
        return
    kind = detect_backend(args.model)
    print(f"model: {args.model}")
    print(f"detected backend: {kind}")

    with tempfile.TemporaryDirectory() as folder:
        Path(folder, "config.json").write_text("{}", encoding="utf-8")
        print(f"fake HF folder detected as: {detect_backend(folder)}")

    try:
        detect_backend("missing.gguf")
    except ValueError as exc:
        print(f"bad path correctly rejected: {exc}")

    print("\n" + "=" * 60)
    print("2. Load the model")
    print("=" * 60)
    engine = ModelEngine(args.model)
    for key, value in engine.info().items():
        print(f"{key}: {value}")

    print("\n" + "=" * 60)
    print("3. Plain prompt, streaming, stop at Finish[...]")
    print("=" * 60)
    result, ttft, inference, speed = timed_generate(
        engine,
        args.prompt,
        max_new_tokens=48,
        stop_strings=("Observation:",),
        should_stop=lambda text: "Finish[" in text and "]" in text,
    )
    print("\n\ntext:", repr(result.text))
    print("completion_tokens:", result.completion_tokens)
    print(f"TTFT: {ttft:.3f} s")
    print(f"inference: {inference:.3f} s")
    print(f"tokens/sec: {speed:.2f}")

    print("\n" + "=" * 60)
    print("4. Chat messages with system + user")
    print("=" * 60)
    result, ttft, inference, speed = timed_generate(
        engine,
        [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "What is 3 + 3? Reply only Finish[number]."},
        ],
        max_new_tokens=48,
        stop_strings=("Observation:",),
        should_stop=lambda text: "Finish[" in text and "]" in text,
    )
    print("\n\ntext:", repr(result.text))
    print("completion_tokens:", result.completion_tokens)
    print(f"TTFT: {ttft:.3f} s")
    print(f"inference: {inference:.3f} s")
    print(f"tokens/sec: {speed:.2f}")

    print("\n" + "=" * 60)
    print("5. Stop string test")
    print("=" * 60)
    result, ttft, inference, speed = timed_generate(
        engine,
        "Write one short line: hello world",
        max_new_tokens=16,
        stop_strings=("\n",),
    )
    print("\n\ntext:", repr(result.text))
    print("completion_tokens:", result.completion_tokens)

    print("\n" + "=" * 60)
    print("6. Unload")
    print("=" * 60)
    engine.unload()
    print("model unloaded")


if __name__ == "__main__":
    main()
