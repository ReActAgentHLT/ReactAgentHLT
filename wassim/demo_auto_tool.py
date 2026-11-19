"""Demo: the model calls a real calculator automatically.

This is only a demonstration of the observation_provider hook. The real
Search/Calculate tools belong to the tools owner (Federico).
"""

from simpleeval import simple_eval

from .engine import ModelEngine
from .session import AgentSession


def calculate(argument: str) -> str:
    print(f"\n>>> TOOL CALLED: Calculate[{argument}]")
    try:
        value = simple_eval(argument.replace("$", "").replace(",", "").strip())
        result = str(value)
    except Exception as exc:
        result = f"Error: {exc}"
    print(f">>> TOOL RESULT: {result}\n")
    return result


def main() -> None:
    engine = ModelEngine("qwen2.5-3b-instruct-q4_k_m.gguf")
    session = AgentSession(
        engine,
        task="gsm8k",
        mode="react",
        max_steps=3,
        observation_provider=lambda tool, argument: calculate(argument),
        on_token=lambda piece: print(piece, end="", flush=True),
    )
    result = session.run("Use the calculator to compute 2 + 2, then finish.")
    print("\n\nAnswer:", result.answer)
    print("Tool calls:", result.metrics.tool_calls)
    print("Stop reason:", result.metrics.stop_reason)
    engine.unload()


if __name__ == "__main__":
    main()
