"""Interactive terminal for the wassim inference package.

The CLI is only a test harness. The reusable pieces are ModelEngine (the brain)
and AgentSession (it chooses Direct/CoT/ReAct/Focused behavior from the mode).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .engine import ModelEngine
from .session import AgentSession

TASKS = {
    "hotpotqa": "HotPotQA - multi-hop QA; ReAct uses Search",
    "gsm8k": "GSM8K - math word problems; ReAct uses Calculate",
}
MODES = {
    "direct": "Direct - answer only; no reasoning, no tools",
    "cot": "Chain of Thought - reason step by step; no tools",
    "react": "ReAct - Thought / Action / Observation; uses Search or Calculate",
    "focused_react": "Focused ReAct - ReAct + question reminder, loop control, forced finish",
}
TOOL_FOR_TASK = {"hotpotqa": "Search", "gsm8k": "Calculate"}


def choose(title: str, options: dict[str, str], default: str | None = None) -> str:
    """Show a numbered menu and return one of the option keys."""
    keys = list(options)
    print(f"\n{title}")
    for number, key in enumerate(keys, 1):
        print(f"  {number}) {key:<14} {options[key]}")
    suffix = f" [{default}]" if default else ""
    while True:
        try:
            raw = input(f"Choose 1-{len(keys)}{suffix}: ").strip()
        except EOFError:
            if default:
                return default
            return keys[0]
        if not raw and default:
            return default
        if raw in options:
            return raw
        if raw.isdigit() and 1 <= int(raw) <= len(keys):
            return keys[int(raw) - 1]
        print("Invalid choice, try again.")


def choose_model(model_arg: str | None) -> str:
    """Use --model if given; otherwise offer the local Qwen file or a custom path."""
    if model_arg:
        return model_arg

    candidate = Path(__file__).resolve().parent.parent / "qwen2.5-3b-instruct-q4_k_m.gguf"
    if candidate.exists():
        print("\nModel")
        print(f"  1) Qwen GGUF  {candidate}")
        print("  2) Enter another .gguf file or Hugging Face directory")
        try:
            choice = input("Choose 1-2 [1]: ").strip() or "1"
        except EOFError:
            choice = "1"
        if choice == "1":
            return str(candidate)
        if choice != "2":
            print("Invalid choice; using the Qwen GGUF.")
            return str(candidate)

    while True:
        path = input("Model path (.gguf or HF directory): ").strip().strip('"')
        if path:
            return path
        print("A model path is required.")


def ask_observation(tool: str, argument: str) -> str:
    print(f"\n[{tool}] {argument}")
    print("Paste the observation, then a blank line:")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "":
            break
        lines.append(line)
    return "\n".join(lines).strip() or "(empty observation)"


def print_summary(result, agent: str, tools: str) -> None:
    m = result.metrics
    print("\n" + "=" * 46)
    print("Performance Summary")
    print("=" * 46)
    print(f"Agent:             {agent}")
    print(f"Tools:             {tools}")
    print(f"Answer:            {result.answer or '(no answer)'}")
    print(f"Prompt tokens:     {m.prompt_tokens}")
    print(f"Completion tokens: {m.completion_tokens}")
    print(f"Total tokens:      {m.total_tokens}")
    print(f"TTFT:              {m.ttft_seconds:.3f} s")
    print(f"Inference time:    {m.inference_seconds:.3f} s")
    print(f"Tokens/sec:        {m.tokens_per_second:.2f}")
    print(f"Steps:             {m.num_steps}")
    print(f"Tool calls:        {len(m.tool_calls)}")
    print(f"Stop reason:       {m.stop_reason}")
    print("=" * 46)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the wassim model/inference layer")
    parser.add_argument("--model", help=".gguf file or HF model directory; omit to choose interactively")
    parser.add_argument("--task", choices=list(TASKS), help="hotpotqa or gsm8k; omit to choose")
    parser.add_argument(
        "--mode",
        "--agent",
        dest="mode",
        choices=list(MODES),
        help="direct, cot, react, or focused_react; omit to choose",
    )
    parser.add_argument("--question", help="one question, then exit")
    parser.add_argument("--max-steps", type=int, default=7)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--n-ctx", type=int, default=8192)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--json-out", help="append each run summary to this JSONL file")
    args = parser.parse_args()

    model_path = choose_model(args.model)
    task = args.task or choose("Task", TASKS, default="hotpotqa")
    agent = args.mode or choose("Agent", MODES, default="react")
    uses_tools = agent in ("react", "focused_react")
    tools = TOOL_FOR_TASK[task] if uses_tools else "none"

    print("\nSelected:")
    print(f"  model: {model_path}")
    print(f"  task:  {task}")
    print(f"  agent: {agent}")
    print(f"  tools: {tools if uses_tools else 'none (closed book)'}")

    try:
        engine = ModelEngine(model_path, n_ctx=args.n_ctx, device_map=args.device_map)
    except Exception as exc:
        print(f"\nFailed to load model: {type(exc).__name__}: {exc}")
        return

    info = engine.info()
    print("\nModel info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    if info.get("cpu_offload"):
        print("WARNING: some layers are on CPU; generation will be slow.")
    if info.get("backend") == "transformers" and info.get("chat_template") is False:
        print("WARNING: this tokenizer has no chat template; the raw prompt is used.")

    observation_provider = ask_observation if uses_tools else None

    def run_one(question: str) -> None:
        print("\n" + "-" * 46)
        print(f"Question: {question}")
        print(f"Agent: {agent} | Tools: {tools if uses_tools else 'none'}")
        print("-" * 46)

        def on_token(piece: str) -> None:
            print(piece, end="", flush=True)

        session = AgentSession(
            engine,
            task,
            agent,
            max_steps=args.max_steps,
            max_new_tokens=args.max_new_tokens,
            on_token=on_token,
            observation_provider=observation_provider,
        )
        result = session.run(question)
        print_summary(result, agent, tools if uses_tools else "none")

        if args.json_out:
            payload = {
                "task": task,
                "agent": agent,
                "tools": tools if uses_tools else "none",
                "question": question,
                "answer": result.answer,
                "metrics": asdict(result.metrics),
                "trajectory": result.trajectory,
            }
            with Path(args.json_out).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            print(f"Saved summary to {args.json_out}")

    try:
        if args.question:
            run_one(args.question)
        else:
            print("\nType a question. Empty line quits.")
            while True:
                try:
                    question = input("\nQuestion: ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not question:
                    break
                run_one(question)
    finally:
        engine.unload()


if __name__ == "__main__":
    main()
