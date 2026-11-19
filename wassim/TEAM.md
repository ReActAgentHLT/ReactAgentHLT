# Team integration guide

This folder is **Wassim's part**: models and inference. It is deliberately
separate from the existing benchmark code.

## What is already done

- Load a `.gguf` model with `llama.cpp` on GPU.
- Load a Hugging Face model directory with `transformers` on GPU.
- Stream text token by token.
- Stop at `Finish[...]`, at one complete `Action[...]`, or at a tool pause.
- Pause generation, wait for an observation, append it, and resume.
- Direct / CoT / ReAct / Focused ReAct prompts.
- Per-call and per-run metrics.
- Interactive CLI for manual testing.

## What is intentionally NOT done here

| Part | Owner | What they add |
|---|---|---|
| `Search` and `Calculate` | Federico | Real tool implementations |
| Dataset loading | Federico | HotPotQA / GSM8K samples |
| Production agent loop | Omar | His own controller, or reuse `AgentSession` |
| EM / F1 / accuracy tables | Nandani | Evaluation on top of `RunResult.metrics` |
| Final prompt policy | Michael | Edits `prompts.py` |

## The two interfaces

### `ModelEngine` — the brain

```python
engine = ModelEngine("qwen2.5-3b-instruct-q4_k_m.gguf")
step = engine.generate(
    prompt,
    max_new_tokens=160,
    stop_policy="action",     # "finish" or "action"
    on_token=print,
)
print(step.text)
print(step.metrics)
```

`StepMetrics` contains:

```text
prompt_tokens
completion_tokens
total_tokens
ttft_seconds
inference_seconds
tokens_per_second
stop_reason
```

### `AgentSession` — the mode-aware test controller

```python
session = AgentSession(
    engine,
    task="hotpotqa",
    mode="react",                    # direct / cot / react / focused_react
    observation_provider=my_tool,    # called only for ReAct / Focused
    on_token=print,
)
result = session.run(question)
```

`RunResult` contains:

```text
answer
metrics.prompt_tokens
metrics.completion_tokens
metrics.total_tokens
metrics.ttft_seconds
metrics.inference_seconds
metrics.tokens_per_second
metrics.num_steps
metrics.tool_calls
metrics.stop_reason
trajectory
```

## Federico — tools

The engine does not care how Search or Calculate work. It only calls your
callback with the tool name and argument, and expects a string back.

```python
from wassim import AgentSession, ModelEngine

search = Search(paragraphs)       # your tool
calculate = calculate             # your tool

def observe(tool, argument):
    if tool == "Search":
        return search(argument)
    if tool == "Calculate":
        return calculate(argument)
    raise ValueError(f"Unknown tool: {tool}")

session = AgentSession(
    engine,
    task="hotpotqa",
    mode="react",
    observation_provider=observe,
)
```

## Omar — agent loop

You have two clean options.

### Option A: reuse the test controller

`AgentSession` already implements the Thought / Action / Observation loop,
ReAct pause/resume, max steps, repeated-action detection, and Focused ReAct.
You can import it and extend it.

### Option B: keep your own controller

Use `ModelEngine` directly and keep your loop in your own file:

```python
step = engine.generate(prompt, stop_policy="action", on_token=print)
text = step.text
# parse the action yourself, call the tool, append "Observation: ..."
```

Both options use the same model backend and the same metrics. Do not change
`engine.py` or `backends.py` for agent logic; change your own controller.

## Nandani — evaluation

The engine gives you the raw counts. Your evaluation script adds gold answers,
EM, F1, and aggregate tables.

```python
from dataclasses import asdict

result = session.run(question)

row = {
    "question": question,
    "prediction": result.answer,
    **asdict(result.metrics),
}
```

`--json-out` already writes this shape to JSONL, one run per line.

## Michael — prompts

All prompt text is in `prompts.py`:

- `direct_prompt`
- `cot_prompt`
- `react_prompt`
- `focused_step_prompt`
- `forced_finish_prompt`

Change those functions without touching the engine. The model layer only needs
the final prompt string.

## Run it

```powershell
python -m wassim.cli --model qwen2.5-3b-instruct-q4_k_m.gguf
```

Then choose the task and agent from the menu. The CLI is only a manual test
harness; the real integration is `ModelEngine` + `AgentSession`.
