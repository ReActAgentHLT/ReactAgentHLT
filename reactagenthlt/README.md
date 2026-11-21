# Package Documentation

## Language Model (`reactagenthlt.language_model`)

- `language_model.py` (`LanguageModel`, `LlamaCppLanguageModel`, `TokenUsage`): provides a uniform `generate()` / `reset_cache()` interface across two backends:
  - **`transformers`** (default): loads a Hugging Face causal LM from a JSON config file, manages KV-cache reuse across turns, and forwards optional `"chat_template_kwargs"` to `apply_chat_template()` (used e.g. to disable native thinking in `configs/lfm3-1.7B.json`).
  - **`llama_cpp`**: connects to a local `llama-server` (`llama.cpp`) via JSON config (`"backend": "llama_cpp"`, `"base_url": "http://127.0.0.1:8080"`) or directly with `LlamaCppLanguageModel(base_url, model, max_new_tokens)`, supporting prefix KV caching, exact token accounting, and regex stop criteria.

## Agents (`reactagenthlt.agents`)

The `agents` package separates action parsing, prompts, and agent loops:

- `base.py` (`BaseAgent`, `Response`): abstract agent interface with shared `system_prompts`, `history`, `add_system_prompt()`, and `clear_ctx()`.
- `standard.py` (`StandardAgent`): single-turn chat agent (`python -m reactagenthlt.agents.standard <config>`).
- `parsing.py` (`parse_action`, `InvalidOutput`, `validate_final_answer`, `normalize_tool_argument`, `truncate_after_action`): strict parser for `Action: Name[arg]` (ReAct modes) and `Finish[answer]` (Direct / CoT baselines).
- `prompts.py`: few-shot prompts for `direct`, `cot`, `react`, and Focused ReAct (`REITERATE`, `FORCE_FINISH`).
- `react.py` (`ReActAgent`, `run_react`, `RunResult`, `StepTrace`): runs the strict ReAct / Focused ReAct loop and baselines.

```python
from reactagenthlt import LanguageModel
from reactagenthlt.agents import ReActAgent, StandardAgent
from reactagenthlt.tools import Calculator, SearchEngine

model = LanguageModel("configs/qwen2.5-3B.json")
agent = StandardAgent(model)
response = agent.ask("Hello!", do_sample=False, stream=False)

react_agent = ReActAgent(
    model,
    search_engine=SearchEngine(kind="sparse", search_by="document", top_k=3),
    calculator=Calculator(),
)
result = react_agent.run("Your question", mode="focused_react")
print(result.answer, result.termination)
```

Supported actions:
- `Search[query]` — retrieves top-`k` passages via `SearchEngine`.
- `Calculate[expression]` — evaluates an arithmetic expression via `Calculator`.
- `Finish[answer]` — terminates the loop and returns the final answer.

## Evaluation (`reactagenthlt.evaluation`)

Run a single evaluation from the repository root:

```bash
python -m reactagenthlt.evaluation configs/qwen2.5-3B.json --dataset hotpotqa --mode focused_react --k 3 --budget 7 --n 100 --seed 0
```

Use `--help` for all options. Results are saved as per-question CSVs in `results/` by default (ignored by git).

Programmatic API:
- `evaluate(model, dataset, k, budget, *, mode="react", n=None, seed=0)` → per-question `DataFrame`.
- `summarize(df)` → aggregated metrics per `(model, dataset, mode, k, budget)`.
- `save_results(df, directory=RESULTS_DIR)` → writes a timestamped CSV.

Metrics per question:
- Common: `actions`, `failed_action`, `seconds`, `input_tokens` (excluding KV-cached prefix), `output_tokens`, `correct`.
- HotpotQA: `em`, `precision`, `recall`, `f1` (official normalization; `correct` when `f1 >= 0.5`), `searches`, `gold_searches`.
- GSM8K: numerical match on the last number in the prediction (with thousands commas stripped).

## Preprocessing (`reactagenthlt.preprocessing`)

- `load_hotpotqa(setting="distractor", train=False, *, include_context=False)` → `DataFrame` with `question`, `answer` (plus `id`, `context`, `supporting_facts` when `include_context=True`).
- `load_gsm8k(setting="main", train=False, reasoning=False)` → `DataFrame` with `question`, `answer` (parsed after `####` unless `reasoning=True`).
- `hotpotqa_corpus(rows=None)` → deduplicated `DataFrame` of `title`, `content`, `document`.
- `hotpotqa_supporting_documents(row)` → set of gold supporting passages for a HotpotQA row.

## Tools (`reactagenthlt.tools`)

- `SearchEngine(kind="sparse", search_by="document", top_k=3, *, corpus=None)`: retrieves top-`k` passages using BM25 (`"sparse"`) or `sentence-transformers/all-MiniLM-L6-v2` (`"dense"`, cached in `.search_embeddings/`).
- `Calculator()`: evaluates arithmetic expressions (`+`, `-`, `*`, `/`, parentheses) after strict regex validation.

## Testing

```bash
python -m unittest discover -s test -v
```
