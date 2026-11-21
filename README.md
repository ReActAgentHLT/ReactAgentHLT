# ReAct Agent HLT

Comparison of **Direct**, **Chain-of-Thought (CoT)**, **ReAct**, and **Focused ReAct** agents on multi-hop QA (**HotpotQA**) and arithmetic word problems (**GSM8K**), equipped with a retrieval search engine and a calculator.

## Installation

Create and activate a Python `>=3.12` virtual environment, then install the package:

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows
pip install -e .
```

## Project Layout

- `configs/` — JSON model configurations (`gemma2-2B.json`, `qwen2.5-3B.json`, `lfm3-1.7B.json`).
- `reactagenthlt/language_model.py` — `LanguageModel` and `TokenUsage` wrapper.
- `reactagenthlt/agents/` — `BaseAgent`, `StandardAgent`, `ReActAgent`, `parsing.py`, `prompts.py`.
- `reactagenthlt/tools/` — `SearchEngine` (BM25 / dense) and `Calculator`.
- `reactagenthlt/preprocessing.py` — HotpotQA and GSM8K loaders and corpus builder.
- `reactagenthlt/evaluation.py` — Benchmark runner, scoring, and summary aggregation.
- `notebooks/` — Exploratory dataset and retrieval notebooks.
- `test/` — Unit and integration tests.

## Quick Usage

Run an evaluation from the repository root:

```bash
python -m reactagenthlt.evaluation configs/qwen2.5-3B.json --dataset hotpotqa --mode focused_react --k 3 --budget 7 --n 100 --seed 0
```

Run the test suite:

```bash
python -m unittest discover -s test -v
```

See `reactagenthlt/README.md` for full module documentation.
