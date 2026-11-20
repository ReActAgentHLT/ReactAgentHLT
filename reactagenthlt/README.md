# Documentation

## Agents

The `agents` package separates model generation from agent behavior:

- `LanguageModel` loads a Hugging Face model once and exposes `generate()`.
- `BaseAgent` defines `ask()` and provides shared conversation state,
  `add_system_prompt()`, and `clear_ctx()`.
- `StandardAgent` performs one generation per chat turn.
- `ReActAgent` runs a question through the shared strict experiment loop.
  `ask()` returns a vanilla ReAct answer and model-call totals, and `run()`
  returns the full trace for a chosen condition.

Agents receive an existing `LanguageModel` rather than a configuration path:

```python
from reactagenthlt.agents import LanguageModel, ReActAgent, StandardAgent
from reactagenthlt.tools import Calculator, SearchEngine

model = LanguageModel("path/to/model-config.json")
agent = StandardAgent(model)
agent.add_system_prompt("You are a helpful assistant.")
response = agent.ask("Hello!", do_sample=False, stream=False)
print(response.output)
print(response.token_usage)
print(response.generation_time)
agent.clear_ctx()  # Keeps system prompts and model configuration.

# The same model can be shared. ReAct starts a fresh context for each question.
react_agent = ReActAgent(
    model,
    task="hotpotqa",
    search_engine=SearchEngine(kind="sparse", search_by="document", top_k=2),
    calculator=Calculator(),
)
result = react_agent.run("Your question", mode="focused_react")
print(result.answer, result.termination)
```

HotpotQA tool-using conditions require a search engine. GSM8K uses the provided
calculator or creates one. The default ordinary step limits are seven for
HotpotQA and nine for GSM8K. An explicit `max_steps` overrides the default.

Sampling and streaming are per-request options on `ask()`. For direct model
calls, `LanguageModel.generate()` also accepts `stop_criteria`, a tuple of regex
patterns. Matching text is retained in the output.

Both `ask()` methods return `Response` with `output`, `token_usage`, and
`generation_time`. `ReActAgent` sums the tokens and generation time over all
calls, including a forced finalization call when applicable. The original
`controller.run_react()` interface is documented below.

Run the interactive standard agent from the repository root:

```bash
python -m reactagenthlt.agents.standard path/to/model-config.json --sample --stream
```

## Preprocessing

For the dataset preprocessing needed for agents evaluation and search engine
knowledge base building all the functions are in `preprocessing.py` file.

---

Both HotPotQA and GSM8K have a utility function (`load_hotpotqa` and
`load_gsm8k`) to fetch them from HuggingFace and do some preprocessing.

Both accept at least 2 parameters, the first to select the _setting_
(**distractor** or **fullwiki** for HotPotQA and **main** or **socratic** for
GSM8K) and the second to choose between training and validation/test set. Also
both functions will return the dataset in the format of a `pandas` `DataFrame`.

For the HotPotQA will be returned only `question` and `answer` columns of the
dataset without any further preprocessing.

Also for GSM8K only `question` and `answer` columns will be returned but since
the `answer` field of GSM8K is in the format

```text
Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.
#### 10
```

with the actual answer after the `####` substring. So if the `reasoning`
parameter of `load_gsm8k` is set to false (default) everything is parsed away
and only the raw number is kept.

## Tools

The implemented tools are a search engine and a calculator whose implementation
can be found in the `tools.py` file.

### Search Engine

The `SearchEngine` class doesn't need to be fed with a corpus since we assume
that the only needed corpus is a collection based on the `context` field present
in the HotPotQA dataset. Therefore, when instanciated, the `SearchEngine` class
will automatically build a knowledge base and keep it as internal state.

---

The knowledge based is built by checking for each `context` of the HotPotQA
dataset the titles and their corresponding lists of sentences. A document is
built by a unique string of all the sentences concatenated and the title on top:

```text
{title}

{content}
```

A copy of the title is also stored separately for other purposes (see later).

---

The parameters are `kind`, `search_by` and `top_k`:

- `kind`: `"sparse"` (the default, BM25) or `"dense"` (Sentence Transformers).
- `search_by` can have `"title"` or `"document"` as allowed values. The first
  will search within a corpus of titles only while the second within a corpus of
  documents composed by one string that includes both title and content of the
  document.
- `top_k`: the number of documents to be returned.

The `search` method accepts a string query and returns a `DataFrame` with
3 columns and up to `top_k` rows.
For each row the records are the title, the content and whole document (title +
content concatenated in one string).

| title            | content                        | document                                          |
| ---------------- | ------------------------------ | ------------------------------------------------- |
| "Doctor Strange" | "Doctor Strange is a movie..." | "Doctor Strange ... Doctor Strange is a movie..." |

```python
se = SearchEngine(kind="sparse", search_by="document", top_k=3)
results = se.search("doctor strange")

print(results["document"].iloc[0]) # will print the first retrieved document
```

Dense retrieval uses `sentence-transformers/all-MiniLM-L6-v2`. The model is
cached by Hugging Face; corpus embeddings are saved in `.search_embeddings/` at
the project root and reused when the model name, search mode, and ordered
corpus match. Delete that directory to force re-encoding, for example after
updating the model weights.

### Calculator

The `Calculator` class is based on the Python built-in `eval` function that
receives in input arbitrary python code as a string and it will execute it.

> [!warning]
> If your agent produces `sudo rm -rf /` your OS could be deleted :)

To prevent anything that is not a mathematical expression to be executed, a
regex will check the input first, allowing only for valid scalar mathematical
expression with the 4 main operators (+, -, \* and /).

```python
calculator = Calculator()
result = calculator.calculate("3 + 1")
print(result) # should be 4 (I guess...)
```

## ReAct Controller

The ReAct Controller is implemented in `controller.py`. It manages the interaction between a language model and the existing SearchEngine and Calculator tools.

The controller follows the ReAct workflow:

**Thought → Action → Observation → Repeat → Finish**

At each step, the controller receives text from the language model, extracts an action, executes the corresponding tool, and adds the resulting observation to the context.

The loop continues until the model produces a final answer or the maximum number of steps is reached.

### Supported Actions

The controller supports three actions:

| Action                  | Description                                                  |
| ----------------------- | ------------------------------------------------------------ |
| `Search[query]`         | Retrieve relevant documents using SearchEngine.              |
| `Calculate[expression]` | Evaluate a supported arithmetic expression using Calculator. |
| `Finish[answer]`        | Stop the loop and return the final answer.                   |

A model response must contain exactly one valid `Action:` line.

For example:

```text
Thought: I need to calculate the sum.
Action: Calculate[15 + 18]
```

### How to Use

Import the `run_react()` function:

```python
from reactagenthlt.controller import run_react
```

The controller receives:

- `question`: The question to answer.
- `generate`: A callable that receives the current context and returns a model-generated string.
- `search_engine`: An optional initialized SearchEngine instance.
- `max_steps`: Maximum number of model calls (default: 7).

The `generate` callable allows different language models to be connected without changing the controller.

#### Example: Calculator

The following example uses predefined model responses to demonstrate the complete ReAct loop without loading a language model.

```python
from reactagenthlt.controller import run_react

responses = iter([
    "Thought: I should calculate the sum.\nAction: Calculate[15 + 18]",
    "Thought: The result is 33.\nAction: Finish[33]",
])

def generate(context: str) -> str:
    return next(responses)

answer = run_react(
    question="What is 15 + 18?",
    generate=generate,
)

print(answer)
```

Expected output:

```text
33
```

#### Example: Search Engine Integration

The controller accepts either retrieval mode through the same SearchEngine interface:

```python
from reactagenthlt.tools import SearchEngine
from reactagenthlt.controller import run_react

search_engine = SearchEngine(
    kind="dense",  # Use "sparse" for BM25.
    search_by="document",
    top_k=3,
)

# Supply your own generate(context) function for the language model.
answer = run_react("Your question", generate, search_engine=search_engine)
```

Pass this initialized instance through the `search_engine` parameter when calling `run_react()`.

The controller handles `Search[query]` actions and adds the retrieved documents to the context as observations.

The SearchEngine loads the HotpotQA-based corpus and builds its retrieval index during initialization.

### Error Handling

The controller includes basic input validation and error handling.

- Invalid action formats raise `ValueError`.
- Empty questions are rejected.
- Expected tool `ValueError` exceptions are returned as error observations.
- Search actions require an initialized SearchEngine.
- The agent raises `RuntimeError` if the maximum number of steps is reached without a `Finish` action.

### Testing

Tests are implemented in `test/test_controller.py` using Python's `unittest` framework.

Run the controller tests from the repository root:

```bash
python -m unittest discover -s test -p "test_controller.py" -v
```

The current implementation has **19 passing tests**, covering action parsing, tool integration, the ReAct loop, context updates, stopping conditions, and error handling.

Integration with the actual SearchEngine has also been tested successfully using predefined model responses.

**Note:** These tests do not yet verify end-to-end reasoning with a real language model. Model integration and benchmark evaluation are separate project tasks.
