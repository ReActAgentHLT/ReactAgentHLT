# Documentation

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

The only parameters are `search_by` and `top_k`:

- `search_by` can have `"title"` or `"document"` as allowed values. The first
  will search within a corpus of titles only while the second within a corpus of
  documents composed by one string that includes both title and content of the
  document.
- `top_k`: the number of documents to be returned.

The only method is `search` that accepts a string as a query (the query will be
tokenized internally) and returns a `DataFrame` with 3 columns and `top_k` rows.
For each row the records are the title, the content and whole document (title +
content concatenated in one string).

| title            | content                        | document                                          |
| ---------------- | ------------------------------ | ------------------------------------------------- |
| "Doctor Strange" | "Doctor Strange is a movie..." | "Doctor Strange ... Doctor Strange is a movie..." |

```python
se = SearchEngine(search_by="document", top_k=3)
results = se.search("doctor strange")

print(results["document"].iloc[0]) # will print the first retrieved document
```

#### Future Updates

From the same exact interface should be possible in future to have an additional
parameter like `retrieval_type` that can be `"sparse"` or `"dense"` but for now
the only sparse version based in BM25 is available.

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
