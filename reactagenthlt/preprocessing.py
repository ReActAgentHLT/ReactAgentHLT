from typing import cast

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


GSM8K_NUMBER = r"[+-]?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?"


def load_hotpotqa(
    setting: str = "distractor", train: bool = False, *, include_context: bool = False
) -> pd.DataFrame:
    """
    Fetch HotPotQA dataset from HuggingFace and returns it as a `pandas`
    `DataFrame`

    ## Parameters
    setting: `str`
        select the HotPotQA setting between `distractor` and `fullwiki`

    train: `bool`
        select if to return the `train` or `validation` split of the dataset

    ## Returns
    Questions and answers. With `include_context`, also return the question ID,
    original context, and supporting facts for corpus building and evaluation.
    """
    choices = {"distractor", "fullwiki"}
    if setting not in choices:
        raise ValueError(f"'{setting}' setting is invalid, possible choices: {choices}")

    split = "validation" if not train else "train"
    ds = cast(
        pd.DataFrame,
        load_dataset("hotpotqa/hotpot_qa", setting)[split].to_pandas(),
    )

    columns = ["question", "answer"]
    if include_context:
        columns.extend(["id", "context", "supporting_facts"])
    return ds[columns]


def load_gsm8k(
    setting: str = "main",
    train: bool = False,
    reasoning: bool = False,
) -> pd.DataFrame:
    """
    Fetch GSM8K dataset from HuggingFace and returns it as a `pandas`
    `DataFrame`

    ## Parameters
    setting: `str`
        select the GSM8K setting between `main` and `socratic`

    train: `bool`
        select if to return the `train` or `validation` split of the dataset

    reasoning: `bool`
        if `False` returns only the numerical answer as a string, with thousands
        separators removed; if `True` preserves the full solution

    ## Returns
    A `DataFrame` containing the dataset with only `question` and `answer`
    fields
    """
    choices = {"main", "socratic"}
    if setting not in choices:
        raise ValueError(f"'{setting}' setting is invalid, possible choices: {choices}")

    split = "test" if not train else "train"
    ds = cast(
        pd.DataFrame,
        load_dataset("openai/gsm8k", setting)[split].to_pandas(),
    )

    if not reasoning:
        answers = ds["answer"].str.extract(
            rf"####\s*({GSM8K_NUMBER})\s*\Z",
            expand=False,
        )
        invalid = answers.isna()
        if invalid.any():
            examples = ds.loc[invalid, "answer"].head(3).to_dict()
            raise ValueError(
                f"Invalid GSM8K answer format in {int(invalid.sum())} row(s) "
                f"(setting={setting!r}, split={split!r}). Expected '#### <number>' "
                f"at the end, with optional sign, decimal point, and correctly "
                f"grouped thousands commas. Example rows (index: answer): {examples!r}"
            )
        # OpenAI's reference evaluator treats commas as thousands separators.
        ds["answer"] = answers.str.replace(",", "", regex=False)

    return ds


def _hotpotqa_documents(context):
    """Yield titles, bodies and rendered passages from one question's context."""
    for title, sentences in zip(context["title"], context["sentences"]):
        content = "".join(sentence.removeprefix(" ") + "\n" for sentence in sentences)
        yield title, content, f"{title}\n\n{content}"


def hotpotqa_supporting_documents(row: pd.Series) -> set[str]:
    """Return the titled passages containing the question's gold supporting facts."""
    titles = set(row["supporting_facts"]["title"])
    return {
        document for title, _, document in _hotpotqa_documents(row["context"])
        if title in titles
    }


def hotpotqa_corpus(rows: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Pool the source and distractor passages across the selected rows.
    By default, use the full distractor validation split.

    ## Returns
    A `DataFrame` containing all the paragraphs contained in each context field
    without duplicates.
    """
    if rows is None:
        rows = load_dataset("hotpotqa/hotpot_qa", "distractor")["validation"].to_pandas()
    documents = []
    for context in tqdm(rows["context"], desc="building corpus", ncols=80):
        documents.extend(_hotpotqa_documents(context))
    return pd.DataFrame(documents, columns=["title", "content", "document"]).drop_duplicates(
        ignore_index=True
    )
