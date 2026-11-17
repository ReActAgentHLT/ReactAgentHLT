from typing import cast

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


def load_hotpotqa(setting: str = "distractor", train: bool = False) -> pd.DataFrame:
    """
    Fetch HotPotQA dataset from HuggingFace and returns it as a `pandas`
    `DataFrame`

    ## Parameters
    setting: `str`
        select the HotPotQA setting between `distractor` and `fullwiki`

    train: `bool`
        select if to return the `train` or `validation` split of the dataset

    ## Returns
    A `DataFrame` containing the dataset with only `question` and `answer`
    fields
    """
    choices = {"distractor", "fullwiki"}
    if setting not in choices:
        raise ValueError(f"'{setting}' setting is invalid, possible choices: {choices}")

    split = "validation" if not train else "train"
    ds = cast(
        pd.DataFrame,
        load_dataset("hotpotqa/hotpot_qa", setting)[split].to_pandas(),
    )

    return ds[["question", "answer"]]


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
            r"####\s*([+-]?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?)\s*\Z",
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


def hotpotqa_corpus() -> pd.DataFrame:
    """
    Build a knowledge base using the `context` field of HotPotQA in the
    `distractor` setting and using the `validation` split to simulate a real web
    search.

    ## Returns
    A `DataFrame` containing all the paragraphs contained in each context field
    without duplicates.
    """
    ds = cast(
        pd.DataFrame,
        load_dataset("hotpotqa/hotpot_qa", "distractor")["validation"].to_pandas(),
    )

    out = {"title": [], "content": [], "document": []}
    for context in tqdm(ds["context"], desc="building corpus", ncols=80):
        titles = context["title"]  # list of titles
        sentences = context["sentences"]  # list of lists of paragraphs

        for title, paragraphs in zip(titles, sentences):
            content = ""
            for paragraph in paragraphs:
                # some paragraphs begins with a whitespace -> remove
                content += paragraph.removeprefix(" ") + "\n"

            out["title"].append(title)
            out["content"].append(content)
            out["document"].append(title + "\n\n" + content)

    df = pd.DataFrame(out)

    return df.drop_duplicates(subset=["title", "content", "document"]).reset_index(
        drop=True
    )
