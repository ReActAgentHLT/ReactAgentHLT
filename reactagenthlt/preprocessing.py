from typing import cast

import pandas as pd
from datasets import load_dataset


def hotpotqa(setting: str = "distractor", train: bool = False) -> pd.DataFrame:
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


def gsm8k(
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
        if `False` returns only the raw numerical answer, if `True` also returns
        the reasoning to build the right expression to evaluate

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
        ds["answer"] = ds["answer"].str.extract(
            r"####\s*([+-]?\d[\d,]*(?:\.\d+)?)\s*$", expand=False
        )

    return ds
