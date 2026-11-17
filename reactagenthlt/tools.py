import re

import numpy as np
import pandas as pd
import rank_bm25

from reactagenthlt.preprocessing import hotpotqa_corpus


# ======== RETRIEVAL ========
class RetrievalTokenizer:
    """
    Simple regex-based tokenizer for the `SearchEngine` class
    """

    def __init__(self):
        self.pattern = re.compile(r"\w+", flags=re.UNICODE)

    def tokenize(self, document: str) -> list[str]:
        return self.pattern.findall(document.lower())

    def tokenize_corpus(self, corpus: pd.Series) -> list[list[str]]:
        return corpus.apply(self.tokenize).to_list()


class SearchEngine:
    """
    Minimal search engine implementation based on `rank_bm25` library that
    embeds by default a collection of documents formed on the base of the
    `context` field of the HotPotQA dataset.
    """

    def __init__(self, search_by: str = "document", top_k: int = 3) -> None:
        """
        search_by: `str`
            select if to search documents by document content or by title alone.
            The title mode will search within a collection of titles while the
            document mode will search within a collection of documents composed
            by title and the actual content.

        top_k: `int`
            number of retrieved documents per query
        """
        if search_by not in {"title", "document"}:
            raise ValueError(
                f"{search_by} is not a valid value for 'seach_by': possible values {'title', 'document'}"
            )

        self.corpus = hotpotqa_corpus()
        self.tokenizer = RetrievalTokenizer()
        self.top_k = top_k

        tokenized_corpus = self.tokenizer.tokenize_corpus(self.corpus[search_by])
        self.retriever = rank_bm25.BM25Okapi(tokenized_corpus)

    def search(self, query: str) -> pd.DataFrame:
        """
        Retrieve the `top_k` results for the given query
        """
        if not query.strip():
            raise ValueError("Search query cannot be empty")

        scores = self.retriever.get_scores(self.tokenizer.tokenize(query))
        indices = np.argsort(scores)[::-1][: self.top_k]

        return self.corpus.iloc[indices]


# ======== MATH CALCULATOR ========
class Calculator:
    """Simple calculator implementation with built-in `eval` function"""

    def calculate(self, expression: str) -> int | float:
        """
        evaluate the expression returning the result. A preliminary check on the
        validity of the expression is made and will result in an `ValueError`
        exception if not satisfied.
        """
        pattern = re.compile(r"[0-9+*/.()\s-]+", flags=re.ASCII)
        if not pattern.fullmatch(expression):
            raise ValueError(
                "Only numbers, arithmetic operators, and parentheses are allowed"
            )

        return eval(expression, {"__builtins__": {}})
