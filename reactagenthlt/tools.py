import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

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
    Search HotPotQA contexts using BM25 or dense sentence embeddings.
    """

    def __init__(
        self,
        kind: str = "sparse",
        search_by: str = "document",
        top_k: int = 3,
    ) -> None:
        """
        kind: `str`
            select BM25 ("sparse") or sentence embeddings ("dense").

        search_by: `str`
            select if to search documents by document content or by title alone.
            The title mode will search within a collection of titles while the
            document mode will search within a collection of documents composed
            by title and the actual content.

        top_k: `int`
            number of retrieved documents per query
        """
        if kind not in {"sparse", "dense"}:
            raise ValueError(
                f"{kind} is not a valid value for 'kind': possible values {'sparse', 'dense'}"
            )

        if search_by not in {"title", "document"}:
            raise ValueError(
                f"{search_by} is not a valid value for 'seach_by': possible values {'title', 'document'}"
            )

        if top_k <= 0:
            raise ValueError(
                f"{top_k} is invalid value for 'top_k': must  be greater than 0"
            )

        self.corpus = hotpotqa_corpus()
        self.kind = kind
        self.top_k = top_k

        if kind == "sparse":
            self.tokenizer = RetrievalTokenizer()
            tokenized_corpus = self.tokenizer.tokenize_corpus(self.corpus[search_by])
            self.retriever = BM25Okapi(tokenized_corpus)
        else:
            from sentence_transformers import SentenceTransformer

            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.model = SentenceTransformer(model_name)
            documents = self.corpus[search_by].tolist()

            # Include the ordered corpus and encoding settings in the cache key.
            digest = hashlib.sha256(f"{model_name}:{search_by}:normalized-v1".encode())
            for document in documents:
                data = document.encode("utf-8")
                digest.update(len(data).to_bytes(8, "big"))
                digest.update(data)
            cache_dir = Path(__file__).resolve().parent.parent / ".search_embeddings"
            cache_path = cache_dir / f"{digest.hexdigest()}.npy"

            if cache_path.exists():
                self.embeddings = np.load(cache_path, allow_pickle=False)
            else:
                self.embeddings = self.model.encode(
                    documents,
                    batch_size=64,
                    normalize_embeddings=True,
                    show_progress_bar=True,
                )
                cache_dir.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, self.embeddings)

    def search(self, query: str) -> pd.DataFrame:
        """
        Retrieve the `top_k` results for the given query
        """
        if not query.strip():
            raise ValueError("Search query cannot be empty")

        if self.kind == "sparse":
            scores = self.retriever.get_scores(self.tokenizer.tokenize(query))
        else:
            query_embedding = self.model.encode(query, normalize_embeddings=True)
            scores = self.embeddings @ query_embedding
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
