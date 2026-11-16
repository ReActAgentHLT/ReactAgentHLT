import pandas as pd


class KnowledgeBase:
    def __init__(self) -> None:
        pass


class SearchEngine:
    def __init__(self, k: int = 10) -> None:
        self.k = k
        self.kb = KnowledgeBase()

    def search(self, query: str) -> list[dict[str, str]]:
        return []
