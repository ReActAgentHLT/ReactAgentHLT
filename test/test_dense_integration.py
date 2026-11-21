"""Exercise dense retrieval through the ReAct loop without external downloads."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from reactagenthlt.language_model import TokenUsage
from reactagenthlt.agents.react import run_react
from reactagenthlt.tools import SearchEngine


class FakeSentenceTransformer:
    document_encodings = 0

    def __init__(self, model_name):
        pass

    def encode(self, text, **kwargs):
        if isinstance(text, list):
            type(self).document_encodings += 1
            return np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        return np.array([0.0, 1.0], dtype=np.float32)


class TestDenseControllerIntegration(unittest.TestCase):
    def test_dense_search_observation_and_cache(self):
        corpus = pd.DataFrame(
            {
                "title": ["Apple", "River"],
                "content": ["A fruit.", "Water flows."],
                "document": ["Apple\n\nA fruit.", "River\n\nWater flows."],
            }
        )
        FakeSentenceTransformer.document_encodings = 0
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch(
                "reactagenthlt.tools.search_engine.__file__",
                str(Path(temp_dir) / "pkg" / "tools" / "search_engine.py"),
            ),
            patch("reactagenthlt.tools.search_engine.hotpotqa_corpus", return_value=corpus),
            patch(
                "reactagenthlt.tools.search_engine.SentenceTransformer",
                FakeSentenceTransformer,
            ),
        ):
            engine = SearchEngine(kind="dense", top_k=1)
            # A second engine should load the saved document embeddings.
            cached_engine = SearchEngine(kind="dense", top_k=1)
            self.assertEqual(FakeSentenceTransformer.document_encodings, 1)
            self.assertEqual(cached_engine.search("water").iloc[0]["title"], "River")

            messages = []
            responses = iter(["Action: Search[water]", "Action: Finish[River]"])

            def generate(chat):
                messages.append(chat)
                return next(responses), TokenUsage(0, 0)

            result = run_react(
                "Where does water flow?", generate, task="hotpotqa", search_engine=engine
            )
            self.assertEqual(result.answer, "River")
            self.assertIn("Observation: River\n\nWater flows.", messages[1][-1]["content"])


if __name__ == "__main__":
    unittest.main()
