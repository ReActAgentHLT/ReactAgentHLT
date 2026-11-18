"""Exercise dense retrieval through the controller without external downloads."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from reactagenthlt.controller import run_react
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
        corpus = pd.DataFrame({
            "title": ["Apple", "River"],
            "content": ["A fruit.", "Water flows."],
            "document": ["Apple\n\nA fruit.", "River\n\nWater flows."],
        })
        FakeSentenceTransformer.document_encodings = 0
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("reactagenthlt.tools.__file__", str(Path(temp_dir) / "pkg" / "tools.py")),
            patch("reactagenthlt.tools.hotpotqa_corpus", return_value=corpus),
            patch.dict("sys.modules", {"sentence_transformers": SimpleNamespace(
                SentenceTransformer=FakeSentenceTransformer
            )}),
        ):
            engine = SearchEngine(kind="dense", top_k=1)
            # A second engine should load the saved document embeddings.
            cached_engine = SearchEngine(kind="dense", top_k=1)
            self.assertEqual(FakeSentenceTransformer.document_encodings, 1)
            self.assertEqual(cached_engine.search("water").iloc[0]["title"], "River")

            contexts = []
            responses = iter(["Action: Search[water]", "Action: Finish[River]"])

            def generate(context):
                contexts.append(context)
                return next(responses)

            answer = run_react("Where does water flow?", generate, engine)
            self.assertEqual(answer, "River")
            self.assertIn("Observation: River\n\nWater flows.", contexts[1])


if __name__ == "__main__":
    unittest.main()
