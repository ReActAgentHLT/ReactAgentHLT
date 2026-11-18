"""Unit tests for dataset preprocessing without network access."""

import unittest
from unittest.mock import Mock, patch

import pandas as pd
from pandas.testing import assert_frame_equal

from reactagenthlt.preprocessing import hotpotqa_corpus, load_gsm8k, load_hotpotqa


def dataset_split(frame):
    """Create a minimal Hugging Face split with a to_pandas method."""
    split = Mock()
    split.to_pandas.return_value = frame.copy(deep=True)
    return split


@patch("reactagenthlt.preprocessing.load_dataset")
class TestHotpotQA(unittest.TestCase):
    def test_settings_splits_and_selected_columns(self, load_dataset):
        train = pd.DataFrame({"question": ["Train?"], "answer": ["Yes"], "extra": [1]})
        validation = pd.DataFrame({"question": ["Validation?"], "answer": ["No"], "extra": [2]})
        load_dataset.return_value = {
            "train": dataset_split(train),
            "validation": dataset_split(validation),
        }
        for setting in ("distractor", "fullwiki"):
            for use_train, frame in ((True, train), (False, validation)):
                with self.subTest(setting=setting, train=use_train):
                    result = load_hotpotqa(setting=setting, train=use_train)
                    assert_frame_equal(result, frame[["question", "answer"]])
                    load_dataset.assert_called_with("hotpotqa/hotpot_qa", setting)

    def test_invalid_setting_does_not_load_dataset(self, load_dataset):
        with self.assertRaises(ValueError):
            load_hotpotqa(setting="invalid")
        load_dataset.assert_not_called()

    def test_missing_required_column_is_rejected(self, load_dataset):
        load_dataset.return_value = {"validation": dataset_split(pd.DataFrame({"question": ["Q"]}))}
        with self.assertRaises(KeyError):
            load_hotpotqa()

    def test_corpus_builds_and_deduplicates_documents(self, load_dataset):
        context = {
            "title": ["First", "Second"],
            "sentences": [[" First sentence.", " Second sentence."], ["Other."]],
        }
        frame = pd.DataFrame({"context": [context, context]})
        load_dataset.return_value = {"validation": dataset_split(frame)}

        result = hotpotqa_corpus()

        expected = pd.DataFrame({
            "title": ["First", "Second"],
            "content": ["First sentence.\nSecond sentence.\n", "Other.\n"],
            "document": ["First\n\nFirst sentence.\nSecond sentence.\n", "Second\n\nOther.\n"],
        })
        assert_frame_equal(result, expected)
        load_dataset.assert_called_once_with("hotpotqa/hotpot_qa", "distractor")


@patch("reactagenthlt.preprocessing.load_dataset")
class TestGSM8K(unittest.TestCase):
    def test_extracts_numbers_and_removes_thousands_separators(self, load_dataset):
        frame = pd.DataFrame({
            "question": ["A?", "B?", "C?", "D?", "E?"],
            "answer": ["Work\n#### 42", "#### -1,234.50", "#### +0.5", "#### 1234", "#### 0\n"],
        })
        load_dataset.return_value = {"test": dataset_split(frame)}
        result = load_gsm8k()
        self.assertEqual(result["answer"].tolist(), ["42", "-1234.50", "+0.5", "1234", "0"])
        self.assertEqual(result["question"].tolist(), frame["question"].tolist())
        load_dataset.assert_called_once_with("openai/gsm8k", "main")

    def test_settings_splits_and_reasoning_preservation(self, load_dataset):
        train = pd.DataFrame({"question": ["Train?"], "answer": ["Steps\n#### 10"]})
        test = pd.DataFrame({"question": ["Test?"], "answer": ["Reasoning\n#### 20"]})
        load_dataset.return_value = {"train": dataset_split(train), "test": dataset_split(test)}
        for setting in ("main", "socratic"):
            for use_train, frame in ((True, train), (False, test)):
                with self.subTest(setting=setting, train=use_train):
                    result = load_gsm8k(setting=setting, train=use_train, reasoning=True)
                    assert_frame_equal(result, frame)
                    load_dataset.assert_called_with("openai/gsm8k", setting)

    def test_invalid_setting_does_not_load_dataset(self, load_dataset):
        with self.assertRaises(ValueError):
            load_gsm8k(setting="invalid")
        load_dataset.assert_not_called()

    def test_invalid_answer_records_are_reported(self, load_dataset):
        for answer in ("no marker 42", "#### 1,23", "#### 3.5 dollars", "####", None):
            with self.subTest(answer=answer):
                frame = pd.DataFrame({"question": ["Good?", "Bad?"], "answer": ["#### 1", answer]})
                load_dataset.return_value = {"test": dataset_split(frame)}
                with self.assertRaisesRegex(ValueError, r"Invalid GSM8K answer format in 1 row\(s\)"):
                    load_gsm8k()

    def test_missing_answer_column_is_rejected(self, load_dataset):
        load_dataset.return_value = {"test": dataset_split(pd.DataFrame({"question": ["Q"]}))}
        with self.assertRaises(KeyError):
            load_gsm8k()


if __name__ == "__main__":
    unittest.main()
