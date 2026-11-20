import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
import torch
from transformers import BatchEncoding

from reactagenthlt.agents.experiment import run_experiment
from reactagenthlt.agents.gguf import GGUFModel
from reactagenthlt.agents.language_model import LanguageModel, TokenUsage
from reactagenthlt.evaluation import diagnose_formats, score_answer, summarize_run
from reactagenthlt.preprocessing import hotpotqa_corpus, hotpotqa_supporting_documents
from reactagenthlt.tools import SearchEngine


def scripted_generator(*outputs):
    return Mock(side_effect=[(text, TokenUsage(5, 2)) for text in outputs])


class TestExperiment(unittest.TestCase):
    def test_baselines_are_closed_book(self):
        for mode, text in [("direct", "Finish[Rome]"), ("cot", "It is in Italy.\nFinish[Rome]")]:
            with self.subTest(mode=mode):
                generate = scripted_generator(text)
                search = Mock(spec=SearchEngine)
                result = run_experiment(
                    "What is the capital of Italy?", generate, task="hotpotqa",
                    mode=mode, search_engine=search,
                )
                prompt = generate.call_args.args[0][0]["content"]
                self.assertTrue(prompt.endswith("Question: What is the capital of Italy?\n"))
                self.assertNotIn("Search[", prompt)
                search.search.assert_not_called()
                self.assertEqual(score_answer(result, "the Rome!"), {"em": 1.0, "f1": 1.0})

    def test_reiteration_and_early_stop_are_independent(self):
        for mode in ("react", "reiteration", "early_stop", "focused_react"):
            with self.subTest(mode=mode):
                generate = scripted_generator(
                    "Thought: Calculate in parts.\nStart with the sum.\nAction: Calculate[ 2 + 3 ]",
                    "Action: Calculate[1 + 1]",
                    "Action: Calculate[2 + 3]",
                    "Action: Finish[+5.00]",
                )
                result = run_experiment("What is 2 + 3?", generate, task="gsm8k", mode=mode)
                history = [call.args[0] for call in generate.call_args_list]
                prompts = [messages[-1]["content"] for messages in history]
                self.assertEqual(
                    [message["role"] for message in history[-1]],
                    ["user", "assistant"] * 3 + ["user"],
                )
                self.assertEqual("Original question:" in prompts[0], mode in {"reiteration", "focused_react"})
                self.assertIn("Observation: 5", prompts[1])
                self.assertEqual(result.steps[0].raw_argument, " 2 + 3 ")
                self.assertTrue(result.steps[2].repeated)
                forced = mode in {"early_stop", "focused_react"}
                self.assertEqual(result.steps[-1].forced, forced)
                if forced:
                    self.assertTrue(prompts[-1].endswith("Action: Finish[number]"))
                    self.assertEqual(
                        sum("Observation: 5" in message["content"] for message in history[-1]),
                        1,
                    )
                summary = summarize_run(result)
                self.assertEqual(summary["ordinary_calls"], 3 if forced else 4)
                self.assertEqual(summary["tool_calls"], 2 if forced else 3)
                self.assertEqual(summary["prompt_tokens"], 20)
                self.assertEqual(summary["completion_tokens"], 8)
                self.assertEqual(score_answer(result, "5"), {"accuracy": 1.0})

    def test_task_budgets_and_extra_finish_call(self):
        outputs = [f"Action: Calculate[{n}+1]" for n in range(9)]
        generate = scripted_generator(*outputs)
        result = run_experiment("Question?", generate, task="gsm8k", mode="focused_react")
        self.assertEqual(result.termination, "budget_exhausted")
        self.assertEqual(generate.call_count, 9)
        self.assertEqual(score_answer(result, "1"), {"accuracy": 0.0})

        search = Mock(spec=SearchEngine)
        search.search.return_value = pd.DataFrame({"document": ["Evidence"]})
        outputs = [f"Action: Search[item {n}]" for n in range(6)]
        generate = scripted_generator(*outputs, "Action: Search[ITEM, 0!]", "Action: Finish[Rome]")
        result = run_experiment("Question?", generate, task="hotpotqa", mode="early_stop", search_engine=search)
        self.assertEqual(result.termination, "forced_finish")
        self.assertEqual(generate.call_count, 8)
        self.assertEqual(search.search.call_count, 6)
        self.assertEqual(result.steps[6].argument, "item 0")

    def test_invalid_replies_end_the_run_and_remain_available(self):
        examples = {
            "No command": "missing_action",
            "```\naction: calculate (2 + 3)\n```": "missing_action",
            "Action: Calculate[2 + 3]\nAction: Finish[5]": "multiple_actions",
            "Thought: I can guess.\nObservation: 5\nAction: Finish[5]": "malformed_action",
            "Action: Search[thing]": "unsupported_action",
            "Action: Calculate[1 / 0]": "invalid_tool_input",
            "Action: Calculate[()]": "invalid_tool_input",
            "Action: Finish[five]": "invalid_final_answer",
        }
        results = []
        for text, reason in examples.items():
            with self.subTest(reason=reason, text=text):
                generate = scripted_generator(text, "Action: Finish[5]")
                result = run_experiment("Question?", generate, task="gsm8k", mode="focused_react")
                results.append(result)
                self.assertEqual(result.termination, reason)
                self.assertEqual(result.steps[0].raw_output, text)
                self.assertEqual(generate.call_count, 1)
                self.assertEqual(score_answer(result, "5"), {"accuracy": 0.0})
        report = diagnose_formats(results)
        self.assertEqual(report["affected_calls"], 3)
        self.assertEqual(report["categories"]["capitalization"], 1)
        self.assertEqual(report["categories"]["markdown_wrapper"], 1)
        self.assertEqual(report["categories"]["multiple_commands"], 1)

    def test_forced_reply_cannot_request_a_tool_or_recover(self):
        for reply, reason in [
            ("Action: Calculate[2+2]", "unsupported_action"),
            ("Action: Finish[answer]", "invalid_final_answer"),
            ("My answer is 4", "missing_action"),
        ]:
            with self.subTest(reply=reply):
                generate = scripted_generator("Action: Calculate[1+1]", "Action: Calculate[1+1]", reply)
                result = run_experiment("Question?", generate, task="gsm8k", mode="early_stop")
                self.assertEqual(result.termination, reason)
                self.assertTrue(result.steps[-1].forced)
                self.assertEqual(summarize_run(result)["tool_calls"], 1)

    def test_infrastructure_failures_are_unscored(self):
        generate = Mock(side_effect=OSError("model unavailable"))
        result = run_experiment("Question?", generate, task="gsm8k")
        self.assertEqual(result.termination, "model_error")
        self.assertIsNone(score_answer(result, "1"))

        search = Mock(spec=SearchEngine)
        search.search.side_effect = ValueError("index unavailable")
        generate = scripted_generator("Action: Search[  A Test!  ]")
        result = run_experiment("Question?", generate, task="hotpotqa", search_engine=search)
        search.search.assert_called_once_with("a test")
        self.assertEqual(result.termination, "tool_error")
        self.assertIsNone(score_answer(result, "Rome"))

    def test_selected_rows_form_one_corpus_and_support_can_be_scored(self):
        rows = pd.DataFrame([
            {"context": {"title": ["Rome", "Paris"], "sentences": [["Italy"], ["France"]]},
             "supporting_facts": {"title": ["Rome", "Paris"]}},
            {"context": {"title": ["Rome", "Lima"], "sentences": [["Italy"], ["Peru"]]},
             "supporting_facts": {"title": ["Lima"]}},
        ])
        corpus = hotpotqa_corpus(rows)
        self.assertEqual(corpus["title"].tolist(), ["Rome", "Paris", "Lima"])
        search = SearchEngine(corpus=corpus, top_k=2)
        generate = scripted_generator("Action: Search[Rome]", "Action: Search[Paris]", "Action: Finish[Rome]")
        result = run_experiment("Question?", generate, task="hotpotqa", search_engine=search)
        summary = summarize_run(result, hotpotqa_supporting_documents(rows.iloc[0]))
        self.assertEqual(summary["supporting_paragraphs_retrieved"], 2)
        self.assertTrue(summary["all_supporting_paragraphs_retrieved"])

    def test_transformers_generator_preserves_full_reply(self):
        model = LanguageModel.__new__(LanguageModel)
        model.config = {"max_new_tokens": 256}
        model.do_sample = model.stream = False
        model.stop_criterias = []
        model.tokenizer = Mock()
        model.tokenizer.apply_chat_template.return_value = BatchEncoding(
            {"input_ids": torch.tensor([[1, 2, 3]])}
        )
        raw = " Action: Finish[5]\nObservation: fabricated\nAction: Finish[9]\n"
        model.tokenizer.decode.return_value = raw
        model.model = Mock(device="cpu")
        model.model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

        result = run_experiment("Question?", model.generate, task="gsm8k")

        self.assertFalse(model.model.generate.call_args.kwargs["stopping_criteria"])
        self.assertEqual(result.steps[0].raw_output, raw)
        self.assertEqual(result.steps[0].token_usage, TokenUsage(3, 2))
        self.assertEqual(result.termination, "multiple_actions")

    def test_gguf_adapter_uses_the_same_generation_interface(self):
        llama = Mock()
        llama.return_value.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Finish[5]\n"}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 4},
        }
        with patch.dict("sys.modules", {"llama_cpp": SimpleNamespace(Llama=llama)}):
            model = GGUFModel("example.gguf", max_new_tokens=64)

        result = run_experiment("What is 2 + 3?", model.generate, task="gsm8k", mode="direct")

        self.assertEqual(result.answer, "5")
        self.assertEqual(result.steps[0].raw_output, "Finish[5]\n")
        self.assertEqual(result.steps[0].token_usage, TokenUsage(20, 4))
        self.assertEqual(llama.return_value.create_chat_completion.call_args.kwargs["max_tokens"], 64)


if __name__ == "__main__":
    unittest.main()
