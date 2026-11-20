import unittest
from functools import partial
from unittest.mock import Mock

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, BatchEncoding, Gemma2Config, Qwen2Config
from transformers.generation import GenerateDecoderOnlyOutput

from reactagenthlt.agents.react import parse_action, run_react
from reactagenthlt.agents.language_model import LanguageModel, TokenUsage
from reactagenthlt.evaluation import diagnose_formats, score_answer, summarize_run
from reactagenthlt.preprocessing import hotpotqa_corpus, hotpotqa_supporting_documents
from reactagenthlt.tools import SearchEngine


def scripted_generator(*outputs):
    return Mock(side_effect=[(text, TokenUsage(5, 2)) for text in outputs])


class TestExperimentRules(unittest.TestCase):
    def test_baselines_are_closed_book(self):
        for mode, text in [("direct", "Finish[Rome]"), ("cot", "It is in Italy.\nFinish[Rome]")]:
            with self.subTest(mode=mode):
                generate = scripted_generator(text)
                search = Mock(spec=SearchEngine)
                result = run_react(
                    "What is the capital of Italy?", generate, task="hotpotqa",
                    mode=mode, search_engine=search,
                )
                prompt = generate.call_args.args[0][0]["content"]
                self.assertTrue(prompt.endswith("Question: What is the capital of Italy?\n"))
                self.assertNotIn("Search[", prompt)
                search.search.assert_not_called()
                self.assertEqual(score_answer(result, "the Rome!"), {"em": 1.0, "f1": 1.0})

    def test_focused_react_adds_reminder_and_early_stop_to_react(self):
        for mode in ("react", "focused_react"):
            with self.subTest(mode=mode):
                generate = scripted_generator(
                    "Thought: Calculate in parts.\nStart with the sum.\nAction: Calculate[ 2 + 3 ]",
                    "Action: Calculate[1 + 1]",
                    "Action: Calculate[2 + 3]",
                    "Action: Finish[+5.00]",
                )
                result = run_react("What is 2 + 3?", generate, task="gsm8k", mode=mode)
                history = [call.args[0] for call in generate.call_args_list]
                prompts = [messages[-1]["content"] for messages in history]
                self.assertEqual(
                    [message["role"] for message in history[-1]],
                    ["user", "assistant"] * 3 + ["user"],
                )
                forced = mode == "focused_react"
                self.assertEqual("Original question:" in prompts[0], forced)
                self.assertIn("Observation: 5", prompts[1])
                self.assertEqual(result.steps[0].raw_argument, " 2 + 3 ")
                self.assertTrue(result.steps[2].repeated)
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
        result = run_react("Question?", generate, task="gsm8k", mode="focused_react")
        self.assertEqual(result.termination, "budget_exhausted")
        self.assertEqual(generate.call_count, 9)
        self.assertEqual(score_answer(result, "1"), {"accuracy": 0.0})

        search = Mock(spec=SearchEngine)
        search.search.return_value = pd.DataFrame({"document": ["Evidence"]})
        outputs = [f"Action: Search[item {n}]" for n in range(6)]
        generate = scripted_generator(*outputs, "Action: Search[ITEM, 0!]", "Action: Finish[Rome]")
        result = run_react("Question?", generate, task="hotpotqa", mode="focused_react", search_engine=search)
        self.assertEqual(result.termination, "forced_finish")
        self.assertEqual(generate.call_count, 8)
        self.assertEqual(search.search.call_count, 6)
        self.assertEqual(result.steps[6].argument, "item 0")

    def test_invalid_replies_end_the_run_and_remain_available(self):
        examples = {
            "No command": "missing_action",
            "```\naction: calculate (2 + 3)\n```": "missing_action",
            "Action: Calculate[2 + 3]\nAction: Finish[5]": "multiple_actions",
            "Action: Calculate(2 + 3)": "malformed_action",
            "Thought: I can guess.\nObservation: 5\nAction: Finish[5]": "malformed_action",
            "Action: calculate[2 + 3]": "unsupported_action",
            "Action: Search[thing]": "unsupported_action",
            "Action: Calculate[1 / 0]": "invalid_tool_input",
            "Action: Finish[five]": "invalid_final_answer",
        }
        results = []
        for text, reason in examples.items():
            with self.subTest(reason=reason, text=text):
                generate = scripted_generator(text, "Action: Finish[5]")
                result = run_react("Question?", generate, task="gsm8k", mode="focused_react")
                results.append(result)
                self.assertEqual(result.termination, reason)
                self.assertEqual(result.steps[0].raw_output, text)
                self.assertEqual(generate.call_count, 1)
                self.assertEqual(score_answer(result, "5"), {"accuracy": 0.0})
                if reason != "invalid_tool_input":
                    self.assertFalse(result.steps[0].tool_called)
        report = diagnose_formats(results)
        self.assertEqual(report["affected_calls"], 5)
        self.assertEqual(report["categories"]["capitalization"], 2)
        self.assertEqual(report["categories"]["parentheses_instead_of_brackets"], 2)
        self.assertEqual(report["categories"]["markdown_wrapper"], 1)
        self.assertEqual(report["categories"]["multiple_commands"], 1)

    def test_calculation_strips_only_grouped_thousands_commas(self):
        generate = scripted_generator(
            "Action: Calculate[1,000,000 / 2,000 + 1,000.5]",
            "Action: Calculate[1000000/2000 + 1000.5]",
            "Action: Finish[1,500.5]",
        )
        result = run_react("Question?", generate, task="gsm8k", mode="focused_react")
        self.assertEqual(result.steps[0].raw_argument, "1,000,000 / 2,000 + 1,000.5")
        self.assertEqual(result.steps[0].argument, "1000000 / 2000 + 1000.5")
        self.assertEqual(result.steps[0].observation, "1500.5")
        self.assertFalse(result.steps[1].repeated)
        self.assertEqual(result.answer, "1500.5")
        for expression in ("1,0000 + 1", "12,34 + 1", "12345,678 + 1", "0.5,000 + 1"):
            with self.subTest(expression=expression):
                generate = scripted_generator(f"Action: Calculate[{expression}]")
                result = run_react("Question?", generate, task="gsm8k")
                self.assertEqual(result.termination, "invalid_tool_input")

    def test_forced_reply_cannot_request_a_tool_or_recover(self):
        for reply, reason in [
            ("Action: Calculate[2+2]", "unsupported_action"),
            ("Action: Finish[answer]", "invalid_final_answer"),
            ("My answer is 4", "missing_action"),
        ]:
            with self.subTest(reply=reply):
                generate = scripted_generator("Action: Calculate[1+1]", "Action: Calculate[1+1]", reply)
                result = run_react("Question?", generate, task="gsm8k", mode="focused_react")
                self.assertEqual(result.termination, reason)
                self.assertTrue(result.steps[-1].forced)
                self.assertEqual(summarize_run(result)["tool_calls"], 1)

    def test_infrastructure_failures_are_unscored(self):
        generate = Mock(side_effect=OSError("model unavailable"))
        result = run_react("Question?", generate, task="gsm8k")
        self.assertEqual(result.termination, "model_error")
        self.assertIsNone(score_answer(result, "1"))

        search = Mock(spec=SearchEngine)
        search.search.side_effect = ValueError("index unavailable")
        generate = scripted_generator("Action: Search[  A Test!  ]")
        result = run_react("Question?", generate, task="hotpotqa", search_engine=search)
        search.search.assert_called_once_with("A Test!")
        self.assertEqual(result.steps[0].argument, "a test")
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
        result = run_react("Question?", generate, task="hotpotqa", search_engine=search)
        summary = summarize_run(result, hotpotqa_supporting_documents(rows.iloc[0]))
        self.assertEqual(summary["supporting_paragraphs_retrieved"], 2)
        self.assertTrue(summary["all_supporting_paragraphs_retrieved"])

    def test_transformers_generator_preserves_full_reply(self):
        model = LanguageModel.__new__(LanguageModel)
        model.config = {"max_new_tokens": 256}
        model.use_kv_cache = False
        model.tokenizer = Mock()
        model.tokenizer.apply_chat_template.return_value = BatchEncoding(
            {"input_ids": torch.tensor([[1, 2, 3]])}
        )
        raw = " Action: Finish[5]\nObservation: fabricated\nAction: Finish[9]\n"
        model.tokenizer.decode.return_value = raw
        model.model = Mock(device="cpu")
        model.model.generate.return_value = GenerateDecoderOnlyOutput(
            sequences=torch.tensor([[1, 2, 3, 4, 5]])
        )

        generate = partial(model.generate, strip_output=False)
        result = run_react("Question?", generate, task="gsm8k")

        self.assertFalse(model.model.generate.call_args.kwargs["stopping_criteria"])
        self.assertEqual(result.steps[0].raw_output, raw)
        self.assertEqual(result.steps[0].token_usage, TokenUsage(3, 2))
        self.assertEqual(result.termination, "multiple_actions")


class TestKVCache(unittest.TestCase):
    """The cached model must generate exactly what the uncached one does."""

    @staticmethod
    def tiny_model(config):
        torch.manual_seed(0)
        model = LanguageModel.__new__(LanguageModel)
        model.config = {"max_new_tokens": 6}
        model.model = AutoModelForCausalLM.from_config(config).eval()
        model.tokenizer = Mock()
        model.tokenizer.decode.side_effect = lambda ids, **_: " ".join(map(str, ids.tolist()))
        model.reset_cache()
        return model

    def run_turns(self, model, use_cache):
        """Three calls whose inputs extend each other, as in a ReAct episode,
        then one unrelated input, as for the next question."""
        model.use_kv_cache = use_cache
        model.reset_cache()
        inputs = [list(range(1, 13)), list(range(1, 25)), list(range(1, 40)), [5, 4, 3, 2]]
        calls = []
        for ids in inputs:
            model.tokenizer.apply_chat_template.return_value = BatchEncoding(
                {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}
            )
            calls.append(model.generate([{"role": "user", "content": "x"}]))
        return calls

    def check(self, config):
        model = self.tiny_model(config)
        uncached = self.run_turns(model, use_cache=False)
        cached = self.run_turns(model, use_cache=True)
        self.assertEqual([text for text, _ in cached], [text for text, _ in uncached])
        self.assertEqual([usage.cached_tokens for _, usage in cached], [0, 12, 24, 0])
        self.assertTrue(all(usage.cached_tokens == 0 for _, usage in uncached))

    def test_full_attention(self):
        self.check(Qwen2Config(
            vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2,
        ))

    def test_sliding_window_past_the_window(self):
        self.check(Gemma2Config(
            vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8, sliding_window=8,
        ))


class TestActionParser(unittest.TestCase):
    """
    Test the action parser used by the ReAct controller (Omar's tests).

    These tests verify that the parser correctly extracts
    supported actions and rejects invalid model outputs.
    """

    def test_calculate_action(self):
        """Check that a calculator action is parsed correctly."""
        self.assertEqual(parse_action("Action: Calculate[15 + 18]"), ("Calculate", "15 + 18"))

    def test_search_action(self):
        """Check that a search query is extracted correctly."""
        output = "Thought: I need information.\nAction: Search[Apple Remote]"
        self.assertEqual(parse_action(output), ("Search", "Apple Remote"))

    def test_finish_action(self):
        """Check that the final answer is extracted correctly."""
        self.assertEqual(parse_action("Action: Finish[33]"), ("Finish", "33"))

    def test_missing_action(self):
        """Reject model outputs that do not contain an action."""
        with self.assertRaises(ValueError):
            parse_action("I need to calculate the answer.")

    def test_multiple_actions(self):
        """Reject outputs containing more than one action."""
        with self.assertRaises(ValueError):
            parse_action("Action: Search[Apple]\nAction: Finish[Apple]")


class TestFedericoTools(unittest.TestCase):
    """
    The loop calls Federico's Calculator and SearchEngine directly: their
    results become the observation, and their own errors decide what is an
    invalid operation. (These replace Omar's tests of his tool wrappers.)
    """

    def test_calculator_results_reach_the_model(self):
        for expression, observation in [
            ("15 + 18", "33"),
            ("10 * (2 + 3)", "50"),
            ("2 ** 10", "1024"),  # Federico's calculator accepts powers
            ("()", "()"),  # and returns its result unchanged
        ]:
            with self.subTest(expression=expression):
                generate = scripted_generator(f"Action: Calculate[{expression}]", "Action: Finish[1]")
                result = run_react("Question?", generate, task="gsm8k")
                self.assertEqual(result.steps[0].observation, observation)
                self.assertEqual(
                    generate.call_args.args[0][-1]["content"], f"Observation: {observation}\n"
                )

    def test_calculator_errors_end_the_run(self):
        for expression, error in [
            ("15 + unknown", "ValueError"),  # rejected by Federico's character check
            ("1 / 0", "ZeroDivisionError"),
            ("1 +", "SyntaxError"),
        ]:
            with self.subTest(expression=expression):
                generate = scripted_generator(f"Action: Calculate[{expression}]")
                result = run_react("Question?", generate, task="gsm8k")
                self.assertEqual(result.termination, "invalid_tool_input")
                self.assertTrue(result.error.startswith(error))
                self.assertTrue(result.steps[0].tool_called)

    def test_search_documents_reach_the_model(self):
        documents = [
            "Apple Remote controls Front Row.",
            "Front Row can also be controlled by a keyboard.",
        ]
        search_engine = Mock()
        search_engine.search.return_value = pd.DataFrame({"document": documents})
        generate = scripted_generator("Action: Search[ Front Row ]", "Action: Finish[Apple]")

        result = run_react(
            "Question?", generate, task="hotpotqa", search_engine=search_engine
        )

        search_engine.search.assert_called_once_with("Front Row")
        self.assertEqual(result.steps[0].documents, documents)
        self.assertEqual(result.steps[0].observation, "\n\n".join(documents))


class TestReActLoop(unittest.TestCase):
    """
    Test the main ReAct reasoning and action loop (Omar's tests).

    These tests use predefined model responses instead of a real
    language model, so no model download or GPU is needed.
    """

    def test_calculator_loop(self):
        """Execute a calculation, return the observation to the model, and finish."""
        generate = scripted_generator("Action: Calculate[15 + 18]", "Action: Finish[33]")
        result = run_react("What is 15 + 18?", generate, task="gsm8k")

        self.assertEqual(result.answer, "33")
        self.assertEqual(generate.call_count, 2)
        # The second model call must contain the calculator result.
        self.assertEqual(generate.call_args.args[0][-1]["content"], "Observation: 33\n")

    def test_search_loop(self):
        """Use the search engine and return the final answer."""
        search_engine = Mock()
        search_engine.search.return_value = pd.DataFrame(
            {"document": ["Hideaki Anno was born in Ube, Japan."]}
        )
        generate = scripted_generator("Action: Search[Hideaki Anno]", "Action: Finish[Ube]")

        result = run_react(
            "Where was Hideaki Anno born?", generate,
            task="hotpotqa", search_engine=search_engine,
        )

        self.assertEqual(result.answer, "Ube")
        search_engine.search.assert_called_once_with("Hideaki Anno")

    def test_search_observation_in_context(self):
        """The retrieved document reaches the model in the next step."""
        search_engine = Mock()
        search_engine.search.return_value = pd.DataFrame(
            {"document": ["The Eiffel Tower is located in Paris."]}
        )
        generate = scripted_generator(
            "Action: Search[Eiffel Tower location]", "Action: Finish[Paris]"
        )

        result = run_react(
            "Where is the Eiffel Tower located?", generate,
            task="hotpotqa", search_engine=search_engine,
        )

        self.assertEqual(result.answer, "Paris")
        self.assertEqual(generate.call_count, 2)
        self.assertIn(
            "Observation: The Eiffel Tower is located in Paris.",
            generate.call_args.args[0][-1]["content"],
        )

    def test_maximum_steps(self):
        """
        Stop when the model does not finish within the allowed steps.

        Changed: the run is recorded as `budget_exhausted` (scored as wrong)
        instead of raising RuntimeError.
        """
        generate = Mock(return_value=("Action: Calculate[15 + 18]", TokenUsage(5, 2)))
        result = run_react("What is 15 + 18?", generate, task="gsm8k", max_steps=2)

        self.assertEqual(result.termination, "budget_exhausted")
        self.assertEqual(generate.call_count, 2)
        self.assertIsNone(result.answer)

    def test_invalid_model_output(self):
        """
        Reject a reply that is not a valid ReAct action.

        Changed: the run ends with the reason `missing_action` and keeps the
        reply, instead of raising ValueError.
        """
        generate = scripted_generator("I do not know what to do.")
        result = run_react("What is 15 + 18?", generate, task="gsm8k")

        self.assertEqual(result.termination, "missing_action")
        self.assertEqual(result.steps[0].raw_output, "I do not know what to do.")

    def test_missing_search_engine(self):
        """
        Report an error when HotpotQA has no search engine.

        Changed: checked before the first model call and recorded as
        `tool_error`, instead of raising ValueError when Search is requested.
        """
        generate = Mock()
        result = run_react("Where was Hideaki Anno born?", generate, task="hotpotqa")

        self.assertEqual(result.termination, "tool_error")
        generate.assert_not_called()

    def test_calculation_error_ends_the_run(self):
        """
        An invalid calculation ends the run as a wrong answer.

        Changed: Omar's loop returned the error to the model as an
        observation and continued; the agreed strict rules end the run with
        `invalid_tool_input` instead. Omar used `2 ** 3`, which Federico's
        calculator accepts, so this uses text that it rejects.
        """
        generate = scripted_generator(
            "Action: Calculate[2 + three]",
            "Action: Finish[The calculation could not be completed.]",
        )
        result = run_react("Calculate 2 plus three.", generate, task="gsm8k")

        self.assertEqual(result.termination, "invalid_tool_input")
        self.assertEqual(
            result.error,
            "ValueError: Only numbers, arithmetic operators, and parentheses are allowed",
        )
        self.assertEqual(generate.call_count, 1)


if __name__ == "__main__":
    unittest.main()
