"""Agent interface tests without loading model weights or a search index."""

import unittest
from unittest.mock import Mock

from reactagenthlt.agents import (
    BaseAgent,
    LanguageModel,
    ReActAgent,
    Response,
    StandardAgent,
    TokenUsage,
)
from reactagenthlt.tools import Calculator, SearchEngine


class TestAgents(unittest.TestCase):
    def setUp(self):
        self.model = Mock(spec=LanguageModel)
        self.model.generate.return_value = ("Hello", TokenUsage(5, 2))

    def test_base_is_abstract(self):
        with self.assertRaises(TypeError):
            BaseAgent(self.model)

    def test_standard_uses_injected_model_and_request_options(self):
        agent = StandardAgent(self.model)
        agent.add_system_prompt("Be helpful.")
        response = agent.ask("Hi", do_sample=True, stream=True)

        self.assertIs(agent.language_model, self.model)
        self.assertIsInstance(response, Response)
        self.assertEqual(response.output, "Hello")
        self.assertEqual(response.token_usage, TokenUsage(5, 2))
        self.assertGreaterEqual(response.generation_time, 0)
        self.model.generate.assert_called_once_with(
            messages=[
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hi"},
            ],
            do_sample=True,
            stream=True,
        )
        self.assertEqual(
            agent.history[-1], {"role": "assistant", "content": "Hello"}
        )

    def test_clear_context_preserves_system_prompts(self):
        agent = StandardAgent(self.model)
        agent.add_system_prompt("Be helpful.")
        agent.ask("Hi")
        agent.clear_ctx()
        self.assertEqual(agent.history, [])
        self.assertEqual(
            agent.system_prompts,
            [{"role": "system", "content": "Be helpful."}],
        )

    def test_shared_model_has_independent_agent_state(self):
        first = StandardAgent(self.model)
        second = ReActAgent(self.model)
        first.ask("Hi")
        first.add_system_prompt("Be helpful.")
        self.assertEqual(second.history, [])
        self.assertEqual(second.system_prompts, [])
        self.assertIs(second.language_model, first.language_model)

    def test_react_accepts_explicit_tools(self):
        search_engine = Mock(spec=SearchEngine)
        calculator = Mock(spec=Calculator)
        agent = ReActAgent(
            self.model,
            search_engine=search_engine,
            calculator=calculator,
            max_steps=3,
        )
        self.assertIs(agent.search_engine, search_engine)
        self.assertIs(agent.calculator, calculator)
        self.assertEqual(agent.max_steps, 3)

    def test_react_defaults(self):
        first = ReActAgent(self.model)
        second = ReActAgent(self.model)
        self.assertIsNone(first.search_engine)
        self.assertIsInstance(first.calculator, Calculator)
        self.assertIsNot(first.calculator, second.calculator)
        self.assertEqual(first.max_steps, 9)

    def test_react_rejects_invalid_step_limit(self):
        for max_steps in (0, -1):
            with self.subTest(max_steps=max_steps):
                with self.assertRaises(ValueError):
                    ReActAgent(self.model, max_steps=max_steps)

    def test_react_ask_uses_injected_calculator_and_totals_model_calls(self):
        calculator = Mock(spec=Calculator)
        calculator.calculate.return_value = 5
        self.model.generate.side_effect = [
            ("Action: Calculate[2 + 3]", TokenUsage(5, 2)),
            ("Action: Finish[5]", TokenUsage(9, 3)),
        ]
        agent = ReActAgent(self.model, calculator=calculator, task="gsm8k")

        response = agent.ask("What is 2 + 3?")

        self.assertEqual(agent.max_steps, 9)
        self.assertEqual(response.output, "5")
        self.assertEqual(response.token_usage, TokenUsage(14, 5))
        self.assertGreaterEqual(response.generation_time, 0)
        calculator.calculate.assert_called_once_with("2 + 3")
        messages = self.model.generate.call_args.args[0]
        self.assertEqual(messages[-1], {"role": "user", "content": "Observation: 5\n"})
        self.assertFalse(self.model.generate.call_args.kwargs["strip_output"])

    def test_react_sends_system_prompts_before_each_question(self):
        self.model.generate.side_effect = [
            ("Action: Finish[5]", TokenUsage(5, 2)),
            ("Action: Finish[6]", TokenUsage(5, 2)),
        ]
        agent = ReActAgent(self.model, task="gsm8k")
        agent.add_system_prompt("You are careful.")

        agent.ask("What is 2 + 3?")
        agent.ask("What is 3 + 3?")

        messages = self.model.generate.call_args.args[0]
        self.assertEqual(messages[0], {"role": "system", "content": "You are careful."})
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertIn("What is 3 + 3?", messages[1]["content"])
        self.assertEqual(len(agent.system_prompts), 1)

    def test_focused_react_run_returns_trace_after_a_repeat(self):
        self.model.generate.side_effect = [
            ("Action: Calculate[2 + 3]", TokenUsage(5, 2)),
            ("Action: Calculate[2 + 3]", TokenUsage(9, 2)),
            ("Action: Finish[5]", TokenUsage(14, 3)),
        ]
        agent = ReActAgent(self.model, task="gsm8k", max_steps=2)

        result = agent.run("What is 2 + 3?", mode="focused_react")

        self.assertEqual(result.termination, "forced_finish")
        self.assertEqual(result.answer, "5")
        self.assertEqual(len(result.steps), 3)
        self.assertFalse(result.steps[1].tool_called)
        self.assertTrue(result.steps[-1].forced)


if __name__ == "__main__":
    unittest.main()
