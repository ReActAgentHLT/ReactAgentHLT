"""Agent interface tests without loading model weights or a search index."""

import unittest
from unittest.mock import Mock

from reactagenthlt.agents import (
    Agent,
    BaseAgent,
    LanguageModel,
    ReActAgent,
    Response,
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
        agent = Agent(self.model)
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
        agent = Agent(self.model)
        agent.add_system_prompt("Be helpful.")
        agent.ask("Hi")
        agent.clear_ctx()
        self.assertEqual(agent.history, [])
        self.assertEqual(
            agent.system_prompts,
            [{"role": "system", "content": "Be helpful."}],
        )

    def test_shared_model_has_independent_agent_state(self):
        first = Agent(self.model)
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
        self.assertEqual(first.max_steps, 7)

    def test_react_rejects_invalid_step_limit(self):
        for max_steps in (0, -1):
            with self.subTest(max_steps=max_steps):
                with self.assertRaises(ValueError):
                    ReActAgent(self.model, max_steps=max_steps)


if __name__ == "__main__":
    unittest.main()
