"""Agent interface tests without loading model weights or a search index."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from reactagenthlt.agents import (
    BaseAgent,
    LanguageModel,
    LlamaCppLanguageModel,
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

    def test_react_modes_stop_generation_after_the_action(self):
        agent = ReActAgent(self.model, task="gsm8k")
        for mode, reply in [
            ("react", "Action: Finish[5]"),
            ("focused_react", "Action: Finish[5]"),
            ("direct", "Finish[5]\nExtra commentary"),
            ("cot", "2 + 3 = 5.\nFinish[5]\nExtra commentary"),
        ]:
            with self.subTest(mode=mode):
                self.model.generate.side_effect = [(reply, TokenUsage(5, 2))]
                result = agent.run("What is 2 + 3?", mode=mode)
                kwargs = self.model.generate.call_args.kwargs
                self.assertEqual(len(kwargs["stop_criteria"]), 1)
                self.assertEqual((result.termination, result.answer), ("finished", "5"))

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


class _FakeHTTPResponse:
    def __init__(self, body: bytes) -> None:
        self._stream = io.BytesIO(body)

    def read(self, *args, **kwargs) -> bytes:
        return self._stream.read(*args, **kwargs)

    def __iter__(self):
        return iter(self._stream)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class TestLlamaCppLanguageModel(unittest.TestCase):
    @staticmethod
    def _sse(*events: dict) -> bytes:
        return b"".join(
            f"data: {json.dumps(event)}\n\n".encode("utf-8") for event in events
        )

    def test_config_backend_stop_criteria_and_kv_prefix_reuse(self):
        requests_seen: list[tuple[str, dict]] = []

        def fake_urlopen(request, timeout=None):
            payload = json.loads(request.data.decode("utf-8"))
            requests_seen.append((request.full_url, payload))
            if request.full_url.endswith("/apply-template"):
                return _FakeHTTPResponse(json.dumps({"prompt": "PROMPT"}).encode())
            if len(requests_seen) == 2:
                return _FakeHTTPResponse(
                    self._sse(
                        {
                            "content": "Thought: calc.\nAction: Calculate[2+3]\nObservation: fake",
                            "tokens_evaluated": 20,
                            "tokens_predicted": 9,
                            "timings": {"cache_n": 0},
                        },
                        {
                            "stop": True,
                            "tokens_evaluated": 20,
                            "tokens_predicted": 9,
                            "timings": {"cache_n": 0},
                        },
                    )
                )
            return _FakeHTTPResponse(
                self._sse(
                    {
                        "content": " Action: Finish[5] ",
                        "tokens_evaluated": 28,
                        "tokens_predicted": 4,
                        "timings": {"cache_n": 20},
                    },
                    {
                        "stop": True,
                        "tokens_evaluated": 28,
                        "tokens_predicted": 4,
                        "timings": {"cache_n": 20},
                    },
                )
            )

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "lfm3-local.json"
            config_path.write_text(
                json.dumps(
                    {
                        "backend": "llama_cpp",
                        "base_url": "http://127.0.0.1:8080",
                        "model": "lfm3-1.7b-q6_k",
                        "max_new_tokens": 256,
                    }
                ),
                encoding="utf-8",
            )
            model = LanguageModel(str(config_path))

        with patch("reactagenthlt.language_model.urlopen", side_effect=fake_urlopen):
            turn1 = [{"role": "user", "content": "What is 2 + 3?"}]
            text1, usage1 = model.generate(
                turn1,
                stop_criteria=(r"Action\s*:\s*[A-Za-z]\w*\s*\[[^\[\]]*\]",),
            )
            turn2 = turn1 + [
                {"role": "assistant", "content": text1},
                {"role": "user", "content": "Observation: 5\n"},
            ]
            text2, usage2 = model.generate(turn2, strip_output=False)

        self.assertEqual(text1, "Thought: calc.\nAction: Calculate[2+3]")
        self.assertEqual(usage1, TokenUsage(20, 9, 0))
        self.assertEqual(text2, " Action: Finish[5] ")
        self.assertEqual(usage2, TokenUsage(28, 4, 20))
        self.assertFalse(requests_seen[1][1]["cache_prompt"])
        self.assertEqual(requests_seen[1][1]["temperature"], 0.0)
        self.assertTrue(requests_seen[3][1]["cache_prompt"])

    def test_direct_llama_cpp_client_and_reset_cache(self):
        model = LlamaCppLanguageModel(
            "http://127.0.0.1:8080/", "lfm3-1.7b-q6_k", max_new_tokens=128
        )
        self.assertEqual(model.config["model"], "lfm3-1.7b-q6_k")
        self.assertEqual(model.config["base_url"], "http://127.0.0.1:8080")
        model.reset_cache()


if __name__ == "__main__":
    unittest.main()
