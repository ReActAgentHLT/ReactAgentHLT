import unittest
from unittest.mock import Mock
import pandas as pd

from reactagenthlt.controller import (
    parse_action,
    execute_calculation,
    execute_search,
    normalize_search_query,
    run_react,
    InvalidReActRun,
    ReActBudgetExceeded,
    ReActToolFailure,
)


class TestActionParser(unittest.TestCase):
    """
    Test the action parser used by the ReAct controller.

    These tests verify that the parser correctly extracts
    supported actions and rejects invalid model outputs.
    """

    def test_calculate_action(self):
        """Check that a calculator action is parsed correctly."""

        output = "Action: Calculate[15 + 18]"

        result = parse_action(output)

        self.assertEqual(result, ("Calculate", "15 + 18"))

    def test_search_action(self):
        """Check that a search query is extracted correctly."""

        output = "Thought: I need information.\nAction: Search[Apple Remote]"

        result = parse_action(output)

        self.assertEqual(result, ("Search", "Apple Remote"))

    def test_finish_action(self):
        """Check that the final answer is extracted correctly."""

        output = "Action: Finish[33]"

        result = parse_action(output)

        self.assertEqual(result, ("Finish", "33"))

    def test_missing_action(self):
        """Reject model outputs that do not contain an action."""

        output = "I need to calculate the answer."

        with self.assertRaises(ValueError):
            parse_action(output)

    def test_multiple_actions(self):
        """Reject outputs containing more than one action."""

        output = "Action: Search[Apple]\nAction: Finish[Apple]"

        with self.assertRaises(ValueError):
            parse_action(output)




class TestCalculatorExecution(unittest.TestCase):
    """
    Test the connection between the ReAct controller and the Calculator.

    These tests verify that valid mathematical expressions are
    evaluated correctly and unsupported inputs are rejected.

    The controller uses Federico's existing Calculator tool
    without modifying its original implementation.
    """

    def test_simple_addition(self):
        """Check that the calculator correctly adds two numbers."""

        result = execute_calculation("15 + 18")

        self.assertEqual(result, "33")

    def test_expression_with_parentheses(self):
        """
        Check that the calculator respects arithmetic precedence
        and correctly evaluates expressions with parentheses.
        """

        result = execute_calculation("10 * (2 + 3)")

        self.assertEqual(result, "50")

    def test_invalid_expression(self):
        """
        Reject expressions containing unsupported text.

        The model may generate invalid input, so the controller
        must validate the expression before calling Calculator.
        """

        with self.assertRaises(ValueError):
            execute_calculation("15 + unknown")

    def test_unsupported_operator(self):
        """
        Reject exponentiation because it is not supported
        by the current controller's calculation rules.
        """

        with self.assertRaises(ValueError):
            execute_calculation("2 ** 10")



class TestSearchExecution(unittest.TestCase):
    """
    Test the integration between the ReAct controller and SearchEngine.

    These tests use a mock search engine to simulate search results.
    This allows us to check the controller without loading a dataset
    or changing Federico's original SearchEngine implementation.
    """

    def test_search_returns_documents(self):
        """
        Check that the controller converts retrieved documents
        into a single text observation for the language model.
        """

        # Create a mock search engine with predefined results.
        search_engine = Mock()

        search_engine.search.return_value = pd.DataFrame({
            "document": [
                "Apple Remote controls Front Row.",
                "Front Row can also be controlled by a keyboard.",
            ]
        })

        # Execute the search through our controller function.
        result = execute_search("Front Row", search_engine)

        # Check that the search engine received the correct query.
        search_engine.search.assert_called_once_with("front row")

        # Check that both retrieved documents appear in the observation.
        self.assertIn("Apple Remote controls Front Row.", result)
        self.assertIn("Front Row can also be controlled by a keyboard.", result)

    def test_empty_search_query(self):
        """
        Check that an empty search query is rejected
        before calling the search engine.
        """

        search_engine = Mock()

        with self.assertRaises(ValueError):
            execute_search("   ", search_engine)

        # The search tool should not be called for invalid input.
        search_engine.search.assert_not_called()

    def test_no_search_results(self):
        """
        Check that the controller returns a clear observation
        when the search engine finds no documents.
        """

        search_engine = Mock()
        search_engine.search.return_value = pd.DataFrame(
            columns=["document"]
        )

        result = execute_search("Unknown topic", search_engine)

        self.assertEqual(result, "No documents found.")



class TestReActLoop(unittest.TestCase):
    """
    Test the main ReAct reasoning and action loop.

    These tests use predefined model responses instead of a real
    language model. This allows us to verify the controller's
    behavior without downloading a model or using a GPU.

    The tests cover successful execution, tool integration,
    stopping conditions, and invalid model outputs.
    """

    def test_calculator_loop(self):
        """
        Check that the controller executes a calculation,
        returns the observation to the model, and finishes.
        """

        # Simulate two responses generated by a language model.
        responses = iter([
            "Action: Calculate[15 + 18]",
            "Action: Finish[33]",
        ])

        # Save the context received during each model call.
        contexts = []

        def fake_model(context):
            contexts.append(context)
            return next(responses)

        # Run the complete ReAct loop.
        answer = run_react("What is 15 + 18?", fake_model)

        # Check the final answer and the number of model calls.
        self.assertEqual(answer, "33")
        self.assertEqual(len(contexts), 2)

        # The second model call must contain the calculator result.
        self.assertIn("Observation: 33", contexts[1])

    def test_search_loop(self):
        """
        Check that the controller uses the existing SearchEngine
        and returns the final answer after retrieving information.
        """

        # Simulate Federico's search engine without loading HotpotQA.
        search_engine = Mock()
        search_engine.search.return_value = pd.DataFrame({
            "document": ["Hideaki Anno was born in Ube, Japan."]
        })

        responses = iter([
            "Action: Search[Hideaki Anno]",
            "Action: Finish[Ube]",
        ])

        # Execute the agent using the simulated model and search tool.
        answer = run_react(
            "Where was Hideaki Anno born?",
            lambda context: next(responses),
            search_engine=search_engine,
        )

        # Verify the final answer and the search query.
        self.assertEqual(answer, "Ube")
        search_engine.search.assert_called_once_with("hideaki anno")


    def test_search_observation_in_context(self):
        """
        Verify that retrieved documents are included in the context
        sent to the language model during the next ReAct step.
        """

        # Simulate the existing search engine.
        search_engine = Mock()
        search_engine.search.return_value = pd.DataFrame({
            "document": ["The Eiffel Tower is located in Paris."]
        })

        # Simulate two consecutive model responses.
        responses = iter([
            "Action: Search[Eiffel Tower location]",
            "Action: Finish[Paris]",
        ])

        # Record the context received by the model.
        contexts = []

        def fake_model(context):
            contexts.append(context)
            return next(responses)

        # Execute the ReAct loop.
        answer = run_react(
            "Where is the Eiffel Tower located?",
            fake_model,
            search_engine=search_engine,
        )

        # Verify the final answer.
        self.assertEqual(answer, "Paris")

        # Verify that the model was called twice.
        self.assertEqual(len(contexts), 2)

        # Verify that the retrieved document reached the model.
        self.assertIn(
            "Observation: The Eiffel Tower is located in Paris.",
            contexts[1],
        )


    def test_maximum_steps(self):
        """
        Check that the controller stops when the model does not
        produce a final answer within the allowed number of steps.
        """

        # The model keeps requesting a calculation instead of finishing.
        def fake_model(context):
            return "Action: Calculate[15 + 18]"

        with self.assertRaises(RuntimeError):
            run_react(
                "What is 15 + 18?",
                fake_model,
                max_steps=2,
            )

    def test_invalid_model_output(self):
        """
        Check that an invalid model response is rejected instead
        of being interpreted as a valid ReAct action.
        """

        def fake_model(context):
            return "I do not know what to do."

        with self.assertRaises(ValueError):
            run_react("What is 15 + 18?", fake_model)

    def test_missing_search_engine(self):
        """
        Check that the controller reports an error when the model
        requests Search but no search engine is available.
        """

        def fake_model(context):
            return "Action: Search[Hideaki Anno]"

        with self.assertRaises(ValueError):
            run_react("Where was Hideaki Anno born?", fake_model)



    def test_calculation_error_becomes_observation(self):
        """
        Verify that a failed calculation is returned to the model
        as a tool error observation without stopping the agent.
        """

        # The model first requests an unsupported calculation,
        # then responds after receiving the error observation.
        responses = iter([
            "Action: Calculate[2 ** 3]",
            "Action: Finish[The calculation could not be completed.]",
        ])

        contexts = []

        def fake_model(context):
            contexts.append(context)
            return next(responses)

        answer = run_react(
            "Calculate 2 to the power of 3.",
            fake_model,
        )

        # The controller should continue after the tool error.
        self.assertEqual(
            answer,
            "The calculation could not be completed.",
        )

        # The second model call must receive the error message.
        self.assertEqual(len(contexts), 2)
        self.assertIn(
            "Observation: Tool error: Invalid or unsupported calculation.",
            contexts[1],
        )


    def test_strict_invalid_action(self):
        """
        Verify that strict evaluation rejects invalid actions
        and preserves the original model output.
        """

        raw_output = (
            "Thought: I need information.\n"
            "Action: Lookup[Paris]"
        )

        def fake_model(context):
            return raw_output

        with self.assertRaises(InvalidReActRun) as result:
            run_react(
                "Where is the Eiffel Tower?",
                fake_model,
                strict=True,
            )

        self.assertEqual(result.exception.reason, "invalid_action")
        self.assertEqual(result.exception.raw_output, raw_output)

    def test_strict_invalid_calculation(self):
        """
        Verify that invalid tool inputs terminate strict runs
        instead of becoming recoverable observations.
        """

        raw_output = "Action: Calculate[2 ** 3]"

        def fake_model(context):
            return raw_output

        with self.assertRaises(InvalidReActRun) as result:
            run_react(
                "Calculate 2 to the power of 3.",
                fake_model,
                strict=True,
            )

        self.assertEqual(
            result.exception.reason,
            "invalid_tool_input",
        )
        self.assertEqual(
            result.exception.raw_output,
            raw_output,
        )

    def test_strict_empty_final_answer(self):
        """
        Verify that strict evaluation rejects empty final
        answers and preserves the original model output.
        """

        raw_output = "Action: Finish[   ]"

        def fake_model(context):
            return raw_output

        with self.assertRaises(InvalidReActRun) as result:
            run_react(
                "What is the answer?",
                fake_model,
                strict=True,
            )

        self.assertEqual(
            result.exception.reason,
            "invalid_final_answer",
        )
        self.assertEqual(
            result.exception.raw_output,
            raw_output,
        )


    def test_budget_exhaustion_metadata(self):
        """
        Verify that an exhausted step budget records the dataset,
        number of model calls, and original final model output.
        """

        raw_output = "Action: Calculate[1 + 1]"

        fake_model = Mock(return_value=raw_output)

        with self.assertRaises(ReActBudgetExceeded) as result:
            run_react(
                question="What is 1 + 1?",
                generate=fake_model,
                dataset="hotpotqa",
                strict=True,
            )

        error = result.exception

        # Verify the structured evaluation failure information.
        self.assertEqual(error.reason, "budget_exhausted")
        self.assertEqual(error.dataset, "hotpotqa")
        self.assertEqual(error.steps_used, 7)
        self.assertEqual(error.last_output, raw_output)

        # The model must not exceed the permitted step budget.
        self.assertEqual(fake_model.call_count, 7)


    def test_search_tool_execution_failure(self):
        """
        Verify that a search engine execution failure is
        reported separately from an invalid model action.
        """

        # Simulate a search engine that fails during execution.
        search_engine = Mock()
        search_engine.search.side_effect = RuntimeError(
            "Search index unavailable."
        )

        raw_output = "Action: Search[Paris]"
        fake_model = Mock(return_value=raw_output)

        with self.assertRaises(ReActToolFailure) as result:
            run_react(
                question="Where is Paris?",
                generate=fake_model,
                search_engine=search_engine,
                strict=True,
            )

        error = result.exception

        # Verify that the failure is classified correctly.
        self.assertEqual(error.reason, "tool_failure")
        self.assertEqual(error.tool_name, "Search")
        self.assertEqual(error.raw_output, raw_output)

        # The search engine was called with the normalized query.
        search_engine.search.assert_called_once_with("paris")


    def test_search_value_error_is_tool_failure(self):
        """
        Verify that a ValueError raised internally by SearchEngine
        is classified as a tool failure, not invalid model input.
        """

        search_engine = Mock()
        search_engine.search.side_effect = ValueError(
            "Search index corrupted."
        )

        raw_output = "Action: Search[Paris]"
        fake_model = Mock(return_value=raw_output)

        with self.assertRaises(ReActToolFailure) as result:
            run_react(
                question="Where is Paris?",
                generate=fake_model,
                search_engine=search_engine,
                strict=True,
            )

        error = result.exception

        self.assertEqual(error.reason, "tool_failure")
        self.assertEqual(error.tool_name, "Search")
        self.assertEqual(error.raw_output, raw_output)

        search_engine.search.assert_called_once_with("paris")


    def test_trace_records_invalid_action_failure(self):
        """
        Verify that strict evaluation records the reason
        for rejecting an invalid model action.
        """

        trace = []
        raw_output = "Action: Lookup[Paris]"

        with self.assertRaises(InvalidReActRun):
            run_react(
                question="Where is Paris?",
                generate=lambda context: raw_output,
                strict=True,
                trace=trace,
            )

        self.assertEqual(len(trace), 1)
        self.assertEqual(trace[0]["raw_output"], raw_output)
        self.assertEqual(trace[0]["status"], "invalid")
        self.assertEqual(
            trace[0]["failure_reason"],
            "invalid_action",
        )

    def test_trace_records_budget_exhaustion(self):
        """
        Verify that the final trace record identifies
        budget exhaustion without exceeding the step limit.
        """

        trace = []

        fake_model = Mock(
            return_value="Action: Calculate[1 + 1]"
        )

        with self.assertRaises(ReActBudgetExceeded):
            run_react(
                question="What is 1 + 1?",
                generate=fake_model,
                dataset="hotpotqa",
                strict=True,
                trace=trace,
            )

        self.assertEqual(len(trace), 7)
        self.assertEqual(fake_model.call_count, 7)

        self.assertEqual(
            trace[-1]["status"],
            "budget_exhausted",
        )

        self.assertEqual(
            trace[-1]["failure_reason"],
            "budget_exhausted",
        )


    def test_trace_records_invalid_final_answer(self):
        """
        Verify that strict evaluation records an empty final
        answer as an invalid run without losing the raw output.
        """

        trace = []
        raw_output = "Action: Finish[   ]"

        with self.assertRaises(InvalidReActRun) as result:
            run_react(
                question="What is the answer?",
                generate=lambda context: raw_output,
                strict=True,
                trace=trace,
            )

        self.assertEqual(
            result.exception.reason,
            "invalid_final_answer",
        )

        self.assertEqual(len(trace), 1)
        self.assertEqual(trace[0]["raw_output"], raw_output)
        self.assertEqual(trace[0]["status"], "invalid")
        self.assertEqual(
            trace[0]["failure_reason"],
            "invalid_final_answer",
        )



class TestSearchNormalization(unittest.TestCase):
    """
    Test query normalization using the existing retrieval tokenizer.
    """

    def test_lowercase_and_punctuation(self):
        """
        Verify that normalization removes punctuation
        and converts uppercase letters to lowercase.
        """

        query = "  Eiffel Tower, PARIS!  "

        result = normalize_search_query(query)

        self.assertEqual(result, "eiffel tower paris")

    def test_unicode_normalization(self):
        """
        Verify that Unicode letters are preserved
        while punctuation is removed.
        """

        query = "Café—PARIS, 2026!"

        result = normalize_search_query(query)

        self.assertEqual(result, "café paris 2026")

    def test_punctuation_only_query(self):
        """
        Verify that queries without searchable words
        are rejected before reaching the search engine.
        """

        search_engine = Mock()

        with self.assertRaises(ValueError):
            execute_search("!!!???", search_engine)

        search_engine.search.assert_not_called()



    def test_hotpotqa_step_budget(self):
        """
        Verify that HotpotQA stops after seven decision steps
        when the model does not produce a final answer.
        """

        # The model keeps requesting calculations without finishing.
        fake_model = Mock(
            return_value="Action: Calculate[1 + 1]"
        )

        with self.assertRaises(RuntimeError):
            run_react(
                question="What is 1 + 1?",
                generate=fake_model,
                dataset="hotpotqa",
            )

        # The default HotpotQA budget allows exactly seven calls.
        self.assertEqual(fake_model.call_count, 7)

    def test_gsm8k_step_budget(self):
        """
        Verify that GSM8K allows nine decision steps,
        including the final Finish action.
        """

        # Eight calculations followed by a final answer.
        responses = [
            "Action: Calculate[1 + 1]"
        ] * 8

        responses.append("Action: Finish[2]")

        fake_model = Mock(side_effect=responses)

        answer = run_react(
            question="What is 1 + 1?",
            generate=fake_model,
            dataset="gsm8k",
        )

        self.assertEqual(answer, "2")
        self.assertEqual(fake_model.call_count, 9)

    def test_unsupported_dataset(self):
        """
        Verify that unsupported datasets are rejected
        before the language model is called.
        """

        fake_model = Mock()

        with self.assertRaises(ValueError):
            run_react(
                question="What is 1 + 1?",
                generate=fake_model,
                dataset="unknown",
            )

        fake_model.assert_not_called()



    def test_trace_records_react_steps(self):
        """
        Verify that the controller records model decisions,
        tool observations, and the final answer step.
        """

        responses = iter([
            "Action: Calculate[15 + 18]",
            "Action: Finish[33]",
        ])

        # Collect the execution trace in a separate list.
        trace = []

        answer = run_react(
            question="What is 15 + 18?",
            generate=lambda context: next(responses),
            trace=trace,
        )

        # Verify the final answer and the recorded steps.
        self.assertEqual(answer, "33")
        self.assertEqual(len(trace), 2)

        self.assertEqual(trace[0]["step"], 1)
        self.assertEqual(
            trace[0]["raw_output"],
            "Action: Calculate[15 + 18]",
        )
        self.assertEqual(trace[0]["action"], "Calculate")
        self.assertEqual(trace[0]["observation"], "33")
        self.assertEqual(trace[0]["status"], "continued")

        self.assertEqual(trace[1]["step"], 2)
        self.assertEqual(trace[1]["action"], "Finish")
        self.assertEqual(trace[1]["status"], "finished")

    def test_trace_preserves_invalid_model_output(self):
        """
        Verify that the original model output is recorded
        even when strict evaluation rejects the action.
        """

        raw_output = "Action: Lookup[Paris]"
        trace = []

        with self.assertRaises(InvalidReActRun):
            run_react(
                question="Where is Paris?",
                generate=lambda context: raw_output,
                strict=True,
                trace=trace,
            )

        self.assertEqual(len(trace), 1)
        self.assertEqual(trace[0]["step"], 1)
        self.assertEqual(trace[0]["raw_output"], raw_output)



if __name__ == "__main__":
    unittest.main()
