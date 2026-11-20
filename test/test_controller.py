import unittest
from unittest.mock import Mock

import pandas as pd

from reactagenthlt.controller import (
    execute_calculation,
    execute_search,
    parse_action,
    run_react,
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

        search_engine.search.return_value = pd.DataFrame(
            {
                "document": [
                    "Apple Remote controls Front Row.",
                    "Front Row can also be controlled by a keyboard.",
                ]
            }
        )

        # Execute the search through our controller function.
        result = execute_search("Front Row", search_engine)

        # Check that the search engine received the correct query.
        search_engine.search.assert_called_once_with("Front Row")

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
        search_engine.search.return_value = pd.DataFrame(columns=["document"])

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
        responses = iter(
            [
                "Action: Calculate[15 + 18]",
                "Action: Finish[33]",
            ]
        )

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
        search_engine.search.return_value = pd.DataFrame(
            {"document": ["Hideaki Anno was born in Ube, Japan."]}
        )

        responses = iter(
            [
                "Action: Search[Hideaki Anno]",
                "Action: Finish[Ube]",
            ]
        )

        # Execute the agent using the simulated model and search tool.
        answer = run_react(
            "Where was Hideaki Anno born?",
            lambda context: next(responses),
            search_engine=search_engine,
        )

        # Verify the final answer and the search query.
        self.assertEqual(answer, "Ube")
        search_engine.search.assert_called_once_with("Hideaki Anno")

    def test_search_observation_in_context(self):
        """
        Verify that retrieved documents are included in the context
        sent to the language model during the next ReAct step.
        """

        # Simulate the existing search engine.
        search_engine = Mock()
        search_engine.search.return_value = pd.DataFrame(
            {"document": ["The Eiffel Tower is located in Paris."]}
        )

        # Simulate two consecutive model responses.
        responses = iter(
            [
                "Action: Search[Eiffel Tower location]",
                "Action: Finish[Paris]",
            ]
        )

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
        responses = iter(
            [
                "Action: Calculate[2 ** 3]",
                "Action: Finish[The calculation could not be completed.]",
            ]
        )

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


if __name__ == "__main__":
    unittest.main()
