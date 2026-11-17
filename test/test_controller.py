import unittest

from reactagenthlt.controller import parse_action, execute_calculation


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



if __name__ == "__main__":
    unittest.main()
