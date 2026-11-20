
import unittest

from reactagenthlt.diagnostics import (
    diagnose_action_format,
    summarize_diagnostic_failures,
)

from unittest.mock import Mock

from reactagenthlt.controller import (
    InvalidReActRun,
    run_react,
)


class TestDiagnosticParser(unittest.TestCase):
    """
    Test the independent diagnostic parser.

    Diagnostics must inspect model outputs without executing
    actions or modifying the main evaluation behavior.
    """

    def test_valid_action(self):
        output = "Action: Search[Paris]"
        self.assertIsNone(diagnose_action_format(output))

    def test_missing_action(self):
        output = "Thought: I should search for information."
        self.assertEqual(
            diagnose_action_format(output),
            "missing_action",
        )

    def test_multiple_actions(self):
        output = (
            "Action: Search[Paris]\n"
            "Action: Finish[France]"
        )
        self.assertEqual(
            diagnose_action_format(output),
            "multiple_actions",
        )

    def test_nonstandard_prefix(self):
        output = "action: Search[Paris]"
        self.assertEqual(
            diagnose_action_format(output),
            "nonstandard_prefix",
        )

    def test_malformed_command(self):
        output = "Action: Search[Paris"
        self.assertEqual(
            diagnose_action_format(output),
            "malformed_command",
        )

    def test_unsupported_action(self):
        output = "Action: Lookup[Paris]"
        self.assertEqual(
            diagnose_action_format(output),
            "unsupported_action",
        )

    def test_empty_argument(self):
        output = "Action: Finish[]"
        self.assertEqual(
            diagnose_action_format(output),
            "empty_argument",
        )


    def test_diagnostic_error_counts(self):
        """
        Verify that formatting errors are counted by category
        and that the total number of affected calls is correct.
        """

        raw_outputs = [
            "Action: Lookup[Paris]",
            "Action: Lookup[London]",
            "action: Search[France]",
            "Action: Search[Paris",
            "Thought: I need to search.",
        ]

        summary = summarize_diagnostic_failures(raw_outputs)

        self.assertEqual(summary["unsupported_action"], 2)
        self.assertEqual(summary["nonstandard_prefix"], 1)
        self.assertEqual(summary["malformed_command"], 1)
        self.assertEqual(summary["missing_action"], 1)
        self.assertEqual(summary["total_affected_calls"], 5)

    def test_valid_actions_are_not_counted(self):
        """
        Verify that valid command formats are excluded
        from the diagnostic formatting-error statistics.
        """

        raw_outputs = [
            "Action: Search[Paris]",
            "Action: Calculate[10 + 5]",
            "Action: Lookup[London]",
        ]

        summary = summarize_diagnostic_failures(raw_outputs)

        self.assertEqual(summary["unsupported_action"], 1)
        self.assertEqual(summary["total_affected_calls"], 1)

    def test_empty_diagnostic_records(self):
        """
        Verify that an empty collection of recorded
        model responses produces zero affected calls.
        """

        summary = summarize_diagnostic_failures([])

        self.assertEqual(
            summary,
            {"total_affected_calls": 0},
        )


    def test_invalid_run_is_diagnosed(self):
        """
        Verify that a real strict-controller failure can be
        analyzed using the independent diagnostic parser.
        """

        # Simulate a model response containing an unsupported action.
        raw_output = "Action: Lookup[Paris]"
        fake_model = Mock(return_value=raw_output)

        # The strict controller must reject the invalid action.
        with self.assertRaises(InvalidReActRun) as result:
            run_react(
                question="Where is Paris?",
                generate=fake_model,
                strict=True,
            )

        error = result.exception

        # Verify that the original model response is preserved.
        self.assertEqual(error.reason, "invalid_action")
        self.assertEqual(error.raw_output, raw_output)

        # Analyze the recorded invalid output separately.
        summary = summarize_diagnostic_failures(
            [error.raw_output]
        )

        self.assertEqual(summary["unsupported_action"], 1)
        self.assertEqual(summary["total_affected_calls"], 1)

    def test_tool_failure_is_not_a_formatting_error(self):
        """
        Verify that invalid tool inputs are kept separate
        from command-formatting errors during diagnostics.
        """

        raw_output = "Action: Calculate[2 ** 3]"
        fake_model = Mock(return_value=raw_output)

        # The command format is valid, but the calculation
        # contains an unsupported operator.
        with self.assertRaises(InvalidReActRun) as result:
            run_react(
                question="Calculate 2 to the power of 3.",
                generate=fake_model,
                strict=True,
            )

        error = result.exception

        # The strict controller identifies a tool-input failure.
        self.assertEqual(error.reason, "invalid_tool_input")
        self.assertEqual(error.raw_output, raw_output)

        # The diagnostic parser must not misclassify
        # a valid command format as a formatting error.
        self.assertIsNone(
            diagnose_action_format(error.raw_output)
        )

        summary = summarize_diagnostic_failures(
            [error.raw_output]
        )

        self.assertEqual(summary["total_affected_calls"], 0)



if __name__ == "__main__":
    unittest.main()
