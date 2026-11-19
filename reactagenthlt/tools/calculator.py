import re


class Calculator:
    """Simple calculator implementation with built-in `eval` function"""

    def calculate(self, expression: str) -> int | float:
        """
        evaluate the expression returning the result. A preliminary check on the
        validity of the expression is made and will result in an `ValueError`
        exception if not satisfied.
        """
        pattern = re.compile(r"[0-9+*/.()\s-]+", flags=re.ASCII)
        if not pattern.fullmatch(expression):
            raise ValueError(
                "Only numbers, arithmetic operators, and parentheses are allowed"
            )

        return eval(expression, {"__builtins__": {}})


if __name__ == "__main__":
    pass
