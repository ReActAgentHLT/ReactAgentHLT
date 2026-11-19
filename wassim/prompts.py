"""Exact prompt templates for the four inference modes."""

DIRECT = {
    "hotpotqa": """Answer the question.
Return only the short final answer in this format:
Finish[answer]""",
    "gsm8k": """Solve the math problem and return only the final numerical answer in this format:
Finish[number]""",
}

COT = {
    "hotpotqa": """Answer the question. Reason step by step before giving your answer.
After your reasoning, provide a short final answer in this format:
Finish[answer]""",
    "gsm8k": """Solve the math problem step by step. Identify the quantities and operations needed to answer the question.
After your reasoning, provide the final numerical answer in this format:
Finish[number]""",
}

REACT = {
    "hotpotqa": """Answer the question by searching the supplied collection of paragraphs.
Some paragraphs may be irrelevant, and answering may require combining information from more than one paragraph.

Available actions:
- Search[query]: search the current question's supplied paragraphs and return the top matching paragraphs with their titles.
- Finish[answer]: provide a short final answer and end the task.

At each step, briefly reason about what to do next, then output exactly one action on a separate line in one of these formats:
Thought: your reasoning
Action: Search[query]
or:
Thought: your reasoning
Action: Finish[answer]

Choose exactly one action. After your action, the system will execute it and provide the result. Use that result to decide your next step. When you have enough information, finish with the answer.""",
    "gsm8k": """Solve the math problem. You can use a calculator for arithmetic.

Available actions:
- Calculate[expression]: evaluate an arithmetic expression and return its result.
- Finish[number]: provide the final numerical answer and end the task.

At each step, briefly reason about what to do next, then output exactly one action on a separate line in one of these formats:
Thought: your reasoning
Action: Calculate[expression]
or:
Thought: your reasoning
Action: Finish[number]

Choose exactly one action. After your action, the system will execute it and provide the result. Use that result to decide your next step. When you have the answer, finish with the final number.""",
}

CONTINUE = """Original question: {question}
Continue solving this question using the information available so far."""

FORCED_FINISH = """Original question: {question}
Using the information collected so far, give your best final answer now. Do not request another tool action. Output only:
Action: Finish[answer]"""


def _task(task: str) -> str:
    if task not in ("hotpotqa", "gsm8k"):
        raise ValueError("task must be 'hotpotqa' or 'gsm8k'")
    return task


def _history(history: str) -> str:
    history = history.rstrip()
    return f"{history}\n" if history else ""


def direct_prompt(task: str, question: str) -> str:
    return f"{DIRECT[_task(task)]}\n\nQuestion: {question}\n"


def cot_prompt(task: str, question: str) -> str:
    return f"{COT[_task(task)]}\n\nQuestion: {question}\n"


def react_prompt(task: str, question: str, history: str = "") -> str:
    return f"{REACT[_task(task)]}\n\nQuestion: {question}\n{_history(history)}"


def focused_step_prompt(task: str, question: str, history: str = "") -> str:
    prompt = react_prompt(task, question, history)
    return f"{prompt}\n{CONTINUE.format(question=question)}\n"


def forced_finish_prompt(task: str, question: str, history: str = "") -> str:
    prompt = react_prompt(task, question, history)
    return f"{prompt}\n{FORCED_FINISH.format(question=question)}\n"
