"""Zero-shot prompts for the two tasks and the Focused ReAct interventions."""

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

REACT = """Solve the task: either answer a question using the supplied collection of paragraphs, or solve a math problem. Some paragraphs may be irrelevant, and answering may require combining information from more than one paragraph.

Available actions:
- Search[query]: search the shared collection of paragraphs and return the top matching paragraphs with their titles.
- Calculate[expression]: evaluate an arithmetic expression and return its result.
- Finish[answer]: provide the final answer and end the task. Give a short answer for a question, or only the final number for a math problem.

Choose the action that best serves your goal. At each step, briefly reason about what to do next, then output exactly one action on a separate line in this format:
Thought: your reasoning
Action: Search[query]

The action can be Search[query], Calculate[expression] or Finish[answer]. Choose exactly one action. After your action, the system will execute it and provide the result. Use that result to decide your next step. When you have the answer, finish with it."""

REITERATE = """Original question: {question}
Continue solving this question using the information available so far."""

FORCE_FINISH = """Original question: {question}
Using the information collected so far, give your best final answer now. Do not request another tool action. Output only:
Action: Finish[{answer}]"""
