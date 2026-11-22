"""Few-shot task prompts and instructions for the Focused ReAct interventions."""

DIRECT = {
    "hotpotqa": """Answer the question in exactly one line: Finish[answer].
Give only the short answer (a name, place, date, or fact). Answer yes or no ONLY if the question is a yes/no question.
Use plain text only. Write Finish[answer] without bullets, bold, backticks, or any other text.

Examples:
Question: In what country was the author of Pride and Prejudice born?
Finish[England]

Question: Are the Eiffel Tower and the Louvre in the same city?
Finish[yes]

Question: Who directed the 2010 science fiction film Inception?
Finish[Christopher Nolan]

Question: In what year did the Berlin Wall fall?
Finish[1989]""",
    "gsm8k": """Solve the math problem. Return exactly one line: Finish[number]. Do not include units or an explanation.
Use plain text only. Write Finish[number] without bullets, bold, backticks, or any other text.

Examples:
Question: Maya buys 3 packs of 8 pencils and gives away 5. How many does she have left?
Finish[19]

Question: A baker shares 30 rolls equally among 6 baskets. How many rolls are in each basket?
Finish[5]

Question: A train travels 60 miles per hour for 3 hours and then 45 miles per hour for 2 hours. How many miles does it travel in total?
Finish[270]

Question: Liam had 50 dollars. He bought 4 notebooks for 6 dollars each and a pen for 3 dollars. How many dollars does he have left?
Finish[23]""",
}

COT = {
    "hotpotqa": """Answer the question. Keep the reasoning brief (1 to 3 short sentences connecting the needed facts), then end with one line in the form Finish[answer].
Give only a short answer inside Finish[...]. Answer yes or no ONLY if the question is a yes/no question.
Use plain text only. Write the final line exactly as Finish[answer], with no bullets, bold, backticks, or text after it.

Examples:
Question: In what country was the author of Pride and Prejudice born?
Jane Austen wrote Pride and Prejudice. She was born in England.
Finish[England]

Question: Are the Eiffel Tower and the Louvre in the same city?
The Eiffel Tower is in Paris, and so is the Louvre.
Finish[yes]

Question: Who directed the 2010 science fiction film Inception?
The 2010 science fiction film Inception was directed by Christopher Nolan.
Finish[Christopher Nolan]

Question: In what year did the Berlin Wall fall?
The Berlin Wall was opened and fell in November 1989.
Finish[1989]""",
    "gsm8k": """Solve the math problem. Show only the key arithmetic steps concisely, then end with one line in the form Finish[number]. Do not include units inside the brackets.
Use plain text only. Write the final line exactly as Finish[number], with no bullets, bold, backticks, or text after it.

Examples:
Question: Maya buys 3 packs of 8 pencils and gives away 5. How many does she have left?
She starts with 3 * 8 = 24 pencils. After giving away 5, she has 24 - 5 = 19.
Finish[19]

Question: A baker shares 30 rolls equally among 6 baskets. How many rolls are in each basket?
An equal share is 30 / 6 = 5 rolls.
Finish[5]

Question: A train travels 60 miles per hour for 3 hours and then 45 miles per hour for 2 hours. How many miles does it travel in total?
First leg: 60 * 3 = 180 miles. Second leg: 45 * 2 = 90 miles. Total: 180 + 90 = 270 miles.
Finish[270]

Question: Liam had 50 dollars. He bought 4 notebooks for 6 dollars each and a pen for 3 dollars. How many dollars does he have left?
He spends 4 * 6 = 24 dollars on notebooks and 24 + 3 = 27 dollars in total. He has 50 - 27 = 23 dollars left.
Finish[23]""",
}

REACT = """Solve the task: either answer a question using the supplied collection of paragraphs, or solve a math problem. Some paragraphs may be irrelevant, and answering may require combining information from more than one paragraph.

Available actions:
- Search[query]: search the shared collection of paragraphs and return the top matching paragraphs with their titles.
- Calculate[expression]: evaluate an arithmetic expression and return its result.
- Finish[answer]: provide the final answer and end the task. Give a short answer for a question (answer yes or no ONLY for yes/no questions), or only the final number for a math problem.

For each reply, give one brief Thought: line (one short sentence), then exactly one Action: line and stop. The system supplies Observation: lines after tool calls; never write one yourself. Use observations to refine your next query or finish when you have enough information. Do not repeat a tool call that already returned a result.
Use plain text for both lines. Do not use bullets, bold, backticks, or add text after the Action: line.

Example transcripts (each Observation: is supplied by the system):

Question: In what country was the author of Pride and Prejudice born?
Thought: First find who wrote the novel.
Action: Search[Pride and Prejudice author]
Observation: Pride and Prejudice is a novel by Jane Austen.
Thought: Now find where Jane Austen was born.
Action: Search[Jane Austen birthplace]
Observation: Jane Austen was born in Steventon, Hampshire, England.
Thought: The observation gives the country.
Action: Finish[England]

Question: Nora fills 4 boxes with 9 marbles each, then gives away 7 marbles. How many are left?
Thought: Multiply to find the total, then subtract the marbles she gives away.
Action: Calculate[4 * 9 - 7]
Observation: 29
Thought: The calculation gives the number left.
Action: Finish[29]

Question: Are the Eiffel Tower and the Louvre in the same city?
Thought: Check which city the Eiffel Tower is in.
Action: Search[Eiffel Tower city]
Observation: The Eiffel Tower is a wrought-iron lattice tower in Paris, France.
Thought: Now check which city the Louvre is in.
Action: Search[Louvre city]
Observation: The Louvre is a national art museum in Paris, France.
Thought: Both landmarks are in Paris.
Action: Finish[yes]

Question: Liam had 50 dollars. He bought 4 notebooks for 6 dollars each and a pen for 3 dollars. How many dollars does he have left?
Thought: First calculate the total amount Liam spent.
Action: Calculate[4 * 6 + 3]
Observation: 27
Thought: Subtract the total spent from 50 dollars.
Action: Calculate[50 - 27]
Observation: 23
Thought: The calculation gives the remaining amount.
Action: Finish[23]"""

REITERATE = """Original question: {question}
Continue solving this question using the information available so far.
Keep the Thought: and Action: lines in plain text, with no bullets, bold, or backticks."""

FORCE_FINISH = """Original question: {question}
Using the information collected so far, give your best final answer now. Do not request another tool action.
Use plain text only, with no bullets, bold, backticks, or text after the action.
Output only:
Action: Finish[{answer}]"""
