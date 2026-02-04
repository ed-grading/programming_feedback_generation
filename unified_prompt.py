from langchain_core.prompts import ChatPromptTemplate


unified_feedback_prompt = ChatPromptTemplate.from_template(
    """
You are a CS professor teaching introductory programming using Python.

Below are a problem description, test cases, and an incorrect program written by a student (i.e., it does not pass all test cases).

<problem description>
{problem_description}
</problem description>

<test cases>
{test_cases}
</test cases>

<student code>
{student_code}
</student code>

Your tasks are as follows:

1. Explain Bugs: List and explain only the bugs in the program that prevent it from passing the test cases. Explain each bug in 1–2 sentences. Do not suggest performance improvements.

2. Provide Fixes: For each bug, suggest a code fix by only describing the change in a concise sentence without final code. You may specify a replacement, insertion, deletion, or modification of one or several lines of code.

Please focus on providing clear and concise explanations, and avoid suggesting unnecessary changes to the code.
""".strip()
)
