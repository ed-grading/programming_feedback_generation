from langchain_core.prompts import ChatPromptTemplate

j1_thinking_judge_prompt = ChatPromptTemplate.from_template(
    """
You are a CS professor and you want to evaluate the quality of feedback provided by a teaching assistant for a student's buggy code.

<problem description>
{problem_description}
</problem description>

<test cases>
{test_cases}
</test cases>

<student code>
{student_code}
</student code>

<ta_feedback>
{ta_feedback}
</ta_feedback>

Your task is to evaluate and check if the given TA feedback follows the grading criteria.

First, provide your reasoning within <think> and </think> tags. This should include:
- Your understanding of what the actual bugs are in the student code
- What constitutes relevant educational feedback for these bugs
- A detailed analysis of the TA's feedback
- When helpful, what you would consider an ideal feedback response for the student's code
- How the TA's feedback aligns with or deviates from good educational practices and teaching pedagogies

Be explicit in your thought process and consider each of the grading criteria below.

<grading_criteria>
{grading_criteria}
</grading_criteria>

<grading_guidelines>
{grading_guidelines}
</grading_guidelines>

<think>
</think>

After your reasoning, provide your evaluation in two parts:

1. Brief Assessment:
Describe the quality of the TA's feedback in 2-3 sentences.

2. Criteria Evaluation: Criteria [yes/no]
""".strip()
)