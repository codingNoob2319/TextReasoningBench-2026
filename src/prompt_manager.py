# IO Prompt
IO_PROMPT_TEMPLATE = """{instruction}

Constraint: You must choose ONE category from the list exactly as it appears. 
Do NOT output any other words, punctuation, or explanations. 

Text: "{text}"

Category:"""



# CoT Prompt
COT_PROMPT_TEMPLATE = """Think step by step. Follow these steps to classify the given text:

1. **Analyze**: Briefly analyze the content, keywords, and {task_type} of the text.
2. **Decide**: Based on your analysis, choose the single most appropriate category.
3. **Output**: strictly follow this format: "Final Answer: [Category Name]"

{instruction}

Text: "{text}"

Analysis:
"""



# ToT Prompt
TOT_PROMPT_TEMPLATE = """You are an expert classifier.

Task: {instruction}

Text to Classify:
"{text}"

--------------------------------------------------
GOAL: Determine the most accurate category for the text above.

INSTRUCTIONS:
1. Analyze the text content deeply.
2. Critique different possibilities.
3. Provide a consolidated Final Answer.

Your Final Answer must clearly state the category name.

Final Answer: [Category]
"""



# SC_CoT Prompt
SC_PROMPT_VARIANTS = [
    # Variant 1: Standard
    """Think step by step.
Task: {instruction}
Text: "{text}"

1. Analyze content.
2. Determine category.
3. Output strictly: "Final Answer: [Category]"
""",

    # Variant 2: Minimalist
    """{instruction}
Text: "{text}"

Let's think step by step.
Reasoning: ...
Final Answer: [Category]
""",

    # Variant 3: Strict Analyst (Role Play)
    """Let's think step by step.You are a strict data annotator.
Task: {instruction}
Text: "{text}"

Scrutinize the text. Weigh the evidence.
Conclusion format: Final Answer: [Category]
""",

    # Variant 4: Process of Elimination
    """Let's think step by step.
Task: {instruction}
Text: "{text}"

Methodology:
1. Identify which categories are definitely WRONG and why.
2. Identify the ONE correct category.

You MUST end your response with:
Final Answer: [Category]
""",
    
    # Variant 5: Intent Analysis
    """Let's think step by step.
Task: {instruction}
Text: "{text}"

Analyze the intent. Is it describing a company, a school, or a person?
Output: Final Answer: [Category]
"""
]



# GoT Prompt

# Node 1: Expansion
GOT_EXPANSION_PROMPT = """You are an analytical assistant.
Task: {instruction}
Text: "{text}"

Please analyze this text from the following cognitive perspective: **{perspective}**.
- Focus ONLY on this perspective.
- Identify evidence relevant to the task.
- Do not draw a final conclusion yet, just provide the analysis.

Analysis:
"""

# Node 2: Aggregation
GOT_AGGREGATE_PROMPT = """You are a Lead Reasoner. You have received three independent analyses of the same text based on different perspectives.

Task: {instruction}
Text: "{text}"

--- Perspective 1 (Literal/Factual) ---
{thought_1}

--- Perspective 2 (Contextual/Intent) ---
{thought_2}

--- Perspective 3 (Critical/Counter-argument) ---
{thought_3}

-------------------------
YOUR JOB:
1. Synthesize these analyses into a single coherent reasoning path.
2. Resolve any conflicts between the perspectives.
3. Determine which evidence is strongest.

Synthesized Reasoning:
"""

# Node 3: Output
GOT_OUTPUT_PROMPT = """Based on the following synthesized reasoning, categorize the text.

Task: {instruction}
Synthesized Reasoning: "{aggregate_thought}"

Constraint: You must choose exactly ONE category from: {labels_list}.
Output ONLY the category name.

Final Answer: [Category]
"""



# Long_CoT Prompt
# Designed specifically for models with native thinking capabilities (e.g., o1, DeepSeek-R1). Kept concise to avoid interfering with internal reasoning chains.
LONG_COT_PROMPT_TEMPLATE = """Task: {instruction}

Text to Classify:
"{text}"

Requirement:
1. You should think deeply about the classification.
2. But your final output must be strictly in the specified format.

Final Answer: [Category]
"""



# BoC Prompt
# Randomly sample a subset of cues to force the model to analyze from specific perspectives, followed by voting.
BOC_PROMPT_TEMPLATE = """You are an expert analyst.
Task: {instruction}

Text: "{text}"

--------------------------------------------------
Please analyze the text specifically focusing on the following cues/aspects:
[{cues}]

Ignore other aspects for a moment. Based strictly on the cues above, determine the most accurate category.
Output strictly in the format: "Final Answer: [Category]"

Final Answer: [Category]
"""



# Few-Shot IO Prompt
FEW_SHOT_IO_PROMPT_TEMPLATE = """{instruction}

Constraint: You must choose ONE category from the list exactly as it appears. 
Do NOT output any other words, punctuation, or explanations. 

Here are some examples:
{few_shot_examples}

Now, classify the following text:
Text: "{text}"

Category:"""




# Few-Shot CoT Prompt
FEW_SHOT_COT_PROMPT_TEMPLATE = """Think step by step. Follow these steps to classify the given text:

1. **Analyze**: Briefly analyze the content, keywords, and {task_type} of the text.
2. **Decide**: Based on your analysis, choose the single most appropriate category.
3. **Output**: strictly follow this format: "Final Answer: [Category Name]"

{instruction}

Here are some examples of how to think and classify:
{few_shot_examples}

Now, classify the following text:
Text: "{text}"

Analysis:
"""