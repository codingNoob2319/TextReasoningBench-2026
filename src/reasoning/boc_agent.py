import time
import random
from collections import Counter
from langchain_core.prompts import ChatPromptTemplate
from src.prompt_manager import BOC_PROMPT_TEMPLATE
from src.parser import extract_and_clean_label
from src.utils import parse_llm_output


# Task-Specific Cue Pools
TASK_CUE_POOLS = {
    # Irony & Metaphor (SemEval, iSarcasm)
    'irony': [
        "Semantic Incongruity (Contradiction between context and statement)",
        "Hyperbole & Exaggeration (Over-the-top descriptions)",
        "Pragmatic Context (Is the situation negative but words positive?)",
        "Punctuation & Emoji Usage (e.g., '...', '?!', inverted commas)",
        "Unexpected Shifts in Sentiment",
        "Mocking or Sarcastic Tone",
        "Explicit vs. Implicit Meaning",
        "Rhetorical Questions",
        "Understatement (Minimizing a significant event)"
    ],

    # Sentiment Classification (SST-2)
    'sentiment': [
        "Explicit Sentiment Adjectives (e.g., terrible, amazing)",
        "Intensity Modifiers (e.g., very, absolutely, barely)",
        "Negation Logic (e.g., not good, hardly exciting)",
        "Subjective vs. Objective Statements",
        "Emotional Connotation of Nouns/Verbs",
        "Conditionals & Concessions (but, although, despite)",
        "Overall Tone & Atmosphere",
        "User Intent (Praise vs. Complaint)"
    ],

    # News/Encyclopedia Topic (AG News)
    'topic': [
        "Key Named Entities (Persons, Organizations, Locations)",
        "Domain-Specific Terminology (e.g., 'touchdown' vs 'merger')",
        "Action Verbs & Events",
        "Geographical Context",
        "Numerical Data & Statistics (Prices, Scores, Dates)",
        "Subject Matter Keywords",
        "Official Titles & Roles",
        "Scientific vs. Political vs. Economic Context"
    ],
    
    # Question Classification (TREC)
    'question_classification': [
        "Wh-word Analysis (Who, What, Where, When, Why, How)",
        "Expected Answer Entity Type (Person, Place, Number...)",
        "Subject of the Question",
        "Grammatical Structure",
        "Contextual Keywords (e.g., 'stand for' implies Abbreviation)",
        "Abstract vs. Concrete Concepts",
        "Definition vs. Fact Retrieval"
    ],
    
    # General Fallback
    'general': [
        "Explicit Keywords",
        "Key Entities",
        "Emotional Tone",
        "Context & Background",
        "Writing Style",
        "Core Subject"
    ]
}

def run_boc(llm, text: str, instruction: str, task_type: str, possible_labels: list, n_estimators: int = 3, cues_per_bag: int = 3, max_retries: int = 5) -> tuple[str, str, float]:
    
    total_pure_time = 0.0
    votes = []
    full_process_log = ""
    
    selected_pool = TASK_CUE_POOLS.get(task_type, TASK_CUE_POOLS['general'])
    # 修改点：移除 StrOutputParser
    chain = ChatPromptTemplate.from_template(BOC_PROMPT_TEMPLATE) | llm 

    for i in range(n_estimators):
        k = min(cues_per_bag, len(selected_pool))
        current_cues = random.sample(selected_pool, k)
        cues_str = "\n- " + "\n- ".join(current_cues)
        
        full_process_log += f"=== Estimator {i+1} (Cues: {current_cues}) ===\n"
        
        response_content = "" # 仅用于提取标签
        
        for attempt in range(max_retries):
            try:
                t_start = time.time()
                msg = chain.invoke({
                    "instruction": instruction, "text": text, "cues": cues_str
                })
                total_pure_time += (time.time() - t_start)
                
                # 解析
                full_log, response_content = parse_llm_output(msg)
                full_process_log += f"{full_log}\n\n"
                break
            except Exception as e:
                time.sleep(1)
        
        if response_content:
            label = extract_and_clean_label(response_content, possible_labels)
            if label != "unparsed": votes.append(label)

    if not votes: return "unparsed", full_process_log, total_pure_time
    
    normalized_votes = [v.title() for v in votes] 
    final_winner = Counter(normalized_votes).most_common(1)[0][0]
    
    return final_winner, full_process_log, total_pure_time