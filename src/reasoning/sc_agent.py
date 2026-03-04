import time
import random
from collections import Counter
from langchain_core.prompts import ChatPromptTemplate
from src.prompt_manager import SC_PROMPT_VARIANTS
from src.parser import extract_and_clean_label
from src.utils import parse_llm_output

def run_sc(llm, text: str, instruction: str, task_type: str, possible_labels: list, n_samples: int = 5) -> tuple[str, str, float]:
    
    answers = []
    raw_responses_log = [] 
    total_pure_time = 0.0
    
    for i in range(n_samples):
        current_prompt_template = random.choice(SC_PROMPT_VARIANTS)
        current_temp = random.uniform(0.5, 0.9)
        prompt = ChatPromptTemplate.from_template(current_prompt_template)
        
        runnable = prompt | llm.bind(temperature=current_temp) 

        try:
            start_time = time.time()
            msg = runnable.invoke({
                "text": text,
                "instruction": instruction,
                "task_type": task_type
            })
            total_pure_time += (time.time() - start_time)
            
            full_log, pure_content = parse_llm_output(msg)
            
            raw_responses_log.append(f"=== Sample {i+1} ===\n{full_log}\n")
            
            parsed_label = extract_and_clean_label(pure_content, possible_labels)
            if parsed_label != "unparsed":
                answers.append(parsed_label)
                
        except Exception as e:
            print(f"SC Sample {i+1} failed: {e}")
            raw_responses_log.append(f"=== Sample {i+1} ===\nError: {e}\n")

    if not answers:
        final_winner = "unparsed"
    else:
        counts = Counter(answers)
        final_winner = counts.most_common(1)[0][0]

    full_log_str = "\n".join(raw_responses_log)
    
    return final_winner, full_log_str, total_pure_time