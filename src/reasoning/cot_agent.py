import time
from langchain_core.prompts import ChatPromptTemplate
from src.prompt_manager import COT_PROMPT_TEMPLATE, FEW_SHOT_COT_PROMPT_TEMPLATE
from src.parser import extract_and_clean_label
from src.utils import parse_llm_output

def run_cot(llm, text: str, instruction: str, task_type: str, possible_labels: list = None, max_retries: int = 5, few_shot_str: str = "") -> tuple[str, str, float]:

    if few_shot_str:
        cot_prompt = ChatPromptTemplate.from_template(FEW_SHOT_COT_PROMPT_TEMPLATE)
        invoke_args = {"text": text, "instruction": instruction, "task_type": task_type, "few_shot_examples": few_shot_str}
    else:
        cot_prompt = ChatPromptTemplate.from_template(COT_PROMPT_TEMPLATE)
        invoke_args = {"text": text, "instruction": instruction, "task_type": task_type}

    cot_chain = cot_prompt | llm 
    
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            response_msg = cot_chain.invoke(invoke_args)
            pure_duration = time.time() - start_time
            
            full_log, pure_content = parse_llm_output(response_msg)
            
            clean_response = "unparsed"
            if possible_labels:
                clean_response = extract_and_clean_label(pure_content, possible_labels)
            
            return clean_response, full_log, pure_duration
            
        except Exception as e:
            print(f"CoT Failed: {e}")
            if attempt < max_retries - 1: time.sleep(2)
            else: return f"Error: {str(e)}", f"Error: {str(e)}", 0.0
                
    return "Error: Unknown", "Error: Unknown", 0.0