import time
import json
from langchain_core.prompts import ChatPromptTemplate
from src.parser import extract_and_clean_label
from src.prompt_manager import LONG_COT_PROMPT_TEMPLATE 

def run_long_cot(llm, text: str, instruction: str, task_type: str, possible_labels: list, max_retries: int = 5) -> tuple[str, str, float]:
    
    prompt = ChatPromptTemplate.from_template(LONG_COT_PROMPT_TEMPLATE)
    chain = prompt | llm 
    
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            message = chain.invoke({
                "text": text,
                "instruction": instruction
            })
            pure_duration = time.time() - start_time
            
            content = message.content
            
            captured_reasoning = ""
            debug_info = {}

            if hasattr(message, 'additional_kwargs'):
                debug_info['additional_kwargs'] = message.additional_kwargs
                if 'reasoning_content' in message.additional_kwargs:
                    captured_reasoning = message.additional_kwargs['reasoning_content']

            if hasattr(message, 'response_metadata'):
                meta = {k:v for k,v in message.response_metadata.items() if k != 'token_usage'}
                debug_info['response_metadata'] = meta
                
                if not captured_reasoning:
                    for key in ['reasoning', 'thoughts', 'thinking', 'reasoning_content']:
                        if key in meta:
                            captured_reasoning = str(meta[key])
                            break

            full_log_parts = []
            
            if captured_reasoning:
                full_log_parts.append(f"=== Captured Reasoning ===\n{captured_reasoning}\n==========================")
            else:
                full_log_parts.append(f"=== [Debug] No Reasoning Found ===\nMetadata Dump: {json.dumps(debug_info, ensure_ascii=False, indent=2)}\n==================================")

            full_log_parts.append(f"=== Model Output ===\n{content}")
            
            raw_response = "\n\n".join(full_log_parts)
            clean_response = extract_and_clean_label(content, possible_labels)
            
            return clean_response, raw_response, pure_duration
            
        except Exception as e:
            print(f"LongCoT Error: {e}")
            if attempt < max_retries - 1: time.sleep(3)
            else: return "Error", str(e), 0.0
                
    return "Error", "Error", 0.0