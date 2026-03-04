import time
import io
import contextlib
from typing import Tuple
from langchain_core.prompts import PromptTemplate
from langchain_experimental.smart_llm import SmartLLMChain
from langchain_core.messages import HumanMessage
from src.prompt_manager import TOT_PROMPT_TEMPLATE
from src.parser import extract_and_clean_label


# SmartLLMChain Version
def _run_tot_smartllm(llm, text: str, instruction: str, possible_labels: list, 
                      num_thoughts: int = 3, max_retries: int = 5) -> Tuple[str, str, float]:
    prompt_content = TOT_PROMPT_TEMPLATE.format(instruction=instruction, text=text)
    prompt = PromptTemplate.from_template(prompt_content)

    chain = SmartLLMChain(llm=llm, prompt=prompt, n_ideas=num_thoughts, verbose=True)

    for attempt in range(max_retries):
        try:
            start_time = time.time()

            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                result = chain.invoke({})

            full_thinking_log = f.getvalue()
            
            pure_duration = time.time() - start_time
            
            if isinstance(result, dict):
                resolution = result.get('resolution', result.get('output', str(result)))
            else:
                resolution = str(result)

            if not full_thinking_log.strip():
                full_thinking_log = f"[Warning: No verbose log captured]\nFinal Resolution: {resolution}"

            if "data_inspection_failed" in resolution.lower() or "inappropriate content" in resolution.lower():
                return "Error: Safety Filter", full_thinking_log, 0.0

            clean_response = extract_and_clean_label(resolution, possible_labels)
            
            return clean_response, full_thinking_log, pure_duration

        except Exception as e:
            err_str = str(e).lower()
            if "data_inspection_failed" in err_str or "inappropriate content" in err_str:
                return "Error: Safety Filter", str(e), 0.0
            
            print(f"ToT SmartLLM Attempt {attempt+1} failed: {str(e)[:200]}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return f"Error: {str(e)[:150]}", str(e), 0.0
    
    return "Error: Unknown", "Error: Unknown", 0.0


# Manual Smart-style ToT
def _run_tot_manual_smart_gemma(llm, text: str, instruction: str, possible_labels: list, 
                                num_thoughts: int = 3, max_retries: int = 5) -> Tuple[str, str, float]:

    start_time = time.time()
    
    full_trace_log = f"=== ToT (Manual) Start ===\nTime: {time.ctime()}\n"

    idea_prompt_base = TOT_PROMPT_TEMPLATE.format(instruction=instruction, text=text)
    
    ideas = []
    critiques = []

    # Step 1: Generate Ideas
    full_trace_log += "\n--- Step 1: Ideation ---\n"
    for i in range(num_thoughts):
        for attempt in range(max_retries):
            try:
                idea_prompt = f"""{idea_prompt_base}\n\nPlease provide one detailed analysis...\nIdea {i+1}:"""
                response = llm.invoke([HumanMessage(content=idea_prompt)])
                idea_text = response.content.strip()
                
                if "data_inspection_failed" in idea_text.lower():
                    idea_text = "Analysis blocked by safety filter."
                
                ideas.append(idea_text)
                full_trace_log += f"\n[Idea {i+1}]:\n{idea_text}\n"
                break
            except Exception as e:
                print(f"  ⚠️ Idea {i+1} attempt {attempt+1} failed: {str(e)[:150]}")
                time.sleep(2)
        else:
            fail_msg = "Failed to generate idea."
            ideas.append(fail_msg)
            full_trace_log += f"\n[Idea {i+1}]: {fail_msg}\n"

    # Step 2: Critique each idea
    full_trace_log += "\n--- Step 2: Critique ---\n"
    for idx, idea in enumerate(ideas):
        for attempt in range(max_retries):
            try:
                critique_prompt = f"""You are a critical evaluator.\nHere is one proposed solution:\n{idea}\n\nTask: {instruction}\nText: "{text}"\n\nCritique this solution..."""
                response = llm.invoke([HumanMessage(content=critique_prompt)])
                critique_text = response.content.strip()
                critiques.append(critique_text)
                full_trace_log += f"\n[Critique {idx+1}]:\n{critique_text}\n"
                break
            except Exception as e:
                print(f"  ⚠️ Critique {idx+1} attempt {attempt+1} failed")
                time.sleep(2)
        else:
            fail_msg = "Failed to generate critique."
            critiques.append(fail_msg)
            full_trace_log += f"\n[Critique {idx+1}]: {fail_msg}\n"

    # Step 3: Resolve / Synthesize
    full_trace_log += "\n--- Step 3: Resolution ---\n"
    resolve_prompt = f"""You are the Lead Reasoner.\nHere are {num_thoughts} proposed analyses with their critiques:\n\n"""
    for i in range(num_thoughts):
        resolve_prompt += f"=== Idea {i+1} ===\n{ideas[i]}\n\n=== Critique {i+1} ===\n{critiques[i]}\n\n"

    resolve_prompt += f"""Task: {instruction}\nText: "{text}"\nCategories: {', '.join(possible_labels)}\nSynthesize the best reasoning..."""

    for attempt in range(max_retries):
        try:
            response = llm.invoke([HumanMessage(content=resolve_prompt)])
            raw_response = response.content.strip()
            
            full_trace_log += f"\n[Final Resolution]:\n{raw_response}\n"
            
            clean_label = extract_and_clean_label(raw_response, possible_labels)
            total_duration = time.time() - start_time
            
            return clean_label, full_trace_log, total_duration
        except Exception as e:
            print(f"  ⚠️ Resolve attempt {attempt+1} failed: {str(e)[:150]}")
            time.sleep(2)

    total_duration = time.time() - start_time
    return "Error: Resolve Failed", full_trace_log, total_duration


def run_tot(llm, text: str, instruction: str, task_type: str, possible_labels: list, 
            num_thoughts: int = 3, max_retries: int = 5, model_name: str = "") -> tuple[str, str, float]:
    
    model_lower = model_name.lower().strip()
    is_gemma = any(name in model_lower for name in ["gemma", "gemma-3", "gemma3", "google/gemma"])

    if is_gemma:
        return _run_tot_manual_smart_gemma(llm, text, instruction, possible_labels, num_thoughts, max_retries)
    else:
        return _run_tot_smartllm(llm, text, instruction, possible_labels, num_thoughts, max_retries)