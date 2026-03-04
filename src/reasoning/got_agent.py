import time
from langchain_core.prompts import ChatPromptTemplate
from src.prompt_manager import GOT_EXPANSION_PROMPT, GOT_AGGREGATE_PROMPT, GOT_OUTPUT_PROMPT
from src.parser import extract_and_clean_label
from src.utils import parse_llm_output

def run_got(llm, text: str, instruction: str, task_type: str, possible_labels: list, max_retries: int = 5) -> tuple[str, str, float]:
    
    total_pure_time = 0.0
    full_process_log = "" 
    
    perspectives = [
        "Literal Meaning & Explicit Keywords",
        "Context, Tone & Implicit Intent",
        "Critical Analysis (Potential Traps)"
    ]
    
    thoughts = []
    expand_chain = ChatPromptTemplate.from_template(GOT_EXPANSION_PROMPT) | llm 

    # Node 1: Expansion
    full_process_log += "=== Node 1: Expansion ===\n"
    for i, p in enumerate(perspectives):
        for attempt in range(3):
            try:
                t_start = time.time()
                msg = expand_chain.invoke({"instruction": instruction, "text": text, "perspective": p})
                total_pure_time += (time.time() - t_start)
                
                full_log, content = parse_llm_output(msg)
                thoughts.append(content)
                full_process_log += f"--- Perspective: {p} ---\n{full_log}\n"
                break 
            except Exception as e:
                time.sleep(1)
        else:
            thoughts.append("Analysis skipped.")
    
    # Node 2: Aggregation
    full_process_log += "\n=== Node 2: Aggregation ===\n"
    agg_chain = ChatPromptTemplate.from_template(GOT_AGGREGATE_PROMPT) | llm 
    aggregated_thought = ""
    
    try:
        t_start = time.time()
        msg = agg_chain.invoke({
            "instruction": instruction, "text": text,
            "thought_1": thoughts[0], "thought_2": thoughts[1], "thought_3": thoughts[2]
        })
        total_pure_time += (time.time() - t_start)
        full_log, aggregated_thought = parse_llm_output(msg)
        full_process_log += f"{full_log}\n"
    except Exception as e:
        full_process_log += f"Aggregation Error: {e}\n"

    # Node 3: Output
    full_process_log += "\n=== Node 3: Final Output ===\n"
    out_chain = ChatPromptTemplate.from_template(GOT_OUTPUT_PROMPT) | llm 
    
    final_raw = ""
    try:
        t_start = time.time()
        msg = out_chain.invoke({
            "instruction": instruction,
            "aggregate_thought": aggregated_thought,
            "labels_list": ", ".join(possible_labels)
        })
        total_pure_time += (time.time() - t_start)
        full_log, final_raw = parse_llm_output(msg)
        full_process_log += f"{full_log}"
    except Exception as e:
        full_process_log += f"Output Error: {e}\n"

    clean_label = extract_and_clean_label(final_raw, possible_labels)
    if clean_label == "unparsed":
         clean_label = extract_and_clean_label(aggregated_thought, possible_labels)

    return clean_label, full_process_log, total_pure_time