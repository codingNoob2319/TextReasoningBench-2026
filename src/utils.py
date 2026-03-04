import json
import os
from datetime import datetime
from langchain_core.messages import AIMessage

def get_checkpoint_path(model_name, dataset_name, method, run_index):
    """Generate the path for the checkpoint save file."""
    safe_model_name = model_name.replace('/', '_')
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_dir = os.path.join(project_root, 'results', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    
    filename = f"ckpt_{safe_model_name}_{dataset_name}_{method}_run{run_index}.jsonl"
    return os.path.join(ckpt_dir, filename)

def load_checkpoint(ckpt_path):
    """Load existing progress from a checkpoint."""
    results = []
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint: {ckpt_path}")
        try:
            with open(ckpt_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"Warning: Checkpoint file corrupted at line {line_num+1} (possibly due to an abnormal exit last time). Skipping this line.")
                        continue
        except Exception as e:
            print(f"❌ Failed to read checkpoint file: {e}")
            
    return results

def append_to_checkpoint(ckpt_path, result_item):
    """Append a single result item to the checkpoint file."""
    try:
        with open(ckpt_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result_item, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"Failed to write to checkpoint: {e}")

def save_results(results_data: dict):
    """Save the final results."""
    config = results_data['config']
    dataset_name = config['dataset']
    model_name_sanitized = config['model'].replace('/', '_')
    method = config['reasoning_method']
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name_sanitized}_{method}_{timestamp}.json"
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(project_root, 'results', dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    
    filepath = os.path.join(save_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=4)
        
    print(f"Experiment results successfully saved to: {filepath}")


def parse_llm_output(message) -> tuple[str, str]:
    """
    General Parser: Extract (content, reasoning) from the model's return message.
    Compatible with:
    1. GPT-5 Wrapper (via additional_kwargs['reasoning_content'])
    2. DeepSeek/Standard (via additional_kwargs['reasoning_content'] or XML tags in message.content)
    3. Standard models
    
    Returns:
        full_raw_log (str): The complete record for JSON storage (includes === Reasoning === separator)
        pure_content (str): Clean text for label extraction
    """
    content = ""
    reasoning = ""
    
    if isinstance(message, AIMessage):
        content = message.content
        if "reasoning_content" in message.additional_kwargs:
            reasoning = message.additional_kwargs["reasoning_content"]
        elif hasattr(message, "response_metadata") and "reasoning_content" in message.response_metadata:
             reasoning = message.response_metadata["reasoning_content"]
    else:
        content = str(message)

    if reasoning:
        full_raw_log = f"=== Reasoning ===\n{reasoning}\n\n=== Final Answer ===\n{content}"
    else:
        full_raw_log = content
        
    return full_raw_log, content