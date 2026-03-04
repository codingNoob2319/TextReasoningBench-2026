import numpy as np
import os
from tqdm import tqdm
import time
from dotenv import load_dotenv

load_dotenv() 

from src.data_loader import get_dataset, get_few_shot_examples
from src.llm_provider import get_llm
from src.evaluator import Evaluator
from src.utils import save_results, get_checkpoint_path, load_checkpoint, append_to_checkpoint

# Agent
from src.reasoning.io_agent import run_io
from src.reasoning.cot_agent import run_cot
from src.reasoning.tot_agent import run_tot
from src.reasoning.sc_agent import run_sc
from src.reasoning.got_agent import run_got
from src.reasoning.long_cot_agent import run_long_cot
from src.reasoning.boc_agent import run_boc

def get_task_info(dataset_name):
    normalized_dataset_name = dataset_name.lower()
    if normalized_dataset_name in ['sst2', 'imdb', 'tweet_eval_emotion']:
        task_type, labels = "sentiment", ['positive', 'negative']
        if normalized_dataset_name == 'tweet_eval_emotion': labels = ['anger', 'joy', 'optimism', 'sadness']
        instruction = f"Classify the text into one of the following {task_type} categories: {', '.join(labels)}."
    elif normalized_dataset_name in ['semeval']:
        task_type = "irony"
        labels = ['ironic', 'non-ironic']
        instruction = f"Classify whether the text is ironic or not. Choose from: {', '.join(labels)}."
    elif normalized_dataset_name in ['ag_news', 'ag_news_subset']:
        task_type, labels = "topic", ['World', 'Sports', 'Business', 'Sci/Tech']
        instruction = f"Classify the text into one of the following {task_type} categories: {', '.join(labels)}."
    elif normalized_dataset_name == 'trec':
        task_type = "question_classification"
        labels = ['Abbreviation', 'Entity', 'Description', 'Human', 'Location', 'Number']
        instruction = (
            "Classify the following question based on the type of answer it requires.\n"
            f"Categories: {', '.join(labels)}.\n"
            "Constraint: Choose the category that best represents the expected answer type."
        )
    elif normalized_dataset_name in ['semeval', 'isarcasm']:
        task_type = "irony"
        labels = ['ironic', 'non-ironic']
        instruction = f"Classify whether the text is ironic or not. Choose from: {', '.join(labels)}."
    else:
        raise ValueError(f"Unsupported dataset: '{dataset_name}' in get_task_info.")
    return instruction, task_type, labels

def run_single_pass(llm, dataset, dataset_name, method, evaluator, sc_samples, model_name, run_index, few_shot_str=""):
    results_details = []
    total_time = 0.0
    
    instruction, task_type, possible_labels = get_task_info(dataset_name)

    ckpt_path = get_checkpoint_path(model_name, dataset_name, method, run_index)
    saved_results = load_checkpoint(ckpt_path)
    processed_count = len(saved_results)
    if processed_count > 0:
        results_details.extend(saved_results)
        total_time = sum([r.get('duration', 0) for r in saved_results])
    remaining_dataset = dataset[processed_count:]

    for sample in tqdm(remaining_dataset, desc=f"处理 {dataset_name} (Run {run_index})"):
        text, true_label = sample['text'], sample['label']
        
        clean_ans = "unparsed"
        raw_output = ""
        duration = 0.0
        
        try:
            if method == 'io':
                clean_ans, raw_output, duration = run_io(llm, text, instruction, few_shot_str=few_shot_str)
            elif method == 'cot':
                clean_ans, raw_output, duration = run_cot(llm, text, instruction, task_type, possible_labels, few_shot_str=few_shot_str) 
            elif method == 'tot':
                clean_ans, raw_output, duration = run_tot(llm, text, instruction, task_type, possible_labels, num_thoughts=3, model_name=model_name)
            elif method == 'sc_cot':
                clean_ans, raw_output, duration = run_sc(llm, text, instruction, task_type, possible_labels, sc_samples)
            elif method == 'got':
                clean_ans, raw_output, duration = run_got(llm, text, instruction, task_type, possible_labels)
            elif method == 'long_cot':
                clean_ans, raw_output, duration = run_long_cot(llm, text, instruction, task_type, possible_labels)
            elif method == 'boc':
                clean_ans, raw_output, duration = run_boc(llm, text, instruction, task_type, possible_labels, n_estimators=3, cues_per_bag=3)
            else:
                raise ValueError(f"Unsupported reasoning method: {method}")
            
            total_time += duration
            
            result_item = {
                'text': text, 
                'true_label': true_label, 
                'model_response': raw_output,
                'parsed_answer': clean_ans,
                'duration': duration
            }
            
            results_details.append(result_item)
            append_to_checkpoint(ckpt_path, result_item)
                
        except Exception as e:
            print(f"Error processing sample, skipping: {e}")
            continue
    
    metrics = evaluator.evaluate(results_details)
    return {'metrics': metrics, 'total_time_seconds': total_time, 'avg_time_per_sample': total_time / len(dataset) if dataset else 0, 'details': results_details}

def run_experiment_suite(experiment_config):
    model_name, dataset_name, method = experiment_config['model'], experiment_config['dataset'], experiment_config['reasoning_method']
    num_runs = experiment_config.get('runs', 1)
    sc_samples = experiment_config.get('consistency_samples', 5)
    temperature = experiment_config.get('temperature', 0.7)
    
    config_enable_thinking = experiment_config.get('enable_thinking', False)
    
    if method == 'long_cot':
        print("Method 'long_cot' detected. API Thinking Mode enabled.")
        enable_thinking = True
    else:
        enable_thinking = config_enable_thinking
    
    print(f"\n{'='*80}\n Starting experiment suite: [Model: {model_name}] - [Dataset: {dataset_name}] - [Method: {method}]\n{'='*80}\n")
    
    try:
        llm = get_llm(model_name, temperature=temperature, enable_thinking=enable_thinking)
        full_dataset = get_dataset(dataset_name)
        if not full_dataset: return
        
        _, _, possible_labels = get_task_info(dataset_name)
        evaluator = Evaluator(dataset_name, possible_labels)
            
        dataset_to_run = full_dataset[:100] if experiment_config.get('debug_mode', True) else full_dataset
        if experiment_config.get('debug_mode', True): print(f"--- [Fast Test Mode] Enabled. Using only {len(dataset_to_run)} samples. ---")
    
    except Exception as e:
        print(f"Experiment initialization failed: {e}"); return


    few_shot_k = experiment_config.get('few_shot_k', 0)
    few_shot_str = ""
    if few_shot_k > 0 and method in ['io', 'cot']:
        raw_examples = get_few_shot_examples(dataset_name, k=few_shot_k)
        for ex in raw_examples:
            if method == 'io':
                few_shot_str += f"Text: \"{ex['text']}\"\nCategory: {ex['label']}\n\n"
            else: 
                few_shot_str += f"Text: \"{ex['text']}\"\nAnalysis: {ex['analysis']}\nFinal Answer: {ex['label']}\n\n"


    all_runs_results = []
    for i in range(num_runs):
        print(f"\n--- Running iteration {i+1}/{num_runs} ---")
        single_run_result = run_single_pass(llm, dataset_to_run, dataset_name, method, evaluator, sc_samples, model_name, i+1, few_shot_str)
        all_runs_results.append(single_run_result)
        
        m = single_run_result['metrics']
        print(f"Iteration {i+1} completed. Acc: {m['accuracy']:.4f}, Macro-F1: {m['macro_f1']:.4f}")

    print(f"\n{'='*80}\n All iterations completed. Starting statistical analysis...")
    
    accuracies = [run['metrics']['accuracy'] for run in all_runs_results]
    macro_f1s = [run['metrics']['macro_f1'] for run in all_runs_results]
    weighted_f1s = [run['metrics']['weighted_f1'] for run in all_runs_results]
    binary_f1s = [run['metrics']['binary_f1'] for run in all_runs_results]
    
    aggregated_metrics = {
        'mean_accuracy': np.mean(accuracies), 'std_accuracy': np.std(accuracies), 
        'mean_macro_f1': np.mean(macro_f1s), 'std_macro_f1': np.std(macro_f1s),
        'mean_weighted_f1': np.mean(weighted_f1s), 'std_weighted_f1': np.std(weighted_f1s),
        'mean_binary_f1': np.mean(binary_f1s), 'std_binary_f1': np.std(binary_f1s)
    }
    
    print("Final Statistical Results:")
    for key, value in aggregated_metrics.items(): 
        if "binary" in key and value == 0: continue
        print(f"  - {key}: {value:.4f}")
    
    run_durations = [run['total_time_seconds'] for run in all_runs_results]
    suite_total_time = np.sum(run_durations)
    suite_avg_time_per_run = np.mean(run_durations)
    
    final_output = {'config': experiment_config, 'aggregated_metrics': aggregated_metrics, 
                    'suite_time_stats': {'total_time_seconds': suite_total_time, 'avg_time_per_run_seconds': suite_avg_time_per_run},
                    'individual_runs': all_runs_results}
    
    save_results(final_output)

    print("\nCleaning up checkpoint files for this experiment...")
    for i in range(num_runs):
        ckpt_path = get_checkpoint_path(model_name, dataset_name, method, i + 1)
        if os.path.exists(ckpt_path):
            try:
                os.remove(ckpt_path)
                print(f"- Deleted: {os.path.basename(ckpt_path)}")
            except Exception as e:
                print(f"  - Failed to delete {ckpt_path}: {e}")
    print("Cleanup completed!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Reasoning Benchmark for Text Classification")
    
    parser.add_argument('--model', type=str, default='qwen3-8b', help='Model name')
    parser.add_argument('--dataset', type=str, default='isarcasm', help='Dataset name')
    parser.add_argument('--method', type=str, default='io', help='Reasoning method: io, cot, tot, sc_cot, got, long_cot, boc')
    parser.add_argument('--runs', type=int, default=1, help='Number of independent runs')
    parser.add_argument('--temp', type=float, default=0.7, help='Generation temperature')
    parser.add_argument('--full', action='store_true', help='Run on full dataset (defaults to Debug mode if omitted)')
    parser.add_argument('--few_shot', type=int, default=0, help='Number of few-shot examples per class (default: 0 for Zero-Shot)')    
    args = parser.parse_args()

    experiment_config = {
        'model': args.model,
        'dataset': args.dataset,
        'reasoning_method': args.method,
        'runs': args.runs,
        'temperature': args.temp,
        'debug_mode': not args.full,
        'few_shot_k': args.few_shot
    }
    
    print(f"⚙️ Startup Config: Dataset={args.dataset} | Method={args.method} | Model={args.model} | FewShot_K={args.few_shot} | FullData={args.full}")
    
    run_experiment_suite(experiment_config)