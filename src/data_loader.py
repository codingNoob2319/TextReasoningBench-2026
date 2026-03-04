import os
from datasets import load_dataset
from typing import List, Dict, Union
import random
from collections import defaultdict

def get_dataset(dataset_name: str) -> List[Dict[str, Union[str, int]]]:
    """
    Supported datasets: 'sst2', 'ag_news', 'semeval', 'trec', 'isarcasm', 'ag_news_subset'
    """
    normalized_name = dataset_name.lower()
    print(f"Loading dataset: {normalized_name}...")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_directory = os.path.join(project_root, 'data', normalized_name)
    print(f"Dataset '{normalized_name}' will be loaded/cached at: {cache_directory}")
    
    os.makedirs(cache_directory, exist_ok=True)

    dataset = None
    label_map = {}
    text_column = 'text'
    label_column = 'label'

    try:
        if normalized_name == 'sst2':
            dataset = load_dataset('sst2', split='validation', cache_dir=cache_directory)
            label_map = {0: 'negative', 1: 'positive'}
            text_column = 'sentence'
            
        elif normalized_name in ['agnews', 'ag_news']:
            dataset = load_dataset('fancyzhx/ag_news', split='test', cache_dir=cache_directory)
            label_names = dataset.features[label_column].names
            label_map = {i: name for i, name in enumerate(label_names)}
            text_column = 'text'

        elif normalized_name == 'semeval':
            print("Loading irony subset from tweet_eval (SemEval 2018)...")
            dataset = load_dataset('tweet_eval', 'irony', split='test', cache_dir=cache_directory)
            label_map = {0: 'non-ironic', 1: 'ironic'}
            text_column = 'text'

        elif normalized_name == 'trec':
            print("Loading TREC-QC (SetFit/TREC-QC Parquet version)...")
            
            try:
                dataset = load_dataset("SetFit/TREC-QC", split='test', cache_dir=cache_directory)
            except Exception as e:
                print(f"API load failed ({e}), attempting direct download of Parquet file...")
                parquet_url = "https://hf-mirror.com/datasets/SetFit/TREC-QC/resolve/main/data/test-00000-of-00001.parquet"
                dataset = load_dataset("parquet", data_files={'test': parquet_url}, split='test', cache_dir=cache_directory)

            raw_to_standard = {
                "abbr": "Abbreviation",
                "abbreviation": "Abbreviation",
                
                "desc": "Description",
                "description": "Description",
                "description and abstract concepts": "Description",
                
                "enty": "Entity",
                "entity": "Entity",
                "entities": "Entity",  
                
                "hum": "Human",
                "human": "Human",
                "human beings": "Human",
                
                "loc": "Location",
                "location": "Location",
                "locations": "Location",
                
                "num": "Number",
                "numeric": "Number",
                "numeric values": "Number"
            }
            
            processed_data = []
            unknown_labels = set()
            
            target_label_col = 'label_coarse_text'
            if target_label_col not in dataset.column_names:
                print("⚠️ Text label column not found, using default ID mapping...")
                target_label_col = 'label_coarse'
                raw_to_standard = {
                    0: "Abbreviation",
                    1: "Description", 
                    2: "Entity",      
                    3: "Human",
                    4: "Location",
                    5: "Number"
                }

            for example in dataset:
                try:
                    txt = example['text']
                    raw_val = example[target_label_col]
                    
                    if isinstance(raw_val, str):
                        lookup_key = raw_val.strip().lower()
                    else:
                        lookup_key = raw_val
                    
                    if lookup_key in raw_to_standard:
                        lbl = raw_to_standard[lookup_key]
                        processed_data.append({
                            'text': txt,
                            'label': lbl
                        })
                    else:
                        unknown_labels.add(str(raw_val))
                except Exception as e:
                    continue
            
            if unknown_labels:
                print(f"⚠️ Warning: Found {len(unknown_labels)} undefined labels that were discarded: {unknown_labels}")
                print("Please add these labels to the mapping dictionary in data_loader.py!")
            
            print(f"Dataset 'trec' loaded and processed. Loaded {len(processed_data)}/{len(dataset)} test samples.")
            return processed_data
        

        elif normalized_name == 'isarcasm':
            print("Loading iSarcasmEval (Source: iabufarha/iSarcasmEval - Task A)...")
            
            # Target URL (Raw file from GitHub)
            github_url = "https://raw.githubusercontent.com/iabufarha/iSarcasmEval/main/test/task_A_En_test.csv"
            
            try:
                # Plan A: Attempt to load CSV directly from GitHub URL
                print(f"Attempting to load from URL: {github_url}")
                dataset = load_dataset('csv', data_files={'test': github_url}, split='test', cache_dir=cache_directory)
            except Exception as e_url:
                print(f"⚠️ Failed to load from URL (possible network issue): {e_url}")
                print(f"Attempting to find local file: {os.path.join(cache_directory, 'task_A_En_test.csv')}")
                
                # Plan B: Fallback to local file if network fails
                local_file = os.path.join(cache_directory, 'task_A_En_test.csv')
                if os.path.exists(local_file):
                    dataset = load_dataset('csv', data_files={'test': local_file}, split='test', cache_dir=cache_directory)
                else:
                    print("❌ Error: Cannot access GitHub, nor found 'task_A_En_test.csv' locally.")
                    print("Please download the file manually and place it in the data/isarcasm/ directory.")
                    return []

            # iSarcasmEval Task A is binary classification
            label_map = {0: 'non-ironic', 1: 'ironic'}
            text_column = 'text'
            label_column = 'sarcastic' 
        
        elif dataset_name == "ag_news_subset":
            base_path = os.getcwd()
            csv_path = os.path.join(base_path, "data/ag_news_subset/test.csv")
            
            dataset = load_dataset("csv", data_files=csv_path, split="train")
            
            label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
            
            text_column = 'text'
            label_column = 'label'
            
        
        else:
            raise ValueError(f"Unsupported dataset: '{dataset_name}'.")

    except Exception as e:
        print(f"Failed to load dataset '{dataset_name}': {e}")
        return []

    processed_data = []
    for example in dataset:
        try:
            txt = example[text_column]
            raw_label = example[label_column]
            
            if raw_label in label_map:
                lbl = label_map[raw_label]
                processed_data.append({
                    'text': txt,
                    'label': lbl
                })
        except Exception as e:
            continue
    
    print(f"Dataset '{dataset_name}' loaded and processed. Total {len(processed_data)} test samples loaded.")
    return processed_data


def get_few_shot_examples(dataset_name: str, k: int = 1) -> list:

    from src.few_shot_data import FEW_SHOT_EXAMPLES
    
    normalized_name = dataset_name.lower()
    if normalized_name in ['agnews', 'ag_news_subset']:
        normalized_name = 'ag_news'
        
    if normalized_name not in FEW_SHOT_EXAMPLES:
        print(f"⚠️ Warning: No hardcoded Few-Shot examples found for '{dataset_name}' in the dictionary!")
        return []

    return FEW_SHOT_EXAMPLES[normalized_name]