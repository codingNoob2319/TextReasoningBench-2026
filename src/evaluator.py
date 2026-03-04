import re
from typing import List, Dict, Any
from sklearn.metrics import accuracy_score, f1_score
from src.parser import extract_and_clean_label

class Evaluator:
    def __init__(self, dataset_name: str, possible_labels: List[str]):
        self.dataset_name = dataset_name.lower()
        self.possible_labels = possible_labels 
        self.possible_labels_lower = [label.lower() for label in possible_labels]
        
        self.binary_config = {
            'sst2': 'positive',
            'semeval': 'ironic',
            'isarcasm': 'ironic'
        }

    def evaluate(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        true_labels = [str(res['true_label']).lower().strip() for res in results]
        pred_labels = []

        for res in results:
            if 'parsed_answer' in res:
                pred = str(res['parsed_answer'])
            else:
                pred = extract_and_clean_label(res['model_response'], self.possible_labels)
            
            pred_labels.append(str(pred).lower().strip())
        
        unparsed_count = pred_labels.count("unparsed")
        
        if unparsed_count > 0:
            print(f"Warning: {unparsed_count}/{len(results)} samples failed to parse a valid label.")
               
        accuracy = accuracy_score(true_labels, pred_labels)

        unique_labels = sorted(list(set(self.possible_labels_lower)))
        
        macro_f1 = f1_score(true_labels, pred_labels, average='macro', labels=unique_labels, zero_division=0)
        weighted_f1 = f1_score(true_labels, pred_labels, average='weighted', labels=unique_labels, zero_division=0)

        metrics = {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'unparsed_ratio': unparsed_count / len(results) if results else 0
        }

        if self.dataset_name in self.binary_config and self.binary_config[self.dataset_name]:
            pos_label = self.binary_config[self.dataset_name].lower()
            
            neg_labels = [l for l in self.possible_labels_lower if l != pos_label]
            neg_label = neg_labels[0] if neg_labels else "negative"

            binary_preds_clean = [pos_label if p == pos_label else neg_label for p in pred_labels]
            
            try:
                metrics['binary_f1'] = f1_score(true_labels, binary_preds_clean, pos_label=pos_label, average='binary', zero_division=0)
            except Exception as e:
                print(f"❌ Binary F1 calculation failed: {e}")
                metrics['binary_f1'] = 0.0
        else:
            metrics['binary_f1'] = 0.0

        return metrics