import json
import torch
import warnings
from transformers import pipeline
from sklearn.metrics import classification_report
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
MODEL_NAME = "cross-encoder/nli-MiniLM2-L6-H768"
DATA_FILE = "clinic_data.json"
SAMPLE_SIZE = 1000  # Small sample just to find the errors
BATCH_SIZE = 32
DEVICE = 0 if torch.cuda.is_available() else -1

# ---------------------------------------------------------
# SEMANTIC MAP (Same as before)
# ---------------------------------------------------------

from test.clinic_experiment_quick import SEMANTIC_MAP, REVERSE_MAP, CANDIDATES

def run_diagnosis():
    print(f"--- CLINIC DIAGNOSIS RUN (1000 Samples) ---")
    
    try:
        classifier = pipeline("zero-shot-classification", model=MODEL_NAME, device=DEVICE)
    except:
        return

    with open(DATA_FILE, 'r') as f:
        raw_data = json.load(f)

    data_points = []
    # Merge splits
    for split in ['train', 'test', 'val']:
        if split in raw_data:
            data_points.extend(raw_data[split])
            
    data_points = data_points[:SAMPLE_SIZE]
    texts = [item[0] for item in data_points]
    true_labels = [item[1] for item in data_points]

    print("Running predictions...")
    pred_labels = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[i:i+BATCH_SIZE]
        results = classifier(batch_texts, CANDIDATES)
        if isinstance(results, dict): results = [results]
        
        for res in results:
            best_desc = res['labels'][0]
            pred_labels.append(REVERSE_MAP[best_desc])

    # ---------------------------------------------------------
    # ERROR ANALYSIS
    # ---------------------------------------------------------
    print("\n" + "="*40)
    print("LOWEST PERFORMING CLASSES (< 20% F1)")
    print("="*40)
    
    report = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)
    
    # Sort by F1 score
    sorted_classes = sorted(report.items(), key=lambda x: x[1]['f1-score'] if isinstance(x[1], dict) else 1.0)
    
    low_performers = []
    for label, metrics in sorted_classes:
        if label in ['accuracy', 'macro avg', 'weighted avg']: continue
        if metrics['f1-score'] < 0.20:
            low_performers.append(label)
            print(f"Intent: {label:<25} | F1: {metrics['f1-score']:.2f} | Precision: {metrics['precision']:.2f} | Recall: {metrics['recall']:.2f}")

    print(f"\nTotal Low Performers: {len(low_performers)} out of {len(set(true_labels))}")

    print("\n" + "="*40)
    print("CONFUSION PAIRS (Top 10)")
    print("="*40)

    confusion_map = {}
    
    for true, pred in zip(true_labels, pred_labels):
        if true != pred:
            pair = f"{true} -> {pred}"
            confusion_map[pair] = confusion_map.get(pair, 0) + 1
            
    sorted_confusion = sorted(confusion_map.items(), key=lambda x: x[1], reverse=True)
    
    for pair, count in sorted_confusion[:15]:
        print(f"{pair:<50} (Count: {count})")

if __name__ == "__main__":
    run_diagnosis()