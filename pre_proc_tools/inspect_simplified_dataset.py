import json
import os
from collections import Counter, defaultdict

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
DATA_FILE = "slurp_simplified.jsonl"

def inspect_dataset():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found. Run the previous script first.")
        return

    print(f"--- INSPECTING {DATA_FILE} ---\n")

    intent_counts = Counter()
    intent_examples = {} 
    total_samples = 0

    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            total_samples += 1
            data = json.loads(line)
            
            label = data['intent']
            sentence = data['sentence']
            
            intent_counts[label] += 1
            
            # Save the shortest sentence as the example (usually easiest to read)
            if label not in intent_examples:
                intent_examples[label] = sentence
            else:
                if len(sentence) < len(intent_examples[label]):
                    intent_examples[label] = sentence

    # ---------------------------------------------------------
    # REPORTING
    # ---------------------------------------------------------
    sorted_intents = sorted(intent_counts.keys())
    
    print(f"TOTAL SAMPLES: {total_samples}")
    print(f"TOTAL UNIQUE INTENTS (CLASSES): {len(sorted_intents)}")
    print("-" * 60)
    print(f"{'INTENT LABEL':<35} | {'COUNT':<5} | {'EXAMPLE SENTENCE'}")
    print("-" * 60)
    
    for intent in sorted_intents:
        count = intent_counts[intent]
        example = intent_examples[intent]
        print(f"{intent:<35} | {count:<5} | {example}")

    print("-" * 60)
    
    # ---------------------------------------------------------
    # GENERATE PYTHON LIST FOR YOUR NEXT SCRIPT
    # ---------------------------------------------------------
    print("\n--- COPY THIS LIST FOR YOUR MAPPING SCRIPT ---")
    print("ALL_INTENTS = [")
    for intent in sorted_intents:
        print(f"    '{intent}',")
    print("]")

    # ---------------------------------------------------------
    # IMBALANCE WARNING
    # ---------------------------------------------------------
    print("\n--- IMBALANCE ANALYSIS ---")
    most_common = intent_counts.most_common(5)
    least_common = intent_counts.most_common()[:-6:-1]
    
    print("Top 5 Frequent Classes:")
    for k, v in most_common:
        print(f"  {k}: {v}")
        
    print("\nBottom 5 Rare Classes (Hard for Model):")
    for k, v in least_common:
        print(f"  {k}: {v}")

if __name__ == "__main__":
    inspect_dataset()