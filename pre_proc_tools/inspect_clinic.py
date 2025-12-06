import json
import os
from collections import Counter

DATA_FILE = "clinic_data.json"

def inspect_clinic_fixed():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return

    print(f"--- INSPECTING {DATA_FILE} (Fixed List Format) ---\n")

    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    target_splits = ['train', 'test', 'val']
    
    all_samples = []
    for split in target_splits:
        if split in raw_data:
            print(f" -> Merging '{split}' ({len(raw_data[split])} samples)...")
            all_samples.extend(raw_data[split])

    intent_counts = Counter()
    intent_examples = {}

    # PARSE [text, label] FORMAT
    for item in all_samples:
        # Safety check
        if isinstance(item, list) and len(item) == 2:
            text = item[0]
            label = item[1]
            
            intent_counts[label] += 1
            
            # Keep shortest text as example
            if label not in intent_examples:
                intent_examples[label] = text
            else:
                if len(text) < len(intent_examples[label]):
                    intent_examples[label] = text

    # REPORT
    sorted_intents = sorted(intent_counts.keys())
    
    print(f"\nTOTAL IN-DOMAIN SAMPLES: {len(all_samples)}")
    print(f"TOTAL UNIQUE INTENTS: {len(sorted_intents)}")
    print("-" * 80)
    print(f"{'INTENT LABEL':<30} | {'COUNT':<5} | {'EXAMPLE SENTENCE'}")
    print("-" * 80)
    
    for intent in sorted_intents:
        # Truncate intent name if too long for display
        display_intent = (intent[:27] + '..') if len(intent) > 27 else intent
        print(f"{display_intent:<30} | {intent_counts[intent]:<5} | {intent_examples[intent]}")

    print("-" * 80)
    
    # Generate list for next step
    print("\n--- COPY THIS LIST FOR YOUR EXPERIMENT ---")
    print("CLINIC_INTENTS = [")
    for intent in sorted_intents:
        print(f"    '{intent}',")
    print("]")

if __name__ == "__main__":
    inspect_clinic_fixed()