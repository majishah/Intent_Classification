import json
import os
import random
from collections import defaultdict

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
INPUT_FILE = "slurp_simplified.jsonl"
OUTPUT_FILE = "dataset_by_intent.txt"

def export_and_preview():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print(f"Reading {INPUT_FILE}...")
    
    # 1. Group Data by Intent
    intent_map = defaultdict(list)
    total_lines = 0
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            intent = data['intent']
            sentence = data['sentence']
            intent_map[intent].append(sentence)
            total_lines += 1

    # 2. Write FULL Report to Text File
    sorted_intents = sorted(intent_map.keys())
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(f"SLURP DATASET ANALYSIS\n")
        f.write(f"Total Samples: {total_lines}\n")
        f.write(f"Total Intents: {len(sorted_intents)}\n")
        f.write("="*60 + "\n\n")
        
        for intent in sorted_intents:
            sentences = intent_map[intent]
            f.write(f"INTENT: {intent} (Count: {len(sentences)})\n")
            f.write("-" * 30 + "\n")
            for s in sorted(sentences):
                f.write(f"  - {s}\n")
            f.write("\n" + "="*60 + "\n\n")

    print(f"\nSuccess! Full detailed report saved to: {OUTPUT_FILE}")
    
    # 3. Print CONSOLE Summary (For AI Analysis)
    # We print the first 5 and random 5 to get a good spread
    print("\n" + "="*80)
    print(f"DATASET SUMMARY (Copy this output for Analysis)")
    print("="*80)
    
    for intent in sorted_intents:
        sentences = intent_map[intent]
        print(f"\n>>> INTENT: {intent} [{len(sentences)} samples]")
        
        # Get a representative sample
        # We take short ones (likely distinct commands) and random ones
        sorted_s = sorted(sentences, key=len)
        samples = sorted_s[:3] # 3 shortest
        
        remaining = [s for s in sentences if s not in samples]
        if remaining:
            samples.extend(random.sample(remaining, min(4, len(remaining)))) # 4 randoms
            
        for s in samples:
            print(f"  - {s}")

if __name__ == "__main__":
    export_and_preview()