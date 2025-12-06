import json
import os
from collections import defaultdict, Counter

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
# Replace this with the actual path to your SLURP .jsonl file
SLURP_FILE_PATH = "slurp_data.jsonl" 
# If you don't have the file yet, download the 'train' or 'test' set 
# from the official repository or Hugging Face.

def analyze_slurp_structure(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    print(f"--- Analyzing {file_path} ---")

    hierarchy = defaultdict(set)
    l1_counts = Counter()
    l2_counts = Counter()
    
    sample_record = None
    total_lines = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                total_lines += 1
                data = json.loads(line)

                # 1. Capture a sample record to see the raw schema
                if sample_record is None:
                    sample_record = data

                # 2. Extract Structure
                # SLURP usually uses 'scenario' for the high level and 'action' for the specific task
                l1 = data.get('scenario') 
                l2 = data.get('action')

                if l1 and l2:
                    hierarchy[l1].add(l2)
                    l1_counts[l1] += 1
                    l2_counts[f"{l1} -> {l2}"] += 1
    
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # --- OUTPUT SECTION ---

    print(f"\n1. RAW DATA STRUCTURE (Keys in one record):")
    print(json.dumps(sample_record, indent=2))

    print(f"\n2. DATASET STATS:")
    print(f"Total Records: {total_lines}")
    print(f"Unique Scenarios (Level 1): {len(hierarchy)}")
    print(f"Unique Actions (Level 2): {len(l2_counts)}")

    print(f"\n3. GENERATED CONFIGURATION (Copy this to your main script):")
    print("-" * 40)
    
    # Generate LABELS_LEVEL_ONE
    sorted_l1 = sorted(hierarchy.keys())
    print(f"LABELS_LEVEL_ONE = {json.dumps(sorted_l1)}")
    print("")

    # Generate LABELS_LEVEL_TWO
    print("LABELS_LEVEL_TWO = {")
    for l1 in sorted_l1:
        l2_list = sorted(list(hierarchy[l1]))
        print(f"    '{l1}': {json.dumps(l2_list)},")
    print("}")
    
    print("-" * 40)

    print(f"\n4. IMBALANCE CHECK (Top 5 Scenarios):")
    for l1, count in l1_counts.most_common(5):
        print(f"  - {l1}: {count} samples")

if __name__ == "__main__":
    # Create a dummy file if you want to test the script without downloading SLURP yet
    if not os.path.exists(SLURP_FILE_PATH):
        print("SLURP file not found. Creating a dummy sample for demonstration...")
        dummy_data = [
            {"scenario": "music", "action": "play", "sentence": "play some jazz"},
            {"scenario": "music", "action": "volume_up", "sentence": "louder please"},
            {"scenario": "calendar", "action": "set_event", "sentence": "schedule a meeting"},
            {"scenario": "email", "action": "send_email", "sentence": "email mom"},
            {"scenario": "email", "action": "query_contact", "sentence": "what is dad's email"}
        ]
        with open(SLURP_FILE_PATH, 'w') as f:
            for item in dummy_data:
                f.write(json.dumps(item) + "\n")

    analyze_slurp_structure(SLURP_FILE_PATH)