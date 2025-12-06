import json
import os

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
INPUT_FILE = "slurp_data.jsonl"  
OUTPUT_FILE = "slurp_simplified.jsonl" 

def create_simple_dataset():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Could not find {INPUT_FILE}")
        return

    print(f"Reading from {INPUT_FILE}...")
    
    count = 0
    unique_intents = set()

    with open(INPUT_FILE, 'r', encoding='utf-8') as fin, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            data = json.loads(line)
            

            intent_label = data.get("intent")
            sentence = data.get("sentence")
            
            if intent_label and sentence:
                # Create the clean entry
                new_entry = {
                    "sentence": sentence,
                    "intent": intent_label
                }
                
                # Write to new file
                fout.write(json.dumps(new_entry) + "\n")
                
                # Stats
                unique_intents.add(intent_label)
                count += 1

    print(f"\nSuccess! Created {OUTPUT_FILE}")
    print(f"Total Samples: {count}")
    print(f"Total Unique Intents Found: {len(unique_intents)}")
    
    # Preview the data
    print("\n--- Preview of Unique Intents (First 10) ---")
    print(sorted(list(unique_intents))[:10])

if __name__ == "__main__":
    create_simple_dataset()