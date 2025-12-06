import pandas as pd
import os
import csv

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

DATA_FILE = "snips_train.csv" 

def inspect_snips():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return

    print(f"--- INSPECTING {DATA_FILE} ---\n")

    try:
        # 1. READ CSV
        # Try reading with pandas for nice formatting
        df = pd.read_csv(DATA_FILE)
        
        print(f"Columns Found: {list(df.columns)}")
        print(f"Total Rows: {len(df)}")
        
        # 2. AUTO-DETECT COLUMNS
        # SNIPS usually has 'text'/'sentence' and 'label'/'intent'
        text_col = None
        intent_col = None
        
        possible_text = ['text', 'sentence', 'query', 'utterance', 'data']
        possible_intent = ['label', 'intent', 'class', 'category']
        
        for col in df.columns:
            if col.lower() in possible_text: text_col = col
            if col.lower() in possible_intent: intent_col = col
            
        if not text_col or not intent_col:
            # Fallback: Assume Col 0 is text, Col 1 is label (or vice versa)
            print("\nCould not auto-detect columns. Showing first row:")
            print(df.iloc[0])
            print("\nWhich column is the TEXT and which is the INTENT?")
            return

        print(f"Using -> Text: '{text_col}' | Intent: '{intent_col}'")

        # 3. ANALYZE INTENTS
        intent_counts = df[intent_col].value_counts()
        sorted_intents = sorted(intent_counts.index.tolist())
        
        print(f"\nTotal Unique Intents: {len(sorted_intents)}")
        print("-" * 80)
        print(f"{'INTENT LABEL':<40} | {'COUNT':<5} | {'EXAMPLE SENTENCE'}")
        print("-" * 80)
        
        for intent in sorted_intents:
            # Get a sample text for this intent
            sample_text = df[df[intent_col] == intent][text_col].iloc[0]
            count = intent_counts[intent]
            print(f"{intent:<40} | {count:<5} | {sample_text}")
            
        print("-" * 80)
        
        # 4. GENERATE LIST
        print("\n--- COPY THIS LIST FOR YOUR EXPERIMENT ---")
        print("SNIPS_INTENTS = [")
        for intent in sorted_intents:
            print(f"    '{intent}',")
        print("]")
        
    except Exception as e:
        print(f"Error reading CSV: {e}")

if __name__ == "__main__":
    inspect_snips()