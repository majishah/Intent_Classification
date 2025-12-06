import pandas as pd
import torch
import os
import warnings
from transformers import pipeline
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
MODEL_NAME = "/home/hari/Desktop/Boss/Intent_Classification/models/nli-MiniLM2-L6-H768/"
DATA_FILE = "snips_train.csv" 
SAMPLE_SIZE = 2000  
BATCH_SIZE = 64
DEVICE = 0 if torch.cuda.is_available() else -1

# ---------------------------------------------------------
# 1. SEMANTIC MAP (Natural Language Definitions)
# ---------------------------------------------------------
SEMANTIC_MAP = {
    # Label: "Add Don and Sherri to my Meditate... playlist"
    'AddToPlaylist': "Add a song or artist to a music playlist",
    
    # Label: "Find a reservation for six..."
    'BookRestaurant': "Book a table at a restaurant",
    
    # Label: "Tell me the weather forecast..."
    'GetWeather': "Check the weather forecast",
    
    # Label: "I wish to listen to some instrumental music"
    'PlayMusic': "Play music",
    
    # Label: "Give 3/6 stars to Doctor in the House"
    'RateBook': "Rate a book or give a review score",
    
    # Label: "Show Force of Nature" (General media search)
    'SearchCreativeWork': "Search for a creative work like a movie, book, or show",
    
    # Label: "Find the schedule for Heart Beats" (Time specific)
    'SearchScreeningEvent': "Find movie screening times or schedules"
}

REVERSE_MAP = {v: k for k, v in SEMANTIC_MAP.items()}
CANDIDATES = list(SEMANTIC_MAP.values())

# ---------------------------------------------------------
# EXECUTION LOGIC
# ---------------------------------------------------------
def run_snips_experiment():
    print(f"\n{'='*60}")
    print(f"SNIPS EXPERIMENT (7 CLASSES)")
    print(f"{'='*60}")
    
    # 1. Load Model
    print(f"Loading Model: {MODEL_NAME}...")
    try:
        classifier = pipeline("zero-shot-classification", model=MODEL_NAME, device=DEVICE)
    except Exception as e:
        print(f"Error: {e}")
        return

    # 2. Load Data
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return
        
    df = pd.read_csv(DATA_FILE)
    
    # Standardize column names based on your inspection
    # Your inspection showed columns: ['text', 'category']
    text_col = 'text'
    label_col = 'category'
    
    # Subset if needed
    if SAMPLE_SIZE and SAMPLE_SIZE < len(df):
        print(f"Sampling {SAMPLE_SIZE} random rows from {len(df)} total...")
        df = df.sample(n=SAMPLE_SIZE, random_state=42)
    else:
        print(f"Using full dataset: {len(df)} rows.")
        
    texts = df[text_col].tolist()
    true_labels = df[label_col].tolist()

    # 3. Run Predictions
    print("Running Predictions...")
    pred_labels = []
    top3_hits = 0 # Less relevant for 7 classes, but good for consistency
    
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[i:i+BATCH_SIZE]
        results = classifier(batch_texts, CANDIDATES)
        if isinstance(results, dict): results = [results]
        
        for j, res in enumerate(results):
            # Top 1
            best_desc = res['labels'][0]
            pred_labels.append(REVERSE_MAP[best_desc])
            
            # Top 3 (Since there are only 7 classes, Top 3 is very broad)
            ranked_desc = res['labels']
            top3_keys = [REVERSE_MAP[d] for d in ranked_desc[:3]]
            
            if true_labels[i+j] in top3_keys:
                top3_hits += 1

    # 4. Metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, 
        pred_labels, 
        average='weighted', 
        zero_division=0
    )
    acc = f1_score(true_labels, pred_labels, average='micro')
    top3_acc = top3_hits / len(texts)

    # 5. Report
    print("\n" + "="*40)
    print("SNIPS RESULTS")
    print("="*40)
    print(f"Accuracy (Exact Match): {acc:.2%}")
    print(f"Accuracy (Top-3):       {top3_acc:.2%}")
    print("-" * 40)
    print(f"Weighted F1 Score:      {f1:.4f}")
    print(f"Weighted Precision:     {precision:.4f}")
    print("="*40)
    
    print("\nDetailed Class Report:")
    print(classification_report(true_labels, pred_labels))

if __name__ == "__main__":
    run_snips_experiment()