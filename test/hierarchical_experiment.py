import json
import torch
import os
import warnings
from transformers import pipeline
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
MODEL_NAME = "cross-encoder/nli-MiniLM2-L6-H768"
DATA_FILE = "slurp_final_clean.jsonl"
DEVICE = 0 if torch.cuda.is_available() else -1
BATCH_SIZE = 32

# ---------------------------------------------------------
# OPTIMIZED SEMANTIC MAP (The Fix)
# ---------------------------------------------------------
SEMANTIC_MAP = {
    'lists_createoradd': "Add item to the grocery or shopping list",
    'lists_query': "Check my shopping list",
    'lists_remove': "Remove item from the shopping list",

    
    'alarm_set': "Set a wake-up alarm or timer", 
    'calendar_set': "Schedule a meeting or calendar event", 
    'alarm_query': "Check my active alarms",
    'alarm_remove': "Delete a wake-up alarm",
    'calendar_query': "Check my calendar schedule",
    'calendar_remove': "Cancel a calendar appointment",

    
    'news_query': "Check current news headlines", 
    'qa_factoid': "Ask a general trivia fact",
    'qa_definition': "Ask for a word definition",
    'qa_maths': "Calculate a numerical equation", 
    'qa_stock': "Check stock market prices",

    
    'email_query': "Check my emails",
    'email_sendemail': "Send an email",
    'email_addcontact': "Add a new contact to address book",
    
    'iot_hue_lightchange': "Change the light color",
    'iot_hue_lightdim': "Dim the lights",
    'iot_hue_lightup': "Brighten the lights",
    'iot_hue_lightoff': "Turn off the lights",
    'iot_hue_lighton': "Turn on the lights",
    'iot_wemo_off': "Turn off the smart plug",
    'iot_wemo_on': "Turn on the smart plug",
    'iot_coffee': "Make some coffee",
    'iot_cleaning': "Start the vacuum cleaner",
    
    'play_music': "Play some music",
    'play_radio': "Play the radio station",
    'play_podcast': "Play a podcast episode",
    'play_game': "Start a video game",
    'music_query': "What song is playing?",
    'music_likeness': "I like this song",
    
    'general_greet': "Say hello",
    'general_joke': "Tell me a joke",
    
    'social_post': "Post a message to social media",
    'social_query': "Check social media notifications",
    
    'takeaway_order': "Order food for delivery",
    'takeaway_query': "Check food delivery status",
    
    'transport_taxi': "Call a taxi or uber",
    'transport_traffic': "Check traffic conditions",
    'transport_ticket': "Book a train or bus ticket",
    'transport_query': "Check travel route info",
    
    'weather_query': "Check the weather forecast"
}

REVERSE_MAP = {v: k for k, v in SEMANTIC_MAP.items()}
CANDIDATES = list(SEMANTIC_MAP.values())
RAW_INTENTS = list(SEMANTIC_MAP.keys())

# ---------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------
def run_experiment():
    print(f"\n{'='*60}")
    print(f"PHD EXPERIMENT: OPTIMIZED FLAT CLASSIFICATION")
    print(f"{'='*60}")
    
    try:
        classifier = pipeline("zero-shot-classification", model=MODEL_NAME, device=DEVICE)
    except:
        return

    data = []
    with open(DATA_FILE, 'r') as f:
        for line in f:
            data.append(json.loads(line))
            
    texts = [d['sentence'] for d in data]
    true_labels = [d['intent'] for d in data]
    
    print(f"Status: Testing {len(texts)} samples on {len(RAW_INTENTS)} classes.")

    # --- PREDICT ---
    print("Running Predictions...")
    pred_mapped = []
    top3_hits = 0

    # Batch processing
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[i:i+BATCH_SIZE]
        results = classifier(batch_texts, CANDIDATES)
        if isinstance(results, dict): results = [results]
        
        for j, res in enumerate(results):
            # Top 1
            best_desc = res['labels'][0]
            pred_mapped.append(REVERSE_MAP[best_desc])
            
            # Top 3
            top3_desc = res['labels'][:3]
            top3_keys = [REVERSE_MAP[d] for d in top3_desc]
            if true_labels[i+j] in top3_keys:
                top3_hits += 1

    # --- METRICS ---
    f1 = f1_score(true_labels, pred_mapped, average='weighted')
    acc = f1_score(true_labels, pred_mapped, average='micro')
    top3_acc = top3_hits / len(texts)

    print("\n" + "="*40)
    print("RESULTS SUMMARY")
    print("="*40)
    print(f"Optimized F1 Score:  {f1:.4f}")
    print(f"Optimized Accuracy:  {acc:.2%}")
    print(f"Top-3 Accuracy:      {top3_acc:.2%}")
    print("="*40)
    
    print("\nDetailed Report:")
    print(classification_report(true_labels, pred_mapped, zero_division=0))

if __name__ == "__main__":
    run_experiment()