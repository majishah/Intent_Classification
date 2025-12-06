import json
import torch
import os
from transformers import pipeline
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
MODEL_NAME = "cross-encoder/nli-MiniLM2-L6-H768"
DATA_FILE = "slurp_focused_domain.jsonl" 
SAMPLE_SIZE = None 
BATCH_SIZE = 32
DEVICE = 0 if torch.cuda.is_available() else -1

# ---------------------------------------------------------
# 1. RAW LABELS (Baseline)
# ---------------------------------------------------------
RAW_INTENTS = [
    'alarm_query', 'alarm_remove', 'alarm_set',
    'calendar_query', 'calendar_remove', 'calendar_set',
    'email_query', 'email_sendemail',
    'iot_hue_lightchange', 'iot_hue_lightdim', 'iot_hue_lightoff', 
    'iot_hue_lighton', 'iot_hue_lightup', 'iot_wemo_off', 'iot_wemo_on',
    'iot_coffee', 'iot_cleaning',
    'play_music', 'play_radio', 'play_podcast',
    'music_query', 'music_settings', 'music_likeness',
    'weather_query'
]

# ---------------------------------------------------------
# 2. IMPROVED SEMANTIC MAP (Contrastive Definitions)
# ---------------------------------------------------------
SEMANTIC_MAP = {
    # ALARM (Focus on 'Status' vs 'Cancel' vs 'Create')
    'alarm_query': "Check the status of active alarms",
    'alarm_remove': "Cancel or delete an existing alarm",
    'alarm_set': "Create and schedule a new alarm",

    # CALENDAR (Focus on 'Check' vs 'Delete' vs 'New Event')
    'calendar_query': "Check the calendar for upcoming appointments",
    'calendar_remove': "Remove an appointment from the calendar",
    'calendar_set': "Add a new event to the calendar",

    # EMAIL (Focus on 'Inbox' vs 'Composing')
    'email_query': "Read unread emails from the inbox",
    'email_sendemail': "Compose and send a new email message",

    # IOT - LIGHTS (Focus on the specific action state)
    'iot_hue_lightchange': "Change the color of the smart lights",
    'iot_hue_lightdim': "Decrease the brightness of the smart lights",
    'iot_hue_lightup': "Increase the brightness of the smart lights",
    'iot_hue_lightoff': "Switch off the smart lights",
    'iot_hue_lighton': "Switch on the smart lights",

    # IOT - APPLIANCES
    'iot_wemo_off': "Turn off the smart plug socket",
    'iot_wemo_on': "Turn on the smart plug socket",
    'iot_coffee': "Start the coffee machine",
    'iot_cleaning': "Start the robot vacuum cleaner",

    # MUSIC - PLAYBACK (Action: Start playing)
    'play_music': "Start playing music tracks",
    'play_radio': "Start playing a radio station stream",
    'play_podcast': "Start playing a podcast episode",

    # MUSIC - INFO/SETTINGS (Action: Query/Adjust)
    'music_query': "Identify the name of the song playing now",
    'music_settings': "Adjust audio playback settings",
    'music_likeness': "Save this song to favorites",

    # WEATHER
    'weather_query': "Get the current weather forecast and temperature"
}

REVERSE_MAP = {v: k for k, v in SEMANTIC_MAP.items()}

# ---------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------
def run_experiment():
    print(f"Loading Model: {MODEL_NAME}...")
    try:
        classifier = pipeline("zero-shot-classification", model=MODEL_NAME, device=DEVICE)
    except:
        print("Error loading model.")
        return

    # Load Data
    data = []
    if not os.path.exists(DATA_FILE):
        print(f"Please run the filter script first to create {DATA_FILE}")
        return
        
    with open(DATA_FILE, 'r') as f:
        for line in f:
            data.append(json.loads(line))
            
    if SAMPLE_SIZE: data = data[:SAMPLE_SIZE]
    
    texts = [d['sentence'] for d in data]
    true_labels = [d['intent'] for d in data]
    
    print(f"Running on {len(texts)} samples across {len(RAW_INTENTS)} focused classes.")

    # --- BASELINE ---
    print("\nRunning Baseline (Raw Tags)...")
    base_preds = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch = texts[i:i+BATCH_SIZE]
        res = classifier(batch, RAW_INTENTS)
        if isinstance(res, dict): res = [res]
        base_preds.extend([r['labels'][0] for r in res])

    # --- SEMANTIC ---
    print("\nRunning Semantic Method (Improved Map)...")
    sem_descriptions = list(SEMANTIC_MAP.values())
    sem_preds_mapped = []
    
    # Top-3 Tracking
    top3_hits = 0

    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch = texts[i:i+BATCH_SIZE]
        res = classifier(batch, sem_descriptions)
        if isinstance(res, dict): res = [res]
        
        for j, r in enumerate(res):
            # Top 1 for F1 Score
            best_desc = r['labels'][0]
            sem_preds_mapped.append(REVERSE_MAP[best_desc])
            
            # Top 3 for Accuracy Check (Standard ZSL Metric)
            top3_desc = r['labels'][:3]
            top3_keys = [REVERSE_MAP[d] for d in top3_desc]
            
            # Since we are processing in batches, we need the correct index for true_labels
            global_idx = i + j
            if true_labels[global_idx] in top3_keys:
                top3_hits += 1

    # --- METRICS ---
    f1_base = f1_score(true_labels, base_preds, average='weighted')
    f1_sem = f1_score(true_labels, sem_preds_mapped, average='weighted')
    acc_sem = f1_score(true_labels, sem_preds_mapped, average='micro')
    top3_acc = top3_hits / len(texts)

    print("\n" + "="*40)
    print("FOCUSED DOMAIN RESULTS")
    print("="*40)
    print(f"Baseline F1 (Raw):     {f1_base:.4f}")
    print(f"Semantic F1 (Map):     {f1_sem:.4f}")
    print(f"Semantic Top-1 Acc:    {acc_sem:.2%}")
    print(f"Semantic Top-3 Acc:    {top3_acc:.2%}")
    print("-" * 40)
    
    if f1_sem > f1_base:
        imp = ((f1_sem - f1_base)/f1_base)*100
        print(f"Improvement: +{imp:.2f}%")
    
    print("\nDetailed Report (Semantic):")
    print(classification_report(true_labels, sem_preds_mapped, zero_division=0))

if __name__ == "__main__":
    run_experiment()