import json
import torch
import os
from transformers import pipeline
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
MODEL_NAME = "./models/nli-MiniLM2-L6-H768" 
DATA_FILE = "slurp_simplified.jsonl"
SAMPLE_SIZE = 500   
BATCH_SIZE = 16     
DEVICE = 0 if torch.cuda.is_available() else -1

print(f"Running on Device: {'GPU (' + str(DEVICE) + ')' if DEVICE != -1 else 'CPU'}")

# ---------------------------------------------------------
# 2. BASELINE LABELS (Raw Strings from Dataset)
# ---------------------------------------------------------
RAW_INTENTS = [
    'alarm_query', 'alarm_remove', 'alarm_set', 'audio_volume_down', 'audio_volume_mute', 
    'audio_volume_other', 'audio_volume_up', 'calendar_query', 'calendar_remove', 'calendar_set', 
    'cleaning', 'cooking_recipe', 'createoradd', 'datetime_convert', 'datetime_query', 
    'email_addcontact', 'email_query', 'email_querycontact', 'email_sendemail', 'game', 
    'general_greet', 'general_joke', 'general_quirky', 'hue_lightdim', 'hue_lightoff', 
    'hue_lightup', 'iot_cleaning', 'iot_coffee', 'iot_hue_lightchange', 'iot_hue_lightdim', 
    'iot_hue_lightoff', 'iot_hue_lighton', 'iot_hue_lightup', 'iot_wemo_off', 'iot_wemo_on', 
    'joke', 'lists_createoradd', 'lists_query', 'lists_remove', 'locations', 'music', 
    'music_dislikeness', 'music_likeness', 'music_query', 'music_settings', 'news_query', 
    'play_audiobook', 'play_game', 'play_music', 'play_podcasts', 'play_radio', 'podcasts', 
    'post', 'qa_currency', 'qa_definition', 'qa_factoid', 'qa_maths', 'qa_stock', 'query', 
    'quirky', 'radio', 'recommendation_events', 'recommendation_locations', 'recommendation_movies', 
    'remove', 'sendemail', 'set', 'social_post', 'social_query', 'takeaway_order', 
    'takeaway_query', 'transport_query', 'transport_taxi', 'transport_ticket', 'transport_traffic', 
    'weather_query', 'wemo_off'
]

# ---------------------------------------------------------
# 3. SEMANTIC MAP (The "Definition" Standard)
# ---------------------------------------------------------

SEMANTIC_MAP = {
    'alarm_query': "Retrieve information about active alerts",
    'alarm_remove': "Cancel or delete an existing alert",
    'alarm_set': "Configure a new scheduled alert",
    'audio_volume_down': "Decrease the sound intensity",
    'audio_volume_mute': "Silence the audio output completely",
    'audio_volume_other': "Modify auxiliary audio configuration",
    'audio_volume_up': "Increase the sound intensity",
    'calendar_query': "Check the agenda for upcoming events",
    'calendar_remove': "Delete an entry from the agenda",
    'calendar_set': "Add a new event to the schedule",
    'cleaning': "Activate floor cleaning device", 
    'iot_cleaning': "Activate automated vacuum cleaner",
    'cooking_recipe': "Search for culinary instructions",
    'createoradd': "Append item to a list",
    'lists_createoradd': "Append item to a shopping or to-do list",
    'datetime_convert': "Calculate time zone differences",
    'datetime_query': "Request the current time or date",
    'email_addcontact': "Save new entry to address book",
    'email_query': "Check inbox for new correspondence",
    'email_querycontact': "Retrieve details from address book",
    'email_sendemail': "Compose and transmit a digital message",
    'sendemail': "Transmit a digital message",
    'game': "Engage in digital gaming",
    'play_game': "Launch a video game application",
    'general_greet': "Initiate social salutation",
    'general_joke': "Request humorous content",
    'joke': "Request humorous content",
    'general_quirky': "Engage in non-functional banter",
    'quirky': "Non-functional banter",
    'hue_lightdim': "Reduce illumination intensity",
    'hue_lightoff': "Deactivate lighting system",
    'hue_lightup': "Increase illumination intensity",
    'iot_hue_lightchange': "Modify lighting color or state",
    'iot_hue_lightdim': "Reduce illumination brightness",
    'iot_hue_lightoff': "Extinguish the lights",
    'iot_hue_lighton': "Activate the lighting system",
    'iot_hue_lightup': "Increase illumination brightness",
    'iot_coffee': "Activate brewing apparatus",
    'iot_wemo_off': "Deactivate smart socket",
    'iot_wemo_on': "Activate smart socket",
    'wemo_off': "Deactivate smart socket",
    'lists_query': "Read back items from a list",
    'lists_remove': "Delete item from a list",
    'remove': "Delete an item",
    'locations': "Geographic navigation inquiry",
    'music': "General music playback interaction",
    'music_dislikeness': "Register negative preference for track",
    'music_likeness': "Register positive preference for track",
    'music_query': "Identify the currently playing track",
    'music_settings': "Adjust playback configuration",
    'play_music': "Initiate music playback",
    'play_audiobook': "Initiate audiobook playback",
    'play_podcasts': "Initiate podcast playback",
    'play_radio': "Initiate radio stream",
    'podcasts': "Podcast playback",
    'radio': "Radio stream playback",
    'news_query': "Retrieve current headlines",
    'qa_currency': "Perform monetary conversion",
    'qa_definition': "Request lexical definition",
    'qa_factoid': "Request factual information",
    'qa_maths': "Perform numerical calculation",
    'qa_stock': "Request financial market data",
    'query': "General status inquiry",
    'set': "Configure a setting",
    'recommendation_events': "Suggest local social activities",
    'recommendation_locations': "Suggest points of interest",
    'recommendation_movies': "Suggest cinematic content",
    'social_post': "Publish content to social media",
    'social_query': "Retrieve social media updates",
    'post': "Publish content",
    'takeaway_order': "Place request for food delivery",
    'takeaway_query': "Track food delivery status",
    'transport_query': "Inquire about travel logistics",
    'transport_taxi': "Request ride-sharing service",
    'transport_ticket': "Reserve transportation passage",
    'transport_traffic': "Inquire about road congestion",
    'weather_query': "Request meteorological forecast"
}

# Reverse Map: Used to translate the model's prediction (Definition) back to the Label (Key)
REVERSE_MAP = {v: k for k, v in SEMANTIC_MAP.items()}

# ---------------------------------------------------------
# 4. EXECUTION LOGIC
# ---------------------------------------------------------
def run_experiment():
    # --- Load Model ---
    print(f"Loading Model: {MODEL_NAME}...")
    try:
        classifier = pipeline("zero-shot-classification", model=MODEL_NAME, device=DEVICE)
    except Exception as e:
        print(f"Critical Error: Failed to load model. {e}")
        return

    # --- Load Data ---
    print(f"Loading Data: {DATA_FILE}...")
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found. Please run the dataset creation script first.")
        return

    data = []
    with open(DATA_FILE, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    if SAMPLE_SIZE: 
        data = data[:SAMPLE_SIZE]
        print(f"Using a subset of {SAMPLE_SIZE} samples for rapid testing.")
    else:
        print(f"Using FULL dataset ({len(data)} samples).")

    texts = [d['sentence'] for d in data]
    true_labels = [d['intent'] for d in data]

    # =========================================================
    # EXPERIMENT PART 1: BASELINE (Raw Labels)
    # =========================================================
    print(f"\n[{'='*15} BASELINE RUN (Raw Labels) {'='*15}]")
    print("Classifying using raw tags (e.g., 'weather_query', 'iot_wemo_on')...")
    
    baseline_preds = []
    
    # Process in batches for GPU efficiency
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[i:i+BATCH_SIZE]
        # candidate_labels = RAW_INTENTS
        results = classifier(batch_texts, RAW_INTENTS)
        
        # Unpack results (handle single vs list)
        if isinstance(results, dict): results = [results]
        
        # Store just the top prediction label
        baseline_preds.extend([r['labels'][0] for r in results])

    # Calculate Baseline Metrics
    f1_base = f1_score(true_labels, baseline_preds, average='weighted')
    acc_base = f1_score(true_labels, baseline_preds, average='micro') # Micro F1 == Accuracy
    print(f" >> Baseline Weighted F1: {f1_base:.4f}")

    # =========================================================
    # EXPERIMENT PART 2: SEMANTIC (Unbiased Definitions)
    # =========================================================
    print(f"\n[{'='*15} SEMANTIC RUN (Definitions) {'='*15}]")
    print("Classifying using formal definitions (e.g., 'Request meteorological forecast')...")
    
    semantic_preds_raw = [] # Will hold the definition string
    semantic_candidates = list(SEMANTIC_MAP.values()) # The list of definitions
    
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[i:i+BATCH_SIZE]
        results = classifier(batch_texts, semantic_candidates)
        
        if isinstance(results, dict): results = [results]
        
        semantic_preds_raw.extend([r['labels'][0] for r in results])


    semantic_preds_mapped = []
    for description in semantic_preds_raw:
        if description in REVERSE_MAP:
            semantic_preds_mapped.append(REVERSE_MAP[description])
        else:
            # Should not happen, but safe fallback
            semantic_preds_mapped.append("unknown")

    # Calculate Semantic Metrics
    f1_sem = f1_score(true_labels, semantic_preds_mapped, average='weighted')
    acc_sem = f1_score(true_labels, semantic_preds_mapped, average='micro')
    print(f" >> Semantic Weighted F1: {f1_sem:.4f}")

    # =========================================================
    # FINAL REPORT
    # =========================================================
    print("\n" + "="*50)
    print("PHD HYPOTHESIS CONCLUSION")
    print("="*50)
    print(f"Total Samples Tested: {len(texts)}")
    print(f"Classes (Intents):    {len(RAW_INTENTS)}")
    print("-" * 50)
    print(f"1. Baseline Accuracy:   {acc_base:.2%}")
    print(f"   Baseline Weighted F1: {f1_base:.4f}")
    print("-" * 50)
    print(f"2. Semantic Accuracy:   {acc_sem:.2%}")
    print(f"   Semantic Weighted F1: {f1_sem:.4f}")
    print("-" * 50)
    
    delta = f1_sem - f1_base
    improvement_pct = (delta / f1_base) * 100
    
    if delta > 0:
        print(f"RESULT: SUCCESS (+{improvement_pct:.2f}%)")
        print("Using formal definitions improved Zero-Shot comprehension.")
    else:
        print(f"RESULT: NO IMPROVEMENT ({improvement_pct:.2f}%)")
    
    print("\nDetailed Semantic Performance Report:")
    print(classification_report(true_labels, semantic_preds_mapped, zero_division=0))

if __name__ == "__main__":
    run_experiment()