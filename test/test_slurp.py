import json
import logging
import torch
from transformers import pipeline
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
# Use a fast, accurate NLI model
MODEL_PATH = "./models/nli-MiniLM2-L6-H768" 
SLURP_FILE = "slurp_data.jsonl"
SAMPLE_SIZE = 500  

# DEVICE CONFIG
device = 0 if torch.cuda.is_available() else -1
print(f"Running on {'GPU' if device == 0 else 'CPU'}")

# ---------------------------------------------------------
# 1. BASELINE LABELS (From your extraction)
# ---------------------------------------------------------
RAW_L1 = ["alarm", "audio", "calendar", "cooking", "datetime", "email", "general", "iot", "lists", "music", "news", "play", "qa", "recommendation", "social", "takeaway", "transport", "weather"]

RAW_L2 = {
    'alarm': ["query", "remove", "set"],
    'audio': ["volume_down", "volume_mute", "volume_other", "volume_up"],
    'calendar': ["query", "remove", "set"],
    'cooking': ["recipe"],
    'datetime': ["convert", "query"],
    'email': ["addcontact", "query", "querycontact", "sendemail"],
    'general': ["greet", "joke", "quirky"],
    'iot': ["cleaning", "coffee", "hue_lightchange", "hue_lightdim", "hue_lightoff", "hue_lighton", "hue_lightup", "wemo_off", "wemo_on"],
    'lists': ["createoradd", "query", "remove"],
    'music': ["dislikeness", "likeness", "query", "settings"],
    'news': ["query"],
    'play': ["audiobook", "game", "music", "podcasts", "radio"],
    'qa': ["currency", "definition", "factoid", "maths", "stock"],
    'recommendation': ["events", "locations", "movies"],
    'social': ["post", "query"],
    'takeaway': ["order", "query"],
    'transport': ["query", "taxi", "ticket", "traffic"],
    'weather': ["query"],
}

# ---------------------------------------------------------
# 2. PHD METHOD: SEMANTIC MAPPING (The Improvement)
# ---------------------------------------------------------

MAP_L1 = {
    "alarm": "Manage Alarms",
    "audio": "Audio Volume Control",
    "calendar": "Calendar and Events",
    "cooking": "Cooking and Recipes",
    "datetime": "Date and Time Queries",
    "email": "Email Communication",
    "general": "General Chit-Chat",
    "iot": "Smart Home Device Control",
    "lists": "Manage Lists",
    "music": "Music Preferences and Info",
    "news": "News and Headlines",
    "play": "Media Playback",
    "qa": "General Knowledge Q&A",
    "recommendation": "Get Recommendations",
    "social": "Social Media Interaction",
    "takeaway": "Order Food Takeaway",
    "transport": "Transportation and Traffic",
    "weather": "Weather Information"
}

MAP_L2 = {
    "iot": {
        "cleaning": "Start Robot Vacuum",
        "coffee": "Make Coffee",
        "hue_lightchange": "Change Light Color",
        "hue_lightdim": "Dim the Lights",
        "hue_lightoff": "Turn Lights Off",
        "hue_lighton": "Turn Lights On",
        "hue_lightup": "Increase Light Brightness",
        "wemo_off": "Turn Off Smart Plug",
        "wemo_on": "Turn On Smart Plug"
    },
    "general": {
        "greet": "Say Hello",
        "joke": "Tell a Joke",
        "quirky": "Ask a Random Question"
    },
    "play": {
        "music": "Play Music",
        "radio": "Play Radio",
        "podcasts": "Play Podcast",
        "audiobook": "Play Audiobook",
        "game": "Play a Game"
    },
}

REVERSE_MAP_L1 = {v: k for k, v in MAP_L1.items()}

# ---------------------------------------------------------
# LOGIC
# ---------------------------------------------------------

def get_mapped_labels(raw_labels, mapping_dict):
    """Converts a list of raw labels to their semantic descriptions."""
    return [mapping_dict.get(lbl, lbl) for lbl in raw_labels]

def predict(classifier, text, candidate_labels):
    """Standard Zero-Shot Prediction."""
    try:
        result = classifier(text, candidate_labels)
        return result['labels'][0]
    except:
        return None

def run_evaluation(test_data, classifier, use_semantic_mapping=False):
    y_true = []
    y_pred = []
    
    print(f"--- Starting Evaluation [{'SEMANTIC' if use_semantic_mapping else 'BASELINE'}] ---")
    
    for item in tqdm(test_data):
        text = item['sentence']
        true_l1_raw = item['scenario']
        
        # 1. SETUP LABELS
        if use_semantic_mapping:
            # Use descriptions
            candidate_labels = list(MAP_L1.values()) 
        else:
            # Use raw words
            candidate_labels = RAW_L1 

        # 2. PREDICT LEVEL 1
        pred_label = predict(classifier, text, candidate_labels)
        
        # 3. NORMALIZE PREDICTION TO RAW FOR COMPARISON
        if use_semantic_mapping:
            # Convert "Smart Home" back to "iot" to compare with truth
            pred_l1_raw = REVERSE_MAP_L1.get(pred_label, "unknown")
        else:
            pred_l1_raw = pred_label

        y_true.append(true_l1_raw)
        y_pred.append(pred_l1_raw)

    return y_true, y_pred

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    # 1. Load Data
    data = []
    with open(SLURP_FILE, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    if SAMPLE_SIZE:
        data = data[:SAMPLE_SIZE]
    
    print(f"Loaded {len(data)} samples.")

    # 2. Load Model
    classifier = pipeline("zero-shot-classification", model=MODEL_PATH, device=device)

    # 3. Run Baseline (Raw Labels)
    true_base, pred_base = run_evaluation(data, classifier, use_semantic_mapping=False)
    f1_base = f1_score(true_base, pred_base, average='weighted')
    
    # 4. Run Semantic (Mapped Labels)
    true_sem, pred_sem = run_evaluation(data, classifier, use_semantic_mapping=True)
    f1_sem = f1_score(true_sem, pred_sem, average='weighted')

    # 5. Print Comparison
    print("\n" + "="*30)
    print("PHD HYPOTHESIS RESULTS")
    print("="*30)
    print(f"Baseline F1 (Raw Tags):     {f1_base:.4f}")
    print(f"Semantic F1 (Descriptions): {f1_sem:.4f}")
    
    improvement = ((f1_sem - f1_base) / f1_base) * 100
    print(f"Improvement:                {improvement:+.2f}%")
    
    print("\nDetailed Report (Semantic):")
    print(classification_report(true_sem, pred_sem))