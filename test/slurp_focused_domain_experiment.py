#  68 % Accuracy
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
# 2. SEMANTIC MAP: "Natural Language" Style
# ---------------------------------------------------------
SEMANTIC_MAP = {
    # ALARM 
    'alarm_query': "Check my alarms",
    'alarm_remove': "Delete an alarm",
    'alarm_set': "Set an alarm",

    # CALENDAR
    'calendar_query': "Check my calendar appointments",
    'calendar_remove': "Cancel a meeting",
    'calendar_set': "Schedule a meeting",

    # EMAIL
    'email_query': "Check my emails",
    'email_sendemail': "Send an email",

    # IOT - LIGHTS 
    'iot_hue_lightchange': "Change the light color",
    'iot_hue_lightdim': "Dim the lights",
    'iot_hue_lightup': "Brighten the lights",
    'iot_hue_lightoff': "Turn off the lights",
    'iot_hue_lighton': "Turn on the lights",

    # IOT - APPLIANCES
    'iot_wemo_off': "Turn off the smart plug",
    'iot_wemo_on': "Turn on the smart plug",
    'iot_coffee': "Make some coffee",
    'iot_cleaning': "Start the vacuum cleaner",

    # MUSIC - PLAYBACK
    'play_music': "Play some music",
    'play_radio': "Play the radio",
    'play_podcast': "Play a podcast",

    # MUSIC - INFO/SETTINGS
    'music_query': "What song is playing?",
    'music_settings': "Change the sound settings",
    'music_likeness': "I like this song",

    # WEATHER
    'weather_query': "Check the weather"
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
    
    print(f"Running on {len(texts)} samples (Natural Language Map)...")

    # --- BASELINE ---
    print("\nRunning Baseline (Raw Tags)...")
    base_preds = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch = texts[i:i+BATCH_SIZE]
        res = classifier(batch, RAW_INTENTS)
        if isinstance(res, dict): res = [res]
        base_preds.extend([r['labels'][0] for r in res])

    # --- SEMANTIC ---
    print("\nRunning Semantic Method (Natural Language Map)...")
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
            
            # Top 3 for Accuracy Check
            top3_desc = r['labels'][:3]
            top3_keys = [REVERSE_MAP[d] for d in top3_desc]
            
            global_idx = i + j
            if true_labels[global_idx] in top3_keys:
                top3_hits += 1

    # --- METRICS ---
    f1_base = f1_score(true_labels, base_preds, average='weighted')
    f1_sem = f1_score(true_labels, sem_preds_mapped, average='weighted')
    acc_sem = f1_score(true_labels, sem_preds_mapped, average='micro')
    top3_acc = top3_hits / len(texts)

    print("\n" + "="*40)
    print("NATURAL LANGUAGE MAP RESULTS")
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

















# import json
# import torch
# import os
# import warnings
# from transformers import pipeline
# from sklearn.metrics import classification_report, f1_score
# from tqdm import tqdm

# warnings.filterwarnings('ignore') 

# # ---------------------------------------------------------
# # CONFIGURATION
# # ---------------------------------------------------------
# MODEL_NAME = "cross-encoder/nli-MiniLM2-L6-H768"
# DATA_FILE = "slurp_expanded.jsonl"
# SAMPLE_SIZE = None
# BATCH_SIZE = 32
# DEVICE = 0 if torch.cuda.is_available() else -1

# RAW_INTENTS = [
#     'alarm_query', 'alarm_remove', 'alarm_set',
#     'calendar_query', 'calendar_remove', 'calendar_set',
#     'email_query', 'email_sendemail',
#     'iot_hue_lightchange', 'iot_hue_lightdim', 'iot_hue_lightoff', 
#     'iot_hue_lighton', 'iot_hue_lightup', 'iot_wemo_off', 'iot_wemo_on',
#     'iot_coffee', 'iot_cleaning',
#     'play_music', 'play_radio', 'play_podcast',
#     'music_query', 'music_settings', 'music_likeness',
#     'weather_query',
#     'news_query', 'general_greet', 'general_joke',
#     'qa_definition', 'qa_maths',
#     'social_post', 'social_query',
#     'takeaway_order', 'takeaway_query',
#     'transport_taxi', 'transport_traffic', 'transport_ticket',
#     'lists_query', 'lists_createoradd'
# ]

# # ---------------------------------------------------------
# # SEMANTIC MAP: "ANCHORED" (Domain + Natural Language)
# # ---------------------------------------------------------
# SEMANTIC_MAP = {
#     # ALARM (Anchored)
#     'alarm_query': "Alarm Clock: Check my alarms",
#     'alarm_remove': "Alarm Clock: Delete an alarm",
#     'alarm_set': "Alarm Clock: Set an alarm",

#     # CALENDAR (Anchored)
#     'calendar_query': "Calendar: Check my appointments",
#     'calendar_remove': "Calendar: Cancel a meeting",
#     'calendar_set': "Calendar: Schedule a meeting",

#     # EMAIL (Anchored)
#     'email_query': "Email: Check my inbox",
#     'email_sendemail': "Email: Send a message",

#     # IOT (Anchored with "Smart Home")
#     'iot_hue_lightchange': "Smart Home: Change the light color",
#     'iot_hue_lightdim': "Smart Home: Dim the lights",
#     'iot_hue_lightup': "Smart Home: Brighten the lights",
#     'iot_hue_lightoff': "Smart Home: Turn off the lights",
#     'iot_hue_lighton': "Smart Home: Turn on the lights",
#     'iot_wemo_off': "Smart Home: Turn off the smart plug",
#     'iot_wemo_on': "Smart Home: Turn on the smart plug",
#     'iot_coffee': "Smart Home: Make some coffee",
#     'iot_cleaning': "Smart Home: Start the vacuum cleaner",

#     # LISTS (Anchored to fix the "Black Hole" issue)
#     'lists_query': "Shopping List: What is on my list?",
#     'lists_createoradd': "Shopping List: Add an item to the list",

#     # MUSIC (Anchored)
#     'play_music': "Music Player: Play some music",
#     'play_radio': "Music Player: Play the radio",
#     'play_podcast': "Music Player: Play a podcast",
#     'music_query': "Music Player: What song is playing?",
#     'music_settings': "Music Player: Change sound settings",
#     'music_likeness': "Music Player: I like this song",

#     # NEWS
#     'news_query': "News: What are the headlines?",

#     # GENERAL
#     'general_greet': "Chat: Say hello",
#     'general_joke': "Chat: Tell me a joke",

#     # QA
#     'qa_definition': "Knowledge: What does this word mean?",
#     'qa_maths': "Knowledge: Solve this math equation",

#     # SOCIAL
#     'social_post': "Social Media: Post a message",
#     'social_query': "Social Media: Check notifications",

#     # TAKEAWAY
#     'takeaway_order': "Food Delivery: Order takeaway food",
#     'takeaway_query': "Food Delivery: Check order status",

#     # TRANSPORT
#     'transport_taxi': "Transport: Call a taxi",
#     'transport_traffic': "Transport: How is the traffic?",
#     'transport_ticket': "Transport: Book a ticket",
    
#     # WEATHER
#     'weather_query': "Weather: Check the forecast"
# }

# REVERSE_MAP = {v: k for k, v in SEMANTIC_MAP.items()}

# def run_experiment():
#     print(f"\n{'='*60}")
#     print(f"PHD EXPERIMENT: ANCHORED SEMANTICS (Soft Hierarchy)")
#     print(f"{'='*60}")
    
#     try:
#         classifier = pipeline("zero-shot-classification", model=MODEL_NAME, device=DEVICE)
#     except:
#         return

#     data = []
#     with open(DATA_FILE, 'r') as f:
#         for line in f:
#             data.append(json.loads(line))
            
#     if SAMPLE_SIZE: data = data[:SAMPLE_SIZE]
    
#     texts = [d['sentence'] for d in data]
#     true_labels = [d['intent'] for d in data]

#     # --- SEMANTIC RUN ---
#     print("Running Anchored Method...")
#     sem_descriptions = list(SEMANTIC_MAP.values())
#     sem_preds_mapped = []
#     top3_hits = 0

#     for i in tqdm(range(0, len(texts), BATCH_SIZE), leave=False):
#         batch = texts[i:i+BATCH_SIZE]
#         res = classifier(batch, sem_descriptions)
#         if isinstance(res, dict): res = [res]
        
#         for j, r in enumerate(res):
#             best_desc = r['labels'][0]
#             sem_preds_mapped.append(REVERSE_MAP[best_desc])
            
#             top3_desc = r['labels'][:3]
#             top3_keys = [REVERSE_MAP[d] for d in top3_desc]
#             if true_labels[i+j] in top3_keys:
#                 top3_hits += 1

#     f1_sem = f1_score(true_labels, sem_preds_mapped, average='weighted')
#     acc_sem = f1_score(true_labels, sem_preds_mapped, average='micro')
#     top3_acc = top3_hits / len(texts)

#     print("\n" + "="*40)
#     print("ANCHORED RESULTS")
#     print("="*40)
#     print(f"Semantic F1:         {f1_sem:.4f}")
#     print(f"Semantic Accuracy:   {acc_sem:.2%}")
#     print(f"Semantic Top-3 Acc:  {top3_acc:.2%}")
#     print("="*40)
    
#     print(classification_report(true_labels, sem_preds_mapped, zero_division=0))

# if __name__ == "__main__":
#     run_experiment()




































# # import json
# # import torch
# # import os
# # import warnings
# # from transformers import pipeline
# # from sklearn.metrics import f1_score, accuracy_score, classification_report
# # from tqdm import tqdm

# # warnings.filterwarnings('ignore')

# # # ---------------------------------------------------------
# # # CONFIGURATION
# # # ---------------------------------------------------------
# # MODEL_NAME = "cross-encoder/nli-MiniLM2-L6-H768"
# # DATA_FILE = "slurp_expanded.jsonl" # Using the 38-class dataset
# # BATCH_SIZE = 1 # Hierarchy is harder to batch efficiently, doing 1-by-1 for safety
# # DEVICE = 0 if torch.cuda.is_available() else -1

# # # ---------------------------------------------------------
# # # 1. LEVEL 1: DOMAIN DEFINITIONS (The Gatekeepers)
# # # ---------------------------------------------------------
# # # We map the prefix (e.g., 'alarm') to a Semantic Description
# # L1_DOMAINS = {
# #     'alarm': "Manage alarms and wake-up calls",
# #     'calendar': "Manage calendar events, meetings, and schedules",
# #     'email': "Manage emails and inbox",
# #     'iot': "Control smart home devices, lights, and appliances",
# #     'lists': "Manage shopping lists and grocery lists", # Specific to avoid confusion!
# #     'music': "Play music and audio",
# #     'news': "Check news and headlines",
# #     'general': "General conversation and jokes",
# #     'qa': "Ask questions, definitions, and math",
# #     'social': "Social media updates and posting",
# #     'takeaway': "Order food delivery",
# #     'transport': "Check traffic, taxis, and tickets",
# #     'weather': "Check the weather forecast"
# # }

# # # ---------------------------------------------------------
# # # 2. LEVEL 2: INTENT DEFINITIONS (The Specialists)
# # # ---------------------------------------------------------
# # # Grouped by Domain
# # L2_INTENTS = {
# #     'alarm': {
# #         'alarm_query': "Check my alarms",
# #         'alarm_remove': "Delete an alarm",
# #         'alarm_set': "Set an alarm"
# #     },
# #     'calendar': {
# #         'calendar_query': "Check my calendar appointments",
# #         'calendar_remove': "Cancel a meeting",
# #         'calendar_set': "Schedule a meeting"
# #     },
# #     'email': {
# #         'email_query': "Check my emails",
# #         'email_sendemail': "Send an email"
# #     },
# #     'iot': {
# #         'iot_hue_lightchange': "Change the light color",
# #         'iot_hue_lightdim': "Dim the lights",
# #         'iot_hue_lightup': "Brighten the lights",
# #         'iot_hue_lightoff': "Turn off the lights",
# #         'iot_hue_lighton': "Turn on the lights",
# #         'iot_wemo_off': "Turn off the smart plug",
# #         'iot_wemo_on': "Turn on the smart plug",
# #         'iot_coffee': "Make some coffee",
# #         'iot_cleaning': "Start the vacuum cleaner"
# #     },
# #     'lists': {
# #         'lists_query': "What is on my shopping list?",
# #         'lists_createoradd': "Add an item to the list"
# #     },
# #     'music': {
# #         'play_music': "Play some music",
# #         'play_radio': "Play the radio",
# #         'play_podcast': "Play a podcast",
# #         'music_query': "What song is playing?",
# #         'music_settings': "Change the sound settings",
# #         'music_likeness': "I like this song"
# #     },
# #     'news': {
# #         'news_query': "What is the news?"
# #     },
# #     'general': {
# #         'general_greet': "Say hello",
# #         'general_joke': "Tell me a joke"
# #     },
# #     'qa': {
# #         'qa_definition': "What does this word mean?",
# #         'qa_maths': "Calculate this math problem"
# #     },
# #     'social': {
# #         'social_post': "Post a message to social media",
# #         'social_query': "Check my social media notifications"
# #     },
# #     'takeaway': {
# #         'takeaway_order': "Order food for delivery",
# #         'takeaway_query': "Check my food delivery status"
# #     },
# #     'transport': {
# #         'transport_taxi': "Call a taxi",
# #         'transport_traffic': "How is the traffic?",
# #         'transport_ticket': "Book a train or bus ticket"
# #     },
# #     'weather': {
# #         'weather_query': "Check the weather"
# #     }
# # }

# # # Helper: Map Intent to Domain for Ground Truth generation
# # INTENT_TO_DOMAIN = {}
# # for domain, intents in L2_INTENTS.items():
# #     for intent_key in intents:
# #         INTENT_TO_DOMAIN[intent_key] = domain

# # # Reverse map for L2 (Description -> Key)
# # REVERSE_L2_MAP = {}
# # for dom in L2_INTENTS:
# #     for k, v in L2_INTENTS[dom].items():
# #         REVERSE_L2_MAP[v] = k

# # # ---------------------------------------------------------
# # # EXECUTION
# # # ---------------------------------------------------------
# # def run_hierarchy():
# #     print(f"\n{'='*60}")
# #     print(f"PHD EXPERIMENT: HIERARCHICAL CASCADING (38 Classes)")
# #     print(f"{'='*60}")
    
# #     try:
# #         classifier = pipeline("zero-shot-classification", model=MODEL_NAME, device=DEVICE)
# #     except:
# #         print("Model Error")
# #         return

# #     # Load Data
# #     data = []
# #     with open(DATA_FILE, 'r') as f:
# #         for line in f:
# #             data.append(json.loads(line))
            
# #     # Prepare Truth Lists
# #     true_domains = []
# #     true_intents = []
# #     texts = []
    
# #     print("Preparing Ground Truths...")
# #     for item in data:
# #         intent = item['intent']
# #         domain = INTENT_TO_DOMAIN.get(intent)
# #         if domain: # Safety check
# #             true_domains.append(domain)
# #             true_intents.append(intent)
# #             texts.append(item['sentence'])
    
# #     print(f"Running Hierarchical Classification on {len(texts)} samples...")
    
# #     pred_domains = []
# #     pred_intents = []
    
# #     # We define the Domain Candidates once
# #     domain_descriptions = list(L1_DOMAINS.values())
# #     # Reverse map for domains (Description -> Key)
# #     REVERSE_L1_MAP = {v: k for k, v in L1_DOMAINS.items()}

# #     # --- THE CASCADE LOOP ---
# #     for text in tqdm(texts):
        
# #         # STEP 1: PREDICT DOMAIN
# #         l1_out = classifier(text, domain_descriptions)
# #         best_domain_desc = l1_out['labels'][0]
# #         pred_domain_key = REVERSE_L1_MAP[best_domain_desc]
# #         pred_domains.append(pred_domain_key)
        
# #         # STEP 2: PREDICT INTENT (Scoped to predicted domain)
# #         # We look up the candidates for this specific domain
# #         possible_intents = L2_INTENTS.get(pred_domain_key)
        
# #         if possible_intents:
# #             # We have sub-intents for this domain
# #             candidates_l2 = list(possible_intents.values())
            
# #             # If there is only 1 intent in this domain (e.g. Weather), we auto-select it
# #             if len(candidates_l2) == 1:
# #                 pred_intent_desc = candidates_l2[0]
# #             else:
# #                 l2_out = classifier(text, candidates_l2)
# #                 pred_intent_desc = l2_out['labels'][0]
            
# #             # Map back to key
# #             pred_intents.append(REVERSE_L2_MAP[pred_intent_desc])
# #         else:
# #             # If prediction failed or no candidates (shouldn't happen with our map)
# #             pred_intents.append("unknown")

# #     # --- METRICS ---
    
# #     # Level 1 Accuracy
# #     l1_acc = accuracy_score(true_domains, pred_domains)
    
# #     # Level 2 Accuracy (The Final Result)
# #     l2_acc = accuracy_score(true_intents, pred_intents)
# #     l2_f1 = f1_score(true_intents, pred_intents, average='weighted')

# #     print("\n" + "="*40)
# #     print("HIERARCHICAL RESULTS")
# #     print("="*40)
# #     print(f"Stage 1 (Domain) Accuracy: {l1_acc:.2%}")
# #     print(f"Stage 2 (Final) Accuracy:  {l2_acc:.2%}")
# #     print(f"Stage 2 Weighted F1:       {l2_f1:.4f}")
# #     print("-" * 40)
# #     print("Comparison to Flat Natural Language:")
# #     print("Previous Flat Accuracy:    52.86%")
    
# #     delta = l2_acc * 100 - 52.86
# #     print(f"IMPROVEMENT:               +{delta:.2f}% (Points)")
# #     print("="*40)

# # if __name__ == "__main__":
# #     run_hierarchy()






































# 50+ Accuracy
# import json
# import torch
# import os
# import warnings
# from transformers import pipeline
# from sklearn.metrics import classification_report, f1_score
# from tqdm import tqdm

# # Suppress sklearn warnings for cleaner output
# warnings.filterwarnings('ignore') 

# # ---------------------------------------------------------
# # CONFIGURATION
# # ---------------------------------------------------------
# MODEL_NAME = "cross-encoder/nli-MiniLM2-L6-H768"
# DATA_FILE = "slurp_expanded.jsonl"
# SAMPLE_SIZE = None 
# BATCH_SIZE = 32
# DEVICE = 0 if torch.cuda.is_available() else -1

# # ---------------------------------------------------------
# # 1. EXPANDED INTENT LIST (38 Classes)
# # ---------------------------------------------------------
# RAW_INTENTS = [
#     'alarm_query', 'alarm_remove', 'alarm_set',
#     'calendar_query', 'calendar_remove', 'calendar_set',
#     'email_query', 'email_sendemail',
#     'iot_hue_lightchange', 'iot_hue_lightdim', 'iot_hue_lightoff', 
#     'iot_hue_lighton', 'iot_hue_lightup', 'iot_wemo_off', 'iot_wemo_on',
#     'iot_coffee', 'iot_cleaning',
#     'play_music', 'play_radio', 'play_podcast',
#     'music_query', 'music_settings', 'music_likeness',
#     'weather_query',
#     # New
#     'news_query', 'general_greet', 'general_joke',
#     'qa_definition', 'qa_maths',
#     'social_post', 'social_query',
#     'takeaway_order', 'takeaway_query',
#     'transport_taxi', 'transport_traffic', 'transport_ticket',
#     'lists_query', 'lists_createoradd'
# ]

# # ---------------------------------------------------------
# # 2. NATURAL LANGUAGE MAP (Expanded)
# # ---------------------------------------------------------
# SEMANTIC_MAP = {
#     # --- ORIGINAL ---
#     'alarm_query': "Check my alarms",
#     'alarm_remove': "Delete an alarm",
#     'alarm_set': "Set an alarm",
#     'calendar_query': "Check my calendar appointments",
#     'calendar_remove': "Cancel a meeting",
#     'calendar_set': "Schedule a meeting",
#     'email_query': "Check my emails",
#     'email_sendemail': "Send an email",
#     'iot_hue_lightchange': "Change the light color",
#     'iot_hue_lightdim': "Dim the lights",
#     'iot_hue_lightup': "Brighten the lights",
#     'iot_hue_lightoff': "Turn off the lights",
#     'iot_hue_lighton': "Turn on the lights",
#     'iot_wemo_off': "Turn off the smart plug",
#     'iot_wemo_on': "Turn on the smart plug",
#     'iot_coffee': "Make some coffee",
#     'iot_cleaning': "Start the vacuum cleaner",
#     'play_music': "Play some music",
#     'play_radio': "Play the radio",
#     'play_podcast': "Play a podcast",
#     'music_query': "What song is playing?",
#     'music_settings': "Change the sound settings",
#     'music_likeness': "I like this song",
#     'weather_query': "Check the weather",
    
#     # --- NEW ADDITIONS (Natural Language Style) ---
#     'news_query': "What is the news?",
#     'general_greet': "Say hello",
#     'general_joke': "Tell me a joke",
#     'qa_definition': "What does this word mean?",
#     'qa_maths': "Calculate this math problem",
#     'social_post': "Post a message to social media",
#     'social_query': "Check my social media notifications",
#     'takeaway_order': "Order food for delivery",
#     'takeaway_query': "Check my food delivery status",
#     'transport_taxi': "Call a taxi",
#     'transport_traffic': "How is the traffic?",
#     'transport_ticket': "Book a train or bus ticket",
#     'lists_query': "What is on my shopping list?",
#     'lists_createoradd': "Add something to my list"
# }

# REVERSE_MAP = {v: k for k, v in SEMANTIC_MAP.items()}

# # ---------------------------------------------------------
# # EXECUTION
# # ---------------------------------------------------------
# def run_experiment():
#     print(f"\n{'='*60}")
#     print(f"PHD EXPERIMENT: EXPANDED DOMAIN (38 Classes)")
#     print(f"{'='*60}")
    
#     # Load Model
#     print(f"Status: Loading Model ({MODEL_NAME})...")
#     try:
#         classifier = pipeline("zero-shot-classification", model=MODEL_NAME, device=DEVICE)
#     except:
#         print("Error: Model load failed.")
#         return

#     # Load Data
#     data = []
#     if not os.path.exists(DATA_FILE):
#         print(f"Error: {DATA_FILE} missing.")
#         return
        
#     with open(DATA_FILE, 'r') as f:
#         for line in f:
#             data.append(json.loads(line))
            
#     if SAMPLE_SIZE: data = data[:SAMPLE_SIZE]
    
#     texts = [d['sentence'] for d in data]
#     true_labels = [d['intent'] for d in data]
    
#     print(f"Status: Loaded {len(texts)} samples.")

#     # --- BASELINE ---
#     print("\n[1/2] Running Baseline (Raw Tags)...")
#     base_preds = []
#     # Disable tqdm for cleaner logs if desired, or keep it
#     for i in tqdm(range(0, len(texts), BATCH_SIZE), leave=False):
#         batch = texts[i:i+BATCH_SIZE]
#         res = classifier(batch, RAW_INTENTS)
#         if isinstance(res, dict): res = [res]
#         base_preds.extend([r['labels'][0] for r in res])

#     # --- SEMANTIC ---
#     print("[2/2] Running Semantic Method (Natural Language Map)...")
#     sem_descriptions = list(SEMANTIC_MAP.values())
#     sem_preds_mapped = []
#     top3_hits = 0

#     for i in tqdm(range(0, len(texts), BATCH_SIZE), leave=False):
#         batch = texts[i:i+BATCH_SIZE]
#         res = classifier(batch, sem_descriptions)
#         if isinstance(res, dict): res = [res]
        
#         for j, r in enumerate(res):
#             # Top 1
#             best_desc = r['labels'][0]
#             sem_preds_mapped.append(REVERSE_MAP[best_desc])
            
#             # Top 3
#             top3_desc = r['labels'][:3]
#             top3_keys = [REVERSE_MAP[d] for d in top3_desc]
#             if true_labels[i+j] in top3_keys:
#                 top3_hits += 1

#     # --- METRICS ---
#     f1_base = f1_score(true_labels, base_preds, average='weighted')
#     f1_sem = f1_score(true_labels, sem_preds_mapped, average='weighted')
#     acc_sem = f1_score(true_labels, sem_preds_mapped, average='micro')
#     top3_acc = top3_hits / len(texts)

#     # --- FINAL CLEAN REPORT ---
#     print("\n" + "="*40)
#     print("RESULTS SUMMARY")
#     print("="*40)
#     print(f"Total Intents:       {len(RAW_INTENTS)}")
#     print(f"Baseline F1:         {f1_base:.4f}")
#     print(f"Semantic F1:         {f1_sem:.4f}")
#     print("-" * 40)
#     print(f"Semantic Accuracy:   {acc_sem:.2%}")
#     print(f"Semantic Top-3 Acc:  {top3_acc:.2%}")
#     print("-" * 40)
    
#     imp = ((f1_sem - f1_base)/f1_base)*100
#     print(f"IMPROVEMENT:         +{imp:.2f}%")
#     print("="*40)
    
#     # Detailed Report
#     print("\nDetailed Semantic Report:")
#     print(classification_report(true_labels, sem_preds_mapped, zero_division=0))

# if __name__ == "__main__":
#     run_experiment()