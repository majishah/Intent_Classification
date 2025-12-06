import json
import os

INPUT_FILE = "slurp_simplified.jsonl"
OUTPUT_FILE = "slurp_expanded.jsonl"

# We expand from 24 to 38 intents
TARGET_INTENTS = {
    # --- ORIGINAL 24 ---
    'alarm_query', 'alarm_remove', 'alarm_set',
    'calendar_query', 'calendar_remove', 'calendar_set',
    'email_query', 'email_sendemail',
    'iot_hue_lightchange', 'iot_hue_lightdim', 'iot_hue_lightoff', 
    'iot_hue_lighton', 'iot_hue_lightup', 'iot_wemo_off', 'iot_wemo_on',
    'iot_coffee', 'iot_cleaning',
    'play_music', 'play_radio', 'play_podcast',
    'music_query', 'music_settings', 'music_likeness',
    'weather_query',
    
    # --- NEW ADDITIONS (14 Intents) ---
    'news_query',              # "What's the news?"
    'general_greet',           # "Hello"
    'general_joke',            # "Tell me a joke"
    'qa_definition',           # "What does X mean?"
    'qa_maths',                # "What is 2+2?"
    'social_post',             # "Tweet this"
    'social_query',            # "Check my facebook"
    'takeaway_order',          # "Order pizza"
    'takeaway_query',          # "Where is my food?"
    'transport_taxi',          # "Call a cab"
    'transport_traffic',       # "How is traffic?"
    'transport_ticket',        # "Book a train ticket"
    'lists_query',             # "What's on my shopping list?"
    'lists_createoradd'        # "Add milk to list"
}

def create_expanded_dataset():
    if not os.path.exists(INPUT_FILE):
        print("Error: slurp_simplified.jsonl not found.")
        return

    count = 0
    with open(INPUT_FILE, 'r') as fin, open(OUTPUT_FILE, 'w') as fout:
        for line in fin:
            data = json.loads(line)
            if data['intent'] in TARGET_INTENTS:
                fout.write(line)
                count += 1
                
    print(f"Created {OUTPUT_FILE}")
    print(f"Total Samples: {count}")
    print(f"Total Intents: {len(TARGET_INTENTS)}")

if __name__ == "__main__":
    create_expanded_dataset()




























# import json
# import os

# INPUT_FILE = "slurp_simplified.jsonl"
# OUTPUT_FILE = "slurp_focused_domain.jsonl"

# # We only keep these specific, high-quality intents.
# # This removes the "noise" (orphans and vague classes).
# TARGET_INTENTS = {
#     # 1. ALARM (Clear intent)
#     'alarm_query', 'alarm_remove', 'alarm_set',
    
#     # 2. CALENDAR (Clear structure)
#     'calendar_query', 'calendar_remove', 'calendar_set',
    
#     # 3. EMAIL (Distinct vocabulary)
#     'email_query', 'email_sendemail', 
    
#     # 4. IOT (We keep the main ones, drop the 'hue_' orphans)
#     'iot_hue_lightchange', 'iot_hue_lightdim', 'iot_hue_lightoff', 
#     'iot_hue_lighton', 'iot_hue_lightup', 'iot_wemo_off', 'iot_wemo_on',
#     'iot_coffee', 'iot_cleaning',
    
#     # 5. MUSIC / PLAY (High volume, need good separation)
#     'play_music', 'play_radio', 'play_podcast',
#     'music_query', 'music_settings', 'music_likeness',
    
#     # 6. WEATHER (The gold standard for intent)
#     'weather_query'
# }

# def filter_dataset():
#     if not os.path.exists(INPUT_FILE):
#         print("Input file not found.")
#         return

#     count = 0
#     with open(INPUT_FILE, 'r') as fin, open(OUTPUT_FILE, 'w') as fout:
#         for line in fin:
#             data = json.loads(line)
#             if data['intent'] in TARGET_INTENTS:
#                 fout.write(line)
#                 count += 1
                
#     print(f"Created {OUTPUT_FILE} with {count} samples.")
#     print(f"Focused on {len(TARGET_INTENTS)} Clean Intents.")

# if __name__ == "__main__":
#     filter_dataset()