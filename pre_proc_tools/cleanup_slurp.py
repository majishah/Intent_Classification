import json
import os

INPUT_FILE = "slurp_simplified.jsonl"
OUTPUT_FILE = "slurp_final_clean.jsonl"


VALID_INTENTS = {
    # ALARM
    'alarm_query', 'alarm_remove', 'alarm_set',
    
    # CALENDAR
    'calendar_query', 'calendar_remove', 'calendar_set',
    
    # EMAIL
    'email_query', 'email_sendemail', 'email_addcontact', 
    
    # GENERAL (Only the clear ones)
    'general_greet', 'general_joke', 
    
    
    # IOT (The main ones)
    'iot_hue_lightchange', 'iot_hue_lightdim', 'iot_hue_lightoff', 
    'iot_hue_lighton', 'iot_hue_lightup', 
    'iot_wemo_off', 'iot_wemo_on',
    'iot_coffee', 'iot_cleaning',
    
    # LISTS
    'lists_query', 'lists_createoradd', 'lists_remove',
    
    # MUSIC / PLAY
    'play_music', 'play_radio', 'play_podcast', 'play_game', 
    'music_query', 'music_likeness',
   
    
    # NEWS
    'news_query',
    
    # QA
    'qa_definition', 'qa_maths', 'qa_stock', 'qa_factoid', 
    
    # SOCIAL
    'social_post', 'social_query',
    
    # TAKEAWAY
    'takeaway_order', 'takeaway_query',
    
    # TRANSPORT
    'transport_taxi', 'transport_traffic', 'transport_ticket', 'transport_query',
    
    # WEATHER
    'weather_query'
}

def clean_dataset():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    count = 0
    skipped = 0
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as fin, open(OUTPUT_FILE, 'w') as fout:
        for line in fin:
            data = json.loads(line)
            if data['intent'] in VALID_INTENTS:
                fout.write(line)
                count += 1
            else:
                skipped += 1
                
    print(f"Created {OUTPUT_FILE}")
    print(f"Kept: {count} samples")
    print(f"Removed: {skipped} samples (Ghost/Garbage classes)")
    print(f"Total Clean Intents: {len(VALID_INTENTS)}")

if __name__ == "__main__":
    clean_dataset()