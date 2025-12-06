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
DATA_FILE = "clinic_data.json"
SAMPLE_SIZE = None # Use ALL 22,500 samples (or set to 1000 for quick test)
BATCH_SIZE = 64    # Higher batch size for speed
DEVICE = 0 if torch.cuda.is_available() else -1

# ---------------------------------------------------------
# 1. SEMANTIC MAP (150 Classes)
# ---------------------------------------------------------
SEMANTIC_MAP = {
    'accept_reservations': "Do you take reservations?",
    'account_blocked': "My account is blocked",
    'alarm': "Set an alarm",
    'application_status': "Check status of my application",
    'apr': "What is the APR interest rate?",
    'are_you_a_bot': "Are you a robot or AI?",
    'balance': "Check my bank balance",
    'bill_balance': "Check my bill balance",
    'bill_due': "When is the bill due?",
    'book_flight': "Book a flight ticket",
    'book_hotel': "Book a hotel room",
    'calculator': "Open calculator",
    'calendar': "Check my calendar",
    'calendar_update': "Update or clear my calendar",
    'calories': "How many calories are in this?",
    'cancel': "Cancel the current action",
    'cancel_reservation': "Cancel my reservation",
    'car_rental': "Rent a car",
    'card_declined': "My card was declined",
    'carry_on': "What are the carry-on luggage rules?",
    'change_accent': "Change your voice accent",
    'change_ai_name': "Change your name",
    'change_language': "Change the language",
    'change_speed': "Speak faster or slower",
    'change_user_name': "Change my user name",
    'change_volume': "Change the volume",
    'confirm_reservation': "Confirm my reservation details",
    'cook_time': "How long do I cook this?",
    'credit_limit': "What is my credit limit?",
    'credit_limit_change': "Change or increase my credit limit",
    'credit_score': "Check my credit score",
    'current_location': "Where am I right now?",
    'damaged_card': "My card is damaged or broken",
    'date': "What is today's date?",
    'definition': "Define this word",
    'direct_deposit': "Set up direct deposit",
    'directions': "Get directions to a place",
    'distance': "How far away is a place?",
    'do_you_have_pets': "Do you have any pets?",
    'exchange_rate': "Check currency exchange rates",
    'expiration_date': "When does my card expire?",
    'find_phone': "Find my lost phone",
    'flight_status': "Check flight status",
    'flip_coin': "Flip a coin",
    'food_last': "How long does food last in the fridge?",
    'freeze_account': "Freeze my bank account",
    'fun_fact': "Tell me a fun fact",
    'gas': "Find a gas station",
    'gas_type': "What kind of gas does my car need?",
    'goodbye': "Say goodbye",
    'greeting': "Say hello",
    'how_busy': "How busy is the restaurant?",
    'how_old_are_you': "How old are you?",
    'improve_credit_score': "How do I improve my credit score?",
    'income': "What is my income?",
    'ingredient_substitution': "What can I substitute for this ingredient?",
    'ingredients_list': "What are the ingredients in this dish?",
    'insurance': "Check my insurance benefits",
    'insurance_change': "Change my insurance plan",
    'interest_rate': "What is my interest rate?",
    'international_fees': "What are the international transaction fees?",
    'international_visa': "Do I need a travel visa?",
    'jump_start': "How do I jump start a car?",
    'last_maintenance': "When was my last car maintenance?",
    'lost_luggage': "I lost my luggage",
    'make_call': "Make a phone call",
    'maybe': "Maybe or unsure",
    'meal_suggestion': "Suggest a meal for dinner",
    'meaning_of_life': "What is the meaning of life?",
    'measurement_conversion': "Convert measurements",
    'meeting_schedule': "Check my meeting schedule",
    'min_payment': "What is my minimum payment?",
    'mpg': "What is my car's gas mileage (MPG)?",
    'new_card': "Request a new credit card",
    'next_holiday': "When is the next holiday?",
    'next_song': "Play the next song",
    'no': "No or negative response",
    'nutrition_info': "Check nutrition information",
    'oil_change_how': "How do I change my oil?",
    'oil_change_when': "When should I change my oil?",
    'order': "Place a purchase order",
    'order_checks': "Order checkbooks",
    'order_status': "Check order status",
    'pay_bill': "Pay a bill",
    'payday': "When is payday?",
    'pin_change': "Change my PIN number",
    'play_music': "Play music",
    'plug_type': "What power plug type is used there?",
    'pto_balance': "Check my paid time off (PTO) balance",
    'pto_request': "Request vacation or time off",
    'pto_request_status': "Check status of my time off request",
    'pto_used': "Check how much time off I used",
    'recipe': "Find a cooking recipe",
    'redeem_rewards': "Redeem credit card rewards",
    'reminder': "Check my reminders",
    'reminder_update': "Update or set a reminder",
    'repeat': "Can you repeat that?",
    'replacement_card_duration': "How long for a replacement card to arrive?",
    'report_fraud': "Report fraudulent activity",
    'report_lost_card': "Report a lost credit card",
    'reset_settings': "Reset to factory settings",
    'restaurant_reservation': "Make a restaurant reservation",
    'restaurant_reviews': "Check restaurant reviews",
    'restaurant_suggestion': "Suggest a restaurant",
    'rewards_balance': "Check rewards point balance",
    'roll_dice': "Roll a dice",
    'rollover_401k': "How do I rollover my 401k?",
    'routing': "What is the routing number?",
    'schedule_maintenance': "Schedule car maintenance",
    'schedule_meeting': "Schedule a meeting",
    'share_location': "Share my location",
    'shopping_list': "Check my shopping list",
    'shopping_list_update': "Update my shopping list",
    'smart_home': "Control smart home devices",
    'spelling': "How do you spell this word?",
    'spending_history': "Check my spending history",
    'sync_device': "Sync my device",
    'taxes': "Questions about taxes",
    'tell_joke': "Tell me a joke",
    'text': "Send a text message",
    'thank_you': "Say thank you",
    'time': "What time is it?",
    'timer': "Set a timer",
    'timezone': "What time zone is this?",
    'tire_change': "How do I change a tire?",
    'tire_pressure': "Check tire pressure",
    'todo_list': "Check my to-do list",
    'todo_list_update': "Update my to-do list",
    'traffic': "Check traffic conditions",
    'transactions': "Check recent transactions",
    'transfer': "Transfer money",
    'translate': "Translate a phrase",
    'travel_alert': "Are there travel alerts for this place?",
    'travel_notification': "Set a travel notification",
    'travel_suggestion': "Suggest a travel destination",
    'uber': "Call an Uber or taxi",
    'update_playlist': "Update my music playlist",
    'user_name': "What is my user name?",
    'vaccines': "Do I need vaccines for travel?",
    'w2': "Where is my W2 tax form?",
    'weather': "Check the weather",
    'what_are_your_hobbies': "What are your hobbies?",
    'what_can_i_ask_you': "What questions can I ask you?",
    'what_is_your_name': "What is your name?",
    'what_song': "What song is playing?",
    'where_are_you_from': "Where are you from?",
    'whisper_mode': "Turn on whisper mode",
    'who_do_you_work_for': "Who is your boss?",
    'who_made_you': "Who created you?",
    'yes': "Yes or positive response"
}

REVERSE_MAP = {v: k for k, v in SEMANTIC_MAP.items()}
CANDIDATES = list(SEMANTIC_MAP.values())
RAW_INTENTS = list(SEMANTIC_MAP.keys())

# ---------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------
def run_experiment():
    print(f"\n{'='*60}")
    print(f"CLINIC150 EXPERIMENT: 150-CLASS ZERO-SHOT")
    print(f"{'='*60}")
    
    print(f"Loading Model ({MODEL_NAME})...")
    try:
        classifier = pipeline("zero-shot-classification", model=MODEL_NAME, device=DEVICE)
    except Exception as e:
        print(f"Model Error: {e}")
        return

    # Load Data
    with open(DATA_FILE, 'r') as f:
        raw_data = json.load(f)

    # Merge Train + Test + Val (Ignore OOS)
    data_points = []
    target_splits = ['train', 'test', 'val']
    
    for split in target_splits:
        if split in raw_data:
            data_points.extend(raw_data[split])
            
    if SAMPLE_SIZE: 
        data_points = data_points[:SAMPLE_SIZE]
        
    texts = [item[0] for item in data_points]
    true_labels = [item[1] for item in data_points]
    
    print(f"Status: Testing {len(texts)} samples on 150 classes.")
    print("Running Predictions (This may take 5-10 mins due to 150 classes)...")

    # PREDICT
    pred_mapped = []
    top3_hits = 0
    top5_hits = 0 # Added Top-5 because 150 classes is huge

    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[i:i+BATCH_SIZE]
        results = classifier(batch_texts, CANDIDATES)
        if isinstance(results, dict): results = [results]
        
        for j, res in enumerate(results):
            # Top 1
            best_desc = res['labels'][0]
            pred_mapped.append(REVERSE_MAP[best_desc])
            
            # Top K
            top_desc = res['labels'] # All sorted labels
            top3_keys = [REVERSE_MAP[d] for d in top_desc[:3]]
            top5_keys = [REVERSE_MAP[d] for d in top_desc[:5]]
            
            true_label = true_labels[i+j]
            if true_label in top3_keys:
                top3_hits += 1
            if true_label in top5_keys:
                top5_hits += 1

    # METRICS
    f1 = f1_score(true_labels, pred_mapped, average='weighted')
    acc = f1_score(true_labels, pred_mapped, average='micro')
    top3_acc = top3_hits / len(texts)
    top5_acc = top5_hits / len(texts)

    print("\n" + "="*40)
    print("CLINIC RESULTS")
    print("="*40)
    print(f"F1 Score:        {f1:.4f}")
    print(f"Accuracy (Top-1): {acc:.2%}")
    print(f"Accuracy (Top-3): {top3_acc:.2%}")
    print(f"Accuracy (Top-5): {top5_acc:.2%}")
    print("="*40)

if __name__ == "__main__":
    run_experiment()