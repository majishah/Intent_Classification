import os
import logging
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

# ==============================================================================
# 1. CONFIGURATION & LABELS
# ==============================================================================

# Model Configuration
# If local path doesn't exist, it falls back to HuggingFace Hub to prevent crash
LOCAL_MODEL_PATH = './models/nli-MiniLM2-L6-H768' 
HF_MODEL_ID = "cross-encoder/nli-MiniLM2-L6-H768"
INTENT_MODEL_PATH = LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else HF_MODEL_ID

DEVICE = 0 if torch.cuda.is_available() else -1
DEVICE_NAME = "cuda:0" if torch.cuda.is_available() else "cpu"

# Logging Configuration
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("IntentService")

# --- Intent Hierarchy Definitions ---
LABELS_LEVEL_ONE = ["Conversation Oriented", "Task Oriented", "Entertainment"]

LABELS_LEVEL_TWO = {
    "Conversation Oriented": ["Greetings", "Farewell", "Gratitude", "Assistance", "Well-being", "Self-assessment", "Emotional-support", "Other"],
    "Task Oriented": ["System Control", "Reminder", "Search", "Information", "Navigation", "Communication", "Other"],
    "Entertainment": ["Music", "Movie", "Games", "Other"]
}

LABELS_LEVEL_THREE = {
    "Greetings": ["Formal Greeting", "Informal Greeting", "Small-talk Starter", "Other"],
    "Farewell": ["Polite Goodbye", "Casual Goodbye", "Sign-off", "Other"],
    "Gratitude": ["Expressing Thanks", "Acknowledging Help", "Other"],
    "Assistance": ["Requesting Help", "Offering Help", "Clarification Request", "Other"],
    "Well-being": ["Inquiring Health", "Expressing Concern", "Sharing Status", "Other"],
    "Self-assessment": ["Stating Capability", "Stating Limitation", "Requesting Feedback", "Other"],
    "Emotional-support": ["Offering Comfort", "Expressing Empathy", "Sharing Feelings", "Other"],
    "System Control": ["Device On", "Device Off", "Adjust Setting", "Query Status", "Other"],
    "Reminder": ["Set Reminder", "Query Reminder", "Cancel Reminder", "Other"],
    "Search": ["Web Search", "Fact Search", "Definition Search", "Other"],
    "Information": ["Requesting News", "Requesting Weather", "Requesting Time", "Requesting Facts", "Other"],
    "Navigation": ["Get Directions", "Traffic Info", "Nearby Places", "Other"],
    "Communication": ["Send Message", "Make Call", "Read Message", "Other"],
    "Music": ["Play Song", "Play Artist", "Play Genre", "Control Playback", "Other"],
    "Movie": ["Find Movie", "Movie Info", "Play Trailer", "Other"],
    "Games": ["Start Game", "Game Suggestion", "Game Score", "Other"],
    "Other": ["General Chit-Chat", "Unclassified"]
}

# Dynamic Label Population (Logic from your script)
for l2_cat in LABELS_LEVEL_TWO:
    if l2_cat not in LABELS_LEVEL_THREE:
         LABELS_LEVEL_THREE[l2_cat] = ["Other", "Unclassified"]

# Global Model Variable
intent_classifier = None

# ==============================================================================
# 2. LIFESPAN & LOGIC
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    global intent_classifier
    logger.info(f"Loading intent model from: {INTENT_MODEL_PATH} on {DEVICE_NAME}...")
    try:
        intent_classifier = pipeline(
            "zero-shot-classification",
            model=INTENT_MODEL_PATH,
            device=DEVICE
        )
        logger.info("Intent classifier loaded successfully.")
    except Exception as e:
        logger.critical(f"Failed to load model: {e}")
        raise e
    yield
    # Cleanup if necessary
    intent_classifier = None
    logger.info("Intent classifier unloaded.")

app = FastAPI(title="Intent Recognition Service", lifespan=lifespan)

def perform_hierarchical_prediction(text: str):
    """
    Executes the cascaded intent classification logic.
    """
    pred_l1, pred_l2, pred_l3 = None, None, None
    
    # --- Level 1 Prediction ---
    res_l1 = intent_classifier(text, LABELS_LEVEL_ONE)
    if res_l1 and res_l1['labels']:
        pred_l1 = res_l1['labels'][0]
    
    # --- Level 2 Prediction ---
    if pred_l1 and pred_l1 in LABELS_LEVEL_TWO:
        candidates_l2 = LABELS_LEVEL_TWO[pred_l1]
        res_l2 = intent_classifier(text, candidates_l2)
        if res_l2 and res_l2['labels']:
            pred_l2 = res_l2['labels'][0]
    
    # --- Level 3 Prediction ---
    if pred_l2:
        # Check explicit mapping first
        if pred_l2 in LABELS_LEVEL_THREE:
            candidates_l3 = LABELS_LEVEL_THREE[pred_l2]
        # Fallback logic if L2 exists but not in L3 dict keys explicitly
        else:
            candidates_l3 = ["Other", "Unclassified"]
            
        res_l3 = intent_classifier(text, candidates_l3)
        if res_l3 and res_l3['labels']:
            pred_l3 = res_l3['labels'][0]

    return pred_l1, pred_l2, pred_l3

# ==============================================================================
# 3. API ENDPOINTS
# ==============================================================================

class TextPayload(BaseModel):
    text: str

class IntentResponse(BaseModel):
    text: str
    level_1: str | None
    level_2: str | None
    level_3: str | None

@app.post("/understand_intent", response_model=IntentResponse)
async def understand_intent(payload: TextPayload):
    if not intent_classifier:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    user_text = payload.text.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    logger.info(f"Processing Text: '{user_text}'")

    # Run Inference
    l1, l2, l3 = perform_hierarchical_prediction(user_text)

    # Print result to console/log as requested
    logger.info(f"IDENTIFIED INTENT -> L1: [{l1}] | L2: [{l2}] | L3: [{l3}]")

    return {
        "text": user_text,
        "level_1": l1,
        "level_2": l2,
        "level_3": l3
    }

@app.get("/health")
def health_check():
    return {"status": "active", "model_loaded": intent_classifier is not None}

# ==============================================================================
# 4. ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    import uvicorn
    # Runs on port 8005 to avoid conflict with your other services (8001, 8002, 8004)
    uvicorn.run(app, host="0.0.0.0", port=8005)