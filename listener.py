

# ==============================================================================
# 1. IMPORTS & INITIAL SETUP
# ==============================================================================

# --- Standard Library Imports ---
import asyncio
import json
import os
import sys
import threading
import time
import datetime 
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Optional, Set, Literal

# --- Third-Party Imports ---
import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Local Application Imports ---
from src.core import ListenDrive
from logger import setup_logger
from standard_tools import get_standardized_timestamps

# ==============================================================================
# 2. CONFIGURATION & GLOBAL STATE
# ==============================================================================

# --- Service Configuration ---
SERVICE_HOST = "0.0.0.0"
SERVICE_PORT = 8002
SHOW_EXTENDED_LOGGING = True
LISTEN_SERVICE_ID = "LD-US-WEST-01" 
CORS_ORIGINS = ["http://localhost:1420", "https://tauri.localhost", "http://localhost"]

# --- Downstream Microservice URLs ---
TEXT_PROCESSING_SERVICE_URL = "http://localhost:8004/understand_natural_language/"
CORRECTED_TEXT_SERVICE_URL = "http://localhost:8001/corrected_transcribed_text/"
HUD_SERVICE_URL = "http://localhost:8001/push_hud_info"
ENVIRONMENTAL_PERCEPTION_SERVICE_URL = "http://localhost:8004/understand_environmental_event/"

# --- ListenDrive Engine & Asset Paths ---
SHARE_PATH = "/usr/share/models/"

# Configuration for the ListenDrive engine
WHISPER_REALTIME_MODEL_PATH = f"{SHARE_PATH}/whisper/faster_whisper_tiny_en/"
WHISPER_FINAL_MODEL_PATH = "./models/faster-distil-whisper-medium.en/"
SPEAKER_MODEL_PATH = f"{SHARE_PATH}/wespeaker/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin"
DIARIZATION_PIPELINE_PATH = f"{SHARE_PATH}/pyannote/speaker-diarization-3.1/config.yaml"
USER_EMBEDDINGS_DIR = "./listen/speaker_embeddings/"
BEEP_FILE_PATH = "/audio_files/verifier.wav"
SR_INPUT_DEVICE = 9
SR_COMPUTE_DEVICE = "cuda"
SR_COMPUTE_TYPE = "default"
SR_LANGUAGE = "en"
VERIFICATION_THRESHOLD = 0.7

# --- Logging Setup ---
SHOW_EXTENDED_LOGGING = True
logger = setup_logger("Listener Live", log_file="listen/logs/listen_live.log", level="DEBUG" if SHOW_EXTENDED_LOGGING else "INFO")

# --- Standardized Time Format Constant ---
# Added this constant to ensure consistency across the entire file
STANDARD_TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f %z"


# --- Preload User Embeddings ---
PRELOADED_EMBEDDINGS = {}
try:
    hari_embedding_path = os.path.join(USER_EMBEDDINGS_DIR, "hari/hari_embedding.npy")
    amma_embedding_path = os.path.join(USER_EMBEDDINGS_DIR, "amma/amma_embedding.npy")

    if os.path.exists(hari_embedding_path):
        PRELOADED_EMBEDDINGS["Hari"] = {
            "embedding": np.load(hari_embedding_path),
            "hashid": "LTM-iN4fsGv9_gC97K_nSE8uN78D9dA3Iut4EzY"
        }
    else:
        logger.warning(f"Hari embedding file not found: {hari_embedding_path}")

    if os.path.exists(amma_embedding_path):
        PRELOADED_EMBEDDINGS["Amma"] = {
            "embedding": np.load(amma_embedding_path),
            "hashid": "hashid_amma"
        }
    else:
        logger.warning(f"Amma embedding file not found: {amma_embedding_path}")
    
    if PRELOADED_EMBEDDINGS:
        logger.info(f"Successfully preloaded {len(PRELOADED_EMBEDDINGS)} user embeddings.")
    else:
        logger.warning("No user embeddings were preloaded.")
except Exception as e:
    logger.error(f"Error preloading embeddings: {e}", exc_info=True)
    PRELOADED_EMBEDDINGS = {}

# --- Global State Management ---
listen_drive_instance: Optional[ListenDrive] = None
connected_amplitude_clients: Set[WebSocket] = set()
connected_realtime_text_clients: Set[WebSocket] = set()
connected_final_text_clients: Set[WebSocket] = set()

amplitude_q: Optional[asyncio.Queue] = None
realtime_text_q: Optional[asyncio.Queue] = None
final_text_q: Optional[asyncio.Queue] = None
processing_q: Optional[asyncio.Queue] = None 
background_tasks: Dict[str, asyncio.Task] = {}

# ==============================================================================
# 3. UTILITY FUNCTIONS (NETWORKING & BROADCASTING)
# ==============================================================================

async def broadcast_text_to_clients(clients: Set[WebSocket], message: str, client_type: str):
    if not clients:
        return
    disconnected_clients = set()
    for client in list(clients):
        try:
            await client.send_text(message)
        except (WebSocketDisconnect, RuntimeError) as e:
            logger.info(f"{client_type} client {client.client} disconnected: {type(e).__name__}")
            disconnected_clients.add(client)
        except Exception as e:
            logger.warning(f"Unexpected error sending to {client_type} client {client.client}: {e}")
            disconnected_clients.add(client)
    for client in disconnected_clients:
        clients.discard(client)

async def broadcast_json_to_clients(clients: Set[WebSocket], data: Dict, client_type: str):
    if not clients:
        return
    try:
        message_str = json.dumps(data)
    except TypeError as e:
        logger.error(f"Could not serialize data to JSON for {client_type}: {data}. Error: {e}")
        return
    await broadcast_text_to_clients(clients, message_str, client_type)

async def send_payload_to_service(url: str, payload: Dict, service_name: str, timeout: float = 10.0):
    if not payload:
        logger.debug(f"Skipping send to {service_name}: empty payload.")
        return

    max_retries = 2
    retry_delay = 1.0

    for attempt in range(max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                logger.debug(f"Payload {json.dumps(payload, indent=4)}")
                logger.debug(f"Sending payload to {service_name} at {url} (Attempt {attempt + 1})")
                response = await client.post(url, json=payload)
                response.raise_for_status() 
                logger.info(f"Successfully sent payload to {service_name}: Status {response.status_code}")
                return 
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from {service_name}: {e.response.status_code} - {e.response.text}")
            if 400 <= e.response.status_code < 500:
                break
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.NetworkError) as e:
            logger.warning(f"Network error sending to {service_name}: {type(e).__name__}")
        except Exception as e:
            logger.error(f"Unexpected error sending to {service_name}: {e}", exc_info=True)
            break
        if attempt < max_retries:
            logger.info(f"Retrying send to {service_name} in {retry_delay:.1f}s...")
            await asyncio.sleep(retry_delay)
            retry_delay *= 2 
        else:
            logger.error(f"Failed to send to {service_name} after {max_retries + 1} attempts.")

# ==============================================================================
# 4. LISTENDRIVE CALLBACKS & DATA HANDLING
# ==============================================================================

def on_amplitude_update_callback(rms_value: float):
    if amplitude_q and not amplitude_q.full():
        try:
            formatted_amplitude = f"{rms_value:.4f}"
            amplitude_q.put_nowait(formatted_amplitude)
        except asyncio.QueueFull:
            logger.warning("Amplitude queue was full, dropping value.")
        except Exception as e:
            logger.error(f"Error queueing amplitude: {e}")

def on_realtime_transcription_update_callback(text: str):
    if realtime_text_q and text and not realtime_text_q.full():
        try:
            realtime_text_q.put_nowait(text)
        except asyncio.QueueFull:
            logger.warning("Real-time text queue was full, dropping update.")
        except Exception as e:
            logger.error(f"Error queueing real-time text: {e}")

def on_segment_processed_for_hud(segment_info: Dict):
    if processing_q and not processing_q.full():
        logger.debug(f"Queueing HUD update with segment info: {segment_info}")
        payload_for_queue = {"type": "hud_update", "data": segment_info}
        try:
            processing_q.put_nowait(payload_for_queue)
        except asyncio.QueueFull:
            logger.warning("Processing queue (for HUD) was full, dropping update.")
        except Exception as e:
            logger.error(f"Error putting HUD update to processing_q: {e}")


def on_wakeword_detected_callback():
    logger.info(">>>> WAKE WORD DETECTED! <<<< System is now actively listening for a command.")


def _handle_verified_transcription(segment_uuid: str, text: str, info: Dict):
    """
    Handles a successful, verified transcription. Queues data for the UI and the Mind API.
    """
    logger.debug(f"Handling verified text for '{info.get('speaker_name')}' (ID: {segment_uuid}): '{text}'")

    # 1. Queue a payload for the main UI (WebSocket)
    ui_payload = {
        "id": segment_uuid,
        "text": text,
        "speaker": info.get("speaker_name"),
        "hashid": info.get("speaker_hashid")
    }
    if final_text_q and not final_text_q.full():
        final_text_q.put_nowait(ui_payload)

    # 2. Get Standardized Timestamps
    timestamps = get_standardized_timestamps()

    # 3. Construct and queue the detailed payload for the Mind API.
    prefixed_uuid = f"listen-{segment_uuid}" 

    # Construct the specific payload for an audio source
    audio_payload = {
        "source_type": "audio", 
        "transcription_id": prefixed_uuid,
        "service_id": LISTEN_SERVICE_ID,
        "raw_text": text,
        "user_id": info.get("speaker_hashid"),
        "speaker_name": info.get("speaker_name"),
        
        # --- UPDATED TIMESTAMPS ---
        "timestamp_utc": timestamps['utc'],
        "timestamp_local": timestamps['local'],
        # --------------------------

        "source_details": "microphone_verified", 
        "audio_file": info.get("saved_filename", "not_saved"),
        "verification_distance": info.get("distance"),
        "overlapped_speech": info.get("overlap", False),
        "speaker_count": info.get("speaker_count", 0),
        "hot_word_detected": info.get("hot_word_detected", False) 
    }

    # Wrap it in the universal envelope
    universal_payload = {
        "payload": audio_payload
    }

    logger.info(f"Sending payload to mind: \n {json.dumps(universal_payload, indent=4)}")

    if processing_q and not processing_q.full():
        processing_q.put_nowait({"type": "mind_initial_process", "data": universal_payload})

def _handle_overlapped_speech(segment_uuid: str, info: Dict):
    """
    Handles a segment where overlapped speech was detected. Queues data for the UI
    and the Environmental Perception Service.
    """
    logger.info(f"Segment ID {segment_uuid} had overlap; dispatching to perception service.")

    # 1. Queue a status update for the UI (WebSocket).
    identified_names = [u['name'] for u in info.get("identified_users", [])]
    speaker_label = f"Overlap ({', '.join(identified_names)})" if identified_names else "Overlap"

    ui_payload = {
        "id": segment_uuid, 
        "text": "", 
        "speaker": speaker_label, 
        "hashid": None, 
        "status": "overlap_detected"
    }
    if final_text_q and not final_text_q.full():
        final_text_q.put_nowait(ui_payload)

    # 2. Get Standardized Timestamps
    timestamps = get_standardized_timestamps()

    # 3. Construct and queue a payload for the Environmental Perception Service.
    prefixed_uuid = f"listen-{segment_uuid}"

    environmental_payload = {
        "event_id": prefixed_uuid,
        "service_id": LISTEN_SERVICE_ID,
        "event_type": "overlapped_speech",
        "speaker_count": info.get("speaker_count", 0),
        "identified_users": info.get("identified_users", []), 
        
        # --- UPDATED TIMESTAMPS ---
        "timestamp_utc": timestamps['utc'],
        "timestamp_local": timestamps['local'],
        # --------------------------

        "source": "microphone_verified",
        "audio_file": info.get("saved_filename", "not_saved"),
    }
    
    if processing_q and not processing_q.full():
        processing_q.put_nowait({"type": "environmental_event", "data": environmental_payload})


def _handle_unverified_speaker(segment_uuid: str, info: Dict):
    """
    Handles a segment where a speaker was detected but could not be verified.
    Dispatches an 'unidentified_speaker' event to the Perception Service.
    """
    logger.info(f"Speaker not verified for segment ID {segment_uuid}; dispatching to perception.")

    # 1. Queue a status for the UI (WebSocket)
    ui_payload = {
        "id": segment_uuid, 
        "text": "", 
        "speaker": "Unknown", 
        "hashid": None, 
        "status": "speaker_unverified"
    }
    if final_text_q and not final_text_q.full():
        final_text_q.put_nowait(ui_payload)

    # 2. Get Standardized Timestamps
    timestamps = get_standardized_timestamps()

    # 3. Construct payload for Environmental Perception Service
    prefixed_uuid = f"listen-{segment_uuid}"
    closest_distance = info.get("distance") 
    
    environmental_payload = {
        "event_id": prefixed_uuid,
        "service_id": LISTEN_SERVICE_ID,
        "event_type": "unidentified_speaker",
        "speaker_count": info.get("speaker_count", 1),
        "identified_users": [], 
        
        # --- UPDATED TIMESTAMPS ---
        "timestamp_utc": timestamps['utc'],
        "timestamp_local": timestamps['local'],
        # --------------------------

        "source": "microphone_unverified",
        "audio_file": info.get("saved_filename", "not_saved"),
        "biometric_metadata": {
            "closest_match_distance": closest_distance,
            "threshold_used": VERIFICATION_THRESHOLD
        }
    }

    if processing_q and not processing_q.full():
        processing_q.put_nowait({"type": "environmental_event", "data": environmental_payload})

def on_final_transcription_callback(transcription_text: str, segment_info: Dict):
    global final_text_q, processing_q, listen_drive_instance

    segment_uuid = str(uuid.uuid4())
    logger.info(f"Processing final segment (UUID: {segment_uuid})")

    cleaned_text = transcription_text.strip() if transcription_text else ""
    user_name = segment_info.get("speaker_name")
    has_overlap = segment_info.get("overlap", False)
    
    # --- NEW: Check Quality Gate ---
    passed_quality = segment_info.get("passed_quality_check", True)
    snr_value = segment_info.get("snr", 0.0)

    if not passed_quality:
        logger.warning(f"Segment rejected by Quality Gate (SNR: {snr_value:.2f}). Not sending to Mind.")
        
        # Send a specific status to the UI/HUD so the user knows why nothing happened
        ui_payload = {
            "id": segment_uuid,
            "text": "", # No text shown
            "speaker": "Ignored (Noise)",
            "hashid": None,
            "status": "low_quality_audio",
            "snr": snr_value
        }
        if final_text_q and not final_text_q.full():
            final_text_q.put_nowait(ui_payload)
            
        return # STOP HERE
    # -------------------------------

    if cleaned_text and user_name and not has_overlap:
        _handle_verified_transcription(segment_uuid, cleaned_text, segment_info)
    elif has_overlap:
        _handle_overlapped_speech(segment_uuid, segment_info)
    else:
        _handle_unverified_speaker(segment_uuid, segment_info)

# ==============================================================================
# 5. BACKGROUND ASYNC TASKS (CONSUMERS)
# ==============================================================================

async def websocket_consumer_task(q: asyncio.Queue, clients: Set[WebSocket], client_type: str, is_json: bool):
    logger.info(f"Starting WebSocket consumer for '{client_type}' (JSON: {is_json}).")
    while True:
        try:
            data = await q.get()
            if is_json:
                await broadcast_json_to_clients(clients, data, client_type)
            else:
                await broadcast_text_to_clients(clients, str(data), client_type)
            q.task_done()
        except asyncio.CancelledError:
            logger.info(f"WebSocket consumer for '{client_type}' is stopping.")
            break
        except Exception as e:
            logger.error(f"Error in WebSocket consumer for '{client_type}': {e}", exc_info=True)
            await asyncio.sleep(0.1) 

def _build_hud_payload(segment_info: Dict) -> Dict:
    distance = segment_info.get('distance')
    has_overlap = segment_info.get('overlap', False)
    speaker_name = segment_info.get('speaker_name')
    
    confidence_str = "Confidence: N/A"
    if distance is not None:
        confidence = max(0.0, 1.0 - distance)
        confidence_str = f"Confidence: {confidence:.2f}"

    speaker_count_str = "N/A"
    if has_overlap:
        speaker_count_str = f"Multiple ({segment_info.get('speaker_count', 0)})"
    elif speaker_name:
        speaker_count_str = "1"

    # NOTE: keeping time.strftime here is fine for simple HUD clock display
    return {
        "primary_title": "- BIOMETRIC INFO -",
        "right_section_1": {
            "title": "Listened Info",
            "field_1": confidence_str,
            "field_2": f"Speaker Count: {speaker_count_str}",
            "field_3": f"Overlapped Speech: {'Yes' if has_overlap else 'No'}"
        },
        "right_section_2": {
            "title": "Listened Source",
            "field_1": f"Device: Mic Index {SR_INPUT_DEVICE}",
            "field_2": f"Time: {time.strftime('%H:%M:%S')}",
            "field_3": "Location: Desk"
        },
        "left_section_1": {"title": "Vision Info", "field_1": "N/A", "field_2": "N/A", "field_3": "N/A"},
        "left_section_2": {"title": "Vision Source", "field_1": "N/A", "field_2": "N/A", "field_3": "N/A"},
    }

async def external_service_consumer_task():
    logger.info("Starting External Service Dispatcher task.")
    while True:
        try:
            item = await processing_q.get()
            item_type = item.get("type")
            item_data = item.get("data")

            if item_type == "hud_update":
                pass 
            elif item_type == "mind_initial_process":
                logger.debug(f"Processing Mind API initial payload.")
                await send_payload_to_service(TEXT_PROCESSING_SERVICE_URL, item_data, "Mind API (Initial)")

            elif item_type == "environmental_event":
                logger.debug(f"Processing environmental event payload.")
                await send_payload_to_service(ENVIRONMENTAL_PERCEPTION_SERVICE_URL, item_data, "Environmental Perception Service")

            elif item_type == "mind_corrected_text":
                logger.debug(f"Processing corrected text payload for Mind API.")
                await send_payload_to_service(CORRECTED_TEXT_SERVICE_URL, item_data, "Mind API (Correction)")

            else:
                logger.warning(f"Unknown item type in processing queue: {item_type}")

            processing_q.task_done()
        except asyncio.CancelledError:
            logger.info("External Service Dispatcher is stopping.")
            break
        except Exception as e:
            logger.error(f"Error in External Service Dispatcher: {e}", exc_info=True)
            await asyncio.sleep(0.5)

# ==============================================================================
# 6. FASTAPI LIFESPAN MANAGER
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global listen_drive_instance, amplitude_q, realtime_text_q, final_text_q, processing_q, background_tasks
    
    logger.info("Listen Service: Lifespan startup sequence initiated...")

    amplitude_q = asyncio.Queue(maxsize=300)
    realtime_text_q = asyncio.Queue(maxsize=100)
    final_text_q = asyncio.Queue(maxsize=50)
    processing_q = asyncio.Queue(maxsize=50)

    recorder_config = {
        'model': WHISPER_FINAL_MODEL_PATH,
        'realtime_model_type': WHISPER_REALTIME_MODEL_PATH,
        'use_main_model_for_realtime': False,
        'language': SR_LANGUAGE,
        'compute_type': SR_COMPUTE_TYPE,
        'device': SR_COMPUTE_DEVICE,
        'input_device_index': SR_INPUT_DEVICE,
        'wakeword_backend': 'openwakeword',
        'openwakeword_model_paths': '/usr/share/listen/wake_word/assistant.onnx',
        'openwakeword_inference_framework': 'onnx',
        'wake_words_sensitivity': 0.4,
        'on_wakeword_detected': on_wakeword_detected_callback,
        'on_amplitude_update': on_amplitude_update_callback,
        'on_realtime_transcription_update': on_realtime_transcription_update_callback,
        'on_segment_processed_callback': on_segment_processed_for_hud,
        'on_final_transcription_for_service_callback': on_final_transcription_callback,
        'enable_realtime_transcription': True,
        'start_callback_in_new_thread': True,
        'spinner': False,
        'no_log_file': True,
        'post_speech_silence_duration': 0.8,
        'faster_whisper_vad_filter': True,
        'enable_speaker_verification': True,
        'speaker_model_path': SPEAKER_MODEL_PATH,
        'diarization_pipeline_path': DIARIZATION_PIPELINE_PATH,
        'preloaded_user_embeddings': PRELOADED_EMBEDDINGS,
        'user_embeddings_dir': None,
        'speaker_verification_threshold': VERIFICATION_THRESHOLD,
        'save_verified_segments': True,
        'play_verification_beep': True,
        'beep_file_path': BEEP_FILE_PATH,
        'min_snr_threshold': 0.5 
    }

    try:
        listen_drive_instance = ListenDrive(**recorder_config)
        logger.info("ListenDrive engine initialized successfully.")
    except Exception as e:
        logger.critical(f"CRITICAL: Failed to initialize ListenDrive engine: {e}", exc_info=True)
        raise RuntimeError("Engine initialization failed, stopping service.") from e

    background_tasks['amplitude'] = asyncio.create_task(
        websocket_consumer_task(amplitude_q, connected_amplitude_clients, "Amplitude", is_json=False)
    )
    background_tasks['realtime_text'] = asyncio.create_task(
        websocket_consumer_task(realtime_text_q, connected_realtime_text_clients, "Real-time Text", is_json=False)
    )
    background_tasks['final_text'] = asyncio.create_task(
        websocket_consumer_task(final_text_q, connected_final_text_clients, "Final Text", is_json=True)
    )
    background_tasks['external_services'] = asyncio.create_task(external_service_consumer_task())

    def run_engine_loop():
        logger.info("Starting ListenDrive main processing loop in a background thread...")
        while listen_drive_instance and listen_drive_instance.is_running:
            try:
                listen_drive_instance.text()
            except Exception as e:
                if not listen_drive_instance.is_shut_down:
                    logger.error(f"Error in ListenDrive main loop: {e}", exc_info=True)
        logger.info("ListenDrive main processing loop has finished.")

    loop_thread = threading.Thread(target=run_engine_loop, daemon=True)
    loop_thread.start()
    app.state.engine_loop_thread = loop_thread

    logger.info("Listen Service: Lifespan startup complete. Service is now running.")
    yield 

    logger.info("Listen Service: Lifespan shutdown sequence initiated...")
    
    if listen_drive_instance:
        logger.info("Requesting ListenDrive engine shutdown...")
        listen_drive_instance.shutdown()
        logger.info("ListenDrive engine shutdown complete.")

    if hasattr(app.state, 'engine_loop_thread') and app.state.engine_loop_thread.is_alive():
        logger.info("Waiting for engine's blocking loop thread to join...")
        app.state.engine_loop_thread.join(timeout=10.0)
        if app.state.engine_loop_thread.is_alive():
            logger.warning("Engine's blocking loop thread did not join in time.")

    logger.info("Cancelling all background asyncio tasks...")
    for task in background_tasks.values():
        task.cancel()
    await asyncio.gather(*background_tasks.values(), return_exceptions=True)
    logger.info("All background tasks cancelled.")

    all_clients = connected_amplitude_clients.union(connected_realtime_text_clients, connected_final_text_clients)
    logger.info(f"Closing {len(all_clients)} remaining WebSocket connections...")
    await asyncio.gather(*(client.close(code=1001) for client in all_clients), return_exceptions=True)

    listen_drive_instance = None
    logger.info("Listen Service: Lifespan shutdown complete.")

# ==============================================================================
# 7. FASTAPI APPLICATION & ENDPOINTS
# ==============================================================================

app = FastAPI(
    title="Listen Drive Service",
    description="Provides real-time transcription, speaker analysis, and voice processing.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ControlPayload(BaseModel):
    state: str

class CorrectedTextPayload(BaseModel):
    id: str
    corrected_text: str
    speaker_hashid: str

@app.get("/status", tags=["Service Status"])
async def get_service_status():
    if not listen_drive_instance:
        return {"status": "initializing", "message": "Engine not yet active.", "listening": False}
    if listen_drive_instance.is_shut_down:
        return {"status": "unavailable", "message": "Engine is shut down.", "listening": False}
    is_mic_processing = listen_drive_instance.use_microphone.value
    return {"status": "active", "listening": is_mic_processing}

@app.post("/control", tags=["Service Control"])
async def post_control(payload: ControlPayload):
    global listen_drive_instance
    if not listen_drive_instance or listen_drive_instance.is_shut_down:
        raise HTTPException(status_code=503, detail="ListenDrive engine not active.")

    action = payload.state.lower()
    if action not in ["pause", "resume"]:
        raise HTTPException(status_code=400, detail="Invalid state. Use 'pause' or 'resume'.")

    is_currently_active = listen_drive_instance.use_microphone.value
    
    if action == "pause" and is_currently_active:
        listen_drive_instance.set_microphone(False)
        message = "Microphone input processing paused."
    elif action == "resume" and not is_currently_active:
        listen_drive_instance.set_microphone(True)
        message = "Microphone input processing resumed."
    else:
        message = f"Microphone was already in the '{action}d' state."
    
    logger.info(f"Control request processed: {message}")
    return {"status": "success", "message": message, "listening": listen_drive_instance.use_microphone.value}

@app.post("/submit_corrected_text", tags=["Data Submission"])
async def submit_corrected_text(payload: CorrectedTextPayload):
    if not payload.id or payload.corrected_text is None or not payload.speaker_hashid:
        raise HTTPException(status_code=400, detail="Payload requires 'id', 'corrected_text', and 'speaker_hashid'.")
    
    logger.info(f"Received correction for ID '{payload.id}': '{payload.corrected_text}'")
    item = {"type": "mind_corrected_text", "data": payload.model_dump()}
    
    if processing_q and not processing_q.full():
        await processing_q.put(item)
        return {"status": "success", "message": "Corrected text queued for processing."}
    else:
        logger.warning("Processing queue full, cannot accept corrected text.")
        raise HTTPException(status_code=503, detail="Service busy, please try again later.")

class SetModePayload(BaseModel):
    mode: Literal["listening", "wakeword"]

class ReleaseModePayload(BaseModel):
    pass

@app.post("/control/set_mode", tags=["Service Control"])
async def set_engine_mode(payload: SetModePayload):
    if not listen_drive_instance or listen_drive_instance.is_shut_down:
        raise HTTPException(status_code=503, detail="ListenDrive engine not active.")

    if payload.mode == "listening":
        listen_drive_instance.force_listening_mode()
    elif payload.mode == "wakeword":
        listen_drive_instance.force_wakeword_mode()
    
    return {"status": "success", "message": f"Engine mode override set to '{payload.mode}'."}

@app.post("/control/release_mode", tags=["Service Control"])
async def release_engine_mode_override(payload: ReleaseModePayload):
    if not listen_drive_instance or listen_drive_instance.is_shut_down:
        raise HTTPException(status_code=503, detail="ListenDrive engine not active.")

    listen_drive_instance.release_mode_override()
    return {"status": "success", "message": "Engine mode override released. Returning to automatic context."}

async def websocket_endpoint_manager(websocket: WebSocket, clients: Set[WebSocket], name: str):
    await websocket.accept()
    clients.add(websocket)
    client_id = f"{websocket.client.host}:{websocket.client.port}"
    logger.info(f"{name} client connected: {client_id}. Total: {len(clients)}")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        logger.info(f"{name} client {client_id} disconnected.")
    finally:
        clients.discard(websocket)
        logger.info(f"{name} client {client_id} removed. Total: {len(clients)}")

@app.websocket("/ws/amplitude")
async def ws_amplitude(websocket: WebSocket):
    await websocket_endpoint_manager(websocket, connected_amplitude_clients, "Amplitude")

@app.websocket("/ws/realtime_text")
async def ws_realtime_text(websocket: WebSocket):
    await websocket_endpoint_manager(websocket, connected_realtime_text_clients, "Real-time Text")

@app.websocket("/ws/final_text")
async def ws_final_text(websocket: WebSocket):
    await websocket_endpoint_manager(websocket, connected_final_text_clients, "Final Text")

if __name__ == "__main__":
    logger.info(f"Starting Listen Service on http://{SERVICE_HOST}:{SERVICE_PORT}")
    uvicorn.run(
        "__main__:app",
        host=SERVICE_HOST,
        port=SERVICE_PORT,
        log_level="info"
    )