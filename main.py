import asyncio
import os
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException

# Import from local modules
import config
from logger_setup import logger
from recognizer import SpeechIntentRecognizer # Import the main class
from streamer import stream_amplitude # Import the background task function
# from utils import ... # utils are used within recognizer

# --- Global State ---
recognizer_instance: SpeechIntentRecognizer | None = None
amplitude_stream_task: asyncio.Task | None = None
ws_amplitude_connections: set[WebSocket] = set()
ws_results_connections: set[WebSocket] = set() # Managed globally here
app_running_state = {"running": True}

# --- FastAPI Lifecycle (MODIFIED to pass global set) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global recognizer_instance, amplitude_stream_task, app_running_state
    global ws_amplitude_connections, ws_results_connections # Make sets global

    logger.info("FastAPI application startup sequence initiated...")
    app_running_state["running"] = True
    ws_amplitude_connections = set() # Reset sets on start
    ws_results_connections = set()   # Reset sets on start

    # Initialize Recognizer
    try:
        logger.info("Initializing Speech Recognizer...")
        # --- PASS the global set to the constructor ---
        recognizer_instance = SpeechIntentRecognizer(ws_results_set_ref=ws_results_connections)
        # ----------------------------------------------
        recognizer_instance.calibrate_audio_model() # Warm-up
        recognizer_instance.start_main_loop() # Starts listening & processing thread
        logger.info("Speech Recognizer started successfully.")
    except Exception as e:
        logger.critical(f"CRITICAL STARTUP FAILURE: Recognizer init failed: {e}", exc_info=True)
        recognizer_instance = None

    # Start Amplitude Streaming Task
    if recognizer_instance:
        logger.info("Starting amplitude streaming task...")
        # Pass state dict and the relevant connections set
        amplitude_stream_task = asyncio.create_task(
            stream_amplitude(app_running_state, ws_amplitude_connections)
        )
    else:
        logger.warning("Skipping amplitude streaming task because recognizer failed to initialize.")

    try:
        yield # Application runs here
    finally:
        # --- Shutdown Logic ---
        logger.info("FastAPI application shutdown sequence initiated...")
        app_running_state["running"] = False # Signal background tasks

        # Stop Recognizer
        if recognizer_instance:
            logger.info("Shutting down recognizer instance...")
            recognizer_instance.stop_listening() # Handles stream stop, thread join, cleanup
            logger.info("Recognizer shutdown process complete.")
        else:
            logger.info("Recognizer was not active, skipping shutdown.")

        # Stop Amplitude Streaming Task
        if amplitude_stream_task and not amplitude_stream_task.done():
            logger.info("Cancelling amplitude streaming task...")
            amplitude_stream_task.cancel()
            try:
                await asyncio.wait_for(amplitude_stream_task, timeout=5.0)
                logger.info("Amplitude streaming task shut down gracefully.")
            except asyncio.CancelledError:
                logger.info("Amplitude streaming task cancellation confirmed.")
            except asyncio.TimeoutError:
                logger.warning("Amplitude streaming task did not finish within timeout during shutdown.")
            except Exception as e:
                logger.error(f"Error during amplitude task shutdown wait: {e}", exc_info=True)
        elif amplitude_stream_task:
             logger.info("Amplitude task was already done.")
        else:
             logger.info("Amplitude task was not started or already finished.")

        # Close remaining WebSocket connections explicitly
        logger.info("Closing any remaining WebSocket connections...")
        close_tasks = []
        # Use list copies for safe iteration while modifying
        for ws in list(ws_amplitude_connections): close_tasks.append(ws.close(code=1001))
        for ws in list(ws_results_connections): close_tasks.append(ws.close(code=1001))
        if close_tasks:
             try:
                  await asyncio.gather(*close_tasks, return_exceptions=True)
                  logger.info("Remaining WebSockets closed.")
             except Exception as ws_close_err:
                  logger.error(f"Error closing WebSockets during shutdown: {ws_close_err}")
        else:
             logger.info("No active WebSockets to close.")

        logger.info("FastAPI application shutdown complete.")


# --- Create FastAPI App ---
app = FastAPI(
    title="Speech API (Modular - User Logic)",
    description="Real-time voice processing.",
    version="2.0.2", # Incremented version
    lifespan=lifespan
)

# --- API Endpoints ---
@app.get("/status", tags=["Recognizer"])
async def get_status():
    if not recognizer_instance: raise HTTPException(status_code=503, detail="Recognizer not initialized")
    processing_alive = recognizer_instance.processing_thread.is_alive() if recognizer_instance.processing_thread else False
    return {"recognizer_status": "listening" if recognizer_instance.listening else "paused",
            "processing_active": processing_alive, "amplitude_clients": len(ws_amplitude_connections),
            "results_clients": len(ws_results_connections), "queue_size": recognizer_instance.data_queue.qsize()}

@app.post("/pause", status_code=200, tags=["Recognizer"])
async def pause_recognizer_endpoint():
    if not recognizer_instance: raise HTTPException(status_code=503, detail="Recognizer not initialized")
    if recognizer_instance.listening: recognizer_instance.pause_listening(); return {"message": "Recognizer paused"}
    else: return {"message": "Recognizer already paused or stopped"}

@app.post("/resume", status_code=200, tags=["Recognizer"])
async def resume_recognizer_endpoint():
    if not recognizer_instance: raise HTTPException(status_code=503, detail="Recognizer not initialized")
    if not recognizer_instance.listening:
        try: recognizer_instance.resume_listening(); return {"message": "Recognizer resumed"}
        except Exception as e: logger.error(f"Resume failed: {e}"); raise HTTPException(status_code=500, detail=f"Failed to resume: {e}")
    else: return {"message": "Recognizer already listening"}

@app.get("/transcripts", tags=["Recognizer"])
async def get_transcripts_endpoint(limit: int = 20):
     if not recognizer_instance: raise HTTPException(status_code=503, detail="Recognizer not initialized")
     return list(recognizer_instance.transcription_log[-limit:])

# --- WebSocket Endpoint for Amplitude ---
@app.websocket("/ws/amplitude")
async def websocket_amplitude_endpoint(websocket: WebSocket):
    global ws_amplitude_connections # Modify global set
    await websocket.accept()
    ws_amplitude_connections.add(websocket)
    client = f"{websocket.client.host}:{websocket.client.port}"
    logger.info(f"[AmpWS] Connect: {client} (N={len(ws_amplitude_connections)})")
    try:
        while True: await websocket.receive_text() # Keep alive
    except WebSocketDisconnect: logger.info(f"[AmpWS] Disconnect: {client}")
    except Exception as e: logger.error(f"[AmpWS] Error {client}: {e}")
    finally: ws_amplitude_connections.discard(websocket); logger.info(f"[AmpWS] Closed: {client} (Rem={len(ws_amplitude_connections)})")

# --- WebSocket Endpoint for Results ---
@app.websocket("/ws/results")
async def websocket_results_endpoint(websocket: WebSocket):
    global ws_results_connections # Modify global set
    await websocket.accept()
    ws_results_connections.add(websocket)
    client = f"{websocket.client.host}:{websocket.client.port}"
    logger.info(f"[ResultsWS] Connect: {client} (N={len(ws_results_connections)})")
    try:
        while True: await websocket.receive_text() # Keep alive
    except WebSocketDisconnect: logger.info(f"[ResultsWS] Disconnect: {client}")
    except Exception as e: logger.error(f"[ResultsWS] Error {client}: {e}")
    finally: ws_results_connections.discard(websocket); logger.info(f"[ResultsWS] Closed: {client} (Rem={len(ws_results_connections)})")

# --- Main Execution ---
if __name__ == "__main__":
    # Environment setup if needed here
    # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.makedirs(config.AUDIO_OUTPUT_DIR, exist_ok=True)

    logger.info(f"Starting Uvicorn server on http://{config.API_HOST}:{config.API_PORT}")
    uvicorn.run(
        "main:app", host=config.API_HOST, port=config.API_PORT,
        log_level=config.LOG_LEVEL_STR.lower(), reload=False
    )













































# # main.py (FastAPI App - CORRECTED Formatting)

# import asyncio
# import os
# from contextlib import asynccontextmanager
# import uvicorn
# from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException

# # Import from local modules
# import config
# from logger_setup import logger
# from recognizer import SpeechIntentRecognizer
# from streamer import stream_amplitude
# # from utils import ... # utils are used within recognizer

# # --- Global State ---
# recognizer_instance: SpeechIntentRecognizer | None = None
# amplitude_stream_task: asyncio.Task | None = None
# ws_amplitude_connections: set[WebSocket] = set()
# ws_results_connections: set[WebSocket] = set()
# app_running_state = {"running": True}

# # --- FastAPI Lifecycle ---
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     global recognizer_instance, amplitude_stream_task, app_running_state
#     global ws_amplitude_connections, ws_results_connections

#     logger.info("FastAPI application startup sequence initiated...")
#     app_running_state["running"] = True
#     ws_amplitude_connections = set() # Reset sets on start
#     ws_results_connections = set()

#     # Initialize Recognizer
#     try:
#         logger.info("Initializing Speech Recognizer...")
#         recognizer_instance = SpeechIntentRecognizer()
#         recognizer_instance.calibrate_audio_model()
#         # Pass the global results_connections set to the processing loop
#         # This couples the modules but simplifies the current structure
#         recognizer_instance.start_main_loop() # This starts the thread that runs _run_async_processor
#         logger.info("Speech Recognizer started successfully.")
#     except Exception as e:
#         logger.critical(f"CRITICAL STARTUP FAILURE: Recognizer init failed: {e}", exc_info=True)
#         recognizer_instance = None

#     # Start Amplitude Streaming Task
#     if recognizer_instance:
#         logger.info("Starting amplitude streaming task...")
#         # Pass state dict and the relevant connections set
#         amplitude_stream_task = asyncio.create_task(
#             stream_amplitude(app_running_state, ws_amplitude_connections)
#         )
#     else:
#         logger.warning("Skipping amplitude streaming task because recognizer failed to initialize.")

#     try:
#         yield # Application runs here
#     finally:
#         # --- Shutdown Logic ---
#         logger.info("FastAPI application shutdown sequence initiated...")
#         app_running_state["running"] = False # Signal background tasks

#         # Stop Recognizer
#         if recognizer_instance:
#             logger.info("Shutting down recognizer instance...")
#             recognizer_instance.stop_listening() # Should handle thread join & cleanup
#             logger.info("Recognizer shutdown process complete.")
#         else:
#             logger.info("Recognizer was not active, skipping shutdown.")

#         # Stop Amplitude Streaming Task
#         if amplitude_stream_task and not amplitude_stream_task.done():
#             logger.info("Cancelling amplitude streaming task...")
#             amplitude_stream_task.cancel()
#             try:
#                 await asyncio.wait_for(amplitude_stream_task, timeout=5.0)
#                 logger.info("Amplitude streaming task shut down gracefully.")
#             except asyncio.CancelledError:
#                 logger.info("Amplitude streaming task cancellation confirmed.")
#             except asyncio.TimeoutError:
#                 logger.warning("Amplitude streaming task did not finish within timeout during shutdown.")
#             except Exception as e:
#                 logger.error(f"Error during amplitude task shutdown wait: {e}", exc_info=True)
#         elif amplitude_stream_task:
#              logger.info("Amplitude task was already done.")
#         else:
#              logger.info("Amplitude task was not started or already finished.")

#         # Close remaining WebSocket connections explicitly
#         logger.info("Closing any remaining WebSocket connections...")
#         close_tasks = []
#         # Use list copies for safe iteration while modifying
#         for ws in list(ws_amplitude_connections): close_tasks.append(ws.close(code=1001))
#         for ws in list(ws_results_connections): close_tasks.append(ws.close(code=1001))
#         if close_tasks:
#              try:
#                   await asyncio.gather(*close_tasks, return_exceptions=True)
#                   logger.info("Remaining WebSockets closed.")
#              except Exception as ws_close_err:
#                   logger.error(f"Error closing WebSockets during shutdown: {ws_close_err}")
#         else:
#              logger.info("No active WebSockets to close.")

#         logger.info("FastAPI application shutdown complete.")


# # --- Create FastAPI App ---
# app = FastAPI(
#     title="Speech API (Modular)",
#     description="Real-time voice processing.",
#     version="2.0.1", # Incremented version
#     lifespan=lifespan
# )

# # --- API Endpoints ---
# @app.get("/status", tags=["Recognizer"])
# async def get_status():
#     if not recognizer_instance: raise HTTPException(status_code=503, detail="Recognizer not initialized")
#     processing_alive = recognizer_instance.processing_thread.is_alive() if recognizer_instance.processing_thread else False
#     return {"recognizer_status": "listening" if recognizer_instance.listening else "paused",
#             "processing_active": processing_alive, "amplitude_clients": len(ws_amplitude_connections),
#             "results_clients": len(ws_results_connections), "queue_size": recognizer_instance.data_queue.qsize()}

# @app.post("/pause", status_code=200, tags=["Recognizer"])
# async def pause_recognizer_endpoint():
#     if not recognizer_instance: raise HTTPException(status_code=503, detail="Recognizer not initialized")
#     if recognizer_instance.listening: recognizer_instance.pause_listening(); return {"message": "Recognizer paused"}
#     else: return {"message": "Recognizer already paused or stopped"}

# @app.post("/resume", status_code=200, tags=["Recognizer"])
# async def resume_recognizer_endpoint():
#     if not recognizer_instance: raise HTTPException(status_code=503, detail="Recognizer not initialized")
#     if not recognizer_instance.listening:
#         try: recognizer_instance.resume_listening(); return {"message": "Recognizer resumed"}
#         except Exception as e: logger.error(f"Resume failed: {e}"); raise HTTPException(status_code=500, detail=f"Failed to resume: {e}")
#     else: return {"message": "Recognizer already listening"}

# @app.get("/transcripts", tags=["Recognizer"])
# async def get_transcripts_endpoint(limit: int = 20):
#      if not recognizer_instance: raise HTTPException(status_code=503, detail="Recognizer not initialized")
#      return list(recognizer_instance.transcription_log[-limit:])

# # --- WebSocket Endpoint for Amplitude ---
# @app.websocket("/ws/amplitude")
# async def websocket_amplitude_endpoint(websocket: WebSocket):
#     global ws_amplitude_connections
#     await websocket.accept()
#     ws_amplitude_connections.add(websocket)
#     client = f"{websocket.client.host}:{websocket.client.port}"
#     logger.info(f"[AmpWS] Connect: {client} (N={len(ws_amplitude_connections)})")
#     try:
#         while True: await websocket.receive_text() # Keep alive
#     except WebSocketDisconnect: logger.info(f"[AmpWS] Disconnect: {client}")
#     except Exception as e: logger.error(f"[AmpWS] Error {client}: {e}", exc_info=True)
#     finally: ws_amplitude_connections.discard(websocket); logger.info(f"[AmpWS] Closed: {client} (Rem={len(ws_amplitude_connections)})")

# # --- WebSocket Endpoint for Results ---
# @app.websocket("/ws/results")
# async def websocket_results_endpoint(websocket: WebSocket):
#     global ws_results_connections
#     await websocket.accept()
#     ws_results_connections.add(websocket)
#     client = f"{websocket.client.host}:{websocket.client.port}"
#     logger.info(f"[ResultsWS] Connect: {client} (N={len(ws_results_connections)})")
#     try:
#         while True: await websocket.receive_text() # Keep alive
#     except WebSocketDisconnect: logger.info(f"[ResultsWS] Disconnect: {client}")
#     except Exception as e: logger.error(f"[ResultsWS] Error {client}: {e}", exc_info=True)
#     finally: ws_results_connections.discard(websocket); logger.info(f"[ResultsWS] Closed: {client} (Rem={len(ws_results_connections)})")

# # --- Main Execution ---
# if __name__ == "__main__":
#     # Environment setup if needed here
#     # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#     os.makedirs(config.AUDIO_OUTPUT_DIR, exist_ok=True)

#     logger.info(f"Starting Uvicorn server on http://{config.API_HOST}:{config.API_PORT}")
#     uvicorn.run(
#         "main:app", host=config.API_HOST, port=config.API_PORT,
#         log_level=config.LOG_LEVEL_STR.lower(), reload=False
#     )