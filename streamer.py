import asyncio
import time
import numpy as np
import pyaudio
import websockets # Not actually used if sending is handled by endpoint

# Import from local modules
import config
from logger_setup import logger

# This version relies on the FastAPI endpoint to manage connections and send data.
# This task just produces the amplitude value. We need a way to communicate it.
# Using asyncio.Queue is a good decoupled approach.

# --- Shared Queue for Amplitude Values ---
# This queue needs to be accessible by both this task and the WebSocket endpoint handler.
# Define it globally here or pass it around (passing is cleaner).
# Let's define it globally here for simplicity, but consider dependency injection later.
# amplitude_value_queue = asyncio.Queue(maxsize=1) # Max size 1: only latest value matters

async def stream_amplitude(app_state: dict, amplitude_connections: set): # Keep connections arg for now, maybe remove later
    """Background task to CALCULATE microphone amplitude (sending handled by endpoint)."""
    p_amp = None; stream_amp = None
    logger.info("[Amplitude] Task started.")
    try:
        p_amp = pyaudio.PyAudio()
        logger.info("[Amplitude] Initializing PyAudio for amplitude.")
        try: # Device Check
             dev_info = p_amp.get_device_info_by_index(config.AMPLITUDE_DEVICE_INDEX)
             logger.info(f"[Amplitude] Checking device {config.AMPLITUDE_DEVICE_INDEX}: {dev_info['name']}")
             if not p_amp.is_format_supported(config.AMPLITUDE_RATE, input_device=config.AMPLITUDE_DEVICE_INDEX,
                                            input_channels=config.CHANNELS, input_format=config.FORMAT):
                  raise ValueError(f"Amplitude format/rate not supported")
        except Exception as e:
            logger.error(f"[Amplitude] Device check failed: {e}", exc_info=True); raise

        stream_amp = p_amp.open(format=config.FORMAT, channels=config.CHANNELS, rate=config.AMPLITUDE_RATE, input=True,
                                frames_per_buffer=config.AMPLITUDE_CHUNK_SIZE, input_device_index=config.AMPLITUDE_DEVICE_INDEX)
        logger.info("[Amplitude] Stream opened.")

        latest_amplitude_str = "0.0000" # Store latest value

        while app_state.get("running", True):
            start_time = time.monotonic()
            try:
                try:
                    data = stream_amp.read(config.AMPLITUDE_CHUNK_SIZE, exception_on_overflow=False)
                except IOError as e:
                    if e.errno == pyaudio.paInputOverflowed:
                        logger.warning("[Amplitude] Input Overflow"); await asyncio.sleep(0.01); continue
                    else:
                         logger.error(f"[Amplitude] Read IOError: {e}", exc_info=True); await asyncio.sleep(0.5); continue

                audio_array = np.frombuffer(data, dtype=np.int16)
                if audio_array.size == 0:
                    await asyncio.sleep(0.01); continue

                # --- Calculation ---
                rms_normalized = np.sqrt(np.mean((audio_array.astype(np.float64) / 32768.0)**2))
                clipped_amplitude = np.clip(rms_normalized * config.AMPLITUDE_MULTIPLIER, 0, config.MAX_SCALED_AMPLITUDE)
                latest_amplitude_str = f"{clipped_amplitude:.4f}" # Update latest value

                # --- Communication (Option 1: Queue - Preferred but needs endpoint change) ---
                # try:
                #     # Clear queue if full (only keep latest)
                #     if amplitude_value_queue.full(): await amplitude_value_queue.get()
                #     await amplitude_value_queue.put(latest_amplitude_str)
                # except asyncio.QueueFull: pass # Should not happen with maxsize=1 and clear

                # --- Communication (Option 2: Direct Send - Kept for now) ---
                if amplitude_connections:
                    current_connections = list(amplitude_connections) # Safe copy
                    tasks = [conn.send_text(latest_amplitude_str) for conn in current_connections]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    # Let endpoint handle disconnects based on send failures
                    for i, result in enumerate(results):
                        if isinstance(result, Exception):
                            # Log warning, but endpoint WS handler should remove client
                            logger.warning(f"[Amplitude] Send failed to {current_connections[i].client}: {result}")
                            # Maybe signal endpoint? Or rely on keepalive check?


                elapsed_time = time.monotonic() - start_time
                await asyncio.sleep(max(0, config.AMPLITUDE_UPDATE_INTERVAL - elapsed_time))

            except Exception as e_inner:
                logger.error(f"[Amplitude] Loop error: {e_inner}", exc_info=True)
                await asyncio.sleep(0.5) # Prevent busy loop

    except asyncio.CancelledError:
        logger.info("[Amplitude] Task cancelled.")
    except Exception as e_outer:
        logger.error(f"[Amplitude] Critical error: {e_outer}", exc_info=True)
    finally:
        logger.info("[Amplitude] Stopping amplitude stream resources...")
        # app_state["running"] = False # Don't signal stop globally unless fatal
        if stream_amp:
            try:
                if stream_amp.is_active(): stream_amp.stop_stream()
                stream_amp.close()
            except Exception as e:
                logger.error(f"[Amplitude] Error closing stream: {e}", exc_info=True)
        if p_amp:
            try:
                p_amp.terminate()
            except Exception as e:
                logger.error(f"[Amplitude] Error terminating PyAudio: {e}", exc_info=True)
        logger.info("[Amplitude] Amplitude stream task finished cleanup.")