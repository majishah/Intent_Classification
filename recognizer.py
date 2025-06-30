# recognizer.py (CORRECTED Connection Set Handling)

import io
import threading
import time
import os
import numpy as np
import torch
from queue import Queue, Empty
import asyncio
from faster_whisper import WhisperModel
import pyaudio
import wave
import logging
import traceback
import pygame.mixer
from transformers import pipeline, AutoTokenizer
import datetime
import json

# Import from local modules
import config # Import config directly
from logger_setup import logger # Use configured logger
from utils import calculate_waveform_amplitudes, audio_data_to_wav_bytes # Import helpers

class SpeechIntentRecognizer:
    """Speech Recognizer using User's Logic + Waveform/Score sending."""

    def __init__(self, ws_results_set_ref: set) -> None: # <<< Accept set reference
        logger.info("Initializing SpeechIntentRecognizer (User Logic)...")
        try:
            self.data_queue = Queue()
            self.pyaudio = None; self.stream = None
            self._init_pyaudio()
            self.load_vad_model()
            self.audio_model = self.load_sr_model()
            os.makedirs(config.AUDIO_OUTPUT_DIR, exist_ok=True)
            logger.info(f"Audio output dir: {config.AUDIO_OUTPUT_DIR}")
            self.transcription_log = []
            self.listening = False; self.running = True; self.processing_thread = None
            self.state_lock = threading.Lock(); self.buffer_lock = threading.Lock()
            self.audio_buffer = np.array([], dtype=np.int16)
            # self.results_connections = set() # <<< REMOVED THIS LINE
            self.ws_results_set_ref = ws_results_set_ref # <<< STORE the passed reference
            self._init_beep_sound()
            logger.info(f"Loading Intent model: {config.INTENT_MODEL_PATH}")
            if not os.path.exists(config.INTENT_MODEL_PATH):
                raise FileNotFoundError(f"Intent model not found: {config.INTENT_MODEL_PATH}")
            self.intent_classifier = pipeline("zero-shot-classification", model=config.INTENT_MODEL_PATH)
            logger.info("Intent classifier initialized.")
            self.last_end_time = 0; self.silence_duration = 0
            logger.info("SpeechIntentRecognizer initialized successfully")
        except Exception as e:
            logger.critical(f"Init failed: {e}", exc_info=True)
            # Cleanup partially initialized resources
            if self.pyaudio:
                 try: self.pyaudio.terminate()
                 except Exception: pass # Ignore errors during cleanup on init failure
            if pygame.mixer.get_init():
                 try: pygame.mixer.quit()
                 except Exception: pass
            raise

    # --- Initialization Methods ---
    def _init_pyaudio(self) -> None:
        logger.info("Initializing PyAudio...")
        try:
            self.pyaudio = pyaudio.PyAudio()
            logger.info("Available audio input devices:")
            default_info = None
            try:
                default_info = self.pyaudio.get_default_input_device_info()
                logger.info(f"---> Default Input Device: Index {default_info['index']} - {default_info['name']}")
            except IOError:
                logger.warning("No default input device found by PyAudio.")

            found_configured_device = False
            for i in range(self.pyaudio.get_device_count()):
                dev = self.pyaudio.get_device_info_by_index(i)
                if dev['maxInputChannels'] >= config.CHANNELS:
                    logger.info(f"  Device {i}: {dev['name']}, Inputs: {dev['maxInputChannels']}, Rate: {dev['defaultSampleRate']}")
                    if i == config.SR_INPUT_DEVICE:
                         logger.info(f"---> Using Configured Device: Index {i}")
                         found_configured_device = True
            if not found_configured_device:
                 logger.error(f"Configured SR_INPUT_DEVICE ({config.SR_INPUT_DEVICE}) NOT FOUND or invalid!")
                 if default_info and default_info['maxInputChannels'] >= config.CHANNELS:
                     logger.warning(f"CRITICAL: Falling back to default device index {default_info['index']}. CHECK SR_INPUT_DEVICE!")
                 else:
                     raise RuntimeError(f"No suitable input device found. Check SR_INPUT_DEVICE ({config.SR_INPUT_DEVICE}).")
        except Exception as e:
            logger.error(f"PyAudio init failed: {e}", exc_info=True)
            raise RuntimeError(f"Audio init failed: {str(e)}")

    def _init_beep_sound(self) -> None:
        logger.info("Initializing beep sound...")
        try:
            pygame.mixer.init(frequency=config.RATE, size=-16, channels=config.CHANNELS)
            if not os.path.exists(config.BEEP_FILE):
                raise FileNotFoundError(f"Beep file not found: {config.BEEP_FILE}")
            self.beep_sound = pygame.mixer.Sound(config.BEEP_FILE)
            self.beep_sound.set_volume(1.0)
            logger.info("Beep sound loaded.")
        except Exception as e:
            logger.error(f"Beep init failed: {e}", exc_info=True)
            logger.warning("Beep disabled."); self.beep_sound = None

    def load_vad_model(self) -> None:
        logger.info(f"Loading VAD model from: {config.LOCAL_SILERO_VAD_PATH}")
        try:
            if not os.path.exists(config.LOCAL_SILERO_VAD_PATH):
                raise FileNotFoundError(f"VAD path not found: {config.LOCAL_SILERO_VAD_PATH}")
            self.vad_model, utils = torch.hub.load(repo_or_dir=config.LOCAL_SILERO_VAD_PATH, model='silero_vad', source='local', trust_repo=True)
            (self.get_speech_timestamps, self.save_audio, self.read_audio,
             self.VADIterator, self.collect_chunks) = utils
            logger.info("VAD model loaded.")
        except Exception as e:
            logger.error(f"VAD load failed: {e}", exc_info=True)
            raise RuntimeError(f"VAD load failed: {str(e)}")

    def load_sr_model(self) -> WhisperModel:
        logger.info(f"Loading SR model from: {config.SRMODEL_PATH}")
        try:
            if not os.path.exists(config.SRMODEL_PATH):
                raise FileNotFoundError(f"SR path not found: {config.SRMODEL_PATH}")
            model = WhisperModel(config.SRMODEL_PATH, device=config.SR_COMPUTE_DEVICE, compute_type=config.SR_COMPUTE_TYPE)
            logger.info(f"SR model loaded: Device={config.SR_COMPUTE_DEVICE}, Compute={config.SR_COMPUTE_TYPE}")
            return model
        except Exception as e:
            logger.error(f"SR load failed: {e}", exc_info=True)
            raise RuntimeError(f"SR load failed: {str(e)}")

    # --- Audio Processing Methods ---
    def apply_gain_and_noise_suppression(self, audio_data: np.ndarray) -> np.ndarray:
        try:
            audio_f = audio_data.astype(np.float32) * config.MICROPHONE_GAIN
            audio_c = np.clip(audio_f, -32768, 32767)
            audio_i = audio_c.astype(np.int16)
            audio_i[np.abs(audio_i) < config.NOISE_THRESHOLD] = 0
            return audio_i
        except Exception as e:
            logger.error(f"Gain/Noise error: {e}", exc_info=True)
            return audio_data # Return original on error

    # --- audio_callback (User's Original Logic - Corrected Formatting) ---
    def audio_callback(self, in_data, frame_count, time_info, status_flags):
        try:
            if status_flags:
                logger.warning(f"Audio callback status: {status_flags}")

            audio_d = np.frombuffer(in_data, dtype=np.int16)
            processed_d = self.apply_gain_and_noise_suppression(audio_d)

            with self.buffer_lock:
                self.audio_buffer = np.append(self.audio_buffer, processed_d)
                if len(self.audio_buffer) > config.BUFFER_MAX_SIZE:
                    self.audio_buffer = self.audio_buffer[-config.BUFFER_MAX_SIZE:]

                buffer_len = len(self.audio_buffer)
                buffer_dur = buffer_len / config.RATE
                buffer_start_est = time.time() - buffer_dur

                try:
                    audio_f32 = self.audio_buffer.astype(np.float32)/32768.0
                    audio_t = torch.from_numpy(audio_f32)
                    timestamps = self.get_speech_timestamps(
                        audio_t,
                        self.vad_model,
                        sampling_rate=config.RATE,
                        min_silence_duration_ms=config.SR_VAD_FILTER_PARAMETERS["min_silence_duration_ms"],
                        min_speech_duration_ms=config.SR_VAD_FILTER_PARAMETERS["min_speech_duration_ms"]
                    )
                except Exception as vad_e:
                    logger.error(f"VAD error: {vad_e}", exc_info=True)
                    return (None, pyaudio.paContinue)

                if timestamps:
                    self.silence_duration = 0
                    for ts in timestamps:
                        start_s = max(0, ts['start'] - config.PRE_ROLL_SAMPLES)
                        end_s = ts['end']
                        # Original Check
                        if end_s < buffer_len - config.CHUNK_SIZE:
                            start_t = buffer_start_est + (start_s / config.RATE)
                            if start_t > self.last_end_time:
                                try:
                                    speech_arr = self.audio_buffer[start_s:end_s]
                                    if speech_arr.size > 0:
                                        speech_bytes = speech_arr.tobytes()
                                        end_t = buffer_start_est + (end_s / config.RATE)
                                        self.data_queue.put({
                                            'audio': speech_bytes,
                                            'start_time': start_t,
                                            'end_time': end_t
                                        })
                                        logger.info(f"Speech detected: ~{start_t:.2f}s->~{end_t:.2f}s")
                                        self.last_end_time = end_t
                                except IndexError:
                                    logger.error(f"IndexError: start={start_s}, end={end_s}, len={buffer_len}")
                                except Exception as seg_e:
                                    logger.error(f"Segment proc error: {seg_e}", exc_info=True)

                    if buffer_len > config.BUFFER_MAX_SIZE: # Trim again if needed
                        self.audio_buffer = self.audio_buffer[-config.BUFFER_MAX_SIZE:]

                else: # No timestamps
                    self.silence_duration += frame_count / config.RATE
                    if self.silence_duration >= 0.5 and buffer_len > config.CHUNK_SIZE:
                        if np.any(self.audio_buffer):
                            logger.debug(f"Silence check ({self.silence_duration:.2f}s), final VAD...")
                            try:
                                audio_f32 = self.audio_buffer.astype(np.float32)/32768.0
                                audio_t = torch.from_numpy(audio_f32)
                                final_ts = self.get_speech_timestamps(
                                    audio_t, self.vad_model, sampling_rate=config.RATE,
                                    min_silence_duration_ms=config.SR_VAD_FILTER_PARAMETERS["min_silence_duration_ms"],
                                    min_speech_duration_ms=config.SR_VAD_FILTER_PARAMETERS["min_speech_duration_ms"]
                                )
                                if final_ts:
                                    ts = final_ts[-1]
                                    start_s = max(0, ts['start'] - config.PRE_ROLL_SAMPLES)
                                    end_s = ts['end']
                                    start_t = buffer_start_est + (start_s / config.RATE)
                                    if start_t > self.last_end_time:
                                        speech_arr = self.audio_buffer[start_s:end_s]
                                        if speech_arr.size > 0:
                                            end_t = buffer_start_est + (end_s / config.RATE)
                                            self.data_queue.put({
                                                'audio': speech_arr.tobytes(),
                                                'start_time': start_t,
                                                'end_time': end_t
                                            })
                                            logger.info(f"Speech (finalized): ~{start_t:.2f}s->~{end_t:.2f}s")
                                            self.last_end_time = end_t
                            except Exception as final_e:
                                logger.error(f"Final segment error: {final_e}", exc_info=True)
                        # Original Buffer Clear
                        logger.debug("Clearing audio buffer after silence check.")
                        self.audio_buffer = np.array([], dtype=np.int16)
                        self.last_end_time = 0

        except Exception as e:
            logger.error(f"Callback error: {e}", exc_info=True)
            return (None, pyaudio.paContinue) # Attempt to continue

        return (None, pyaudio.paContinue)

    # --- audio_data_to_wav_bytes (Calls Util function) ---
    def audio_data_to_wav_bytes(self, audio_data: bytes, sample_rate=config.RATE, save_to_file: bool = False) -> tuple[io.BytesIO | None, str | None]:
         # Use the utility function, passing the PyAudio instance
         return audio_data_to_wav_bytes(audio_data, self.pyaudio, sample_rate, save_to_file)

    # --- Intent Classification Methods ---
    def predict_intent(self, user_input, labels):
        if not user_input or not labels: logger.warning("Missing input/labels for intent."); return None
        try:
            return self.intent_classifier(user_input, labels)
        except Exception as e:
            logger.error(f"Intent predict failed: {e}", exc_info=True); return None

    def print_intent_details(self, level, intent_result):
        if not intent_result or not isinstance(intent_result, dict): return
        seq=intent_result.get('sequence','N/A'); labels=intent_result.get('labels',[]); scores=intent_result.get('scores',[])
        if not labels or not scores: return
        pred_l=labels[0]; pred_s=scores[0]
        print(f"\n{'='*20} L{level} Intent {'='*20}")
        print(f"Input: '{seq}'")
        print(f"Pred: {pred_l} ({pred_s:.4f})")
        print("Details:")
        for l,s in zip(labels,scores): print(f"  - {l:<20}: {s:.4f}")
        print(f"{'='*50}")

    # --- process_audio_data (MODIFIED to use self.ws_results_set_ref) ---
    async def process_audio_data(self) -> None: # <<< REMOVE parameter results_connections
        # Use self.ws_results_set_ref instead of passing the set
        logger.info("Audio processing loop started (User Logic + Scores/Metadata).")
        while self.running:
            item = None; processed = False
            try:
                try:
                    item = self.data_queue.get(timeout=0.1)
                except Empty:
                    await asyncio.sleep(0.05); continue

                if item is None: logger.info("Poison pill. Exiting loop."); break
                if not isinstance(item, dict) or 'audio' not in item: logger.warning(f"Invalid item: {type(item)}"); continue

                speech_data = item['audio']
                start_time = item.get('start_time', 0)
                end_time = item.get('end_time', 0)

                audio_duration_s = 0.0
                if self.pyaudio:
                    try: sw=self.pyaudio.get_sample_size(config.FORMAT); ns=len(speech_data)/(sw*config.CHANNELS); audio_duration_s=ns/config.RATE if config.RATE>0 else 0
                    except Exception as de: logger.error(f"Duration err: {de}")

                logger.debug("Calculating waveform..."); waveform_amps = calculate_waveform_amplitudes(speech_data, sample_rate=config.RATE)
                wav_data_io, saved_filename = self.audio_data_to_wav_bytes(speech_data, sample_rate=config.RATE, save_to_file=config.SAVE_AUDIO_FILES)
                if not wav_data_io: logger.warning("WAV conversion failed"); del speech_data; continue

                payload = {"transcription": "[No speech]", "intent": "N/A", "waveform_amplitudes": waveform_amps,
                           "start_time": start_time, "end_time": end_time, "error": None, "intent_score": None,
                           "transcription_avg_logprob": None, "audio_duration_s": audio_duration_s, "processing_timestamp_iso": None}
                try:
                    if self.beep_sound: asyncio.create_task(self.play_beep_async())
                    wav_data_io.seek(0)
                    try:
                        logger.debug(f"Transcribing: {saved_filename or 'In Mem'}"); segments, info = self.audio_model.transcribe(audio=wav_data_io, language=config.SR_LANGUAGE, vad_filter=config.SR_VAD_FILTER, vad_parameters=config.SR_VAD_FILTER_PARAMETERS, temperature=config.MODEL_TEMPERATURE, word_timestamps=True)
                        logger.debug(f"Transcribed: Lang={info.language}, Prob={info.language_probability:.2f}")
                        found = False; texts = []; logprobs = []; current_end = start_time
                        for s in segments:
                            txt = s.text.strip()
                            if not txt: continue
                            found = True
                            texts.append(txt)
                            s_start=start_time+s.start; s_end=start_time+s.end; current_end=max(current_end,s_end)
                            if hasattr(s,'avg_logprob') and s.avg_logprob is not None and not np.isnan(s.avg_logprob): logprobs.append(s.avg_logprob)
                            logger.info(f"Seg: \"{txt}\" [~{s_start:.2f}s->~{s_end:.2f}s] (LP={s.avg_logprob:.3f})")
                            self.transcription_log.append({"text":txt,"start_time":s_start,"end_time":s_end,"file":saved_filename}); self.transcription_log=self.transcription_log[-100:]
                        if found:
                            payload["transcription"] = " ".join(texts); payload["end_time"] = current_end
                            if logprobs: payload["transcription_avg_logprob"] = float(np.mean(logprobs))
                            lp_val=payload['transcription_avg_logprob']; lp_str=f"{lp_val:.3f}" if lp_val is not None else "N/A"
                            logger.info(f"Final Utterance: \"{payload['transcription']}\" [~{start_time:.2f}s->~{current_end:.2f}s] (AvgLP: {lp_str})")
                            print(f"\n{'#'*25} Intent {'#'*25}\nInput: '{payload['transcription']}'\n{'#'*60}")
                            l1=self.predict_intent(payload["transcription"], config.LABELS_LEVEL_ONE)
                            if l1: self.print_intent_details(1,l1); p_l1=l1['labels'][0]; payload["intent"]=p_l1; payload["intent_score"]=float(l1['scores'][0])
                            if p_l1 in config.LABELS_LEVEL_TWO: l2=self.predict_intent(payload["transcription"],config.LABELS_LEVEL_TWO[p_l1])
                            if l2: self.print_intent_details(2,l2); p_l2=l2['labels'][0]; payload["intent"]=p_l2; payload["intent_score"]=float(l2['scores'][0])
                            if p_l2 in config.LABELS_LEVEL_THREE: l3=self.predict_intent(payload["transcription"],config.LABELS_LEVEL_THREE[p_l2])
                            if l3: self.print_intent_details(3,l3); payload["intent"]=l3['labels'][0]; payload["intent_score"]=float(l3['scores'][0])
                        elif saved_filename: logger.info(f"No speech by Whisper in {saved_filename}")
                    except Exception as tx_e: logger.error(f"Tx error: {tx_e}"); payload["error"]=f"Tx Failed: {tx_e}"
                except Exception as proc_e: logger.error(f"Proc error: {proc_e}"); payload["error"]=f"Proc Failed: {proc_e}"
                finally: del speech_data; del wav_data_io

                payload["processing_timestamp_iso"]=datetime.datetime.now().isoformat()

                # --- MODIFIED Send Results Block ---
                if self.ws_results_set_ref: # <<< Use the stored instance reference
                    logger.info(f"Sending results ({len(self.ws_results_set_ref)} clients)."); dc=set(); conns=list(self.ws_results_set_ref)
                    try: payload_str = json.dumps(payload)
                    except TypeError as json_e: logger.error(f"JSON err: {json_e}"); payload_str=json.dumps({"error":f"Result serialization failed: {json_e}"})
                    tasks=[asyncio.create_task(c.send_text(payload_str)) for c in conns]
                    results=await asyncio.gather(*tasks,return_exceptions=True)
                    for i,r in enumerate(results):
                        if isinstance(r,Exception):
                            logger.warning(f"[ResultsWS] Send fail {conns[i].client}: {r}")
                            dc.add(conns[i])
                    # Let the endpoint handler remove disconnected clients
                    # for c in dc: self.ws_results_set_ref.discard(c)
                else:
                    logger.debug("No results clients connected (or set reference not available).")
                # -----------------------------------
                processed = True
            except Exception as outer_e: logger.error(f"Outer loop err: {outer_e}"); await asyncio.sleep(1)
            finally:
                if item:
                    self.data_queue.task_done()
        logger.info("Processing loop finished.")


    # --- Control Methods ---
    def start_listening(self) -> None:
        with self.state_lock:
            if self.listening: logger.warning("Already listening."); return
            if not self.pyaudio: raise RuntimeError("PyAudio dead.")
            if not self.running: logger.warning("Recognizer stopped."); return
            logger.info("Starting stream...")
            try:
                actual_rate = config.RATE
                if not self.pyaudio.is_format_supported(config.RATE, input_device=config.SR_INPUT_DEVICE, input_channels=config.CHANNELS, input_format=config.FORMAT):
                     supported=False; rates=[16000, 48000, 44100]; logger.warning(f"Rate {config.RATE} not supported, trying {rates}")
                     for r in rates:
                         if self.pyaudio.is_format_supported(r,input_device=config.SR_INPUT_DEVICE,input_channels=config.CHANNELS,input_format=config.FORMAT):
                             logger.warning(f"Using {r}Hz"); actual_rate=r; supported=True; break
                     if not supported: raise ValueError("No supported rate found")
                self.stream = self.pyaudio.open(format=config.FORMAT, channels=config.CHANNELS, rate=actual_rate, input=True, input_device_index=config.SR_INPUT_DEVICE, frames_per_buffer=config.CHUNK_SIZE, stream_callback=self.audio_callback)
                if actual_rate != config.RATE: logger.warning(f"Actual rate {actual_rate}Hz differs from config {config.RATE}Hz.")
                self.stream.start_stream(); self.listening = True; self.audio_buffer = np.array([],dtype=np.int16); self.last_end_time = 0
                logger.info(f"Listening: Dev {config.SR_INPUT_DEVICE} @ {actual_rate}Hz")
            except Exception as e:
                logger.error(f"Start failed: {e}", exc_info=True)
                if self.stream: self.stream.close(); self.stream = None
                raise RuntimeError(f"Start failed: {str(e)}")

    def pause_listening(self) -> None:
        with self.state_lock:
            if self.stream and self.listening:
                try:
                    if self.stream.is_active():
                        self.stream.stop_stream(); logger.info("Listening Paused")
                    else:
                        logger.warning("Stream inactive on pause.")
                    self.listening = False
                except Exception as e:
                    logger.error(f"Error pausing: {e}", exc_info=True)
            elif not self.stream:
                 logger.warning("Cannot pause, stream inactive.")
            else:
                 logger.info("Already paused.")

    def resume_listening(self) -> None:
        with self.state_lock:
            if self.stream and not self.listening:
                try:
                    if not self.stream.is_active():
                        self.stream.start_stream(); logger.info("Listening Resumed")
                    else:
                        logger.warning("Stream active on resume.")
                    self.listening = True
                except Exception as e:
                    logger.error(f"Error resuming: {e}", exc_info=True)
            elif not self.stream:
                 logger.warning("Cannot resume, stream inactive.")
            else:
                 logger.info("Already listening.")

    def stop_listening(self) -> None:
        logger.info("Attempting full stop...")
        with self.state_lock:
            if not self.running:
                logger.info("Stop already initiated."); return
            self.running = False # Signal loops FIRST
            if self.stream:
                logger.info("Stopping PyAudio stream...")
                try:
                    if self.stream.is_active():
                        self.stream.stop_stream()
                except Exception as e:
                    logger.error(f"Error stopping stream: {e}", exc_info=True)
                self.listening = False
            else:
                logger.info("Stream already inactive.")

        logger.info("Signalling processing queue stop...")
        self.data_queue.put(None) # Poison pill

        if self.processing_thread and self.processing_thread.is_alive():
            logger.info("Waiting for processing thread (max 5s)...")
            self.processing_thread.join(timeout=5.0)
            logger.info(f"Processing thread {'alive' if self.processing_thread.is_alive() else 'joined'}.")
        else:
            logger.info("Processing thread inactive.")

        with self.state_lock: # Final cleanup
             if self.stream:
                 try:
                     self.stream.close(); logger.info("PyAudio stream closed.")
                 except Exception as e:
                     logger.error(f"Error closing stream: {e}", exc_info=True)
                 finally:
                      self.stream = None
             if self.pyaudio:
                 try:
                     self.pyaudio.terminate(); logger.info("PyAudio terminated.")
                 except Exception as e:
                     logger.error(f"Error terminating PyAudio: {e}", exc_info=True)
                 finally:
                      self.pyaudio = None

        try: # Stop Pygame
            if pygame.mixer.get_init():
                pygame.mixer.quit(); logger.info("Pygame mixer stopped")
        except Exception as e:
            logger.error(f"Error stopping pygame mixer: {e}", exc_info=True)

        logger.info("Full stop process complete.")

    # --- play_beep_async ---
    async def play_beep_async(self):
        if not self.beep_sound or not pygame.mixer.get_init(): return
        try:
            self.beep_sound.play()
            await asyncio.sleep(config.BEEP_DURATION)
            logger.debug("Beep played")
        except Exception as e:
            logger.error(f"Beep play error: {e}", exc_info=True)
        finally:
            try:
                while pygame.mixer.get_busy():
                    await asyncio.sleep(0.01)
                pygame.mixer.stop()
                logger.debug("Beep stopped.")
            except Exception as e_stop:
                 logger.error(f"Beep stop error: {e_stop}", exc_info=True)

    # --- calibrate_audio_model ---
    def calibrate_audio_model(self) -> None:
        logger.info("Calibrating SR Model...");
        try:
            if not os.path.exists(config.SR_CALIB_AUDIO_FILES_PATH):
                logger.warning(f"Cal file not found: {config.SR_CALIB_AUDIO_FILES_PATH}. Skipping."); return
            cal_segments, cal_info = self.audio_model.transcribe(config.SR_CALIB_AUDIO_FILES_PATH, language=config.SR_LANGUAGE)
            logger.info(f"Warm-up: Lang={cal_info.language}, Prob={cal_info.language_probability:.2f}")
            for s in cal_segments: logger.info(f"[Cal: {s.start:.2f}s->{s.end:.2f}s] {s.text}")
            logger.info("Calibration Complete.")
        except Exception as e:
            logger.error(f"Calibration error: {e}"); logger.warning("Continuing anyway.")

    # --- Methods for FastAPI Integration ---
    def start_main_loop(self) -> None:
        if self.processing_thread and self.processing_thread.is_alive(): logger.warning("Proc loop running."); return
        try:
            logger.info("Starting main loop...");
            if not self.listening: self.start_listening()
            self.processing_thread = threading.Thread(target=self._run_async_processor, name="ProcessorThread", daemon=True)
            self.processing_thread.start()
            logger.info("Proc thread started.")
        except Exception as e:
            logger.critical(f"Loop startup failed: {e}"); self.stop_listening()

    # --- MODIFIED _run_async_processor (No longer passes set) ---
    def _run_async_processor(self):
        tname=threading.current_thread().name; logger.info(f"Async proc thread [{tname}] started."); loop = None
        try:
            loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
            # Calls process_audio_data without the set argument now
            loop.run_until_complete(self.process_audio_data())
        except Exception as e:
            logger.error(f"Async proc err [{tname}]: {e}", exc_info=True)
        finally:
             if loop:
                 try: # Cleanup loop resources
                    tasks = asyncio.all_tasks(loop);
                    if tasks: logger.info(f"[{tname}] Cancelling {len(tasks)} tasks..."); [task.cancel() for task in tasks]; loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True)); logger.info(f"[{tname}] Tasks cancelled.")
                    loop.run_until_complete(loop.shutdown_asyncgens())
                 except Exception as el_e: logger.error(f"Loop cleanup error [{tname}]: {el_e}")
                 finally:
                     if not loop.is_closed(): loop.close(); logger.info(f"[{tname}] Event loop closed.")
             logger.info(f"Async proc thread [{tname}] finished.")