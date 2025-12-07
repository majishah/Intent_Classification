"""
The Listen class orchestrates the complete real-time voice processing pipeline. 
It is the central engine that manages the entire lifecycle of an utterance, from 
initial audio capture to final, biometrically verified transcription.

This engine employs a multi-stage Voice Activity Detection (VAD) system to 
accurately detect human speech. For speaker analysis, it leverages `pyannote.audio` 
pipelines to perform speaker diarization, which identifies the number of speakers 
and detects overlapped speech. If a single speaker is detected, it can perform 
biometric speaker verification to identify the individual against pre-registered 
voice embeddings.

Transcription is handled by the `faster_whisper` library, running in a separate 
process for maximum performance on a CPU or GPU, while also providing stabilized, 
real-time transcription previews. It also integrates wake word detection 
(supporting `pvporcupine` and `openwakeword`) for hands-free activation. The entire 
system is built on a concurrent architecture to remain responsive and is highly 
customizable through a rich set of callbacks.

Key Features:
- Advanced Voice Activity Detection (VAD): Uses a two-stage process (`webrtcvad` 
  and `silero-vad`) for fast and accurate speech detection.
- Speaker Diarization: Counts the number of speakers in an audio segment and 
  detects overlapped speech using `pyannote.audio`.
- Speaker Verification: Performs biometric identification of a single speaker 
  against known voice embeddings.
- Wake Word Detection: Supports multiple backends (`pvporcupine`, `openwakeword`) 
  to initiate recording on a spoken command.
- High-Performance Transcription: Utilizes `faster-whisper` in a separate 
  process for fast and accurate speech-to-text conversion, with real-time 
  previews.
- Concurrent Architecture: Leverages multiprocessing and multithreading to 
  ensure a non-blocking, responsive system.
- Extensible via Callbacks: Provides a rich set of event callbacks for easy 
  integration into larger applications.

"""

# ==============================================================================
# 1. IMPORTS
# ==============================================================================

# ==============================================================================
# Standard Library Imports
# ==============================================================================
from typing import Iterable, List, Optional, Tuple, Union
import signal as system_signal
from ctypes import c_bool
import collections
import traceback
import threading
import platform
import datetime
import struct
import base64
import queue
import copy
import time
import wave
import gc
import io
import os
import re
import psutil

# ==============================================================================
# 2. ENVIRONMENT CONFIGURATION
# ==============================================================================
# Environment variables must be set BEFORE the libraries they affect are imported.
# This section should come immediately after the initial imports.

# Suppress the Pygame welcome message.
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"

# Set OpenMP runtime duplicate library handling to OK.
# This is often needed for PyTorch/Numpy in certain environments.
# (Use only for development if it causes issues).
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ==============================================================================
# Third-Party Library Imports
# ==============================================================================
from pyannote.audio import (Audio as PyannoteAudio, Inference as PyannoteInference, Model as PyannoteModel, Pipeline as PyannotePipeline)
from faster_whisper import BatchedInferencePipeline, WhisperModel
from openwakeword.model import Model as OpenWakeWordModel
from scipy.spatial.distance import cdist
import torch.multiprocessing as mp
from scipy.signal import resample
from scipy import signal
import soundfile as sf
import faster_whisper
import openwakeword
import numpy as np
import pvporcupine
import webrtcvad
import pygame
import torch
import halo

# ==============================================================================
# Local Application Imports
# ==============================================================================
from listen.utils_layer.logger import setup_logger
from .safepipe import SafePipe

# ==============================================================================
# 3. LOGGER SETUP
# ==============================================================================
# It's good practice to set up the logger early so that any issues during
# the module's initialization can be properly logged.

# --- Configuration for the logger ---
# This flag can be changed to control the verbosity of the log output.
SHOW_EXTENDED_LOGGER = True

# --- Logger Initialization ---
# Assuming `setup_logger` is a function you've defined elsewhere.
# If not, the standard logger setup would go here.
logger = setup_logger(
    "Listen", 
    log_file="listen/logs/listen_driver.log", 
    level="DEBUG" if SHOW_EXTENDED_LOGGER else "INFO"
)
# Prevents log messages from being duplicated by parent (e.g., root) loggers.
logger.propagate = False

# ==============================================================================
# 4. GLOBAL CONSTANTS & DEFAULTS
# ==============================================================================
# This section defines all the "magic numbers" and default configuration values.
# Grouping them makes the code easier to tune and understand.

# --- Core Audio & Processing Constants ---
SAMPLE_RATE = 16000
CHANNELS = 1
BUFFER_SIZE = 512
TIME_SLEEP = 0.02
INT16_MAX_ABS_VALUE = 32768.0
ALLOWED_LATENCY_LIMIT = 100
# PyAudio format constant (using int to avoid pyaudio import if not needed elsewhere).
AUDIO_FORMAT = 16  # Corresponds to pyaudio.paInt16

# --- Transcription Model Defaults ---
INIT_MODEL_TRANSCRIPTION = "tiny"
INIT_MODEL_TRANSCRIPTION_REALTIME = "tiny"

# --- Real-Time Transcription Behavior ---
INIT_REALTIME_PROCESSING_PAUSE = 0.2
INIT_REALTIME_INITIAL_PAUSE = 0.2

# --- Voice Activity Detection (VAD) Defaults ---
INIT_WEBRTC_SENSITIVITY = 3
INIT_SILERO_SENSITIVITY = 0.4
INIT_POST_SPEECH_SILENCE_DURATION = 0.6

# --- Recording Behavior Defaults ---
INIT_MIN_LENGTH_OF_RECORDING = 0.5
INIT_MIN_GAP_BETWEEN_RECORDINGS = 0
INIT_PRE_RECORDING_BUFFER_DURATION = 1.0

# --- Wake Word Defaults ---
INIT_WAKE_WORDS_SENSITIVITY = 0.6
INIT_WAKE_WORD_ACTIVATION_DELAY = 0.0
INIT_WAKE_WORD_TIMEOUT = 5.0
INIT_WAKE_WORD_BUFFER_DURATION = 0.1

# --- Platform-Specific Defaults ---
INIT_HANDLE_BUFFER_OVERFLOW = platform.system() != 'Darwin'

# --- Speaker Analysis Defaults (Paths and Thresholds) ---
DEFAULT_SPEAKER_MODEL_PATH = "./models/speaker_verification_wespeaker.bin"
DEFAULT_USER_EMBEDDINGS_DIR = "./user_embeddings/"
DEFAULT_SPEAKER_VERIFICATION_THRESHOLD = 0.7

# --- Output & Asset Paths ---
DEFAULT_AUDIO_OUTPUT_DIR_SEGMENTS = "./listen/recorded_speech_segments/"
DEFAULT_BEEP_FILE_PATH = "./beep.wav"

# ==============================================================================
# 5. HELPER FUNCTIONS
# ==============================================================================
# Any standalone functions that support the main classes of the module go here.

def get_wav_info(filepath: str) -> Optional[Tuple[int, int, int]]:
    """
    Gets sample rate, channels, and sample width (in bytes) from a WAV file.
    Returns (sample_rate, channels, sampwidth_bytes) or None if error.
    """
    if not os.path.exists(filepath):
        logger.error(f"WAV file not found for info: {filepath}")
        return None
    try:
        with wave.open(filepath, 'rb') as wf:
            rate = wf.getframerate()
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth() # Sample width in bytes
            logger.debug(f"WAV Info for {filepath}: Rate={rate}, Channels={channels}, SampWidthBytes={sampwidth}")
            return rate, channels, sampwidth
    except wave.Error as e:
        logger.error(f"Error reading WAV file info for {filepath}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error getting WAV info for {filepath}: {e}")
        return None

class bcolors:
    OKGREEN = '\033[92m'  # Green for active speech detection
    WARNING = '\033[93m'  # Yellow for silence detection
    ENDC = '\033[0m'      # Reset to default color

class ListenDriveScriber:
    def __init__(self, conn, stdout_pipe, model_path, download_root, compute_type, gpu_device_index, device,
                 ready_event, shutdown_event, interrupt_stop_event, beam_size, initial_prompt, suppress_tokens,
                 batch_size, faster_whisper_vad_filter, normalize_audio):
        self.conn = conn
        self.stdout_pipe = stdout_pipe
        self.model_path = model_path
        self.download_root = download_root
        self.compute_type = compute_type
        self.gpu_device_index = gpu_device_index
        self.device = device
        self.ready_event = ready_event
        self.shutdown_event = shutdown_event
        self.interrupt_stop_event = interrupt_stop_event
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt
        self.suppress_tokens = suppress_tokens
        self.batch_size = batch_size
        self.faster_whisper_vad_filter = faster_whisper_vad_filter
        self.normalize_audio = normalize_audio
        self.queue = queue.Queue()

    def custom_print(self, *args, **kwargs):
        message = ' '.join(map(str, args))
        try:
            self.stdout_pipe.send(message)
        except (BrokenPipeError, EOFError, OSError):
            pass
    
    def poll_connection(self):
        while not self.shutdown_event.is_set(): # Primary check
            try:
                if self.conn.poll(0.1):  # poll with a timeout
                    if self.shutdown_event.is_set(): # CHECK AGAIN
                        break
                    data = self.conn.recv()
                    self.queue.put(data)
            except (EOFError, BrokenPipeError, OSError) as e:
                if not self.shutdown_event.is_set():
                    logger.error(f"TranscriptionWorker: Pipe error in poll_connection (unexpected): {e}")
                else:
                    logger.debug(f"TranscriptionWorker: Pipe error in poll_connection (expected during shutdown): {e}")
                break # Exit loop on these errors
            except Exception as e:
                if not self.shutdown_event.is_set():
                    logger.error(f"TranscriptionWorker: Unexpected error in poll_connection: {e}", exc_info=True)
                break
        logger.debug("TranscriptionWorker: poll_connection thread finishing.")

    # In class TranscriptionWorker:
    def run(self):
        if __name__ == "__main__": # This check is for when a module is run directly
             pass # Signal handling is better at the start of the process's run method
        # system_signal.signal(system_signal.SIGINT, system_signal.SIG_IGN) # Ignore Ctrl+C in worker
        # __builtins__['print'] = self.custom_print # Re-evaluate

        logger.info(f"Initializing faster_whisper main transcription model {self.model_path}")

        try:
            model = faster_whisper.WhisperModel(
                model_size_or_path=self.model_path,
                device=self.device,
                compute_type=self.compute_type,
                device_index=self.gpu_device_index,
                download_root=self.download_root,
            )
            if self.batch_size > 0:
                model = BatchedInferencePipeline(model=model)

            current_dir = os.path.dirname(os.path.realpath(__file__))
            warmup_audio_path = os.path.join(current_dir, "warmup_audio.wav")
            if os.path.exists(warmup_audio_path):
                warmup_audio_data, _ = sf.read(warmup_audio_path, dtype="float32")
                # The result of transcribe is an iterator of Segment objects and an Info object
                segments_iterator, info_object = model.transcribe(warmup_audio_data, language="en", beam_size=1)
                # Consume the iterator to perform the warmup
                _ = [segment.text for segment in segments_iterator]
            else:
                logger.warning(f"Warmup audio file not found: {warmup_audio_path}. Skipping model warmup.")

        except Exception as e:
            logger.exception(f"Error initializing main faster_whisper transcription model: {e}")
            self.ready_event.set() 
            raise

        self.ready_event.set()
        logger.debug("Faster_whisper main speech to text transcription model initialized successfully")

        polling_thread = threading.Thread(target=self.poll_connection, daemon=True)
        polling_thread.start()

        try:
            while not self.shutdown_event.is_set():
                try:
                    if self.shutdown_event.is_set(): break
                    audio_data_np, language_hint, use_prompt_flag = self.queue.get(timeout=0.1)
                    
                    try:
                        logger.debug(f"Transcribing audio with language {language_hint}")
                        start_t = time.time()

                        if audio_data_np is None or audio_data_np.size == 0:
                            logger.error("Received None or empty audio for transcription")
                            if not self.shutdown_event.is_set(): self.conn.send(('error', "Received None or empty audio for transcription"))
                            continue
                        
                        if self.normalize_audio:
                            peak = np.max(np.abs(audio_data_np))
                            if peak > 0:
                                audio_data_np = (audio_data_np / peak) * 0.95
                        
                        current_prompt = self.initial_prompt if use_prompt_flag and self.initial_prompt else None

                        # faster_whisper.WhisperModel.transcribe returns an iterator and an info object
                        segments_iterator, info = model.transcribe(
                            audio_data_np,
                            language=language_hint if language_hint else None,
                            beam_size=self.beam_size,
                            initial_prompt=current_prompt,
                            suppress_tokens=self.suppress_tokens,
                            vad_filter=self.faster_whisper_vad_filter,
                            # batch_size is handled by BatchedInferencePipeline if used,
                            # or not applicable for direct WhisperModel.transcribe
                            **(dict(batch_size=self.batch_size) if isinstance(model, BatchedInferencePipeline) else {})
                        )
                        
                        # Convert iterator to list of Segment objects to send back
                        list_of_segments = list(segments_iterator)
                        
                        elapsed = time.time() - start_t
                        # For debugging, you can still create the joined string here
                        # transcription_debug_string = " ".join(seg.text for seg in list_of_segments).strip()
                        # logger.debug(f"Final text detected (worker): {transcription_debug_string} in {elapsed:.4f}s")
                        
                        if self.shutdown_event.is_set(): break
                        
                        try:
                            # --- THIS IS THE KEY CHANGE ---
                            self.conn.send(('success', (list_of_segments, info))) 
                            # --- END KEY CHANGE ---
                        except (BrokenPipeError, EOFError, OSError):
                            logger.debug("TranscriptionWorker: Failed to send result, pipe likely closed.")
                            break 
                    except Exception as e:
                        logger.error(f"General error in transcription: {e}", exc_info=True)
                        if not self.shutdown_event.is_set():
                            try: self.conn.send(('error', str(e)))
                            except (BrokenPipeError, EOFError, OSError): pass
                except queue.Empty:
                    continue
                except KeyboardInterrupt: 
                    self.interrupt_stop_event.set()
                    logger.debug("Transcription worker process finished due to KeyboardInterrupt (inferred).")
                    break
                except Exception as e:
                    logger.error(f"General error in TranscriptionWorker processing queue item: {e}", exc_info=True)
        finally:
            logger.debug("TranscriptionWorker: Main run loop finished.")
            self.shutdown_event.set() 

            logger.debug("TranscriptionWorker: Waiting for poll_connection thread to join...")
            polling_thread.join(timeout=1.0) 
            if polling_thread.is_alive():
                logger.warning("TranscriptionWorker: poll_connection thread did not join in time.")

            logger.debug("TranscriptionWorker: Closing its connection pipe.")
            if hasattr(self.conn, 'close') and not getattr(self.conn, 'closed', True): # Check if conn has close and is not closed
                try: self.conn.close()
                except Exception as e: logger.debug(f"TranscriptionWorker: Error closing its connection: {e}")
            
            if hasattr(self.stdout_pipe, 'close') and not getattr(self.stdout_pipe, 'closed', True):
                try: self.stdout_pipe.close()
                except Exception as e: logger.debug(f"TranscriptionWorker: Error closing stdout_pipe: {e}")

            logger.info("TranscriptionWorker process finished.")

class ListenDrive:
    """
    An drive responsible for the full voice processing pipeline: capturing audio,
    detecting voice activity, verifying the speaker, and transcribing the audio.
    """

    def __init__(self,
                 model: str = INIT_MODEL_TRANSCRIPTION,
                 download_root: str = None, 
                 language: str = "",
                 compute_type: str = "default",
                 input_device_index: int = None,
                 gpu_device_index: Union[int, List[int]] = 0,
                 device: str = "cuda",
                 on_recording_start=None,
                 on_recording_stop=None,
                 on_transcription_start=None,
                 ensure_sentence_starting_uppercase=True,
                 ensure_sentence_ends_with_period=True,
                 use_microphone=True,
                 spinner=True,
                 batch_size: int = 16,
                 

                 # Realtime transcription parameters
                 enable_realtime_transcription=False,
                 use_main_model_for_realtime=False,
                 realtime_model_type=INIT_MODEL_TRANSCRIPTION_REALTIME,
                 realtime_processing_pause=INIT_REALTIME_PROCESSING_PAUSE,
                 init_realtime_after_seconds=INIT_REALTIME_INITIAL_PAUSE,
                 on_realtime_transcription_update=None,
                 on_realtime_transcription_stabilized=None,
                 realtime_batch_size: int = 16,

                 # Voice activation parameters
                 silero_sensitivity: float = INIT_SILERO_SENSITIVITY,
                 silero_use_onnx: bool = False,
                 silero_deactivity_detection: bool = False,
                 webrtc_sensitivity: int = INIT_WEBRTC_SENSITIVITY,
                 post_speech_silence_duration: float = (
                     INIT_POST_SPEECH_SILENCE_DURATION
                 ),
                 min_length_of_recording: float = (
                     INIT_MIN_LENGTH_OF_RECORDING
                 ),
                 min_gap_between_recordings: float = (
                     INIT_MIN_GAP_BETWEEN_RECORDINGS
                 ),
                 pre_recording_buffer_duration: float = (
                     INIT_PRE_RECORDING_BUFFER_DURATION
                 ),
                 on_vad_start=None,
                 on_vad_stop=None,
                 on_vad_detect_start=None,
                 on_vad_detect_stop=None,
                 on_turn_detection_start=None,
                 on_turn_detection_stop=None,

                 min_snr_threshold: float = 10.0, # <--- NEW PARAM (dB)

                 # Wake word parameters
                 wakeword_backend: str = "",
                 openwakeword_model_paths: str = None,
                 openwakeword_inference_framework: str = "onnx",
                 wake_words: str = "",
                 wake_words_sensitivity: float = INIT_WAKE_WORDS_SENSITIVITY,
                #  wake_word_activation_delay: float = (
                #     INIT_WAKE_WORD_ACTIVATION_DELAY
                #  ),

                 wake_word_timeout: float = INIT_WAKE_WORD_TIMEOUT,
                 wake_word_buffer_duration: float = INIT_WAKE_WORD_BUFFER_DURATION,
                 on_wakeword_detected=None,
                 on_wakeword_timeout=None,
                 on_wakeword_detection_start=None,
                 on_wakeword_detection_end=None,
                 on_recorded_chunk=None,
                 debug_mode=False,
                 handle_buffer_overflow: bool = INIT_HANDLE_BUFFER_OVERFLOW,
                 beam_size: int = 5,
                 beam_size_realtime: int = 3,
                 buffer_size: int = BUFFER_SIZE,
                 sample_rate: int = SAMPLE_RATE,
                 initial_prompt: Optional[Union[str, Iterable[int]]] = None,
                 initial_prompt_realtime: Optional[Union[str, Iterable[int]]] = None,
                 suppress_tokens: Optional[List[int]] = [-1],
                 print_transcription_time: bool = False,
                 early_transcription_on_silence: int = 0,
                 allowed_latency_limit: int = ALLOWED_LATENCY_LIMIT,
                 no_log_file: bool = False,
                 use_extended_logger: bool = False,
                 faster_whisper_vad_filter: bool = True,
                 normalize_audio: bool = False,
                 on_amplitude_update=None,
                 start_callback_in_new_thread: bool = False,
                 # --- New Parameters for Speaker Verification ---
                 enable_speaker_verification: bool = True,
                 speaker_model_path: str = DEFAULT_SPEAKER_MODEL_PATH,
                 diarization_pipeline_path: str = "pyannote/speaker-diarization-3.1", # Add a default
                 user_embeddings_dir: str = DEFAULT_USER_EMBEDDINGS_DIR,
                 speaker_verification_threshold: float = DEFAULT_SPEAKER_VERIFICATION_THRESHOLD,
                 audio_segment_output_dir: str = DEFAULT_AUDIO_OUTPUT_DIR_SEGMENTS,
                 save_verified_segments: bool = False,
                 play_verification_beep: bool = True,
                 beep_file_path: str = DEFAULT_BEEP_FILE_PATH,
                 on_segment_processed_callback = None,
                 on_final_transcription_for_service_callback = None,
                 preloaded_user_embeddings: Optional[dict] = None,  
                 channels: int = CHANNELS # Parameter for channels       
                 ):
 
        """
        Initializes the ListenDrive engine with all its components.

        Args:
        -- Transcription Parameters --
        - model (str): Name of the `faster-whisper` model to use (e.g., 'tiny.en', 'base').
        - language (str): Language code for transcription (e.g., 'en', 'es'). Autodetects if empty.
        - compute_type (str): The quantization type for the model (e.g., 'float16', 'int8').
        - device (str): The device to run the model on, 'cuda' or 'cpu'.
        - download_root (str): A custom path to store downloaded Whisper models.
        - beam_size (int): Beam size for the final transcription decoding.
        - batch_size (int): Batch size for the main transcription model.
        - initial_prompt (str|Iterable[int]): An initial prompt to guide the transcription model.
        - suppress_tokens (List[int]): A list of token IDs to suppress during transcription.
        - faster_whisper_vad_filter (bool): If True, uses the VAD filter from `faster-whisper` for additional
          speech detection during the transcription process.
        - print_transcription_time (bool): If True, prints the processing time for each final transcription.

        -- Real-Time Transcription Parameters --
        - enable_realtime_transcription (bool): If True, enables the live transcription preview.
        - realtime_model_type (str): The `faster-whisper` model to use for real-time processing.
        - use_main_model_for_realtime (bool): If True, uses the main model for real-time tasks.
        - realtime_processing_pause (float): Seconds to wait between real-time transcription updates.
        - init_realtime_after_seconds (float): Initial delay before the first real-time preview is shown.
        - beam_size_realtime (int): Beam size for the real-time transcription decoding.
        - realtime_batch_size (int): Batch size for the real-time transcription model.
        - initial_prompt_realtime (str|Iterable[int]): An initial prompt for the real-time model.

        -- Speaker Analysis Parameters --
        - enable_speaker_verification (bool): If True, enables speaker analysis features.
        - speaker_model_path (str): Path to the `wespeaker` speaker embedding model (`.bin` file).
        - diarization_pipeline_path (str): Path to a local `pyannote.audio` speaker diarization 
          pipeline configuration (`.yaml` file).
        - user_embeddings_dir (str): Path to a directory containing pre-computed user voice embeddings.
        - preloaded_user_embeddings (dict): A dictionary of user embeddings to load directly.
        - speaker_verification_threshold (float): Cosine distance threshold for a speaker match.
        
        -- Voice Activity Detection (VAD) Parameters --
        - webrtc_sensitivity (int): VAD sensitivity for `webrtcvad` (0-3). 3 is most aggressive.
        - silero_sensitivity (float): VAD sensitivity for `silero-vad` (0.0-1.0). 1 is most sensitive.
        - silero_use_onnx (bool): If True, uses the ONNX version of the Silero VAD model for speed.
        - silero_deactivity_detection (bool): If True, uses Silero VAD for end-of-speech detection.
        - post_speech_silence_duration (float): Seconds of silence to wait after speech before stopping.
        - pre_recording_buffer_duration (float): Seconds of audio to keep before speech starts.
        - min_length_of_recording (float): Minimum duration in seconds for a valid recording.
        - min_gap_between_recordings (float): Minimum seconds of silence between two separate recordings.
        - early_transcription_on_silence (int): Milliseconds of silence before triggering an early
          (and potentially discarded) final transcription for faster results.

        -- Wake Word Parameters --
        - wakeword_backend (str): The wake word engine to use, 'pvporcupine' or 'openwakeword'.
        - wake_words (str): Comma-separated list of wake words for Porcupine.
        - wake_words_sensitivity (float): Sensitivity for wake word detection (0.0-1.0).
        - openwakeword_model_paths (str): Comma-separated paths to custom openwakeword models.
        - openwakeword_inference_framework (str): Inference framework for oww, 'onnx' or 'tflite'.
        - wake_word_activation_delay (float): Seconds to wait before activating wake word detection.
        - wake_word_timeout (float): Seconds to listen for speech after a wake word is detected.
        - wake_word_buffer_duration (float): Seconds of audio to trim from the start of a recording
          after a wake word is detected.
        
        -- Audio & System Parameters --
        - input_device_index (int): The index of the microphone to use.
        - gpu_device_index (int|List[int]): The GPU device index/indices to use.
        - sample_rate (int): The audio sample rate. Must match model requirements (e.g., 16000).
        - buffer_size (int): The audio buffer size in samples.
        - channels (int): The number of audio channels (e.g., 1 for mono).
        - use_microphone (bool): If False, the engine will process audio fed via the `feed_audio` method.
        - handle_buffer_overflow (bool): If True, logs and discards overflowing audio chunks.
        - allowed_latency_limit (int): Max number of chunks to queue before discarding.
        - normalize_audio (bool): If True, normalizes audio volume before transcription.

        -- Callbacks & Event Handlers --
        - on_recording_start, on_recording_stop: Called when recording starts or stops.
        - on_transcription_start: Called when the final transcription task begins.
        - on_realtime_transcription_update, on_realtime_transcription_stabilized: For live text.
        - on_vad_start, on_vad_stop: Called when speech activity starts or stops.
        - on_vad_detect_start, on_vad_detect_stop: Called when the VAD listener starts or stops.
        - on_turn_detection_start, on_turn_detection_stop: Called when listening for end-of-turn.
        - on_wakeword_detected, on_wakeword_timeout: Wake word event callbacks.
        - on_wakeword_detection_start, on_wakeword_detection_end: Wake word listener callbacks.
        - on_recorded_chunk: Called with every raw audio chunk recorded.
        - on_amplitude_update: Called frequently with the real-time RMS amplitude of the audio.
        - on_segment_processed_callback: Called after speaker analysis is complete for a segment.
        - on_final_transcription_for_service_callback: Called with the final verified text and metadata.
        - start_callback_in_new_thread (bool): If True, all callbacks will be run in a separate thread.

        -- Output & Logger --
        - audio_segment_output_dir (str): Directory to save verified audio segments.
        - save_verified_segments (bool): If True, saves a .wav file for each verified segment.
        - play_verification_beep (bool): If True, plays a sound on successful speaker verification.
        - beep_file_path (str): Path to the .wav file for the verification beep.
        - spinner (bool): If True, displays a console spinner indicating the current state.
        - debug_mode (bool): Enables extra verbose logger.
        - level: The logger level for the console.
        - no_log_file (bool): If True, disables writing to a log file.
        - use_extended_logger (bool): Enables extremely verbose logger for debugging.
        - ensure_sentence_starting_uppercase (bool): If True, capitalizes the first letter of sentences.
        - ensure_sentence_ends_with_period (bool): If True, adds a period to sentences if needed.

        Raises:
            Exception: On errors related to initializing models, audio devices, or wake word engines.
        """

        self.language = language
        self.compute_type = compute_type
        self.input_device_index = input_device_index
        self.gpu_device_index = gpu_device_index
        self.device = device
        self.wake_words = wake_words
        # self.wake_word_activation_delay = wake_word_activation_delay
        self.min_snr_threshold = min_snr_threshold # <--- NEW ASSIGNMENT
        self.wake_word_timeout = wake_word_timeout
        self.wake_word_buffer_duration = wake_word_buffer_duration
        self.ensure_sentence_starting_uppercase = (
            ensure_sentence_starting_uppercase
        )
        self.ensure_sentence_ends_with_period = (
            ensure_sentence_ends_with_period
        )
        self.use_microphone = mp.Value(c_bool, use_microphone)
        self.min_gap_between_recordings = min_gap_between_recordings
        self.min_length_of_recording = min_length_of_recording
        self.pre_recording_buffer_duration = pre_recording_buffer_duration
        self.post_speech_silence_duration = post_speech_silence_duration
        self.on_recording_start = on_recording_start
        self.on_recording_stop = on_recording_stop
        self.on_wakeword_detected = on_wakeword_detected
        self.on_wakeword_timeout = on_wakeword_timeout
        self.on_vad_start = on_vad_start
        self.on_vad_stop = on_vad_stop
        self.on_vad_detect_start = on_vad_detect_start
        self.on_vad_detect_stop = on_vad_detect_stop
        self.on_turn_detection_start = on_turn_detection_start
        self.on_turn_detection_stop = on_turn_detection_stop
        self.on_wakeword_detection_start = on_wakeword_detection_start
        self.on_wakeword_detection_end = on_wakeword_detection_end
        self.on_recorded_chunk = on_recorded_chunk
        self.on_transcription_start = on_transcription_start
        self.enable_realtime_transcription = enable_realtime_transcription
        self.use_main_model_for_realtime = use_main_model_for_realtime
        self.main_model_type = model
        if not download_root:
            download_root = None
        self.download_root = download_root
        self.realtime_model_type = realtime_model_type
        self.realtime_processing_pause = realtime_processing_pause
        self.init_realtime_after_seconds = init_realtime_after_seconds
        self.on_realtime_transcription_update = (
            on_realtime_transcription_update
        )
        self.on_realtime_transcription_stabilized = (
            on_realtime_transcription_stabilized
        )
        self.debug_mode = debug_mode
        self.handle_buffer_overflow = handle_buffer_overflow
        self.beam_size = beam_size
        self.beam_size_realtime = beam_size_realtime
        self.allowed_latency_limit = allowed_latency_limit
        self.batch_size = batch_size
        self.realtime_batch_size = realtime_batch_size

        self.vad_perf_log = []  # <--- Add this line to store VAD execution times

        self.audio_queue = mp.Queue()
        self.amplitude_queue = mp.Queue() # New queue for RMS amplitude 
        self.buffer_size = buffer_size
        self.sample_rate = sample_rate
        self.recording_start_time = 0
        self.recording_stop_time = 0
        self.last_recording_start_time = 0
        self.last_recording_stop_time = 0
        self.wake_word_detect_time = 0
        self.silero_check_time = 0
        self.silero_working = False
        self.speech_end_silence_start = 0
        # --- NEW: Add a dedicated timer for the wakeword buffer ---
        self.wakeword_buffer_silence_start = 0
        self.silero_sensitivity = silero_sensitivity
        self.silero_deactivity_detection = silero_deactivity_detection
        self.listen_start = 0
        self.spinner = spinner
        self.halo = None
        self.state = "inactive"
        self.wakeword_detected = False
        self.wakeword_detection_flag = False

        # --- NEW: Attributes for Smart Wake Word Buffering ---
        self.is_buffering_for_wakeword = False
        self.potential_command_buffer = []

        # --- NEW: Attributes for Context-Aware State Control ---
        self.last_speaker_count = 0  # Remembers the number of speakers from the last segment.
        self.mode_override = None    # Allows an external API call to lock the mode.
        # --- NEW: Attribute for Context-Aware State Control ---
        self.last_speaker_count = 0  # Remembers the number of speakers from the last segment.
        self.text_storage = []
        self.realtime_stabilized_text = ""
        self.realtime_stabilized_safetext = ""
        self.is_webrtc_speech_active = False
        self.is_silero_speech_active = False
        self.recording_thread = None
        self.realtime_thread = None
        self.audio_interface = None
        self.audio = None
        self.stream = None
        self.start_recording_event = threading.Event()
        self.stop_recording_event = threading.Event()
        self.backdate_stop_seconds = 0.0
        self.backdate_resume_seconds = 0.0
        self.last_transcription_bytes = None
        self.last_transcription_bytes_b64 = None
        self.initial_prompt = initial_prompt
        self.initial_prompt_realtime = initial_prompt_realtime
        self.suppress_tokens = suppress_tokens
        self.use_wake_words = wake_words or wakeword_backend in {'oww', 'openwakeword', 'openwakewords'}
        self.detected_language = None
        self.detected_language_probability = 0
        self.detected_realtime_language = None
        self.detected_realtime_language_probability = 0
        self.transcription_lock = threading.Lock()
        self.shutdown_lock = threading.Lock()
        self.transcribe_count = 0
        self.print_transcription_time = print_transcription_time
        self.early_transcription_on_silence = early_transcription_on_silence
        self.use_extended_logger = use_extended_logger
        self.faster_whisper_vad_filter = faster_whisper_vad_filter
        self.normalize_audio = normalize_audio
        self.awaiting_speech_end = False
        self.start_callback_in_new_thread = start_callback_in_new_thread
        self.on_amplitude_update = on_amplitude_update 
        # --- Speaker Verification Attributes ---
        self.enable_speaker_verification = enable_speaker_verification
        self.speaker_model_path = speaker_model_path
        self.diarization_pipeline_path = diarization_pipeline_path
        self.user_embeddings_dir = user_embeddings_dir
        self.speaker_verification_threshold = speaker_verification_threshold
        self.audio_segment_output_dir = audio_segment_output_dir
        self.save_verified_segments = save_verified_segments
        self.play_verification_beep = play_verification_beep
        self.beep_file_path = beep_file_path
        self.on_segment_processed_callback = on_segment_processed_callback
        self.on_final_transcription_for_service_callback_ref = on_final_transcription_for_service_callback

        self.speaker_model_instance = None # Renamed from self.speaker_model to avoid conflict
        self.speaker_inference_instance = None # Renamed
        self.diarization_pipeline_instance = None
        self.preloaded_user_embeddings = preloaded_user_embeddings # <<< STORE IT
        self.user_embeddings = {}
        self.beep_sound = None
        self.current_segment_raw_bytes: Optional[bytes] = None
        self.channels = channels # <<< ADD THIS LINE to store channels
        

        self.is_shut_down = False
        self.shutdown_event = mp.Event()
        
        try:
            # Only set the start method if it hasn't been set already
            if mp.get_start_method(allow_none=True) is None:
                mp.set_start_method("spawn")
        except RuntimeError as e:
            logger.info(f"Start method has already been set. Details: {e}")

        logger.info("Starting Listen Drive")

        if use_extended_logger:
            logger.info("Listen Drive was called with these parameters:")
            for param, value in locals().items():
                logger.info(f"{param}: {value}")

        self.interrupt_stop_event = mp.Event()
        self.was_interrupted = mp.Event()
        self.main_transcription_ready_event = mp.Event()

        self.parent_transcription_pipe, child_transcription_pipe = SafePipe()
        self.parent_stdout_pipe, child_stdout_pipe = SafePipe()

        # Set device for model
        self.device = "cuda" if self.device == "cuda" and torch.cuda.is_available() else "cpu"

        self.transcript_process = self._start_thread(
            target=ListenDrive._transcription_worker,
            args=(
                child_transcription_pipe,
                child_stdout_pipe,
                self.main_model_type,
                self.download_root,
                self.compute_type,
                self.gpu_device_index,
                self.device,
                self.main_transcription_ready_event,
                self.shutdown_event,
                self.interrupt_stop_event,
                self.beam_size,
                self.initial_prompt,
                self.suppress_tokens,
                self.batch_size,
                self.faster_whisper_vad_filter,
                self.normalize_audio,
            )
        )

        # Start audio data reading process
        if self.use_microphone.value:
            logger.info("Initializing audio recording"
                         " (creating pyAudio input stream,"
                         f" sample rate: {self.sample_rate}"
                         f" buffer size: {self.buffer_size}"
                         )
            self.reader_process = self._start_thread(
                target=ListenDrive._audio_data_worker,
                args=(
                    self.audio_queue,
                    self.amplitude_queue,
                    self.sample_rate,
                    self.buffer_size,
                    self.input_device_index,
                    self.shutdown_event,
                    self.interrupt_stop_event,
                    self.use_microphone
                )
            )

        # Initialize the realtime transcription model
        if self.enable_realtime_transcription and not self.use_main_model_for_realtime:
            try:
                logger.info("Initializing faster_whisper realtime "
                             f"transcription model {self.realtime_model_type}, "
                             f"default device: {self.device}, "
                             f"compute type: {self.compute_type}, "
                             f"device index: {self.gpu_device_index}, "
                             f"download root: {self.download_root}"
                             )
                self.realtime_model_type = faster_whisper.WhisperModel(
                    model_size_or_path=self.realtime_model_type,
                    device=self.device,
                    compute_type=self.compute_type,
                    device_index=self.gpu_device_index,
                    download_root=self.download_root,
                )
                if self.realtime_batch_size > 0:
                    self.realtime_model_type = BatchedInferencePipeline(model=self.realtime_model_type)

                # Run a warm-up transcription
                current_dir = os.path.dirname(os.path.realpath(__file__))
                warmup_audio_path = os.path.join(
                    current_dir, "warmup_audio.wav"
                )
                warmup_audio_data, _ = sf.read(warmup_audio_path, dtype="float32")
                segments, info = self.realtime_model_type.transcribe(warmup_audio_data, language="en", beam_size=1)
                model_warmup_transcription = " ".join(segment.text for segment in segments)
            except Exception as e:
                logger.exception("Error initializing faster_whisper "
                                  f"realtime transcription model: {e}"
                                  )
                raise

            logger.debug("Faster_whisper realtime speech to text "
                          "transcription model initialized successfully")

        # --- Load Speaker Verification Resources ---
        if self.enable_speaker_verification:
            self._load_speaker_verification_resources() # Call new method
            if self.play_verification_beep:
                self._initialize_beep_sound() # Call new method
            if self.save_verified_segments:
                try:
                    os.makedirs(self.audio_segment_output_dir, exist_ok=True)
                    if not os.access(self.audio_segment_output_dir, os.W_OK):
                        logger.error(f"No write permission for {self.audio_segment_output_dir}. Disabling segment saving.")
                        self.save_verified_segments = False
                except Exception as e_dir:
                    logger.error(f"Could not create segment output dir {self.audio_segment_output_dir}: {e_dir}. Disabling segment saving.")
                    self.save_verified_segments = False
        # --- End Load Speaker Verification ---

        # Setup wake word detection
        if wake_words or wakeword_backend in {'oww', 'openwakeword', 'openwakewords', 'pvp', 'pvporcupine'}:
            self.wakeword_backend = wakeword_backend

            self.wake_words_list = [
                word.strip() for word in wake_words.lower().split(',')
            ]
            self.wake_words_sensitivity = wake_words_sensitivity
            self.wake_words_sensitivities = [
                float(wake_words_sensitivity)
                for _ in range(len(self.wake_words_list))
            ]

            if wake_words and self.wakeword_backend in {'pvp', 'pvporcupine'}:

                try:
                    self.porcupine = pvporcupine.create(
                        keywords=self.wake_words_list,
                        sensitivities=self.wake_words_sensitivities
                    )
                    self.buffer_size = self.porcupine.frame_length
                    self.sample_rate = self.porcupine.sample_rate

                except Exception as e:
                    logger.exception(
                        "Error initializing porcupine "
                        f"wake word detection engine: {e}. "
                        f"Wakewords: {self.wake_words_list}."
                    )
                    raise

                logger.debug(
                    "Porcupine wake word detection engine initialized successfully"
                )

            elif self.wakeword_backend in {'oww', 'openwakeword', 'openwakewords'}:
                    
                openwakeword.utils.download_models()

                try:
                    if openwakeword_model_paths:
                        model_paths = openwakeword_model_paths.split(',')
                        self.owwModel = OpenWakeWordModel(
                            wakeword_models=model_paths,
                            inference_framework=openwakeword_inference_framework
                        )
                        logger.info(
                            "Successfully loaded wakeword model(s): "
                            f"{openwakeword_model_paths}"
                        )
                    else:
                        self.owwModel = OpenWakeWordModel(
                            inference_framework=openwakeword_inference_framework)
                    
                    self.oww_n_models = len(self.owwModel.models.keys())
                    if not self.oww_n_models:
                        logger.error(
                            "No wake word models loaded."
                        )

                    for model_key in self.owwModel.models.keys():
                        logger.info(
                            "Successfully loaded openwakeword model: "
                            f"{model_key}"
                        )

                except Exception as e:
                    logger.exception(
                        "Error initializing openwakeword "
                        f"wake word detection engine: {e}"
                    )
                    raise

                logger.debug(
                    "Open wake word detection engine initialized successfully"
                )
            
            else:
                logger.exception(f"Wakeword engine {self.wakeword_backend} unknown/unsupported or wake_words not specified. Please specify one of: pvporcupine, openwakeword.")


        # Setup voice activity detection model WebRTC
        try:
            logger.info("Initializing WebRTC voice with "
                         f"Sensitivity {webrtc_sensitivity}"
                         )
            self.webrtc_vad_model = webrtcvad.Vad()
            self.webrtc_vad_model.set_mode(webrtc_sensitivity)

        except Exception as e:
            logger.exception("Error initializing WebRTC voice "
                              f"activity detection engine: {e}"
                              )
            raise

        logger.debug("WebRTC VAD voice activity detection "
                      "engine initialized successfully"
                      )

        # Setup voice activity detection model Silero VAD
        try:
            self.silero_vad_model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                verbose=False,
                onnx=silero_use_onnx
            )

        except Exception as e:
            logger.exception(f"Error initializing Silero VAD "
                              f"voice activity detection engine: {e}"
                              )
            raise

        logger.debug("Silero VAD voice activity detection "
                      "engine initialized successfully"
                      )

        self.audio_buffer = collections.deque(
            maxlen=int((self.sample_rate // self.buffer_size) *
                       self.pre_recording_buffer_duration)
        )
        self.last_words_buffer = collections.deque(
            maxlen=int((self.sample_rate // self.buffer_size) *
                       0.3)
        )
        self.frames = []
        self.last_frames = []

        # Recording control flags
        self.is_recording = False
        self.is_running = True
        self.start_recording_on_voice_activity = False
        self.stop_recording_on_voice_deactivity = False

        # Start the recording worker thread
        self.recording_thread = threading.Thread(target=self._recording_worker)
        self.recording_thread.daemon = True
        self.recording_thread.start()

        # Start the realtime transcription worker thread
        self.realtime_thread = threading.Thread(target=self._realtime_worker)
        self.realtime_thread.daemon = True
        self.realtime_thread.start()
                   
        # Wait for transcription models to start
        logger.debug('Waiting for main transcription model to start')
        self.main_transcription_ready_event.wait()
        logger.debug('Main transcription model ready')

        self.stdout_thread = threading.Thread(target=self._read_stdout)
        self.stdout_thread.daemon = True
        self.stdout_thread.start()
        

        # --- ADDED AMPLITUDE READER THREAD START ---
        self.amplitude_reader_thread = None
        if self.on_amplitude_update:
            self.amplitude_reader_thread = threading.Thread(target=self._amplitude_reader_worker, daemon=True)
            self.amplitude_reader_thread.start()
        self.recording_thread = threading.Thread(target=self._recording_worker)
        # -----------------------------------------

        logger.debug('Initialization completed successfully')

    # In class AudioToTextRecorder:
    def _load_speaker_verification_resources(self):
        """Loads speaker verification model, overlap pipeline, and user embeddings."""
        logger.info("Loading speaker verification resources...")
        if not self.enable_speaker_verification:
            logger.info("Speaker verification is disabled by configuration.")
            return
        try:
            # Load Speaker Verification Model (as before)
            if not os.path.exists(self.speaker_model_path):
                raise FileNotFoundError(f"Speaker verification model not found: {self.speaker_model_path}")
            self.speaker_model_instance = PyannoteModel.from_pretrained(self.speaker_model_path)
            self.speaker_inference_instance = PyannoteInference(self.speaker_model_instance, window="whole")
            self.speaker_model_instance.eval()
            logger.info("Speaker verification model loaded.")

            # Load the diarization pipeline
            self.diarization_pipeline_instance = PyannotePipeline.from_pretrained(self.diarization_pipeline_path)
            logger.info("Speaker diarization pipeline loaded.")

            # --- Load User Embeddings ---
            if self.preloaded_user_embeddings and isinstance(self.preloaded_user_embeddings, dict):
                self.user_embeddings = copy.deepcopy(self.preloaded_user_embeddings) # Use a copy
                logger.info(f"Using {len(self.user_embeddings)} preloaded user embeddings.")
                # Basic validation of preloaded structure
                for username, data in self.user_embeddings.items():
                    if not isinstance(data, dict) or "embedding" not in data or "hashid" not in data:
                        logger.error(f"Invalid structure for preloaded embedding for user '{username}'. Check format.")
                        # Optionally remove invalid entry: del self.user_embeddings[username]
                    elif not isinstance(data["embedding"], np.ndarray):
                        logger.error(f"Preloaded embedding for user '{username}' is not a NumPy array.")
            
            elif self.user_embeddings_dir and os.path.isdir(self.user_embeddings_dir):
                logger.info(f"Loading user embeddings from directory: {self.user_embeddings_dir}")
                for item_name in os.listdir(self.user_embeddings_dir):
                    if item_name.endswith(".npy"):
                        username = os.path.splitext(item_name)[0]
                        embedding_path = os.path.join(self.user_embeddings_dir, item_name)
                        hashid_path = os.path.join(self.user_embeddings_dir, f"{username}.hashid.txt")

                        try:
                            embedding = np.load(embedding_path)
                            hashid = f"hashid_{username}" # Default if file not found
                            if os.path.exists(hashid_path):
                                with open(hashid_path, 'r') as hf:
                                    hashid_from_file = hf.read().strip()
                                    if hashid_from_file: hashid = hashid_from_file
                            
                            self.user_embeddings[username] = {"embedding": embedding, "hashid": hashid}
                            logger.info(f"Loaded embedding for user: {username} (Hash: {hashid})")
                        except Exception as e:
                            logger.error(f"Failed to load embedding/hashid for {username} from directory: {e}")
            else:
                logger.warning("Neither preloaded embeddings nor a valid user_embeddings_dir provided. No users for speaker verification.")
            
            if self.user_embeddings:
                logger.info(f"Total {len(self.user_embeddings)} user embeddings configured.")


        except Exception as e:
            logger.error(f"Failed to load speaker verification resources: {e}", exc_info=True)
            self.enable_speaker_verification = False # Disable if loading fails    

    def _initialize_beep_sound(self):
        if not self.play_verification_beep:
            logger.info("Beep sound playback is disabled by configuration.")
            return

        if not os.path.exists(self.beep_file_path):
            logger.error(f"Beep sound file not found: {self.beep_file_path}")
            self.play_verification_beep = False # Disable if file not found
            return

        # Get beep file's actual parameters
        beep_info = get_wav_info(self.beep_file_path) 
        # OR: beep_info_sf = get_wav_info_sf(self.beep_file_path)
        # if using soundfile, you'd then map subtype to pygame's size parameter

        if not beep_info:
            logger.error(f"Could not get info for beep file {self.beep_file_path}. Cannot initialize mixer correctly.")
            self.play_verification_beep = False
            return

        beep_rate, beep_channels, beep_sampwidth_bytes = beep_info
        
        # Pygame's 'size' parameter:
        # -16 for 16-bit signed, 16 for 16-bit unsigned
        # -8 for 8-bit signed, 8 for 8-bit unsigned
        # We assume signed for PCM WAVs.
        pygame_size = 0
        if beep_sampwidth_bytes == 2: # 16-bit
            pygame_size = -16
        elif beep_sampwidth_bytes == 1: # 8-bit
            pygame_size = -8 
        else:
            logger.error(f"Unsupported beep file sample width: {beep_sampwidth_bytes*8}-bit. Pygame supports 8-bit or 16-bit.")
            self.play_verification_beep = False
            return

        try:
            # Quit mixer if it was already initialized with different parameters
            if pygame.mixer.get_init():
                current_freq, current_format, current_channels = pygame.mixer.get_init()
                # Pygame format is bit_depth (positive for unsigned, negative for signed)
                # We need to compare absolute value and sign for bit depth.
                # And channels.
                is_different = False
                if current_freq != beep_rate: is_different = True
                if abs(current_format) != beep_sampwidth_bytes * 8: is_different = True
                if (current_format < 0) != (pygame_size < 0) : is_different = True # Sign mismatch
                if current_channels != beep_channels: is_different = True
                
                if is_different:
                    logger.info(f"Pygame mixer was initialized with ({current_freq}, {current_format}, {current_channels}). Quitting to re-init for beep file ({beep_rate}, {pygame_size}, {beep_channels}).")
                    pygame.mixer.quit()
            
            # Initialize mixer with parameters matching the beep file
            logger.info(f"Initializing Pygame mixer for beep: Freq={beep_rate}, Size={pygame_size}, Channels={beep_channels}")
            pygame.mixer.init(frequency=beep_rate, size=pygame_size, channels=beep_channels, buffer=512) # Added buffer
            
            self.beep_sound = pygame.mixer.Sound(self.beep_file_path)
            self.beep_sound.set_volume(1.0)
            logger.info(f"Beep sound '{self.beep_file_path}' loaded successfully.")

        except pygame.error as pg_error: # Catch Pygame specific errors
            logger.error(f"Pygame error initializing mixer or loading beep sound: {pg_error}", exc_info=True)
            self.play_verification_beep = False
            if pygame.mixer.get_init(): pygame.mixer.quit() # Attempt to clean up
        except Exception as e:
            logger.error(f"Unexpected error initializing pygame mixer or beep sound: {e}", exc_info=True)
            self.play_verification_beep = False
            if pygame.mixer.get_init(): pygame.mixer.quit() # Attempt to clean up

    def _play_beep_thread_target(self):
        if self.beep_sound and self.play_verification_beep and pygame.mixer.get_init():
            try:
                self.beep_sound.play()
                time.sleep(self.beep_sound.get_length() + 0.1) # Wait for sound to finish + small buffer
            except Exception as e:
                logger.error(f"Error playing beep sound in thread: {e}")

    def _audio_segment_to_wav_bytes(self, audio_data_bytes: bytes, sample_rate: int, 
                                   save_to_file: bool = False, filename_prefix="segment") -> Tuple[Optional[io.BytesIO], Optional[str]]:
        if not audio_data_bytes:
            logger.warning("Empty audio data bytes for WAV conversion.")
            return None, None
        
        saved_filepath = None
        try:
            wav_io = io.BytesIO()
            with wave.open(wav_io, 'wb') as wf:
                wf.setnchannels(self.channels) 
                # Determine sample width from self.audio_format (default pyaudio.paInt16 -> 2 bytes)
                sample_width = 2 # Default for paInt16
                # Add more cases if other formats are supported by AUDIO_FORMAT constant
                # For example: if self.audio_format == pyaudio.paInt8: sample_width = 1
                wf.setsampwidth(sample_width) 
                wf.setframerate(sample_rate)
                wf.writeframes(audio_data_bytes)
            wav_io.seek(0)

            if save_to_file and self.audio_segment_output_dir:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = f"{filename_prefix}_{timestamp}.wav"
                saved_filepath = os.path.join(self.audio_segment_output_dir, filename)
                try:
                    with open(saved_filepath, 'wb') as f_disk:
                        f_disk.write(wav_io.getvalue()) # Write the content of BytesIO
                    logger.info(f"Audio segment saved: {saved_filepath}")
                except Exception as e:
                    logger.error(f"Failed to save audio segment to {saved_filepath}: {e}")
                    saved_filepath = None 
            
            return wav_io, saved_filepath
        except Exception as e:
            logger.error(f"Error converting audio to WAV: {e}", exc_info=True)
            return None, None

    def _calculate_snr(self, audio_data: np.ndarray) -> float:
            """
            Calculates the Signal-to-Noise Ratio (SNR) in dB using a percentile-based approach.
            Assumes audio_data is a float32 array, typically normalized.
            """
            if audio_data is None or len(audio_data) == 0:
                return 0.0

            # Frame length for energy calculation (approx 30ms at 16kHz)
            frame_length = 512
            num_frames = len(audio_data) // frame_length
            
            if num_frames == 0: 
                return 0.0

            # Reshape to frames and calculate energy (RMS squared)
            frames = audio_data[:num_frames * frame_length].reshape(num_frames, frame_length)
            frame_energies = np.mean(frames ** 2, axis=1)

            # Estimate Signal Power (top 95th percentile) and Noise Power (bottom 1st percentile)
            # Using 1st percentile helps find true silence floor even in tight recordings
            signal_power = np.percentile(frame_energies, 95)
            noise_power = np.percentile(frame_energies, 1)

            # Prevent divide by zero
            if noise_power <= 1e-12: 
                return 100.0 
            
            snr = 10 * np.log10(signal_power / noise_power)
            return float(snr)

    def _diarize_and_analyze_segment(self, wav_data_io: io.BytesIO):
        """
        Performs speaker diarization.
        Returns: (has_overlap, speaker_count, diarization_result)
        """
        if not wav_data_io or not self.diarization_pipeline_instance or not self.enable_speaker_verification:
            return (False, 0, None)

        try:
            wav_data_io.seek(0)
            # Perform diarization
            diarization = self.diarization_pipeline_instance(wav_data_io)

            speaker_labels = diarization.labels()
            speaker_count = len(speaker_labels)
            
            # Check for overlap (if two distinct speakers speak at the same time)
            # Pyannote has a specific method for finding overlap, 
            # but checking speaker count > 1 is a decent proxy for "multiple people present"
            has_overlap = speaker_count > 1
            
            if has_overlap:
                logger.debug(f"Diarization: {speaker_count} speakers found. Labels: {speaker_labels}")
                
            return (has_overlap, speaker_count, diarization) # <<< RETURN THE OBJECT

        except Exception as e:
            logger.error(f"Error during speaker diarization: {e}", exc_info=True)
            return (False, 0, None)

    def _verify_speaker_segment(self, wav_data_io: io.BytesIO) -> Tuple[Optional[str], Optional[str], Optional[float]]:
        if not wav_data_io or not self.speaker_inference_instance or not self.user_embeddings or not self.enable_speaker_verification:
            return None, None, None
        
        try:
            wav_data_io.seek(0)
            embedding = self.speaker_inference_instance(wav_data_io)
            if embedding is None or len(embedding) == 0:
                 logger.warning("Speaker inference returned no embedding.")
                 return None, None, None
            embedding_2d = embedding.reshape(1, -1)

            min_dist = float('inf')
            identified_user = None
            identified_hashid = None

            for username, data in self.user_embeddings.items():
                user_emb = data["embedding"]
                dist = cdist(user_emb.reshape(1, -1), embedding_2d, metric="cosine")[0, 0]
                logger.debug(f"Speaker verification: User '{username}', Distance: {dist:.4f}")
                if dist < min_dist:
                    min_dist = dist
                    if dist < self.speaker_verification_threshold:
                        identified_user = username
                        identified_hashid = data["hashid"]
            
            if identified_user:
                logger.info(f"Speaker verified: {identified_user} (Distance: {min_dist:.4f})")
                if self.play_verification_beep:
                     threading.Thread(target=self._play_beep_thread_target, daemon=True).start()
            else:
                logger.info(f"Speaker not recognized. Min distance: {min_dist:.4f} (Threshold: {self.speaker_verification_threshold})")
            
            return identified_user, identified_hashid, min_dist

        except Exception as e:
            logger.error(f"Error during speaker verification: {e}", exc_info=True)
            return None, None, None


    def _get_transcription_from_worker(self, audio_data_np: np.ndarray, use_prompt: bool) -> Tuple[Optional[list], Optional[object], Optional[str]]:
        """Sends audio to TranscriptionWorker and returns (segments_list, info_obj, error_str)."""
        try:
            self.parent_transcription_pipe.send((audio_data_np, self.language, use_prompt))
            
            start_poll_time = time.time()
            # Using a longer timeout for transcription, e.g., 60 seconds
            # The SafePipe itself has internal timeouts for its queue operations.
            # This poll is for the parent waiting for the child.
            while not self.parent_transcription_pipe.poll(0.2): # Poll every 200ms
                if self.shutdown_event.is_set() or self.interrupt_stop_event.is_set():
                    logger.info("Transcription worker call interrupted during polling.")
                    return None, None, "Interrupted"
                if time.time() - start_poll_time > 60: 
                    logger.error("Timeout waiting for transcription result from worker.")
                    return None, None, "Timeout"
            
            status, result = self.parent_transcription_pipe.recv()

            if status == 'success':
                segments_list, info_obj = result # Expecting list of segments
                return segments_list, info_obj, None
            else:
                return None, None, str(result) 
        except Exception as e:
            logger.error(f"Error communicating with transcription worker: {e}", exc_info=True)
            return None, None, str(e)

    def _get_resource_usage(self):
        """
        Returns a dictionary containing:
        - parent_ram_mb: Memory of the main driver logic
        - children_ram_mb: Memory of the transcription workers (Whisper)
        - total_ram_mb: Combined CPU RAM
        - gpu_vram_mb: GPU Memory usage (if available)
        """
        try:
            # 1. CPU RAM (Parent + Children)
            process = psutil.Process(os.getpid())
            parent_mem = process.memory_info().rss  # Resident Set Size
            
            children_mem = 0
            for child in process.children(recursive=True):
                try:
                    children_mem += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            # Convert to MB
            parent_mb = parent_mem / (1024 * 1024)
            children_mb = children_mem / (1024 * 1024)
            total_mb = parent_mb + children_mb

            # 2. GPU VRAM
            gpu_mb = 0.0
            if torch.cuda.is_available():
                # memory_allocated gets the actual tensor usage
                # memory_reserved gets the cached memory held by pytorch
                gpu_mb = torch.cuda.memory_allocated() / (1024 * 1024)

            return {
                "parent_ram": parent_mb,
                "children_ram": children_mb,
                "total_ram": total_mb,
                "gpu_vram": gpu_mb
            }
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {"parent_ram": 0, "children_ram": 0, "total_ram": 0, "gpu_vram": 0}

    def _execute_final_transcription_task(self, audio_data_np: np.ndarray, result_queue: queue.Queue):
        # Start Timer
        t_start = time.perf_counter()
        
        segments, info, error = self._get_transcription_from_worker(audio_data_np, use_prompt=True)
        
        # End Timer
        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000

        # Send result WITH latency
        if error:
            result_queue.put({'type': 'transcription', 'data': None, 'error': error, 'latency': latency_ms})
        else:
            result_queue.put({'type': 'transcription', 'data': (segments, info), 'error': None, 'latency': latency_ms})

    def _execute_speaker_verification_task(self, audio_segment_bytes: bytes, result_queue: queue.Queue, hot_word_was_detected: bool):
        # Total Task Timer
        t_task_start = time.perf_counter() 
        
        overlap_latency_ms = 0.0
        fast_path_used = False

        if not self.enable_speaker_verification:
            verification_data = {
                "speaker_name": None, "speaker_hashid": None, "distance": None, 
                "overlap": False, "saved_filename": None, "identified_users": [],
                "latency_metrics": {"overlap_detection_ms": 0.0, "total_verification_ms": 0.0}
            }
            result_queue.put({'type': 'verification', 'data': verification_data, 'error': None})
            return

        try:
            # 1. Prepare Main Wav File
            filename_prefix = "verified_seg" if self.save_verified_segments else "temp_ver_seg"
            wav_data_io, saved_filename = self._audio_segment_to_wav_bytes(
                audio_segment_bytes, self.sample_rate, 
                save_to_file=self.save_verified_segments,
                filename_prefix=filename_prefix
            )

            if not wav_data_io:
                result_queue.put({'type': 'verification', 'data': None, 'error': 'WAV conversion failed'})
                return

            
            wav_data_io.seek(0)
            quick_name, quick_hash, quick_dist = self._verify_speaker_segment(wav_data_io)
            
            # STRICT THRESHOLD for Fast Path
            # If distance is very low (e.g. 0.4), it is definitely the user and signal is clean.
            # Normal threshold is usually 0.7.
            FAST_PATH_THRESHOLD = 0.70 

            if quick_name and quick_dist < FAST_PATH_THRESHOLD:
                # SUCCESS: We found the user quickly. Skip heavy diarization.
                fast_path_used = True
                
                t_task_end = time.perf_counter()
                total_latency_ms = (t_task_end - t_task_start) * 1000
                
                logger.info(f"⚡ Fast Path Verification used for {quick_name} (Dist: {quick_dist:.4f})")

                verification_data = {
                    "speaker_name": quick_name,
                    "speaker_hashid": quick_hash,
                    "distance": quick_dist,
                    "overlap": False, # Assume no overlap if match is this good
                    "speaker_count": 1,
                    "saved_filename": saved_filename,
                    "hot_word_detected": hot_word_was_detected,
                    "identified_users": [{
                        "name": quick_name, 
                        "hashid": quick_hash, 
                        "confidence": 1.0 - quick_dist
                    }],
                    "latency_metrics": {
                        "overlap_detection_ms": 0.0, # Skipped
                        "total_verification_ms": total_latency_ms,
                        "method": "fast_path"
                    }
                }
                result_queue.put({'type': 'verification', 'data': verification_data, 'error': None})
                return

            
            logger.debug(f"Fast path failed (Dist: {quick_dist if quick_name else 'N/A'}). Running full diarization.")

            # Start Overlap Timer
            t_diar_start = time.perf_counter()
            
            has_overlap, speaker_count, diarization_result = self._diarize_and_analyze_segment(wav_data_io)
            
            # End Overlap Timer
            t_diar_end = time.perf_counter()
            overlap_latency_ms = (t_diar_end - t_diar_start) * 1000
            
            identified_users = []
            
            # ISOLATION AND VERIFICATION LOGIC (Same as before)
            if diarization_result and speaker_count > 0:
                wav_data_io.seek(0)
                full_audio_data, sr = sf.read(wav_data_io)
                
                for label in diarization_result.labels():
                    try:
                        timeline = diarization_result.label_timeline(label)
                        speaker_audio_parts = []
                        total_duration = 0.0
                        for segment in timeline:
                            start_sample = int(segment.start * sr)
                            end_sample = int(segment.end * sr)
                            if start_sample < len(full_audio_data):
                                chunk = full_audio_data[start_sample:end_sample]
                                speaker_audio_parts.append(chunk)
                                total_duration += segment.duration

                        if total_duration > 0.3 and speaker_audio_parts:
                            isolated_audio = np.concatenate(speaker_audio_parts)
                            isolated_io = io.BytesIO()
                            sf.write(isolated_io, isolated_audio, sr, format='WAV')
                            isolated_io.seek(0)
                            u_name, u_hash, u_dist = self._verify_speaker_segment(isolated_io)
                            if u_name and u_hash:
                                identified_users.append({
                                    "name": u_name, "hashid": u_hash, "confidence": max(0.0, 1.0 - u_dist), "label": label
                                })
                    except Exception as inner_e:
                        logger.error(f"Error verifying specific label {label}: {inner_e}")
                        continue
            else:
                # If diarization failed but fast path also failed, 
                # we return the result of the whole file verify from the start of the function
                # or try again if logic requires. Here we use the quick result if valid.
                if quick_name:
                     identified_users.append({
                        "name": quick_name, "hashid": quick_hash, "confidence": max(0.0, 1.0 - quick_dist)
                    })

            # Determine Primary Speaker
            primary_name = None
            primary_hash = None
            primary_dist = None
            
            if identified_users:
                identified_users.sort(key=lambda x: x['confidence'], reverse=True)
                best_match = identified_users[0]
                primary_name = best_match['name']
                primary_hash = best_match['hashid']
                primary_dist = 1.0 - best_match['confidence']
            
            if wav_data_io: wav_data_io.close()

            # Final Payload
            t_task_end = time.perf_counter()
            total_latency_ms = (t_task_end - t_task_start) * 1000

            verification_data = {
                "speaker_name": primary_name,
                "speaker_hashid": primary_hash,
                "distance": primary_dist,
                "overlap": has_overlap,
                "speaker_count": speaker_count,
                "saved_filename": saved_filename,
                "hot_word_detected": hot_word_was_detected,
                "identified_users": identified_users,
                "latency_metrics": {
                    "overlap_detection_ms": overlap_latency_ms,
                    "total_verification_ms": total_latency_ms,
                    "method": "full_pipeline"
                }
            }
            result_queue.put({'type': 'verification', 'data': verification_data, 'error': None})

        except Exception as e:
            logger.error(f"Exception in speaker verification task: {e}", exc_info=True)
            result_queue.put({'type': 'verification', 'data': None, 'error': str(e)})

    def _start_thread(self, target=None, args=()):
        """
        Implement a consistent threading model across the library.

        This method is used to start any thread in this library. It uses the
        standard threading. Thread for Linux and for all others uses the pytorch
        MultiProcessing library 'Process'.
        Args:
            target (callable object): is the callable object to be invoked by
              the run() method. Defaults to None, meaning nothing is called.
            args (tuple): is a list or tuple of arguments for the target
              invocation. Defaults to ().
        """
        if (platform.system() == 'Linux'):
            thread = threading.Thread(target=target, args=args)
            thread.deamon = True
            thread.start()
            return thread
        else:
            thread = mp.Process(target=target, args=args)
            thread.start()
            return thread

    def _read_stdout(self):
        while not self.shutdown_event.is_set():
            try:
                if self.parent_stdout_pipe.poll(0.1):
                    logger.debug("Receive from stdout pipe")
                    message = self.parent_stdout_pipe.recv()
                    logger.info(message)
            except (BrokenPipeError, EOFError, OSError):
                # The pipe probably has been closed, so we ignore the error
                pass
            except KeyboardInterrupt:  # handle manual interruption (Ctrl+C)
                logger.info("KeyboardInterrupt in read from stdout detected, exiting...")
                break
            except Exception as e:
                logger.error(f"Unexpected error in read from stdout: {e}", exc_info=True)
                logger.error(traceback.format_exc())  # Log the full traceback here
                break 
            time.sleep(0.1)

    def _transcription_worker(*args, **kwargs):
        worker = ListenDriveScriber(*args, **kwargs)
        worker.run()

    def _run_callback(self, cb, *args, **kwargs):
        if self.start_callback_in_new_thread:
            # Run the callback in a new thread to avoid blocking the main thread
            threading.Thread(target=cb, args=args, kwargs=kwargs, daemon=True).start()
        else:
            # Run the callback in the main thread to avoid threading issues
            cb(*args, **kwargs)

    @staticmethod
    def _audio_data_worker(
        audio_queue, amplitude_queue, target_sample_rate,
        vad_chunk_buffer_size, # Renamed from buffer_size for clarity
        input_device_index, shutdown_event, interrupt_stop_event, use_microphone_value_proxy
    ):
        import pyaudio # Import here as it's only used by this worker
        import numpy as np # Local import for process
        from scipy import signal as sp_signal # Local import for process

        # Use the global constants for format and channels
        PYAUDIO_FORMAT = pyaudio.paInt16 # Directly use pyaudio constant
        PYAUDIO_CHANNELS = CHANNELS     # Use the global CHANNELS

        if __name__ == '__main__': # This condition is usually for script execution, not for methods inside a class
            system_signal.signal(system_signal.SIGINT, system_signal.SIG_IGN)

        # ... (get_highest_sample_rate, initialize_audio_stream, preprocess_audio as before, but ensure they use PYAUDIO_FORMAT, PYAUDIO_CHANNELS) ...
        def get_highest_sample_rate(audio_interface_local, device_idx):
            try:
                dev_info = audio_interface_local.get_device_info_by_index(device_idx)
                max_sr = int(dev_info['defaultSampleRate'])
                # Try common rates
                common_rates = [8000, 11025, 16000, 22050, 44100, 48000]
                supported = []
                for r in common_rates:
                    try:
                        if audio_interface_local.is_format_supported(r, input_device=device_idx, input_channels=PYAUDIO_CHANNELS, input_format=PYAUDIO_FORMAT):
                            supported.append(r)
                    except ValueError: pass
                if supported: max_sr = max(supported)
                if target_sample_rate in supported and target_sample_rate >= max_sr : max_sr = target_sample_rate
                logger.debug(f"Device {dev_info['name']} best SR: {max_sr}Hz (Target: {target_sample_rate}Hz)")
                return max_sr
            except Exception as e:
                logger.warning(f"Error getting highest SR for device {device_idx}: {e}. Defaulting to {target_sample_rate or 48000}Hz.")
                return target_sample_rate or 48000

        def initialize_audio_stream(audio_interface_local, dev_idx, sr_to_use, frames_per_buffer_local):
            try:
                stream_obj = audio_interface_local.open(
                    format=PYAUDIO_FORMAT, channels=PYAUDIO_CHANNELS, rate=sr_to_use,
                    input=True, frames_per_buffer=frames_per_buffer_local,
                    input_device_index=dev_idx,
                )
                logger.info(f"Mic stream opened (Device: {dev_idx}, Rate: {sr_to_use}Hz, Chunk: {frames_per_buffer_local})")
                return stream_obj
            except Exception as e:
                logger.error(f"Failed to open stream for device {dev_idx} at {sr_to_use}Hz: {e}")
                raise

        def preprocess_audio(raw_bytes, original_sr, target_sr):
            if original_sr == target_sr: return raw_bytes
            audio_np = np.frombuffer(raw_bytes, dtype=np.int16)
            if audio_np.size == 0: return b''
            
            num_samples_target = int(len(audio_np) * target_sr / original_sr)
            if num_samples_target == 0: return b'' # Avoid resampling to zero samples
            
            resampled_np = sp_signal.resample(audio_np, num_samples_target)
            return resampled_np.astype(np.int16).tobytes()


        audio_interface_instance = None
        stream_instance = None
        actual_device_sample_rate_val = None # Renamed
        pyaudio_read_chunk_size_val = 1024 # Renamed

        def setup_audio_stream(): # Renamed from setup_audio
            nonlocal audio_interface_instance, stream_instance, actual_device_sample_rate_val, input_device_index
            retries = 5; delay = 3
            for attempt in range(retries):
                if shutdown_event.is_set(): return False
                try:
                    if audio_interface_instance is None: audio_interface_instance = pyaudio.PyAudio()
                    
                    current_input_device_index = input_device_index # Use the one passed to worker
                    if current_input_device_index is None:
                        try:
                            default_dev_info = audio_interface_instance.get_default_input_device_info()
                            current_input_device_index = default_dev_info['index']
                            logger.info(f"Using default input device: {default_dev_info['name']} (Index {current_input_device_index})")
                        except Exception as e_def:
                            logger.error(f"Could not get default input device: {e_def}. Listing devices:")
                            for i in range(audio_interface_instance.get_device_count()):
                                try: logger.info(f"  Dev {i}: {audio_interface_instance.get_device_info_by_index(i)['name']}")
                                except: pass
                            return False # Critical if no device can be selected

                    actual_device_sample_rate_val = get_highest_sample_rate(audio_interface_instance, current_input_device_index)
                    stream_instance = initialize_audio_stream(audio_interface_instance, current_input_device_index,
                                                              actual_device_sample_rate_val, pyaudio_read_chunk_size_val)
                    if stream_instance: return True
                except Exception as e_setup:
                    logger.error(f"Audio setup attempt {attempt+1}/{retries} failed: {e_setup}")
                    if audio_interface_instance:
                        try: audio_interface_instance.terminate()
                        except: pass
                        audio_interface_instance = None
                    if attempt < retries -1 : time.sleep(delay)
                    else: return False
            return False

        if not setup_audio_stream():
            logger.error("Audio data worker: Failed to set up audio. Exiting.")
            return

        vad_accumulated_bytes = bytearray()
        # vad_chunk_buffer_size is in SAMPLES, Silero VAD expects 16-bit PCM (2 bytes/sample)
        vad_model_input_chunk_bytes = vad_chunk_buffer_size * 2 
        last_debug_log_time = 0

        try:
            while not shutdown_event.is_set():
                try:
                    if not stream_instance or not stream_instance.is_active():
                        logger.warning("Stream inactive. Re-initializing...")
                        if stream_instance:
                            try: stream_instance.close()
                            except: pass
                        if not setup_audio_stream():
                            logger.error("Failed to re-initialize stream. Worker stopping.")
                            break
                        else: continue # Restart loop with new stream

                    raw_mic_bytes = stream_instance.read(pyaudio_read_chunk_size_val, exception_on_overflow=False)

                    if use_microphone_value_proxy.value:
                        # Amplitude
                        if raw_mic_bytes:
                            audio_int16_raw = np.frombuffer(raw_mic_bytes, dtype=np.int16)
                            if audio_int16_raw.size > 0:
                                audio_float_norm = audio_int16_raw.astype(np.float32) / 32768.0
                                rms_amp = np.sqrt(np.mean(np.clip(audio_float_norm, -1.0, 1.0)**2))
                                try: amplitude_queue.put_nowait(rms_amp)
                                except queue.Full: logger.warning("Amplitude queue full in worker.")
                                except Exception as e_ampq: logger.error(f"Error putting to amp_q: {e_ampq}")
                        
                        # Preprocess for VAD/STT
                        processed_for_vad_bytes = preprocess_audio(raw_mic_bytes, actual_device_sample_rate_val, target_sample_rate)
                        vad_accumulated_bytes.extend(processed_for_vad_bytes)

                        # Feed to VAD audio_queue
                        while len(vad_accumulated_bytes) >= vad_model_input_chunk_bytes:
                            chunk_to_queue = vad_accumulated_bytes[:vad_model_input_chunk_bytes]
                            vad_accumulated_bytes = vad_accumulated_bytes[vad_model_input_chunk_bytes:]
                            try: audio_queue.put_nowait(bytes(chunk_to_queue)) # Convert bytearray to bytes
                            except queue.Full: logger.warning("Main audio_queue full in worker.")
                            except Exception as e_aq: logger.error(f"Error putting to audio_q: {e_aq}")


                            if time.time() - last_debug_log_time > 5: # Log less frequently
                                logger.debug(f"_audio_data_worker: Queued VAD chunk. vad_accumulated_bytes len: {len(vad_accumulated_bytes)}")
                                last_debug_log_time = time.time()
                    else:
                        time.sleep(0.02) # Mic muted, sleep

                except pyaudio.PaError as pa_e:
                    if pa_e.args[0] == pyaudio.paInputOverflowed: logger.warning("PyAudio Input Overflowed.")
                    else: logger.error(f"PyAudio error: {pa_e}", exc_info=True); stream_instance = None # Force re-setup
                except OSError as os_e:
                    logger.error(f"OSError in recording loop: {os_e}", exc_info=True); stream_instance = None
                except Exception as e_loop:
                    logger.error(f"Unknown error in recording loop: {e_loop}", exc_info=True)
                    if "stream" in str(e_loop).lower(): stream_instance = None 
                    else: time.sleep(0.1)
        
        except KeyboardInterrupt: interrupt_stop_event.set()
        except Exception as main_e: logger.critical(f"Audio worker unrecoverable error: {main_e}", exc_info=True)
        finally:
            logger.info("Audio data worker finalizing...")
            if stream_instance:
                try:
                    if stream_instance.is_active(): stream_instance.stop_stream()
                    stream_instance.close()
                except: pass
            if audio_interface_instance:
                try: audio_interface_instance.terminate()
                except: pass
            logger.info("Audio data worker process finished.")


    # Inside class AudioToTextRecorder:
    def _amplitude_reader_worker(self):
        """Worker thread to read RMS values from the queue and call the callback."""
        logger.debug("Amplitude reader worker started.")
        while self.is_running: # Use the existing flag
            try:
                rms_value = self.amplitude_queue.get(timeout=0.1) # Small timeout

                if self.on_amplitude_update:
                    try:
                        # Use the existing _run_callback helper if you have one for other callbacks
                        # self._run_callback(self.on_amplitude_update, rms_value)
                        # Or directly:
                        if self.start_callback_in_new_thread:
                            threading.Thread(target=self.on_amplitude_update, args=(rms_value,), daemon=True).start()
                        else:
                            self.on_amplitude_update(rms_value)
                    except Exception as cb_err:
                        logger.error(f"Error executing on_amplitude_update callback: {cb_err}", exc_info=True)

            except queue.Empty:
                # Queue is empty, loop checks self.is_running
                continue
            except Exception as q_err:
                logger.error(f"Error reading from amplitude queue: {q_err}")
                time.sleep(0.5) # Avoid busy loop on persistent error

        logger.debug("Amplitude reader worker finished.")

    def wakeup(self):
        """
        If in wake work modus, wake up as if a wake word was spoken.
        """
        self.listen_start = time.time()

    def abort(self):
        state = self.state
        self.start_recording_on_voice_activity = False
        self.stop_recording_on_voice_deactivity = False
        self.interrupt_stop_event.set()
        if self.state != "inactive": # if inactive, was_interrupted will never be set
            self.was_interrupted.wait()
            self._set_state("transcribing")
        self.was_interrupted.clear()
        if self.is_recording: # if recording, make sure to stop the recorder
            self.stop()

    # --- wait_audio: Modified to store raw_bytes ---
    def wait_audio(self):
        try:
            if self.listen_start == 0: self.listen_start = time.time()

            if not self.is_recording and not self.frames:
                # self._set_state("listening")
                self.start_recording_on_voice_activity = True
                logger.debug('Waiting for recording start event...')
                while not self.interrupt_stop_event.is_set() and not self.shutdown_event.is_set():
                    if self.start_recording_event.wait(timeout=0.02): break
                if self.interrupt_stop_event.is_set() or self.shutdown_event.is_set(): return


            if self.is_recording:
                self.stop_recording_on_voice_deactivity = True
                logger.debug('Waiting for recording stop event...')
                while not self.interrupt_stop_event.is_set() and not self.shutdown_event.is_set():
                    if self.stop_recording_event.wait(timeout=0.02): break
                if self.interrupt_stop_event.is_set() or self.shutdown_event.is_set(): return

            # --- Store raw bytes of the completed segment ---
            current_frames_for_segment = self.frames if self.frames else self.last_frames
            if current_frames_for_segment:
                self.current_segment_raw_bytes = b''.join(current_frames_for_segment)
            else:
                self.current_segment_raw_bytes = None
            # --- End Store raw bytes ---

            # Process frames into self.audio (np.float32 array)
            # This logic for self.audio (float32) and backdating remains
            full_audio_array = np.frombuffer(b''.join(current_frames_for_segment), dtype=np.int16) if current_frames_for_segment else np.array([], dtype=np.int16)
            
            if full_audio_array.size > 0:
                full_audio_float = full_audio_array.astype(np.float32) / INT16_MAX_ABS_VALUE
            else:
                full_audio_float = np.array([], dtype=np.float32)

            # Backdating logic (applies to self.audio which is float32)
            samples_to_remove = int(self.sample_rate * self.backdate_stop_seconds)
            if samples_to_remove > 0 and samples_to_remove < len(full_audio_float):
                self.audio = full_audio_float[:-samples_to_remove]
            else:
                self.audio = full_audio_float

            # Prepare frames for potential resume (backdating resume)
            samples_to_keep_for_resume = int(self.sample_rate * self.backdate_resume_seconds)
            frames_for_resume_bytes = b''
            if samples_to_keep_for_resume > 0 and len(full_audio_float) > 0:
                resume_audio_float = full_audio_float[-samples_to_keep_for_resume:]
                resume_audio_int16 = (resume_audio_float * INT16_MAX_ABS_VALUE).astype(np.int16)
                frames_for_resume_bytes = resume_audio_int16.tobytes()

            self.frames.clear()
            self.last_frames.clear() # Clear last_frames too after processing

            # Repopulate self.frames with resume data if any
            # This logic might need adjustment if self.frames should always be raw VAD chunks
            # For now, assuming it's for the next recording's pre-roll based on original code
            if frames_for_resume_bytes:
                 # Split into VAD-sized chunks (self.buffer_size * 2 bytes)
                 vad_chunk_byte_size = self.buffer_size * 2 
                 for i in range(0, len(frames_for_resume_bytes), vad_chunk_byte_size):
                     self.frames.append(frames_for_resume_bytes[i : i + vad_chunk_byte_size])
            
            self.backdate_stop_seconds = 0.0
            self.backdate_resume_seconds = 0.0
            self.listen_start = 0
            self._set_state("inactive")

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt in wait_audio, shutting down")
            self.shutdown()
            raise
        except Exception as e:
            logger.error(f"Error in wait_audio: {e}", exc_info=True)
            # Potentially reset state or shutdown if critical
            self._set_state("inactive") # Fallback state



    def perform_final_transcription(self, audio_bytes=None, use_prompt=True):
        start_time = 0
        with self.transcription_lock:
            if audio_bytes is None:
                audio_bytes = copy.deepcopy(self.audio)

            if audio_bytes is None or len(audio_bytes) == 0:
                print("No audio data available for transcription")
                #logger.info("No audio data available for transcription")
                return ""

            try:
                if self.transcribe_count == 0:
                    logger.debug("Adding transcription request, no early transcription started")
                    start_time = time.time()  # Start timing
                    self.parent_transcription_pipe.send((audio_bytes, self.language, use_prompt))
                    self.transcribe_count += 1

                while self.transcribe_count > 0:
                    logger.debug(F"Receive from parent_transcription_pipe after sendiung transcription request, transcribe_count: {self.transcribe_count}")
                    if not self.parent_transcription_pipe.poll(0.1): # check if transcription done
                        if self.interrupt_stop_event.is_set(): # check if interrupted
                            self.was_interrupted.set()
                            self._set_state("inactive")
                            return "" # return empty string if interrupted
                        continue
                    status, result = self.parent_transcription_pipe.recv()
                    self.transcribe_count -= 1

                self.allowed_to_early_transcribe = True
                self._set_state("inactive")
                if status == 'success':
                    segments, info = result
                    self.detected_language = info.language if info.language_probability > 0 else None
                    self.detected_language_probability = info.language_probability
                    self.last_transcription_bytes = copy.deepcopy(audio_bytes)
                    self.last_transcription_bytes_b64 = base64.b64encode(self.last_transcription_bytes.tobytes()).decode('utf-8')
                    transcription = self._preprocess_output(segments)
                    end_time = time.time()  # End timing
                    transcription_time = end_time - start_time

                    if start_time:
                        if self.print_transcription_time:
                            print(f"Model {self.main_model_type} completed transcription in {transcription_time:.2f} seconds")
                        else:
                            logger.debug(f"Model {self.main_model_type} completed transcription in {transcription_time:.2f} seconds")
                    return "" if self.interrupt_stop_event.is_set() else transcription # if interrupted return empty string
                else:
                    logger.error(f"Transcription error: {result}")
                    raise Exception(result)
            except Exception as e:
                logger.error(f"Error during transcription: {str(e)}", exc_info=True)
                raise e

    def transcribe(self, was_wake_word_triggered: bool = False):
            # PROFILE: Start Total Timer
            t_start = time.perf_counter()

            audio_to_transcribe_np = np.copy(self.audio)
            raw_bytes_for_verification = self.current_segment_raw_bytes

            if not audio_to_transcribe_np.any() or raw_bytes_for_verification is None:
                # Handle empty audio
                if self.on_segment_processed_callback:
                    self._run_callback(self.on_segment_processed_callback, {"audio_processed": False, "error": "No audio data"})
                if self.on_final_transcription_for_service_callback_ref:
                    self.on_final_transcription_for_service_callback_ref("", {"processed": False, "error": "No audio data"})
                self._set_state("inactive") 
                return ""

            # PROFILE: Start SNR Timer
            t_snr_start = time.perf_counter()

            # ==============================================================================
            #              HARD SNR QUALITY GATING
            # ==============================================================================
            
            # 1. Calculate SNR
            snr_value = self._calculate_snr(audio_to_transcribe_np)
            
            # 2. Check Threshold
            threshold = getattr(self, 'min_snr_threshold', 0.5) 
            passed_quality_check = bool(snr_value >= threshold)

            # 3. Log Status
            duration_seconds = len(audio_to_transcribe_np) / self.sample_rate
            color = bcolors.OKGREEN if passed_quality_check else bcolors.WARNING
            logger.info(f"{color}Audio Quality Check: SNR = {snr_value:.2f} dB (Threshold: {threshold} dB) | Duration: {duration_seconds:.2f}s{bcolors.ENDC}")

            # 4. HARD EXIT (The Gating Logic)
            if not passed_quality_check:
                logger.warning(f"SNR too low. Aborting Processing to save resources.")
                
                # Notify UI (via callback)
                if self.on_segment_processed_callback:
                    self._run_callback(self.on_segment_processed_callback, {
                        "passed_quality_check": False,
                        "snr": snr_value,
                        "transcription_pending": False,
                        "speaker_name": "Ignored (Noise)",
                        "error": "Low SNR"
                    })

                # Reset State and Exit Immediately
                self.listen_start = 0
                self._set_state("inactive")
                
                # Print Profile for the "Rejection" path
                t_reject_end = time.perf_counter()
                print("\n" + "="*45)
                print(f"⚡ LATENCY PROFILE (REJECTED NOISE)")
                print(f"="*45)
                print(f"1. SNR Calculation:    {(t_snr_start - t_start)*1000:8.2f} ms")
                print(f"2. Gating Decision:    {(t_reject_end - t_snr_start)*1000:8.2f} ms")
                print(f"-"*45)
                print(f"✅ TOTAL SAVED TIME:   ~1700.00 ms")
                print(f"="*45 + "\n")
                
                return "" 

            # ==============================================================================
            # AI PROCESSING (Only runs if SNR passed)
            # ==============================================================================
            
            t_snr_end = time.perf_counter()

            self._set_state("transcribing")
            if self.on_transcription_start:
                if self._run_callback(self.on_transcription_start, audio_to_transcribe_np) is False:
                    self._set_state("inactive")
                    return ""

            transcription_q = queue.Queue()
            verification_q = queue.Queue()

            trans_thread = threading.Thread(target=self._execute_final_transcription_task,
                                            args=(audio_to_transcribe_np, transcription_q), daemon=True)
            
            ver_thread = threading.Thread(target=self._execute_speaker_verification_task,
                                            args=(raw_bytes_for_verification, verification_q, was_wake_word_triggered), daemon=True)

            logger.debug("Starting parallel transcription and verification threads.")

            # PROFILE: Start AI Timer
            t_threads_start = time.perf_counter()

            trans_thread.start()
            ver_thread.start()

            trans_thread.join(timeout=70) 
            ver_thread.join(timeout=35)   

            # PROFILE: End AI Timer
            t_threads_end = time.perf_counter()

            # Retrieve Results
            trans_result = {'type': 'transcription', 'data': None, 'error': 'Timeout', 'latency': 0.0}
            if not trans_thread.is_alive():
                try: trans_result = transcription_q.get_nowait()
                except queue.Empty: trans_result['error'] = 'Transcription queue empty after join'
            
            ver_result = {'type': 'verification', 'data': None, 'error': 'Timeout'}
            if not ver_thread.is_alive():
                try: ver_result = verification_q.get_nowait()
                except queue.Empty: ver_result['error'] = 'Verification queue empty after join'

            # ---------------------------------------------------------
            # EXTRACT LATENCY METRICS
            # ---------------------------------------------------------
            
            # 1. Transcription Latency (from worker)
            trans_latency_ms = trans_result.get('latency', 0.0)

            # 2. Verification & Overlap Latency
            ver_data_safe = ver_result.get('data', {}) if isinstance(ver_result.get('data'), dict) else {}
            ver_metrics = ver_data_safe.get('latency_metrics', {})
            overlap_latency_ms = ver_metrics.get('overlap_detection_ms', 0.0)
            total_verify_ms = ver_metrics.get('total_verification_ms', 0.0)

            # 3. VAD Latency (Average of the frames processed)
            avg_vad_latency_ms = 0.0
            if hasattr(self, 'vad_perf_log') and self.vad_perf_log:
                avg_vad_latency_ms = (sum(self.vad_perf_log) / len(self.vad_perf_log)) * 1000
                # Reset log for next turn
                self.vad_perf_log = []

            # ---------------------------------------------------------

            # Construct Metadata
            verification_data_for_callbacks = ver_data_safe.copy()
            verification_data_for_callbacks["hot_word_detected"] = was_wake_word_triggered
            verification_data_for_callbacks["snr"] = snr_value
            verification_data_for_callbacks["passed_quality_check"] = True
            
            # Pass latency info to callbacks (optional, good for debugging)
            verification_data_for_callbacks["perf_metrics"] = {
                "transcription_ms": trans_latency_ms,
                "overlap_ms": overlap_latency_ms,
                "verification_ms": total_verify_ms,
                "avg_vad_ms": avg_vad_latency_ms
            }

            if was_wake_word_triggered:
                self.release_mode_override()

            if ver_result.get('error'): verification_data_for_callbacks["verification_error"] = ver_result.get('error')
            if trans_result.get('error'): verification_data_for_callbacks["transcription_error"] = trans_result.get('error')
            verification_data_for_callbacks["transcription_pending"] = False 

            # Intermediate callback
            if self.on_segment_processed_callback:
                self._run_callback(self.on_segment_processed_callback, verification_data_for_callbacks)

            # Final Decision Logic
            final_text_output = ""
            should_use_text = False

            if trans_result.get('error') or (self.enable_speaker_verification and ver_result.get('error')):
                logger.error("Task failed in threads.")
            else:
                if self.enable_speaker_verification:
                    if verification_data_for_callbacks.get("speaker_name") and not verification_data_for_callbacks.get("overlap"):
                        should_use_text = True
                    else:
                        logger.info("Speaker unverified or overlap detected.")
                else:
                    should_use_text = True 

                if should_use_text and trans_result.get('data'):
                    segments_list, _ = trans_result['data'] 
                    final_text_output = " ".join(seg.text for seg in segments_list).strip()
                    final_text_output = self._preprocess_output(final_text_output)
                    logger.info(f"Final text: \"{final_text_output}\"")

            # Send to Service Layer
            if self.on_final_transcription_for_service_callback_ref:
                self.on_final_transcription_for_service_callback_ref(final_text_output, verification_data_for_callbacks)

            # Cleanup
            self.last_speaker_count = verification_data_for_callbacks.get("speaker_count", 0)
            self.listen_start = 0
            self._set_state("inactive")

            t_end = time.perf_counter()
            
            snr_time = (t_snr_end - t_snr_start) * 1000
            total_time = (t_end - t_start) * 1000
            
            # Calculate overhead. Note: Threads run in parallel, so we subtract the MAX of the two longest threads
            # plus the serial SNR time from the total wall-clock time.
            parallel_execution_time = max(trans_latency_ms, total_verify_ms)
            # Fallback if metrics missing: use wall clock difference of threads
            if parallel_execution_time == 0: 
                parallel_execution_time = (t_threads_end - t_threads_start) * 1000
                
            overhead = total_time - (snr_time + parallel_execution_time)

            mem_stats = self._get_resource_usage()

            print("\n" + "="*45)
            print(f" LATENCY PROFILE (DETAILED)")
            print(f"="*45)
            print(f"1. Audio Analysis:")
            print(f"   - Avg VAD (per frame):{avg_vad_latency_ms:8.2f} ms")
            print(f"   - SNR Calculation:    {snr_time:8.2f} ms")
            print(f"2. Parallel Processing:")
            print(f"   - Transcription:      {trans_latency_ms:8.2f} ms")
            print(f"   - Overlap Detection:  {overlap_latency_ms:8.2f} ms")
            print(f"   - Full Spkr Verify:   {total_verify_ms:8.2f} ms")
            print(f"3. System Overhead:      {overhead:8.2f} ms")
            print(f"-"*45)
            print(f" TOTAL LATENCY:        {total_time:8.2f} ms")
            print(f"="*45 + "\n")

            # --- NEW MEMORY SECTION ---
            print(f"💾 MEMORY FOOTPRINT")
            print(f"="*45)
            print(f"1. CPU RAM (System):")
            print(f"   - Driver Process:     {mem_stats['parent_ram']:8.2f} MB")
            print(f"   - Worker (Whisper):   {mem_stats['children_ram']:8.2f} MB")
            print(f"   - TOTAL RAM USED:     {mem_stats['total_ram']:8.2f} MB")
            print(f"2. GPU VRAM (Cuda):")
            print(f"   - Model Allocation:   {mem_stats['gpu_vram']:8.2f} MB")
            print(f"="*45 + "\n")

            return final_text_output

    def _process_wakeword(self, data):
        """
        Processes audio data to detect wake words.
        """
        if self.wakeword_backend in {'pvp', 'pvporcupine'}:
            pcm = struct.unpack_from(
                "h" * self.buffer_size,
                data
            )
            porcupine_index = self.porcupine.process(pcm)
            # if self.debug_mode:
            #     logger.info(f"wake words porcupine_index: {porcupine_index}")
            # return porcupine_index
            for i in range(0, len(pcm) - self.porcupine.frame_length + 1, self.porcupine.frame_length):
                frame = pcm[i:i + self.porcupine.frame_length]
                porcupine_index = self.porcupine.process(frame)
                if porcupine_index >= 0:
                    logger.debug(f"Porcupine detected wake word in buffer.")
                    return porcupine_index
            return -1 # Return -1 if no keyword is found in the entire buffer

        elif self.wakeword_backend in {'oww', 'openwakeword', 'openwakewords'}:
            pcm = np.frombuffer(data, dtype=np.int16)
            prediction = self.owwModel.predict(pcm)
            max_score = -1
            max_index = -1
            wake_words_in_prediction = len(self.owwModel.prediction_buffer.keys())
            self.wake_words_sensitivities
            if wake_words_in_prediction:
                for idx, mdl in enumerate(self.owwModel.prediction_buffer.keys()):
                    scores = list(self.owwModel.prediction_buffer[mdl])
                    if scores[-1] >= self.wake_words_sensitivity and scores[-1] > max_score:
                        max_score = scores[-1]
                        max_index = idx
                if self.debug_mode:
                    logger.info(f"wake words oww max_index, max_score: {max_index} {max_score}")
                return max_index  
            else:
                if self.debug_mode:
                    logger.info(f"wake words oww_index: -1")
                return -1

        if self.debug_mode:        
            logger.info("wake words no match")

        return -1

    # text() method calls transcribe() after wait_audio()
    def text(self, on_transcription_finished=None): # on_transcription_finished is now effectively self.on_final_transcription_for_service_callback_ref
        self.interrupt_stop_event.clear()
        self.was_interrupted.clear()
        try:
            self.wait_audio()
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt in text() method, shutting down.")
            self.shutdown()
            raise
        except Exception as e:
            logger.error(f"Exception during wait_audio in text(): {e}", exc_info=True)
            self._set_state("inactive") # Ensure state reset
            return "" # Return empty on error during wait_audio

        if self.is_shut_down or self.interrupt_stop_event.is_set():
            if self.interrupt_stop_event.is_set(): self.was_interrupted.set()
            return ""

        # The `on_transcription_finished` argument to `text()` is a bit redundant now
        # as the main callback is `self.on_final_transcription_for_service_callback_ref`.
        # If `on_transcription_finished` is still needed for some specific local use of `text()`,
        # it would need careful handling alongside the main service callback.
        # For now, assuming the main service callback is primary.
        
        return self.transcribe() # transcribe() now handles the callbacks


    def format_number(self, num):
        # Convert the number to a string
        num_str = f"{num:.10f}"  # Ensure precision is sufficient
        # Split the number into integer and decimal parts
        integer_part, decimal_part = num_str.split('.')
        # Take the last two digits of the integer part and the first two digits of the decimal part
        result = f"{integer_part[-2:]}.{decimal_part[:2]}"
        return result

    def start(self, frames = None):
        """
        Starts recording audio directly without waiting for voice activity.
        """

        # Ensure there's a minimum interval
        # between stopping and starting recording
        if (time.time() - self.recording_stop_time
                < self.min_gap_between_recordings):
            logger.info("Attempted to start recording "
                         "too soon after stopping."
                         )
            return self

        logger.info("recording started")
        self._set_state("recording")
        self.vad_perf_log = []  # <--- Add this: Reset VAD logs on new recording
        self.text_storage = []
        self.realtime_stabilized_text = ""
        self.realtime_stabilized_safetext = ""
        self.wakeword_detected = False
        self.wake_word_detect_time = 0
        self.frames = []
        if frames:
            self.frames = frames
        self.is_recording = True

        self.recording_start_time = time.time()
        self.is_silero_speech_active = False
        self.is_webrtc_speech_active = False
        self.stop_recording_event.clear()
        self.start_recording_event.set()

        if self.on_recording_start:
            self._run_callback(self.on_recording_start)

        return self

    def stop(self,
             backdate_stop_seconds: float = 0.0,
             backdate_resume_seconds: float = 0.0,
        ):
        """
        Stops recording audio.

        Args:
        - backdate_stop_seconds (float, default="0.0"): Specifies the number of
            seconds to backdate the stop time. This is useful when the stop
            command is issued after the actual stop time.
        - backdate_resume_seconds (float, default="0.0"): Specifies the number
            of seconds to backdate the time relistening is initiated.
        """

        # Ensure there's a minimum interval
        # between starting and stopping recording
        if (time.time() - self.recording_start_time
                < self.min_length_of_recording):
            logger.info("Attempted to stop recording "
                         "too soon after starting."
                         )
            return self

        logger.info("recording stopped")
        self.last_frames = copy.deepcopy(self.frames)
        self.backdate_stop_seconds = backdate_stop_seconds
        self.backdate_resume_seconds = backdate_resume_seconds

        # 1. Reset all wake word related flags to prepare for the next event.
        self.wakeword_detected = False
        self.wakeword_detection_flag = False

        # 2. Reset the internal state of the openwakeword model. This is crucial
        #    to prevent it from getting stuck after a detection.
        if self.use_wake_words and self.wakeword_backend in {'oww', 'openwakeword', 'openwakewords'}:
            if hasattr(self, 'owwModel'):
                self.owwModel.reset()
                logger.debug("OpenWakeWord model state has been reset.")

        self.is_recording = False
        self.recording_stop_time = time.time()
        self.is_silero_speech_active = False
        self.is_webrtc_speech_active = False
        self.silero_check_time = 0
        self.start_recording_event.clear()
        self.stop_recording_event.set()

        self.last_recording_start_time = self.recording_start_time
        self.last_recording_stop_time = self.recording_stop_time

        if self.on_recording_stop:
            self._run_callback(self.on_recording_stop)

        return self

    def listen(self):
        """
        Puts recorder in immediate "listen" state.
        This is the state after a wake word detection, for example.
        The recorder now "listens" for voice activation.
        Once voice is detected we enter "recording" state.
        """
        self.listen_start = time.time()
        # self._set_state("listening")
        self.start_recording_on_voice_activity = True

    def feed_audio(self, chunk, original_sample_rate=16000):
        """
        Feed an audio chunk into the processing pipeline. Chunks are
        accumulated until the buffer size is reached, and then the accumulated
        data is fed into the audio_queue.
        """
        # Check if the buffer attribute exists, if not, initialize it
        if not hasattr(self, 'buffer'):
            self.buffer = bytearray()

        # Check if input is a NumPy array
        if isinstance(chunk, np.ndarray):
            # Handle stereo to mono conversion if necessary
            if chunk.ndim == 2:
                chunk = np.mean(chunk, axis=1)

            # Resample to 16000 Hz if necessary
            if original_sample_rate != 16000:
                num_samples = int(len(chunk) * 16000 / original_sample_rate)
                chunk = resample(chunk, num_samples)

            # Ensure data type is int16
            chunk = chunk.astype(np.int16)

            # Convert the NumPy array to bytes
            chunk = chunk.tobytes()

        # Append the chunk to the buffer
        self.buffer += chunk
        buf_size = 2 * self.buffer_size  # silero complains if too short

        # Check if the buffer has reached or exceeded the buffer_size
        while len(self.buffer) >= buf_size:
            # Extract self.buffer_size amount of data from the buffer
            to_process = self.buffer[:buf_size]
            self.buffer = self.buffer[buf_size:]

            # Feed the extracted data to the audio_queue
            self.audio_queue.put(to_process)

    def set_microphone(self, microphone_on=True):
        """
        Set the microphone on or off.
        """
        logger.info("Setting microphone to: " + str(microphone_on))
        self.use_microphone.value = microphone_on
    
    def shutdown(self):
        """
        Safely shuts down all components of ListenDrive.
        """
        with self.shutdown_lock:
            if self.is_shut_down:
                logger.info("Shutdown already in progress or completed.")
                return

            logger.info("Initiating ListenDrive shutdown...")
            print("\033[91mListenDrive shutting down...\033[0m") # Or your chosen package name

            # 1. Set flags to signal all threads/processes to stop
            self.is_shut_down = True    # General shutdown flag
            self.is_running = False     # Signals _recording_worker, _realtime_worker, _amplitude_reader_worker
            self.shutdown_event.set()   # Signals _audio_data_worker (process) & _transcription_worker (process)

            # 2. Unblock any blocking calls in main interaction methods
            if hasattr(self, 'start_recording_event'): # Check if attribute exists
                self.start_recording_event.set()
            if hasattr(self, 'stop_recording_event'):
                self.stop_recording_event.set()
            # If self.text() or wait_audio() might be waiting on other events, set them too.

            # 3. Join threads started by the main AudioToTextRecorder process
            # These threads check self.is_running
            logger.debug('Waiting for _recording_worker thread to finish...')
            if hasattr(self, 'recording_thread') and self.recording_thread and self.recording_thread.is_alive():
                self.recording_thread.join(timeout=2.0) # Increased timeout slightly
                if self.recording_thread.is_alive():
                    logger.warning("_recording_worker thread did not join in time.")

            logger.debug('Waiting for _realtime_worker thread to finish...')
            if hasattr(self, 'realtime_thread') and self.realtime_thread and self.realtime_thread.is_alive():
                self.realtime_thread.join(timeout=2.0)
                if self.realtime_thread.is_alive():
                    logger.warning("_realtime_worker thread did not join in time.")

            logger.debug('Waiting for _amplitude_reader_worker thread to finish...')
            if hasattr(self, 'amplitude_reader_thread') and self.amplitude_reader_thread and self.amplitude_reader_thread.is_alive():
                self.amplitude_reader_thread.join(timeout=2.0)
                if self.amplitude_reader_thread.is_alive():
                    logger.warning("_amplitude_reader_worker thread did not join in time.")

            logger.debug('Waiting for _stdout_thread to finish...')
            if hasattr(self, 'stdout_thread') and self.stdout_thread and self.stdout_thread.is_alive():
                # This thread polls a pipe, which will close with the transcription process
                self.stdout_thread.join(timeout=1.0)
                if self.stdout_thread.is_alive():
                    logger.warning("_stdout_thread did not join in time.")


            # 4. Join processes started by AudioToTextRecorder
            # These processes primarily check self.shutdown_event
            logger.debug('Waiting for _audio_data_worker (reader) process to finish...')
            if hasattr(self, 'reader_process') and self.reader_process and self.reader_process.is_alive():
                self.reader_process.join(timeout=5.0) # Give it ample time to close pyaudio
                if self.reader_process.is_alive():
                    logger.warning("_audio_data_worker process did not join in time. Terminating forcefully.")
                    self.reader_process.terminate() # Force terminate if stuck
                    self.reader_process.join(timeout=1.0) # Wait for termination to complete

            logger.debug('Waiting for _transcription_worker process to finish...')
            if hasattr(self, 'transcript_process') and self.transcript_process and self.transcript_process.is_alive():
                # The transcription worker also checks self.shutdown_event and its pipe
                # Attempt to close the pipe from this end too, which might help it exit
                if hasattr(self, 'parent_transcription_pipe'):
                    try:
                        # Sending a None or a specific shutdown message can help the worker exit its recv loop
                        # self.parent_transcription_pipe.send(('shutdown_signal', None))
                        self.parent_transcription_pipe.close() # Close our end
                        logger.debug("Closed parent_transcription_pipe.")
                    except Exception as e:
                        logger.warning(f"Error closing parent_transcription_pipe: {e}")

                self.transcript_process.join(timeout=5.0) # Give time for model cleanup
                if self.transcript_process.is_alive():
                    logger.warning("_transcription_worker process did not join in time. Terminating forcefully.")
                    self.transcript_process.terminate()
                    self.transcript_process.join(timeout=1.0)


            # 5. Close any remaining pipes from the parent side
            if hasattr(self, 'parent_transcription_pipe') and not self.parent_transcription_pipe._closed: # Assuming _closed flag from SafePipe
                try:
                    self.parent_transcription_pipe.close()
                except Exception as e:
                    logger.warning(f"Error on final close of parent_transcription_pipe: {e}")
            if hasattr(self, 'parent_stdout_pipe') and not self.parent_stdout_pipe._closed:
                try:
                    self.parent_stdout_pipe.close()
                except Exception as e:
                    logger.warning(f"Error on final close of parent_stdout_pipe: {e}")


            # 6. Clean up resources like models (if loaded in this main process)
            logger.debug('Cleaning up transcription models...')
            if self.enable_realtime_transcription and not self.use_main_model_for_realtime:
                if hasattr(self, 'realtime_model_type') and self.realtime_model_type:
                    logger.debug("Deleting real-time transcription model instance.")
                    try:
                        # If WhisperModel or BatchedInferencePipeline has specific cleanup, call it.
                        # Otherwise, rely on Python's garbage collection.
                        del self.realtime_model_type
                    except Exception as e:
                        logger.warning(f"Error deleting realtime_model_type: {e}")
                    finally:
                        self.realtime_model_type = None

            if self.speaker_model_instance: del self.speaker_model_instance; self.speaker_model_instance = None
            if self.speaker_inference_instance: del self.speaker_inference_instance; self.speaker_inference_instance = None
            if self.overlap_pipeline_instance: del self.overlap_pipeline_instance; self.overlap_pipeline_instance = None
            # The main transcription model is in a separate process, so its cleanup
            # happens within that process when it exits.

            # 7. Garbage collect
            logger.debug("Performing garbage collection.")
            gc.collect()

            logger.info("ListenDrive shutdown sequence complete.")
            print("\033[92mListenDrive shutdown complete.\033[0m")



    def force_listening_mode(self):
        """
        Forces the engine into the active 'listening' state via an external command.
        This overrides the default contextual logic.
        """
        logger.info("API CALL: Forcing engine into LISTENING mode.")
        self.mode_override = "listening"
        # Directly set the state and the necessary flag. This is the most
        # authoritative way to handle an external command and avoids race conditions.
        self._set_state("listening")
        self.start_recording_on_voice_activity = True

    def force_wakeword_mode(self):
        """
        Forces the engine into the passive 'wakeword' state via an external command.
        This overrides the default contextual logic.
        """
        logger.info("API CALL: Forcing engine into WAKEWORD mode.")
        self.mode_override = "wakeword"
        self._set_state("wakeword") # Set state immediately for responsiveness

    def release_mode_override(self):
        """
        Releases any forced state, allowing the engine to return to its
        default context-aware behavior.
        """
        logger.info("API CALL: Releasing mode override. Returning to automatic context.")
        self.mode_override = None


    def _trim_silence_from_buffer(self, audio_buffer: bytes) -> Optional[bytes]:
        """
        Uses WebRTC VAD to trim leading and trailing silence from an audio buffer.

        Args:
            audio_buffer (bytes): The raw audio data as a byte string.

        Returns:
            Optional[bytes]: The trimmed audio data, or None if no speech is found.
        """
        if not audio_buffer:
            return None

        # WebRTC VAD requires audio chunks of 10, 20, or 30 ms. We'll use 30ms.
        frame_duration_ms = 30
        frame_size = int(self.sample_rate * (frame_duration_ms / 1000.0) * 2) # 2 bytes per sample
        
        num_frames = len(audio_buffer) // frame_size
        if num_frames == 0:
            return None

        # VAD aggressiveness (0=least aggressive, 3=most aggressive)
        vad = webrtcvad.Vad(3)

        # Find the first frame of speech
        first_speech_frame = -1
        for i in range(num_frames):
            frame = audio_buffer[i*frame_size : (i+1)*frame_size]
            if len(frame) < frame_size:
                break
            if vad.is_speech(frame, self.sample_rate):
                first_speech_frame = i
                break
        
        # If no speech was found at all, return None.
        if first_speech_frame == -1:
            return None

        # Find the last frame of speech by checking in reverse.
        last_speech_frame = first_speech_frame
        for i in range(num_frames - 1, first_speech_frame, -1):
            frame = audio_buffer[i*frame_size : (i+1)*frame_size]
            if len(frame) < frame_size:
                continue
            if vad.is_speech(frame, self.sample_rate):
                last_speech_frame = i
                break
        
        # "Chop" the audio from the start of the first speech frame to the end of the last one.
        start_byte = first_speech_frame * frame_size
        end_byte = (last_speech_frame + 1) * frame_size
        
        logger.debug(f"VAD trimmed buffer from {len(audio_buffer)} bytes to {end_byte - start_byte} bytes.")
        return audio_buffer[start_byte:end_byte]


    def _process_buffered_command(self, audio_buffer: list, was_wake_word: bool):
        """
        Takes a buffer of audio frames and sends it directly for full processing.
        This is called by the worker when a wake word is confirmed in a buffer.
        """
        logger.info("Processing buffered audio as a command.")
        self._set_state("transcribing")
        
        # Combine the audio frames into a single byte string.
        raw_bytes = b''.join(audio_buffer)
        
        # Convert to the float32 NumPy array that transcribe() expects.
        audio_np = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / INT16_MAX_ABS_VALUE
        
        # We need to manually set the attributes that the transcribe() method will use.
        self.audio = audio_np
        self.current_segment_raw_bytes = raw_bytes
        
        # Add the 'hot_word_detected' flag to the metadata.
        # This requires a small change to transcribe()
        
        # Directly call the transcription and analysis method.
        # We run this in a new thread so the _recording_worker is not blocked.
        # threading.Thread(target=self.transcribe, args=(was_wake_word,)).start()
        threading.Thread(target=lambda: self.transcribe(was_wake_word_triggered=was_wake_word)).start()

    def _recording_worker(self):
        """
        The main worker method which constantly monitors the audio input and
        manages the engine's state machine (listening, wakeword, recording).
        """
        if self.use_extended_logger:
            logger.debug("Recording worker thread started.")

        # --- Main Loop ---
        while self.is_running:
            try:
                # Block until a new audio chunk is available from the capture process.
                data = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                # If the queue is empty, just continue the loop to check `self.is_running`.
                continue
            except BrokenPipeError:
                logger.error("Pipe to audio capture process was broken. Stopping worker.", exc_info=True)
                self.is_running = False
                break

            if self.on_recorded_chunk:
                self._run_callback(self.on_recorded_chunk, data)

            if self.handle_buffer_overflow and self.audio_queue.qsize() > self.allowed_latency_limit:
                logger.warning(
                    f"Audio queue size ({self.audio_queue.qsize()}) exceeds latency limit. Discarding old chunks."
                )
                while self.audio_queue.qsize() > self.allowed_latency_limit:
                    try: self.audio_queue.get_nowait()
                    except queue.Empty: break

            # ==============================================================================
            # STATE MACHINE LOGIC
            # ==============================================================================

            if self.is_recording:
                # --- State: Actively Recording Speech (from VAD) ---
                self.frames.append(data)
                is_speech = self._is_webrtc_speech(data, all_frames_must_be_true=True)

                if not is_speech:
                    if self.speech_end_silence_start == 0 and \
                       (time.time() - self.recording_start_time > self.min_length_of_recording):
                        self.speech_end_silence_start = time.time()
                else:
                    if self.speech_end_silence_start != 0:
                        self.speech_end_silence_start = 0

                if self.speech_end_silence_start and \
                   (time.time() - self.speech_end_silence_start >= self.post_speech_silence_duration):
                    logger.info("Voice deactivity detected. Stopping recording.")
                    if self.on_vad_stop: self._run_callback(self.on_vad_stop)
                    self.stop()
                    self.speech_end_silence_start = 0
            else:
                # --- State: Idle (Not Recording) ---
                self.audio_buffer.append(data)
                
                # Step 1: Determine the desired next state ONLY if we are in the
                # 'inactive' transition state (i.e., just finished a recording).
                if self.state == "inactive":
                    logger.debug(f"STATE TRANSITION: last_speaker_count={self.last_speaker_count}, mode_override={self.mode_override}")
                    
                    next_state = "listening"
                    if self.mode_override:
                        next_state = self.mode_override
                    elif self.use_wake_words:
                        if self.last_speaker_count > 1:
                            next_state = "wakeword"
                    
                    self._set_state(next_state)

                # Step 2: Execute actions based on the CURRENT state.
                if self.state in ["transcribing", "inactive"]:
                    pass

                elif self.state == "listening":
                    self.start_recording_on_voice_activity = True
                    if self._is_voice_active():
                        logger.info("VAD detected voice activity, starting recording.")
                        if self.on_vad_start: self._run_callback(self.on_vad_start)
                        
                        self.start()
                        self.frames.extend(list(self.audio_buffer))
                        self.audio_buffer.clear()
                        self.silero_vad_model.reset_states()
                    else:
                        self._check_voice_activity(data)

                elif self.state == "wakeword":
                    # <<< --- THIS IS THE FINAL, CORRECTED LOGIC --- >>>
                    is_speech = self._is_silero_speech(data)

                    # If speech is detected, we are either starting or continuing an utterance.
                    if is_speech:
                        # If we weren't already buffering, this is the start of a new potential command.
                        if not self.is_buffering_for_wakeword:
                            self.is_buffering_for_wakeword = True
                            self.potential_command_buffer.clear()
                            # Include the pre-roll audio from the main buffer for context.
                            self.potential_command_buffer.extend(list(self.audio_buffer))
                            logger.debug("Potential command started. Buffering audio in wakeword mode.")
                        
                        # Always append the current speech chunk to our buffer.
                        self.potential_command_buffer.append(data)
                        
                        # Crucially, reset the silence timer because speech is actively happening.
                        self.wakeword_buffer_silence_start = 0

                    # If silence is detected AND we were in the middle of buffering an utterance...
                    elif not is_speech and self.is_buffering_for_wakeword:
                        # Append the silence chunk to the buffer to capture the natural end of the phrase.
                        self.potential_command_buffer.append(data)

                        # If this is the first chunk of silence, start the timer.
                        if self.wakeword_buffer_silence_start == 0:
                            self.wakeword_buffer_silence_start = time.time()
                        
                        # Now, check if the silence has lasted long enough to declare the utterance finished.
                        if time.time() - self.wakeword_buffer_silence_start >= self.post_speech_silence_duration:
                            logger.debug("Utterance ended. Analyzing buffer for wake word...")
                            
                            full_utterance_bytes = b"".join(self.potential_command_buffer)
                            speech_only_bytes = self._trim_silence_from_buffer(full_utterance_bytes)
                            
                            if speech_only_bytes:
                                wakeword_index = self._process_wakeword(speech_only_bytes)
                                if wakeword_index >= 0:
                                    logger.info("Wake word CONFIRMED in buffered utterance. Processing as a command.")
                                    if self.on_wakeword_detected:
                                        self._run_callback(self.on_wakeword_detected)
                                    
                                    self._process_buffered_command(
                                        audio_buffer=self.potential_command_buffer.copy(),
                                        was_wake_word=True
                                    )
                                else:
                                    logger.debug("No wake word found in trimmed speech. Discarding.")
                            else:
                                logger.debug("VAD found no speech in buffer. Discarding.")

                            # IMPORTANT: Reset everything for the next potential command.
                            self.potential_command_buffer.clear()
                            self.is_buffering_for_wakeword = False
                            self.wakeword_buffer_silence_start = 0
                            
        logger.info("Recording worker thread has stopped.")

    def _realtime_worker(self):
        """
        Performs real-time transcription if the feature is enabled.

        The method is responsible transcribing recorded audio frames
          in real-time based on the specified resolution interval.
        The transcribed text is stored in `self.realtime_transcription_text`
          and a callback
        function is invoked with this text if specified.
        """

        try:

            logger.debug('Starting realtime worker')

            # Return immediately if real-time transcription is not enabled
            if not self.enable_realtime_transcription:
                return

            # Track time of last transcription
            last_transcription_time = time.time()

            while self.is_running:

                if self.is_recording:

                    # MODIFIED SLEEP LOGIC:
                    # Wait until realtime_processing_pause has elapsed,
                    # but check often so we can respond to changes quickly.
                    while (
                        time.time() - last_transcription_time
                    ) < self.realtime_processing_pause:
                        time.sleep(0.001)
                        if not self.is_running or not self.is_recording:
                            break

                    if self.awaiting_speech_end:
                        time.sleep(0.001)
                        continue

                    # Update transcription time
                    last_transcription_time = time.time()

                    # Convert the buffer frames to a NumPy array
                    audio_array = np.frombuffer(
                        b''.join(self.frames),
                        dtype=np.int16
                        )

                    logger.debug(f"Current realtime buffer size: {len(audio_array)}")

                    # Normalize the array to a [-1, 1] range
                    audio_array = audio_array.astype(np.float32) / \
                        INT16_MAX_ABS_VALUE

                    if self.use_main_model_for_realtime:
                        with self.transcription_lock:
                            try:
                                self.parent_transcription_pipe.send((audio_array, self.language, True))
                                if self.parent_transcription_pipe.poll(timeout=5):  # Wait for 5 seconds
                                    logger.debug("Receive from realtime worker after transcription request to main model")
                                    status, result = self.parent_transcription_pipe.recv()
                                    if status == 'success':
                                        segments, info = result
                                        self.detected_realtime_language = info.language if info.language_probability > 0 else None
                                        self.detected_realtime_language_probability = info.language_probability
                                        realtime_text = segments
                                        logger.debug(f"Realtime text detected with main model: {realtime_text}")
                                    else:
                                        logger.error(f"Realtime transcription error: {result}")
                                        continue
                                else:
                                    logger.warning("Realtime transcription timed out")
                                    continue
                            except Exception as e:
                                logger.error(f"Error in realtime transcription: {str(e)}", exc_info=True)
                                continue
                    else:
                        # Perform transcription and assemble the text
                        if self.normalize_audio:
                            # normalize audio to -0.95 dBFS
                            if audio_array is not None and audio_array.size > 0:
                                peak = np.max(np.abs(audio_array))
                                if peak > 0:
                                    audio_array = (audio_array / peak) * 0.95

                        if self.realtime_batch_size > 0:
                            segments, info = self.realtime_model_type.transcribe(
                                audio_array,
                                language=self.language if self.language else None,
                                beam_size=self.beam_size_realtime,
                                initial_prompt=self.initial_prompt_realtime,
                                suppress_tokens=self.suppress_tokens,
                                batch_size=self.realtime_batch_size,
                                vad_filter=self.faster_whisper_vad_filter
                            )
                        else:
                            segments, info = self.realtime_model_type.transcribe(
                                audio_array,
                                language=self.language if self.language else None,
                                beam_size=self.beam_size_realtime,
                                initial_prompt=self.initial_prompt_realtime,
                                suppress_tokens=self.suppress_tokens,
                                vad_filter=self.faster_whisper_vad_filter
                            )

                        self.detected_realtime_language = info.language if info.language_probability > 0 else None
                        self.detected_realtime_language_probability = info.language_probability
                        realtime_text = " ".join(
                            seg.text for seg in segments
                        )
                        logger.debug(f"Realtime text detected: {realtime_text}")

                    # double check recording state
                    # because it could have changed mid-transcription
                    if self.is_recording and time.time() - \
                            self.recording_start_time > self.init_realtime_after_seconds:

                        self.realtime_transcription_text = realtime_text
                        self.realtime_transcription_text = \
                            self.realtime_transcription_text.strip()

                        self.text_storage.append(
                            self.realtime_transcription_text
                            )

                        # Take the last two texts in storage, if they exist
                        if len(self.text_storage) >= 2:
                            last_two_texts = self.text_storage[-2:]

                            # Find the longest common prefix
                            # between the two texts
                            prefix = os.path.commonprefix(
                                [last_two_texts[0], last_two_texts[1]]
                                )

                            # This prefix is the text that was transcripted
                            # two times in the same way
                            # Store as "safely detected text"
                            if len(prefix) >= \
                                    len(self.realtime_stabilized_safetext):

                                # Only store when longer than the previous
                                # as additional security
                                self.realtime_stabilized_safetext = prefix

                        # Find parts of the stabilized text
                        # in the freshly transcripted text
                        matching_pos = self._find_tail_match_in_text(
                            self.realtime_stabilized_safetext,
                            self.realtime_transcription_text
                            )

                        if matching_pos < 0:
                            # pick which text to send
                            text_to_send = (
                                self.realtime_stabilized_safetext
                                if self.realtime_stabilized_safetext
                                else self.realtime_transcription_text
                            )
                            # preprocess once
                            processed = self._preprocess_output(text_to_send, True)
                            # invoke on its own thread
                            self._run_callback(self._on_realtime_transcription_stabilized, processed)

                        else:
                            # We found parts of the stabilized text
                            # in the transcripted text
                            # We now take the stabilized text
                            # and add only the freshly transcripted part to it
                            output_text = self.realtime_stabilized_safetext + \
                                self.realtime_transcription_text[matching_pos:]

                            # This yields us the "left" text part as stabilized
                            # AND at the same time delivers fresh detected
                            # parts on the first run without the need for
                            # two transcriptions
                            self._run_callback(self._on_realtime_transcription_stabilized, self._preprocess_output(output_text, True))

                        # Invoke the callback with the transcribed text
                        self._run_callback(self._on_realtime_transcription_update, self._preprocess_output(self.realtime_transcription_text,True))

                # If not recording, sleep briefly before checking again
                else:
                    time.sleep(TIME_SLEEP)

        except Exception as e:
            logger.error(f"Unhandled exeption in _realtime_worker: {e}", exc_info=True)
            raise

    def _is_silero_speech(self, chunk):
        """
        Returns true if speech is detected in the provided audio data
        """
        if self.sample_rate != 16000:
            pcm_data = np.frombuffer(chunk, dtype=np.int16)
            data_16000 = signal.resample_poly(
                pcm_data, 16000, self.sample_rate)
            chunk = data_16000.astype(np.int16).tobytes()

        self.silero_working = True
        audio_chunk = np.frombuffer(chunk, dtype=np.int16)
        audio_chunk = audio_chunk.astype(np.float32) / INT16_MAX_ABS_VALUE
        

        t_start = time.perf_counter()
        
        vad_prob = self.silero_vad_model(
            torch.from_numpy(audio_chunk),
            SAMPLE_RATE).item()
        

        t_end = time.perf_counter()
        if hasattr(self, 'vad_perf_log'):
            self.vad_perf_log.append(t_end - t_start)
            
        is_silero_speech_active = vad_prob > (1 - self.silero_sensitivity)
        if is_silero_speech_active:
            if not self.is_silero_speech_active and self.use_extended_logger:
                logger.info(f"{bcolors.OKGREEN}Silero VAD detected speech{bcolors.ENDC}")
        elif self.is_silero_speech_active and self.use_extended_logger:
            logger.info(f"{bcolors.WARNING}Silero VAD detected silence{bcolors.ENDC}")
        self.is_silero_speech_active = is_silero_speech_active
        self.silero_working = False
        return is_silero_speech_active

    def _is_webrtc_speech(self, chunk, all_frames_must_be_true=False):
        """
        Returns true if speech is detected in the provided audio data
        """

        t_start = time.perf_counter()

        speech_str = f"{bcolors.OKGREEN}WebRTC VAD detected speech{bcolors.ENDC}"
        silence_str = f"{bcolors.WARNING}WebRTC VAD detected silence{bcolors.ENDC}"
        if self.sample_rate != 16000:
            pcm_data = np.frombuffer(chunk, dtype=np.int16)
            data_16000 = signal.resample_poly(
                pcm_data, 16000, self.sample_rate)
            chunk = data_16000.astype(np.int16).tobytes()

        # Number of audio frames per millisecond
        frame_length = int(16000 * 0.01)  # for 10ms frame
        num_frames = int(len(chunk) / (2 * frame_length))
        speech_frames = 0

        for i in range(num_frames):
            start_byte = i * frame_length * 2
            end_byte = start_byte + frame_length * 2
            frame = chunk[start_byte:end_byte]
            if self.webrtc_vad_model.is_speech(frame, 16000):
                speech_frames += 1
                if not all_frames_must_be_true:
                    if self.debug_mode:
                        logger.info(f"Speech detected in frame {i + 1} of {num_frames}")
                    if not self.is_webrtc_speech_active and self.use_extended_logger:
                        logger.info(speech_str)
                    self.is_webrtc_speech_active = True
                    

                    t_end = time.perf_counter()
                    if hasattr(self, 'vad_perf_log'): self.vad_perf_log.append(t_end - t_start)
                    
                    return True


        t_end = time.perf_counter()
        if hasattr(self, 'vad_perf_log'): self.vad_perf_log.append(t_end - t_start)

        if all_frames_must_be_true:
            if self.debug_mode and speech_frames == num_frames:
                logger.info(f"Speech detected in {speech_frames} of {num_frames} frames")
            elif self.debug_mode:
                logger.info(f"Speech not detected in all {num_frames} frames")
            speech_detected = speech_frames == num_frames
            if speech_detected and not self.is_webrtc_speech_active and self.use_extended_logger:
                logger.info(speech_str)
            elif not speech_detected and self.is_webrtc_speech_active and self.use_extended_logger:
                logger.info(silence_str)
            self.is_webrtc_speech_active = speech_detected
            return speech_detected
        else:
            if self.debug_mode:
                logger.info(f"Speech not detected in any of {num_frames} frames")
            if self.is_webrtc_speech_active and self.use_extended_logger:
                logger.info(silence_str)
            self.is_webrtc_speech_active = False
            return False


    def _check_voice_activity(self, data):
        """
        Initiate check if voice is active based on the provided data.

        Args:
            data: The audio data to be checked for voice activity.
        """
        self._is_webrtc_speech(data)

        # First quick performing check for voice activity using WebRTC
        if self.is_webrtc_speech_active:

            if not self.silero_working:
                self.silero_working = True

                # Run the intensive check in a separate thread
                threading.Thread(
                    target=self._is_silero_speech,
                    args=(data,)).start()

    def clear_audio_queue(self):
        """
        Safely empties the audio queue to ensure no remaining audio 
        fragments get processed e.g. after waking up the recorder.
        """
        self.audio_buffer.clear()
        try:
            while True:
                self.audio_queue.get_nowait()
        except:
            # PyTorch's mp.Queue doesn't have a specific Empty exception
            # so we catch any exception that might occur when the queue is empty
            pass

    def _is_voice_active(self):
        """
        Determine if voice is active.

        Returns:
            bool: True if voice is active, False otherwise.
        """
        return self.is_webrtc_speech_active and self.is_silero_speech_active

    def _set_state(self, new_state):
        """
        Update the current state of the recorder and execute
        corresponding state-change callbacks.

        Args:
            new_state (str): The new state to set.

        """
        # Check if the state has actually changed
        if new_state == self.state:
            return

        # Store the current state for later comparison
        old_state = self.state

        # Update to the new state
        self.state = new_state

        # Log the state change
        logger.info(f"State changed from '{old_state}' to '{new_state}'")

        # Execute callbacks based on transitioning FROM a particular state
        if old_state == "listening":
            if self.on_vad_detect_stop:
                self._run_callback(self.on_vad_detect_stop)
        elif old_state == "wakeword":
            if self.on_wakeword_detection_end:
                self._run_callback(self.on_wakeword_detection_end)

        # Execute callbacks based on transitioning TO a particular state
        if new_state == "listening":
            if self.on_vad_detect_start:
                self._run_callback(self.on_vad_detect_start)
            self._set_spinner("speak now")
            if self.spinner and self.halo:
                self.halo._interval = 250
        elif new_state == "wakeword":
            if self.on_wakeword_detection_start:
                self._run_callback(self.on_wakeword_detection_start)
            self._set_spinner(f"say {self.wake_words}")
            if self.spinner and self.halo:
                self.halo._interval = 500
        elif new_state == "transcribing":
            self._set_spinner("transcribing")
            if self.spinner and self.halo:
                self.halo._interval = 50
        elif new_state == "recording":
            self._set_spinner("recording")
            if self.spinner and self.halo:
                self.halo._interval = 100
        elif new_state == "inactive":
            if self.spinner and self.halo:
                self.halo.stop()
                self.halo = None

    def _set_spinner(self, text):
        """
        Update the spinner's text or create a new
        spinner with the provided text.

        Args:
            text (str): The text to be displayed alongside the spinner.
        """
        if self.spinner:
            # If the Halo spinner doesn't exist, create and start it
            if self.halo is None:
                self.halo = halo.Halo(text=text)
                self.halo.start()
            # If the Halo spinner already exists, just update the text
            else:
                self.halo.text = text

    def _preprocess_output(self, text, preview=False):
        """
        Preprocesses the output text by removing any leading or trailing
        whitespace, converting all whitespace sequences to a single space
        character, and capitalizing the first character of the text.

        Args:
            text (str): The text to be preprocessed.

        Returns:
            str: The preprocessed text.
        """
        text = re.sub(r'\s+', ' ', text.strip())

        if self.ensure_sentence_starting_uppercase:
            if text:
                text = text[0].upper() + text[1:]

        # Ensure the text ends with a proper punctuation
        # if it ends with an alphanumeric character
        if not preview:
            if self.ensure_sentence_ends_with_period:
                if text and text[-1].isalnum():
                    text += '.'

        return text

    def _find_tail_match_in_text(self, text1, text2, length_of_match=10):
        """
        Find the position where the last 'n' characters of text1
        match with a substring in text2.

        This method takes two texts, extracts the last 'n' characters from
        text1 (where 'n' is determined by the variable 'length_of_match'), and
        searches for an occurrence of this substring in text2, starting from
        the end of text2 and moving towards the beginning.

        Parameters:
        - text1 (str): The text containing the substring that we want to find
          in text2.
        - text2 (str): The text in which we want to find the matching
          substring.
        - length_of_match(int): The length of the matching string that we are
          looking for

        Returns:
        int: The position (0-based index) in text2 where the matching
          substring starts. If no match is found or either of the texts is
          too short, returns -1.
        """

        # Check if either of the texts is too short
        if len(text1) < length_of_match or len(text2) < length_of_match:
            return -1

        # The end portion of the first text that we want to compare
        target_substring = text1[-length_of_match:]

        # Loop through text2 from right to left
        for i in range(len(text2) - length_of_match + 1):
            # Extract the substring from text2
            # to compare with the target_substring
            current_substring = text2[len(text2) - i - length_of_match:
                                      len(text2) - i]

            # Compare the current_substring with the target_substring
            if current_substring == target_substring:
                # Position in text2 where the match starts
                return len(text2) - i

        return -1

    def _on_realtime_transcription_stabilized(self, text):
        """
        Callback method invoked when the real-time transcription stabilizes.

        This method is called internally when the transcription text is
        considered "stable" meaning it's less likely to change significantly
        with additional audio input. It notifies any registered external
        listener about the stabilized text if recording is still ongoing.
        This is particularly useful for applications that need to display
        live transcription results to users and want to highlight parts of the
        transcription that are less likely to change.

        Args:
            text (str): The stabilized transcription text.
        """
        if self.on_realtime_transcription_stabilized:
            if self.is_recording:
                self._run_callback(self.on_realtime_transcription_stabilized, text)

    def _on_realtime_transcription_update(self, text):
        """
        Callback method invoked when there's an update in the real-time
        transcription.

        This method is called internally whenever there's a change in the
        transcription text, notifying any registered external listener about
        the update if recording is still ongoing. This provides a mechanism
        for applications to receive and possibly display live transcription
        updates, which could be partial and still subject to change.

        Args:
            text (str): The updated transcription text.
        """
        if self.on_realtime_transcription_update:
            if self.is_recording:
                self._run_callback(self.on_realtime_transcription_update, text)

    def __enter__(self):
        """
        Method to setup the context manager protocol.

        This enables the instance to be used in a `with` statement, ensuring
        proper resource management. When the `with` block is entered, this
        method is automatically called.

        Returns:
            self: The current instance of the class.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Method to define behavior when the context manager protocol exits.

        This is called when exiting the `with` block and ensures that any
        necessary cleanup or resource release processes are executed, such as
        shutting down the system properly.

        Args:
            exc_type (Exception or None): The type of the exception that
              caused the context to be exited, if any.
            exc_value (Exception or None): The exception instance that caused
              the context to be exited, if any.
            traceback (Traceback or None): The traceback corresponding to the
              exception, if any.
        """
        self.shutdown()