# SpeechIntentRecognizer System: Professional Report 

## 1. Introduction
The `SpeechIntentRecognizer` system is a modular Python application designed to perform real-time speech recognition and intent classification. It captures audio input from your microphone, converts it into text using the `faster-whisper` model, and determines the intent using a zero-shot classification model from the `transformers` library. The system is neatly organised into five modules: `config.py`, `utils.py`, `speech_recognizer.py`, `intent_classifier.py`, and `main.py`. This report provides a detailed explanation of the required packages, their installation process, and a line-by-line workflow analysis.

---

## 2. Required Packages

This system depends on several Python packages to manage audio processing, speech recognition, intent classification, and overall system coordination. Below is the list of packages, their purposes, and how to install them.

### 2.1 Packages and Their Purposes
1. **`numpy`**  
   - **Purpose**: Handles numerical operations for audio buffer manipulation.
   - **Version**: Latest available (e.g., 1.26.4 as of March 2025).

2. **`torch`**  
   - **Purpose**: Core library for loading and running the Silero VAD model.
   - **Version**: Compatible with your system (e.g., 2.2.0).

3. **`torchaudio`**  
   - **Purpose**: Provides audio processing utilities required by Silero VAD.
   - **Version**: Must align with `torch` (e.g., 2.2.0).

4. **`faster-whisper`**  
   - **Purpose**: An efficient version of OpenAI’s Whisper for speech-to-text transcription.
   - **Version**: Latest (e.g., 0.10.0).

5. **`pyaudio`**  
   - **Purpose**: Interfaces with the microphone for real-time audio capture.
   - **Version**: Latest (e.g., 0.2.14).

6. **`pygame`**  
   - **Purpose**: Plays a beep sound to signal speech detection.
   - **Version**: Latest (e.g., 2.5.2).

7. **`transformers`**  
   - **Purpose**: Offers the zero-shot classification pipeline for intent recognition.
   - **Version**: Latest (e.g., 4.38.2).

8. **`asyncio`, `threading`, `logging`**  
   - **Purpose**: Standard Python modules for asynchronous processing, multi-threading, and logging.
   - **Version**: Included with Python (3.8 or higher recommended).

### 2.2 Installation Instructions
To install all these packages, run this command in your terminal or command prompt:
```bash
pip install numpy torch torchaudio faster-whisper pyaudio pygame transformers
```

#### Additional Setup
- **System Dependencies**:
  - **Windows**: Ensure Microsoft Visual C++ Build Tools are installed for `pyaudio`.
  - **Linux**: Install `portaudio` using `sudo apt-get install portaudio19-dev`.
  - **macOS**: Install `portaudio` with `brew install portaudio`.
- **Model Files**:
  - **Speech Recognition**: Place `faster-whisper` model files in `./Res/LLMs/Speech_Recognition/faster_whisper_tiny_en/`.
  - **VAD**: Store Silero VAD files in `./Res/VAD_Models/silero_vad`.
  - **Intent Classification**: Keep `nli-MiniLM2-L6-H768` in `D:/Project_Files/Destiny/Destiny_Dev/Res/LLMs/Encoder/`.
  - Download from respective repositories (e.g., Hugging Face) if not already available.
- **Verification**: Test the imports with:
  ```python
  import numpy, torch, torchaudio, faster_whisper, pyaudio, pygame, transformers
  print("All packages installed successfully, no issues!")
  ```

---

## 3. System Architecture
The system is divided into five files, each with a specific role:
- **`config.py`**: Holds all configuration settings and constants.
- **`utils.py`**: Provides utility functions for logging and formatting output.
- **`speech_recognizer.py`**: Manages audio capture and transcription.
- **`intent_classifier.py`**: Performs intent classification on transcribed text.
- **`main.py`**: Coordinates the entire workflow.

---

## 4. Workflow and Line-by-Line Explanation

### 4.1 `config.py` - Configuration Module
This module defines all the constants and settings used throughout the system.

```python
import os  # Assists in managing files and folders on the computer.
```
- Imports the `os` module to deal with file paths and environment variables.

```python
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "handlers": [
        {"type": "FileHandler", "filename": "speech_recognition.log"},
        {"type": "StreamHandler"}
    ]
}
```
- Configures logging: `INFO` level for normal details, a timestamped format, and output to both a file and the console.

```python
SRMODEL_PATH = "./Res/LLMs/Speech_Recognition/faster_whisper_tiny_en/"
LOCAL_SILERO_VAD_PATH = './Res/VAD_Models/silero_vad'
SR_CALIB_AUDIO_FILES_PATH = "./Res/Audio_Files/recording.wav"
RECORDER_SAMPLE_RATE = 16000
SR_INPUT_DEVICE = 1
CHUNK_SIZE = 4096
FORMAT = "paInt16"
CHANNELS = 1
RATE = RECORDER_SAMPLE_RATE
MICROPHONE_GAIN = 1.2
NOISE_THRESHOLD = 300
PRE_ROLL_SAMPLES = int(0.3 * RECORDER_SAMPLE_RATE)
```
- Specifies paths for models and a calibration audio file, along with audio settings: 16kHz sample rate, device index 1, 4096-sample chunks, 20% gain, noise threshold at 300, and 0.3 seconds of pre-roll audio.

```python
SR_COMPUTE_DEVICE = "cpu"
SR_COMPUTE_TYPE = "int8"
SR_LANGUAGE = "en"
SR_VAD_FILTER = True
SR_VAD_FILTER_PARAMETERS = {"min_silence_duration_ms": 500, "min_speech_duration_ms": 200}
MODEL_TEMPERATURE = 0.4
AUDIO_OUTPUT_DIR = "./listen/recorded_speech_segments"
SAVE_AUDIO_FILES = True
BEEP_FILE = "./listen/verifier.wav"
BEEP_DURATION = 0.1
```
- Sets computation to CPU with int8 type, English language, VAD filtering (500ms silence, 200ms speech), model temperature at 0.4, audio output directory, saves audio files, and uses a 0.1-second beep.

```python
INTENT_MODEL_PATH = "D:/Project_Files/Destiny/Destiny_Dev/Res/LLMs/Encoder/nli-MiniLM2-L6-H768"
LABELS_LEVEL_ONE = ["Conversation Oriented", "Task Oriented", "Entertainment"]
LABELS_LEVEL_TWO = {
    "Conversation Oriented": ["Greetings", "Farewell", "Gratitude", "Assistance", "Well-being", "Self-assessment", "Emotional-support", "Other"],
    "Task Oriented": ["System Control", "Reminder", "Search", "Information", "Navigation", "Communication", "Other"],
    "Entertainment": ["Music", "Movie", "Games"]
}
LABELS_LEVEL_THREE = {
    "Greetings": ["Greeting", "Small-talk"],
    "Level 2-A2": ["Level 3-A2-1", "Level 3-A2-2", "Other"],
    "Level 2-B1": ["Level 3-B1-1", "Level 3-B1-2", "Other"],
    "Level 2-B2": ["Level 3-B2-1", "Level 3-B2-2", "Other"],
}
```
- Defines the intent model path and hierarchical intent labels across three levels.

```python
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```
- Sets an environment variable to prevent duplicate library errors with Intel MKL.

---

### 4.2 `utils.py` - Utility Module
This module offers helper functions for the system.

```python
import logging
```
- Imports the `logging` module for system logging.

```python
def setup_logging(config):
    """Set up logging based on configuration."""
    handlers = []
    for handler in config["handlers"]:
        if handler["type"] == "FileHandler":
            handlers.append(logging.FileHandler(handler["filename"]))
        elif handler["type"] == "StreamHandler":
            handlers.append(logging.StreamHandler())
    
    logging.basicConfig(
        level=getattr(logging, config["level"]),
        format=config["format"],
        handlers=handlers
    )
    return logging.getLogger("SpeechRecognition")
```
- Configures logging with file and console handlers based on `LOGGING_CONFIG`, returning a logger named "SpeechRecognition".

```python
def print_intent_details(level, intent_result):
    """Print detailed intent classification results in a clean format."""
    predicted_label = intent_result['labels'][0]
    predicted_score = intent_result['scores'][0]
    
    print(f"\n{'=' * 50}")
    print(f"Level {level} Classification")
    print(f"{'=' * 50}")
    print(f"Input: '{intent_result['sequence']}'")
    print(f"Predicted Intent: {predicted_label}")
    print(f"Confidence Score: {predicted_score:.4f}")
    print("\nAll Possible Intents:")
    print("-" * 20)
    for label, score in zip(intent_result['labels'], intent_result['scores']):
        print(f"{label:<20} | {score:.4f}")
    print(f"{'=' * 50}")
```
- Displays intent classification results neatly, showing the predicted label, score, and all possible intents.

---

### 4.3 `speech_recognizer.py` - Speech Recognition Module
This module handles audio capture and transcription.

```python
import io
import threading
import time
import os
import numpy as np
import torch
import asyncio
from queue import Queue
from faster_whisper import WhisperModel
import pyaudio
import wave
import pygame.mixer
from config import *
```
- Imports libraries needed for audio processing, threading, and model handling.

```python
class SpeechRecognizer:
    """A class for real-time speech recognition."""
    
    def __init__(self, logger):
        """Initialize the speech recognizer."""
        self.logger = logger
        self.data_queue = Queue()
        self.pyaudio = None
        self.stream = None
        self._init_pyaudio()
        self.load_vad_model()
        self.audio_model = self.load_sr_model()
        os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
        self.logger.info(f"Audio output directory ensured: {AUDIO_OUTPUT_DIR}")
        
        self.transcription = ['']
        self.listening = False
        self.running = True
        self.state_lock = threading.Lock()
        self.audio_buffer = np.array([], dtype=np.int16)
        self.buffer_lock = threading.Lock()
        self.buffer_max_size = RECORDER_SAMPLE_RATE * 5
        self.last_end_time = 0
        self.silence_duration = 0
        self._init_beep_sound()
```
- Initialises the class with a logger, audio queue, PyAudio setup, VAD and Whisper models, creates an output directory, and sets up transcription storage, state flags, locks, and a 5-second buffer.

```python
    def _init_pyaudio(self):
        try:
            self.pyaudio = pyaudio.PyAudio()
            for i in range(self.pyaudio.get_device_count()):
                dev = self.pyaudio.get_device_info_by_index(i)
                self.logger.info(f"Device {i}: {dev['name']}, Input Channels: {dev['maxInputChannels']}")
        except Exception as e:
            self.logger.error(f"Failed to initialize PyAudio: {e}")
            raise RuntimeError(f"Audio initialization failed: {str(e)}")
```
- Initialises PyAudio and logs details of available audio devices.

```python
    def _init_beep_sound(self):
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=1)
            if not os.path.exists(BEEP_FILE):
                raise FileNotFoundError(f"Beep sound file not found at {BEEP_FILE}")
            self.beep_sound = pygame.mixer.Sound(BEEP_FILE)
            self.beep_sound.set_volume(1.0)
            self.logger.info("Beep sound loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize pygame mixer or load beep sound: {e}")
            raise RuntimeError(f"Pygame initialization failed: {str(e)}")
```
- Sets up `pygame.mixer` and loads the beep sound file.

```python
    def load_vad_model(self):
        try:
            self.vad_model, utils = torch.hub.load(repo_or_dir=LOCAL_SILERO_VAD_PATH, model='silero_vad', source='local')
            (self.get_speech_timestamps, self.save_audio, self.read_audio,
             self.VADIterator, self.collect_chunks) = utils
            self.logger.info("VAD model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading VAD model: {e}")
            raise RuntimeError(f"Failed to load VAD model: {str(e)}")
```
- Loads the Silero VAD model and its utility functions.

```python
    def load_sr_model(self):
        try:
            model = WhisperModel(SRMODEL_PATH, device=SR_COMPUTE_DEVICE, compute_type=SR_COMPUTE_TYPE)
            self.logger.info(f"Speech recognition model loaded successfully from {SRMODEL_PATH}")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load speech recognition model: {e}")
            raise RuntimeError(f"Failed to load speech recognition model: {str(e)}")
```
- Loads the `faster-whisper` model for transcription.

```python
    def apply_gain_and_noise_suppression(self, audio_data: np.ndarray) -> np.ndarray:
        try:
            audio_data = audio_data * MICROPHONE_GAIN
            audio_data = np.clip(audio_data, -32768, 32767).astype(np.int16)
            audio_data[np.abs(audio_data) < NOISE_THRESHOLD] = 0
            return audio_data
        except Exception as e:
            self.logger.error(f"Error in gain/noise suppression: {e}")
            return audio_data
```
- Applies gain to amplify audio and suppresses noise below the threshold.

```python
    def audio_callback(self, in_data, frame_count, time_info, status_flags):
        try:
            if status_flags:
                self.logger.warning(f"Audio callback status: {status_flags}")
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            audio_data = self.apply_gain_and_noise_suppression(audio_data)
            with self.buffer_lock:
                self.audio_buffer = np.append(self.audio_buffer, audio_data)
                if len(self.audio_buffer) > self.buffer_max_size:
                    self.audio_buffer = self.audio_buffer[-self.buffer_max_size:]
                chunk_start_time = time.time() - (len(self.audio_buffer) / RECORDER_SAMPLE_RATE)
                timestamps = self.get_speech_timestamps(
                    self.audio_buffer, self.vad_model, sampling_rate=RECORDER_SAMPLE_RATE,
                    min_silence_duration_ms=SR_VAD_FILTER_PARAMETERS["min_silence_duration_ms"],
                    min_speech_duration_ms=SR_VAD_FILTER_PARAMETERS["min_speech_duration_ms"]
                )
                if timestamps:
                    self.silence_duration = 0
                    for ts in timestamps:
                        start_sample = max(0, ts['start'] - PRE_ROLL_SAMPLES)
                        end_sample = ts['end']
                        if end_sample < len(self.audio_buffer) - CHUNK_SIZE:
                            start_time = chunk_start_time + (start_sample / RECORDER_SAMPLE_RATE)
                            if start_time > self.last_end_time:
                                speech_array = self.audio_buffer[start_sample:end_sample]
                                speech_data = speech_array.tobytes()
                                end_time = chunk_start_time + (end_sample / RECORDER_SAMPLE_RATE)
                                self.data_queue.put({'audio': speech_data, 'start_time': start_time, 'end_time': end_time})
                                self.logger.info(f"Speech detected: {start_time:.2f}s to {end_time:.2f}s")
                                self.last_end_time = end_time
                else:
                    self.silence_duration += CHUNK_SIZE / RECORDER_SAMPLE_RATE
                    # ... (similar logic for finalizing speech segments)
            return (None, pyaudio.paContinue)
        except Exception as e:
            self.logger.error(f"Error in audio callback: {e}")
            return (None, pyaudio.paContinue)
```
- Processes microphone input, applies gain and noise suppression, detects speech with VAD, and queues segments.

```python
    def audio_data_to_wav_bytes(self, audio_data, sample_rate=RECORDER_SAMPLE_RATE, save_to_file: bool = False):
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"speech_segment_{timestamp}.wav"
            filepath = os.path.join(AUDIO_OUTPUT_DIR, filename) if save_to_file else None
            if save_to_file:
                with wave.open(filepath, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_data)
                self.logger.info(f"Audio segment saved to {filepath}")
            wav_io = io.BytesIO()
            with wave.open(wav_io, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data)
            wav_io.seek(0)
            return wav_io, filename
        except Exception as e:
            self.logger.error(f"Error converting audio data to WAV: {e}")
            return None, None
```
- Converts audio data into WAV format and saves it if specified.

```python
    async def play_beep_async(self):
        try:
            self.beep_sound.play()
            await asyncio.sleep(BEEP_DURATION)
            self.logger.debug("Beep sound played successfully")
        except Exception as e:
            self.logger.error(f"Error playing beep sound: {e}")
        finally:
            while pygame.mixer.get_busy():
                await asyncio.sleep(0.01)
            pygame.mixer.stop()
```
- Plays a beep sound asynchronously to indicate speech detection.

```python
    def calibrate_audio_model(self):
        self.logger.info("Calibrating Speech Model...")
        try:
            if not os.path.exists(SR_CALIB_AUDIO_FILES_PATH):
                self.logger.warning(f"Calibration file not found at {SR_CALIB_AUDIO_FILES_PATH}, skipping calibration")
                return
            cal_segments, cal_info = self.audio_model.transcribe(SR_CALIB_AUDIO_FILES_PATH)
            for segment in cal_segments:
                self.logger.info("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            self.logger.info("Calibration Complete... Ready To Listen.")
        except Exception as e:
            self.logger.error(f"Error during model calibration: {e}")
            self.logger.warning("Continuing despite calibration error")
```
- Calibrates the Whisper model using a sample audio file.

```python
    def start_listening(self):
        with self.state_lock:
            if not self.listening:
                try:
                    device_info = self.pyaudio.get_device_info_by_index(SR_INPUT_DEVICE)
                    if device_info['maxInputChannels'] < CHANNELS:
                        raise ValueError(f"Device {SR_INPUT_DEVICE} does not support {CHANNELS} channels")
                    if not self.pyaudio.is_format_supported(RATE, input_device=SR_INPUT_DEVICE, 
                                                          input_channels=CHANNELS, input_format=getattr(pyaudio, FORMAT)):
                        raise ValueError(f"Device {SR_INPUT_DEVICE} does not support {RATE}Hz, {FORMAT}")
                    self.stream = self.pyaudio.open(
                        format=getattr(pyaudio, FORMAT),
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=SR_INPUT_DEVICE,
                        frames_per_buffer=CHUNK_SIZE,
                        stream_callback=self.audio_callback
                    )
                    self.stream.start_stream()
                    self.listening = True
                    self.logger.info("Listening Started")
                except Exception as e:
                    self.logger.error(f"Failed to start audio stream: {e}")
                    raise RuntimeError(f"Failed to start listening: {str(e)}")
```
- Initiates the audio stream for real-time listening.

```python
    def pause_listening(self):
        with self.state_lock:
            if self.stream and self.listening:
                try:
                    self.stream.stop_stream()
                    self.listening = False
                    self.logger.info("Listening Paused")
                except Exception as e:
                    self.logger.error(f"Error pausing stream: {e}")
```
- Pauses the audio stream.

```python
    def resume_listening(self):
        with self.state_lock:
            if self.stream and not self.listening:
                try:
                    self.stream.start_stream()
                    self.listening = True
                    self.logger.info("Listening Resumed")
                except Exception as e:
                    self.logger.error(f"Error resuming stream: {e}")
```
- Resumes the audio stream.

```python
    def stop_listening(self):
        with self.state_lock:
            self.running = False
            if self.stream:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                    self.listening = False
                except Exception as e:
                    self.logger.error(f"Error stopping stream: {e}")
            if self.pyaudio:
                try:
                    self.pyaudio.terminate()
                except Exception as e:
                    self.logger.error(f"Error terminating PyAudio: {e}")
            try:
                pygame.mixer.quit()
                self.logger.info("Pygame mixer stopped")
            except Exception as e:
                self.logger.error(f"Error stopping pygame mixer: {e}")
            self.logger.info("Listening Stopped")
```
- Stops the audio stream and performs cleanup.

---

### 4.4 `intent_classifier.py` - Intent Classification Module
This module classifies the intent of transcribed text.

```python
from transformers import pipeline
from config import *
```
- Imports the classification pipeline and configuration settings.

```python
class IntentClassifier:
    """A class for intent classification."""
    
    def __init__(self, logger):
        """Initialize the intent classifier."""
        self.logger = logger
        try:
            self.classifier = pipeline("zero-shot-classification", model=INTENT_MODEL_PATH)
            self.logger.info("Intent classifier initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize intent classifier: {e}")
            raise RuntimeError(f"Intent classifier initialization failed: {str(e)}")
```
- Initialises the classifier with the specified model.

```python
    def predict_intent(self, user_input, labels):
        """Predict intent for given input and labels."""
        try:
            return self.classifier(user_input, labels)
        except Exception as e:
            self.logger.error(f"Intent prediction error: {e}")
            return None
```
- Predicts the intent from the input text using the provided labels.

---

### 4.5 `main.py` - Main Execution Module
This module integrates and executes the system.

```python
import asyncio
import threading
import time
import traceback
from speech_recognizer import SpeechRecognizer
from intent_classifier import IntentClassifier
from utils import setup_logging, print_intent_details
from config import *
```
- Imports necessary modules and classes.

```python
class SpeechIntentRecognizer:
    """A class integrating speech recognition and intent classification."""
    
    def __init__(self):
        """Initialize the integrated system."""
        self.logger = setup_logging(LOGGING_CONFIG)
        self.speech_recognizer = SpeechRecognizer(self.logger)
        self.intent_classifier = IntentClassifier(self.logger)
```
- Initialises the system with logging, speech recognition, and intent classification components.

```python
    async def process_audio_data(self):
        """Process audio data for transcription and intent classification."""
        while self.speech_recognizer.running:
            try:
                if not self.speech_recognizer.data_queue.empty():
                    item = self.speech_recognizer.data_queue.get()
                    if not isinstance(item, dict) or 'audio' not in item:
                        self.logger.warning(f"Invalid item in queue: {type(item)}")
                        continue
                    speech_data = item['audio']
                    start_time = item.get('start_time', 0)
                    end_time = item.get('end_time', 0)
                    wav_data, saved_filename = self.speech_recognizer.audio_data_to_wav_bytes(speech_data, save_to_file=SAVE_AUDIO_FILES)
                    if not wav_data or not saved_filename:
                        self.logger.warning("Failed to convert audio to WAV format")
                        continue
                    try:
                        asyncio.create_task(self.speech_recognizer.play_beep_async())
                        wav_data.seek(0)
                        segments, info = self.speech_recognizer.audio_model.transcribe(
                            audio=wav_data,
                            language=SR_LANGUAGE,
                            vad_filter=SR_VAD_FILTER,
                            vad_parameters=SR_VAD_FILTER_PARAMETERS,
                            temperature=MODEL_TEMPERATURE,
                            word_timestamps=True
                        )
                        transcript_found = False
                        for segment in segments:
                            text = segment.text.strip()
                            if not text:
                                continue
                            transcript_found = True
                            seg_start = start_time + segment.start
                            seg_end = start_time + segment.end
                            self.logger.info(f"Transcribed: {text} | Time: {seg_start:.2f}s - {seg_end:.2f}s | File: {saved_filename}")
                            self.speech_recognizer.transcription.append((text, seg_start, seg_end))
                            print(f"\n{'#' * 50}")
                            print(f"Transcription: '{text}' [{seg_start:.2f}s - {seg_end:.2f}s]")
                            print(f"{'#' * 50}")
                            level_one_intent = self.intent_classifier.predict_intent(text, LABELS_LEVEL_ONE)
                            if level_one_intent:
                                print_intent_details(1, level_one_intent)
                                predicted_level_one = level_one_intent['labels'][0]
                                if predicted_level_one in LABELS_LEVEL_TWO:
                                    level_two_intent = self.intent_classifier.predict_intent(text, LABELS_LEVEL_TWO[predicted_level_one])
                                    if level_two_intent:
                                        print_intent_details(2, level_two_intent)
                                        predicted_level_two = level_two_intent['labels'][0]
                                        if predicted_level_two in LABELS_LEVEL_THREE:
                                            level_three_intent = self.intent_classifier.predict_intent(text, LABELS_LEVEL_THREE[predicted_level_two])
                                            if level_three_intent:
                                                print_intent_details(3, level_three_intent)
                        if not transcript_found:
                            self.logger.info(f"No transcribable speech found in {saved_filename}")
                    except Exception as transcribe_error:
                        self.logger.error(f"Transcription error for {saved_filename}: {transcribe_error}")
                    finally:
                        del speech_data
                        if wav_data:
                            del wav_data
                else:
                    await asyncio.sleep(0.1)
            except Exception as outer_error:
                self.logger.error(f"Critical error in audio processing loop: {outer_error}")
                await asyncio.sleep(1)
```
- Processes audio from the queue asynchronously, transcribes it, classifies intent, and logs and displays the results.

```python
    def run(self):
        """Run the integrated system."""
        try:
            self.logger.info("Starting speech and intent recognition system")
            self.speech_recognizer.calibrate_audio_model()
            self.speech_recognizer.start_listening()
            processing_thread = threading.Thread(target=asyncio.run, args=(self.process_audio_data(),), daemon=True)
            processing_thread.start()
            try:
                while self.speech_recognizer.running:
                    time.sleep(0.1)
                    cmd = input("Enter command (pause/resume/stop): ").lower()
                    if cmd == "pause":
                        self.speech_recognizer.pause_listening()
                    elif cmd == "resume":
                        self.speech_recognizer.resume_listening()
                    elif cmd == "stop":
                        self.speech_recognizer.stop_listening()
                        break
            except KeyboardInterrupt:
                self.logger.info("Keyboard Interrupt received. Shutting Down.")
            finally:
                self.speech_recognizer.stop_listening()
            processing_thread.join(timeout=2.0)
        except Exception as e:
            self.logger.error(f"Error in main run loop: {e}")
            self.logger.debug(traceback.format_exc())
            self.speech_recognizer.stop_listening()
```
- Executes the system, calibrates the model, starts listening, processes audio in a thread, and responds to user commands.

```python
if __name__ == "__main__":
    try:
        recognizer = SpeechIntentRecognizer()
        recognizer.run()
    except Exception as e:
        logger = setup_logging(LOGGING_CONFIG)
        logger.critical(f"Failed to start speech and intent recognition: {e}")
        logger.debug(traceback.format_exc())
```
- Entry point: creates and runs the `SpeechIntentRecognizer` instance, handling any critical errors.

---

## 5. Workflow Summary
1. **Initialisation**: `main.py` sets up `SpeechIntentRecognizer`, configuring logging, speech recognition, and intent classification.
2. **Audio Capture**: `speech_recognizer.py` starts the audio stream, detects speech using VAD, and queues the segments.
3. **Transcription**: `main.py` processes queued audio, transcribes it with `faster-whisper`, and logs the text.
4. **Intent Classification**: `intent_classifier.py` classifies the transcription across three levels, with results formatted by `utils.py`.
5. **Control**: User inputs (`pause`, `resume`, `stop`) manage the system, which shuts down cleanly on command.

---

## 6. Conclusion
The `SpeechIntentRecognizer` system seamlessly integrates advanced speech recognition and intent classification into a modular, real-time application. With proper package installation and model setup, it offers a reliable platform for voice-based interaction analysis, suitable for further enhancement or deployment. Quite a nice solution, indeed!

