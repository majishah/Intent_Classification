import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import torch
import wave
import logging
from faster_whisper import WhisperModel

# -------------------
# Configuration
# -------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SegmentVisualizer")

# --- Paths (ADJUST THESE if your structure is different) ---
AUDIO_OUTPUT_DIR = "./recorded_speech_segments" # Directory containing the saved .wav segments
LOCAL_SILERO_VAD_PATH = './models/silero_vad'    # Path to Silero VAD model directory
SRMODEL_PATH = "./models/faster_whisper_tiny_en/" # Path to Faster Whisper model directory

# --- Model/Audio Parameters (Should match the original script's settings) ---
TARGET_SAMPLE_RATE = 16000
SR_COMPUTE_DEVICE = "cpu" # Or "cuda" if you have a compatible GPU and CUDA installed
SR_COMPUTE_TYPE = "int8"  # Or "float16", "float32" etc.
SR_LANGUAGE = "en"
VAD_MIN_SILENCE_MS = 500 # From original VAD parameters
VAD_MIN_SPEECH_MS = 200 # From original VAD parameters

# --- VAD Threshold (adjust if needed for visualization) ---
# Lower threshold might pick up quieter sounds, higher might be stricter.
VAD_THRESHOLD = 0.5 # Default threshold for Silero VAD utils is often 0.5

# -------------------
# Model Loading
# -------------------

def load_vad_model():
    """Loads the Silero VAD model and utility functions."""
    logger.info(f"Loading VAD model from: {LOCAL_SILERO_VAD_PATH}")
    if not os.path.exists(LOCAL_SILERO_VAD_PATH):
        logger.error(f"VAD path not found: {LOCAL_SILERO_VAD_PATH}")
        raise FileNotFoundError(f"VAD path not found: {LOCAL_SILERO_VAD_PATH}")
    try:
        # Force reload might be needed if models were updated or cache is stale
        # model, utils = torch.hub.load(repo_or_dir=LOCAL_SILERO_VAD_PATH, model='silero_vad', source='local', force_reload=True, trust_repo=True)

        # Standard load:
        model, utils = torch.hub.load(repo_or_dir=LOCAL_SILERO_VAD_PATH,
                                      model='silero_vad',
                                      source='local',
                                      trust_repo=True) # trust_repo=True is needed for local loading
        logger.info("VAD model loaded successfully.")
        return model, utils
    except Exception as e:
        logger.error(f"Failed to load VAD model: {e}", exc_info=True)
        raise

def load_sr_model():
    """Loads the Faster Whisper speech recognition model."""
    logger.info(f"Loading SR model from: {SRMODEL_PATH}")
    if not os.path.exists(SRMODEL_PATH):
        logger.error(f"SR path not found: {SRMODEL_PATH}")
        raise FileNotFoundError(f"SR path not found: {SRMODEL_PATH}")
    try:
        model = WhisperModel(SRMODEL_PATH, device=SR_COMPUTE_DEVICE, compute_type=SR_COMPUTE_TYPE)
        logger.info(f"SR model loaded: Device={SR_COMPUTE_DEVICE}, Compute={SR_COMPUTE_TYPE}")
        return model
    except Exception as e:
        logger.error(f"Failed to load SR model: {e}", exc_info=True)
        raise

# -------------------
# Helper Functions
# -------------------

def get_speech_timestamps_from_file(audio_path: str, vad_model, get_speech_ts_func, sample_rate: int):
    """
    Runs VAD on a loaded audio file and returns speech timestamps relative to the file start.
    """
    try:
        # Load audio using librosa - ensures consistent sample rate
        wav, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
        audio_tensor = torch.from_numpy(wav)

        # Use the loaded utility function to get timestamps
        speech_timestamps = get_speech_ts_func(
            audio_tensor,
            vad_model,
            threshold=VAD_THRESHOLD,
            sampling_rate=sample_rate,
            min_silence_duration_ms=VAD_MIN_SILENCE_MS,
            min_speech_duration_ms=VAD_MIN_SPEECH_MS
            # Add other VAD parameters if needed (e.g., window_size_samples)
        )
        # Convert sample indices to seconds
        timestamps_sec = [{'start': ts['start'] / sample_rate, 'end': ts['end'] / sample_rate} for ts in speech_timestamps]
        return timestamps_sec
    except Exception as e:
        logger.error(f"Error getting VAD timestamps for {audio_path}: {e}", exc_info=True)
        return []

def get_transcription(audio_path: str, sr_model):
    """Runs transcription on an audio file."""
    try:
        segments, info = sr_model.transcribe(audio_path, language=SR_LANGUAGE)
        full_text = " ".join([segment.text.strip() for segment in segments])
        logger.info(f"Transcription for {os.path.basename(audio_path)}: '{full_text}'")
        return full_text if full_text else "[No speech transcribed]"
    except Exception as e:
        logger.error(f"Error getting transcription for {audio_path}: {e}", exc_info=True)
        return "[Transcription Error]"

# -------------------
# Main Visualization Logic
# -------------------

if __name__ == "__main__":
    logger.info("Starting audio segment visualization...")

    if not os.path.isdir(AUDIO_OUTPUT_DIR):
        logger.error(f"Audio segment directory not found: {AUDIO_OUTPUT_DIR}")
        exit(1)

    try:
        vad_model, vad_utils = load_vad_model()
        get_speech_ts_func = vad_utils[0] # Usually the first function is get_speech_timestamps
        sr_model = load_sr_model()
    except Exception as model_load_error:
        logger.critical(f"Failed to load models. Exiting. Error: {model_load_error}")
        exit(1)

    # Find all .wav files
    audio_files = sorted(glob.glob(os.path.join(AUDIO_OUTPUT_DIR, "speech_segment_*.wav")))

    if not audio_files:
        logger.warning(f"No 'speech_segment_*.wav' files found in {AUDIO_OUTPUT_DIR}")
        exit(0)

    logger.info(f"Found {len(audio_files)} audio segments to visualize.")

    for audio_file in audio_files:
        filename = os.path.basename(audio_file)
        logger.info(f"\nProcessing: {filename}")

        try:
            # 1. Load Audio Waveform
            audio_data, sr = librosa.load(audio_file, sr=TARGET_SAMPLE_RATE, mono=True)
            duration = librosa.get_duration(y=audio_data, sr=sr)
            time_axis = np.linspace(0, duration, len(audio_data))

            # 2. Get VAD Timestamps for this segment
            vad_timestamps = get_speech_timestamps_from_file(audio_file, vad_model, get_speech_ts_func, sr)

            # 3. Get Transcription for this segment
            transcription = get_transcription(audio_file, sr_model)

            # 4. Create Plot
            fig, ax = plt.subplots(figsize=(12, 4)) # Adjust figure size as needed

            # Plot waveform
            librosa.display.waveshow(audio_data, sr=sr, ax=ax, color='blue', alpha=0.7)

            # Plot VAD segments
            vad_detected = False
            for i, ts in enumerate(vad_timestamps):
                label = "VAD Speech" if i == 0 else None # Only label the first segment for legend
                ax.axvspan(ts['start'], ts['end'], color='red', alpha=0.3, label=label)
                vad_detected = True

            # Configure plot
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.set_xlim(0, duration)
            ax.set_title(f"File: {filename}\nTranscription: {transcription}", wrap=True)
            ax.grid(True, linestyle='--', alpha=0.6)
            if vad_detected:
                ax.legend(loc='upper right')
            else:
                 # Add text if no VAD detected, helps clarify
                 ax.text(0.5, 0.5, 'No speech detected by VAD in this segment',
                         horizontalalignment='center', verticalalignment='center',
                         transform=ax.transAxes, fontsize=10, color='gray')


            plt.tight_layout()
            plt.show() # Display the plot - it will pause execution until closed

            plt.close(fig) # Close the figure to free memory before the next loop

        except Exception as e:
            logger.error(f"Failed to process or plot {filename}: {e}", exc_info=True)
            # Ensure figure is closed even if error occurs mid-plot
            if 'fig' in locals() and plt.fignum_exists(fig.number):
                 plt.close(fig)


    logger.info("Visualization finished.")