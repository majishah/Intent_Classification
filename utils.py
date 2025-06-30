import numpy as np
import io
import wave
import os
import datetime
import librosa
import pyaudio # Needed for FORMAT constant comparison

# Import necessary components from other modules
from logger_setup import logger
import config # Import config to access constants directly

def calculate_waveform_amplitudes(audio_bytes: bytes, sample_rate: int, target_points: int = 150) -> list[float]:
    """Calculates normalized RMS amplitude, downsampled to target_points."""
    if not audio_bytes: return [0.0] * target_points
    try:
        # Use constants directly from imported config
        dtype = np.int16 if config.FORMAT == pyaudio.paInt16 else np.int32
        max_val = 32767.0 if dtype == np.int16 else 2147483647.0
        audio_np = np.frombuffer(audio_bytes, dtype=dtype).astype(np.float32) / max_val

        if audio_np.size == 0: return [0.0] * target_points

        frame_length = 2048; hop_length = 512
        rms_values = librosa.feature.rms(y=audio_np, frame_length=frame_length, hop_length=hop_length)[0]

        max_rms = np.max(rms_values) if rms_values.size > 0 else 1.0
        if max_rms < 0.01: max_rms = 0.1
        normalized_rms = rms_values / max_rms
        normalized_rms = np.clip(normalized_rms, 0.0, 1.0)

        num_frames = len(normalized_rms)
        if num_frames == 0: return [0.0] * target_points

        current_indices = np.linspace(0, 1, num_frames)
        target_indices = np.linspace(0, 1, target_points)
        downsampled_amplitudes = np.interp(target_indices, current_indices, normalized_rms).tolist()

        return downsampled_amplitudes
    except Exception as e:
        logger.error(f"Error calculating waveform amplitudes: {e}", exc_info=True)
        return [0.0] * target_points

def audio_data_to_wav_bytes(audio_data: bytes, pyaudio_instance, # Pass PyAudio instance
                           sample_rate=config.RATE, save_to_file: bool = False) -> tuple[io.BytesIO | None, str | None]:
    """Convert raw audio bytes to WAV format BytesIO and optionally save to file."""
    if not audio_data: logger.warning("Empty audio data for WAV"); return None, None
    if not pyaudio_instance: logger.error("PyAudio instance needed for WAV"); return None, None
    try:
        now = datetime.datetime.now(); ts_str = now.strftime("%Y%m%d_%H%M%S_%f")
        fname = f"speech_segment_{ts_str}.wav"; fpath = os.path.join(config.AUDIO_OUTPUT_DIR, fname) if save_to_file else None
        wav_io = io.BytesIO(); wf = wave.open(wav_io, 'wb')
        try:
            wf.setnchannels(config.CHANNELS)
            # Use passed PyAudio instance
            sample_width = pyaudio_instance.get_sample_size(config.FORMAT)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)
        finally:
            wf.close()
        wav_io.seek(0)
        if save_to_file and fpath:
            try:
                os.makedirs(os.path.dirname(fpath), exist_ok=True)
                with open(fpath,'wb') as f:
                    f.write(wav_io.getvalue())
                logger.info(f"Audio saved: {fpath}")
            except Exception as save_e:
                logger.error(f"Failed save to {fpath}: {save_e}", exc_info=True)
        return wav_io, fname
    except Exception as e:
        logger.error(f"WAV conversion error: {e}", exc_info=True)
        return None, None