import numpy as np
from scipy import signal
import logging
from typing import Optional

logger = logging.getLogger('audio-processor.processor')

class AudioProcessor:
    def __init__(self, target_sample_rate: int = 16000,
                 monitoring_buffer_duration: float = 0.5):
        self.target_sample_rate = target_sample_rate
        self.monitoring_buffer_duration = monitoring_buffer_duration
        self.monitoring_buffer_samples = int(target_sample_rate * monitoring_buffer_duration)
        self.monitoring_buffer = np.array([], dtype=np.float32)
    
    def resample_audio(self, audio_data: np.ndarray, orig_sr: int,
                      target_sr: Optional[int] = None) -> np.ndarray:
        """Resample audio data to target sample rate while preserving signal characteristics"""
        if target_sr is None:
            target_sr = self.target_sample_rate
            
        if orig_sr == target_sr:
            return audio_data.astype(np.float32)
        
        # Calculate resampling ratio
        ratio = target_sr / orig_sr
        
        # Ensure input is float32
        audio_data = audio_data.astype(np.float32)
        
        # Normalize the signal to [-1, 1] range before resampling
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Apply low-pass filter to prevent aliasing
        nyquist = min(orig_sr, target_sr) / 2
        cutoff = 0.9 * nyquist  # Leave some margin
        b = signal.firwin(101, cutoff, fs=orig_sr)
        audio_data = signal.filtfilt(b, [1.0], audio_data)
        
        # Calculate number of samples for output
        output_samples = int(len(audio_data) * ratio)
        
        # Resample using scipy's resample function
        resampled = signal.resample(audio_data, output_samples)
        
        # Ensure consistent amplitude after resampling
        if np.max(np.abs(resampled)) > 0:
            resampled = resampled / np.max(np.abs(resampled))
        
        return resampled.astype(np.float32)
    
    def update_monitoring_buffer(self, audio_data: np.ndarray) -> np.ndarray:
        """Update the monitoring buffer with new audio data"""
        self.monitoring_buffer = np.concatenate((self.monitoring_buffer, audio_data))
        
        # Trim monitoring buffer to save memory (keep last N seconds)
        max_samples = 2 * self.target_sample_rate
        if len(self.monitoring_buffer) > max_samples:
            self.monitoring_buffer = self.monitoring_buffer[-max_samples:]
        
        return self.monitoring_buffer
    
    def get_context_buffer(self, duration: float = 0.5) -> Optional[np.ndarray]:
        """Get a portion of the monitoring buffer for context"""
        samples = int(duration * self.target_sample_rate)
        if len(self.monitoring_buffer) >= samples:
            return self.monitoring_buffer[-samples:]
        return None
    
    def normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio data to [-1, 1] range"""
        audio_data = np.array(audio_data, dtype=np.float32)
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        return audio_data
