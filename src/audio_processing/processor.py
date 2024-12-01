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
    
    def _apply_bandpass_filter(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        """Apply bandpass filter optimized for speech frequencies (80Hz-8kHz)"""
        nyquist = sr / 2
        low_cutoff = 80 / nyquist
        high_cutoff = min(8000, nyquist-100) / nyquist
        
        # Use higher order for better frequency response
        filter_order = min(301, len(audio_data) // 4)
        if filter_order % 2 == 0:
            filter_order += 1
            
        if filter_order >= 3:
            b = signal.firwin(filter_order, [low_cutoff, high_cutoff], 
                            window='hamming', pass_zero=False)
            try:
                return signal.filtfilt(b, [1.0], audio_data)
            except ValueError:
                return signal.lfilter(b, [1.0], audio_data)
        return audio_data

    def _reduce_noise(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        """Simple noise reduction using spectral gating"""
        if len(audio_data) < 256:
            return audio_data
            
        # Compute spectrogram
        nperseg = min(256, len(audio_data))
        f, t, Zxx = signal.stft(audio_data, fs=sr, nperseg=nperseg)
        
        # Estimate noise floor from lowest 10% of magnitudes
        magnitude = np.abs(Zxx)
        noise_floor = np.percentile(magnitude, 10, axis=1, keepdims=True)
        
        # Apply spectral gating
        gain_mask = (magnitude > 2 * noise_floor)  # 6dB threshold
        Zxx_clean = Zxx * gain_mask
        
        # Inverse STFT
        _, audio_clean = signal.istft(Zxx_clean, fs=sr, nperseg=nperseg)
        return audio_clean

    def _normalize_audio(self, audio_data: np.ndarray, headroom_db: float = 6.0) -> np.ndarray:
        """Normalize audio with headroom and soft knee"""
        if len(audio_data) == 0 or np.max(np.abs(audio_data)) == 0:
            return audio_data
            
        # Convert headroom to linear gain
        headroom_linear = 10 ** (-headroom_db/20)
        
        # Find peak value
        peak = np.max(np.abs(audio_data))
        
        # Apply soft knee normalization
        gain = min(headroom_linear / peak, 1.0)
        return audio_data * gain

    def resample_audio(self, audio_data: np.ndarray, orig_sr: int,
                      target_sr: Optional[int] = None) -> np.ndarray:
        """Resample audio data to target sample rate with enhanced processing"""
        if target_sr is None:
            target_sr = self.target_sample_rate
            
        if orig_sr == target_sr:
            return audio_data.astype(np.float32)
        
        # Ensure input is float32
        audio_data = audio_data.astype(np.float32)
        
        # Initial normalization
        audio_data = self._normalize_audio(audio_data)
        
        # Apply bandpass filter optimized for speech
        audio_data = self._apply_bandpass_filter(audio_data, orig_sr)
        
        # Apply noise reduction
        audio_data = self._reduce_noise(audio_data, orig_sr)
        
        # Calculate resampling ratio and output size
        ratio = target_sr / orig_sr
        output_samples = int(len(audio_data) * ratio)
        
        # High-quality resampling
        resampled = signal.resample(audio_data, output_samples, window=('kaiser', 5.0))
        
        # Final normalization with headroom
        resampled = self._normalize_audio(resampled, headroom_db=6.0)
        
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
