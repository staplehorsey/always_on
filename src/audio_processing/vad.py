import os
import time
import logging
import numpy as np
import torch
from typing import Optional, List, Tuple

logger = logging.getLogger('audio-processor.vad')

class VoiceActivityDetector:
    def __init__(self, target_sample_rate: int = 16000, frame_size: int = 512,
                 speech_threshold: float = 0.5, min_speech_duration: float = 1.0,
                 speech_cooldown: float = 1.5):
        self.target_sample_rate = target_sample_rate
        self.frame_size = frame_size
        self.speech_threshold = speech_threshold
        self.min_speech_duration = min_speech_duration
        self.speech_cooldown = speech_cooldown
        
        # Initialize buffer
        self.audio_buffer = np.zeros(self.frame_size, dtype=np.float32)
        self.audio_buffer_idx = 0
        
        # State tracking
        self.is_recording = False
        self.current_recording: List[float] = []
        self.last_speech_time = 0
        
        # Load VAD model
        logger.info("Loading VAD model...")
        torch.hub.set_dir(os.path.expanduser('~/.cache/torch/hub'))
        self.vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                         model='silero_vad',
                                         force_reload=False)
        self.vad_model.eval()
    
    def process_frame(self, frame: np.ndarray) -> Optional[Tuple[bool, np.ndarray]]:
        """Process a frame of audio and detect speech"""
        # Add samples to buffer
        samples_to_add = min(len(frame), self.frame_size - self.audio_buffer_idx)
        self.audio_buffer[self.audio_buffer_idx:self.audio_buffer_idx + samples_to_add] = \
            frame[:samples_to_add]
        
        self.audio_buffer_idx += samples_to_add
        
        # If buffer is full, process it
        if self.audio_buffer_idx >= self.frame_size:
            # Normalize buffer for VAD
            vad_buffer = self.audio_buffer.copy()
            if np.max(np.abs(vad_buffer)) > 0:
                vad_buffer = vad_buffer / np.max(np.abs(vad_buffer))
            
            # Convert to torch tensor
            tensor = torch.from_numpy(vad_buffer).to('cuda' if torch.cuda.is_available() else 'cpu')
            tensor = tensor.unsqueeze(0)  # Add batch dimension
            
            # Run VAD
            speech_prob = self.vad_model(tensor, self.target_sample_rate).item()
            current_time = time.time()
            
            # Reset buffer
            self.audio_buffer = np.zeros(self.frame_size, dtype=np.float32)
            self.audio_buffer_idx = 0
            
            return speech_prob > self.speech_threshold, vad_buffer
        
        return None
    
    def update_recording_state(self, is_speech: bool, audio_data: np.ndarray,
                             context_buffer: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Update recording state based on speech detection"""
        current_time = time.time()
        
        if is_speech:
            if not self.is_recording:
                logger.info("Speech detected, starting recording...")
                self.is_recording = True
                self.current_recording = []
                # Include context buffer if provided
                if context_buffer is not None:
                    self.current_recording.extend(context_buffer.tolist())
            
            self.last_speech_time = current_time
            self.current_recording.extend(audio_data.tolist())
            return None
        
        elif self.is_recording:
            # Calculate silence duration
            silence_duration = current_time - self.last_speech_time
            
            # Keep recording during silence up to cooldown
            self.current_recording.extend(audio_data.tolist())
            
            if silence_duration >= self.speech_cooldown:
                # Check if recording meets minimum duration
                recording_duration = len(self.current_recording) / self.target_sample_rate
                if recording_duration >= self.min_speech_duration:
                    logger.info("Speech ended, finalizing recording...")
                    # Add trailing silence if context buffer provided
                    if context_buffer is not None:
                        self.current_recording.extend(context_buffer.tolist())
                    
                    # Get the recording and reset state
                    recording = np.array(self.current_recording)
                    self.is_recording = False
                    self.current_recording = []
                    return recording
                else:
                    logger.debug(f"Discarding short speech segment ({recording_duration:.2f}s)")
                    self.is_recording = False
                    self.current_recording = []
        
        return None
