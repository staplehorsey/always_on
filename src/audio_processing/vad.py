import os
import time
import logging
import numpy as np
import torch
from typing import Optional, List, Tuple
from scipy import signal

logger = logging.getLogger('audio-processor.vad')

class VoiceActivityDetector:
    def __init__(self, target_sample_rate: int = 16000, frame_size: int = 512,
                 speech_threshold: float = 0.5, min_speech_duration: float = 1.0,
                 speech_cooldown: float = 30.0):
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
        self.current_recording = np.array([], dtype=np.float32)
        self.last_speech_time = 0
        self.consecutive_silence = 0  # Track consecutive silence frames
        
        # Load VAD model
        logger.info("Loading VAD model...")
        torch.hub.set_dir(os.path.expanduser('~/.cache/torch/hub'))
        self.vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                         model='silero_vad',
                                         force_reload=False)
        self.vad_model.eval()

    def process_frame(self, frame: np.ndarray, context_buffer: Optional[np.ndarray] = None,
                     server_sample_rate: int = 44100) -> Optional[np.ndarray]:
        """Process a frame of audio and detect speech"""
        try:
            # Debug input
            logger.debug(f"VAD input frame: shape={frame.shape}, dtype={frame.dtype}")
            if context_buffer is not None:
                logger.debug(f"Context buffer: shape={context_buffer.shape}, dtype={context_buffer.dtype}")
            
            # Process frame in VAD-sized chunks
            remaining_samples = len(frame)
            processed_samples = 0
            result = None
            
            while remaining_samples > 0:
                # Calculate how many samples we can add to the current buffer
                space_in_buffer = self.frame_size - self.audio_buffer_idx
                samples_to_add = min(remaining_samples, space_in_buffer)
                
                # Add samples to buffer
                self.audio_buffer[self.audio_buffer_idx:self.audio_buffer_idx + samples_to_add] = \
                    frame[processed_samples:processed_samples + samples_to_add]
                
                self.audio_buffer_idx += samples_to_add
                processed_samples += samples_to_add
                remaining_samples -= samples_to_add
                
                # If buffer is full, process it
                if self.audio_buffer_idx >= self.frame_size:
                    # Normalize buffer for VAD
                    vad_buffer = self.audio_buffer.copy()
                    if np.max(np.abs(vad_buffer)) > 0:
                        vad_buffer = vad_buffer / np.max(np.abs(vad_buffer))
                    
                    # Debug VAD buffer
                    logger.debug(f"VAD buffer stats: min={np.min(vad_buffer)}, max={np.max(vad_buffer)}")
                    
                    # Convert to torch tensor
                    tensor = torch.from_numpy(vad_buffer).to('cuda' if torch.cuda.is_available() else 'cpu')
                    tensor = tensor.unsqueeze(0)  # Add batch dimension
                    
                    # Run VAD
                    speech_prob = self.vad_model(tensor, self.target_sample_rate).item()
                    logger.debug(f"Speech probability: {speech_prob:.3f}")
                    current_time = time.time()
                    
                    # Handle speech detection with cooldown and padding
                    if speech_prob > self.speech_threshold:
                        if not self.is_recording:
                            logger.info("Speech detected, starting recording...")
                            self.is_recording = True
                            self.current_recording = np.array([], dtype=np.float32)
                            # Include context buffer if provided
                            if context_buffer is not None:
                                padding_samples = min(int(10.0 * self.target_sample_rate), len(context_buffer))
                                logger.debug(f"Adding {padding_samples} samples of context")
                                self.current_recording = np.concatenate([
                                    self.current_recording,
                                    context_buffer[-padding_samples:]
                                ])
                        
                        self.consecutive_silence = 0
                        self.last_speech_time = current_time
                        self.current_recording = np.concatenate([self.current_recording, self.audio_buffer])
                    elif self.is_recording:
                        # Calculate silence duration
                        silence_duration = current_time - self.last_speech_time
                        self.consecutive_silence += 1
                        logger.debug(f"Silence duration: {silence_duration:.1f}s, consecutive={self.consecutive_silence}")
                        
                        # Keep recording during silence up to cooldown
                        self.current_recording = np.concatenate([self.current_recording, self.audio_buffer])
                        
                        if silence_duration >= self.speech_cooldown:
                            # Check if recording meets minimum duration
                            recording_duration = len(self.current_recording) / self.target_sample_rate
                            if recording_duration >= self.min_speech_duration:
                                logger.info(f"Speech ended, finalizing recording ({recording_duration:.1f}s)")
                                # Add trailing silence if context buffer provided
                                if context_buffer is not None:
                                    padding_samples = min(int(10.0 * self.target_sample_rate), len(context_buffer))
                                    logger.debug(f"Adding {padding_samples} samples of trailing context")
                                    self.current_recording = np.concatenate([
                                        self.current_recording,
                                        context_buffer[-padding_samples:]
                                    ])
                                
                                result = self.current_recording.copy()
                                logger.debug(f"Final recording: shape={result.shape}, min={np.min(result)}, max={np.max(result)}")
                            else:
                                logger.debug(f"Discarding short speech segment ({recording_duration:.2f}s)")
                            
                            # Reset recording state
                            self.is_recording = False
                            self.current_recording = np.array([], dtype=np.float32)
                            self.consecutive_silence = 0
                    
                    # Reset buffer
                    self.audio_buffer = np.zeros(self.frame_size, dtype=np.float32)
                    self.audio_buffer_idx = 0
            
            return result
            
        except Exception as e:
            logger.error(f"Error in VAD processing: {e}", exc_info=True)
            return None
