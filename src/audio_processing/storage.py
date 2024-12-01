import os
import logging
import numpy as np
from datetime import datetime
import scipy.io.wavfile as wavfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger('audio-processor.storage')

class StorageManager:
    def __init__(self, base_dir: str = "recordings"):
        self.base_dir = base_dir
    
    def save_recording(self, audio_data: np.ndarray, transcript: str,
                      title: str, summary: str, sample_rate: int) -> Optional[str]:
        """Save recording and metadata to disk"""
        try:
            # Ensure audio data is contiguous and float32
            audio_data = np.ascontiguousarray(audio_data, dtype=np.float32)
            
            # Normalize if needed
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Get current timestamp
            now = datetime.now()
            year = now.strftime("%Y")
            month = now.strftime("%m-%B")
            day = now.strftime("%d-%A")
            timestamp = now.strftime("%H-%M-%S")
            
            # Clean the title
            clean_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
            
            # Create directory structure
            dir_name = os.path.join(self.base_dir, year, month, day,
                                  f"{timestamp}_{clean_title}")
            os.makedirs(dir_name, exist_ok=True)
            
            # Save files
            logger.info(f"Saving recording to {dir_name}...")
            
            # Save audio (ensure it's normalized and contiguous)
            wavfile.write(f"{dir_name}/audio.wav", sample_rate, audio_data)
            
            # Save transcript
            with open(f"{dir_name}/transcript.txt", 'w') as f:
                f.write(transcript)
            
            # Save summary
            with open(f"{dir_name}/summary.txt", 'w') as f:
                f.write(f"Title: {title}\n\nSummary:\n{summary}\n\n"
                       f"Full Transcript:\n{transcript}")
            
            logger.info(f"Files saved to: {dir_name}")
            return dir_name
            
        except Exception as e:
            logger.error(f"Error saving recording: {e}")
            return None
