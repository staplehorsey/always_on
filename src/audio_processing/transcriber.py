import logging
import numpy as np
import whisper
from typing import Dict, Any, Optional

logger = logging.getLogger('audio-processor.transcriber')

class AudioTranscriber:
    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        logger.info(f"Loading Whisper model ({model_name})...")
        self.model = whisper.load_model(model_name)
    
    def transcribe(self, audio_array: np.ndarray) -> Optional[Dict[str, Any]]:
        """Transcribe audio using Whisper"""
        try:
            # Ensure audio is float32 and normalized for Whisper
            audio_array = np.array(audio_array, dtype=np.float32)
            if np.max(np.abs(audio_array)) > 1.0:
                audio_array = audio_array / np.max(np.abs(audio_array))
            
            # Transcribe with Whisper
            logger.info("Transcribing audio...")
            result = self.model.transcribe(audio_array)
            
            transcript = result["text"].strip()
            if not transcript:
                logger.warning("No transcript generated")
                return None
            
            logger.info(f"Transcription complete: {transcript}")
            return result
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return None
