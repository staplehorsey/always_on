import socket
import logging
import numpy as np
import sounddevice as sd
from typing import Optional, Tuple

logger = logging.getLogger('audio-processor.io')

class AudioIOHandler:
    def __init__(self, server_host: str, server_port: int, enable_monitoring: bool = True,
                 server_sample_rate: int = 44100):
        self.server_host = server_host
        self.server_port = server_port
        self.enable_monitoring = enable_monitoring
        self.server_sample_rate = server_sample_rate
        
        # Initialize socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(0.1)  # 100ms timeout for non-blocking recv
        
        # Audio monitoring setup
        self.audio_output_stream: Optional[sd.OutputStream] = None
        if self.enable_monitoring:
            self.setup_monitoring()
    
    def setup_monitoring(self):
        """Setup audio monitoring stream"""
        self.audio_output_stream = sd.OutputStream(
            channels=1,
            samplerate=self.server_sample_rate,
            blocksize=1024,
            dtype=np.float32
        )
        
    def connect(self) -> None:
        """Connect to the audio server"""
        logger.info(f"Connecting to {self.server_host}:{self.server_port}...")
        self.sock.connect((self.server_host, self.server_port))
        logger.info("Connected to audio server")
        
        if self.enable_monitoring and self.audio_output_stream:
            logger.info("Audio monitoring enabled - you should hear the incoming audio")
            self.audio_output_stream.start()
    
    def receive_audio(self) -> Tuple[Optional[np.ndarray], Optional[Exception]]:
        """Receive audio data from the socket"""
        try:
            data = self.sock.recv(4096)
            if not data:
                return None, None
            return np.frombuffer(data, dtype=np.float32), None
        except socket.timeout:
            return None, None
        except Exception as e:
            return None, e
    
    def play_audio(self, audio_data: np.ndarray, gain: float = 1.5) -> None:
        """Play audio through the monitoring stream"""
        if self.enable_monitoring and self.audio_output_stream:
            try:
                samples_to_play = audio_data * gain
                samples_to_play = np.clip(samples_to_play, -1.0, 1.0)
                self.audio_output_stream.write(samples_to_play)
            except Exception as e:
                logger.error(f"Error playing audio: {e}")
    
    def cleanup(self) -> None:
        """Clean up IO resources"""
        if self.enable_monitoring and hasattr(self, 'audio_output_stream'):
            try:
                self.audio_output_stream.stop()
                self.audio_output_stream.close()
            except Exception as e:
                logger.error(f"Error closing output stream: {e}")
        
        try:
            self.sock.close()
        except Exception as e:
            logger.error(f"Error closing socket: {e}")
