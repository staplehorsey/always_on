import logging
import queue
import threading
from typing import Optional
import numpy as np

from .io_handler import AudioIOHandler
from .vad import VoiceActivityDetector
from .processor import AudioProcessor
from .transcriber import AudioTranscriber
from .llm_processor import LLMProcessor
from .storage import StorageManager

logger = logging.getLogger('audio-processor.orchestrator')

class AudioOrchestrator:
    def __init__(self, server_host: str = 'staple.local', server_port: int = 12345,
                 enable_monitoring: bool = True, recordings_dir: str = 'recordings',
                 server_sample_rate: int = 44100):
        # Initialize components
        self.io_handler = AudioIOHandler(server_host, server_port, enable_monitoring, server_sample_rate)
        self.vad = VoiceActivityDetector()
        self.processor = AudioProcessor()
        self.transcriber = AudioTranscriber()
        self.llm_processor = LLMProcessor()
        self.storage = StorageManager(recordings_dir)
        
        # Processing queue
        self.max_queue_size = 10
        self.processing_queue = queue.Queue(maxsize=self.max_queue_size)
        self.shutdown_flag = threading.Event()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_queue)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def _process_queue(self) -> None:
        """Process recordings from the queue"""
        while not self.shutdown_flag.is_set():
            try:
                # Get the next recording from the queue with timeout
                try:
                    audio_array = self.processing_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process the recording
                self._process_recording(audio_array)
                
                # Mark task as done
                self.processing_queue.task_done()
            except Exception as e:
                logger.error(f"Error in queue processor: {e}", exc_info=True)
                continue
    
    def _add_to_queue(self, audio_array) -> None:
        """Add recording to processing queue"""
        try:
            if self.processing_queue.qsize() >= self.max_queue_size:
                logger.warning("Queue full, dropping recording")
                return
            
            self.processing_queue.put(audio_array)
            logger.info(f"Added recording to queue. Queue size: {self.processing_queue.qsize()}")
        except Exception as e:
            logger.error(f"Error adding to queue: {e}")
    
    def _process_recording(self, audio_array: np.ndarray) -> None:
        """Process a single recording"""
        try:
            # Debug info about input
            logger.debug(f"Processing recording: type={type(audio_array)}, shape={audio_array.shape}, dtype={audio_array.dtype}")
            
            # Convert to float32 if needed
            audio_array = np.array(audio_array, dtype=np.float32)
            logger.debug(f"After float32 conversion: shape={audio_array.shape}, dtype={audio_array.dtype}")
            
            # Skip empty recordings
            if len(audio_array) == 0:
                logger.warning("Empty recording - skipping processing")
                return
                
            if not np.any(audio_array):
                logger.warning("Silent recording (all zeros) - skipping processing")
                return
            
            # Debug array stats
            logger.debug(f"Array stats: min={np.min(audio_array)}, max={np.max(audio_array)}, mean={np.mean(audio_array)}")
            
            # Audio is already at target rate from VAD, no need to resample
            
            # Normalize once for Whisper
            if np.max(np.abs(audio_array)) > 0:
                audio_array = audio_array / np.max(np.abs(audio_array))
                logger.debug(f"After normalization: min={np.min(audio_array)}, max={np.max(audio_array)}")
            
            # Transcribe with Whisper
            logger.info("Transcribing audio...")
            result = self.transcriber.transcribe(audio_array)
            
            if not result:
                logger.warning("No transcript generated - skipping processing")
                return
            
            transcript = result["text"].strip()
            logger.debug(f"Generated transcript: {transcript[:100]}...")
            
            # Generate title and summary using LLM
            title, summary = self.llm_processor.generate_title_and_summary(transcript)
            
            # Save recording and metadata
            self.storage.save_recording(
                audio_array, transcript, title, summary,
                self.processor.target_sample_rate
            )
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}", exc_info=True)
    
    def process_audio_chunk(self, chunk) -> None:
        """Process a chunk of audio data"""
        try:
            # Convert bytes to float32
            samples = np.frombuffer(chunk, dtype=np.float32)
            
            # Play audio if monitoring enabled
            self.io_handler.play_audio(chunk)
            
            # Update monitoring buffer with resampled audio
            resampled = self.processor.resample_audio(samples, self.io_handler.server_sample_rate)
            self.processor.update_monitoring_buffer(resampled)
            
            # Process audio frame for VAD with context buffer
            recording = self.vad.process_frame(resampled, self.processor.monitoring_buffer)
            
            # Add to queue if we got a valid recording (non-empty array)
            if recording is not None and len(recording) > 0:
                logger.debug(f"Got recording: shape={recording.shape}, non-zero={np.any(recording)}")
                self._add_to_queue(recording)
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}", exc_info=True)
    
    def run(self) -> None:
        """Main processing loop"""
        try:
            logger.info("Starting audio processing...")
            
            # Connect to audio server
            self.io_handler.connect()
            
            while True:
                # Receive audio data
                chunk, error = self.io_handler.receive_audio()
                if error:
                    logger.error(f"Error receiving audio: {error}")
                    continue
                if chunk is None:
                    continue
                
                # Process the chunk
                self.process_audio_chunk(chunk)
                
        except KeyboardInterrupt:
            logger.info("Stopping audio processing...")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up all resources"""
        logger.info("Cleaning up resources...")
        
        # Stop queue processor
        logger.info("Stopping queue processor...")
        self.shutdown_flag.set()
        
        # Wait for queue to finish with timeout
        try:
            logger.info("Waiting for processing queue to finish...")
            self.processing_queue.join()
        except Exception as e:
            logger.error(f"Error waiting for queue: {e}")
        
        # Clean up components
        self.io_handler.cleanup()
        self.llm_processor.cleanup()
