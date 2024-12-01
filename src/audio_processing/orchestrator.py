import logging
import queue
import threading
from typing import Optional

from .io_handler import AudioIOHandler
from .vad import VoiceActivityDetector
from .processor import AudioProcessor
from .transcriber import AudioTranscriber
from .llm_processor import LLMProcessor
from .storage import StorageManager

logger = logging.getLogger('audio-processor.orchestrator')

class AudioOrchestrator:
    def __init__(self, server_host: str = 'staple.local', server_port: int = 12345,
                 enable_monitoring: bool = True, recordings_dir: str = 'recordings'):
        # Initialize components
        self.io_handler = AudioIOHandler(server_host, server_port, enable_monitoring)
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
                logger.error(f"Error in queue processor: {e}")
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
    
    def _process_recording(self, audio_array) -> None:
        """Process a single recording"""
        try:
            # Normalize audio for processing
            audio_array = self.processor.normalize_audio(audio_array)
            
            # Transcribe audio
            result = self.transcriber.transcribe(audio_array)
            if not result:
                logger.warning("No transcript generated - skipping processing")
                return
            
            transcript = result["text"].strip()
            
            # Generate title and summary
            title, summary = self.llm_processor.generate_title_and_summary(transcript)
            
            # Save recording and metadata
            self.storage.save_recording(
                audio_array, transcript, title, summary,
                self.processor.target_sample_rate
            )
            
        except Exception as e:
            logger.error(f"Error processing recording: {e}")
    
    def process_audio_chunk(self, chunk) -> None:
        """Process a chunk of audio data"""
        try:
            # Convert to float32 and play if monitoring enabled
            self.io_handler.play_audio(chunk)
            
            # Resample audio for processing
            resampled = self.processor.resample_audio(
                chunk, self.io_handler.server_sample_rate
            )
            
            # Update monitoring buffer
            self.processor.update_monitoring_buffer(resampled)
            
            # Process audio frame for VAD
            vad_result = self.vad.process_frame(resampled)
            if vad_result:
                is_speech, audio_frame = vad_result
                
                # Get context buffer for padding
                context_buffer = self.processor.get_context_buffer()
                
                # Update recording state
                recording = self.vad.update_recording_state(
                    is_speech, audio_frame, context_buffer
                )
                
                # If recording complete, add to processing queue
                if recording is not None:
                    self._add_to_queue(recording)
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
    
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
