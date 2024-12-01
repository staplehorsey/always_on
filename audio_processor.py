import os
import numpy as np
import sounddevice as sd
import whisper
import torch
import time
import queue
import threading
import json
import re
import openai
import logging
import logging.handlers
from datetime import datetime

# Set up logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.handlers.RotatingFileHandler(
            'logs/audio_processor.log',
            maxBytes=10485760,  # 10MB
            backupCount=5
        ),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('audio-processor')

class AudioProcessor:
    def __init__(self):
        logger.info("Initializing audio processor...")
        # Set tokenizer parallelism to avoid fork warnings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Initialize device
        self.device = torch.device('cpu')  # Using CPU for VAD model
        logger.info(f"Using device: {self.device}")
        
        # Initialize queues
        self.processing_queue = queue.Queue()
        self.summarization_queue = queue.Queue()
        self.max_queue_size = 10  # Maximum number of pending recordings
        
        # Initialize OpenAI client for local server
        openai.api_key = "dummy"  # Not used but required
        openai.api_base = "http://localhost:8000/v1"
        self.openai_client = openai.Client(
            api_key="dummy",
            base_url="http://localhost:8000/v1"
        )
        
        # Initialize models in separate threads
        self.model_ready = threading.Event()
        self.shutdown_flag = threading.Event()
        
        # Start model initialization thread
        self.init_thread = threading.Thread(target=self._initialize_models)
        self.init_thread.start()
        
        # Start processing threads
        self.processing_thread = threading.Thread(target=self._process_queue)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Audio parameters
        self.sample_rate = 16000
        self.vad_frame_samples = 512  # Silero VAD expects 512 samples for 16kHz
        self.buffer_duration = 60  # Total buffer duration in seconds
        self.padding_duration = 30  # Padding before and after speech
        
        # State variables
        self.recording_buffer = np.array([], dtype=np.float32)
        self.monitoring_buffer = np.array([], dtype=np.float32)
        self.vad_buffer = np.array([], dtype=np.float32)  # Buffer for VAD processing
        self.is_recording = False
        self.speech_detected = False
        self.last_speech_end = 0
        
        # Wait for models to be ready
        logger.info("Waiting for models to initialize...")
        self.model_ready.wait()
        logger.info("Initialization complete")

    def _initialize_models(self):
        try:
            logger.info("Initializing models...")
            logger.info(f"Device: {self.device}")
            
            logger.info("Loading VAD model...")
            start_time = time.time()
            self.vad_model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad",
                                                 model="silero_vad",
                                                 force_reload=False,
                                                 onnx=False)
            self.vad_model.eval()
            logger.info(f"VAD model loaded in {time.time() - start_time:.2f} seconds")
            
            logger.info("Loading Whisper model...")
            start_time = time.time()
            self.whisper_model = whisper.load_model("base")
            logger.info(f"Whisper model loaded in {time.time() - start_time:.2f} seconds")
            
            # Test LLM server connection
            try:
                response = self.openai_client.chat.completions.create(
                    model="mistral",
                    messages=[{"role": "user", "content": "Test connection"}],
                    max_tokens=10
                )
                logger.info("LLM server connection successful!")
            except Exception as e:
                logger.warning(f"Could not connect to LLM server: {str(e)}")
                logger.warning("Make sure the server is running on http://localhost:8000")
            
            # Signal that models are ready
            self.model_ready.set()
            logger.info("All models initialized successfully!")
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error("Traceback:", exc_info=True)
            self.shutdown_flag.set()
    
    def _process_queue(self):
        """Process audio recordings from the queue"""
        while not self.shutdown_flag.is_set():
            try:
                # Get the next recording from the queue
                audio_array = self.processing_queue.get(timeout=1.0)
                
                # Process the recording
                self._process_recording(audio_array)
                
                # Mark task as done
                self.processing_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing recording: {str(e)}")
                continue
    
    def audio_callback(self, indata, frames, time_info, status):
        if status:
            logger.warning(f"Status: {status}")
        
        # Convert to mono if necessary and add to buffer
        audio_data = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        audio_data = audio_data.astype(np.float32)
        
        # Add to monitoring buffer
        self.monitoring_buffer = np.concatenate((self.monitoring_buffer, audio_data))
        
        # Add to VAD buffer
        self.vad_buffer = np.concatenate((self.vad_buffer, audio_data))
        
        # Process complete VAD frames
        while len(self.vad_buffer) >= self.vad_frame_samples:
            # Get a frame
            frame = self.vad_buffer[:self.vad_frame_samples]
            self.vad_buffer = self.vad_buffer[self.vad_frame_samples:]
            
            # Process with Silero VAD
            audio_tensor = torch.FloatTensor(frame).to(self.device)
            speech_prob = self.vad_model(audio_tensor, self.sample_rate).item()
            
            current_time = time.time()
            if speech_prob > 0.5:
                self.speech_detected = True
                self.last_speech_end = current_time
                if not self.is_recording:
                    self.start_recording()
            elif self.speech_detected and (current_time - self.last_speech_end) > self.padding_duration:
                self.stop_recording()
        
        # If we're recording, add to recording buffer
        if self.is_recording:
            self.recording_buffer = np.concatenate((self.recording_buffer, audio_data))
    
    def start_recording(self):
        logger.info("Speech detected, starting recording...")
        self.is_recording = True
        # Initialize recording buffer with the padding buffer
        self.recording_buffer = self.monitoring_buffer.astype(np.float32)  # Ensure float32
    
    def stop_recording(self):
        if not self.is_recording:
            return
            
        logger.info("Stopping recording...")
        self.is_recording = False
        self.speech_detected = False
        
        # Convert recording buffer to numpy array
        audio_array = self.recording_buffer.astype(np.float32)
        
        # Only process if recording is longer than 10 seconds
        if len(audio_array) > 10 * self.sample_rate:
            # Add to processing queue if not full
            if self.processing_queue.qsize() < self.max_queue_size:
                logger.info("Adding recording to processing queue...")
                self.processing_queue.put(audio_array)
            else:
                logger.warning("Warning: Processing queue full, skipping recording")
        
        # Clear recording buffer
        self.recording_buffer = np.array([], dtype=np.float32)
        self.monitoring_buffer = np.array([], dtype=np.float32)
    
    def _process_recording(self, audio_array):
        """Process a single recording (called from the processing thread)"""
        try:
            # Transcribe with Whisper
            logger.info("Transcribing audio...")
            result = self.whisper_model.transcribe(audio_array)
            transcript = result["text"]
            
            # Generate title and summary using Mistral
            logger.info("\nGenerating title and summary...")
            logger.info(f"Transcript length: {len(transcript)} chars")
            
            # Truncate transcript if too long (Mistral has a 2048 token limit)
            max_transcript_chars = 1000  # Conservative limit to leave room for prompt
            truncated_transcript = transcript[:max_transcript_chars]
            if len(transcript) > max_transcript_chars:
                truncated_transcript += "..."
                logger.info(f"Truncated to {len(truncated_transcript)} chars")
            
            prompt = f'''<s>[INST] Create a title and summary for this transcript. The output must be valid JSON.

Transcript: "{truncated_transcript}"

Requirements:
- Title: 2-5 meaningful words (no filler words like "okay", "um", "so")
- Summary: 2-3 complete, informative sentences

Output format:
{{"title": "Your Title Here", "summary": "Your summary here."}}

Remember to output ONLY valid JSON. [/INST]'''

            logger.info("\nPrompt length:", len(prompt))
            
            # Try multiple times with different temperatures
            attempts = 0
            max_attempts = 3
            temperatures = [0.7, 0.9, 0.5]
            
            while attempts < max_attempts:
                try:
                    logger.info(f"\nAttempt {attempts + 1} with temperature {temperatures[attempts]}")
                    start_time = time.time()
                    
                    response = self.openai_client.chat.completions.create(
                        model="mistral",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=200,
                        temperature=temperatures[attempts],
                        top_p=0.9,
                        num_return_sequences=1,
                        return_full_text=False
                    )['choices'][0]['message']['content'].strip()
                    
                    generation_time = time.time() - start_time
                    logger.info(f"Generation took {generation_time:.2f} seconds")
                    logger.info(f"Raw response ({len(response)} chars):\n{response[:500]}...")
                    
                    # Find JSON content
                    json_match = re.search(r'\{[^{]*\}', response)
                    if json_match:
                        json_str = json_match.group()
                        logger.info(f"\nExtracted JSON:\n{json_str}")
                        
                        result_dict = json.loads(json_str)
                        title = result_dict.get('title', '').strip()
                        summary = result_dict.get('summary', '').strip()
                        
                        logger.info(f"\nTitle: {title}")
                        logger.info(f"Summary: {summary}")
                        
                        # Validate title
                        title_words = [w for w in title.split() if w.strip()]
                        if len(title_words) >= 2 and len(title_words) <= 5 and not any(title.lower().startswith(w) for w in ["okay", "um", "uh", "so"]):
                            # Validate summary
                            summary_sentences = [s.strip() for s in summary.split('.') if s.strip()]
                            if len(summary_sentences) >= 2 and len(summary_sentences) <= 3:
                                logger.info("\nValidation passed!")
                                break
                        
                        logger.info("Failed validation:")
                        logger.info(f"Title words: {len(title_words)}")
                        logger.info(f"Summary sentences: {len(summary_sentences)}")
                    else:
                        logger.info("No JSON found in response")
                    
                    logger.info(f"\nAttempt {attempts + 1} failed validation, trying again...")
                    attempts += 1
                except json.JSONDecodeError as e:
                    logger.error(f"\nJSON parsing error in attempt {attempts + 1}: {str(e)}")
                    attempts += 1
                except Exception as e:
                    logger.error(f"\nError in attempt {attempts + 1}: {str(e)}")
                    logger.error(f"Error type: {type(e)}")
                    import traceback
                    logger.error("Traceback:", exc_info=True)
                    attempts += 1
            
            if attempts == max_attempts:
                logger.info("All attempts failed, using fallback")
                # Extract meaningful words for title
                words = transcript.split()
                skip_words = {"okay", "um", "uh", "like", "so", "just", "i", "the", "a", "an", "and", "but", "or", "if", "then"}
                title_words = []
                for word in words:
                    word = word.lower().strip('.,!?')
                    if len(word) > 2 and word not in skip_words and len(title_words) < 5:
                        title_words.append(word)
                    if len(title_words) == 5:
                        break
                
                # Ensure we have at least 2 words
                if len(title_words) < 2:
                    title_words.extend(['audio', 'recording'][:2 - len(title_words)])
                
                title = " ".join(word.capitalize() for word in title_words)
                
                # Create a meaningful summary from transcript content
                first_sentence = f"Discussion about {' '.join(title_words)}."
                second_sentence = "The conversation explores system functionality and performance."
                summary = f"{first_sentence} {second_sentence}"
            
            # Get current date/time information
            now = datetime.now()
            year = now.strftime("%Y")
            month = now.strftime("%m-%B")  # 01-January, 02-February, etc.
            day = now.strftime("%d-%A")    # 01-Monday, 02-Tuesday, etc.
            timestamp = now.strftime("%H-%M-%S")  # 24-hour format for better sorting
            
            # Clean the title to be filesystem-friendly
            clean_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
            
            # Create the directory structure: recordings/YYYY/MM-Month/DD-Day/HHMMSS_Title
            dir_name = os.path.join(
                "recordings",
                year,
                month,
                day,
                f"{timestamp}_{clean_title}"
            )
            os.makedirs(dir_name, exist_ok=True)
            
            # Save audio file
            logger.info(f"Saving recording to {dir_name}...")
            wavfile.write(f"{dir_name}/audio.wav", self.sample_rate, audio_array.astype(np.float32))
            
            # Save transcript
            with open(f"{dir_name}/transcript.txt", 'w') as f:
                f.write(transcript)
            
            # Save summary
            with open(f"{dir_name}/summary.txt", 'w') as f:
                f.write(f"Title: {title}\n\nSummary:\n{summary}\n\nFull Transcript:\n{transcript}")
            
            logger.info(f"Processing complete. Files saved to: {dir_name}")
        except Exception as e:
            logger.error(f"Error processing recording: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def shutdown(self):
        """Clean shutdown of the audio processor"""
        logger.info("\nShutting down audio processor...")
        
        # Stop the main loop
        self.shutdown_flag.set()
        
        # Wait for processing queue to finish
        logger.info("Waiting for processing queue to finish...")
        self.processing_queue.join()
        
        # Wait for model initialization thread if still running
        if self.init_thread.is_alive():
            logger.info("Waiting for model initialization to finish...")
            self.init_thread.join()
        
        logger.info("Shutdown complete")
    
    def start(self):
        """Start audio monitoring"""
        try:
            logger.info("\nStarting audio monitoring...")
            # Use a larger block size than VAD frame size to reduce CPU usage
            block_samples = self.vad_frame_samples * 4  # Process 4 VAD frames at a time
            with sd.InputStream(callback=self.audio_callback,
                              channels=1,
                              samplerate=self.sample_rate,
                              blocksize=block_samples):
                logger.info("\nListening for speech... Press Ctrl+C to stop.")
                while not self.shutdown_flag.is_set():
                    time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("\nStopping audio monitoring...")
        except Exception as e:
            logger.error(f"Error: {str(e)}")
        finally:
            self.shutdown()

if __name__ == "__main__":
    # Create recordings directory if it doesn't exist
    os.makedirs("recordings", exist_ok=True)
    
    # Start the audio processor
    processor = AudioProcessor()
    try:
        processor.start()
    except KeyboardInterrupt:
        processor.shutdown()
