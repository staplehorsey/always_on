import os
import sys
import json
import time
import queue
import socket
import logging
import threading
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import whisper
import openai
import sounddevice as sd
from scipy import signal

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('audio-processor')

class AudioProcessor:
    def __init__(self, server_host='staple.local', server_port=12345):
        self.server_host = server_host
        self.server_port = server_port
        
        # Initialize socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(0.1)  # 100ms timeout for non-blocking recv
        
        # Initialize Whisper model
        logger.info("Loading Whisper model...")
        self.model = whisper.load_model("base")
        
        # Initialize VAD model
        logger.info("Loading VAD model...")
        torch.hub.set_dir(os.path.expanduser('~/.cache/torch/hub'))
        self.vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                             model='silero_vad',
                                             force_reload=False)
        self.vad_model.eval()
        
        # Initialize OpenAI client for local server
        self.openai_client = openai.OpenAI(
            base_url="http://rat.local:8080/v1",
            api_key="sk-no-key-required"
        )
        
        # Processing queue
        self.max_queue_size = 10
        self.processing_queue = queue.Queue(maxsize=self.max_queue_size)
        self.shutdown_flag = threading.Event()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_queue)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Audio parameters
        self.target_sample_rate = 16000  # Required by Whisper
        self.server_sample_rate = 44100  # Server's sample rate
        self.is_recording = False
        self.current_recording = []
        self.monitoring_buffer = np.array([], dtype=np.float32)
        self.monitoring_buffer_duration = 0.5  # seconds
        self.monitoring_buffer_samples = int(self.target_sample_rate * self.monitoring_buffer_duration)
        
        # VAD parameters
        self.vad_frame_size = 512  # VAD requires 512 samples for 16kHz
        self.audio_buffer = np.zeros(self.vad_frame_size, dtype=np.float32)
        self.audio_buffer_idx = 0
        
        # Speech detection parameters
        self.speech_threshold = 0.5  # Silero VAD threshold
        self.min_speech_duration = 1.0  # seconds
        self.speech_cooldown = 1.5  # seconds
        self.consecutive_silence = 0
        self.last_speech_time = 0
        
        # Audio monitoring setup
        self.enable_monitoring = True
        if self.enable_monitoring:
            # Setup output stream for monitoring at server's sample rate
            self.audio_output_stream = sd.OutputStream(
                channels=1,
                samplerate=self.server_sample_rate,  # Use server's sample rate for playback
                blocksize=1024,
                dtype=np.float32
            )
        
        logger.info("Initialization complete")

    def resample_audio(self, audio_data, orig_sr, target_sr):
        """Resample audio data to target sample rate while preserving signal characteristics"""
        if orig_sr == target_sr:
            return audio_data
            
        # Calculate resampling ratio
        ratio = target_sr / orig_sr
        
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
        
        return resampled

    def process_audio_chunk(self, chunk):
        """Process a chunk of audio data"""
        try:
            # Convert bytes to float32 array
            original_samples = np.frombuffer(chunk, dtype=np.float32)
            
            # Play audio if monitoring is enabled (use original samples)
            if self.enable_monitoring:
                try:
                    gain = 1.5  # Reduced gain to prevent distortion
                    samples_to_play = original_samples * gain
                    # Clip to prevent distortion
                    samples_to_play = np.clip(samples_to_play, -1.0, 1.0)
                    self.audio_output_stream.write(samples_to_play)
                except Exception as e:
                    logger.error(f"Error playing audio: {e}")
            
            # Resample from server rate to target rate for VAD/processing
            resampled = self.resample_audio(original_samples, self.server_sample_rate, self.target_sample_rate)
            
            # Add to monitoring buffer
            self.monitoring_buffer = np.concatenate((self.monitoring_buffer, resampled))
            
            # Process resampled audio in VAD-sized chunks
            remaining_samples = len(resampled)
            processed_samples = 0
            
            while remaining_samples > 0:
                # Calculate how many samples we can add to the current buffer
                space_in_buffer = self.vad_frame_size - self.audio_buffer_idx
                samples_to_add = min(remaining_samples, space_in_buffer)
                
                # Add samples to buffer
                self.audio_buffer[self.audio_buffer_idx:self.audio_buffer_idx + samples_to_add] = \
                    resampled[processed_samples:processed_samples + samples_to_add]
                
                self.audio_buffer_idx += samples_to_add
                processed_samples += samples_to_add
                remaining_samples -= samples_to_add
                
                # If buffer is full, process it
                if self.audio_buffer_idx >= self.vad_frame_size:
                    # Ensure audio buffer is normalized for VAD
                    vad_buffer = self.audio_buffer.copy()
                    if np.max(np.abs(vad_buffer)) > 0:
                        vad_buffer = vad_buffer / np.max(np.abs(vad_buffer))
                    
                    # Convert to torch tensor
                    tensor = torch.from_numpy(vad_buffer).to('cuda' if torch.cuda.is_available() else 'cpu')
                    tensor = tensor.unsqueeze(0)  # Add batch dimension
                    
                    # Run VAD
                    speech_prob = self.vad_model(tensor, self.target_sample_rate).item()
                    current_time = time.time()
                    
                    # Handle speech detection with cooldown and padding
                    if speech_prob > self.speech_threshold:
                        if not self.is_recording:
                            logger.info("Speech detected, starting recording...")
                            self.is_recording = True
                            # Include some previous audio for context
                            buffer_duration = len(self.monitoring_buffer) / self.target_sample_rate
                            padding_samples = min(int(0.5 * self.target_sample_rate), len(self.monitoring_buffer))
                            self.current_recording = self.monitoring_buffer[-padding_samples:].tolist()
                        
                        self.consecutive_silence = 0
                        self.last_speech_time = current_time
                        self.current_recording.extend(self.audio_buffer.tolist())
                    elif self.is_recording:
                        # Calculate silence duration
                        silence_duration = current_time - self.last_speech_time
                        
                        # Keep recording during silence up to cooldown
                        self.current_recording.extend(self.audio_buffer.tolist())
                        
                        if silence_duration >= self.speech_cooldown:
                            # Check if recording meets minimum duration
                            recording_duration = len(self.current_recording) / self.target_sample_rate
                            if recording_duration >= self.min_speech_duration:
                                logger.info("Speech ended, processing recording...")
                                # Add a bit of trailing silence
                                padding_samples = min(int(0.5 * self.target_sample_rate), len(self.monitoring_buffer))
                                self.current_recording.extend(self.monitoring_buffer[-padding_samples:].tolist())
                                # Process the recording
                                self._add_to_queue(np.array(self.current_recording))
                            else:
                                logger.debug(f"Discarding short speech segment ({recording_duration:.2f}s)")
                            
                            # Reset recording state
                            self.is_recording = False
                            self.current_recording = []
                    
                    # Reset buffer
                    self.audio_buffer = np.zeros(self.vad_frame_size, dtype=np.float32)
                    self.audio_buffer_idx = 0
            
            # Trim monitoring buffer to save memory (keep last 2 seconds)
            max_samples = 2 * self.target_sample_rate
            if len(self.monitoring_buffer) > max_samples:
                self.monitoring_buffer = self.monitoring_buffer[-max_samples:]
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            logger.error(traceback.format_exc())
            self.audio_buffer = np.zeros(self.vad_frame_size, dtype=np.float32)
            self.audio_buffer_idx = 0

    def _process_queue(self):
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
                logger.error(f"Error in queue processor: {str(e)}")
                logger.error(traceback.format_exc())
                continue

    def _add_to_queue(self, audio_array):
        """Add recording to processing queue"""
        try:
            if self.processing_queue.qsize() >= self.max_queue_size:
                logger.warning("Queue full, dropping recording")
                return
                
            self.processing_queue.put(audio_array)
            logger.info(f"Added recording to queue. Queue size: {self.processing_queue.qsize()}")
        except Exception as e:
            logger.error(f"Error adding to queue: {str(e)}")

    def _process_recording(self, audio_array):
        """Process a single recording"""
        try:
            # Transcribe with Whisper
            logger.info("Transcribing audio...")
            result = self.model.transcribe(audio_array)
            transcript = result["text"].strip()
            
            if not transcript:
                logger.warning("No transcript generated - skipping title/summary generation")
                return
                
            logger.info(f"Transcription complete: {transcript}")
            
            # Skip title/summary generation if OpenAI client is not available
            if self.openai_client is None:
                logger.warning("OpenAI client not available - skipping title/summary generation")
                title = "Untitled Recording"
                summary = "Summary not available (LLM service not connected)"
            else:
                # Generate title and summary using local LLM
                logger.info("Generating title and summary...")
                
                # Truncate transcript if too long
                max_transcript_chars = 1000  # Conservative limit to leave room for prompt
                truncated_transcript = transcript[:max_transcript_chars]
                if len(transcript) > max_transcript_chars:
                    truncated_transcript += "..."
                    logger.info(f"Truncated to {len(truncated_transcript)} chars")
                
                system_prompt = """You are a helpful AI assistant that creates titles and summaries for audio transcripts.
Your task is to create a concise title and informative summary. The output must be valid JSON."""

                user_prompt = f"""Create a title and summary for this transcript:

"{truncated_transcript}"

Requirements:
- Title: 2-5 meaningful words (no filler words like "okay", "um", "so")
- Summary: 2-3 complete, informative sentences

Output format:
{{"title": "Your Title Here", "summary": "Your summary here."}}"""

                try:
                    response = self.openai_client.chat.completions.create(
                        model="mistral",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.7,
                        max_tokens=200
                    )
                    
                    if response and response.choices:
                        # Parse the JSON response
                        response_text = response.choices[0].message.content.strip()
                        try:
                            response_json = json.loads(response_text)
                            title = response_json.get("title", "Untitled Recording")
                            summary = response_json.get("summary", "No summary available.")
                        except json.JSONDecodeError as e:
                            logger.error(f"Error parsing JSON response: {e}")
                            logger.error(f"Raw response: {response_text}")
                            raise
                    else:
                        raise Exception("No response from LLM")
                        
                except Exception as e:
                    logger.error(f"Error generating title/summary: {e}")
                    # Use fallback title/summary
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
                    summary = f"Audio recording about {' '.join(title_words)}."
            
            # Save the recording
            now = datetime.now()
            year = now.strftime("%Y")
            month = now.strftime("%m-%B")
            day = now.strftime("%d-%A")
            timestamp = now.strftime("%H-%M-%S")
            
            # Clean the title
            clean_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
            
            # Create directory structure
            dir_name = os.path.join("recordings", year, month, day, f"{timestamp}_{clean_title}")
            os.makedirs(dir_name, exist_ok=True)
            
            # Save files
            logger.info(f"Saving recording to {dir_name}...")
            import scipy.io.wavfile as wavfile
            wavfile.write(f"{dir_name}/audio.wav", self.target_sample_rate, audio_array)
            
            with open(f"{dir_name}/transcript.txt", 'w') as f:
                f.write(transcript)
            
            with open(f"{dir_name}/summary.txt", 'w') as f:
                f.write(f"Title: {title}\n\nSummary:\n{summary}\n\nFull Transcript:\n{transcript}")
            
            logger.info(f"Processing complete. Files saved to: {dir_name}")
        except Exception as e:
            logger.error(f"Error processing recording: {e}")
            logger.error(traceback.format_exc())

    def run(self):
        """Main processing loop"""
        try:
            logger.info("Starting audio processing...")
            
            # Connect to audio server
            logger.info(f"Connecting to {self.server_host}:{self.server_port}...")
            self.sock.connect((self.server_host, self.server_port))
            logger.info("Connected to audio server")
            
            # Start audio monitoring if enabled
            if self.enable_monitoring:
                logger.info("Audio monitoring enabled - you should hear the incoming audio")
                self.audio_output_stream.start()
            
            while True:
                try:
                    data = self.sock.recv(4096)
                    if not data:
                        break
                    
                    # Convert bytes to float32 array
                    audio_chunk = np.frombuffer(data, dtype=np.float32)
                    self.process_audio_chunk(audio_chunk)
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    logger.error(f"Error processing chunk: {e}")
                    continue
                    
        except KeyboardInterrupt:
            logger.info("Stopping audio monitoring...")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
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
        
        # Stop audio monitoring
        if self.enable_monitoring:
            if hasattr(self, 'audio_output_stream'):
                try:
                    self.audio_output_stream.stop()
                    self.audio_output_stream.close()
                except Exception as e:
                    logger.error(f"Error closing output stream: {e}")
        
        # Close socket
        try:
            self.sock.close()
        except Exception as e:
            logger.error(f"Error closing socket: {e}")
        
        # Close OpenAI client
        if self.openai_client is not None:
            try:
                self.openai_client.close()
            except Exception as e:
                logger.error(f"Error closing OpenAI client: {e}")
        
        logger.info("Cleanup complete")

    def _audio_callback(self, indata, frames, time, status):
        """Audio callback for monitoring"""
        if status:
            logger.error(f"Audio callback error: {status}")
        if self.enable_monitoring:
            try:
                gain = 1.5  # Reduced gain to prevent distortion
                samples_to_play = indata * gain
                # Clip to prevent distortion
                samples_to_play = np.clip(samples_to_play, -1.0, 1.0)
                self.audio_output_stream.write(samples_to_play)
            except Exception as e:
                logger.error(f"Error playing audio: {e}")

if __name__ == "__main__":
    processor = AudioProcessor()
    try:
        processor.run()
    except KeyboardInterrupt:
        pass
