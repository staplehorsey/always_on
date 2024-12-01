import os
import sys
import json
import time
import queue
import socket
import logging
import asyncio
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
        
        # Audio processing parameters
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
        self.speech_threshold = 0.01
        self.min_speech_duration = 1.0  # seconds
        self.speech_cooldown = 1.5  # seconds
        self.silence_threshold = 0.005
        self.consecutive_silence = 0
        
        # Processing queue
        self.max_queue_size = 10
        self.processing_queue = asyncio.Queue(maxsize=self.max_queue_size)
        self.queue_processor_task = None  # Will store the queue processor task
        self.queue_processor_running = True  # Control flag for the processor
        
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

    async def process_audio_chunk(self, chunk):
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
            
            # Resample and normalize for VAD processing
            resampled = self.resample_audio(original_samples, self.server_sample_rate, self.target_sample_rate)
            
            # Add to monitoring buffer (use resampled audio for VAD)
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
                        self.is_recording = True
                        self.consecutive_silence = 0
                        
                        if not self.current_recording:
                            logger.info("Speech detected, starting recording...")
                            self.current_recording = self.monitoring_buffer.tolist()
                        
                        self.current_recording.extend(self.audio_buffer.tolist())
                    else:
                        # Only stop recording if we've been silent for longer than the cooldown
                        if self.is_recording:
                            self.consecutive_silence += len(self.audio_buffer) / self.target_sample_rate
                            if self.consecutive_silence >= self.speech_cooldown:
                                recording_duration = len(self.current_recording) / self.target_sample_rate
                                if recording_duration >= self.min_speech_duration:
                                    logger.info("Speech ended, processing recording...")
                                    self.is_recording = False
                                    
                                    # Add trailing silence as padding
                                    pad_samples = int(self.monitoring_buffer_duration * self.target_sample_rate)
                                    if len(self.monitoring_buffer) > pad_samples:
                                        self.current_recording.extend(self.monitoring_buffer[-pad_samples:].tolist())
                                    
                                    # Add to processing queue
                                    await self._add_to_queue(np.array(self.current_recording))
                                else:
                                    logger.debug(f"Discarding short speech segment ({recording_duration:.2f}s)")
                                self.current_recording = []
                                self.consecutive_silence = 0
                            else:
                                # Keep recording during the cooldown period
                                self.current_recording.extend(self.audio_buffer.tolist())
                    
                    # Reset buffer
                    self.audio_buffer = np.zeros(self.vad_frame_size, dtype=np.float32)
                    self.audio_buffer_idx = 0
            
            # Trim monitoring buffer to save memory
            max_buffer_samples = int((self.monitoring_buffer_duration * 2) * self.target_sample_rate)
            if len(self.monitoring_buffer) > max_buffer_samples:
                self.monitoring_buffer = self.monitoring_buffer[-max_buffer_samples:]
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            logger.error(traceback.format_exc())
            self.audio_buffer = np.zeros(self.vad_frame_size, dtype=np.float32)

    async def _process_recording(self, audio_array):
        """Process a single recording"""
        try:
            # Convert audio to float32 for Whisper
            audio_array = np.array(audio_array, dtype=np.float32)
            
            # Ensure audio is in the correct range [-1, 1]
            if audio_array.max() > 1 or audio_array.min() < -1:
                audio_array = np.clip(audio_array, -1, 1)
            
            # Transcribe with Whisper
            logger.info("Transcribing audio with Whisper...")
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
                logger.info(f"Transcript length: {len(transcript)} chars")
                
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

                # Use non-async OpenAI client since llama.cpp doesn't support async
                try:
                    response = self.openai_client.chat.completions.create(
                        model="mistral",  # Model name doesn't matter for llama.cpp
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
                    
                    # Create a meaningful summary from transcript content
                    first_sentence = f"Discussion about {' '.join(title_words)}."
                    second_sentence = "The conversation explores various topics and ideas."
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
            import scipy.io.wavfile as wavfile
            wavfile.write(f"{dir_name}/audio.wav", self.target_sample_rate, audio_array.astype(np.float32))
            
            # Save transcript
            with open(f"{dir_name}/transcript.txt", 'w') as f:
                f.write(transcript)
            
            # Save summary
            with open(f"{dir_name}/summary.txt", 'w') as f:
                f.write(f"Title: {title}\n\nSummary:\n{summary}\n\nFull Transcript:\n{transcript}")
            
            logger.info(f"Processing complete. Files saved to: {dir_name}")
        except Exception as e:
            logger.error(f"Error processing recording: {str(e)}")
            logger.error(traceback.format_exc())

    async def _process_queue(self):
        """Process recordings in the queue"""
        logger.info("Starting queue processor...")
        while self.queue_processor_running:
            try:
                # Get recording from queue with timeout
                try:
                    audio_array = await asyncio.wait_for(self.processing_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Process recording
                try:
                    await self._process_recording(audio_array)
                except Exception as e:
                    logger.error(f"Error processing recording: {str(e)}")
                finally:
                    # Mark task as done
                    self.processing_queue.task_done()
                    
            except Exception as e:
                logger.error(f"Error in queue processor: {str(e)}")
                continue

    async def _add_to_queue(self, audio_array):
        """Add recording to processing queue"""
        try:
            if self.processing_queue.qsize() >= self.max_queue_size:
                logger.warning("Queue full, dropping recording")
                return
                
            await self.processing_queue.put(audio_array)
            logger.info(f"Added recording to queue. Queue size: {self.processing_queue.qsize()}")
        except Exception as e:
            logger.error(f"Error adding to queue: {str(e)}")

    async def run(self):
        """Main processing loop"""
        try:
            logger.info("Starting audio processing...")
            
            # Start queue processor task
            self.queue_processor_running = True
            self.queue_processor_task = asyncio.create_task(self._process_queue())
            
            # Connect to audio server
            logger.info(f"Connecting to {self.server_host}:{self.server_port}...")
            self.sock.connect((self.server_host, self.server_port))
            logger.info("Connected to audio server")
            
            # Start audio monitoring if enabled
            if self.enable_monitoring:
                logger.info("Audio monitoring enabled - you should hear the incoming audio")
                # Setup input stream for recording
                self.audio_input_stream = sd.InputStream(
                    channels=1,
                    samplerate=self.target_sample_rate,
                    blocksize=1024,
                    dtype=np.float32
                )
                self.audio_input_stream.start()
                self.audio_output_stream.start()
            
            while True:
                try:
                    data = self.sock.recv(4096)
                    if not data:
                        break
                    
                    # Convert bytes to float32 array
                    audio_chunk = np.frombuffer(data, dtype=np.float32)
                    await self.process_audio_chunk(audio_chunk)
                    
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
            await self.cleanup()
            
    async def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up resources...")
        
        # Stop queue processor
        logger.info("Stopping queue processor...")
        self.queue_processor_running = False
        if self.queue_processor_task:
            try:
                await asyncio.wait_for(self.queue_processor_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Queue processor task did not finish in time")
            except Exception as e:
                logger.error(f"Error stopping queue processor: {e}")
        
        # Stop audio monitoring
        if self.enable_monitoring:
            if hasattr(self, 'audio_input_stream'):
                try:
                    self.audio_input_stream.stop()
                    self.audio_input_stream.close()
                except Exception as e:
                    logger.error(f"Error closing input stream: {e}")
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
        
        # Wait for processing queue to finish with timeout
        try:
            logger.info("Waiting for processing queue to finish (max 10s)...")
            await asyncio.wait_for(self.processing_queue.join(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning("Timed out waiting for processing queue")
        except Exception as e:
            logger.error(f"Error waiting for processing queue: {e}")
        
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
        asyncio.run(processor.run())
    except KeyboardInterrupt:
        pass
