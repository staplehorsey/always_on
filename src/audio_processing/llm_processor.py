import json
import logging
import openai
from typing import Dict, Optional, Tuple

logger = logging.getLogger('audio-processor.llm')

class LLMProcessor:
    def __init__(self, base_url: str = "http://rat.local:8080/v1",
                 api_key: str = "sk-no-key-required"):
        self.base_url = base_url
        self.api_key = api_key
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
    
    def generate_title_and_summary(self, transcript: str,
                                 max_transcript_chars: int = 1000) -> Tuple[str, str]:
        """Generate title and summary for transcript using local LLM"""
        try:
            # Truncate transcript if too long
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

            response = self.client.chat.completions.create(
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
                    return title, summary
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing JSON response: {e}")
                    logger.error(f"Raw response: {response_text}")
                    raise
            
            raise Exception("No response from LLM")
            
        except Exception as e:
            logger.error(f"Error generating title/summary: {e}")
            return self.generate_fallback_title_summary(transcript)
    
    def generate_fallback_title_summary(self, transcript: str) -> Tuple[str, str]:
        """Generate fallback title and summary when LLM fails"""
        words = transcript.split()
        skip_words = {"okay", "um", "uh", "like", "so", "just", "i", "the",
                     "a", "an", "and", "but", "or", "if", "then"}
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
        
        return title, summary
    
    def cleanup(self) -> None:
        """Clean up LLM resources"""
        if self.client is not None:
            try:
                self.client.close()
            except Exception as e:
                logger.error(f"Error closing OpenAI client: {e}")
