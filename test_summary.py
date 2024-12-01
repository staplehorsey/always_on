import os
import json
import torch
from transformers import pipeline

def test_summarization(transcript_path):
    print("\n=== Testing Summarization ===")
    
    # Initialize the model
    print("Loading Phi-2 model...")
    summarizer = pipeline(
        "text-generation",
        model="microsoft/phi-2",
        torch_dtype=torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Read the transcript
    print(f"\nReading transcript from: {transcript_path}")
    with open(transcript_path, 'r') as f:
        transcript = f.read()
    print(f"\nTranscript content:\n{transcript}\n")
    
    # Prepare prompt
    max_transcript_chars = 1000
    truncated_transcript = transcript[:max_transcript_chars]
    if len(transcript) > max_transcript_chars:
        truncated_transcript += "..."
    
    prompt = f'''Create a NEW and UNIQUE JSON summary of this transcript.

Transcript: "{truncated_transcript}"

Requirements:
1. Title must be EXACTLY 2-5 words
2. Summary must be EXACTLY 2-3 sentences
3. Response must be ONLY valid JSON, nothing else
4. Do NOT copy the example below - create a NEW summary
5. Be creative - use your own words!

Here is an example of the format (DO NOT COPY ANY TEXT FROM THIS):
{{"title": "Speech Pipeline Status", "summary": "Implemented a voice detection and transcription system with state management. Currently facing challenges with parallel processing and model reliability."}}

Your NEW and CREATIVE response (ONLY valid JSON):'''
    
    print(f"Generated prompt:\n{prompt}\n")
    
    # Generate response
    print("Generating response...")
    response = summarizer(
        prompt,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1,
        return_full_text=False
    )[0]['generated_text']
    
    print(f"\nRaw response:\n{response}\n")
    
    # Parse response
    try:
        # Find the first { and last }
        start = response.find('{')
        end = response.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = response[start:end]
            print(f"\nExtracted JSON string:\n{json_str}\n")
            result_dict = json.loads(json_str)
            
            # Validate title and summary
            title = result_dict.get('title', '')
            summary = result_dict.get('summary', '')
            
            title_words = title.split()
            if len(title_words) < 2 or len(title_words) > 5:
                print(f"\nWarning: Title '{title}' has {len(title_words)} words, should be 2-5 words")
            
            summary_sentences = summary.split('.')
            summary_sentences = [s.strip() for s in summary_sentences if s.strip()]
            if len(summary_sentences) < 2 or len(summary_sentences) > 3:
                print(f"\nWarning: Summary has {len(summary_sentences)} sentences, should be 2-3 sentences")
            
            print(f"\nParsed dictionary:\n{json.dumps(result_dict, indent=2)}\n")
            title = result_dict.get('title', 'Untitled Recording')
            summary = result_dict.get('summary', 'No summary available.')
        else:
            print("\nNo JSON found in response")
            words = transcript.split()[:10]
            title = " ".join(words[:5]) + "..."
            summary = transcript[:200] + "..."
    except Exception as e:
        print(f"\nError parsing model output: {str(e)}")
        words = transcript.split()[:10]
        title = " ".join(words[:5]) + "..."
        summary = transcript[:200] + "..."
    
    print("\nFinal output:")
    print(f"Title: {title}")
    print(f"Summary: {summary}")

if __name__ == "__main__":
    test_path = "/Users/msainz/Projects/always_on/recordings/2024/11-November/30-Saturday/20-58-01_Project Status Update/transcript.txt"
    test_summarization(test_path)
