import requests
import json

def test_llm_server():
    url = "http://localhost:8000/v1/chat/completions"
    
    # Test data
    data = {
        "prompt": "What are three interesting facts about neural networks?",
        "max_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    try:
        print("Sending request to LLM server...")
        response = requests.post(url, json=data)
        
        if response.status_code == 200:
            result = response.json()
            print("\nResponse received!")
            print("\nGenerated text:")
            print(result["choices"][0]["message"]["content"])
            print("\nToken usage:")
            print(json.dumps(result["usage"], indent=2))
        else:
            print(f"\nError: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_llm_server()
