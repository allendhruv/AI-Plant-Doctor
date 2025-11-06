# ai_helper.py

import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key securely
api_key = os.getenv("OPENROUTER_API_KEY")

def get_disease_solution(disease_name):
    """
    Fetches a practical treatment solution for the given plant disease using OpenRouter API.
    """
    if not api_key:
        raise RuntimeError("API key for OpenRouter is missing. Check your .env file.")

    prompt = (
        f"Suggest the best treatment solution for the plant disease in points: {disease_name}. "
        "Keep it practical, and actionable for farmers."
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5000",  # Optional but good practice
    }

    payload = {
        "model": "openai/gpt-3.5-turbo",  # we can also try "anthropic/claude-3" if needed
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150,
        "temperature": 0.7,
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        content = response.json().get('choices', [{}])[0].get('message', {}).get('content')
        
        if not content:
            raise ValueError("No content found in AI response.")
        
        return content.strip()

    except requests.RequestException as e:
        print(f"[ERROR] OpenRouter API Request failed: {e}")
        raise RuntimeError("Failed to fetch AI solution from OpenRouter.") from e

    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        raise
