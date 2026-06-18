from google import genai
from config.settings import GEMINI_API_KEY

def get_gemini_client():
    return genai.Client(api_key=GEMINI_API_KEY)