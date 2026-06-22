import os
from google import genai
from google.genai.errors import ClientError, ServerError
import random
import re
import time
import requests
from config.settings import GEMINI_API_KEY

def get_gemini_client():
    return genai.Client(api_key=GEMINI_API_KEY)

def call_gemini(prompt, max_tries = 8):
        GEMINI_API_KEY = os.getenv('PAID_API_KEY')
        client = genai.Client(api_key=GEMINI_API_KEY)
        last_err = None
        for attempt in range(1, max_tries +1):
            try:
                return client.models.generate_content(model="gemini-2.5-flash", contents=[prompt])
            except ClientError as e:
                msg = str(e).lower()
                if ("resource exhausted" in msg) or ("quota" in msg) or ("429" in msg):
                    s = str(e)
                    m = re.search(r"retryDelay'\s*:\s*'(\d+)s'", s) or re.search(r"retryDelay\s*[:=]\s*'?(\d+)s'?", s, flags=re.I)
                    retry_delay = int(m.group(1)) if m else None
                    if retry_delay is None:
                        retry_delay = min(120, (2 ** (attempt - 1))) + random.uniform(0, 1.5)
                    print(f"Gemini quota/rate limit (attempt {attempt}/{max_tries}). Sleeping {retry_delay:.1f}s...", flush=True)
                    time.sleep(retry_delay)
                    last_err = e
                    continue
                raise
            except (ServerError, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                # transient backend/network
                wait = min(120, (2 ** (attempt - 1))) + random.uniform(0, 1.5)
                print(f"⚠️ Gemini transient error (attempt {attempt}/{max_tries}): {e} | Sleeping {wait:.1f}s...", flush=True)
                time.sleep(wait)
                last_err = e
                continue
    
            except Exception as e:
                # Unknown error: small backoff a few times, then fail
                wait = min(30, (2 ** (attempt - 1))) + random.uniform(0, 1.0)
                print(f"⚠️ Gemini unexpected error (attempt {attempt}/{max_tries}): {e} | Sleeping {wait:.1f}s...", flush=True)
                time.sleep(wait)
                last_err = e
                continue
    
        raise RuntimeError(f"Gemini failed after {max_tries} attempts. Last error: {last_err}")