import streamlit as st
import pandas as pd
import json
from datetime import datetime
from datetime import timedelta
from google import genai
from google.genai.errors import ClientError
import toml
import time
import requests
import re
import backoff
import os

@backoff.on_exception(backoff.expo,
                        (genai.errors.ServerError, requests.exceptions.ConnectionError),
                        max_tries = 6,
                        jitter = None,
                        on_backoff = lambda details: print(
                            f"Retrying after error: {details['exception']} (try {details['tries']} after {details['wait']}s)"
                        ))

def call_gemini(client, prompt):
    return client.models.generate_content(model = "gemini-1.5-flash", contents = [prompt])
st.title("Article Risk Review Portal")
#give me a filter to filter articles by date range
st.sidebar.header("Filter Articles")
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=30))
end_date = st.sidebar.date_input("End Date", datetime.now())


if start_date > end_date:
    st.sidebar.error("Start date must be before end date.")
# Load articles and risks

articles = pd.read_csv('Model_training/BERTopic_results.csv')
print(len(articles))
def university_label(articles, batch_size = 10, delay =5):
    GEMINI_API_KEY = os.getenv('PAID_API_KEY')
    client = genai.Client(api_key=GEMINI_API_KEY)
    results = []
    for start in range(0, len(articles), batch_size):
        batch = articles.iloc[start:start + batch_size]
        batch_number = start // batch_size + 1
        total_batches = (len(articles) + batch_size - 1) // batch_size
        print(f"📦 Processing batch {batch_number} of {total_batches}...", flush = True)
        for _, article in batch.iterrows():
            content = article['Content']
            title = article['Title']
            if pd.isna(content) or pd.isna(title):
                continue
            try:
                prompt = (
                    f"""
                    Read the following title and content from the following article: \
                    Title: {article['Title']}"
                    Article: {" ".join(str(article['Content']).split()[:200])}
                    If the article refers to higher education, university lawsuits, or research funding in higher education, 
                    return a JSON object like:
                    {{
                        "Title":"same title",
                        "Content":"same content",
                        "University Label": 1
                    }}
                    Else, set "University Label" to 0
                    """
                )
                response = call_gemini(client, prompt)
                
                if response.text:
                    response_text = response.text
                    json_str = re.search(r"```json\s*(\{.*\})\s*```", response_text, re.DOTALL)
                    if json_str:
                        parsed = json.loads(json_str.group(1))
                        results.append(parsed)
                    else:
                        try:
                            # Try parsing whole text as raw JSON (no backticks)
                            parsed = json.loads(response_text)
                            results.append(parsed)
                        except Exception:
                            print("⚠️ Could not parse Gemini output:", response_text[:200])
                            continue
            except ClientError as e:
                if "RESOURCE_EXHAUSTED" in str(e):
                    wait_time = 60  # Default wait time (1 minute)
                    retry_delay_match = re.search(r"'retryDelay': '(\d+)s'", str(e))
                    if retry_delay_match:
                        wait_time = int(retry_delay_match.group(1))  # Use API's recommended delay
            
                    print(f"⚠️ API quota exceeded. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ API request failed: {e}")
                    return "❌ API error encountered."
                continue
            except requests.exceptions.ConnectionError:
                wait_time = 2
                print(f"Connection error. Waiting for {wait_time:.2f} seconds before retrying...")
                time.sleep(wait_time)
                continue
    return results


results = university_label(articles)
results_df = pd.DataFrame(results)
results_df.to_csv('BERTopic_before.csv', index = False)
