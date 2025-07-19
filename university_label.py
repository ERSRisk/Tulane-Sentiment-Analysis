import streamlit as st
import pandas as pd
import json
from datetime import datetime, timedelta
from google import genai
from google.genai.errors import ClientError
import requests
import re
import backoff
import os
import asyncio

# Initialize Gemini client
GEMINI_API_KEY = os.getenv("PAID_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

# Gemini API call
def call_gemini(client, prompt):
    return client.models.generate_content(model="gemini-1.5-flash", contents=[prompt])

# Async error handling decorator (used inside process_article)
@backoff.on_exception(backoff.expo,
                      (genai.errors.ServerError, requests.exceptions.ConnectionError),
                      max_tries=6,
                      jitter=None,
                      on_backoff=lambda details: print(
                          f"Retrying after error: {details['exception']} (try {details['tries']} after {details['wait']}s)")
)
async def process_article(article, sem):
    async with sem:
        content = article['Content']
        title = article['Title']
        if pd.isna(content) or pd.isna(title):
            return None

        prompt = f"""
        Read the following title and content from the following article: 
        Title: {title}
        Article: {" ".join(str(content).split()[:200])}
        If the article refers to higher education, university lawsuits, or research funding in higher education, 
        return a **compact and valid JSON object**, properly escaped, without explanations:
        {{
            "Title":"same title",
            "Content":"same content",
            "University Label": 1
        }}
        Else, set "University Label" to 0
        """

        try:
            response = await asyncio.to_thread(call_gemini, client, prompt)
            if hasattr(response, "text") and response.text:
                response_text = response.text
                json_str = re.search(r"```json\s*(\{.*\})\s*```", response_text, re.DOTALL)
                try:
                    if json_str:
                        return json.loads(json_str.group(1))
                    else:
                        return json.loads(response_text)
                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON decode error: {e}")
                    return None
        except ClientError as e:
            print(f"❌ ClientError: {e}")
            return None
        except requests.exceptions.ConnectionError:
            print("⚠️ Connection error, skipping...")
            return None

# Run all in async
async def university_label_async(articles, concurrency=10):
    sem = asyncio.Semaphore(concurrency)
    tasks = [process_article(row, sem) for _, row in articles.iterrows()]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]

# Streamlit UI
st.title("Article Risk Review Portal")

st.sidebar.header("Filter Articles")
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=30))
end_date = st.sidebar.date_input("End Date", datetime.now())

if start_date > end_date:
    st.sidebar.error("Start date must be before end date.")

articles = pd.read_csv('Model_training/BERTopic_results.csv')
print(len(articles))

results = asyncio.run(university_label_async(articles))
results_df = pd.DataFrame(results)
results_df.to_csv('BERTopic_before.csv', index=False)
