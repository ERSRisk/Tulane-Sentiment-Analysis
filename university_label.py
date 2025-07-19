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
import ast

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
async def process_article(article, sem, batch_number=None, total_batches=None, article_index=None):
    if batch_number is not None and total_batches is not None and article_index is not None:
        print(f"📦 Processing Batch {batch_number} of {total_batches} | Article {article_index}", flush=True)
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
                
                # Try parsing from triple backticks
                if json_str:
                    raw = json_str.group(1)
                else:
                    raw = response_text
                
                # Attempt robust parsing
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError as e1:
                    try:
                        parsed = ast.literal_eval(raw)  # fallback (still secure)
                    except Exception as e2:
                        print(f"⚠️ JSON decode fallback error: {e1} | Eval error: {e2}")
                        return None
        except ClientError as e:
            print(f"❌ ClientError: {e}")
            return None
        except requests.exceptions.ConnectionError:
            print("⚠️ Connection error, skipping...")
            return None

# Run all in async
async def university_label_async(articles, batch_size=15, concurrency=10):
    sem = asyncio.Semaphore(concurrency)
    tasks = []

    total_articles = len(articles)
    total_batches = (len(articles) + batch_size - 1) // batch_size
    for start in range(0, total_articles, batch_size):
        batch_number = (start // batch_size) + 1
        batch = articles.iloc[start:start+batch_size]
        for i, (_, row) in enumerate(batch.iterrows()):
            tasks.append(process_article(row, sem,
                                         batch_number=batch_number,
                                         total_batches=total_batches,
                                         article_index=i+1))
    
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
