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

def university_label(articles, batch_size = 5, delay =5):
    def call_gemini(prompt):
        GEMINI_API_KEY = os.getenv("PAID_API_KEY")
        client = genai.Client(api_key=GEMINI_API_KEY)
        return client.models.generate_content(model="gemini-1.5-flash", contents=[prompt])
    
    # 🧠 Async article processor
    @backoff.on_exception(backoff.expo,
                          (genai.errors.ServerError, requests.exceptions.ConnectionError),
                          max_tries=6,
                          jitter=None,
                          on_backoff=lambda details: print(
                              f"Retrying after error: {details['exception']} (try {details['tries']} after {details['wait']}s)", flush=True)
    )
    async def process_article(article, sem, batch_number=None, total_batches=None, article_index=None):
        async with sem:
            try:
                if batch_number is not None and total_batches is not None and article_index is not None:
                    print(f"📦 Processing Batch {batch_number} of {total_batches} | Article {article_index}", flush=True)
    
                content = article['Content']
                title = article['Title']
                if pd.isna(content) or pd.isna(title):
                    return None
    
                prompt = f"""
                Read the following title and content from the following article: 
                Title: {title}
                Content: {" ".join(str(content).split()[:200])}
                Check each article Title and Content for news regarding higher education, university news, or
                university funding. If the article refers to higher education or university news, 
                return a **compact and valid JSON object**, properly escaped, without explanations:
                {{
                    "Title":"same title",
                    "Content":"same content",
                    "University Label": 1
                }}
                Else, set "University Label" to 0
                """
    
                response = await asyncio.to_thread(call_gemini, prompt)
                if hasattr(response, "text") and response.text:
                    response_text = response.text
                    json_str = re.search(r"```json\s*(\{.*\})\s*```", response_text, re.DOTALL)
                    raw = json_str.group(1) if json_str else response_text
    
                    try:
                        return json.loads(raw)
                    except json.JSONDecodeError as e1:
                        try:
                            return ast.literal_eval(raw)
                        except Exception as e2:
                            print(f"⚠️ JSON decode fallback error: {e1} | Eval error: {e2}", flush=True)
                            return None
            except Exception as e:
                print(f"🔥 Uncaught error in article {article_index} of batch {batch_number}: {e}", flush=True)
                return None

    # 🚀 Async batch runner
async def university_label_async(articles, batch_size=15, concurrency=10):
    sem = asyncio.Semaphore(concurrency)
    tasks = []

    total_articles = len(articles)
    total_batches = (total_articles + batch_size - 1) // batch_size
    for start in range(0, total_articles, batch_size):
        batch_number = (start // batch_size) + 1
        print(f"🚚 Starting Batch {batch_number} of {total_batches}", flush=True)
        batch = articles.iloc[start:start+batch_size]
        for i, (_, row) in enumerate(batch.iterrows()):
            tasks.append(process_article(row, sem,
                                         batch_number=batch_number,
                                         total_batches=total_batches,
                                         article_index=i+1))
    
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]

def load_university_label():
    all_articles = pd.read_csv('Model_training/BERTopic_results.csv')
    try:
        existing = pd.read_csv('BERTopic_before.csv')
        labeled_titles = set(existing['Title']) if 'Title' in existing else set()
    except FileNotFoundError:
        existing = pd.DataFrame()
        labeled_titles = set()

    new_articles = all_articles[~all_articles['Title'].isin(labeled_titles)]
    print(f"🔎 Total articles: {len(all_articles)} | Unlabeled: {len(new_articles)}", flush=True)
    
    results = asyncio.run(university_label_async(new_articles))
    new_df = pd.DataFrame(results)
    if not existing.empty:
        combined = pd.concat([existing, new_df], ignore_index = True)
    else:
        combined = new_df
    return combined

def combine_into_dataframe():
    all_articles = pd.read_csv('Model_training/BERTopic_results.csv')
    labeled_df = pd.read_csv('BERTopic_before.csv')
    merged = pd.merge(all_articles, labeled_df[['Title', 'University Label']], on = 'Title', how = 'left')
    merged['University Label'] = merged['University Label'].fillna(0).astype(int)
    merged.to_csv('BERTopic_results_test.csv', index = False)

results_df = load_university_label()
results_df.to_csv('BERTopic_before.csv', index=False)
print("✅ Done! Saved as BERTopic_before.csv", flush=True)
combine_into_dataframe()

