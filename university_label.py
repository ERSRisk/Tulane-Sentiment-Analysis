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
import os

articles = pd.read_csv('Model_training/BERTopic_results.csv')

def university_label(articles, batch_size = 5, delay =5):
    GEMINI_API_KEY = os.getenv("PAID_API_KEY")
    client = genai.Client(api_key=GEMINI_API_KEY)
    results = []
    for start in range(0, len(articles), batch_size):
        batch = articles.iloc[start:start + batch_size]
        batch_number = start // batch_size + 1
        total_batches = (len(articles) + batch_size - 1) // batch_size
        print(f"📦 Processing batch {batch_number} of {total_batches}...")
        for _, article in batch.iterrows():
            try:
                prompt = (
                    f"""
                    Read the following title and content from the following article: \
                    Title: {article['Title']}"
                    Article: {article['Content']}
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
                response = client.models.generate_content(model="gemini-1.5-flash", contents=[prompt])
                
                if response.text:
                    response_text = response.text
                    json_str = re.search(r"```json\s*(\{.*\})\s*```", response_text, re.DOTALL)
                    parsed = json.loads(json_str.group(1))
                    results.append(parsed)
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
labeled_df = pd.DataFrame(results)
labeled_df.rename(columns = {"University Label": "University_Label"}, inplace = True)

merged_articles = pd.merge(
    articles,
    labeled_df[['Title', "Content", "University_Label"]],
    on = ["Title", "Content"],
    how = "left"
)

merged_articles['University_Label'] = merged_articles["University_Label"].fillna(-1).astype(int)

merged_articles.to_csv("Model_training/BERTopic_test.csv", index = False)
