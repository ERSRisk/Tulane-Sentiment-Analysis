from google import genai
import pandas as pd
import re
from datetime import timedelta
from google.genai.errors import ClientError
import time
import asyncio
import datetime
import tweepy
import json
import re
import os
import toml

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY_NEWS")
GEMINI_API_KEY_X = os.getenv("GEMINI_API_KEY_X")
X_API_KEY = os.getenv("X_API_KEY")

start_date = (datetime.date.today() - timedelta(days = 7))
end_date = datetime.date.today()

semaphore = asyncio.Semaphore(3)

async def limited_process(batch_df, search, batch_size, total_batches):
        async with semaphore:
            return await asyncio.to_thread(process_batch, batch_df, search, batch_size, total_batches)

async def analyze_in_batches_concurrent_X(tweets, search, batch_size=10):
        all_responses = []
        total_batches = len(tweets) // batch_size + (1 if len(tweets) % batch_size != 0 else 0)
        print(f"Total batches: {total_batches}")
        batch_tasks = []
        for i in range(0, len(tweets), batch_size):
            batch_df = tweets[i:i + batch_size]




            task = limited_process(batch_df, search, i // batch_size + 1, total_batches)
            batch_tasks.append(task)

        all_responses = await asyncio.gather(*batch_tasks)
        all_responses = [pd.DataFrame(response) for response in all_responses if response]
        return pd.concat(all_responses)

async def run_async_batches_X(tweets, search, batch_size = 10):
    return await analyze_in_batches_concurrent_X(tweets, search, batch_size = 10)

def process_batch(batch_df, search, i, total_batches):
    print(f"Processing batch {i} of {total_batches}...")
    formatted_tweets = "\n\n".join([f"{tweet.text}" for tweet in batch_df])
    analysis_list = analyze_sentiment_X(formatted_tweets, search)
    response = []
    if not analysis_list:
        print("analysis list is missing")
    elif len(analysis_list) != len(batch_df):
        print(f"Got {len(analysis_list)} responses for {len(batch_df)} tweets.")
    for tweet, analysis in zip(batch_df, analysis_list):
        result = {
        "created_at": tweet.created_at.isoformat(),
        "text": tweet.text,
        "link": f"https://twitter.com/{tweet.author_id}/status/{tweet.id}",
        "description": analysis.get("description", "parse error"),
        "sentiment": analysis.get("sentiment", 0),
        "summary": analysis.get("summary", "parse error"),
        "is_sport": analysis.get("is_sport", 0),
        "affiliation": analysis.get("affiliation", 0)
        }
        if result['is_sport'] == 0 and result['affiliation'] == 1:
            response.append(result)
    return response

#This function fetches tweets from Twitter API based on the search term and date range provided by the user.
# It uses the Tweepy library to interact with the Twitter API and returns a DataFrame with the tweet data.
# The function takes the following parameters:
# search: The search term to look for in tweets.
# start_date: The start date for the tweet search.
# end_date: The end date for the tweet search.
# no_of_tweets: The number of tweets to fetch.
def fetch_twits(search, start_date, end_date, no_of_tweets):
    client = tweepy.Client(bearer_token=X_API_KEY)
    response = client.search_recent_tweets(
        query=search,
        max_results=100,
        tweet_fields=["created_at"],
        start_time=datetime.datetime.combine(start_date, datetime.time.min).isoformat() + "Z",
        end_time=(datetime.datetime.combine(end_date, datetime.time.min)).isoformat() + "Z"
    )
    tweets = response.data
    if not tweets:
        print('No tweets found for the given search term and date range.')
    return tweets

def analyze_sentiment_X(formatted_tweet_block, search, retries=5):
    flagged_keywords = ["1. Civil Rights", "2. Antisemitism", "3. Federal Grants", "4. Contracts", 
                            "5. Discrimination", "6. Education Secretary", "7. Investigation", "8. Lawsuit", 
                            "9. Executive Order", "10. Title IX", "11. Transgender Athletes", "12. Diversity, Equity, and Inclusion (DEI)", 
                            "13. Funding Freeze, funding frost", "14. University Policies", "15. Student Success", '16. Allegations', "17. Compliance", 
                            "18. Oversight", "19. Political Activity", 
                            "20. Community Relations"]
    client = genai.Client(api_key=GEMINI_API_KEY_X)
    prompt = f"""
            Analyze the following tweets. For each one, return a JSON object with:
            - "description": a short description of the tweet,
            - "sentiment": a sentiment score from -1 to 1 (where -1 is very negative, 0 is neutral, and 1 is very positive) based on its relation with the keywords '{search}' and {flagged_keywords}. If it's an academic study conducted by Tulane that is recognized, give it a high score. Very low scores should be given to topics regarding the following issues if shed in a negative light: {flagged_keywords},
            - "summary": a summary of the sentiment and key reasons why the sentiment is positive, neutral, or negative based on its relation with the keywords '{search}',
            - "is_sport": 1 if the tweet is related to sports, if the word 'player', 'playoffs', 'NFL', 'season', or any other sport related is found in the text of the post, or the description contains references to sports seasons, match results, baseball, football, player recruitment; else, give it a 0.
            - "affiliation": 1 if the tweet is affiliated with Tulane directly and has bearing on Tulane's reputation as an organization; else, if the post is indirectly related to Tulane or has no bearing on the organization, give it a 0.

            Respond as a JSON array of objects, one per tweet, in the order presented.

            Tweets:
            {formatted_tweet_block}
            """
    models = ["gemini-1.5-flash", "gemini-2.5-pro-preview-05-06", 'gemini-2.0-flash']
    for attempt in range(retries): 
        for model in models:
            try:
                
                response = client.models.generate_content(model=model, contents=prompt)
                if response.candidates and response.candidates[0].content.parts:
                    raw_text = response.candidates[0].content.parts[0].text
                    print(raw_text)
                    cleaned_text = re.sub(r"```json|```|\n|\s{2,}", "", raw_text).strip()
                    return json.loads(cleaned_text)
            
            except ClientError as e:
                if "RESOURCE_EXHAUSTED" in str(e):
                    wait_time = 60
                    retry_delay_match = re.search(r"'retryDelay': '(\d+)s'", str(e))
                    if retry_delay_match:
                        wait_time = int(retry_delay_match.group(1))
                    print(f"⚠️ API quota exceeded. Retrying in {wait_time} seconds...")
                    time.sleep(60)
            except Exception:
                continue
    return print("API Failed. Check API Key or model.")

def load_existing_posts_X():
    if os.path.exists('Online_Extraction/tweets.json'):
        with open('Online_Extraction/tweets.json', 'r', encoding = 'utf-8') as f:
            return json.load(f)
    return []

def save_new_posts_X(existing_posts, new_posts):
    new_posts = new_posts.to_dict(orient="records")
    existing_urls_X = {post['link'] for post in existing_posts}
    unique_new_posts = [post for post in new_posts if post['link'] not in existing_urls_X]
    
    updated_posts = existing_posts + unique_new_posts
    with open('Online_Extraction/tweets.json', 'w', encoding = 'utf-8') as f:
            json.dump(updated_posts, f, indent = 2, ensure_ascii=False)
    return updated_posts
