import news as test
import datetime
from google import genai
import asyncio
from datetime import timedelta
import json
import toml
import tweets_extract as te
import rss
import os

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY_NEWS")
GEMINI_API_KEY_X = os.getenv("GEMINI_API_KEY_X")
X_API_KEY = os.getenv("X_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)
search = 'Tulane'
start_date = (datetime.date.today() - timedelta(days = 7))
start_date_X = (datetime.date.today() - timedelta(days = 6))
end_date = datetime.date.today()
timezone_option = 'CDT'

articles = test.fetch_news(search, start_date, end_date)
articles = test.get_articles_with_full_content(articles, timezone=timezone_option)
unique_articles = []
seen_titles = set()
for article in articles:
    if article['title'] not in seen_titles:
        unique_articles.append(article)
        seen_titles.add(article['title'])
articles = unique_articles

gemini_response_text = test.run_async_batches(articles, search, timezone_option, batch_size=10)
results = test.text_to_dataframe(gemini_response_text, articles)
existing_articles_news = test.load_existing_articles_news()
new_articles_news = test.save_new_articles_news(existing_articles_news, results)
with open('Online_Extraction/extracted_news.json', 'w') as f:
        json.dump(new_articles_news, f)

tweets = te.fetch_twits(search, start_date_X, end_date, 100)
df = asyncio.run(te.run_async_batches_X(tweets, search, batch_size=10))
existing_posts = te.load_existing_posts_X()
new_posts = te.save_new_posts_X(existing_posts, df)
with open('Online_Extraction/tweets.json', 'w') as f:
    json.dump(new_posts, f)
