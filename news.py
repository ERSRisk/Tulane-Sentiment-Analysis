import requests
from google import genai
import re
from datetime import timedelta
from newspaper import Article
from google.genai.errors import ClientError
import random
import time
import asyncio
from dateutil import parser
import datetime
import json
import re
import os

NEWS_API_KEY = st.secrets["all_my_api_keys"]["NEWS_API_KEY"]
GEMINI_API_KEY = st.secrets["all_my_api_keys"]["GEMINI_API_KEY_X"]

client = genai.Client(api_key=GEMINI_API_KEY)
search = 'Tulane'
start_date = (datetime.date.today() - timedelta(days = 7))
end_date = datetime.date.today()
timezone_option = 'CDT'

def fetch_news(search, start_date, end_date):
    news_url = (
            f"https://newsapi.org/v2/everything?q={search} NOT sports NOT Football NOT basketball&"
            f"from={start_date}&to={end_date}&sortBy=popularity&apiKey={NEWS_API_KEY}"
        )
    response = requests.get(news_url)
    if response.status_code == 200:
        news_data = response.json()
        return news_data.get("articles", [])  # Return the articles
    else:
        return []


## This function fetches the full content of an article using the Newspaper3k library.
## The News API only provides a snippet of the article, so this function is used to get the full text.
def fetch_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        return None


# This function combines the truncated content from the News API with the full content fetched from the article URL.
# It also extracts the date and time from the article's publishedAt field and formats it according to the selected timezone.
def get_articles_with_full_content(articles, timezone="CDT"):
    """Replace truncated content with full article text and extract formatted date and time"""
    updated_articles = []
    seen_titles = set()

    #Determine offset based on selected timezone
    if timezone == "UTC":
        offset = timedelta(hours=0)
        tz_label = "UTC"
    elif timezone == "CST":
        offset = timedelta(hours=-6)
        tz_label = "CST"
    elif timezone == "CDT":
        offset = timedelta(hours=-5)
        tz_label = "CDT"
    else:
        offset = timedelta(hours=0)
        tz_label = "UTC"

    for article in articles:
        title = article["title"]
        if title in seen_titles:
            continue  # Skip duplicate
        seen_titles.add(title)

        #Get full text if the content is truncated
        full_text = fetch_content(article['url']) or article.get('content')
        #Parse publishedAt and split into date and time
        original_dt_str = article.get("publishedAt", "N/A")
        #try to parse and convert
        try:
            original_dt = parser.parse(original_dt_str)
            adjusted_dt = original_dt + offset  # Convert from UTC to CST
            adjusted_date = adjusted_dt.strftime("%m/%d/%Y")
            adjusted_time = adjusted_dt.strftime("%I:%M %p ") + tz_label
        except Exception:
            adjusted_date = "N/A"
            adjusted_time = "N/A"

        updated_articles.append({
            "title": article["title"],
            "description": article.get("description", "No description available."),
            "content": full_text if full_text else article["content"],
            "url": article["url"],
            "original_datetime": original_dt_str,
            "adjusted_date": adjusted_date,
            "adjusted_time": adjusted_time
        })
    return updated_articles



# This function formats the articles into a string that can be sent to the Gemini API for sentiment analysis.
# It includes the title, description, content, and URL of each article.
def format_articles_for_prompt(articles):
    """Format the articles in a way that can be sent to Gemini."""
    return "\n\n".join(
        [f"Title: {article['title']}\nDescription: {article['description']}\nContent: {article['content']}\nURL: {article['url']}"
        for article in articles])
# This function analyzes the sentiment of the articles using the Gemini API.
# It sends a prompt to the API and returns the response.
# The prompt includes instructions for the API to analyze the sentiment based on the keywords provided.
def analyze_sentiment(text_to_analyze, search, retries=5):
    for attempt in range(retries):
        try:
            sentiment_prompt = (
                "Analyze the sentiment of the following news articles in relation to the keywords: "
                f"'{search}'.\n"
                "Assume all articles affect Tulane's reputation positively, neutrally, or negatively. \n"
                "Then, consider how the keywords also get discussed or portrayed in the article.\n"
                "Provide an overall sentiment score (-1 to 1, where -1 is very negative, 0 is neutral, and 1 is very positive(This is a continuous range)) \n"
                "Provide a summary of the sentiment and key reasons why the sentiment is positive, neutral, or negative, "
                "specifically in relation to the keywords.\n"
                "Make sure that you include the score from -1 to 1 in a continuous range (with decimal places) and include the title, "
                "sentiment score, summary, and a statement explaining how the article relates to the keywords.\n"
                "Separate article info by double newlines and always include 'Title:' before the headline and 'Sentiment:' before the score.\n"
                "If you encounter any articles related to sports, please exclude them from the analysis. Sports articles do not need to be summarized. \n"
                "Only judge the sentiment for each article in terms of how it mentions the keywords. Max amount of titles should be 100.\n\n"
                "If an article title has already been seen before, do not analyze it again.\n"
                "If Tulane is mentioned anywhere in the article — even in passing or in an author affiliation — state that clearly in your summary.\n"
                "Tulane was found in the text. Here is the full content for analysis.\n"
                f"{text_to_analyze}"
            )
            gemini_response = client.models.generate_content(model="gemini-1.5-flash", contents=[sentiment_prompt])
            return gemini_response.text if gemini_response and gemini_response.text else ""
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
        except requests.exceptions.ConnectionError:
            wait_time = 2 ** attempt + random.uniform(0, 1)
            print(f"Connection error. Waiting for {wait_time:.2f} seconds before retrying...")
            time.sleep(wait_time)
        return "❌ API failed after multiple attempts."




# This function processes a batch of articles concurrently using asyncio.
semaphore = asyncio.Semaphore(5)


# This function limits the number of concurrent tasks to avoid overwhelming the API.
async def limited_process(batch_df, search, batch_size, total_batches, timezone):
    async with semaphore:
        return await asyncio.to_thread(process_batch, batch_df, search, batch_size, total_batches, timezone)




# This function processes the articles in batches and returns the responses.
# It divides the articles into smaller batches and processes them concurrently.
# It also handles the pagination of the articles by keeping track of the batch number and total batches.
async def analyze_in_batches_concurrent(articles, search, timezone, batch_size=10):
    all_responses = []
    total_batches = len(articles) // batch_size + (1 if len(articles) % batch_size != 0 else 0)
    print(f"Total batches: {total_batches}")
    batch_tasks = []
    for i in range(0, len(articles), batch_size):
        batch_df = articles[i:i + batch_size]




        task = limited_process(batch_df, search, i // batch_size + 1, total_batches, timezone)
        batch_tasks.append(task)

    all_responses = await asyncio.gather(*batch_tasks)
    return '\n\n'.join(all_responses)




# This function runs the asynchronous batch processing and returns the final responses.
# It uses asyncio.run to execute the asynchronous function and waits for it to complete.
# It also handles the pagination of the articles by keeping track of the batch number and total batches.
def run_async_batches(articles, search, timezone, batch_size = 10):
    return asyncio.run(analyze_in_batches_concurrent(articles, search, timezone, batch_size))


# This function caches the responses from the Gemini API to avoid redundant API calls.
# It uses Streamlit's caching mechanism to store the responses based on the input parameters.
def cached_gemini_response(articles, search, timezone):
    return run_async_batches(articles, search, timezone, batch_size=10)


# This function processes a batch of articles and returns the response from the Gemini API.
# It formats the articles for the prompt and sends them to the API for sentiment analysis.
# It also handles the pagination of the articles by keeping track of the batch number and total batches.
def process_batch(batch_df, search, i, total_batches, timezone):


    processed_articles = get_articles_with_full_content(batch_df, timezone=timezone)
    formatted_batch = format_articles_for_prompt(processed_articles)
    print(f"Processing batch {i} of {total_batches}...")
    response = analyze_sentiment(formatted_batch, search)
    return response


                # This function processes the response from the Gemini API and formats it into a DataFrame.
                # It extracts the title, sentiment score, summary, and other relevant information from the response.
def text_to_dataframe(text, articles):
    rows = []
    sections = re.split(r'Title:\s*', text)[1:]

    for section in sections:
        title_match = re.match(r'(.*?)\nSentiment:', section, re.DOTALL)
        sentiment_match = re.search(r'Sentiment:\s*(-?\d+\.?\d*)', section)
        summary = re.search(r'Summary:\s*(.*?)(?=\n|$)', section, re.DOTALL)

        if title_match and sentiment_match:
            title = title_match.group(1).strip()
            sentiment = float(sentiment_match.group(1))
            summary = summary.group(1).strip() if summary else "No summary available."
            article_data = next((article for article in articles if article['title'].strip().lower() == title.lower()), None)
            if article_data is not None:
                rows.append({
                'Title': title,
                'Sentiment': sentiment,
                'URL': article_data.get('url'),
                'Original Datetime': article_data.get('original_datetime', 'N/A'),
                'Adjusted Date': article_data.get('adjusted_date', 'N/A'),
                'Adjusted Time': article_data.get('adjusted_time', 'N/A'),
                'Full Article Text':article_data.get('content', 'N/A'),
                'Summary': summary
                })
            else:
                rows.append({
                'Title': title,
                'Sentiment': sentiment,
                'URL': 'Not Found',
                'Original Datetime': 'Not Found',
                'Adjusted Date': 'Not Found',
                'Adjusted Time': 'Not Found'
                })
    return rows

def load_existing_articles_news():
    if os.path.exists('articles.json'):
        with open('articles.json', 'r', encoding = 'utf-8') as f:
            return json.load(f)
    return []

def save_new_articles_news(existing_articles, new_articles):
    existing_urls = {article['URL'] for article in existing_articles}
    unique_new_articles = [article for article in new_articles if article['URL'] not in existing_urls]
    
    if unique_new_articles:
        updated_articles = existing_articles + unique_new_articles
    return updated_articles
