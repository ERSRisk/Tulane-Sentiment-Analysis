import streamlit as st
import requests
from google import genai
import pandas as pd
import re
from datetime import timedelta
from streamlit_tags import st_tags
import plotly.express as px
from newspaper import Article
from google.genai.errors import ClientError
import random
import time
import asyncio
from lxml.html.clean import Cleaner
from dateutil import parser
import datetime
from textblob import TextBlob
import tweepy
import nest_asyncio
import json
import re
import altair as alt
import matplotlib.pyplot as plt


st.set_page_config(page_title="Tulane Risk Dashboard")
st.sidebar.title("Navigation")
st.sidebar.markdown("Select a tool:")
selection = st.sidebar.selectbox("Choose a tool:", ["News Sentiment", "X Sentiment", "Article Risk Review", "Unmatched Topic Analysis"])

if "current_tab" not in st.session_state:
    st.session_state.current_tab = selection

# If switching tabs, clear session except the current tab
if st.session_state.current_tab != selection:
    keys_to_keep = {"current_tab"}
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]
    st.session_state.current_tab = selection

if selection == "News Sentiment":
    # Setting up the APIs for News API and Gemini API
    NEWS_API_KEY = st.secrets["all_my_api_keys"]["NEWS_API_KEY"]
    GEMINI_API_KEY = st.secrets["all_my_api_keys"]["GEMINI_API_KEY_X"]
    





    # Configure Gemini API
    client = genai.Client(api_key=GEMINI_API_KEY)


    # Streamlit UI
    st.title("Tulane University: Sentiment Analysis from News")




    # Make it so someone can type in their own keywords to customize the search
    search = st_tags(label="Enter your values (press Enter to separate keywords):",
                    text="Add a new value...",
                    value=["Tulane"],  # Default values
                    suggestions=["Tulane University"],  # Optional suggestions
                    key="1")

    # Date range selection
    start_date = st.date_input("Start Date", value= datetime.date.today() - timedelta(days = 7))
    end_date = st.date_input("End Date", value=datetime.date.today())
    timezone_option = st.selectbox(
        "Select Timezone for Article Timestamps:",
        options=["UTC", "CST", "CDT"],
        index=2  # Default to CDT
    )

    if search:
        st.session_state.search_ran = True

    ## Checkbox for including sports news. Selecting the checkbox will include sports news in the search.
    sports = st.checkbox("Include sports news")


    ## Checkbox for using cache. Unchecking this will allow for debugging.
    use_cache = st.checkbox("Use cache (uncheck for debugging purposes)", value=True)

    if st.session_state.get("search_ran"):
        ## This first function fetches the news articles from the News API based on the search keywords and date range.
        @st.cache_data(show_spinner=False, persist=True)
        def fetch_news(search, start_date, end_date, sports):
            if sports:
                news_url = (
                    f"https://newsapi.org/v2/everything?q={search}&"
                    f"from={start_date}&to={end_date}&sortBy=popularity&apiKey={NEWS_API_KEY}"
                )
            else:
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
        @st.cache_data(show_spinner=False, persist=True)
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
        @st.cache_data(show_spinner=False, persist=True)
        def format_articles_for_prompt(articles):
            """Format the articles in a way that can be sent to Gemini."""
            return "\n\n".join(
                [f"Title: {article['title']}\nDescription: {article['description']}\nContent: {article['content']}\nURL: {article['url']}"
                for article in articles])


        # This function analyzes the sentiment of the articles using the Gemini API.
        # It sends a prompt to the API and returns the response.
        # The prompt includes instructions for the API to analyze the sentiment based on the keywords provided.
        @st.cache_data(show_spinner=False, persist=True)
        def analyze_sentiment(text_to_analyze, search, sports, retries=5):
            for attempt in range(retries):
                try:
                    if sports:
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
                            "Only judge the sentiment for each article in terms of how it mentions the keywords. Max amount of titles should be 100.\n\n"
                            "If an article title has already been seen before, do not analyze it again.\n"
                            "If Tulane is mentioned anywhere in the article — even in passing or in an author affiliation — state that clearly in your summary.\n"
                            "Tulane was found in the text. Here is the full content for analysis.\n"
                            f"{text_to_analyze}"
                        )
                    else:
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
        async def analyze_in_batches_concurrent(articles, search, sports, timezone, batch_size=10):
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
        def run_async_batches(articles, search, sports, timezone, batch_size = 10):
            return asyncio.run(analyze_in_batches_concurrent(articles, search, sports, timezone, batch_size))


        # This function caches the responses from the Gemini API to avoid redundant API calls.
        # It uses Streamlit's caching mechanism to store the responses based on the input parameters.
        @st.cache_data(show_spinner=False, persist=True)
        def cached_gemini_response(articles, search, sports, timezone):
            return run_async_batches(articles, search, sports, timezone, batch_size=10)


        # This function processes a batch of articles and returns the response from the Gemini API.
        # It formats the articles for the prompt and sends them to the API for sentiment analysis.
        # It also handles the pagination of the articles by keeping track of the batch number and total batches.
        def process_batch(batch_df, search, i, total_batches, timezone):


            processed_articles = get_articles_with_full_content(batch_df, timezone=timezone)
            formatted_batch = format_articles_for_prompt(processed_articles)
            print(f"Processing batch {i} of {total_batches}...")
            response = analyze_sentiment(formatted_batch, search, sports)
            return response


        if "slider_value" not in st.session_state:
            st.session_state.slider_value = (-1.0, 1.0)


        # This function handles the display of the slider for filtering the sentiment scores.
        if st.button('Search') or "slider_shown" in st.session_state:
            search = '+'.join(search)
            if use_cache:
                articles = fetch_news(search, start_date, end_date, sports)
                articles = get_articles_with_full_content(articles, timezone=timezone_option)
                unique_articles = []
                seen_titles = set()
                for article in articles:
                    if article['title'] not in seen_titles:
                        unique_articles.append(article)
                        seen_titles.add(article['title'])
                articles = unique_articles
            else:
                fetch_news.clear()
                articles = fetch_news(search, start_date, end_date, sports)
                articles = get_articles_with_full_content(articles, timezone=timezone_option)
                unique_articles = []
                seen_titles = set()


                for article in articles:
                    if article['title'] not in seen_titles:
                        unique_articles.append(article)
                        seen_titles.add(article['title'])
                articles = unique_articles
            
            if not articles:
                st.write("No articles found.")
            else:
                if use_cache:
                    gemini_response_text = cached_gemini_response(articles, search, sports, timezone_option)
                else:
                    gemini_response_text = run_async_batches(articles, search, sports, timezone_option, batch_size=10)
            
                # This function processes the response from the Gemini API and formats it into a DataFrame.
                # It extracts the title, sentiment score, summary, and other relevant information from the response.
                def text_to_dataframe(text, articles):
                    rows = []
                    sections = re.split(r'Title:\s*', text)[1:]




                    for section in sections:
                        title_match = re.match(r'(.*?)\nSentiment:', section, re.DOTALL)
                        sentiment_match = re.search(r'Sentiment:\s*(-?\d+\.?\d*)', section)




                        if title_match and sentiment_match:
                            title = title_match.group(1).strip()
                            sentiment = float(sentiment_match.group(1))
                            article_data = next((article for article in articles if article['title'].strip().lower() == title.lower()), None)
                            if article_data is not None:
                                rows.append({
                                'Title': title,
                                'Sentiment': sentiment,
                                'URL': article_data.get('url'),
                                'Original Datetime': article_data.get('original_datetime', 'N/A'),
                                'Adjusted Date': article_data.get('adjusted_date', 'N/A'),
                                'Adjusted Time': article_data.get('adjusted_time', 'N/A'),
                                'Full Article Text':article_data.get('content', 'N/A')
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
                




                    return pd.DataFrame(rows)












                df = text_to_dataframe(gemini_response_text, articles)
                sentiment_counts = df['Sentiment'].value_counts()
        
                st.header("Sentiment Score Summary")
                st.write("")
                # Plot sentiment score summary
                st.bar_chart(sentiment_counts)
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Average Sentiment Score", round(df['Sentiment'].mean(), 2))
                with col2:
                    st.metric("Number of News Stories", len(df))


        # This function checks the overall sentiment of the articles and displays a message accordingly.
                if df['Sentiment'].mean() >= 0.1:
                    st.write("Overall sentiment is positive.")
                elif df['Sentiment'].mean() <= -0.1:
                    st.write("Overall sentiment is negative.")
                else:
                    st.write("Overall sentiment is neutral.")
        
                st.write("---")
                st.header("News Stories")


                st.session_state.slider_shown = True
                st.session_state.slider_value = st.slider("Sentiment Filter", -1.0, 1.0, (-1.0, 1.0), 0.1,)
        
                st.write("")
                filtered_df = df[(df['Sentiment'] >= st.session_state.slider_value[0]) &
                        (df['Sentiment'] <= st.session_state.slider_value[1])]


        # This function displays the filtered articles based on the sentiment score.
                for _, row in filtered_df.iterrows():
                    st.markdown("---") #before each row
                    st.markdown(f"###  **[{row['Title']}]({row['URL']})**")
                    st.markdown(f"**Date & Time:** {row.get('Adjusted Date', 'N/A')} at {row.get('Adjusted Time', 'N/A')}")
                    st.markdown(f"🔹 **Sentiment Score:** `{row['Sentiment']}`")
            
                    # Grab summary from the original text using the title
                    pattern = rf"Title:\s*{re.escape(row['Title'])}\s*.*?Sentiment:\s*-?\d+\.?\d*\s*Summary:\s*(.*?)(?:\n|$)"
                    match = re.search(pattern, gemini_response_text, re.DOTALL)
                    if match:
                        summary = match.group(1).strip()
                        st.markdown(f"**Summary:** {summary}")
                    else:
                        st.markdown("⚠️ Summary not found.")
                    st.write("---")
        
                #format the date range into a MM-DD-YYYY format
                start_str = start_date.strftime("%m-%d-%Y")
                end_str = end_date.strftime("%m-%d-%Y")




                #create a dynamic name
                df_name = f"TU_News_{start_str}_to_{end_str}"




                #display the Dataframe with the dynamic name
                st.header(f"Dataframe of Results")
                st.dataframe(df)




                #download button for the dataframe
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Dataframe",
                    data=csv,
                    file_name=f"{df_name}.csv",
                    mime="text/csv",
                )




        st.markdown("---")
    #st.markdown("🔍 Built by Luke Roosa, Samuel Centeno, & Illia Kunin | Powered by NewsAPI & Gemini")
if selection == "X Sentiment":
    nest_asyncio.apply()
    #This function is used to analyze the tweets using Google Gemini API
    # It takes a tweet as input and returns a JSON response with a description and whether the tweet is related to sports or not.

    #Define the API keys for X and Gemini
    GEMINI_API_KEY_X = st.secrets["all_my_api_keys"]["GEMINI_API_KEY_X"]
    X_API_KEY = st.secrets["all_my_api_keys"]["X_API_KEY"]

   
    #adding this option to run batches
    semaphore = asyncio.Semaphore(3)

    async def limited_process(batch_df, search, batch_size, total_batches):
            async with semaphore:
                return await asyncio.to_thread(process_batch, batch_df, search, batch_size, total_batches)

    async def analyze_in_batches_concurrent_X(tweets, search, sports, batch_size=10):
            all_responses = []
            total_batches = len(tweets) // batch_size + (1 if len(tweets) % batch_size != 0 else 0)
            print(f"Total batches: {total_batches}")
            batch_tasks = []
            for i in range(0, len(tweets), batch_size):
                batch_df = tweets[i:i + batch_size]




                task = limited_process(batch_df, search, i // batch_size + 1, total_batches)
                batch_tasks.append(task)
    
            all_responses = await asyncio.gather(*batch_tasks)
            return pd.concat(all_responses)
    
    async def run_async_batches_X(tweets, search, sports, batch_size = 10):
        return asyncio.run(analyze_in_batches_concurrent_X(tweets, search, sports, batch_size = 10))
    
    def process_batch(batch_df, search, i, total_batches):
        print(f"Processing batch {i} of {total_batches}...")
        formatted_tweets = "\n\n".join([f"{tweet.text}" for tweet in batch_df])
        analysis_list = analyze_sentiment_X(formatted_tweets, search, sports)
        response = []
        if not analysis_list:
            print("analysis list is missing")
        elif len(analysis_list) != len(batch_df):
            print(f"Got {len(analysis_list)} responses for {len(batch_df)} tweets.")
        for tweet, analysis in zip(batch_df, analysis_list):
            result = {
            "created_at": tweet.created_at,
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
        return pd.DataFrame(response)



    #This function fetches tweets from Twitter API based on the search term and date range provided by the user.
    # It uses the Tweepy library to interact with the Twitter API and returns a DataFrame with the tweet data.
    # The function takes the following parameters:
    # search: The search term to look for in tweets.
    # start_date: The start date for the tweet search.
    # end_date: The end date for the tweet search.
    # no_of_tweets: The number of tweets to fetch.
    def fetch_twits(search, start_date, end_date, no_of_tweets):
        import datetime
        client = tweepy.Client(bearer_token=X_API_KEY)
        response = client.search_recent_tweets(
            query=search,
            max_results=100,
            tweet_fields=["created_at"],
            start_time=start_date.isoformat() + "Z",
            end_time=(datetime.datetime.combine(end_date, datetime.time.min)).isoformat() + "Z"
        )
        tweets = response.data
        if not tweets:
            st.warning('No tweets were extracted. Check X source')
            st.session_state.x_search_ran = False
            st.stop()
        return tweets

    def analyze_sentiment_X(formatted_tweet_block, search, sports, retries=5):
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
        models = ["gemini-1.5-pro", "gemini-2.5-pro-preview-05-06"]
        for attempt in range(retries): 
            for model in models:
                try:
                    
                    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
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


    #Setting the page title and layout for the UI
    st.title("Tulane University: Sentiment Analysis from X")
    search = st_tags(
        label="Enter your values (press Enter to separate keywords):",
        text="Add a new value...",
        value=["Tulane"],  # Default values
        suggestions=["Tulane University"],  # Optional suggestions
        key="1"
    )
    #Adding a date input for the user to select the start and end dates for the tweet search
    start_date = st.date_input("Start Date", value= datetime.date.today() - datetime.timedelta(days = 6))
    start_date= datetime.datetime.combine(start_date, datetime.time(0, 0)) + datetime.timedelta(hours=1)
    end_date = st.date_input("End Date", value=datetime.date.today())
    search_button = st.button("Search")

    sports= st.checkbox("Include sports news")
    pass
    pass


    # Store search trigger persistently
    # Track button click
    if search_button:
        st.session_state.x_search_ran = True
        st.session_state.x_results_ready = False  # Reset
    # Fetch only when search button is pressed
    if st.session_state.get("x_search_ran", False):
        tweets = fetch_twits(search, start_date, end_date, 100)
        with open("tweets.json", "w") as f:
            json.dump([tweet.data for tweet in tweets], f)
        print('Fetched tweets:', tweets[:2] if tweets else "None")
        if not tweets:
            st.warning('No posts were extracted. Check X source.')
            st.session_state.x_search_ran = False
            st.stop()    
        df = asyncio.get_event_loop().run_until_complete(run_async_batches_X(tweets, search, sports, batch_size=10))
        st.session_state.x_df = df
        st.session_state.x_search_ran = False
        st.session_state.x_results_ready = True

    # === Display results if ready ===
    if st.session_state.get("x_results_ready", False):
        df = st.session_state.get("x_df", None)

        if df is None or df.empty:
            st.warning("No tweets found for the given search term and date range.")
            st.stop()

        slider_value = st.slider(
            "Sentiment Filter",
            min_value=-1.0,
            max_value=1.0,
            value=(-1.0, 1.0),
            step=0.1,
            key="slider_value"
        )

        # Filter tweets based on sentiment
        df_filtered = df[
            (df['sentiment'] >= slider_value[0]) &
            (df['sentiment'] <= slider_value[1])
        ]

        if df_filtered.empty:
            st.warning("No tweets found in this sentiment range.")
            st.stop()

        # Clean usernames for display
        df_filtered['text'] = df_filtered['text'].apply(lambda x: re.sub(r"@\w+", "@user", x))

        # Chart
        st.subheader("Sentiment Distribution")
        sentiment_counts = df_filtered['sentiment'].value_counts().sort_index()
        st.bar_chart(sentiment_counts)

        # Display tweets
        for _, row in df_filtered.iterrows():
            st.markdown(f"**Created At:** {row['created_at'].strftime('%Y-%m-%d %H:%M')}")
            st.markdown(f"**Link:** [Tweet Link]({row['link']})")
            st.markdown(f"**Text:** {row['text']}")
            st.markdown(f"**Description:** {row['description']}")
            st.markdown(f"**Sentiment:** {row['sentiment']}")
            st.markdown(f"**Summary:** {row['summary']}")
            st.write("---")

        # Full cleaned table
        st.write(df_filtered.drop(columns=["is_sport"]))
    if selection == "Unmatched Topic Analysis":
        with open('Online_Extraction/unmatched_topics.json', 'r') as f:
            unmatched = json.load(f)
        
        with open('Online_Extraction/topics_BERT.json', 'r') as f:
            saved_topics = json.load(f)
        
        try:
            with open('Online_Extraction/discarded_topics', 'r') as f:
                discarded_topics = json.load(f)
                if not isinstance(discarded_topics, list):
                    discarded_topics = [discarded_topics]
        except FileNotFoundError:
            discarded_topics = []
        
        st.title('Unmatched Topics Analysis')
        
        for topic in unmatched:
            skip_key = f"skip_{topic['topic']}"
            if st.session_state.get(skip_key):
                continue
        
            st.subheader(f"Topic {topic['topic']}: {topic['name']}")
            st.markdown(f"**Keywords:** {(topic['keywords'])}")
            with st.expander("**Sample Articles:**"):
                docs = topic['documents']
                random.shuffle(docs)
                for doc in docs:
                    words = doc.split()
                    st.markdown("**Sample Titles:**")
                    st.markdown(f"{' '.join(words[:40]) + '...' if len(words)>40 else ''}")
            radio_key = str(topic['topic'])
            reset_flag = f"reset_{radio_key}"
            
        
            if st.session_state.get(reset_flag):
                st.session_state[radio_key] = ''
                st.session_state[reset_flag] = False
            decision = st.radio("What would you like to do with this topic?",['','Keep as new topic', 'Merge with existing topic', 'Discard'],
                key=radio_key, index = 0)
            if decision == 'Keep as new topic':
                st.session_state['confirm_new'] = True
                if st.session_state.get('confirm_new'):
                    st.warning("Are you sure you want to create a new topic?")
                    col1, col2= st.columns(2)
                    with col1:
                        if st.button("Yes, create new topic", key=f"create_new_{radio_key}"):
                            st.session_state['confirm_new'] = False
                            new_topic = {
                                'topic': topic['topic'],
                                'name': topic['name'],
                                'keywords': topic['keywords'],
                                'documents': topic['documents']
                            }
                            saved_topics.append(new_topic)
                            with open('topics_BERT.json', 'w') as f:
                                json.dump(saved_topics, f)
                            st.success(f"New topic {topic['topic']} created successfully!")
                    with col2:
                        if st.button("Cancel", key=f"cancel_new_{radio_key}"):
                            st.session_state['confirm_new'] = False
                            st.session_state[reset_flag] = True
                            st.rerun()
            if decision == 'Merge with existing topic':
                st.session_state['confirm_merge'] = True
                if st.session_state.get('confirm_merge'):
                    st.warning("Are you sure you want to merge this topic with an existing one?")
                    col1, col2= st.columns(2)
                    with col1:
                        if st.button("Yes, merge topic", key=f"merge_{radio_key}"):
                            st.session_state['confirm_merge'] = False
                            existing_topic = st.selectbox("Select existing topic to merge with:", ['--Select a topic--'],[t['name'] for t in saved_topics], key=f"existing_topic_{radio_key}")
                            for t in saved_topics:
                                if t['name'] == existing_topic:
                                    if isinstance(t['documents'], str):
                                        t['documents'] = [t['documents']]
                                    t['documents'].extend(topic['documents'])
        
                                # Ensure keywords are lists
                                    if isinstance(t['keywords'], str):
                                        t['keywords'] = [k.strip() for k in t['keywords'].split(',')]
                                        new_keywords = [k.strip() for k in topic['keywords'].split(',')] if isinstance(topic['keywords'], str) else topic['keywords']
                                    t['keywords'].extend(new_keywords)
                                    with open('topics_BERT.json', 'w') as f:
                                        json.dump(saved_topics, f)
                                    st.success(f"Topic {topic['topic']} merged successfully!")
                    with col2:
                        if st.button("Cancel", key=f"cancel_merge_{radio_key}"):
                            st.session_state['confirm_merge'] = False
                            st.session_state[reset_flag] = True
                            st.rerun()
            if decision == 'Discard':
                st.session_state[reset_flag] = True
                st.session_state[skip_key] = True
        
                st.warning(f"Topic {topic['topic']} discarded.")
        
                discarded_topic = {
                    'topic': topic['topic'],
                    'name': topic['name'],
                    'keywords': topic['keywords'],
                    'documents': topic['documents']
                }
                discarded_topics.append(discarded_topic)
                with open('Online_Extraction/discarded_topics', 'w') as f:
                    json.dump(discarded_topic, f)
        
                unmatched_json = [t for t in unmatched if t['topic'] != topic['topic']]
                with open('Online_Extraction/unmatched_topics.json', 'w') as f:
                    json.dump(unmatched_json, f)
                
                st.success(f"Topic {topic['topic']} discarded successfully!")

if selection == "Article Risk Review":
    import streamlit as st
    import pandas as pd
    import json
    from datetime import datetime
    from datetime import timedelta
    import os
    import ast

    st.title("Article Risk Review Portal")
    #give me a filter to filter articles by date range
    st.sidebar.header("Filter Articles")
    start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=30))
    end_date = st.sidebar.date_input("End Date", datetime.now())
    
    
    if start_date > end_date:
        st.sidebar.error("Start date must be before end date.")
    # Load articles and risks
    
    if 'articles' not in st.session_state:
        if os.path.exists('BERTopic_results.csv'):
            st.session_state.articles = pd.read_csv('BERTopic_results.csv')
        
    base_df = st.session_state.articles

    #articles = articles[articles['Published']> start_date.strftime('%Y-%m-%d')]
    #articles = articles[articles['Published']< end_date.strftime('%Y-%m-%d')]
    filtered_df = base_df[base_df['University Label'] == 1]
    filtered_df = filtered_df.drop_duplicates(subset=['Title', 'Link'])
    with open('Model_training/risks.json', 'r') as f:
        data = json.load(f)

    risks = data['risks']
    all_possible_risks = [risk['name'] for risk in risks]

    all_possible_risks = [r for r in all_possible_risks if isinstance(r, str)]
    filter_risks = [r for r in all_possible_risks if r != "no risk"]

    filtered_risks = st.multiselect("Select Risks to Filter Articles", options = all_possible_risks, default=filter_risks, key="risk_filter")

    def match_any(predicted, selected):
        if not isinstance(predicted, list) or not predicted:
            return False
        predicted = [str(p).strip().lower() for p in predicted if isinstance(p, str)]
        selected = [s.strip().lower() for s in selected]
        return any(p in selected for p in predicted)
    
    
    for _, article in filtered_df.iterrows():
        idx = article.name
        if pd.isna(article.get('Title')) or pd.isna(article.get('Content')):
            continue
    
        raw = article.get("Predicted_Risks", "[]")
        if isinstance(raw, list):
            predicted = raw
        elif isinstance(raw, str):
            raw = raw.strip()
            if raw.startswith("[") and raw.endswith("]"):
                try:
                    predicted = ast.literal_eval(raw)
                except:
                    predicted = []
            elif raw.lower() in ("no risk", "none", ""):
                predicted = []
            else:
                predicted = [raw]  # single risk string gets wrapped in a list
        else:
            predicted = []

    
        if not match_any(predicted, filtered_risks):
            continue
    
        title = str(article.get("Title", ""))[:100]
        if title:
            with st.expander(f"{title}..."):
                st.markdown(f"[Read full article]({article['Link']})")
                st.write(article['Content'][:1000])
                st.metric('Risk Score', article['Risk_Score'])
    
                st.markdown("**Predicted Risks:**")
                valid_defaults = [opt for opt in all_possible_risks if any(opt.lower() == str(p).lower() for p in predicted if isinstance(p, str))]
                selected_risks = st.multiselect(
                    "Edit risks if necessary:",
                    options=all_possible_risks,
                    default=valid_defaults,
                    key=f"edit_{idx}"
                )
                with st.expander('View Risk Labels'):
                    col1, col2, col3, col4, col5, col6, col7 =  st.columns(7)
                    with col1:
                        st.metric('Recency', article['Recency'])
                    with col2:
                        st.metric('Acceleration', article['Acceleration_value'])
                    with col3:
                        st.metric('Source Accuracy', article['Source_Accuracy'])
                    with col4:
                        st.metric('Impact Score', article['Impact_Score'])
                    with col5:
                        st.metric('Location', article['Location'])
                    with col6:
                        st.metric('Industry Risk', article['Industry_Risk'])
                    with col7:
                        st.metric('Frequency', article['Frequency_Score'])
                    
                    with st.expander("Manually update risk labels:"):
                        options = [0.0, 1.0,2.0,3.0,4.0,5.0]
                        with st.form(f"manual_edit_form_{idx}"):
                            col1, col2, col3, col4, col5, col6, col7 =  st.columns(7)
                            with col1:
                                upd_recency_value = st.text_area('Recency', value= article['Recency'], key =f"recency_input_{idx}")
                            with col2:
                                upd_acceleration_value = st.number_input('Acceleration',  min_value=0.0, max_value = 5.0, step = 1.0, value=article['Acceleration_value'],key =f"acceleration_input_{idx}")
                            with col3:
                                upd_source_accuracy =st.number_input('Source Accuracy',  min_value=0.0, max_value = 5.0, step = 1.0, value=article['Source_Accuracy'],key =f"source_input_{idx}")
                            with col4:
                                upd_impact_score = st.number_input('Impact Score',  min_value=0.0, max_value = 5.0, step = 1.0, value=article['Impact_Score'],key =f"impact_input_{idx}")
                            with col5:
                                upd_location=st.number_input('Location',  min_value=0.0, max_value = 5.0, step = 1.0, value=article['Location'],key =f"location_input_{idx}")
                            with col6:
                                upd_industry_risk = st.number_input('Industry Risk',  min_value=0.0, max_value = 5.0, step = 1.0, value=article['Industry_Risk'],key =f"industry_input_{idx}")
                            with col7:
                                upd_frequency_score = st.number_input('Frequency Score', min_value=0.0, max_value = 5.0, step = 1.0, value=article['Frequency_Score'],key =f"frequency_input_{idx}")

                            st.markdown('Please provide a reason for the changes made to the risk labels:')
                            reason = st.text_area("Reason for changes", placeholder="Explain the changes made to the risk labels.", key=f"reason_{idx}")
                            submitted =  st.form_submit_button("Update Risk Labels")
                        if submitted:
                            st.session_state.articles.at[idx, 'Recency_Upd'] = upd_recency_value
                            st.session_state.articles.at[idx, 'Acceleration_value_Upd'] = upd_acceleration_value
                            st.session_state.articles.at[idx, 'Source_Accuracy_Upd'] = upd_source_accuracy
                            st.session_state.articles.at[idx, 'Impact_Score_Upd'] = upd_impact_score
                            st.session_state.articles.at[idx, 'Location_Upd'] = upd_location
                            st.session_state.articles.at[idx, 'Industry_Risk_Upd'] = upd_industry_risk
                            st.session_state.articles.at[idx, 'Frequency_Score_Upd'] = upd_frequency_score
                            st.session_state.articles.at[idx, 'Change reason'] = reason
                            st.session_state.articles.to_csv('BERTopic_results.csv', index=False)
                            st.success("Risk labels updated successfully.")

                if st.button("Save Correction", key=f"save_{idx}"):
                    st.session_state.articles.at[idx, 'Predicted_Risks'] = selected_risks
                    st.session_state.articles.to_csv('BERTopic_results.csv', index=False)
                    st.success("Correction saved.")
