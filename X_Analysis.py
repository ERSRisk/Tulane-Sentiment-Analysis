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
import concurrent.futures
import json
import re
import altair as alt
import matplotlib.pyplot as plt


st.set_page_config(page_title="Tulane Risk Dashboard")
st.sidebar.title("Navigation")
st.sidebar.markdown("Select a tool:")
selection = st.sidebar.selectbox("Choose a tool:", ["News Sentiment", "X Sentiment"])

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
    GEMINI_API_KEY = st.secrets["all_my_api_keys"]["GEMINI_API_KEY_NEWS"]
    





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
                    gemini_response = client.models.generate_content(model="gemini-2.5-pro-exp-03-25", contents=[sentiment_prompt])
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
    #This function is used to analyze the tweets using Google Gemini API
    # It takes a tweet as input and returns a JSON response with a description and whether the tweet is related to sports or not.

    #Define the API keys for X and Gemini
    GEMINI_API_KEY_X = st.secrets["all_my_api_keys"]["GEMINI_API_KEY_X"]
    X_API_KEY = st.secrets["all_my_api_keys"]["X_API_KEY"]

    def tweet_analysis(tweet):
        client = genai.Client(api_key=GEMINI_API_KEY_X)
        prompt = f"""
    Analyze the following tweet and provide a description in 1–2 sentences.
    Also, analyze whether the tweet is related to sports.
    Respond in JSON format with two keys:
    - "description": a short description of the tweet,
    - "is_sport": 1 if the tweet is related to sports, otherwise 0.
    Tweet:
    {tweet}
    """
        response = client.models.generate_content(model="gemini-2.5-pro-exp-03-25", contents=prompt)
        #function returns the response in JSON format(description and is_sport)
        if response.candidates and response.candidates[0].content.parts:
            raw_text = response.candidates[0].content.parts[0].text
            cleaned_text = re.sub(r"```json|```|\n|\s{2,}", "", raw_text).strip()
            data = json.loads(cleaned_text)
            return data
   
        else:
            return "No analysis available."
   
    @st.cache_data(show_spinner=False)
    @st.cache_data(show_spinner=False)
    #This function fetches tweets from Twitter API based on the search term and date range provided by the user.
    # It uses the Tweepy library to interact with the Twitter API and returns a DataFrame with the tweet data.
    # The function takes the following parameters:
    # search: The search term to look for in tweets.
    # start_date: The start date for the tweet search.
    # end_date: The end date for the tweet search.
    # no_of_tweets: The number of tweets to fetch.
    def fetch_twits(search, start_date, end_date, no_of_tweets):
        import time
        client = tweepy.Client(bearer_token=X_API_KEY)
        response = client.search_recent_tweets(
            query=search,
            max_results=10,
            tweet_fields=["created_at"],
            start_time=start_date.isoformat() + "Z",
            end_time=(datetime.datetime.combine(end_date, datetime.time.min)).isoformat() + "Z"
        )
        tweets = response.data
        if not tweets:
            return None


        results = []
        for i, tweet in enumerate(tweets):
            print(f"🔍 Analyzing tweet {i+1}/{len(tweets)}...")
            try:
                analysis = tweet_analysis(tweet.text)
            except Exception as e:
                print(f"⚠️ Failed to analyze tweet: {e}")
                analysis = {"description": "parse error", "is_sport": 0}
            result = {
                "created_at": tweet.created_at,
                "text": tweet.text,
                "sentiment": round(TextBlob(tweet.text).sentiment.polarity, 1),
                "link": f"https://twitter.com/{tweet.author_id}/status/{tweet.id}",
                "description": analysis.get("description", "parse error"),
                "is_sport": analysis.get("is_sport", 0)
            }
            results.append(result)
            time.sleep(4)  # 15 requests per minute = 1 request every 4 seconds
        return pd.DataFrame(results)




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
    if search_button:
        st.session_state.x_search_ran = True
        st.session_state.x_slider_shown = False  # reset slider visibility

    # Main logic runs ONLY when this is True
    if st.session_state.get("x_search_ran", False):

        #This part of the code is used to display the results of the tweet analysis.
        #"action" afer the button is pressed
        if "slider_value" not in st.session_state:
            st.session_state.slider_value = (-1.0, 1.0)
        if search_button or "slider_shown" in st.session_state:
            df=fetch_twits(search, start_date, end_date,10)
            if df is None:
                st.write("No tweets found for the given search term and date range.")
                st.stop()
    
            #Giving all the twits(including the ones that are realted to sports)
            if sports:
                # Adding a title above the bar chart
                st.subheader("Sentiment Distribution")




                # Creating bar chart for the sentiment
                sentiment_counts = df['sentiment'].value_counts().sort_index()
                fig, ax = plt.subplots()
                sentiment_counts.plot(kind='bar', ax=ax, color='skyblue')
                for p in ax.patches:
                    ax.annotate(f'{p.get_height()}',
                                (p.get_x() + p.get_width() / 2., p.get_height()),
                                xytext=(0, 5),  # отступы от столбца
                                textcoords='offset points',
                                ha='center', va='bottom', fontsize=10)


                ax.set_ylabel("Count")
                ax.set_xlabel("Sentiment")
                st.pyplot(fig)
    
                #making the slider to filter outputs by sentiment
                st.session_state.slider_shown = True
                st.session_state.slider_value = st.slider("Sentiment Filter", -1.0, 1.0, (-1.0, 1.0), 0.1,)
                st.write("")




                #filtering df by user's sentiment range
                df_filtered = df[(df['sentiment'] >= st.session_state.slider_value[0]) & (df['sentiment'] <= st.session_state.slider_value[1])]
                df_filtered['text'] = df_filtered['text'].apply(lambda x: re.sub(r"@\w+", "@user", x))
                # Display the filtered DataFrame
                for index, row in df_filtered.iterrows():
                    st.write(f"**Created At:** {row['created_at'].strftime('%Y-%m-%d %H:%M')}")
                    st.write(f"**Link:** [Tweet Link]({row['link']})")
                    st.write(f"**Text:** {row['text']}")
                    st.write(f"**Description:** {row['description']}")
                    st.write(f"**Sentiment:** {row['sentiment']}")
                    st.write("---")  # Separator between tweets
            
                df['text'] = df['text'].apply(lambda x: re.sub(r"@\w+", "@user", x))
                st.write(df.drop(columns=["is_sport"]))




            #presenting all the twits that are not related to sports(if user has selected it in the checkbox)
            else:
                # Filter out tweets related to sports
                df=df[df['is_sport'] == 0]


                # Adding a title above the bar chart
                st.subheader("Sentiment Distribution")
                # Creating bar chart for the sentiment
                sentiment_counts = df['sentiment'].value_counts().sort_index()
                fig, ax = plt.subplots()
                sentiment_counts.plot(kind='bar', ax=ax, color='skyblue')


                # Adding value labels on top of the bars
                for p in ax.patches:
                    ax.annotate(f'{p.get_height()}',
                                (p.get_x() + p.get_width() / 2., p.get_height()),
                                xytext=(0, 5),  # отступы от столбца
                                textcoords='offset points',
                                ha='center', va='bottom', fontsize=10)


                # Setting the title and labels for the bar chart
                ax.set_title("Sentiment Distribution")
                ax.set_ylabel("Count")
                ax.set_xlabel("Sentiment")
                # Displaying the bar chart
                st.pyplot(fig)
            
                #making the slider ti filter outputs by sentiment
                st.session_state.slider_shown = True
                st.session_state.slider_value = st.slider("Sentiment Filter", -1.0, 1.0, (-1.0, 1.0), 0.1,)
                st.write("")


                #filtering df by user's sentiment range
                df_filtered = df[(df['sentiment'] >= st.session_state.slider_value[0]) & (df['sentiment'] <= st.session_state.slider_value[1])]
                df_filtered['text'] = df_filtered['text'].apply(lambda x: re.sub(r"@\w+", "@user", x))
                # Display the filtered DataFrame
                for index, row in df_filtered.iterrows():
                    st.write(f"**Created At:** {row['created_at'].strftime('%Y-%m-%d %H:%M')}")
                    st.write(f"**Link:** [Tweet Link]({row['link']})")
                    st.write(f"**Text:** {row['text']}")
                    st.write(f"**Description:** {row['description']}")
                    st.write(f"**Sentiment:** {row['sentiment']}")
                    st.write("---")  # Separator between tweets
                df['text'] = df['text'].apply(lambda x: re.sub(r"@\w+", "@user", x))
                st.write(df.drop(columns=["is_sport"]))