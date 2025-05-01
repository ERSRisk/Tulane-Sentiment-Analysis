#X Analysis


import streamlit as st
import pandas as pd
from google import genai
import pandas as pd
import datetime
from textblob import TextBlob
import tweepy
import concurrent.futures
from streamlit_tags import st_tags
import json
import re
import altair as alt
import matplotlib.pyplot as plt


# Load API keys from Streamlit secrets
Gemini_API_Key_X = st.secrets["all_my_api_keys"]["GEMINI_API_KEY_X"]
X_API_Key = st.secrets["all_my_api_keys"]["X_API_KEY"]


@st.cache_data(show_spinner=False)
def tweet_analysis(tweet):
    client = genai.Client(api_key=Gemini_API_Key_X)
    prompt = f"""
Analyze the following tweet and provide a description in 1–2 sentences.
Also, analyze whether the tweet is related to sports.
Respond in JSON format with two keys:
- "description": a short description of the tweet,
- "is_sport": 1 if the tweet is related to sports, otherwise 0.
Tweet:
{tweet}
"""
    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
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
def fetch_twits(search, start_date, end_date, no_of_tweets):
    import time


    client = tweepy.Client(bearer_token=X_API_Key)
    response = client.search_recent_tweets(
        query=search,
        max_results=100,
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





st.title("Tulane University: Sentiment Analysis from X")
search = st_tags(
    label="Enter your values (press Enter to separate keywords):",
    text="Add a new value...",
    value=["Tulane"],  # Default values
    suggestions=["Tulane University"],  # Optional suggestions
    key="1"
)
start_date = st.date_input("Start Date", value= datetime.date.today() - datetime.timedelta(days = 6))
start_date= datetime.datetime.combine(start_date, datetime.time(0, 0)) + datetime.timedelta(hours=1)
end_date = st.date_input("End Date", value=datetime.date.today())
search_button = st.button("Search")
sports= st.checkbox("Include sports news")
pass
pass












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
                        xytext=(0, 5),  #Offsets from the bar
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
        # Display the filtered DataFrame
        for index, row in df_filtered.iterrows():
            st.write(f"**Created At:** {row['created_at'].strftime('%Y-%m-%d %H:%M')}")
            st.write(f"**Link:** [Tweet Link]({row['link']})")
            st.write(f"**Text:** {row['text']}")
            st.write(f"**Description:** {row['description']}")
            st.write(f"**Sentiment:** {row['sentiment']}")
            st.write("---")  # Separator between tweets
        st.write(df.drop(columns=["is_sport"]))




    #presenting all the twits that are not related to sports(if user has selected it in the checkbox)
    else:
        # Filter out tweets related to sports
        df=df[df['is_sport'] == 0]




        st.subheader("Sentiment Distribution")
        # Creating bar chart for the sentiment
        sentiment_counts = df['sentiment'].value_counts().sort_index()
        fig, ax = plt.subplots()
        sentiment_counts.plot(kind='bar', ax=ax, color='skyblue')




        #Add annotations with the number of occurrences
        for p in ax.patches:
            ax.annotate(f'{p.get_height()}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        xytext=(0, 5),  #spacing from the bar
                        textcoords='offset points',
                        ha='center', va='bottom', fontsize=10)




        #Let's configure the display
        ax.set_title("Sentiment Distribution")
        ax.set_ylabel("Count")
        ax.set_xlabel("Sentiment")




        #Let's show the chart using Streamlit
        st.pyplot(fig)
    




    
        #making the slider to filter outputs by sentiment
        st.session_state.slider_shown = True
        st.session_state.slider_value = st.slider("Sentiment Filter", -1.0, 1.0, (-1.0, 1.0), 0.1,)
        st.write("")




        #filtering df by user's sentiment range
        df_filtered = df[(df['sentiment'] >= st.session_state.slider_value[0]) & (df['sentiment'] <= st.session_state.slider_value[1])]
        # Display the filtered DataFrame
        for index, row in df_filtered.iterrows():
            st.write(f"**Created At:** {row['created_at'].strftime('%Y-%m-%d %H:%M')}")
            st.write(f"**Link:** [Tweet Link]({row['link']})")
            st.write(f"**Text:** {row['text']}")
            st.write(f"**Description:** {row['description']}")
            st.write(f"**Sentiment:** {row['sentiment']}")
            st.write("---")  # Separator between tweets
        st.write(df.drop(columns=["is_sport"]))