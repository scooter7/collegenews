import streamlit as st
import requests
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from streamlit_echarts import st_echarts
import pandas as pd
import boto3
from datetime import datetime
from collections import Counter
from io import StringIO

# Download necessary NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("vader_lexicon", quiet=True)
nltk.download("stopwords", quiet=True)

# Predefined keywords for analysis
KEYWORDS = ["technology", "health", "finance", "sports", "entertainment"]

def get_custom_stopwords(url):
    response = requests.get(url)
    stopwords = set(response.text.split())
    return stopwords

def plot_wordcloud(words):
    custom_stopwords_url = "https://github.com/aneesha/RAKE/raw/master/SmartStoplist.txt"
    custom_stopwords = get_custom_stopwords(custom_stopwords_url)
    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          stopwords=custom_stopwords,
                          min_font_size=10).generate(words)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(wordcloud)
    ax.axis("off")
    st.pyplot(fig)

def render_sentiment_gauge(score):
    color = '#6DD400' if score > 0 else '#FFD93D' if score == 0 else '#FF4500'
    option = {
        "series": [
            {
                "type": 'gauge',
                "startAngle": 180,
                "endAngle": 0,
                "min": -100,
                "max": 100,
                "splitNumber": 4,
                "pointer": {"show": True, "length": '90%', "width": 8},
                "axisLine": {
                    "lineStyle": {
                        "width": 15,
                        "color": [
                            [0.33, '#FF4500'], 
                            [0.5, '#FFD93D'],    
                            [0.67, '#FFD93D'],   
                            [1, '#6DD400']       
                        ]
                    }
                },
                "axisLabel": {"show": False},
                "axisTick": {"show": False},
                "splitLine": {"show": False},
                "detail": {
                    "formatter": '{value}%',
                    "offsetCenter": [0, '80%'],
                    "fontSize": 16,
                    "color": color,
                    "fontWeight": 'bold'
                },
                "data": [{'value': score, "name": "Sentiment Score"}],
            }
        ]
    }
    st_echarts(options=option, height="400px")

def analyze_sentiment(text):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    sentiment = sid.polarity_scores(text)
    sentiment_score = sentiment['compound'] * 100
    return sentiment_score

def fetch_news(query):
    ENDPOINT = 'https://newsapi.org/v2/everything'
    params = {
        'q': query,
        'apiKey': st.secrets["newsapi"]["api_key"],
        'pageSize': 10,
    }
    response = requests.get(ENDPOINT, params=params)
    return response.json()

def upload_csv_to_s3(df):
    try:
        if "aws" in st.secrets:
            s3 = boto3.client(
                's3',
                aws_access_key_id=st.secrets["aws"]["aws_access_key_id"],
                aws_secret_access_key=st.secrets["aws"]["aws_secret_access_key"]
            )
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            response = s3.put_object(
                Bucket=st.secrets["aws"]["bucket_name"], 
                Key=st.secrets["aws"]["object_key"], 
                Body=csv_buffer.getvalue()
            )
            st.write(f"Data saved to S3 bucket. Response: {response}")
        else:
            st.error("AWS credentials not found in Streamlit secrets.")
    except Exception as e:
        st.error(f"Failed to upload data to S3: {e}")

def main():
    st.title("News Feed Analyzer")

    keyword = st.selectbox("Select a keyword to analyze:", KEYWORDS)

    try:
        if "aws" in st.secrets:
            s3 = boto3.client(
                's3',
                aws_access_key_id=st.secrets["aws"]["aws_access_key_id"],
                aws_secret_access_key=st.secrets["aws"]["aws_secret_access_key"]
            )
            obj = s3.get_object(Bucket=st.secrets["aws"]["bucket_name"], Key=st.secrets["aws"]["object_key"])
            historical_data = pd.read_csv(obj['Body'])
            st.write("Loaded historical data successfully from S3.")
        else:
            historical_data = pd.DataFrame(columns=["Date", "Keyword", "Topics", "Sentiment"])
            st.write("Initialized empty dataset.")
    except Exception as e:
        st.error(f"Error loading historical data: {e}")
        historical_data = pd.DataFrame(columns=["Date", "Keyword", "Topics", "Sentiment"])

    if st.button("Search"):
        results = fetch_news(keyword)
        news_text = ""
        if results.get("articles"):
            for article in results["articles"]:
                title = article['title']
                description = article['description'] or "No description available"
                url = article['url']
                news_text += f"{description} "
                st.markdown(f"#### [{title}]({url})")
                st.markdown(f"*{description}*")
                st.markdown("---")

            if news_text:
                st.write("Aggregate Word Cloud:")
                plot_wordcloud(news_text)
                sentiment_score = analyze_sentiment(news_text)
                st.write("Aggregate Sentiment:")
                render_sentiment_gauge(sentiment_score)

                words = nltk.word_tokenize(news_text)
                words = [word for word in words if word.isalpha()]
                most_common_words = Counter(words).most_common(5)
                top_words = ', '.join(word for word, count in most_common_words)

                update_df = pd.DataFrame({
                    "Date": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                    "Keyword": [keyword],
                    "Topics": [top_words],
                    "Sentiment": [sentiment_score]
                })

                st.table(update_df)

                historical_data = pd.concat([historical_data, update_df])
        else:
            st.write("No results found.")

    if not historical_data.empty:
        st.write("Sentiment Trend Analysis:")
        for key in KEYWORDS:
            key_data = historical_data[historical_data['Keyword'] == key]
            if not key_data.empty:
                st.line_chart(key_data.set_index('Date')['Sentiment'])

    if st.button("Update"):
        st.write("Attempting to save data to S3...")
        upload_csv_to_s3(historical_data)

if __name__ == "__main__":
    main()
