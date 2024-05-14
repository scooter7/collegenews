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

# Ensure necessary NLTK downloads
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

def upload_to_s3(bucket, filepath, data):
    try:
        s3 = boto3.client(
            's3',
            aws_access_key_id=st.secrets["aws"]["aws_access_key_id"],
            aws_secret_access_key=st.secrets["aws"]["aws_secret_access_key"]
        )
        s3.put_object(Bucket=bucket, Key=filepath, Body=data)
        return True
    except Exception as e:
        st.error(f"Failed to upload data to S3: {e}")
        return False

def download_from_s3(bucket, filepath):
    try:
        s3 = boto3.client(
            's3',
            aws_access_key_id=st.secrets["aws"]["aws_access_key_id"],
            aws_secret_access_key=st.secrets["aws"]["aws_secret_access_key"]
        )
        response = s3.get_object(Bucket=bucket, Key=filepath)
        return pd.read_csv(response['Body'])
    except Exception as e:
        st.error(f"Failed to download data from S3: {e}")
        return pd.DataFrame(columns=["Date", "Keyword", "Topics", "Sentiment"])

def main():
    st.title("News Feed Analyzer")

    keyword = st.selectbox("Select a keyword to analyze:", KEYWORDS)

    # Attempt to load historical data from S3
    historical_data = download_from_s3('strategicinsights', 'news.csv')

    if st.button("Search"):
        results = fetch_news(keyword)
        news_text = ""
        if results.get("articles"):
            for article in results["articles"]:
                title = article['title']
                description = article['description'] or "No description available"
                url = article['url']
                news_text += description + " "
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

                st.table(pd.DataFrame({
                    "Keyword": [keyword],
                    "Top Words": [top_words],
                    "Sentiment Score": [sentiment_score]
                }))

                new_data = pd.DataFrame({
                    "Date": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                    "Keyword": [keyword],
                    "Topics": [top_words],
                    "Sentiment": [sentiment_score]
                })
                historical_data = pd.concat([historical_data, new_data])
        else:
            st.write("No results found.")

    if not historical_data.empty:
        st.write("Sentiment Trend Analysis:")
        for key in KEYWORDS:
            key_data = historical_data[historical_data['Keyword'] == key]
            if not key_data.empty:
                st.line_chart(key_data.set_index('Date')['Sentiment'])

    if st.button("Update"):
        csv_data = historical_data.to_csv(index=False)
        if upload_to_s3('strategicinsights', 'news.csv', csv_data.encode()):
            st.success("Updated data uploaded to S3.")
        else:
            st.error("Failed to update data on S3.")

if __name__ == "__main__":
    main()
