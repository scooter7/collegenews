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

# Ensure necessary NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("vader_lexicon", quiet=True)
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

# Predefined keywords for analysis, with phrases wrapped in quotes
KEYWORDS = ['"Columbia University"', '"Yale University"', '"Brown University"', 
            '"Cornell University"', '"Princeton University"', '"Harvard University"']

def get_custom_stopwords(url):
    try:
        response = requests.get(url)
        stopwords = set(response.text.split())
        return stopwords
    except Exception as e:
        st.error(f"Failed to fetch custom stopwords: {e}")
        return set()

def plot_wordcloud(words):
    custom_stopwords_url = "https://github.com/aneesha/RAKE/raw/master/SmartStoplist.txt"
    custom_stopwords = get_custom_stopwords(custom_stopwords_url)
    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          stopwords=custom_stopwords,
                          min_font_size=10).generate(' '.join(words))
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
        'sortBy': 'publishedAt'  # Ensures the most recent news are fetched
    }
    response = requests.get(ENDPOINT, params=params)
    return response.json()

def upload_csv_to_s3(df, bucket, object_key):
    s3 = boto3.client(
        's3',
        aws_access_key_id=st.secrets["aws"]["aws_access_key_id"],
        aws_secret_access_key=st.secrets["aws"]["aws_secret_access_key"]
    )
    try:
        response = s3.get_object(Bucket=bucket, Key=object_key)
        existing_data = pd.read_csv(response['Body'])
        st.write("Existing data loaded from S3 for appending.")
        combined_data = pd.concat([existing_data, df], ignore_index=True)
    except Exception as e:
        st.write(f"Could not load existing data from S3, assuming new file. Error: {e}")
        combined_data = df

    csv_buffer = StringIO()
    combined_data.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    s3.put_object(Bucket=bucket, Key=object_key, Body=csv_buffer.getvalue())
    st.write(f"Data appended and uploaded to S3 bucket `{bucket}` at `{object_key}`.")

def main():
    st.title("News Feed Analyzer")

    if 'historical_data' not in st.session_state:
        st.session_state.historical_data = pd.DataFrame(columns=["Date", "Keyword", "Topics", "Sentiment"])

    for keyword in KEYWORDS:
        st.header(f"Keyword: {keyword}")
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
                # Word Cloud
                st.write("Aggregate Word Cloud:")
                plot_wordcloud(nltk.word_tokenize(news_text.lower()))

                # Sentiment Analysis
                sentiment_score = analyze_sentiment(news_text)
                st.write("Aggregate Sentiment:")
                render_sentiment_gauge(sentiment_score)

                # Process text for topics
                nltk_stopwords = set(stopwords.words('english'))
                custom_stopwords = get_custom_stopwords("https://github.com/aneesha/RAKE/raw/master/SmartStoplist.txt")
                all_stopwords = nltk_stopwords.union(custom_stopwords)

                words = nltk.word_tokenize(news_text.lower())
                filtered_words = [word for word in words if word.isalpha() and word not in all_stopwords]
                
                most_common_words = Counter(filtered_words).most_common(5)
                top_words = ', '.join(word for word, count in most_common_words)

                update_df = pd.DataFrame({
                    "Date": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                    "Keyword": [keyword],
                    "Topics": [top_words],
                    "Sentiment": [sentiment_score]
                })

                st.table(update_df)

                # Append new data to session state
                st.session_state.historical_data = pd.concat([st.session_state.historical_data, update_df])

                # Sentiment Trend Analysis for the current keyword
                try:
                    plot_data = st.session_state.historical_data.copy()
                    plot_data['Date'] = pd.to_datetime(plot_data['Date'])
                    key_data = plot_data[plot_data['Keyword'] == keyword]
                    if not key_data.empty:
                        st.subheader(f"Sentiment Trend for '{keyword}':")
                        st.line_chart(key_data.set_index('Date')['Sentiment'])
                except Exception as e:
                    st.error(f"Failed to plot sentiment data for '{keyword}': {e}")
        else:
            st.write("No results found for this keyword.")

    if st.button("Update All Data to S3"):
        st.write("Attempting to save all data to S3...")
        if not st.session_state.historical_data.empty:
            upload_csv_to_s3(st.session_state.historical_data, st.secrets["aws"]["bucket_name"], st.secrets["aws"]["object_key"])
        else:
            st.write("No data to save.")

if __name__ == "__main__":
    main()
