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
from io import StringIO, BytesIO
from GoogleNews import GoogleNews
import altair as alt
import os

# Set NLTK data directory
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
nltk.data.path.append(nltk_data_dir)

# Download necessary NLTK data and models
nltk.download("punkt", download_dir=nltk_data_dir, quiet=True)
nltk.download("vader_lexicon", download_dir=nltk_data_dir, quiet=True)
nltk.download("stopwords", download_dir=nltk_data_dir, quiet=True)
from nltk.corpus import stopwords

# Initialize GoogleNews
googlenews = GoogleNews(lang='en', region='US')

# Keywords for analysis
KEYWORDS = ['Troy University', 'University of South Alabama', 'Jacksonville State University',
            'University of Alabama', 'Auburn University', 'Columbus State University']

@st.cache_data
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
    options = {
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
    st_echarts(options=options, height="400px")

def analyze_sentiment(text):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    sentiment = sid.polarity_scores(text)
    sentiment_score = sentiment['compound'] * 100
    return sentiment_score

def fetch_news(keyword):
    try:
        googlenews.clear()
        googlenews.search(keyword)
        result = googlenews.result()
        if not result:
            st.error(f"No results found for {keyword}.")
            return []
        return result
    except Exception as e:
        st.error(f"Failed to fetch news for {keyword}: {e}")
        return []

@st.cache_data
def load_historical_data(bucket, object_key):
    s3 = boto3.client(
        's3',
        aws_access_key_id=st.secrets["aws"]["aws_access_key_id"],
        aws_secret_access_key=st.secrets["aws"]["aws_secret_access_key"]
    )
    try:
        response = s3.get_object(Bucket=bucket, Key=object_key)
        historical_data = pd.read_csv(response['Body'])
        historical_data['Date'] = pd.to_datetime(historical_data['Date'], errors='coerce').dt.date
        return historical_data
    except s3.exceptions.NoSuchKey:
        st.write(f"The file {object_key} does not exist in the S3 bucket {bucket}. A new file will be created.")
        return pd.DataFrame(columns=["Date", "Keyword", "Topics", "Sentiment"])
    except Exception as e:
        st.write(f"Could not load historical data from S3. Error: {e}")
        return pd.DataFrame(columns=["Date", "Keyword", "Topics", "Sentiment"])

def upload_csv_to_s3(df, bucket, object_key):
    s3 = boto3.client(
        's3',
        aws_access_key_id=st.secrets["aws"]["aws_access_key_id"],
        aws_secret_access_key=st.secrets["aws"]["aws_secret_access_key"]
    )
    try:
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        s3.put_object(Bucket=bucket, Key=object_key, Body=csv_buffer.getvalue())
        st.write(f"Data uploaded to S3 bucket `{bucket}` at `{object_key}`.")
    except Exception as e:
        st.error(f"Failed to upload data to S3: {e}")

def main():
    hide_button_css = """
            <style>
            .st-emotion-cache-ztfqz8.ef3psqc5 {
                display: none !important;
            }
            </style>
            """

    st.markdown(hide_button_css, unsafe_allow_html=True)

    st.title("News Feed Analyzer")

    # Load historical data from S3
    historical_data = load_historical_data(st.secrets["aws"]["bucket_name"], st.secrets["aws"]["object_key"])
    combined_data = historical_data.copy()
    current_sentiments = []

    for keyword in KEYWORDS:
        st.header(f"Keyword: {keyword}")

        # Fetch news articles for the keyword
        articles = fetch_news(keyword)
        if not articles:
            st.error(f"No results found for {keyword}.")
            continue

        # Display news links
        news_text = ""
        for article in articles:
            title = article['title']
            description = article['desc']
            url = article['link']
            if not url.startswith("http"):
                url = "https://news.google.com" + url  # Ensure the URL is correct
            news_text += f"{title} {description} "
            st.markdown(f"#### [{title}]({url})")
            st.markdown(f"*{description}*")
            st.markdown("---")

        # Generate and display word cloud
        if news_text:
            st.write("Aggregate Word Cloud:")
            plot_wordcloud(nltk.word_tokenize(news_text.lower()))

            # Analyze sentiment
            sentiment_score = analyze_sentiment(news_text)
            st.write("Aggregate Sentiment:")
            render_sentiment_gauge(sentiment_score)

            # Update historical data
            update_df = pd.DataFrame({
                "Date": [datetime.now().strftime("%Y-%m-%d")],
                "Keyword": [keyword],
                "Topics": [', '.join(word for word, count in Counter([word for word in nltk.word_tokenize(news_text.lower()) if word.isalpha() and word not in get_custom_stopwords("https://github.com/aneesha/RAKE/raw/master/SmartStoplist.txt")]).most_common(5))],
                "Sentiment": [sentiment_score]
            })

            combined_data = pd.concat([combined_data, update_df])

            current_sentiments.append({
                "Keyword": keyword,
                "Sentiment": sentiment_score
            })

            # Process the data for sentiment trend chart
            keyword_data = combined_data[combined_data['Keyword'] == keyword].copy()
            keyword_data['Date'] = pd.to_datetime(keyword_data['Date'], errors='coerce').dt.date
            keyword_data = keyword_data.dropna(subset=['Date'])
            keyword_data = keyword_data.sort_values('Date')

            # Ensure each day has a sentiment score
            date_range = pd.date_range(start=keyword_data['Date'].min(), end=keyword_data['Date'].max())
            keyword_data = keyword_data.set_index('Date').reindex(date_range).fillna(method='ffill').reset_index()
            keyword_data.columns = ['Date', 'Keyword', 'Topics', 'Sentiment']

            # Display sentiment trend chart
            if not keyword_data.empty:
                st.subheader(f"Sentiment Trend for \"{keyword}\":")
                point_chart = alt.Chart(keyword_data).mark_point().encode(
                    x=alt.X('Date:T', axis=alt.Axis(title='Date')),
                    y=alt.Y('Sentiment:Q', axis=alt.Axis(title='Sentiment Score')),
                    tooltip=['Date:T', 'Sentiment:Q']
                ).properties(width=700, height=400).interactive()
                st.altair_chart(point_chart)

    # Update S3 with the combined data after processing all keywords
    if st.button("Update All Data to S3"):
        upload_csv_to_s3(combined_data, st.secrets["aws"]["bucket_name"], st.secrets["aws"]["object_key"])

    # Display current sentiment column chart
    if current_sentiments:
        st.subheader("Current Sentiment of Each School")
        sentiment_df = pd.DataFrame(current_sentiments)
        
        column_chart = alt.Chart(sentiment_df).mark_bar().encode(
            x=alt.X('Keyword', sort=None, axis=alt.Axis(title='School')),
            y=alt.Y('Sentiment', axis=alt.Axis(title='Sentiment Score')),
            color=alt.condition(
                alt.datum.Sentiment > 0,  # Positive sentiment
                alt.value('#6DD400'),
                alt.value('#FF4500')  # Negative sentiment
            ),
            tooltip=['Keyword', 'Sentiment']
        ).properties(width=700, height=400).interactive()
        st.altair_chart(column_chart)

        # Download button for sentiment table
        csv_buffer = BytesIO()
        sentiment_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        st.download_button(
            label="Download Sentiment Data as CSV",
            data=csv_buffer,
            file_name='current_sentiments.csv',
            mime='text/csv'
        )

if __name__ == "__main__":
    main()
