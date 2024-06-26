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
from GoogleNews import GoogleNews
import altair as alt

# Download necessary NLTK data and models
nltk.download("punkt", quiet=True)
nltk.download("vader_lexicon", quiet=True)
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

# Initialize GoogleNews
googlenews = GoogleNews()

# Keywords for analysis
KEYWORDS = ['Columbia University', 'Yale University', 'Brown University',
            'Cornell University', 'Princeton University', 'Harvard University']

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
    if 'last_successful_data' not in st.session_state:
        st.session_state.last_successful_data = {}

    for keyword in KEYWORDS:
        st.header(f"Keyword: {keyword}")
        articles = fetch_news(keyword)

        if not articles:
            st.error(f"No results found for {keyword}.")
            continue

        news_text = ""
        for article in articles:
            title = article['title']
            description = article['desc']
            url = article['link']
            news_text += f"{title} {description} "
            st.markdown(f"#### [{title}]({url})")
            st.markdown(f"*{description}*")
            st.markdown("---")

        if news_text:
            st.write("Aggregate Word Cloud:")
            plot_wordcloud(nltk.word_tokenize(news_text.lower()))

            sentiment_score = analyze_sentiment(news_text)
            st.write("Aggregate Sentiment:")
            render_sentiment_gauge(sentiment_score)

            nltk_stopwords = set(stopwords.words('english'))
            custom_stopwords = get_custom_stopwords("https://github.com/aneesha/RAKE/raw/master/SmartStoplist.txt")
            all_stopwords = nltk_stopwords.union(custom_stopwords)

            words = nltk.word_tokenize(news_text.lower())
            filtered_words = [word for word in words if word.isalpha() and word not in all_stopwords]
            
            most_common_words = Counter(filtered_words).most_common(5)
            top_words = ', '.join(word for word, count in most_common_words)

            update_df = pd.DataFrame({
                "Date": [datetime.now().strftime("%Y-%m-%d")],
                "Keyword": [keyword],
                "Topics": [top_words],
                "Sentiment": [sentiment_score]
            })

            st.table(update_df)

            st.session_state.historical_data = pd.concat([st.session_state.historical_data, update_df])

            # Aggregate data to get the most recent sentiment score per day
            aggregated_data = st.session_state.historical_data
            aggregated_data['Date'] = pd.to_datetime(aggregated_data['Date']).dt.date
            aggregated_data = (aggregated_data.sort_values(by='Date')
                               .groupby(['Date', 'Keyword'], as_index=False)
                               .last())

            # Filter data for the current keyword
            key_data = aggregated_data[aggregated_data['Keyword'] == keyword]
            if not key_data.empty:
                st.subheader(f"Sentiment Trend for \"{keyword}\":")
                
                # Create a line chart with points using Altair
                line = alt.Chart(key_data).mark_line(point=True).encode(
                    x=alt.X('Date:T', axis=alt.Axis(title='Date')),
                    y=alt.Y('Sentiment:Q', axis=alt.Axis(title='Sentiment Score')),
                    tooltip=['Date:T', 'Sentiment:Q']
                ).properties(
                    width=700,
                    height=400
                ).interactive()

                st.altair_chart(line)

    if st.button("Update All Data to S3"):
        st.write("Attempting to save all data to S3...")
        if not st.session_state.historical_data.empty:
            upload_csv_to_s3(st.session_state.historical_data, st.secrets["aws"]["bucket_name"], st.secrets["aws"]["object_key"])
        else:
            st.write("No data to save.")

if __name__ == "__main__":
    main()
