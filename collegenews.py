import streamlit as st
import requests
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from streamlit_echarts import st_echarts  # Import for echarts

# Function to fetch custom stopwords
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
    # Determine the color based on the sentiment score
    color = '#6DD400' if score > 0 else '#FFD93D' if score == 0 else '#FF4500'
    
    option = {
        "series": [
            {
                "type": 'gauge',
                "startAngle": 180,  # Start angle for semi-circle
                "endAngle": 0,      # End angle for semi-circle
                "min": -100,        # Minimum value of the gauge
                "max": 100,         # Maximum value of the gauge
                "splitNumber": 4,   # Divide scale into 3 parts (negative, neutral, positive)
                "pointer": {
                    "show": True,   # Show pointer
                    "length": '90%',
                    "width": 8
                },
                "axisLine": {
                    "lineStyle": {
                        "width": 15,
                        "color": [
                            [0.33, '#FF4500'],   # Red from -100 to -33
                            [0.5, '#FFD93D'],    # Yellow from -33 to 0
                            [0.67, '#FFD93D'],   # Yellow from 0 to 33
                            [1, '#6DD400']       # Green from 33 to 100
                        ]
                    }
                },
                "axisLabel": {
                    "show": False
                },
                "axisTick": {
                    "show": False
                },
                "splitLine": {
                    "show": False
                },
                "detail": {
                    "formatter": '{value}%',
                    "offsetCenter": [0, '80%'],  # Position of the detail (score)
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
    nltk.download("vader_lexicon", quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    sentiment = sid.polarity_scores(text)
    sentiment_score = sentiment['compound'] * 100  # Scale to percentage for gauge
    return sentiment_score

def fetch_news(query):
    ENDPOINT = 'https://newsapi.org/v2/everything'
    params = {
        'q': query,
        'apiKey': st.secrets["newsapi"]["api_key"],
        'pageSize': 10,  # Retrieve the top 10 articles
    }
    response = requests.get(ENDPOINT, params=params)
    return response.json()

def main():
    st.title("News Feed Analyzer")
    query = st.text_input("Enter news keyword to search:")
    if st.button("Search"):
        results = fetch_news(query)
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
        else:
            st.write("No results found.")

if __name__ == "__main__":
    main()
