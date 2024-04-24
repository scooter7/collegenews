import streamlit as st
import requests
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud

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

def analyze_sentiment(text):
    nltk.download("vader_lexicon", quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    sentiment = sid.polarity_scores(text)
    return "Positive" if sentiment["compound"] > 0 else "Negative" if sentiment["compound"] < 0 else "Neutral"

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
                sentiment = analyze_sentiment(news_text)
                st.write("Aggregate Sentiment:", sentiment)
        else:
            st.write("No results found.")

if __name__ == "__main__":
    main()
