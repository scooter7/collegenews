import streamlit as st
import requests
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from googleapiclient.discovery import build

# Function to download and return custom stopwords
def get_custom_stopwords(url):
    response = requests.get(url)
    # Split by line break to get a list of stopwords
    stopwords = set(response.text.split())
    return stopwords

def plot_wordcloud(words):
    # Use custom stopwords from the provided URL
    custom_stopwords_url = "https://github.com/aneesha/RAKE/raw/master/SmartStoplist.txt"
    custom_stopwords = get_custom_stopwords(custom_stopwords_url)
    wordcloud = WordCloud(width=800, height=800, 
                          background_color='white', 
                          stopwords=custom_stopwords, 
                          min_font_size=10).generate(words)
    
    # Create a matplotlib figure
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(wordcloud)
    ax.axis("off")
    ax.set_title("Word Cloud")
    st.pyplot(fig)  # Pass the figure to streamlit.pyplot

def analyze_sentiment(text):
    nltk.download("vader_lexicon", quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    sentiment = sid.polarity_scores(text)
    if sentiment["compound"] > 0:
        return "Positive"
    elif sentiment["compound"] < 0:
        return "Negative"
    else:
        return "Neutral"

def google_search(query):
    api_key = st.secrets["google_search"]["api_key"]
    cse_id = st.secrets["google_search"]["cse_id"]
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=query, cx=cse_id, num=3).execute()
    return res

def main():
    st.title("News Feed Analyzer")
    query = st.text_input("Enter news keyword to search:")
    if st.button("Search"):
        results = google_search(query)
        news_text = ""
        if "items" in results:
            for result in results["items"]:
                news_text += result.get("snippet", "") + " "
        if news_text:
            st.write("Word Cloud:")
            plot_wordcloud(news_text)
            sentiment = analyze_sentiment(news_text)
            st.write("Sentiment:", sentiment)
        else:
            st.write("No results found.")

if __name__ == "__main__":
    main()
