import streamlit as st
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from googleapiclient.discovery import build

def plot_wordcloud(words):
    wordcloud = WordCloud(width=800, height=800, 
                          background_color='white', 
                          stopwords=set(nltk.corpus.stopwords.words("english")), 
                          min_font_size=10).generate(words) 
    
    plt.figure(figsize=(8, 8), facecolor=None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad=0) 
    st.pyplot()

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
