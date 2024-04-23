import streamlit as st
import requests
from textblob import TextBlob

API_KEY = '0debed01aa29475f9ff512e806bea611'
ENDPOINT = 'https://newsapi.org/v2/everything'

def fetch_news(keyword):
    headers = {'Authorization': f'Bearer {API_KEY}'}
    params = {'q': keyword, 'pageSize': 10, 'language': 'en'}
    response = requests.get(ENDPOINT, headers=headers, params=params)
    news_items = response.json().get('articles', [])
    
    articles = []
    for item in news_items:
        description = item.get('description') or 'No description available'
        articles.append({
            "title": item.get('title'),
            "url": item.get('url'),
            "description": description,
            "sentiment": TextBlob(description).sentiment
        })
    return articles

def sentiment_label(polarity):
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

def main():
    st.title("News Sentiment Analyzer")
    keyword = st.text_input("Enter a keyword to search news:", "")
    if st.button("Fetch News"):
        if keyword:
            with st.spinner('Fetching news articles...'):
                articles = fetch_news(keyword)
                for article in articles:
                    sentiment = sentiment_label(article['sentiment'].polarity)
                    st.subheader(article['title'])
                    st.write(f"URL: [{article['title']}]({article['url']})")
                    st.write(f"Sentiment: {sentiment}")
                    st.write(f"Description: {article['description']}")
                    st.markdown("---")

if __name__ == '__main__':
    main()
