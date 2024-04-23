import streamlit as st
import requests
from textblob import TextBlob
from newspaper import Article
from newspaper import Config
import nltk

# Global constants
API_KEY = '0debed01aa29475f9ff512e806bea611'
ENDPOINT = 'https://newsapi.org/v2/everything'

# Download the punkt tokenizer for content parsing
nltk.download('punkt')

# Configuration for newspaper3k to avoid being blocked
config = Config()
config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'

def fetch_full_text(url):
    try:
        article = Article(url, config=config)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        st.error(f"Failed to retrieve full article content: {str(e)}")
        return None

def fetch_news(keyword):
    headers = {'Authorization': f'Bearer {API_KEY}'}
    params = {'q': keyword, 'pageSize': 10, 'language': 'en'}
    response = requests.get(ENDPOINT, headers=headers, params=params)
    news_items = response.json().get('articles', [])
    
    articles = []
    for item in news_items:
        full_content = fetch_full_text(item.get('url'))
        if full_content:
            sentiment = TextBlob(full_content).sentiment
        else:
            sentiment = TextBlob(item.get('description') or "").sentiment
        articles.append({
            "title": item.get('title'),
            "url": item.get('url'),
            "content": full_content if full_content else item.get('description'),
            "sentiment": sentiment
        })
    return articles

def main():
    st.title("News Sentiment Analyzer")
    keyword = st.text_input("Enter a keyword to search news:", "")
    if st.button("Fetch News"):
        with st.spinner('Fetching news articles...'):
            articles = fetch_news(keyword)
            for article in articles:
                sentiment = sentiment_label(article['sentiment'].polarity)
                st.subheader(article['title'])
                st.write(f"URL: [{article['title']}]({article['url']})")
                st.write(f"Sentiment: {sentiment}")
                st.write(f"Content: {article['content'][:500]}...")  # Display first 500 characters
                st.markdown("---")

def sentiment_label(polarity):
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

if __name__ == '__main__':
    main()
