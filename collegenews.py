import streamlit as st
import requests
from newspaper import Article, Config
from nltk.sentiment import SentimentIntensityAnalyzer
from rake_nltk import Rake
import nltk

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('vader_lexicon')

# Initialize SentimentIntensityAnalyzer and Rake
sia = SentimentIntensityAnalyzer()
rake = Rake()

API_KEY = '0debed01aa29475f9ff512e806bea611'
ENDPOINT = 'https://newsapi.org/v2/everything'

config = Config()
config.browser_user_agent = 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)'

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
        url = item.get('url')
        full_content = fetch_full_text(url)
        if full_content:
            sentiment = sia.polarity_scores(full_content)
            rake.extract_keywords_from_text(full_content)
            keywords = rake.get_ranked_phrases()[:5]  # Extract top 5 keywords
            articles.append({
                "title": item.get('title'),
                "url": url,
                "content": full_content,
                "sentiment": sentiment,
                "keywords": keywords
            })
        else:
            description = item.get('description') or "No description available"
            sentiment = sia.polarity_scores(description)
            rake.extract_keywords_from_text(description)
            keywords = rake.get_ranked_phrases()[:5]
            articles.append({
                "title": item.get('title'),
                "url": url,
                "content": description,
                "sentiment": sentiment,
                "keywords": keywords
            })
    return articles

def main():
    st.title("News Sentiment and Keyword Analyzer")
    keyword = st.text_input("Enter a keyword to search news:", "")
    if st.button("Fetch News"):
        with st.spinner('Fetching news articles...'):
            articles = fetch_news(keyword)
            for article in articles:
                st.subheader(article['title'])
                st.write(f"URL: [{article['title']}]({article['url']})")
                st.write("Sentiment Analysis:")
                st.write(f"Positive: {article['sentiment']['pos']:.2f}")
                st.write(f"Neutral: {article['sentiment']['neu']:.2f}")
                st.write(f"Negative: {article['sentiment']['neg']:.2f}")
                st.write(f"Compound: {article['sentiment']['compound']:.2f}")
                st.write("Top Keywords: ", ', '.join(article['keywords']))
                st.write(f"Content: {article['content'][:500]}...")  # Display first 500 characters
                st.markdown("---")

if __name__ == '__main__':
    main()
