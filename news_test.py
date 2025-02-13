import requests
import pandas as pd
from datetime import datetime, timedelta

# Replace with your API key from NewsAPI
API_KEY = "2a65a97a01a449ca94aef7ac07ee8fc3"

# Define date range (last 7 days)
END_DATE = datetime.today().strftime('%Y-%m-%d')
START_DATE = (datetime.today() - timedelta(days=7)).strftime('%Y-%m-%d')

# Updated Query: Searching for banking call volume-related events
QUERY = "interest rates OR banking fraud OR regulatory changes"

# Financial News Sources (Filter to include only these)
FINANCIAL_SOURCES = [
    "bloomberg", "cnbc", "forbes", "business-insider", "financial-times",
    "the-wall-street-journal", "reuters", "the-economic-times"
]

# API Endpoint
URL = f"https://newsapi.org/v2/everything?q={QUERY}&from={START_DATE}&to={END_DATE}&language=en&sortBy=publishedAt&apiKey={API_KEY}"

# Make request
response = requests.get(URL)

if response.status_code == 200:
    articles = response.json().get("articles", [])
    
    # Extract relevant details and filter by financial sources
    data = []
    for article in articles:
        if article["source"]["name"].lower().replace(" ", "-") in FINANCIAL_SOURCES:
            data.append({
                "title": article["title"],
                "source": article["source"]["name"],
                "published_at": article["publishedAt"],
                "url": article["url"],  # Include URL for reference
                "content": article["content"]
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    df =df[['content']]
    # Display DataFrame
    print(df)
else:
    print("Error:", response.json())
