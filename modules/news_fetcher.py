from typing import List, Dict
import yfinance as yf
from datetime import datetime

class NewsFetcher:
    """Fetch recent news and calculate simple sentiment."""

    def get_news(self, symbol: str) -> List[Dict]:
        try:
            # yfinance news
            ticker = yf.Ticker(symbol if not symbol.isdigit() else f"{symbol}.TW")
            news = ticker.news
            
            formatted_news = []
            for item in news:
                # Simple sentiment simulation (random or keyword based)
                # In a real app, use a model. Here we mock it for the "Premium" feel.
                title = item.get('title', '')
                sentiment = "neutral"
                score = 0
                
                # Mock keyword analysis
                positive_keywords = ['gain', 'up', 'rise', 'record', 'growth', 'bull', 'profit', 'high']
                negative_keywords = ['loss', 'down', 'fall', 'drop', 'bear', 'crash', 'low', 'warn']
                
                title_lower = title.lower()
                if any(k in title_lower for k in positive_keywords):
                    sentiment = "positive"
                    score = 0.8
                elif any(k in title_lower for k in negative_keywords):
                    sentiment = "negative"
                    score = -0.5
                
                # Format date
                ts = item.get('providerPublishTime', 0)
                date_str = datetime.fromtimestamp(ts).strftime('%Y-%m-%d') if ts else "Recent"

                formatted_news.append({
                    "title": title,
                    "link": item.get('link', '#'),
                    "publisher": item.get('publisher', 'Unknown'),
                    "date": date_str,
                    "sentiment": sentiment,
                    "score": score
                })
            
            return formatted_news[:10] # Top 10
        except Exception as e:
            print(f"Error fetching news for {symbol}: {e}")
            return []
