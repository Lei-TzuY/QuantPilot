"""
NLP Sentiment Analysis Module
Analyzes news headlines and text for market sentiment.
"""

import re
from typing import Dict, List, Optional
from datetime import datetime

# Try to import NLP libraries
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class SentimentAnalyzer:
    """
    Multi-method sentiment analyzer for financial news.
    Supports TextBlob (simple) and Transformers (advanced).
    """
    
    def __init__(self, method: str = "auto"):
        """
        Initialize sentiment analyzer.
        
        Args:
            method: "textblob", "transformers", or "auto"
        """
        self.method = method
        self.transformer_model = None
        
        if method == "auto":
            if TRANSFORMERS_AVAILABLE:
                self.method = "transformers"
            elif TEXTBLOB_AVAILABLE:
                self.method = "textblob"
            else:
                self.method = "lexicon"
        
        # Financial sentiment lexicon (fallback)
        self.positive_words = {
            'surge', 'jump', 'gain', 'rise', 'soar', 'rally', 'boom', 'profit',
            'growth', 'bullish', 'upgrade', 'beat', 'outperform', 'strong',
            'positive', 'optimistic', 'recovery', 'expand', 'success', 'breakthrough',
            '上漲', '大漲', '利多', '創高', '突破', '成長', '獲利', '看好'
        }
        
        self.negative_words = {
            'fall', 'drop', 'decline', 'crash', 'plunge', 'slump', 'loss', 'bearish',
            'downgrade', 'miss', 'underperform', 'weak', 'negative', 'pessimistic',
            'recession', 'crisis', 'default', 'bankruptcy', 'warning', 'concern',
            '下跌', '大跌', '利空', '崩盤', '虧損', '衰退', '警訊', '擔憂'
        }
        
        self.intensity_modifiers = {
            'very': 1.5, 'extremely': 2.0, 'slightly': 0.5, 'somewhat': 0.7,
            'significantly': 1.5, 'dramatically': 2.0, 'sharply': 1.8
        }
    
    def _load_transformer(self):
        """Load FinBERT or similar financial sentiment model."""
        if self.transformer_model is None and TRANSFORMERS_AVAILABLE:
            try:
                # Try FinBERT first, fall back to general sentiment
                self.transformer_model = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    tokenizer="ProsusAI/finbert"
                )
            except Exception:
                try:
                    self.transformer_model = pipeline("sentiment-analysis")
                except Exception as e:
                    print(f"Failed to load transformer model: {e}")
                    self.method = "textblob" if TEXTBLOB_AVAILABLE else "lexicon"
    
    def _analyze_textblob(self, text: str) -> Dict:
        """Analyze sentiment using TextBlob."""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # -1 to 1
        subjectivity = blob.sentiment.subjectivity  # 0 to 1
        
        return {
            "score": polarity,
            "subjectivity": subjectivity,
            "label": "positive" if polarity > 0.1 else ("negative" if polarity < -0.1 else "neutral")
        }
    
    def _analyze_transformer(self, text: str) -> Dict:
        """Analyze sentiment using transformer model."""
        self._load_transformer()
        
        if self.transformer_model is None:
            return self._analyze_lexicon(text)
        
        try:
            result = self.transformer_model(text[:512])[0]  # Limit text length
            label = result['label'].lower()
            score = result['score']
            
            # Convert to -1 to 1 scale
            if label == 'positive':
                normalized_score = score
            elif label == 'negative':
                normalized_score = -score
            else:  # neutral
                normalized_score = 0
            
            return {
                "score": normalized_score,
                "confidence": score,
                "label": label
            }
        except Exception as e:
            return self._analyze_lexicon(text)
    
    def _analyze_lexicon(self, text: str) -> Dict:
        """Simple lexicon-based sentiment analysis."""
        text_lower = text.lower()
        words = re.findall(r'\w+', text_lower)
        
        positive_count = sum(1 for w in words if w in self.positive_words)
        negative_count = sum(1 for w in words if w in self.negative_words)
        
        total = positive_count + negative_count
        if total == 0:
            return {"score": 0, "label": "neutral", "confidence": 0}
        
        score = (positive_count - negative_count) / total
        
        # Apply intensity modifiers
        for word, modifier in self.intensity_modifiers.items():
            if word in text_lower:
                score *= modifier
        
        # Clamp to [-1, 1]
        score = max(-1, min(1, score))
        
        return {
            "score": score,
            "label": "positive" if score > 0.1 else ("negative" if score < -0.1 else "neutral"),
            "confidence": abs(score)
        }
    
    def analyze(self, text: str) -> Dict:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict with score (-1 to 1), label, and confidence
        """
        if not text or not text.strip():
            return {"score": 0, "label": "neutral", "confidence": 0}
        
        if self.method == "transformers":
            return self._analyze_transformer(text)
        elif self.method == "textblob":
            return self._analyze_textblob(text)
        else:
            return self._analyze_lexicon(text)
    
    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Analyze multiple texts."""
        return [self.analyze(text) for text in texts]
    
    def analyze_news(self, news_items: List[Dict]) -> Dict:
        """
        Analyze a list of news items and compute aggregate sentiment.
        
        Args:
            news_items: List of dicts with 'title' and optionally 'summary', 'date'
            
        Returns:
            Aggregate sentiment analysis result
        """
        if not news_items:
            return {
                "overall_score": 0,
                "overall_label": "neutral",
                "news_count": 0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "details": []
            }
        
        details = []
        scores = []
        
        for item in news_items:
            # Analyze title (most important)
            title = item.get('title', '')
            summary = item.get('summary', '')
            
            # Combine title and summary with title weighted more
            text = f"{title}. {summary}" if summary else title
            result = self.analyze(text)
            
            details.append({
                "title": title[:100],
                "score": result['score'],
                "label": result['label'],
                "date": item.get('date', '')
            })
            scores.append(result['score'])
        
        # Calculate aggregate
        avg_score = sum(scores) / len(scores) if scores else 0
        positive_count = sum(1 for d in details if d['label'] == 'positive')
        negative_count = sum(1 for d in details if d['label'] == 'negative')
        neutral_count = len(details) - positive_count - negative_count
        
        return {
            "overall_score": round(avg_score, 4),
            "overall_label": "positive" if avg_score > 0.1 else ("negative" if avg_score < -0.1 else "neutral"),
            "news_count": len(news_items),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
            "details": details[:10]  # Limit to 10 for API response
        }
    
    def get_trading_signal(self, sentiment_result: Dict) -> Dict:
        """
        Convert sentiment analysis to trading signal.
        
        Args:
            sentiment_result: Result from analyze_news()
            
        Returns:
            Trading signal with direction and strength
        """
        score = sentiment_result.get('overall_score', 0)
        news_count = sentiment_result.get('news_count', 0)
        
        if news_count < 3:
            confidence = "low"
        elif news_count < 10:
            confidence = "medium"
        else:
            confidence = "high"
        
        if score > 0.3:
            signal = "STRONG_BUY"
            direction = "bullish"
        elif score > 0.1:
            signal = "BUY"
            direction = "bullish"
        elif score < -0.3:
            signal = "STRONG_SELL"
            direction = "bearish"
        elif score < -0.1:
            signal = "SELL"
            direction = "bearish"
        else:
            signal = "HOLD"
            direction = "neutral"
        
        return {
            "signal": signal,
            "direction": direction,
            "sentiment_score": score,
            "confidence": confidence,
            "news_analyzed": news_count
        }
