from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os

from modules.data_fetcher import DataFetcher
from modules.technical_analysis import TechnicalAnalyzer
from modules.backtester import Backtester
from modules.signal_generator import SignalGenerator
from modules.portfolio_manager import PortfolioManager
from modules.news_fetcher import NewsFetcher
from modules.batch_processor import BatchProcessor
from modules.paper_trader import PaperTrader
from modules.ml_signal import MLSignalGenerator
from modules.alert_manager import AlertManager
from modules.monte_carlo import MonteCarloSimulator
from modules.webhook_notifier import WebhookNotifier
import threading
import time

app = Flask(__name__, static_folder="static")
CORS(app)

data_fetcher = DataFetcher()
technical_analyzer = TechnicalAnalyzer()
backtester = Backtester()
signal_generator = SignalGenerator()
portfolio_manager = PortfolioManager()
news_fetcher = NewsFetcher()
batch_processor = BatchProcessor(data_fetcher, backtester)
paper_trader = PaperTrader()
ml_signal = MLSignalGenerator()
alert_manager = AlertManager()
monte_carlo = MonteCarloSimulator()
webhook_notifier = WebhookNotifier()

# Background worker for alerts
def alert_worker():
    while True:
        try:
            alert_manager.check_alerts(data_fetcher)
        except Exception as e:
            print(f"Alert worker error: {e}")
        time.sleep(60)  # Check every 60 seconds

alert_thread = threading.Thread(target=alert_worker, daemon=True)
alert_thread.start()


# ==================== Static files ====================
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory("static", filename)


# ==================== Stock data APIs ====================
@app.route("/api/stock/<symbol>", methods=["GET"])
def get_stock_data(symbol):
    period = request.args.get("period", "1y")
    interval = request.args.get("interval", "1d")
    try:
        data = data_fetcher.get_stock_data(symbol, period, interval)
        return jsonify({"success": True, "data": data, "symbol": symbol})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/stock/<symbol>/info", methods=["GET"])
def get_stock_info(symbol):
    try:
        info = data_fetcher.get_stock_info(symbol)
        return jsonify({"success": True, "info": info})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/stock/<symbol>/realtime", methods=["GET"])
def get_realtime_price(symbol):
    try:
        price = data_fetcher.get_realtime_price(symbol)
        return jsonify({"success": True, "price": price})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


# ==================== Technical analysis APIs ====================
@app.route("/api/analysis/<symbol>", methods=["GET"])
def get_technical_analysis(symbol):
    period = request.args.get("period", "1y")
    indicators = request.args.get("indicators", "ma,rsi,macd").split(",")
    try:
        raw_data = data_fetcher.get_stock_data_df(symbol, period)
        analysis = technical_analyzer.analyze(raw_data, indicators)
        return jsonify({"success": True, "analysis": analysis, "symbol": symbol})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/analysis/<symbol>/signals", methods=["GET"])
def get_trading_signals(symbol):
    period = request.args.get("period", "6mo")
    try:
        raw_data = data_fetcher.get_stock_data_df(symbol, period)
        signals = signal_generator.generate_signals(raw_data)
        return jsonify({"success": True, "signals": signals, "symbol": symbol})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


# ==================== Backtest APIs ====================
@app.route("/api/backtest", methods=["POST"])
def run_backtest():
    payload = request.json or {}
    symbol = payload.get("symbol")
    strategy = payload.get("strategy", "ma_crossover")
    period = payload.get("period", "2y")
    initial_capital = payload.get("initial_capital", 1_000_000)
    initial_capital = payload.get("initial_capital", 1_000_000)
    params = payload.get("params", {})
    risk_params = payload.get("risk_params", {})

    if not symbol:
        return jsonify({"success": False, "error": "symbol is required"}), 400

    try:
        stock_data = data_fetcher.get_stock_data_df(symbol, period)
        result = backtester.run(stock_data, strategy, initial_capital, params, risk_params)
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/backtest/optimize", methods=["POST"])
def optimize_strategy():
    payload = request.json or {}
    symbol = payload.get("symbol")
    strategy = payload.get("strategy", "ma_crossover")
    period = payload.get("period", "2y")
    initial_capital = payload.get("initial_capital", 1_000_000)
    param_ranges = payload.get("param_ranges", {})
    risk_params = payload.get("risk_params", {})

    if not symbol:
        return jsonify({"success": False, "error": "symbol is required"}), 400

    try:
        stock_data = data_fetcher.get_stock_data_df(symbol, period)
        results = backtester.optimize(stock_data, strategy, initial_capital, param_ranges, risk_params)
        return jsonify({"success": True, "results": results})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/strategies", methods=["GET"])
def get_strategies():
    strategies = backtester.get_available_strategies()
    return jsonify({"success": True, "strategies": strategies})


@app.route("/api/backtest/montecarlo", methods=["POST"])
def run_monte_carlo():
    payload = request.json or {}
    symbol = payload.get("symbol")
    strategy = payload.get("strategy", "ma_crossover")
    period = payload.get("period", "2y")
    initial_capital = payload.get("initial_capital", 1_000_000)
    params = payload.get("params", {})
    risk_params = payload.get("risk_params", {})
    forecast_days = payload.get("forecast_days", 252)

    if not symbol:
        return jsonify({"success": False, "error": "symbol is required"}), 400

    try:
        stock_data = data_fetcher.get_stock_data_df(symbol, period)
        backtest_result = backtester.run(stock_data, strategy, initial_capital, params, risk_params)
        
        # Run Monte Carlo simulation on the equity curve
        mc_result = monte_carlo.run_simulation(
            backtest_result["equity_curve"],
            initial_capital=backtest_result["final_value"],
            forecast_days=forecast_days
        )
        
        return jsonify({
            "success": True,
            "backtest": {
                "return_pct": backtest_result["return_pct"],
                "sharpe_ratio": backtest_result["sharpe_ratio"],
                "max_drawdown_pct": backtest_result["max_drawdown_pct"]
            },
            "monte_carlo": mc_result
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/stock/<symbol>/news", methods=["GET"])
def get_stock_news(symbol):
    try:
        news = news_fetcher.get_news(symbol)
        return jsonify({"success": True, "news": news})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/batch/backtest", methods=["POST"])
def run_batch_backtest():
    data = request.json or {}
    symbols = data.get("symbols", [])
    if not symbols:
         return jsonify({"success": False, "error": "symbols list required"}), 400
         
    strategy = data.get("strategy", "ma_crossover")
    period = data.get("period", "1y")
    initial_capital = data.get("initial_capital", 1_000_000)
    params = data.get("params", {})
    risk_params = data.get("risk_params", {})
    
    try:
        results = batch_processor.run_batch_backtest(
            symbols, strategy, period, initial_capital, None, None, params, risk_params
        )
        return jsonify({"success": True, "results": results})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== Portfolio APIs ====================
@app.route("/api/portfolio", methods=["GET"])
def get_portfolio():
    try:
        portfolio = portfolio_manager.get_portfolio()
        return jsonify({"success": True, "portfolio": portfolio})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/portfolio/add", methods=["POST"])
def add_to_portfolio():
    data = request.json or {}
    try:
        symbol = data.get("symbol")
        shares = float(data.get("shares", 0))
        buy_price = float(data.get("buy_price", 0))
        if not symbol:
            raise ValueError("symbol is required")
        result = portfolio_manager.add_position(symbol, shares, buy_price)
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/portfolio/remove", methods=["POST"])
def remove_from_portfolio():
    data = request.json or {}
    try:
        symbol = data.get("symbol")
        if not symbol:
            raise ValueError("symbol is required")
        result = portfolio_manager.remove_position(symbol)
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/portfolio/performance", methods=["GET"])
def get_portfolio_performance():
    try:
        performance = portfolio_manager.get_performance(data_fetcher)
        return jsonify({"success": True, "performance": performance})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


# ==================== Popular stocks ====================
@app.route("/api/popular", methods=["GET"])
def get_popular_stocks():
    popular = [
        {"symbol": "2330", "name": "TSMC", "industry": "Semiconductors"},
        {"symbol": "2317", "name": "Hon Hai", "industry": "Electronics"},
        {"symbol": "2454", "name": "MediaTek", "industry": "Semiconductors"},
        {"symbol": "2412", "name": "Chunghwa Telecom", "industry": "Telecom"},
        {"symbol": "2882", "name": "Cathay Financial", "industry": "Banking"},
        {"symbol": "2881", "name": "Fubon Financial", "industry": "Banking"},
        {"symbol": "1301", "name": "Formosa Plastics", "industry": "Materials"},
        {"symbol": "2308", "name": "Delta Electronics", "industry": "Electronics"},
        {"symbol": "2303", "name": "UMC", "industry": "Semiconductors"},
        {"symbol": "3711", "name": "ASE", "industry": "Semiconductors"},
    ]
    return jsonify({"success": True, "stocks": popular})


# ==================== Paper Trading APIs ====================
@app.route("/api/paper/status", methods=["GET"])
def get_paper_status():
    try:
        status = paper_trader.get_portfolio_value(data_fetcher)
        return jsonify({"success": True, **status})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/paper/buy", methods=["POST"])
def paper_buy():
    data = request.json or {}
    symbol = data.get("symbol")
    shares = int(data.get("shares", 0))
    
    if not symbol or shares <= 0:
        return jsonify({"success": False, "error": "symbol and shares required"}), 400
    
    try:
        price_data = data_fetcher.get_realtime_price(symbol)
        price = price_data.get("price", 0)
        result = paper_trader.execute_buy(symbol, shares, price)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/paper/sell", methods=["POST"])
def paper_sell():
    data = request.json or {}
    symbol = data.get("symbol")
    shares = int(data.get("shares", 0))
    
    if not symbol or shares <= 0:
        return jsonify({"success": False, "error": "symbol and shares required"}), 400
    
    try:
        price_data = data_fetcher.get_realtime_price(symbol)
        price = price_data.get("price", 0)
        result = paper_trader.execute_sell(symbol, shares, price)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/paper/history", methods=["GET"])
def paper_history():
    return jsonify({"success": True, "trades": paper_trader.get_trade_history()})


@app.route("/api/paper/reset", methods=["POST"])
def paper_reset():
    result = paper_trader.reset()
    return jsonify(result)


# ==================== ML Signal APIs ====================
@app.route("/api/ml/train/<symbol>", methods=["POST"])
def train_ml_model(symbol):
    data = request.json or {}
    period = data.get("period", "2y")
    
    try:
        stock_df = data_fetcher.get_stock_data_df(symbol, period)
        result = ml_signal.train(stock_df, symbol)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/ml/predict/<symbol>", methods=["GET"])
def predict_ml_signal(symbol):
    period = request.args.get("period", "6mo")
    
    try:
        stock_df = data_fetcher.get_stock_data_df(symbol, period)
        result = ml_signal.predict(stock_df, symbol)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/ml/importance/<symbol>", methods=["GET"])
def get_ml_importance(symbol):
    try:
        result = ml_signal.get_feature_importance(symbol)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


# ==================== Alert APIs ====================
@app.route("/api/alerts", methods=["GET"])
def get_alerts():
    alerts = alert_manager.get_alerts()
    return jsonify({"success": True, "alerts": alerts})


@app.route("/api/alerts", methods=["POST"])
def create_alert():
    data = request.json or {}
    symbol = data.get("symbol")
    condition = data.get("condition", "above")
    target_value = float(data.get("target_value", 0))
    note = data.get("note", "")
    
    if not symbol or target_value <= 0:
        return jsonify({"success": False, "error": "symbol and target_value required"}), 400
    
    alert = alert_manager.create_alert(symbol, condition, target_value, note)
    return jsonify({"success": True, "alert": alert})


@app.route("/api/alerts/<alert_id>", methods=["DELETE"])
def delete_alert(alert_id):
    success = alert_manager.delete_alert(alert_id)
    if success:
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "Alert not found"}), 404


@app.route("/api/alerts/triggered", methods=["GET"])
def get_triggered_alerts():
    alerts = alert_manager.get_triggered_alerts()
    return jsonify({"success": True, "alerts": alerts})


# ==================== Webhook APIs ====================
@app.route("/api/webhook/config", methods=["GET"])
def get_webhook_config():
    config = webhook_notifier.get_config()
    return jsonify({"success": True, "config": config})


@app.route("/api/webhook/config", methods=["POST"])
def set_webhook_config():
    data = request.json or {}
    try:
        result = webhook_notifier.configure(
            discord_webhook=data.get("discord_webhook"),
            slack_webhook=data.get("slack_webhook"),
            enabled=data.get("enabled")
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/webhook/test", methods=["POST"])
def test_webhook():
    data = request.json or {}
    platform = data.get("platform", "discord")
    
    try:
        if platform == "discord":
            success = webhook_notifier.send_discord(
                "🧪 Test Notification",
                "This is a test message from QuantPilot!",
                0x58A6FF,
                [{"name": "Status", "value": "Working!", "inline": True}]
            )
        else:
            success = webhook_notifier.send_slack(
                "🧪 Test Notification",
                "This is a test message from QuantPilot!",
                [{"name": "Status", "value": "Working!"}]
            )
        
        return jsonify({"success": success})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


# ==================== Advanced ML APIs ====================
@app.route("/api/ml/lstm/train/<symbol>", methods=["POST"])
def train_lstm(symbol):
    """Train LSTM model for a symbol."""
    try:
        from modules.ml_lstm import LSTMPredictor
        
        payload = request.json or {}
        period = payload.get("period", "2y")
        epochs = payload.get("epochs", 50)
        
        df = data_fetcher.get_stock_data_df(symbol, period)
        
        lstm = LSTMPredictor()
        result = lstm.train(df, epochs=epochs)
        
        if result.get("success"):
            lstm.save(symbol)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/ml/lstm/predict/<symbol>", methods=["GET"])
def predict_lstm(symbol):
    """Get LSTM prediction for a symbol."""
    try:
        from modules.ml_lstm import LSTMPredictor
        
        df = data_fetcher.get_stock_data_df(symbol, "6mo")
        
        lstm = LSTMPredictor()
        if not lstm.load(symbol):
            return jsonify({"success": False, "error": "Model not trained. Train first."}), 400
        
        result = lstm.predict(df)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/ml/sentiment/<symbol>", methods=["GET"])
def analyze_sentiment(symbol):
    """Analyze news sentiment for a symbol."""
    try:
        from modules.sentiment_analyzer import SentimentAnalyzer
        
        news = news_fetcher.get_news(symbol, limit=20)
        
        analyzer = SentimentAnalyzer()
        sentiment_result = analyzer.analyze_news(news)
        signal = analyzer.get_trading_signal(sentiment_result)
        
        return jsonify({
            "success": True,
            "symbol": symbol,
            "sentiment": sentiment_result,
            "signal": signal
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/ml/rl/train/<symbol>", methods=["POST"])
def train_rl(symbol):
    """Train DQN agent for a symbol."""
    try:
        from modules.rl_agent import DQNAgent
        
        payload = request.json or {}
        period = payload.get("period", "2y")
        episodes = payload.get("episodes", 50)
        
        df = data_fetcher.get_stock_data_df(symbol, period)
        
        agent = DQNAgent(state_shape=(20, 12))
        result = agent.train(df, episodes=episodes, verbose=False)
        
        if result.get("success"):
            agent.save(symbol)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/ml/factors/<symbol>", methods=["GET"])
def get_factors(symbol):
    """Get multi-factor analysis for a symbol."""
    try:
        from modules.multi_factor import MultiFactor
        
        period = request.args.get("period", "1y")
        df = data_fetcher.get_stock_data_df(symbol, period)
        
        factor_model = MultiFactor()
        ic_results = factor_model.calculate_factor_ic(df)
        signal = factor_model.get_factor_signal(df)
        
        return jsonify({
            "success": True,
            "symbol": symbol,
            "signal": signal,
            "factor_ic": ic_results
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/ml/ensemble/<symbol>", methods=["GET"])
def get_ensemble(symbol):
    """Get ensemble prediction combining all models."""
    try:
        from modules.ml_ensemble import QuantEnsemble
        
        period = request.args.get("period", "6mo")
        df = data_fetcher.get_stock_data_df(symbol, period)
        
        # Get news for sentiment
        news = news_fetcher.get_news(symbol, limit=10)
        
        ensemble = QuantEnsemble()
        ensemble.load_models(symbol)
        result = ensemble.predict(df, news_items=news)
        
        return jsonify({
            "success": True,
            "symbol": symbol,
            **result
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/ml/xgboost/train/<symbol>", methods=["POST"])
def train_xgboost(symbol):
    """Train XGBoost/LightGBM model."""
    try:
        from modules.ml_xgboost import GradientBoostPredictor
        
        payload = request.json or {}
        period = payload.get("period", "2y")
        task = payload.get("task", "classification")
        
        df = data_fetcher.get_stock_data_df(symbol, period)
        
        model = GradientBoostPredictor(task=task)
        result = model.train(df, **payload)
        
        if result.get("success"):
            model.save(symbol)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/ml/xgboost/predict/<symbol>", methods=["GET"])
def predict_xgboost(symbol):
    """Get XGBoost prediction."""
    try:
        from modules.ml_xgboost import GradientBoostPredictor
        
        df = data_fetcher.get_stock_data_df(symbol, "6mo")
        
        model = GradientBoostPredictor()
        if not model.load(symbol):
            return jsonify({"success": False, "error": "Model not trained"}), 400
        
        result = model.predict(df)
        result["importance"] = model.get_feature_importance()
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/ml/transformer/train/<symbol>", methods=["POST"])
def train_transformer(symbol):
    """Train Transformer model."""
    try:
        from modules.ml_transformer import TransformerPredictor
        
        payload = request.json or {}
        period = payload.get("period", "2y")
        epochs = payload.get("epochs", 50)
        
        df = data_fetcher.get_stock_data_df(symbol, period)
        
        model = TransformerPredictor()
        result = model.train(df, epochs=epochs)
        
        if result.get("success"):
            model.save(symbol)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/ml/transformer/predict/<symbol>", methods=["GET"])
def predict_transformer(symbol):
    """Get Transformer prediction."""
    try:
        from modules.ml_transformer import TransformerPredictor
        
        df = data_fetcher.get_stock_data_df(symbol, "6mo")
        
        model = TransformerPredictor()
        if not model.load(symbol):
            return jsonify({"success": False, "error": "Model not trained"}), 400
        
        result = model.predict(df)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/ml/explain/<symbol>", methods=["GET"])
def explain_model(symbol):
    """Get model explanation using SHAP."""
    try:
        from modules.ml_xgboost import GradientBoostPredictor
        from modules.ml_explainer import ModelExplainer
        
        df = data_fetcher.get_stock_data_df(symbol, "1y")
        
        # Load XGBoost model
        gb = GradientBoostPredictor()
        if not gb.load(symbol):
            return jsonify({"success": False, "error": "XGBoost model not trained"}), 400
        
        # Get features
        data = gb._prepare_features(df)
        X = data[gb.feature_columns].values
        
        # Create explainer
        explainer = ModelExplainer()
        setup_result = explainer.create_explainer(gb.model, X, gb.feature_columns, "tree")
        
        if not setup_result.get("success"):
            return jsonify(setup_result), 400
        
        # Get global importance
        importance = explainer.get_global_importance(X)
        
        # Get explanation for latest prediction
        latest_explanation = explainer.explain_prediction(X[-1:])
        
        return jsonify({
            "success": True,
            "symbol": symbol,
            "global_importance": importance,
            "latest_prediction_explanation": latest_explanation
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/ml/anomaly/train/<symbol>", methods=["POST"])
def train_anomaly(symbol):
    """Train autoencoder for anomaly detection."""
    try:
        from modules.ml_autoencoder import AnomalyDetector
        
        payload = request.json or {}
        period = payload.get("period", "2y")
        
        df = data_fetcher.get_stock_data_df(symbol, period)
        
        detector = AnomalyDetector()
        result = detector.train(df)
        
        if result.get("success"):
            detector.save(symbol)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/ml/anomaly/detect/<symbol>", methods=["GET"])
def detect_anomaly(symbol):
    """Detect market anomalies."""
    try:
        from modules.ml_autoencoder import AnomalyDetector
        
        df = data_fetcher.get_stock_data_df(symbol, "6mo")
        
        detector = AnomalyDetector()
        if not detector.load(symbol):
            return jsonify({"success": False, "error": "Model not trained"}), 400
        
        result = detector.get_market_state(df)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/timeseries/decompose/<symbol>", methods=["GET"])
def decompose_series(symbol):
    """Decompose time series into components."""
    try:
        from modules.time_series import TimeSeriesDecomposer
        
        period = request.args.get("period", "1y")
        df = data_fetcher.get_stock_data_df(symbol, period)
        
        decomposer = TimeSeriesDecomposer()
        result = decomposer.decompose(df, period=5)
        
        # Don't return full components in API (too large)
        result.pop("components", None)
        
        return jsonify({"success": True, "symbol": symbol, **result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/timeseries/forecast/<symbol>", methods=["GET"])
def forecast_series(symbol):
    """Forecast future prices."""
    try:
        from modules.time_series import TimeSeriesDecomposer
        
        period = request.args.get("period", "1y")
        horizon = int(request.args.get("horizon", 10))
        
        df = data_fetcher.get_stock_data_df(symbol, period)
        
        decomposer = TimeSeriesDecomposer()
        decomposer.decompose(df)
        result = decomposer.forecast(df, horizon=horizon)
        
        return jsonify({"success": True, "symbol": symbol, **result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/timeseries/seasonal/<symbol>", methods=["GET"])
def seasonal_analysis(symbol):
    """Analyze seasonal patterns."""
    try:
        from modules.time_series import TimeSeriesDecomposer
        
        df = data_fetcher.get_stock_data_df(symbol, "2y")
        
        decomposer = TimeSeriesDecomposer()
        result = decomposer.seasonal_analysis(df)
        
        return jsonify({"success": True, "symbol": symbol, **result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/ml/gan/train/<symbol>", methods=["POST"])
def train_gan(symbol):
    """Train GAN for synthetic data generation."""
    try:
        from modules.ml_gan import FinancialGAN
        
        payload = request.json or {}
        period = payload.get("period", "2y")
        epochs = payload.get("epochs", 100)
        
        df = data_fetcher.get_stock_data_df(symbol, period)
        
        gan = FinancialGAN()
        result = gan.train(df, epochs=epochs)
        
        if result.get("success"):
            gan.save(symbol)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/ml/gan/generate/<symbol>", methods=["GET"])
def generate_synthetic(symbol):
    """Generate synthetic price data."""
    try:
        from modules.ml_gan import FinancialGAN
        
        n_samples = int(request.args.get("n_samples", 10))
        base_price = float(request.args.get("base_price", 100))
        
        gan = FinancialGAN()
        if not gan.load(symbol):
            return jsonify({"success": False, "error": "Model not trained"}), 400
        
        result = gan.generate(n_samples, base_price)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/optimize/bayesian/<symbol>", methods=["POST"])
def bayesian_optimize(symbol):
    """Optimize strategy parameters using Bayesian optimization."""
    try:
        from modules.bayesian_opt import optimize_strategy
        
        payload = request.json or {}
        strategy = payload.get("strategy", "ma_crossover")
        n_iterations = payload.get("n_iterations", 20)
        metric = payload.get("metric", "sharpe_ratio")
        period = payload.get("period", "2y")
        
        # Default param ranges
        if strategy == "ma_crossover":
            param_ranges = {
                "short_window": (5, 30),
                "long_window": (30, 100)
            }
        elif strategy == "rsi":
            param_ranges = {
                "rsi_period": (7, 21),
                "rsi_lower": (20, 40),
                "rsi_upper": (60, 80)
            }
        else:
            param_ranges = payload.get("param_ranges", {})
        
        # Override with user-provided ranges
        if "param_ranges" in payload:
            param_ranges.update(payload["param_ranges"])
        
        df = data_fetcher.get_stock_data_df(symbol, period)
        
        result = optimize_strategy(
            backtester, df, strategy, param_ranges,
            n_iterations=n_iterations, metric=metric
        )
        
        return jsonify({"success": True, "symbol": symbol, **result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


# ==================== Advanced ML APIs ====================
@app.route("/api/ml/features/generate", methods=["POST"])
def generate_ml_features():
    """Generate advanced ML features."""
    try:
        from modules.ml_feature_engineering import FeatureEngineering
        
        payload = request.json or {}
        symbol = payload.get("symbol")
        period = payload.get("period", "2y")
        
        if not symbol:
            return jsonify({"success": False, "error": "symbol is required"}), 400
        
        # Get stock data
        df = data_fetcher.get_stock_data_df(symbol, period)
        
        # Generate features
        fe = FeatureEngineering()
        features_df = fe.generate_all_features(df)
        
        # Get feature statistics
        stats = {
            'num_features': len(features_df.columns),
            'num_samples': len(features_df),
            'feature_names': list(features_df.columns),
            'missing_values': features_df.isnull().sum().to_dict()
        }
        
        return jsonify({
            "success": True,
            "symbol": symbol,
            "stats": stats
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/ml/train/advanced", methods=["POST"])
def train_advanced_ml():
    """Train advanced ML model."""
    try:
        from modules.ml_advanced import AdvancedMLManager
        from modules.ml_feature_engineering import FeatureEngineering
        from modules.ml_model_manager import MLModelManager
        
        payload = request.json or {}
        symbol = payload.get("symbol")
        period = payload.get("period", "2y")
        model_type = payload.get("model_type", "random_forest")
        tune_hyperparams = payload.get("tune_hyperparams", False)
        test_size = payload.get("test_size", 0.2)
        
        if not symbol:
            return jsonify({"success": False, "error": "symbol is required"}), 400
        
        # Get stock data
        df = data_fetcher.get_stock_data_df(symbol, period)
        
        # Generate features
        fe = FeatureEngineering()
        features_df = fe.generate_all_features(df)
        
        # Create ML manager
        ml_manager = AdvancedMLManager()
        
        # Create target
        features_df = ml_manager.create_target(features_df)
        
        # Remove NaN
        features_df = features_df.dropna()
        
        if len(features_df) < 100:
            return jsonify({"success": False, "error": "Insufficient data"}), 400
        
        # Prepare features
        feature_cols = [col for col in features_df.columns 
                       if col not in ['target', 'open', 'high', 'low', 'close', 'volume']]
        
        X = features_df[feature_cols]
        y = features_df['target']
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        # Train model
        train_result = ml_manager.train_model(
            X_train, y_train,
            model_type=model_type,
            tune_hyperparams=tune_hyperparams
        )
        
        # Evaluate
        eval_result = ml_manager.evaluate_model(X_test, y_test)
        
        # Get feature importance
        importance = ml_manager.get_feature_importance(top_n=20)
        
        # Save model
        model_manager = MLModelManager()
        model_id = model_manager.save_model(
            ml_manager.model,
            model_name=f"{symbol}_ml",
            model_type=model_type,
            metadata={
                'symbol': symbol,
                'period': period,
                'num_features': len(feature_cols),
                'train_accuracy': train_result['accuracy'],
                'test_accuracy': eval_result['accuracy'],
                'test_f1': eval_result['f1_score']
            }
        )
        
        return jsonify({
            "success": True,
            "symbol": symbol,
            "model_id": model_id,
            "model_type": model_type,
            "train_result": train_result,
            "test_result": eval_result,
            "feature_importance": importance
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/ml/predict/advanced", methods=["POST"])
def predict_advanced_ml():
    """Predict using advanced ML model."""
    try:
        from modules.ml_model_manager import MLModelManager
        from modules.ml_feature_engineering import FeatureEngineering
        
        payload = request.json or {}
        model_id = payload.get("model_id")
        symbol = payload.get("symbol")
        period = payload.get("period", "3mo")
        
        if not model_id:
            return jsonify({"success": False, "error": "model_id is required"}), 400
        
        # Load model
        model_manager = MLModelManager()
        model = model_manager.load_model(model_id)
        
        # Get stock data
        df = data_fetcher.get_stock_data_df(symbol, period)
        
        # Generate features
        fe = FeatureEngineering()
        features_df = fe.generate_all_features(df)
        features_df = features_df.dropna()
        
        # Prepare features
        feature_cols = [col for col in features_df.columns 
                       if col not in ['target', 'open', 'high', 'low', 'close', 'volume']]
        
        X = features_df[feature_cols]
        
        # Predict
        predictions = model.predict(X)
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)
            proba_list = probabilities.tolist()
        else:
            proba_list = None
        
        # Get latest prediction
        latest_pred = int(predictions[-1])
        latest_proba = proba_list[-1] if proba_list else None
        latest_signal = "BUY" if latest_pred == 1 else "SELL"
        confidence = max(latest_proba) if latest_proba else None
        
        return jsonify({
            "success": True,
            "symbol": symbol,
            "model_id": model_id,
            "latest_prediction": {
                "signal": latest_signal,
                "prediction": latest_pred,
                "probability": latest_proba,
                "confidence": confidence
            },
            "num_predictions": len(predictions)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/ml/backtest/ml_strategy", methods=["POST"])
def backtest_ml_strategy():
    """Backtest ML-based strategy."""
    try:
        from modules.ml_backtester import MLBacktester
        from modules.ml_model_manager import MLModelManager
        from modules.ml_feature_engineering import FeatureEngineering
        
        payload = request.json or {}
        model_id = payload.get("model_id")
        symbol = payload.get("symbol")
        period = payload.get("period", "2y")
        initial_capital = payload.get("initial_capital", 1_000_000)
        confidence_threshold = payload.get("confidence_threshold", 0.6)
        
        if not model_id:
            return jsonify({"success": False, "error": "model_id is required"}), 400
        
        # Load model
        model_manager = MLModelManager()
        model = model_manager.load_model(model_id)
        
        # Get stock data
        df = data_fetcher.get_stock_data_df(symbol, period)
        
        # Generate features
        fe = FeatureEngineering()
        features_df = fe.generate_all_features(df)
        features_df = features_df.dropna()
        
        # Prepare features
        feature_cols = [col for col in features_df.columns 
                       if col not in ['target', 'open', 'high', 'low', 'close', 'volume']]
        
        X = features_df[feature_cols]
        
        # Predict
        predictions = model.predict(X)
        probabilities = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
        
        # Align price data with predictions
        price_df = df.loc[features_df.index]
        
        # Run backtest
        backtester_ml = MLBacktester()
        result = backtester_ml.backtest_ml_strategy(
            price_df,
            predictions,
            probabilities if probabilities is not None else np.zeros((len(predictions), 2)),
            initial_capital=initial_capital,
            confidence_threshold=confidence_threshold
        )
        
        return jsonify({
            "success": True,
            "symbol": symbol,
            "model_id": model_id,
            "backtest_result": {
                'initial_capital': result['initial_capital'],
                'final_value': result['final_value'],
                'total_return': result['total_return'],
                'total_return_pct': result['total_return_pct'],
                'num_trades': result['num_trades'],
                'metrics': result['metrics']
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/ml/models", methods=["GET"])
def list_ml_models():
    """List all saved ML models."""
    try:
        from modules.ml_model_manager import MLModelManager
        
        model_manager = MLModelManager()
        
        model_name = request.args.get("model_name")
        model_type = request.args.get("model_type")
        
        models = model_manager.list_models(
            model_name=model_name,
            model_type=model_type
        )
        
        return jsonify({
            "success": True,
            "num_models": len(models),
            "models": models
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/ml/models/<model_id>", methods=["DELETE"])
def delete_ml_model(model_id):
    """Delete a saved ML model."""
    try:
        from modules.ml_model_manager import MLModelManager
        
        model_manager = MLModelManager()
        model_manager.delete_model(model_id)
        
        return jsonify({
            "success": True,
            "message": f"Model {model_id} deleted"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/ml/compare", methods=["POST"])
def compare_ml_models():
    """Compare multiple ML models."""
    try:
        from modules.ml_model_manager import MLModelManager
        from modules.ml_feature_engineering import FeatureEngineering
        
        payload = request.json or {}
        model_ids = payload.get("model_ids", [])
        symbol = payload.get("symbol")
        period = payload.get("period", "1y")
        
        if not model_ids:
            return jsonify({"success": False, "error": "model_ids is required"}), 400
        
        # Get stock data
        df = data_fetcher.get_stock_data_df(symbol, period)
        
        # Generate features
        fe = FeatureEngineering()
        features_df = fe.generate_all_features(df)
        
        # Create target
        from modules.ml_advanced import AdvancedMLManager
        ml_manager = AdvancedMLManager()
        features_df = ml_manager.create_target(features_df)
        features_df = features_df.dropna()
        
        # Prepare test data
        feature_cols = [col for col in features_df.columns 
                       if col not in ['target', 'open', 'high', 'low', 'close', 'volume']]
        
        X_test = features_df[feature_cols]
        y_test = features_df['target'].values
        
        # Compare models
        model_manager = MLModelManager()
        comparison = model_manager.compare_models(model_ids, X_test, y_test)
        
        return jsonify({
            "success": True,
            "symbol": symbol,
            "comparison": comparison
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/ml/walk_forward", methods=["POST"])
def walk_forward_analysis():
    """Perform walk-forward analysis."""
    try:
        from modules.ml_backtester import MLBacktester
        from modules.ml_advanced import AdvancedMLManager
        from modules.ml_feature_engineering import FeatureEngineering
        
        payload = request.json or {}
        symbol = payload.get("symbol")
        period = payload.get("period", "3y")
        model_type = payload.get("model_type", "random_forest")
        train_window = payload.get("train_window", 252)
        test_window = payload.get("test_window", 63)
        step_size = payload.get("step_size", 63)
        
        if not symbol:
            return jsonify({"success": False, "error": "symbol is required"}), 400
        
        # Get stock data
        df = data_fetcher.get_stock_data_df(symbol, period)
        
        if len(df) < train_window + test_window:
            return jsonify({"success": False, "error": "Insufficient data"}), 400
        
        # Create instances
        backtester_ml = MLBacktester()
        ml_manager = AdvancedMLManager(model_type=model_type)
        fe = FeatureEngineering()
        
        # Run walk-forward analysis
        result = backtester_ml.walk_forward_analysis(
            df,
            ml_manager,
            fe,
            train_window=train_window,
            test_window=test_window,
            step_size=step_size
        )
        
        return jsonify({
            "success": True,
            "symbol": symbol,
            "model_type": model_type,
            "walk_forward_result": result
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    os.makedirs("modules", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("=" * 50)
    print("Starting trading web app...")
    print("Browse: http://localhost:5000")
    print("=" * 50)

    app.run(debug=True, host="0.0.0.0", port=5000)
