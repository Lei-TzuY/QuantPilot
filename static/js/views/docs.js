import { API } from '../api.js';

export default class DocsView {
    async getHtml() {
        return `
            <div class="card">
                <h2><i class="fa-solid fa-book"></i> API Documentation</h2>
                <p style="color: var(--text-muted);">QuantPilot REST API reference for quantitative research and integration.</p>
            </div>

            <div class="card">
                <h2>Stock Data</h2>
                
                <div class="api-endpoint">
                    <div class="method get">GET</div>
                    <code>/api/stock/{symbol}</code>
                    <p>Get historical OHLCV data for a stock.</p>
                    <details>
                        <summary>Parameters & Response</summary>
                        <div class="params">
                            <p><strong>Query Parameters:</strong></p>
                            <ul>
                                <li><code>period</code> - Data period: 1mo, 3mo, 6mo, 1y, 2y, 5y (default: 1y)</li>
                                <li><code>interval</code> - Data interval: 1d, 1wk, 1mo (default: 1d)</li>
                            </ul>
                            <p><strong>Response:</strong></p>
                            <pre>{ "success": true, "data": [...], "symbol": "2330" }</pre>
                        </div>
                    </details>
                </div>

                <div class="api-endpoint">
                    <div class="method get">GET</div>
                    <code>/api/stock/{symbol}/realtime</code>
                    <p>Get real-time price for a stock.</p>
                </div>

                <div class="api-endpoint">
                    <div class="method get">GET</div>
                    <code>/api/stock/{symbol}/info</code>
                    <p>Get stock information (name, sector, etc.).</p>
                </div>

                <div class="api-endpoint">
                    <div class="method get">GET</div>
                    <code>/api/stock/{symbol}/news</code>
                    <p>Get recent news for a stock.</p>
                </div>
            </div>

            <div class="card">
                <h2>Technical Analysis</h2>
                
                <div class="api-endpoint">
                    <div class="method get">GET</div>
                    <code>/api/analysis/{symbol}</code>
                    <p>Get technical indicators for a stock.</p>
                    <details>
                        <summary>Parameters & Response</summary>
                        <div class="params">
                            <p><strong>Query Parameters:</strong></p>
                            <ul>
                                <li><code>period</code> - Data period (default: 1y)</li>
                                <li><code>indicators</code> - Comma-separated: ma,rsi,macd,bollinger</li>
                            </ul>
                        </div>
                    </details>
                </div>

                <div class="api-endpoint">
                    <div class="method get">GET</div>
                    <code>/api/analysis/{symbol}/signals</code>
                    <p>Get trading signals based on technical analysis.</p>
                </div>
            </div>

            <div class="card">
                <h2>Backtesting</h2>
                
                <div class="api-endpoint">
                    <div class="method post">POST</div>
                    <code>/api/backtest</code>
                    <p>Run a backtest with specified strategy.</p>
                    <details>
                        <summary>Request Body & Response</summary>
                        <div class="params">
                            <pre>{
  "symbol": "2330",
  "strategy": "ma_crossover",
  "period": "2y",
  "initial_capital": 1000000,
  "params": { "short_window": 20, "long_window": 60 },
  "risk_params": {
    "stop_loss_pct": 0.05,
    "take_profit_pct": 0.1,
    "commission_rate": 0.001425,
    "tax_rate": 0.003,
    "slippage_pct": 0.001
  }
}</pre>
                            <p><strong>Response includes:</strong> return_pct, cagr, sharpe_ratio, win_rate, profit_factor, fee_summary</p>
                        </div>
                    </details>
                </div>

                <div class="api-endpoint">
                    <div class="method post">POST</div>
                    <code>/api/backtest/optimize</code>
                    <p>Run parameter optimization (grid search).</p>
                </div>

                <div class="api-endpoint">
                    <div class="method post">POST</div>
                    <code>/api/backtest/montecarlo</code>
                    <p>Run Monte Carlo simulation for risk analysis.</p>
                    <details>
                        <summary>Response</summary>
                        <div class="params">
                            <p><strong>Returns:</strong> VaR (95%/99%), CVaR, confidence bands, probability of profit</p>
                        </div>
                    </details>
                </div>

                <div class="api-endpoint">
                    <div class="method post">POST</div>
                    <code>/api/batch/backtest</code>
                    <p>Run backtest on multiple symbols at once.</p>
                </div>

                <div class="api-endpoint">
                    <div class="method get">GET</div>
                    <code>/api/strategies</code>
                    <p>Get available trading strategies.</p>
                </div>
            </div>

            <div class="card">
                <h2>Portfolio & Paper Trading</h2>
                
                <div class="api-endpoint">
                    <div class="method get">GET</div>
                    <code>/api/portfolio</code>
                    <p>Get current portfolio holdings.</p>
                </div>

                <div class="api-endpoint">
                    <div class="method post">POST</div>
                    <code>/api/portfolio/add</code>
                    <p>Add a position to portfolio.</p>
                </div>

                <div class="api-endpoint">
                    <div class="method get">GET</div>
                    <code>/api/paper/status</code>
                    <p>Get paper trading account status.</p>
                </div>

                <div class="api-endpoint">
                    <div class="method post">POST</div>
                    <code>/api/paper/buy</code>
                    <code>/api/paper/sell</code>
                    <p>Execute paper trades.</p>
                </div>
            </div>

            <div class="card">
                <h2>Alerts & Webhooks</h2>
                
                <div class="api-endpoint">
                    <div class="method get">GET</div>
                    <code>/api/alerts</code>
                    <p>Get all active alerts.</p>
                </div>

                <div class="api-endpoint">
                    <div class="method post">POST</div>
                    <code>/api/alerts</code>
                    <p>Create a new alert.</p>
                    <details>
                        <summary>Supported Conditions</summary>
                        <div class="params">
                            <ul>
                                <li><code>above</code> / <code>below</code> - Price alerts</li>
                                <li><code>volatility_spike</code> - ATR-based volatility</li>
                                <li><code>volume_surge</code> - Relative volume spike</li>
                                <li><code>rsi_oversold</code> / <code>rsi_overbought</code> - RSI conditions</li>
                            </ul>
                        </div>
                    </details>
                </div>

                <div class="api-endpoint">
                    <div class="method get">GET</div>
                    <code>/api/webhook/config</code>
                    <p>Get webhook configuration.</p>
                </div>

                <div class="api-endpoint">
                    <div class="method post">POST</div>
                    <code>/api/webhook/config</code>
                    <p>Configure Discord/Slack webhooks.</p>
                    <details>
                        <summary>Request Body</summary>
                        <div class="params">
                            <pre>{ "discord_webhook": "https://...", "slack_webhook": "https://...", "enabled": true }</pre>
                        </div>
                    </details>
                </div>

                <div class="api-endpoint">
                    <div class="method post">POST</div>
                    <code>/api/webhook/test</code>
                    <p>Send a test notification.</p>
                </div>
            </div>

            <div class="card">
                <h2>ML Signals</h2>
                
                <div class="api-endpoint">
                    <div class="method post">POST</div>
                    <code>/api/ml/train/{symbol}</code>
                    <p>Train a Random Forest model for a symbol.</p>
                </div>

                <div class="api-endpoint">
                    <div class="method get">GET</div>
                    <code>/api/ml/predict/{symbol}</code>
                    <p>Get ML-based signal prediction.</p>
                </div>

                <div class="api-endpoint">
                    <div class="method get">GET</div>
                    <code>/api/ml/importance/{symbol}</code>
                    <p>Get feature importance from trained model.</p>
                </div>
            </div>

            <style>
                .api-endpoint {
                    border: 1px solid var(--border);
                    border-radius: 8px;
                    padding: 1rem;
                    margin-bottom: 1rem;
                    background: rgba(0,0,0,0.2);
                }
                .api-endpoint code {
                    background: rgba(88, 166, 255, 0.2);
                    padding: 4px 8px;
                    border-radius: 4px;
                    font-family: 'JetBrains Mono', monospace;
                }
                .api-endpoint p {
                    margin: 0.5rem 0 0 0;
                    color: var(--text-muted);
                }
                .method {
                    display: inline-block;
                    padding: 2px 8px;
                    border-radius: 4px;
                    font-size: 0.8rem;
                    font-weight: bold;
                    margin-right: 0.5rem;
                }
                .method.get { background: #3fb950; color: #0d1117; }
                .method.post { background: #58a6ff; color: #0d1117; }
                .method.delete { background: #f85149; color: white; }
                details {
                    margin-top: 0.75rem;
                }
                summary {
                    cursor: pointer;
                    color: var(--primary);
                    font-size: 0.9rem;
                }
                .params {
                    margin-top: 0.75rem;
                    padding: 0.75rem;
                    background: rgba(0,0,0,0.3);
                    border-radius: 6px;
                    font-size: 0.9rem;
                }
                .params pre {
                    background: rgba(0,0,0,0.4);
                    padding: 0.75rem;
                    border-radius: 4px;
                    overflow-x: auto;
                    font-size: 0.85rem;
                }
                .params ul {
                    margin: 0.5rem 0;
                    padding-left: 1.5rem;
                }
                .params li {
                    margin: 0.25rem 0;
                }
            </style>
        `;
    }

    async init() {
        // No dynamic initialization needed for docs
    }
}
