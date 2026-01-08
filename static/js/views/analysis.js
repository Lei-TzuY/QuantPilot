import { API } from '../api.js';
import { Charts } from '../charts.js';

export default class AnalysisView {
    constructor() {
        this.chart = null;
        this.indicatorChart = null;
    }

    async getHtml() {
        return `
            <div class="card">
                <div class="flex-between" style="margin-bottom: 1rem; align-items: flex-end;">
                    <div class="input-group" style="max-width: 300px; display: flex; align-items: center;">
                        <input type="text" id="symbol-input" placeholder="Stock Symbol (e.g., 2330)" value="2330">
                        <button class="btn" id="analyze-btn"><i class="fa-solid fa-search"></i></button>
                        <button class="btn-secondary" id="watchlist-btn" style="margin-left: 0.5rem; font-size: 1.2rem; padding: 0.5rem;" title="Add to Watchlist">
                            <i class="fa-regular fa-star"></i>
                        </button>
                    </div>
                    
                    <div style="display: flex; gap: 1rem; align-items: center;">
                        <label style="display: flex; align-items: center; gap: 0.5rem; cursor: pointer;">
                            <input type="checkbox" id="check-ma" checked> MA
                        </label>
                        <label style="display: flex; align-items: center; gap: 0.5rem; cursor: pointer;">
                            <input type="checkbox" id="check-bb"> BB
                        </label>
                        <label style="display: flex; align-items: center; gap: 0.5rem; cursor: pointer;">
                            <input type="checkbox" id="check-rsi"> RSI
                        </label>
                        <label style="display: flex; align-items: center; gap: 0.5rem; cursor: pointer;">
                            <input type="checkbox" id="check-macd"> MACD
                        </label>
                        <select id="period-select" style="width: auto;">
                            <option value="1mo">1M</option>
                            <option value="6mo" selected>6M</option>
                            <option value="1y">1Y</option>
                            <option value="5y">5Y</option>
                        </select>
                    </div>
                </div>
            </div>

            <div class="card" style="height: 400px;">
                <canvas id="analysis-chart"></canvas>
            </div>
            
            <div class="card" id="indicator-chart-card" style="height: 250px; margin-top: 1rem; display: none;">
                <canvas id="indicator-chart"></canvas>
            </div>

            <div class="grid-3" id="stats-container"></div>
            <div class="grid-4" id="fundamentals-container" style="margin-top: 1rem;"></div>

            <div class="card" style="margin-top: 1rem;">
                <h3>Strategy Optimization</h3>
                <div class="grid-2">
                    <div>
                        <label>Strategy</label>
                        <select id="opt-strategy" style="width: 100%; margin-top: 5px;">
                            <option value="ma_crossover">MA Crossover</option>
                        </select>
                    </div>
                    <div>
                        <label>Initial Capital</label>
                        <input type="number" id="opt-capital" value="1000000" style="width: 100%; margin-top: 5px;">
                    </div>
                </div>
                
                <div id="opt-params-ma" class="grid-2" style="margin-top: 1rem; gap: 1rem;">
                    <div>
                        <label>Short Window</label>
                        <input type="text" id="ma-short-range" value="5,10,20" style="width: 100%; margin-top: 5px;">
                    </div>
                    <div>
                        <label>Long Window</label>
                        <input type="text" id="ma-long-range" value="20,60,120" style="width: 100%; margin-top: 5px;">
                    </div>
                </div>

                <div class="card" style="margin-top: 1rem; padding: 0.8rem; background: rgba(56, 189, 248, 0.05); border: 1px solid rgba(56, 189, 248, 0.2);">
                    <h4 style="margin-top: 0; margin-bottom: 0.5rem; font-size: 1rem;">Risk Management</h4>
                    <div class="grid-3">
                        <div>
                            <label style="font-size: 0.9rem;">Stop Loss %</label>
                            <input type="number" id="risk-sl" value="5" min="0" max="100" style="width: 100%; margin-top: 5px;">
                        </div>
                        <div>
                            <label style="font-size: 0.9rem;">Take Profit %</label>
                            <input type="number" id="risk-tp" value="10" min="0" style="width: 100%; margin-top: 5px;">
                        </div>
                        <div>
                            <label style="font-size: 0.9rem;">Position %</label>
                            <input type="number" id="risk-size" value="100" min="1" max="100" style="width: 100%; margin-top: 5px;">
                        </div>
                    </div>
                </div>
                
                <button class="btn" id="optimize-btn" style="margin-top: 1rem; width: 100%;">
                    <i class="fa-solid fa-bolt"></i> Run Optimization
                </button>
                
                <div id="optimization-results" style="margin-top: 1rem; overflow-x: auto;"></div>
            </div>

            <div class="card" style="margin-top: 1rem;">
                <h3>Latest News & Sentiment</h3>
                <div id="news-container" style="margin-top: 1rem;">Loading...</div>
            </div>

            <div class="grid-2" style="margin-top: 1rem;">
                <div class="card">
                    <h3><i class="fa-solid fa-brain"></i> ML Prediction</h3>
                    <div id="ml-prediction-container" style="margin-top: 1rem;">
                        <p style="color: var(--text-muted)">Train a model first</p>
                    </div>
                    <div style="display: flex; gap: 0.5rem; margin-top: 1rem;">
                        <button class="btn-secondary" id="ml-train-btn" style="flex: 1;">
                            <i class="fa-solid fa-graduation-cap"></i> Train
                        </button>
                        <button class="btn" id="ml-predict-btn" style="flex: 1;">
                            <i class="fa-solid fa-magic"></i> Predict
                        </button>
                    </div>
                </div>

                <div class="card">
                    <h3><i class="fa-solid fa-paper-plane"></i> Paper Trading</h3>
                    <div id="paper-status-container" style="margin-top: 1rem;">
                        <p style="color: var(--text-muted)">Cash: Loading...</p>
                    </div>
                    <div class="input-group" style="margin-top: 1rem;">
                        <input type="number" id="paper-shares" value="100" min="1" placeholder="Shares">
                    </div>
                    <div style="display: flex; gap: 0.5rem; margin-top: 0.5rem;">
                        <button class="btn" id="paper-buy-btn" style="flex: 1; background: var(--success);">
                            <i class="fa-solid fa-arrow-up"></i> Buy
                        </button>
                        <button class="btn" id="paper-sell-btn" style="flex: 1; background: var(--danger);">
                            <i class="fa-solid fa-arrow-down"></i> Sell
                        </button>
                    </div>
                </div>
            </div>
        `;
    }

    async init() {
        const load = () => {
            const symbol = document.getElementById('symbol-input').value;
            this.loadStock(symbol);
        };

        document.getElementById('analyze-btn').addEventListener('click', load);
        document.getElementById('period-select').addEventListener('change', load);
        document.getElementById('check-ma').addEventListener('change', load);
        document.getElementById('check-bb').addEventListener('change', load);
        document.getElementById('check-rsi').addEventListener('change', () => { document.getElementById('check-macd').checked = false; load(); });
        document.getElementById('check-macd').addEventListener('change', () => { document.getElementById('check-rsi').checked = false; load(); });
        document.getElementById('optimize-btn').addEventListener('click', () => this.runOptimization());
        document.getElementById('ml-train-btn').addEventListener('click', () => this.trainML());
        document.getElementById('ml-predict-btn').addEventListener('click', () => this.predictML());
        document.getElementById('paper-buy-btn').addEventListener('click', () => this.paperBuy());
        document.getElementById('paper-sell-btn').addEventListener('click', () => this.paperSell());

        this.loadStock('2330');
        this.loadPaperStatus();
    }

    async loadStock(symbol) {
        const input = document.getElementById('symbol-input');
        if (input) input.value = symbol;

        this.loadNews(symbol);
        this.loadFundamentals(symbol);
        this.updateWatchlistIcon(symbol);

        const period = document.getElementById('period-select').value;
        const showMa = document.getElementById('check-ma').checked;
        const showBb = document.getElementById('check-bb').checked;
        const showRsi = document.getElementById('check-rsi').checked;
        const showMacd = document.getElementById('check-macd').checked;

        const ctx = document.getElementById('analysis-chart').getContext('2d');
        const indicatorCtx = document.getElementById('indicator-chart').getContext('2d');
        const indicatorCard = document.getElementById('indicator-chart-card');

        try {
            Charts.destroy(this.chart);
            Charts.destroy(this.indicatorChart);

            const priceResponse = await API.get(`/api/stock/${symbol}`, { period });

            if (priceResponse.success && priceResponse.data) {
                const dates = priceResponse.data.map(d => d.date);
                const prices = priceResponse.data.map(d => d.close);

                let datasets = [{
                    label: `${symbol} Price`, data: prices, borderColor: '#38bdf8', backgroundColor: '#38bdf820',
                    borderWidth: 2, pointRadius: 0, fill: true, tension: 0.1
                }];

                let indicatorsList = [];
                if (showMa) indicatorsList.push('ma');
                if (showBb) indicatorsList.push('bollinger');
                if (showRsi) indicatorsList.push('rsi');
                if (showMacd) indicatorsList.push('macd');

                indicatorCard.style.display = (showRsi || showMacd) ? 'block' : 'none';

                if (indicatorsList.length > 0) {
                    const analysisResponse = await API.get(`/api/analysis/${symbol}`, { period, indicators: indicatorsList.join(',') });
                    if (analysisResponse.success && analysisResponse.analysis) {
                        const analysis = analysisResponse.analysis;
                        if (showMa && analysis.ma) {
                            analysis.ma.periods.forEach((p, i) => {
                                const colors = ['#fbbf24', '#f87171', '#a78bfa'];
                                datasets.push({ label: `MA ${p}`, data: analysis.ma.data.map(d => d[`ma_${p}`]), borderColor: colors[i % 3], borderWidth: 1.5, pointRadius: 0, fill: false });
                            });
                        }
                        if (showBb && analysis.bollinger) {
                            datasets.push({ label: 'Upper BB', data: analysis.bollinger.data.map(d => d.upper), borderColor: '#34d399', borderWidth: 1, pointRadius: 0, fill: false });
                            datasets.push({ label: 'Lower BB', data: analysis.bollinger.data.map(d => d.lower), borderColor: '#34d399', borderWidth: 1, pointRadius: 0, fill: false });
                        }
                        let indicatorDatasets = [];
                        if (showRsi && analysis.rsi) {
                            indicatorDatasets.push({ label: 'RSI', data: analysis.rsi.data.map(d => d.rsi), borderColor: '#a78bfa', borderWidth: 2, pointRadius: 0, fill: false });
                        } else if (showMacd && analysis.macd) {
                            indicatorDatasets.push({ label: 'MACD', data: analysis.macd.data.map(d => d.macd), borderColor: '#38bdf8', borderWidth: 2, pointRadius: 0, fill: false });
                            indicatorDatasets.push({ label: 'Signal', data: analysis.macd.data.map(d => d.signal), borderColor: '#fbbf24', borderWidth: 2, pointRadius: 0, fill: false });
                        }
                        if (indicatorDatasets.length > 0) {
                            this.indicatorChart = Charts.createLineChart(indicatorCtx, 'Indicators', dates, indicatorDatasets, '#a78bfa');
                        }
                    }
                }

                this.chart = Charts.createLineChart(ctx, `${symbol} Analysis`, dates, datasets, '#38bdf8');
                this.updateStats(priceResponse.data);
            }
        } catch (error) {
            console.error("Failed to load stock data", error);
        }
    }

    updateStats(data) {
        if (data.length > 1) {
            const prices = data.map(d => d.close);
            const lastPrice = prices[prices.length - 1];
            const prevPrice = prices[prices.length - 2];
            const change = ((lastPrice - prevPrice) / prevPrice * 100).toFixed(2);
            const volume = data[data.length - 1].volume;
            document.getElementById('stats-container').innerHTML = `
                <div class="card"><h3>Last Price</h3><p style="font-size: 1.5rem; font-weight: bold; color: ${change >= 0 ? 'var(--success)' : 'var(--danger)'}">${lastPrice.toFixed(2)} <span style="font-size: 1rem;">(${change}%)</span></p></div>
                <div class="card"><h3>Volume</h3><p style="font-size: 1.5rem;">${(volume / 1000).toFixed(0)} K</p></div>
                <div class="card"><h3>High/Low</h3><p>H: ${Math.max(...prices).toFixed(2)}<br>L: ${Math.min(...prices).toFixed(2)}</p></div>
            `;
        }
    }

    async loadFundamentals(symbol) {
        const container = document.getElementById('fundamentals-container');
        try {
            const response = await API.get(`/api/stock/${symbol}/info`);
            if (response.success && response.info) {
                const i = response.info;
                container.innerHTML = `
                    <div class="card"><small>Market Cap</small><p style="font-weight: bold;">${i.marketCap ? (i.marketCap / 1e8).toFixed(2) + ' E' : 'N/A'}</p></div>
                    <div class="card"><small>PE Ratio</small><p style="font-weight: bold;">${i.trailingPE ? i.trailingPE.toFixed(2) : 'N/A'}</p></div>
                    <div class="card"><small>EPS</small><p style="font-weight: bold;">${i.trailingEps ? i.trailingEps.toFixed(2) : 'N/A'}</p></div>
                    <div class="card"><small>Div Yield</small><p style="font-weight: bold;">${i.dividendYield ? (i.dividendYield * 100).toFixed(2) + '%' : 'N/A'}</p></div>
                `;
            } else { container.innerHTML = ''; }
        } catch { container.innerHTML = ''; }
    }

    async runOptimization() {
        const symbol = document.getElementById('symbol-input').value;
        const initialCapital = parseFloat(document.getElementById('opt-capital').value);
        const strategy = document.getElementById('opt-strategy').value;
        const resultsDiv = document.getElementById('optimization-results');
        const sl = parseFloat(document.getElementById('risk-sl').value) / 100;
        const tp = parseFloat(document.getElementById('risk-tp').value) / 100;
        const size = parseFloat(document.getElementById('risk-size').value) / 100;

        resultsDiv.innerHTML = '<p>Running... <i class="fa-solid fa-spinner fa-spin"></i></p>';

        let paramRanges = {};
        if (strategy === 'ma_crossover') {
            paramRanges = {
                short_window: document.getElementById('ma-short-range').value.split(',').map(Number),
                long_window: document.getElementById('ma-long-range').value.split(',').map(Number)
            };
        }

        try {
            const response = await API.post('/api/backtest/optimize', {
                symbol, strategy, initial_capital: initialCapital, period: '2y', param_ranges: paramRanges,
                risk_params: { stop_loss_pct: sl, take_profit_pct: tp, position_size_pct: size }
            });

            if (response.success && response.results) {
                this.renderOptimizationResults(response.results);
            } else {
                resultsDiv.innerHTML = `<p style="color: var(--danger)">Failed: ${response.error}</p>`;
            }
        } catch (error) {
            resultsDiv.innerHTML = `<p style="color: var(--danger)">Error: ${error.message}</p>`;
        }
    }

    renderOptimizationResults(results) {
        const resultsDiv = document.getElementById('optimization-results');
        if (!results || results.length === 0) { resultsDiv.innerHTML = '<p>No results.</p>'; return; }
        let html = `<table style="width: 100%; border-collapse: collapse;"><thead><tr style="border-bottom: 1px solid var(--border);"><th style="padding: 8px;">Params</th><th>Return %</th><th>Sharpe</th><th>Max DD %</th></tr></thead><tbody>`;
        results.forEach(res => {
            const p = Object.entries(res.params).map(([k, v]) => `${k}:${v}`).join(', ');
            html += `<tr style="border-bottom: 1px solid var(--border-light);"><td style="padding: 8px;">${p}</td><td style="color: ${res.return_pct >= 0 ? 'var(--success)' : 'var(--danger)'}">${res.return_pct.toFixed(2)}%</td><td>${res.sharpe_ratio.toFixed(2)}</td><td style="color: var(--danger)">${res.max_drawdown_pct.toFixed(2)}%</td></tr>`;
        });
        html += '</tbody></table>';
        resultsDiv.innerHTML = html;
    }

    async loadNews(symbol) {
        const container = document.getElementById('news-container');
        container.innerHTML = 'Loading... <i class="fa-solid fa-spinner fa-spin"></i>';
        try {
            const response = await API.get(`/api/stock/${symbol}/news`);
            if (response.success && response.news && response.news.length > 0) {
                container.innerHTML = response.news.map(item => `
                    <div style="padding: 0.8rem; border-bottom: 1px solid var(--border-light);">
                        <a href="${item.link}" target="_blank" style="text-decoration: none; color: var(--text-main);">${item.title}</a>
                        <div style="font-size: 0.85rem; color: var(--text-muted); margin-top: 0.3rem;">${item.publisher} • <span style="color: ${this.getSentimentColor(item.sentiment)}">${item.sentiment}</span></div>
                    </div>
                `).join('');
            } else { container.innerHTML = '<p>No recent news.</p>'; }
        } catch { container.innerHTML = '<p>Failed to load news.</p>'; }
    }

    getSentimentColor(s) { return s === 'positive' ? 'var(--success)' : s === 'negative' ? 'var(--danger)' : 'var(--secondary)'; }

    updateWatchlistIcon(symbol) {
        const btn = document.getElementById('watchlist-btn');
        if (!btn) return;
        const icon = btn.querySelector('i');
        const watchlist = JSON.parse(localStorage.getItem('watchlist') || '[]');
        if (watchlist.includes(symbol)) { icon.classList.remove('fa-regular'); icon.classList.add('fa-solid'); icon.style.color = '#fbbf24'; }
        else { icon.classList.remove('fa-solid'); icon.classList.add('fa-regular'); icon.style.color = 'var(--text-main)'; }
        const newBtn = btn.cloneNode(true);
        btn.parentNode.replaceChild(newBtn, btn);
        newBtn.addEventListener('click', () => this.toggleWatchlist(symbol));
    }

    toggleWatchlist(symbol) {
        let watchlist = JSON.parse(localStorage.getItem('watchlist') || '[]');
        if (watchlist.includes(symbol)) { watchlist = watchlist.filter(s => s !== symbol); }
        else { watchlist.push(symbol); }
        localStorage.setItem('watchlist', JSON.stringify(watchlist));
        this.updateWatchlistIcon(symbol);
    }

    // ==================== ML Methods ====================
    async trainML() {
        const symbol = document.getElementById('symbol-input').value;
        const container = document.getElementById('ml-prediction-container');
        container.innerHTML = 'Training... <i class="fa-solid fa-spinner fa-spin"></i>';
        try {
            const response = await API.post(`/api/ml/train/${symbol}`, { period: '2y' });
            if (response.success) {
                container.innerHTML = `<p style="color: var(--success)">Model trained! Accuracy: <strong>${(response.accuracy * 100).toFixed(1)}%</strong></p>`;
            } else {
                container.innerHTML = `<p style="color: var(--danger)">${response.error}</p>`;
            }
        } catch (e) { container.innerHTML = `<p style="color: var(--danger)">Error: ${e.message}</p>`; }
    }

    async predictML() {
        const symbol = document.getElementById('symbol-input').value;
        const container = document.getElementById('ml-prediction-container');
        container.innerHTML = 'Predicting... <i class="fa-solid fa-spinner fa-spin"></i>';
        try {
            const response = await API.get(`/api/ml/predict/${symbol}`);
            if (response.success) {
                const color = response.signal === 'BUY' ? 'var(--success)' : 'var(--danger)';
                container.innerHTML = `
                    <p style="font-size: 1.5rem; font-weight: bold; color: ${color};">${response.signal}</p>
                    <p>Confidence: ${response.confidence}%</p>
                `;
            } else {
                container.innerHTML = `<p style="color: var(--danger)">${response.error}</p>`;
            }
        } catch (e) { container.innerHTML = `<p style="color: var(--danger)">Error: ${e.message}</p>`; }
    }

    // ==================== Paper Trading Methods ====================
    async loadPaperStatus() {
        const container = document.getElementById('paper-status-container');
        try {
            const response = await API.get('/api/paper/status');
            if (response.success) {
                container.innerHTML = `
                    <p><strong>Cash:</strong> $${response.cash.toLocaleString()}</p>
                    <p><strong>Total Value:</strong> $${response.total_value.toLocaleString()}</p>
                    <p><strong>Positions:</strong> ${response.holdings.length}</p>
                `;
            }
        } catch { container.innerHTML = '<p>Failed to load status.</p>'; }
    }

    async paperBuy() {
        const symbol = document.getElementById('symbol-input').value;
        const shares = parseInt(document.getElementById('paper-shares').value);
        try {
            const response = await API.post('/api/paper/buy', { symbol, shares });
            if (response.success) {
                alert(`Bought ${shares} shares of ${symbol}!`);
                this.loadPaperStatus();
            } else { alert(`Error: ${response.error}`); }
        } catch (e) { alert(`Error: ${e.message}`); }
    }

    async paperSell() {
        const symbol = document.getElementById('symbol-input').value;
        const shares = parseInt(document.getElementById('paper-shares').value);
        try {
            const response = await API.post('/api/paper/sell', { symbol, shares });
            if (response.success) {
                alert(`Sold ${shares} shares of ${symbol}! PnL: $${response.pnl.toFixed(2)}`);
                this.loadPaperStatus();
            } else { alert(`Error: ${response.error}`); }
        } catch (e) { alert(`Error: ${e.message}`); }
    }
}
