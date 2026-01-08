import { API } from '../api.js';

export default class BatchView {
    async getHtml() {
        return `
            <div class="card">
                <h2>Batch Strategy Analysis</h2>
                <p>Run strategies on multiple stocks simultaneously to compare performance.</p>
                
                <div class="grid-2" style="margin-top: 1rem;">
                    <div>
                        <label>Stock Symbols (comma separated)</label>
                        <textarea id="batch-symbols" rows="4" style="width: 100%; margin-top: 5px;" placeholder="e.g. 2330, 2317, 2454, 2881">2330, 2317, 2454, 2881, 1301</textarea>
                    </div>
                    <div>
                        <label>Strategy Config</label>
                        <select id="batch-strategy" style="width: 100%; margin-top: 5px;">
                            <option value="ma_crossover">MA Crossover</option>
                        </select>
                        
                        <div style="margin-top: 10px;">
                            <label>Period</label>
                            <select id="batch-period" style="width: 100%;">
                                <option value="6mo">6 Months</option>
                                <option value="1y" selected>1 Year</option>
                                <option value="2y">2 Years</option>
                            </select>
                        </div>
                    </div>
                </div>

                <div class="card" style="margin-top: 1rem; background: rgba(56, 189, 248, 0.05); border: 1px solid rgba(56, 189, 248, 0.2);">
                    <h4 style="margin-top: 0; margin-bottom: 0.5rem; color: var(--text-main); font-size: 1rem;">Risk Settings (Optional)</h4>
                    <div class="grid-3">
                         <div>
                            <label style="font-size: 0.9rem;">Stop Loss %</label>
                            <input type="number" id="batch-sl" value="5" min="0" max="100" style="width: 100%;">
                        </div>
                         <div>
                            <label style="font-size: 0.9rem;">Take Profit %</label>
                            <input type="number" id="batch-tp" value="10" min="0" max="1000" style="width: 100%;">
                        </div>
                        <div>
                            <label style="font-size: 0.9rem;">Position Size %</label>
                            <input type="number" id="batch-size" value="100" min="1" max="100" style="width: 100%;">
                        </div>
                    </div>
                </div>

                <button class="btn" id="run-batch-btn" style="width: 100%; margin-top: 1rem;">
                    <i class="fa-solid fa-play"></i> Run Batch Analysis
                </button>
            </div>

            <div class="card" style="margin-top: 1rem;">
                <h3>Results</h3>
                <div id="batch-results">
                    <p style="color: var(--text-muted);">Results will appear here...</p>
                </div>
            </div>
        `;
    }

    async init() {
        document.getElementById('run-batch-btn').addEventListener('click', () => {
            this.runBatch();
        });
    }

    async runBatch() {
        const symbolsStr = document.getElementById('batch-symbols').value;
        const symbols = symbolsStr.split(',').map(s => s.trim()).filter(s => s);
        const strategy = document.getElementById('batch-strategy').value;
        const period = document.getElementById('batch-period').value;

        const sl = parseFloat(document.getElementById('batch-sl').value) / 100;
        const tp = parseFloat(document.getElementById('batch-tp').value) / 100;
        const size = parseFloat(document.getElementById('batch-size').value) / 100;

        const container = document.getElementById('batch-results');
        container.innerHTML = 'Running analysis... <i class="fa-solid fa-spinner fa-spin"></i>';

        try {
            const response = await API.post('/api/batch/backtest', {
                symbols,
                strategy,
                period,
                risk_params: {
                    stop_loss_pct: sl,
                    take_profit_pct: tp,
                    position_size_pct: size
                }
            });

            if (response.success && response.results) {
                this.renderResults(response.results);
            } else {
                container.innerHTML = `<p style="color: var(--danger)">Error: ${response.error}</p>`;
            }
        } catch (error) {
            console.error(error);
            container.innerHTML = `<p style="color: var(--danger)">Failed to run batch analysis.</p>`;
        }
    }

    renderResults(results) {
        if (!results || results.length === 0) {
            document.getElementById('batch-results').innerHTML = '<p>No results returned.</p>';
            return;
        }

        let html = `
            <table style="width: 100%; border-collapse: collapse;">
                <thead>
                    <tr style="border-bottom: 2px solid var(--border); text-align: left;">
                        <th style="padding: 10px;">Symbol</th>
                        <th style="padding: 10px;">Return %</th>
                        <th style="padding: 10px;">Sharpe</th>
                        <th style="padding: 10px;">Max DD %</th>
                        <th style="padding: 10px;">Action</th>
                    </tr>
                </thead>
                <tbody>
        `;

        results.forEach(res => {
            if (res.error) {
                html += `
                    <tr style="border-bottom: 1px solid var(--border-light); opacity: 0.6;">
                        <td style="padding: 10px;">${res.symbol}</td>
                        <td colspan="4" style="padding: 10px; color: var(--danger);">Error: ${res.error}</td>
                    </tr>
                `;
            } else {
                html += `
                    <tr style="border-bottom: 1px solid var(--border-light);">
                        <td style="padding: 10px; font-weight: bold;">${res.symbol}</td>
                        <td style="padding: 10px; color: ${res.return_pct >= 0 ? 'var(--success)' : 'var(--danger)'}">
                            ${res.return_pct.toFixed(2)}%
                        </td>
                        <td style="padding: 10px;">${res.sharpe_ratio.toFixed(2)}</td>
                        <td style="padding: 10px; color: var(--danger);">${res.max_drawdown_pct.toFixed(2)}%</td>
                        <td style="padding: 10px;">
                            <button class="btn-secondary" style="padding: 4px 8px; font-size: 0.8rem;" 
                                onclick="window.app.navigate('analysis'); setTimeout(() => window.app.views.analysis.loadStock('${res.symbol}'), 100);">
                                <i class="fa-solid fa-chart-line"></i> View
                            </button>
                        </td>
                    </tr>
                `;
            }
        });

        html += '</tbody></table>';
        document.getElementById('batch-results').innerHTML = html;
    }
}
