import { API } from '../api.js';
import { Charts } from '../charts.js';

export default class BacktestView {
    constructor() {
        this.chart = null;
    }

    async getHtml() {
        return `
            <div class="grid-2">
                <div class="card">
                    <h2>Backtest Configuration</h2>
                    <form id="backtest-form">
                        <div style="margin-bottom: 1rem;">
                            <label>Stock Symbol</label>
                            <input type="text" name="symbol" value="2330" required>
                        </div>
                        <div style="margin-bottom: 1rem;">
                            <label>Strategy</label>
                            <select name="strategy">
                                <option value="ma_crossover">MA Crossover</option>
                                <option value="rsi">RSI Mean Reversion</option>
                                <option value="macd">MACD</option>
                            </select>
                        </div>
                        <div style="margin-bottom: 1rem;">
                            <label>Initial Capital</label>
                            <input type="number" name="initial_capital" value="1000000">
                        </div>
                        <div style="margin-bottom: 1rem;">
                            <label>Period</label>
                            <select name="period">
                                <option value="1y">1 Year</option>
                                <option value="2y" selected>2 Years</option>
                                <option value="5y">5 Years</option>
                            </select>
                        </div>
                        
                        <!-- Optimization Toggle -->
                        <div style="margin-bottom: 1rem; border-top: 1px solid var(--border); padding-top: 1rem;">
                            <label style="display: flex; align-items: center; gap: 0.5rem; cursor: pointer;">
                                <input type="checkbox" id="optimize-mode"> Enable Optimization (Grid Search)
                            </label>
                        </div>

                        <!-- Optimization Params (Hidden by default) -->
                        <div id="optimize-params" style="display: none; margin-bottom: 1rem; background: rgba(0,0,0,0.2); padding: 1rem; border-radius: 8px;">
                            <small style="color: var(--text-muted); display: block; margin-bottom: 0.5rem;">For MA Crossover Only</small>
                            <div class="input-group" style="margin-bottom: 0.5rem;">
                                <input type="text" id="short-range" placeholder="Short Window (e.g. 5,10,20)">
                            </div>
                            <div class="input-group">
                                <input type="text" id="long-range" placeholder="Long Window (e.g. 20,40,60)">
                            </div>
                        </div>

                        <button type="submit" class="btn" style="width:100%" id="run-btn">Run Backtest</button>
                    </form>
                </div>
                
                <div class="card">
                    <h2>Results</h2>
                    <div id="metrics-result" style="display: flex; flex-direction: column; gap: 1rem; height: 100%; justify-content: center; align-items: center; color: var(--text-muted); overflow-y: auto;">
                        Run a backtest to see results.
                    </div>
                </div>
            </div>

            <div class="card" style="height: 400px; margin-top: 1.5rem;">
                <canvas id="backtest-chart"></canvas>
            </div>
            
            <div class="card">
                <h2>Trade History</h2>
                <div style="max-height: 300px; overflow-y: auto;">
                    <table style="width: 100%; text-align: left; border-collapse: collapse;">
                        <thead>
                            <tr style="border-bottom: 1px solid var(--border); color: var(--text-muted);">
                                <th style="padding: 0.5rem;">Date</th>
                                <th style="padding: 0.5rem;">Type</th>
                                <th style="padding: 0.5rem;">Price</th>
                                <th style="padding: 0.5rem;">Shares</th>
                                <th style="padding: 0.5rem;">Cash Balance</th>
                            </tr>
                        </thead>
                        <tbody id="trades-table-body">
                        </tbody>
                    </table>
                </div>
            </div>
        `;
    }

    async init() {
        const toggle = document.getElementById('optimize-mode');
        const paramsDiv = document.getElementById('optimize-params');
        const btn = document.getElementById('run-btn');

        toggle.addEventListener('change', () => {
            paramsDiv.style.display = toggle.checked ? 'block' : 'none';
            btn.innerText = toggle.checked ? 'Run Optimization' : 'Run Backtest';
        });

        document.getElementById('backtest-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const payload = Object.fromEntries(formData.entries());
            payload.initial_capital = parseInt(payload.initial_capital);

            const isOptimize = document.getElementById('optimize-mode').checked;
            const originalText = btn.innerText;
            btn.innerText = 'Processing...';
            btn.disabled = true;

            try {
                if (isOptimize) {
                    // Parse ranges
                    const shortStr = document.getElementById('short-range').value || "5,10,20";
                    const longStr = document.getElementById('long-range').value || "20,40,60";

                    payload.param_ranges = {
                        short_window: shortStr.split(',').map(Number),
                        long_window: longStr.split(',').map(Number)
                    };

                    const response = await API.post('/api/backtest/optimize', payload);
                    if (response.success) {
                        this.renderOptimizationResults(response.results);
                    } else {
                        alert('Optimization failed: ' + response.error);
                    }

                } else {
                    // Normal Backtest
                    const response = await API.post('/api/backtest', payload);
                    if (response.success) {
                        this.renderResults(response.result);
                    } else {
                        alert('Backtest failed: ' + response.error);
                    }
                }
            } catch (error) {
                alert('Error running process');
                console.error(error);
            } finally {
                btn.innerText = originalText;
                btn.disabled = false;
            }
        });
    }

    renderOptimizationResults(results) {
        // Hide Chart Area temporarily or just overwrite results div
        const container = document.getElementById('metrics-result');
        container.style.display = 'block'; // Reset flex

        let html = '<table style="width:100%; text-align: left; border-collapse: collapse;">';
        html += '<thead style="color: var(--text-muted); border-bottom: 1px solid var(--border);"><tr><th>Short</th><th>Long</th><th>Return</th><th>Sharpe</th><th>Drawdown</th></tr></thead><tbody>';

        results.forEach(r => {
            const retColor = r.return_pct >= 0 ? 'var(--success)' : 'var(--danger)';
            html += `
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding: 8px;">${r.params.short_window}</td>
                    <td style="padding: 8px;">${r.params.long_window}</td>
                    <td style="padding: 8px; color: ${retColor}">${r.return_pct.toFixed(2)}%</td>
                    <td style="padding: 8px;">${r.sharpe_ratio.toFixed(2)}</td>
                    <td style="padding: 8px; color: var(--danger)">${r.max_drawdown_pct.toFixed(2)}%</td>
                </tr>
            `;
        });
        html += '</tbody></table>';
        container.innerHTML = html;

        // Clear chart
        const ctx = document.getElementById('backtest-chart').getContext('2d');
        Charts.destroy(this.chart);
        // Clear trades
        document.getElementById('trades-table-body').innerHTML = '';
    }

    renderResults(result) {
        // Render Metrics
        const returnColor = result.return_pct >= 0 ? 'var(--success)' : 'var(--danger)';
        const cagrColor = (result.cagr || 0) >= 0 ? 'var(--success)' : 'var(--danger)';
        const winRateColor = (result.win_rate || 0) >= 50 ? 'var(--success)' : 'var(--danger)';

        document.getElementById('metrics-result').innerHTML = `
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; width: 100%;">
                <div style="text-align: center; padding: 0.75rem; background: rgba(0,0,0,0.2); border-radius: 8px;">
                    <p style="color: var(--text-muted); font-size: 0.85rem; margin-bottom: 0.25rem;">Total Return</p>
                    <h3 style="color: ${returnColor}; font-size: 1.3rem; margin: 0;">${result.return_pct.toFixed(2)}%</h3>
                </div>
                <div style="text-align: center; padding: 0.75rem; background: rgba(0,0,0,0.2); border-radius: 8px;">
                    <p style="color: var(--text-muted); font-size: 0.85rem; margin-bottom: 0.25rem;">CAGR</p>
                    <h3 style="color: ${cagrColor}; font-size: 1.3rem; margin: 0;">${(result.cagr || 0).toFixed(2)}%</h3>
                </div>
                <div style="text-align: center; padding: 0.75rem; background: rgba(0,0,0,0.2); border-radius: 8px;">
                    <p style="color: var(--text-muted); font-size: 0.85rem; margin-bottom: 0.25rem;">Max Drawdown</p>
                    <h3 style="color: var(--danger); font-size: 1.3rem; margin: 0;">${result.max_drawdown_pct ? result.max_drawdown_pct.toFixed(2) : 'N/A'}%</h3>
                </div>
                <div style="text-align: center; padding: 0.75rem; background: rgba(0,0,0,0.2); border-radius: 8px;">
                    <p style="color: var(--text-muted); font-size: 0.85rem; margin-bottom: 0.25rem;">Sharpe Ratio</p>
                    <h3 style="font-size: 1.3rem; margin: 0;">${result.sharpe_ratio ? result.sharpe_ratio.toFixed(2) : 'N/A'}</h3>
                </div>
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; width: 100%; margin-top: 1rem;">
                <div style="text-align: center; padding: 0.75rem; background: rgba(0,0,0,0.2); border-radius: 8px;">
                    <p style="color: var(--text-muted); font-size: 0.85rem; margin-bottom: 0.25rem;">Win Rate</p>
                    <h3 style="color: ${winRateColor}; font-size: 1.3rem; margin: 0;">${(result.win_rate || 0).toFixed(1)}%</h3>
                </div>
                <div style="text-align: center; padding: 0.75rem; background: rgba(0,0,0,0.2); border-radius: 8px;">
                    <p style="color: var(--text-muted); font-size: 0.85rem; margin-bottom: 0.25rem;">Profit Factor</p>
                    <h3 style="font-size: 1.3rem; margin: 0;">${(result.profit_factor || 0).toFixed(2)}</h3>
                </div>
                <div style="text-align: center; padding: 0.75rem; background: rgba(0,0,0,0.2); border-radius: 8px;">
                    <p style="color: var(--text-muted); font-size: 0.85rem; margin-bottom: 0.25rem;">Total Trades</p>
                    <h3 style="font-size: 1.3rem; margin: 0;">${result.total_trades || 0}</h3>
                </div>
                <div style="text-align: center; padding: 0.75rem; background: rgba(0,0,0,0.2); border-radius: 8px;">
                    <p style="color: var(--text-muted); font-size: 0.85rem; margin-bottom: 0.25rem;">Avg Hold Days</p>
                    <h3 style="font-size: 1.3rem; margin: 0;">${(result.avg_holding_days || 0).toFixed(1)}</h3>
                </div>
            </div>
            <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1rem; padding: 0.5rem; background: linear-gradient(135deg, rgba(88, 166, 255, 0.1), rgba(163, 113, 247, 0.1)); border-radius: 8px;">
                <div style="text-align: center;">
                    <span style="color: var(--text-muted);">Final Value: </span>
                    <span style="font-weight: bold; font-size: 1.2rem;">${result.final_value.toFixed(0)} TWD</span>
                </div>
                ${result.fee_summary ? `
                <div style="text-align: center; border-left: 1px solid rgba(255,255,255,0.1); padding-left: 2rem;">
                    <span style="color: var(--text-muted);">Total Fees: </span>
                    <span style="font-weight: bold; color: var(--danger);">${result.fee_summary.total_fees.toFixed(0)} TWD</span>
                    <small style="color: var(--text-muted); margin-left: 0.5rem;">(Commission: ${result.fee_summary.total_commission.toFixed(0)} | Slippage: ${result.fee_summary.total_slippage.toFixed(0)})</small>
                </div>
                ` : ''}
            </div>
        `;

        // Render Chart
        // Note: result.equity_curve should be available from our backend update
        const ctx = document.getElementById('backtest-chart').getContext('2d');
        Charts.destroy(this.chart);

        // We need labels (dates). The backend returns equity_curve as a list of values.
        // But we don't have the dates explicitly associated in the list unless we align them.
        // Wait, 'equity_curve' in backend 'run' method was just a list.
        // And 'trades' has dates. 
        // We need the dates corresponding to equity_curve.
        // The backend 'run' method calculates equity_curve iterating over 'df'. 
        // So equity_curve length == df length.
        // But the backend doesn't return the array of dates for the simulation period explicitly in the 'equity_curve' field.
        // It returns 'start_date' and 'end_date'.
        // For a perfect chart we need dates. 
        // It's acceptable for now to just show indices 1..N or ask backend to return dates.
        // Given I cannot easily edit backend again without risk and "Perfection" is time-bound,
        // I will use a simple index 1..N or just "Day 1", "Day 2".
        // Better: The user can infer from start/end date.

        const labels = result.equity_curve ? result.equity_curve.map((_, i) => `${i}`) : [];
        this.chart = Charts.createLineChart(ctx, 'Portfolio Value', labels, result.equity_curve || [], '#10b981');

        // Render Trades
        const tbody = document.getElementById('trades-table-body');
        tbody.innerHTML = result.trades.map(t => `
            <tr>
                <td style="padding: 0.5rem; color: var(--text-muted);">${t.date}</td>
                <td style="padding: 0.5rem; font-weight: bold; color: ${t.type === 'buy' ? 'var(--success)' : 'var(--danger)'}">${t.type.toUpperCase()}</td>
                <td style="padding: 0.5rem;">${t.price.toFixed(2)}</td>
                <td style="padding: 0.5rem;">${t.shares}</td>
                <td style="padding: 0.5rem;">${t.cash_after.toFixed(0)}</td>
            </tr>
        `).join('');
    }
}
