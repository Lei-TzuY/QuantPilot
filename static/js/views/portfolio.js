import { API } from '../api.js';

export default class PortfolioView {
    async getHtml() {
        return `
            <div class="card">
                <div class="flex-between">
                    <h2>My Portfolio</h2>
                    <button class="btn" id="refresh-btn"><i class="fa-solid fa-sync"></i> Refresh</button>
                </div>
            </div>

            <div class="grid-2">
                <div class="card">
                    <h3>Add Position</h3>
                    <form id="add-position-form" style="margin-top: 1rem;">
                        <div class="input-group" style="margin-bottom: 1rem;">
                            <input type="text" name="symbol" placeholder="Symbol (e.g. 2330)" required>
                            <input type="number" name="shares" placeholder="Shares" required>
                        </div>
                         <div style="margin-bottom: 1rem;">
                            <input type="number" name="buy_price" placeholder="Buy Price (Optional)" step="0.01">
                        </div>
                        <button type="submit" class="btn btn-secondary" style="width:100%">Add to Portfolio</button>
                    </form>
                </div>

                <div class="card">
                    <h3>Performance</h3>
                    <div id="portfolio-performance">
                       Loading...
                    </div>
                </div>
            </div>

            <div class="card">
                <h3>Holdings</h3>
                <table style="width: 100%; text-align: left; border-collapse: collapse; margin-top: 1rem;">
                     <thead>
                        <tr style="border-bottom: 1px solid var(--border); color: var(--text-muted);">
                            <th style="padding: 0.5rem;">Symbol</th>
                            <th style="padding: 0.5rem;">Shares</th>
                            <th style="padding: 0.5rem;">Avg Price</th>
                            <th style="padding: 0.5rem;">Total Cost</th>
                            <th style="padding: 0.5rem;">Current Price</th>
                             <th style="padding: 0.5rem;">Value</th>
                            <th style="padding: 0.5rem;">P&L</th>
                            <th style="padding: 0.5rem;">Action</th>
                        </tr>
                    </thead>
                    <tbody id="portfolio-table-body">
                    </tbody>
                </table>
            </div>
        `;
    }

    async init() {
        document.getElementById('refresh-btn').addEventListener('click', () => this.loadPortfolio());

        document.getElementById('add-position-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const payload = Object.fromEntries(formData.entries());

            try {
                await API.post('/api/portfolio/add', payload);
                e.target.reset();
                this.loadPortfolio();
            } catch (err) {
                alert("Failed to add position");
            }
        });

        this.loadPortfolio();
    }

    async loadPortfolio() {
        try {
            const data = await API.get('/api/portfolio');
            if (data.success) {
                this.renderHoldings(data.portfolio);
            }

            // Allow failing on performance if no data
            try {
                const perf = await API.get('/api/portfolio/performance');
                if (perf.success) {
                    this.renderPerformance(perf.performance);
                }
            } catch (e) { console.log("Performance api error", e); } // Silently fail if API errors (e.g. no internet for realtime)

        } catch (error) {
            console.error(error);
        }
    }

    async renderHoldings(portfolio) {
        const tbody = document.getElementById('portfolio-table-body');
        const entries = Object.entries(portfolio);

        if (entries.length === 0) {
            tbody.innerHTML = '<tr><td colspan="8" style="padding: 1rem; text-align: center;">No holdings yet. Add one above!</td></tr>';
            return;
        }

        tbody.innerHTML = entries.map(([symbol, data]) => `
            <tr id="row-${symbol}">
                <td style="padding: 0.5rem; font-weight: bold;">${symbol}</td>
                <td style="padding: 0.5rem;">${data.shares}</td>
                <td style="padding: 0.5rem;">${data.buy_price.toFixed(2)}</td>
                <td style="padding: 0.5rem;">${(data.shares * data.buy_price).toFixed(0)}</td>
                <td class="current-price" style="padding: 0.5rem;">Loading...</td>
                <td class="market-value" style="padding: 0.5rem;">-</td>
                <td class="pnl" style="padding: 0.5rem;">-</td>
                <td style="padding: 0.5rem;">
                    <button class="btn-secondary" style="padding: 0.25rem 0.5rem; font-size: 0.8rem; background: var(--danger); color: white;" 
                            onclick="window.app.views.portfolio.remove('${symbol}')">Remove</button>
                </td>
            </tr>
        `).join('');

        // Fetch prices asynchronously
        for (const [symbol, data] of entries) {
            this.updateRowData(symbol, data.shares, data.buy_price);
        }
    }

    async updateRowData(symbol, shares, buyPrice) {
        try {
            const response = await API.get(`/api/stock/${symbol}/realtime`);
            if (response.success && response.price) {
                const currentPrice = response.price;
                const marketValue = currentPrice * shares;
                const cost = shares * buyPrice;
                const pnl = marketValue - cost;
                const pnlPct = cost > 0 ? (pnl / cost * 100) : 0;

                const row = document.getElementById(`row-${symbol}`);
                if (row) {
                    row.querySelector('.current-price').textContent = currentPrice.toFixed(2);
                    row.querySelector('.market-value').textContent = marketValue.toFixed(0);

                    const pnlCell = row.querySelector('.pnl');
                    pnlCell.innerHTML = `
                        <span style="color: ${pnl >= 0 ? 'var(--success)' : 'var(--danger)'}">
                            ${pnl >= 0 ? '+' : ''}${pnl.toFixed(0)} <small>(${pnlPct.toFixed(2)}%)</small>
                        </span>
                    `;
                }
            }
        } catch (error) {
            console.error(`Failed to update ${symbol}`, error);
        }
    }

    // Helper for delete
    async remove(symbol) {
        if (confirm(`Remove ${symbol}?`)) {
            try {
                await API.post('/api/portfolio/remove', { symbol });
                this.loadPortfolio();
            } catch (err) {
                alert("Failed to remove");
            }
        }
    }

    renderPerformance(perf) {
        document.getElementById('portfolio-performance').innerHTML = `
            <div style="font-size: 2rem; font-weight: bold; color: var(--primary);">
                ${perf.total_value.toFixed(0)} TWD
            </div>
            <div style="color: ${perf.total_profit >= 0 ? 'var(--success)' : 'var(--danger)'}">
                ${perf.total_profit >= 0 ? '+' : ''}${perf.total_profit.toFixed(0)} (${perf.total_return_pct.toFixed(2)}%)
            </div>
        `;

        // Also update table rows with real prices if available in perf data
        // perf object structure depends on backend. 
        // Backend 'get_performance' returns total stats.
        // It doesn't return per-stock realtime data explicitly in the 'performance' dict unless checked.
        // modules/portfolio_manager.py: calculate_performance iterates and fetches price.
        // It returns { "total_invested": ..., "total_value": ..., "total_profit": ..., "total_return_pct": ... }
        // It does NOT return per-item details. 
        // For "Perfect" UI, we would want per-item details.
        // I will accept this limitation for now or fetch prices individually.
        // Given complexity, showing Total Performance is a good enough "Premium" feature.
    }
}
