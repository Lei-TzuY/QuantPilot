import { API } from '../api.js';

export default class DashboardView {
    async getHtml() {
        return `
            <div class="header">
                <h2>Market Overview</h2>
                <div class="date">${new Date().toLocaleDateString()}</div>
            </div>
            
            <div class="grid-3" id="popular-stocks-grid">
                <!-- Popular Stocks injected here -->
                <div class="card">Loading market data...</div>
            </div>

            <h3 style="margin-top: 2rem;">My Watchlist</h3>
            <div class="grid-3" id="watchlist-grid">
                <div class="card" style="opacity: 0.7;">No stocks in watchlist</div>
            </div>

            <h3 style="margin-top: 2rem;"><i class="fa-solid fa-bell"></i> Price Alerts</h3>
            <div class="card" style="margin-top: 1rem;">
                <div class="grid-4" style="gap: 0.5rem; align-items: end;">
                    <div>
                        <label>Symbol</label>
                        <input type="text" id="alert-symbol" placeholder="2330" style="width: 100%;">
                    </div>
                    <div>
                        <label>Condition</label>
                        <select id="alert-condition" style="width: 100%;">
                            <option value="above">Above</option>
                            <option value="below">Below</option>
                        </select>
                    </div>
                    <div>
                        <label>Price</label>
                        <input type="number" id="alert-target" placeholder="1000" style="width: 100%;">
                    </div>
                    <button class="btn" id="create-alert-btn" style="height: 40px;">
                        <i class="fa-solid fa-plus"></i> Add
                    </button>
                </div>
                <div id="alerts-list" style="margin-top: 1rem;"></div>
            </div>
        `;
    }

    async init() {
        try {
            // 1. Load Popular
            const data = await API.get('/api/popular');
            const popularContainer = document.getElementById('popular-stocks-grid');

            if (data.success && data.stocks) {
                popularContainer.innerHTML = data.stocks.map(stock => this.createStockCardHtml(stock)).join('');
                data.stocks.forEach(stock => this.updateStockCard(stock.symbol));
            }

            // 2. Load Watchlist
            const watchlist = JSON.parse(localStorage.getItem('watchlist') || '[]');
            const watchlistContainer = document.getElementById('watchlist-grid');

            if (watchlist.length > 0) {
                // Mock objects for watchlist (we don't have name/industry derived easily without API, 
                // but we can just show Symbol for now or fetch info)
                // For MVP, just show Symbol
                watchlistContainer.innerHTML = watchlist.map(symbol => this.createStockCardHtml({
                    symbol: symbol,
                    name: "Watchlist Item",
                    industry: "Watched"
                })).join('');

                watchlist.forEach(symbol => this.updateStockCard(symbol));
            } else {
                watchlistContainer.innerHTML = '<p style="grid-column: 1/-1; color: var(--text-muted);">Star stocks in Analysis view to add them here.</p>';
            }

            // 3. Load Alerts
            this.loadAlerts();
            document.getElementById('create-alert-btn').addEventListener('click', () => this.createAlert());

        } catch (error) {
            console.error(error);
        }
    }

    async loadAlerts() {
        try {
            const response = await API.get('/api/alerts');
            const container = document.getElementById('alerts-list');
            if (response.success && response.alerts.length > 0) {
                container.innerHTML = response.alerts.map(a => `
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem; border-bottom: 1px solid var(--border-light);">
                        <span><strong>${a.symbol}</strong> ${a.condition} <strong>${a.target_value}</strong></span>
                        <button class="btn-secondary" style="padding: 4px 8px; font-size: 0.8rem;" onclick="window.app.views.dashboard.deleteAlert('${a.id}')">
                            <i class="fa-solid fa-trash"></i>
                        </button>
                    </div>
                `).join('');
            } else {
                container.innerHTML = '<p style="color: var(--text-muted);">No active alerts.</p>';
            }
        } catch { }
    }

    async createAlert() {
        const symbol = document.getElementById('alert-symbol').value;
        const condition = document.getElementById('alert-condition').value;
        const target_value = parseFloat(document.getElementById('alert-target').value);
        if (!symbol || !target_value) { alert('Please fill all fields'); return; }
        try {
            await API.post('/api/alerts', { symbol, condition, target_value });
            this.loadAlerts();
            document.getElementById('alert-symbol').value = '';
            document.getElementById('alert-target').value = '';
        } catch { alert('Failed to create alert'); }
    }

    async deleteAlert(id) {
        try {
            await API.delete(`/api/alerts/${id}`);
            this.loadAlerts();
        } catch { }
    }

    createStockCardHtml(stock) {
        return `
            <div class="card hover-effect" id="card-${stock.symbol}" onclick="window.app.navigate('analysis'); setTimeout(() => window.app.views.analysis.loadStock('${stock.symbol}'), 100);">
                <div class="flex-between">
                    <h3 style="color: var(--primary)">${stock.symbol}</h3>
                    <span class="badge">${stock.industry}</span>
                </div>
                <p style="font-size: 1.1rem; font-weight: 600; margin-top: 0.5rem;">${stock.name}</p>
                <div class="price-info" style="margin-top: 1rem; font-size: 1.2rem; font-weight: bold;">
                    Loading...
                </div>
            </div>
        `;
    }

    async updateStockCard(symbol) {
        try {
            // We use the same realtime endpoint
            const response = await API.get(`/api/stock/${symbol}/realtime`);
            if (response.success) {
                const card = document.getElementById(`card-${symbol}`);
                if (card) {
                    const priceDiv = card.querySelector('.price-info');
                    // For now, realtime endpoint only returns price, not change. 
                    // To show change, we might need a better endpoint or just show price.
                    // For "WOW" factor, let's just show Price in big bold text.
                    // Ideally we'd get change % too.
                    priceDiv.innerHTML = `${response.price.toFixed(2)}`;
                    priceDiv.style.color = 'var(--text-main)';
                }
            }
        } catch (e) {
            console.warn(`Failed to update dashboard card for ${symbol}`);
        }
    }
}

