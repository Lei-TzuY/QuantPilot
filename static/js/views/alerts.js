import { API } from '../api.js';

export default class AlertsView {
    async getHtml() {
        return `
            <div class="grid-2">
                <div class="card">
                    <h2>Create Alert</h2>
                    <form id="alert-form">
                        <div style="margin-bottom: 1rem;">
                            <label>Stock Symbol</label>
                            <input type="text" name="symbol" placeholder="e.g. 2330" required>
                        </div>
                        <div style="margin-bottom: 1rem;">
                            <label>Condition Type</label>
                            <select name="condition" id="condition-type">
                                <option value="above">Price Above</option>
                                <option value="below">Price Below</option>
                                <option value="volatility_spike">Volatility Spike (ATR)</option>
                                <option value="volume_surge">Volume Surge</option>
                                <option value="rsi_oversold">RSI Oversold (&lt;30)</option>
                                <option value="rsi_overbought">RSI Overbought (&gt;70)</option>
                            </select>
                        </div>
                        <div style="margin-bottom: 1rem;" id="target-value-group">
                            <label id="target-label">Target Price</label>
                            <input type="number" name="target_value" id="target-value" step="0.01" required>
                        </div>
                        <div style="margin-bottom: 1rem;">
                            <label>Note (Optional)</label>
                            <input type="text" name="note" placeholder="e.g. Breakout signal">
                        </div>
                        <button type="submit" class="btn" style="width:100%">Create Alert</button>
                    </form>
                </div>

                <div class="card">
                    <h2>Alert Statistics</h2>
                    <div id="alert-stats" style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
                        <div style="text-align: center; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 8px;">
                            <p style="color: var(--text-muted); font-size: 0.9rem;">Active Alerts</p>
                            <h3 id="active-count" style="color: var(--primary); font-size: 2rem;">0</h3>
                        </div>
                        <div style="text-align: center; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 8px;">
                            <p style="color: var(--text-muted); font-size: 0.9rem;">Triggered Today</p>
                            <h3 id="triggered-count" style="color: var(--success); font-size: 2rem;">0</h3>
                        </div>
                    </div>
                </div>
            </div>

            <div class="card">
                <div class="flex-between">
                    <h2>Active Alerts</h2>
                    <button class="btn btn-secondary" id="refresh-alerts"><i class="fa-solid fa-sync"></i> Refresh</button>
                </div>
                <div style="max-height: 350px; overflow-y: auto; margin-top: 1rem;">
                    <table style="width: 100%; text-align: left; border-collapse: collapse;">
                        <thead>
                            <tr style="border-bottom: 1px solid var(--border); color: var(--text-muted);">
                                <th style="padding: 0.5rem;">Symbol</th>
                                <th style="padding: 0.5rem;">Condition</th>
                                <th style="padding: 0.5rem;">Target</th>
                                <th style="padding: 0.5rem;">Note</th>
                                <th style="padding: 0.5rem;">Created</th>
                                <th style="padding: 0.5rem;">Action</th>
                            </tr>
                        </thead>
                        <tbody id="alerts-table-body">
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="card">
                <h2>Triggered Alerts</h2>
                <div style="max-height: 250px; overflow-y: auto; margin-top: 1rem;">
                    <table style="width: 100%; text-align: left; border-collapse: collapse;">
                        <thead>
                            <tr style="border-bottom: 1px solid var(--border); color: var(--text-muted);">
                                <th style="padding: 0.5rem;">Symbol</th>
                                <th style="padding: 0.5rem;">Condition</th>
                                <th style="padding: 0.5rem;">Target</th>
                                <th style="padding: 0.5rem;">Triggered At</th>
                                <th style="padding: 0.5rem;">Triggered Price</th>
                            </tr>
                        </thead>
                        <tbody id="triggered-table-body">
                        </tbody>
                    </table>
                </div>
            </div>
        `;
    }

    async init() {
        // Condition type change handler - update label
        const conditionSelect = document.getElementById('condition-type');
        const targetLabel = document.getElementById('target-label');
        const targetInput = document.getElementById('target-value');
        const targetGroup = document.getElementById('target-value-group');

        conditionSelect.addEventListener('change', () => {
            const val = conditionSelect.value;
            if (val === 'above' || val === 'below') {
                targetLabel.textContent = 'Target Price';
                targetInput.placeholder = '';
                targetGroup.style.display = 'block';
            } else if (val === 'volatility_spike') {
                targetLabel.textContent = 'ATR Multiplier (e.g. 2.0)';
                targetInput.placeholder = '2.0';
                targetInput.value = '2.0';
                targetGroup.style.display = 'block';
            } else if (val === 'volume_surge') {
                targetLabel.textContent = 'Volume Ratio (e.g. 2.0 = 2x avg)';
                targetInput.placeholder = '2.0';
                targetInput.value = '2.0';
                targetGroup.style.display = 'block';
            } else if (val === 'rsi_oversold' || val === 'rsi_overbought') {
                targetLabel.textContent = 'RSI Threshold';
                targetInput.value = val === 'rsi_oversold' ? '30' : '70';
                targetGroup.style.display = 'block';
            }
        });

        // Form submit
        document.getElementById('alert-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const payload = Object.fromEntries(formData.entries());
            payload.target_value = parseFloat(payload.target_value);

            try {
                const response = await API.post('/api/alerts', payload);
                if (response.success) {
                    e.target.reset();
                    this.loadAlerts();
                } else {
                    alert('Failed: ' + response.error);
                }
            } catch (err) {
                alert('Error creating alert');
                console.error(err);
            }
        });

        // Refresh button
        document.getElementById('refresh-alerts').addEventListener('click', () => this.loadAlerts());

        // Initial load
        this.loadAlerts();
    }

    async loadAlerts() {
        try {
            const [activeRes, triggeredRes] = await Promise.all([
                API.get('/api/alerts'),
                API.get('/api/alerts/triggered')
            ]);

            if (activeRes.success) {
                this.renderActiveAlerts(activeRes.alerts);
                document.getElementById('active-count').textContent = activeRes.alerts.length;
            }

            if (triggeredRes.success) {
                this.renderTriggeredAlerts(triggeredRes.alerts);
                document.getElementById('triggered-count').textContent = triggeredRes.alerts.length;
            }
        } catch (err) {
            console.error('Failed to load alerts', err);
        }
    }

    renderActiveAlerts(alerts) {
        const tbody = document.getElementById('alerts-table-body');
        if (alerts.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" style="padding: 1rem; text-align: center; color: var(--text-muted);">No active alerts</td></tr>';
            return;
        }

        const conditionLabels = {
            'above': 'Price Above',
            'below': 'Price Below',
            'volatility_spike': 'Volatility Spike',
            'volume_surge': 'Volume Surge',
            'rsi_oversold': 'RSI Oversold',
            'rsi_overbought': 'RSI Overbought'
        };

        tbody.innerHTML = alerts.map(a => `
            <tr>
                <td style="padding: 0.5rem; font-weight: bold;">${a.symbol}</td>
                <td style="padding: 0.5rem;">
                    <span style="background: rgba(88, 166, 255, 0.2); padding: 2px 8px; border-radius: 4px; font-size: 0.85rem;">
                        ${conditionLabels[a.condition] || a.condition}
                    </span>
                </td>
                <td style="padding: 0.5rem;">${a.target_value}</td>
                <td style="padding: 0.5rem; color: var(--text-muted);">${a.note || '-'}</td>
                <td style="padding: 0.5rem; color: var(--text-muted); font-size: 0.85rem;">${new Date(a.created_at).toLocaleDateString()}</td>
                <td style="padding: 0.5rem;">
                    <button class="btn-secondary" style="padding: 0.25rem 0.5rem; font-size: 0.8rem; background: var(--danger); color: white; border: none; border-radius: 4px; cursor: pointer;"
                            onclick="window.alertsView.deleteAlert('${a.id}')">Delete</button>
                </td>
            </tr>
        `).join('');
    }

    renderTriggeredAlerts(alerts) {
        const tbody = document.getElementById('triggered-table-body');
        if (alerts.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" style="padding: 1rem; text-align: center; color: var(--text-muted);">No triggered alerts</td></tr>';
            return;
        }

        tbody.innerHTML = alerts.map(a => `
            <tr style="background: rgba(63, 185, 80, 0.1);">
                <td style="padding: 0.5rem; font-weight: bold;">${a.symbol}</td>
                <td style="padding: 0.5rem;">${a.condition}</td>
                <td style="padding: 0.5rem;">${a.target_value}</td>
                <td style="padding: 0.5rem; color: var(--success);">${a.triggered_at ? new Date(a.triggered_at).toLocaleString() : '-'}</td>
                <td style="padding: 0.5rem; font-weight: bold;">${a.triggered_price || '-'}</td>
            </tr>
        `).join('');
    }

    async deleteAlert(alertId) {
        if (!confirm('Delete this alert?')) return;
        try {
            const response = await API.delete(`/api/alerts/${alertId}`);
            if (response.success) {
                this.loadAlerts();
            } else {
                alert('Failed to delete');
            }
        } catch (err) {
            alert('Error deleting alert');
        }
    }
}
