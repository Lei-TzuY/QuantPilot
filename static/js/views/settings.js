import { API } from '../api.js';

export default class SettingsView {
    async getHtml() {
        return `
            <div class="card">
                <h2><i class="fa-solid fa-gear"></i> Settings</h2>
                <p style="color: var(--text-muted);">Configure QuantPilot platform settings</p>
            </div>

            <div class="card">
                <h2><i class="fa-brands fa-discord"></i> / <i class="fa-brands fa-slack"></i> Webhook Notifications</h2>
                <p style="color: var(--text-muted); margin-bottom: 1rem;">Receive alerts and backtest results via Discord or Slack.</p>
                
                <div id="webhook-status" style="margin-bottom: 1.5rem; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 8px;">
                    Loading configuration...
                </div>

                <form id="webhook-form">
                    <div style="margin-bottom: 1rem;">
                        <label style="display: flex; align-items: center; gap: 0.5rem; cursor: pointer;">
                            <input type="checkbox" id="webhook-enabled" style="width: auto;">
                            <span>Enable Webhook Notifications</span>
                        </label>
                    </div>
                    
                    <div style="margin-bottom: 1rem;">
                        <label><i class="fa-brands fa-discord"></i> Discord Webhook URL</label>
                        <input type="text" id="discord-webhook" placeholder="https://discord.com/api/webhooks/...">
                        <small style="color: var(--text-muted);">Get this from Discord Server Settings → Integrations → Webhooks</small>
                    </div>
                    
                    <div style="margin-bottom: 1rem;">
                        <label><i class="fa-brands fa-slack"></i> Slack Webhook URL</label>
                        <input type="text" id="slack-webhook" placeholder="https://hooks.slack.com/services/...">
                        <small style="color: var(--text-muted);">Create one at api.slack.com/apps → Incoming Webhooks</small>
                    </div>
                    
                    <div style="display: flex; gap: 1rem;">
                        <button type="submit" class="btn" style="flex: 1;">Save Configuration</button>
                        <button type="button" class="btn btn-secondary" id="test-discord">Test Discord</button>
                        <button type="button" class="btn btn-secondary" id="test-slack">Test Slack</button>
                    </div>
                </form>
                
                <div id="webhook-message" style="margin-top: 1rem; display: none;"></div>
            </div>

            <div class="card">
                <h2><i class="fa-solid fa-calculator"></i> Default Trading Parameters</h2>
                <p style="color: var(--text-muted); margin-bottom: 1rem;">Set default values for backtesting (Taiwan market defaults shown).</p>
                
                <div class="grid-2">
                    <div>
                        <label>Commission Rate (%)</label>
                        <input type="number" id="commission-rate" value="0.1425" step="0.01" min="0">
                        <small style="color: var(--text-muted);">Default: 0.1425% per trade</small>
                    </div>
                    <div>
                        <label>Transaction Tax (%)</label>
                        <input type="number" id="tax-rate" value="0.3" step="0.01" min="0">
                        <small style="color: var(--text-muted);">Default: 0.3% on sell (Taiwan stocks)</small>
                    </div>
                    <div>
                        <label>Slippage (%)</label>
                        <input type="number" id="slippage-rate" value="0.1" step="0.01" min="0">
                        <small style="color: var(--text-muted);">Default: 0.1% price slippage</small>
                    </div>
                    <div>
                        <label>Default Position Size (%)</label>
                        <input type="number" id="position-size" value="100" step="1" min="1" max="100">
                        <small style="color: var(--text-muted);">Percentage of capital to allocate</small>
                    </div>
                </div>
                
                <button class="btn" style="margin-top: 1rem;" id="save-trading-params">Save Trading Defaults</button>
            </div>
        `;
    }

    async init() {
        await this.loadWebhookConfig();

        // Webhook form submit
        document.getElementById('webhook-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            await this.saveWebhookConfig();
        });

        // Test buttons
        document.getElementById('test-discord').addEventListener('click', () => this.testWebhook('discord'));
        document.getElementById('test-slack').addEventListener('click', () => this.testWebhook('slack'));

        // Trading params save
        document.getElementById('save-trading-params').addEventListener('click', () => this.saveTradingParams());

        // Load saved trading params
        this.loadTradingParams();
    }

    async loadWebhookConfig() {
        try {
            const response = await API.get('/api/webhook/config');
            if (response.success) {
                const config = response.config;
                document.getElementById('webhook-enabled').checked = config.enabled;

                const statusDiv = document.getElementById('webhook-status');
                statusDiv.innerHTML = `
                    <div style="display: flex; gap: 2rem;">
                        <div>
                            <span style="color: var(--text-muted);">Discord:</span>
                            <span style="color: ${config.discord_configured ? 'var(--success)' : 'var(--text-muted)'};">
                                ${config.discord_configured ? '✓ Configured' : '✗ Not configured'}
                            </span>
                        </div>
                        <div>
                            <span style="color: var(--text-muted);">Slack:</span>
                            <span style="color: ${config.slack_configured ? 'var(--success)' : 'var(--text-muted)'};">
                                ${config.slack_configured ? '✓ Configured' : '✗ Not configured'}
                            </span>
                        </div>
                        <div>
                            <span style="color: var(--text-muted);">Status:</span>
                            <span style="color: ${config.enabled ? 'var(--success)' : 'var(--danger)'};">
                                ${config.enabled ? 'Enabled' : 'Disabled'}
                            </span>
                        </div>
                    </div>
                `;
            }
        } catch (err) {
            console.error('Failed to load webhook config', err);
        }
    }

    async saveWebhookConfig() {
        const messageDiv = document.getElementById('webhook-message');

        try {
            const payload = {
                enabled: document.getElementById('webhook-enabled').checked
            };

            const discordUrl = document.getElementById('discord-webhook').value.trim();
            const slackUrl = document.getElementById('slack-webhook').value.trim();

            if (discordUrl) payload.discord_webhook = discordUrl;
            if (slackUrl) payload.slack_webhook = slackUrl;

            const response = await API.post('/api/webhook/config', payload);

            if (response.success) {
                this.showMessage(messageDiv, 'Configuration saved successfully!', 'success');
                await this.loadWebhookConfig();

                // Clear the input fields for security
                document.getElementById('discord-webhook').value = '';
                document.getElementById('slack-webhook').value = '';
            } else {
                this.showMessage(messageDiv, 'Failed to save: ' + response.error, 'error');
            }
        } catch (err) {
            this.showMessage(messageDiv, 'Error saving configuration', 'error');
        }
    }

    async testWebhook(platform) {
        const messageDiv = document.getElementById('webhook-message');

        try {
            this.showMessage(messageDiv, `Sending test to ${platform}...`, 'info');

            const response = await API.post('/api/webhook/test', { platform });

            if (response.success) {
                this.showMessage(messageDiv, `✓ Test message sent to ${platform}!`, 'success');
            } else {
                this.showMessage(messageDiv, `✗ Test failed. Make sure ${platform} is configured.`, 'error');
            }
        } catch (err) {
            this.showMessage(messageDiv, `Error testing ${platform}`, 'error');
        }
    }

    showMessage(div, text, type) {
        const colors = {
            success: 'var(--success)',
            error: 'var(--danger)',
            info: 'var(--primary)'
        };
        div.style.display = 'block';
        div.style.color = colors[type] || 'var(--text-muted)';
        div.textContent = text;

        if (type !== 'info') {
            setTimeout(() => { div.style.display = 'none'; }, 5000);
        }
    }

    loadTradingParams() {
        const saved = localStorage.getItem('tradingParams');
        if (saved) {
            try {
                const params = JSON.parse(saved);
                if (params.commission) document.getElementById('commission-rate').value = params.commission * 100;
                if (params.tax) document.getElementById('tax-rate').value = params.tax * 100;
                if (params.slippage) document.getElementById('slippage-rate').value = params.slippage * 100;
                if (params.positionSize) document.getElementById('position-size').value = params.positionSize * 100;
            } catch { }
        }
    }

    saveTradingParams() {
        const params = {
            commission: parseFloat(document.getElementById('commission-rate').value) / 100,
            tax: parseFloat(document.getElementById('tax-rate').value) / 100,
            slippage: parseFloat(document.getElementById('slippage-rate').value) / 100,
            positionSize: parseFloat(document.getElementById('position-size').value) / 100
        };
        localStorage.setItem('tradingParams', JSON.stringify(params));
        alert('Trading parameters saved!');
    }
}
