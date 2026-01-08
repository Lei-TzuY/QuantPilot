import DashboardView from './views/dashboard.js';
import AnalysisView from './views/analysis.js';
import BacktestView from './views/backtest.js';
import PortfolioView from './views/portfolio.js';
import BatchView from './views/batch.js';
import AlertsView from './views/alerts.js';
import DocsView from './views/docs.js';
import SettingsView from './views/settings.js';

class App {
    constructor() {
        this.currentView = null;
        this.views = {
            dashboard: new DashboardView(),
            analysis: new AnalysisView(),
            backtest: new BacktestView(),
            portfolio: new PortfolioView(),
            batch: new BatchView(),
            alerts: new AlertsView(),
            docs: new DocsView(),
            settings: new SettingsView(),
        };

        this.init();
    }


    init() {
        // Navigation Handling
        document.querySelectorAll('.nav-item').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const viewName = e.currentTarget.getAttribute('data-view');
                this.navigate(viewName);
            });
        });

        // Load default view
        this.navigate('dashboard');
    }

    async navigate(viewName) {
        // Update Nav UI
        document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
        document.querySelector(`.nav-item[data-view="${viewName}"]`)?.classList.add('active');

        // Clean up current view
        const content = document.getElementById('content');
        content.innerHTML = '<div style="text-align:center; padding: 2rem; color: #94a3b8;">Loading...</div>';

        // Load new view
        if (this.views[viewName]) {
            html = await this.views[viewName].getHtml(); // This line is buggy in standard JS if variable not declared?
            // Wait, checking my syntax. 
            // `html` implicitly global or error in module? 
            // Better declare it.
            const htmlContent = await this.views[viewName].getHtml();
            content.innerHTML = htmlContent;

            // Execute view logic after render
            if (this.views[viewName].init) {
                await this.views[viewName].init();
            }
        } else {
            content.innerHTML = `<h2>Page Not Found</h2>`;
        }
    }
}

// Initialize App
document.addEventListener('DOMContentLoaded', () => {
    window.app = new App();
});
