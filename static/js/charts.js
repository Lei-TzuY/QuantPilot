export class Charts {
    static createLineChart(ctx, label, labels, dataOrDatasets, color) {
        let datasets = [];

        // Check if dataOrDatasets is an array of dataset objects (custom structure) or raw numbers
        if (Array.isArray(dataOrDatasets) && dataOrDatasets.length > 0 && typeof dataOrDatasets[0] === 'object') {
            datasets = dataOrDatasets;
        } else {
            // Legacy single line mode
            datasets = [{
                label: label,
                data: dataOrDatasets,
                borderColor: color,
                backgroundColor: color + '20',
                borderWidth: 2,
                pointRadius: 0,
                pointHoverRadius: 4,
                fill: true,
                tension: 0.3
            }];
        }

        return new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: true, labels: { color: '#94a3b8' } },
                    tooltip: { mode: 'index', intersect: false }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(148, 163, 184, 0.1)' },
                        ticks: { color: '#94a3b8' }
                    },
                    y: {
                        grid: { color: 'rgba(148, 163, 184, 0.1)' },
                        ticks: { color: '#94a3b8' }
                    }
                },
                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                }
            }
        });
    }

    static createCandlestickChart() {
        // Chart.js standard doesn't support candlestick well without plugins.
        // For simplicity in this "perfect" MVP, we stick to line charts for Close price
        // or we could use 'chartjs-chart-financial' if we had a build step.
        // We will stick to Line Chart for 'Price' for now.
        console.warn("Candlestick not implemented in simple Chart.js wrapper. Using Line.");
    }

    static destroy(chartInstance) {
        if (chartInstance) {
            chartInstance.destroy();
        }
    }
}
