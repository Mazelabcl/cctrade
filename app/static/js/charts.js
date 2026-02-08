// TradingView Lightweight Charts initialization
// Will be fully implemented in Phase 4
document.addEventListener('DOMContentLoaded', function() {
    const container = document.getElementById('chart-container');
    if (!container) return;

    if (typeof LightweightCharts === 'undefined') {
        container.innerHTML = '<p class="text-muted p-4">Chart library loading... Refresh if this persists.</p>';
        return;
    }

    const chart = LightweightCharts.createChart(container, {
        layout: {
            background: { color: '#1a1a2e' },
            textColor: '#d1d4dc',
        },
        grid: {
            vertLines: { color: 'rgba(42, 46, 57, 0.5)' },
            horzLines: { color: 'rgba(42, 46, 57, 0.5)' },
        },
        timeScale: {
            borderColor: '#2a2e39',
            timeVisible: true,
        },
    });

    const candleSeries = chart.addSeries(LightweightCharts.CandlestickSeries, {
        upColor: '#26a69a',
        downColor: '#ef5350',
        borderVisible: false,
        wickUpColor: '#26a69a',
        wickDownColor: '#ef5350',
    });

    // Load candle data from API
    fetch('/api/candles?tf=1h&limit=500')
        .then(r => r.json())
        .then(data => {
            if (data.length === 0) {
                container.innerHTML = '<p class="text-muted p-4">No candle data available. Import data first.</p>';
                return;
            }
            const chartData = data.map(c => ({
                time: Math.floor(new Date(c.open_time).getTime() / 1000),
                open: c.open,
                high: c.high,
                low: c.low,
                close: c.close,
            }));
            candleSeries.setData(chartData);
            chart.timeScale().fitContent();
        })
        .catch(err => {
            container.innerHTML = '<p class="text-muted p-4">Failed to load chart data.</p>';
            console.error(err);
        });

    // Responsive resize
    new ResizeObserver(entries => {
        const { width, height } = entries[0].contentRect;
        chart.applyOptions({ width, height });
    }).observe(container);
});
