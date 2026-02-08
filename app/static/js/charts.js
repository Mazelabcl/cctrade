// TradingView Lightweight Charts — full implementation
document.addEventListener('DOMContentLoaded', function() {
    const container = document.getElementById('chart-container');
    if (!container || typeof LightweightCharts === 'undefined') return;

    // State
    let currentTf = '1h';
    let currentCount = 500;
    let showFractals = true;
    let enabledLevels = { Fractal: true, HTF: true, Fib: true, VP: true };
    let levelLines = [];

    // Create chart
    const chart = LightweightCharts.createChart(container, {
        layout: { background: { color: '#1a1a2e' }, textColor: '#d1d4dc' },
        grid: {
            vertLines: { color: 'rgba(42, 46, 57, 0.3)' },
            horzLines: { color: 'rgba(42, 46, 57, 0.3)' },
        },
        crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
        timeScale: { borderColor: '#2a2e39', timeVisible: true, secondsVisible: false },
        rightPriceScale: { borderColor: '#2a2e39' },
    });

    const candleSeries = chart.addSeries(LightweightCharts.CandlestickSeries, {
        upColor: '#26a69a', downColor: '#ef5350',
        borderVisible: false,
        wickUpColor: '#26a69a', wickDownColor: '#ef5350',
    });

    // Level color mapping
    const levelColors = {
        Fractal: '#ef5350',
        HTF: '#42a5f5',
        Fib: '#ffd54f',
        VP: '#ab47bc',
    };

    function getLevelCategory(levelType) {
        if (levelType.startsWith('Fractal')) return 'Fractal';
        if (levelType.startsWith('HTF')) return 'HTF';
        if (levelType.startsWith('Fib')) return 'Fib';
        if (levelType.startsWith('VP') || levelType.startsWith('vp')) return 'VP';
        return 'HTF';
    }

    function clearLevelLines() {
        levelLines.forEach(line => {
            try { candleSeries.removePriceLine(line); } catch(e) {}
        });
        levelLines = [];
    }

    function loadData() {
        const url = `/api/candles?tf=${currentTf}&limit=${currentCount}`;
        fetch(url)
            .then(r => r.json())
            .then(data => {
                if (!data.length) {
                    container.innerHTML = '<p class="text-muted p-4">No candle data for this timeframe.</p>';
                    return;
                }

                const chartData = data.map(c => ({
                    time: Math.floor(new Date(c.open_time).getTime() / 1000),
                    open: c.open, high: c.high, low: c.low, close: c.close,
                }));
                candleSeries.setData(chartData);

                // Fractal markers
                if (showFractals) {
                    const markers = [];
                    data.forEach(c => {
                        const t = Math.floor(new Date(c.open_time).getTime() / 1000);
                        if (c.bullish_fractal) {
                            markers.push({
                                time: t, position: 'belowBar', color: '#26a69a',
                                shape: 'arrowUp', text: '',
                            });
                        }
                        if (c.bearish_fractal) {
                            markers.push({
                                time: t, position: 'aboveBar', color: '#ef5350',
                                shape: 'arrowDown', text: '',
                            });
                        }
                    });
                    candleSeries.setMarkers(markers.sort((a, b) => a.time - b.time));
                } else {
                    candleSeries.setMarkers([]);
                }

                chart.timeScale().fitContent();
                loadLevels(data);
            })
            .catch(err => console.error('Failed to load candles:', err));
    }

    function loadLevels(candleData) {
        clearLevelLines();

        // Get price range from visible candles to filter levels
        if (!candleData.length) return;
        const prices = candleData.flatMap(c => [c.high, c.low]);
        const minPrice = Math.min(...prices);
        const maxPrice = Math.max(...prices);
        const margin = (maxPrice - minPrice) * 0.1;

        fetch(`/api/levels?active_only=true`)
            .then(r => r.json())
            .then(levels => {
                // Filter to visible price range and deduplicate by rounding
                const seen = new Set();
                const roundTo = maxPrice > 10000 ? 10 : 1;

                levels.forEach(l => {
                    if (l.price_level < minPrice - margin || l.price_level > maxPrice + margin) return;

                    const cat = getLevelCategory(l.level_type);
                    if (!enabledLevels[cat]) return;

                    const rounded = Math.round(l.price_level / roundTo) * roundTo;
                    const key = `${cat}-${rounded}`;
                    if (seen.has(key)) return;
                    seen.add(key);

                    const totalTouches = (l.support_touches || 0) + (l.resistance_touches || 0);
                    const opacity = Math.max(0.2, 1 - totalTouches * 0.15);

                    const color = levelColors[cat] || '#888';
                    const line = candleSeries.createPriceLine({
                        price: l.price_level,
                        color: color,
                        lineWidth: 1,
                        lineStyle: LightweightCharts.LineStyle.Dotted,
                        axisLabelVisible: false,
                        title: '',
                    });
                    levelLines.push(line);
                });
            })
            .catch(err => console.error('Failed to load levels:', err));
    }

    // Timeframe buttons
    document.querySelectorAll('#tf-selector .btn').forEach(btn => {
        btn.addEventListener('click', function() {
            document.querySelectorAll('#tf-selector .btn').forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            currentTf = this.dataset.tf;
            loadData();
        });
    });

    // Count buttons
    document.querySelectorAll('#count-selector .btn').forEach(btn => {
        btn.addEventListener('click', function() {
            document.querySelectorAll('#count-selector .btn').forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            currentCount = parseInt(this.dataset.count);
            loadData();
        });
    });

    // Level toggle buttons
    document.querySelectorAll('#level-toggles .btn').forEach(btn => {
        btn.addEventListener('click', function() {
            this.classList.toggle('active');
            enabledLevels[this.dataset.level] = this.classList.contains('active');
            loadData();
        });
    });

    // Fractal marker toggle
    document.getElementById('btn-fractals').addEventListener('click', function() {
        showFractals = !showFractals;
        this.classList.toggle('active', showFractals);
        loadData();
    });
    document.getElementById('btn-fractals').classList.add('active');

    // Responsive resize
    new ResizeObserver(entries => {
        const { width, height } = entries[0].contentRect;
        chart.applyOptions({ width, height });
    }).observe(container);

    // Initial load
    loadData();
});
