// TradingView Lightweight Charts v4.1.3
document.addEventListener('DOMContentLoaded', function() {
    const container = document.getElementById('chart-container');
    if (!container || typeof LightweightCharts === 'undefined') return;

    // State
    let currentTf = '1h';
    let currentCount = 500;
    let showFractals = true;
    let showPredictions = true;
    let enabledLevels = { Fractal: true, HTF: true, Fib: true, VP: true };
    let levelLines = [];

    // Create chart
    const chart = LightweightCharts.createChart(container, {
        layout: { background: { color: '#1a1a2e' }, textColor: '#d1d4dc' },
        grid: {
            vertLines: { color: 'rgba(42, 46, 57, 0.3)' },
            horzLines: { color: 'rgba(42, 46, 57, 0.3)' },
        },
        crosshair: { mode: 0 },  // Normal
        timeScale: { borderColor: '#2a2e39', timeVisible: true, secondsVisible: false },
        rightPriceScale: { borderColor: '#2a2e39' },
    });

    // v4.1.3 API: addCandlestickSeries
    const candleSeries = chart.addCandlestickSeries({
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

                // Build markers from fractals and predictions
                const markers = [];

                if (showFractals) {
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
                }

                if (showPredictions) {
                    loadPredictions(markers);
                } else {
                    candleSeries.setMarkers(markers.sort((a, b) => a.time - b.time));
                }

                chart.timeScale().fitContent();
                loadLevels(data);
            })
            .catch(err => console.error('Failed to load candles:', err));
    }

    function loadPredictions(existingMarkers) {
        fetch(`/api/predictions/overlay?limit=${currentCount}`)
            .then(r => r.json())
            .then(preds => {
                const markers = [...existingMarkers];
                preds.forEach(p => {
                    const t = Math.floor(new Date(p.time).getTime() / 1000);
                    if (p.predicted_class === 1) {
                        markers.push({
                            time: t, position: 'belowBar',
                            color: 'rgba(38, 166, 154, 0.5)',
                            shape: 'circle', text: 'P',
                        });
                    } else if (p.predicted_class === 2) {
                        markers.push({
                            time: t, position: 'aboveBar',
                            color: 'rgba(239, 83, 80, 0.5)',
                            shape: 'circle', text: 'P',
                        });
                    }
                });
                candleSeries.setMarkers(markers.sort((a, b) => a.time - b.time));
            })
            .catch(err => {
                console.error('Failed to load predictions:', err);
                candleSeries.setMarkers(existingMarkers.sort((a, b) => a.time - b.time));
            });
    }

    function loadLevels(candleData) {
        clearLevelLines();

        if (!candleData.length) return;
        const prices = candleData.flatMap(c => [c.high, c.low]);
        const minPrice = Math.min(...prices);
        const maxPrice = Math.max(...prices);
        const margin = (maxPrice - minPrice) * 0.1;

        fetch(`/api/levels?active_only=true`)
            .then(r => r.json())
            .then(levels => {
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

                    const line = candleSeries.createPriceLine({
                        price: l.price_level,
                        color: levelColors[cat] || '#888',
                        lineWidth: 1,
                        lineStyle: 2,  // Dotted
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
    const fractalBtn = document.getElementById('btn-fractals');
    if (fractalBtn) {
        fractalBtn.addEventListener('click', function() {
            showFractals = !showFractals;
            this.classList.toggle('active', showFractals);
            loadData();
        });
        fractalBtn.classList.add('active');
    }

    // Prediction marker toggle
    const predBtn = document.getElementById('btn-predictions');
    if (predBtn) {
        predBtn.addEventListener('click', function() {
            showPredictions = !showPredictions;
            this.classList.toggle('active', showPredictions);
            loadData();
        });
        predBtn.classList.add('active');
    }

    // Responsive resize
    new ResizeObserver(entries => {
        const { width, height } = entries[0].contentRect;
        chart.applyOptions({ width, height });
    }).observe(container);

    // Initial load
    loadData();
});
