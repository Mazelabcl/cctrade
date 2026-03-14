// TradingView Lightweight Charts v4.1.3 — Levels start at birth candle
document.addEventListener('DOMContentLoaded', function() {
    const container = document.getElementById('chart-container');
    if (!container || typeof LightweightCharts === 'undefined') return;

    // State
    let currentTf = '1d';
    let currentCount = 500;
    let showFractals = true;
    let showPredictions = true;
    let enabledLevels = { Fractal: true, HTF: true, Fib: true, VP: true };
    let enabledSourceTfs = { daily: true, weekly: true, monthly: true };
    let enabledStatus = { naked: true, touched: true };
    let levelSeriesList = [];   // line series for levels (start at birth)
    let lastChartData = [];     // cache for level redraws

    const MAX_LEVEL_SERIES = 300;  // perf cap

    // Create chart
    const chart = LightweightCharts.createChart(container, {
        layout: { background: { color: '#1a1a2e' }, textColor: '#d1d4dc' },
        grid: {
            vertLines: { color: 'rgba(42, 46, 57, 0.3)' },
            horzLines: { color: 'rgba(42, 46, 57, 0.3)' },
        },
        crosshair: { mode: 0 },
        timeScale: { borderColor: '#2a2e39', timeVisible: true, secondsVisible: false },
        rightPriceScale: { borderColor: '#2a2e39' },
    });

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

    // Short labels for display
    const tfShort = {
        hourly: 'H', '4hourly': '4H', daily: 'D', weekly: 'W', monthly: 'M',
    };

    function getLevelCategory(levelType) {
        if (levelType.startsWith('Fractal')) return 'Fractal';
        if (levelType.startsWith('HTF')) return 'HTF';
        if (levelType.startsWith('Fib')) return 'Fib';
        if (levelType.startsWith('VP') || levelType.startsWith('vp')) return 'VP';
        return 'HTF';
    }

    function isNaked(level) {
        const totalTouches = (level.support_touches || 0) + (level.resistance_touches || 0);
        return totalTouches < 1;
    }

    function clearLevelLines() {
        levelSeriesList.forEach(s => {
            try { chart.removeSeries(s); } catch(e) {}
        });
        levelSeriesList = [];
    }

    function buildLevelQueryParams() {
        const params = new URLSearchParams({ active_only: 'true' });
        const tfs = Object.entries(enabledSourceTfs)
            .filter(([_, on]) => on)
            .map(([tf]) => tf);
        if (tfs.length > 0 && tfs.length < 3) {
            params.set('timeframe', tfs.join(','));
        }
        if (enabledStatus.naked && !enabledStatus.touched) {
            params.set('naked_only', 'true');
        }
        return params.toString();
    }

    function loadData() {
        const url = `/api/candles?tf=${currentTf}&limit=${currentCount}`;
        fetch(url)
            .then(r => r.json())
            .then(data => {
                if (!data.length) {
                    candleSeries.setData([]);
                    clearLevelLines();
                    const countEl = document.getElementById('level-count');
                    if (countEl) countEl.textContent = 'No candle data for this timeframe';
                    return;
                }

                lastChartData = data.map(c => ({
                    time: Math.floor(new Date(c.open_time).getTime() / 1000),
                    open: c.open, high: c.high, low: c.low, close: c.close,
                }));
                candleSeries.setData(lastChartData);

                // Build markers
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

        if (!candleData.length || !lastChartData.length) return;
        const prices = candleData.flatMap(c => [c.high, c.low]);
        const minPrice = Math.min(...prices);
        const maxPrice = Math.max(...prices);
        const margin = (maxPrice - minPrice) * 0.1;

        const firstTime = lastChartData[0].time;
        const lastTime = lastChartData[lastChartData.length - 1].time;

        const queryParams = buildLevelQueryParams();
        fetch(`/api/levels?${queryParams}`)
            .then(r => r.json())
            .then(levels => {
                const seen = new Set();
                const roundTo = maxPrice > 10000 ? 10 : 1;
                let visibleCount = 0;
                let capped = false;

                // Sort by created_at so most recent levels are drawn last (on top)
                levels.sort((a, b) => new Date(a.created_at) - new Date(b.created_at));

                levels.forEach(l => {
                    if (visibleCount >= MAX_LEVEL_SERIES) { capped = true; return; }
                    if (l.price_level < minPrice - margin || l.price_level > maxPrice + margin) return;

                    const cat = getLevelCategory(l.level_type);
                    if (!enabledLevels[cat]) return;

                    // Source TF filter
                    if (!enabledSourceTfs[l.timeframe]) return;

                    // Status filter
                    const naked = isNaked(l);
                    if (naked && !enabledStatus.naked) return;
                    if (!naked && !enabledStatus.touched) return;

                    // Dedup by category + rounded price + timeframe
                    const rounded = Math.round(l.price_level / roundTo) * roundTo;
                    const key = `${cat}-${l.timeframe}-${rounded}`;
                    if (seen.has(key)) return;
                    seen.add(key);

                    // Birth time — where the level line starts
                    const birthTime = Math.floor(new Date(l.created_at).getTime() / 1000);
                    // Clamp: if born before chart range, start at chart start
                    const startTime = Math.max(birthTime, firstTime);

                    // Build label
                    const tfLabel = tfShort[l.timeframe] || l.timeframe;
                    const typeName = l.level_type.replace('_level', '').replace('VP_', 'VP ');
                    const title = `${typeName} ${tfLabel}`;

                    // Style
                    const lineStyle = naked ? 0 : 2;  // 0=solid, 2=dashed
                    const lineWidth = naked ? 2 : 1;
                    const color = levelColors[cat] || '#888';

                    // End time: naked levels extend to chart end, touched levels end at first touch
                    let endTime = lastTime;
                    if (!naked && l.first_touched_at) {
                        endTime = Math.floor(new Date(l.first_touched_at).getTime() / 1000);
                        // Clamp: don't end before start
                        if (endTime <= startTime) endTime = startTime + 86400;
                    }

                    // Skip levels that ended before the visible chart range
                    if (endTime < firstTime) return;

                    // Create a line series that starts at birth and extends to end
                    const series = chart.addLineSeries({
                        color: color,
                        lineWidth: lineWidth,
                        lineStyle: lineStyle,
                        lastValueVisible: naked,
                        priceLineVisible: false,
                        crosshairMarkerVisible: false,
                        title: title,
                    });

                    series.setData([
                        { time: startTime, value: l.price_level },
                        { time: endTime, value: l.price_level },
                    ]);

                    levelSeriesList.push(series);
                    visibleCount++;
                });

                // Update level count
                const countEl = document.getElementById('level-count');
                if (countEl) {
                    let text = `${visibleCount} levels shown`;
                    if (capped) text += ` (capped at ${MAX_LEVEL_SERIES}, use filters)`;
                    countEl.textContent = text;
                }
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

    // Level type toggle buttons
    document.querySelectorAll('#level-toggles .btn').forEach(btn => {
        btn.addEventListener('click', function() {
            this.classList.toggle('active');
            enabledLevels[this.dataset.level] = this.classList.contains('active');
            loadData();
        });
    });

    // Source timeframe toggle buttons
    document.querySelectorAll('#source-tf-toggles .btn').forEach(btn => {
        btn.addEventListener('click', function() {
            this.classList.toggle('active');
            enabledSourceTfs[this.dataset.stf] = this.classList.contains('active');
            loadData();
        });
    });

    // Status toggle buttons (naked / touched)
    document.querySelectorAll('#status-toggles .btn').forEach(btn => {
        btn.addEventListener('click', function() {
            this.classList.toggle('active');
            enabledStatus[this.dataset.status] = this.classList.contains('active');
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
