// TradingView Lightweight Charts v4.1.3 — Levels start at birth candle
document.addEventListener('DOMContentLoaded', function() {
    const container = document.getElementById('chart-container');
    if (!container || typeof LightweightCharts === 'undefined') return;

    // Read URL params for pre-set state (e.g., /charts/?tf=1d&count=1000)
    const urlParams = new URLSearchParams(window.location.search);

    // State
    let currentTf = urlParams.get('tf') || '1d';
    let currentCount = parseInt(urlParams.get('count')) || 500;
    let showFractals = true;
    let showPredictions = true;
    let viewMode = 'levels';    // 'levels', 'trades', 'all'
    let selectedTradeIdx = 'all'; // 'all' or trade index
    let tradeConfig = 'fractals_fib_cc'; // API config preset
    let enabledLevels = { SFP: true, HTF: true, CC: true, Igor: true, VP: true };
    let enabledSourceTfs = { daily: true, weekly: true, monthly: true };
    let enabledStatus = { naked: true, touched: true };
    let levelSeriesList = [];   // line series for levels (start at birth)
    let tradeSeriesList = [];   // line series for trade SL/entry lines
    let lastChartData = [];     // cache for level redraws
    let allTrades = [];         // cached forward trades

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

    // Short labels for display
    const tfShort = {
        hourly: 'H', '4hourly': '4H', daily: 'D', weekly: 'W', monthly: 'M',
    };

    // Chart Champions color mapping by HTF timeframe
    const htfColors = { daily: '#00e5ff', weekly: '#ffd54f', monthly: '#ab47bc' };

    function getLevelCategory(levelType) {
        if (levelType.startsWith('Fractal')) return 'SFP';
        if (levelType.startsWith('HTF')) return 'HTF';
        if (levelType === 'Fib_CC') return 'CC';
        if (levelType.startsWith('Fib')) return 'Igor';
        if (levelType.startsWith('VP') || levelType.startsWith('vp')) return 'VP';
        return 'HTF';
    }

    function getLevelStyle(level) {
        const lt = level.level_type;
        const tf = level.timeframe;

        // HTF — color by timeframe
        if (lt === 'HTF_level') return { color: htfColors[tf] || '#42a5f5', lineWidth: 1, lineStyle: 0 };

        // SFP (Fractal levels)
        if (lt.startsWith('Fractal')) return { color: '#e0e0e0', lineWidth: 1, lineStyle: 0 };

        // CC — Daniel's golden pocket (dotted yellow)
        if (lt === 'Fib_CC') return { color: '#ffd54f', lineWidth: 1, lineStyle: 1 };

        // Igor quarters
        if (lt === 'Fib_0.50') return { color: '#ffd54f', lineWidth: 1, lineStyle: 0 };
        if (lt === 'Fib_0.25' || lt === 'Fib_0.75') return { color: '#ef5350', lineWidth: 1, lineStyle: 0 };

        // VP — POC thicker red, VAH/VAL blue
        if (lt === 'VP_POC') return { color: '#ef5350', lineWidth: 2, lineStyle: 0 };
        if (lt === 'VP_VAH' || lt === 'VP_VAL') return { color: '#42a5f5', lineWidth: 2, lineStyle: 0 };

        // Old Fib types (pre-migration) — fallback gray
        if (lt.startsWith('Fib')) return { color: '#888', lineWidth: 1, lineStyle: 0 };

        return { color: '#888', lineWidth: 1, lineStyle: 0 };
    }

    function getLevelLabel(level) {
        const lt = level.level_type;
        const tf = tfShort[level.timeframe] || level.timeframe;

        if (lt === 'HTF_level') return `HTF ${tf}`;
        if (lt.startsWith('Fractal')) return 'SFP';
        if (lt === 'Fib_CC') return `CC ${tf}`;
        if (lt === 'Fib_0.25') return `Igor .25 ${tf}`;
        if (lt === 'Fib_0.50') return `Igor .50 ${tf}`;
        if (lt === 'Fib_0.75') return `Igor .75 ${tf}`;
        if (lt === 'VP_POC') return `POC ${tf}`;
        if (lt === 'VP_VAH') return `VAH ${tf}`;
        if (lt === 'VP_VAL') return `VAL ${tf}`;

        return `${lt} ${tf}`;
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

    function buildLevelQueryParams(forTrade) {
        // In trade mode, get ALL levels (including invalidated) so we can filter by trade time
        const params = new URLSearchParams({ active_only: forTrade ? 'false' : 'true' });
        const tfs = Object.entries(enabledSourceTfs)
            .filter(([_, on]) => on)
            .map(([tf]) => tf);
        if (tfs.length > 0 && tfs.length < 3) {
            params.set('timeframe', tfs.join(','));
        }
        if (!forTrade && enabledStatus.naked && !enabledStatus.touched) {
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
                    window._baseMarkers = markers;
                    const tradeM = (viewMode === 'trades' || viewMode === 'all') ? (window._tradeMarkers || []) : [];
                    candleSeries.setMarkers([...markers, ...tradeM].sort((a, b) => a.time - b.time));
                }

                chart.timeScale().fitContent();

                // Load trades first (so we know which trade is selected for level filtering)
                if (viewMode === 'trades' || viewMode === 'all') {
                    loadTrades();
                } else {
                    clearTradeLines();
                    window._tradeMarkers = [];
                }

                // Load levels (filtered by trade time if a specific trade is selected)
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
                window._baseMarkers = markers;
                const tradeM = (viewMode === 'trades' || viewMode === 'all') ? (window._tradeMarkers || []) : [];
                candleSeries.setMarkers([...markers, ...tradeM].sort((a, b) => a.time - b.time));
            })
            .catch(err => {
                console.error('Failed to load predictions:', err);
                window._baseMarkers = existingMarkers;
                candleSeries.setMarkers(existingMarkers.sort((a, b) => a.time - b.time));
            });
    }

    function clearTradeLines() {
        tradeSeriesList.forEach(s => chart.removeSeries(s));
        tradeSeriesList = [];
    }

    // 15m candle duration in seconds
    const TF_SECONDS = { '1h': 3600, '4h': 14400, '1d': 86400, '1w': 604800, '15m': 900 };
    function barDuration() { return TF_SECONDS[currentTf] || 900; }

    function loadTrades() {
        clearTradeLines();

        const renderTrades = (trades) => {
            const tradeMarkers = [];
            const fiveBars = barDuration() * 5;

            // Update trade info text
            const infoEl = document.getElementById('trade-info');

            trades.forEach(t => {
                const entryTime = Math.floor(new Date(t.entry_time).getTime() / 1000);
                const exitTime = Math.floor(new Date(t.exit_time).getTime() / 1000);
                const lineEnd = entryTime + fiveBars;
                const won = t.pnl_r > 0;
                const isLong = t.direction === 'LONG';

                // Entry marker (arrow)
                tradeMarkers.push({
                    time: entryTime,
                    position: isLong ? 'belowBar' : 'aboveBar',
                    color: '#ffffff',
                    shape: isLong ? 'arrowUp' : 'arrowDown',
                    text: `${t.direction} ${t.pnl_r > 0 ? '+' : ''}${t.pnl_r}R`,
                });

                // 1) Entry line — WHITE thick, 5 bars
                const entrySeries = chart.addLineSeries({
                    color: '#ffffff', lineWidth: 3, lineStyle: 0,
                    priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false,
                });
                entrySeries.setData([
                    { time: entryTime, value: t.entry_price },
                    { time: lineEnd, value: t.entry_price },
                ]);
                tradeSeriesList.push(entrySeries);

                // 2) SL line — RED thick, 5 bars
                const slSeries = chart.addLineSeries({
                    color: '#ff1744', lineWidth: 3, lineStyle: 0,
                    priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false,
                });
                slSeries.setData([
                    { time: entryTime, value: t.stop_loss },
                    { time: lineEnd, value: t.stop_loss },
                ]);
                tradeSeriesList.push(slSeries);

                // 3) Exit line — GREEN thick, 5 bars at exit price
                const exitSeries = chart.addLineSeries({
                    color: won ? '#00e676' : '#ff5252', lineWidth: 3, lineStyle: 0,
                    priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false,
                });
                const exitLineStart = Math.max(entryTime, exitTime - fiveBars);
                exitSeries.setData([
                    { time: exitLineStart, value: t.exit_price },
                    { time: exitTime + fiveBars, value: t.exit_price },
                ]);
                tradeSeriesList.push(exitSeries);

                // Exit marker
                tradeMarkers.push({
                    time: exitTime,
                    position: won ? 'aboveBar' : 'belowBar',
                    color: won ? '#00e676' : '#ff1744',
                    shape: 'circle',
                    text: `Exit: ${t.exit_reason} ${t.pnl_r > 0 ? '+' : ''}${t.pnl_r}R`,
                });

                // Show trade info
                if (infoEl && trades.length === 1) {
                    infoEl.innerHTML = `<b>${t.direction}</b> @ $${t.entry_price.toLocaleString()} | ` +
                        `SL: $${t.stop_loss.toLocaleString()} (${t.risk_pct || '?'}%) | ` +
                        `Exit: $${t.exit_price.toLocaleString()} <b>(${t.exit_reason})</b> | ` +
                        `P&L: <span style="color:${won ? '#00e676' : '#ff1744'}">${t.pnl_r > 0 ? '+' : ''}${t.pnl_r}R</span> | ` +
                        `Levels: ${(t.zone_levels || []).slice(0, 5).join(', ')}`;
                } else if (infoEl) {
                    infoEl.innerHTML = '';
                }
            });

            window._tradeMarkers = tradeMarkers;
            const base = window._baseMarkers || [];
            candleSeries.setMarkers([...base, ...tradeMarkers].sort((a, b) => a.time - b.time));
        };

        // Fetch trades if not cached
        if (allTrades.length > 0) {
            const filtered = filterTrades(allTrades);
            renderTrades(filtered);
            updateTradeStats(filtered);
            // Reload levels now that we know which trade is selected
            if (selectedTradeIdx !== 'all' && lastChartData.length) loadLevels(lastChartData);
            return;
        }

        fetch(`/api/forward-trades?config=${tradeConfig}`)
            .then(r => r.json())
            .then(trades => {
                if (!Array.isArray(trades)) return;
                allTrades = trades;
                if (!document.querySelector('#trade-selector option[value="1"]')) {
                    populateTradeSelector(trades);
                }
                const filtered = filterTrades(trades);
                renderTrades(filtered);
                updateTradeStats(filtered);
                // Now reload levels with trade filter active
                if (selectedTradeIdx !== 'all' && lastChartData.length) loadLevels(lastChartData);
            })
            .catch(err => console.error('Failed to load trades:', err));
    }

    function filterTrades(trades) {
        if (selectedTradeIdx === 'all') return trades;
        const idx = parseInt(selectedTradeIdx);
        return isNaN(idx) ? trades : [trades[idx]];
    }

    function populateTradeSelector(trades) {
        const sel = document.getElementById('trade-selector');
        if (!sel) return;
        const currentVal = selectedTradeIdx;  // Preserve selection
        sel.innerHTML = '<option value="all">All trades</option>';
        trades.forEach((t, i) => {
            const won = t.pnl_r > 0;
            const icon = won ? '+' : '';
            const label = `#${i+1} ${t.direction} ${t.entry_price.toFixed(0)} | ${icon}${t.pnl_r}R | ${t.exit_reason}`;
            const opt = document.createElement('option');
            opt.value = i;
            opt.textContent = label;
            opt.style.color = won ? '#00e676' : '#ff5252';
            sel.appendChild(opt);
        });
        // Restore previous selection
        if (currentVal !== 'all') {
            sel.value = currentVal;
        }
    }

    function updateTradeStats(trades) {
        const el = document.getElementById('trade-stats');
        if (!el || !trades.length) return;
        const wins = trades.filter(t => t.pnl_r > 0).length;
        const totalR = trades.reduce((s, t) => s + t.pnl_r, 0);
        el.textContent = `${trades.length} trades | ${wins}W ${trades.length - wins}L | ${totalR > 0 ? '+' : ''}${totalR.toFixed(1)}R`;
    }

    function loadLevels(candleData) {
        clearLevelLines();

        // In Trades mode with a SPECIFIC trade selected, show only that trade's levels
        // In Trades mode with 'all' selected, skip levels (too cluttered)
        if (viewMode === 'trades' && selectedTradeIdx === 'all') return;

        if (!candleData.length || !lastChartData.length) return;
        const prices = candleData.flatMap(c => [c.high, c.low]);
        const minPrice = Math.min(...prices);
        const maxPrice = Math.max(...prices);

        // When a specific trade is selected, filter levels to only those valid at trade time
        let tradeTimeFilter = null;
        if (selectedTradeIdx !== 'all' && allTrades.length > 0) {
            const idx = parseInt(selectedTradeIdx);
            if (!isNaN(idx) && allTrades[idx]) {
                tradeTimeFilter = Math.floor(new Date(allTrades[idx].entry_time).getTime() / 1000);
            }
        }
        const margin = (maxPrice - minPrice) * 0.1;

        const firstTime = lastChartData[0].time;
        const lastTime = lastChartData[lastChartData.length - 1].time;

        const queryParams = buildLevelQueryParams(!!tradeTimeFilter);
        fetch(`/api/levels?${queryParams}`)
            .then(r => r.json())
            .then(levels => {
                const seen = new Set();
                const roundTo = maxPrice > 10000 ? 10 : 1;
                let visibleCount = 0;
                let capped = false;

                // Sort: when filtering by trade, sort by proximity to trade price
                // Otherwise sort by created_at descending (newest first)
                if (tradeTimeFilter && selectedTradeIdx !== 'all' && allTrades.length > 0) {
                    const tp = allTrades[parseInt(selectedTradeIdx)]?.entry_price || 0;
                    levels.sort((a, b) => Math.abs(a.price_level - tp) - Math.abs(b.price_level - tp));
                } else {
                    levels.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
                }

                levels.forEach(l => {
                    if (visibleCount >= MAX_LEVEL_SERIES) { capped = true; return; }
                    // Skip price range filter when trade filter is active (trade range filter handles it)
                    if (!tradeTimeFilter && (l.price_level < minPrice - margin || l.price_level > maxPrice + margin)) return;

                    const cat = getLevelCategory(l.level_type);
                    if (!enabledLevels[cat]) { _dbg.catOut++; return; }

                    // Source TF filter
                    if (!enabledSourceTfs[l.timeframe]) return;

                    // Status filter (in trade mode, show all - time filter handles it)
                    const naked = isNaked(l);
                    if (!tradeTimeFilter) {
                        if (naked && !enabledStatus.naked) return;
                        if (!naked && !enabledStatus.touched) return;
                    }

                    // Birth time — where the level line starts
                    const birthTime = Math.floor(new Date(l.created_at).getTime() / 1000);

                    // End time: structural levels end at first_touched_at, mobile at superseded_at
                    let endTime = lastTime;
                    if (l.first_touched_at) {
                        endTime = Math.floor(new Date(l.first_touched_at).getTime() / 1000);
                    }
                    if (l.superseded_at) {
                        // Mobile levels (PrevSession, VP) end when superseded by next one
                        const supersededTime = Math.floor(new Date(l.superseded_at).getTime() / 1000);
                        endTime = Math.min(endTime, supersededTime);
                    }

                    // Trade time filter: only show levels valid at trade time + near trade price
                    if (tradeTimeFilter) {
                        if (birthTime > tradeTimeFilter) return;
                        // Skip if already touched/superseded before trade
                        if (l.first_touched_at) {
                            const touchTime = Math.floor(new Date(l.first_touched_at).getTime() / 1000);
                            if (touchTime < tradeTimeFilter) return;
                        }
                        if (l.superseded_at) {
                            const supTime = Math.floor(new Date(l.superseded_at).getTime() / 1000);
                            if (supTime < tradeTimeFilter) return;
                        }
                        const tradeIdx = parseInt(selectedTradeIdx);
                        if (!isNaN(tradeIdx) && allTrades[tradeIdx]) {
                            const tradePrice = allTrades[tradeIdx].entry_price;
                            if (l.price_level < tradePrice * 0.95 || l.price_level > tradePrice * 1.05) return;
                        }
                    }

                    // Skip levels whose entire life is outside the visible chart range
                    if (endTime < firstTime) return;
                    if (birthTime > lastTime) return;

                    // Clamp start/end to chart range
                    const startTime = Math.max(birthTime, firstTime);
                    if (endTime <= startTime) endTime = startTime + 86400;

                    // Dedup by category + rounded price + timeframe
                    const rounded = Math.round(l.price_level / roundTo) * roundTo;
                    const key = `${cat}-${l.timeframe}-${rounded}`;
                    if (seen.has(key)) return;
                    seen.add(key);

                    // Build label and style from CC methodology
                    const title = getLevelLabel(l);
                    const style = getLevelStyle(l);
                    const color = style.color;
                    const lineWidth = naked ? style.lineWidth : 1;
                    const lineStyle = naked ? style.lineStyle : 2;  // dashed for touched

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

    // View mode toggles
    document.querySelectorAll('#view-mode .btn').forEach(btn => {
        btn.addEventListener('click', function() {
            viewMode = this.dataset.view;
            document.querySelectorAll('#view-mode .btn').forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            // Show/hide trade controls
            const tsg = document.getElementById('trade-selector-group');
            const tcg = document.getElementById('trade-config-group');
            if (tsg) tsg.style.display = (viewMode === 'trades') ? 'flex' : 'none';
            if (tcg) tcg.style.display = (viewMode === 'trades') ? 'flex' : 'none';
            loadData();
        });
    });

    // Trade selector
    // Strategy config selector
    const tradeConfigSel = document.getElementById('trade-config');
    if (tradeConfigSel) {
        tradeConfigSel.addEventListener('change', function() {
            tradeConfig = this.value;
            allTrades = [];  // Force re-fetch
            selectedTradeIdx = 'all';
            _forward_trades_cache = {};
            loadData();
        });
    }

    const tradeSel = document.getElementById('trade-selector');
    if (tradeSel) {
        tradeSel.addEventListener('change', function() {
            selectedTradeIdx = this.value;
            // Lightweight re-render: just trade lines + filtered levels
            loadTrades();  // Uses cached allTrades, no API call
            if (lastChartData.length) {
                // Small delay to ensure trades are rendered first
                setTimeout(() => loadLevels(lastChartData), 100);
            }
        });
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

    // Sync button active states with URL params
    document.querySelectorAll('#tf-selector .btn').forEach(b => {
        b.classList.toggle('active', b.dataset.tf === currentTf);
    });
    document.querySelectorAll('#count-selector .btn').forEach(b => {
        b.classList.toggle('active', parseInt(b.dataset.count) === currentCount);
    });

    // Responsive resize
    new ResizeObserver(entries => {
        const { width, height } = entries[0].contentRect;
        chart.applyOptions({ width, height });
    }).observe(container);

    // Initial load
    loadData();
});
