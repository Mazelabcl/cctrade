"""Exit Feature Analysis — What happens at the MAX candle?

For each Fractal trade in the MFE results, examines what was happening at the
candle where the trade reached its maximum favorable excursion (MFE). The goal
is to discover exit signals automatically.

Analyzes:
1. Level proximity — naked levels within 0.5% of max_price
2. Fractal formation — did a reversal fractal form within 5 candles after MAX?
3. Volume change — volume spike at MAX candle vs 20-candle average
4. Fibonacci extension — is max_price near a fib extension from entry?
5. Candle pattern — wick rejection ratio at MAX candle
6. Time in trade — candles since entry

Usage:
    python scripts/exit_feature_analysis.py [--min-rr 0.5]
"""
import sys
import os
import json
import argparse
from datetime import datetime, timedelta
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def p(msg):
    print(msg, flush=True)


def run_analysis(min_rr=0.5):
    from app import create_app
    from app.extensions import db
    from app.models import Candle, Level

    app = create_app()

    with app.app_context():
        # ------------------------------------------------------------------
        # Load MFE results, filter to Fractal trades with meaningful MFE
        # ------------------------------------------------------------------
        mfe_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'mfe_results_4h.json')
        with open(mfe_path) as f:
            all_trades = json.load(f)

        trades = [t for t in all_trades
                  if t['level_type'].startswith('Fractal')
                  and t['max_rr'] >= min_rr]

        p(f"Total MFE trades: {len(all_trades)}")
        p(f"Fractal trades with max_rr >= {min_rr}: {len(trades)}")
        if not trades:
            p("No trades to analyze.")
            return

        # ------------------------------------------------------------------
        # Load all 4h candles into a fast lookup
        # ------------------------------------------------------------------
        p("Loading 4h candles...")
        candles_q = (Candle.query
                     .filter_by(timeframe='4h', symbol='BTCUSDT')
                     .order_by(Candle.open_time)
                     .all())
        p(f"  Loaded {len(candles_q)} candles")

        # Build time->index map + arrays
        candle_by_time = {}
        candle_list = []
        for c in candles_q:
            candle_by_time[c.open_time] = len(candle_list)
            candle_list.append(c)

        # ------------------------------------------------------------------
        # Load all levels
        # ------------------------------------------------------------------
        p("Loading levels...")
        levels_q = Level.query.all()
        p(f"  Loaded {len(levels_q)} levels")

        # ------------------------------------------------------------------
        # Analyze each trade
        # ------------------------------------------------------------------
        p(f"\nAnalyzing {len(trades)} trades...\n")

        results = []
        stats = {
            'total': 0,
            'level_nearby': 0,
            'level_nearby_by_type': Counter(),
            'fractal_formed': 0,
            'volume_spike': 0,
            'fib_ext_hit': 0,
            'fib_ext_which': Counter(),
            'rejection_wick': 0,
            'candles_to_max_dist': [],
        }

        for i, trade in enumerate(trades):
            if i % 500 == 0 and i > 0:
                p(f"  Processed {i}/{len(trades)}...")

            entry_time = datetime.strptime(trade['entry_time'],
                                           '%Y-%m-%d %H:%M:%S')
            entry_price = trade['entry_price']
            direction = trade['direction']
            max_price = trade['max_price']
            max_rr = trade['max_rr']
            candles_to_max = trade['candles_to_max']
            risk = trade['risk']

            # Compute stop_loss from entry_price and risk
            if direction == 'LONG':
                stop_loss = entry_price - risk
            else:
                stop_loss = entry_price + risk

            # Compute max_candle_time: entry_time + candles_to_max * 4h
            max_candle_time = entry_time + timedelta(hours=4 * candles_to_max)

            # Find the MAX candle in our data
            max_idx = candle_by_time.get(max_candle_time)
            if max_idx is None:
                continue

            max_candle = candle_list[max_idx]
            stats['total'] += 1

            result = {
                'trade_id': trade['trade_id'],
                'direction': direction,
                'level_type': trade['level_type'],
                'entry_price': entry_price,
                'max_price': max_price,
                'max_rr': max_rr,
                'candles_to_max': candles_to_max,
                'max_candle_time': max_candle_time.isoformat(),
            }

            # ==============================================================
            # 1. Level proximity — naked levels within 0.5% of max_price
            # ==============================================================
            threshold = max_price * 0.005
            nearby_levels = []
            for lv in levels_q:
                # Level must exist before the max candle
                if lv.created_at > max_candle_time:
                    continue
                # Level must not be invalidated before max candle
                if lv.invalidated_at and lv.invalidated_at < max_candle_time:
                    continue
                # Check proximity
                if abs(lv.price_level - max_price) <= threshold:
                    # For LONG, we care about resistance levels above
                    # For SHORT, we care about support levels below
                    nearby_levels.append({
                        'type': lv.level_type,
                        'tf': lv.timeframe,
                        'price': lv.price_level,
                        'distance_pct': round(
                            (lv.price_level - max_price) / max_price * 100, 4),
                    })

            result['nearby_levels'] = len(nearby_levels)
            result['nearby_level_types'] = [l['type'] for l in nearby_levels]

            if nearby_levels:
                stats['level_nearby'] += 1
                for lv in nearby_levels:
                    stats['level_nearby_by_type'][lv['type']] += 1

            # ==============================================================
            # 2. Fractal formation — reversal fractal within 5 candles
            # ==============================================================
            fractal_formed = False
            for offset in range(1, 6):
                check_idx = max_idx + offset
                if check_idx >= len(candle_list):
                    break
                check_candle = candle_list[check_idx]
                if direction == 'LONG' and check_candle.bearish_fractal:
                    fractal_formed = True
                    result['fractal_offset'] = offset
                    break
                elif direction == 'SHORT' and check_candle.bullish_fractal:
                    fractal_formed = True
                    result['fractal_offset'] = offset
                    break

            result['fractal_formed'] = fractal_formed
            if fractal_formed:
                stats['fractal_formed'] += 1

            # ==============================================================
            # 3. Volume change — spike vs 20-candle average
            # ==============================================================
            vol_start = max(0, max_idx - 20)
            vol_window = [candle_list[j].volume for j in range(vol_start, max_idx)]
            avg_vol = sum(vol_window) / len(vol_window) if vol_window else 1
            max_vol = max_candle.volume
            vol_ratio = round(max_vol / avg_vol, 2) if avg_vol > 0 else 0

            result['volume_ratio'] = vol_ratio
            result['volume_spike'] = vol_ratio > 1.5
            if vol_ratio > 1.5:
                stats['volume_spike'] += 1

            # ==============================================================
            # 4. Fibonacci extension from entry
            # ==============================================================
            fib_ratios = [1.0, 1.618, 2.0, 2.618]
            risk_size = abs(entry_price - stop_loss)
            fib_hit = None
            for ratio in fib_ratios:
                if direction == 'LONG':
                    ext_price = entry_price + risk_size * ratio
                else:
                    ext_price = entry_price - risk_size * ratio
                # Within 0.3% of the extension level
                if abs(max_price - ext_price) / max_price < 0.003:
                    fib_hit = ratio
                    break

            result['fib_extension_hit'] = fib_hit
            if fib_hit:
                stats['fib_ext_hit'] += 1
                stats['fib_ext_which'][str(fib_hit)] += 1

            # ==============================================================
            # 5. Candle pattern — wick rejection ratio
            # ==============================================================
            candle_range = max_candle.high - max_candle.low
            if candle_range > 0:
                body_top = max(max_candle.open, max_candle.close)
                body_bottom = min(max_candle.open, max_candle.close)
                if direction == 'LONG':
                    # Upper wick = rejection for longs
                    upper_wick = max_candle.high - body_top
                    wick_ratio = round(upper_wick / candle_range, 3)
                else:
                    # Lower wick = rejection for shorts
                    lower_wick = body_bottom - max_candle.low
                    wick_ratio = round(lower_wick / candle_range, 3)
            else:
                wick_ratio = 0

            result['wick_ratio'] = wick_ratio
            result['rejection_wick'] = wick_ratio > 0.5
            if wick_ratio > 0.5:
                stats['rejection_wick'] += 1

            # ==============================================================
            # 6. Time in trade
            # ==============================================================
            result['candles_in_trade'] = candles_to_max
            stats['candles_to_max_dist'].append(candles_to_max)

            results.append(result)

        # ------------------------------------------------------------------
        # Print summary
        # ------------------------------------------------------------------
        total = stats['total']
        if total == 0:
            p("No trades matched candle data.")
            return

        p("=" * 60)
        p(f"EXIT FEATURE ANALYSIS — {total} Fractal trades (max_rr >= {min_rr})")
        p("=" * 60)

        pct = lambda n: f"{n}/{total} ({n/total*100:.1f}%)"

        p(f"\n1. LEVEL PROXIMITY (within 0.5% of max_price):")
        p(f"   Trades with nearby level: {pct(stats['level_nearby'])}")
        p(f"   Breakdown by level type:")
        for lt, cnt in stats['level_nearby_by_type'].most_common(15):
            p(f"     {lt}: {cnt}")

        p(f"\n2. FRACTAL FORMATION (reversal within 5 candles after MAX):")
        p(f"   Fractal formed: {pct(stats['fractal_formed'])}")
        if stats['fractal_formed'] > 0:
            offsets = [r['fractal_offset'] for r in results if r.get('fractal_formed')]
            avg_offset = sum(offsets) / len(offsets)
            p(f"   Avg candles to fractal: {avg_offset:.1f}")
            offset_dist = Counter(offsets)
            for k in sorted(offset_dist):
                p(f"     Offset {k}: {offset_dist[k]}")

        p(f"\n3. VOLUME SPIKE (>1.5x 20-candle avg):")
        p(f"   Volume spikes: {pct(stats['volume_spike'])}")
        all_ratios = [r['volume_ratio'] for r in results]
        if all_ratios:
            p(f"   Avg volume ratio: {sum(all_ratios)/len(all_ratios):.2f}")
            p(f"   Median volume ratio: {sorted(all_ratios)[len(all_ratios)//2]:.2f}")

        p(f"\n4. FIBONACCI EXTENSION (within 0.3% of extension level):")
        p(f"   Fib extension hit: {pct(stats['fib_ext_hit'])}")
        for ratio, cnt in stats['fib_ext_which'].most_common():
            p(f"     {ratio}x: {cnt}")

        p(f"\n5. CANDLE REJECTION WICK (>50% wick ratio):")
        p(f"   Rejection wicks: {pct(stats['rejection_wick'])}")
        all_wicks = [r['wick_ratio'] for r in results]
        if all_wicks:
            p(f"   Avg wick ratio: {sum(all_wicks)/len(all_wicks):.3f}")

        p(f"\n6. TIME IN TRADE (candles to MAX):")
        dist = stats['candles_to_max_dist']
        if dist:
            p(f"   Avg: {sum(dist)/len(dist):.1f} candles")
            p(f"   Median: {sorted(dist)[len(dist)//2]} candles")
            p(f"   Max: {max(dist)} candles")
            # Bucket distribution
            buckets = Counter()
            for d in dist:
                if d <= 1:
                    buckets['0-1'] += 1
                elif d <= 3:
                    buckets['2-3'] += 1
                elif d <= 6:
                    buckets['4-6'] += 1
                elif d <= 12:
                    buckets['7-12'] += 1
                elif d <= 24:
                    buckets['13-24'] += 1
                else:
                    buckets['25+'] += 1
            for bucket in ['0-1', '2-3', '4-6', '7-12', '13-24', '25+']:
                if bucket in buckets:
                    p(f"     {bucket} candles: {buckets[bucket]}")

        # ------------------------------------------------------------------
        # Combined signal analysis
        # ------------------------------------------------------------------
        p(f"\n{'=' * 60}")
        p("COMBINED SIGNAL ANALYSIS")
        p(f"{'=' * 60}")

        multi_signal = 0
        for r in results:
            signals = 0
            if r['nearby_levels'] > 0:
                signals += 1
            if r['fractal_formed']:
                signals += 1
            if r['volume_spike']:
                signals += 1
            if r['fib_extension_hit']:
                signals += 1
            if r['rejection_wick']:
                signals += 1
            r['signal_count'] = signals
            if signals >= 2:
                multi_signal += 1

        p(f"Trades with 2+ exit signals: {pct(multi_signal)}")
        signal_dist = Counter(r['signal_count'] for r in results)
        for k in sorted(signal_dist):
            p(f"  {k} signals: {signal_dist[k]} ({signal_dist[k]/total*100:.1f}%)")

        # ------------------------------------------------------------------
        # Save results
        # ------------------------------------------------------------------
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'exit_feature_analysis_results.json')
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        p(f"\nResults saved to {out_path}")
        p(f"Total analyzed: {total}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Exit Feature Analysis')
    parser.add_argument('--min-rr', type=float, default=0.5,
                        help='Minimum max_rr to include (default: 0.5)')
    args = parser.parse_args()
    run_analysis(min_rr=args.min_rr)
