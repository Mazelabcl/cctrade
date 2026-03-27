#!/usr/bin/env python
"""AutoResearch Agent — automated experiment loop for exit optimization.

Mutates configuration, runs fast backtest, evaluates if improved, logs results.
Inspired by Karpathy's autoresearch concept.

Usage:
    python scripts/autoresearch/agent.py --experiments 50
    python scripts/autoresearch/agent.py --experiments 100 --mode exit_optimization
"""
import sys
import os
import time
import json
import copy
import random
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import yaml


def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path) as f:
        return yaml.safe_load(f)


def save_config(config, config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


MUTATIONS_EXIT = [
    {
        'name': 'change_strategy',
        'field': 'exit.strategy',
        'options': ['swing_trail', 'breakeven_trail', 'atr_trail', 'partial'],
    },
    {
        'name': 'swing_lookback',
        'field': 'exit.swing_lookback',
        'range': [2, 15],
        'type': 'int',
    },
    {
        'name': 'atr_multiplier',
        'field': 'exit.atr_multiplier',
        'range': [0.5, 4.0],
        'step': 0.25,
    },
    {
        'name': 'breakeven_rr',
        'field': 'exit.breakeven_at_rr',
        'range': [0.5, 3.0],
        'step': 0.25,
    },
    {
        'name': 'partial_pct',
        'field': 'exit.partial_pct',
        'range': [0.2, 0.8],
        'step': 0.1,
    },
    {
        'name': 'partial_rr',
        'field': 'exit.partial_rr',
        'range': [1.0, 5.0],
        'step': 0.5,
    },
    {
        'name': 'timeout_candles',
        'field': 'exit.timeout_candles',
        'range': [50, 1000],
        'step': 50,
        'type': 'int',
    },
    {
        'name': 'add_level_type',
        'field': 'levels.types',
        'action': 'add',
        'pool': [
            'Fractal_support', 'Fractal_resistance', 'HTF_level', 'Fib_CC',
            'PrevSession_High', 'PrevSession_Low', 'PrevSession_VWAP',
            'PrevSession_VP_VAH', 'PrevSession_VP_VAL',
        ],
    },
    {
        'name': 'remove_level_type',
        'field': 'levels.types',
        'action': 'remove',
        'min_items': 1,
    },
    {
        'name': 'change_timeframe',
        'field': 'execution.timeframe',
        'options': ['15m', '30m', '1h', '4h'],
    },
]


def get_nested(d, key_path):
    keys = key_path.split('.')
    for k in keys:
        d = d[k]
    return d


def set_nested(d, key_path, value):
    keys = key_path.split('.')
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def propose_mutation(config, history=None):
    """Propose a random mutation to the config."""
    # Avoid repeating recent failed mutations
    recent_fails = set()
    if history:
        for h in history[-10:]:
            if not h.get('improved'):
                recent_fails.add(h['mutation']['name'])

    # Pick a mutation type
    candidates = [m for m in MUTATIONS_EXIT if m['name'] not in recent_fails]
    if not candidates:
        candidates = MUTATIONS_EXIT

    mut_def = random.choice(candidates)
    mutation = {'name': mut_def['name'], 'field': mut_def['field']}

    if 'options' in mut_def:
        current = get_nested(config, mut_def['field'])
        options = [o for o in mut_def['options'] if o != current]
        new_val = random.choice(options) if options else current
        mutation['old'] = current
        mutation['new'] = new_val
        mutation['description'] = f"{mut_def['name']}: {current} -> {new_val}"

    elif 'range' in mut_def:
        current = get_nested(config, mut_def['field'])
        lo, hi = mut_def['range']
        step = mut_def.get('step', 1)
        delta = random.choice([-2, -1, 1, 2]) * step
        new_val = max(lo, min(hi, current + delta))
        if mut_def.get('type') == 'int':
            new_val = int(new_val)
        else:
            new_val = round(new_val, 2)
        mutation['old'] = current
        mutation['new'] = new_val
        mutation['description'] = f"{mut_def['name']}: {current} -> {new_val}"

    elif mut_def.get('action') == 'add':
        current = get_nested(config, mut_def['field'])
        pool = [t for t in mut_def['pool'] if t not in current]
        if pool:
            to_add = random.choice(pool)
            mutation['new'] = to_add
            mutation['description'] = f"add {to_add} to level_types"
        else:
            mutation['description'] = "no types to add (skip)"
            mutation['skip'] = True

    elif mut_def.get('action') == 'remove':
        current = get_nested(config, mut_def['field'])
        min_items = mut_def.get('min_items', 1)
        if len(current) > min_items:
            to_remove = random.choice(current)
            mutation['new'] = to_remove
            mutation['description'] = f"remove {to_remove} from level_types"
        else:
            mutation['description'] = "can't remove (min 1)"
            mutation['skip'] = True

    return mutation


def apply_mutation(config, mutation):
    """Apply a mutation to a config copy. Returns new config."""
    new_config = copy.deepcopy(config)

    if mutation.get('skip'):
        return new_config

    field = mutation['field']

    if mutation['name'] == 'add_level_type':
        types = get_nested(new_config, field)
        types.append(mutation['new'])

    elif mutation['name'] == 'remove_level_type':
        types = get_nested(new_config, field)
        types.remove(mutation['new'])

    elif 'new' in mutation:
        set_nested(new_config, field, mutation['new'])

    return new_config


def run_experiments(n_experiments=50, config_path=None):
    """Main AutoResearch loop."""
    from evaluate import evaluate
    from app import create_app
    from app.extensions import db

    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, 'experiments.jsonl')

    app = create_app()
    with app.app_context():
        # Baseline evaluation
        config = load_config(config_path)
        print("=" * 70, flush=True)
        print("AUTORESEARCH — Exit Optimization", flush=True)
        print("=" * 70, flush=True)
        print(f"Running {n_experiments} experiments...", flush=True)
        print(flush=True)

        print("Evaluating baseline...", flush=True)
        t0 = time.time()
        baseline = evaluate(config, db.session)
        print(f"Baseline ({time.time()-t0:.1f}s): total_r={baseline['total_r']}, "
              f"PF={baseline['profit_factor']}, WR={baseline['win_rate']}%, "
              f"trades={baseline['total_trades']}", flush=True)
        print(flush=True)

        history = []
        best = baseline.copy()
        best_config = copy.deepcopy(config)
        improvements = 0

        for i in range(n_experiments):
            config_current = load_config(config_path)
            mutation = propose_mutation(config_current, history)

            if mutation.get('skip'):
                print(f"[{i+1}/{n_experiments}] SKIP: {mutation['description']}", flush=True)
                continue

            new_config = apply_mutation(config_current, mutation)

            t0 = time.time()
            metrics = evaluate(new_config, db.session)
            elapsed = time.time() - t0

            improved = metrics.get('total_r', 0) > best.get('total_r', 0)

            experiment = {
                'id': i,
                'timestamp': datetime.now().isoformat(),
                'mutation': mutation,
                'metrics': metrics,
                'improved': improved,
                'best_total_r': best['total_r'],
                'elapsed_sec': round(elapsed, 1),
            }
            history.append(experiment)

            # Log to JSONL
            with open(results_file, 'a') as f:
                f.write(json.dumps(experiment, default=str) + '\n')

            if improved:
                improvements += 1
                best = metrics
                best_config = copy.deepcopy(new_config)
                save_config(new_config, config_path)
                print(f"[{i+1}/{n_experiments}] ** IMPROVED ** {mutation['description']} "
                      f"-> total_r={metrics['total_r']} PF={metrics['profit_factor']} "
                      f"({elapsed:.1f}s)", flush=True)
            else:
                print(f"[{i+1}/{n_experiments}] no gain: {mutation['description']} "
                      f"-> total_r={metrics.get('total_r', 0)} ({elapsed:.1f}s)", flush=True)

        # Summary
        print(flush=True)
        print("=" * 70, flush=True)
        print("AUTORESEARCH SUMMARY", flush=True)
        print("=" * 70, flush=True)
        print(f"Experiments: {n_experiments}", flush=True)
        print(f"Improvements: {improvements}", flush=True)
        print(f"Baseline total_r: {baseline['total_r']}", flush=True)
        print(f"Best total_r: {best['total_r']}", flush=True)
        print(f"Improvement: {best['total_r'] - baseline['total_r']:.1f}R "
              f"({(best['total_r'] / max(baseline['total_r'], 1) - 1) * 100:.1f}%)", flush=True)
        print(flush=True)
        print("Best config:", flush=True)
        for k, v in best.items():
            print(f"  {k}: {v}", flush=True)
        print(flush=True)
        print(f"Results logged to: {results_file}", flush=True)
        print(f"Best config saved to: {config_path or 'config.yaml'}", flush=True)

        return best, best_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AutoResearch Agent')
    parser.add_argument('--experiments', type=int, default=50, help='Number of experiments')
    parser.add_argument('--config', type=str, default=None, help='Path to config YAML')
    args = parser.parse_args()

    run_experiments(n_experiments=args.experiments, config_path=args.config)
