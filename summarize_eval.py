"""Summarizes the seed-run evaluation (evaluate.py CSVs of launch_eval_seeds.sh).

Pipelines are named <cfg>-s<seed>-<criterion>[+2x1] (cfg: ep, tr, cur, rep;
criterion: mean or 400-500), plus the phase 1 checkpoints and the greedy /
random baselines. Prints, per range, the mean apx-ratio of each config and
criterion as mean ± std over the 3 seeds (raw and with 2x1), then the
baselines and the phase 1 checkpoints. Baselines appear once per CSV; the
copies are identical, so the first is kept.

    python summarize_eval.py experiments/eval_seeds/g*.csv [--ranges 10_11 ...]
"""
import argparse
import re

import pandas as pd

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument('csv', nargs='+')
parser.add_argument('--ranges', nargs='*', default=None)
parser.add_argument('--seeds_table', action='store_true',
                    help='also print the per-seed values.')

CFG = {'ep': 'episode', 'tr': 'transition', 'cur': 'current', 'rep': 'replace'}
SEED = re.compile(r'(ep|tr|cur|rep)-s(\d)-(mean|400-500)(\+2x1)?$')


def range_key(r):
    return int(r.split('_')[0])


if __name__ == '__main__':
    args = parser.parse_args()
    frames = []
    for f in args.csv:
        d = pd.read_csv(f)
        if frames:  # baselines are in every file
            d = d[~d.pipeline.str.match(r'^(greedy|random)(\+2x1)?$')]
        frames.append(d)
    df = pd.concat(frames)
    if args.ranges:
        df = df[df.range.isin(args.ranges)]
    ranges = sorted(df.range.unique(), key=range_key)
    per_pipe = df.groupby(['pipeline', 'range']).apx_ratio.mean().unstack()[ranges]

    rows = []
    for pipe, vals in per_pipe.iterrows():
        m = SEED.match(pipe)
        if m:
            rows.append(dict(cfg=CFG[m[1]], seed=int(m[2]), crit=m[3],
                             ls='+2x1' if m[4] else 'raw', **vals))
    seeds = pd.DataFrame(rows)
    pd.set_option('display.width', 250)
    for ls in ('raw', '+2x1'):
        s = seeds[seeds.ls == ls]
        g = s.groupby(['crit', 'cfg'])[ranges]
        table = g.mean().round(4).astype(str) + ' ± ' + g.std().round(4).astype(str)
        print(f'\n== {ls}: mean apx-ratio, mean ± std over seeds')
        print(table.to_string())
        if args.seeds_table:
            print(s.set_index(['crit', 'cfg', 'seed'])[ranges].sort_index().round(4).to_string())
    other = per_pipe[[not SEED.match(p) for p in per_pipe.index]]
    print('\n== baselines and phase 1 checkpoints')
    print(other.round(4).to_string())
