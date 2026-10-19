"""Paired significance tests for the seed-run evaluation (evaluate.py CSVs).

Every pipeline is evaluated on the same graphs, so pipelines are compared graph
by graph on the cover size (vertices; negative difference = smaller cover):
- RL vs greedy, with the same local search: for each config / criterion / seed
  and range, the Wilcoxon signed-rank test on the per-graph difference
  RL - greedy (ties dropped), win / tie / loss counts and the mean difference,
  with a Holm correction over the whole table. Seed level: the mean of the 3
  per-seed mean differences, its 95% t-interval, and how many seeds are
  significant for / against RL. The interval uses n_seeds - 1 df.
- Config vs config, same criterion and local search: per graph, the cover
  averaged over the seeds, and the Wilcoxon test on the difference between
  two configs (Holm over the table). This tests the seed-averaged policies,
  not the training recipe (that needs enough seeds for the seed-level interval).

    python paired_tests.py experiments/eval_seeds/g*.csv [--out paired_tests.csv]
"""
import argparse
import itertools
import re

import numpy as np
import pandas as pd
from scipy import stats

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument('csv', nargs='+')
parser.add_argument('--out', default=None, help='prefix for the CSVs with every test.')
parser.add_argument('--alpha', type=float, default=0.05)

CFG = {'ep': 'episode', 'tr': 'transition', 'cur': 'current', 'rep': 'replace'}
SEED = re.compile(r'(ep|tr|cur|rep)-s(\d)-(mean|400-500)(\+2x1)?$')


def range_key(r):
    return int(r.split('_')[0])


def wilcoxon(d):
    d = d[d != 0]
    return stats.wilcoxon(d).pvalue if len(d) else 1.0


def holm(p):
    p = np.asarray(p, float)
    adj, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


if __name__ == '__main__':
    args = parser.parse_args()
    frames = []
    for f in args.csv:
        d = pd.read_csv(f)
        if frames:  # baselines are in every file
            d = d[~d.pipeline.str.match(r'^(greedy|random)(\+2x1)?$')]
        frames.append(d)
    df = pd.concat(frames)
    sizes = df.set_index(['range', 'graph', 'pipeline'])['size'].unstack()
    ref = df.groupby(['range', 'graph'])['ref'].first()
    ranges = sorted(df.range.unique(), key=range_key)
    runs = [(p, SEED.match(p)) for p in sizes.columns if SEED.match(p)]
    n_seeds = len({int(m[2]) for _, m in runs})

    # RL vs greedy, per seed
    rows = []
    for pipe, m in runs:
        ls = m[4] or ''
        for r in ranges:
            pair = sizes.loc[r, [pipe, 'greedy' + ls]].dropna()
            d = pair[pipe] - pair['greedy' + ls]
            rows.append(dict(ls=ls or 'raw', crit=m[3], cfg=CFG[m[1]], seed=int(m[2]), range=r,
                             n=len(d), wins=int((d < 0).sum()), ties=int((d == 0).sum()),
                             losses=int((d > 0).sum()), mean_diff=d.mean(),
                             ratio_diff=(d / ref.loc[r].loc[d.index]).mean(), p=wilcoxon(d.values)))
    vs = pd.DataFrame(rows)
    vs['p_holm'] = holm(vs.p)
    vs['sig'] = np.where(vs.p_holm < args.alpha, np.sign(-vs.mean_diff), 0).astype(int)

    def seed_level(g):
        x = g.mean_diff.values
        t975 = stats.t.ppf(0.975, len(x) - 1)
        return pd.Series(dict(mean=x.mean(), ci=t975 * x.std(ddof=1) / np.sqrt(len(x)),
                              n_for=int((g.sig > 0).sum()), n_against=int((g.sig < 0).sum()),
                              wins=g.wins.mean(), ties=g.ties.mean(), losses=g.losses.mean()))

    summary = (vs.groupby(['ls', 'crit', 'cfg', 'range'])[['mean_diff', 'sig', 'wins', 'ties', 'losses']]
               .apply(seed_level).reset_index())

    # config vs config, seed-averaged per graph
    rows = []
    for ls, crit in itertools.product(['', '+2x1'], ['mean', '400-500']):
        avg = {c: sizes[[p for p, m in runs if m[1] == k and m[3] == crit and (m[4] or '') == ls]].mean(axis=1)
               for k, c in CFG.items()}
        for (a, b), r in itertools.product(itertools.combinations(CFG.values(), 2), ranges):
            d = (avg[a].loc[r] - avg[b].loc[r]).dropna()
            rows.append(dict(ls=ls or 'raw', crit=crit, a=a, b=b, range=r, n=len(d),
                             mean_diff=d.mean(), p=wilcoxon(d.values)))
    cc = pd.DataFrame(rows)
    cc['p_holm'] = holm(cc.p)

    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 20)
    short = {r: r.replace('_', '-') for r in ranges}
    print(f'{len(sizes)} graph rows, {len(runs)} seed pipelines, {len(vs)} RL-vs-greedy tests, '
          f'{len(cc)} config-vs-config tests; Holm at alpha = {args.alpha}')
    for ls in ('raw', '+2x1'):
        s = summary[summary.ls == ls].copy()
        s['cell'] = (s['mean'].map('{:+.2f}'.format) + '±' + s.ci.map('{:.2f}'.format) + ' '
                     + s.n_for.astype(int).astype(str) + '/' + s.n_against.astype(int).astype(str))
        t = s.pivot_table(index=['crit', 'cfg'], columns='range', values='cell', aggfunc='first')
        print(f'\n== RL - greedy{"" if ls == "raw" else " (both +2x1)"}: mean cover difference in vertices '
              f'(negative = RL better), mean ± 95% CI over {n_seeds} seeds, seeds significant for / against RL')
        print(t[ranges].rename(columns=short).to_string())
        w = s.pivot_table(index=['crit', 'cfg'], columns='range', values=['wins', 'ties', 'losses'])
        cell = (w['wins'].round(0).astype(int).astype(str) + '/' + w['ties'].round(0).astype(int).astype(str)
                + '/' + w['losses'].round(0).astype(int).astype(str))
        print(f'-- per-graph wins / ties / losses of RL (mean over seeds, out of the graphs per range)')
        print(cell[ranges].rename(columns=short).to_string())
    for ls in ('raw', '+2x1'):
        c = cc[cc.ls == ls].copy()
        c['cell'] = c.mean_diff.map('{:+.2f}'.format) + np.where(c.p_holm < args.alpha, '*', ' ')
        t = c.pivot_table(index=['crit', 'a', 'b'], columns='range', values='cell', aggfunc='first')
        print(f'\n== config a - config b ({ls}): mean difference of the seed-averaged cover, vertices '
              f'(* = Holm-significant)')
        print(t[ranges].rename(columns=short).to_string())
    if args.out:
        vs.to_csv(f'{args.out}_rl_vs_greedy.csv', index=False)
        cc.to_csv(f'{args.out}_config_pairs.csv', index=False)
        print(f'\nwrote {args.out}_rl_vs_greedy.csv and {args.out}_config_pairs.csv')
