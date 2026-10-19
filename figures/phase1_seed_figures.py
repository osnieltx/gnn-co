"""Static figures for the phase 1 seed evaluation (evaluate.py CSVs of launch_eval_seeds.sh).

    python figures/phase1_seed_figures.py experiments/eval_seeds/g*.csv [--out figures/phase1]

fig1_ratio_vs_size   mean apx-ratio vs range, config lines with min-max seed band
fig2_seed_dots       one dot per seed on four key ranges
fig3_excess_vertices excess vertices (cover - reference) vs range
fig4_pick_pairs      per run: mean pick vs 400-500 pick on 400-500 and 1000-1200
Each figure is written as PDF (paper) and PNG (vault).
"""
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument('csv', nargs='+')
parser.add_argument('--out', default='figures/phase1')

CFG = {'ep': 'cumulative-uniform', 'tr': 'cumulative-balanced', 'cur': 'cumulative-current',
       'rep': 'replace'}
ORDER = ['cumulative-current', 'cumulative-balanced', 'cumulative-uniform', 'replace']
# reference palette slots 1-3 and 7 (blue, orange, aqua, violet), light mode
COLOR = dict(zip(ORDER, ['#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7']))
INK, INK2, GRID, GRAY = '#0b0b0b', '#52514e', '#e4e3df', '#8a8984'
SEED = re.compile(r'(ep|tr|cur|rep)-s(\d)-(mean|400-500)(\+2x1)?$')
PICK_TITLE = {'mean': 'mean pick (generalist)', '400-500': '400-500 pick (big-graph specialist)'}
# curriculum stage of each pick (journal, 30-09): (cfg, seed) -> (mean stage, 400-500 stage)
STAGE = {('cumulative-uniform', 1): (3, 7), ('cumulative-uniform', 2): (4, 4), ('cumulative-uniform', 3): (3, 7),
         ('cumulative-balanced', 1): (4, 4), ('cumulative-balanced', 2): (2, 2), ('cumulative-balanced', 3): (4, 6),
         ('cumulative-current', 1): (2, 2), ('cumulative-current', 2): (3, 3), ('cumulative-current', 3): (4, 5),
         ('replace', 1): (4, 7), ('replace', 2): (3, 7), ('replace', 3): (3, 3),
         # seeds 4-5 (eval_seeds45/checkpoints.txt)
         ('cumulative-uniform', 4): (6, 6), ('cumulative-uniform', 5): (4, 4),
         ('cumulative-balanced', 4): (4, 3), ('cumulative-balanced', 5): (4, 4),
         ('cumulative-current', 4): (3, 2), ('cumulative-current', 5): (4, 5),
         ('replace', 4): (4, 5), ('replace', 5): (2, 7)}

plt.rcParams.update({
    'font.size': 9, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'text.color': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID,
    'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titlesize': 9.5, 'axes.titleweight': 'normal', 'legend.frameon': False,
})


def range_key(r):
    return int(r.split('_')[0])


def label(r):
    a, b = r.split('_')
    b = int(b) - 1  # the CSV ranges are half-open
    return a if int(a) == b else f'{a}–{b}'


def load(files):
    frames = []
    for f in files:
        d = pd.read_csv(f)
        if frames:  # baselines are in every file; keep the first copy
            d = d[~d.pipeline.str.match(r'^(greedy|random)(\+2x1)?$')]
        frames.append(d)
    df = pd.concat(frames)
    df['excess'] = (df.apx_ratio - 1) * df.ref
    return df


def per_run(df, col):
    g = df.groupby(['pipeline', 'range'])[col].mean().reset_index()
    m = g.pipeline.str.extract(SEED)
    g, m = g[m[0].notna()].copy(), m[m[0].notna()]
    g['cfg'] = m[0].map(CFG)
    g['seed'] = m[1].astype(int)
    g['crit'] = m[2]
    g['ls'] = np.where(m[3].notna(), '+2x1', 'raw')
    return g


def baselines(df, col):
    g = df.groupby(['pipeline', 'range'])[col].mean().reset_index()
    return g[g.pipeline.isin(['greedy', 'greedy+2x1', 'random', 'random+2x1'])]


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'{name}.{ext}', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def lines_figure(df, col, ylabel, out, name, title, logy_raw=True):
    runs, base = per_run(df, col), baselines(df, col)
    ranges = sorted(runs.range.unique(), key=range_key)
    x = np.arange(len(ranges))
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6), sharex=True, sharey='row')
    for i, ls in enumerate(['raw', '+2x1']):
        for j, crit in enumerate(['mean', '400-500']):
            ax = axes[i, j]
            for cfg in ORDER:
                s = runs[(runs.cfg == cfg) & (runs.crit == crit) & (runs.ls == ls)]
                p = s.pivot(index='seed', columns='range', values=col)[ranges]
                ax.fill_between(x, p.min(), p.max(), color=COLOR[cfg], alpha=0.15, lw=0)
                ax.plot(x, p.mean(), color=COLOR[cfg], lw=2, marker='o', ms=4, label=cfg)
            for b, ls_style in (('greedy', ':'), ('random', '--')):
                key = b if ls == 'raw' else b + '+2x1'
                v = base[base.pipeline == key].set_index('range')[col].reindex(ranges)
                ax.plot(x, v, color=GRAY, lw=1.4, ls=ls_style, label=b)
            if col == 'apx_ratio' and ls == 'raw':
                ax.set_ylim(0.995, 1.10)
                top = runs[(runs.crit == crit) & (runs.ls == ls)].groupby(['cfg', 'range'])[col].mean()
                off = [f'{cfg} @ {label(r)}: {v:.2f}' for (cfg, r), v in top[top > 1.10].items()]
                if off:
                    ax.text(0.98, 0.97, 'off scale\n' + '\n'.join(off), transform=ax.transAxes,
                            ha='right', va='top', fontsize=7.5, color=INK2,
                            bbox=dict(fc='white', ec=GRID, boxstyle='round,pad=0.3'))
            ax.set_title(f"{PICK_TITLE[crit]}, {'raw RL' if ls == 'raw' else 'RL + 2×1'}", loc='left')
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == 1:
                ax.set_xticks(x)
                ax.set_xticklabels([label(r) for r in ranges], rotation=40, ha='right')
                ax.set_xlabel('graph size range (nodes)')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(title, y=1.04, fontsize=11)
    fig.tight_layout()
    save(fig, out, name)


def fig2(df, out):
    runs = per_run(df, 'apx_ratio')
    n_seeds = runs.seed.nunique()
    keep = ['40_51', '100_201', '400_501', '1000_1201']
    keep = [r for r in keep if r in set(runs.range)]
    fig, axes = plt.subplots(2, len(keep), figsize=(2.9 * len(keep), 6), sharex=True)
    for i, crit in enumerate(['mean', '400-500']):
        for j, r in enumerate(keep):
            ax = axes[i, j]
            for k, cfg in enumerate(ORDER):
                s = runs[(runs.cfg == cfg) & (runs.crit == crit) & (runs.ls == 'raw') & (runs.range == r)]
                ax.scatter(s.apx_ratio, k + (s.seed.values - (n_seeds + 1) / 2) * 0.36 / n_seeds, s=46, color=COLOR[cfg], edgecolor='white',
                           linewidth=1.2, zorder=3)
                ax.plot([s.apx_ratio.mean()] * 2, [k - .3, k + .3], color=INK, lw=1.6, zorder=4)
            ax.set_yticks(range(len(ORDER)))
            ax.set_yticklabels(ORDER if j == 0 else [])
            ax.invert_yaxis()
            ax.grid(axis='y', visible=False)
            ax.set_title(f'{label(r)} nodes' if i == 0 else '', loc='left')
            if j == 0:
                ax.set_ylabel(PICK_TITLE[crit].split(' (')[0], fontsize=9)
            if i == 1:
                ax.set_xlabel('apx-ratio (raw RL)')
    fig.suptitle(f'One dot per seed; black tick = mean over the {n_seeds} seeds', y=1.0, fontsize=11)
    fig.tight_layout()
    save(fig, out, 'fig2_seed_dots')


def fig4(df, out):
    runs = per_run(df, 'apx_ratio')
    keep = [r for r in ['400_501', '1000_1201'] if r in set(runs.range)]
    fig, axes = plt.subplots(1, len(keep), figsize=(4.8 * len(keep), 4.6), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, keep):
        for cfg in ORDER:
            for seed in sorted(runs.seed.unique()):
                s = runs[(runs.cfg == cfg) & (runs.seed == seed) & (runs.ls == 'raw') & (runs.range == r)]
                v = s.set_index('crit').apx_ratio
                late = STAGE[(cfg, seed)][1] >= 6
                ax.plot([0, 1], [v['mean'], v['400-500']], color=COLOR[cfg], lw=1.8, alpha=.9,
                        ls='-' if late else (0, (3, 2)))
                ax.scatter([0, 1], [v['mean'], v['400-500']], s=34, color=COLOR[cfg],
                           edgecolor='white', linewidth=1, zorder=3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['mean pick', '400-500 pick'])
        ax.set_xlim(-.25, 1.25)
        ax.set_title(f'{label(r)} nodes (raw RL)', loc='left')
        ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('apx-ratio')
    hs = [plt.Line2D([], [], color=COLOR[c], lw=2) for c in ORDER]
    hs += [plt.Line2D([], [], color=GRAY, lw=1.8), plt.Line2D([], [], color=GRAY, lw=1.8, ls=(0, (3, 2)))]
    fig.legend(hs, ORDER + ['400-500 pick after stage ≥ 6', '400-500 pick on stage ≤ 5'],
               loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    save(fig, out, 'fig4_pick_pairs')


if __name__ == '__main__':
    args = parser.parse_args()
    df, out = load(args.csv), Path(args.out)
    n_seeds = per_run(df, 'apx_ratio').seed.nunique()
    lines_figure(df, 'apx_ratio', 'mean apx-ratio', out, 'fig1_ratio_vs_size',
                 f'Phase 1 seed runs: apx-ratio vs graph size (band = min–max over {n_seeds} seeds; raw panels clipped at 1.10)')
    lines_figure(df, 'excess', 'mean excess vertices over reference', out, 'fig3_excess_vertices',
                 f'Excess vertices (cover − reference) vs graph size (band = min–max over {n_seeds} seeds)')
    fig2(df, out)
    fig4(df, out)
    print('wrote', sorted(p.name for p in out.glob('*')))
