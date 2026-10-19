"""Gurobi incumbent and bound over time on the traced test graphs (gap_trace.py).

    python figures/gap_trace_figure.py experiments/gap_trace [--variant default] [--out figures/phase1]

One panel per range, one pair of lines per graph, values divided by n. Dotted lines: the
random-graph lower bound n - alpha_99 (holds with probability >= 99%) and the estimate of the
optimum n - alpha_1m + 2 (see gap_analysis.py; offset calibrated on proven optima, n <= 200).
"""
import argparse
from math import lgamma, log
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument('dir')
parser.add_argument('--variant', default='default', choices=['default', 'bound'])
parser.add_argument('--out', default='figures/phase1')
P = 0.15
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
C_INC, C_BND = '#2a78d6', '#eb6834'
TITLE = {'300_401': '300–400 nodes', '400_501': '400–500 nodes', '500_601': '500–600 nodes'}


def log_ex(n, k):
    """log E[X_k]: expected number of independent sets of size k in G(n, P)."""
    return (lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)) + k * (k - 1) / 2 * log(1 - P)


def alphas(n):
    """(alpha_99, alpha_1m): smallest k with E[X_{k+1}] <= 0.01, largest k with E[X_k] >= 1."""
    a99 = next(k for k in range(1, n) if log_ex(n, k + 1) <= log(0.01))
    a1m = max(k for k in range(1, n) if log_ex(n, k) >= 0)
    return a99, a1m


plt.rcParams.update({
    'font.size': 9, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'text.color': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID,
    'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False,
})

if __name__ == '__main__':
    args = parser.parse_args()
    d = Path(args.dir)
    tr = pd.read_csv(d / 'trace.csv')
    sm = pd.read_csv(d / 'trace_summary.csv')
    tr, sm = tr[tr.variant == args.variant], sm[sm.variant == args.variant]
    ranges = [r for r in TITLE if r in set(tr.range)]
    fig, axes = plt.subplots(1, len(ranges), figsize=(4.6 * len(ranges), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, ranges):
        for g, t in tr[tr.range == r].groupby('graph'):
            n = int(sm[(sm.range == r) & (sm.graph == g)].n.iloc[0])
            t = t[(t.incumbent < 1e9) & (t.bound > 0) & (t.seconds > 0)]
            ax.step(t.seconds, t.incumbent / n, where='post', color=C_INC, lw=1.2, alpha=.75)
            ax.step(t.seconds, t.bound / n, where='post', color=C_BND, lw=1.2, alpha=.75)
            a99, a1m = alphas(n)
            ax.axhline((n - a99) / n, color=INK2, lw=0.7, ls=':', alpha=.6)
            ax.axhline((n - a1m + 2) / n, color=INK, lw=0.7, ls='--', alpha=.5)
        ax.set_xscale('log')
        ax.set_xlim(1, 3840)
        ax.set_title(TITLE[r], loc='left')
        ax.set_xlabel('solver time (s)')
    axes[0].set_ylabel('value / n (vertices of the cover)')
    h = [plt.Line2D([], [], color=C_INC, lw=2), plt.Line2D([], [], color=C_BND, lw=2),
         plt.Line2D([], [], color=INK2, lw=1, ls=':'), plt.Line2D([], [], color=INK, lw=1, ls='--')]
    fig.legend(h, ['Gurobi incumbent', 'Gurobi bound', 'random-graph bound n − α₉₉ (≥ 99%)',
                   'estimated optimum n − α₁ₘ + 2'], loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'fig5_gurobi_gap_trace.{ext}', dpi=200, bbox_inches='tight', facecolor='white')
    print('wrote fig5_gurobi_gap_trace')
