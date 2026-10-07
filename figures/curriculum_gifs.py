"""One GIF per curriculum type, showing which stages the training graphs come from.

    python figures/curriculum_gifs.py [--out figures/phase1/gifs]

Black marks are the expected fraction of transitions per stage, computed from the weights
in dql.Agent.sample_graph. Bars are the replay buffer filled by sampled episodes (fixed seed):
as in dql.advance_curriculum, the buffer is cleared at every stage advance and refilled from
the current mix. An episode adds ~n transitions (n = mean size of its stage). The time axis is
schematic: stages advance at equal intervals (the real schedule depends on the run).
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument('--out', default='figures/phase1/gifs')

STAGES = [(10, 11), (15, 21), (40, 51), (50, 101), (100, 201), (200, 301), (300, 401), (400, 501)]
LABELS = ['10', '15–20', '40–50', '50–100', '100–200', '200–300', '300–400', '400–500']
MEAN_N = np.array([(a + b - 1) / 2 for a, b in STAGES])
CURRENT_SHARE = 0.5
FRAMES_PER_STAGE = 12
EPISODES_PER_FRAME = 80
WARM_EPISODES = 40  # episodes sampled right after a stage advance
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
# one hue, light to dark with stage size (stage identity, not a config color)
STAGE_COLOR = plt.cm.Blues(np.linspace(0.35, 0.95, len(STAGES)))
TYPES = {
    'replace': 'replace: only the current stage is sampled',
    'cumulative-uniform': 'cumulative-uniform: stage drawn uniformly per episode',
    'cumulative-balanced': 'cumulative-balanced: stage drawn with p ∝ 1/n̄, equal transitions per stage',
    'cumulative-current': f'cumulative-current: {CURRENT_SHARE:.0%} of transitions on the current stage',
}


def shares(kind, k):
    """Expected share of transitions from each of the k active stages (0-indexed last = current)."""
    n = MEAN_N[:k]
    if kind == 'replace':
        w = np.zeros(k)
        w[-1] = 1
        return w
    if kind == 'cumulative-uniform':   # equal per episode -> transitions ∝ mean n
        w = n.copy()
    elif kind == 'cumulative-balanced':  # p ∝ 1/n -> equal transitions
        w = np.ones(k)
    else:                               # current stage gets CURRENT_SHARE, rest split evenly
        w = np.full(k, (1 - CURRENT_SHARE) / (k - 1) if k > 1 else 0.0)
        w[-1] = CURRENT_SHARE if k > 1 else 1.0
    return w / w.sum()


def draw(kind, out):
    fig, ax_b = plt.subplots(figsize=(7.4, 3.9))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.84, bottom=0.16)
    total = FRAMES_PER_STAGE * len(STAGES)
    rng = np.random.default_rng(0)
    history, counts = [], np.zeros(len(STAGES))
    for f in range(total):
        k = f // FRAMES_PER_STAGE + 1
        w = np.zeros(len(STAGES))
        w[:k] = shares(kind, k)
        q = np.zeros(len(STAGES))  # per-episode probability: transition share / episode length
        q[:k] = w[:k] / MEAN_N[:k]
        q /= q.sum()
        n_ep = EPISODES_PER_FRAME
        if f % FRAMES_PER_STAGE == 0:
            counts[:] = 0  # buffer cleared at the stage advance
            n_ep = WARM_EPISODES
        for i in rng.choice(len(STAGES), size=n_ep, p=q):
            counts[i] += MEAN_N[i]
        history.append((k, w, counts / counts.sum()))

    def update(f):
        k, w, full = history[min(f, total - 1)]
        ax_b.clear()
        ax_b.bar(range(len(STAGES)), full, color=STAGE_COLOR, width=0.8,
                 label='replay buffer (cleared at each stage advance)')
        ax_b.scatter(range(k), w[:k], marker='_', s=420, color=INK, linewidths=2, zorder=3,
                     label='expected share')
        for i in range(k):
            if full[i] > 0:
                ax_b.text(i, max(full[i], w[i]) + 0.02, f'{full[i]:.0%}' if full[i] >= 0.01 else f'{full[i]:.1%}',
                          ha='center', fontsize=8, color=INK)
        ax_b.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=8)
        ax_b.set_xticks(range(len(STAGES)))
        ax_b.set_xticklabels(LABELS, fontsize=8, color=INK2)
        ax_b.set_ylim(0, 1.05)
        ax_b.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax_b.set_yticklabels(['0', '25%', '50%', '75%', '100%'], color=INK2, fontsize=8)
        ax_b.set_ylabel('share of transitions', fontsize=9, color=INK2)
        ax_b.set_xlabel('graph size range (nodes)', fontsize=9, color=INK2)
        ax_b.grid(axis='y', color=GRID, linewidth=0.6)
        ax_b.set_axisbelow(True)
        for s in ('top', 'right'):
            ax_b.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax_b.spines[s].set_color(GRID)
        fig.suptitle(TYPES[kind], fontsize=10.5, color=INK, x=0.1, ha='left', y=0.97)

    anim = FuncAnimation(fig, update, frames=total + 10, interval=120)
    anim.save(out / f'{kind}.gif', writer=PillowWriter(fps=8), dpi=110,
              savefig_kwargs={'facecolor': 'white'})
    plt.close(fig)


if __name__ == '__main__':
    out = Path(parser.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    for kind in TYPES:
        draw(kind, out)
    print('wrote', sorted(p.name for p in out.glob('*.gif')))
