"""Illustrates MVC(G) = n - alpha(G) on a small G(n, p) graph.

    python figures/mvc_alpha_duality.py [--n 12] [--p 0.25] [--out figures/phase1]

Left: a minimum vertex cover (every edge has an endpoint in it). Right: its complement,
a maximum independent set (no edges inside). Found by brute force.
"""
import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument('--n', type=int, default=12)
parser.add_argument('--p', type=float, default=0.25)
parser.add_argument('--out', default='figures/phase1')
COVER, INDEP, EDGE, INK, INK2 = '#eb6834', '#2a78d6', '#a8a79f', '#0b0b0b', '#52514e'


def min_cover(g):
    nodes = list(g)
    for k in range(len(nodes) + 1):
        for c in combinations(nodes, k):
            s = set(c)
            if all(u in s or v in s for u, v in g.edges):
                return s


def pick_graph(n, p):
    """First connected G(n, p) from seed 0, 1, ... (so the drawing is one piece)."""
    seed = 0
    while True:
        g = nx.gnp_random_graph(n, p, seed=seed)
        if nx.is_connected(g):
            return g, seed
        seed += 1


if __name__ == '__main__':
    args = parser.parse_args()
    g, seed = pick_graph(args.n, args.p)
    cover = min_cover(g)
    indep = set(g) - cover
    assert all(not g.has_edge(u, v) for u, v in combinations(indep, 2))
    pos = nx.spring_layout(g, seed=3, k=1.1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, chosen, color, title in (
            (axes[0], cover, COVER, f'minimum vertex cover: {len(cover)} vertices'),
            (axes[1], indep, INDEP, f'maximum independent set: {len(indep)} vertices')):
        nx.draw_networkx_edges(g, pos, ax=ax, edge_color=EDGE, width=1.6)
        nx.draw_networkx_nodes(g, pos, ax=ax, nodelist=sorted(set(g) - chosen), node_color='white',
                               edgecolors=INK2, linewidths=1.4, node_size=420)
        nx.draw_networkx_nodes(g, pos, ax=ax, nodelist=sorted(chosen), node_color=color,
                               edgecolors='white', linewidths=1.6, node_size=420)
        nx.draw_networkx_labels(g, pos, ax=ax, font_size=9,
                                font_color='white', labels={v: v for v in chosen})
        nx.draw_networkx_labels(g, pos, ax=ax, font_size=9, font_color=INK2,
                                labels={v: v for v in set(g) - chosen})
        ax.set_title(title, loc='left', fontsize=10.5, color=INK)
        ax.set_axis_off()
    fig.suptitle(f'MVC(G) = n − α(G):  {len(cover)} = {g.number_of_nodes()} − {len(indep)}'
                 f'   (G(n={args.n}, p={args.p}), {g.number_of_edges()} edges)',
                 fontsize=11.5, color=INK, y=0.99)
    fig.tight_layout()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'fig6_mvc_alpha_duality.{ext}', dpi=200, bbox_inches='tight', facecolor='white')
    print('seed', seed, 'cover', sorted(cover), 'independent', sorted(indep))
