"""Evaluates MVC (or, baselines only, MDS) pipelines on a test set built with build_val_set.py.

The problem comes from the set's meta.pt. For MDS the starts are the
max-gain greedy and a random minimal dominating set, and 2x1 is its MDS
version (drop redundant vertices, then swap two set vertices for one outside
vertex that keeps the set dominating).

Pipelines = a start (a trained checkpoint's greedy rollout, max-degree
greedy, or a random minimal cover) optionally followed by the 2x1 local
search (drop redundant vertices, then swap two cover vertices for one common
neighbour). For each graph it records the cover size, the reference size
(Gurobi's best cover), the apx-ratio and the wall-clock time; per range and
pipeline it prints the mean ratio, the share of graphs matching or beating
the reference, and the time per graph.

    python evaluate.py --test_dir /test_sets/<set> \
        --checkpoint 2135-mean=experiments/<id>/checkpoints/step=N.ckpt ...
"""
import argparse
import glob
import os
import random
import time

import torch
from torch_geometric.loader import DataLoader

from dql import DQGNS2V
from graph import is_vc_vectorized

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument('--test_dir', required=True)
parser.add_argument('--checkpoint', nargs='*', default=[],
                    help='label=path pairs of checkpoints to evaluate.')
parser.add_argument('--ranges', nargs='*', default=None,
                    help="ranges to evaluate, e.g. 15_21 (default: all finished).")
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--out', default=None,
                    help='CSV with one row per graph and pipeline, written after each range.')
parser.add_argument('--subsample', nargs='*', default=[],
                    help='RANGE=N pairs: evaluate only the first N graphs of that range, '
                         'e.g. 1000_1201=250.')
parser.add_argument('--seed', type=int, default=0)


def adjacency(g):
    adj = [set() for _ in range(g.num_nodes)]
    for u, v in g.edge_index.t().tolist():
        adj[u].add(v)
    return adj


def is_cover(adj, s):
    return all(u in s or nbrs <= s for u, nbrs in enumerate(adj))


def max_degree_greedy(adj):
    deg = [len(n) for n in adj]
    alive = [set(n) for n in adj]
    s = set()
    while max(deg) > 0:
        v = max(range(len(deg)), key=deg.__getitem__)
        s.add(v)
        for u in alive[v]:
            alive[u].discard(v)
            deg[u] -= 1
        alive[v] = set()
        deg[v] = 0
    return s


def random_minimal(adj, rng):
    """Adds vertices in random order while they still have an uncovered
    edge, then drops redundant ones."""
    order = list(range(len(adj)))
    rng.shuffle(order)
    s = set()
    for v in order:
        if v not in s and any(u not in s for u in adj[v]):
            s.add(v)
    return prune(adj, s)


def prune(adj, s):
    """Drops vertices whose neighbours are all in the cover, in order of
    degree (ties by index)."""
    s = set(s)
    for v in sorted(s, key=lambda v: (len(adj[v]), v)):
        if adj[v] <= s - {v}:
            s.discard(v)
    return s


def local_search_2x1(adj, s):
    """Remove redundant vertices, then replace two cover vertices u, v by
    one common neighbour w outside the cover whenever the result is a cover
    (first improvement: lowest w, then lowest pair; until no move applies).

    For each vertex it keeps the number of neighbours outside the cover and
    the sum of their ids: a move at w needs two non-adjacent cover vertices
    whose only outside neighbour is w (the sum then names it), and only
    neighbours of w can become redundant after it.
    """
    n = len(adj)
    s = set(s)
    out = [0] * n
    out_sum = [0] * n
    for u in range(n):
        for x in adj[u]:
            if x not in s:
                out[u] += 1
                out_sum[u] += x

    def remove(v):  # v leaves the cover
        s.discard(v)
        for x in adj[v]:
            out[x] += 1
            out_sum[x] += v

    def add(v):  # v joins the cover
        s.add(v)
        for x in adj[v]:
            out[x] -= 1
            out_sum[x] -= v

    def prune_among(vs):
        for v in sorted(vs, key=lambda v: (len(adj[v]), v)):
            if v in s and out[v] == 0:
                remove(v)

    prune_among(list(s))
    while True:
        groups = {}
        for u in s:
            if out[u] == 1:
                groups.setdefault(out_sum[u], []).append(u)
        move = None
        for w in sorted(groups):
            cand = sorted(groups[w])
            for i, u in enumerate(cand):
                v = next((v for v in cand[i + 1:] if v not in adj[u]), None)
                if v is not None:
                    move = u, v, w
                    break
            if move:
                break
        if move is None:
            return s
        u, v, w = move
        remove(u)
        remove(v)
        add(w)
        prune_among([x for x in adj[w] if x in s])


# MDS counterparts, used when the test set's meta.pt says problem='mds'.
# Closed neighbourhoods: N[v] = adj[v] | {v}.

def closed(adj):
    return [nbrs | {v} for v, nbrs in enumerate(adj)]


def is_ds(adj, s):
    return all(u in s or not nbrs.isdisjoint(s) for u, nbrs in enumerate(adj))


def max_gain_greedy_ds(adj):
    """Repeatedly adds the vertex that dominates the most undominated
    vertices (ties to the lowest index)."""
    nb = closed(adj)
    gain = [len(c) for c in nb]
    dominated = [False] * len(adj)
    s = set()
    while max(gain) > 0:
        v = max(range(len(gain)), key=gain.__getitem__)
        s.add(v)
        for x in nb[v]:
            if not dominated[x]:
                dominated[x] = True
                for w in nb[x]:
                    gain[w] -= 1
    return s


def random_minimal_ds(adj, rng):
    """Adds vertices in random order while they dominate a new vertex, then
    drops redundant ones."""
    nb = closed(adj)
    order = list(range(len(adj)))
    rng.shuffle(order)
    s, dominated = set(), set()
    for v in order:
        if not nb[v] <= dominated:
            s.add(v)
            dominated |= nb[v]
    return local_search_2x1_ds(adj, s, swaps=False)


def local_search_2x1_ds(adj, s, swaps=True):
    """MDS version of the 2x1 local search: remove redundant vertices, then
    replace two set vertices u, v by one vertex w outside the set whenever
    the result still dominates (first improvement: lowest w, then lowest
    pair; until no move applies).

    cnt[x] = |N[x] & s|. The private vertices of u (cnt == 1, dominated
    only by u) must all lie in N[w], so w is searched only in the
    intersection of their closed neighbourhoods; each candidate move is then
    checked on N[u] | N[v].
    """
    nb = closed(adj)
    s = set(s)
    cnt = [len(c & s) for c in nb]

    def remove(v):
        s.discard(v)
        for x in nb[v]:
            cnt[x] -= 1

    def add(v):
        s.add(v)
        for x in nb[v]:
            cnt[x] += 1

    def prune_all():
        for v in sorted(s, key=lambda v: (len(adj[v]), v)):
            if all(cnt[x] >= 2 for x in nb[v]):
                remove(v)

    prune_all()
    if not swaps:
        return s
    while True:
        # After pruning every set vertex has at least one private vertex.
        cand = {}  # w -> set vertices u whose private vertices all lie in N[w]
        for u in s:
            priv = sorted((x for x in nb[u] if cnt[x] == 1), key=lambda x: len(nb[x]))
            ws = set(nb[priv[0]])
            for x in priv[1:]:
                ws &= nb[x]
                if not ws:
                    break
            for w in ws - s:
                cand.setdefault(w, []).append(u)
        move = None
        for w in sorted(cand):
            us = sorted(cand[w])
            for i, u in enumerate(us):
                for v in us[i + 1:]:
                    if all(cnt[x] - (x in nb[u]) - (x in nb[v]) + (x in nb[w]) >= 1
                           for x in nb[u] | nb[v]):
                        move = u, v, w
                        break
                if move:
                    break
            if move:
                break
        if move is None:
            return s
        u, v, w = move
        remove(u)
        remove(v)
        add(w)
        prune_all()


@torch.no_grad()
def rl_rollout(net, graphs, device, batch_size):
    """Greedy rollout of a Q-network: repeatedly add the highest-Q vertex of
    each unsolved graph (ties to the lowest index) until all are covers."""
    covers, times = [], []
    for batch in DataLoader(graphs, batch_size=batch_size):
        t0 = time.perf_counter()
        batch = batch.to(device)
        x = torch.zeros(batch.num_nodes, 1, device=device)
        num_graphs = batch.num_graphs
        # Nodes of a graph are contiguous: lay the Q-values out as a
        # (graph, local index) grid and take each row's argmax at once.
        start = batch.ptr[:-1]
        local = torch.arange(batch.num_nodes, device=device) - start[batch.batch]
        grid = torch.full((num_graphs, int(local.max()) + 1), float('-inf'), device=device)
        unsolved = torch.ones(num_graphs, dtype=torch.bool, device=device)
        while unsolved.any():
            q = net(x, batch.edge_index, batch.batch).squeeze(-1)
            q = q.masked_fill(x[:, 0] == 1, float('-inf'))
            grid[batch.batch, local] = q
            pick = start + grid.argmax(dim=1)  # first maximum: lowest index
            x[pick[unsolved], 0] = 1
            unsolved = ~is_vc_vectorized(batch.edge_index, x[:, 0] == 1,
                                         batch.batch, num_graphs)
        if device != 'cpu':
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / num_graphs
        chosen = x[:, 0].cpu() == 1
        for gi, (a, b) in enumerate(zip(batch.ptr[:-1].tolist(), batch.ptr[1:].tolist())):
            covers.append(set(torch.nonzero(chosen[a:b]).flatten().tolist()))
            times.append(dt)
    return covers, times


def load_net(path, device):
    ckpt = torch.load(path, map_location=device)
    hp = ckpt.get('hyper_parameters', {})
    kwargs = {k: hp[k] for k in ('num_iterations', 'msg_norm', 'graph_pool') if k in hp}
    net = DQGNS2V(c_in=1, **kwargs).to(device)
    net.load_state_dict({k[4:]: v for k, v in ckpt['state_dict'].items()
                         if k.startswith('net.')})
    return net.eval()


if __name__ == '__main__':
    args = parser.parse_args()
    rng = random.Random(args.seed)
    files = sorted(glob.glob(f'{args.test_dir}/*_*.pt'),
                   key=lambda f: int(os.path.basename(f).split('_')[0]))
    files = [f for f in files if '.part' not in f]
    if args.ranges:
        files = [f for f in files if os.path.basename(f)[:-3] in args.ranges]
    problem = torch.load(f'{args.test_dir}/meta.pt').get('problem', 'mvc')
    if problem == 'mds':
        if args.checkpoint:
            raise NotImplementedError('the RL rollout only builds vertex covers; '
                                      'MDS sets evaluate the baselines only.')
        is_feasible, local_search = is_ds, local_search_2x1_ds
        heuristics = (('greedy', max_gain_greedy_ds),
                      ('random', lambda a: random_minimal_ds(a, rng)))
    else:
        is_feasible, local_search = is_cover, local_search_2x1
        heuristics = (('greedy', max_degree_greedy),
                      ('random', lambda a: random_minimal(a, rng)))
    nets = {}
    for pair in args.checkpoint:
        label, path = pair.split('=', 1)
        nets[label] = load_net(path, args.device)

    subsample = {r: int(k) for r, k in (pair.split('=') for pair in args.subsample)}
    out = open(args.out, 'w') if args.out else None
    if out:
        out.write('range,graph,pipeline,size,ref,ref_gap,apx_ratio,seconds\n')
    rows = []
    for f in files:
        rng_name = os.path.basename(f)[:-3]
        graphs = torch.load(f)[:subsample.get(rng_name)]
        refs = [int((g.y == 1).sum()) for g in graphs]
        adjs = [adjacency(g) for g in graphs]
        starts = {}
        for label, net in nets.items():
            starts[label] = rl_rollout(net, graphs, args.device, args.batch_size)
        for label, fn in heuristics:
            covers, times = [], []
            for a in adjs:
                t0 = time.perf_counter()
                covers.append(fn(a))
                times.append(time.perf_counter() - t0)
            starts[label] = (covers, times)
        for label, (covers, times) in starts.items():
            for i, (s, t) in enumerate(zip(covers, times)):
                assert is_feasible(adjs[i], s), (label, rng_name, i)
                t0 = time.perf_counter()
                s_ls = local_search(adjs[i], s)
                t_ls = time.perf_counter() - t0
                assert is_feasible(adjs[i], s_ls)
                for pipe, size, tt in ((label, len(s), t),
                                       (f'{label}+2x1', len(s_ls), t + t_ls)):
                    rows.append((rng_name, i, pipe, size, refs[i],
                                 getattr(graphs[i], 'gap', 0.0), size / refs[i], tt))

        print(f'\n== {rng_name}: {len(graphs)} graphs')
        print(f"{'pipeline':24s} {'apx-ratio':>9s} {'<= ref':>7s} {'ms/graph':>9s}")
        for pipe in dict.fromkeys(r[2] for r in rows if r[0] == rng_name):
            rr = [r for r in rows if r[0] == rng_name and r[2] == pipe]
            print(f"{pipe:24s} {sum(r[6] for r in rr) / len(rr):9.4f} "
                  f"{100 * sum(r[3] <= r[4] for r in rr) / len(rr):6.1f}% "
                  f"{1000 * sum(r[7] for r in rr) / len(rr):9.2f}")
        if out:
            for r in rows:
                if r[0] == rng_name:
                    out.write(','.join(str(x) for x in r) + '\n')
            out.flush()
