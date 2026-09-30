"""Phase 2: NuMVC from different starts under a common time budget.

For each test graph, builds the start covers of evaluate.py (a checkpoint's
greedy rollout, max-degree greedy, a random minimal cover) and runs NuMVC
(numvc/build.sh: the authors' code, patched to take a start cover) from each
one, plus NuMVC's own start ('numvc': its max-degree greedy with random
ties), for --seeds random seeds and --budget seconds. NuMVC prints its best
cover size at every improvement, so one run gives the whole anytime curve:
the best cover within a budget B is the last improvement at start time +
NuMVC time <= B.

Time accounting: the start time is evaluate.py's (the RL rollout's GPU wall
time amortized over its batch, the Python greedy's wall time); NuMVC's is its
own CPU time, which includes reading the start cover (or building its own
start). Runs go --processes at a time, one core each.

Writes one row per improvement to --out and prints, per range and start, the
mean apx-ratio (vs the test set reference) at each of --report budgets.

    python numvc_eval.py --test_dir /test_sets/<set> --ranges 300_401 --graphs 100 \
        --checkpoint 1231=experiments/<id>/checkpoints/step=N.ckpt --budget 10 \
        --out experiments/numvc/r300_401.csv
"""
import argparse
import csv
import os
import random
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import torch

from evaluate import (adjacency, is_cover, load_net, max_degree_greedy,
                      random_minimal, rl_rollout)

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument('--test_dir', required=True)
parser.add_argument('--ranges', nargs='+', required=True)
parser.add_argument('--graphs', type=int, default=None,
                    help='use the first this many graphs of each range (default: all).')
parser.add_argument('--checkpoint', nargs='*', default=[],
                    help='label=path pairs of checkpoints whose rollouts are starts.')
parser.add_argument('--no_baselines', action='store_true',
                    help="skip the greedy, random and NuMVC's own starts.")
parser.add_argument('--budget', type=float, default=10,
                    help='NuMVC cutoff in CPU seconds per run.')
parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
parser.add_argument('--report', type=float, nargs='+',
                    default=[0.01, 0.03, 0.1, 0.3, 1, 3, 10],
                    help='budgets (start + NuMVC seconds) to print ratios at.')
parser.add_argument('--numvc', default=os.path.join(os.path.dirname(__file__), 'numvc/bin/numvc'))
parser.add_argument('--processes', type=int, default=os.cpu_count())
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--out', required=True)
parser.add_argument('--seed', type=int, default=0, help='seed of the random starts.')


def write_dimacs(path, adj):
    edges = [(u, v) for u, nbrs in enumerate(adj) for v in nbrs if u < v]
    with open(path, 'w') as f:
        f.write(f'p edge {len(adj)} {len(edges)}\n')
        f.writelines(f'e {u + 1} {v + 1}\n' for u, v in edges)


def run_numvc(binary, graph_file, seed, budget, start_file=None):
    """Runs NuMVC; returns its improvements [(cpu seconds, size)] and final cover."""
    cmd = [binary, graph_file, '0', str(seed), str(budget)]
    if start_file:
        cmd.append(start_file)
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    lines = out.splitlines()
    trace = [(float(t), int(s)) for _, t, s in (l.split() for l in lines if l.startswith('t '))]
    # the line after "c The following output is the found independent set."
    i = next(k for k, l in enumerate(lines) if l.startswith('c The following output'))
    indep = {int(v) - 1 for v in lines[i + 1].split()}
    return trace, indep


def best_within(trace, t_start, budget):
    sizes = [s for t, s in trace if t_start + t <= budget]
    return min(sizes) if sizes else None


if __name__ == '__main__':
    args = parser.parse_args()
    if not os.path.exists(args.numvc):
        raise SystemExit(f'{args.numvc} not found: run sh numvc/build.sh first.')
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    rng = random.Random(args.seed)
    nets = {label: load_net(path, args.device) for label, path in
            (pair.split('=', 1) for pair in args.checkpoint)}

    with open(args.out, 'w', newline='') as fo, tempfile.TemporaryDirectory() as tmp:
        w = csv.writer(fo)
        w.writerow(['range', 'graph', 'start', 'seed', 'n', 'ref', 'start_size',
                    'start_seconds', 'numvc_seconds', 'size'])
        for rng_name in args.ranges:
            graphs = torch.load(f'{args.test_dir}/{rng_name}.pt')[:args.graphs]
            refs = [int((g.y == 1).sum()) for g in graphs]
            adjs = [adjacency(g) for g in graphs]
            starts = {label: rl_rollout(net, graphs, args.device, args.batch_size)
                      for label, net in nets.items()}
            if not args.no_baselines:
                for label, fn in (('greedy', max_degree_greedy),
                                  ('random', lambda a: random_minimal(a, rng))):
                    covers, times = [], []
                    for a in adjs:
                        t0 = time.perf_counter()
                        covers.append(fn(a))
                        times.append(time.perf_counter() - t0)
                    starts[label] = (covers, times)

            jobs = []
            for i, a in enumerate(adjs):
                gfile = f'{tmp}/{rng_name}_{i}.dimacs'
                write_dimacs(gfile, a)
                for label, (covers, times) in starts.items():
                    assert is_cover(a, covers[i]), (label, rng_name, i)
                    sfile = f'{tmp}/{rng_name}_{i}_{label}.start'
                    with open(sfile, 'w') as f:
                        f.write(' '.join(str(v + 1) for v in sorted(covers[i])))
                    for seed in args.seeds:
                        jobs.append((i, label, seed, gfile, sfile, len(covers[i]), times[i]))
                if not args.no_baselines:
                    for seed in args.seeds:  # NuMVC's own start; its time is in NuMVC's
                        jobs.append((i, 'numvc', seed, gfile, None, None, 0.0))

            def work(job):
                i, label, seed, gfile, sfile, start_size, t_start = job
                trace, indep = run_numvc(args.numvc, gfile, seed, args.budget, sfile)
                cover = set(range(len(adjs[i]))) - indep
                assert is_cover(adjs[i], cover) and len(cover) == trace[-1][1], (rng_name, i, label)
                return job, trace

            t0 = time.time()
            results = []
            with ThreadPoolExecutor(args.processes) as pool:
                for k, (job, trace) in enumerate(pool.map(work, jobs)):
                    i, label, seed, _, _, start_size, t_start = job
                    for t, s in trace:
                        w.writerow([rng_name, i, label, seed, len(adjs[i]), refs[i],
                                    start_size if start_size is not None else trace[0][1],
                                    f'{t_start:.6f}', f'{t:.6f}', s])
                    results.append((i, label, trace, t_start))
                    if (k + 1) % 500 == 0:
                        print(f'{rng_name}: {k + 1}/{len(jobs)} runs, {time.time() - t0:.0f} s', flush=True)
            fo.flush()

            print(f'\n== {rng_name}: {len(graphs)} graphs x {len(args.seeds)} seeds, '
                  f'mean apx-ratio at start + NuMVC budget (s); "-" = start slower than budget')
            print(f"{'start':24s}" + ''.join(f'{b:>9g}' for b in args.report))
            for label in dict.fromkeys(r[1] for r in results):
                row = f'{label:24s}'
                for b in args.report:
                    ratios = [best_within(tr, ts, b) / refs[i]
                              for i, l, tr, ts in results
                              if l == label and best_within(tr, ts, b) is not None]
                    n_runs = sum(l == label for _, l, _, _ in results)
                    row += (f'{sum(ratios) / len(ratios):9.4f}' if len(ratios) == n_runs
                            else f"{'-':>9s}")
                print(row, flush=True)
