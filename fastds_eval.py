"""FastDS on an MDS test set under a CPU-time budget.

Runs FastDS (fastds/build.sh: the authors' code, patched to print its anytime
trace) on each test graph from its own start (inference rules + greedy
construction) for --seeds random seeds and --budget CPU seconds. FastDS prints
its best set size at every improvement, so one run gives the whole anytime
curve: the best set within a budget B is the last improvement at time <= B.
Its time counts reading the graph and the inference rules. Runs go
--processes at a time, one core each.

Writes one row per improvement to --out and prints, per range, the mean
apx-ratio, mean excess vertices and share of graphs at the reference (the
test set's Gurobi solution) at each of --report budgets; with --baselines (an
evaluate.py CSV), also those of its pipelines on the same graphs.

    python fastds_eval.py --test_dir gnn-co-data/test_sets/s2v_ext_ba4_mds_t3840 \
        --ranges 100_201 400_501 --graphs 100 --budget 10 \
        --baselines gnn-co-data/eval/ba4_mds_baselines.csv --out gnn-co-data/eval/ba4_mds_fastds.csv
"""
import argparse
import csv
import os
import subprocess
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import torch

from evaluate import adjacency, is_ds
from numvc_eval import best_within, write_dimacs

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument('--test_dir', required=True)
parser.add_argument('--ranges', nargs='+', required=True)
parser.add_argument('--graphs', type=int, default=None,
                    help='use the first this many graphs of each range (default: all).')
parser.add_argument('--budget', type=float, default=10,
                    help='FastDS cutoff in CPU seconds per run.')
parser.add_argument('--seeds', type=int, nargs='+', default=[1])
parser.add_argument('--report', type=float, nargs='+', default=[0.01, 0.1, 1, 10],
                    help='budgets (FastDS seconds) to print results at.')
parser.add_argument('--baselines', default=None,
                    help='evaluate.py CSV whose pipelines to print on the same graphs.')
parser.add_argument('--pipelines', nargs='+', default=['greedy+2x1'])
parser.add_argument('--fastds', default=os.path.join(os.path.dirname(__file__), 'fastds/bin/fastds'))
parser.add_argument('--processes', type=int, default=os.cpu_count())
parser.add_argument('--out', required=True)


def run_fastds(binary, graph_file, seed, budget):
    """Runs FastDS; returns its improvements [(cpu seconds, size)] and final set."""
    out = subprocess.run([binary, graph_file, str(budget), str(seed)],
                         capture_output=True, text=True, check=True).stdout
    lines = out.splitlines()
    trace = [(float(t), int(s)) for _, t, s in (l.split() for l in lines if l.startswith('t '))]
    ds = next({int(v) - 1 for v in l.split()[1:]} for l in lines if l.startswith('s'))
    return trace, ds


def summary(label, sizes, refs):
    """One printed row: mean apx-ratio, mean excess vertices, % at the reference."""
    n = len(sizes)
    return (f'{label:16s} {sum(s / r for s, r in zip(sizes, refs)) / n:9.4f} '
            f'{sum(s - r for s, r in zip(sizes, refs)) / n:8.2f} '
            f'{100 * sum(s <= r for s, r in zip(sizes, refs)) / n:6.1f}%')


if __name__ == '__main__':
    args = parser.parse_args()
    if not os.path.exists(args.fastds):
        raise SystemExit(f'{args.fastds} not found: run sh fastds/build.sh first.')
    assert torch.load(f'{args.test_dir}/meta.pt').get('problem') == 'mds'
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    base = defaultdict(dict)  # (range, pipeline) -> {graph: (size, seconds)}
    if args.baselines:
        with open(args.baselines) as f:
            for r in csv.DictReader(f):
                base[r['range'], r['pipeline']][int(r['graph'])] = (int(r['size']), float(r['seconds']))

    with open(args.out, 'w', newline='') as fo, tempfile.TemporaryDirectory() as tmp:
        w = csv.writer(fo)
        w.writerow(['range', 'graph', 'seed', 'n', 'ref', 'ref_gap', 'seconds', 'size'])
        for rng_name in args.ranges:
            graphs = torch.load(f'{args.test_dir}/{rng_name}.pt')[:args.graphs]
            refs = [int((g.y == 1).sum()) for g in graphs]
            adjs = [adjacency(g) for g in graphs]
            jobs = []
            for i, a in enumerate(adjs):
                gfile = f'{tmp}/{rng_name}_{i}.dimacs'
                write_dimacs(gfile, a)
                jobs += [(i, seed, gfile) for seed in args.seeds]

            def work(job):
                i, seed, gfile = job
                trace, ds = run_fastds(args.fastds, gfile, seed, args.budget)
                assert is_ds(adjs[i], ds) and len(ds) == trace[-1][1], (rng_name, i, seed)
                return job, trace

            t0 = time.time()
            results = []
            with ThreadPoolExecutor(args.processes) as pool:
                for (i, seed, _), trace in pool.map(work, jobs):
                    for t, s in trace:
                        w.writerow([rng_name, i, seed, len(adjs[i]), refs[i],
                                    getattr(graphs[i], 'gap', 0.0), f'{t:.6f}', s])
                    results.append((i, trace))
            fo.flush()

            print(f'\n== {rng_name}: {len(graphs)} graphs x {len(args.seeds)} seeds, '
                  f'{time.time() - t0:.0f} s wall; vs the reference: mean apx-ratio, '
                  f'mean excess vertices, % at the reference ("-" = no solution yet)')
            print(f"{'pipeline':16s} {'apx-ratio':>9s} {'excess':>8s} {'<= ref':>7s}")
            for b in args.report:
                sizes = [best_within(tr, 0, b) for _, tr in results]
                label = f'fastds@{b:g}s'
                print(summary(label, sizes, [refs[i] for i, _ in results])
                      if None not in sizes else f'{label:16s} {"-":>9s}')
            firsts = [tr[0][0] for _, tr in results]
            print(f'{"fastds first":16s} {summary("", [tr[0][1] for _, tr in results], [refs[i] for i, _ in results])[17:]}'
                  f'  at {1000 * sum(firsts) / len(firsts):.2f} ms/graph mean, {1000 * max(firsts):.2f} max')
            for pipe in args.pipelines:
                rows = base[rng_name, pipe]
                if all(i in rows for i in range(len(graphs))):
                    print(summary(pipe, [rows[i][0] for i in range(len(graphs))], refs)
                          + f'  at {1000 * sum(rows[i][1] for i in range(len(graphs))) / len(graphs):.2f} ms/graph')
