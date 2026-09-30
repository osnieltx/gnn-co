"""Compares the test set's Gurobi covers and bounds with the size of the
minimum vertex cover predicted by random graph theory for G(n, p).

A cover's complement is an independent set, so MVC = n - alpha. In G(n, p)
the expected number of independent sets of size k is
E[X_k] = C(n, k) (1 - p)^C(k, 2), and by Markov P(alpha >= k) <= E[X_k]:
- alpha_99 = the smallest k with E[X_{k+1}] <= 0.01, so MVC >= n - alpha_99
  with probability >= 99% over the sampling of the graph (a lower bound that
  holds for the distribution, not a proof for one graph);
- alpha_1m = the largest k with E[X_k] >= 1 (first moment), so MVC ~ n - alpha_1m
  (alpha concentrates on a few values near it; biased high at finite n).
Ranges whose references are proven optimal calibrate the estimate: they give
the true alpha - alpha_1m.

For each range, prints per graph: n, the reference cover and Gurobi's bound
(ref * (1 - gap)), the implied independent set n - ref, alpha_1m, alpha_99,
and, with --eval, the best cover of any pipeline in evaluate.py's CSVs; with
--trace, the bound reached by gap_trace.py's variants.

    python gap_analysis.py --test_dir /test_sets/<set> --ranges 100_201 300_401 \
        --eval experiments/eval/r300_401.csv --trace experiments/gap_trace/trace_summary.csv
"""
import argparse
import csv
from collections import defaultdict
from math import lgamma, log

import torch

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument('--test_dir', required=True)
parser.add_argument('--ranges', nargs='+', required=True)
parser.add_argument('-p', type=float, default=.15)
parser.add_argument('--graphs', type=int, default=None,
                    help='per-graph rows for the first this many graphs (default: none).')
parser.add_argument('--eval', nargs='*', default=[], help="evaluate.py CSVs.")
parser.add_argument('--trace', default=None, help="gap_trace.py's summary CSV.")


def log_expected_indep_sets(n, k, p):
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1) + k * (k - 1) / 2 * log(1 - p)


def alpha_estimates(n, p):
    a_1m = max(k for k in range(1, n) if log_expected_indep_sets(n, k, p) >= 0)
    a_99 = min(k for k in range(1, n) if log_expected_indep_sets(n, k + 1, p) <= log(.01))
    return a_1m, a_99


def mean(xs):
    return sum(xs) / len(xs) if xs else float('nan')


if __name__ == '__main__':
    args = parser.parse_args()
    best = defaultdict(lambda: (10 ** 9, ''))  # (range, graph) -> (size, pipeline)
    for f in args.eval:
        for r in csv.DictReader(open(f)):
            key = (r['range'], int(r['graph']))
            if int(r['size']) < best[key][0]:
                best[key] = (int(r['size']), r['pipeline'])
    trace = {}
    if args.trace:
        for r in csv.DictReader(open(args.trace)):
            trace[(r['range'], int(r['graph']), r['variant'])] = r

    for rng_name in args.ranges:
        graphs = torch.load(f'{args.test_dir}/{rng_name}.pt')
        rows = []
        for i, g in enumerate(graphs):
            n = g.num_nodes
            ref = int((g.y == 1).sum())
            gap = float(getattr(g, 'gap', 0.0))
            a_1m, a_99 = alpha_estimates(n, args.p)
            rows.append(dict(i=i, n=n, ref=ref, gap=gap, bound=ref * (1 - gap),
                             alpha_ref=n - ref, a_1m=a_1m, a_99=a_99,
                             best=best.get((rng_name, i), (None, ''))))
        opt = [r for r in rows if r['gap'] < 1e-4]
        print(f'\n== {rng_name}: {len(rows)} graphs, {len(opt)} proven optimal')
        print(f"mean n {mean([r['n'] for r in rows]):.1f}; ref {mean([r['ref'] for r in rows]):.1f}; "
              f"Gurobi bound {mean([r['bound'] for r in rows]):.1f}; "
              f"random-graph bound n - alpha_99 {mean([r['n'] - r['a_99'] for r in rows]):.1f} "
              f"(above Gurobi's on {sum(r['n'] - r['a_99'] > r['bound'] for r in rows)} graphs)")
        print(f"independent set implied by ref (n - ref) {mean([r['alpha_ref'] for r in rows]):.2f}; "
              f"alpha_1m {mean([r['a_1m'] for r in rows]):.2f}; alpha_99 {mean([r['a_99'] for r in rows]):.2f}")
        if opt:
            diff = [r['alpha_ref'] - r['a_1m'] for r in opt]
            hist = {d: diff.count(d) for d in sorted(set(diff))}
            print(f'calibration on proven optima: true alpha - alpha_1m = {hist}, '
                  f'alpha > alpha_99 on {sum(r["alpha_ref"] > r["a_99"] for r in opt)} graphs')
        withbest = [r for r in rows if r['best'][0] is not None]
        if withbest:
            print(f"best pipeline cover {mean([r['best'][0] for r in withbest]):.1f} vs ref "
                  f"{mean([r['ref'] for r in withbest]):.1f} "
                  f"(below ref on {sum(r['best'][0] < r['ref'] for r in withbest)} / {len(withbest)})")
        for r in rows[:args.graphs or 0]:
            extra = ''
            for v in ('default', 'bound'):
                t = trace.get((rng_name, r['i'], v))
                if t:
                    extra += (f" | {v}: cover {t['cover']} bound {float(t['final_bound']):.1f} "
                              f"final inc at {t['seconds_to_final_incumbent'] or '?'} s")
            print(f"#{r['i']:<3d} n {r['n']:4d} ref {r['ref']:4d} Gurobi bound {r['bound']:6.1f} "
                  f"n-alpha_99 {r['n'] - r['a_99']:4d} n-alpha_1m {r['n'] - r['a_1m']:4d} "
                  f"best {r['best'][0]} ({r['best'][1]}){extra}")
