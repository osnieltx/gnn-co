"""Traces Gurobi's incumbent and lower bound over time on test set graphs.

Checks where the MIP gap of the big MVC ranges comes from: if the incumbent
reaches its final value early (and is close to the best cover our pipelines
find) while the bound crawls, the gap is the lower bound, and better starts
won't shorten the proofs.

Solves the first --graphs graphs of each range again, with the same setup
as build_val_set.py (1 thread per solve, --time_limit seconds), once per
--variant:
- default: Gurobi's defaults (should reproduce the stored references),
- bound: MIPFocus=3, Gurobi working on the bound.
Writes one row per change of incumbent or bound (and at least every 30 s)
to --out, and a summary per solve to <out>_summary.csv, appending as solves
finish so partial runs are usable.

    python gap_trace.py --test_dir /test_sets/<set> --ranges 300_401 400_501 \
        --graphs 8 --out experiments/gap_trace/trace.csv
"""
import argparse
import csv
import multiprocessing
import os

import torch

import graph

VARIANTS = {'default': {}, 'bound': {'MIPFocus': 3}}

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument('--test_dir', required=True)
parser.add_argument('--ranges', nargs='+', required=True)
parser.add_argument('--graphs', type=int, default=8,
                    help='solve the first this many graphs of each range.')
parser.add_argument('--variants', nargs='+', default=list(VARIANTS),
                    choices=list(VARIANTS))
parser.add_argument('--time_limit', type=float, default=3840)
parser.add_argument('--processes', type=int, default=16)
parser.add_argument('--out', required=True)


def solve(job):
    rng_name, i, variant, edge_index, n, time_limit = job
    import gurobipy as gp
    trace = []
    last = {'t': -1e9, 'inc': None, 'bnd': None}

    def cb(model, where):
        C = gp.GRB.Callback
        if where == C.MIP:
            inc, bnd = model.cbGet(C.MIP_OBJBST), model.cbGet(C.MIP_OBJBND)
        elif where == C.MIPSOL:  # a new incumbent, maybe with no MIP call after it
            inc, bnd = model.cbGet(C.MIPSOL_OBJBST), model.cbGet(C.MIPSOL_OBJBND)
        else:
            return
        t = model.cbGet(C.RUNTIME)
        if inc != last['inc'] or bnd != last['bnd'] or t - last['t'] >= 30:
            trace.append((t, inc, bnd))
            last.update(t=t, inc=inc, bnd=bnd)

    cover, gap = graph.milp_solve_mvc(edge_index, n, time_limit, return_gap=True,
                                      params=VARIANTS[variant], callback=cb)
    return rng_name, i, variant, n, len(cover), gap, trace


if __name__ == '__main__':
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    jobs = []
    for rng_name in args.ranges:
        graphs = torch.load(f'{args.test_dir}/{rng_name}.pt')[:args.graphs]
        for i, g in enumerate(graphs):
            for variant in args.variants:
                jobs.append((rng_name, i, variant, g.edge_index, g.num_nodes,
                             args.time_limit))
        refs = {i: (int((g.y == 1).sum()), float(g.gap)) for i, g in enumerate(graphs)}
        print(rng_name, 'references (size, gap):', refs, flush=True)

    summary_path = os.path.splitext(args.out)[0] + '_summary.csv'
    with open(args.out, 'w', newline='') as ft, \
            open(summary_path, 'w', newline='') as fs:
        wt, ws = csv.writer(ft), csv.writer(fs)
        wt.writerow(['range', 'graph', 'variant', 'seconds', 'incumbent', 'bound'])
        ws.writerow(['range', 'graph', 'variant', 'n', 'cover', 'gap', 'final_bound',
                     'first_bound', 'seconds_to_final_incumbent'])
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(args.processes, initializer=graph.init_worker) as pool:
            for rng_name, i, variant, n, size, gap, trace in \
                    pool.imap_unordered(solve, jobs):
                for t, inc, bnd in trace:
                    wt.writerow([rng_name, i, variant, f'{t:.1f}', inc, f'{bnd:.3f}'])
                t_final = next((t for t, inc, _ in trace if inc <= size + 1e-6), None)
                ws.writerow([rng_name, i, variant, n, size, f'{gap:.4f}',
                             f'{trace[-1][2]:.3f}' if trace else '',
                             next((f'{b:.3f}' for _, _, b in trace if b > -1e30), ''),
                             '' if t_final is None else f'{t_final:.1f}'])
                ft.flush()
                fs.flush()
                print(f'{rng_name} #{i} {variant}: cover {size}, gap {gap:.4f}, '
                      f'final incumbent at {t_final and round(t_final, 1)} s', flush=True)
