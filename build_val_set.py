"""Builds a fixed, solved validation set once, so training runs can load it
with --val_dir instead of re-solving MILPs on every run.

Each stage is saved as <out>/<start>_<stop>.pt (stop exclusive, as in -n),
next to a meta.pt with the problem, p and time limit used. Graphs are stored
without attributes; script_rl adds them at load time when needed.

Solves are capped at --time_limit seconds and keep the best solution found,
as in Khalil et al. (2017); each graph's MIP gap is stored in g.gap.
"""
import argparse
import os

import torch

from curricula import curricula, parse_graph_size, stage_ranges

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument('--out', required=True, help='output directory.')
parser.add_argument('--curriculum', choices=curricula.keys(), default=None,
                    help='use a named curriculum for the graph sizes; overrides -n.')
parser.add_argument('-n', nargs='+', type=parse_graph_size, default=[10],
                    help='list of n parameters or ranges (e.g., 10 20,25 30).')
parser.add_argument('-p', type=float, default=.15,
                    help='the p paramether of G(n,p) model')
parser.add_argument('-v', type=int, default=100,
                    help='number of graphs per stage.')
parser.add_argument('--problem', default='mvc', choices=['mvc', 'mds'])
parser.add_argument('--time_limit', type=float, default=120,
                    help='seconds per MILP solve.')
parser.add_argument('--processes', type=int, default=None,
                    help='parallel solves (default: min(cores, 16)).')
parser.add_argument('--chunk', type=int, default=0,
                    help='save progress every this many graphs, as '
                         '<start>_<stop>.part<k>.pt, and resume from existing '
                         'parts on restart (0: save each stage only at the end). '
                         'Use a multiple of --processes.')
parser.add_argument('--store_nx', action='store_true',
                    help='also store a networkx copy of each graph (g.nx); '
                         'several GB for 1000+ node graphs, and rebuildable '
                         'from edge_index.')


if __name__ == '__main__':
    args = parser.parse_args()
    import glob
    from graph import iter_graphs, milp_solve_mds, milp_solve_mvc
    solver = {'mvc': milp_solve_mvc, 'mds': milp_solve_mds}[args.problem]
    sizes = curricula[args.curriculum] if args.curriculum else args.n

    os.makedirs(args.out, exist_ok=True)
    torch.save({'problem': args.problem, 'p': args.p,
                'time_limit': args.time_limit, 'store_nx': args.store_nx},
               f'{args.out}/meta.pt')

    for n_range in stage_ranges(sizes):
        path = f'{args.out}/{n_range.start}_{n_range.stop}.pt'
        if os.path.exists(path):
            print(f'{path} exists, skipping.')
            continue
        # Resume from the parts a previous (interrupted) run already saved.
        part_prefix = f'{args.out}/{n_range.start}_{n_range.stop}.part'
        parts = sorted(glob.glob(f'{part_prefix}*.pt'),
                       key=lambda f: int(f[len(part_prefix):-3]))
        graphs = [g for f in parts for g in torch.load(f)]
        if graphs:
            print(f'{path}: resuming with {len(graphs)} graphs from {len(parts)} parts')
        buffer = []
        for g in iter_graphs(
                n_range, args.p, args.v - len(graphs), solver=solver,
                processes=args.processes, g_nx=args.store_nx,
                solver_kwargs={'time_limit': args.time_limit, 'return_gap': True}):
            buffer.append(g)
            if args.chunk and len(buffer) == args.chunk:
                torch.save(buffer, f'{part_prefix}{len(parts)}.pt')
                parts.append(f'{part_prefix}{len(parts)}.pt')
                graphs += buffer
                buffer = []
        graphs += buffer
        torch.save(graphs, path)
        for f in parts:
            os.remove(f)

        gaps = torch.tensor([g.gap for g in graphs])
        print(f'{path}: {(gaps < 1e-4).sum()}/{len(gaps)} proven optimal, '
              f'mean gap {gaps.mean():.4f}, max gap {gaps.max():.4f}')
