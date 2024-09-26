import argparse
import os
from datetime import datetime
from functools import partial
from pathlib import Path

from graph import milp_solve, milp_solve_mds, prepare_graph

solvers = {'mvc': milp_solve, 'mds': milp_solve_mds}
parser = argparse.ArgumentParser(
    description='Trains a GNN to solve a given CO problem.')
parser.add_argument('-b', '--batch_size', type=int, default=1,
                    help='the batch size.')
parser.add_argument('--problem', dest='milp_solver', default='mds',
                    choices=solvers.keys(), help='the CO to train.')
parser.add_argument('-d', '--devices', type=int, default=1,
                    help='number of gpu devices.')
parser.add_argument('--data', type=Path, default=None, help='data source to use')
parser.add_argument('-p', type=float, default=.15,
                    help='the p paramether of G(n,p) model')
parser.add_argument('-n', type=int, default=10,
                    help='the n paramether of G(n,p) model')
parser.add_argument('-s', '--sample_size', type=int, default=10000,
                    help='the size of the sample to be generated.')
args = parser.parse_args()


if __name__ == '__main__':
    import torch
    from tqdm.contrib.concurrent import process_map

    from models import train_node_classifier

    if args.data:
        data_attrs = ['milp_solver', 'p', 'n', 'sample_size']
        print(f'Importing data related params {data_attrs} from {args.data}')
        data_source_args = torch.load(f'{args.data}/params.pt')
        for atr in data_attrs:
            setattr(args, atr, data_source_args[atr])

    print(f'Starting script. Training for {args.milp_solver.upper()}. \n'
          f'Sample of {args.sample_size} graphs from the G({args.n}, {args.p}) '
          f'distribution. \n'
          f'Batch size: {args.batch_size}.')

    torch.multiprocessing.set_sharing_strategy('file_system')

    date = str(datetime.now())[:16]
    date = date.replace(':', '')
    model_dir = f'experiments/{date}'
    os.makedirs(model_dir)
    dataset_dir = f'{model_dir}/dataset'
    os.makedirs(dataset_dir)
    params = vars(args)
    torch.save(params, f'{dataset_dir}/params.pt')

    if args.data:
        get_graphs = lambda i: torch.load(f'{dataset_dir}/{i}.pt')
    else:
        get_graph = partial(prepare_graph, n=n, p=args.p,
                            dataset_dir=dataset_dir,
                            solver=solvers[args.milp_solver])
    graphs = process_map(get_graph, range(args.sample_size), unit='graph',
                         chunksize=1 if args.sample_size <= 1000 else 10)


    n = args.n

    max_d = 0
    # print('Normalizing degrees')
    # for g in graphs:
    #     g.x = g.x.unsqueeze(1) / max_d

    c_in = graphs[0].x.size(dim=1)
    print(f'{max_d=} {c_in=}')
    node_gnn_model, node_gnn_result = train_node_classifier(
        devices=args.devices,
        layer_name="GCN",
        dataset=graphs,
        c_in=c_in,
        c_hidden=30,
        num_layers=20,
        m=16,
        max_epochs=350,
        dp_rate=0,
        batch_size=args.batch_size,
        model_dir=model_dir,
    )
