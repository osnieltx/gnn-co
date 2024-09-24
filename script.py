import argparse
import os
from datetime import datetime

from graph import create_graph, milp_solve, milp_solve_mds,\
    clustering_coefficient, jaccard_coefficient

solvers = {'mvc': milp_solve, 'mds': milp_solve_mds}
parser = argparse.ArgumentParser(
    description='Trains a GNN to solve a given CO problem.')
parser.add_argument('-b', '--batch_size', type=int, default=1,
                    help='the batch size.')
parser.add_argument('--problem', dest='milp_solver', default='mds',
                    choices=solvers.keys(), help='the CO to train.')
parser.add_argument('-d', '--devices', type=int, default=1,
                    help='number of gpu devices.')
parser.add_argument('-p', type=float, default=.15,
                    help='the p paramether of G(n,p) model')
parser.add_argument('-n', type=int, default=10,
                    help='the n paramether of G(n,p) model')
args = parser.parse_args()

if __name__ == '__main__':
    print(f'Starting script. Training for {args.milp_solver.upper()}. '
          f'Graphs sampled from the G({args.n}, {args.p}) distribution.')
    import torch
    from tqdm import tqdm

    from pyg import geom_data
    from models import train_node_classifier

    graphs_to_generate = 10000
    torch.multiprocessing.set_sharing_strategy('file_system')

    graphs = []
    n = args.n
    max_d = 0
    for g_i in tqdm(range(graphs_to_generate), unit='graph'):
        edge_index = create_graph(n, args.p)
        s = solvers[args.milp_solver](edge_index, n)
        y = torch.FloatTensor([[n in s] for n in range(n)])
        x = torch.FloatTensor([[1]]*n)
        # x = clustering_coefficient(edge_index)[:, 1]
        # d_g = x.max().item()
        # if d_g > max_d:
        #     max_d = d_g
        tg = geom_data.Data(x=x, y=y, edge_index=edge_index, )
        graphs.append(tg)

    date = str(datetime.now())[:16]
    date = date.replace(':', '')
    model_dir = f'experiments/{date}'
    os.makedirs(model_dir)
    print('Saving dataset... 💽')
    torch.save(graphs, f'{model_dir}/dataset.pt')

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
        logger_name=date,
        batch_size=args.batch_size,
    )
