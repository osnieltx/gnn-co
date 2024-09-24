import argparse

from graph import create_graph, milp_solve, milp_solve_mds

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
parser.add_argument('-s', '--sample_size', type=int, default=10000,
                    help='the size of the sample to be generated.')
args = parser.parse_args()

if __name__ == '__main__':
    print(f'Starting script. Training for {args.milp_solver.upper()}. \n'
          f'Sample of {args.sample_size} graphs from the G({args.n}, {args.p}) '
          f'distribution. \n'
          f'Batch size: {args.batch_size}.')
    import torch
    from tqdm import tqdm

    from pyg import geom_data
    from models import train_node_classifier

    torch.multiprocessing.set_sharing_strategy('file_system')

    graphs = []
    n = args.n
    max_d = 0
    for g_i in tqdm(range(args.sample_size), unit='graph'):
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
    )
