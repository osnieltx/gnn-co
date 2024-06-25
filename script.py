import argparse

from graph import create_graph, milp_solve, milp_solve_mds

solvers = {'mvc': milp_solve, 'mds': milp_solve_mds}
parser = argparse.ArgumentParser(description='Trains a GNN to solve a given CO problem.')
parser.add_argument('-p', '--problem', dest='milp_solver', choices=solvers.keys(),
                    default='mvc', help='the CO to train.')
parser.add_argument('-d', '--devices', type=int,
                    default=1, help='number of gpu devices.')
args = parser.parse_args()

if __name__ == '__main__':
    print(f'Starting script. Training for {args.milp_solver.upper()}.')
    import torch
    from tqdm import tqdm

    from pyg import geom_data
    from models import train_node_classifier

    graphs_to_generate = 10000
    torch.multiprocessing.set_sharing_strategy('file_system')

    graphs = []
    for g_i in tqdm(range(graphs_to_generate), unit='graph'):
        n = 10
        edge_index = create_graph(n)
        s = solvers[args.milp_solver](edge_index, n)

        y = torch.FloatTensor([[n in s] for n in range(n)])
        x = torch.Tensor([[1.]] * 10)
        tg = geom_data.Data(x=x, y=y, edge_index=edge_index)
        graphs.append(tg)

    node_gnn_model, node_gnn_result = train_node_classifier(
        devices=args.devices,
        layer_name="GCN",
        dataset=graphs,
        c_hidden=30,
        num_layers=20,
        node_dim=0,
        m=16,
        max_epochs=300,
        dp_rate=0
    )
