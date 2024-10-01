import argparse

parser = argparse.ArgumentParser(
    description='Trains a GNN to solve a given CO problem with RL.')
parser.add_argument('-e', '--epochs', type=int, default=300,
                    help='the n paramether of G(n,p) model')
parser.add_argument('-p', type=float, default=.15,
                    help='the p paramether of G(n,p) model')
parser.add_argument('-n', type=int, default=10,
                    help='the n paramether of G(n,p) model')
parser.add_argument('-s', '--sample_size', type=int, default=10000,
                    help='the size of the sample to be generated.')
args = parser.parse_args()

if __name__ == '__main__':
    from datetime import datetime
    from functools import partial
    from multiprocessing import Pool

    from tqdm import tqdm
    from torch import save

    from graph import prepare_graph
    from rl import RLAgent, MDSRL, train_rl_agent

    date = str(datetime.now())[:16]
    date = date.replace(':', '')
    print(f'Starting script. Experiment {date}. '
          f'Sample of {args.sample_size} graphs from the G({args.n}, {args.p}) '
          f'distribution. \n')

    with Pool() as p:
        gf = partial(prepare_graph, n=args.n, p=args.p, g_nx=True)
        graphs = list(tqdm(
            p.imap_unordered(gf, range(args.sample_size)),
            total=args.sample_size, unit='graph'))

    gnn = RLAgent(c_in=graphs[0].x.size(dim=1), c_hidden=64, c_out=32)
    rl_agent = MDSRL(gnn)
    train_rl_agent(rl_agent, graphs, args.epochs, args.n)
    save(rl_agent, f'./experiments/{date}_rlmodel.pt')
