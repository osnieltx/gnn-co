import glob
import os
from pathlib import Path

import networkx as nx
import torch
import yaml
from tqdm import tqdm

from dql import DQNLightning
from graph import milp_solve_mds, is_ds, generate_graphs, dominanting_potential

ids = [
    '2025-04-16-1736',
    '2025-04-17-0803',
    '2025-04-17-1012',
    '2025-04-17-1253',
    '2025-04-17-1715',
    '2025-04-18-0645',
    '2025-04-18-0651',
    '2025-04-18-1307',
    '2025-04-30-0504',
    '2025-04-18-1306',
    '2025-04-21-0809',
    '2025-04-23-0844',
    '2025-04-21-0810',
    '2025-04-23-0846',
    '2025-04-24-0802',
    '2025-04-21-0814',
    '2025-04-23-0848',
    '2025-04-24-0807',
    '2025-04-30-0500',
    '2025-04-16-1844',
    '2025-04-17-0752',
    '2025-04-18-0648',
    '2025-04-24-0804',
    '2025-04-24-1016',
    '2025-04-24-1942',
    '2025-04-27-0923',
    '2025-04-24-1946',
    '2025-04-27-0925',
    '2025-04-27-0926',
    '2025-04-30-0513',
    '2025-04-29-1059',
    '2025-04-29-1101',
    '2025-04-30-0502',
    '2025-04-29-1103',
    '2025-04-29-1903',
    '2025-04-29-1104',
    '2025-04-30-0509',
    '2025-04-29-1358',
    '2025-04-29-1559',
    '2025-04-29-1722',
    '2025-04-29-1900',
    '2025-04-29-1724',
    '2025-04-29-2033',
    '2025-04-29-2045',
    '2025-04-29-2101',
]


def greedy(g):
    g = g.clone()
    s = set()
    while not is_ds(g.nx, s):
        s.add(torch.argmax(g.x[:, 1]).item())
        g.x[:, 1] = dominanting_potential(g.edge_index, s)
    return s


def local_search(g: nx.Graph, s: set):
    s_ = s.copy()
    s_list = list(s)

    for u in s_list:
        s__ = s_.copy()
        s__.remove(u)
        if is_ds(g, s__):
            s_ = s__

    s_list = list(s_)
    for i, u in enumerate(s_list[:-1]):
        for v in s_list[i+1:]:
            if not(u in s_ and v in s_):
                continue
            for w in set(g[u]) & set(g[v]):
                s__ = s_.copy()
                s__.remove(u)
                s__.remove(v)
                s__.add(w)
                if is_ds(g, s__):
                    s_ = s__
                    break
    return s_


if __name__ == '__main__':
    with open('./results.csv', 'w') as results:
        results.write(
            'model,n,p,apx-ratio,avg-S_gnn,avg-S*,local-s\n')
        for i in tqdm(ids, unit='model'):
            os.system(
                f'scp -r osnielteixeira2@200.20.15.153:~/experiments/{i}/ '
                '~/Documents/UFF/mestrado/2o\ Sem/EO/gnn-co/experiments/')
            base_path = f'../experiments/{i}/version_0'
            list_of_checkpoints = glob.glob(base_path + '/checkpoints/*')
            latest_chekpoint = max(list_of_checkpoints, key=os.path.getctime)
            hparams_path = base_path + '/hparams.yaml'

            dqn_model: DQNLightning = DQNLightning.load_from_checkpoint(
                latest_chekpoint, map_location=torch.device("cpu"),
                hparams_file=hparams_path, s=1, warm_start_steps=0)
            dqn_model.eval()

            conf = yaml.safe_load(Path(hparams_path).read_text())
            n, p = conf['n'], conf['p']
            tt_g = 800
            graphs = generate_graphs(range(n, conf['delta_n'] + 1), p, tt_g,
                                     milp_solve_mds, attrs=conf['graph_attr'])

            print(f'solving mds')
            for g in graphs:
                # g.greedy_s = greedy(g)

                # Perform an episode of actions
                s = set()
                dqn_model.agent.reset(g)
                for step in range(n):
                    action = dqn_model.agent.get_action(dqn_model.net, 0, 'cpu')
                    s.add(action)
                    g.x = g.x.clone()
                    g.x[action][0] = 1
                    dqn_model.agent.state = g
                    if is_ds(g.nx, s):
                        break
                g.s = s

                g.local_s = local_search(g.nx, s)

            metrics = (
                f'{i},{n},{p},'
                f'{sum(len(g.s) / (g.y==1).sum() for g in graphs) / tt_g:.3f},'
                f'{sum(len(g.s) for g in graphs) / tt_g:.2f},'
                f'{sum((g.y == 1).sum() for g in graphs) / tt_g:.2f},'
                # f'{sum(len(g.greedy_s) / (g.y==1).sum() for g in graphs)/ tt_g:.3f},'
                f'{sum(len(g.local_s) / (g.y==1).sum() for g in graphs) / tt_g:.3f}\n')

            results.write(metrics)
            results.flush()
