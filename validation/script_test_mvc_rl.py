import sys
import os

# Get the directory of the current script
current_script_dir = os.path.dirname(__file__)

# Get the path to the parent directory (one level up)
parent_dir = os.path.abspath(os.path.join(current_script_dir, os.pardir))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

import glob
import os
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from dql import DQNLightning
from graph import is_vc, generate_graphs, milp_solve_mvc

ids = [
    '2025-07-17-0923',
    '2025-07-17-0924',
    '2025-07-17-0925',
    '2025-07-17-1019',
    '2025-07-17-1026',
    '2025-07-17-1032',
    '2025-07-17-1210',
    '2025-07-17-1240',
    '2025-07-17-1253',
    '2025-07-17-1515',
    '2025-07-17-1623',
    '2025-07-17-1644',
    '2025-07-17-1947',
    '2025-07-17-2134',
    '2025-07-17-2244',
]


# def greedy(g):
#     g = g.clone()
#     s = set()
#     while not is_ds(g.nx, s):
#         s.add(torch.argmax(g.x[:, 1]).item())
#         g.x[:, 1] = dominable_neighbors(g.edge_index, s)
#     return s
#
#
# def local_search(g: nx.Graph, s: set):
#     s_ = s.copy()
#     s_list = list(s)
#
#     for u in s_list:
#         s__ = s_.copy()
#         s__.remove(u)
#         if is_ds(g, s__):
#             s_ = s__
#
#     s_list = list(s_)
#     for i, u in enumerate(s_list[:-1]):
#         for v in s_list[i+1:]:
#             if not(u in s_ and v in s_):
#                 continue
#             for w in set(g[u]) & set(g[v]):
#                 s__ = s_.copy()
#                 s__.remove(u)
#                 s__.remove(v)
#                 s__.add(w)
#                 if is_ds(g, s__):
#                     s_ = s__
#                     break
#     return s_


if __name__ == '__main__':
    with open('./validation/results_mvc.csv', 'w') as results:
        results.write('model,n,p,apx-ratio,avg-S_gnn,avg-S*\n')
        for i in tqdm(ids, unit='model'):
            os.system(
                f'scp -r osnielteixeira2@200.20.15.153:~/experiments/{i}/ '
                '~/Documents/UFF/mestrado/2o\ Sem/EO/gnn-co/experiments/')
            base_path = f'./experiments/{i}/version_0'
            list_of_checkpoints = glob.glob(base_path + '/checkpoints/*')
            latest_chekpoint = max(list_of_checkpoints, key=os.path.getctime)
            hparams_path = base_path + '/hparams.yaml'

            dqn_model: DQNLightning = DQNLightning.load_from_checkpoint(
                latest_chekpoint, map_location=torch.device("cpu"),
                hparams_file=hparams_path, s=1, warm_start_steps=0,
                check_solved=milp_solve_mvc,
            )
            dqn_model.eval()

            conf = yaml.unsafe_load(Path(hparams_path).read_text())
            n, p = conf['n'], conf['p']
            tt_g = 800
            graphs = generate_graphs(range(n, conf['delta_n'] + 1), p, tt_g,
                                     milp_solve_mvc, attrs=conf['graph_attr'])

            print(f'solving mvc')
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
                    if is_vc(g.nx, s):
                        break
                g.s = s

                # g.local_s = local_search(g.nx, s)

            metrics = (
                f'{i},{n},{p},'
                f'{sum(len(g.s) / (g.y==1).sum() for g in graphs) / tt_g:.3f},'
                f'{sum(len(g.s) for g in graphs) / tt_g:.2f},'
                f'{sum((g.y == 1).sum() for g in graphs) / tt_g:.2f},'
                # f'{sum(len(g.greedy_s) / (g.y==1).sum() for g in graphs)/ tt_g:.3f},'
                # f'{sum(len(g.local_s) / (g.y==1).sum() for g in graphs) / tt_g:.3f}'
                '\n'
            )

            results.write(metrics)
            results.flush()
