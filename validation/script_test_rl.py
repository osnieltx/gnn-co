import glob
import os
from datetime import datetime
from pathlib import Path

import networkx as nx
import torch
import yaml
from tqdm import tqdm

from dql import DQNLightning
from graph import (milp_solve_mds, milp_solve_mvc, is_ds, generate_graphs,
                   dominating_potential, is_vc, covering_potential)


def greedy(g, checker, attr_func):
    g = g.clone()
    s = set()
    while not checker(g.nx, s):
        s.add(torch.argmax(g.x[:, 1]).item())
        g.x[:, 1] = attr_func(g.edge_index, s)
    return s


def local_search(g: nx.Graph, s: set, checker):
    s_ = s.copy()
    s_list = list(s)

    for u in s_list:
        s__ = s_.copy()
        s__.remove(u)
        if checker(g, s__):
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
                if checker(g, s__):
                    s_ = s__
                    break
    return s_


if __name__ == '__main__':
    problems = {'mvc': (milp_solve_mvc, is_vc, covering_potential),
                'mds': (milp_solve_mds, is_ds, dominating_potential)}

    ids = []
    cutoff_datetime = datetime.strptime("2025-08-08-2050", "%Y-%m-%d-%H%M")
    for entry in os.listdir('./experiments'):
        full_path = os.path.join('./experiments', entry)
        if os.path.isdir(full_path):
            creation_datetime = datetime.strptime(entry, "%Y-%m-%d-%H%M")

            if creation_datetime >= cutoff_datetime:
                ids.append(entry)

    with open('./experiments/rl_results.csv', 'w') as results:
        results.write(
            'model,n,p,apx-ratio,avg-S_gnn,avg-S*,greedy-s,local-s,problem,attr,comment\n')
        for i in tqdm(ids, unit='model'):
            n, p = None, None
            script_params = {'problem': None, 'attr': None}
            try:
                # os.system(
                #     f'scp -r osnielteixeira2@200.20.15.153:~/experiments/{i}/ '
                #     '~/Documents/UFF/mestrado/2o\ Sem/EO/gnn-co/experiments/')
                base_path = f'./experiments/{i}/version_0'
                list_of_checkpoints = glob.glob(base_path + '/checkpoints/*')
                latest_chekpoint = max(list_of_checkpoints, key=os.path.getctime)
                hparams_path = base_path + '/hparams.yaml'

                dqn_model: DQNLightning = DQNLightning.load_from_checkpoint(
                    latest_chekpoint, map_location=torch.device("cpu"),
                    hparams_file=hparams_path, s=1, warm_start_steps=0)
                dqn_model.eval()

                script_params = torch.load(f'../experiments/{i}/params.pt')
                solver, checker, attr_func = problems[script_params['problem']]
                attr_func = attr_func if script_params['attr'] else None

                conf = yaml.safe_load(Path(hparams_path).read_text())
                n, p = conf['n'], conf['p']
                tt_g = 800
                graphs = generate_graphs(range(n, conf['delta_n'] + 1), p, tt_g,
                                         solver, attrs=attr_func)

                print(f'solving {script_params["problem"]}')
                for g in graphs:
                    if attr_func:
                        g.greedy_s = greedy(g, checker, attr_func)
                    else:
                        g.greedy_s = set()

                    # Perform an episode of actions
                    s = set()
                    dqn_model.agent.reset(g)
                    for step in range(n):
                        action = dqn_model.agent.get_action(dqn_model.net, 0, 'cpu')
                        s.add(action)
                        g.x = g.x.clone()
                        g.x[action][0] = 1
                        dqn_model.agent.state = g
                        if checker(g.nx, s):
                            break
                    g.s = s

                    if attr_func:
                        g.local_s = local_search(g.nx, s)
                    else:
                        g.local_s = set()

                metrics = (
                    f'{i},{n},{p},'
                    f'{sum(len(g.s) / (g.y==1).sum() for g in graphs) / tt_g:.3f},'
                    f'{sum(len(g.s) for g in graphs) / tt_g:.2f},'
                    f'{sum((g.y == 1).sum() for g in graphs) / tt_g:.2f},'
                    f'{sum(len(g.greedy_s) / (g.y==1).sum() for g in graphs)/ tt_g:.3f},'
                    f'{sum(len(g.local_s) / (g.y==1).sum() for g in graphs) / tt_g:.3f},'
                    f'{script_params["problem"]},{script_params["attr"]},,\n')
            except Exception as e:
                print(e)
                metrics = (
                    f'{i},{n},{p},'
                    f','
                    f','
                    f','
                    f','
                    f','
                    f'{script_params["problem"]},{script_params["attr"]},'
                    f'failed with {e}\n')
            results.write(metrics)
            results.flush()
