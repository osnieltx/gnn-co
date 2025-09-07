import glob
import os
import traceback
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import networkx as nx
import torch
import yaml
from tqdm import tqdm

from dql import DQNLightning
from graph import (milp_solve_mds, milp_solve_mvc, is_ds, generate_graphs,
                   dominating_potential, is_vc, covering_potential, load_graph)

torch.multiprocessing.set_sharing_strategy('file_system')


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
    cutoff_datetime = datetime.strptime("2025-09-04-1719", "%Y-%m-%d-%H%M")
    use_validation_ds = False
    for entry in os.listdir('./experiments'):
        full_path = os.path.join('./experiments', entry)
        if os.path.isdir(full_path):
            try:
                creation_datetime = datetime.strptime(entry, "%Y-%m-%d-%H%M")

                if creation_datetime >= cutoff_datetime:
                    ids.append(entry)
            except Exception as e:
                pass

    with open('./experiments/rl_results.csv', 'w') as results:
        results.write(
            'model,n,p,apx-ratio,avg-S_gnn,avg-S*,greedy-s,local-s,problem,attr,comment\n')
        for i in tqdm(ids, unit='model'):
            print(i)
            n, p = None, None
            script_params = {'problem': None, 'attr': None}
            try:
                os.system(
                    f'scp -r osnielteixeira2@200.20.15.153:~/experiments/{i}/ '
                    f'~/Documents/UFF/mestrado/2o\ Sem/EO/gnn-co/experiments/')
                base_path = f'./experiments/{i}/version_0'
                list_of_checkpoints = glob.glob(base_path + '/checkpoints/*')
                latest_chekpoint = max(list_of_checkpoints, key=os.path.getctime)
                hparams_path = base_path + '/hparams.yaml'

                script_params = torch.load(f'./experiments/{i}/params.pt')
                print(script_params)
                problem = script_params.get('problem',
                                            script_params.get('milp_solver'))
                solver, checker, attr_func = problems[problem]
                attr_func = attr_func if script_params['attr'] else None

                dqn_model: DQNLightning = DQNLightning.load_from_checkpoint(
                    latest_chekpoint, map_location=torch.device("cpu"),
                    hparams_file=hparams_path, s=1, warm_start_steps=0,
                    graph_attr=attr_func, check_solved=checker
                )
                dqn_model.eval()

                conf = yaml.full_load(Path(hparams_path).read_text())
                n, p = conf['n'], conf['p']
                if use_validation_ds:
                    print('Loading graphs from experiment directory.')
                    v = script_params['v']
                    tt_g = v
                    get_graph = partial(load_graph,
                                        path=f'./experiments/{i}/dataset')
                    with Pool() as pool:
                        graphs = list(tqdm(
                            pool.imap_unordered(get_graph, range(v)),
                            total=v,
                            unit='graph')
                        )
                else:
                    tt_g = 800
                    graphs = generate_graphs(range(n, conf['delta_n'] + 1), p,
                                             tt_g, solver, attrs=attr_func)

                print(f'solving {problem}')
                for g in tqdm(graphs, unit='graph'):
                    if attr_func:
                        g.greedy_s = greedy(g, checker, attr_func)
                    else:
                        g.greedy_s = set()

                    # Perform an episode of actions
                    dqn_model.agent.reset(g)
                    for step in range(n):
                        _, done = dqn_model.agent.play_validation_step(
                            dqn_model.net, 'cpu'
                        )
                        if done:
                            break
                    indices = (dqn_model.agent.state.x[:, 0] == 1).nonzero(
                        as_tuple=True
                    )[0]
                    s = set(indices.tolist())
                    g.s = s

                    g.local_s = local_search(g.nx, s, checker)

                metrics = (
                    f'{i},{n},{p},'
                    f'{sum(len(g.s) / (g.y==1).sum() for g in graphs) / tt_g:.3f},'
                    f'{sum(len(g.s) for g in graphs) / tt_g:.2f},'
                    f'{sum((g.y == 1).sum() for g in graphs) / tt_g:.2f},'
                    f'{sum(len(g.greedy_s) / (g.y==1).sum() for g in graphs)/ tt_g:.3f},'
                    f'{sum(len(g.local_s) / (g.y==1).sum() for g in graphs) / tt_g:.3f},'
                    f'{problem},{script_params["attr"]},\n')
            except Exception:
                tb = traceback.format_exc()
                e_str = repr(tb)
                print(e_str)
                metrics = (
                    f'{i},{n},{p},'
                    f','
                    f','
                    f','
                    f','
                    f','
                    f'{problem},{script_params["attr"]},'
                    f'failed with {tb}\n')
            results.write(metrics)
            results.flush()