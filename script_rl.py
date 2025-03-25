import argparse
import os
from datetime import datetime

import torch

from dql import DQNLightning
from ppo import PPO

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(
    description='Trains a RL Agente with GNN to solve a given CO problem.')
algorithms = {'DQN': DQNLightning, 'PPO': PPO}
parser.add_argument('-a', '--algorithm', dest='rl_alg', default='ppo',
                    choices=algorithms.keys(), help='the RL algorithm train.')
parser.add_argument('-b', '--batch_size', type=int, default=5000,
                    help='the batch size.')
parser.add_argument('-d', '--devices', type=int, default=1,
                    help='number of gpu devices.')
parser.add_argument('-p', type=float, default=.15,
                    help='the p paramether of G(n,p) model')
parser.add_argument('-n', type=int, default=10,
                    help='the n paramether of G(n,p) model')
parser.add_argument('--delta_n', type=int, default=10,
                    help='the max n paramether of G(n,p) model')
parser.add_argument('-s', type=int, default=10000,
                    help='the size of the sample to be generated.')
parser.add_argument('-v', type=int, default=300,
                    help='the size of the validation sample to be generated.')
args = parser.parse_args()


if __name__ == '__main__':
    import pytz
    import warnings
    from pytorch_lightning import Trainer
    from torch_geometric.loader import DataLoader
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger

    from graph import generate_graphs, milp_solve_mds

    warnings.filterwarnings("ignore", ".*does not have many workers.*")

    date = str(datetime.now(pytz.timezone('Brazil/East')))[:16]
    date = date.replace(':', '').replace(' ', '-')
    print(f'Starting script. Experiment {date}\n'
          f'Batch size: {args.batch_size}.')

    model_dir = f'experiments/{date}'
    os.makedirs(model_dir)
    dataset_dir = f'{model_dir}/dataset'
    os.makedirs(dataset_dir)
    params = vars(args)
    torch.save(params, f'{model_dir}/params.pt')

    devices = params.pop('devices')
    v = params.pop('v')
    rl_alg = algorithms[params.pop('rl_alg')]
    graph_attr = ['dominable_neighbors']
    model = rl_alg(**params, graph_attr=graph_attr)
    logger = CSVLogger('experiments/', name=date)
    trainer = Trainer(
        callbacks=[
            ModelCheckpoint(save_weights_only=True,
                            mode="min",
                            monitor="val_apx_ratio")],
        accelerator='gpu',
        devices=devices,
        max_epochs=150000,
        enable_progress_bar=True,
        logger=logger,
        log_every_n_steps=1,
        profiler="simple",
        check_val_every_n_epoch=20,
        gradient_clip_val=0.5,
        gradient_clip_algorithm="value",
    )
    n = params['n']
    delta_n = params['delta_n']
    if delta_n == n:
        delta_n += 1
    n_r = range(n, delta_n)
    graphs = generate_graphs(n_r, params['p'], v, solver=milp_solve_mds,
                             dataset_dir=dataset_dir,
                             attrs=graph_attr)
    val_data_loader = DataLoader(graphs, batch_size=params['batch_size'])

    trainer.fit(model, val_dataloaders=val_data_loader)
