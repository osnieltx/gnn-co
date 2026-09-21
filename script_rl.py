import argparse
import os
from datetime import datetime, timedelta

import torch

from dql import DQNLightning
from ppo import PPO

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(
    description='Trains a RL Agente with GNN to solve a given CO problem.')
def parse_graph_size(arg):
    """Parses comma-separated strings into tuples, or returns an int."""
    if ',' in arg:
        return tuple(map(int, arg.split(',')))
    return int(arg)

algorithms = {'DQN': DQNLightning, 'PPO': PPO}
problems = {'mvc', 'mds'}
batch_size = 512
parser.add_argument('-a', '--algorithm', dest='rl_alg', default='DQN',
                    choices=algorithms.keys(), help='the RL algorithm train.')
parser.add_argument('-b', '--batch_size', type=int, default=batch_size,
                    help='the batch size.')
parser.add_argument('--curriculum_mode', type=str, default='replace',
                    choices=['replace', 'cumulative'],
                    help='Curriculum mode: "replace" discards old sizes;'
                         '"cumulative" retains all previous graph sizes.')
parser.add_argument('-d', '--devices', type=int, default=1,
                    help='number of gpu devices.')
parser.add_argument('--eps_last_frame', type=int, default=15000,
                    help='The global step at which epsilon reaches its minimum (eps_end).')
parser.add_argument('--lr', type=float, default=7e-4,
                    help='the learning rate for the optimizer.')
parser.add_argument('-p', type=float, default=.15,
                    help='the p paramether of G(n,p) model')
parser.add_argument('-n', nargs='+', type=parse_graph_size, default=[10],
                    help='list of n parameters or ranges (e.g., 10 20,25 30) for G(n,p)')
parser.add_argument('-s', type=int, default=10000,
                    help='the size of the sample to be generated.')
parser.add_argument('-v', type=int, default=batch_size,
                    help='the size of the validation sample to be generated.')
parser.add_argument('--problem', default='mds', choices=problems,
                    help='the CO to train.')
parser.add_argument('--no_attr', dest='attr', action='store_false',
                    default=True, help='if the graph have attributes')
parser.add_argument('--sync_rate', type=int, default=1000,
                    help='Target network sync frequency.')
parser.add_argument('--n_step', type=int, default=5,
                    help='N-step return size.')
parser.add_argument('--num_iterations', type=int, default=5,
                    help='S2V message passing steps.')
parser.add_argument('--gamma', type=float, default=1.0, help='Discount factor.')

args = parser.parse_args()

if __name__ == '__main__':
    import pytz
    import warnings
    from pytorch_lightning import Trainer
    from torch_geometric.loader import DataLoader
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    from pytorch_lightning.loggers import WandbLogger
    import wandb

    wandb.login()

    from graph import (generate_graphs, milp_solve_mds, milp_solve_mvc,
                       is_vc_vectorized, is_ds_vectorized, covering_potential,
                       dominating_potential)

    problems = {'mvc': (milp_solve_mvc, is_vc_vectorized, covering_potential),
                'mds': (milp_solve_mds, is_ds_vectorized, dominating_potential)}

    warnings.filterwarnings("ignore", ".*does not have many workers.*")

    minutes = 0
    while True:
        now = datetime.now(pytz.timezone('Brazil/East'))
        now += timedelta(minutes=minutes)
        date = str(now)[:16]
        date = date.replace(':', '').replace(' ', '-')
        model_dir = f'experiments/{date}'
        try:
            os.makedirs(model_dir)
        except FileExistsError:
            minutes += 1
        else:
            break

    print(f'Starting script. Experiment {date}\n'
          f'Batch size: {args.batch_size}.\n'
          f'RL alg: {args.rl_alg}')
    dataset_dir = f'{model_dir}/dataset'
    os.makedirs(dataset_dir)
    params = vars(args)
    torch.save(params, f'{model_dir}/params.pt')
    params['n_sizes'] = params.pop('n')
    devices = params.pop('devices')
    v = params.pop('v')
    rl_alg = algorithms[params.pop('rl_alg')]
    problem = params.pop('problem')
    solver, check_solved, attr = problems[problem]
    attr_func = attr if params.pop('attr') else None
    max_epochs = 9 * 10 ** 4
    model = rl_alg(**params, graph_attr=attr_func, check_solved=check_solved,
                   max_epochs=max_epochs)

    early_stop_callback = EarlyStopping(
        monitor="val_apx_ratio_all",
        min_delta=0.0001,
        patience=55,  # * check_val_every_n_epoch
        verbose=True,
        mode="min",
        check_on_train_epoch_end=False  # Check after validation
    )
    # logger = CSVLogger('experiments/', name=date)
    wandb_logger = WandbLogger(log_model="all", name=date)
    trainer = Trainer(
        callbacks=[
            ModelCheckpoint(save_weights_only=True,
                            mode="min",
                            monitor="val_apx_ratio_all"),
            early_stop_callback
        ],
        accelerator='auto',
        devices=devices,
        max_epochs=max_epochs,
        enable_progress_bar=True,
        logger=wandb_logger,
        log_every_n_steps=1,
        check_val_every_n_epoch=400,
    )

    graphs = []
    for n in params['n_sizes']:
        n_range = range(n, n+1) if type(n) is int else range(n[0], n[1]) if type(n) is tuple else n
        graphs.extend(generate_graphs(n_range, params['p'], v, solver=solver, dataset_dir=dataset_dir, attrs=attr_func))
    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available() else 'cpu')
    graphs = [g.to(device) for g in graphs]
    val_data_loader = DataLoader(graphs, batch_size=params['batch_size'])
    trainer.fit(model, val_dataloaders=val_data_loader)
