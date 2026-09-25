import argparse
import os
from datetime import datetime, timedelta

import torch

from curricula import curricula, parse_graph_size, stage_ranges
from dql import DQNLightning, StageEarlyStopping
from ppo import PPO

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(
    description='Trains a RL Agente with GNN to solve a given CO problem.')
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
parser.add_argument('--curriculum', choices=curricula.keys(), default=None,
                    help='use a named curriculum for the graph sizes; overrides -n.')
parser.add_argument('-d', '--devices', type=int, default=1,
                    help='number of gpu devices.')
parser.add_argument('--accelerator', type=str, default='auto',
                    choices=['auto', 'cpu', 'gpu', 'mps'],
                    help='compute backend for training.')
parser.add_argument('--eps_last_frame', type=int, default=15000,
                    help='The global step at which epsilon reaches its minimum (eps_end).')
parser.add_argument('--lr', type=float, default=7e-4,
                    help='the learning rate for the optimizer.')
parser.add_argument('-p', type=float, default=.15,
                    help='the p paramether of G(n,p) model')
parser.add_argument('-n', nargs='+', type=parse_graph_size, default=[10],
                    help='list of n parameters or ranges (e.g., 10 20,25 30) for G(n,p)')
parser.add_argument('-s', type=int, default=10000,
                    help='the size of the sample to be generated (PPO only; '
                         'DQN samples a new graph every episode).')
parser.add_argument('-v', type=int, default=batch_size,
                    help='the size of the validation sample to be generated.')
parser.add_argument('--val_dir', default=None,
                    help='load a validation set made by build_val_set.py '
                         'instead of generating one; ignores -v.')
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
parser.add_argument('--no_msg_norm', dest='msg_norm', action='store_false',
                    default=True,
                    help="don't divide S2V neighbor sums by the graph's mean "
                         "degree (old behaviour; embeddings explode on large graphs).")
parser.add_argument('--graph_pool', default='mean', choices=['add', 'mean'],
                    help='how node embeddings are pooled into the graph embedding.')
parser.add_argument('--loss', default='huber', choices=['mse', 'huber'],
                    help='TD loss.')
parser.add_argument('--reward_norm', default='graph', choices=['graph', 'stage'],
                    help="step reward -1/n of the graph being solved ('graph') or "
                         "-1/max n of the current stage ('stage', as in S2V-DQN).")
parser.add_argument('--stage_patience', type=int, default=20,
                    help='advance to the next curriculum stage after this many '
                         'validations without improving the stage apx-ratio, '
                         'even if the target was not reached (0 disables).')
parser.add_argument('--grad_clip', type=float, default=10,
                    help='clip the gradient norm to this value (0 disables).')

args = parser.parse_args()
if args.curriculum:
    args.n = curricula[args.curriculum]

if __name__ == '__main__':
    import pytz
    import warnings
    from pytorch_lightning import Trainer
    from torch_geometric.loader import DataLoader
    from pytorch_lightning.callbacks import ModelCheckpoint
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
    params.pop('curriculum')
    devices = params.pop('devices')
    accelerator = params.pop('accelerator')
    v = params.pop('v')
    val_dir = params.pop('val_dir')
    grad_clip = params.pop('grad_clip') or None
    rl_alg = algorithms[params.pop('rl_alg')]
    problem = params.pop('problem')
    solver, check_solved, attr = problems[problem]
    attr_func = attr if params.pop('attr') else None
    max_epochs = 10 * 10 ** 4
    model = rl_alg(**params, graph_attr=attr_func, check_solved=check_solved,
                   max_epochs=max_epochs)

    # Per-stage metric, reset on every advance: val_apx_ratio_all mixes in
    # stages not trained yet (or being forgotten) and hides progress.
    early_stop_callback = StageEarlyStopping(
        monitor="val_apx_ratio/stage",
        min_delta=0.0001,
        patience=55,  # * check_val_every_n_epoch
        verbose=True,
        mode="min",
        check_on_train_epoch_end=False  # Check after validation
    )
    if accelerator == 'cpu':
        device = torch.device('cpu')
    elif accelerator == 'gpu':
        device = torch.device('cuda')
    elif accelerator == 'mps':
        device = torch.device('mps')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    accelerator = device.type

    # logger = CSVLogger('experiments/', name=date)
    wandb_logger = WandbLogger(log_model="all", name=date)
    wandb_logger.experiment  # forces wandb.init() before define_metric
    wandb.define_metric("val_apx_ratio_all", summary="min")
    trainer = Trainer(
        callbacks=[
            ModelCheckpoint(save_weights_only=True,
                            mode="min",
                            monitor="val_apx_ratio_all"),
            early_stop_callback
        ],
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
        enable_progress_bar=True,
        logger=wandb_logger,
        log_every_n_steps=1,
        check_val_every_n_epoch=400,
        gradient_clip_val=grad_clip,
    )

    graphs = []
    if val_dir:
        meta = torch.load(f'{val_dir}/meta.pt')
        if (meta['problem'], meta['p']) != (problem, params['p']):
            raise ValueError(f'{val_dir} was built for {meta}, not '
                             f'problem={problem}, p={params["p"]}')
    for n_range in stage_ranges(params['n_sizes']):
        if val_dir:
            stage_graphs = torch.load(f'{val_dir}/{n_range.start}_{n_range.stop}.pt')
            if attr_func:
                for g in stage_graphs:
                    g.x = torch.cat((g.x, attr_func(g.edge_index).unsqueeze(1)), 1)
        else:
            # One folder per stage, since files are named by index within it.
            stage_dir = f'{dataset_dir}/{n_range.start}_{n_range.stop}'
            os.makedirs(stage_dir)
            stage_graphs = generate_graphs(n_range, params['p'], v, solver=solver, dataset_dir=stage_dir, attrs=attr_func)
        graphs.extend(stage_graphs)
    graphs = [g.to(device) for g in graphs]
    val_data_loader = DataLoader(graphs, batch_size=params['batch_size'])
    trainer.fit(model, val_dataloaders=val_data_loader)
