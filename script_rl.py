import argparse
import os
from datetime import datetime

import torch

from dql import DQNLightning

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(
    description='Trains a RL Agente with GNN to solve a given CO problem.')
parser.add_argument('-b', '--batch_size', type=int, default=500,
                    help='the batch size.')
parser.add_argument('-d', '--devices', type=int, default=1,
                    help='number of gpu devices.')
parser.add_argument('-p', type=float, default=.15,
                    help='the p paramether of G(n,p) model')
parser.add_argument('-n', type=int, default=10,
                    help='the n paramether of G(n,p) model')
parser.add_argument('-s', type=int, default=10000,
                    help='the size of the sample to be generated.')
args = parser.parse_args()


if __name__ == '__main__':
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger

    date = str(datetime.now())[:16]
    date = date.replace(':', '')
    print(f'Starting script. Experiment {date}. '
          f'Batch size: {args.batch_size}.')

    model_dir = f'experiments/{date}'
    os.makedirs(model_dir)
    params = vars(args)
    torch.save(params, f'{model_dir}/params.pt')

    devices = params.pop('devices')
    model = DQNLightning(**params)
    logger = CSVLogger('experiments/', name=date)
    trainer = Trainer(
        val_check_interval=100,
        callbacks=[
            ModelCheckpoint(save_weights_only=True,
                            mode="min",
                            monitor="loss")],
        accelerator='gpu',
        devices=devices,
        max_epochs=1500,
        enable_progress_bar=True,
        logger=logger,
        profiler="simple"
    )

    trainer.fit(model)
