"""Picks a checkpoint from a training run using its validation history.

script_rl.py keeps a checkpoint from every validation
(experiments/<id>/checkpoints/step=<N>.ckpt); this scores each validation on
a chosen criterion and prints the best step and its checkpoint:

- a stage, e.g. --metric 400-500 (a size specialist), or
- --metric mean: the mean of the per-stage apx-ratios (a generalist), with
  --exclude for stages whose ratio is too coarse to compare (e.g. 10-10,
  where one extra node costs ~0.15).

--smooth k averages the criterion over the last k validations before picking,
so a single lucky validation doesn't win. The chosen value is still biased
low (it is the minimum of a noisy series): report results on a separate test
set, not this number.
"""
import argparse
import glob
import os
import re

import wandb

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument('experiment', help='experiment id, e.g. 2026-09-25-0620')
parser.add_argument('--metric', default='mean',
                    help="a stage like '400-500', or 'mean' over stages.")
parser.add_argument('--exclude', nargs='*', default=['10-10'],
                    help="stages left out of 'mean'.")
parser.add_argument('--smooth', type=int, default=1,
                    help='rolling mean over this many validations.')
parser.add_argument('--project', default='osnieltx-uff/lightning_logs')
parser.add_argument('--experiments_dir', default='experiments')


def stage_columns(history):
    return [c for c in history.columns
            if re.fullmatch(r'val_apx_ratio/\d+-\d+', c)]


if __name__ == '__main__':
    args = parser.parse_args()
    runs = wandb.Api().runs(args.project,
                            filters={'display_name': args.experiment})
    if len(runs) != 1:
        raise SystemExit(f'{len(runs)} wandb runs named {args.experiment}')
    history = runs[0].history(samples=100000, pandas=True)
    val = history[history['val_apx_ratio_all'].notna()].copy()

    if args.metric == 'mean':
        cols = [c for c in stage_columns(val)
                if c.split('/')[1] not in args.exclude]
        val['criterion'] = val[cols].mean(axis=1)
    else:
        val['criterion'] = val[f'val_apx_ratio/{args.metric}']
    val['criterion'] = val['criterion'].rolling(args.smooth).mean()

    best = val.loc[val['criterion'].idxmin()]
    step = int(best['trainer/global_step'])
    # Checkpoints are named by the step count after that validation's step.
    ckpt_dir = os.path.join(args.experiments_dir, args.experiment, 'checkpoints')
    saved = {int(re.search(r'step=(\d+)', p).group(1)): p
             for p in glob.glob(f'{ckpt_dir}/step=*.ckpt')}
    ckpt = saved.get(step + 1) or saved.get(step)

    print(f"criterion={args.metric} smooth={args.smooth}: best at step {step}, "
          f"value {best['criterion']:.4f} (stage {int(best['curriculum/stage'])})")
    print(ckpt or f'no checkpoint for step {step} in {ckpt_dir}')
