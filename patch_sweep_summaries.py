import argparse
import re

import wandb


def _digits(s):
    return int(re.sub(r"\D", "", s))


def patch_summaries(project_path, sweep_id, metric, cutoff, apply):
    api = wandb.Api()
    sweep = api.sweep(f"{project_path}/{sweep_id}")
    cutoff_key = _digits(cutoff)

    for run in sweep.runs:
        try:
            run_key = _digits(run.name)
        except ValueError:
            print(f"skip {run.id} ({run.name}): name isn't a date")
            continue

        if run_key >= cutoff_key:
            continue

        history = run.history(keys=[metric])
        if history.empty:
            print(f"skip {run.id} ({run.name}): no '{metric}' history")
            continue

        best = history[metric].min()
        current = run.summary.get(metric)
        if current is not None and current == best:
            continue

        print(f"{run.id} ({run.name}): {metric} {current} -> {best}")
        if apply:
            run.summary[metric] = best
            run.summary.update()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-path", default="osnieltx-uff/lightning_logs")
    parser.add_argument("--sweep-id", default="2yqvtuw2")
    parser.add_argument("--metric", default="val_apx_ratio_all")
    parser.add_argument("--cutoff", default="2026-09-21-22-00")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    patch_summaries(
        args.project_path, args.sweep_id, args.metric, args.cutoff, args.apply
    )
