import wandb
import pandas as pd


def find_optimal_config(sweep_id, project_path, tail_fraction=0.2, penalty_lambda=1.5):
    api = wandb.Api()
    sweep = api.sweep(f"{project_path}/{sweep_id}")

    run_scores = []

    for run in sweep.runs:
        # Ignore crashed or incomplete runs
        if run.state != "finished":
            continue

        # Fetch the validation history
        history = run.history(keys=["val_apx_ratio_all"])
        if history.empty:
            continue

        # Extract the tail end of the run
        tail_length = int(len(history) * tail_fraction)
        tail_data = history["val_apx_ratio_all"].tail(tail_length)

        # Calculate the components
        tail_mean = tail_data.mean()
        tail_std = tail_data.std()

        # Apply the composite scoring function
        composite_score = tail_mean + (penalty_lambda * tail_std)

        run_scores.append({
            "run_id": run.id,
            "mean": tail_mean,
            "std": tail_std,
            "score": composite_score,
            "params": run.config
        })

    # Convert to DataFrame and sort by the lowest composite score
    df = pd.DataFrame(run_scores).dropna()
    df_sorted = df.sort_values(by="score", ascending=True).reset_index(drop=True)

    return df_sorted.iloc[0]


# Execution
best_run = find_optimal_config(
    sweep_id="wi92vk1w",
    project_path="osnieltx-uff/lightning_logs"
)

print(f"Optimal Score: {best_run['score']:.4f}")
print("Optimal Parameters:")
for param, value in best_run['params'].items():
    print(f" - {param}: {value}")