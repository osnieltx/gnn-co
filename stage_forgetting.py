"""Per stage: apx-ratio just before a run left it vs at the end of the run (seed runs, from wandb).

    python stage_forgetting.py out.csv
"""
import re, sys
import pandas as pd
import wandb

RUNS = {'episode': ['2026-09-29-2242', '2026-09-29-2246', '2026-09-29-2250'],
        'transition': ['2026-09-29-2243', '2026-09-29-2247', '2026-09-29-2251'],
        'current': ['2026-09-29-2244', '2026-09-29-2248', '2026-09-29-2252'],
        'replace': ['2026-09-29-2245', '2026-09-29-2249', '2026-09-29-2253']}
K = 5
api = wandb.Api()
rows = []
for cfg, ids in RUNS.items():
    for seed, rid in enumerate(ids, 1):
        run = api.runs('osnieltx-uff/lightning_logs', filters={'display_name': rid})[0]
        h = run.history(samples=100000, pandas=True)
        val = h[h['val_apx_ratio_all'].notna()].copy()
        stages = sorted((c for c in val.columns if re.fullmatch(r'val_apx_ratio/\d+-\d+', c)),
                        key=lambda c: int(c.split('/')[1].split('-')[0]))
        val['st'] = val['curriculum/stage'].ffill().astype(int)
        final = int(val['st'].iloc[-1])
        end = val.tail(K)
        for i, c in enumerate(stages):
            on = val[val['st'] == i]
            rows.append(dict(cfg=cfg, seed=seed, run=run.state, final_stage=final, stage=c.split('/')[1],
                             idx=i, trained=i <= final,
                             learned=on[c].tail(K).mean() if len(on) else float('nan'),
                             end=end[c].mean()))
df = pd.DataFrame(rows)
df['forget'] = df['end'] - df['learned']
df.to_csv(sys.argv[1], index=False)
pd.set_option('display.width', 250)
t = df[df.trained & (df.idx < df.final_stage)]  # stages the run trained on and then left
print('Stages the run left: apx-ratio when leaving -> at the end (mean of 5 validations each)')
print(t.pivot_table(index=['cfg', 'seed'], columns='stage', values='forget',
                    aggfunc='first').reindex(columns=[s for s in ['10-10','15-20','40-50','50-100','100-200','200-300','300-400'] if s in set(t.stage)]).round(3).to_string())
print('\nEnd-of-run apx-ratio per stage (all stages, * = never trained on)')
e = df.assign(v=df.apply(lambda r: f"{r.end:.3f}" + ('' if r.trained else '*'), axis=1))
print(e.pivot_table(index=['cfg', 'seed'], columns='stage', values='v', aggfunc='first')
       .reindex(columns=['10-10','15-20','40-50','50-100','100-200','200-300','300-400','400-500']).to_string())
