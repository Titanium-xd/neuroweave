import json, glob, pathlib, numpy as np

# Load and summarize campaign_1 results
c1 = {}
for f in glob.glob('artifacts/campaign_1/*.json'):
    name = pathlib.Path(f).stem
    d = json.load(open(f))
    agg = d.get('aggregate', {})
    meta = d.get('metadata', {})
    c1[name] = {
        'mean': round(agg.get('test_acc_mean', 0)*100, 1),
        'sd':   round(agg.get('test_acc_std', 0)*100, 1),
        'ci95': round(agg.get('test_acc_ci95', 0)*100, 1),
        'params': meta.get('trainable_parameters'),
        'frozen': meta.get('frozen_parameters'),
        'seeds': len(d.get('seeds', [])),
        'time':  round(agg.get('wall_time_s_mean', 0), 1),
    }

print('=== CAMPAIGN 1 (T-001) ===')
for k,v in sorted(c1.items()):
    print(f'{k}: mean={v["mean"]}%, sd={v["sd"]}%, ci95=+/-{v["ci95"]}%, params={v["params"]}, frozen={v["frozen"]}, seeds={v["seeds"]}, time={v["time"]}s')

# Load confirmation T-002
c3 = {}
for f in glob.glob('artifacts/confirmation_t002/*.json'):
    name = pathlib.Path(f).stem
    d = json.load(open(f))
    agg = d.get('aggregate', {})
    meta = d.get('metadata', {})
    dev_accs = [sr['best_dev_acc'] for sr in d.get('results_per_seed', [])]
    c3[name] = {
        'mean': round(agg.get('test_acc_mean', 0)*100, 1),
        'sd':   round(agg.get('test_acc_std', 0)*100, 1),
        'ci95': round(agg.get('test_acc_ci95', 0)*100, 1),
        'dev_mean': round(np.mean(dev_accs)*100, 1) if dev_accs else None,
        'params': meta.get('trainable_parameters'),
        'frozen': meta.get('frozen_parameters'),
        'seeds': len(d.get('seeds', [])),
        'time':  round(agg.get('wall_time_s_mean', 0), 1),
    }

print('\n=== CONFIRMATION T-002 ===')
for k,v in sorted(c3.items()):
    print(f'{k}: mean={v["mean"]}%, sd={v["sd"]}%, ci95=+/-{v["ci95"]}%, dev={v["dev_mean"]}%, params={v["params"]}, frozen={v["frozen"]}')
