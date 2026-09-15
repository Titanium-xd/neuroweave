import pandas as pd
import scipy.sparse as sp
import numpy as np
import glob

n_paths = list(glob.glob('data/raw/malecns_e2e_validation/neurons_*.parquet'))
e_paths = list(glob.glob('data/raw/malecns_e2e_validation/edges_*.parquet'))

neurons = pd.read_parquet(n_paths[0]).head(150)
all_edges = pd.read_parquet(e_paths[0])
body_ids = set(neurons['bodyId'])
edges = all_edges[all_edges['bodyId_pre'].isin(body_ids) & all_edges['bodyId_post'].isin(body_ids)]

n = len(neurons)
e = len(edges)
density = e / (n * (n - 1))
mean_degree = (e * 2) / n

print(f'nodes={n}')
print(f'edges={e}')
print(f'density={density:.5f}')
print(f'mean_degree={mean_degree:.2f}')

bid_to_idx = {int(b): i for i, b in enumerate(neurons['bodyId'])}
rows = edges['bodyId_pre'].map(bid_to_idx).to_numpy(dtype=int)
cols = edges['bodyId_post'].map(bid_to_idx).to_numpy(dtype=int)
col = 'signed_weight' if 'signed_weight' in edges.columns else 'weight'
vals = edges[col].to_numpy(dtype=float)
print(f'weight_col={col}')
print(f'weight_min={vals.min():.2f}')
print(f'weight_max={vals.max():.2f}')
adj = sp.coo_matrix((np.ones(e), (rows, cols)), shape=(n, n)).tocsr()
out_deg = np.array(adj.sum(axis=1)).flatten()
in_deg  = np.array(adj.sum(axis=0)).flatten()
print(f'max_out_degree={out_deg.max():.0f}')
print(f'max_in_degree={in_deg.max():.0f}')
print(f'body_ids_sample={list(neurons["bodyId"].head(5))}')
print('DONE')
