"""GPU smoke: torch + PyG + pyg-lib on CUDA, neighbor search speed vs pure torch.

Run inside one of variants/ (uv run --locked python ../../probe.py [OUT_DIR]).
torch-cluster is timed only if the variant installs it. Writes
OUT_DIR/gpu_smoke.json, the current directory by default.
"""

import json
import os
import sys
import time

import pyg_lib
import torch

try:
    import torch_cluster
except ImportError:  # gone after torch 2.11: merged into pyg-lib
    torch_cluster = None
import torch_geometric
from torch_geometric.nn import GCNConv, knn_graph, radius_graph


def radius_graph_torch(pos, r, batch, max_rows=4096):
    src, dst, s0 = [], [], 0
    for s1 in torch.bincount(batch).cumsum(0).tolist():
        p = pos[s0:s1]
        for a in range(0, p.size(0), max_rows):
            i, j = (torch.cdist(p[a : a + max_rows], p) < r).nonzero(as_tuple=True)
            keep = (i + a) != j
            dst.append(i[keep] + a + s0)
            src.append(j[keep] + s0)
        s0 = s1
    return torch.stack([torch.cat(src), torch.cat(dst)])


def bench(fn, *args, reps=5, **kwargs):
    fn(*args, **kwargs)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        out = fn(*args, **kwargs)
    torch.cuda.synchronize()
    return out, 1000 * (time.perf_counter() - t) / reps


info = {
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "pyg": torch_geometric.__version__,
    "pyg_lib": pyg_lib.__version__,
    "torch_cluster": torch_cluster.__version__ if torch_cluster else None,
    "cuda_available": torch.cuda.is_available(),
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "bf16": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else None,
}
print(info, flush=True)
dev = "cuda"
x = torch.randn(1000, 16, device=dev)
ei = torch.randint(0, 1000, (2, 8000), device=dev)
info["gcnconv"] = list(GCNConv(16, 8).to(dev)(x, ei).shape)

rows = []
torch.manual_seed(0)
for n, b in [(500, 32), (2000, 8), (10000, 4), (30000, 1), (100000, 1)]:
    pos = (torch.rand(n * b, 3) * (n ** (1 / 3)) * 2.2).to(dev)
    batch = torch.arange(b, device=dev).repeat_interleave(n)
    ref, t_lib = bench(radius_graph, pos, 5.0, batch, max_num_neighbors=256)
    out, t_torch = bench(radius_graph_torch, pos, 5.0, batch)
    _, t_knn = bench(knn_graph, pos, 16, batch)
    tc, t_tc = (
        bench(torch_cluster.radius_graph, pos, 5.0, batch, max_num_neighbors=256) if torch_cluster else (None, None)
    )
    row = {
        "atoms": n,
        "graphs": b,
        "edges": ref.size(1),
        "same_count": ref.size(1) == out.size(1),
        "radius_pyg_lib_ms": round(t_lib, 2),
        "radius_torch_ms": round(t_torch, 2),
        "knn16_pyg_lib_ms": round(t_knn, 2),
        "radius_torch_cluster_ms": round(t_tc, 2) if t_tc is not None else None,
        "torch_cluster_edges": tc.size(1) if tc is not None else None,
    }
    print(row, flush=True)
    rows.append(row)
info["bench"] = rows
out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
with open(os.path.join(out_dir, "gpu_smoke.json"), "w") as f:
    json.dump(info, f, indent=2)
