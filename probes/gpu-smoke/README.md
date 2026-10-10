# GPU smoke: torch + PyG + pyg-lib

Checks that a torch + torch-geometric + pyg-lib combination installs from
`uv.lock` and works on a GPU, and compares the speed of neighbour search:
pyg-lib, torch-cluster (if installed) and a pure torch implementation.
Results and conclusions:
[`references/torch-gpu.md`](../../references/torch-gpu.md).

```
probe.py [OUT_DIR]        checks and timings; writes OUT_DIR/gpu_smoke.json (default .)
variants/<torch>-<pyg-lib>/
  pyproject.toml          one combination of versions, indexes with platform markers
  uv.lock                 locked for Mac arm64 and Linux x86_64
```

Running one variant on a machine with a GPU:

```bash
cd variants/torch211-pyglib060-cluster
uv sync --locked
uv run --locked python ../../probe.py
```

If `uv lock` fails with `Failed to read from zip file`, the network is
cutting downloads from `data.pyg.org`; the lock can be built on the machine
that runs the check and brought back with the results.
