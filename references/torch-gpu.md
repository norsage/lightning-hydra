# torch + PyG on GPU: which versions to install

Checked on 2026-10-04 with the probe `probes/gpu-smoke`. The environment
layout: `templates/project/pyproject.toml`.

## Versions

The CUDA version of the wheels is set by the NVIDIA driver on the nodes
(`nvidia-smi`, the `Driver Version` line).

| driver | wheels' CUDA | torch | pyg-lib | torch index for Linux | checked |
|---|---|---|---|---|---|
| 535 (CUDA 12.2) | 12.6 | 2.11 | 0.6.0 | `download.pytorch.org/whl/cu126` | A100, works |
| 535 | 12.6 | 2.12–2.14 | 0.7.0–0.9.0 | `cu126` | `CUDA error: device kernel image is invalid` in the first pyg-lib kernel |
| 580+ (CUDA 13) | 13.0 | 2.14 | 0.9 | PyPI | not checked |

- CUDA 12.6 runs on driver 535 through minor version compatibility.
  CUDA 13 needs driver 580+.
- pyg-lib 0.7.0 and later are built with `--compress-mode=size` and without
  `libcudart`; the likely cause of the error on 535 is compressed kernels
  the driver cannot read. Not proven.
- pyg-lib 0.6.0 is built only for torch 2.10 and 2.11. The index
  `torch-2.11.0+cu126` also has 0.7–0.9, hence `==0.6.0` in the template.
- The next pyg-lib drops CUDA 12: on driver 535 there is nothing to upgrade
  to.
- torch from PyPI for Linux is built for CUDA 13.0. CUDA 12 needs the
  PyTorch index: `cu126` goes up to torch 2.14.1, `cu128` up to 2.11.

## PyG packages

- torch-cluster and torch-spline-conv moved into pyg-lib 0.6.0,
  torch-scatter into 0.7.0. The last torch-cluster wheels are for torch
  2.11, torch-scatter and torch-sparse for 2.12. The template has none of
  them.
- PyG 2.8 without extensions: `scatter`, `softmax`, GCNConv, GATConv,
  SAGEConv, TransformerConv and pooling work on CPU, CUDA and MPS.
- `radius_graph`, `knn_graph`, `radius`, `knn` need `pyg-lib>=0.6.0`.
- pyg-lib is only on `data.pyg.org` (a flat index per torch + CUDA pair);
  it is not on PyPI. The macOS arm64 wheels are in the CPU index.

## The `pyproject.toml` layout

- `environments`: lock for Mac and Linux only. Without it `uv lock` fails
  on Windows, where pyg-lib is not found.
- `required-environments`: uv must find a wheel for macOS arm64 and Linux
  x86_64. Without it uv picks the highest local version (`+pt211cpu`, Linux
  only), and `uv sync` fails on Mac.
- Indexes with `explicit = true` and `sys_platform` markers in
  `[tool.uv.sources]`: on Linux torch from `cu126`, on Mac from PyPI.

## Pitfalls

- **Neighbour search is not implemented on MPS.** `radius_graph` and
  `knn_graph` fail with `NotImplementedError: The operator 'pyg::radius' is
  not currently implemented for the MPS device`. Build the graph on CPU and
  move it to the device; neighbour search on MPS and in pure torch is slower
  than CPU at every size checked.
- **`max_num_neighbors`.** pyg-lib on CUDA preallocates
  `atoms × max_num_neighbors` indices: `10**6` on 16,000 atoms asked for
  119 GiB. Use a realistic estimate (64–256 is enough for 5 Å in a protein).
- **Broken downloads from `data.pyg.org`.** From some networks CloudFront
  serves the index but cuts wheel files at ~16 KB: `uv lock` fails with
  `Failed to read from zip file`, `uv sync` with a timeout. Not every time.
  Workaround: another network, or build the lock and the environment where
  downloads work (on cluster nodes they do).

## Neighbour search speed (A100, torch 2.11, pyg-lib 0.6.0)

radius 5 Å, `max_num_neighbors=256`, uniform points at atomic density.
pyg-lib and torch-cluster are the same kernel, the same speed.

| atoms × graphs | radius pyg-lib | radius pure torch |
|---|---|---|
| 500 × 32 | 0.56 ms | 12.4 ms |
| 10,000 × 4 | 4.1 ms | 14.0 ms |
| 100,000 × 1 | 41.5 ms | 244 ms |

Pure torch (`cdist` in blocks) is 3–6 times slower; with
`compute_mode="donot_use_mm_for_euclid_dist"` it gives the same edges.
