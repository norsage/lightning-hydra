---
name: lightning-hydra
description: Training skeleton for PyTorch Lightning + Hydra projects with a torch + PyG environment that installs on Mac (MPS) and Linux (CUDA) from one uv.lock. One entry point scripts/train.py, code in src/<pkg>/{data,model,utils}, Hydra config groups data/ model/ trainer/ callbacks/ logger/ and experiment/ files, outputs under logs/. Trains locally with no external service; tracking, HPC transport and data plug in at docking points by config. Use when starting a training project on Lightning and Hydra, when adding Lightning or Hydra to a project, when a project has scripts/train.py with configs/train.yaml, when choosing torch, PyG or pyg-lib versions for a CUDA driver, when uv lock fails on pyg-lib, or when connecting MLflow, a transport or data to such a project.
---

# lightning-hydra

Lays out a training project and keeps it laid out the same way, so that a
person or an agent finds the entry point, the experiments and the outputs
without reading the code. The skill knows no tracking server, cluster or
dataset: those are other skills, connected at the [docking
points](#docking-points).

| file | what |
|---|---|
| [`templates/project/`](templates/project/) | the skeleton; `__pkg__` stands for the package name |
| [`scripts/scaffold.py`](scripts/scaffold.py) | deploys the skeleton, or compares a project with it |
| [`references/torch-gpu.md`](references/torch-gpu.md) | driver → CUDA → torch → pyg-lib, traps |

Templates are copied, not installed as a package: the project owns its copy
and decides when to take changes.

## Deploying into a project

1. Pick the package name (lowercase, underscores) and the Python version
   (default 3.12; see [Environment](#environment) for the others), then:
   ```bash
   python3 <skill>/scripts/scaffold.py <project> --pkg <name> [--python 3.13]
   ```
   A version the skeleton does not run on is refused with the reason: tell
   the user that reason and the supported versions, and let them choose.
   It writes the files that are missing and lists the ones that `exists`.
   Those it does not touch: in an existing project merge them by hand
   (`pyproject.toml`: dependencies, `[tool.uv]`, indexes and sources;
   `.gitignore`: `logs/`). `--dry-run` shows the list first.
2. `uv lock && uv sync` in the project. Needs `data.pyg.org` and
   `download.pytorch.org`; if `uv lock` fails with
   `Failed to read from zip file`, see [Environment](#environment).
3. `uv run ruff check . && uv run pyright && uv run pytest`: all three are
   in the `dev` group and the skeleton passes them. The smoke test runs
   `scripts/train.py trainer=debug` (one batch of train, val and test on CPU).
4. Replace the stubs (`RandomRegressionDataModule`, `MLP`, `Regressor`)
   with the project's code and point `configs/data/`, `configs/model/` at
   it. `src/README.md` in the project says how.
5. Add to the project's `AGENTS.md` (or `CLAUDE.md`):
   `Training follows the lightning-hydra skill: entry point scripts/train.py, configs in configs/ (see configs/README.md), outputs in logs/.`

No credentials and no administrator are needed.

## Layout

```
src/<pkg>/{data,model,utils}/   created from configs via _target_; LightningModule in model/lightning.py
scripts/train.py                the only entry point; imports nothing from the package
configs/train.yaml              group choice, seed, ckpt_path, paths, Hydra output directories
configs/{data,model,trainer,callbacks,logger}/
configs/experiment/<name>.yaml  `# @package _global_`, only what differs from train.yaml
tests/test_train.py             smoke: trainer=debug
```

Rules the skeleton keeps; keep them when changing it:

- Everything that is built is a `_target_` in a group; `train.py` only
  orders the steps. Optimizer and scheduler with `_partial_: true`.
- `self.hparams["name"]`, not `self.hparams.name`: pyright types
  `hparams` as a mapping.
- Lightning callbacks are a dict, so an experiment can drop one
  (`callbacks.early_stopping=null`). Whole groups off: `callbacks=none`,
  `logger=none` (`logger=null` fails in Hydra 1.3).
- `accelerator: auto`, `deterministic: warn` (`true` fails on PyG's scatter
  on CUDA).
- Outputs: `logs/runs/<name>/<date_time>/`, multirun
  `logs/multiruns/<name>/<date_time>/<job>/`; config in `.hydra/`, metrics
  in `csv/`, checkpoints in `checkpoints/`.

The project's `configs/README.md` is the user-facing version of this.

## Docking points

The skeleton fills each with a local default; another skill replaces the
default by editing configs, not `train.py` (the one exception is marked).

| point | default | who replaces it |
|---|---|---|
| `configs/logger/` | `csv.yaml`, CSV in the output directory | run tracker: its logger as a new file, `logger=<file>` |
| `hydra.callbacks` in `train.yaml` | `{}` | run tracker: its Hydra callback (a run per job, a parent per sweep) |
| `configs/callbacks/model_checkpoint` | `ModelCheckpoint` into `paths.ckpt_dir` | run tracker: checkpoints by run id |
| `resolve_ckpt_path` in `scripts/train.py` | `ckpt_path`: `null`, a path, or `last` | run tracker: its resume function (code, one function) |
| `hydra.run.dir`, `hydra.sweep.dir`, `paths.log_dir` | under `logs/` | transport: where it collects results from |
| `configs/data/` fields | stub without files | data: paths to imported datasets |

Roles: the skill requires none; it offers these points to a run tracker
(mlflow-tracking, `integrations/lightning.md` and `integrations/hydra.md`),
a transport (runs the project on a remote cluster) and a data source (a
dataset registry). Recipes that name a particular one belong to the
extension or to that skill, not here.

## Environment

Python is pinned by `.python-version`, which `scaffold.py` writes, together
with `requires-python` and pyright's `pythonVersion`. `uv.lock` pins the
packages, not the interpreter: without `.python-version` uv takes whatever
installed Python fits `requires-python`, and on a machine without one (a
remote task) downloads the newest allowed, so the laptop and the task run
different Pythons. Commit `.python-version`.

| Python | |
|---|---|
| 3.12 | default |
| 3.11, 3.13 | supported: the lock resolves; `trainer=debug` passes on Linux (a cluster task, CPU, 2026-10-05); on Mac not run yet |
| 3.14 | no: Hydra 1.3 fails before reading the config (argparse `badly formed help string`) |
| 3.10 and older | no: older than the skeleton is tested on |

`requires-python` keeps the upper bound (`<3.14`) while Hydra 1.3 does not
run on 3.14; raise it, and `SUPPORTED_PYTHON` in `scaffold.py`, when it does.

`templates/project/pyproject.toml` locks torch + PyG for macOS arm64 and
Linux x86_64 in one `uv.lock`, without extras: indexes with platform
markers, and `environments` / `required-environments` in `[tool.uv]`.

- Linux: torch from the PyTorch `cu126` index, pyg-lib 0.6.0 from
  `data.pyg.org` (`torch-2.11.0+cu126`). Mac: torch from PyPI (MPS),
  pyg-lib from the CPU index.
- torch 2.11 + pyg-lib 0.6.0 is the newest pair whose kernels run on
  driver 535. A newer driver allows newer versions: the table in
  [`references/torch-gpu.md`](references/torch-gpu.md).
- No torch-scatter, torch-sparse, torch-cluster: PyG 2.8 uses torch and
  pyg-lib.
- On MPS `radius_graph` and `knn_graph` raise `NotImplementedError`: build
  graphs on CPU.
- `data.pyg.org` truncates downloads at ~16 KB from some networks, and
  `uv lock`/`uv sync` fail with `Failed to read from zip file`. Retry from
  another network or through a mirror; the skill does not work around it.

## Checking

- Locally: `uv run ruff check .`, `uv run pyright`, `uv run pytest` in the project; `python scripts/train.py
  trainer=debug` for one batch; `python scripts/train.py --cfg job` for the
  final config without running.
- On GPU: not yet part of the skill.

## Updating a deployed project

```bash
python3 <skill>/scripts/scaffold.py <project> --pkg <name> --compare
```

prints `same`, `differs` or `missing` for each template file and writes
nothing. Take `missing` files with a plain run of `scaffold.py`; merge
`differs` by hand, since the project has changed its copy on purpose.
Changes to `pyproject.toml` versions need `uv lock` again.
