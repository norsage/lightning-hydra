# lightning-hydra

A skill for Claude Code and other agents: a training skeleton on PyTorch
Lightning + Hydra with one entry point, Hydra config groups and
experiments, and a torch + PyG environment that installs on Mac (MPS) and
Linux (CUDA) from one `uv.lock`. It trains locally without external
services; run tracking, HPC transport and data plug in through config. See
[SKILL.md](SKILL.md).

## Install

```sh
bash install.sh            # every project on this machine
bash install.sh --symlink  # link the checkout: the agent reads your working copy

cd my-project
bash /path/to/lightning-hydra/install.sh --project  # into the current directory, committed with this project
```

Re-running replaces the install, after a prompt. A copy leaves out `probes/`
(the GPU compatibility spike) and the test setup.

## Deploying the skeleton

```sh
python3 scripts/scaffold.py ../my-project --pkg my_project
cd ../my-project && uv sync && uv run ruff check . && uv run pyright && uv run pytest
```

Existing files are never overwritten. `--compare` lists what differs
from the current templates when updating a deployed project.

## Development

```sh
uv sync --group templates   # torch, lightning, hydra: pyright resolves the templates
uv run ruff check . && uv run ruff format --check . && uv run pyright
uv run pytest               # scaffold.py and install.sh
LH_E2E=1 uv run pytest      # also deploys a project: uv sync from data.pyg.org, ruff, pyright, pytest there
```
