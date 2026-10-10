# configs

Hydra configs for `scripts/train.py`.

```
train.yaml       choice of groups, seed, ckpt_path, paths and output directories
data/            DataModule                   (data=<file>)
model/           LightningModule, network, optimizer, scheduler (model=<file>)
trainer/         lightning Trainer            (trainer=debug: one batch per stage)
callbacks/       Lightning callbacks as a mapping (callbacks=none: no callbacks)
logger/          Lightning logger, CSV by default (logger=none: no logger)
experiment/      experiments: overrides with `# @package _global_`
```

## Running

```bash
python scripts/train.py                          # all defaults
python scripts/train.py experiment=example       # an experiment
python scripts/train.py trainer=debug            # a check that everything builds
python scripts/train.py -m seed=1,2,3            # a series of jobs
python scripts/train.py model.optimizer.lr=1e-4  # any field from the command line
```

The final config without running: `python scripts/train.py --cfg job`.

## Experiments

One experiment is one file `experiment/<name>.yaml`, first line
`# @package _global_`, with only what differs from `train.yaml`. The `name`
field sets the output directory. A new experiment starts as a copy of
`experiment/example.yaml`.

## Outputs

| what | where |
|---|---|
| a single run | `logs/runs/<name>/<date_time>/` |
| `-m` | `logs/multiruns/<name>/<date_time>/<job number>/` |
| final config and overrides | `<output>/.hydra/` |
| metrics (CSV) | `<output>/csv/version_0/metrics.csv` |
| checkpoints | `<output>/checkpoints/` (`last.ckpt` and the best by `val/loss`) |

## Docking points

The skeleton trains without external services. Other skills plug in by
editing configs, not code:

| point | default | what plugs in |
|---|---|---|
| `logger/` | CSV | a file with the tracker's logger, `logger=<file>` |
| `hydra.callbacks` in `train.yaml` | empty | the run tracker's callback |
| `callbacks/` | `ModelCheckpoint` in `paths.ckpt_dir` | your own callbacks, another checkpoint directory |
| `ckpt_path` | `null` | a path, `last`, or your own `resolve_ckpt_path` in `train.py` |
| `hydra.run.dir`, `hydra.sweep.dir`, `paths.log_dir` | `logs/` | a remote transport's directories |
