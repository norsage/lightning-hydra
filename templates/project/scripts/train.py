"""Training: the single entry point.

    python scripts/train.py                       # the default config
    python scripts/train.py experiment=example    # an experiment
    python scripts/train.py -m seed=1,2,3         # a series of jobs

Everything that is instantiated is described in configs/ with `_target_`;
here is only the order of steps.
"""

import gc
import logging
from pathlib import Path

import hydra
import lightning.pytorch as L
import torch
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def resolve_ckpt_path(cfg: DictConfig) -> str | None:
    """The checkpoint to resume fit from, or None to train from scratch.

    A docking point: run tracking replaces this function with its own
    (for example, last.ckpt of the run being resumed).
    """
    if cfg.ckpt_path is None:
        return None
    if cfg.ckpt_path == "last":
        last = Path(cfg.paths.ckpt_dir) / "last.ckpt"
        if last.exists():
            return str(last)
        log.info("No %s, training from scratch", last)
        return None
    return str(cfg.ckpt_path)


def instantiate_callbacks(cfg: DictConfig | None) -> list[L.Callback]:
    if not cfg:
        return []
    return [hydra.utils.instantiate(c) for c in cfg.values() if c is not None]


def free_memory() -> None:
    """Free memory between --multirun jobs in one process."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    if cfg.seed is not None:
        L.seed_everything(cfg.seed, workers=True)

    datamodule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)
    callbacks = instantiate_callbacks(cfg.get("callbacks"))
    logger = hydra.utils.instantiate(cfg.logger) if cfg.get("logger") else False
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    try:
        if trainer.logger is not None:
            trainer.logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False))
        trainer.fit(model, datamodule=datamodule, ckpt_path=resolve_ckpt_path(cfg))
        if cfg.test:
            # The best checkpoint if there is one; with fast_dev_run or
            # trainer.enable_checkpointing=false, the current weights.
            ckpt = trainer.checkpoint_callback
            best = "best" if ckpt is not None and ckpt.best_model_path else None
            trainer.test(model, datamodule=datamodule, ckpt_path=best)
    finally:
        del trainer, model, datamodule, callbacks, logger
        free_memory()


if __name__ == "__main__":
    main()
