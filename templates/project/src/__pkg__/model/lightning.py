"""LightningModule: training steps, metrics, the optimizer.

The network, optimizer and scheduler come from the config: the network as a
ready module, the optimizer and scheduler as functions (`_partial_: true`)
that lack only the model's parameters and the optimizer, respectively.
"""

from collections.abc import Callable

import lightning.pytorch as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn
from torch.nn import functional as F


class Regressor(L.LightningModule):
    def __init__(
        self,
        net: nn.Module,
        optimizer: Callable[..., torch.optim.Optimizer],
        scheduler: Callable[..., torch.optim.lr_scheduler.LRScheduler] | None = None,
        monitor: str = "val/loss",
    ) -> None:
        super().__init__()
        # The checkpoint keeps the network as weights, the optimizer as state.
        self.save_hyperparameters(ignore=["net", "optimizer", "scheduler"])
        self.net = net
        self.optimizer = optimizer
        self.scheduler = scheduler

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _step(self, batch, stage: str) -> torch.Tensor:
        x, y = batch
        loss = F.mse_loss(self(x), y)
        self.log(f"{stage}/loss", loss, prog_bar=stage != "test", batch_size=len(x))
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = self.optimizer(params=self.parameters())
        if self.scheduler is None:
            return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": self.scheduler(optimizer), "monitor": self.hparams["monitor"]},
        }
