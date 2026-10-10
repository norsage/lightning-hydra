"""Stub DataModule: synthetic regression y = Xw + noise.

It lets the skeleton train right after it is deployed. Replace it with your
own DataModule and update configs/data/.
"""

import lightning.pytorch as L
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


class RandomRegressionDataModule(L.LightningDataModule):
    def __init__(
        self,
        num_samples: int = 1024,
        num_features: int = 16,
        batch_size: int = 64,
        num_workers: int = 0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str | None = None) -> None:
        hp = self.hparams
        g = torch.Generator().manual_seed(hp["seed"])
        x = torch.randn(hp["num_samples"], hp["num_features"], generator=g)
        w = torch.randn(hp["num_features"], 1, generator=g)
        y = x @ w + 0.1 * torch.randn(hp["num_samples"], 1, generator=g)
        n_val = n_test = hp["num_samples"] // 10
        n_train = hp["num_samples"] - n_val - n_test
        self.train_set, self.val_set, self.test_set = random_split(
            TensorDataset(x, y), [n_train, n_val, n_test], generator=g
        )

    def _loader(self, dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.hparams["batch_size"],
            shuffle=shuffle,
            num_workers=self.hparams["num_workers"],
            persistent_workers=self.hparams["num_workers"] > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_set)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_set)
