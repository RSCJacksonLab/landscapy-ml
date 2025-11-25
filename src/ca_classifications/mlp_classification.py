from __future__ import annotations

from typing import Iterable, List, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceMLPClassifier(pl.LightningModule):
    """Simple MLP classifier for sequence embeddings."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_sizes: Iterable[int] | None = (256, 128),
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        if num_features <= 0:
            raise ValueError("num_features must be positive")
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than 1 for classification")

        layers: List[nn.Module] = []
        input_dim = num_features
        for h in hidden_sizes or []:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = h
        layers.append(nn.Linear(input_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _shared_step(self, batch):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()
        return loss, acc

    def training_step(self, batch, batch_idx: int):
        loss, acc = self._shared_step(batch)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        loss, acc = self._shared_step(batch)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/acc", acc, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx: int):
        loss, acc = self._shared_step(batch)
        self.log("test/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test/acc", acc, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )


class SequenceMLPEnsembleClassifier(pl.LightningModule):
    """Deep ensemble of simple MLP classifiers for uncertainty estimation."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        num_models: int = 5,
        hidden_sizes: Iterable[int] | None = (256, 128),
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        if num_features <= 0:
            raise ValueError("num_features must be positive")
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than 1 for classification")
        if num_models < 1:
            raise ValueError("num_models must be at least 1")

        self.models = nn.ModuleList(
            [
                SequenceMLPClassifier(
                    num_features=num_features,
                    num_classes=num_classes,
                    hidden_sizes=hidden_sizes,
                    dropout=dropout,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                ).net
                for _ in range(num_models)
            ]
        )
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return [m(x) for m in self.models]

    def _shared_step(self, batch):
        x, y = batch
        logits_list = self.forward(x)
        losses = [F.cross_entropy(logits, y) for logits in logits_list]
        loss = torch.stack(losses).mean()
        probs = torch.stack([F.softmax(logits, dim=-1) for logits in logits_list], dim=0)
        mean_probs = probs.mean(dim=0)
        preds = mean_probs.argmax(dim=-1)
        acc = (preds == y).float().mean()
        return loss, acc, probs

    def training_step(self, batch, batch_idx: int):
        loss, acc, _ = self._shared_step(batch)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        loss, acc, _ = self._shared_step(batch)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/acc", acc, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx: int):
        loss, acc, _ = self._shared_step(batch)
        self.log("test/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test/acc", acc, prog_bar=True, on_step=False, on_epoch=True)

    def predict_with_uncertainty(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ensemble mean probs and variance across models."""
        self.eval()
        with torch.no_grad():
            probs = torch.stack([F.softmax(m(x), dim=-1) for m in self.models], dim=0)
            mean_probs = probs.mean(dim=0)
            var_probs = probs.var(dim=0)
        return mean_probs, var_probs

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Alias for predict_with_uncertainty for consistency."""
        return self.predict_with_uncertainty(x)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
