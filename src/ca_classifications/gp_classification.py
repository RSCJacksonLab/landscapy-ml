from __future__ import annotations

from typing import Optional, Tuple

import gpytorch
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger


class SequenceGPModel(gpytorch.models.ApproximateGP):
    """Variational GP model for sequence embeddings."""

    def __init__(self, inducing_points: torch.Tensor, num_classes: int) -> None:
        if inducing_points.dim() != 2:
            raise ValueError("inducing_points must have shape [num_inducing, num_features]")

        batch_shape = torch.Size([num_classes])
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=batch_shape
        )
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=True,
            ),
            num_tasks=num_classes,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=batch_shape),
            batch_shape=batch_shape,
        )

    def forward(self, inputs: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(inputs)
        covar_x = self.covar_module(inputs)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SequenceGPClassifier(pl.LightningModule):
    """Lightning module wrapping a GPyTorch softmax classifier."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        *,
        inducing_points: Optional[torch.Tensor] = None,
        num_inducing: int = 64,
        learning_rate: float = 0.01,
        weight_decay: float = 0.0,
        num_data: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["inducing_points"])
        if num_features <= 0:
            raise ValueError("num_features must be positive")
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than 1 for classification")
        if inducing_points is None:
            inducing_points = torch.randn(num_inducing, num_features)

        self.model = SequenceGPModel(inducing_points, num_classes)
        self.likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_classes=num_classes)
        self.elbo = gpytorch.mlls.VariationalELBO(
            self.likelihood, self.model, num_data=num_data or 1
        )
        self._num_data = num_data

    def forward(self, inputs: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return self.model(inputs)

    def _maybe_update_num_data(self, batch_size: int) -> None:
        if self._num_data is None and self.trainer is not None:
            estimated = self.trainer.num_training_batches * batch_size
            if estimated > 0:
                self.elbo.num_data = estimated
                self._num_data = estimated

    def _shared_step(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        inputs, labels = batch
        labels = labels.long().view(-1)
        self._maybe_update_num_data(inputs.size(0))
        output = self.model(inputs)
        loss = -self.elbo(output, labels)
        with torch.no_grad():
            probs = self.likelihood(output).probs.mean(0)
            predictions = probs.argmax(dim=-1)
            accuracy = (predictions == labels).float().mean()
        return loss, accuracy, probs

    def training_step(self, batch, batch_idx: int):
        loss, accuracy, _ = self._shared_step(batch)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/acc", accuracy, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        loss, accuracy, _ = self._shared_step(batch)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/acc", accuracy, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx: int):
        loss, accuracy, _ = self._shared_step(batch)
        self.log("test/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test/acc", accuracy, prog_bar=True, on_step=False, on_epoch=True)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        inputs, _ = batch
        with torch.no_grad():
            posterior = self.likelihood(self.model(inputs))
            probs = posterior.probs.mean(0)
            variance = posterior.probs.var(0)
        return probs, variance

    def predict_with_uncertainty(
        self, inputs: torch.Tensor, num_likelihood_samples: int = 32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return mean class probabilities and variance for given embeddings."""
        self.model.eval()
        self.likelihood.eval()
        inputs = inputs.to(self.device)
        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(
            num_likelihood_samples
        ):
            posterior = self.likelihood(self.model(inputs))
            probs = posterior.probs
            mean_probs = probs.mean(0)
            variance = probs.var(0)
        return mean_probs, variance

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )


def create_trainer(
    *,
    max_epochs: int = 50,
    accelerator: Optional[str] = None,
    devices: Optional[int] = None,
    log_every_n_steps: int = 10,
    log_dir: str = "logs",
    experiment_name: str = "ca_classifications",
) -> pl.Trainer:
    """Convenience factory for a basic Lightning trainer with TensorBoard logging."""
    logger = TensorBoardLogger(save_dir=log_dir, name=experiment_name, default_hp_metric=False)
    return pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=log_every_n_steps,
        logger=logger,
        enable_progress_bar=True,
        enable_checkpointing=False,
    )

