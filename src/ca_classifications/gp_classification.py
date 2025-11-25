from __future__ import annotations

from typing import Optional, Tuple
import warnings

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
        # Disable mixing_weights to avoid requiring num_features with newer GPyTorch.
        self.likelihood = gpytorch.likelihoods.SoftmaxLikelihood(
            num_classes=num_classes, mixing_weights=False
        )
        self.elbo = gpytorch.mlls.VariationalELBO(
            self.likelihood, self.model, num_data=num_data or 1
        )
        self._num_data = num_data
        self.num_classes = num_classes
        # Epoch accumulators
        self._train_correct = 0
        self._train_total = 0
        self._val_correct = 0
        self._val_total = 0
        self._train_class_correct: Optional[torch.Tensor] = None
        self._train_class_total: Optional[torch.Tensor] = None
        self._val_class_correct: Optional[torch.Tensor] = None
        self._val_class_total: Optional[torch.Tensor] = None

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
            correct = (predictions == labels)
            accuracy = correct.float().mean()
            correct_per_class = torch.bincount(
                labels[correct], minlength=self.num_classes
            )
            total_per_class = torch.bincount(labels, minlength=self.num_classes)
        return loss, accuracy, probs, correct.sum().item(), labels.numel(), correct_per_class, total_per_class

    def training_step(self, batch, batch_idx: int):
        loss, accuracy, _, correct, total, correct_per_class, total_per_class = self._shared_step(
            batch
        )
        self._train_correct += correct
        self._train_total += total
        if self._train_class_correct is not None:
            self._train_class_correct += correct_per_class
            self._train_class_total += total_per_class
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train/acc_step", accuracy, prog_bar=False, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx: int):
        loss, accuracy, _, correct, total, correct_per_class, total_per_class = self._shared_step(
            batch
        )
        self._val_correct += correct
        self._val_total += total
        if self._val_class_correct is not None:
            self._val_class_correct += correct_per_class
            self._val_class_total += total_per_class
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx: int):
        loss, accuracy, _, _, _, _, _ = self._shared_step(batch)
        self.log("test/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test/acc", accuracy, prog_bar=True, on_step=False, on_epoch=True)

    def predict(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Alias for predict_with_uncertainty for consistency across models."""
        return self.predict_with_uncertainty(inputs)

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

    def on_train_epoch_start(self):
        self._train_correct = 0
        self._train_total = 0
        self._train_class_correct = torch.zeros(self.num_classes, device=self.device)
        self._train_class_total = torch.zeros(self.num_classes, device=self.device)

    def on_validation_epoch_start(self):
        self._val_correct = 0
        self._val_total = 0
        self._val_class_correct = torch.zeros(self.num_classes, device=self.device)
        self._val_class_total = torch.zeros(self.num_classes, device=self.device)

    def on_train_epoch_end(self):
        if self._train_total > 0:
            acc = float(self._train_correct) / float(self._train_total)
            self.log("train/acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        if self._train_class_total is not None:
            denom = torch.clamp(self._train_class_total, min=1)
            per_class = (self._train_class_correct / denom).float()
            macro = per_class.mean().item()
            self.log("train/acc_macro", macro, prog_bar=False, on_step=False, on_epoch=True)
            # Per-class accuracy logging for deeper inspection
            per_class_cpu = per_class.detach().cpu()
            self.log_dict(
                {f"train/acc_class_{i}": float(v) for i, v in enumerate(per_class_cpu)},
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

    def on_validation_epoch_end(self):
        if self._val_total > 0:
            acc = float(self._val_correct) / float(self._val_total)
            self.log("val/acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        if self._val_class_total is not None:
            denom = torch.clamp(self._val_class_total, min=1)
            per_class = (self._val_class_correct / denom).float()
            macro = per_class.mean().item()
            self.log("val/acc_macro", macro, prog_bar=False, on_step=False, on_epoch=True)
            per_class_cpu = per_class.detach().cpu()
            self.log_dict(
                {f"val/acc_class_{i}": float(v) for i, v in enumerate(per_class_cpu)},
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )


def create_trainer(
    *,
    max_epochs: int = 50,
    accelerator: Optional[str] = "auto",
    devices: Optional[int] = 1,
    log_every_n_steps: int = 10,
    log_dir: str = "logs",
    experiment_name: str = "ca_classifications",
    checkpoint_dir: str = "checkpoints",
    checkpoint_monitor: Optional[str] = "val/loss",
    checkpoint_mode: str = "min",
    checkpoint_every_n_epochs: int = 1,
    save_top_k: int = 1,
    use_wandb: bool = True,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_tags: Optional[list[str]] = None,
    wandb_dir: Optional[str] = None,
    num_sanity_val_steps: int = 0,
) -> pl.Trainer:
    """Convenience factory for a basic Lightning trainer with TensorBoard logging."""
    tensorboard_logger = TensorBoardLogger(
        save_dir=log_dir,
        name=experiment_name,
        default_hp_metric=False,
    )
    loggers = [tensorboard_logger]

    if use_wandb:
        try:
            from pytorch_lightning.loggers import WandbLogger
        except Exception as exc:  # pragma: no cover - optional dependency
            warnings.warn(
                f"W&B logging requested but wandb is not available: {exc}. "
                "Proceeding with TensorBoard only.",
                RuntimeWarning,
            )
        else:
            if wandb_project is None:
                warnings.warn(
                    "wandb_project is None; WandbLogger will use the default W&B project.",
                    RuntimeWarning,
                )
            wandb_logger = WandbLogger(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_run_name,
                tags=wandb_tags,
                save_dir=wandb_dir,
            )
            loggers.append(wandb_logger)
    callbacks = []
    if checkpoint_monitor is not None:
        checkpoint_cb = pl.callbacks.ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"{{epoch}}-{{step}}-{checkpoint_monitor.replace('/', '-')}",
            monitor=checkpoint_monitor,
            mode=checkpoint_mode,
            save_top_k=save_top_k,
            every_n_epochs=checkpoint_every_n_epochs,
        )
        callbacks.append(checkpoint_cb)
    acc = accelerator if accelerator is not None else "auto"

    devs = devices if devices is not None else 1

    return pl.Trainer(
        max_epochs=max_epochs,
        accelerator=acc,
        devices=devs,
        log_every_n_steps=log_every_n_steps,
        logger=loggers,
        enable_progress_bar=True,
        enable_checkpointing=bool(callbacks),
        callbacks=callbacks,
        num_sanity_val_steps=num_sanity_val_steps,
    )
