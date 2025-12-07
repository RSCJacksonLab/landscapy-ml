from __future__ import annotations

from typing import Optional, Tuple
import warnings

import gpytorch
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger


class FeatureNormalizer(torch.nn.Module):
    """
    Track running mean/variance and standardize inputs.

    This keeps deterministic buffers (no learnable params) so that the same
    transform can be applied at inference.
    """

    def __init__(self, num_features: int, eps: float = 1e-6) -> None:
        super().__init__()
        if num_features <= 0:
            raise ValueError("num_features must be positive.")
        self.register_buffer("mean", torch.zeros(num_features))
        self.register_buffer("var", torch.ones(num_features))
        self.register_buffer("count", torch.tensor(0.0))
        self.eps = eps

    @property
    def fitted(self) -> bool:
        return bool(self.count.item() > 0)

    def reset(self) -> None:
        self.mean.zero_()
        self.var.fill_(1.0)
        self.count.zero_()

    def update(self, x: torch.Tensor) -> None:
        """Update running stats from a batch."""
        with torch.no_grad():
            if x.ndim == 1:
                x = x.unsqueeze(0)
            batch_count = float(x.shape[0])
            if batch_count == 0:
                return
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            total = self.count + batch_count
            delta = batch_mean - self.mean
            new_mean = self.mean + delta * batch_count / total
            m_a = self.var * self.count
            m_b = batch_var * batch_count
            m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total
            new_var = m2 / total
            self.mean.copy_(new_mean)
            self.var.copy_(new_var)
            self.count.copy_(total)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / torch.sqrt(self.var + self.eps)


class SequenceGPModel(gpytorch.models.ApproximateGP):
    """
    Variational Gaussian process model for sequence embeddings.

    Parameters
    ----------
    inducing_points : torch.Tensor
        Initial inducing point locations of shape ``(num_inducing, num_features)``.
    num_classes : int
        Number of output classes/tasks.
    """

    def __init__(self, inducing_points: torch.Tensor, num_classes: int) -> None:
        if inducing_points.dim() != 2:
            raise ValueError(
                "inducing_points must have shape [num_inducing, num_features]"
            )

        batch_shape = torch.Size([num_classes])
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=batch_shape
        )
        variational_strategy = (
            gpytorch.variational.IndependentMultitaskVariationalStrategy(
                gpytorch.variational.VariationalStrategy(
                    self,
                    inducing_points,
                    variational_distribution,
                    learn_inducing_locations=True,
                ),
                num_tasks=num_classes,
            )
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=batch_shape),
            batch_shape=batch_shape,
        )

    def forward(
        self, inputs: torch.Tensor
    ) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(inputs)
        covar_x = self.covar_module(inputs)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SequenceGPClassifier(pl.LightningModule):
    """
    Lightning module wrapping a variational GPyTorch softmax classifier.

    Parameters
    ----------
    num_features : int
        Feature dimension of the input embeddings.
    num_classes : int
        Number of output classes.
    inducing_points : torch.Tensor, optional
        Optional initial inducing point locations. If ``None``, random points are used.
    num_inducing : int, default=64
        Number of inducing points to initialize when ``inducing_points`` is ``None``.
    learning_rate : float, default=0.01
        Learning rate for the optimizer.
    weight_decay : float, default=0.0
        Weight decay for the optimizer.
    num_data : int, optional
        Optional dataset size hint for the ELBO; inferred lazily when ``None``.
    embedding_domain : str, optional
        Optional embedding domain metadata used for compatibility checks at inference time.
    embedding_model : str, optional
        Optional embedding model identifier used for compatibility checks at inference time.
    normalize_features : bool, default=True
        Whether to standardize input embeddings before feeding them to the GP.
    normalization_fit_batches : int, optional
        Maximum number of training batches used to fit normalization statistics at the
        start of training. ``None`` uses the full training loader once.
    """

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
        embedding_domain: Optional[str] = None,
        embedding_model: Optional[str] = None,
        normalize_features: bool = True,
        normalization_fit_batches: Optional[int] = None,
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
        self.embedding_domain = embedding_domain
        self.embedding_model = embedding_model
        # Epoch accumulators
        self._train_correct = 0
        self._train_total = 0
        self._val_correct = 0
        self._val_total = 0
        self._train_class_correct: Optional[torch.Tensor] = None
        self._train_class_total: Optional[torch.Tensor] = None
        self._val_class_correct: Optional[torch.Tensor] = None
        self._val_class_total: Optional[torch.Tensor] = None
        self.normalize_features = normalize_features
        self.normalization_fit_batches = normalization_fit_batches
        self.normalizer = FeatureNormalizer(num_features) if normalize_features else None

    def forward(
        self, inputs: torch.Tensor
    ) -> gpytorch.distributions.MultivariateNormal:
        inputs = self._normalize_inputs(inputs)
        return self.model(inputs)

    def _normalize_inputs(self, inputs: torch.Tensor, *, update_stats: bool = False):
        if not self.normalize_features or self.normalizer is None:
            return inputs
        if update_stats:
            self.normalizer.update(inputs)
        return self.normalizer(inputs)

    def _inducing_points(self) -> torch.Tensor:
        strategy = self.model.variational_strategy
        # IndependentMultitaskVariationalStrategy wraps a base strategy.
        base_strategy = getattr(strategy, "base_variational_strategy", strategy)
        return base_strategy.inducing_points

    def _normalize_inducing_points(self) -> None:
        if not self.normalize_features or self.normalizer is None:
            return
        inducing = self._inducing_points()
        with torch.no_grad():
            normalized = self.normalizer(inducing)
            inducing.data.copy_(normalized)

    def _fit_normalizer_from_train_loader(self) -> None:
        """
        Fit normalization stats from the training loader once before training.

        Avoids per-step drift while keeping inference deterministic.
        """
        if not self.normalize_features or self.normalizer is None:
            return
        if self.normalizer.fitted:
            return
        if self.trainer is None or self.trainer.datamodule is None:
            return
        loader = self.trainer.datamodule.train_dataloader()
        max_batches = self.normalization_fit_batches
        device = self.device
        for idx, batch in enumerate(loader):
            feats = batch[0] if isinstance(batch, (tuple, list)) else batch
            feats = feats.to(device)
            self._normalize_inputs(feats, update_stats=True)
            if max_batches is not None and (idx + 1) >= max_batches:
                break
        self._normalize_inducing_points()

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
        output = self(inputs)
        loss = -self.elbo(output, labels)
        with torch.no_grad():
            probs = self.likelihood(output).probs.mean(0)
            predictions = probs.argmax(dim=-1)
            correct = predictions == labels
            accuracy = correct.float().mean()
            correct_per_class = torch.bincount(
                labels[correct], minlength=self.num_classes
            )
            total_per_class = torch.bincount(labels, minlength=self.num_classes)
        return (
            loss,
            accuracy,
            probs,
            correct.sum().item(),
            labels.numel(),
            correct_per_class,
            total_per_class,
        )

    def training_step(self, batch, batch_idx: int):
        loss, accuracy, _, correct, total, correct_per_class, total_per_class = (
            self._shared_step(batch)
        )
        self._train_correct += correct
        self._train_total += total
        if self._train_class_correct is not None:
            self._train_class_correct += correct_per_class
            self._train_class_total += total_per_class
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log(
            "train/acc_step", accuracy, prog_bar=False, on_step=True, on_epoch=False
        )
        return loss

    def validation_step(self, batch, batch_idx: int):
        loss, accuracy, _, correct, total, correct_per_class, total_per_class = (
            self._shared_step(batch)
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
        """
        Predict class probabilities and variances for input embeddings.

        Parameters
        ----------
        inputs : torch.Tensor
            Input embedding matrix of shape ``(N, D)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Mean probabilities and variances per class.
        """
        return self.predict_with_uncertainty(inputs)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        inputs, _ = batch
        with torch.no_grad():
            posterior = self.likelihood(self(inputs))
            probs = posterior.probs.mean(0)
            variance = posterior.probs.var(0)
        return probs, variance

    def predict_with_uncertainty(
        self, inputs: torch.Tensor, num_likelihood_samples: int = 32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return mean class probabilities and variance for embeddings.

        Parameters
        ----------
        inputs : torch.Tensor
            Input embedding matrix of shape ``(N, D)``.
        num_likelihood_samples : int, default=32
            Number of likelihood samples used to estimate uncertainty.

        Returns
        -------
        mean_probs : torch.Tensor
            Mean class probabilities of shape ``(N, num_classes)``.
        variance : torch.Tensor
            Variance of class probabilities of shape ``(N, num_classes)``.
        """
        self.model.eval()
        self.likelihood.eval()
        inputs = inputs.to(self.device)
        inputs = self._normalize_inputs(inputs)
        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(
            num_likelihood_samples
        ):
            posterior = self.likelihood(self(inputs))
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
        if self.trainer is not None and self.trainer.global_step == 0:
            self._fit_normalizer_from_train_loader()
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
            self.log(
                "train/acc_macro", macro, prog_bar=False, on_step=False, on_epoch=True
            )
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
            self.log(
                "val/acc_macro", macro, prog_bar=False, on_step=False, on_epoch=True
            )
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
    experiment_name: str = "landscapyml",
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
    """
    Build a PyTorch Lightning ``Trainer`` with TensorBoard (and optional W&B) logging.

    Parameters
    ----------
    max_epochs : int, default=50
        Maximum number of training epochs.
    accelerator : str or None, default="auto"
        Accelerator passed to Lightning (e.g., ``"cpu"``, ``"gpu"``, or ``"auto"``).
    devices : int or None, default=1
        Number of devices to use; forwarded to Lightning.
    log_every_n_steps : int, default=10
        Logging frequency in steps.
    log_dir : str, default="logs"
        Base directory for TensorBoard logs.
    experiment_name : str, default="landscapyml"
        Experiment subdirectory name used by loggers.
    checkpoint_dir : str, default="checkpoints"
        Directory for model checkpoints.
    checkpoint_monitor : str or None, default="val/loss"
        Metric name to monitor for checkpointing. ``None`` disables checkpointing.
    checkpoint_mode : str, default="min"
        Whether to minimize or maximize the monitored metric.
    checkpoint_every_n_epochs : int, default=1
        Frequency in epochs for checkpointing.
    save_top_k : int, default=1
        Number of best checkpoints to keep.
    use_wandb : bool, default=True
        Whether to enable Weights & Biases logging (if available).
    wandb_project : str, optional
        Optional W&B project name.
    wandb_entity : str, optional
        Optional W&B entity/organization.
    wandb_run_name : str, optional
        Optional W&B run name.
    wandb_tags : list[str], optional
        Optional W&B tags.
    wandb_dir : str, optional
        Optional W&B log directory.
    num_sanity_val_steps : int, default=0
        Number of validation sanity steps to run before training.

    Returns
    -------
    pytorch_lightning.Trainer
        Configured Lightning trainer instance.
    """
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
