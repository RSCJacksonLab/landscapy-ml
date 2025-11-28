import torch

from landscapyml.gp_classification import (
    SequenceGPClassifier,
    SequenceGPModel,
    create_trainer,
)


def test_sequence_gp_classifier_forward_and_predict():
    model = SequenceGPClassifier(
        num_features=2, num_classes=2, num_inducing=2, num_data=4
    )
    x = torch.randn(4, 2)
    labels = torch.tensor([0, 1, 0, 1])
    loss = model.training_step((x, labels), 0)
    assert loss is not None
    mean, var = model.predict_with_uncertainty(x)
    assert mean.shape == (4, 2)
    assert var.shape == (4, 2)


def test_sequence_gp_model_shapes():
    inducing = torch.randn(3, 2)
    gp_model = SequenceGPModel(inducing_points=inducing, num_classes=2)
    out = gp_model(torch.randn(2, 2))
    assert out.mean.shape[-1] == 2


def test_create_trainer_returns_trainer(tmp_path):
    trainer = create_trainer(
        max_epochs=1,
        log_dir=str(tmp_path / "logs"),
        checkpoint_dir=str(tmp_path / "ckpts"),
        use_wandb=False,
    )
    assert trainer.max_epochs == 1
