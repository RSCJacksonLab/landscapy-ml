import torch

from landscapyml.mlp_classification import (
    SequenceMLPClassifier,
    SequenceMLPEnsembleClassifier,
)


def test_sequence_mlp_classifier_forward():
    model = SequenceMLPClassifier(num_features=3, num_classes=2)
    x = torch.randn(5, 3)
    logits = model(x)
    assert logits.shape == (5, 2)
    loss = model.training_step((x, torch.tensor([0, 1, 0, 1, 0])), 0)
    assert loss is not None


def test_sequence_mlp_ensemble_predict_with_uncertainty():
    model = SequenceMLPEnsembleClassifier(num_features=2, num_classes=3, num_models=3)
    x = torch.randn(4, 2)
    mean, var = model.predict_with_uncertainty(x)
    assert mean.shape == (4, 3)
    assert var.shape == (4, 3)
