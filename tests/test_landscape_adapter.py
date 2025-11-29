import torch

from landscapyml.landscape_adapter import (
    datamodule_from_landscape,
    records_from_landscape,
)


class DummyLayer:
    def __init__(self):
        self.categories = ["x", "y"]


class DummyLandscape:
    def __init__(self):
        self.fitness_layers = {"label": DummyLayer()}

    def to_sequence_tensors(self, **kwargs):
        self.kwargs = kwargs
        return [
            {
                "sequence_tensor": torch.tensor([1.0, 0.0]),
                "fitness_tensors": {"label": torch.tensor([0, 1])},
                "embedding": torch.tensor([0.5, 0.5]),
                "attention_mask": torch.tensor([1, 1]),
            }
        ]


def test_records_from_landscape_extracts_mapping():
    land = DummyLandscape()
    records, mapping = records_from_landscape(land, label_layer="label")
    assert mapping == ["x", "y"]
    assert len(records) == 1
    rec = records[0]
    assert "embedding" in rec
    assert "attention_mask" in rec
    assert rec["fitness_tensors"]["label"].shape[-1] == 2


def test_datamodule_from_landscape_wraps_records():
    land = DummyLandscape()
    dm = datamodule_from_landscape(
        land,
        label_layer="label",
        batch_size=1,
        shuffle=False,
    )
    dm.setup("fit")
    loader = dm.train_dataloader()
    features, labels = next(iter(loader))
    assert features.shape[0] == 1
    assert labels.shape[0] == 1
    assert set(labels.tolist()) == {1}
