import torch

from landscapyml.landscape_pipeline import (
    LandscapeDataModule,
    LandscapeDataset,
    export_landscape_records,
    make_fitness_target_getter,
)


class DummyLayer:
    def __init__(self, categories=None):
        self.categories = categories


class DummyLandscape:
    def __init__(self):
        self.fitness_layers = {
            "family": DummyLayer(["a", "b"]),
            "score": DummyLayer(),
        }

    def to_sequence_tensors(self, **kwargs):
        self.kwargs = kwargs
        return [
            {
                "sequence_tensor": torch.tensor([1.0, 0.0]),
                "fitness_tensors": {
                    "family": torch.tensor([0, 1]),
                    "score": torch.tensor([0.5]),
                },
                "embedding": torch.tensor([0.25, 0.75]),
            },
            {
                "sequence_tensor": torch.tensor([0.0, 1.0]),
                "fitness_tensors": {
                    "family": torch.tensor([1, 0]),
                    "score": torch.tensor([1.5]),
                },
                "embedding": torch.tensor([0.75, 0.25]),
            },
        ]


def test_export_landscape_records_preserves_multiple_layers():
    landscape = DummyLandscape()
    exported = export_landscape_records(landscape, feature_view="embedding")

    assert len(exported.records) == 2
    assert set(exported.records[0]["fitness_tensors"]) == {"family", "score"}
    assert exported.fitness_mappings["family"] == ["a", "b"]
    assert exported.fitness_mappings["score"] is None


def test_landscape_dataset_supports_custom_target_getter():
    records = [
        {
            "embedding": torch.tensor([1.0, 2.0]),
            "fitness_tensors": {"family": torch.tensor([0, 1])},
        }
    ]
    dataset = LandscapeDataset(
        records,
        target_getter=make_fitness_target_getter(
            "family",
            collapse_one_hot=True,
            dtype=torch.long,
        ),
    )

    features, target = dataset[0]
    assert torch.allclose(features, torch.tensor([1.0, 2.0]))
    assert target.item() == 1


def test_landscape_datamodule_builds_supervised_loaders():
    landscape = DummyLandscape()
    exported = export_landscape_records(landscape, feature_view="embedding")
    dm = LandscapeDataModule(
        train_data=exported.records,
        batch_size=2,
        shuffle=False,
        dataset_kwargs={
            "target_getter": make_fitness_target_getter(
                "family",
                collapse_one_hot=True,
                dtype=torch.long,
            )
        },
    )

    dm.setup("fit")
    features, labels = next(iter(dm.train_dataloader()))
    assert features.shape == (2, 2)
    assert set(labels.tolist()) == {0, 1}
