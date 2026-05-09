import sys
import types

import pytest
import torch

from landscapyml.core.data import (
    LandscapeDataModule,
    LandscapeDataset,
    LandscapeGraphDataset,
    LandscapeGraphRegressionDataModule,
    build_regression_graph_from_landscape,
)
from landscapyml.core.data_utils import (
    _pad_tokens,
    embed_sequences,
    embed_sequences_to_records,
)


class DummyEmbedder:
    pad_token_id = 0

    def __init__(self, model_name: str, device=None, batch_size: int = 32):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

    def batch_iterator(self, seq_list, batch_size: int = 32):
        max_len = max(len(s) for s in seq_list) + 1
        tokens = []
        masks = []
        lengths = []
        for idx, seq in enumerate(seq_list):
            length = len(seq)
            lengths.append(length)
            tok = torch.arange(length + 1, dtype=torch.long)
            mask = torch.ones_like(tok)
            # pad to max_len
            tok = torch.nn.functional.pad(tok, (0, max_len - tok.numel()), value=0)
            mask = torch.nn.functional.pad(mask, (0, max_len - mask.numel()), value=0)
            tokens.append(tok)
            masks.append(mask)
        token_batch = torch.stack(tokens)
        mask_batch = torch.stack(masks)
        batch_indices = list(range(len(seq_list)))
        yield token_batch, mask_batch, lengths, batch_indices

    def forward_pass(self, token_batch, mask_batch):
        # Produce deterministic hidden states per sequence
        batch_size, seq_len = token_batch.shape
        embed_dim = 4
        hs = []
        for i in range(batch_size):
            hs.append(torch.full((seq_len, embed_dim), float(i + 1)))
        hidden_states = torch.stack(hs, dim=0)
        return types.SimpleNamespace(hidden_states=[hidden_states])


class DummySoftEmbedder(DummyEmbedder):
    pad_token_id = None


@pytest.fixture(autouse=True)
def stub_landscapy_embedder(monkeypatch):
    hard_mod = types.SimpleNamespace(ESMEmbedder=DummyEmbedder)
    soft_mod = types.SimpleNamespace(ESMEmbedder=DummySoftEmbedder)
    monkeypatch.setitem(
        sys.modules, "fitness_landscape.embedding.hard_embedding", hard_mod
    )
    monkeypatch.setitem(
        sys.modules, "fitness_landscape.embedding.soft_embedding", soft_mod
    )
    yield
    monkeypatch.setitem(
        sys.modules, "fitness_landscape.embedding.hard_embedding", hard_mod
    )
    monkeypatch.setitem(
        sys.modules, "fitness_landscape.embedding.soft_embedding", soft_mod
    )


def test_pad_tokens_and_attention_masks():
    tokens = [torch.tensor([1, 2]), torch.tensor([3])]
    masks = [torch.tensor([1, 1]), torch.tensor([1])]
    padded_tokens, padded_masks = _pad_tokens(tokens, masks, pad_value=0)
    assert padded_tokens[0].tolist() == [1, 2]
    assert padded_tokens[1].tolist() == [3, 0]
    assert padded_masks[0].tolist() == [1, 1]
    assert padded_masks[1].tolist() == [1, 0]


def test_embed_sequences_with_tokens(monkeypatch):
    seqs = ["AAA", "BC"]
    embeddings, tokens, masks = embed_sequences(
        seqs, embedding_mode="hard", include_tokens=True
    )
    assert embeddings.shape[0] == 2
    assert tokens is not None and masks is not None
    assert all(t.shape[0] == tokens[0].shape[0] for t in tokens)
    assert all(m.shape[0] == masks[0].shape[0] for m in masks)


def test_embed_sequences_soft_mode(monkeypatch):
    seqs = ["AAA"]
    embeddings, tokens, masks = embed_sequences(
        seqs, embedding_mode="soft", include_tokens=True
    )
    assert embeddings.shape[0] == 1
    # include_tokens should be disabled for soft mode
    assert tokens is None and masks is None


def test_embed_sequences_to_records_creates_expected_fields():
    seqs = ["AAA", "BBB"]
    labels = [0, 1]
    records = embed_sequences_to_records(seqs, labels, label_key="label")
    assert len(records) == 2
    for rec in records:
        assert "fitness_tensors" in rec and "label" in rec["fitness_tensors"]
        assert "embedding" in rec


def test_landscape_data_specializations_share_core_abstractions():
    assert issubclass(LandscapeGraphDataset, LandscapeDataset)
    assert issubclass(LandscapeGraphRegressionDataModule, LandscapeDataModule)


def test_generic_datamodule_splits_train_val():
    records = [
        {"sequence_tensor": torch.tensor([1.0, 0.0]), "fitness_tensors": {"label": 0}},
        {"sequence_tensor": torch.tensor([0.0, 1.0]), "fitness_tensors": {"label": 1}},
        {"sequence_tensor": torch.tensor([1.0, 1.0]), "fitness_tensors": {"label": 0}},
    ]
    dm = LandscapeDataModule(
        train_data=records,
        val_data=None,
        test_data=None,
        predict_data=None,
        batch_size=2,
        val_split=0.34,
        val_seed=42,
    )
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    assert len(train_loader.dataset) + len(val_loader.dataset) == len(records)


def test_graph_regression_falls_back_for_variable_length_sequences():
    import networkx as nx
    import numpy as np

    class DummySequence:
        def __init__(self, value):
            self.value = value

        def to_array(self):
            return np.asarray(list(self.value), dtype=object)

    class DummyLayer:
        dtype = "numeric"

        def to_scalar(self, aggregate_func=np.mean):  # noqa: ARG002
            return np.asarray([1.0, np.nan, 3.0], dtype=float)

    class DummyLandscape:
        def __init__(self):
            self.sequences = [
                DummySequence("AA"),
                DummySequence("AAA"),
                DummySequence("AB"),
            ]
            self.graph = nx.path_graph(3)
            for idx, sequence in enumerate(self.sequences):
                self.graph.nodes[idx]["sequence"] = sequence
            self._node_order = list(self.graph.nodes())
            self.fitness_layers = {"score": DummyLayer()}

        def to_graph_tensor(self, tokenizer=None):  # noqa: ARG002
            raise ValueError("inhomogeneous shape after 1 dimensions")

    graph = build_regression_graph_from_landscape(
        DummyLandscape(),
        target_layer="score",
    )
    assert graph.x.shape[0] == 3
    assert graph.edge_index.shape[0] == 2
    assert graph.known_mask.tolist() == [True, False, True]
    assert graph.predict_mask.tolist() == [False, True, False]
