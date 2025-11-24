from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union, Literal

import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

SequenceLike = Union[str, Sequence[int], torch.Tensor]
LabelLike = Union[int, Sequence[int], torch.Tensor]


def _pad_tokens(
    tokens: List[torch.Tensor],
    masks: List[Optional[torch.Tensor]],
    *,
    pad_value: Union[int, float],
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Pad token/attention tensors to a shared length."""
    max_len = max(int(t.shape[0]) for t in tokens if t is not None)
    padded_tokens: List[torch.Tensor] = []
    padded_masks: List[torch.Tensor] = []
    for tok, mask in zip(tokens, masks):
        if tok is None:
            raise RuntimeError("Token tensor missing; cannot pad.")
        if mask is None:
            raise RuntimeError("Attention mask missing; cannot pad.")
        pad_len = max_len - int(tok.shape[0])
        if tok.dim() == 1:
            padded_tokens.append(torch.nn.functional.pad(tok, (0, pad_len), value=pad_value))
        else:
            padded_tokens.append(
                torch.nn.functional.pad(tok, (0, 0, 0, pad_len), value=float(pad_value))
            )

        attn = torch.as_tensor(mask)
        padded_masks.append(torch.nn.functional.pad(attn.long(), (0, pad_len), value=0))
    return padded_tokens, padded_masks


def embed_sequences_to_records(
    sequences: Sequence[SequenceLike],
    labels: Sequence[LabelLike],
    *,
    label_key: str,
    embedding_mode: Literal["hard", "soft"] = "hard",
    model_name: str = "facebook/esm2_t6_8M_UR50D",
    device: Optional[str] = None,
    embedding_batch_size: int = 32,
    include_tokens: bool = True,
) -> List[Dict[str, Any]]:
    """
    Embed raw sequences using landscapy's ESM embedders and return records
    matching FitnessLandscape.to_sequence_tensors output structure.
    """
    seq_list = list(sequences)
    if not seq_list:
        raise ValueError("sequences must not be empty.")
    if len(seq_list) != len(labels):
        raise ValueError("sequences and labels must have the same length.")

    mode = embedding_mode.lower()
    if mode not in {"hard", "soft"}:
        raise ValueError("embedding_mode must be 'hard' or 'soft'.")

    try:
        if mode == "hard":
            from fitness_landscape.embedding.hard_embedding import ESMEmbedder as HardESMEmbedder

            embedder = HardESMEmbedder(
                model_name=model_name, device=device, batch_size=embedding_batch_size
            )
            pad_value: Union[int, float] = (
                int(embedder.pad_token_id) if embedder.pad_token_id is not None else 0
            )
        else:
            from fitness_landscape.embedding.soft_embedding import ESMEmbedder as SoftESMEmbedder

            embedder = SoftESMEmbedder(
                model_name=model_name, device=device, batch_size=embedding_batch_size
            )
            pad_value = 0.0
    except Exception as exc:  # pragma: no cover - imported lazily
        raise ImportError(
            "Embedding raw sequences requires the landscapy package to be available."
        ) from exc

    n = len(seq_list)
    embeddings: List[Optional[torch.Tensor]] = [None] * n
    seq_tokens: List[Optional[torch.Tensor]] = [None] * n
    attn_masks: List[Optional[torch.Tensor]] = [None] * n

    for token_batch, mask_batch, lengths, batch_indices in embedder.batch_iterator(
        seq_list, batch_size=embedding_batch_size
    ):
        with torch.no_grad():
            output = embedder.forward_pass(token_batch, mask_batch)
            hidden_states = output.hidden_states[-1]

        for j, original_idx in enumerate(batch_indices):
            length = int(lengths[j])
            emb = hidden_states[j, 1 : length + 1].mean(dim=0).detach().cpu().float()
            embeddings[original_idx] = emb
            if include_tokens:
                seq_tokens[original_idx] = token_batch[j].detach().cpu()
                attn_masks[original_idx] = mask_batch[j].detach().cpu()

    if any(e is None for e in embeddings):
        raise RuntimeError("Failed to compute embeddings for all sequences.")

    embedding_stack = torch.stack([emb for emb in embeddings], dim=0).float()

    padded_tokens: List[torch.Tensor]
    padded_masks: List[torch.Tensor]
    if include_tokens:
        if any(tok is None for tok in seq_tokens) or any(mask is None for mask in attn_masks):
            raise RuntimeError("Tokenization failed for one or more sequences.")
        padded_tokens, padded_masks = _pad_tokens(seq_tokens, attn_masks, pad_value=pad_value)
    else:
        padded_tokens = [embedding_stack[i] for i in range(n)]
        padded_masks = [torch.empty(0, dtype=torch.long) for _ in range(n)]

    records: List[Dict[str, Any]] = []
    for i in range(n):
        record = {
            "sequence_tensor": padded_tokens[i],
            "fitness_tensors": {label_key: torch.as_tensor(labels[i])},
            "embedding": embedding_stack[i],
        }
        if include_tokens:
            record["attention_mask"] = padded_masks[i]
        records.append(record)

    return records


def _expand_batch_dict(batch: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Convert batched dict from FitnessLandscape.to_sequence_tensors(as_batch=True) to list."""
    if "sequence_tensor" not in batch or "fitness_tensors" not in batch:
        raise ValueError("Batch dictionary must contain 'sequence_tensor' and 'fitness_tensors'.")
    seqs = torch.as_tensor(batch["sequence_tensor"])
    fitness = {k: torch.as_tensor(v) for k, v in batch["fitness_tensors"].items()}
    n = seqs.shape[0]
    attention_mask = batch.get("attention_mask")
    attention_mask_t = torch.as_tensor(attention_mask) if attention_mask is not None else None
    embedding = batch.get("embedding")
    embedding_t = torch.as_tensor(embedding) if embedding is not None else None

    records: List[Dict[str, Any]] = []
    for i in range(n):
        rec = {
            "sequence_tensor": seqs[i],
            "fitness_tensors": {name: tensor[i] for name, tensor in fitness.items()},
        }
        if attention_mask_t is not None:
            rec["attention_mask"] = attention_mask_t[i]
        if embedding_t is not None:
            rec["embedding"] = embedding_t[i]
        records.append(rec)
    return records


def _normalize_records(data: Any) -> List[Dict[str, Any]]:
    """Normalize inputs to a list of per-sequence records."""
    if data is None:
        return []
    if isinstance(data, Mapping):
        return _expand_batch_dict(data)
    if isinstance(data, Iterable) and not isinstance(data, (str, bytes, torch.Tensor)):
        items = list(data)
        if not items:
            return []
        if isinstance(items[0], Mapping):
            return items  # type: ignore[return-value]
    raise ValueError("Data must be a batch dict or an iterable of record dictionaries.")


class SequenceClassificationDataset(Dataset):
    """Dataset turning FitnessLandscape exports into (features, label) pairs."""

    def __init__(self, records: Sequence[Mapping[str, Any]], label_key: str) -> None:
        if not records:
            raise ValueError("records must be a non-empty sequence.")
        self.records = list(records)
        self.label_key = label_key

    def __len__(self) -> int:
        return len(self.records)

    def _feature_tensor(self, record: Mapping[str, Any]) -> torch.Tensor:
        # Prefer precomputed embeddings; fall back to sequence_tensor.
        feature = record.get("embedding", record.get("sequence_tensor"))
        if feature is None:
            raise ValueError("Record missing 'embedding' or 'sequence_tensor'.")
        tensor = torch.as_tensor(feature)
        if not tensor.is_floating_point():
            tensor = tensor.float()
        return tensor

    def _label_tensor(self, record: Mapping[str, Any]) -> torch.Tensor:
        fitness = record.get("fitness_tensors")
        if not isinstance(fitness, Mapping) or self.label_key not in fitness:
            raise ValueError(f"Record missing fitness label '{self.label_key}'.")
        label = torch.as_tensor(fitness[self.label_key])
        if label.ndim > 0 and label.numel() > 1:
            label = label.argmax(dim=-1)
        label = label.long().squeeze()
        return label

    def __getitem__(self, idx: int):
        record = self.records[idx]
        features = self._feature_tensor(record)
        label = self._label_tensor(record)
        return features, label


class SequenceClassificationDataModule(pl.LightningDataModule):
    """Lightning DataModule for CA sequence classification."""

    def __init__(
        self,
        *,
        train_data: Any,
        label_key: str,
        val_data: Any = None,
        test_data: Any = None,
        predict_data: Any = None,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle: bool = True,
    ) -> None:
        super().__init__()
        self.train_records = _normalize_records(train_data)
        self.val_records = _normalize_records(val_data) if val_data is not None else []
        self.test_records = _normalize_records(test_data) if test_data is not None else []
        self.predict_records = (
            _normalize_records(predict_data) if predict_data is not None else []
        )
        self.label_key = label_key
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle

        if not self.train_records:
            raise ValueError("train_data must not be empty.")

        self._train_ds: Optional[SequenceClassificationDataset] = None
        self._val_ds: Optional[SequenceClassificationDataset] = None
        self._test_ds: Optional[SequenceClassificationDataset] = None
        self._predict_ds: Optional[SequenceClassificationDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self._train_ds = SequenceClassificationDataset(self.train_records, self.label_key)
            if self.val_records:
                self._val_ds = SequenceClassificationDataset(self.val_records, self.label_key)
        if stage in (None, "test"):
            if self.test_records:
                self._test_ds = SequenceClassificationDataset(self.test_records, self.label_key)
        if stage in (None, "predict"):
            if self.predict_records:
                self._predict_ds = SequenceClassificationDataset(
                    self.predict_records, self.label_key
                )

    def _loader(self, dataset: Optional[Dataset], shuffle: bool = False) -> Optional[DataLoader]:
        if dataset is None:
            return None
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self) -> DataLoader:
        loader = self._loader(self._train_ds, shuffle=self.shuffle)
        if loader is None:
            raise RuntimeError("Training dataset was not initialized.")
        return loader

    def val_dataloader(self) -> Optional[DataLoader]:
        return self._loader(self._val_ds)

    def test_dataloader(self) -> Optional[DataLoader]:
        return self._loader(self._test_ds)

    def predict_dataloader(self) -> Optional[DataLoader]:
        return self._loader(self._predict_ds)

    @classmethod
    def from_sequences(
        cls,
        *,
        train_sequences: Sequence[SequenceLike],
        train_labels: Sequence[LabelLike],
        label_key: str,
        val_sequences: Optional[Sequence[SequenceLike]] = None,
        val_labels: Optional[Sequence[LabelLike]] = None,
        test_sequences: Optional[Sequence[SequenceLike]] = None,
        test_labels: Optional[Sequence[LabelLike]] = None,
        predict_sequences: Optional[Sequence[SequenceLike]] = None,
        predict_labels: Optional[Sequence[LabelLike]] = None,
        embedding_mode: Literal["hard", "soft"] = "hard",
        model_name: str = "facebook/esm2_t6_8M_UR50D",
        device: Optional[str] = None,
        embedding_batch_size: int = 32,
        include_tokens: bool = True,
        **datamodule_kwargs: Any,
    ) -> "SequenceClassificationDataModule":
        """
        Convenience constructor to embed raw sequences with landscapy before training.
        """

        def _maybe_embed(
            seqs: Optional[Sequence[SequenceLike]],
            labels: Optional[Sequence[LabelLike]],
        ) -> Optional[List[Dict[str, Any]]]:
            if seqs is None and labels is None:
                return None
            if seqs is None or labels is None:
                raise ValueError("Sequences and labels must be provided together.")
            return embed_sequences_to_records(
                seqs,
                labels,
                label_key=label_key,
                embedding_mode=embedding_mode,
                model_name=model_name,
                device=device,
                embedding_batch_size=embedding_batch_size,
                include_tokens=include_tokens,
            )

        train_records = _maybe_embed(train_sequences, train_labels)
        if train_records is None:
            raise ValueError("train_sequences and train_labels are required.")

        return cls(
            train_data=train_records,
            label_key=label_key,
            val_data=_maybe_embed(val_sequences, val_labels),
            test_data=_maybe_embed(test_sequences, test_labels),
            predict_data=_maybe_embed(predict_sequences, predict_labels),
            **datamodule_kwargs,
        )
