from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Callable, Literal, Optional, Union

import numpy as np
import torch

LandscapeRecord = dict[str, Any]
InputGetter = Callable[[Mapping[str, Any]], Any]
TargetGetter = Callable[[Mapping[str, Any]], Any]
SequenceLike = Union[str, Sequence[int], torch.Tensor]
LabelLike = Union[int, Sequence[int], torch.Tensor]


def expand_record_batch(batch: Mapping[str, Any]) -> list[LandscapeRecord]:
    """
    Convert a batched landscapy tensor export into per-record dictionaries.
    """

    if "sequence_tensor" not in batch or "fitness_tensors" not in batch:
        raise ValueError(
            "Batch dictionary must contain 'sequence_tensor' and 'fitness_tensors'."
        )

    seqs = torch.as_tensor(batch["sequence_tensor"])
    fitness = {k: torch.as_tensor(v) for k, v in batch["fitness_tensors"].items()}
    attention_mask = batch.get("attention_mask")
    attention_mask_t = (
        torch.as_tensor(attention_mask) if attention_mask is not None else None
    )
    embedding = batch.get("embedding")
    embedding_t = torch.as_tensor(embedding) if embedding is not None else None

    records: list[LandscapeRecord] = []
    for idx in range(seqs.shape[0]):
        record: LandscapeRecord = {
            "sequence_tensor": seqs[idx],
            "fitness_tensors": {name: tensor[idx] for name, tensor in fitness.items()},
        }
        if attention_mask_t is not None:
            record["attention_mask"] = attention_mask_t[idx]
        if embedding_t is not None:
            record["embedding"] = embedding_t[idx]
        records.append(record)
    return records


def normalize_records(data: Any) -> list[LandscapeRecord]:
    """
    Normalize supported landscape record inputs into a list of records.
    """

    if data is None:
        return []
    if isinstance(data, Mapping):
        return expand_record_batch(data)
    if isinstance(data, Iterable) and not isinstance(
        data, (str, bytes, bytearray, torch.Tensor)
    ):
        items = list(data)
        if not items:
            return []
        if isinstance(items[0], Mapping):
            return [dict(item) for item in items]
    raise ValueError("Data must be a batch dict or an iterable of record dictionaries.")


def make_preferred_input_getter(
    *feature_keys: str, cast_float: bool = True
) -> InputGetter:
    keys = feature_keys or ("embedding", "sequence_tensor")

    def _getter(record: Mapping[str, Any]) -> torch.Tensor:
        for key in keys:
            feature = record.get(key)
            if feature is None:
                continue
            tensor = torch.as_tensor(feature)
            if cast_float and not tensor.is_floating_point():
                tensor = tensor.float()
            return tensor
        key_list = ", ".join(keys)
        raise ValueError(f"Record missing feature view. Tried: {key_list}.")

    return _getter


def make_fitness_target_getter(
    layer_name: str,
    *,
    collapse_one_hot: bool = False,
    dtype: Optional[torch.dtype] = None,
    squeeze: bool = True,
) -> TargetGetter:
    def _getter(record: Mapping[str, Any]) -> torch.Tensor:
        fitness = record.get("fitness_tensors")
        if not isinstance(fitness, Mapping) or layer_name not in fitness:
            raise ValueError(f"Record missing fitness label '{layer_name}'.")

        target = torch.as_tensor(fitness[layer_name])
        if collapse_one_hot and target.ndim > 0 and target.numel() > 1:
            target = target.argmax(dim=-1)
        if dtype is not None:
            target = target.to(dtype=dtype)
        if squeeze:
            target = target.squeeze()
        return target

    return _getter


def _pad_tokens(
    tokens: list[torch.Tensor],
    masks: list[Optional[torch.Tensor]],
    *,
    pad_value: Union[int, float],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Pad token and attention tensors to a shared sequence length.
    """

    max_len = max(int(t.shape[0]) for t in tokens if t is not None)
    padded_tokens: list[torch.Tensor] = []
    padded_masks: list[torch.Tensor] = []
    for tok, mask in zip(tokens, masks):
        if tok is None:
            raise RuntimeError("Token tensor missing; cannot pad.")
        if mask is None:
            raise RuntimeError("Attention mask missing; cannot pad.")
        pad_len = max_len - int(tok.shape[0])
        if tok.dim() == 1:
            padded_tokens.append(
                torch.nn.functional.pad(tok, (0, pad_len), value=pad_value)
            )
        else:
            padded_tokens.append(
                torch.nn.functional.pad(tok, (0, 0, 0, pad_len), value=float(pad_value))
            )

        attn = torch.as_tensor(mask)
        padded_masks.append(torch.nn.functional.pad(attn.long(), (0, pad_len), value=0))
    return padded_tokens, padded_masks


def embed_sequences(
    sequences: Sequence[SequenceLike],
    *,
    embedding_mode: Literal["hard", "soft"] = "hard",
    model_name: str = "facebook/esm2_t6_8M_UR50D",
    device: Optional[str] = None,
    embedding_batch_size: int = 32,
    include_tokens: bool = True,
) -> tuple[torch.Tensor, Optional[list[torch.Tensor]], Optional[list[torch.Tensor]]]:
    """
    Embed raw sequences using landscapy's ESM embedders.
    """

    if include_tokens and embedding_mode != "hard":
        warnings.warn(
            "include_tokens=True is only supported for hard tokenization; disabling tokens.",
            RuntimeWarning,
        )
        include_tokens = False
    seq_list = list(sequences)
    if not seq_list:
        raise ValueError("sequences must not be empty.")

    mode = embedding_mode.lower()
    if mode not in {"hard", "soft"}:
        raise ValueError("embedding_mode must be 'hard' or 'soft'.")
    logger = logging.getLogger("landscapyml")
    logger.info(
        "Embedding %d sequences (mode=%s, model=%s, batch_size=%d)",
        len(seq_list),
        mode,
        model_name,
        embedding_batch_size,
    )

    try:
        if mode == "hard":
            from fitness_landscape.embedding.hard_embedding import (
                ESMEmbedder as HardESMEmbedder,
            )

            embedder = HardESMEmbedder(
                model_name=model_name, device=device, batch_size=embedding_batch_size
            )
            pad_value: Union[int, float] = (
                int(embedder.pad_token_id) if embedder.pad_token_id is not None else 0
            )
        else:
            from fitness_landscape.embedding.soft_embedding import (
                ESMEmbedder as SoftESMEmbedder,
            )

            embedder = SoftESMEmbedder(
                model_name=model_name, device=device, batch_size=embedding_batch_size
            )
            pad_value = 0.0
    except Exception as exc:  # pragma: no cover - imported lazily
        raise ImportError(
            "Embedding raw sequences requires the landscapy package to be available."
        ) from exc

    n = len(seq_list)
    embeddings: list[Optional[torch.Tensor]] = [None] * n
    seq_tokens: list[Optional[torch.Tensor]] = [None] * n
    attn_masks: list[Optional[torch.Tensor]] = [None] * n

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

    if include_tokens:
        if any(tok is None for tok in seq_tokens) or any(
            mask is None for mask in attn_masks
        ):
            raise RuntimeError("Tokenization failed for one or more sequences.")
        padded_tokens, padded_masks = _pad_tokens(
            seq_tokens, attn_masks, pad_value=pad_value
        )
        logger.info("Finished embedding %d sequences", len(seq_list))
        return embedding_stack, padded_tokens, padded_masks

    logger.info("Finished embedding %d sequences", len(seq_list))
    return embedding_stack, None, None


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
) -> list[dict[str, Any]]:
    """
    Embed raw sequences and emit FitnessLandscape-compatible record dictionaries.
    """

    embeddings, tokens, masks = embed_sequences(
        sequences,
        embedding_mode=embedding_mode,
        model_name=model_name,
        device=device,
        embedding_batch_size=embedding_batch_size,
        include_tokens=include_tokens,
    )

    if len(sequences) != len(labels):
        raise ValueError("sequences and labels must have the same length.")

    records: list[dict[str, Any]] = []
    for i, label in enumerate(labels):
        record = {
            "sequence_tensor": tokens[i] if tokens is not None else embeddings[i],
            "fitness_tensors": {label_key: torch.as_tensor(label)},
            "embedding": embeddings[i],
        }
        if masks is not None:
            record["attention_mask"] = masks[i]
        records.append(record)

    return records


def aggregate_numeric_targets(
    layer: Any, *, aggregate_func: Optional[Callable[..., Any]] = np.mean
) -> np.ndarray:
    if getattr(layer, "dtype", None) != "numeric":
        raise ValueError("Landscape graph regression supports numeric fitness layers only.")
    if aggregate_func is None:
        values = layer.to_scalar()
    else:
        try:
            values = layer.to_scalar(aggregate_func=aggregate_func)
        except TypeError:
            values = layer.to_scalar()
    return np.asarray(values, dtype=float).reshape(-1)


def build_mask(num_nodes: int, indices: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    if indices.numel() > 0:
        mask[indices] = True
    return mask


def feature_normalization_stats(
    x: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.as_tensor(x, dtype=torch.float32)
    if features.ndim != 2:
        raise ValueError("Graph node features must be a 2D tensor.")
    if mask is not None:
        mask = torch.as_tensor(mask, dtype=torch.bool, device=features.device)
        if mask.shape[0] != features.shape[0]:
            raise ValueError("Feature normalization mask length does not match graph nodes.")
        if int(mask.sum().item()) > 0:
            features = features[mask]
    if not bool(torch.isfinite(features).all()):
        raise ValueError("Cannot normalize graph features containing NaN or Inf values.")

    mean = features.mean(dim=0)
    scale = features.std(dim=0, unbiased=False)
    scale = torch.where(scale > float(eps), scale, torch.ones_like(scale))
    return mean.detach(), scale.detach()


def sequence_composition_features(
    sequences: Sequence[Any],
    *,
    alphabet: Optional[Sequence[Any]] = None,
) -> np.ndarray:
    """
    Build simple variable-length-safe sequence features.

    The first feature is normalized length; remaining features are residue/token
    frequencies over either the supplied alphabet or the observed symbols.
    """

    arrays: list[list[str]] = []
    observed: list[str] = []
    seen: set[str] = set()
    lengths: list[int] = []

    for sequence in sequences:
        if hasattr(sequence, "to_array"):
            arr = [str(value) for value in np.asarray(sequence.to_array()).reshape(-1)]
        else:
            arr = [str(value) for value in sequence]
        arrays.append(arr)
        lengths.append(len(arr))
        for symbol in arr:
            if symbol not in seen:
                seen.add(symbol)
                observed.append(symbol)

    alphabet_keys = (
        [str(symbol) for symbol in alphabet]
        if alphabet is not None
        else sorted(observed)
    )
    alphabet_index = {symbol: idx for idx, symbol in enumerate(alphabet_keys)}
    max_len = max(max(lengths), 1) if lengths else 1
    features = np.zeros((len(sequences), len(alphabet_keys) + 1), dtype=float)

    for row, arr in enumerate(arrays):
        length = max(len(arr), 1)
        features[row, 0] = length / max_len
        for symbol in arr:
            idx = alphabet_index.get(symbol)
            if idx is not None:
                features[row, idx + 1] += 1.0
        features[row, 1:] /= length
    return features


def _as_index_tensor(
    indices: Sequence[int] | torch.Tensor | None,
    *,
    num_nodes: int,
    name: str,
) -> torch.Tensor | None:
    if indices is None:
        return None
    tensor = torch.as_tensor(indices, dtype=torch.long).view(-1)
    if tensor.numel() == 0:
        return tensor
    if bool((tensor < 0).any()) or bool((tensor >= num_nodes).any()):
        raise ValueError(f"{name} contains node indices outside [0, {num_nodes}).")
    return torch.unique(tensor, sorted=True)


def _sample_without_replacement(
    indices: torch.Tensor,
    n: int,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if n <= 0 or indices.numel() == 0:
        return indices.new_empty((0,), dtype=torch.long), indices
    n = min(int(n), int(indices.numel()))
    perm = torch.randperm(indices.numel(), generator=generator)
    selected = indices[perm[:n]]
    remaining = indices[perm[n:]]
    return selected, remaining


def resolve_split_indices(
    *,
    num_nodes: int,
    known_idx: torch.Tensor,
    train_indices: Sequence[int] | torch.Tensor | None = None,
    val_indices: Sequence[int] | torch.Tensor | None = None,
    test_indices: Sequence[int] | torch.Tensor | None = None,
    val_fraction: float = 0.0,
    test_fraction: float = 0.0,
    seed: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    explicit_train = _as_index_tensor(
        train_indices, num_nodes=num_nodes, name="train_indices"
    )
    explicit_val = _as_index_tensor(val_indices, num_nodes=num_nodes, name="val_indices")
    explicit_test = _as_index_tensor(
        test_indices, num_nodes=num_nodes, name="test_indices"
    )

    explicit = [
        idx for idx in (explicit_train, explicit_val, explicit_test) if idx is not None
    ]
    if explicit:
        known_mask = build_mask(num_nodes, known_idx)
        for name, idx in (
            ("train_indices", explicit_train),
            ("val_indices", explicit_val),
            ("test_indices", explicit_test),
        ):
            if idx is not None and idx.numel() and not bool(known_mask[idx].all()):
                raise ValueError(f"{name} contains indices without finite target values.")

        assigned = torch.zeros(num_nodes, dtype=torch.bool)
        for name, idx in (
            ("train_indices", explicit_train),
            ("val_indices", explicit_val),
            ("test_indices", explicit_test),
        ):
            if idx is None or idx.numel() == 0:
                continue
            if bool(assigned[idx].any()):
                raise ValueError(f"{name} overlaps with another supplied split.")
            assigned[idx] = True

        remaining = known_idx[~assigned[known_idx]]
        train_idx = explicit_train if explicit_train is not None else remaining
        val_idx = explicit_val if explicit_val is not None else known_idx.new_empty((0,))
        test_idx = (
            explicit_test if explicit_test is not None else known_idx.new_empty((0,))
        )

        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)

        if explicit_val is None and val_fraction > 0:
            n_val = int(round(float(val_fraction) * max(int(train_idx.numel()), 1)))
            sampled, train_idx = _sample_without_replacement(
                train_idx, n_val, generator=generator
            )
            val_idx = sampled
        if explicit_test is None and test_fraction > 0:
            n_test = int(round(float(test_fraction) * max(int(train_idx.numel()), 1)))
            sampled, train_idx = _sample_without_replacement(
                train_idx, n_test, generator=generator
            )
            test_idx = sampled

        if train_idx.numel() <= 0:
            raise ValueError("At least one finite target must be available for training.")
        return train_idx, val_idx, test_idx

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    perm = torch.randperm(known_idx.numel(), generator=generator)
    shuffled = known_idx[perm]

    n_known = shuffled.numel()
    n_test = int(round(float(test_fraction) * n_known))
    n_val = int(round(float(val_fraction) * n_known))
    n_train = n_known - n_val - n_test
    if n_train <= 0:
        raise ValueError("At least one known node must remain in the training mask.")

    train_idx = shuffled[:n_train]
    val_idx = shuffled[n_train : n_train + n_val]
    test_idx = shuffled[n_train + n_val :]
    return train_idx, val_idx, test_idx


__all__ = [
    "InputGetter",
    "LabelLike",
    "LandscapeRecord",
    "SequenceLike",
    "TargetGetter",
    "_pad_tokens",
    "aggregate_numeric_targets",
    "build_mask",
    "embed_sequences",
    "embed_sequences_to_records",
    "expand_record_batch",
    "feature_normalization_stats",
    "make_fitness_target_getter",
    "make_preferred_input_getter",
    "normalize_records",
    "resolve_split_indices",
    "sequence_composition_features",
]
