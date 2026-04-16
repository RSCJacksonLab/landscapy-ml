from __future__ import annotations

import logging
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    Literal,
)

import torch
import warnings

from .landscape_pipeline import (
    LandscapeDataModule,
    LandscapeDataset,
    expand_record_batch as _expand_batch_dict,
    make_fitness_target_getter,
    make_preferred_input_getter,
    normalize_records as _normalize_records,
)

SequenceLike = Union[str, Sequence[int], torch.Tensor]
LabelLike = Union[int, Sequence[int], torch.Tensor]


def _pad_tokens(
    tokens: List[torch.Tensor],
    masks: List[Optional[torch.Tensor]],
    *,
    pad_value: Union[int, float],
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Pad token and attention tensors to a shared sequence length.

    Parameters
    ----------
    tokens : list[Tensor]
        Token tensors to be padded.
    masks : list[Tensor | None]
        Attention mask tensors aligned with ``tokens``.
    pad_value : int or float
        Value used when padding tokens or masks.

    Returns
    -------
    tuple[list[Tensor], list[Tensor]]
        Padded token tensors and attention masks with uniform length.

    Raises
    ------
    RuntimeError
        If any token or mask tensor is missing.
    """
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
) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]:
    """
    Embed raw sequences using landscapy's ESM embedders.

    Parameters
    ----------
    sequences : Sequence[SequenceLike]
        Raw sequences to embed.
    embedding_mode : {"hard", "soft"}, default="hard"
        Tokenization/embedding strategy.
    model_name : str, default="facebook/esm2_t6_8M_UR50D"
        HuggingFace model identifier used by landscapy.
    device : str or None, optional
        Device string forwarded to the embedder.
    embedding_batch_size : int, default=32
        Batch size used during embedding.
    include_tokens : bool, default=True
        Whether to return token tensors and attention masks.

    Returns
    -------
    embeddings : torch.Tensor
        Embedding matrix of shape ``(N, D)``.
    tokens : list[torch.Tensor] or None
        Padded token tensors per sequence, or ``None`` when ``include_tokens`` is ``False``.
    attention_masks : list[torch.Tensor] or None
        Attention masks aligned with ``tokens``, or ``None`` when ``include_tokens`` is ``False``.

    Raises
    ------
    ValueError
        If sequences are empty or an invalid embedding mode is provided.
    ImportError
        If landscapy embedders are unavailable.
    RuntimeError
        If embedding fails to produce outputs for every sequence.
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
) -> List[Dict[str, Any]]:
    """
    Embed raw sequences and emit FitnessLandscape-compatible record dictionaries.

    Parameters
    ----------
    sequences : Sequence[SequenceLike]
        Raw sequences to embed.
    labels : Sequence[LabelLike]
        Corresponding labels for each sequence.
    label_key : str
        Name of the label inside the output ``fitness_tensors`` mapping.
    embedding_mode : {"hard", "soft"}, default="hard"
        Tokenization/embedding strategy.
    model_name : str, default="facebook/esm2_t6_8M_UR50D"
        HuggingFace model identifier used by landscapy.
    device : str or None, optional
        Device string forwarded to the embedder.
    embedding_batch_size : int, default=32
        Batch size used during embedding.
    include_tokens : bool, default=True
        Whether to include padded token and attention mask tensors.

    Returns
    -------
    list[dict[str, Any]]
        Record dictionaries containing ``sequence_tensor`` or ``embedding``, ``fitness_tensors``,
        and optional ``attention_mask``.

    Raises
    ------
    ValueError
        If the lengths of sequences and labels differ.
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

    records: List[Dict[str, Any]] = []
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


class SequenceClassificationDataset(LandscapeDataset):
    """
    Classification specialization built on top of the generic landscape dataset.
    """

    def __init__(self, records: Sequence[Dict[str, Any]], label_key: str) -> None:
        self.label_key = label_key
        super().__init__(
            records,
            input_getter=make_preferred_input_getter("embedding", "sequence_tensor"),
            target_getter=make_fitness_target_getter(
                label_key,
                collapse_one_hot=True,
                dtype=torch.long,
            ),
        )


class SequenceClassificationDataModule(LandscapeDataModule):
    """
    Classification-focused wrapper around the generic landscape datamodule.
    """

    def __init__(
        self,
        *,
        train_data: Any,
        label_key: str,
        label_mapping: Optional[Sequence[str]] = None,
        val_data: Any = None,
        test_data: Any = None,
        predict_data: Any = None,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle: bool = True,
        val_split: float = 0.0,
        val_seed: Optional[int] = None,
    ) -> None:
        self.label_key = label_key
        self.label_mapping = list(label_mapping) if label_mapping is not None else None
        super().__init__(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            predict_data=predict_data,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            val_split=val_split,
            val_seed=val_seed,
            dataset_factory=SequenceClassificationDataset,
            dataset_kwargs={"label_key": label_key},
            predict_dataset_kwargs={"label_key": label_key},
        )

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
        Embed raw sequences with landscapy and construct a DataModule.

        Parameters
        ----------
        train_sequences : Sequence[SequenceLike]
            Raw sequences for training.
        train_labels : Sequence[LabelLike]
            Labels aligned with ``train_sequences``.
        label_key : str
            Label key used inside ``fitness_tensors``.
        val_sequences : Sequence[SequenceLike], optional
            Validation sequences.
        val_labels : Sequence[LabelLike], optional
            Validation labels.
        test_sequences : Sequence[SequenceLike], optional
            Test sequences.
        test_labels : Sequence[LabelLike], optional
            Test labels.
        predict_sequences : Sequence[SequenceLike], optional
            Sequences for prediction.
        predict_labels : Sequence[LabelLike], optional
            Optional labels for prediction records.
        embedding_mode : {"hard", "soft"}, default="hard"
            Tokenization/embedding strategy.
        model_name : str, default="facebook/esm2_t6_8M_UR50D"
            HuggingFace model identifier used by landscapy.
        device : str or None, optional
            Device string forwarded to the embedder.
        embedding_batch_size : int, default=32
            Batch size used during embedding.
        include_tokens : bool, default=True
            Whether to include token/attention tensors.
        **datamodule_kwargs : Any
            Additional keyword arguments forwarded to ``SequenceClassificationDataModule``.

        Returns
        -------
        SequenceClassificationDataModule
            Initialized DataModule containing embedded records.

        Raises
        ------
        ValueError
            If sequences and labels are not provided in matching pairs.
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
