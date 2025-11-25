from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

import torch

from .data import embed_sequences
from .gp_classification import SequenceGPClassifier


def predict_sequences(
    model: SequenceGPClassifier,
    sequences: Sequence[Any],
    *,
    embedding_mode: str = "hard",
    model_name: str = "facebook/esm2_t6_8M_UR50D",
    device: Optional[str] = None,
    embedding_batch_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Embed raw sequences and return (mean_probs, variance) from the classifier.
    """
    model.eval()
    embeddings, _, _ = embed_sequences(
        sequences,
        embedding_mode=embedding_mode,
        model_name=model_name,
        device=device,
        embedding_batch_size=embedding_batch_size,
        include_tokens=False,
    )
    embeddings = embeddings.to(model.device)
    mean_probs, variance = model.predict_with_uncertainty(embeddings)
    return mean_probs.cpu(), variance.cpu()


def predict_landscape_records(
    model: SequenceGPClassifier,
    records: Iterable[Mapping[str, Any]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run inference on records exported from FitnessLandscape.to_sequence_tensors.
    """
    model.eval()
    feats = []
    for rec in records:
        feature = rec.get("embedding", rec.get("sequence_tensor"))
        if feature is None:
            raise ValueError("Record missing 'embedding' or 'sequence_tensor'.")
        tensor = torch.as_tensor(feature, dtype=torch.float32)
        feats.append(tensor)
    inputs = torch.stack(feats, dim=0).to(model.device)
    mean_probs, variance = model.predict_with_uncertainty(inputs)
    return mean_probs.cpu(), variance.cpu()
