"""Real/Bogus inference engine: robust preprocessing, mini-batching, validated outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from LUMIA.core.runtime import configure_deterministic_torch, resolve_torch_device
from LUMIA.core.schemas import (
    CandidateDetection,
    ClassificationResult,
    CutoutBatchResult,
    CutoutStampSize,
)
from LUMIA.ml.config import CNNClassifierConfig
from LUMIA.ml.models import INPUT_CHANNELS, RealBogusNet, build_model

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

__all__ = [
    "ClassificationResult",
    "CutoutClassificationBatchResult",
    "InferenceMemoryError",
    "RealBogusClassifier",
    "attach_classifications_to_detections",
    "classify_cutout_batch",
    "preprocess_triplets",
]

_MAD_TO_SIGMA: float = 1.4826
_MIN_SCALE: float = 1e-6
_CLIP_SIGMA: float = 100.0


class InferenceMemoryError(MemoryError):
    """Raised when a GPU out-of-memory condition occurs during mini-batch inference."""


class CutoutClassificationBatchResult(BaseModel):
    """Validated batch classification contract bridging CutoutBatchResult to Module 5/2."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    classifications: tuple[ClassificationResult, ...]
    accepted_candidate_ids: tuple[str, ...]
    rejected_candidate_ids: tuple[str, ...]
    stamp_size: CutoutStampSize
    model_version: str = Field(..., min_length=1)
    decision_threshold: float = Field(..., ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_partition_invariants(self) -> CutoutClassificationBatchResult:
        seen_ids = [item.candidate_id for item in self.classifications]
        if len(set(seen_ids)) != len(seen_ids):
            msg = "Duplicate candidate_id entries are prohibited in classifications."
            raise ValueError(msg)
        for item in self.classifications:
            if item.model_version != self.model_version:
                msg = (
                    f"ClassificationResult model_version {item.model_version!r} does not "
                    f"match batch model_version {self.model_version!r}."
                )
                raise ValueError(msg)
        expected_accepted = tuple(
            item.candidate_id for item in self.classifications if item.is_real
        )
        expected_rejected = tuple(
            item.candidate_id for item in self.classifications if not item.is_real
        )
        if self.accepted_candidate_ids != expected_accepted:
            msg = "accepted_candidate_ids does not match is_real=True classifications."
            raise ValueError(msg)
        if self.rejected_candidate_ids != expected_rejected:
            msg = "rejected_candidate_ids does not match is_real=False classifications."
            raise ValueError(msg)
        return self

    def score_map(self) -> dict[str, ClassificationResult]:
        """Return an immutable-entry lookup dictionary keyed by ``candidate_id``."""
        return {item.candidate_id: item for item in self.classifications}


def _validate_triplet_shape(x: torch.Tensor) -> None:
    if x.ndim != 4:
        msg = f"Expected 4D tensor (B, 3, H, W), got shape {tuple(x.shape)}."
        raise ValueError(msg)
    if x.shape[1] != INPUT_CHANNELS:
        msg = (
            f"Expected {INPUT_CHANNELS} channels (Science, Reference, Difference), "
            f"got {x.shape[1]}."
        )
        raise ValueError(msg)


def _masked_lower_median(values: torch.Tensor, finite: torch.Tensor) -> torch.Tensor:
    """Lower median over the last dim ignoring non-finite entries (0 if none finite).

    Implemented with ``torch.sort``/``gather`` so it is deterministic on CPU and CUDA.
    """
    sortable = torch.where(finite, values, torch.full_like(values, float("inf")))
    sorted_vals, _ = torch.sort(sortable, dim=-1)
    n_finite = finite.sum(dim=-1, keepdim=True)
    index = torch.clamp((n_finite - 1) // 2, min=0)
    median = torch.gather(sorted_vals, -1, index)
    return torch.where(n_finite > 0, median, torch.zeros_like(median))


def preprocess_triplets(triplets: torch.Tensor) -> torch.Tensor:
    """Robustly normalize each channel of each cutout (median/MAD) and sanitize pixels.

    Non-finite pixels are replaced by the channel median (zero after centering). The
    input tensor is never modified.
    """
    _validate_triplet_shape(triplets)
    x = triplets.to(dtype=torch.float32)
    batch, channels, height, width = x.shape
    flat = x.reshape(batch, channels, height * width)

    finite = torch.isfinite(flat)
    median = _masked_lower_median(flat, finite)
    centered = torch.where(finite, flat - median, torch.zeros_like(flat))

    mad = _masked_lower_median(centered.abs(), finite)
    scale = torch.clamp(_MAD_TO_SIGMA * mad, min=_MIN_SCALE)

    normalized = torch.clamp(centered / scale, min=-_CLIP_SIGMA, max=_CLIP_SIGMA)
    return normalized.reshape(batch, channels, height, width)


class RealBogusClassifier:
    """Deterministic CNN Real/Bogus classifier with device routing and mini-batching."""

    def __init__(self, config: CNNClassifierConfig | None = None) -> None:
        self.config: CNNClassifierConfig = (
            config if config is not None else CNNClassifierConfig()
        )
        # Resolve the device first so an unavailable GPU fails before any allocation.
        self.device: torch.device = resolve_torch_device(self.config.device)

        configure_deterministic_torch(
            self.config.seed,
            deterministic=self.config.deterministic,
        )
        if self.config.num_cpu_threads is not None:
            torch.set_num_threads(self.config.num_cpu_threads)

        # Weights are drawn on CPU from the seeded global RNG so that they are
        # identical regardless of the execution device.
        model: RealBogusNet = build_model(self.config)
        if self.config.checkpoint_path is not None:
            state = torch.load(
                self.config.checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            if not isinstance(state, Mapping):
                msg = (
                    "Checkpoint state_dict must be a mapping of parameter tensors, "
                    f"got {type(state).__name__}."
                )
                raise ValueError(msg)
            model.load_state_dict(state)
        self.model: RealBogusNet = model.to(self.device).eval()

    def _to_tensor(self, triplets: torch.Tensor | ArrayLike) -> torch.Tensor:
        tensor = (
            triplets
            if isinstance(triplets, torch.Tensor)
            else torch.as_tensor(np.asarray(triplets), dtype=torch.float32)
        )
        _validate_triplet_shape(tensor)
        size = self.config.stamp_size
        if tensor.shape[2] != size or tensor.shape[3] != size:
            msg = (
                f"Expected stamp size {size}x{size}, got "
                f"{tensor.shape[2]}x{tensor.shape[3]}."
            )
            raise ValueError(msg)
        return tensor

    def predict_scores(
        self,
        triplets: CutoutBatchResult | torch.Tensor | ArrayLike,
    ) -> torch.Tensor:
        """Return Real probabilities of shape (N,) on CPU for (N, 3, H, W) cutouts."""
        raw_input: torch.Tensor | ArrayLike = (
            triplets.triplets if isinstance(triplets, CutoutBatchResult) else triplets
        )
        tensor = self._to_tensor(raw_input)
        n_samples = int(tensor.shape[0])
        if n_samples == 0:
            return torch.empty(0, dtype=torch.float32)

        self.model.eval()
        chunk_size = self.config.inference_batch_size
        outputs: list[torch.Tensor] = []
        try:
            with torch.no_grad():
                for start in range(0, n_samples, chunk_size):
                    chunk = tensor[start : start + chunk_size].to(self.device)
                    probs = self.model(preprocess_triplets(chunk))
                    outputs.append(probs.cpu())
        except torch.OutOfMemoryError as exc:
            msg = (
                "GPU out-of-memory during CNN inference; "
                f"reduce inference_batch_size: {exc}"
            )
            raise InferenceMemoryError(msg) from exc
        return torch.cat(outputs, dim=0)

    def predict(
        self,
        triplets: CutoutBatchResult | torch.Tensor | ArrayLike,
        candidate_ids: Sequence[str] | None = None,
    ) -> list[ClassificationResult]:
        """Classify a ``CutoutBatchResult`` or ``(B, 3, S, S)`` tensor into Real/Bogus scores."""
        if isinstance(triplets, CutoutBatchResult):
            batch_ids = triplets.candidate_ids
            if candidate_ids is not None and list(candidate_ids) != batch_ids:
                msg = (
                    "Explicit candidate_ids do not match "
                    "CutoutBatchResult.candidate_ids."
                )
                raise ValueError(msg)
            resolved_ids: Sequence[str] = batch_ids
            tensor = self._to_tensor(triplets.triplets)
        else:
            if candidate_ids is None:
                msg = (
                    "candidate_ids must be provided when classifying a raw tensor or array."
                )
                raise ValueError(msg)
            resolved_ids = candidate_ids
            tensor = self._to_tensor(triplets)

        if len(resolved_ids) != int(tensor.shape[0]):
            msg = (
                f"candidate_ids length ({len(resolved_ids)}) must match batch size "
                f"({int(tensor.shape[0])})."
            )
            raise ValueError(msg)

        scores = self.predict_scores(tensor).tolist()
        return [
            ClassificationResult(
                candidate_id=cand_id,
                score=min(max(float(score), 0.0), 1.0),
                is_real=float(score) >= self.config.decision_threshold,
                model_version=self.config.model_version,
            )
            for cand_id, score in zip(resolved_ids, scores, strict=True)
        ]


def classify_cutout_batch(
    classifier: RealBogusClassifier,
    batch: CutoutBatchResult,
) -> CutoutClassificationBatchResult:
    """Pure bridge classifying an in-memory ``CutoutBatchResult`` into a validated summary."""
    if batch.stamp_size != classifier.config.stamp_size:
        msg = (
            f"CutoutBatchResult stamp_size ({batch.stamp_size}) does not match "
            f"classifier stamp_size ({classifier.config.stamp_size})."
        )
        raise ValueError(msg)

    predictions = classifier.predict(batch)
    accepted = tuple(item.candidate_id for item in predictions if item.is_real)
    rejected = tuple(item.candidate_id for item in predictions if not item.is_real)
    return CutoutClassificationBatchResult(
        classifications=tuple(predictions),
        accepted_candidate_ids=accepted,
        rejected_candidate_ids=rejected,
        stamp_size=batch.stamp_size,
        model_version=classifier.config.model_version,
        decision_threshold=classifier.config.decision_threshold,
    )


def attach_classifications_to_detections(
    detections: Sequence[CandidateDetection],
    classifications: CutoutClassificationBatchResult | Sequence[ClassificationResult],
    *,
    filter_real_only: bool = False,
) -> list[CandidateDetection]:
    """Purely return immutable ``CandidateDetection`` copies populated with CNN provenance."""
    items: Sequence[ClassificationResult] = (
        classifications.classifications
        if isinstance(classifications, CutoutClassificationBatchResult)
        else classifications
    )
    lookup: dict[str, ClassificationResult] = {}
    for cls_res in items:
        if cls_res.candidate_id in lookup:
            msg = f"Duplicate ClassificationResult for candidate_id {cls_res.candidate_id!r}."
            raise ValueError(msg)
        lookup[cls_res.candidate_id] = cls_res

    enriched: list[CandidateDetection] = []
    for det in detections:
        if det.candidate_id not in lookup:
            msg = f"Missing ClassificationResult for candidate_id {det.candidate_id!r}."
            raise KeyError(msg)
        matched = lookup[det.candidate_id]
        if filter_real_only and not matched.is_real:
            continue
        enriched.append(
            CandidateDetection(
                candidate_id=det.candidate_id,
                field_id=det.field_id,
                ra_deg=det.ra_deg,
                dec_deg=det.dec_deg,
                bjd_tdb=det.bjd_tdb,
                mag=det.mag,
                snr=det.snr,
                cnn_score=matched.score,
                model_version=matched.model_version,
                astrometric_uncertainty_arcsec=det.astrometric_uncertainty_arcsec,
                astrometric_residual_arcsec=det.astrometric_residual_arcsec,
            )
        )
    return enriched
