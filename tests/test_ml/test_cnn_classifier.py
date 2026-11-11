"""Strict TDD tests for Module 7: PyTorch CNN Real/Bogus classifier."""

from __future__ import annotations

import ast
import builtins
from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from LUMIA.core.schemas import (
    CandidateDetection,
    CutoutBatchResult,
    SNRPeakCandidate,
)
from LUMIA.ml.classifier import (
    ClassificationResult,
    CutoutClassificationBatchResult,
    RealBogusClassifier,
    attach_classifications_to_detections,
    classify_cutout_batch,
    preprocess_triplets,
)
from LUMIA.ml.config import CNNClassifierConfig
from LUMIA.ml.models import ResNetLiteClassifier, VGGClassifier, build_model
from LUMIA.ml.runtime import configure_deterministic_torch, resolve_torch_device
from LUMIA.physics.cutout_extraction import (
    CutoutConfig,
    PeakExtractionConfig,
    extract_zogy_cutout_batch,
)
from LUMIA.physics.gpu_processing import ZOGYConfig, zogy_subtract

_REPO_ROOT: Path = Path(__file__).resolve().parents[2]

BACKBONES: list[str] = ["vgg", "resnet_lite"]
STAMP_SIZES: list[int] = [31, 51]


def _random_triplets(
    batch: int,
    size: int,
    *,
    seed: int = 7,
) -> torch.Tensor:
    """Build a batched (B, 3, H, W) Science/Reference/Difference cutout tensor."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, 3, size, size, generator=generator) * 5.0 + 100.0


class TestCNNConfigValidation:
    """Pydantic validation of the CNN classifier configuration."""

    def test_defaults_are_valid(self) -> None:
        cfg = CNNClassifierConfig()
        assert cfg.backbone in {"vgg", "resnet_lite"}
        assert cfg.stamp_size in {31, 51}
        assert 0.0 <= cfg.dropout_rate < 1.0
        assert 0.0 <= cfg.decision_threshold <= 1.0
        assert cfg.deterministic is True

    @pytest.mark.parametrize("backbone", BACKBONES)
    @pytest.mark.parametrize("size", STAMP_SIZES)
    def test_accepts_supported_backbones_and_resolutions(
        self,
        backbone: str,
        size: int,
    ) -> None:
        cfg = CNNClassifierConfig(
            backbone=backbone,  # type: ignore[arg-type]
            stamp_size=size,  # type: ignore[arg-type]
            dropout_rate=0.3,
        )
        assert cfg.backbone == backbone
        assert cfg.stamp_size == size
        assert cfg.dropout_rate == pytest.approx(0.3)

    @pytest.mark.parametrize(
        "bad_kwargs",
        [
            {"backbone": "alexnet"},
            {"stamp_size": 30},  # even-sized stamp has no center pixel
            {"stamp_size": 41},  # odd but unsupported resolution
            {"stamp_size": "51"},  # strict mode forbids str -> int coercion
            {"dropout_rate": 1.0},
            {"dropout_rate": -0.1},
            {"inference_batch_size": 0},
            {"decision_threshold": 1.5},
            {"decision_threshold": -0.2},
            {"device": "tpu"},
            {"seed": -1},
            {"model_version": ""},
            {"unknown_field": 1},
        ],
    )
    def test_rejects_invalid_parameters(self, bad_kwargs: dict[str, object]) -> None:
        with pytest.raises(ValidationError):
            CNNClassifierConfig(**bad_kwargs)  # type: ignore[arg-type]

    def test_config_is_immutable(self) -> None:
        cfg = CNNClassifierConfig()
        with pytest.raises(ValidationError):
            cfg.dropout_rate = 0.9  # type: ignore[misc]


class TestModelArchitectures:
    """VGG-style and ResNet-Lite backbones accept (B, 3, H, W) and emit probabilities."""

    @pytest.mark.parametrize("backbone", BACKBONES)
    @pytest.mark.parametrize("size", STAMP_SIZES)
    def test_forward_returns_sigmoid_probabilities(
        self,
        backbone: str,
        size: int,
    ) -> None:
        cfg = CNNClassifierConfig(
            backbone=backbone,  # type: ignore[arg-type]
            stamp_size=size,  # type: ignore[arg-type]
        )
        configure_deterministic_torch(cfg.seed, deterministic=True)
        model = build_model(cfg).eval()

        batch = torch.randn(4, 3, size, size)
        with torch.no_grad():
            probs = model(batch)
            logits = model.forward_logits(batch)

        assert probs.shape == (4,)
        assert logits.shape == (4,)
        assert probs.dtype == torch.float32
        assert bool(torch.all(torch.isfinite(probs)))
        assert float(probs.min()) >= 0.0
        assert float(probs.max()) <= 1.0
        assert torch.allclose(probs, torch.sigmoid(logits), atol=1e-6)

    def test_build_model_dispatches_on_backbone(self) -> None:
        vgg = build_model(CNNClassifierConfig(backbone="vgg"))
        res = build_model(CNNClassifierConfig(backbone="resnet_lite"))
        assert isinstance(vgg, VGGClassifier)
        assert isinstance(res, ResNetLiteClassifier)

    @pytest.mark.parametrize("backbone", BACKBONES)
    def test_models_are_resolution_agnostic_via_global_pooling(
        self,
        backbone: str,
    ) -> None:
        """Global adaptive pooling lets one model handle both supported stamp sizes."""
        model = build_model(
            CNNClassifierConfig(backbone=backbone),  # type: ignore[arg-type]
        ).eval()
        with torch.no_grad():
            for size in STAMP_SIZES:
                out = model(torch.randn(2, 3, size, size))
                assert out.shape == (2,)

    @pytest.mark.parametrize("backbone", BACKBONES)
    def test_rejects_wrong_channel_count(self, backbone: str) -> None:
        model = build_model(
            CNNClassifierConfig(backbone=backbone),  # type: ignore[arg-type]
        ).eval()
        with pytest.raises(ValueError, match="channel"):
            model(torch.randn(2, 2, 31, 31))

    @pytest.mark.parametrize("backbone", BACKBONES)
    def test_rejects_non_4d_input(self, backbone: str) -> None:
        model = build_model(
            CNNClassifierConfig(backbone=backbone),  # type: ignore[arg-type]
        ).eval()
        with pytest.raises(ValueError, match="4D"):
            model(torch.randn(3, 31, 31))

    @pytest.mark.parametrize("backbone", BACKBONES)
    def test_logits_support_gradient_for_training(self, backbone: str) -> None:
        """Training path uses logits + BCE-with-logits; gradients must flow."""
        model = build_model(
            CNNClassifierConfig(backbone=backbone),  # type: ignore[arg-type]
        ).train()
        logits = model.forward_logits(torch.randn(4, 3, 31, 31))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            torch.tensor([1.0, 0.0, 1.0, 0.0]),
        )
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert all(g is not None for g in grads)
        assert any(float(g.abs().sum()) > 0.0 for g in grads if g is not None)


class TestDeterminismAndDeviceRouting:
    """Deterministic PyTorch flags and CPU/GPU routing."""

    def test_configure_deterministic_torch_sets_global_flags(self) -> None:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = True

        configure_deterministic_torch(123, deterministic=True)

        assert torch.are_deterministic_algorithms_enabled() is True
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False

    def test_classifier_construction_enforces_deterministic_execution(self) -> None:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = True

        RealBogusClassifier(CNNClassifierConfig(device="cpu"))

        assert torch.are_deterministic_algorithms_enabled() is True
        assert torch.backends.cudnn.benchmark is False

    @pytest.mark.parametrize("backbone", BACKBONES)
    def test_same_seed_gives_identical_weights_and_scores(self, backbone: str) -> None:
        cfg = CNNClassifierConfig(
            backbone=backbone,  # type: ignore[arg-type]
            device="cpu",
            seed=2026,
        )
        first = RealBogusClassifier(cfg)
        second = RealBogusClassifier(cfg)

        for p1, p2 in zip(
            first.model.parameters(),
            second.model.parameters(),
            strict=True,
        ):
            assert torch.equal(p1, p2)

        data = _random_triplets(6, cfg.stamp_size)
        assert torch.equal(first.predict_scores(data), second.predict_scores(data))

    def test_different_seeds_give_different_weights(self) -> None:
        a = RealBogusClassifier(CNNClassifierConfig(device="cpu", seed=1))
        b = RealBogusClassifier(CNNClassifierConfig(device="cpu", seed=2))
        first_a = next(a.model.parameters())
        first_b = next(b.model.parameters())
        assert not torch.equal(first_a, first_b)

    def test_classifier_uses_eval_mode_and_disables_gradients(self) -> None:
        clf = RealBogusClassifier(CNNClassifierConfig(device="cpu"))
        assert clf.model.training is False

        scores = clf.predict_scores(_random_triplets(3, 31))
        assert scores.requires_grad is False

    def test_cpu_device_routing(self) -> None:
        clf = RealBogusClassifier(CNNClassifierConfig(device="cpu"))
        assert clf.device == torch.device("cpu")
        assert all(p.device.type == "cpu" for p in clf.model.parameters())

    def test_resolve_auto_matches_cuda_availability(self) -> None:
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        assert resolve_torch_device("auto").type == expected

    @pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA is available")
    def test_explicit_cuda_without_gpu_fails_fast(self) -> None:
        with pytest.raises(RuntimeError, match="CUDA"):
            RealBogusClassifier(CNNClassifierConfig(device="cuda"))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device_routing_and_cpu_inputs_are_moved(self) -> None:
        clf = RealBogusClassifier(CNNClassifierConfig(device="cuda"))
        assert all(p.device.type == "cuda" for p in clf.model.parameters())
        scores = clf.predict_scores(_random_triplets(3, 31))
        assert scores.device.type == "cpu"
        assert scores.shape == (3,)


class TestPreprocessing:
    """Robust per-channel normalization and non-finite sanitization."""

    def test_median_centering_and_mad_scaling(self) -> None:
        data = _random_triplets(4, 31)
        out = preprocess_triplets(data)

        assert out.shape == data.shape
        assert bool(torch.all(torch.isfinite(out)))
        medians = out.flatten(2).median(dim=-1).values
        assert float(medians.abs().max()) < 1e-4

        mad = (out - medians[..., None, None]).abs().flatten(2).median(dim=-1).values
        # Scaling by 1.4826 * MAD leaves a unit MAD of 1 / 1.4826 in every channel.
        assert torch.allclose(mad, torch.full_like(mad, 1.0 / 1.4826), rtol=1e-3)

    def test_invariant_to_channel_gain_and_offset(self) -> None:
        data = _random_triplets(2, 31)
        scaled = data * 10.0 + 500.0
        assert torch.allclose(
            preprocess_triplets(data),
            preprocess_triplets(scaled),
            atol=1e-3,
        )

    def test_replaces_non_finite_pixels(self) -> None:
        data = _random_triplets(2, 31)
        data[0, 0, 0, 0] = float("nan")
        data[0, 1, 5, 5] = float("inf")
        data[1, 2, 30, 30] = float("-inf")

        out = preprocess_triplets(data)
        assert bool(torch.all(torch.isfinite(out)))

    def test_constant_and_fully_masked_channels_stay_finite(self) -> None:
        data = _random_triplets(2, 31)
        data[0, 0] = 42.0  # zero-MAD flat channel
        data[1, 1] = float("nan")  # fully masked channel

        out = preprocess_triplets(data)
        assert bool(torch.all(torch.isfinite(out)))
        assert float(out[0, 0].abs().max()) == pytest.approx(0.0, abs=1e-6)

    def test_does_not_mutate_input(self) -> None:
        data = _random_triplets(2, 31)
        data[0, 0, 0, 0] = float("nan")
        before = data.clone()
        preprocess_triplets(data)
        assert torch.equal(torch.nan_to_num(data, nan=-1.0), torch.nan_to_num(before, nan=-1.0))

    def test_rejects_invalid_shapes(self) -> None:
        with pytest.raises(ValueError, match="4D"):
            preprocess_triplets(torch.zeros(3, 31, 31))
        with pytest.raises(ValueError, match="channel"):
            preprocess_triplets(torch.zeros(1, 4, 31, 31))


class TestClassifierInference:
    """End-to-end Real/Bogus inference with validated result schemas."""

    @pytest.mark.parametrize("backbone", BACKBONES)
    @pytest.mark.parametrize("size", STAMP_SIZES)
    def test_predict_scores_in_unit_interval(self, backbone: str, size: int) -> None:
        clf = RealBogusClassifier(
            CNNClassifierConfig(
                backbone=backbone,  # type: ignore[arg-type]
                stamp_size=size,  # type: ignore[arg-type]
                device="cpu",
            ),
        )
        scores = clf.predict_scores(_random_triplets(5, size))
        assert scores.shape == (5,)
        assert bool(torch.all((scores >= 0.0) & (scores <= 1.0)))

    def test_accepts_numpy_input(self) -> None:
        clf = RealBogusClassifier(CNNClassifierConfig(device="cpu", stamp_size=31))
        arr = np.random.default_rng(0).normal(size=(3, 3, 31, 31)).astype(np.float64)
        scores = clf.predict_scores(arr)
        assert scores.shape == (3,)
        assert scores.dtype == torch.float32

    def test_minibatching_matches_single_pass(self) -> None:
        data = _random_triplets(7, 31)
        small = RealBogusClassifier(
            CNNClassifierConfig(device="cpu", stamp_size=31, inference_batch_size=2),
        )
        large = RealBogusClassifier(
            CNNClassifierConfig(device="cpu", stamp_size=31, inference_batch_size=64),
        )
        assert torch.allclose(
            small.predict_scores(data),
            large.predict_scores(data),
            atol=1e-6,
        )

    def test_empty_batch_returns_empty_results(self) -> None:
        clf = RealBogusClassifier(CNNClassifierConfig(device="cpu", stamp_size=31))
        assert clf.predict_scores(torch.zeros(0, 3, 31, 31)).shape == (0,)
        assert clf.predict(torch.zeros(0, 3, 31, 31), []) == []

    def test_predict_returns_validated_result_schema(self) -> None:
        cfg = CNNClassifierConfig(
            device="cpu",
            stamp_size=31,
            decision_threshold=0.5,
            model_version="rb-test-0.1",
        )
        clf = RealBogusClassifier(cfg)
        ids = ["cand-001", "cand-002", "cand-003"]
        data = _random_triplets(3, 31)

        results = clf.predict(data, ids)
        scores = clf.predict_scores(data)

        assert [r.candidate_id for r in results] == ids
        for result, score in zip(results, scores.tolist(), strict=True):
            assert isinstance(result, ClassificationResult)
            assert 0.0 <= result.score <= 1.0
            assert result.score == pytest.approx(score, abs=1e-6)
            assert result.is_real == (result.score >= cfg.decision_threshold)
            assert result.model_version == "rb-test-0.1"

    def test_threshold_controls_pass_reject_flag(self) -> None:
        data = _random_triplets(4, 31)
        ids = [f"c{i}" for i in range(4)]
        accept_all = RealBogusClassifier(
            CNNClassifierConfig(device="cpu", stamp_size=31, decision_threshold=0.0),
        ).predict(data, ids)
        reject_all = RealBogusClassifier(
            CNNClassifierConfig(device="cpu", stamp_size=31, decision_threshold=1.0),
        ).predict(data, ids)
        assert all(r.is_real for r in accept_all)
        # Scores strictly below 1.0 are rejected at threshold 1.0.
        assert all((not r.is_real) or r.score >= 1.0 for r in reject_all)

    def test_non_finite_pixels_do_not_produce_nan_scores(self) -> None:
        clf = RealBogusClassifier(CNNClassifierConfig(device="cpu", stamp_size=31))
        data = _random_triplets(2, 31)
        data[0, 2, 10, 10] = float("nan")
        data[1, 0, :, 0] = float("inf")
        assert bool(torch.all(torch.isfinite(clf.predict_scores(data))))

    def test_rejects_shape_mismatches(self) -> None:
        clf = RealBogusClassifier(CNNClassifierConfig(device="cpu", stamp_size=51))

        with pytest.raises(ValueError, match="stamp"):
            clf.predict_scores(_random_triplets(2, 31))
        with pytest.raises(ValueError, match="channel"):
            clf.predict_scores(torch.zeros(2, 2, 51, 51))
        with pytest.raises(ValueError, match="4D"):
            clf.predict_scores(torch.zeros(3, 51, 51))
        with pytest.raises(ValueError, match="candidate_ids"):
            clf.predict(_random_triplets(2, 51), ["only-one"])

    def test_classification_result_schema_validation(self) -> None:
        with pytest.raises(ValidationError):
            ClassificationResult(
                candidate_id="x",
                score=1.2,
                is_real=True,
                model_version="v",
            )
        with pytest.raises(ValidationError):
            ClassificationResult(
                candidate_id="",
                score=0.5,
                is_real=True,
                model_version="v",
            )


class TestCheckpointLoading:
    """Optional state_dict checkpoint loading (no real training is performed)."""

    def test_loads_checkpoint_weights(self, tmp_path: Path) -> None:
        source = RealBogusClassifier(CNNClassifierConfig(device="cpu", seed=11))
        ckpt = tmp_path / "rb.pt"
        torch.save(source.model.state_dict(), ckpt)

        restored = RealBogusClassifier(
            CNNClassifierConfig(device="cpu", seed=99, checkpoint_path=ckpt),
        )
        data = _random_triplets(4, 31)
        assert torch.equal(source.predict_scores(data), restored.predict_scores(data))

    def test_missing_checkpoint_fails_fast(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            RealBogusClassifier(
                CNNClassifierConfig(device="cpu", checkpoint_path=tmp_path / "nope.pt"),
            )


class TestCutoutExtractionBridgeIntegration:
    """Verify Module 4 -> Module 7 in-memory CutoutBatchResult classification bridge."""

    @pytest.mark.parametrize("backbone", BACKBONES)
    @pytest.mark.parametrize("size", STAMP_SIZES)
    def test_predict_and_classify_cutout_batch_from_zogy_stamps(
        self,
        backbone: str,
        size: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify ZOGY cutouts flow into RealBogusClassifier.predict with zero disk I/O."""
        rng = np.random.default_rng(1042)
        science = rng.normal(120.0, 1.1, size=(112, 112)).astype(np.float32)
        reference = rng.normal(120.0, 1.1, size=(112, 112)).astype(np.float32)

        center = 7
        yy, xx = np.mgrid[0:15, 0:15]
        psf = np.exp(-0.5 * ((xx - center) ** 2 + (yy - center) ** 2) / (1.5**2))
        psf = (psf / np.sum(psf)).astype(np.float32)

        # Inject two science-only moving transients
        img_y, img_x = np.mgrid[0:112, 0:112]
        for x0, y0, amp in [(34.0, 42.0, 42.0), (76.0, 68.0, 28.0)]:
            science += (
                amp * np.exp(-0.5 * ((img_x - x0) ** 2 + (img_y - y0) ** 2) / (1.5**2))
            ).astype(np.float32)

        zogy_res = zogy_subtract(
            science,
            reference,
            psf,
            psf,
            ZOGYConfig(
                flux_scale_science=1.0,
                flux_scale_reference=1.0,
                noise_std_science=1.1,
                noise_std_reference=1.1,
            ),
        )

        clf = RealBogusClassifier(
            CNNClassifierConfig(
                backbone=backbone,  # type: ignore[arg-type]
                stamp_size=size,  # type: ignore[arg-type]
                device="cpu",
                decision_threshold=0.5,
                seed=314,
                model_version="lumia-rb-m4m7-1.0",
            )
        )

        def _forbid_io(*args: object, **kwargs: object) -> object:
            msg = f"Unexpected filesystem access during bridge execution: {args!r}"
            raise AssertionError(msg)

        monkeypatch.setattr(builtins, "open", _forbid_io)
        monkeypatch.setattr(Path, "open", _forbid_io)

        cutout_batch: CutoutBatchResult = extract_zogy_cutout_batch(
            science,
            reference,
            zogy_res,
            peak_config=PeakExtractionConfig(snr_threshold=6.0, id_prefix="bridge"),
            cutout_config=CutoutConfig(
                stamp_size=size,  # type: ignore[arg-type]
                boundary_mode="pad",
            ),
        )
        assert len(cutout_batch.candidates) == 2

        direct_preds = clf.predict(cutout_batch)
        tensor_preds = clf.predict(cutout_batch.triplets, cutout_batch.candidate_ids)
        direct_scores = clf.predict_scores(cutout_batch)
        batch_summary: CutoutClassificationBatchResult = classify_cutout_batch(
            clf,
            cutout_batch,
        )

        assert direct_preds == tensor_preds
        assert list(batch_summary.classifications) == direct_preds
        assert torch.allclose(
            direct_scores,
            clf.predict_scores(cutout_batch.triplets),
            atol=1e-6,
        )
        assert batch_summary.stamp_size == size
        assert batch_summary.model_version == "lumia-rb-m4m7-1.0"
        assert set(batch_summary.accepted_candidate_ids).isdisjoint(
            set(batch_summary.rejected_candidate_ids)
        )
        assert (
            len(batch_summary.accepted_candidate_ids)
            + len(batch_summary.rejected_candidate_ids)
        ) == 2
        assert set(batch_summary.score_map().keys()) == set(cutout_batch.candidate_ids)

    def test_predict_and_bridge_validate_candidate_ids_and_stamp_sizes(self) -> None:
        """Verify mismatched candidate_ids, missing IDs, and stamp mismatches fail fast."""
        clf_31 = RealBogusClassifier(CNNClassifierConfig(device="cpu", stamp_size=31))
        clf_51 = RealBogusClassifier(CNNClassifierConfig(device="cpu", stamp_size=51))

        peak = SNRPeakCandidate(
            candidate_id="cand-bridge-00",
            x_px=25,
            y_px=25,
            x_centroid_px=25.0,
            y_centroid_px=25.0,
            snr=9.2,
        )
        batch_31 = CutoutBatchResult(
            triplets=_random_triplets(1, 31),
            candidates=[peak],
            stamp_size=31,
        )

        # Missing candidate_ids when passing raw tensor must raise ValueError
        with pytest.raises(ValueError, match="candidate_ids"):
            clf_31.predict(_random_triplets(1, 31))

        # Conflicting explicit candidate_ids alongside CutoutBatchResult must fail fast
        with pytest.raises(ValueError, match="candidate_ids"):
            clf_31.predict(batch_31, ["wrong-id"])

        # Stamp size mismatch (31 batch into 51 classifier) must fail fast
        with pytest.raises(ValueError, match="stamp"):
            classify_cutout_batch(clf_51, batch_31)

        # Empty CutoutBatchResult returns empty CutoutClassificationBatchResult cleanly
        empty_batch = CutoutBatchResult(
            triplets=torch.empty((0, 3, 31, 31), dtype=torch.float32),
            candidates=[],
            stamp_size=31,
        )
        empty_res = classify_cutout_batch(clf_31, empty_batch)
        assert empty_res.classifications == ()
        assert empty_res.accepted_candidate_ids == ()
        assert empty_res.rejected_candidate_ids == ()

        # Strict ADR-0002 policy check on CutoutClassificationBatchResult
        cfg = CutoutClassificationBatchResult.model_config
        assert cfg.get("strict") is True
        assert cfg.get("extra") == "forbid"
        assert cfg.get("frozen") is True
        assert cfg.get("allow_inf_nan") is False

    def test_attach_classifications_to_detections_enriches_immutably(self) -> None:
        """Verify pure provenance attachment onto canonical CandidateDetection records."""
        d1 = CandidateDetection(
            candidate_id="cand-01",
            field_id="ZTF_01",
            ra_deg=120.0,
            dec_deg=15.0,
            bjd_tdb=2460755.5,
            snr=10.5,
        )
        d2 = CandidateDetection(
            candidate_id="cand-02",
            field_id="ZTF_01",
            ra_deg=120.2,
            dec_deg=15.1,
            bjd_tdb=2460755.5,
            snr=7.1,
        )
        cls_results = [
            ClassificationResult(
                candidate_id="cand-01",
                score=0.91,
                is_real=True,
                model_version="rb-v2",
            ),
            ClassificationResult(
                candidate_id="cand-02",
                score=0.18,
                is_real=False,
                model_version="rb-v2",
            ),
        ]

        enriched_all = attach_classifications_to_detections([d1, d2], cls_results)
        assert d1.cnn_score is None  # Input remains untouched
        assert len(enriched_all) == 2
        assert enriched_all[0].cnn_score == pytest.approx(0.91)
        assert enriched_all[0].model_version == "rb-v2"
        assert enriched_all[1].cnn_score == pytest.approx(0.18)

        real_only = attach_classifications_to_detections(
            [d1, d2],
            cls_results,
            filter_real_only=True,
        )
        assert [det.candidate_id for det in real_only] == ["cand-01"]

        # Missing candidate_id in classifications raises KeyError
        with pytest.raises(KeyError):
            attach_classifications_to_detections([d1, d2], cls_results[:1])

    def test_ml_package_never_imports_physics_modules(self) -> None:
        """AST guard: src/LUMIA/ml must communicate via LUMIA.core, never LUMIA.physics."""
        ml_dir = _REPO_ROOT / "src" / "LUMIA" / "ml"
        violations: list[str] = []
        for py_file in sorted(ml_dir.rglob("*.py")):
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "LUMIA.physics" or alias.name.startswith(
                            "LUMIA.physics."
                        ):
                            violations.append(f"{py_file.name}:{node.lineno} -> {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    if mod == "LUMIA.physics" or mod.startswith("LUMIA.physics."):
                        violations.append(f"{py_file.name}:{node.lineno} -> {mod}")
        assert violations == []

