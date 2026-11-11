"""Unit tests for solar system taxonomy domain entities and deterministic dataset splitting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from LUMIA.data_mining.taxonomy import (
    CatalogObjectEntry,
    DatasetSplitsManifest,
    SplitName,
    TargetObjectSpec,
    generate_dataset_splits,
    load_objects_catalog,
)


def test_catalog_object_entry_strict_validation() -> None:
    """Verify strict Pydantic v2 validation and immutability of CatalogObjectEntry."""
    valid_entry = CatalogObjectEntry(
        object_id="99942",
        name="Apophis",
        taxonomy_class="NEO",
        magnitude_range=[15.0, 21.5],
        spk_id="2099942",
        designation="99942 Apophis",
        has_coma=False,
        is_candidate=False,
        h_mag=19.7,
    )
    assert valid_entry.object_id == "99942"
    assert not valid_entry.is_candidate

    # Verify frozen immutability
    with pytest.raises(ValidationError):
        valid_entry.object_id = "101955"  # type: ignore[misc]

    # Verify extra fields forbidden (e.g. deprecated split_assignment)
    with pytest.raises(ValidationError):
        CatalogObjectEntry(
            object_id="99942",
            name="Apophis",
            taxonomy_class="NEO",
            magnitude_range=[15.0, 21.5],
            split_assignment="train",  # type: ignore[call-arg]
        )

    # Verify to_target_spec conversion
    spec = valid_entry.to_target_spec()
    assert isinstance(spec, TargetObjectSpec)
    assert spec.target_id == valid_entry.object_id
    assert spec.is_candidate == valid_entry.is_candidate


def test_dataset_splits_manifest_schema_validation() -> None:
    """Verify DatasetSplitsManifest schema constraints and immutability."""
    manifest = DatasetSplitsManifest(
        schema_version="1.0.0",
        catalog_checksum_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        created_at_utc="2026-11-10T00:00:00Z",
        splits={
            "train": ["99942"],
            "val": ["101955"],
            "test": ["65803"],
            "ood": ["2I"],
        },
        class_distribution={
            "NEO": {"train": 1, "val": 1, "test": 1, "ood": 0},
            "ISO": {"train": 0, "val": 0, "test": 0, "ood": 1},
        },
    )
    assert manifest.schema_version == "1.0.0"
    assert manifest.splits["train"] == ["99942"]

    # Immutability
    with pytest.raises(ValidationError):
        manifest.schema_version = "2.0.0"  # type: ignore[misc]

    # Extra fields forbidden
    with pytest.raises(ValidationError):
        DatasetSplitsManifest(
            schema_version="1.0.0",
            catalog_checksum_sha256="test",
            created_at_utc="2026-11-10T00:00:00Z",
            splits={"train": []},  # type: ignore[arg-type]
            class_distribution={},
            unsupported_extra_field=123,  # type: ignore[call-arg]
        )


def test_objects_catalog_json_validity() -> None:
    """Verify objects_catalog.json contents adhere to Phase 1 domain requirements."""
    catalog_path = Path("configs/objects_catalog.json")
    assert catalog_path.exists(), "configs/objects_catalog.json must exist"

    with catalog_path.open("r", encoding="utf-8") as f:
        raw_catalog = json.load(f)

    raw_targets = raw_catalog.get("targets", [])
    assert len(raw_targets) > 0

    target_ids = {t["object_id"] for t in raw_targets}

    # 1I/'Oumuamua must be pruned (unobservable in ZTF operations)
    assert "1I" not in target_ids
    assert "'Oumuamua" not in [t["name"] for t in raw_targets]

    # Observable ISOs must be present
    assert "2I" in target_ids
    assert "3I" in target_ids

    # All entries must parse into CatalogObjectEntry without extra attributes
    entries = load_objects_catalog(catalog_path)
    assert len(entries) == len(raw_targets)

    for entry in entries:
        assert entry.spk_id is not None, f"Target {entry.object_id} must have non-null SPK-ID"
        assert len(entry.spk_id) > 0


def test_generate_dataset_splits_proportions_and_floors(tmp_path: Path) -> None:
    """Verify deterministic stratified partitioning proportions and small-class floor constraints."""
    mock_catalog_data = {
        "catalog_version": "test-1.0.0",
        "targets": [
            # 11 Comets: should result in 9 train, 1 val, 1 test, 0 ood
            *[
                {
                    "object_id": f"COMET_{i:02d}",
                    "name": f"Comet {i}",
                    "taxonomy_class": "COMET",
                    "magnitude_range": [15.0, 20.0],
                    "spk_id": f"9000{i:04d}",
                    "has_coma": True,
                }
                for i in range(11)
            ],
            # 5 NEOs (small class 3 <= N < 10): should have at least 1 val and 1 test (e.g. 3 train, 1 val, 1 test)
            *[
                {
                    "object_id": f"NEO_{i:02d}",
                    "name": f"NEO {i}",
                    "taxonomy_class": "NEO",
                    "magnitude_range": [18.0, 22.0],
                    "spk_id": f"2000{i:04d}",
                }
                for i in range(5)
            ],
            # 2 ISOs: both routed strictly to ood
            {
                "object_id": "2I",
                "name": "Borisov",
                "taxonomy_class": "ISO",
                "magnitude_range": [15.0, 22.0],
                "spk_id": "1003632",
                "has_coma": True,
            },
            {
                "object_id": "3I",
                "name": "ATLAS",
                "taxonomy_class": "ISO",
                "magnitude_range": [16.0, 22.0],
                "spk_id": "1003988",
                "has_coma": True,
                "is_candidate": True,
            },
        ],
    }

    mock_catalog_path = tmp_path / "mock_objects_catalog.json"
    mock_manifest_path = tmp_path / "mock_dataset_splits.json"
    mock_catalog_path.write_text(json.dumps(mock_catalog_data, indent=2), encoding="utf-8")

    manifest = generate_dataset_splits(mock_catalog_path, mock_manifest_path)

    assert mock_manifest_path.exists()
    assert isinstance(manifest, DatasetSplitsManifest)

    # Checksum verification
    expected_hash = hashlib.sha256(mock_catalog_path.read_bytes()).hexdigest()
    assert manifest.catalog_checksum_sha256 == expected_hash

    # Verify COMET partition (11 objects: 9 train, 1 val, 1 test, 0 ood)
    comet_dist = manifest.class_distribution["COMET"]
    assert comet_dist["train"] == 9
    assert comet_dist["val"] == 1
    assert comet_dist["test"] == 1
    assert comet_dist["ood"] == 0

    # Verify small-class NEO partition (5 objects: 3 train, 1 val, 1 test, 0 ood)
    neo_dist = manifest.class_distribution["NEO"]
    assert neo_dist["train"] == 3
    assert neo_dist["val"] >= 1
    assert neo_dist["test"] >= 1
    assert neo_dist["ood"] == 0

    # Verify ISO partition (2 objects strictly in ood)
    iso_dist = manifest.class_distribution["ISO"]
    assert iso_dist["train"] == 0
    assert iso_dist["val"] == 0
    assert iso_dist["test"] == 0
    assert iso_dist["ood"] == 2
    assert "2I" in manifest.splits["ood"]
    assert "3I" in manifest.splits["ood"]

    # Verify zero overlap across any partitions
    all_splits: list[SplitName] = ["train", "val", "test", "ood"]
    for i, s1 in enumerate(all_splits):
        for s2 in all_splits[i + 1 :]:
            overlap = set(manifest.splits[s1]) & set(manifest.splits[s2])
            assert not overlap, f"Target leakage between {s1} and {s2}: {overlap}"

    # Verify total targets partitioned equals input targets
    total_partitioned = sum(len(manifest.splits[s]) for s in all_splits)
    assert total_partitioned == 11 + 5 + 2


def test_real_dataset_splits_manifest_integrity() -> None:
    """Verify production configs/dataset_splits.json matches configs/objects_catalog.json."""
    catalog_path = Path("configs/objects_catalog.json")
    manifest_path = Path("configs/dataset_splits.json")

    assert catalog_path.exists(), "configs/objects_catalog.json must exist"
    assert manifest_path.exists(), "configs/dataset_splits.json must exist"

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    manifest = DatasetSplitsManifest.model_validate(manifest_data)

    # SHA256 checksum check
    catalog_bytes = catalog_path.read_bytes()
    expected_hash = hashlib.sha256(catalog_bytes).hexdigest()
    assert (
        manifest.catalog_checksum_sha256 == expected_hash
    ), "Split manifest catalog checksum does not match configs/objects_catalog.json"

    # Verify zero target overlap
    all_splits: list[SplitName] = ["train", "val", "test", "ood"]
    for i, s1 in enumerate(all_splits):
        for s2 in all_splits[i + 1 :]:
            overlap = set(manifest.splits[s1]) & set(manifest.splits[s2])
            assert not overlap, f"Target leakage between {s1} and {s2}: {overlap}"

    # Verify ISO isolation
    assert set(manifest.splits["ood"]) == {"2I", "3I"}

    # Verify all catalog objects are partitioned
    entries = load_objects_catalog(catalog_path)
    catalog_ids = {e.object_id for e in entries}
    partitioned_ids = set().union(*(manifest.splits[s] for s in all_splits))
    assert partitioned_ids == catalog_ids
