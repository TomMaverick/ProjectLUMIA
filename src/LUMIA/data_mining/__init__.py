"""Project LUMIA Data-Mining Subsystem.

Automated ephemeris retrieval, archive cutout crawling, DeepStreaks artifact rejection,
dihedral D4 augmentation, faint synthetic moving-object injection, and out-of-core
HDF5 training dataset curation for solar system small bodies.
"""

from __future__ import annotations

from LUMIA.data_mining.archive_crawler import (
    ArchiveCrawler,
    CutoutTriplet,
    ExposureMetadata,
    StampSize,
    TrackletSeries,
    WCSMetadata,
    build_ztf_ibe_cutout_url,
    compute_fixed_sky_midpoint,
    extract_wcs_metadata,
    partition_tracklet_sliding_window,
    slice_subpixel_cutout,
    verify_wcs_containment_and_quality,
)
from LUMIA.data_mining.artifact_filter import (
    ARTIFACT_TAXONOMY_MAP,
    ArtifactAssessment,
    ArtifactFilterConfig,
    ArtifactType,
    DeepStreaksArtifactFilter,
)
from LUMIA.data_mining.augmentation import (
    DihedralOp,
    FaintSyntheticConfig,
    FaintSyntheticInjector,
    InjectedSampleMetadata,
    apply_d4_transform,
    apply_d4_transform_with_coords,
    apply_spatial_crop_jitter,
    transform_coordinates_d4,
)
from LUMIA.data_mining.dataset import (
    DEFAULT_ZTF_DATASET_PATH,
    ZTFTrackletDataset,
    create_ztf_tracklet_dataloader,
    pad_tracklet_batch,
    ztf_tracklet_collate_fn,
)
from LUMIA.data_mining.pipeline import (
    DataMiningPipelineConfig,
    PipelineCampaignReport,
    SolarSystemDataMiningPipeline,
    TargetProcessingReport,
)
from LUMIA.data_mining.qc_export import (
    export_cutout_triplet_qc_png,
)
from LUMIA.data_mining.storage import (
    AsteroidTrainingSample,
    HDF5TrainingStorageManager,
    SplitName,
    TrackletSample,
)
from LUMIA.data_mining.taxonomy import (
    CatalogObjectEntry,
    EphemerisPoint,
    HorizonsClient,
    HorizonsQueryConfig,
    TargetObjectSpec,
    TaxonomyClass,
    compute_optimal_cadence_window,
    load_objects_catalog,
)

__all__ = [
    "ARTIFACT_TAXONOMY_MAP",
    "ArchiveCrawler",
    "ArtifactAssessment",
    "ArtifactFilterConfig",
    "ArtifactType",
    "AsteroidTrainingSample",
    "CatalogObjectEntry",
    "CutoutTriplet",
    "DEFAULT_ZTF_DATASET_PATH",
    "DataMiningPipelineConfig",
    "DeepStreaksArtifactFilter",
    "DihedralOp",
    "EphemerisPoint",
    "ExposureMetadata",
    "FaintSyntheticConfig",
    "FaintSyntheticInjector",
    "HDF5TrainingStorageManager",
    "HorizonsClient",
    "HorizonsQueryConfig",
    "InjectedSampleMetadata",
    "PipelineCampaignReport",
    "SolarSystemDataMiningPipeline",
    "SplitName",
    "StampSize",
    "TargetObjectSpec",
    "TargetProcessingReport",
    "TaxonomyClass",
    "TrackletSample",
    "TrackletSeries",
    "WCSMetadata",
    "ZTFTrackletDataset",
    "apply_d4_transform",
    "apply_d4_transform_with_coords",
    "apply_spatial_crop_jitter",
    "build_ztf_ibe_cutout_url",
    "compute_fixed_sky_midpoint",
    "compute_optimal_cadence_window",
    "create_ztf_tracklet_dataloader",
    "export_cutout_triplet_qc_png",
    "extract_wcs_metadata",
    "load_objects_catalog",
    "pad_tracklet_batch",
    "partition_tracklet_sliding_window",
    "slice_subpixel_cutout",
    "transform_coordinates_d4",
    "verify_wcs_containment_and_quality",
    "ztf_tracklet_collate_fn",
]
