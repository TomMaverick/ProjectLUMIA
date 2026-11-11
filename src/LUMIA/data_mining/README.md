# LUMIA Data Mining

## Overview
LUMIA Data Mining provides an automated ingestion, filtering, synthetic injection, and out-of-core HDF5 curation pipeline. It crawls wide-field astronomical archives (such as ZTF and Tomo-e Gozen), correlates target coordinates with NASA JPL Horizons orbital ephemerides, filters optical artifacts using morphological heuristics, and packages multi-epoch tracklets into compressed HDF5 training datasets.

## Interface
- `DataMiningPipeline`: Master pipeline class executing archive crawling, artifact screening, synthetic augmentation, and dataset persistence.
- `DataMiningPipelineConfig`: Validated configuration governing output HDF5 paths, stamp dimensions, crop jitter offsets, and synthetic target ratios.
- `ArchiveCrawler`: Crawls local or remote astronomical image repositories and extracts calibrated cutout triplets aligned across epochs via in-memory SODA streaming (`tempfile.SpooledTemporaryFile`) supporting ZTF, Pan-STARRS, and Vera C. Rubin.
- `DeepStreaksArtifactFilter` / `DIAArtifactMiner`: Evaluates cutouts for linear trails, diffraction spikes, subtraction dipoles, bleeding columns, and optical ghosts.
- `FaintSyntheticInjector`: Injects simulated asteroid PSF profiles into science cutouts to enrich datasets with low-SNR moving sources.
- `HDF5TrainingStorageManager` / `WorkerPartitionedZarrStore`: Handles structured dataset storage, chunking, and worker-partitioned Zarr v3 files avoiding Windows NTFS lock collisions.
- `HorizonsClient`: Interface for querying the NASA JPL Horizons ephemeris service to compute expected sky positions and rates.
- `ZTFTrackletDataset`: PyTorch Dataset implementation providing out-of-core tensor loading from curated HDF5 files.

## Invariants
- `DataMiningPipelineConfig`: `stamp_size` must be in the range `[32, 256]` and `crop_jitter_max_offset_px` in `[1, 10]`.
- All cutouts written to HDF5 must be strictly 3-channel triplets `(Science, Reference, Difference)` with finite float32 values.
- Synthetic injections must strictly conserve background flux variance and avoid non-physical negative counts.

## Side Effects
- Performs network HTTP requests to NASA JPL Horizons API services.
- Reads FITS and image files from archive directories on the local filesystem.
- Creates and modifies large compressed HDF5 dataset files (`.h5`) on disk.
- Generates PNG preview and quality control diagnostic images.

## Verification
- Run: `uv run pytest tests/test_data_mining`
