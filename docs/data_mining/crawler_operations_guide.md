# ZTF Archive Crawler & Tracklet Mining Operations Guide

## Overview

This guide details the operational procedures, command-line interface (CLI) options, and maintenance workflows for the ZTF Tracklet Mining Pipeline and Archive Crawler (`scripts/run_mining.py`).

The crawler queries JPL Horizons for high-precision topocentric ephemerides (observatory code `I41`), searches the IPAC/IRSA ZTF Image Buffer Engine (IBE) for matching science exposures, retrieves difference/science/reference FITS triplets, partitions observations into sliding-window tracklets ($\Delta \theta_{\text{max}} \le 52.6\text{ arcsec}$), and persists validated data directly into an HDF5 CSR storage container (`ztf_sso_tracklets_144x144.h5`).

---

## Command-Line Interface Reference

The mining pipeline is executed via `uv run python scripts/run_mining.py` with the following arguments:

```bash
uv run python scripts/run_mining.py [OPTIONS]
```

### Options and Parameters

| Flag | Type / Default | Description |
| :--- | :--- | :--- |
| `--catalog` | `Path` (default: `configs/objects_catalog.json`) | Path to the source target objects JSON catalog. |
| `--taxonomy` | `Path` (default: `configs/dataset_splits.json`) | Path to the dataset splits and taxonomy manifest. |
| `--output-hdf5` | `Path` (default: `data/ground_truth/ztf_sso_tracklets_144x144.h5`) | Output HDF5 file path (schema version 2.1.0, stamp size 144x144). |
| `--objects` | Space-separated strings | Specific object names to mine (e.g., `--objects "1P/Halley" "C/2020 F3"`). Defaults to all objects in catalog. |
| `--obs-days` | `Integer` (default: `30`) | Number of ephemeris observation days to span from start epoch. |
| `--step-minutes` | `Integer` (default: `30`) | JPL Horizons ephemeris time resolution in minutes. |
| `--max-cutouts` | `Integer` (default: `1000`) | Maximum number of cutouts/frames to harvest per target object. |
| `--stamp-size` | `Integer` (default: `144`) | Cutout stamp dimensions in pixels ($144 \times 144$). |
| `--survey` | `String` (default: `ZTF`) | Survey platform adapter (`ZTF`, `Pan-STARRS`, `Rubin`). |
| `--faint-injection` | Flag (switch) | Injects faint moving asteroid profiles ($1.5\sigma \le SNR < 5.0\sigma$) for covariate shift mitigation. |
| `--no-synthetic` | Flag (switch) | Disables synthetic tracklet fallback. Downloads real survey FITS data only. |
| `--dry-run` | Flag (switch) | Queries Horizons and survey archives without downloading FITS frames or writing to storage. |
| `--reset-ledger` | Flag (switch) | Clears the SQLite crawler ledger (`crawler_checkpoints.db`) completely before running. |
| `--reset-targets` | Space-separated strings | Clears crawler checkpoint history for specific target objects only. |
| `--qc-dir` | `Path` (default: `data/mining_qc`) | Directory to store diagnostic quality control plots and figures. |
| `--log-level` | `String` (default: `INFO`) | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |

---

## Common Operational Workflows

### 1. Harvesting Real ZTF Data for Specific Objects (Production Mode)

To download real ZTF difference, science, and reference triplets without generating synthetic fallback cutouts:

```bash
uv run python scripts/run_mining.py \
  --objects "1P/Halley" "C/2020 F3" \
  --obs-days 60 \
  --step-minutes 30 \
  --no-synthetic
```

### 2. Mining the Entire Catalog

To crawl all 20 target objects defined in `configs/objects_catalog.json` with strict real-data mode:

```bash
uv run python scripts/run_mining.py \
  --catalog configs/objects_catalog.json \
  --obs-days 90 \
  --no-synthetic
```

### 3. Dry-Run / Ephemeris Inspection

To verify ephemeris coverage and identify existing exposures in the ZTF archive without downloading FITS images or modifying storage:

```bash
uv run python scripts/run_mining.py \
  --objects "3I/ATLAS" \
  --obs-days 14 \
  --dry-run
```

### 4. Overwriting or Retrying Crawls (Ledger Reset)

The crawler tracks processed exposures in an SQLite Write-Ahead Logging (WAL) ledger (`data/ground_truth/crawler_checkpoints.db`). Once an exposure or tracklet is marked as completed, it will be skipped in subsequent runs to conserve network bandwidth and avoid duplicate work.

#### To re-crawl specific objects:
```bash
uv run python scripts/run_mining.py \
  --objects "1P/Halley" \
  --reset-targets "1P/Halley" \
  --no-synthetic
```

#### To wipe the ledger and re-mine from scratch:
```bash
uv run python scripts/run_mining.py \
  --reset-ledger \
  --no-synthetic
```

---

## Cleaning Out Legacy / Synthetic HDF5 Tracklets

If an existing HDF5 file (`data/ground_truth/ztf_sso_tracklets_144x144.h5` or `data/ground_truth/ztf_sso_tracklets_128x128.h5`) contains legacy mock/synthetic tracklets:

1. **Option A: Fresh Dataset Creation (Recommended)**
   Delete or move the old `.h5` file along with the checkpoint database:
   ```powershell
   Remove-Item data/ground_truth/ztf_sso_tracklets_144x144.h5 -ErrorAction SilentlyContinue
   Remove-Item data/ground_truth/crawler_checkpoints.db* -ErrorAction SilentlyContinue
   ```
   Then initiate the crawler with `--no-synthetic`. The storage manager will automatically create a fresh HDF5 container structured with Schema 2.1.0, 144x144 stamps, and deterministic train/val/test/ood splits.

2. **Option B: Distinct Storage Target**
   Direct the output to a dedicated clean file:
   ```bash
   uv run python scripts/run_mining.py \
     --output-hdf5 data/ground_truth/ztf_real_production_144x144.h5 \
     --no-synthetic
   ```

---

## Resiliency and Transport Mechanisms

The crawler includes defensive networking policies conforming to IRSA guidelines:

1. **Token-Bucket Rate Limiting:**
   A thread-safe token bucket enforces a 2–4 requests/sec baseline rate to comply with IRSA guidelines and avoid IP blacklisting.

2. **Automatic Retry & Exponential Backoff:**
   `urllib3.util.Retry` is configured with up to 5 retries and exponential backoff ($1.5 \times 2^{\text{retry}}$) on HTTP status codes `429`, `500`, `502`, `503`, and `504`.

3. **Reference Image Coadd Caching:**
   ZTF coadd reference images are cached in memory indexed by `(field, filter, ccd, qid, ra_bin, dec_bin)`. Multiple detections across the same quadrant reuse the reference coadd rather than re-downloading identical images.

4. **Sliding-Window Tracklet Partitioning:**
   Frames with temporal gaps $> 2.0\text{ hours}$ or sky separations $> 52.6\text{ arcsec}$ are partitioned into separate tracklets, satisfying physical astrometric motion limits.

5. **Two-Phase Commit:**
   Tracklet records and cutouts are committed to HDF5/Zarr only after passing shape validation, followed by immediate status updates in the SQLite WAL ledger. If interrupted, the startup CSR integrity check (`repair_hdf5_csr_integrity()`) automatically repairs any truncated pointer arrays.

6. **IVOA SODA Cutout Streaming via `SpooledTemporaryFile`:**
   Pan-STARRS and Vera C. Rubin endpoints retrieve server-side cutouts directly into `tempfile.SpooledTemporaryFile` buffers in RAM (10 MB threshold), eliminating the overhead and disk writes of multi-hundred-megabyte raw frame downloads.

7. **Worker-Partitioned Storage (`worker_{id}.zarr`) on Windows NTFS:**
   To guarantee zero file locking collisions (`PermissionError: [WinError 32]`) when multiple multiprocessing crawler workers write simultaneously, workers write strictly to isolated directory stores (`store/worker_{id}.zarr`). A sequential compaction process (`compact_worker_stores`) merges worker stores offline.
