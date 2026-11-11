"""Tests for worker-partitioned storage concurrency and offline compaction."""

from __future__ import annotations

import concurrent.futures
from pathlib import Path

import numpy as np
import pytest

from LUMIA.data_mining.storage import WorkerPartitionedZarrStore, compact_worker_stores


def _worker_write_task(root_dir: Path, worker_id: int, num_items: int) -> int:
    """Simulates a crawler worker process writing cutouts to its isolated partition."""
    store = WorkerPartitionedZarrStore(root_dir=root_dir, worker_id=worker_id, stamp_size=48)
    for i in range(num_items):
        triplet = np.full((3, 48, 48), fill_value=float(worker_id * 100 + i), dtype=np.float32)
        cand_id = f"worker_{worker_id}_cand_{i}"
        # Include an overlapping candidate across workers to test de-duplication
        if i == 0:
            cand_id = "shared_duplicate_candidate"
        store.append_cutout(
            candidate_id=cand_id,
            triplet=triplet,
            metadata={"snr": 3.5, "survey": "rubin", "filter": "r"},
        )
    return store.get_record_count()


def test_concurrent_worker_partitioned_writes_no_lock_collision(tmp_path: Path) -> None:
    """Validate 0% lock collisions (WinError 32) when multiple workers write concurrently."""
    num_workers = 4
    items_per_worker = 10

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(_worker_write_task, tmp_path, w_id, items_per_worker)
            for w_id in range(num_workers)
        ]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    # Every worker successfully writes all items
    assert len(results) == num_workers
    for count in results:
        assert count == items_per_worker

    # Verify each worker store partition contains expected records
    partition_paths = []
    for w_id in range(num_workers):
        part_path = tmp_path / f"worker_{w_id}.zarr"
        assert part_path.exists()
        partition_paths.append(part_path)
        store = WorkerPartitionedZarrStore(root_dir=tmp_path, worker_id=w_id, stamp_size=48)
        assert store.get_record_count() == items_per_worker
        records = store.read_records()
        assert len(records) == items_per_worker
        assert records[0]["candidate_id"] == "shared_duplicate_candidate"

    # Compact all partitions into a single destination store
    dest_path = tmp_path / "compacted.zarr"
    total_unique = compact_worker_stores(
        partition_paths=partition_paths,
        destination_store_path=dest_path,
        deduplicate_by="candidate_id",
        stamp_size=48,
    )

    # 4 workers * 10 items = 40 items total.
    # "shared_duplicate_candidate" was written by all 4 workers (3 duplicates).
    # Expected unique items = 40 - 3 = 37.
    assert total_unique == 37
