# Configs

## Overview
The `configs/` directory stores declarative reference data, target asteroid catalogs, and configuration specifications utilized across data mining, synthetic injection, and ephemeris lookups in Project LUMIA.

## Interface
- `objects_catalog.json`: Master catalog specifying targeted solar system objects (NEOs, MBAs, Jupiter Trojans, TNOs, Comets) with designations, SPK IDs, taxonomic classifications, expected magnitude ranges, and physical flags (`has_coma`).

## Invariants
- Catalog JSON entries must validate against target schemas defined in `LUMIA.data_mining.taxonomy.TargetObjectSpec`.
- Each object must define unique `object_id` and `spk_id` identifiers.
- Magnitude ranges must specify valid float pairs `[min_mag, max_mag]` with `min_mag <= max_mag`.

## Side Effects
- None. This directory provides static, read-only declarative configuration data.

## Verification
- Run: `uv run pytest tests/test_data_mining/test_data_mining.py`
