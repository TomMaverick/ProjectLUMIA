# Architecture Decision Records (ADRs)

## Overview
The `docs/adr/` directory houses the binding Architecture Decision Records (ADRs) that govern the design, boundaries, type safety, and data persistence models of Project LUMIA.

## Interface
- [0001-module-boundaries-and-numbering.md](./0001-module-boundaries-and-numbering.md): Establishes the canonical 10-module pipeline architecture and enforces strict unidirectional dependency flow.
- [0002-strict-pydantic-contracts.md](./0002-strict-pydantic-contracts.md): Mandates Pydantic v2 models with `strict=True`, `extra="forbid"`, `frozen=True`, and `allow_inf_nan=False` across all module boundaries.
- [0003-duckdb-vs-postgresql-persistence.md](./0003-duckdb-vs-postgresql-persistence.md): Mandates the polyglot persistence architecture, assigning DuckDB and Apache Parquet to analytical OLAP tasks and PostgreSQL 17 to transactional OLTP operations.

## Invariants
- ADRs document architectural decisions and cannot be altered without an explicit superseded record and stakeholder approval.
- Code changes in production and test suites must strictly adhere to all active ADRs.

## Side Effects
- None. These files are static documentation records.

## Verification
- Run: `uv run pytest tests/test_security_and_boundaries.py`
