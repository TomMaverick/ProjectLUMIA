# Documentation

## Overview
The `docs/` directory contains the complete architectural, mathematical, and procedural documentation for Project LUMIA. It centralizes master system requirements, Architecture Decision Records (ADRs), module specifications, database schemas, audits, and subsystem blueprints.

## Local Interface & Invariants
- [lastenheft.md](./lastenheft.md): Master requirements specification (Lastenheft) defining functional requirements, scientific constraints, and performance targets.
- [database_schema_blueprint.md](./database_schema_blueprint.md): Physical database schema blueprint defining PostgreSQL 17 3NF DDL and DuckDB/Parquet analytical table structures.
- [architecture_compliance_report.md](./architecture_compliance_report.md): Formal compliance assessment and verification status against architectural directives.
- [backlog.md](./backlog.md): Prioritized implementation backlog and sprint roadmap.

## Subdirectories & Boundaries
- [adr](./adr/README.md): Binding Architecture Decision Records governing boundaries, strict Pydantic contracts, and polyglot persistence.
- [assets](./assets/README.md): Vector logos and graphic assets.
- [audits](./audits/README.md): Architecture compliance reviews and consistency audit baselines.
- [data_mining](./data_mining/README.md): Architectural blueprints for astronomical survey data mining, artifact screening, and HDF5 curation.
- [designs](./designs/README.md): Application sitemaps and interaction models.
- [frontend](./frontend/README.md): UI/UX design specifications, dark-mode ergonomics, and component contracts.
- [modules](./modules/README.md): Technical specifications for the ten canonical pipeline modules (Modules 1–10).
- [training](./training/README.md): Deep learning blueprints, tracklet network architectures, and training optimization proposals.
