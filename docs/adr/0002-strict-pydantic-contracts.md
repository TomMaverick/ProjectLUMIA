# ADR-0002: Strict Pydantic v2 Contracts Across All Module Boundaries

* **Status:** Accepted
* **Date:** 2026-11-07
* **Deciders:** Project LUMIA Architecture Team
* **Related Documents:** `.agentrules.md` (Rules 3, 4, 7), `docs/audits/architecture_audit_v1.md` (Findings `P0-2`, `P0-3`, `P1-2`, `P1-3`)

---

## Context

Project LUMIA exchanges astrometric sky coordinates, Barycentric Julian Dates, calibrated photometric fluxes, GPU image tensors, neural classification probabilities, and REST/WebSocket JSON payloads across multiple computational and I/O layers. An architectural review (`docs/audits/architecture_audit_v1.md`) highlighted four contract vulnerabilities in early modules when compared against the Module 7 (`src/LUMIA/ml/`) reference baseline:

1. **Mixed Container Paradigms:** Module 4 (`gpu_processing.py`) returned standard-library `@dataclass(frozen=True, slots=True)` containers (`ShiftAndStackResult`, `PsiPhiStackResult`, `ZOGYResult`) and Module 3 returned raw `tuple[np.ndarray, np.ndarray]` pairs, whereas `.agentrules.md` Rule 4 mandates `pydantic` models for all configurations, inter-module data passing, and API schemas. Standard dataclasses perform zero runtime type, range, or finiteness validation.
2. **Implicit Type Coercion & Silent Extra Keys:** Without `strict=True` and `extra="forbid"`, standard Pydantic models silently coerce strings or booleans into numeric types (for example, `"51"` -> `51` or `True` -> `1.0`) and silently ignore misspelled configuration keys in YAML/JSON telescope profiles, risking subtle scientific corruption.
3. **IEEE-754 `NaN` / `Inf` Boundary Leakage:** In Pydantic v2, `ConfigDict(strict=True)` alone still accepts `float("nan")` and `float("inf")` on unconstrained `float` fields unless `allow_inf_nan=False` is explicitly enabled. Unchecked non-finite floats can poison downstream least-squares orbital fits, database aggregations, and JSON serialization (`500 Internal Server Error`).
4. **Schema Duplication & Impure Models:** Defining separate ad-hoc `CandidateDetection` schemas in `physics/linking.py` and `api/schemas.py` introduced field-name and nullability drift, while attaching YAML file I/O methods directly onto configuration models coupled pure schemas to filesystem side effects.

---

## Decision

1. **Universal Pydantic v2 Mandate:**
   * All configuration schemas, inter-module domain entities, computational result containers, and FastAPI request/response payloads must inherit from Pydantic v2 `BaseModel` (via a shared project base class such as `LumiaBaseModel`).
   * Python standard-library `@dataclass`, `NamedTuple`, `TypedDict`, and untyped `dict[str, Any]` payloads are prohibited as public module contracts.

2. **Mandatory Strict `ConfigDict` Baseline:**
   * Every Pydantic model in Project LUMIA must enforce:
     * `strict=True` — Disables implicit type coercion across scalars and collections (JSON/YAML boundary parsing must use `model_validate_json` or explicit loader parsing where appropriate, while Python-to-Python calls enforce exact runtime types).
     * `extra="forbid"` — Immediately raises `ValidationError` on unrecognized or misspelled fields.
     * `frozen=True` — Enforces immutability after instantiation so domain records behave as value objects safe for concurrent caching and DAG passing.
     * `allow_inf_nan=False` — Rejects `NaN`, `+Inf`, and `-Inf` across all floating-point fields automatically without requiring repetitive per-field custom validators.
   * Models wrapping in-memory `torch.Tensor` or `numpy.ndarray` buffers (such as Module 4 GPU result containers) must additionally set `arbitrary_types_allowed=True` paired with `@model_validator(mode="after")` checks verifying dimensionality, shape compatibility, dtype, and finiteness.

3. **Canonical Single-Source-of-Truth Domain Schemas:**
   * Core scientific entities (`ObservatoryLocation`, `TelescopeConfig`, `CandidateDetection`, `ClassificationResult`, `Tracklet`) must be defined once in shared, persistence-free domain/configuration modules and imported or composed by orchestration (`data_pipeline`) and presentation (`api`) layers rather than redefined.

4. **Pure Schema Classes (No Embedded I/O):**
   * Pydantic models must remain pure data and validation definitions. External filesystem reading (YAML, JSON, FITS, HDF5) and SQL serialization must reside in dedicated loader or repository functions that translate low-level parser failures into descriptive domain exceptions.

---

## Consequences

### Positive
* Guarantees fail-fast, self-documenting validation with exact field locations (`loc`) whenever malformed configs, wrong scalar types, unknown keys, or non-finite `NaN`/`Inf` numbers reach any module boundary.
* Eliminates contract drift between the physics/ML engines, DuckDB/PostgreSQL repositories, and FastAPI OpenAPI/TypeScript schemas.
* Enforces functional immutability (`frozen=True`), preventing accidental state mutation across Prefect tasks and multi-threaded API handlers.

### Negative / Trade-offs
* Constructing thousands of individual Pydantic models inside tight inner loops incurs measurable CPU overhead compared to raw NumPy structs or `@dataclass(slots=True)`. High-throughput stages (such as Module 4 pixel grids and Module 7 mini-batches) must keep bulk pixel/coordinate arrays inside batched tensor/array containers and materialize per-candidate Pydantic models only at filtered candidate/tracklet boundaries.
* `strict=True` requires callers (including unit tests and database row mappers) to pass exact Python types (`float` vs. `int`, `tuple` vs. `list` where annotated) rather than relying on implicit coercion.
