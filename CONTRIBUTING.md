# Contributing to Project LUMIA

Thank you for your interest in contributing to Project LUMIA (Low-SNR Unidentified Moving Interplanetary Analyzer).

Project LUMIA is a research-grade computational astrophysics framework governed by formal Architecture Decision Records (ADRs) and strict automated continuous integration verification.

---

## Architectural Governance & Core Invariants

All contributions must strictly adhere to the project invariants defined in the architecture documentation:

1. **Language & Editorial Standards:**
   - All code, comments, identifiers, documentation, commit messages, and pull request descriptions must be written in English.
   - Emojis are strictly prohibited across all code, comments, documentation, UI strings, and terminal outputs.

2. **Strict Pydantic v2 Data Contracts (ADR-0002):**
   - All inter-module configurations, domain entities, computational result containers, and API schemas must inherit from `LumiaBaseModel` (`strict=True, extra="forbid", frozen=True, allow_inf_nan=False`).
   - Implicit type casting, undeclared fields, post-initialization mutation, and non-finite IEEE-754 floats (`NaN`, `+Inf`, `-Inf`) are rejected immediately.
   - Standard-library `@dataclass` containers are prohibited as inter-module contracts.

3. **Strict Layering & AST Import Firewall (ADR-0001, ADR-0003):**
   - Computational domain packages (`LUMIA.physics`, `LUMIA.ml`, `LUMIA.models`, `LUMIA.data_mining`) must remain purely mathematical.
   - Database engines (`psycopg`, `duckdb`, `sqlalchemy`) and web frameworks (`fastapi`) must never be imported inside computational domain packages. Persistence is owned strictly by `LUMIA.data_pipeline`.
   - Architectural firewalls are enforced automatically via AST inspection in `tests/test_security_and_boundaries.py`.

4. **Deterministic Numerical Execution:**
   - Deterministic PyTorch execution must be preserved (`torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=":4096:8"`).

---

## Development Setup

1. **Install Prerequisites:**
   - Python `>= 3.14`
   - [`uv`](https://docs.astral.sh/uv/) package manager
   - Node.js `>= 20.0.0` and `npm` (for frontend components)
   - NVIDIA CUDA Toolkit (optional, for GPU acceleration)

2. **Clone and Synchronize:**
   ```bash
   git clone https://github.com/project-lumia/ProjectLUMIA.git
   cd ProjectLUMIA
   uv sync
   ```

3. **Frontend Setup:**
   ```bash
   cd frontend
   npm ci
   cd ..
   ```

---

## Testing & Quality Assurance

Before submitting a pull request, verify that all linting, formatting, type checking, and tests pass:

1. **Lint and Format:**
   ```bash
   uv run ruff check .
   uv run ruff format --check .
   ```

2. **Static Type Checking:**
   ```bash
   uv run mypy
   ```

3. **Execute Test Suite:**
   ```bash
   uv run pytest
   ```

4. **Verify Architectural Boundaries:**
   ```bash
   uv run pytest tests/test_security_and_boundaries.py
   ```

---

## Pull Request Guidelines

1. **Test-Driven Development:** Any new feature, algorithm, or bug fix must include dedicated unit tests demonstrating the problem and verifying the resolution.
2. **Commit Messages:** Use clear, descriptive commit messages following the Conventional Commits specification (e.g., `feat(physics): implement lorentz aberration correction in BCRS ephemerides`).
3. **Documentation:** Any change affecting public APIs, data contracts, or astrodynamic assumptions must be reflected in the relevant module documentation and Architecture Decision Records.
