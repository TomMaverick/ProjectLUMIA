# Module 2: Data Engineering & Task Orchestration Specification

## Overview
Module 2 manages the execution flow, job orchestration, and data persistence across the entire pipeline. It ensures fault tolerance, asynchronous processing, and strict separation between transactional state and analytical catalog data.

## Key Components
* **DAG Orchestration:** Job management powered by `Prefect`, featuring automated retries, exponential backoff, and task failure logging.
* **Polyglot Persistence & Repository Encapsulation:** 
  * `PostgreSQL`: Transactional OLTP store for job statuses, pipeline execution states, human Active Learning triage dispositions, long-term discovery records, and audit logs.
  * `DuckDB` + `Apache Parquet`: Columnar OLAP engine for wide-field stellar reference catalogs, bulk candidate detection tables, and fast triage filtering/aggregation.
  * All SQL DDL/DML and query execution must remain encapsulated behind typed repository interfaces in `data_pipeline`, forbidding raw SQL in `physics`, `ml`, or `api`.
* **Resource Optimization & SSD Protection:** In-memory buffering via `tempfile.SpooledTemporaryFile` to minimize SSD write cycles, VRAM chunking mechanisms for large matrices, and lazy imports to optimize startup performance and memory consumption.